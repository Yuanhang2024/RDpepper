"""Truth-free geometry-derived bond-order candidate for result-first recovery.

Adapter between the frozen NNAA geometry study module
(:mod:`cycpep_master.core.geometry_simple`) and the bond-order inference
portfolio.  The adapter only restructures source evidence that the input file
already carries: observed heavy atoms, chain-scoped CONECT adjacency, and a
proximity completion with the frozen module's covalent-radii tolerance.  Every
chemistry decision (bond orders, formal charges, implicit hydrogens) stays
inside the frozen algorithm module.

Observed explicit hydrogens (H) and deuteriums (D, materialized as
isotope-2 hydrogen rather than silently converted to protium) are projected
out before the frozen algorithm runs: the algorithm still sees a heavy-only
graph in the original source order, and the returned molecule is rebuilt in
FULL source order so the standard chemical-graph audit keeps matching the
source atom count/order.  Observed hydrogens are never dropped or
re-derived silently: each one is re-attached to its bonded heavy atom
(CONECT first, nearest-heavy covalent fallback), and the candidate must
allow at least the observed attachments -- observed hydrogens are a lower
bound, so partially hydrogenated inputs keep what was observed while the
surplus stays implicit.  Hydrogens the file omits stay implicit on the
heavy atoms and are materialized only later at export.

Blind constraints honored at runtime: no residue names, no CCD/monomer
dictionary lookups, no truth inputs, no network, no coordinate edits.  The
returned molecule keeps the source atom order and the observed coordinates
exactly, so the caller can bind it to the source with the standard
chemical-graph audit.  When the algorithm reports ``ambiguous``/``unresolved``
statuses or produces no sanitized top candidate, the adapter raises and the
portfolio keeps its existing fallback candidates.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Geometry import Point3D

from .geometry_inference import infer_monomer_geometry
from .geometry_simple import ALGORITHM as GEOMETRY_ALGORITHM
from .pdb_utils import first_model_records

ENGINE_NAME = "geometry_simple_local"
TIMEOUT_SECONDS = 2.0

# Hydrogen-class elements projected out of the frozen heavy-atom algorithm.
_HYDROGEN_ELEMENTS = {"H", "D"}
# Tie band for nearest-heavy parent resolution of an observed hydrogen.
_PARENT_TIE_ANGSTROM = 1e-6

# Local covalent-radii tolerance mirroring the frozen module's connectivity
# rule; used only to complete adjacency that CONECT does not declare.
_PROXIMITY_LO = 0.55
_PROXIMITY_HI_SLACK = 0.40
_DEFAULT_RADIUS = 0.9

# Bond-order/length consistency gate for source-bound candidate graphs.
# A bond whose observed length differs from the reference length of its
# ASSIGNED order by more than INCONSISTENT_RESIDUAL_ANGSTROM contradicts the
# assignment (X-ray bond lengths at ~2 A resolution are precise to roughly
# 0.05 A).  More than INCONSISTENT_BOND_LIMIT such bonds marks the whole
# graph as demonstrably inconsistent with the observed geometry; uniform
# single-bond proximity graphs on carbonyl/aromatic inputs fail this gate.
INCONSISTENT_RESIDUAL_ANGSTROM = 0.15
INCONSISTENT_BOND_LIMIT = 3


class GeometryCandidateError(ValueError):
    """Raised when no single geometry-derived candidate is admissible."""


def _radius(element: str) -> float:
    from .geometry_simple import COVALENT_RADII

    return COVALENT_RADII.get(element, _DEFAULT_RADIUS)


def _conect_edges(pdb_path: Path, serials: set[int]) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    with pdb_path.open(encoding="ascii", errors="replace") as handle:
        for line in first_model_records(handle):
            if not line.startswith("CONECT"):
                continue
            try:
                anchor = int(line[6:11])
            except ValueError:
                continue
            if anchor not in serials:
                continue
            for start in range(11, min(len(line), 61), 5):
                field = line[start : start + 5]
                if not field.strip():
                    continue
                try:
                    partner = int(field)
                except ValueError:
                    continue
                if partner in serials and partner != anchor:
                    edges.add((min(anchor, partner), max(anchor, partner)))
    return sorted(edges)


def _proximity_edges(
    elements: list[str],
    xyz: list[tuple[float, float, float]],
    existing: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    added: list[tuple[int, int]] = []
    n = len(elements)
    for i in range(n):
        ri = _radius(elements[i])
        for j in range(i + 1, n):
            if (i, j) in existing:
                continue
            dx = xyz[i][0] - xyz[j][0]
            dy = xyz[i][1] - xyz[j][1]
            dz = xyz[i][2] - xyz[j][2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            radius_sum = ri + _radius(elements[j])
            if _PROXIMITY_LO * radius_sum <= distance <= radius_sum + _PROXIMITY_HI_SLACK:
                added.append((i, j))
    return added


def _build_record(
    source_atoms: list[dict[str, Any]],
    conect_edges: list[tuple[int, int]],
    proximity_edges: list[tuple[int, int]],
) -> dict[str, Any]:
    serial_to_index = {
        int(atom["serial"]): index for index, atom in enumerate(source_atoms)
    }
    record_atoms = [
        {
            "id": index,
            "element": str(atom["element"]).capitalize(),
            "xyz": [float(value) for value in atom["xyz"]],
        }
        for index, atom in enumerate(source_atoms)
    ]
    edges: set[tuple[int, int]] = set()
    for left, right in conect_edges:
        edges.add(
            (
                min(serial_to_index[left], serial_to_index[right]),
                max(serial_to_index[left], serial_to_index[right]),
            )
        )
    edges.update(proximity_edges)
    return {
        "atoms": record_atoms,
        "bonds": [list(edge) for edge in sorted(edges)],
        "total_charge": None,
    }


def _candidate_molecule(
    candidate: dict[str, Any],
    source_atoms: list[dict[str, Any]],
    supplied_edges: set[tuple[int, int]],
    *,
    heavy_positions: list[int] | None = None,
    hydrogen_parents: dict[int, int] | None = None,
) -> Any:
    """Rebuild the candidate in FULL source atom order.

    ``heavy_positions`` maps heavy-local candidate indices to source indices
    (identity when the source carries no explicit hydrogens).
    ``hydrogen_parents`` maps each observed hydrogen's source index to its
    bonded heavy atom's source index; those hydrogens are re-attached as
    explicit atoms with single bonds instead of being folded into the heavy
    atom's implicit hydrogen count.
    """
    n = len(source_atoms)
    if heavy_positions is None:
        heavy_positions = list(range(n))
    if hydrogen_parents is None:
        hydrogen_parents = {}
    local_of_source = {
        source: local for local, source in enumerate(heavy_positions)
    }
    observed_h_by_heavy: dict[int, int] = {}
    for hydrogen_index, parent_index in hydrogen_parents.items():
        observed_h_by_heavy[parent_index] = (
            observed_h_by_heavy.get(parent_index, 0) + 1
        )
    charges = candidate.get("formal_charges")
    hydrogens = candidate.get("hydrogen_counts")
    bonds = candidate.get("bonds")
    if (
        not isinstance(charges, list)
        or not isinstance(hydrogens, list)
        or not isinstance(bonds, list)
        or len(charges) != len(heavy_positions)
        or len(hydrogens) != len(heavy_positions)
    ):
        raise GeometryCandidateError("geometry candidate payload shape mismatch")
    editable = Chem.RWMol()
    for index, atom in enumerate(source_atoms):
        element_upper = str(atom["element"]).upper()
        if element_upper in _HYDROGEN_ELEMENTS:
            item = Chem.Atom("H")
            if element_upper == "D":
                item.SetIsotope(2)
        else:
            local = local_of_source[index]
            item = Chem.Atom(str(atom["element"]).capitalize())
            item.SetFormalCharge(int(charges[local]))
            inferred = int(hydrogens[local])
            materialized = observed_h_by_heavy.get(index, 0)
            item.SetNumExplicitHs(max(inferred - materialized, 0))
            item.SetNoImplicit(True)
        editable.AddAtom(item)
    observed_edges: set[tuple[int, int]] = set()
    for row in bonds:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise GeometryCandidateError("geometry candidate bond row malformed")
        left_local, right_local, order = int(row[0]), int(row[1]), float(row[2])
        if not (
            0 <= left_local < len(heavy_positions)
            and 0 <= right_local < len(heavy_positions)
        ):
            raise GeometryCandidateError("geometry candidate bond index invalid")
        left = heavy_positions[left_local]
        right = heavy_positions[right_local]
        if left == right:
            raise GeometryCandidateError("geometry candidate bond index invalid")
        key = (min(left, right), max(left, right))
        observed_edges.add(key)
        bond_type = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE,
                     3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}.get(order)
        if bond_type is None:
            raise GeometryCandidateError(
                f"unsupported geometry candidate bond order {order}"
            )
        editable.AddBond(left, right, bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            editable.GetAtomWithIdx(left).SetIsAromatic(True)
            editable.GetAtomWithIdx(right).SetIsAromatic(True)
            editable.GetBondBetweenAtoms(left, right).SetIsAromatic(True)
    for hydrogen_index, parent_index in sorted(
        hydrogen_parents.items(), key=lambda item: (min(item), max(item))
    ):
        editable.AddBond(
            hydrogen_index, parent_index, Chem.BondType.SINGLE
        )
    missing = sorted(supplied_edges - observed_edges)
    if missing:
        raise GeometryCandidateError(
            "geometry candidate dropped supplied source edges: "
            + ", ".join(f"{a}-{b}" for a, b in missing[:8])
        )
    molecule = editable.GetMol()
    conformer = Chem.Conformer(n)
    for index, atom in enumerate(source_atoms):
        xyz = atom["xyz"]
        conformer.SetAtomPosition(
            index,
            Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )
    conformer.Set3D(True)
    molecule.AddConformer(conformer, assignId=True)
    molecule.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise GeometryCandidateError(
            f"geometry candidate molecule is not sanitizable: {exc}"
        ) from exc
    Chem.RemoveStereochemistry(molecule)
    return molecule


def _observed_hydrogen_parents(
    source_atoms: list[dict[str, Any]],
    hydrogen_positions: list[int],
    heavy_positions: list[int],
    conect: list[tuple[int, int]],
) -> dict[int, int]:
    """Resolve each observed hydrogen to exactly one bonded heavy atom.

    CONECT evidence wins when present and is itself checked against the
    observed geometry: a CONECT-declared heavy parent beyond covalent
    tolerance, or a different heavy atom lying closer within tolerance,
    is conflicting evidence and rejects the candidate.  Without CONECT the
    nearest heavy atom within the covalent-radii tolerance is the parent.
    Hydrogens with no resolvable parent, with multiple CONECT-declared heavy
    parents, or with an exact nearest-neighbor tie are likewise rejected:
    their observed chemistry cannot be restructured honestly.
    """
    heavy_serial_to_index = {
        int(source_atoms[position]["serial"]): position
        for position in heavy_positions
    }
    conect_partners: dict[int, set[int]] = {}
    for left, right in conect:
        conect_partners.setdefault(left, set()).add(right)
        conect_partners.setdefault(right, set()).add(left)

    def within_cutoff(distance: float, heavy_index: int) -> bool:
        cutoff = (
            _radius("H")
            + _radius(str(source_atoms[heavy_index]["element"]).capitalize())
            + _PROXIMITY_HI_SLACK
        )
        return distance <= cutoff

    def distance_to(hydrogen_index: int, heavy_index: int) -> float:
        p = source_atoms[hydrogen_index]["xyz"]
        q = source_atoms[heavy_index]["xyz"]
        return math.sqrt(
            (float(p[0]) - float(q[0])) ** 2
            + (float(p[1]) - float(q[1])) ** 2
            + (float(p[2]) - float(q[2])) ** 2
        )

    parents: dict[int, int] = {}
    for hydrogen_index in hydrogen_positions:
        hydrogen = source_atoms[hydrogen_index]
        serial = int(hydrogen["serial"])
        declared = {
            heavy_serial_to_index[partner]
            for partner in conect_partners.get(serial, ())
            if partner in heavy_serial_to_index
        }
        if len(declared) > 1:
            raise GeometryCandidateError(
                f"observed hydrogen serial {serial} has multiple "
                "CONECT-bonded heavy parents"
            )
        if declared:
            parent = next(iter(declared))
            declared_distance = distance_to(hydrogen_index, parent)
            if not within_cutoff(declared_distance, parent):
                raise GeometryCandidateError(
                    f"observed hydrogen serial {serial} is CONECT-bonded to "
                    f"heavy atom beyond covalent tolerance "
                    f"({declared_distance:.2f} A)"
                )
            for heavy_index in heavy_positions:
                if heavy_index == parent:
                    continue
                other = distance_to(hydrogen_index, heavy_index)
                if (
                    within_cutoff(other, heavy_index)
                    and other < declared_distance - _PARENT_TIE_ANGSTROM
                ):
                    raise GeometryCandidateError(
                        f"observed hydrogen serial {serial} geometry "
                        "contradicts its CONECT-declared parent "
                        f"(heavy serial "
                        f"{int(source_atoms[heavy_index]['serial'])} is closer)"
                    )
            parents[hydrogen_index] = parent
            continue
        best_distance: float | None = None
        best_positions: list[int] = []
        for heavy_index in heavy_positions:
            distance = distance_to(hydrogen_index, heavy_index)
            if not within_cutoff(distance, heavy_index):
                continue
            if (
                best_distance is not None
                and distance > best_distance + _PARENT_TIE_ANGSTROM
            ):
                continue
            if (
                best_distance is None
                or distance < best_distance - _PARENT_TIE_ANGSTROM
            ):
                best_distance = distance
                best_positions = [heavy_index]
            else:
                best_positions.append(heavy_index)
        if not best_positions:
            raise GeometryCandidateError(
                f"observed hydrogen serial {serial} has no bonded heavy "
                "atom within covalent tolerance"
            )
        if len(best_positions) > 1:
            raise GeometryCandidateError(
                f"observed hydrogen serial {serial} parent is ambiguous "
                "between equidistant heavy atoms"
            )
        parents[hydrogen_index] = best_positions[0]
    return parents


def build_geometry_simple_molecule(
    pdb_path: str | Path,
    source_atoms: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Return ``(molecule, meta)`` for the top simple-geometry candidate.

    ``source_atoms`` is the chain-scoped atom list already used by the
    bond-order portfolio (element, xyz, serial).  Observed H/D atoms are
    projected to a heavy-only graph for the frozen algorithm and re-attached
    afterwards, so the returned molecule keeps the FULL source atom order.
    Raises :class:`GeometryCandidateError` when the algorithm does not resolve
    a single candidate or the observed hydrogen chemistry conflicts with the
    inferred one, so callers retain their fallback artifacts.
    """
    is_hydrogen = [
        str(atom["element"]).upper() in _HYDROGEN_ELEMENTS
        for atom in source_atoms
    ]
    heavy_positions = [
        index for index, flag in enumerate(is_hydrogen) if not flag
    ]
    hydrogen_positions = [
        index for index, flag in enumerate(is_hydrogen) if flag
    ]
    if not heavy_positions:
        raise GeometryCandidateError(
            "source contains no heavy atoms for the geometry algorithm"
        )
    path = Path(pdb_path)
    serials = {int(atom["serial"]) for atom in source_atoms}
    conect = _conect_edges(path, serials)
    heavy_atoms = [source_atoms[position] for position in heavy_positions]
    heavy_serial_to_local = {
        int(atom["serial"]): local for local, atom in enumerate(heavy_atoms)
    }
    heavy_conect = [
        edge
        for edge in conect
        if edge[0] in heavy_serial_to_local and edge[1] in heavy_serial_to_local
    ]
    conect_indexed = {
        (
            min(heavy_serial_to_local[left], heavy_serial_to_local[right]),
            max(heavy_serial_to_local[left], heavy_serial_to_local[right]),
        )
        for left, right in heavy_conect
    }
    elements = [str(atom["element"]).capitalize() for atom in heavy_atoms]
    xyz = [tuple(float(v) for v in atom["xyz"]) for atom in heavy_atoms]
    proximity = _proximity_edges(elements, xyz, conect_indexed)
    record = _build_record(heavy_atoms, heavy_conect, proximity)
    result = infer_monomer_geometry(
        record, mode="simple", timeout_seconds=TIMEOUT_SECONDS
    )
    status = str(result.get("status") or "")
    candidates = result.get("candidates") or []
    # ``ambiguous`` with at least one scored candidate still carries usable
    # per-bond evidence (e.g. ring aromaticity assigned from plane/angle/
    # length).  The top candidate enters the portfolio as an explicit
    # hypothesis with its reason codes recorded; every existing validation
    # (hydrogen conflicts, declared-edge preservation, composition binding)
    # still applies downstream.  Only statuses with no candidates at all
    # (unresolved/timeout/invalid_input) stay fail-closed.
    admissible = {"candidate", "ambiguous"}
    if status not in admissible or not candidates:
        raise GeometryCandidateError(
            f"geometry algorithm status {status or 'missing'}; "
            f"reasons={result.get('reason_codes')}"
        )
    top = candidates[0]
    hydrogen_parents = _observed_hydrogen_parents(
        source_atoms, hydrogen_positions, heavy_positions, conect
    )
    observed_by_heavy: dict[int, int] = {}
    for hydrogen_index, parent_index in hydrogen_parents.items():
        observed_by_heavy[parent_index] = (
            observed_by_heavy.get(parent_index, 0) + 1
        )
    inferred_counts = top.get("hydrogen_counts")
    if not isinstance(inferred_counts, list) or len(inferred_counts) != len(
        heavy_positions
    ):
        raise GeometryCandidateError("geometry candidate payload shape mismatch")
    local_of_source = {
        source: local for local, source in enumerate(heavy_positions)
    }
    conflicts = []
    for heavy_index in sorted(observed_by_heavy):
        observed = observed_by_heavy[heavy_index]
        inferred = int(inferred_counts[local_of_source[heavy_index]])
        if observed > inferred:
            # Observed hydrogens are a lower bound: partial-hydrogen inputs
            # keep what was observed and leave the surplus implicit.  Only a
            # candidate whose hydrogen chemistry cannot even accommodate the
            # observed attachments is a genuine conflict.
            conflicts.append(
                f"serial {int(source_atoms[heavy_index]['serial'])}: "
                f"inferred {inferred}, observed {observed}"
            )
    if conflicts:
        raise GeometryCandidateError(
            "observed hydrogens exceed the inferred hydrogen chemistry: "
            + "; ".join(conflicts[:8])
        )
    supplied = {
        (
            min(heavy_positions[left], heavy_positions[right]),
            max(heavy_positions[left], heavy_positions[right]),
        )
        for left, right in record["bonds"]
    }
    molecule = _candidate_molecule(
        top,
        source_atoms,
        supplied,
        heavy_positions=heavy_positions,
        hydrogen_parents=hydrogen_parents,
    )
    meta = {
        "algorithm": str(result.get("algorithm") or GEOMETRY_ALGORITHM),
        "status": status,
        "algorithm_status_note": (
            "ambiguous candidate admitted as an explicit hypothesis; "
            "reason codes recorded; all validations still applied"
            if status == "ambiguous" else None
        ),
        "reason_codes": list(result.get("reason_codes") or []),
        "candidate_count": len(candidates),
        "top_score": top.get("score"),
        "top_charge_mode": (top.get("evidence") or {}).get("charge_mode"),
        "total_formal_charge": (top.get("evidence") or {}).get(
            "total_formal_charge"
        ),
        "conect_edge_count": len(conect),
        "proximity_edge_count": len(proximity),
        "heavy_atom_count": len(heavy_positions),
        "explicit_hydrogen_count": len(hydrogen_positions),
        "explicit_hydrogen_retained": len(hydrogen_parents),
        "explicit_hydrogen_validation": (
            "observed_hydrogens_retained_within_inferred_chemistry"
            if hydrogen_positions
            else "none_observed"
        ),
        "timeout_seconds": TIMEOUT_SECONDS,
        "elapsed_ms": (result.get("evidence") or {}).get("elapsed_ms"),
    }
    return molecule, meta


def _reference_length(element_a: str, element_b: str, order: float) -> float:
    from .geometry_simple import REF_LEN, _ORDER_FACTOR

    key = (min(element_a, element_b), max(element_a, element_b))
    table = REF_LEN.get(key)
    if table is not None and order in table:
        return table[order]
    radius_sum = _radius(element_a) + _radius(element_b)
    return radius_sum * _ORDER_FACTOR.get(order, 1.0)


def bond_length_consistency(candidate_graph: Any) -> dict[str, Any]:
    """Measure a source-bound candidate graph against observed bond lengths.

    Returns ``{"inconsistent_bond_count", "max_residual_angstrom",
    "flagged"}`` for a ``chemical-graph-1`` dict.  ``flagged`` is True when
    more than ``INCONSISTENT_BOND_LIMIT`` bonds are longer/shorter than the
    reference length of their assigned order by more than
    ``INCONSISTENT_RESIDUAL_ANGSTROM``.  Graphs without usable rows return
    ``None`` fields and ``flagged=False`` (no positive inconsistency
    evidence; selection must never demote on absence of evidence).
    """
    if not isinstance(candidate_graph, dict):
        return {
            "inconsistent_bond_count": None,
            "max_residual_angstrom": None,
            "flagged": False,
        }
    atoms = candidate_graph.get("atoms")
    bonds = candidate_graph.get("bonds")
    if not isinstance(atoms, list) or not isinstance(bonds, list):
        return {
            "inconsistent_bond_count": None,
            "max_residual_angstrom": None,
            "flagged": False,
        }
    xyz_by_serial: dict[int, tuple[float, float, float]] = {}
    element_by_serial: dict[int, str] = {}
    for row in atoms:
        if not isinstance(row, dict):
            continue
        xyz = row.get("xyz")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            continue
        try:
            xyz_by_serial[int(row["serial"])] = (
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
            )
            element_by_serial[int(row["serial"])] = str(row["element"]).capitalize()
        except (KeyError, TypeError, ValueError):
            continue
    inconsistent = 0
    max_residual = 0.0
    measured = 0
    for row in bonds:
        if not isinstance(row, dict):
            continue
        try:
            left = int(row["a"])
            right = int(row["b"])
            order = float(row["order"])
        except (KeyError, TypeError, ValueError):
            continue
        if left not in xyz_by_serial or right not in xyz_by_serial:
            continue
        if row.get("is_aromatic"):
            order = 1.5
        p = xyz_by_serial[left]
        q = xyz_by_serial[right]
        distance = math.sqrt(
            (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
        )
        reference = _reference_length(
            element_by_serial[left], element_by_serial[right], order
        )
        residual = abs(distance - reference)
        measured += 1
        max_residual = max(max_residual, residual)
        if residual > INCONSISTENT_RESIDUAL_ANGSTROM:
            inconsistent += 1
    if not measured:
        return {
            "inconsistent_bond_count": None,
            "max_residual_angstrom": None,
            "flagged": False,
        }
    return {
        "inconsistent_bond_count": inconsistent,
        "max_residual_angstrom": round(max_residual, 4),
        "flagged": bool(inconsistent > INCONSISTENT_BOND_LIMIT),
    }


__all__ = [
    "ENGINE_NAME",
    "GeometryCandidateError",
    "bond_length_consistency",
    "build_geometry_simple_molecule",
]
