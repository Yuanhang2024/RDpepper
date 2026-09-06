"""Fail-closed, entity-local inference of previously unknown peptide monomers.

The inferer intentionally covers only chemistry uniquely recoverable from one
PDB residue. Explicit intra-residue CONECT edges are mandatory evidence when
present; short covalent distances create connectivity candidates, not silent
repairs. A candidate is released only when connectivity, bond order and 3D
stereochemistry collapse to one neutral free-monomer graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Geometry import Point3D

from .cyclization import _COVALENT_RADII
from .cyclization import detect_cyclization
from .pdb_parser import get_pdb_atoms, get_res_seq, read_conect


_BACKBONE_NAMES = ("N", "CA", "C", "O")
_MAX_AMBIGUOUS_EDGES = 12
_MAX_BOND_ORDER_ASSIGNMENTS = 256
_MAX_CANDIDATES = 64
_MAX_SINGLE_BOND_DEGREE = {
    "B": 4, "C": 4, "N": 4, "O": 2, "P": 6, "S": 6,
    "F": 1, "CL": 1, "BR": 1, "I": 1, "SE": 6,
}


@dataclass
class LocalMonomerInferenceResult:
    status: str
    pdb_resname: str
    residue_key: tuple
    candidate_smiles: str | None = None
    candidate_r3_mapped_smiles: str | None = None
    r3_port: dict[str, Any] | None = None
    candidate_graph_count: int = 0
    reason_codes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def unique(self) -> bool:
        return self.status == "unique" and bool(self.candidate_smiles)

    def quarantine_row(self, *, record_id: str | None = None) -> dict[str, str]:
        input_sha = str(self.evidence.get("input_sha256", ""))
        stable_id = record_id or hashlib.sha256(
            f"{input_sha}|{self.residue_key}|{','.join(self.reason_codes)}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "record_id": stable_id,
            "pdb_resname": self.pdb_resname,
            "status": "rejected" if self.status == "rejected" else "not_supported",
            "reason_codes": ";".join(self.reason_codes),
            "input_sha256": input_sha,
            "candidate_graph_count": str(self.candidate_graph_count),
            "details_json": json.dumps(self.evidence, sort_keys=True, separators=(",", ":")),
        }


@dataclass
class PDBMonomerBootstrapResult:
    status: str
    derived_rows: list[dict] = field(default_factory=list)
    manifest_entries: list[dict] = field(default_factory=list)
    pdb_aliases: list[dict] = field(default_factory=list)
    quarantine_rows: list[dict] = field(default_factory=list)
    inference_results: list[LocalMonomerInferenceResult] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.status in {"no_unknown_monomers", "ready"}


def _distance(left: dict, right: dict) -> float:
    return math.dist(left["xyz"], right["xyz"])


def _pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _edge_ratio(left: dict, right: dict) -> float | None:
    first = _COVALENT_RADII.get(str(left["elem"]).upper())
    second = _COVALENT_RADII.get(str(right["elem"]).upper())
    if first is None or second is None:
        return None
    return _distance(left, right) / (first + second)


def _explicit_intra_residue_edges(
    pdb_path: str | Path, serials: set[int]
) -> set[tuple[int, int]]:
    raw = read_conect(str(pdb_path))
    return {
        _pair(source, target)
        for source, targets in raw.items()
        for target in targets
        if source in serials and target in serials and source != target
    }


def _connectivity_candidates(
    atoms: list[dict], explicit_edges: set[tuple[int, int]]
) -> tuple[list[set[tuple[int, int]]], dict[str, Any], list[str]]:
    by_serial = {atom["num"]: atom for atom in atoms}
    strong = set(explicit_edges)
    ambiguous = []
    conflicts = []
    for edge in explicit_edges:
        ratio = _edge_ratio(by_serial[edge[0]], by_serial[edge[1]])
        if ratio is None or ratio > 1.30 or ratio < 0.45:
            conflicts.append({"serials": list(edge), "covalent_radius_ratio": ratio})
    if conflicts:
        return [], {"explicit_geometry_conflicts": conflicts}, [
            "EXPLICIT_GEOMETRY_CONFLICT"
        ]
    for left_index in range(len(atoms)):
        for right_index in range(left_index + 1, len(atoms)):
            left, right = atoms[left_index], atoms[right_index]
            edge = _pair(left["num"], right["num"])
            if edge in explicit_edges:
                continue
            ratio = _edge_ratio(left, right)
            if ratio is None or ratio < 0.45 or ratio > 1.30:
                continue
            if ratio <= 1.12:
                strong.add(edge)
            else:
                ambiguous.append(edge)
    if len(ambiguous) > _MAX_AMBIGUOUS_EDGES:
        return [], {
            "strong_edges": sorted(map(list, strong)),
            "ambiguous_edges": sorted(map(list, ambiguous)),
        }, ["CONNECTIVITY_CANDIDATE_LIMIT_EXCEEDED"]

    candidates = []
    for choices in product((False, True), repeat=len(ambiguous)):
        edges = set(strong)
        edges.update(edge for edge, include in zip(ambiguous, choices) if include)
        degree = {serial: 0 for serial in by_serial}
        for left, right in edges:
            degree[left] += 1
            degree[right] += 1
        if any(
            degree[serial] > _MAX_SINGLE_BOND_DEGREE.get(
                str(by_serial[serial]["elem"]).upper(), 4
            )
            for serial in degree
        ):
            continue
        if not _connected(set(by_serial), edges):
            continue
        candidates.append(edges)
    ledger = {
        "explicit_edges": sorted(map(list, explicit_edges)),
        "strong_geometry_edges": sorted(map(list, strong - explicit_edges)),
        "ambiguous_geometry_edges": sorted(map(list, ambiguous)),
        "connectivity_candidate_count": len(candidates),
    }
    if not candidates:
        return [], ledger, ["NO_CONNECTED_VALENCE_COMPATIBLE_GRAPH"]
    return candidates, ledger, []


def _connected(serials: set[int], edges: set[tuple[int, int]]) -> bool:
    if not serials:
        return False
    neighbors = {serial: set() for serial in serials}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    seen = set()
    stack = [next(iter(serials))]
    while stack:
        serial = stack.pop()
        if serial in seen:
            continue
        seen.add(serial)
        stack.extend(neighbors[serial] - seen)
    return seen == serials


def _unique_named_atoms(atoms: list[dict]) -> tuple[dict[str, dict], list[str]]:
    by_name: dict[str, list[dict]] = {}
    for atom in atoms:
        by_name.setdefault(str(atom["name"]).strip().upper(), []).append(atom)
    missing = [name for name in _BACKBONE_NAMES if len(by_name.get(name, [])) != 1]
    if missing:
        return {}, ["INCOMPLETE_OR_AMBIGUOUS_PEPTIDE_BACKBONE"]
    return {name: rows[0] for name, rows in by_name.items() if len(rows) == 1}, []


def _explicit_r3_port(
    pdb_path: str | Path,
    chain_id: str,
    residues: list[dict],
    position: int,
    atoms: list[dict],
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Resolve one atom-level side-chain port from explicit topology only."""
    topology = detect_cyclization(
        str(pdb_path), chain_id, allow_geometric_inference=False
    )
    local_by_name: dict[str, list[dict]] = {}
    for atom in atoms:
        local_by_name.setdefault(str(atom["name"]).strip().upper(), []).append(atom)
    endpoints = []
    unresolved = []
    for bond in topology.bonds:
        for first in (True, False):
            endpoint_position = int(bond.pos1 if first else bond.pos2)
            rgroup = str(bond.rgroup1 if first else bond.rgroup2).upper()
            if endpoint_position != position or rgroup != "R3":
                continue
            atom_name = str(bond.atom1 if first else bond.atom2).strip().upper()
            matches = local_by_name.get(atom_name, [])
            if len(matches) != 1:
                unresolved.append(atom_name)
                continue
            partner_position = int(bond.pos2 if first else bond.pos1)
            partner_name = str(bond.atom2 if first else bond.atom1).strip().upper()
            partner_serial = None
            if 1 <= partner_position <= len(residues):
                partner_atoms = get_pdb_atoms(
                    str(pdb_path), tuple(residues[partner_position - 1]["key"]), chain_id
                )
                partner_matches = [
                    atom for atom in partner_atoms
                    if str(atom["name"]).strip().upper() == partner_name
                ]
                if len(partner_matches) == 1:
                    partner_serial = int(partner_matches[0]["num"])
            if partner_serial is None:
                unresolved.append(f"partner:{partner_position}:{partner_name}")
                continue
            local = matches[0]
            endpoints.append({
                "pdb_serial": int(local["num"]),
                "atom_name": atom_name,
                "atom_element": str(local["elem"]).strip().upper(),
                "evidence_source": str(bond.evidence_source),
                "bond_type": str(bond.bond_type),
                "partner_residue_position": partner_position,
                "partner_atom_name": partner_name,
                "partner_pdb_serial": partner_serial,
                "bond_order": 1,
            })
    if unresolved:
        return None, "not_supported", ["R3_ENDPOINT_NOT_UNIQUELY_RESOLVED"]
    if not endpoints:
        geometry_topology = detect_cyclization(
            str(pdb_path), chain_id, allow_geometric_inference=True
        )
        geometry_r3 = any(
            bond.evidence_source == "geometry"
            and (
                (int(bond.pos1) == position and str(bond.rgroup1).upper() == "R3")
                or (int(bond.pos2) == position and str(bond.rgroup2).upper() == "R3")
            )
            for bond in geometry_topology.bonds
        )
        if geometry_r3:
            return None, "not_supported", ["GEOMETRY_ONLY_R3_NOT_AUTHORIZED"]
        return None, None, []
    unique = {
        (
            row["pdb_serial"], row["partner_residue_position"],
            row["partner_atom_name"], row["partner_pdb_serial"],
        ): row
        for row in endpoints
    }
    by_local = {}
    for key, row in unique.items():
        by_local.setdefault(row["pdb_serial"], []).append((key, row))
    if any(len(rows) > 1 for rows in by_local.values()):
        return None, "rejected", ["R3_PORT_REUSED_BY_MULTIPLE_PARTNERS"]
    if len(by_local) > 1:
        unsupported = {
            "representation": "multiple_independent_r3_ports",
            "observed_ports": [
                dict(rows[0][1])
                for _serial, rows in sorted(by_local.items())
            ],
        }
        return unsupported, "not_supported", [
            "MULTIPLE_R3_PORTS_NOT_REPRESENTABLE"
        ]
    port = next(iter(unique.values()))
    element = port["atom_element"]
    if element == "C" and port["bond_type"] in {"isopeptide", "ester"}:
        port["cap"] = "OH"
    elif element in {"C", "N", "O", "S", "SE"}:
        port["cap"] = "H"
    else:
        return None, "not_supported", ["R3_LEAVING_GROUP_NOT_UNIQUELY_SUPPORTED"]
    return port, None, []


def _bond_order_options(
    first: dict,
    second: dict,
    *,
    backbone_carbonyl: tuple[int, int],
    terminal_hydroxyl: tuple[int, int] | None = None,
) -> tuple[Chem.BondType, ...]:
    edge = _pair(first["num"], second["num"])
    if edge == backbone_carbonyl:
        return (Chem.BondType.DOUBLE,)
    if edge == terminal_hydroxyl:
        return (Chem.BondType.SINGLE,)
    elements = frozenset((str(first["elem"]).upper(), str(second["elem"]).upper()))
    distance = _distance(first, second)
    if elements == frozenset(("C", "N")):
        if distance <= 1.20:
            return (Chem.BondType.TRIPLE,)
        if distance <= 1.29:
            return (Chem.BondType.DOUBLE,)
        if distance <= 1.42:
            return (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
    elif elements == frozenset(("C", "C")):
        if distance <= 1.24:
            return (Chem.BondType.TRIPLE,)
        if distance <= 1.38:
            return (Chem.BondType.DOUBLE,)
        if distance <= 1.46:
            return (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
    elif elements == frozenset(("C", "O")):
        oxygen = first if str(first["elem"]).upper() == "O" else second
        if str(oxygen["name"]).upper() != "O":
            if distance <= 1.30:
                return (Chem.BondType.DOUBLE,)
            if distance <= 1.38:
                return (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
    return (Chem.BondType.SINGLE,)


def _added_hydroxyl_position(carbon: dict, oxygen: dict) -> tuple[float, float, float]:
    vector = tuple(
        carbon["xyz"][axis] - oxygen["xyz"][axis] for axis in range(3)
    )
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(
        carbon["xyz"][axis] + 1.34 * vector[axis] / length for axis in range(3)
    )


def _candidate_molecules(
    atoms: list[dict],
    edges: set[tuple[int, int]],
    named: dict[str, dict],
    r3_port: dict[str, Any] | None = None,
) -> tuple[list[Chem.Mol], dict[str, Any], list[str]]:
    serial_to_index = {atom["num"]: index for index, atom in enumerate(atoms)}
    by_serial = {atom["num"]: atom for atom in atoms}
    required = {
        _pair(named["N"]["num"], named["CA"]["num"]),
        _pair(named["CA"]["num"], named["C"]["num"]),
        _pair(named["C"]["num"], named["O"]["num"]),
    }
    if not required.issubset(edges):
        return [], {"missing_backbone_edges": sorted(map(list, required - edges))}, [
            "PEPTIDE_BACKBONE_CONNECTIVITY_MISSING"
        ]
    backbone_carbonyl = _pair(named["C"]["num"], named["O"]["num"])
    observed_oxt = [
        atom for atom in atoms
        if str(atom["name"]).strip().upper() == "OXT"
        and str(atom["elem"]).strip().upper() == "O"
    ]
    if len(observed_oxt) > 1:
        return [], {"observed_terminal_oxt_count": len(observed_oxt)}, [
            "AMBIGUOUS_TERMINAL_OXT"
        ]
    terminal_hydroxyl = (
        _pair(named["C"]["num"], observed_oxt[0]["num"])
        if observed_oxt else None
    )
    if terminal_hydroxyl is not None and terminal_hydroxyl not in edges:
        return [], {"observed_terminal_oxt_edge": list(terminal_hydroxyl)}, [
            "TERMINAL_OXT_NOT_BOUND_TO_BACKBONE_CARBONYL"
        ]
    ordered_edges = sorted(edges)
    options = [
        _bond_order_options(
            by_serial[left], by_serial[right],
            backbone_carbonyl=backbone_carbonyl,
            terminal_hydroxyl=terminal_hydroxyl,
        )
        for left, right in ordered_edges
    ]
    assignment_count = math.prod(len(values) for values in options)
    if assignment_count > _MAX_BOND_ORDER_ASSIGNMENTS:
        return [], {"bond_order_assignment_count": assignment_count}, [
            "BOND_ORDER_CANDIDATE_LIMIT_EXCEEDED"
        ]

    molecules = []
    failures = []
    for assignment in product(*options):
        editable = Chem.RWMol()
        add_terminal_hydroxyl = terminal_hydroxyl is None
        add_r3_hydroxyl = bool(r3_port and r3_port.get("cap") == "OH")
        conformer = Chem.Conformer(
            len(atoms) + int(add_terminal_hydroxyl) + int(add_r3_hydroxyl)
        )
        for index, atom in enumerate(atoms):
            editable.AddAtom(Chem.Atom(str(atom["elem"]).capitalize()))
            conformer.SetAtomPosition(index, Point3D(*atom["xyz"]))
        hydroxyl_index = None
        if add_terminal_hydroxyl:
            hydroxyl_index = editable.AddAtom(Chem.Atom("O"))
            hydroxyl_position = _added_hydroxyl_position(named["C"], named["O"])
            conformer.SetAtomPosition(hydroxyl_index, Point3D(*hydroxyl_position))
        for (left, right), bond_type in zip(ordered_edges, assignment):
            editable.AddBond(
                serial_to_index[left], serial_to_index[right], bond_type
            )
        if hydroxyl_index is not None:
            editable.AddBond(
                serial_to_index[named["C"]["num"]],
                hydroxyl_index,
                Chem.BondType.SINGLE,
            )
        if add_r3_hydroxyl:
            anchor_serial = int(r3_port["pdb_serial"])
            anchor = by_serial[anchor_serial]
            oxygen_neighbors = [
                by_serial[other]
                for edge in ordered_edges if anchor_serial in edge
                for other in edge if other != anchor_serial
                if str(by_serial[other]["elem"]).upper() == "O"
                and assignment[ordered_edges.index(edge)] == Chem.BondType.DOUBLE
            ]
            if len(oxygen_neighbors) != 1:
                continue
            r3_oxygen_index = editable.AddAtom(Chem.Atom("O"))
            r3_position = _added_hydroxyl_position(anchor, oxygen_neighbors[0])
            conformer.SetAtomPosition(r3_oxygen_index, Point3D(*r3_position))
            editable.AddBond(
                serial_to_index[anchor_serial],
                r3_oxygen_index,
                Chem.BondType.SINGLE,
            )
        molecule = editable.GetMol()
        molecule.AddConformer(conformer)
        try:
            molecule.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(molecule)
            Chem.AssignStereochemistryFrom3D(molecule)
            Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
            if Chem.GetFormalCharge(molecule) != 0:
                continue
            molecules.append(molecule)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    return molecules, {
        "bond_order_assignment_count": assignment_count,
        "invalid_assignment_count": len(failures),
        "terminal_hydroxyl_added": add_terminal_hydroxyl,
        "terminal_hydroxyl_source": (
            "synthetic_free_monomer_cap"
            if add_terminal_hydroxyl else "observed_OXT"
        ),
        "r3_hydroxyl_added": bool(r3_port and r3_port.get("cap") == "OH"),
    }, [] if molecules else ["NO_SANITIZABLE_NEUTRAL_MONOMER_GRAPH"]


def _ionization_ambiguity(molecule: Chem.Mol, backbone_indices: set[int]) -> list[int]:
    ambiguous = []
    for atom in molecule.GetAtoms():
        if atom.GetIdx() in backbone_indices:
            continue
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 0:
            acylated = any(
                neighbor.GetSymbol() == "C"
                and any(
                    bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetOtherAtom(neighbor).GetSymbol() in {"O", "S"}
                    for bond in neighbor.GetBonds()
                )
                for neighbor in atom.GetNeighbors()
            )
            if (
                atom.GetTotalDegree() <= 3
                and not atom.GetIsAromatic()
                and not acylated
                and all(
                    bond.GetBondType() == Chem.BondType.SINGLE
                    for bond in atom.GetBonds()
                )
            ):
                ambiguous.append(atom.GetIdx())
    return ambiguous


def _stereo_ambiguity(molecule: Chem.Mol) -> tuple[list[int], list[int]]:
    tetrahedral = [
        index for index, assignment in Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
        if assignment == "?"
    ]
    unspecified_bonds = []
    try:
        for info in Chem.FindPotentialStereo(molecule):
            if "Unspecified" in str(info.specified):
                unspecified_bonds.append(int(info.centeredOn))
    except Exception:
        pass
    return tetrahedral, unspecified_bonds


def infer_residue_monomer(
    pdb_path: str | Path,
    chain_id: str,
    residue_key: tuple,
) -> LocalMonomerInferenceResult:
    """Infer one residue as a neutral free monomer or quarantine it."""
    path = Path(pdb_path)
    pdb_resname = str(residue_key[0]).strip().upper()
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    atoms = get_pdb_atoms(str(path), residue_key, chain_id)
    evidence: dict[str, Any] = {
        "inference_version": "local-monomer-inference-1",
        "resolution_mode": "locally_inferred",
        "input_sha256": input_hash,
        "chain_id": chain_id,
        "residue_key": list(residue_key),
        "observed_heavy_atom_count": len(atoms),
        "observed_serials": [atom["num"] for atom in atoms],
    }
    named, reasons = _unique_named_atoms(atoms)
    if reasons:
        return LocalMonomerInferenceResult(
            "quarantined", pdb_resname, residue_key,
            reason_codes=reasons, evidence=evidence,
        )
    residues = get_res_seq(str(path), chain_id, include_het=True)
    positions = [
        index + 1 for index, residue in enumerate(residues)
        if tuple(residue["key"]) == tuple(residue_key)
    ]
    if len(positions) != 1:
        return LocalMonomerInferenceResult(
            "quarantined", pdb_resname, residue_key,
            reason_codes=["RESIDUE_POSITION_NOT_UNIQUE"], evidence=evidence,
        )
    position = positions[0]
    r3_port, r3_failure_status, r3_reasons = _explicit_r3_port(
        path, chain_id, residues, position, atoms
    )
    evidence["ports"] = {
        "R1": {"pdb_serial": named["N"]["num"], "atom_name": "N", "default": "H"},
        "R2": {"pdb_serial": named["C"]["num"], "atom_name": "C", "default": "OH"},
        "R3": r3_port,
    }
    if r3_reasons:
        return LocalMonomerInferenceResult(
            r3_failure_status or "quarantined", pdb_resname, residue_key,
            reason_codes=r3_reasons,
            evidence=evidence,
        )
    explicit = _explicit_intra_residue_edges(
        path, {atom["num"] for atom in atoms}
    )
    connectivity, connectivity_evidence, reasons = _connectivity_candidates(
        atoms, explicit
    )
    evidence["connectivity"] = connectivity_evidence
    if reasons:
        return LocalMonomerInferenceResult(
            "quarantined", pdb_resname, residue_key,
            candidate_graph_count=len(connectivity),
            reason_codes=reasons, evidence=evidence,
        )

    canonical: dict[str, Chem.Mol] = {}
    chemistry_ledgers = []
    chemistry_reasons = []
    uncertainty_reasons = set()
    uncertain_candidate_count = 0
    for edges in connectivity:
        molecules, ledger, candidate_reasons = _candidate_molecules(
            atoms, edges, named, r3_port
        )
        chemistry_ledgers.append(ledger)
        chemistry_reasons.extend(candidate_reasons)
        for molecule in molecules:
            backbone_indices = {
                next(
                    atom.GetIdx() for atom in molecule.GetAtoms()
                    if atom.GetIdx() < len(atoms)
                    and atoms[atom.GetIdx()]["num"] == named[name]["num"]
                )
                for name in ("N", "CA", "C", "O")
            }
            ionizable = _ionization_ambiguity(molecule, backbone_indices)
            if ionizable:
                chemistry_reasons.append("IONIZATION_STATE_UNRESOLVED")
                uncertainty_reasons.add("IONIZATION_STATE_UNRESOLVED")
                uncertain_candidate_count += 1
                continue
            unresolved_atoms, unresolved_bonds = _stereo_ambiguity(molecule)
            if unresolved_atoms or unresolved_bonds:
                chemistry_reasons.append("UNRESOLVED_STEREOCHEMISTRY")
                uncertainty_reasons.add("UNRESOLVED_STEREOCHEMISTRY")
                uncertain_candidate_count += 1
                continue
            smiles = Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            )
            canonical.setdefault(smiles, molecule)
            if len(canonical) > _MAX_CANDIDATES:
                break
    evidence["chemistry_candidates"] = chemistry_ledgers
    evidence["candidate_smiles"] = sorted(canonical)
    evidence["candidate_graph_count"] = len(canonical) + uncertain_candidate_count
    evidence["uncertain_candidate_count"] = uncertain_candidate_count
    evidence["uncertainty_reason_codes"] = sorted(uncertainty_reasons)
    evidence["atom_serial_to_candidate_index"] = {
        str(atom["num"]): index for index, atom in enumerate(atoms)
    }
    if uncertainty_reasons:
        return LocalMonomerInferenceResult(
            "quarantined", pdb_resname, residue_key,
            candidate_graph_count=len(canonical) + uncertain_candidate_count,
            reason_codes=sorted(uncertainty_reasons), evidence=evidence,
        )
    if len(canonical) != 1:
        if len(canonical) > _MAX_CANDIDATES:
            reasons = ["CHEMICAL_GRAPH_CANDIDATE_LIMIT_EXCEEDED"]
        elif len(canonical) > 1:
            reasons = ["AMBIGUOUS_CONSTITUTIONAL_OR_STEREOCHEMICAL_GRAPH"]
        else:
            reasons = sorted(set(chemistry_reasons)) or ["NO_UNIQUE_MONOMER_GRAPH"]
        return LocalMonomerInferenceResult(
            "quarantined", pdb_resname, residue_key,
            candidate_graph_count=len(canonical),
            reason_codes=reasons, evidence=evidence,
        )
    smiles = next(iter(canonical))
    selected = canonical[smiles]
    r3_mapped_smiles = None
    if r3_port:
        r3_index = next(
            index for index, atom in enumerate(atoms)
            if int(atom["num"]) == int(r3_port["pdb_serial"])
        )
        marked = Chem.Mol(selected)
        marked.GetAtomWithIdx(r3_index).SetAtomMapNum(9003)
        r3_mapped_smiles = Chem.MolToSmiles(
            marked, canonical=True, isomericSmiles=True
        )
        r3_port = dict(r3_port)
        r3_port["candidate_atom_index"] = r3_index
        r3_port["mapped_atom_number"] = 9003
        evidence["ports"]["R3"] = dict(r3_port)
    return LocalMonomerInferenceResult(
        "unique", pdb_resname, residue_key,
        candidate_smiles=smiles,
        candidate_r3_mapped_smiles=r3_mapped_smiles,
        r3_port=r3_port,
        candidate_graph_count=1,
        evidence=evidence,
    )


def _library_first_residue_match(
    pdb_path: str | Path,
    chain_id: str,
    residue_key: tuple,
    *,
    source_identity_template: Any | None = None,
) -> tuple[LocalMonomerInferenceResult | None, dict[str, Any]]:
    """Resolve an unknown PDB code from base Unified evidence before inference."""
    from ..paths.residue_template_factory import (
        find_unified_residue_matches,
        mapped_free_smiles_for_r3,
    )

    path = Path(pdb_path)
    pdb_resname = str(residue_key[0]).strip().upper()
    atoms = get_pdb_atoms(str(path), residue_key, chain_id)
    attempt: dict[str, Any] = {
        "resolution_mode": "unified_library_match",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chain_id": chain_id,
        "residue_key": list(residue_key),
        "observed_heavy_atom_count": len(atoms),
        "released": False,
    }
    if source_identity_template is not None:
        attempt["source_identity_constraint"] = {
            "declared_name": str(source_identity_template.pdb_resname),
            "declared_symbol": str(source_identity_template.symbol),
            "declared_template_heavy_atom_count": int(
                source_identity_template.mol.GetNumAtoms()
            ),
        }
    named, reasons = _unique_named_atoms(atoms)
    if reasons:
        attempt["fallback_reason_codes"] = reasons
        return None, attempt
    residues = get_res_seq(str(path), chain_id, include_het=True)
    positions = [
        index + 1 for index, residue in enumerate(residues)
        if tuple(residue["key"]) == tuple(residue_key)
    ]
    if len(positions) != 1:
        attempt["fallback_reason_codes"] = ["RESIDUE_POSITION_NOT_UNIQUE"]
        return None, attempt
    r3_port, _r3_status, r3_reasons = _explicit_r3_port(
        path, chain_id, residues, positions[0], atoms
    )
    if r3_reasons:
        attempt["fallback_reason_codes"] = list(r3_reasons)
        return None, attempt
    explicit = _explicit_intra_residue_edges(
        path, {int(atom["num"]) for atom in atoms}
    )
    connectivity, connectivity_evidence, connectivity_reasons = (
        _connectivity_candidates(atoms, explicit)
    )
    attempt["connectivity"] = connectivity_evidence
    if connectivity_reasons:
        attempt["fallback_reason_codes"] = list(connectivity_reasons)
        return None, attempt
    external_serials = (
        [int(r3_port["pdb_serial"])] if isinstance(r3_port, dict) else []
    )
    matches, search = find_unified_residue_matches(
        pdb_resname,
        atoms,
        explicit_edges=explicit,
        external_serials=external_serials,
        r3_cap=(str(r3_port.get("cap")) if isinstance(r3_port, dict) else None),
        observed_edges=(connectivity[0] if len(connectivity) == 1 else None),
        source_identity_symbol=(
            str(source_identity_template.symbol)
            if source_identity_template is not None else None
        ),
    )
    attempt["audited_connectivity_used_for_mapping"] = len(connectivity) == 1
    groups: dict[tuple[str, str], list[dict]] = {}
    for match in matches:
        groups.setdefault(tuple(match["identity"]), []).append(match)
    attempt["search"] = search
    attempt["strict_identity_count"] = len(groups)
    attempt["strict_identities"] = [list(identity) for identity in sorted(groups)]
    if len(groups) != 1:
        attempt["fallback_reason_codes"] = [
            "NO_UNIQUE_BASE_UNIFIED_STRUCTURE_MATCH"
        ]
        return None, attempt
    equivalent = next(iter(groups.values()))

    def rank(match: dict) -> tuple[int, str]:
        value = str(match["evidence"].get("monomer_id", "")).strip()
        try:
            monomer_id = int(value)
        except ValueError:
            monomer_id = 2**63 - 1
        return monomer_id, str(match["template"].symbol)

    selected = min(equivalent, key=rank)
    template = selected["template"]
    mapped_r3 = None
    if isinstance(r3_port, dict):
        try:
            mapped_r3 = mapped_free_smiles_for_r3(template)
        except ValueError as exc:
            attempt["fallback_reason_codes"] = ["R3_FREE_GRAPH_MAPPING_NOT_UNIQUE"]
            attempt["r3_mapping_error"] = str(exc)
            return None, attempt
    attempt.update({
        "released": True,
        "selected_symbol": template.symbol,
        "equivalent_symbols": sorted(
            str(match["template"].symbol) for match in equivalent
        ),
        "selected_identity": list(selected["identity"]),
        "selected_match_evidence": selected["evidence"],
    })
    evidence = {
        "inference_version": "unified-library-match-1",
        "resolution_mode": "unified_library_match",
        "input_sha256": attempt["input_sha256"],
        "chain_id": chain_id,
        "residue_key": list(residue_key),
        "observed_heavy_atom_count": len(atoms),
        "observed_serials": [int(atom["num"]) for atom in atoms],
        "ports": {
            "R1": {"pdb_serial": named["N"]["num"], "atom_name": "N", "default": "H"},
            "R2": {"pdb_serial": named["C"]["num"], "atom_name": "C", "default": "OH"},
            "R3": r3_port,
        },
        "library_first_match": attempt,
    }
    return LocalMonomerInferenceResult(
        "unique",
        pdb_resname,
        residue_key,
        candidate_smiles=template.free_smiles,
        candidate_r3_mapped_smiles=mapped_r3,
        r3_port=r3_port,
        candidate_graph_count=1,
        evidence=evidence,
    ), attempt


def _embedded_component_residue_match(
    pdb_path: str | Path,
    chain_id: str,
    residue_key: tuple,
    component: dict,
) -> LocalMonomerInferenceResult:
    """Resolve one unknown residue from its input-embedded chemical dictionary."""
    from .mmcif_chem_comp import (
        EmbeddedChemCompError,
        resolve_embedded_peptide_component,
    )

    path = Path(pdb_path)
    pdb_resname = str(residue_key[0]).strip().upper()
    atoms = get_pdb_atoms(str(path), residue_key, chain_id)
    evidence: dict[str, Any] = {
        "inference_version": "embedded-mmcif-chem-comp-1",
        "resolution_mode": "embedded_mmcif_chem_comp",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chain_id": chain_id,
        "residue_key": list(residue_key),
        "observed_heavy_atom_count": len(atoms),
        "observed_serials": [int(atom["num"]) for atom in atoms],
        "component_snapshot_sha256": component.get("component_snapshot_sha256"),
        "component_source_input_sha256": component.get("source_input_sha256"),
    }
    residues = get_res_seq(str(path), chain_id, include_het=True)
    positions = [
        index + 1 for index, residue in enumerate(residues)
        if tuple(residue["key"]) == tuple(residue_key)
    ]
    if len(positions) != 1:
        return LocalMonomerInferenceResult(
            "rejected", pdb_resname, residue_key,
            reason_codes=["RESIDUE_POSITION_NOT_UNIQUE"], evidence=evidence,
        )
    r3_port, r3_status, r3_reasons = _explicit_r3_port(
        path, chain_id, residues, positions[0], atoms
    )
    named, reasons = _unique_named_atoms(atoms)
    evidence["ports"] = {
        "R1": (
            {"pdb_serial": named["N"]["num"], "atom_name": "N", "default": "H"}
            if not reasons else None
        ),
        "R2": (
            {"pdb_serial": named["C"]["num"], "atom_name": "C", "default": "OH"}
            if not reasons else None
        ),
        "R3": r3_port,
    }
    if r3_reasons:
        return LocalMonomerInferenceResult(
            r3_status or "rejected", pdb_resname, residue_key,
            reason_codes=r3_reasons, evidence=evidence,
        )
    if reasons:
        return LocalMonomerInferenceResult(
            "rejected", pdb_resname, residue_key,
            reason_codes=reasons, evidence=evidence,
        )
    explicit = _explicit_intra_residue_edges(
        path, {int(atom["num"]) for atom in atoms}
    )
    connectivity, connectivity_evidence, connectivity_reasons = (
        _connectivity_candidates(atoms, explicit)
    )
    evidence["connectivity"] = connectivity_evidence
    if connectivity_reasons or len(connectivity) != 1:
        return LocalMonomerInferenceResult(
            "quarantined", pdb_resname, residue_key,
            candidate_graph_count=len(connectivity),
            reason_codes=(
                list(connectivity_reasons)
                or ["CONNECTIVITY_NOT_UNIQUE_FOR_EMBEDDED_CHEM_COMP"]
            ),
            evidence=evidence,
        )
    try:
        resolution = resolve_embedded_peptide_component(
            component,
            atoms,
            connectivity[0],
            r3_atom_name=(
                str(r3_port.get("atom_name"))
                if isinstance(r3_port, dict) else None
            ),
            r3_cap=(
                str(r3_port.get("cap")) if isinstance(r3_port, dict) else None
            ),
        )
    except EmbeddedChemCompError as exc:
        evidence["embedded_chem_comp_error"] = str(exc)
        return LocalMonomerInferenceResult(
            "rejected" if exc.rejected else "quarantined",
            pdb_resname,
            residue_key,
            reason_codes=[exc.code],
            evidence=evidence,
        )
    evidence["embedded_chem_comp_resolution"] = resolution
    evidence["released"] = True
    return LocalMonomerInferenceResult(
        "unique",
        pdb_resname,
        residue_key,
        candidate_smiles=resolution["free_smiles"],
        candidate_r3_mapped_smiles=resolution["r3_mapped_smiles"],
        r3_port=r3_port,
        candidate_graph_count=1,
        evidence=evidence,
    )


def _source_identity_row_for_residue(
    source_identity_audit: dict[str, Any] | None,
    residue_key: tuple,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve one normalized coordinate residue to a source declaration."""
    if not isinstance(source_identity_audit, dict):
        return None, "absent"
    rows = source_identity_audit.get("rows")
    if not isinstance(rows, list):
        return None, str(source_identity_audit.get("status") or "absent")
    audit_status = str(source_identity_audit.get("status") or "absent")
    if audit_status != "unique":
        return None, audit_status
    try:
        residue_name = str(residue_key[0]).strip().upper()
        residue_number = int(residue_key[1])
        insertion_code = (
            str(residue_key[3]).strip() if len(residue_key) >= 4 else ""
        )
    except (TypeError, ValueError, IndexError):
        return None, "unmapped"
    candidates = []
    malformed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        mapping_state = str(
            row.get("mapping_state") or row.get("mapping_status") or ""
        )
        if mapping_state != "unique":
            continue
        raw_resseq = row.get("coordinate_resseq")
        if raw_resseq is None:
            continue
        try:
            coordinate_resseq = int(raw_resseq)
        except (TypeError, ValueError, OverflowError):
            malformed = True
            continue
        if (
            str(row.get("coordinate_residue_name") or "").strip().upper()
            == residue_name
            and coordinate_resseq == residue_number
            and str(row.get("coordinate_icode") or "").strip()
            == insertion_code
        ):
            candidates.append(dict(row))
    if len(candidates) == 1:
        return candidates[0], "unique"
    if len(candidates) > 1:
        return None, "ambiguous"
    if malformed:
        return None, "invalid"
    return None, "unmapped"


def _source_identity_constraint(
    source_identity_audit: dict[str, Any] | None,
    residue_key: tuple,
    observed_heavy_atom_count: int,
) -> tuple[dict[str, Any], Any | None]:
    """Materialize a declared source template for one unknown coordinate row."""
    from ..paths.residue_template_factory import get_residue_template

    row, mapping_status = _source_identity_row_for_residue(
        source_identity_audit, residue_key
    )
    constraint: dict[str, Any] = {
        "status": mapping_status,
        "mapping_state": mapping_status,
        "coordinate_residue_key": list(residue_key),
        "observed_heavy_atom_count": int(observed_heavy_atom_count),
    }
    if row is None:
        if mapping_status in {"invalid", "partial", "ambiguous"}:
            constraint.update({
                "status": "incomplete",
                "reason_code": "SOURCE_IDENTITY_MAPPING_UNRESOLVED",
                "reason": "source_identity_audit_not_uniquely_mapped",
            })
        return constraint, None
    declared_name = str(row.get("declared_name") or "").strip().upper()
    constraint.update({
        "status": "unique",
        "declared_name": declared_name,
        "source_sha256": row.get("source_sha256"),
        "sequence_position": row.get("sequence_position"),
        "subchain": row.get("subchain"),
        "entity_id": row.get("entity_id"),
    })
    try:
        template = get_residue_template(declared_name)
    except (TypeError, ValueError, RuntimeError) as exc:
        constraint.update({
            "status": "declared_template_unavailable",
            "declared_template_error": str(exc),
        })
        return constraint, None
    declared_count = int(template.mol.GetNumAtoms())
    constraint.update({
        "declared_template_symbol": template.symbol,
        "declared_template_heavy_atom_count": declared_count,
        "declared_free_graph_sha256": template.free_graph_sha256,
    })
    # One source-bound R3 leaving atom can be consumed by an explicit closure;
    # larger deficits are source-identity incompleteness, not an alias choice.
    if (
        observed_heavy_atom_count > declared_count
        or declared_count - observed_heavy_atom_count > 1
    ):
        constraint.update({
            "status": "incomplete",
            "reason_code": "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE",
        })
        return constraint, template
    constraint["status"] = (
        "constrained"
        if declared_count == observed_heavy_atom_count
        else "partial_allowed"
    )
    return constraint, template


def _inchikey_identity_relation(left: str | None, right: str | None) -> str:
    if not left or not right:
        return "identity_unavailable"
    if left == right:
        return "exact_full_inchikey"
    left_blocks = left.split("-")
    right_blocks = right.split("-")
    if len(left_blocks) != 3 or len(right_blocks) != 3:
        return "identity_unavailable"
    if left_blocks[0] != right_blocks[0]:
        return "constitutional_graph_conflict"
    if left_blocks[1] != right_blocks[1]:
        return "stereochemistry_or_isotope_conflict"
    return "protonation_or_charge_conflict"


def audit_known_monomers_against_embedded_components(
    pdb_path: str | Path,
    chain_id: str,
    embedded_chem_comp_templates: dict[str, dict] | None,
    *,
    coordinate_source_sha256: str | None,
) -> dict[str, Any]:
    """Audit, but never override, known Unified residues using embedded CCD."""
    from ..paths.residue_template_factory import (
        get_residue_template,
        resolve_symbol,
    )

    templates = embedded_chem_comp_templates or {}
    rows = []
    known_without_component = []
    for residue in get_res_seq(str(pdb_path), chain_id, include_het=True):
        pdb_resname = str(residue["name"]).strip().upper()
        try:
            symbol = resolve_symbol(pdb_resname)
        except ValueError:
            continue
        component = templates.get(pdb_resname)
        component_type = " ".join(
            str((component or {}).get("component_type") or "").upper().split()
        )
        if (
            component is None
            or not component.get("atom_rows")
            or not component.get("bond_rows")
            or component_type not in {
                "L-PEPTIDE LINKING",
                "D-PEPTIDE LINKING",
                "PEPTIDE LINKING",
            }
        ):
            known_without_component.append(list(residue["key"]))
            continue
        template = get_residue_template(pdb_resname)
        unified_molecule = Chem.MolFromSmiles(template.free_smiles)
        unified_key = (
            Chem.MolToInchiKey(unified_molecule)
            if unified_molecule is not None else None
        )
        result = _embedded_component_residue_match(
            pdb_path,
            chain_id,
            tuple(residue["key"]),
            component,
        )
        resolution = result.evidence.get("embedded_chem_comp_resolution") or {}
        embedded_key = resolution.get("full_inchikey")
        source_bound = bool(coordinate_source_sha256) and (
            component.get("source_input_sha256") == coordinate_source_sha256
            == result.evidence.get("component_source_input_sha256")
        )
        relation = _inchikey_identity_relation(unified_key, embedded_key)
        if not source_bound:
            status = "rejected"
            reason_code = "MMCIF_CHEM_COMP_SOURCE_BINDING_INVALID"
        elif result.status == "rejected":
            status = "rejected"
            reason_code = (
                result.reason_codes[0]
                if result.reason_codes else "MMCIF_CHEM_COMP_EVIDENCE_CONFLICT"
            )
        elif not result.unique or relation == "identity_unavailable":
            status = "not_supported"
            reason_code = (
                result.reason_codes[0]
                if result.reason_codes
                else "MMCIF_CHEM_COMP_IDENTITY_NOT_RESOLVABLE"
            )
        elif relation == "protonation_or_charge_conflict":
            status = "not_supported"
            reason_code = "LIBRARY_VS_EMBEDDED_PROTONATION_OR_CHARGE_CONFLICT"
        elif relation != "exact_full_inchikey":
            status = "rejected"
            reason_code = "LIBRARY_VS_EMBEDDED_COMPONENT_IDENTITY_CONFLICT"
        else:
            status = "pass"
            reason_code = None
        rows.append({
            "residue_key": list(residue["key"]),
            "pdb_resname": pdb_resname,
            "unified_symbol": symbol,
            "unified_source": template.source,
            "unified_free_graph_sha256": template.free_graph_sha256,
            "unified_full_inchikey": unified_key,
            "embedded_full_inchikey": embedded_key,
            "identity_relation": relation,
            "embedded_resolution_status": result.status,
            "embedded_resolution_reason_codes": list(result.reason_codes),
            "component_snapshot_sha256": component.get(
                "component_snapshot_sha256"
            ),
            "component_source_input_sha256": component.get(
                "source_input_sha256"
            ),
            "source_payload_bound": source_bound,
            "status": status,
            "reason_code": reason_code,
        })

    statuses = {row["status"] for row in rows}
    if "rejected" in statuses:
        status = "rejected"
    elif "not_supported" in statuses:
        status = "not_supported"
    elif rows:
        status = "pass"
    else:
        status = "not_required"
    return {
        "schema_version": "known-embedded-component-audit-1",
        "status": status,
        "override_performed": False,
        "coordinate_source_sha256": coordinate_source_sha256,
        "known_residue_count": len(rows) + len(known_without_component),
        "audited_residue_count": len(rows),
        "exact_full_inchikey_count": sum(
            row["identity_relation"] == "exact_full_inchikey" for row in rows
        ),
        "not_supported_count": sum(
            row["status"] == "not_supported" for row in rows
        ),
        "conflict_count": sum(row["status"] == "rejected" for row in rows),
        "known_residues_without_embedded_component": known_without_component,
        "rows": rows,
    }


def bootstrap_unknown_monomers(
    pdb_path: str | Path,
    chain_id: str,
    *,
    embedded_chem_comp_templates: dict[str, dict] | None = None,
    source_identity_audit: dict[str, Any] | None = None,
) -> PDBMonomerBootstrapResult:
    """Infer every Unified-unknown residue without persisting entity state."""
    from .derived_monomers import build_derived_batch
    from ..paths.residue_template_factory import resolve_symbol

    residues = get_res_seq(str(pdb_path), chain_id, include_het=True)
    unknown = []
    for residue in residues:
        try:
            resolve_symbol(residue["name"])
        except ValueError:
            unknown.append(residue)
    if not unknown:
        return PDBMonomerBootstrapResult(status="no_unknown_monomers")

    results = []
    for residue in unknown:
        key = tuple(residue["key"])
        atoms = get_pdb_atoms(str(pdb_path), key, chain_id)
        source_constraint, source_template = _source_identity_constraint(
            source_identity_audit,
            key,
            len(atoms),
        )
        component = (embedded_chem_comp_templates or {}).get(
            str(residue["name"]).strip().upper()
        )
        if source_constraint.get("status") == "incomplete":
            source_reason = str(
                source_constraint.get("reason_code")
                or "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"
            )
            result = LocalMonomerInferenceResult(
                "rejected",
                str(residue["name"]).strip().upper(),
                key,
                reason_codes=[source_reason],
                evidence={
                    "resolution_mode": "source_identity_constraint",
                    "input_sha256": hashlib.sha256(
                        Path(pdb_path).read_bytes()
                    ).hexdigest(),
                    "chain_id": chain_id,
                    "residue_key": list(key),
                    "observed_heavy_atom_count": len(atoms),
                    "source_identity_constraint": source_constraint,
                },
            )
            results.append(result)
            continue
        if source_constraint.get("status") == "declared_template_unavailable":
            result = LocalMonomerInferenceResult(
                "rejected",
                str(residue["name"]).strip().upper(),
                key,
                reason_codes=["SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"],
                evidence={
                    "resolution_mode": "source_identity_constraint",
                    "input_sha256": hashlib.sha256(
                        Path(pdb_path).read_bytes()
                    ).hexdigest(),
                    "chain_id": chain_id,
                    "residue_key": list(key),
                    "observed_heavy_atom_count": len(atoms),
                    "source_identity_constraint": source_constraint,
                },
            )
            results.append(result)
            continue
        if component is not None:
            result = _embedded_component_residue_match(
                pdb_path, chain_id, key, component
            )
            result.evidence["source_identity_constraint"] = source_constraint
            if result.unique and source_template is not None:
                resolution = result.evidence.get(
                    "embedded_chem_comp_resolution", {}
                )
                candidate = Chem.MolFromSmiles(str(resolution.get("free_smiles", "")))
                declared = Chem.MolFromSmiles(str(source_template.free_smiles))
                if (
                    candidate is None
                    or declared is None
                    or Chem.MolToInchiKey(candidate) != Chem.MolToInchiKey(declared)
                ):
                    result.status = "rejected"
                    result.reason_codes = ["SOURCE_IDENTITY_CONSTRAINT_CONFLICT"]
            results.append(result)
            continue
        result, match_attempt = _library_first_residue_match(
            pdb_path,
            chain_id,
            key,
            source_identity_template=(
                source_template
                if source_constraint.get("status") == "constrained"
                else None
            ),
        )
        if result is None:
            if source_template is not None and source_constraint.get("status") == "constrained":
                result = LocalMonomerInferenceResult(
                    "rejected",
                    str(residue["name"]).strip().upper(),
                    key,
                    reason_codes=["SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"],
                    evidence={
                        "resolution_mode": "source_identity_constraint",
                        "input_sha256": hashlib.sha256(
                            Path(pdb_path).read_bytes()
                        ).hexdigest(),
                        "chain_id": chain_id,
                        "residue_key": list(key),
                        "observed_heavy_atom_count": len(atoms),
                        "source_identity_constraint": source_constraint,
                        "library_first_match": match_attempt,
                    },
                )
                results.append(result)
                continue
            result = infer_residue_monomer(pdb_path, chain_id, key)
            result.evidence["library_first_match"] = match_attempt
        result.evidence["source_identity_constraint"] = source_constraint
        if (
            result.unique
            and source_template is not None
            and str(result.candidate_smiles or "")
        ):
            candidate = Chem.MolFromSmiles(str(result.candidate_smiles))
            declared = Chem.MolFromSmiles(str(source_template.free_smiles))
            if candidate is None or declared is None or Chem.MolToInchiKey(candidate) != Chem.MolToInchiKey(declared):
                result.status = "rejected"
                result.reason_codes = ["SOURCE_IDENTITY_CONSTRAINT_CONFLICT"]
        results.append(result)
    failed = [result for result in results if not result.unique]
    if failed:
        return PDBMonomerBootstrapResult(
            status=(
                "rejected" if any(result.status == "rejected" for result in failed)
                else "quarantined"
            ),
            quarantine_rows=[result.quarantine_row() for result in failed],
            inference_results=results,
        )

    graphs_by_code: dict[str, set[tuple[str, str]]] = {}
    for result in results:
        molecule = Chem.MolFromSmiles(result.candidate_smiles)
        key = Chem.MolToInchiKey(molecule) if molecule is not None else ""
        port_signature = json.dumps(
            {
                "mapped_smiles": result.candidate_r3_mapped_smiles,
                "cap": (result.r3_port or {}).get("cap"),
            },
            sort_keys=True,
        )
        graphs_by_code.setdefault(result.pdb_resname, set()).add(
            (key, port_signature)
        )
    conflicting_codes = sorted(
        code for code, graphs in graphs_by_code.items() if len(graphs) != 1
    )
    if conflicting_codes:
        quarantine = []
        for result in results:
            if result.pdb_resname not in conflicting_codes:
                continue
            result.status = "quarantined"
            result.reason_codes = ["PDB_RESNAME_MAPS_TO_MULTIPLE_LOCAL_GRAPHS"]
            quarantine.append(result.quarantine_row())
        return PDBMonomerBootstrapResult(
            status="quarantined",
            quarantine_rows=quarantine,
            inference_results=results,
        )

    inferred_candidates = [
        {
            "pdb_resname": result.pdb_resname,
            "smiles": result.candidate_smiles,
            "r3_mapped_smiles": result.candidate_r3_mapped_smiles,
            "r3_port": result.r3_port,
            "input_sha256": result.evidence["input_sha256"],
            "source_entity_id": f"{Path(pdb_path).stem}:{chain_id}",
            "component_snapshot_sha256": result.evidence.get(
                "component_snapshot_sha256"
            ),
            "component_source_input_sha256": result.evidence.get(
                "component_source_input_sha256"
            ),
            "resolution_mode": result.evidence.get("resolution_mode"),
        }
        for result in results
        if result.evidence.get("resolution_mode") != "unified_library_match"
    ]
    direct_results = [
        result for result in results
        if result.evidence.get("resolution_mode") == "unified_library_match"
    ]
    from .derived_monomers import PortSemanticMismatchError
    try:
        if inferred_candidates:
            rows, manifests, aliases = build_derived_batch(inferred_candidates)
        else:
            rows, manifests, aliases = [], [], []
    except PortSemanticMismatchError as exc:
        quarantine = []
        for result in results:
            result.status = "not_supported"
            result.reason_codes = ["R3_PORT_SEMANTICS_DO_NOT_MATCH_LIBRARY"]
            row = result.quarantine_row()
            row["details_json"] = json.dumps(
                {"error": str(exc)}, sort_keys=True
            )
            quarantine.append(row)
        return PDBMonomerBootstrapResult(
            status="quarantined",
            quarantine_rows=quarantine,
            inference_results=results,
        )
    except ValueError as exc:
        quarantine = []
        for result in results:
            result.status = "quarantined"
            result.reason_codes = ["DERIVED_MONOMER_MATERIALIZATION_FAILED"]
            row = result.quarantine_row()
            row["details_json"] = json.dumps(
                {"error": str(exc)}, sort_keys=True
            )
            quarantine.append(row)
        return PDBMonomerBootstrapResult(
            status="quarantined",
            quarantine_rows=quarantine,
            inference_results=results,
        )
    aliases.extend({
        "pdb_resname": entry["pdb_resname"],
        "target_symbol": entry["symbol"],
        "graph_sha256": entry["graph_sha256"],
        "full_inchikey": entry["full_inchikey"],
        "input_sha256": entry.get("input_sha256", ""),
        "source_entity_id": entry.get("source_entity_id", ""),
        "resolution_mode": entry.get("resolution_mode", ""),
    } for entry in manifests)
    for result in direct_results:
        molecule = Chem.MolFromSmiles(str(result.candidate_smiles))
        canonical = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        match = result.evidence["library_first_match"]
        aliases.append({
            "pdb_resname": result.pdb_resname,
            "target_symbol": str(match["selected_symbol"]),
            "graph_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "full_inchikey": Chem.MolToInchiKey(molecule),
            "input_sha256": str(result.evidence.get("input_sha256", "")),
            "source_entity_id": f"{Path(pdb_path).stem}:{chain_id}",
            "resolution_mode": "unified_library_match",
        })
    deduplicated = {}
    for alias in aliases:
        key = (str(alias["pdb_resname"]).upper(), str(alias["target_symbol"]))
        deduplicated[key] = dict(alias)
    return PDBMonomerBootstrapResult(
        status="ready",
        derived_rows=rows,
        manifest_entries=manifests,
        pdb_aliases=[deduplicated[key] for key in sorted(deduplicated)],
        inference_results=results,
    )
