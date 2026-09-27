"""Generated-coordinate owner for the V5 typed-artifact pipeline.

Legacy compatibility exporters retain their historical embedding paths; only
the V5 ``ChemicalGraphArtifact`` pipeline is governed by this materializer.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdMolAlign, rdMolTransforms
from rdkit.Geometry import Point3D

from ..core.artifacts import (
    ArtifactStatus,
    ArtifactType,
    ClaimBoundary,
    ConformerEnsembleArtifact,
    ConformerMember,
    CoordinateLevel,
    CoordinateOrigin,
    ENSEMBLE_SCHEMA_VERSION,
    FormatLevel,
    ValidatedMol2Artifact,
    artifact_payload_sha256,
    evidence_from_dict,
    inherit_evidence,
    make_artifact_id,
    validate_artifact_identity,
    validate_artifact_record,
    validate_inherited_evidence,
)
from ..docking.mol2_input import (
    sha256_path,
    write_validation_receipt,
)
from ..docking.template_library import find_coordinate_evidence
from ..core.monomer_resolution import needs_monomer_resolution_scope
from ..docking.torsion_prior import (
    build_query_keys,
    load_torsion_prior,
    prior_guidance_quartet,
    classify_macrocycle_topology,
)
from .conformer import mol_to_mol2


_BACKBONE = Chem.MolFromSmarts(
    "[N;X3,X4][C;X4][C;X3](=[O;X1])"
)
_STANDARD_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
V5_ETKDG_ATTEMPT_TIMEOUT_SECONDS = 30
PRIOR_APPLY_MAX_STD_DEG = 30.0
PRIOR_RELAX_TOLERANCE_DEG = 10.0
PRIOR_RELAX_MAX_ITERATIONS = 3
PRIOR_CONSTRAINT_FORCE_CONSTANT = 100.0


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _portable_template_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    portable = dict(value)
    matches = []
    for raw in value.get("matches", []):
        row = dict(raw)
        path_value = row.pop("template_pdb_path", None)
        if path_value:
            path = Path(str(path_value)).resolve()
            try:
                locator = path.relative_to(package_root).as_posix()
            except ValueError:
                locator = path.name
            row["template_pdb_locator"] = locator
        matches.append(row)
    portable["matches"] = matches
    return portable


def _single_conformer_copy(
    molecule: Chem.Mol, conformer_id: int = 0
) -> Chem.Mol:
    result = Chem.Mol(molecule)
    conformer = Chem.Conformer(molecule.GetConformer(conformer_id))
    result.RemoveAllConformers()
    result.AddConformer(conformer, assignId=True)
    return result


def _backbone_matches(molecule: Chem.Mol):
    return list(molecule.GetSubstructMatches(_BACKBONE))


def _annotate_residues(
    molecule: Chem.Mol,
    exact_document: Mapping[str, Any],
) -> dict[str, Any]:
    rows = sorted(
        exact_document["monomers"],
        key=lambda row: (
            str(row["chain_id"]),
            int(row["canonical_position"]),
        ),
    )
    matches = _backbone_matches(molecule)
    if len(matches) != len(rows):
        return {
            "status": "unavailable",
            "reason": "backbone/residue cardinality mismatch",
            "matched_residues": len(matches),
            "expected_residues": len(rows),
        }
    anchors = []
    atom_to_residue: dict[int, int] = {}
    names: dict[int, str] = {}
    for residue_index, (row, match) in enumerate(
        zip(rows, matches), 1
    ):
        n_atom, ca_atom, c_atom, o_atom = match
        anchors.append(ca_atom)
        for atom_index, name in (
            (n_atom, "N"),
            (ca_atom, "CA"),
            (c_atom, "C"),
            (o_atom, "O"),
        ):
            atom_to_residue[atom_index] = residue_index
            names[atom_index] = name
    distance_matrix = Chem.GetDistanceMatrix(molecule)
    for atom in molecule.GetAtoms():
        atom_index = atom.GetIdx()
        if atom_index in atom_to_residue:
            continue
        residue_index = min(
            range(1, len(anchors) + 1),
            key=lambda value: (
                float(distance_matrix[atom_index, anchors[value - 1]]),
                value,
            ),
        )
        atom_to_residue[atom_index] = residue_index
    counters: dict[tuple[int, str], int] = {}
    for atom in molecule.GetAtoms():
        atom_index = atom.GetIdx()
        residue_index = atom_to_residue[atom_index]
        row = rows[residue_index - 1]
        symbol = str(row["monomer_symbol"])
        residue_name = _STANDARD_THREE.get(
            symbol, symbol[:3].upper() or "UNK"
        )
        atom.SetProp("_TriposResidueName", residue_name)
        atom.SetProp("_TriposChainId", str(row["chain_id"]))
        atom.SetIntProp("_TriposResidueNumber", residue_index)
        atom.SetProp("_TriposInsertionCode", "")
        name = names.get(atom_index)
        if name is None:
            key = (residue_index, atom.GetSymbol())
            counters[key] = counters.get(key, 0) + 1
            name = f"{atom.GetSymbol()}{counters[key]}"
        atom.SetProp("_TriposAtomName", name)
        atom.SetProp(
            "_CycPepAtomUID",
            f"{exact_document.get('graph_sha256')}:{row['node_id']}:{name}",
        )
    return {
        "status": "complete",
        "matched_residues": len(matches),
        "atom_metadata_count": molecule.GetNumAtoms(),
    }


def _template_coord_map(
    molecule: Chem.Mol,
    match: Mapping[str, Any],
) -> dict[int, Point3D]:
    backbone = _backbone_matches(molecule)
    coordinates = list(match.get("ca_coordinate_map") or [])
    if len(backbone) != len(coordinates):
        return {}
    return {
        int(backbone[index][1]): Point3D(
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        )
        for index, row in enumerate(coordinates)
    }


def _embed(
    parent: Chem.Mol,
    *,
    random_seed: int,
    num_threads: int,
    coord_map: Mapping[int, Point3D] | None = None,
) -> tuple[Chem.Mol | None, dict[str, Any]]:
    molecule = Chem.Mol(parent)
    molecule.RemoveAllConformers()
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(random_seed)
    parameters.numThreads = int(num_threads)
    parameters.useMacrocycleTorsions = True
    parameters.useRandomCoords = coord_map is None
    parameters.timeout = V5_ETKDG_ATTEMPT_TIMEOUT_SECONDS
    if coord_map:
        parameters.SetCoordMap(dict(coord_map))
    status = int(AllChem.EmbedMolecule(molecule, parameters))
    if status != 0:
        return None, {
            "status": "embed_failed",
            "embed_status": status,
            "random_seed": int(random_seed),
            "coord_map_size": len(coord_map or {}),
            "timeout_seconds": V5_ETKDG_ATTEMPT_TIMEOUT_SECONDS,
        }
    return molecule, {
        "status": "embedded",
        "embed_status": status,
        "random_seed": int(random_seed),
        "coord_map_size": len(coord_map or {}),
        "timeout_seconds": V5_ETKDG_ATTEMPT_TIMEOUT_SECONDS,
    }


def _derived_topology(
    exact_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the shared-vocabulary topology and macrocycle ring size.

    Mirrors ``docking/torsion_observations.py`` ``_topology`` on the
    exact_v1 monomer-port graph: non-PEPTIDE bonds between residues
    separated by more than one position are closures; HT closures flag
    head-to-tail, SS flags disulfide, and every other closure counts as
    unclassified.  The result feeds both the runtime prior query keys and
    the MOL2 validation receipt so the materialization and PDBQT layers
    query one table with one vocabulary.
    """
    monomers = list(exact_document.get("monomers") or [])
    if not monomers:
        return {
            "topology_class": "unknown",
            "macrocycle_ring_size": None,
            "cyclic_backbone": False,
        }
    positions: dict[int, tuple[str, int]] = {}
    per_chain_counts: dict[str, int] = {}
    for row in monomers:
        positions[int(row["node_id"])] = (
            str(row["chain_id"]),
            int(row["canonical_position"]),
        )
        chain_id = str(row["chain_id"])
        per_chain_counts[chain_id] = per_chain_counts.get(chain_id, 0) + 1
    head_to_tail = False
    disulfide = False
    other = False
    spans: list[int] = []
    for bond in exact_document.get("bonds") or []:
        if str(bond.get("bond_type") or "") == "PEPTIDE":
            continue
        source = positions.get(int(bond["src"]["node_id"]))
        destination = positions.get(int(bond["dst"]["node_id"]))
        if source is None or destination is None:
            continue
        (source_chain, source_position), (
            destination_chain,
            destination_position,
        ) = source, destination
        if source_chain != destination_chain:
            other = True
            continue
        separation = abs(source_position - destination_position)
        if separation <= 1:
            continue
        spans.append(separation + 1)
        bond_type = str(bond.get("bond_type") or "")
        if bond_type in {"HT", "HEAD_TO_TAIL"}:
            head_to_tail = True
        elif bond_type in {"SS", "DISULFIDE"}:
            disulfide = True
        else:
            other = True
    topology, ring_size = classify_macrocycle_topology(
        residue_count=max(per_chain_counts.values(), default=0),
        head_to_tail=head_to_tail,
        disulfide=disulfide,
        other=other,
        closure_residue_spans=spans,
    )
    return {
        "topology_class": topology,
        "macrocycle_ring_size": ring_size,
        "cyclic_backbone": head_to_tail,
    }


def _prior_constraints(
    molecule: Chem.Mol,
    prior,
    *,
    topology_class: str | None,
    macrocycle_ring_size: int | None,
    cyclic_backbone: bool,
    excluded_atoms: frozenset[int] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect table-guided dihedral constraints for rotatable bonds.

    A bond is guided when the runtime table matches it with high or medium
    confidence and a sufficiently concentrated circular mean.  Bonds whose
    central atoms carry template coordinates are excluded so template
    evidence keeps priority.  ``eligible_to_freeze`` is deliberately not
    consulted: it gates PDBQT freezing decisions, not materialization
    guidance.
    """
    if prior is None:
        return [], {"status": "unavailable", "applied_count": 0}
    constraints: list[dict[str, Any]] = []
    not_applicable = 0
    for left, right in molecule.GetSubstructMatches(
        Lipinski.RotatableBondSmarts
    ):
        if int(left) in excluded_atoms or int(right) in excluded_atoms:
            continue
        quartet = prior_guidance_quartet(
            molecule,
            (int(left), int(right)),
            cyclic_backbone=cyclic_backbone,
        )
        if quartet is None:
            not_applicable += 1
            continue
        try:
            keys = build_query_keys(
                molecule,
                (int(left), int(right)),
                topology_class=topology_class,
                macrocycle_ring_size=macrocycle_ring_size,
            )
            match = prior.query(keys, flexibility_mode="balanced")
        except Exception:
            not_applicable += 1
            continue
        statistics = dict(match.statistics or {})
        mean = statistics.get("circular_mean_deg")
        spread = statistics.get("circular_std_deg")
        if (
            match.status != "matched"
            or match.confidence not in {"high", "medium"}
            or mean is None
            or spread is None
            or not math.isfinite(float(mean))
            or not math.isfinite(float(spread))
            or float(spread) > PRIOR_APPLY_MAX_STD_DEG
        ):
            not_applicable += 1
            continue
        constraints.append({
            "atom_indices": [int(left), int(right)],
            "quartet": [int(value) for value in quartet],
            "lookup_level": match.lookup_level,
            "confidence": str(match.confidence),
            "target_deg": float(mean),
            "circular_std_deg": float(spread),
        })
    audit = {
        "status": "applied" if constraints else "no_applicable_prior",
        "applied_count": len(constraints),
        "not_applicable_bond_count": not_applicable,
        "applied": [
            {
                "atom_indices": row["atom_indices"],
                "quartet": row["quartet"],
                "lookup_level": row["lookup_level"],
                "confidence": row["confidence"],
                "target_deg": row["target_deg"],
            }
            for row in constraints
        ],
        "runtime_sha256": getattr(prior, "runtime_sha256", None),
        "manifest_sha256": getattr(prior, "manifest_sha256", None),
    }
    return constraints, audit


def _circular_delta_deg(observed: float, target: float) -> float:
    return (observed - target + 180.0) % 360.0 - 180.0


def _measure_constraint_deg(
    molecule: Chem.Mol, quartet: Sequence[int]
) -> float | None:
    try:
        value = float(
            rdMolTransforms.GetDihedralDeg(
                molecule.GetConformer(), *quartet
            )
        )
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _force_field_for(molecule: Chem.Mol):
    try:
        properties = AllChem.MMFFGetMoleculeProperties(molecule)
        if properties is not None:
            return (
                AllChem.MMFFGetMoleculeForceField(
                    molecule, properties, confId=0
                ),
                "MMFF94",
            )
    except Exception:
        pass
    try:
        return AllChem.UFFGetMoleculeForceField(molecule, confId=0), "UFF"
    except Exception:
        return None, None


def _constraint_bond_audit(
    molecule: Chem.Mol,
    constraints: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    bonds = []
    unsatisfied = []
    for row in constraints:
        quartet = [int(value) for value in row["quartet"]]
        observed = _measure_constraint_deg(molecule, quartet)
        delta = (
            _circular_delta_deg(observed, float(row["target_deg"]))
            if observed is not None
            else None
        )
        satisfied = delta is not None and abs(delta) <= (
            PRIOR_RELAX_TOLERANCE_DEG
        )
        bonds.append({
            "atom_indices": list(row["atom_indices"]),
            "target_deg": float(row["target_deg"]),
            "final_deg": observed,
            "delta_deg": delta,
            "satisfied": satisfied,
        })
        if not satisfied:
            unsatisfied.append(list(row["atom_indices"]))
    return bonds, unsatisfied


def _constrained_relaxation(
    molecule: Chem.Mol,
    constraints: Sequence[Mapping[str, Any]],
    *,
    fixed_atoms: Sequence[int] = (),
) -> dict[str, Any]:
    """Relax a conformer while respecting table dihedral constraints.

    Initial dihedrals are projected onto the table means; minimization then
    runs under RDKit torsion constraints when the installed backend exposes
    MMFFAddTorsionConstraint / UFFAddTorsionConstraint and falls back to a
    deterministic project-and-relax loop otherwise.  Every constrained bond
    reports its final deviation; bonds beyond tolerance are recorded as
    prior_unsatisfied and never silently accepted.
    """
    conformer = molecule.GetConformer()
    for row in constraints:
        rdMolTransforms.SetDihedralDeg(
            conformer,
            *[int(value) for value in row["quartet"]],
            float(row["target_deg"]),
        )
    force_field, method = _force_field_for(molecule)
    if force_field is None:
        bonds, unsatisfied = _constraint_bond_audit(
            molecule, constraints
        )
        return {
            "status": "not_parameterized",
            "method": None,
            "result": None,
            "energy": None,
            "mechanism": "projection_only",
            "iterations": 0,
            "tolerance_deg": PRIOR_RELAX_TOLERANCE_DEG,
            "bonds": bonds,
            "prior_unsatisfied": unsatisfied,
        }
    for atom_index in sorted(set(int(value) for value in fixed_atoms)):
        force_field.AddFixedPoint(atom_index)
    native_adder = getattr(
        force_field,
        {
            "MMFF94": "MMFFAddTorsionConstraint",
            "UFF": "UFFAddTorsionConstraint",
        }.get(str(method), ""),
        None,
    )
    if not constraints:
        mechanism = "no_constraints"
        iterations = 1
        result = int(force_field.Minimize(maxIts=500))
    elif native_adder is not None:
        mechanism = "rdkit_torsion_constraint"
        for row in constraints:
            quartet = [int(value) for value in row["quartet"]]
            native_adder(
                quartet[0],
                quartet[1],
                quartet[2],
                quartet[3],
                False,
                float(row["target_deg"]) - PRIOR_RELAX_TOLERANCE_DEG,
                float(row["target_deg"]) + PRIOR_RELAX_TOLERANCE_DEG,
                PRIOR_CONSTRAINT_FORCE_CONSTANT,
            )
        force_field.Initialize()
        iterations = 1
        result = int(force_field.Minimize(maxIts=500))
    else:
        mechanism = "project_relax"
        iterations = 0
        result = 1
        for iteration in range(1, PRIOR_RELAX_MAX_ITERATIONS + 1):
            result = int(force_field.Minimize(maxIts=500))
            iterations = iteration
            bonds, unsatisfied = _constraint_bond_audit(
                molecule, constraints
            )
            if not unsatisfied:
                break
            if iteration < PRIOR_RELAX_MAX_ITERATIONS:
                for row in constraints:
                    rdMolTransforms.SetDihedralDeg(
                        conformer,
                        *[int(value) for value in row["quartet"]],
                        float(row["target_deg"]),
                    )
    bonds, unsatisfied = _constraint_bond_audit(molecule, constraints)
    try:
        energy = float(force_field.CalcEnergy())
    except Exception:
        energy = None
    return {
        "status": "converged" if result == 0 else "iteration_limit",
        "method": method,
        "result": result,
        "energy": energy if energy is not None and math.isfinite(
            energy
        ) else None,
        "mechanism": mechanism,
        "iterations": iterations,
        "tolerance_deg": PRIOR_RELAX_TOLERANCE_DEG,
        "bonds": bonds,
        "prior_unsatisfied": unsatisfied,
    }


def _optimize(
    molecule: Chem.Mol,
    *,
    fixed_atoms: Sequence[int] = (),
) -> dict[str, Any]:
    force_field, method = _force_field_for(molecule)
    if force_field is None:
        return {
            "status": "not_parameterized",
            "method": None,
            "energy": None,
        }
    for atom_index in sorted(set(int(value) for value in fixed_atoms)):
        force_field.AddFixedPoint(atom_index)
    result = int(force_field.Minimize(maxIts=500))
    energy = float(force_field.CalcEnergy())
    return {
        "status": "converged" if result == 0 else "iteration_limit",
        "method": method,
        "result": result,
        "energy": energy if math.isfinite(energy) else None,
    }


def _severe_clashes(molecule: Chem.Mol) -> list[list[Any]]:
    conformer = molecule.GetConformer()
    distance = Chem.GetDistanceMatrix(molecule)
    clashes = []
    heavy = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    for offset, left in enumerate(heavy):
        for right in heavy[offset + 1:]:
            if distance[left, right] <= 2:
                continue
            a = conformer.GetAtomPosition(left)
            b = conformer.GetAtomPosition(right)
            observed = a.Distance(b)
            if observed < 0.75:
                clashes.append([left, right, observed])
    return clashes


def _bond_geometry(molecule: Chem.Mol) -> dict[str, Any]:
    conformer = molecule.GetConformer()
    invalid = []
    ring_distances = []
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if (
            molecule.GetAtomWithIdx(left).GetAtomicNum() == 1
            or molecule.GetAtomWithIdx(right).GetAtomicNum() == 1
        ):
            continue
        distance = conformer.GetAtomPosition(left).Distance(
            conformer.GetAtomPosition(right)
        )
        if bond.IsInRing():
            ring_distances.append(distance)
        if not math.isfinite(distance) or distance < 0.65 or distance > 2.60:
            invalid.append([left, right, distance])
    return {
        "heavy_bond_count": sum(
            bond.GetBeginAtom().GetAtomicNum() > 1
            and bond.GetEndAtom().GetAtomicNum() > 1
            for bond in molecule.GetBonds()
        ),
        "ring_bond_count": len(ring_distances),
        "ring_bond_distance_min": (
            min(ring_distances) if ring_distances else None
        ),
        "ring_bond_distance_max": (
            max(ring_distances) if ring_distances else None
        ),
        "invalid_bond_count": len(invalid),
        "invalid_bonds": invalid[:20],
    }


def _qa(
    molecule: Chem.Mol,
    *,
    expected_inchikey: str,
    optimization: Mapping[str, Any],
    topology_class: str | None,
) -> dict[str, Any]:
    coordinates_finite = all(
        math.isfinite(value)
        for atom_index in range(molecule.GetNumAtoms())
        for value in tuple(
            molecule.GetConformer().GetAtomPosition(atom_index)
        )
    )
    observed_key = Chem.MolToInchiKey(molecule)
    clashes = _severe_clashes(molecule)
    bond_geometry = _bond_geometry(molecule)
    expected_ring = str(topology_class or "linear") != "linear"
    closure_valid = bool(
        not expected_ring or bond_geometry["ring_bond_count"] > 0
    )
    energy = optimization.get("energy")
    heavy_count = max(1, molecule.GetNumHeavyAtoms())
    energy_per_heavy_atom = (
        float(energy) / heavy_count
        if energy is not None and math.isfinite(float(energy))
        else None
    )
    strain_valid = bool(
        energy_per_heavy_atom is None
        or energy_per_heavy_atom <= 250.0
    )
    return {
        "coordinates_finite": coordinates_finite,
        "full_inchikey": observed_key,
        "full_inchikey_match": observed_key == expected_inchikey,
        "severe_clash_count": len(clashes),
        "severe_clashes": clashes[:20],
        "bond_geometry": bond_geometry,
        "closure_valid": closure_valid,
        "strain_screen": {
            "energy_per_heavy_atom": energy_per_heavy_atom,
            "maximum_energy_per_heavy_atom": 250.0,
            "passed": strain_valid,
        },
        "optimization": dict(optimization),
        "passed": bool(
            coordinates_finite
            and observed_key == expected_inchikey
            and not clashes
            and bond_geometry["invalid_bond_count"] == 0
            and closure_valid
            and strain_valid
        ),
    }


def _is_duplicate(
    molecule: Chem.Mol,
    accepted: Sequence[Chem.Mol],
    *,
    threshold: float = 0.15,
) -> bool:
    for existing in accepted:
        try:
            if rdMolAlign.GetBestRMS(molecule, existing) < threshold:
                return True
        except Exception:
            continue
    return False


def materialize_mol2_ensemble(
    chemical_graph: Mapping[str, Any],
    output_dir: str | Path,
    *,
    ensemble_size: int = 4,
    template_strategy: str = "full",
    torsion_prior_path: str | Path | None = None,
    random_seed: int = 42,
    num_threads: int = 1,
    monomer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate zero or more independently validated MOL2 artifacts."""
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                chemical_graph.get("exact_v1") or {}
            ),
        ):
            return materialize_mol2_ensemble(
                chemical_graph,
                output_dir,
                ensemble_size=ensemble_size,
                template_strategy=template_strategy,
                torsion_prior_path=torsion_prior_path,
                random_seed=random_seed,
                num_threads=num_threads,
            )
    if type(ensemble_size) is not int or ensemble_size < 1:
        raise ValueError("ensemble_size must be a positive integer")
    if type(num_threads) is not int or num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty ensemble directory: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if chemical_graph.get("artifact_type") != (
        ArtifactType.CHEMICAL_GRAPH.value
    ):
        raise ValueError("ChemicalGraphArtifact is required")
    graph_payload = {
        "exact_v1": chemical_graph.get("exact_v1"),
        "smiles": chemical_graph.get("smiles"),
        "full_inchikey": chemical_graph.get("full_inchikey"),
        "formal_charge": chemical_graph.get("formal_charge"),
        "topology_class": chemical_graph.get("topology_class"),
        "microstate_policy": chemical_graph.get("microstate_policy"),
        "parent_full_inchikey": chemical_graph.get(
            "parent_full_inchikey"
        ),
        "exact_v1_identity_match": chemical_graph.get(
            "exact_v1_identity_match"
        ),
        "materializable": chemical_graph.get("materializable"),
        "monomer_resolution": dict(
            chemical_graph.get("monomer_resolution") or {}
        ),
        "symbolic_graph": dict(
            (chemical_graph.get("provenance") or {}).get(
                "symbolic_graph"
            )
            or {}
        ),
    }
    validate_artifact_record(
        chemical_graph,
        expected_type=ArtifactType.CHEMICAL_GRAPH,
        payload=graph_payload,
    )
    if chemical_graph.get("status") != ArtifactStatus.MATERIALIZED.value:
        raise ValueError("chemical graph artifact is not materialized")
    if not chemical_graph.get("materializable"):
        raise ValueError("chemical graph is not materializable")
    smiles = str(chemical_graph.get("smiles") or "")
    expected_key = str(chemical_graph.get("full_inchikey") or "")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not expected_key:
        raise ValueError("chemical graph lacks a parseable identity")
    if Chem.MolToInchiKey(molecule) != expected_key:
        raise ValueError("chemical graph InChIKey differs from its SMILES")
    parent = Chem.AddHs(molecule)
    exact_document = dict(chemical_graph.get("exact_v1") or {})
    if exact_document.get("exactness_status") != "EXACT":
        raise ValueError("exact_v1 EXACT graph is required for materialization")
    metadata_audit = _annotate_residues(parent, exact_document)
    if metadata_audit.get("status") != "complete":
        raise ValueError(
            "chemical graph atom provenance is not materializable: "
            + str(metadata_audit.get("reason") or "unknown reason")
        )
    generated_map = str(
        (chemical_graph.get("provenance") or {}).get("map")
        or ""
    )
    template_evidence = _portable_template_evidence(
        find_coordinate_evidence(
            generated_map,
            smiles,
            max_matches=2,
            template_strategy=template_strategy,
            random_seed=int(random_seed),
        )
    )
    prior = None
    prior_error = None
    runtime_prior_path = (
        Path(torsion_prior_path)
        if torsion_prior_path is not None
        else (
            Path(__file__).resolve().parents[1]
            / "data"
            / "torsion_priors"
            / "torsion_priors_runtime.json"
        )
    )
    if runtime_prior_path.is_file():
        try:
            prior = load_torsion_prior(runtime_prior_path)
        except Exception as exc:
            prior_error = f"{type(exc).__name__}: {exc}"
    else:
        prior_error = f"torsion prior is missing: {runtime_prior_path}"

    derived_topology = _derived_topology(exact_document)

    attempts = []
    accepted_molecules: list[Chem.Mol] = []
    members: list[ConformerMember] = []
    strategies = []
    for index, match in enumerate(
        template_evidence.get("matches", []), 1
    ):
        strategies.append((
            f"template_{index}",
            match,
            CoordinateOrigin.TEMPLATE_BORROWED,
        ))
    strategies.append((
        "torsion_prior_guided",
        None,
        CoordinateOrigin.TORSION_GUIDED,
    ))
    for index in range(max(ensemble_size * 4, 12)):
        strategies.append((
            f"diverse_etkdg_{index + 1}",
            None,
            CoordinateOrigin.GENERATED,
        ))

    for attempt_index, (strategy, match, origin) in enumerate(
        strategies, 1
    ):
        if len(members) >= ensemble_size:
            break
        coord_map = (
            _template_coord_map(parent, match)
            if match is not None
            else {}
        )
        candidate, embed_audit = _embed(
            parent,
            random_seed=int(random_seed) + attempt_index - 1,
            num_threads=int(num_threads),
            coord_map=coord_map,
        )
        attempt = {
            "attempt_index": attempt_index,
            "strategy": strategy,
            "coordinate_origin": origin.value,
            "template": match,
            "embed": embed_audit,
        }
        if candidate is None:
            attempt["status"] = "embed_failed"
            attempts.append(attempt)
            continue
        if strategy.startswith("diverse_etkdg"):
            prior_audit = {
                "status": "not_requested",
                "applied_count": 0,
            }
            optimization = _optimize(
                candidate, fixed_atoms=tuple(coord_map)
            )
        else:
            constraints, prior_audit = _prior_constraints(
                candidate,
                prior,
                topology_class=derived_topology["topology_class"],
                macrocycle_ring_size=derived_topology[
                    "macrocycle_ring_size"
                ],
                cyclic_backbone=derived_topology["cyclic_backbone"],
                excluded_atoms=(
                    frozenset(int(value) for value in coord_map)
                    if match is not None
                    else frozenset()
                ),
            )
            optimization = _constrained_relaxation(
                candidate,
                constraints,
                fixed_atoms=tuple(coord_map),
            )
            prior_audit["constraint"] = {
                key: optimization[key]
                for key in (
                    "mechanism",
                    "iterations",
                    "tolerance_deg",
                    "bonds",
                    "prior_unsatisfied",
                )
            }
        qa = _qa(
            candidate,
            expected_inchikey=expected_key,
            optimization=optimization,
            topology_class=chemical_graph.get("topology_class"),
        )
        qa["metadata"] = metadata_audit
        qa["prior_guidance"] = prior_audit
        attempt["qa"] = qa
        if not qa["passed"]:
            attempt["status"] = "qa_rejected"
            attempts.append(attempt)
            continue
        if _is_duplicate(candidate, accepted_molecules):
            attempt["status"] = "duplicate"
            attempts.append(attempt)
            continue
        member_index = len(members) + 1
        conformer_id = f"conf_{member_index:03d}"
        mol2_path = destination / f"{conformer_id}.mol2"
        single = _single_conformer_copy(candidate)
        try:
            written, error = mol_to_mol2(
                single, output_path=str(mol2_path)
            )
            if error or not written or not mol2_path.is_file():
                raise RuntimeError(error or "MOL2 writer produced no file")
            coordinate_mode = (
                "template_completed"
                if origin == CoordinateOrigin.TEMPLATE_BORROWED
                else "regenerated"
            )
            parent_evidence = chemical_graph["evidence"]
            receipt = write_validation_receipt(
                mol2_path,
                coordinate_mode=coordinate_mode,
                rigor=str(parent_evidence["chemical_rigor"]),
                quality=(
                    "specified"
                    if parent_evidence["chemical_basis"] == "S"
                    else "hypothesis"
                ),
                source_heavy_atom_mapping_complete=False,
                atom_provenance_complete=True,
                source_input_sha256=str(
                    chemical_graph["payload_sha256"]
                ),
                topology_class=derived_topology["topology_class"],
                macrocycle_ring_size=derived_topology[
                    "macrocycle_ring_size"
                ],
                max_source_coordinate_delta_angstrom=None,
                evidence_manifest_sha256=str(
                    exact_document.get("graph_sha256")
                ),
                expected_full_inchikey=expected_key,
            )
        except Exception as exc:
            attempt["status"] = "mol2_validation_failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
            try:
                mol2_path.unlink(missing_ok=True)
                Path(str(mol2_path) + ".validation.json").unlink(
                    missing_ok=True
                )
            except OSError:
                pass
            continue
        member = ConformerMember(
            conformer_id=conformer_id,
            coordinate_origin=origin,
            strategy=strategy,
            mol2_path=mol2_path.relative_to(destination).as_posix(),
            mol2_sha256=sha256_path(mol2_path),
            receipt_path=receipt.relative_to(destination).as_posix(),
            receipt_sha256=sha256_path(receipt),
            energy=optimization.get("energy"),
            qa=qa,
            status="validated",
            warnings=(),
        )
        members.append(member)
        accepted_molecules.append(candidate)
        attempt["status"] = "accepted"
        attempt["conformer_id"] = conformer_id
        attempts.append(attempt)

    parent_profile = evidence_from_dict(chemical_graph["evidence"])
    best_origin = (
        members[0].coordinate_origin
        if members
        else CoordinateOrigin.NONE
    )
    ensemble_evidence = inherit_evidence(
        parent_profile,
        coordinate_origin=best_origin,
        coordinate_level=(
            CoordinateLevel.X2
            if any(
                member.coordinate_origin
                == CoordinateOrigin.TEMPLATE_BORROWED
                for member in members
            )
            else CoordinateLevel.X1
            if members
            else CoordinateLevel.X0
        ),
    )
    payload = {
        "parent_graph_artifact_id": chemical_graph["artifact_id"],
        "parent_graph_sha256": exact_document.get("graph_sha256"),
        "requested_count": ensemble_size,
        "members": [member.to_dict() for member in members],
        "attempts": attempts,
        "template_evidence": template_evidence,
        "torsion_prior_error": prior_error,
        "random_seed": int(random_seed),
        "num_threads": int(num_threads),
        "rdkit_version": rdkit.__version__,
    }
    payload_hash = artifact_payload_sha256(payload)
    artifact_id = make_artifact_id(
        ArtifactType.CONFORMER_ENSEMBLE,
        (str(chemical_graph["artifact_id"]),),
        payload,
    )
    status = (
        ArtifactStatus.MATERIALIZED
        if len(members) == ensemble_size
        else ArtifactStatus.PARTIAL
        if members
        else ArtifactStatus.FAILED
    )
    artifact = ConformerEnsembleArtifact(
        artifact_type=ArtifactType.CONFORMER_ENSEMBLE,
        artifact_id=artifact_id,
        parent_artifact_ids=(str(chemical_graph["artifact_id"]),),
        status=status,
        payload_sha256=payload_hash,
        evidence=ensemble_evidence,
        warnings=tuple(
            ["ENSEMBLE_PARTIAL"]
            if members and len(members) < ensemble_size
            else ["ENSEMBLE_MATERIALIZATION_FAILED"]
            if not members
            else []
        ),
        provenance={
            "materializer": "materialize_mol2_ensemble",
            "schema_version": ENSEMBLE_SCHEMA_VERSION,
            "template_evidence": template_evidence,
            "torsion_prior_error": prior_error,
            "attempts": attempts,
        },
        claim_boundary=ClaimBoundary(
            allowed=(
                "generated coordinate provenance",
                "MOL2 format qualification",
            ),
            forbidden=(
                "experimental structure accuracy",
                "biological flexibility",
                "docking benefit",
            ),
        ),
        requested_count=int(ensemble_size),
        produced_count=len(members),
        members=tuple(members),
    )
    validate_inherited_evidence(
        parent_profile,
        ensemble_evidence,
        allow_coordinate=True,
    )
    validate_artifact_identity(artifact, payload=payload)
    manifest_path = destination / "ensemble_manifest.json"
    manifest = {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "artifact": artifact.to_dict(),
        "payload": payload,
    }
    _atomic_json(manifest_path, manifest)
    final = replace(
        artifact,
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_path(manifest_path),
    )
    validated_artifacts = []
    for member in members:
        member_profile = inherit_evidence(
            parent_profile,
            coordinate_origin=member.coordinate_origin,
            coordinate_level=(
                CoordinateLevel.X2
                if member.coordinate_origin
                == CoordinateOrigin.TEMPLATE_BORROWED
                else CoordinateLevel.X1
            ),
            format_level=FormatLevel.Q2,
        )
        member_payload = {
            "conformer_id": member.conformer_id,
            "path": str(
                destination / str(member.mol2_path)
            ),
            "sha256": member.mol2_sha256,
            "receipt_path": str(
                destination / str(member.receipt_path)
            ),
            "receipt_sha256": member.receipt_sha256,
            "full_inchikey": expected_key,
        }
        member_artifact_id = make_artifact_id(
            ArtifactType.VALIDATED_MOL2,
            (artifact_id,),
            member_payload,
        )
        validated_artifact = ValidatedMol2Artifact(
                artifact_type=ArtifactType.VALIDATED_MOL2,
                artifact_id=member_artifact_id,
                parent_artifact_ids=(artifact_id,),
                status=ArtifactStatus.MATERIALIZED,
                payload_sha256=artifact_payload_sha256(
                    member_payload
                ),
                evidence=member_profile,
                provenance={
                    "ensemble_artifact_id": artifact_id,
                    "strategy": member.strategy,
                    "qa": dict(member.qa),
                },
                claim_boundary=artifact.claim_boundary,
                conformer_id=member.conformer_id,
                path=str(destination / str(member.mol2_path)),
                sha256=str(member.mol2_sha256),
                receipt_path=str(
                    destination / str(member.receipt_path)
                ),
                receipt_sha256=str(member.receipt_sha256),
                full_inchikey=expected_key,
        )
        validate_inherited_evidence(
            parent_profile,
            member_profile,
            allow_coordinate=True,
            allow_format=True,
        )
        validate_artifact_identity(
            validated_artifact, payload=member_payload
        )
        validated_artifacts.append(validated_artifact.to_dict())
    return {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "ensemble": final.to_dict(),
        "validated_mol2_artifacts": validated_artifacts,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "attempts": attempts,
    }


__all__ = [
    "ENSEMBLE_SCHEMA_VERSION",
    "materialize_mol2_ensemble",
]
