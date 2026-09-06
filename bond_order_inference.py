"""Auditable bond-order candidate inference for result-first recovery.

The module deliberately separates candidate materialization from chemical
qualification.  Every admitted candidate must preserve the observed heavy-atom
composition.  Agreement is measured across implementation families, while all
engine attempts and disagreements remain visible to callers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem

from .core.pdb_utils import first_model_records, pdb_atom_element


SCHEMA_VERSION = "bond-order-inference-1"

_ENGINE_PRIORITY = {
    "strict_candidate_assessment": 0,
    "path_g_template": 1,
    "openbabel_pdb": 2,
    "rdkit_pdb_proximity": 3,
    "path_h_geometry": 4,
}
_GRAPH_ENGINE_PRIORITY = {
    "openbabel_pdb": 0,
    "rdkit_pdb_proximity": 1,
}
_FAMILY_PRIORITY = {
    "template": 0,
    "bond_order_perception": 1,
    "geometry": 2,
}


def _source_atoms(pdb_path: Path, chain_id: str) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    serials: set[int] = set()
    with pdb_path.open(encoding="ascii", errors="replace") as handle:
        for line in first_model_records(handle):
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54 or line[21:22] != chain_id:
                continue
            serial = int(line[6:11])
            if serial in serials:
                raise ValueError(f"duplicate source atom serial {serial}")
            serials.add(serial)
            element = pdb_atom_element(line)
            if not element:
                raise ValueError(
                    f"source atom {serial} has no resolvable element"
                )
            xyz = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError(f"source atom {serial} has non-finite coordinates")
            try:
                residue_number = int(line[22:26])
            except ValueError:
                residue_number = 0
            atoms.append(
                {
                    "serial": serial,
                    "name": line[12:16].strip(),
                    "residue": line[17:20].strip(),
                    "chain": line[21:22].strip(),
                    "residue_number": residue_number,
                    "insertion_code": line[26:27].strip(),
                    "element": element.capitalize(),
                    "xyz": list(xyz),
                }
            )
    if not atoms:
        raise ValueError("no source atoms available for bond-order inference")
    return atoms


def _heavy_composition_from_atoms(
    atoms: list[dict[str, Any]],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(atom["element"]).upper()
                for atom in atoms
                if str(atom["element"]).upper() != "H"
            ).items()
        )
    )


def _heavy_composition_from_mol(molecule: Any) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                atom.GetSymbol().upper()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() > 1
            ).items()
        )
    )


def _identity_from_smiles(smiles: str) -> dict[str, Any]:
    token = str(smiles).strip().split()[0] if str(smiles).strip() else ""
    if not token:
        raise ValueError("empty candidate SMILES")
    molecule = Chem.MolFromSmiles(token)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("candidate SMILES is not RDKit-parseable")
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise ValueError("candidate SMILES contains dummy atoms")
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    full_inchikey = Chem.MolToInchiKey(molecule)
    if not full_inchikey:
        raise ValueError("candidate has no Standard InChIKey")
    blocks = full_inchikey.split("-")
    return {
        "canonical_smiles": canonical,
        "full_inchikey": full_inchikey,
        "inchi_connectivity_block": blocks[0],
        "inchi_nonprotonation_key": "-".join(blocks[:2]),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "heavy_atom_composition": _heavy_composition_from_mol(molecule),
    }


def _set_pdb_info(atom: Any, source: dict[str, Any]) -> None:
    info = Chem.AtomPDBResidueInfo()
    info.SetSerialNumber(int(source["serial"]))
    info.SetName(str(source["name"]))
    info.SetResidueName(str(source["residue"]))
    info.SetResidueNumber(int(source["residue_number"]))
    info.SetInsertionCode(str(source["insertion_code"]))
    info.SetChainId(str(source["chain"]))
    atom.SetPDBResidueInfo(info)


def _chemical_graph(
    molecule: Any,
    source_atoms: list[dict[str, Any]],
    *,
    engine: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if molecule.GetNumAtoms() != len(source_atoms):
        raise ValueError(
            "candidate/source atom count mismatch: "
            f"{molecule.GetNumAtoms()} != {len(source_atoms)}"
        )
    if molecule.GetNumConformers() == 0:
        raise ValueError("candidate molecule has no source-bound conformer")
    conformer = molecule.GetConformer()
    max_coordinate_delta = 0.0
    atom_rows: list[dict[str, Any]] = []
    serial_by_index: list[int] = []
    for index, (atom, source) in enumerate(
        zip(molecule.GetAtoms(), source_atoms)
    ):
        if atom.GetSymbol().upper() != str(source["element"]).upper():
            raise ValueError(
                "candidate/source element mismatch at index "
                f"{index}: {atom.GetSymbol()} != {source['element']}"
            )
        position = conformer.GetAtomPosition(index)
        delta = max(
            abs(float(position.x) - float(source["xyz"][0])),
            abs(float(position.y) - float(source["xyz"][1])),
            abs(float(position.z) - float(source["xyz"][2])),
        )
        max_coordinate_delta = max(max_coordinate_delta, delta)
        if delta > 0.001:
            raise ValueError(
                "candidate/source coordinate mismatch at serial "
                f"{source['serial']}: {delta:.6f} A"
            )
        serial = int(source["serial"])
        serial_by_index.append(serial)
        atom_rows.append(
            {
                **source,
                "formal_charge": int(atom.GetFormalCharge()),
                "isotope": int(atom.GetIsotope()),
                "num_explicit_hs": int(atom.GetNumExplicitHs()),
                "no_implicit": bool(atom.GetNoImplicit()),
                "num_radical_electrons": int(
                    atom.GetNumRadicalElectrons()
                ),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "chiral_tag": str(atom.GetChiralTag()),
            }
        )
    bond_rows: list[dict[str, Any]] = []
    for bond in molecule.GetBonds():
        left = serial_by_index[bond.GetBeginAtomIdx()]
        right = serial_by_index[bond.GetEndAtomIdx()]
        bond_rows.append(
            {
                "a": min(left, right),
                "b": max(left, right),
                "order": float(bond.GetBondTypeAsDouble()),
                "is_aromatic": bool(bond.GetIsAromatic()),
                "stereo": str(bond.GetStereo()),
            }
        )
    bond_rows.sort(key=lambda row: (row["a"], row["b"]))
    graph = {
        "schema_version": "chemical-graph-1",
        "atoms": atom_rows,
        "bonds": bond_rows,
    }
    audit = {
        "engine": engine,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "bond_count": molecule.GetNumBonds(),
        "source_atom_mapping_complete": True,
        "source_atom_mapping_mode": "preserved_input_order",
        "max_coordinate_delta_angstrom": max_coordinate_delta,
        "graph_full_inchikey": Chem.MolToInchiKey(molecule),
    }
    return graph, audit


def _prepare_rdkit_candidate(
    pdb_path: Path,
    source_atoms: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    molecule = Chem.MolFromPDBFile(
        str(pdb_path),
        proximityBonding=True,
        sanitize=False,
        removeHs=False,
    )
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("RDKit PDB reader produced no molecule")
    Chem.SanitizeMol(molecule)
    Chem.RemoveStereochemistry(molecule)
    graph, mapping_audit = _chemical_graph(
        molecule, source_atoms, engine="rdkit_pdb_proximity"
    )
    identity = _identity_from_smiles(
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    )
    identity.update(
        {
            "engine": "rdkit_pdb_proximity",
            "family": "geometry",
            "candidate_graph": graph,
            "mapping_audit": mapping_audit,
            "engine_version": Chem.rdBase.rdkitVersion,
            "stereo_status": "unverified_removed",
        }
    )
    return identity, mapping_audit


def _openbabel_rdkit_mol(
    pdb_path: Path,
    source_atoms: list[dict[str, Any]],
) -> tuple[Any, str]:
    from openbabel import openbabel as ob
    from openbabel import pybel

    pybel_molecule = next(pybel.readfile("pdb", str(pdb_path)))
    ob_atoms = list(ob.OBMolAtomIter(pybel_molecule.OBMol))
    if len(ob_atoms) != len(source_atoms):
        raise ValueError(
            "Open Babel/source atom count mismatch: "
            f"{len(ob_atoms)} != {len(source_atoms)}"
        )
    editable = Chem.RWMol()
    for source, ob_atom in zip(source_atoms, ob_atoms):
        atom = Chem.Atom(int(ob_atom.GetAtomicNum()))
        atom.SetFormalCharge(int(ob_atom.GetFormalCharge()))
        _set_pdb_info(atom, source)
        editable.AddAtom(atom)
    for ob_bond in ob.OBMolBondIter(pybel_molecule.OBMol):
        order = int(ob_bond.GetBondOrder())
        if ob_bond.IsAromatic():
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }.get(order)
        if bond_type is None:
            raise ValueError(f"unsupported Open Babel bond order {order}")
        left = int(ob_bond.GetBeginAtomIdx()) - 1
        right = int(ob_bond.GetEndAtomIdx()) - 1
        editable.AddBond(left, right, bond_type)
        if ob_bond.IsAromatic():
            editable.GetAtomWithIdx(left).SetIsAromatic(True)
            editable.GetAtomWithIdx(right).SetIsAromatic(True)
            editable.GetBondBetweenAtoms(left, right).SetIsAromatic(True)
    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(source_atoms))
    for index, ob_atom in enumerate(ob_atoms):
        conformer.SetAtomPosition(
            index,
            (
                float(ob_atom.GetX()),
                float(ob_atom.GetY()),
                float(ob_atom.GetZ()),
            ),
        )
    conformer.Set3D(True)
    molecule.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(molecule)
    Chem.RemoveStereochemistry(molecule)
    return molecule, str(ob.OBReleaseVersion())


def _prepare_openbabel_candidate(
    pdb_path: Path,
    source_atoms: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    molecule, version = _openbabel_rdkit_mol(pdb_path, source_atoms)
    graph, mapping_audit = _chemical_graph(
        molecule, source_atoms, engine="openbabel_pdb"
    )
    identity = _identity_from_smiles(
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    )
    identity.update(
        {
            "engine": "openbabel_pdb",
            "family": "bond_order_perception",
            "candidate_graph": graph,
            "mapping_audit": mapping_audit,
            "engine_version": version,
            "stereo_status": "unverified_removed",
        }
    )
    return identity, mapping_audit


def _prepare_path_candidate(
    engine: str,
    family: str,
    function: Callable[..., Any],
    pdb_path: Path,
    chain_id: str,
) -> dict[str, Any]:
    value = function(str(pdb_path), str(chain_id))
    error = None
    smiles = value
    if isinstance(value, tuple):
        smiles = value[0] if value else None
        error = value[1] if len(value) > 1 else None
    if error or not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(str(error or "engine produced no SMILES"))
    identity = _identity_from_smiles(smiles)
    identity.update({"engine": engine, "family": family})
    return identity


def _candidate_id(group: dict[str, Any]) -> str:
    payload = {
        "inchi_connectivity_block": group["inchi_connectivity_block"],
        "engines": group["supporting_engines"],
        "full_inchikey_variants": group["full_inchikey_variants"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _admit_candidate(
    candidate: dict[str, Any],
    source_composition: dict[str, int],
    attempts: list[dict[str, Any]],
    admitted: list[dict[str, Any]],
) -> None:
    attempt = {
        "engine": candidate["engine"],
        "family": candidate["family"],
        "status": "rejected",
        "canonical_smiles": candidate["canonical_smiles"],
        "full_inchikey": candidate["full_inchikey"],
        "heavy_atom_composition": candidate["heavy_atom_composition"],
    }
    if candidate["heavy_atom_composition"] != source_composition:
        attempt["reason"] = "source_heavy_atom_composition_mismatch"
        attempts.append(attempt)
        return
    attempt["status"] = "admitted"
    attempt["reason"] = None
    attempts.append(attempt)
    admitted.append(candidate)


def infer_bond_order_candidates(
    pdb_path: str | Path,
    chain_id: str,
    *,
    strict_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Infer and rank source-composition-bound chemical candidates."""
    path = Path(pdb_path)
    source_atoms = _source_atoms(path, str(chain_id))
    source_composition = _heavy_composition_from_atoms(source_atoms)
    attempts: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []

    for index, source in enumerate(strict_candidates or []):
        try:
            identity = _identity_from_smiles(source["canonical_smiles"])
            identity.update(
                {
                    "engine": "strict_candidate_assessment",
                    "family": "template",
                    "strict_candidate_index": index,
                    "strict_routes": list(source.get("routes") or []),
                    "strict_supporting_route_count": int(
                        source.get("supporting_route_count") or 0
                    ),
                }
            )
            _admit_candidate(
                identity, source_composition, attempts, admitted
            )
        except Exception as exc:
            attempts.append(
                {
                    "engine": "strict_candidate_assessment",
                    "family": "template",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    from .paths.path_g import generate_g
    from .paths.path_h import generate_h

    for engine, family, function in (
        ("path_g_template", "template", generate_g),
        ("path_h_geometry", "geometry", generate_h),
    ):
        try:
            candidate = _prepare_path_candidate(
                engine, family, function, path, str(chain_id)
            )
            _admit_candidate(
                candidate, source_composition, attempts, admitted
            )
        except Exception as exc:
            attempts.append(
                {
                    "engine": engine,
                    "family": family,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    try:
        candidate, _mapping = _prepare_openbabel_candidate(path, source_atoms)
        _admit_candidate(candidate, source_composition, attempts, admitted)
    except ImportError as exc:
        attempts.append(
            {
                "engine": "openbabel_pdb",
                "family": "bond_order_perception",
                "status": "not_supported",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    except Exception as exc:
        attempts.append(
            {
                "engine": "openbabel_pdb",
                "family": "bond_order_perception",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    try:
        candidate, _mapping = _prepare_rdkit_candidate(path, source_atoms)
        _admit_candidate(candidate, source_composition, attempts, admitted)
    except Exception as exc:
        attempts.append(
            {
                "engine": "rdkit_pdb_proximity",
                "family": "geometry",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in admitted:
        key = (
            candidate["engine"],
            candidate["inchi_connectivity_block"],
            candidate["full_inchikey"],
        )
        deduplicated.setdefault(key, candidate)
    admitted = list(deduplicated.values())

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in admitted:
        grouped.setdefault(
            candidate["inchi_connectivity_block"], []
        ).append(candidate)
    identity_groups: list[dict[str, Any]] = []
    for connectivity_key, members in grouped.items():
        members.sort(
            key=lambda item: (
                _ENGINE_PRIORITY.get(item["engine"], 99),
                item["full_inchikey"],
                item["canonical_smiles"],
            )
        )
        graph_members = [
            member for member in members if member.get("candidate_graph")
        ]
        representative = min(
            graph_members or members,
            key=lambda item: (
                _GRAPH_ENGINE_PRIORITY.get(item["engine"], 99),
                _ENGINE_PRIORITY.get(item["engine"], 99),
                item["full_inchikey"],
                item["canonical_smiles"],
            ),
        )
        families = sorted(
            {member["family"] for member in members},
            key=lambda value: (_FAMILY_PRIORITY.get(value, 99), value),
        )
        engines = sorted(
            {member["engine"] for member in members},
            key=lambda value: (_ENGINE_PRIORITY.get(value, 99), value),
        )
        full_keys = sorted({member["full_inchikey"] for member in members})
        variants = sorted({member["canonical_smiles"] for member in members})
        if len(families) >= 2:
            evidence_class = "multi_family_consensus"
            quality = "high"
            rigor = "L2:R"
        elif families == ["template"]:
            evidence_class = "template_formula_bound"
            quality = "high"
            rigor = "L2:R"
        elif families == ["bond_order_perception"]:
            evidence_class = "single_engine_bond_order_perception"
            quality = "candidate"
            rigor = "L2:H"
        else:
            evidence_class = "geometry_only_bond_order_hypothesis"
            quality = "hypothesis"
            rigor = "L1:H"
        group = {
            "canonical_smiles": representative["canonical_smiles"],
            "full_inchikey": representative["full_inchikey"],
            "inchi_connectivity_block": connectivity_key,
            "inchi_nonprotonation_key": representative[
                "inchi_nonprotonation_key"
            ],
            "formal_charge": representative["formal_charge"],
            "heavy_atom_composition": source_composition,
            "supporting_engines": engines,
            "supporting_families": families,
            "engine_count": len(engines),
            "independent_family_count": len(families),
            "evidence_class": evidence_class,
            "quality": quality,
            "rigor": rigor,
            "qualified_chemistry": False,
            "full_inchikey_variants": full_keys,
            "canonical_smiles_variants": variants,
            "selected_engine": representative["engine"],
            "candidate_graph": representative.get("candidate_graph"),
            "mapping_audit": representative.get("mapping_audit"),
        }
        group["candidate_id"] = _candidate_id(group)
        identity_groups.append(group)

    def rank(group: dict[str, Any]) -> tuple[Any, ...]:
        return (
            -int(group["independent_family_count"]),
            min(
                _FAMILY_PRIORITY.get(family, 99)
                for family in group["supporting_families"]
            ),
            -int(group["engine_count"]),
            min(
                _ENGINE_PRIORITY.get(engine, 99)
                for engine in group["supporting_engines"]
            ),
            group["inchi_connectivity_block"],
            group["full_inchikey"],
        )

    identity_groups.sort(key=rank)
    selected = identity_groups[0] if identity_groups else None
    selection_tied = bool(
        selected
        and len(identity_groups) > 1
        and rank(identity_groups[0])[:-2] == rank(identity_groups[1])[:-2]
    )
    if selected is not None and selection_tied:
        selected = dict(selected)
        selected["quality"] = "candidate"
        selected["rigor"] = "L2:H"
        selected["evidence_class"] = (
            f"{selected['evidence_class']}_selection_tied"
        )

    public_groups = []
    for group in identity_groups:
        public_groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "candidate_graph"
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "parseable" if selected is not None else "not_supported",
        "source_heavy_atom_composition": source_composition,
        "candidate_count": len(identity_groups),
        "selected_candidate_id": (
            selected["candidate_id"] if selected is not None else None
        ),
        "selected_candidate": selected,
        "selection_tied": selection_tied,
        "identity_groups": public_groups,
        "engine_attempts": attempts,
        "openbabel_available": any(
            attempt["engine"] == "openbabel_pdb"
            and attempt["status"] in {"admitted", "rejected"}
            for attempt in attempts
        ),
        "qualified_chemistry": False,
    }


__all__ = ["SCHEMA_VERSION", "infer_bond_order_candidates"]
