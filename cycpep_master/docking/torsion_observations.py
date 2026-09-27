"""Structure parsing and torsion extraction for offline V4 prior builds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from .torsion_prior import build_query_keys, classify_macrocycle_topology


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
CAP_RESIDUES = frozenset({"ACE", "NME", "NH2"})
CHI_DEFINITIONS = {
    "ARG": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
    "ASN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "ASP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "HIS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "ND1"),
    ],
    "ILE": [
        ("N", "CA", "CB", "CG1"),
        ("CA", "CB", "CG1", "CD1"),
    ],
    "LEU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "LYS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
    "MET": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ],
    "PHE": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "PRO": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
    ],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "TYR": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "VAL": [("N", "CA", "CB", "CG1")],
}


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    alternate_location: str
    occupancy: float
    element: str
    xyz: tuple[float, float, float]
    line: str

    @property
    def residue_key(self) -> tuple[int, str, str]:
        return (
            self.residue_number,
            self.insertion_code,
            self.residue_name,
        )


def _atom_record(line: str) -> AtomRecord | None:
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
        return None
    try:
        serial = int(line[6:11])
        residue_number = int(line[22:26])
        occupancy = float(line[54:60] or 0.0)
        xyz = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in xyz):
        return None
    element = (
        line[76:78].strip()
        if len(line) >= 78
        else ""
    )
    if not element:
        element = "".join(
            character for character in line[12:16] if character.isalpha()
        )[:2].strip().title()
    return AtomRecord(
        serial=serial,
        name=line[12:16].strip().upper(),
        residue_name=line[17:20].strip().upper(),
        chain_id=line[21:22].strip(),
        residue_number=residue_number,
        insertion_code=line[26:27].strip(),
        alternate_location=line[16:17].strip(),
        occupancy=occupancy,
        element=element.title(),
        xyz=xyz,
        line=line,
    )


def _select_alternate_locations(
    atoms: list[AtomRecord],
) -> list[AtomRecord]:
    selected = {}
    for atom in atoms:
        key = (atom.residue_key, atom.name)
        prior = selected.get(key)
        rank = (
            atom.alternate_location not in {"", "A"},
            -atom.occupancy,
            atom.alternate_location,
            atom.serial,
        )
        if prior is None:
            selected[key] = (rank, atom)
            continue
        if rank < prior[0]:
            selected[key] = (rank, atom)
    return sorted(
        (value[1] for value in selected.values()),
        key=lambda atom: atom.serial,
    )


def _parse_models(
    lines: list[str],
    chain_id: str,
) -> tuple[dict[int, list[AtomRecord]], list[str], list[str]]:
    models: dict[int, list[AtomRecord]] = {}
    model_id = 1
    explicit_models = False
    conect = []
    links = []
    for line in lines:
        if line.startswith("MODEL"):
            explicit_models = True
            try:
                model_id = int(line[10:14])
            except ValueError:
                model_id = len(models) + 1
            continue
        if line.startswith("ENDMDL"):
            continue
        if line.startswith("CONECT"):
            conect.append(line)
            continue
        if line.startswith("LINK"):
            links.append(line)
            continue
        atom = _atom_record(line)
        if atom is None or atom.chain_id != chain_id:
            continue
        models.setdefault(model_id, []).append(atom)
    if not explicit_models and models:
        models = {1: next(iter(models.values()))}
    return models, conect, links


def _filtered_pdb_block(
    atoms: list[AtomRecord],
    conect_lines: list[str],
    link_lines: list[str],
) -> str:
    serials = {atom.serial for atom in atoms}
    lines = [atom.line for atom in atoms]
    chain = atoms[0].chain_id if atoms else ""
    for line in link_lines:
        if len(line) >= 52 and (
            line[21:22].strip() == chain
            and line[51:52].strip() == chain
        ):
            lines.append(line)
    for line in conect_lines:
        fields = line.split()
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError:
            continue
        if len(values) >= 2 and all(
            value in serials for value in values
        ):
            lines.append(line)
    lines.append("END")
    return "\n".join(lines) + "\n"


def _molecule_from_atoms(
    atoms: list[AtomRecord],
    conect_lines: list[str],
    link_lines: list[str],
) -> tuple[Chem.Mol, dict[int, int]]:
    graph_atoms = [
        atom for atom in atoms if atom.element.upper() != "H"
    ]
    block = _filtered_pdb_block(
        graph_atoms, conect_lines, link_lines
    )
    molecule = Chem.MolFromPDBBlock(
        block,
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    if molecule is None:
        raise ValueError("RDKit could not parse isolated peptide chain")
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise ValueError(f"isolated peptide graph is invalid: {exc}") from exc
    if molecule.GetNumAtoms() != len(graph_atoms):
        raise ValueError(
            "isolated peptide atom count changed during RDKit parsing"
        )
    serial_to_index = {}
    for index, atom_record in enumerate(graph_atoms):
        atom = molecule.GetAtomWithIdx(index)
        info = atom.GetPDBResidueInfo()
        serial = (
            info.GetSerialNumber()
            if info is not None
            else atom_record.serial
        )
        serial_to_index[int(serial)] = index
        atom.SetProp("_TriposAtomName", atom_record.name)
        atom.SetProp("_TriposResidueName", atom_record.residue_name)
        atom.SetProp("_TriposChainId", atom_record.chain_id)
        atom.SetIntProp(
            "_TriposResidueNumber", atom_record.residue_number
        )
    if set(serial_to_index) != {
        atom.serial for atom in graph_atoms
    }:
        raise ValueError("PDB serial-to-RDKit mapping is incomplete")
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey:
        raise ValueError("isolated peptide did not produce a full InChIKey")
    return molecule, serial_to_index


def _molecule_from_path_a(
    path: Path,
    chain_id: str,
    atoms: list[AtomRecord],
) -> tuple[Chem.Mol, dict[int, int]]:
    from ..paths.path_a import _build_combo

    combo, serial_to_index = _build_combo(
        str(path),
        chain_id,
        geometric_cyclization=False,
    )
    molecule = combo.GetMol()
    Chem.SanitizeMol(molecule)
    records = {
        atom.serial: atom
        for atom in atoms
        if atom.element.upper() != "H"
    }
    if set(serial_to_index) != set(records):
        raise ValueError("Path A source-heavy-atom mapping is incomplete")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    assigned = set()
    for serial, atom_index in serial_to_index.items():
        record = records[serial]
        conformer.SetAtomPosition(atom_index, record.xyz)
        assigned.add(atom_index)
        atom = molecule.GetAtomWithIdx(atom_index)
        info = Chem.AtomPDBResidueInfo()
        info.SetSerialNumber(int(record.serial))
        info.SetName(f"{record.name:<4}"[:4])
        info.SetResidueName(record.residue_name)
        info.SetResidueNumber(int(record.residue_number))
        info.SetInsertionCode(record.insertion_code)
        info.SetChainId(record.chain_id)
        atom.SetPDBResidueInfo(info)
        atom.SetProp("_TriposAtomName", record.name)
        atom.SetProp("_TriposResidueName", record.residue_name)
        atom.SetProp("_TriposChainId", record.chain_id)
        atom.SetIntProp(
            "_TriposResidueNumber", record.residue_number
        )
    if len(assigned) != molecule.GetNumAtoms():
        raise ValueError("Path A coordinate mapping is incomplete")
    conformer.Set3D(True)
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey:
        raise ValueError("Path A peptide did not produce a full InChIKey")
    return molecule, {
        int(serial): int(index)
        for serial, index in serial_to_index.items()
    }


def _residue_atom_indices(
    molecule: Chem.Mol,
) -> tuple[list[tuple[int, str, str]], dict[tuple[int, str, str], dict[str, int]]]:
    residue_atoms: dict[
        tuple[int, str, str], dict[str, int]
    ] = {}
    order = []
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (
            int(info.GetResidueNumber()),
            info.GetInsertionCode().strip(),
            info.GetResidueName().strip().upper(),
        )
        if key not in residue_atoms:
            residue_atoms[key] = {}
            order.append(key)
        residue_atoms[key][info.GetName().strip().upper()] = atom.GetIdx()
    peptide_order = [
        key for key in order if key[2] not in CAP_RESIDUES
    ]
    return peptide_order, residue_atoms


def _has_bond(molecule: Chem.Mol, left: int, right: int) -> bool:
    return molecule.GetBondBetweenAtoms(left, right) is not None


def _topology(
    molecule: Chem.Mol,
    residue_order: list[tuple[int, str, str]],
    residue_atoms: dict[tuple[int, str, str], dict[str, int]],
) -> tuple[str, int | None, list[tuple[int, int]]]:
    if not residue_order:
        return "unknown", None, []
    position_by_atom = {}
    for position, key in enumerate(residue_order):
        for atom_index in residue_atoms.get(key, {}).values():
            position_by_atom[atom_index] = position
    closures = []
    head_to_tail = False
    disulfide = False
    other = False
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left not in position_by_atom or right not in position_by_atom:
            continue
        left_position = position_by_atom[left]
        right_position = position_by_atom[right]
        separation = abs(left_position - right_position)
        if separation <= 1:
            continue
        closures.append(tuple(sorted((left, right))))
        left_name = molecule.GetAtomWithIdx(left).GetPDBResidueInfo()
        right_name = molecule.GetAtomWithIdx(right).GetPDBResidueInfo()
        atom_names = {
            left_name.GetName().strip().upper(),
            right_name.GetName().strip().upper(),
        }
        if (
            {left_position, right_position}
            == {0, len(residue_order) - 1}
            and atom_names == {"N", "C"}
        ):
            head_to_tail = True
        elif atom_names == {"SG"}:
            disulfide = True
        else:
            other = True
    topology, ring_size = classify_macrocycle_topology(
        residue_count=len(residue_order),
        head_to_tail=head_to_tail,
        disulfide=disulfide,
        other=other,
        closure_residue_spans=[
            abs(position_by_atom[left] - position_by_atom[right]) + 1
            for left, right in closures
        ],
    )
    return topology, ring_size, closures


def _angle(
    conformer: Chem.Conformer,
    indices: tuple[int, int, int, int],
) -> float | None:
    if len(set(indices)) != 4:
        return None
    try:
        value = float(
            rdMolTransforms.GetDihedralDeg(conformer, *indices)
        )
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _observation(
    *,
    molecule: Chem.Mol,
    conformer: Chem.Conformer,
    indices: tuple[int, int, int, int],
    central_bond: tuple[int, int],
    torsion_name: str,
    topology_class: str,
    ring_size: int | None,
    common: dict[str, Any],
) -> dict[str, Any] | None:
    value = _angle(conformer, indices)
    if value is None:
        return None
    if not _has_bond(molecule, *central_bond):
        return None
    keys = build_query_keys(
        molecule,
        central_bond,
        topology_class=topology_class,
        macrocycle_ring_size=ring_size,
    )
    return {
        **common,
        "torsion_name": torsion_name,
        "torsion_kind": keys.torsion_kind,
        "angle_deg": value,
        "quartet_atom_indices": list(indices),
        "central_bond_atom_indices": list(central_bond),
        "exact_key": keys.exact,
        "residue_class_ring_key": keys.residue_class_ring,
        "morgan_key": keys.morgan,
        "generic_key": keys.generic,
    }


def parse_structure_file(
    path: str | Path,
    *,
    source_name: str,
    source_class: str,
    chain_id: str,
    expected_residue_count: int | None = None,
    expected_topology_class: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    payload = source.read_bytes()
    file_sha256 = hashlib.sha256(payload).hexdigest()
    lines = payload.decode("ascii", errors="replace").splitlines()
    models, conect, links = _parse_models(lines, chain_id)
    if not models:
        return {
            "status": "not_supported",
            "error": f"chain {chain_id!r} is absent",
            "file_path": str(source),
            "file_sha256": file_sha256,
            "observations": [],
            "structures": [],
        }
    observations = []
    structures = []
    errors = []
    for model_id, raw_atoms in sorted(models.items()):
        atoms = _select_alternate_locations(raw_atoms)
        try:
            if source_name == "AfCycDesign":
                molecule, _serial_to_index = _molecule_from_path_a(
                    source, chain_id, atoms
                )
                graph_route = "cycpep_path_a"
            else:
                molecule, _serial_to_index = _molecule_from_atoms(
                    atoms, conect, links
                )
                graph_route = "rdkit_pdb_proximity"
            residue_order, residue_atoms = _residue_atom_indices(
                molecule
            )
            if len(residue_order) < 3:
                raise ValueError("peptide has fewer than three residues")
            if (
                expected_residue_count is not None
                and len(residue_order) != int(expected_residue_count)
            ):
                raise ValueError(
                    "peptide residue count differs from source metadata: "
                    f"{len(residue_order)} != {expected_residue_count}"
                )
            sequence = "".join(
                AA3_TO_1.get(key[2], f"[{key[2]}]")
                for key in residue_order
            )
            topology_class, ring_size, closures = _topology(
                molecule, residue_order, residue_atoms
            )
            if expected_topology_class is not None:
                compatibility = {
                    "head_to_tail": {
                        "head_to_tail",
                        "mixed",
                    },
                    "disulfide": {
                        "disulfide",
                        "mixed",
                        "mixed_sidechain",
                    },
                    "isopeptide": {
                        "sidechain",
                        "mixed",
                        "mixed_sidechain",
                    },
                }
                allowed = compatibility.get(
                    expected_topology_class,
                    {expected_topology_class},
                )
                if topology_class not in allowed:
                    raise ValueError(
                        "explicit closure graph differs from source "
                        f"topology metadata: {topology_class} != "
                        f"{expected_topology_class}"
                    )
                if not topology_class.startswith("mixed"):
                    topology_class = expected_topology_class
            inchikey = Chem.MolToInchiKey(molecule)
            entity_key = inchikey
            structure_id = source.stem
            common = {
                "entity_key": entity_key,
                "full_inchikey": inchikey,
                "source_name": source_name,
                "source_class": source_class,
                "structure_id": structure_id,
                "model_id": int(model_id),
                "chain_id": chain_id,
                "sequence": sequence,
                "residue_count": len(residue_order),
                "topology_class": topology_class,
                "macrocycle_ring_size": ring_size,
                "file_path": str(source),
                "file_sha256": file_sha256,
                "graph_route": graph_route,
            }
            structures.append({
                **common,
                "atom_count": molecule.GetNumAtoms(),
                "heavy_atom_count": molecule.GetNumHeavyAtoms(),
                "formal_charge": int(Chem.GetFormalCharge(molecule)),
                "closure_bond_count": len(closures),
                "status": "success",
            })
            conformer = molecule.GetConformer()
            cyclic = topology_class in {"head_to_tail", "mixed"}
            for position, residue_key in enumerate(residue_order):
                atoms_by_name = residue_atoms[residue_key]
                previous_key = (
                    residue_order[position - 1]
                    if position > 0
                    else residue_order[-1] if cyclic else None
                )
                next_key = (
                    residue_order[position + 1]
                    if position + 1 < len(residue_order)
                    else residue_order[0] if cyclic else None
                )
                if previous_key is not None:
                    previous = residue_atoms[previous_key]
                    required = (
                        previous.get("C"),
                        atoms_by_name.get("N"),
                        atoms_by_name.get("CA"),
                        atoms_by_name.get("C"),
                    )
                    if all(value is not None for value in required):
                        row = _observation(
                            molecule=molecule,
                            conformer=conformer,
                            indices=required,
                            central_bond=(required[1], required[2]),
                            torsion_name=f"phi:{position + 1}",
                            topology_class=topology_class,
                            ring_size=ring_size,
                            common=common,
                        )
                        if row:
                            observations.append(row)
                if next_key is not None:
                    following = residue_atoms[next_key]
                    psi = (
                        atoms_by_name.get("N"),
                        atoms_by_name.get("CA"),
                        atoms_by_name.get("C"),
                        following.get("N"),
                    )
                    omega = (
                        atoms_by_name.get("CA"),
                        atoms_by_name.get("C"),
                        following.get("N"),
                        following.get("CA"),
                    )
                    if all(value is not None for value in psi):
                        row = _observation(
                            molecule=molecule,
                            conformer=conformer,
                            indices=psi,
                            central_bond=(psi[1], psi[2]),
                            torsion_name=f"psi:{position + 1}",
                            topology_class=topology_class,
                            ring_size=ring_size,
                            common=common,
                        )
                        if row:
                            observations.append(row)
                    if all(value is not None for value in omega):
                        row = _observation(
                            molecule=molecule,
                            conformer=conformer,
                            indices=omega,
                            central_bond=(omega[1], omega[2]),
                            torsion_name=f"omega:{position + 1}",
                            topology_class=topology_class,
                            ring_size=ring_size,
                            common=common,
                        )
                        if row:
                            observations.append(row)
                for chi_index, names in enumerate(
                    CHI_DEFINITIONS.get(residue_key[2], []), 1
                ):
                    indices = tuple(
                        atoms_by_name.get(name) for name in names
                    )
                    if not all(value is not None for value in indices):
                        continue
                    row = _observation(
                        molecule=molecule,
                        conformer=conformer,
                        indices=indices,
                        central_bond=(indices[1], indices[2]),
                        torsion_name=f"chi{chi_index}:{position + 1}",
                        topology_class=topology_class,
                        ring_size=ring_size,
                        common=common,
                    )
                    if row:
                        observations.append(row)
            for closure_index, (left, right) in enumerate(closures, 1):
                left_neighbors = sorted(
                    atom.GetIdx()
                    for atom in molecule.GetAtomWithIdx(left).GetNeighbors()
                    if atom.GetIdx() != right
                    and atom.GetAtomicNum() > 1
                )
                right_neighbors = sorted(
                    atom.GetIdx()
                    for atom in molecule.GetAtomWithIdx(right).GetNeighbors()
                    if atom.GetIdx() != left
                    and atom.GetAtomicNum() > 1
                )
                if not left_neighbors or not right_neighbors:
                    continue
                indices = (
                    left_neighbors[0],
                    left,
                    right,
                    right_neighbors[0],
                )
                row = _observation(
                    molecule=molecule,
                    conformer=conformer,
                    indices=indices,
                    central_bond=(left, right),
                    torsion_name=f"closure:{closure_index}",
                    topology_class=topology_class,
                    ring_size=ring_size,
                    common=common,
                )
                if row:
                    observations.append(row)
        except Exception as exc:
            errors.append({
                "model_id": int(model_id),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return {
        "status": (
            "success"
            if structures and not errors
            else "partial" if structures
            else "not_supported"
        ),
        "error": errors or None,
        "file_path": str(source),
        "file_sha256": file_sha256,
        "observations": observations,
        "structures": structures,
    }
