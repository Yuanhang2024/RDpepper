"""Fail-closed use of chemical-component definitions embedded in mmCIF.

The coordinate projection intentionally emits legacy PDB for the established
reconstruction paths.  This module preserves the mmCIF chemical dictionary as
separate, hashable evidence so unknown peptide residues do not lose official
atom, bond-order, aromaticity, protonation, or stereochemistry information.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import gemmi
from rdkit import Chem


class EmbeddedChemCompError(ValueError):
    def __init__(self, code: str, message: str, *, rejected: bool = False):
        super().__init__(message)
        self.code = code
        self.rejected = rejected


def _value(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw in {"", ".", "?"}:
        return None
    try:
        return str(gemmi.cif.as_string(raw)).strip()
    except ValueError:
        return raw


def _rows(block: gemmi.cif.Block, category: str) -> list[dict[str, str | None]]:
    table = block.find_mmcif_category(category)
    if len(table) == 0:
        return []
    tags = [str(tag).lower() for tag in table.tags]
    return [
        {tag: _value(row[index]) for index, tag in enumerate(tags)}
        for row in table
    ]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _formula_formal_charge(formula: Any) -> str | None:
    """Parse the terminal CCD formula charge token (for example ``1``/``2-``)."""
    raw = _value(formula)
    if raw is None:
        return None
    tokens = raw.split()
    if len(tokens) < 2:
        return None
    token = tokens[-1]
    if not re.fullmatch(r"(?:[+-]?\d+|\d+[+-])", token):
        return None
    if token.endswith("-"):
        return str(-int(token[:-1]))
    if token.endswith("+"):
        return str(int(token[:-1]))
    return str(int(token))


def extract_embedded_chem_comp_templates(path: str | Path) -> dict[str, dict]:
    try:
        return _extract_embedded_chem_comp_templates(path)
    except EmbeddedChemCompError:
        raise
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_SCHEMA_INVALID",
            f"mmCIF chemical-component schema is invalid: {type(exc).__name__}: {exc}",
            rejected=True,
        ) from exc


def _extract_embedded_chem_comp_templates(path: str | Path) -> dict[str, dict]:
    """Extract serializable component rows without silently repairing them."""
    source = Path(path)
    try:
        source_payload = source.read_bytes()
        document = gemmi.cif.read_string(source_payload.decode("utf-8"))
        if len(document) != 1:
            raise ValueError(
                f"mmCIF chemical dictionary requires one data block, observed {len(document)}"
            )
        block = document[0]
    except (IndexError, OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise EmbeddedChemCompError(
            "MALFORMED_MMCIF_CHEM_COMP",
            f"mmCIF chemical components cannot be parsed: {type(exc).__name__}: {exc}",
            rejected=True,
        ) from exc

    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    metadata = {}
    for row in _rows(block, "_chem_comp."):
        component_id = _value(row.get("_chem_comp.id"))
        if component_id is None:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                "_chem_comp row lacks comp_id",
                rejected=True,
            )
        key = component_id.upper()
        if key in metadata:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                f"duplicate _chem_comp metadata for {key}",
                rejected=True,
            )
        metadata[key] = row

    atoms: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(block, "_chem_comp_atom."):
        component_id = _value(row.get("_chem_comp_atom.comp_id"))
        if component_id is None:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                "_chem_comp_atom row lacks comp_id",
                rejected=True,
            )
        atoms[component_id.upper()].append(row)

    bonds: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(block, "_chem_comp_bond."):
        component_id = _value(row.get("_chem_comp_bond.comp_id"))
        if component_id is None:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                "_chem_comp_bond row lacks comp_id",
                rejected=True,
            )
        bonds[component_id.upper()].append(row)

    templates = {}
    for component_id in sorted(set(metadata) | set(atoms) | set(bonds)):
        component_metadata = metadata.get(component_id) or {}
        explicit_charge = (
            component_metadata.get("_chem_comp.pdbx_formal_charge")
            or component_metadata.get("_chem_comp.formal_charge")
        )
        formula_charge = _formula_formal_charge(
            component_metadata.get("_chem_comp.formula")
        )
        snapshot = {
            "component_id": component_id,
            "source_input_sha256": source_sha256,
            "component_type": (metadata.get(component_id) or {}).get(
                "_chem_comp.type"
            ),
            "component_name": (metadata.get(component_id) or {}).get(
                "_chem_comp.name"
            ),
            "formula": (metadata.get(component_id) or {}).get(
                "_chem_comp.formula"
            ),
            "component_formal_charge": explicit_charge or formula_charge,
            "component_formal_charge_source": (
                "explicit_chem_comp_field" if explicit_charge is not None
                else "terminal_formula_token" if formula_charge is not None
                else None
            ),
            "atom_rows": atoms.get(component_id, []),
            "bond_rows": bonds.get(component_id, []),
        }
        snapshot["component_snapshot_sha256"] = _canonical_hash(snapshot)
        templates[component_id] = snapshot
    return templates


def _required(row: dict, field: str) -> str:
    value = _value(row.get(field))
    if value is None:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_SCHEMA_INCOMPLETE",
            f"embedded chemical component lacks {field}",
            rejected=True,
        )
    return value


def _formal_charge(row: dict) -> int:
    raw = _value(
        row.get("_chem_comp_atom.charge")
        or row.get("_chem_comp_atom.pdbx_formal_charge")
    )
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError as exc:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_SCHEMA_INVALID",
            f"invalid embedded formal charge: {raw}",
            rejected=True,
        ) from exc


def _bond_type(order: str, aromatic: str | None) -> Chem.BondType:
    normalized = order.strip().upper()
    if str(aromatic or "").strip().upper() == "Y" or normalized == "AROM":
        return Chem.BondType.AROMATIC
    mapping = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "QUAD": Chem.BondType.QUADRUPLE,
    }
    if normalized not in mapping:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_BOND_ORDER_NOT_SUPPORTED",
            f"unsupported embedded bond order: {order}",
        )
    return mapping[normalized]


def _assign_declared_stereochemistry(
    molecule: Chem.Mol, rows: list[dict], index_by_name: dict[str, int]
) -> None:
    targets = {}
    for row in rows:
        config = str(row.get("_chem_comp_atom.pdbx_stereo_config") or "N").upper()
        if config in {"R", "S"}:
            targets[_required(row, "_chem_comp_atom.atom_id")] = config
        elif config not in {"N", ".", "?", ""}:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_STEREO_CONFIG_INVALID",
                f"unsupported embedded atom stereo configuration: {config}",
                rejected=True,
            )

    for name in targets:
        molecule.GetAtomWithIdx(index_by_name[name]).SetChiralTag(
            Chem.ChiralType.CHI_TETRAHEDRAL_CW
        )
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    for name, desired in targets.items():
        atom = molecule.GetAtomWithIdx(index_by_name[name])
        observed = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        if observed != desired:
            atom.InvertChirality()
            Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
            observed = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        if observed != desired:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_STEREOCHEMISTRY_UNRESOLVED",
                f"declared {desired} configuration cannot be materialized for {name}",
            )

    unresolved = [
        index
        for index, assignment in Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False
        )
        if assignment == "?"
    ]
    if unresolved:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_STEREOCHEMISTRY_UNRESOLVED",
            f"embedded component has undeclared stereocenters: {unresolved}",
        )


def _assign_declared_bond_stereochemistry(
    molecule: Chem.Mol,
    index_by_name: dict[str, int],
    declarations: list[tuple[str, str, str]],
) -> None:
    """Materialize unambiguous CCD E/Z declarations on double bonds."""
    for left_name, right_name, desired in declarations:
        left_index = index_by_name[left_name]
        right_index = index_by_name[right_name]
        bond = molecule.GetBondBetweenAtoms(left_index, right_index)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_BOND_STEREOCHEMISTRY_NOT_SUPPORTED",
                f"{desired} declaration is not attached to a double bond: "
                f"{left_name}-{right_name}",
                rejected=True,
            )
        left_neighbors = [
            atom
            for atom in molecule.GetAtomWithIdx(left_index).GetNeighbors()
            if atom.GetIdx() != right_index and atom.GetAtomicNum() != 1
        ]
        right_neighbors = [
            atom
            for atom in molecule.GetAtomWithIdx(right_index).GetNeighbors()
            if atom.GetIdx() != left_index and atom.GetAtomicNum() != 1
        ]
        if len(left_neighbors) != 1 or len(right_neighbors) != 1:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_BOND_STEREOCHEMISTRY_UNRESOLVED",
                f"{desired} declaration lacks unique non-hydrogen stereo "
                f"references: {left_name}-{right_name}",
            )
        bond.SetStereoAtoms(
            left_neighbors[0].GetIdx(),
            right_neighbors[0].GetIdx(),
        )
        bond.SetStereo(
            Chem.BondStereo.STEREOE
            if desired == "E"
            else Chem.BondStereo.STEREOZ
        )
    if declarations:
        Chem.AssignStereochemistry(
            molecule, cleanIt=False, force=True
        )
    for left_name, right_name, desired in declarations:
        bond = molecule.GetBondBetweenAtoms(
            index_by_name[left_name],
            index_by_name[right_name],
        )
        expected = (
            Chem.BondStereo.STEREOE
            if desired == "E"
            else Chem.BondStereo.STEREOZ
        )
        if bond is None or bond.GetStereo() != expected:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_BOND_STEREOCHEMISTRY_UNRESOLVED",
                f"declared {desired} configuration could not be materialized "
                f"for {left_name}-{right_name}",
            )


def _component_molecule(component: dict) -> tuple[Chem.Mol, dict[str, int]]:
    atom_rows = list(component.get("atom_rows") or [])
    bond_rows = list(component.get("bond_rows") or [])
    if not atom_rows or not bond_rows:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_SCHEMA_INCOMPLETE",
            "embedded component requires atom and bond categories",
        )

    editable = Chem.RWMol()
    index_by_name = {}
    periodic = Chem.GetPeriodicTable()
    component_id = str(component.get("component_id") or "").strip().upper()
    if not component_id:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_SCHEMA_INVALID",
            "embedded component lacks component_id",
            rejected=True,
        )
    for row in atom_rows:
        if _required(row, "_chem_comp_atom.comp_id").upper() != component_id:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                "embedded atom row component identity mismatch",
                rejected=True,
            )
        name = _required(row, "_chem_comp_atom.atom_id")
        raw_element = _required(row, "_chem_comp_atom.type_symbol")
        element = raw_element[:1].upper() + raw_element[1:].lower()
        if name in index_by_name:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                f"duplicate embedded component atom id: {name}",
                rejected=True,
            )
        try:
            atomic_number = int(periodic.GetAtomicNumber(element))
        except RuntimeError as exc:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_ELEMENT_NOT_SUPPORTED",
                f"unsupported embedded component element: {element}",
            ) from exc
        if atomic_number <= 0:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_ELEMENT_NOT_SUPPORTED",
                f"unsupported embedded component element: {element}",
            )
        atom = Chem.Atom(atomic_number)
        atom.SetFormalCharge(_formal_charge(row))
        atom.SetProp("_ccd_atom_id", name)
        if str(row.get("_chem_comp_atom.pdbx_aromatic_flag") or "").upper() == "Y":
            atom.SetIsAromatic(True)
        index_by_name[name] = editable.AddAtom(atom)

    seen_bonds = set()
    bond_stereo_declarations = []
    for row in bond_rows:
        if _required(row, "_chem_comp_bond.comp_id").upper() != component_id:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                "embedded bond row component identity mismatch",
                rejected=True,
            )
        left = _required(row, "_chem_comp_bond.atom_id_1")
        right = _required(row, "_chem_comp_bond.atom_id_2")
        if left not in index_by_name or right not in index_by_name or left == right:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                f"embedded component bond has invalid endpoints: {left}-{right}",
                rejected=True,
            )
        edge = tuple(sorted((left, right)))
        if edge in seen_bonds:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                f"duplicate embedded component bond: {left}-{right}",
                rejected=True,
            )
        seen_bonds.add(edge)
        bond_stereo = str(
            row.get("_chem_comp_bond.pdbx_stereo_config") or "N"
        ).strip().upper()
        bond_type = _bond_type(
            _required(row, "_chem_comp_bond.value_order"),
            row.get("_chem_comp_bond.pdbx_aromatic_flag"),
        )
        if bond_stereo not in {"", ".", "?", "N", "E", "Z"}:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_BOND_STEREOCHEMISTRY_NOT_SUPPORTED",
                f"embedded bond stereochemistry is not supported: {bond_stereo}",
            )
        if bond_stereo in {"E", "Z"}:
            if bond_type != Chem.BondType.DOUBLE:
                raise EmbeddedChemCompError(
                    "MMCIF_CHEM_COMP_BOND_STEREOCHEMISTRY_NOT_SUPPORTED",
                    f"{bond_stereo} declaration requires a double bond: "
                    f"{left}-{right}",
                    rejected=True,
                )
            bond_stereo_declarations.append(
                (left, right, bond_stereo)
            )
        editable.AddBond(index_by_name[left], index_by_name[right], bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            bond = editable.GetBondBetweenAtoms(index_by_name[left], index_by_name[right])
            bond.SetIsAromatic(True)

    molecule = editable.GetMol()
    component_charge = _value(component.get("component_formal_charge"))
    declared_charge = None
    if component_charge is not None:
        try:
            declared_charge = int(component_charge)
        except ValueError as exc:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_SCHEMA_INVALID",
                f"invalid component formal charge: {component_charge}",
                rejected=True,
            ) from exc
        molecule.UpdatePropertyCache(strict=False)
        current_charge = Chem.GetFormalCharge(molecule)
        delta = declared_charge - current_charge
        candidates = []
        if delta > 0:
            candidates = [
                atom for atom in molecule.GetAtoms()
                if atom.GetFormalCharge() == 0
                and atom.GetAtomicNum() in {7, 8, 15, 16}
                and round(sum(
                    bond.GetBondTypeAsDouble() for bond in atom.GetBonds()
                )) == ({7: 4, 8: 3, 15: 5, 16: 5}[atom.GetAtomicNum()])
            ]
        elif delta < 0:
            candidates = [
                atom for atom in molecule.GetAtoms()
                if atom.GetFormalCharge() == 0
                and atom.GetAtomicNum() in {8, 16}
                and round(sum(
                    bond.GetBondTypeAsDouble() for bond in atom.GetBonds()
                )) == 1
                and not any(
                    neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
                )
            ]
        if delta and len(candidates) == abs(delta):
            assigned_charge = 1 if delta > 0 else -1
            for atom in candidates:
                atom.SetFormalCharge(assigned_charge)
    try:
        Chem.SanitizeMol(molecule)
    except (RuntimeError, ValueError) as exc:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_MATERIALIZATION_FAILED",
            f"embedded component cannot be sanitized: {type(exc).__name__}: {exc}",
        ) from exc
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_DISCONNECTED",
            "embedded component contains more than one connected component",
            rejected=True,
        )
    if declared_charge is not None:
        if Chem.GetFormalCharge(molecule) != declared_charge:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_FORMAL_CHARGE_MISMATCH",
                "component and atom-level formal charges disagree",
                rejected=True,
            )
    _assign_declared_stereochemistry(molecule, atom_rows, index_by_name)
    _assign_declared_bond_stereochemistry(
        molecule,
        index_by_name,
        bond_stereo_declarations,
    )
    return molecule, index_by_name


def _source_ported_cxsmiles(
    molecule: Chem.Mol,
    index_by_name: dict[str, int],
    *,
    r3_atom_name: str | None,
    r3_cap: str | None,
) -> tuple[str, dict[str, str]]:
    """Replace source-declared leaving groups with explicit R1/R2/R3 ports."""
    editable = Chem.RWMol(molecule)
    removals: set[int] = set()

    def add_dummy(anchor_index: int, label: str) -> None:
        dummy = Chem.Atom(0)
        dummy.SetNoImplicit(True)
        dummy.SetProp("atomLabel", label)
        dummy_index = editable.AddAtom(dummy)
        editable.AddBond(
            int(anchor_index), int(dummy_index), Chem.BondType.SINGLE
        )

    n_index = index_by_name["N"]
    n_atom = editable.GetAtomWithIdx(n_index)
    n_hydrogens = [
        atom.GetIdx()
        for atom in n_atom.GetNeighbors()
        if atom.GetAtomicNum() == 1
    ]
    if not n_hydrogens:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_PEPTIDE_BACKBONE_INVALID",
            "standalone component cannot establish an R1 leaving hydrogen",
            rejected=True,
        )
    removals.add(n_hydrogens[0])
    add_dummy(n_index, "_R1")

    carbon_index = index_by_name["C"]
    carbon = editable.GetAtomWithIdx(carbon_index)
    excluded = {index_by_name["CA"], index_by_name["O"]}
    r2_candidates = [
        atom
        for atom in carbon.GetNeighbors()
        if atom.GetIdx() not in excluded
        and editable.GetBondBetweenAtoms(
            carbon_index, atom.GetIdx()
        ).GetBondType()
        == Chem.BondType.SINGLE
        and atom.GetSymbol() in {"O", "N", "S"}
    ]
    if len(r2_candidates) != 1:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_PEPTIDE_BACKBONE_INVALID",
            "standalone component cannot establish one R2 cap",
            rejected=True,
        )
    r2_atom = r2_candidates[0]
    r2_element = r2_atom.GetSymbol()
    removals.update(
        neighbor.GetIdx()
        for neighbor in r2_atom.GetNeighbors()
        if neighbor.GetAtomicNum() == 1
    )
    r2_atom.SetAtomicNum(0)
    r2_atom.SetFormalCharge(0)
    r2_atom.SetNumExplicitHs(0)
    r2_atom.SetNoImplicit(True)
    r2_atom.SetAtomMapNum(0)
    r2_atom.SetProp("atomLabel", "_R2")

    ports = {
        "R1": "H",
        "R2": {"O": "OH", "N": "NH2", "S": "SH"}[r2_element],
        "R3": "-",
    }
    if r3_atom_name is not None:
        anchor_index = index_by_name[r3_atom_name]
        anchor = editable.GetAtomWithIdx(anchor_index)
        normalized_cap = str(r3_cap or "").upper()
        if normalized_cap == "H":
            hydrogens = [
                atom.GetIdx()
                for atom in anchor.GetNeighbors()
                if atom.GetAtomicNum() == 1
                and atom.GetIdx() not in removals
            ]
            if not hydrogens:
                raise EmbeddedChemCompError(
                    "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                    f"R3 atom {r3_atom_name} lacks a leaving hydrogen",
                    rejected=True,
                )
            removals.add(hydrogens[0])
            add_dummy(anchor_index, "_R3")
        elif normalized_cap == "OH":
            hydroxyls = [
                atom
                for atom in anchor.GetNeighbors()
                if atom.GetSymbol() == "O"
                and atom.GetIdx() != r2_atom.GetIdx()
                and editable.GetBondBetweenAtoms(
                    anchor_index, atom.GetIdx()
                ).GetBondType()
                == Chem.BondType.SINGLE
            ]
            if len(hydroxyls) != 1:
                raise EmbeddedChemCompError(
                    "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                    f"R3 atom {r3_atom_name} lacks one hydroxyl cap",
                    rejected=True,
                )
            hydroxyl = hydroxyls[0]
            removals.update(
                atom.GetIdx()
                for atom in hydroxyl.GetNeighbors()
                if atom.GetAtomicNum() == 1
            )
            hydroxyl.SetAtomicNum(0)
            hydroxyl.SetFormalCharge(0)
            hydroxyl.SetNumExplicitHs(0)
            hydroxyl.SetNoImplicit(True)
            hydroxyl.SetAtomMapNum(0)
            hydroxyl.SetProp("atomLabel", "_R3")
        else:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                f"unsupported standalone R3 cap {normalized_cap!r}",
            )
        ports["R3"] = normalized_cap

    for index in sorted(removals, reverse=True):
        editable.RemoveAtom(index)
    ported = editable.GetMol()
    ported.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(ported)
        ported = Chem.RemoveHs(ported)
        Chem.SetDoubleBondNeighborDirections(ported)
        cxsmiles = Chem.MolToCXSmiles(
            ported, isomericSmiles=True
        )
    except (RuntimeError, ValueError) as exc:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_MATERIALIZATION_FAILED",
            f"source-preserving port graph failed: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    reparsed = Chem.MolFromSmiles(cxsmiles)
    labels = [
        atom.GetProp("atomLabel")
        for atom in reparsed.GetAtoms()
        if atom.HasProp("atomLabel")
    ] if reparsed is not None else []
    expected = ["_R1", "_R2"] + (
        ["_R3"] if r3_atom_name is not None else []
    )
    if any(labels.count(label) != 1 for label in expected):
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
            "source-preserving CXSMILES lost a required port",
            rejected=True,
        )
    return cxsmiles, ports


def materialize_standalone_peptide_component(
    component: dict,
    *,
    r3_atom_name: str | None = None,
    r3_cap: str | None = None,
) -> dict:
    """Compile one standalone CCD component into audited peptide chemistry.

    This is the coordinate-free counterpart of
    :func:`resolve_embedded_peptide_component`.  The component's own declared
    heavy-atom graph is supplied as the observed graph, so bond order,
    formal charge, stereochemistry, and atom-name-specific R3 evidence remain
    source-bound.  No persistent monomer registry state is changed here.
    """
    molecule, _index_by_name = _component_molecule(component)
    heavy_atoms = [
        atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    observed_atoms = [
        {
            "num": int(index + 1),
            "name": atom.GetProp("_ccd_atom_id"),
            "elem": atom.GetSymbol(),
        }
        for index, atom in enumerate(heavy_atoms)
    ]
    serial_by_index = {
        atom.GetIdx(): index + 1
        for index, atom in enumerate(heavy_atoms)
    }
    observed_edges = [
        (
            serial_by_index[bond.GetBeginAtomIdx()],
            serial_by_index[bond.GetEndAtomIdx()],
        )
        for bond in molecule.GetBonds()
        if bond.GetBeginAtomIdx() in serial_by_index
        and bond.GetEndAtomIdx() in serial_by_index
    ]
    normalized_r3 = (
        str(r3_atom_name).strip().upper()
        if r3_atom_name is not None
        else None
    )
    normalized_cap = (
        str(r3_cap).strip().upper()
        if r3_cap is not None
        else None
    )
    if normalized_r3 is not None and normalized_cap is None:
        anchor = next(
            (
                atom
                for atom in molecule.GetAtoms()
                if atom.HasProp("_ccd_atom_id")
                and atom.GetProp("_ccd_atom_id").upper()
                == normalized_r3
            ),
            None,
        )
        if anchor is None:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                f"standalone component lacks R3 atom {normalized_r3}",
                rejected=True,
            )
        if any(
            neighbor.GetAtomicNum() == 1
            for neighbor in anchor.GetNeighbors()
        ):
            normalized_cap = "H"
        elif anchor.GetSymbol() == "C" and any(
            neighbor.GetSymbol() == "O"
            and molecule.GetBondBetweenAtoms(
                anchor.GetIdx(), neighbor.GetIdx()
            ).GetBondType()
            == Chem.BondType.SINGLE
            and any(
                hydrogen.GetAtomicNum() == 1
                for hydrogen in neighbor.GetNeighbors()
            )
            for neighbor in anchor.GetNeighbors()
        ):
            normalized_cap = "OH"
        else:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                f"standalone component cannot infer an R3 cap at "
                f"{normalized_r3}",
            )
    resolved = resolve_embedded_peptide_component(
        component,
        observed_atoms,
        observed_edges,
        r3_atom_name=normalized_r3,
        r3_cap=normalized_cap,
    )
    ported_cxsmiles, ports = _source_ported_cxsmiles(
        molecule,
        _index_by_name,
        r3_atom_name=normalized_r3,
        r3_cap=normalized_cap,
    )
    return {
        **resolved,
        "resolution_mode": "standalone_ccd_component",
        "r3_atom_name": normalized_r3,
        "r3_cap": normalized_cap,
        "ported_cxsmiles": ported_cxsmiles,
        "r_groups": ports,
    }


def resolve_embedded_peptide_component(
    component: dict,
    observed_atoms: Iterable[dict],
    observed_edges: Iterable[tuple[int, int]],
    *,
    r3_atom_name: str | None = None,
    r3_cap: str | None = None,
) -> dict:
    try:
        return _resolve_embedded_peptide_component(
            component,
            observed_atoms,
            observed_edges,
            r3_atom_name=r3_atom_name,
            r3_cap=r3_cap,
        )
    except EmbeddedChemCompError:
        raise
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_OBSERVED_INPUT_INVALID",
            f"embedded component inputs are invalid: {type(exc).__name__}: {exc}",
            rejected=True,
        ) from exc


def _resolve_embedded_peptide_component(
    component: dict,
    observed_atoms: Iterable[dict],
    observed_edges: Iterable[tuple[int, int]],
    *,
    r3_atom_name: str | None = None,
    r3_cap: str | None = None,
) -> dict:
    """Validate one embedded peptide component against the projected residue."""
    snapshot_hash = component.get("component_snapshot_sha256")
    snapshot_payload = {
        key: value for key, value in component.items()
        if key != "component_snapshot_sha256"
    }
    if (
        not isinstance(snapshot_hash, str)
        or _canonical_hash(snapshot_payload) != snapshot_hash
        or not _is_sha256(component.get("source_input_sha256"))
    ):
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_SOURCE_BINDING_INVALID",
            "embedded component snapshot/source binding is invalid",
            rejected=True,
        )
    component_type = " ".join(
        str(component.get("component_type") or "").upper().split()
    )
    if component_type not in {
        "L-PEPTIDE LINKING",
        "D-PEPTIDE LINKING",
        "PEPTIDE LINKING",
    }:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_NOT_PEPTIDE_LINKING",
            f"embedded component type is not peptide linking: {component_type or '<missing>'}",
        )

    molecule, index_by_name = _component_molecule(component)
    required_backbone = {"N", "CA", "C", "O"}
    if not required_backbone <= set(index_by_name):
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_PEPTIDE_BACKBONE_INVALID",
            "embedded peptide component lacks N/CA/C/O backbone atoms",
            rejected=True,
        )
    n_atom = molecule.GetAtomWithIdx(index_by_name["N"])
    ca_atom = molecule.GetAtomWithIdx(index_by_name["CA"])
    carbon = molecule.GetAtomWithIdx(index_by_name["C"])
    oxygen = molecule.GetAtomWithIdx(index_by_name["O"])
    backbone_bonds = [
        molecule.GetBondBetweenAtoms(n_atom.GetIdx(), ca_atom.GetIdx()),
        molecule.GetBondBetweenAtoms(ca_atom.GetIdx(), carbon.GetIdx()),
        molecule.GetBondBetweenAtoms(carbon.GetIdx(), oxygen.GetIdx()),
    ]
    r2_candidates = [
        neighbor for neighbor in carbon.GetNeighbors()
        if neighbor.GetIdx() not in {ca_atom.GetIdx(), oxygen.GetIdx()}
        and molecule.GetBondBetweenAtoms(
            carbon.GetIdx(), neighbor.GetIdx()
        ).GetBondType() == Chem.BondType.SINGLE
    ]
    r2_cap = r2_candidates[0] if len(r2_candidates) == 1 else None
    r2_hydrogen_count = (
        sum(neighbor.GetAtomicNum() == 1 for neighbor in r2_cap.GetNeighbors())
        if r2_cap is not None else 0
    )
    if (
        [bond.GetBondType() if bond is not None else None for bond in backbone_bonds]
        != [
            Chem.BondType.SINGLE,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
        ]
        or not any(neighbor.GetAtomicNum() == 1 for neighbor in n_atom.GetNeighbors())
        or r2_cap is None
        or r2_cap.GetSymbol() not in {"O", "N", "S"}
        or not (
            r2_hydrogen_count >= 1
            or (
                r2_cap.GetSymbol() in {"O", "S"}
                and r2_cap.GetFormalCharge() < 0
            )
        )
    ):
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_PEPTIDE_BACKBONE_INVALID",
            "embedded peptide component has invalid R1/R2 backbone semantics",
            rejected=True,
        )
    r2_cap_name = r2_cap.GetProp("_ccd_atom_id")
    atoms = list(observed_atoms)
    observed_by_name = {}
    serial_to_name = {}
    for atom in atoms:
        name = str(atom.get("name", "")).strip()
        if not name or name in observed_by_name:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_OBSERVED_ATOM_MISMATCH",
                f"observed residue atom names are not unique: {name}",
                rejected=True,
            )
        serial = int(atom["num"])
        if serial in serial_to_name:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_OBSERVED_ATOM_MISMATCH",
                f"observed residue serials are not unique: {serial}",
                rejected=True,
            )
        observed_by_name[name] = str(atom.get("elem", "")).upper()
        serial_to_name[serial] = name

    component_elements = {
        name: molecule.GetAtomWithIdx(index).GetSymbol().upper()
        for name, index in index_by_name.items()
        if molecule.GetAtomWithIdx(index).GetAtomicNum() != 1
    }
    observed_names = set(observed_by_name)
    component_names = set(component_elements)
    missing_component_names = component_names - observed_names
    if (
        observed_names - component_names
        or missing_component_names not in (set(), {r2_cap_name})
        or any(
            component_elements[name] != observed_by_name[name]
            for name in observed_names & component_names
        )
    ):
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_OBSERVED_ATOM_MISMATCH",
            "embedded component heavy atoms do not match the observed residue",
            rejected=True,
        )

    expected_edges = set()
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtom()
        right = bond.GetEndAtom()
        if left.GetAtomicNum() == 1 or right.GetAtomicNum() == 1:
            continue
        left_name = left.GetProp("_ccd_atom_id")
        right_name = right.GetProp("_ccd_atom_id")
        if left_name in observed_names and right_name in observed_names:
            expected_edges.add(tuple(sorted((left_name, right_name))))
    observed_name_edges = set()
    raw_observed_edges = [
        tuple(sorted((int(left), int(right))))
        for left, right in observed_edges
    ]
    if (
        any(left == right for left, right in raw_observed_edges)
        or len(raw_observed_edges) != len(set(raw_observed_edges))
    ):
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_OBSERVED_CONNECTIVITY_MISMATCH",
            "observed residue edges are duplicated or self-referential",
            rejected=True,
        )
    for left_serial, right_serial in raw_observed_edges:
        if int(left_serial) not in serial_to_name or int(right_serial) not in serial_to_name:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_OBSERVED_CONNECTIVITY_MISMATCH",
                "observed residue edge contains an unknown serial",
                rejected=True,
            )
        observed_name_edges.add(tuple(sorted((
            serial_to_name[int(left_serial)], serial_to_name[int(right_serial)]
        ))))
    if observed_name_edges != expected_edges:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_OBSERVED_CONNECTIVITY_MISMATCH",
            "embedded component connectivity does not match the observed residue",
            rejected=True,
        )

    try:
        free = Chem.RemoveHs(molecule)
        Chem.AssignStereochemistry(free, cleanIt=False, force=True)
        Chem.SetDoubleBondNeighborDirections(free)
        free_smiles = Chem.MolToSmiles(
            free, canonical=True, isomericSmiles=True
        )
        full_inchikey = Chem.MolToInchiKey(free)
    except (RuntimeError, ValueError) as exc:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_MATERIALIZATION_FAILED",
            f"embedded free component cannot be materialized: {type(exc).__name__}: {exc}",
        ) from exc
    if not full_inchikey:
        raise EmbeddedChemCompError(
            "MMCIF_CHEM_COMP_MATERIALIZATION_FAILED",
            "embedded free component did not yield an InChIKey",
        )
    mapped_smiles = None
    if r3_atom_name is not None:
        if r3_atom_name not in index_by_name:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                f"embedded component lacks R3 atom {r3_atom_name}",
                rejected=True,
            )
        source_anchor = molecule.GetAtomWithIdx(index_by_name[r3_atom_name])
        normalized_cap = str(r3_cap or "").upper()
        cap_valid = False
        if normalized_cap == "H":
            cap_valid = any(
                neighbor.GetAtomicNum() == 1
                for neighbor in source_anchor.GetNeighbors()
            )
        elif normalized_cap == "OH" and source_anchor.GetSymbol() == "C":
            hydroxyls = [
                neighbor for neighbor in source_anchor.GetNeighbors()
                if neighbor.GetSymbol() == "O"
                and molecule.GetBondBetweenAtoms(
                    source_anchor.GetIdx(), neighbor.GetIdx()
                ).GetBondType() == Chem.BondType.SINGLE
                and any(
                    hydrogen.GetAtomicNum() == 1
                    for hydrogen in neighbor.GetNeighbors()
                )
            ]
            cap_valid = len(hydroxyls) == 1
        if not cap_valid:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                f"embedded component does not support {normalized_cap or '<missing>'} substitution at {r3_atom_name}",
                rejected=True,
            )
        anchor = next(
            (
                atom for atom in free.GetAtoms()
                if atom.HasProp("_ccd_atom_id")
                and atom.GetProp("_ccd_atom_id") == r3_atom_name
            ),
            None,
        )
        if anchor is None:
            raise EmbeddedChemCompError(
                "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH",
                f"embedded component lost R3 atom {r3_atom_name}",
                rejected=True,
            )
        marked = Chem.Mol(free)
        marked.GetAtomWithIdx(anchor.GetIdx()).SetAtomMapNum(9003)
        mapped_smiles = Chem.MolToSmiles(
            marked, canonical=True, isomericSmiles=True
        )

    return {
        "free_smiles": free_smiles,
        "r3_mapped_smiles": mapped_smiles,
        "full_inchikey": full_inchikey,
        "source_input_sha256": component.get("source_input_sha256"),
        "component_snapshot_sha256": component.get("component_snapshot_sha256"),
        "component_id": component.get("component_id"),
        "component_type": component.get("component_type"),
        "observed_heavy_atom_count": len(observed_names),
        "free_heavy_atom_count": len(component_names),
        "polymer_leaving_heavy_atom_names": sorted(missing_component_names),
        "r2_cap_atom_name": r2_cap_name,
        "r2_cap_element": r2_cap.GetSymbol(),
        "r2_cap_explicit_hydrogen_count": r2_hydrogen_count,
        "observed_connectivity_exact": True,
        "stereochemistry_source": "_chem_comp_atom.pdbx_stereo_config",
        "bond_order_source": "_chem_comp_bond.value_order",
        "protonation_source": "embedded_explicit_hydrogen_graph",
    }
