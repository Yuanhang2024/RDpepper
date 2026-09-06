"""Unified-library-backed residue templates for the Path A/C/E family."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from typing import Iterable

from rdkit import Chem
from rdkit.Geometry import Point3D

from ..core.cyclization import _is_covalent_bond
from ..core.pdb_parser import parse_backbone, standard_pdb_atom_name_map
from . import _map_utils


_STANDARD_PDB_TO_SYMBOL = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# PDB identifiers are aliases only. Every molecular graph still comes from a
# manifest-selected Unified library row.
_PDB_TO_UNIFIED_SYMBOL = {
    **_STANDARD_PDB_TO_SYMBOL,
    "ACE": "ac", "NME": "nme", "NH2": "nh2",
    "DAL": "dA", "DSN": "dS", "DSG": "dN", "DAS": "dD",
    "DGL": "dE", "DTH": "dT", "DVA": "dV", "DLE": "dL",
    "DIL": "dI", "DPR": "dP", "DTR": "dW", "DTY": "dY",
    "DPN": "dF", "DAR": "dR", "DLY": "dK", "DHI": "dH",
    "DCY": "dC", "MED": "dM", "DGN": "dQ", "DBB": "dAbu",
    "ORN": "Orn", "ABU": "Abu", "AIB": "Aib", "NLE": "Nle",
    "NVA": "Nva", "SAR": "Sar", "HYP": "Hyp", "PCA": "Glp",
    "KYN": "Kyn", "M3G": "3MeGlu", "DHA": "Dha", "DHB": "dhB",
    "GGL": "gGlu",
}

for _alias_code, _alias_target in _map_utils._active_pdb_aliases.items():
    _built_in_target = _PDB_TO_UNIFIED_SYMBOL.get(_alias_code)
    if _built_in_target is not None and _built_in_target != _alias_target:
        raise ValueError(
            f"derived PDB alias {_alias_code!r} conflicts with built-in symbol "
            f"{_built_in_target!r}"
        )


@dataclass(frozen=True)
class ResidueTemplate:
    pdb_resname: str
    symbol: str
    smiles: str
    cxsmiles: str
    source: str
    r1: str
    r2: str
    r3: str
    free_smiles: str
    free_graph_sha256: str
    r3_anchor_index: int | None = None

    @property
    def mol(self) -> Chem.Mol:
        molecule = Chem.MolFromSmiles(self.smiles)
        if molecule is None:
            raise ValueError(f"Unified template {self.symbol!r} is not parseable")
        return molecule


@dataclass(frozen=True)
class CappedTemplate:
    smiles: str
    base: ResidueTemplate
    cap: ResidueTemplate
    base_to_combined: tuple[int, ...]
    cap_to_combined: tuple[int, ...]


def _row_for_symbol(symbol: str) -> dict:
    row = _map_utils._active_user_rows.get(symbol)
    if row is None:
        row = _map_utils._unified_by_symbol.get(symbol)
    if row is None:
        folded = [
            value for key, value in {
                **_map_utils._unified_by_symbol,
                **_map_utils._active_user_rows,
            }.items()
            if str(key).casefold() == symbol.casefold()
        ]
        if len(folded) == 1:
            row = folded[0]
    if row is None:
        raise ValueError(f"No Unified monomer for {symbol}")
    return row


def _normalize_port_labels(editable: Chem.RWMol, symbol: str) -> None:
    """Move anchor-side CX labels onto their adjacent attachment dummy."""
    for atom in list(editable.GetAtoms()):
        if atom.GetAtomicNum() == 0 or not atom.HasProp("atomLabel"):
            continue
        label = atom.GetProp("atomLabel")
        if label not in {"_R1", "_R2", "_R3"}:
            continue
        dummies = [
            neighbor for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 0
            and not neighbor.HasProp("atomLabel")
        ]
        if len(dummies) != 1:
            raise ValueError(
                f"Unified monomer {symbol} has ambiguous anchor-labelled {label}"
            )
        dummies[0].SetProp("atomLabel", label)
        atom.ClearProp("atomLabel")


def _materialize_free_monomer_smiles(
    cxsmiles: str,
    r1: str,
    r2: str,
    r3: str,
) -> str:
    """Cap every CXSMILES port to recover the corresponding free monomer."""
    molecule = Chem.MolFromSmiles(cxsmiles)
    if molecule is None:
        raise ValueError("Unified CXSMILES is not parseable")
    editable = Chem.RWMol(molecule)
    _normalize_port_labels(editable, "<unresolved>")
    removals: list[int] = []
    defaults = {"_R1": r1, "_R2": r2, "_R3": r3}
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = atom.GetProp("atomLabel") if atom.HasProp("atomLabel") else ""
        cap = str(defaults.get(label, "-")).strip().upper()
        if cap in {"OH", "NH2", "SH"}:
            atom.SetAtomicNum({"OH": 8, "NH2": 7, "SH": 16}[cap])
            atom.SetAtomMapNum(0)
            atom.SetNoImplicit(False)
            if atom.HasProp("atomLabel"):
                atom.ClearProp("atomLabel")
        elif cap in {"", "-", "H"}:
            if cap == "H":
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) == 1:
                    anchor = neighbors[0]
                    explicit_h = anchor.GetNumExplicitHs()
                    if explicit_h > 0:
                        anchor.SetNumExplicitHs(explicit_h + 1)
                        anchor.SetNoImplicit(True)
            removals.append(atom.GetIdx())
        else:
            raise ValueError(f"unsupported free-monomer cap {label}={cap}")
    for index in sorted(removals, reverse=True):
        editable.RemoveAtom(index)
    free_molecule = editable.GetMol()
    free_molecule.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(free_molecule)
    return Chem.MolToSmiles(
        free_molecule, canonical=True, isomericSmiles=True
    )


def resolve_symbol(pdb_resname: str) -> str:
    name = str(pdb_resname).strip()
    mapped = _PDB_TO_UNIFIED_SYMBOL.get(name.upper())
    if mapped:
        return mapped
    active_alias = _map_utils.resolve_pdb_alias(name)
    if active_alias:
        return active_alias
    if name in _map_utils._active_user_rows:
        return name
    if name in _map_utils._unified_by_symbol:
        return name
    folded = [
        key for key in {
            **_map_utils._unified_by_symbol,
            **_map_utils._active_user_rows,
        }
        if str(key).casefold() == name.casefold()
    ]
    if len(folded) == 1:
        return str(folded[0])
    raise ValueError(f"No Unified symbol mapping for PDB residue {name}")


@lru_cache(maxsize=4096)
def _template_from_values(
    pdb_resname: str,
    symbol: str,
    cxsmiles: str,
    source: str,
    r1: str,
    r2: str,
    r3: str,
    free_smiles: str,
) -> ResidueTemplate:
    molecule = Chem.MolFromSmiles(cxsmiles)
    if molecule is None:
        raise ValueError(f"Unified CXSMILES for {symbol} is not parseable")
    editable = Chem.RWMol(molecule)
    # Some source rows place an _R label on the attachment atom while the
    # adjacent dummy is unlabeled. Normalize both encodings before ports are
    # materialized so R3 anchor provenance is not silently lost.
    _normalize_port_labels(editable, symbol)
    removals = []
    r3_anchor_original = None
    defaults = {"_R1": r1, "_R2": r2, "_R3": r3}
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = atom.GetProp("atomLabel") if atom.HasProp("atomLabel") else ""
        default = defaults.get(label, "-")
        if label == "_R3":
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                raise ValueError(
                    f"Unified monomer {symbol} has invalid {label} attachment"
                )
            r3_anchor_original = neighbors[0].GetIdx()
        if label == "_R2" and default.upper() in {"NH2", "SH"}:
            atom.SetAtomicNum({"NH2": 7, "SH": 16}[default.upper()])
            atom.SetAtomMapNum(0)
            atom.SetNoImplicit(False)
            atom.ClearProp("atomLabel")
            removals.extend(
                neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                if neighbor.GetAtomicNum() == 1
            )
        elif label == "_R3" and default.upper() == "OH":
            atom.SetAtomicNum(8)
            atom.SetNoImplicit(False)
            atom.ClearProp("atomLabel")
        elif label == "_R3" and default.upper() == "H":
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                raise ValueError(
                    f"Unified monomer {symbol} has invalid {label} attachment"
                )
            # Bracketed CXSMILES atoms can retain an explicit-H count after
            # their dummy is deleted.  Make H implicit so a later crosslink
            # substitutes it automatically instead of creating bad valence.
            anchor = neighbors[0]
            explicit_h = anchor.GetNumExplicitHs()
            if explicit_h > 0:
                anchor.SetNumExplicitHs(explicit_h + 1)
                anchor.SetNoImplicit(True)
            else:
                anchor.SetNumExplicitHs(0)
                anchor.SetNoImplicit(False)
            removals.append(atom.GetIdx())
        elif default.upper() in ("", "-", "H", "OH", "NH2", "SH"):
            removals.append(atom.GetIdx())
        else:
            raise ValueError(
                f"Unified monomer {symbol} has unsupported attachment default {label}={default}"
            )
    removals = sorted(set(removals))
    for index in reversed(removals):
        editable.RemoveAtom(index)
    molecule = editable.GetMol()
    molecule.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(molecule)
    r3_anchor_index = None
    if r3_anchor_original is not None:
        materialized_anchor = r3_anchor_original - sum(
            index < r3_anchor_original for index in removals
        )
        marked = Chem.Mol(molecule)
        marked.GetAtomWithIdx(materialized_anchor).SetAtomMapNum(9003)
        marked_smiles = Chem.MolToSmiles(
            marked, canonical=True, isomericSmiles=True
        )
        ordered = Chem.MolFromSmiles(marked_smiles)
        marked_atoms = [
            atom for atom in ordered.GetAtoms() if atom.GetAtomMapNum() == 9003
        ]
        if len(marked_atoms) != 1:
            raise ValueError(f"Unified monomer {symbol} lost its R3 anchor")
        r3_anchor_index = marked_atoms[0].GetIdx()
        for atom in ordered.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(
            ordered, canonical=False, isomericSmiles=True
        )
    else:
        smiles = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
    free_molecule = Chem.MolFromSmiles(free_smiles)
    if free_molecule is None:
        raise ValueError(f"Unified free monomer SMILES for {symbol} is not parseable")
    free_canonical = Chem.MolToSmiles(
        free_molecule, canonical=True, isomericSmiles=True
    )
    free_graph_sha256 = hashlib.sha256(
        free_canonical.encode("utf-8")
    ).hexdigest()
    return ResidueTemplate(
        pdb_resname, symbol, smiles, cxsmiles, source, r1, r2, r3,
        free_canonical, free_graph_sha256, r3_anchor_index,
    )


def _template_from_row(
    pdb_resname: str, symbol: str, row: dict
) -> ResidueTemplate:
    cxsmiles = str(row.get("CXSMILES", "")).strip()
    if not cxsmiles:
        raise ValueError(f"Unified monomer {symbol} lacks CXSMILES")
    source = str(row.get("source", "unified")).strip() or "unified"
    r1 = str(row.get("R1", "-")).strip() or "-"
    r2 = str(row.get("R2", "-")).strip() or "-"
    r3 = str(row.get("R3", "-")).strip() or "-"
    free_smiles = str(
        row.get("smiles_canonical")
        or row.get("smiles_original")
        or row.get("replaced_SMILES")
        or ""
    ).strip()
    if not free_smiles:
        free_smiles = _materialize_free_monomer_smiles(cxsmiles, r1, r2, r3)
    return _template_from_values(
        str(pdb_resname), symbol, cxsmiles, source, r1, r2, r3, free_smiles
    )


def get_residue_template_for_symbol(
    symbol: str, *, pdb_resname: str | None = None
) -> ResidueTemplate:
    """Materialize one active Unified row without requiring a PDB alias."""
    row = _row_for_symbol(symbol)
    return _template_from_row(pdb_resname or symbol, symbol, row)


def get_residue_template(pdb_resname: str) -> ResidueTemplate:
    symbol = resolve_symbol(pdb_resname)
    return get_residue_template_for_symbol(symbol, pdb_resname=pdb_resname)


def _cap_name_map(template: ResidueTemplate) -> dict[str, int]:
    molecule = template.mol
    if template.symbol == "nh2":
        return {"N": next(atom.GetIdx() for atom in molecule.GetAtoms())}
    if template.symbol == "nme":
        n_atom = next(atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "N")
        carbon = next(atom for atom in n_atom.GetNeighbors() if atom.GetSymbol() == "C")
        return {"N": n_atom.GetIdx(), "C": carbon.GetIdx()}
    if template.symbol == "ac":
        carbonyl = next(
            atom for atom in molecule.GetAtoms()
            if atom.GetSymbol() == "C" and any(
                neighbor.GetSymbol() == "O"
                and molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
                == Chem.BondType.DOUBLE
                for neighbor in atom.GetNeighbors()
            )
        )
        oxygen = next(neighbor for neighbor in carbonyl.GetNeighbors() if neighbor.GetSymbol() == "O")
        methyl = next(neighbor for neighbor in carbonyl.GetNeighbors() if neighbor.GetSymbol() == "C")
        return {"CH3": methyl.GetIdx(), "C": carbonyl.GetIdx(), "O": oxygen.GetIdx()}
    return {}


def _adjacency(molecule: Chem.Mol) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
        for atom in molecule.GetAtoms()
    )


def _normalize_mapping_override(mapping_override) -> dict[int, int]:
    if not isinstance(mapping_override, dict) or not mapping_override:
        raise ValueError("Atom mapping override must be a non-empty mapping")
    normalized = {}
    for serial, template_index in mapping_override.items():
        try:
            normalized_serial = int(serial)
            normalized_index = int(template_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Atom mapping override contains non-integer values") from exc
        if normalized_serial in normalized:
            raise ValueError("Atom mapping override contains duplicate serials")
        normalized[normalized_serial] = normalized_index
    return normalized


def _mapping_fingerprint(mapping: dict[int, int]) -> str:
    return hashlib.sha256(
        repr(
            tuple(
                sorted((int(serial), int(index)) for serial, index in mapping.items())
            )
        ).encode("utf-8")
    ).hexdigest()


def _mapping_matches_observed_graph(
    template: ResidueTemplate,
    pdb_atoms: list[dict],
    mapping: dict[int, int],
    *,
    observed_edges: Iterable[tuple[int, int]] | None = None,
) -> bool:
    molecule = template.mol
    if len(mapping) != len(pdb_atoms):
        return False
    serial_to_observed = {
        int(atom["num"]): index for index, atom in enumerate(pdb_atoms)
    }
    if set(mapping) != set(serial_to_observed):
        return False
    if set(mapping.values()) != set(range(molecule.GetNumAtoms())):
        return False
    observed_adj = [set() for _ in pdb_atoms]
    if observed_edges is None:
        for left in range(len(pdb_atoms)):
            for right in range(left + 1, len(pdb_atoms)):
                if _is_covalent_bond(pdb_atoms[left], pdb_atoms[right]):
                    observed_adj[left].add(right)
                    observed_adj[right].add(left)
    else:
        serial_to_index = {
            int(atom["num"]): index for index, atom in enumerate(pdb_atoms)
        }
        for left_serial, right_serial in observed_edges:
            left = serial_to_index.get(int(left_serial))
            right = serial_to_index.get(int(right_serial))
            if left is None or right is None or left == right:
                return False
            observed_adj[left].add(right)
            observed_adj[right].add(left)

    template_adj = _adjacency(molecule)
    n_idx, ca_idx, _cb_idx, c_idx, o_idx = parse_backbone(molecule)
    for name, template_index in (("N", n_idx), ("CA", ca_idx), ("C", c_idx), ("O", o_idx)):
        observed = [
            index for index, atom in enumerate(pdb_atoms)
            if atom["name"] == name
        ]
        if template_index is not None and len(observed) == 1:
            serial = int(pdb_atoms[observed[0]]["num"])
            if mapping.get(serial) != template_index:
                return False
    for observed_index, atom in enumerate(pdb_atoms):
        template_index = mapping[int(atom["num"])]
        template_atom = molecule.GetAtomWithIdx(template_index)
        if template_atom.GetSymbol().upper() != str(atom["elem"]).upper():
            return False
        if len(template_adj[template_index]) != len(observed_adj[observed_index]):
            return False
        for other_index, other_atom in enumerate(pdb_atoms):
            if other_index == observed_index:
                continue
            observed_edge = other_index in observed_adj[observed_index]
            other_template_index = mapping[int(other_atom["num"])]
            template_edge = other_template_index in template_adj[template_index]
            if observed_edge != template_edge:
                return False
    return True


def _generic_graph_mappings(
    template: ResidueTemplate,
    pdb_atoms: list[dict],
    limit: int | None = 64,
    *,
    observed_edges: Iterable[tuple[int, int]] | None = None,
) -> list[dict[int, int]]:
    molecule = template.mol
    if molecule.GetNumAtoms() != len(pdb_atoms):
        return []
    template_adj = _adjacency(molecule)
    observed_adj = [set() for _ in pdb_atoms]
    if observed_edges is None:
        for left in range(len(pdb_atoms)):
            for right in range(left + 1, len(pdb_atoms)):
                if _is_covalent_bond(pdb_atoms[left], pdb_atoms[right]):
                    observed_adj[left].add(right)
                    observed_adj[right].add(left)
    else:
        serial_to_index = {
            int(atom["num"]): index for index, atom in enumerate(pdb_atoms)
        }
        for left_serial, right_serial in observed_edges:
            left = serial_to_index.get(int(left_serial))
            right = serial_to_index.get(int(right_serial))
            if left is None or right is None or left == right:
                return []
            observed_adj[left].add(right)
            observed_adj[right].add(left)

    anchors = {}
    n_idx, ca_idx, _cb_idx, c_idx, o_idx = parse_backbone(molecule)
    for name, index in (("N", n_idx), ("CA", ca_idx), ("C", c_idx), ("O", o_idx)):
        observed = [i for i, atom in enumerate(pdb_atoms) if atom["name"] == name]
        if index is not None and len(observed) == 1:
            anchors[observed[0]] = index

    candidates: dict[int, list[int]] = {}
    for observed_index, atom in enumerate(pdb_atoms):
        if observed_index in anchors:
            candidates[observed_index] = [anchors[observed_index]]
            continue
        candidates[observed_index] = [
            template_index
            for template_index, template_atom in enumerate(molecule.GetAtoms())
            if template_atom.GetSymbol().upper() == str(atom["elem"]).upper()
            and len(template_adj[template_index]) == len(observed_adj[observed_index])
        ]
        if not candidates[observed_index]:
            return []

    order = sorted(
        range(len(pdb_atoms)),
        key=lambda index: (len(candidates[index]), -len(observed_adj[index]), pdb_atoms[index]["name"]),
    )
    results: list[dict[int, int]] = []

    def visit(position: int, assigned: dict[int, int], used: set[int]) -> None:
        if limit is not None and len(results) >= limit:
            return
        if position == len(order):
            results.append({pdb_atoms[i]["num"]: assigned[i] for i in assigned})
            return
        observed_index = order[position]
        for template_index in candidates[observed_index]:
            if template_index in used:
                continue
            compatible = True
            for other_observed, other_template in assigned.items():
                observed_edge = other_observed in observed_adj[observed_index]
                template_edge = other_template in template_adj[template_index]
                if observed_edge != template_edge:
                    compatible = False
                    break
            if not compatible:
                continue
            assigned[observed_index] = template_index
            used.add(template_index)
            visit(position + 1, assigned, used)
            used.remove(template_index)
            del assigned[observed_index]

    visit(0, {}, set())
    return results


def enumerate_residue_mapping_candidates(
    template: ResidueTemplate,
    pdb_atoms: Iterable[dict],
    external_serials: Iterable[int] = (),
    *,
    observed_edges: Iterable[tuple[int, int]] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Enumerate every legal residue graph mapping for diagnostics.

    Unlike the strict mapper, this function never selects a primary mapping.
    ``limit`` is an explicit diagnostic cap; pass ``None`` to enumerate all
    mappings. Each row includes template atom indices, external endpoint
    targets, and a deterministic candidate fingerprint.
    """
    atoms = list(pdb_atoms)
    external = {int(serial) for serial in external_serials}
    molecule = template.mol
    by_serial = {int(atom["num"]): atom for atom in atoms}
    mappings = _generic_graph_mappings(
        template, atoms, limit=limit, observed_edges=observed_edges
    )
    candidates = []
    for mapping in mappings:
        serial_mapping = {
            str(serial): int(template_index)
            for serial, template_index in sorted(mapping.items())
        }
        endpoint_targets = {
            str(serial): {
                "template_atom_index": int(mapping[serial]),
                "template_atom_name": str(
                    molecule.GetAtomWithIdx(int(mapping[serial])).GetSymbol()
                ),
                "pdb_atom_name": str(by_serial[serial].get("name", "")),
            }
            for serial in sorted(external)
            if serial in mapping and serial in by_serial
        }
        fingerprint_payload = (
            tuple(sorted(serial_mapping.items())),
            tuple(sorted(
                (key, tuple(sorted(value.items())))
                for key, value in endpoint_targets.items()
            )),
        )
        candidates.append({
            "serial_to_template_atom_index": serial_mapping,
            "external_endpoint_targets": endpoint_targets,
            "candidate_fingerprint": hashlib.sha256(
                repr(fingerprint_payload).encode("utf-8")
            ).hexdigest(),
        })
    return candidates


def map_pdb_atoms(
    template: ResidueTemplate,
    pdb_atoms: Iterable[dict],
    external_serials: Iterable[int] = (),
    *,
    observed_edges: Iterable[tuple[int, int]] | None = None,
    consumed_r3_serials: Iterable[int] = (),
) -> dict[int, int]:
    mapping, evidence = map_pdb_atoms_with_evidence(
        template,
        pdb_atoms,
        external_serials,
        observed_edges=observed_edges,
        consumed_r3_serials=consumed_r3_serials,
    )
    if not (
        evidence.get("mapping_complete") is True
        and evidence.get("mapping_injective") is True
        and evidence.get("template_mapping_complete") is True
        and evidence.get("mapping_unique") is True
        and evidence.get("external_attachment_mapping_unique") is True
    ):
        raise ValueError(
            "Incomplete atom mapping for Unified monomer "
            f"{template.symbol}; ambiguous mapping or missing mapping"
        )
    return mapping


def map_pdb_atoms_with_evidence(
    template: ResidueTemplate,
    pdb_atoms: Iterable[dict],
    external_serials: Iterable[int] = (),
    *,
    observed_edges: Iterable[tuple[int, int]] | None = None,
    consumed_r3_serials: Iterable[int] = (),
    mapping_override: dict[int, int] | None = None,
) -> tuple[dict[int, int], dict]:
    """Map observed atoms and expose the evidence needed for strict adjudication."""
    atoms = list(pdb_atoms)
    external = {int(serial) for serial in external_serials}
    source_bound_r3 = {int(serial) for serial in consumed_r3_serials}
    graph_sha256 = hashlib.sha256(template.smiles.encode("utf-8")).hexdigest()

    def evidence(
        mapping,
        method,
        candidate_count,
        deferred_serials=(),
        deferred_r3_atoms: dict[int, str] | None = None,
        deferred_r3_anchor_serials=(),
        mapping_candidates=None,
        mapping_candidates_truncated=False,
        external_attachment_mapping_unique=True,
        mapping_override=None,
    ):
        deferred = {int(serial) for serial in deferred_serials}
        deferred_r3 = dict(deferred_r3_atoms or {})
        template_atom_count = template.mol.GetNumAtoms()
        mapped_template_atom_count = len(set(mapping.values()))
        raw_unmapped_template_count = (
            template_atom_count - mapped_template_atom_count
        )
        effective_unmapped_template_count = (
            raw_unmapped_template_count - len(deferred_r3)
        )
        unassigned_stereocenters = [
            index
            for index, assignment in Chem.FindMolChiralCenters(
                template.mol,
                includeUnassigned=True,
                useLegacyImplementation=False,
            )
            if assignment == "?"
        ]
        coordinate_stereo = _coordinate_stereo_evidence(
            template,
            atoms,
            mapping,
            deferred_template_indices=deferred_r3,
        ) if (
            len(mapping) + len(deferred_r3) == template_atom_count
            and len(set(mapping.values())) == len(mapping)
        ) else {
            "passed": False,
            "reason": "incomplete_mapping_for_coordinate_stereochemistry",
        }
        return {
            "pdb_resname": template.pdb_resname,
            "unified_symbol": template.symbol,
            "unified_source": template.source,
            "monomer_graph_sha256": graph_sha256,
            "free_monomer_graph_sha256": template.free_graph_sha256,
            "rgroup_defaults": {
                "R1": template.r1, "R2": template.r2, "R3": template.r3,
            },
            "r3_anchor_template_atom_index": template.r3_anchor_index,
            "mapping_method": method,
            "observed_heavy_atom_count": len(atoms),
            "mapped_heavy_atom_count": len(mapping),
            "template_heavy_atom_count": template_atom_count,
            "mapped_template_heavy_atom_count": mapped_template_atom_count,
            "mapping_complete": len(mapping) + len(deferred) == len(atoms),
            "mapping_injective": len(set(mapping.values())) == len(mapping),
            "template_mapping_complete": (
                mapped_template_atom_count + len(deferred_r3)
                == template_atom_count
            ),
            "unmapped_template_heavy_atom_count": raw_unmapped_template_count,
            "effective_unmapped_template_heavy_atom_count": (
                effective_unmapped_template_count
            ),
            "deferred_consumed_r3_template_atoms": [
                {"template_atom_index": int(index), "atom_name": name}
                for index, name in sorted(deferred_r3.items())
            ],
            "deferred_consumed_r3_count": len(deferred_r3),
            "deferred_consumed_r3_anchor_serials": sorted(
                int(serial) for serial in deferred_r3_anchor_serials
            ),
            "source_bound_explicit_r3_attachment_serials": sorted(
                source_bound_r3
            ),
            "template_unassigned_stereocenter_indices": unassigned_stereocenters,
            "template_stereo_complete": not unassigned_stereocenters,
            "coordinate_stereochemistry": coordinate_stereo,
            "mapping_candidate_count": candidate_count,
            "mapping_unique": candidate_count == 1,
            "external_attachment_mapping_unique": bool(
                external_attachment_mapping_unique
            ),
            "mapping_candidates": list(mapping_candidates or []),
            "mapping_candidates_truncated": bool(mapping_candidates_truncated),
            "mapping_override": dict(mapping_override) if mapping_override else None,
            "deferred_terminal_oxt_serials": sorted(deferred),
            "deferred_terminal_oxt_count": len(deferred),
            "serial_to_template_atom_index": {
                str(serial): int(index) for serial, index in sorted(mapping.items())
            },
            "external_attachment_serials": sorted(external),
            "external_attachment_indices": {
                str(serial): int(mapping[serial])
                for serial in sorted(external) if serial in mapping
            },
        }

    if template.pdb_resname.upper() in _STANDARD_PDB_TO_SYMBOL:
        by_name = standard_pdb_atom_name_map(template.pdb_resname, template.smiles)
        mapped = {atom["num"]: by_name[atom["name"]] for atom in atoms if atom["name"] in by_name}
        if len(mapped) == len(atoms) and len(set(mapped.values())) == len(mapped):
            deferred_r3, anchor_serials = _deferred_consumed_standard_r3_atoms(
                template,
                mapped,
                by_name,
                source_bound_r3,
            )
            method = (
                "standard_pdb_atom_names_with_consumed_r3_leaving_atom"
                if deferred_r3 else "standard_pdb_atom_names"
            )
            return mapped, evidence(
                mapped,
                method,
                1,
                deferred_r3_atoms=deferred_r3,
                deferred_r3_anchor_serials=anchor_serials,
            )
        deferred_oxt = [
            int(atom["num"]) for atom in atoms
            if str(atom["name"]).strip().upper() == "OXT"
            and int(atom["num"]) not in mapped
        ]
        if (
            len(deferred_oxt) == 1
            and len(mapped) + 1 == len(atoms)
            and len(set(mapped.values())) == len(mapped)
            and len(mapped) == template.mol.GetNumAtoms()
            and str(template.r2).strip().upper() == "OH"
            and deferred_oxt[0] not in external
        ):
            return mapped, evidence(
                mapped,
                "standard_pdb_atom_names_with_deferred_terminal_oxt",
                1,
                deferred_oxt,
            )

    cap_map = _cap_name_map(template)
    if cap_map:
        mapped = {atom["num"]: cap_map[atom["name"]] for atom in atoms if atom["name"] in cap_map}
        if len(mapped) == len(atoms) and len(set(mapped.values())) == len(mapped):
            return mapped, evidence(mapped, "cap_pdb_atom_names", 1)

    if any("xyz" not in atom for atom in atoms):
        raise ValueError("PDB coordinates are required for dynamic template mapping")

    mappings = _generic_graph_mappings(
        template, atoms, limit=65, observed_edges=observed_edges
    )
    if not mappings:
        raise ValueError(f"No complete atom mapping for Unified monomer {template.symbol}")
    mapping_candidates_truncated = len(mappings) > 64
    retained_mappings = mappings[:64]
    override = (
        _normalize_mapping_override(mapping_override)
        if mapping_override is not None else None
    )
    if override is not None:
        if not _mapping_matches_observed_graph(
            template, atoms, override, observed_edges=observed_edges
        ):
            raise ValueError(
                f"Illegal atom mapping override for Unified monomer {template.symbol}"
            )
        selected = override
    else:
        selected = min(
            retained_mappings,
            key=lambda mapping: tuple(
                mapping[serial] for serial in sorted(mapping)
            ),
        )
    external_attachment_mapping_unique = True
    for serial in external:
        values = {mapping.get(serial) for mapping in retained_mappings}
        if mapping_candidates_truncated:
            external_attachment_mapping_unique = False
        elif len(values) > 1:
            external_attachment_mapping_unique = False
            break
    else:
        external_attachment_mapping_unique = (
            not mapping_candidates_truncated
            and external_attachment_mapping_unique
        )
    method = (
        "element_adjacency_audited_connectivity"
        if observed_edges is not None
        else "element_adjacency_geometry"
    )
    if override is not None:
        method = f"{method}_legal_override"
    return selected, evidence(
        selected,
        method,
        len(retained_mappings),
        mapping_candidates=[
            {
                "serial_to_template_atom_index": {
                    str(serial): int(index)
                    for serial, index in sorted(candidate.items())
                },
                "candidate_fingerprint": _mapping_fingerprint(candidate),
            }
            for candidate in retained_mappings
        ],
        mapping_candidates_truncated=mapping_candidates_truncated,
        external_attachment_mapping_unique=external_attachment_mapping_unique,
        mapping_override=(
            {
                "serial_to_template_atom_index": {
                    str(serial): int(index)
                    for serial, index in sorted(override.items())
                },
                "candidate_fingerprint": _mapping_fingerprint(override),
                "validated": True,
            }
            if override is not None else None
        ),
    )


def _deferred_consumed_standard_r3_atoms(
    template: ResidueTemplate,
    mapping: dict[int, int],
    atom_name_map: dict[str, int],
    source_bound_r3_serials: set[int],
) -> tuple[dict[int, str], list[int]]:
    residue_name = template.pdb_resname.strip().upper()
    leaving_name = {"ASP": "OD2", "GLU": "OE2"}.get(residue_name)
    if (
        leaving_name is None
        or str(template.r3).strip().upper() != "OH"
        or template.r3_anchor_index is None
    ):
        return {}, []
    leaving_index = atom_name_map.get(leaving_name)
    if leaving_index is None:
        return {}, []
    mapped_indices = set(mapping.values())
    missing_indices = set(range(template.mol.GetNumAtoms())) - mapped_indices
    if missing_indices != {int(leaving_index)}:
        return {}, []
    anchor_index = int(template.r3_anchor_index)
    anchor_serials = sorted(
        int(serial)
        for serial, template_index in mapping.items()
        if int(template_index) == anchor_index
    )
    if len(anchor_serials) != 1 or anchor_serials[0] not in source_bound_r3_serials:
        return {}, []
    bond = template.mol.GetBondBetweenAtoms(anchor_index, int(leaving_index))
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return {}, []
    return {int(leaving_index): leaving_name}, anchor_serials


def _row_polymer_heavy_atom_count(row: dict) -> int | None:
    value = str(row.get("HeavyAtomCount", "")).strip()
    try:
        count = int(float(value))
    except ValueError:
        return None
    if str(row.get("R2", "")).strip().upper() == "OH":
        count -= 1
    return count


def _element_signature(molecule: Chem.Mol) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(
        atom.GetSymbol().upper() for atom in molecule.GetAtoms()
    ).items()))


def _coordinate_stereo_evidence(
    template: ResidueTemplate,
    atoms: list[dict],
    mapping: dict[int, int],
    *,
    deferred_template_indices: Iterable[int] = (),
) -> dict:
    expected = Chem.Mol(template.mol)
    deferred = sorted(
        {int(index) for index in deferred_template_indices}, reverse=True
    )
    if deferred:
        if any(index < 0 or index >= expected.GetNumAtoms() for index in deferred):
            return {"passed": False, "reason": "invalid_deferred_template_atom"}
        editable = Chem.RWMol(expected)
        for index in deferred:
            editable.RemoveAtom(index)
        expected = editable.GetMol()
        old_to_new = {
            old_index: old_index - sum(
                deferred_index < old_index for deferred_index in deferred
            )
            for old_index in range(template.mol.GetNumAtoms())
            if old_index not in deferred
        }
        mapping = {
            serial: old_to_new[template_index]
            for serial, template_index in mapping.items()
            if template_index in old_to_new
        }
    expected_centers = dict(Chem.FindMolChiralCenters(
        expected,
        includeUnassigned=True,
        useLegacyImplementation=False,
    ))
    potential = list(Chem.FindPotentialStereo(expected))
    unspecified = [
        int(item.centeredOn) for item in potential
        if str(item.specified) != "Specified"
    ]
    if unspecified or any(value == "?" for value in expected_centers.values()):
        return {
            "passed": False,
            "reason": "unassigned_template_stereochemistry",
            "template_unassigned_stereo_indices": sorted(unspecified),
        }

    observed = Chem.Mol(expected)
    Chem.RemoveStereochemistry(observed)
    conformer = Chem.Conformer(observed.GetNumAtoms())
    by_serial = {int(atom["num"]): atom for atom in atoms}
    for serial, template_index in mapping.items():
        xyz = by_serial[int(serial)].get("xyz")
        if xyz is None or len(xyz) != 3:
            return {"passed": False, "reason": "missing_coordinate"}
        conformer.SetAtomPosition(int(template_index), Point3D(*map(float, xyz)))
    observed.RemoveAllConformers()
    observed.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(observed, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
    observed_centers = dict(Chem.FindMolChiralCenters(
        observed,
        includeUnassigned=True,
        useLegacyImplementation=False,
    ))
    expected_smiles = Chem.MolToSmiles(
        expected, canonical=True, isomericSmiles=True
    )
    observed_smiles = Chem.MolToSmiles(
        observed, canonical=True, isomericSmiles=True
    )
    passed = expected_smiles == observed_smiles
    return {
        "passed": passed,
        "reason": None if passed else "coordinate_stereochemistry_mismatch",
        "template_centers": {
            str(index): value for index, value in sorted(expected_centers.items())
        },
        "observed_centers": {
            str(index): value for index, value in sorted(observed_centers.items())
        },
        "template_polymer_smiles": expected_smiles,
        "observed_polymer_smiles": observed_smiles,
    }


def _distance_supported_bond_orders(left: dict, right: dict) -> set[Chem.BondType]:
    first = str(left["elem"]).upper()
    second = str(right["elem"]).upper()
    elements = frozenset((first, second))
    xyz_left = tuple(map(float, left["xyz"]))
    xyz_right = tuple(map(float, right["xyz"]))
    distance = sum(
        (xyz_left[index] - xyz_right[index]) ** 2 for index in range(3)
    ) ** 0.5
    names = {str(left.get("name", "")).upper(), str(right.get("name", "")).upper()}
    if elements == frozenset(("C", "O")) and names == {"C", "O"}:
        return (
            {Chem.BondType.DOUBLE}
            if 1.05 <= distance <= 1.38 else {Chem.BondType.SINGLE}
        )
    if elements == frozenset(("C", "N")):
        if distance <= 1.20:
            return {Chem.BondType.TRIPLE}
        if distance <= 1.29:
            return {Chem.BondType.DOUBLE}
        if distance <= 1.42:
            return {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}
    elif elements == frozenset(("C", "C")):
        if distance <= 1.24:
            return {Chem.BondType.TRIPLE}
        if distance <= 1.38:
            return {Chem.BondType.DOUBLE, Chem.BondType.AROMATIC}
        if distance <= 1.46:
            return {
                Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                Chem.BondType.AROMATIC,
            }
    elif elements == frozenset(("C", "O")):
        if distance <= 1.30:
            return {Chem.BondType.DOUBLE}
        if distance <= 1.38:
            return {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}
    return {Chem.BondType.SINGLE}


def _bond_order_geometry_evidence(
    template: ResidueTemplate,
    atoms: list[dict],
    mapping: dict[int, int],
) -> dict:
    by_serial = {int(atom["num"]): atom for atom in atoms}
    inverse = {int(index): int(serial) for serial, index in mapping.items()}
    mismatches = []
    ambiguous = []
    for bond in template.mol.GetBonds():
        left_index = bond.GetBeginAtomIdx()
        right_index = bond.GetEndAtomIdx()
        left = by_serial[inverse[left_index]]
        right = by_serial[inverse[right_index]]
        supported = _distance_supported_bond_orders(left, right)
        if len(supported) != 1:
            ambiguous.append({
                "pdb_serials": [int(left["num"]), int(right["num"])],
                "supported_bond_types": sorted(str(value) for value in supported),
            })
        if bond.GetBondType() not in supported:
            mismatches.append({
                "pdb_serials": [int(left["num"]), int(right["num"])],
                "template_bond_type": str(bond.GetBondType()),
                "supported_bond_types": sorted(str(value) for value in supported),
            })
    return {
        "passed": not mismatches and not ambiguous,
        "mismatches": mismatches,
        "ambiguous_bond_orders": ambiguous,
    }


def find_unified_residue_matches(
    pdb_resname: str,
    pdb_atoms: Iterable[dict],
    *,
    explicit_edges: Iterable[tuple[int, int]] = (),
    external_serials: Iterable[int] = (),
    r3_cap: str | None = None,
    observed_edges: Iterable[tuple[int, int]] | None = None,
    source_identity_symbol: str | None = None,
) -> tuple[list[dict], dict]:
    """Return strict base-Unified matches for one otherwise unknown residue.

    Count and element filters only reduce the search space. A released match
    still requires a unique full atom mapping, explicit-edge consistency,
    coordinate stereochemistry, and (when used) the exact R3 port semantics.
    """
    atoms = list(pdb_atoms)
    target_elements = tuple(sorted(Counter(
        str(atom["elem"]).upper() for atom in atoms
    ).items()))
    external = {int(serial) for serial in external_serials}
    explicit = {
        tuple(sorted((int(left), int(right))))
        for left, right in explicit_edges
    }
    matches = []
    prefiltered = 0
    rejected = Counter()
    for symbol, row in sorted(_map_utils._unified_by_symbol.items()):
        if source_identity_symbol is not None and str(symbol) != str(
            source_identity_symbol
        ):
            rejected["source_identity_symbol_filtered"] += 1
            continue
        reference = _map_utils._full_unified_reference_rows().get(symbol)
        source = str(
            row.get("source") or (reference or {}).get("source", "")
        ).strip()
        if source in {"local_structure_derived", "user_registered"}:
            continue
        monomer_type = str(
            row.get("Monomer_Type")
            or (reference or {}).get("Monomer_Type", "")
        ).strip().upper()
        if monomer_type and monomer_type != "BACKBONE":
            continue
        polymer_type = str(
            row.get("Polymer_Type")
            or (reference or {}).get("Polymer_Type", "")
        ).strip().upper()
        if polymer_type and polymer_type != "PEPTIDE":
            continue
        cxsmiles = str(
            row.get("CXSMILES") or (reference or {}).get("CXSMILES", "")
        )
        if "_R1" not in cxsmiles or "_R2" not in cxsmiles:
            continue
        estimated = _row_polymer_heavy_atom_count(reference or row)
        if estimated is not None and estimated != len(atoms):
            continue
        try:
            template = _template_from_row(pdb_resname, str(symbol), row)
        except (ValueError, RuntimeError):
            rejected["template_materialization"] += 1
            continue
        if template.mol.GetNumAtoms() != len(atoms):
            continue
        if _element_signature(template.mol) != target_elements:
            continue
        prefiltered += 1
        try:
            mapping, mapping_evidence = map_pdb_atoms_with_evidence(
                template, atoms, external, observed_edges=observed_edges
            )
        except ValueError:
            rejected["atom_mapping"] += 1
            continue
        if not all((
            mapping_evidence["mapping_complete"],
            mapping_evidence["mapping_injective"],
            mapping_evidence["template_mapping_complete"],
            mapping_evidence["mapping_unique"],
            mapping_evidence["external_attachment_mapping_unique"],
        )):
            rejected["nonunique_or_incomplete_mapping"] += 1
            continue
        if any(
            template.mol.GetBondBetweenAtoms(mapping[left], mapping[right]) is None
            for left, right in explicit
            if left in mapping and right in mapping
        ):
            rejected["explicit_edge_mismatch"] += 1
            continue
        if external:
            if len(external) != 1 or template.r3_anchor_index is None:
                rejected["r3_anchor_mismatch"] += 1
                continue
            serial = next(iter(external))
            if mapping.get(serial) != template.r3_anchor_index:
                rejected["r3_anchor_mismatch"] += 1
                continue
            if str(template.r3).strip().upper() != str(r3_cap or "").strip().upper():
                rejected["r3_cap_mismatch"] += 1
                continue
        bond_orders = _bond_order_geometry_evidence(template, atoms, mapping)
        if not bond_orders["passed"]:
            rejected["bond_order_geometry_mismatch"] += 1
            continue
        stereo = _coordinate_stereo_evidence(template, atoms, mapping)
        if not stereo["passed"]:
            rejected[str(stereo.get("reason") or "stereo_mismatch")] += 1
            continue
        molecule = Chem.MolFromSmiles(template.free_smiles)
        if molecule is None:
            rejected["free_graph_parse"] += 1
            continue
        full_inchikey = Chem.MolToInchiKey(molecule)
        port_identity = (
            f"{str(template.r3).strip().upper()}:{template.r3_anchor_index}"
            if template.r3_anchor_index is not None else "-"
        )
        matches.append({
            "template": replace(template, pdb_resname=str(pdb_resname)),
            "mapping": mapping,
            "identity": (full_inchikey, port_identity),
            "evidence": {
                **mapping_evidence,
                "monomer_id": str(row.get("monomer_id", "")),
                "full_inchikey": full_inchikey,
                "port_identity": port_identity,
                "explicit_intra_residue_edges": [
                    list(edge) for edge in sorted(explicit)
                ],
                "bond_order_geometry": bond_orders,
                "coordinate_stereochemistry": stereo,
            },
        })
    return matches, {
        "base_unified_rows_considered": len(_map_utils._unified_by_symbol),
        "source_identity_symbol": source_identity_symbol,
        "source_identity_constraint_applied": source_identity_symbol is not None,
        "signature_prefilter_match_count": prefiltered,
        "strict_match_count": len(matches),
        "rejection_counts": dict(sorted(rejected.items())),
    }


def mapped_free_smiles_for_r3(
    template: ResidueTemplate,
) -> str:
    """Mark the selected Unified R3 anchor in its canonical free monomer."""
    if template.r3_anchor_index is None:
        raise ValueError(f"Unified monomer {template.symbol} has no R3 anchor")
    polymer = template.mol
    free = Chem.MolFromSmiles(template.free_smiles)
    if free is None:
        raise ValueError(f"Unified monomer {template.symbol} free graph is invalid")
    mappings = free.GetSubstructMatches(polymer, useChirality=True, uniquify=True)
    if len(mappings) != 1:
        raise ValueError(
            f"Unified monomer {template.symbol} has no unique polymer-to-free mapping"
        )
    anchor = int(mappings[0][template.r3_anchor_index])
    marked = Chem.Mol(free)
    marked.GetAtomWithIdx(anchor).SetAtomMapNum(9003)
    return Chem.MolToSmiles(marked, canonical=True, isomericSmiles=True)


def compose_capped_template(pdb_resname: str, cap_resname: str) -> CappedTemplate:
    base = get_residue_template(pdb_resname)
    cap = get_residue_template(cap_resname)
    base_mol = base.mol
    cap_mol = cap.mol
    if cap.symbol == "ac":
        combined = Chem.RWMol(Chem.CombineMols(cap_mol, base_mol))
        cap_anchor = _cap_name_map(cap)["C"]
        base_anchor = parse_backbone(base_mol)[0] + cap_mol.GetNumAtoms()
    elif cap.symbol in {"nme", "nh2"}:
        combined = Chem.RWMol(Chem.CombineMols(base_mol, cap_mol))
        base_anchor = parse_backbone(base_mol)[3]
        cap_anchor = _cap_name_map(cap)["N"] + base_mol.GetNumAtoms()
    else:
        raise ValueError(f"Unsupported Unified cap {cap.symbol}")
    if base_anchor is None:
        raise ValueError(f"Unified monomer {base.symbol} lacks a peptide backbone anchor")
    combined.AddBond(int(base_anchor), int(cap_anchor), Chem.BondType.SINGLE)
    molecule = combined.GetMol()
    Chem.SanitizeMol(molecule)
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(smiles)
    if reparsed is None:
        raise ValueError("Capped Unified template did not round-trip")
    base_matches = reparsed.GetSubstructMatches(base_mol, useChirality=False)
    cap_matches = reparsed.GetSubstructMatches(cap_mol, useChirality=False)
    pair = next(
        (
            (base_match, cap_match)
            for base_match in base_matches
            for cap_match in cap_matches
            if set(base_match).isdisjoint(cap_match)
            and len(set(base_match) | set(cap_match)) == reparsed.GetNumAtoms()
        ),
        None,
    )
    if pair is None:
        raise ValueError("Could not recover capped-template atom provenance")
    return CappedTemplate(smiles, base, cap, tuple(pair[0]), tuple(pair[1]))


def cap_atom_name_map(template: ResidueTemplate) -> dict[str, int]:
    return _cap_name_map(template)
