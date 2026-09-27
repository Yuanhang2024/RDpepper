"""Prospective family-aware fail-closed cyclic-peptide reconstruction.

This module is additive.  ``remediation_v3`` remains unchanged so historical
benchmark results keep their original decision rule.  The v5 entry point is
intended for newly frozen evaluations whose protocol binds this source before
target execution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from rdkit import Chem

from . import __version__
from .chemical_audit import audit_pdb_file
from .core.cyclization import _is_covalent_bond, detect_cyclization
from .core.identity_memo import identity_memo_context, molecular_identity
from .core.pdb_parser import standard_pdb_atom_name_map
from .paths.residue_template_factory import get_residue_template
from .remediation_v3 import reconstruct_pdb_fail_closed


ROUTE_FAMILIES = {
    "a": "residue_template",
    "c": "residue_template",
    "e": "residue_template",
    "b": "monomer_library",
    "g": "monomer_library",
    "f": "coordinate_heuristic",
    "h": "coordinate_heuristic",
}
QUALIFYING_FAMILIES = frozenset({"residue_template", "monomer_library"})
ROUTE_ORDER = tuple("abcefgh")
_SKIP_RESIDUES = {
    "HOH",
    "WAT",
    "DOD",
    "NA",
    "CL",
    "K",
    "MG",
    "CA",
    "ZN",
    "SO4",
    "PO4",
}


def _expected_standard_atom_element(
    residue_name: str,
    atom_name: str,
) -> str | None:
    """Return the authoritative element for a named standard-residue atom."""
    residue_name = str(residue_name).strip().upper()
    atom_name = str(atom_name).strip().upper()
    # OXT has an unambiguous PDB element independent of residue-template
    # availability, including for otherwise unknown monomers.
    if atom_name == "OXT":
        return "O"
    try:
        template = get_residue_template(residue_name)
    except ValueError:
        return None
    atom_name_map = standard_pdb_atom_name_map(residue_name, template.smiles)
    atom_index = atom_name_map.get(atom_name)
    if atom_index is not None:
        return template.mol.GetAtomWithIdx(atom_index).GetSymbol().upper()
    return None


def _standard_residue_geometry_conflicts(
    residue_inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find observed standard-residue template bonds contradicted by coordinates."""
    conflicts: list[dict[str, Any]] = []
    for residue in residue_inventory:
        residue_name = str(residue["residue_name"]).upper()
        try:
            residue_template = get_residue_template(residue_name)
        except ValueError:
            continue
        template_smiles = residue_template.smiles
        template = Chem.MolFromSmiles(template_smiles)
        atom_name_map = standard_pdb_atom_name_map(residue_name, template_smiles)
        if template is None or not atom_name_map:
            continue
        name_by_template_index = {
            template_index: atom_name
            for atom_name, template_index in atom_name_map.items()
        }
        observed = {
            str(atom["atom_name"]).upper(): atom
            for atom in residue["atoms"]
        }
        for bond in template.GetBonds():
            atom_name_1 = name_by_template_index.get(bond.GetBeginAtomIdx())
            atom_name_2 = name_by_template_index.get(bond.GetEndAtomIdx())
            if atom_name_1 not in observed or atom_name_2 not in observed:
                continue
            atom_1 = observed[atom_name_1]
            atom_2 = observed[atom_name_2]
            if _is_covalent_bond(
                {"elem": atom_1["element"], "xyz": atom_1["xyz"]},
                {"elem": atom_2["element"], "xyz": atom_2["xyz"]},
            ):
                continue
            conflicts.append(
                {
                    "position": residue["position"],
                    "residue_number": residue["residue_number"],
                    "residue_name": residue_name,
                    "atom_1": atom_name_1,
                    "atom_2": atom_name_2,
                }
            )
    return conflicts


def _audit_chem_comp_evidence(
    pdb_path: str | Path,
    residue_inventory: list[dict[str, Any]],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate the narrow evidence contract used to override geometry conflicts."""
    import hashlib

    audit: dict[str, Any] = {
        "schema_version": "chem-comp-evidence-1",
        "status": "not_present",
        "covered_conflicts": [],
        "uncovered_conflicts": [],
        "stereo_centers_audit": {
            "declared_count": 0,
            "bound_count": 0,
            "unbound_count": 0,
            "ambiguous_count": 0,
            "conflict_count": 0,
            "coverage_complete": False,
            "failure_reason": None,
        },
        "error": None,
    }
    if evidence is None:
        return audit
    if not isinstance(evidence, dict):
        audit.update(status="rejected", error="evidence must be an object")
        return audit
    audit["schema_version"] = evidence.get("schema_version")
    if evidence.get("schema_version") != "chem-comp-evidence-1":
        audit.update(status="rejected", error="unsupported evidence schema")
        return audit
    coordinate_hash = evidence.get("coordinate_projection_sha256")
    observed_hash = hashlib.sha256(Path(pdb_path).read_bytes()).hexdigest()
    if coordinate_hash != observed_hash:
        audit.update(
            status="rejected",
            error="coordinate projection hash does not match the strict input",
        )
        return audit
    source_hash = evidence.get("source_payload_sha256")
    snapshot_hash = evidence.get("component_snapshot_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash.lower())
        or not isinstance(snapshot_hash, str)
        or len(snapshot_hash) != 64
        or any(char not in "0123456789abcdef" for char in snapshot_hash.lower())
    ):
        audit.update(status="rejected", error="source or component hash is invalid")
        return audit
    if evidence.get("bond_order_source") != "_chem_comp_bond.value_order":
        audit.update(status="rejected", error="bond-order source is not authoritative")
        return audit
    if evidence.get("stereo_source") != "_chem_comp_atom.pdbx_stereo_config":
        audit.update(status="rejected", error="stereo source is not authoritative")
        return audit

    selected_atoms = {
        (int(atom["serial"]), str(residue["residue_name"]).upper(),
         int(residue["residue_number"]), str(atom["atom_name"]).upper()):
        str(atom["element"]).upper()
        for residue in residue_inventory
        for atom in residue["atoms"]
    }
    evidence_atoms = evidence.get("atom_records")
    if not isinstance(evidence_atoms, list):
        audit.update(status="rejected", error="atom_records are required")
        return audit
    supplied_atoms = {}
    for row in evidence_atoms:
        if not isinstance(row, dict):
            audit.update(status="rejected", error="atom_records contain a malformed row")
            return audit
        try:
            key = (
                int(row["serial"]), str(row["residue_name"]).upper(),
                int(row["residue_number"]), str(row["atom_name"]).upper(),
            )
            element = str(row["element"]).upper()
        except (KeyError, TypeError, ValueError):
            audit.update(status="rejected", error="atom_records contain invalid identity")
            return audit
        if key in supplied_atoms or not element:
            audit.update(status="rejected", error="atom_records are not unique")
            return audit
        supplied_atoms[key] = element
    if supplied_atoms != selected_atoms:
        audit.update(status="rejected", error="atom_records do not match the selected chain")
        return audit

    pair_rows = evidence.get("bond_order_pairs")
    if not isinstance(pair_rows, list):
        audit.update(status="rejected", error="bond_order_pairs are required")
        return audit
    serials = {key[0] for key in selected_atoms}
    pairs = set()
    for row in pair_rows:
        try:
            left, right, order = int(row[0]), int(row[1]), str(row[2]).upper()
        except (IndexError, TypeError, ValueError):
            audit.update(status="rejected", error="bond_order_pairs contain an invalid row")
            return audit
        if left == right or left not in serials or right not in serials or order not in {"SING", "DOUB", "TRIP", "QUAD", "AROM"}:
            audit.update(status="rejected", error="bond_order_pairs contain an invalid endpoint")
            return audit
        pairs.add(tuple(sorted((left, right))))

    stereo_summary = audit["stereo_centers_audit"]
    stereo_rows = evidence.get("stereo_centers")
    if stereo_rows is None:
        # Absent stereo_centers is legitimate for achiral components; treat as
        # empty so the stereo audit is skipped and the V6 stereochemistry
        # dimension (which still requires authoritative stereo for chiral
        # centers) makes the fail-closed decision. Only a present-but-malformed
        # value is rejected here.
        stereo_rows = []
    if not isinstance(stereo_rows, list):
        stereo_summary["failure_reason"] = "stereo_centers must be a list when present"
        audit.update(status="rejected", error="stereo_centers must be a list when present")
        return audit
    seen_stereo: dict[int, str] = {}
    for row in stereo_rows:
        stereo_summary["declared_count"] += 1
        try:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError("stereo row must contain serial and R/S")
            serial = row[0]
            config = row[1]
            if type(serial) is not int:
                raise ValueError("stereo serial must be an integer")
            config = str(config).upper()
        except (TypeError, ValueError):
            stereo_summary["conflict_count"] += 1
            stereo_summary["failure_reason"] = "stereo_centers contain an invalid row"
            audit.update(status="rejected", error=stereo_summary["failure_reason"])
            return audit
        if config not in {"R", "S"}:
            stereo_summary["conflict_count"] += 1
            stereo_summary["failure_reason"] = "stereo_centers contain an invalid R/S value"
            audit.update(status="rejected", error=stereo_summary["failure_reason"])
            return audit
        if serial not in serials:
            stereo_summary["unbound_count"] += 1
            stereo_summary["failure_reason"] = "stereo center serial is outside the selected chain"
            audit.update(status="rejected", error=stereo_summary["failure_reason"])
            return audit
        previous = seen_stereo.get(serial)
        if previous is not None:
            if previous == config:
                stereo_summary["ambiguous_count"] += 1
                stereo_summary["failure_reason"] = "stereo center serial is duplicated"
            else:
                stereo_summary["conflict_count"] += 1
                stereo_summary["failure_reason"] = "stereo center serial has conflicting R/S declarations"
            audit.update(status="rejected", error=stereo_summary["failure_reason"])
            return audit
        seen_stereo[serial] = config
        stereo_summary["bound_count"] += 1

    stereo_summary["coverage_complete"] = (
        stereo_summary["declared_count"] == stereo_summary["bound_count"]
        and stereo_summary["ambiguous_count"] == 0
        and stereo_summary["conflict_count"] == 0
        and stereo_summary["unbound_count"] == 0
    )
    if not stereo_summary["coverage_complete"]:
        stereo_summary["failure_reason"] = "stereo center coverage is incomplete"
        audit.update(status="rejected", error=stereo_summary["failure_reason"])
        return audit

    audit["_stereo_centers"] = dict(seen_stereo)
    audit["stereo_declared_count"] = stereo_summary["declared_count"]
    audit["stereo_bound_count"] = stereo_summary["bound_count"]
    audit["stereo_unbound_count"] = stereo_summary["unbound_count"]
    audit["stereo_ambiguous_count"] = stereo_summary["ambiguous_count"]
    audit["stereo_conflict_count"] = stereo_summary["conflict_count"]
    audit["stereo_coverage_complete"] = stereo_summary["coverage_complete"]
    audit["stereo_failure_reason"] = stereo_summary["failure_reason"]

    serial_by_identity = {
        (residue["residue_name"].upper(), int(residue["residue_number"]),
         str(atom["atom_name"]).upper()): int(atom["serial"])
        for residue in residue_inventory for atom in residue["atoms"]
    }
    conflicts = evidence.get("geometry_conflicts")
    if not isinstance(conflicts, list):
        # The strict validator computes the conflict list; this field is only
        # an optional caller receipt and is deliberately not trusted.
        conflicts = []
    audit["source_payload_sha256"] = source_hash
    audit["component_snapshot_sha256"] = snapshot_hash
    audit["coordinate_projection_sha256"] = observed_hash
    audit["authoritative_bond_count"] = len(pairs)
    audit["status"] = "validated"
    audit["_pairs"] = pairs
    audit["_serial_by_identity"] = serial_by_identity
    return audit


def _evidence_covers_conflict(conflict: dict[str, Any], audit: dict[str, Any]) -> bool:
    if audit.get("status") != "validated":
        return False
    key_left = (
        str(conflict["residue_name"]).upper(), int(conflict["residue_number"]),
        str(conflict["atom_1"]).upper(),
    )
    key_right = (
        str(conflict["residue_name"]).upper(), int(conflict["residue_number"]),
        str(conflict["atom_2"]).upper(),
    )
    serials = audit.get("_serial_by_identity", {})
    left, right = serials.get(key_left), serials.get(key_right)
    return left is not None and right is not None and tuple(sorted((left, right))) in audit.get("_pairs", set())


@dataclass
class StrictReconstructionResult:
    status: str
    support_status: str = "unknown"
    output_smiles: str | None = None
    output_inchikey: str | None = None
    rejection_reason: str | None = None
    warning_codes: list[str] = field(default_factory=list)
    path_used: str | None = None
    route_results: list[dict[str, Any]] = field(default_factory=list)
    evidence_families: list[str] = field(default_factory=list)
    input_evidence: dict[str, Any] = field(default_factory=dict)
    output_evidence: dict[str, Any] = field(default_factory=dict)
    repair_codes: list[str] = field(default_factory=list)
    qualified_success: bool = False


@dataclass
class StrictInputValidationResult:
    accepted: bool
    warning_codes: list[str] = field(default_factory=list)
    reason: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    pdb_audit: Any = None


class _StrictInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _atom_element_counts(mol: Chem.Mol) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                atom.GetSymbol().upper()
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() > 1
            ).items()
        )
    )


def _family_support_context(residue_names: list[str]) -> dict[str, Any]:
    from .paths import path_g

    path_g._register_special_residues()
    monomer_symbols = [path_g._residue_symbol(name) for name in residue_names]
    missing_residue_templates = []
    for name in sorted(set(residue_names)):
        try:
            get_residue_template(name)
        except ValueError:
            missing_residue_templates.append(name)
    missing_monomer_symbols = sorted(
        {
            symbol
            for symbol in monomer_symbols
            if symbol not in path_g.monomers2smi_dict
        }
    )
    supported_families = []
    if not missing_residue_templates:
        supported_families.append("residue_template")
    if not missing_monomer_symbols:
        supported_families.append("monomer_library")
    return {
        "residue_names": list(residue_names),
        "monomer_symbols": monomer_symbols,
        "missing_residue_templates": missing_residue_templates,
        "missing_monomer_symbols": missing_monomer_symbols,
        "supported_qualifying_families": supported_families,
        "strict_consensus_supported": set(supported_families)
        == set(QUALIFYING_FAMILIES),
    }


def _canonical_identity(smiles: str | None) -> tuple[str | None, str | None]:
    identity = molecular_identity(smiles)
    if identity is None:
        return None, None
    return identity.canonical_smiles, identity.full_inchikey


def _strict_connection_context(
    lines: list[str],
    chain_id: str,
    *,
    selected_serials: set[int],
    all_serials: set[int],
    serial_by_atom: dict[tuple[str, str, int, str], int],
    serial_to_residue: dict[int, tuple[int, str]],
    residue_positions: dict[tuple[int, str], int],
) -> dict[str, Any]:
    """Resolve selected-chain connection records without silent skipping."""
    explicit_edges: dict[tuple[int, int], set[str]] = defaultdict(set)
    selected_cross_residue_edges: set[tuple[int, int]] = set()

    def add_edge(first: int, second: int, source: str) -> None:
        if first == second:
            raise _StrictInputError(
                "V5_SELF_CONNECTION",
                f"{source.upper()} record contains a self-connection for serial {first}",
            )
        edge = tuple(sorted((first, second)))
        explicit_edges[edge].add(source)
        if serial_to_residue.get(first) != serial_to_residue.get(second):
            selected_cross_residue_edges.add(edge)

    for line_number, line in enumerate(lines, start=1):
        if line.startswith("SSBOND"):
            if len(line) < 35:
                raise _StrictInputError(
                    "V5_MALFORMED_SSBOND_RECORD",
                    f"SSBOND record is truncated at line {line_number}",
                )
            try:
                chain1, chain2 = line[15:16], line[29:30]
                num1 = int(line[17:21])
                num2 = int(line[31:35])
            except ValueError as exc:
                raise _StrictInputError(
                    "V5_MALFORMED_SSBOND_RECORD",
                    f"SSBOND residue number is malformed at line {line_number}",
                ) from exc
            if chain_id not in {chain1, chain2}:
                continue
            if chain1 != chain_id or chain2 != chain_id:
                raise _StrictInputError(
                    "V5_INTERCHAIN_CONNECTION_UNSUPPORTED",
                    f"selected chain participates in inter-chain SSBOND at line {line_number}",
                )
            first = serial_by_atom.get((
                "SG", line[11:14].strip(), num1, line[21:22].strip()
            ))
            second = serial_by_atom.get((
                "SG", line[25:28].strip(), num2, line[35:36].strip()
            ))
            if first is None or second is None:
                raise _StrictInputError(
                    "V5_TRUNCATED_CONNECTION_ENDPOINT",
                    f"SSBOND endpoint is absent from the selected chain at line {line_number}",
                )
            add_edge(first, second, "ssbond")

        elif line.startswith("LINK"):
            if len(line) < 57:
                raise _StrictInputError(
                    "V5_MALFORMED_LINK_RECORD",
                    f"LINK record is truncated at line {line_number}",
                )
            try:
                chain1, chain2 = line[21:22], line[51:52]
                num1 = int(line[22:26])
                num2 = int(line[52:56])
            except ValueError as exc:
                raise _StrictInputError(
                    "V5_MALFORMED_LINK_RECORD",
                    f"LINK residue number is malformed at line {line_number}",
                ) from exc
            if chain_id not in {chain1, chain2}:
                continue
            if chain1 != chain_id or chain2 != chain_id:
                raise _StrictInputError(
                    "V5_INTERCHAIN_CONNECTION_UNSUPPORTED",
                    f"selected chain participates in inter-chain LINK at line {line_number}",
                )
            first = serial_by_atom.get(
                (line[12:16].strip(), line[17:20].strip(), num1, line[26:27].strip())
            )
            second = serial_by_atom.get(
                (line[42:46].strip(), line[47:50].strip(), num2, line[56:57].strip())
            )
            if first is None or second is None:
                raise _StrictInputError(
                    "V5_TRUNCATED_CONNECTION_ENDPOINT",
                    f"LINK endpoint is absent from the selected chain at line {line_number}",
                )
            add_edge(first, second, "link")

        elif line.startswith("CONECT"):
            tokens = line.split()[1:]
            if len(tokens) < 2:
                raise _StrictInputError(
                    "V5_MALFORMED_CONECT_RECORD",
                    f"CONECT record has fewer than two serials at line {line_number}",
                )
            try:
                numbers = [int(token) for token in tokens]
            except ValueError as exc:
                raise _StrictInputError(
                    "V5_MALFORMED_CONECT_RECORD",
                    f"CONECT record contains a non-integer serial at line {line_number}",
                ) from exc
            source = numbers[0]
            selected_targets = [target for target in numbers[1:] if target in selected_serials]
            if source in selected_serials:
                for target in numbers[1:]:
                    if target not in all_serials:
                        raise _StrictInputError(
                            "V5_TRUNCATED_CONNECTION_ENDPOINT",
                            f"CONECT target serial {target} is absent at line {line_number}",
                        )
                    if target not in selected_serials:
                        raise _StrictInputError(
                            "V5_INTERCHAIN_CONNECTION_UNSUPPORTED",
                            f"selected atom {source} connects outside chain {chain_id!r} at line {line_number}",
                        )
                    add_edge(source, target, "conect")
            elif selected_targets:
                if source not in all_serials:
                    raise _StrictInputError(
                        "V5_TRUNCATED_CONNECTION_ENDPOINT",
                        f"CONECT source serial {source} is absent at line {line_number}",
                    )
                raise _StrictInputError(
                    "V5_INTERCHAIN_CONNECTION_UNSUPPORTED",
                    f"selected-chain atom connects to external serial {source} at line {line_number}",
                )

    partners: dict[int, set[int]] = defaultdict(set)
    for first, second in selected_cross_residue_edges:
        partners[first].add(second)
        partners[second].add(first)
    conflicts = {
        serial: sorted(values)
        for serial, values in partners.items()
        if len(values) > 1
    }
    if conflicts:
        raise _StrictInputError(
            "V5_EXPLICIT_CONNECTION_VALENCE_CONFLICT",
            f"selected-chain atoms have multiple cross-residue partners: {conflicts}",
        )

    closure_edges = 0
    for first, second in selected_cross_residue_edges:
        first_pos = residue_positions[serial_to_residue[first]]
        second_pos = residue_positions[serial_to_residue[second]]
        if abs(first_pos - second_pos) > 1:
            closure_edges += 1
    return {
        "explicit_edges": [
            {
                "serial_1": first,
                "serial_2": second,
                "sources": sorted(sources),
            }
            for (first, second), sources in sorted(explicit_edges.items())
        ],
        "same_chain_conect_closure_edges": sum(
            1
            for (first, second), sources in explicit_edges.items()
            if "conect" in sources
            and serial_to_residue.get(first) != serial_to_residue.get(second)
            and abs(
                residue_positions[serial_to_residue[first]]
                - residue_positions[serial_to_residue[second]]
            ) > 1
        ),
        "selected_cross_residue_edge_count": len(selected_cross_residue_edges),
        "selected_closure_edge_count": closure_edges,
    }


def _strict_seqres_context(
    lines: list[str],
    chain_id: str,
    coordinate_residue_names: list[str],
) -> dict[str, Any]:
    """Validate optional chain-level SEQRES declarations against coordinates."""
    records: list[tuple[int, int, list[str]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.startswith("SEQRES") or line[11:12] != chain_id:
            continue
        if len(line) < 19:
            raise _StrictInputError(
                "V5_MALFORMED_SEQRES_RECORD",
                f"SEQRES record is truncated at line {line_number}",
            )
        try:
            serial = int(line[7:10])
            declared_count = int(line[13:17])
        except ValueError as exc:
            raise _StrictInputError(
                "V5_MALFORMED_SEQRES_RECORD",
                f"SEQRES numbering is malformed at line {line_number}",
            ) from exc
        records.append((serial, declared_count, line[19:70].split()))
    if not records:
        return {
            "status": "absent",
            "declared_residue_count": None,
            "coordinate_residue_count": len(coordinate_residue_names),
        }
    serials = [row[0] for row in records]
    if serials != list(range(1, len(records) + 1)):
        raise _StrictInputError(
            "V5_MALFORMED_SEQRES_RECORD",
            f"SEQRES serials for chain {chain_id!r} are not contiguous from one: {serials}",
        )
    declared = {row[1] for row in records}
    if len(declared) != 1:
        raise _StrictInputError(
            "V5_MALFORMED_SEQRES_RECORD",
            f"SEQRES records disagree on declared chain length: {sorted(declared)}",
        )
    sequence = [name for _, _, names in records for name in names]
    declared_count = next(iter(declared))
    if declared_count != len(sequence):
        raise _StrictInputError(
            "V5_MALFORMED_SEQRES_RECORD",
            f"SEQRES declares {declared_count} residues but lists {len(sequence)}",
        )
    if sequence != coordinate_residue_names:
        raise _StrictInputError(
            "V5_SEQRES_COORDINATE_MISMATCH",
            f"SEQRES chain sequence of {len(sequence)} residues differs from the "
            f"{len(coordinate_residue_names)} coordinate residues",
        )
    return {
        "status": "exact_match",
        "declared_residue_count": declared_count,
        "coordinate_residue_count": len(coordinate_residue_names),
        "residue_names": sequence,
    }


def _selected_chain_context(
    pdb_path: str | Path,
    chain_id: str,
    *,
    allow_linear_topology: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the strict selected-chain input audit context.

    With ``allow_linear_topology=True``, a selected chain that carries neither
    explicit closure records, nor a geometrically-inferred closure, is accepted
    as ``topology_class="linear"`` instead of failing the fail-closed cyclization
    gate.  It is an opt-in relaxation; the cyclic contract is unchanged when the
    flag is left at its default ``False``.
    """
    path = Path(pdb_path)
    if path.suffix.lower() == ".gz":
        raise _StrictInputError(
            "V5_PLAIN_PDB_REQUIRED",
            "v5 reconstruction requires a normalized plain-text PDB input",
        )
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines()
    model_count = sum(line.startswith("MODEL") for line in lines)
    if model_count > 1:
        raise _StrictInputError(
            "V5_MULTIMODEL_INPUT_REJECTED",
            f"v5 reconstruction requires at most one model; observed {model_count}",
        )

    serials: set[int] = set()
    all_serials: set[int] = set()
    atom_keys: set[tuple[str, int, str, str]] = set()
    serial_by_atom: dict[tuple[str, str, int, str], int] = {}
    residue_names: dict[tuple[int, str], str] = {}
    residue_atoms: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    element_counts: Counter[str] = Counter()
    residue_keys: set[tuple[str, int, str, bool]] = set()
    serial_to_residue: dict[int, tuple[int, str]] = {}
    selected_chain_atoms_seen = False
    selected_chain_terminated = False
    for line_number, line in enumerate(lines, start=1):
        if line.startswith("TER"):
            ter_chain = line[21:22] if len(line) > 21 else ""
            if selected_chain_atoms_seen and ter_chain == chain_id:
                selected_chain_terminated = True
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            any_serial = int(line[6:11])
        except ValueError as exc:
            raise _StrictInputError(
                "V5_MALFORMED_PDB_NUMBER",
                f"atom serial is malformed at line {line_number}",
            ) from exc
        if any_serial in all_serials:
            raise _StrictInputError(
                "V5_DUPLICATE_ATOM_SERIAL",
                f"PDB input reuses atom serial {any_serial}",
            )
        all_serials.add(any_serial)
        if len(line) < 22:
            raise _StrictInputError(
                "V5_MALFORMED_ATOM_RECORD",
                f"atom record is too short to contain a chain identifier at line {line_number}",
            )
        if line[21:22] != chain_id:
            continue
        if len(line) < 78:
            raise _StrictInputError(
                "V5_MISSING_ELEMENT_FIELD",
                f"selected-chain atom record is too short for an element field at line {line_number}",
            )
        residue_name = line[17:20].strip()
        if residue_name in _SKIP_RESIDUES:
            continue
        if selected_chain_terminated:
            raise _StrictInputError(
                "V5_MULTISEGMENT_CHAIN_REJECTED",
                f"selected chain {chain_id!r} contains coordinate atoms after TER at line {line_number}",
            )
        selected_chain_atoms_seen = True
        altloc = line[16:17]
        insertion_code = line[26:27]
        if altloc.strip():
            raise _StrictInputError(
                "V5_ALTLOC_INPUT_REJECTED",
                f"selected chain contains alternate location {altloc!r} at line {line_number}",
            )
        try:
            serial = int(line[6:11])
            residue_number = int(line[22:26])
            xyz = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except ValueError as exc:
            raise _StrictInputError(
                "V5_MALFORMED_PDB_NUMBER",
                "selected chain has a malformed atom, residue number, or coordinate "
                f"at line {line_number}",
            ) from exc
        if serial in serials:
            raise _StrictInputError(
                "V5_DUPLICATE_ATOM_SERIAL",
                f"selected chain reuses atom serial {serial}",
            )
        serials.add(serial)
        residue_identity = (residue_number, insertion_code.strip())
        serial_to_residue[serial] = residue_identity
        atom_name = line[12:16].strip()
        atom_key = (
            residue_name,
            residue_number,
            insertion_code.strip(),
            atom_name,
        )
        if atom_key in atom_keys:
            raise _StrictInputError(
                "V5_DUPLICATE_ATOM_IDENTITY",
                f"selected chain repeats atom identity {atom_key}",
            )
        atom_keys.add(atom_key)
        serial_by_atom[(atom_name, residue_name, residue_number, insertion_code.strip())] = serial
        prior_name = residue_names.setdefault(residue_identity, residue_name)
        if prior_name != residue_name:
            raise _StrictInputError(
                "V5_AMBIGUOUS_RESIDUE_NUMBER",
                f"residue {residue_number}{insertion_code.strip()} maps to both "
                f"{prior_name} and {residue_name}",
            )
        element = line[76:78].strip().upper()
        if not element:
            raise _StrictInputError(
                "V5_MISSING_ELEMENT_FIELD",
                f"selected chain lacks an element field at line {line_number}",
            )
        expected_element = _expected_standard_atom_element(
            residue_name, atom_name
        )
        if expected_element is not None and element != expected_element:
            raise _StrictInputError(
                "V5_ATOM_NAME_ELEMENT_CONFLICT",
                f"selected-chain atom {residue_name} "
                f"{residue_number}{insertion_code.strip()}:{atom_name} "
                f"declares element {element}, expected {expected_element}",
            )
        if element != "H":
            element_counts[element] += 1
            residue_atoms[residue_identity].append(
                {
                    "serial": serial,
                    "atom_name": atom_name,
                    "element": element,
                    "record_type": line[:6].strip(),
                    "xyz": xyz,
                }
            )
        residue_keys.add((
            residue_name,
            residue_number,
            insertion_code.strip(),
            line.startswith("HETATM"),
        ))

    if not serials:
        raise _StrictInputError(
            "V5_NO_SELECTED_CHAIN_ATOMS",
            f"no non-solvent atoms were found for chain {chain_id!r}",
        )
    if len(residue_keys) < 2:
        raise _StrictInputError(
            "V5_INSUFFICIENT_RESIDUES",
            "v5 cyclic-peptide reconstruction requires at least two residues",
        )

    ordered_residue_identities = sorted(residue_names)
    residue_positions = {
        residue_identity: position
        for position, residue_identity in enumerate(ordered_residue_identities, start=1)
    }
    residue_inventory = [
        {
            "position": residue_positions[residue_identity],
            "residue_number": residue_identity[0],
            "insertion_code": residue_identity[1],
            "residue_name": residue_names[residue_identity],
            "atoms": sorted(
                residue_atoms[residue_identity], key=lambda row: row["serial"]
            ),
        }
        for residue_identity in ordered_residue_identities
    ]
    geometry_conflicts = _standard_residue_geometry_conflicts(residue_inventory)
    chem_comp_audit = _audit_chem_comp_evidence(
        pdb_path, residue_inventory, chem_comp_evidence
    )
    if chem_comp_audit.get("status") == "rejected":
        raise _StrictInputError(
            "V5_CHEM_COMP_EVIDENCE_INVALID",
            str(chem_comp_audit.get("error") or "chem_comp evidence is invalid"),
        )
    if geometry_conflicts and chem_comp_audit.get("status") == "validated":
        covered, uncovered = [], []
        for conflict in geometry_conflicts:
            (covered if _evidence_covers_conflict(conflict, chem_comp_audit)
             else uncovered).append(conflict)
        chem_comp_audit["covered_conflicts"] = covered
        chem_comp_audit["uncovered_conflicts"] = uncovered
        geometry_conflicts = uncovered
    if geometry_conflicts:
        raise _StrictInputError(
            "V5_STANDARD_RESIDUE_BOND_GEOMETRY_CONFLICT",
            "one or more observed standard-residue template bonds are contradicted "
            f"by their coordinates: {geometry_conflicts}",
        )
    coordinate_residue_names = [
        residue_names[residue_identity]
        for residue_identity in ordered_residue_identities
    ]
    seqres_context = _strict_seqres_context(
        lines,
        chain_id,
        coordinate_residue_names,
    )
    connection_context = _strict_connection_context(
        lines,
        chain_id,
        selected_serials=serials,
        all_serials=all_serials,
        serial_by_atom=serial_by_atom,
        serial_to_residue=serial_to_residue,
        residue_positions=residue_positions,
    )
    same_chain_ssbond = sum(
        line.startswith("SSBOND")
        and len(line) >= 30
        and line[15:16] == chain_id
        and line[29:30] == chain_id
        for line in lines
    )
    same_chain_link = sum(
        line.startswith("LINK")
        and len(line) >= 52
        and line[21:22] == chain_id
        and line[51:52] == chain_id
        for line in lines
    )

    topology = detect_cyclization(str(path), chain_id)
    if not topology.bonds:
        if not allow_linear_topology:
            raise _StrictInputError(
                "V5_NO_CYCLIZATION_EVIDENCE",
                "no explicit or geometric cyclization evidence was found for the selected chain",
            )
        # `detect_cyclization` runs with `allow_geometric_inference=True` by
        # default, so an empty cross-link set means neither the explicit
        # records (SSBOND/LINK/CONECT) nor the geometric covalent-radius
        # fallback produced any closure candidate.  That complete absence is
        # the evidence basis for asserting a linear topology here.
        topology_class = "linear"
    else:
        topology_class = topology.topology
    endpoint_partners: dict[tuple[int, str], set[tuple[int, str]]] = defaultdict(set)
    for bond in topology.bonds:
        left = (bond.pos1, str(bond.atom1 or bond.rgroup1 or "").upper())
        right = (bond.pos2, str(bond.atom2 or bond.rgroup2 or "").upper())
        endpoint_partners[left].add(right)
        endpoint_partners[right].add(left)
    conflicting_endpoints = {
        f"{position}:{atom}": sorted(f"{p}:{a}" for p, a in partners)
        for (position, atom), partners in endpoint_partners.items()
        if len(partners) > 1
    }
    if conflicting_endpoints:
        raise _StrictInputError(
            "V5_CONFLICTING_CYCLIZATION_ENDPOINT",
            "one or more selected-chain closure atoms have multiple partners: "
            f"{conflicting_endpoints}",
        )
    evidence_sources = [
        str(getattr(bond, "evidence_source", "unknown"))
        for bond in topology.bonds
    ]
    coordinate_inference_used = "geometry" in evidence_sources
    has_accepted_explicit_source = any(
        source in {"ssbond", "link", "conect"} for source in evidence_sources
    )
    repair_codes: list[str] = []
    if topology_class != "linear" and (
        coordinate_inference_used or (
            not has_accepted_explicit_source
            and same_chain_ssbond + same_chain_link
            + connection_context["same_chain_conect_closure_edges"] == 0
        )
    ):
        repair_codes.append("CONNECTIVITY_INFERRED_FROM_COORDINATES")
    return {
        "chain_id": chain_id,
        "topology_class": topology_class,
        "residue_names": coordinate_residue_names,
        "residue_atom_inventory": residue_inventory,
        "heavy_atom_count": sum(element_counts.values()),
        "heavy_element_counts": dict(sorted(element_counts.items())),
        "residue_count": len(residue_keys),
        "model_count": model_count or 1,
        "topology": topology.topology,
        "cyclization_bonds": [
            {
                "bond_type": bond.bond_type,
                "position_1": bond.pos1,
                "position_2": bond.pos2,
                "atom_1": bond.atom1,
                "atom_2": bond.atom2,
                "rgroup_1": bond.rgroup1,
                "rgroup_2": bond.rgroup2,
                "evidence_source": getattr(bond, "evidence_source", "unknown"),
            }
            for bond in topology.bonds
        ],
        "accepted_cyclization_evidence_sources": evidence_sources,
        "same_chain_record_counts": {
            "ssbond": same_chain_ssbond,
            "link": same_chain_link,
            "conect_closure_edges": connection_context[
                "same_chain_conect_closure_edges"
            ],
        },
        "explicit_connection_audit": connection_context,
        "seqres_audit": seqres_context,
        "repair_codes": repair_codes,
        "chem_comp_evidence_audit": {
            key: value for key, value in chem_comp_audit.items()
            if not str(key).startswith("_")
        },
    }


def validate_pdb_reconstruction_input_v5(
    pdb_path: str | Path,
    chain_id: str = "L",
    *,
    allow_linear_topology: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
) -> StrictInputValidationResult:
    """Apply the shared fail-closed input gate used by v5 reconstruction.

    This validates record-level syntax and selected-chain connection semantics;
    it does not claim that the input graph is chemically complete or correct.

    ``allow_linear_topology`` is forwarded to the selected-chain audit; when
    set, a chain with no explicit or geometric cyclization evidence is accepted
    as ``topology_class="linear"`` rather than rejected for lacking closure
    evidence.
    """
    pdb_audit = audit_pdb_file(pdb_path)
    if not pdb_audit.accepted:
        return StrictInputValidationResult(
            accepted=False,
            warning_codes=list(pdb_audit.warning_codes),
            reason=pdb_audit.reason or "input PDB audit failed",
            pdb_audit=pdb_audit,
        )
    try:
        context = _selected_chain_context(
            pdb_path,
            chain_id,
            allow_linear_topology=bool(allow_linear_topology),
            chem_comp_evidence=chem_comp_evidence,
        )
    except (OSError, UnicodeError, _StrictInputError, ValueError) as exc:
        code = exc.code if isinstance(exc, _StrictInputError) else "V5_INPUT_CONTEXT_FAILED"
        return StrictInputValidationResult(
            accepted=False,
            warning_codes=[code],
            reason=f"{type(exc).__name__}: {exc}",
        )
    return StrictInputValidationResult(
        accepted=True, context=context, pdb_audit=pdb_audit
    )


def _route_key(row: dict[str, Any]) -> str | None:
    value = row.get("output_inchikey")
    return str(value) if value else None


def _connection_signature(
    left_position: int,
    left_rgroup: str,
    right_position: int,
    right_rgroup: str,
) -> tuple[tuple[int, str], tuple[int, str]]:
    return tuple(
        sorted(
            (
                (int(left_position), str(left_rgroup).upper()),
                (int(right_position), str(right_rgroup).upper()),
            )
        )
    )


def _helm_connection_signatures(helm: str) -> list[tuple[tuple[int, str], tuple[int, str]]]:
    parts = helm.split("$")
    if len(parts) < 2 or not parts[1]:
        return []
    signatures = []
    for connection in parts[1].split("|"):
        fields = connection.split(",")
        if len(fields) != 3 or "-" not in fields[2]:
            raise ValueError(f"malformed HELM connection: {connection!r}")
        left, right = fields[2].split("-", 1)
        left_position, left_rgroup = left.split(":", 1)
        right_position, right_rgroup = right.split(":", 1)
        signatures.append(
            _connection_signature(
                int(left_position),
                left_rgroup,
                int(right_position),
                right_rgroup,
            )
        )
    return signatures


def _trace_topology_construction(
    route: str,
    pdb_path: str | Path,
    chain_id: str,
    selected_key: str,
    input_evidence: dict[str, Any],
    *,
    allow_geometric_inference: bool = True,
    forward_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify that each declared closure changes the actual B/G assembly."""
    try:
        from .paths._map_utils import get_smi_from_map, helm_to_map
        if route == "b":
            from .paths.path_b import build_helm_from_pdb as build_helm
        elif route == "g":
            from .paths.path_g import build_helm_with_special as build_helm
        else:
            raise ValueError(f"route {route!r} has no topology construction trace")

        if forward_artifact is None:
            helm = build_helm(
                str(pdb_path),
                chain_id,
                allow_geometric_inference=allow_geometric_inference,
            )
            mapped = None
            assembled = None
        else:
            if forward_artifact.get("route") != route:
                raise ValueError("forward artifact route differs from trace route")
            if forward_artifact.get("allow_geometric_inference") is not bool(
                allow_geometric_inference
            ):
                raise ValueError("forward artifact geometry mode differs")
            helm = forward_artifact.get("helm")
            mapped = forward_artifact.get("map_payload")
            assembled = forward_artifact.get("output_smiles")
        if not helm:
            raise ValueError("route builder emitted no HELM")
        parts = helm.split("$")
        if len(parts) < 5:
            raise ValueError("route builder emitted malformed HELM")
        observed = _helm_connection_signatures(helm)
        expected = [
            _connection_signature(
                bond["position_1"],
                bond["rgroup_1"],
                bond["position_2"],
                bond["rgroup_2"],
            )
            for bond in input_evidence.get("cyclization_bonds", [])
        ]
        if Counter(observed) != Counter(expected):
            raise ValueError(
                f"builder connection set differs from input ledger: "
                f"observed={observed}, expected={expected}"
            )
        if mapped is None:
            mapped = helm_to_map(helm)
        if assembled is None:
            assembled = (
                get_smi_from_map(mapped)
                if mapped and not mapped.startswith("ERROR")
                else None
            )
        _, assembled_key = _canonical_identity(assembled)
        if assembled_key != selected_key:
            raise ValueError(
                f"traced builder identity {assembled_key} differs from selected {selected_key}"
            )

        effective: list[dict[str, Any]] = []
        connections = parts[1].split("|") if parts[1] else []
        for index, signature in enumerate(observed):
            counterfactual_parts = list(parts)
            counterfactual_parts[1] = "|".join(
                connection
                for offset, connection in enumerate(connections)
                if offset != index
            )
            counterfactual_helm = "$".join(counterfactual_parts)
            try:
                counterfactual_map = helm_to_map(counterfactual_helm)
                counterfactual_smiles = (
                    get_smi_from_map(counterfactual_map)
                    if counterfactual_map
                    and not counterfactual_map.startswith("ERROR")
                    else None
                )
                _, counterfactual_key = _canonical_identity(counterfactual_smiles)
            except Exception:
                counterfactual_key = None
            effective.append(
                {
                    "signature": [list(endpoint) for endpoint in signature],
                    "removal_inchikey": counterfactual_key,
                    "identity_changes_when_removed": counterfactual_key != selected_key,
                }
            )
        if not all(row["identity_changes_when_removed"] for row in effective):
            raise ValueError("one or more declared closures do not affect assembled identity")
        return {
            "status": "verified",
            "route": route,
            "selected_inchikey": selected_key,
            "connection_count": len(observed),
            "connections": effective,
            "unexpected_connection_count": 0,
            "allow_geometric_inference": allow_geometric_inference,
        }
    except Exception as exc:
        return {
            "status": "unverified",
            "route": route,
            "selected_inchikey": selected_key,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _attach_topology_construction_traces(
    route_rows: list[dict[str, Any]],
    pdb_path: str | Path,
    chain_id: str,
    input_evidence: dict[str, Any],
    *,
    execution_artifacts: Mapping[Any, Any] | None = None,
) -> None:
    for row in route_rows:
        route = str(row.get("route", ""))
        key = _route_key(row)
        if route not in {"b", "g"} or row.get("status") != "success" or not key:
            continue
        row["topology_construction_trace"] = _trace_topology_construction(
            route,
            pdb_path,
            chain_id,
            key,
            input_evidence,
            forward_artifact=(
                (execution_artifacts or {}).get((route, True))
            ),
        )


def _attach_explicit_only_monomer_evidence(
    route_rows: list[dict[str, Any]],
    pdb_path: str | Path,
    chain_id: str,
    input_evidence: dict[str, Any],
) -> None:
    """Re-run B/G with coordinate inference disabled for bounded recovery."""
    from .paths.path_b import generate_with_artifacts as generate_b
    from .paths.path_g import generate_g_with_artifacts as generate_g

    by_route = {str(row.get("route")): row for row in route_rows}
    for route, generator in (("b", generate_b), ("g", generate_g)):
        started = perf_counter()
        try:
            smiles, error, forward_artifact = generator(
                str(pdb_path),
                chain_id,
                allow_geometric_inference=False,
            )
            canonical, key = _canonical_identity(smiles)
            evidence: dict[str, Any] = {
                "status": "success" if canonical and key and error is None else "failed",
                "output_smiles": canonical,
                "output_inchikey": key,
                "error": error,
                "runtime_sec": perf_counter() - started,
                "geometry_inference_disabled": True,
            }
            if evidence["status"] == "success":
                evidence["topology_construction_trace"] = _trace_topology_construction(
                    route,
                    pdb_path,
                    chain_id,
                    str(key),
                    input_evidence,
                    allow_geometric_inference=False,
                    forward_artifact=forward_artifact,
                )
        except Exception as exc:
            evidence = {
                "status": "failed",
                "output_smiles": None,
                "output_inchikey": None,
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_sec": perf_counter() - started,
                "geometry_inference_disabled": True,
            }
        if route in by_route:
            by_route[route]["explicit_only_evidence"] = evidence


def _raw_geometric_keys(route_rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in route_rows:
        if row.get("route") not in {"f", "h"}:
            continue
        identity = row.get("candidate_identity")
        if isinstance(identity, dict) and identity.get("output_inchikey"):
            keys.add(str(identity["output_inchikey"]))
    return keys


def _standard_residue_completion_ledger(
    input_evidence: dict[str, Any],
    output_counts: dict[str, int],
) -> dict[str, Any]:
    """Account for template-added heavy atoms without consulting test truth.

    Completion is intentionally limited to the 20 standard residues, whose PDB
    atom names can be generated independently by RDKit. Every observed atom
    must map injectively to one expected residue/name slot with the same
    element, and the output-minus-input formula must equal the missing slots.
    Nonstandard monomers therefore remain fail-closed until an independently
    validated atom-name/port registry is frozen for them.
    """
    from .paths.path_b import _AA_3TO1

    inventory = input_evidence.get("residue_atom_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise _StrictInputError(
            "V5_TEMPLATE_COMPLETION_PROVENANCE_UNAVAILABLE",
            "atom-level residue inventory is required to authorize template completion",
        )
    residue_names = [str(row.get("residue_name", "")).upper() for row in inventory]
    unsupported = sorted({name for name in residue_names if name not in _AA_3TO1})
    if unsupported:
        raise _StrictInputError(
            "V5_TEMPLATE_COMPLETION_MONOMER_UNVERIFIED",
            f"atom-level completion is not authorized for nonstandard residues {unsupported}",
        )
    reference = Chem.MolFromSequence("".join(_AA_3TO1[name] for name in residue_names))
    if reference is None:
        raise _StrictInputError(
            "V5_TEMPLATE_COMPLETION_REFERENCE_FAILED",
            "RDKit could not construct the standard-residue atom-name reference",
        )
    expected: dict[tuple[int, str], str] = {}
    for atom in reference.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or atom.GetAtomicNum() <= 1:
            continue
        expected[(int(info.GetResidueNumber()), info.GetName().strip())] = (
            atom.GetSymbol().upper()
        )

    bonds = list(input_evidence.get("cyclization_bonds", []))
    for bond in bonds:
        endpoints = {
            (int(bond["position_1"]), str(bond["rgroup_1"])),
            (int(bond["position_2"]), str(bond["rgroup_2"])),
        }
        if (len(inventory), "R2") in endpoints:
            expected.pop((len(inventory), "OXT"), None)
        for position, rgroup in endpoints:
            if rgroup != "R3" or not (1 <= position <= len(residue_names)):
                continue
            leaving_name = {"ASP": "OD2", "GLU": "OE2"}.get(
                residue_names[position - 1]
            )
            if leaving_name:
                expected.pop((position, leaving_name), None)

    observed_keys: set[tuple[int, str]] = set()
    observed_rows: list[dict[str, Any]] = []
    for row in inventory:
        position = int(row["position"])
        for atom in row.get("atoms", []):
            key = (position, str(atom.get("atom_name", "")))
            element = str(atom.get("element", "")).upper()
            expected_element = expected.get(key)
            if expected_element is None or expected_element != element or key in observed_keys:
                raise _StrictInputError(
                    "V5_OBSERVED_ATOM_TEMPLATE_MAPPING_FAILED",
                    f"observed atom {position}:{key[1]} ({element}) has no unique matching standard-template slot",
                )
            observed_keys.add(key)
            observed_rows.append(
                {
                    "position": position,
                    "atom_name": key[1],
                    "element": element,
                    "serial": atom.get("serial"),
                }
            )

    missing = [
        {"position": position, "atom_name": name, "element": element}
        for (position, name), element in sorted(expected.items())
        if (position, name) not in observed_keys
    ]
    missing_counts = dict(sorted(Counter(row["element"] for row in missing).items()))
    input_counts = dict(input_evidence["heavy_element_counts"])
    delta = {
        element: output_counts.get(element, 0) - input_counts.get(element, 0)
        for element in sorted(set(output_counts) | set(input_counts))
        if output_counts.get(element, 0) != input_counts.get(element, 0)
    }
    if any(value < 0 for value in delta.values()):
        raise _StrictInputError(
            "V5_HEAVY_ATOM_DELETION_REJECTED",
            f"selected output deletes observed heavy atoms: {delta}",
        )
    positive_delta = {element: value for element, value in delta.items() if value > 0}
    if positive_delta != missing_counts:
        raise _StrictInputError(
            "V5_TEMPLATE_COMPLETION_LEDGER_MISMATCH",
            f"output additions {positive_delta} do not equal missing standard-template slots {missing_counts}",
        )
    return {
        "authority": "rdkit_standard_residue_pdb_atom_names",
        "observed_atom_mapping_injective": True,
        "observed_atom_count": len(observed_rows),
        "expected_atom_count": len(expected),
        "added_atoms": missing,
        "added_element_counts": missing_counts,
        "ledger_closed": True,
    }


def _output_context(smiles: str, input_evidence: dict[str, Any], min_ring_size: int) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _StrictInputError(
            "V5_OUTPUT_PARSE_FAILED",
            "selected consensus identity could not be parsed by RDKit",
        )
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise _StrictInputError(
            "V5_OUTPUT_SANITIZATION_FAILED",
            f"selected consensus identity failed sanitization: {type(exc).__name__}: {exc}",
        ) from exc
    output_counts = _atom_element_counts(mol)
    input_counts = dict(input_evidence["heavy_element_counts"])
    completion_ledger = None
    inventory = input_evidence.get("residue_atom_inventory")
    from .paths.path_b import _AA_3TO1

    standard_inventory = (
        isinstance(inventory, list)
        and bool(inventory)
        and all(
            str(row.get("residue_name", "")).upper() in _AA_3TO1
            for row in inventory
        )
    )
    if standard_inventory or output_counts != input_counts:
        if any(
            output_counts.get(element, 0) < count
            for element, count in input_counts.items()
        ):
            raise _StrictInputError(
                "V5_HEAVY_ATOM_DELETION_REJECTED",
                f"selected output elements {output_counts} delete atoms from selected input {input_counts}",
            )
        completion_ledger = _standard_residue_completion_ledger(
            input_evidence,
            output_counts,
        )
    ring_sizes = sorted((len(ring) for ring in Chem.GetSymmSSSR(mol)), reverse=True)
    largest_ring = ring_sizes[0] if ring_sizes else 0
    if largest_ring < min_ring_size:
        if str(input_evidence.get("topology_class")) != "linear":
            raise _StrictInputError(
                "V5_MACROCYCLE_NOT_DEMONSTRATED",
                f"largest perceived output ring has {largest_ring} atoms; required >= {min_ring_size}",
            )
        # Linear inputs assert acyclicity through the closed input audit; keep
        # the audit fields (largest_ring=0, threshold) for downstream auditors.
    return {
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "heavy_element_counts": output_counts,
        "connected_component_count": len(Chem.GetMolFrags(mol)),
        "ring_sizes": ring_sizes,
        "largest_ring_size": largest_ring,
        "minimum_required_ring_size": min_ring_size,
        "sanitized": True,
        "heavy_atom_conserved": output_counts == input_counts,
        "template_completion_ledger": completion_ledger,
    }


def _accepted_closure_provenance(
    route_rows: list[dict[str, Any]],
    selected_key: str,
    input_evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Bind detected closure edges to a selected identity through topology users.

    A route label or whole-molecule identity is insufficient. A supporting B/G
    row must carry a verified construction trace showing an exact connection
    set and a changed identity when each declared edge is removed.
    """
    supporters_by_signature: dict[
        tuple[tuple[int, str], tuple[int, str]], list[str]
    ] = defaultdict(list)
    for row in route_rows:
        route = str(row.get("route", ""))
        trace = row.get("topology_construction_trace")
        if (
            route not in {"b", "g"}
            or row.get("status") != "success"
            or _route_key(row) != selected_key
            or not isinstance(trace, dict)
            or trace.get("status") != "verified"
            or trace.get("selected_inchikey") != selected_key
            or trace.get("unexpected_connection_count") != 0
        ):
            continue
        for connection in trace.get("connections", []):
            if not connection.get("identity_changes_when_removed"):
                continue
            signature_raw = connection.get("signature")
            if not isinstance(signature_raw, list) or len(signature_raw) != 2:
                continue
            signature = _connection_signature(
                signature_raw[0][0],
                signature_raw[0][1],
                signature_raw[1][0],
                signature_raw[1][1],
            )
            supporters_by_signature[signature].append(route)
    detected = list(input_evidence.get("cyclization_bonds", []))
    provenance = []
    for bond in detected:
        signature = _connection_signature(
            bond["position_1"],
            bond["rgroup_1"],
            bond["position_2"],
            bond["rgroup_2"],
        )
        supporters = sorted(set(supporters_by_signature.get(signature, [])))
        provenance.append(
            {
                **dict(bond),
                "selected_identity_supporters": supporters,
                "traceable_to_selected_identity": bool(supporters),
            }
        )
    complete = bool(provenance) and all(
        row["traceable_to_selected_identity"] for row in provenance
    )
    return provenance, complete


def _reject(
    code: str,
    reason: str,
    *,
    route_rows: list[dict[str, Any]] | None = None,
    input_evidence: dict[str, Any] | None = None,
    output_evidence: dict[str, Any] | None = None,
) -> StrictReconstructionResult:
    return StrictReconstructionResult(
        status="rejected",
        support_status="supported",
        rejection_reason=reason,
        warning_codes=[code],
        path_used="V5_FAMILY_AUDITED_CONSENSUS",
        route_results=list(route_rows or []),
        input_evidence=dict(input_evidence or {}),
        output_evidence=dict(output_evidence or {}),
        repair_codes=list((input_evidence or {}).get("repair_codes", [])),
        qualified_success=False,
    )


def _not_supported(
    code: str,
    reason: str,
    *,
    input_evidence: dict[str, Any],
) -> StrictReconstructionResult:
    return StrictReconstructionResult(
        status="not_supported",
        support_status="not_supported",
        rejection_reason=reason,
        warning_codes=[code],
        path_used="V5_FAMILY_SUPPORT_PREFLIGHT",
        input_evidence=dict(input_evidence),
        repair_codes=list(input_evidence.get("repair_codes", [])),
        qualified_success=False,
    )


def _explicit_monomer_recovery_context(
    route_rows: list[dict[str, Any]],
    input_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a bounded B/G candidate only when every recovery gate closes."""
    support = input_evidence.get("support_assessment", {})
    if not support.get("strict_consensus_supported"):
        return None
    if input_evidence.get("repair_codes"):
        return None
    if not input_evidence.get("cyclization_bonds"):
        return None
    if any(
        str(bond.get("evidence_source")) not in {"ssbond", "link", "conect"}
        for bond in input_evidence["cyclization_bonds"]
    ):
        return None

    evidence_by_route: dict[str, dict[str, Any]] = {}
    for row in route_rows:
        route = str(row.get("route", ""))
        evidence = row.get("explicit_only_evidence")
        if route in {"b", "g"} and isinstance(evidence, dict):
            evidence_by_route[route] = evidence
    if set(evidence_by_route) != {"b", "g"}:
        return None
    keys = {
        str(evidence.get("output_inchikey"))
        for evidence in evidence_by_route.values()
        if evidence.get("status") == "success" and evidence.get("output_inchikey")
    }
    if len(keys) != 1 or any(
        evidence.get("status") != "success" for evidence in evidence_by_route.values()
    ):
        return None
    selected_key = next(iter(keys))
    signatures = {
        _connection_signature(
            bond["position_1"],
            bond["rgroup_1"],
            bond["position_2"],
            bond["rgroup_2"],
        )
        for bond in input_evidence["cyclization_bonds"]
    }
    supporters: dict[tuple[tuple[int, str], tuple[int, str]], set[str]] = defaultdict(set)
    for route, evidence in evidence_by_route.items():
        trace = evidence.get("topology_construction_trace")
        if (
            not isinstance(trace, dict)
            or trace.get("status") != "verified"
            or trace.get("selected_inchikey") != selected_key
            or trace.get("unexpected_connection_count") != 0
            or trace.get("allow_geometric_inference") is not False
        ):
            return None
        for connection in trace.get("connections", []):
            if not connection.get("identity_changes_when_removed"):
                return None
            raw = connection.get("signature")
            if not isinstance(raw, list) or len(raw) != 2:
                return None
            supporters[
                _connection_signature(raw[0][0], raw[0][1], raw[1][0], raw[1][1])
            ].add(route)
    if set(supporters) != signatures or any(
        routes != {"b", "g"} for routes in supporters.values()
    ):
        return None
    return {
        "selected_key": selected_key,
        "selected_smiles": evidence_by_route["g"]["output_smiles"],
        "routes": ["b", "g"],
        "closure_supporters": {
            str(signature): sorted(routes) for signature, routes in supporters.items()
        },
    }


def _adjudicate_route_rows(
    route_rows: list[dict[str, Any]],
    input_evidence: dict[str, Any],
    *,
    minimum_evidence_families: int = 2,
    minimum_macrocycle_ring_size: int = 8,
) -> StrictReconstructionResult:
    accepted = [
        row
        for row in route_rows
        if row.get("status") == "success"
        and row.get("route") in ROUTE_FAMILIES
        and _route_key(row)
        and row.get("output_smiles")
    ]
    family_keys: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        family_keys[ROUTE_FAMILIES[str(row["route"])]].add(str(_route_key(row)))
    conflicting_families = {
        family: sorted(keys)
        for family, keys in family_keys.items()
        if family in QUALIFYING_FAMILIES and len(keys) != 1
    }
    regular_family_keys = {
        next(iter(keys))
        for family, keys in family_keys.items()
        if family in QUALIFYING_FAMILIES and len(keys) == 1
    }
    regular_consensus = (
        not conflicting_families
        and len(
            [family for family in family_keys if family in QUALIFYING_FAMILIES]
        )
        >= minimum_evidence_families
        and len(regular_family_keys) == 1
    )
    recovery = (
        None
        if regular_consensus
        else _explicit_monomer_recovery_context(route_rows, input_evidence)
    )
    monomer_conflict = "monomer_library" in conflicting_families
    if conflicting_families and (recovery is None or monomer_conflict):
        return _reject(
            "V5_WITHIN_FAMILY_IDENTITY_CONFLICT",
            f"one or more implementation families emitted conflicting identities: {conflicting_families}",
            route_rows=route_rows,
            input_evidence=input_evidence,
        )
    qualifying_family_keys = {
        family: keys
        for family, keys in family_keys.items()
        if family in QUALIFYING_FAMILIES
    }
    if len(qualifying_family_keys) < minimum_evidence_families and recovery is None:
        return _reject(
            "V5_INSUFFICIENT_EVIDENCE_FAMILIES",
            f"observed {len(qualifying_family_keys)} successful qualifying implementation "
            f"family/families; required {minimum_evidence_families}",
            route_rows=route_rows,
            input_evidence=input_evidence,
        )
    accepted_keys = {next(iter(keys)) for keys in qualifying_family_keys.values() if len(keys) == 1}
    if len(accepted_keys) != 1 and recovery is None:
        return _reject(
            "V5_CROSS_FAMILY_IDENTITY_CONFLICT",
            f"successful implementation families emitted {len(accepted_keys)} distinct full InChIKeys",
            route_rows=route_rows,
            input_evidence=input_evidence,
        )
    selected_key = str(recovery["selected_key"]) if recovery else next(iter(accepted_keys))
    geometric_keys = _raw_geometric_keys(route_rows)
    disagreeing_geometric = sorted(
        key for key in geometric_keys if key != selected_key
    )
    if disagreeing_geometric:
        return _reject(
            "V5_GEOMETRIC_RAW_IDENTITY_CONFLICT",
            "one or more F/H diagnostic candidates disagree with the selected "
            f"full InChIKey: {disagreeing_geometric}",
            route_rows=route_rows,
            input_evidence=input_evidence,
        )

    by_route = {str(row["route"]): row for row in accepted}
    selected_smiles = recovery["selected_smiles"] if recovery else next(
        by_route[route]["output_smiles"]
        for route in ROUTE_ORDER
        if route in by_route and _route_key(by_route[route]) == selected_key
    )
    canonical, canonical_key = _canonical_identity(str(selected_smiles))
    if not canonical or canonical_key != selected_key:
        return _reject(
            "V5_SELECTED_IDENTITY_DRIFT",
            "canonicalization changed or removed the selected full InChIKey",
            route_rows=route_rows,
            input_evidence=input_evidence,
        )
    try:
        output_evidence = _output_context(
            canonical,
            input_evidence,
            minimum_macrocycle_ring_size,
        )
        if recovery:
            atom_ledger = _standard_residue_completion_ledger(
                input_evidence,
                dict(output_evidence["heavy_element_counts"]),
            )
            output_evidence["observed_atom_provenance_ledger"] = atom_ledger
            if atom_ledger["added_atoms"]:
                output_evidence["template_completion_ledger"] = atom_ledger
    except _StrictInputError as exc:
        return _reject(
            exc.code,
            str(exc),
            route_rows=route_rows,
            input_evidence=input_evidence,
        )
    families = ["monomer_library"] if recovery else sorted(qualifying_family_keys)
    repair_codes = sorted(set(input_evidence.get("repair_codes", [])))
    if recovery:
        closure_provenance = [
            {
                **dict(bond),
                "selected_identity_supporters": ["b", "g"],
                "traceable_to_selected_identity": True,
                "geometry_inference_disabled": True,
            }
            for bond in input_evidence.get("cyclization_bonds", [])
        ]
        closure_provenance_complete = bool(closure_provenance)
    else:
        closure_provenance, closure_provenance_complete = _accepted_closure_provenance(
            route_rows,
            selected_key,
            input_evidence,
        )
    output_evidence["accepted_closure_provenance"] = closure_provenance
    output_evidence["closure_provenance_complete"] = closure_provenance_complete
    if recovery:
        output_evidence["explicit_monomer_recovery"] = recovery
        repair_codes.extend(
            [
                "EXPLICIT_MONOMER_FAMILY_RECOVERY",
                "RESIDUE_FAMILY_CONFLICT_BYPASSED",
            ]
        )
        if output_evidence.get("template_completion_ledger"):
            repair_codes.append("TEMPLATE_HEAVY_ATOM_COMPLETION")
    if closure_provenance and not closure_provenance_complete:
        repair_codes.append("ACCEPTED_CLOSURE_PROVENANCE_UNRESOLVED")
    repair_codes = sorted(set(repair_codes))
    return StrictReconstructionResult(
        status="success",
        support_status="supported",
        output_smiles=canonical,
        output_inchikey=selected_key,
        warning_codes=list(repair_codes),
        path_used=(
            "V5_EXPLICIT_MONOMER_FAMILY_RECOVERY"
            if recovery
            else f"V5_FAMILY_CONSENSUS:{','.join(families)}"
        ),
        route_results=route_rows,
        evidence_families=families,
        input_evidence=input_evidence,
        output_evidence=output_evidence,
        repair_codes=repair_codes,
        qualified_success=not repair_codes,
    )


def reconstruct_pdb_fail_closed_v5(
    pdb_path: str | Path,
    chain_id: str = "L",
    *,
    minimum_evidence_families: int = 2,
    minimum_macrocycle_ring_size: int = 8,
    monomer_context: Mapping[str, Any] | None = None,
) -> StrictReconstructionResult:
    """Reconstruct only with family-level agreement and conservation checks."""
    with identity_memo_context():
        return _reconstruct_pdb_fail_closed_v5_impl(
            pdb_path,
            chain_id,
            minimum_evidence_families=minimum_evidence_families,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            monomer_context=monomer_context,
        )


def _reconstruct_pdb_fail_closed_v5_impl(
    pdb_path: str | Path,
    chain_id: str,
    *,
    minimum_evidence_families: int,
    minimum_macrocycle_ring_size: int,
    monomer_context: Mapping[str, Any] | None,
) -> StrictReconstructionResult:

    from .core.monomer_resolution import needs_monomer_resolution_scope

    if needs_monomer_resolution_scope(monomer_context):
        from .core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            (
                monomer_context
                if monomer_context is not None
                else {"include_persistent_user": True}
            ),
            required_symbols=monomer_symbol_hints(
                pdb_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            resolved = reconstruct_pdb_fail_closed_v5(
                pdb_path,
                chain_id,
                minimum_evidence_families=minimum_evidence_families,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
            )
        if monomer_context is not None:
            resolved.input_evidence.setdefault(
                "monomer_resolution", dict(resolution_ledger)
            )
        return resolved
    if minimum_evidence_families < 2:
        return _reject(
            "V5_INVALID_FAMILY_THRESHOLD",
            "minimum_evidence_families must be at least 2",
        )
    if minimum_macrocycle_ring_size < 3:
        return _reject(
            "V5_INVALID_MACROCYCLE_THRESHOLD",
            "minimum_macrocycle_ring_size must be at least 3",
        )
    validation = validate_pdb_reconstruction_input_v5(pdb_path, chain_id)
    if not validation.accepted:
        return _reject(
            validation.warning_codes[0]
            if validation.warning_codes
            else "V5_INPUT_AUDIT_FAILED",
            validation.reason or "input PDB audit failed",
        )
    input_evidence = dict(validation.context)

    support = _family_support_context(list(input_evidence["residue_names"]))
    input_evidence["support_assessment"] = support
    if not support["strict_consensus_supported"]:
        return _not_supported(
            "V5_QUALIFYING_FAMILY_MONOMER_COVERAGE_UNSUPPORTED",
            "strict v5 consensus requires both residue-template and monomer-library "
            f"coverage; missing residue templates={support['missing_residue_templates']}, "
            f"missing monomer symbols={support['missing_monomer_symbols']}",
            input_evidence=input_evidence,
        )

    execution_artifacts: dict[Any, Any] = {}
    previous = reconstruct_pdb_fail_closed(
        pdb_path,
        chain_id,
        _execution_artifacts=execution_artifacts,
        _pdb_audit=validation.pdb_audit,
    )
    route_rows = [dict(row) for row in previous.route_results]
    _attach_topology_construction_traces(
        route_rows,
        pdb_path,
        chain_id,
        input_evidence,
        execution_artifacts=execution_artifacts,
    )
    _attach_explicit_only_monomer_evidence(
        route_rows,
        pdb_path,
        chain_id,
        input_evidence,
    )
    return _adjudicate_route_rows(
        route_rows,
        input_evidence,
        minimum_evidence_families=minimum_evidence_families,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
    )


def remediation_version() -> str:
    return f"{__version__}+remediation.5"
