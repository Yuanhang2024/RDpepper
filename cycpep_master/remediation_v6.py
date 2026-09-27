"""Evidence-dimension fail-closed cyclic-peptide reconstruction.

Version 6 is additive: it does not alter the frozen v3/v5 adjudicators.  Paths
remain candidate generators, while acceptance is based on auditable evidence
dimensions rather than counting implementations that share Unified chemistry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from dataclasses import asdict

from rdkit import Chem, rdBase
from rdkit.Chem import inchi

from . import __version__
from .core.identity_memo import identity_memo_context, molecular_identity
from .paths.path_a import generate_with_evidence
from .remediation_v3 import reconstruct_pdb_fail_closed
from .remediation_v5 import StrictReconstructionResult
from . import remediation_v5 as _v5


_CHEMICAL_ROUTES = frozenset({"a", "b", "c", "e", "g"})
_EXPLICIT_SOURCES = frozenset({"ssbond", "link", "conect"})
_ROUTE_ORDER = tuple("abceg")

_CANDIDATE_ASSESSMENT_SCHEMA = "1.0.0-v6-candidate-assessment.2"
_CANDIDATE_IDENTITY_POLICY = "standard-inchi-equivalence-v1"


def _inchi_version() -> str:
    getter = getattr(inchi, "GetInchiVersion", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            pass
    return "unavailable"


def _candidate_identity(smiles: str) -> dict[str, Any] | None:
    """Return deterministic identity layers for one route candidate."""
    identity = molecular_identity(smiles)
    if identity is None:
        return None
    return {
        "canonical_smiles": identity.canonical_smiles,
        "full_inchikey": identity.full_inchikey,
        "inchi_connectivity_block": identity.inchi_connectivity_block,
        "inchi_second_block": identity.inchi_second_block,
        "inchi_nonprotonation_key": identity.inchi_nonprotonation_key,
        "inchi_protonation_flag": identity.inchi_protonation_flag,
        "heavy_atom_composition": dict(identity.heavy_atom_composition),
        "formal_charge": identity.formal_charge,
    }


def _candidate_assessment(
    route_rows: list[dict[str, Any]],
    evidence_dimensions: dict[str, Any] | None = None,
    *,
    status: str,
    qualified_success: bool,
    support_status: str = "unknown",
    repair_codes: list[str] | None = None,
    output_smiles: str | None = None,
    output_inchikey: str | None = None,
    evidence_candidate_full_inchikey: str | None = None,
) -> dict[str, Any]:
    """Describe exploratory candidates without weakening strict V6 decisions.

    Candidate classes are truth-blind Standard InChIKey equivalence classes.
    L1-L3 describe identity layers, not benchmark correctness. Candidate
    agreement alone never permits unattended selection.
    """
    grouped: dict[str, dict[str, Any]] = {}
    route_audits: list[dict[str, Any]] = []
    inadmissible_routes: list[str] = []
    identity_resolution_failed_routes: list[str] = []
    counts = {
        "total_route_rows": len(route_rows),
        "chemical_route_rows": 0,
        "successful_chemical_route_rows": 0,
        "nonchemical_route_rows": 0,
        "unsuccessful_chemical_route_rows": 0,
        "missing_output_smiles_rows": 0,
        "identity_resolution_failed_rows": 0,
        "missing_declared_identity_rows": 0,
        "declared_identity_mismatch_rows": 0,
        "admitted_candidate_rows": 0,
        "distinct_candidate_equivalence_classes": 0,
    }

    def identity_input_sha256(row: dict[str, Any]) -> str:
        payload = {
            "route": row.get("route"),
            "status": row.get("status"),
            "output_smiles": row.get("output_smiles"),
            "output_inchikey": row.get("output_inchikey"),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    for row in route_rows:
        route = str(row.get("route", ""))
        audit = {
            "route": route,
            "row_status": str(row.get("status", "")),
            "declared_full_inchikey": (
                str(row["output_inchikey"])
                if row.get("output_inchikey") is not None else None
            ),
            "recomputed_full_inchikey": None,
            "canonical_smiles": None,
            "admitted": False,
            "reason": "",
            "identity_input_sha256": identity_input_sha256(row),
        }
        if route not in _CHEMICAL_ROUTES:
            counts["nonchemical_route_rows"] += 1
            audit["reason"] = "nonchemical_route"
            route_audits.append(audit)
            continue
        counts["chemical_route_rows"] += 1
        if row.get("status") != "success":
            counts["unsuccessful_chemical_route_rows"] += 1
            audit["reason"] = "route_status_not_success"
            route_audits.append(audit)
            continue
        counts["successful_chemical_route_rows"] += 1
        if not row.get("output_smiles"):
            counts["missing_output_smiles_rows"] += 1
            inadmissible_routes.append(route)
            audit["reason"] = "output_smiles_missing"
            route_audits.append(audit)
            continue
        identity = _candidate_identity(str(row["output_smiles"]))
        if identity is None:
            counts["identity_resolution_failed_rows"] += 1
            inadmissible_routes.append(route)
            identity_resolution_failed_routes.append(route)
            audit["reason"] = "identity_resolution_failed"
            route_audits.append(audit)
            continue
        audit["recomputed_full_inchikey"] = identity["full_inchikey"]
        audit["canonical_smiles"] = identity["canonical_smiles"]
        declared_key = audit["declared_full_inchikey"]
        if declared_key is None:
            counts["missing_declared_identity_rows"] += 1
            inadmissible_routes.append(route)
            audit["reason"] = "declared_full_inchikey_missing"
            route_audits.append(audit)
            continue
        if declared_key != identity["full_inchikey"]:
            counts["declared_identity_mismatch_rows"] += 1
            inadmissible_routes.append(route)
            audit["reason"] = "declared_recomputed_identity_mismatch"
            route_audits.append(audit)
            continue
        full_key = identity["full_inchikey"]
        candidate = grouped.setdefault(
            full_key,
            {
                "full_inchikey": full_key,
                "inchi_connectivity_block": identity["inchi_connectivity_block"],
                "inchi_second_block": identity["inchi_second_block"],
                "inchi_nonprotonation_key": identity["inchi_nonprotonation_key"],
                "inchi_protonation_flag": identity["inchi_protonation_flag"],
                "canonical_smiles_variants": set(),
                "heavy_atom_composition_variants": {},
                "formal_charge_values": set(),
                "routes": set(),
                "route_identity_input_sha256": set(),
                "admitted_route_row_count": 0,
            },
        )
        composition_key = json.dumps(
            identity["heavy_atom_composition"],
            sort_keys=True,
            separators=(",", ":"),
        )
        candidate["canonical_smiles_variants"].add(identity["canonical_smiles"])
        candidate["heavy_atom_composition_variants"][composition_key] = identity[
            "heavy_atom_composition"
        ]
        candidate["formal_charge_values"].add(identity["formal_charge"])
        candidate["routes"].add(route)
        candidate["route_identity_input_sha256"].add(
            audit["identity_input_sha256"]
        )
        candidate["admitted_route_row_count"] += 1
        counts["admitted_candidate_rows"] += 1
        audit["admitted"] = True
        audit["reason"] = "admitted"
        route_audits.append(audit)

    candidates = []
    for full_key in sorted(grouped):
        candidate = grouped[full_key]
        canonical_variants = sorted(candidate["canonical_smiles_variants"])
        composition_variants = [
            candidate["heavy_atom_composition_variants"][key]
            for key in sorted(candidate["heavy_atom_composition_variants"])
        ]
        formal_charge_values = sorted(candidate["formal_charge_values"])
        routes = sorted(candidate["routes"], key=_ROUTE_ORDER.index)
        candidates.append({
            "canonical_smiles": canonical_variants[0],
            "canonical_smiles_variants": canonical_variants,
            "full_inchikey": candidate["full_inchikey"],
            "inchi_connectivity_block": candidate["inchi_connectivity_block"],
            "inchi_second_block": candidate["inchi_second_block"],
            "inchi_nonprotonation_key": candidate["inchi_nonprotonation_key"],
            "inchi_protonation_flag": candidate["inchi_protonation_flag"],
            "heavy_atom_composition": composition_variants[0],
            "heavy_atom_composition_variants": composition_variants,
            "formal_charge": (
                formal_charge_values[0]
                if len(formal_charge_values) == 1 else None
            ),
            "formal_charge_values": formal_charge_values,
            "routes": routes,
            "supporting_route_count": len(routes),
            "admitted_route_row_count": candidate["admitted_route_row_count"],
            "route_identity_input_sha256": sorted(
                candidate["route_identity_input_sha256"]
            ),
        })
    counts["distinct_candidate_equivalence_classes"] = len(candidates)
    route_audits.sort(key=lambda audit: (
        _ROUTE_ORDER.index(audit["route"])
        if audit["route"] in _ROUTE_ORDER else len(_ROUTE_ORDER),
        audit["route"],
        audit["identity_input_sha256"],
    ))

    def distinct(field: str) -> list[Any]:
        values = {
            json.dumps(candidate[field], sort_keys=True, separators=(",", ":"))
            for candidate in candidates
        }
        return [json.loads(value) for value in sorted(values)]

    composition_values = sorted({
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for candidate in candidates
        for value in candidate["heavy_atom_composition_variants"]
    })
    composition_values = [json.loads(value) for value in composition_values]
    connectivity_values = distinct("inchi_connectivity_block")
    nonprotonation_values = distinct("inchi_nonprotonation_key")
    full_values = distinct("full_inchikey")
    canonical_values = sorted({
        value
        for candidate in candidates
        for value in candidate["canonical_smiles_variants"]
    })

    shared_level = None
    if len(composition_values) == 1:
        shared_level = "L1"
    if len(connectivity_values) == 1:
        shared_level = "L2"
    if len(nonprotonation_values) == 1:
        shared_level = "L3"

    conflicts = []
    if len(composition_values) > 1:
        conflicts.append("heavy_atom_composition")
    if len(connectivity_values) > 1:
        conflicts.append("connectivity")
    connectivity_groups: dict[str, list[dict[str, Any]]] = {}
    nonprotonation_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        connectivity_groups.setdefault(
            candidate["inchi_connectivity_block"], []
        ).append(candidate)
        nonprotonation_groups.setdefault(
            candidate["inchi_nonprotonation_key"], []
        ).append(candidate)
    if any(
        len({row["inchi_second_block"] for row in rows}) > 1
        for rows in connectivity_groups.values()
    ):
        conflicts.append("nonprotonation_identity_within_connectivity")
    if any(
        len({row["inchi_protonation_flag"] for row in rows}) > 1
        or len({
            charge
            for row in rows
            for charge in row["formal_charge_values"]
        }) > 1
        for rows in nonprotonation_groups.values()
    ):
        conflicts.append(
            "protonation_or_formal_charge_within_nonprotonation_identity"
        )
    if len(full_values) > 1:
        conflicts.append("standard_inchikey_equivalence")

    dimensions = evidence_dimensions if isinstance(evidence_dimensions, dict) else {}

    def passed(name: str) -> bool:
        value = dimensions.get(name)
        return isinstance(value, dict) and value.get("passed") is True

    evidence_candidate_identity_bound = bool(
        len(candidates) == 1
        and evidence_candidate_full_inchikey == candidates[0]["full_inchikey"]
    )
    qualified_level = None
    if (
        evidence_candidate_identity_bound
        and
        shared_level in {"L1", "L2", "L3"}
        and passed("chemical_graph_audit")
        and passed("atom_mapping")
    ):
        qualified_level = "L1"
    if (
        shared_level in {"L2", "L3"}
        and qualified_level == "L1"
        and passed("explicit_connectivity")
        and passed("closure_identity_trace")
        and passed("mapping_evidence_binding")
        and passed("path_evidence_fresh_replay")
    ):
        qualified_level = "L2"
    if (
        shared_level == "L3"
        and qualified_level == "L2"
        and passed("stereochemistry")
    ):
        qualified_level = "L3"

    selected_identity = (
        _candidate_identity(output_smiles) if output_smiles else None
    )
    strict_output_emitted = bool(
        status == "success" and output_smiles and output_inchikey
    )
    if not strict_output_emitted:
        selected_output_reason = "not_emitted"
    elif selected_identity is None:
        selected_output_reason = "identity_resolution_failed"
    elif selected_identity["full_inchikey"] != output_inchikey:
        selected_output_reason = "declared_recomputed_identity_mismatch"
    elif output_inchikey not in grouped:
        selected_output_reason = "not_in_candidate_set"
    else:
        selected_output_reason = "bound"
    selected_output_identity_bound = selected_output_reason == "bound"
    unattended_selection_qualified = bool(
        strict_output_emitted
        and selected_output_identity_bound
        and qualified_success
        and len(candidates) == 1
    )

    if not candidates:
        mode = "no_candidate"
    elif len(candidates) == 1:
        mode = "candidate_unique"
    else:
        mode = "candidate_ensemble"

    return {
        "schema_version": _CANDIDATE_ASSESSMENT_SCHEMA,
        "producer": {
            "entrypoint": "cycpep_master.remediation_v6._candidate_assessment",
            "package_version": __version__,
            "rdkit_version": rdBase.rdkitVersion,
            "inchi_version": _inchi_version(),
            "identity_policy": _CANDIDATE_IDENTITY_POLICY,
            "truth_access": False,
        },
        "result_context": {
            "status": status,
            "support_status": support_status,
            "qualified_success": bool(qualified_success),
            "repair_codes": sorted(set(repair_codes or [])),
        },
        "candidate_basis": (
            "successful chemical route rows with parseable SMILES and matching "
            "declared/recomputed Standard InChIKey"
        ),
        "mode": mode,
        "exploratory_only": not unattended_selection_qualified,
        "unattended_selection_qualified": unattended_selection_qualified,
        "strict_output_emitted": strict_output_emitted,
        "selected_output_identity_bound": selected_output_identity_bound,
        "selected_output_audit": {
            "declared_full_inchikey": output_inchikey,
            "recomputed_full_inchikey": (
                selected_identity["full_inchikey"]
                if selected_identity is not None else None
            ),
            "canonical_smiles": (
                selected_identity["canonical_smiles"]
                if selected_identity is not None else None
            ),
            "reason": selected_output_reason,
        },
        "automatic_selection_permitted": unattended_selection_qualified,
        "candidate_count": len(candidates),
        "successful_chemical_route_row_count": counts[
            "successful_chemical_route_rows"
        ],
        "admitted_candidate_route_row_count": counts[
            "admitted_candidate_rows"
        ],
        "candidates": candidates,
        "route_identity_audits": route_audits,
        "route_row_counts": counts,
        "identity_resolution_failed_routes": sorted(
            set(identity_resolution_failed_routes)
        ),
        "inadmissible_candidate_routes": sorted(set(inadmissible_routes)),
        "invalid_candidate_routes": sorted(set(inadmissible_routes)),
        "highest_shared_candidate_level": shared_level,
        "highest_evidence_qualified_level": qualified_level,
        "evidence_qualification_scope": "unique_candidate_only",
        "evidence_candidate_full_inchikey": evidence_candidate_full_inchikey,
        "evidence_candidate_identity_bound": evidence_candidate_identity_bound,
        "standard_inchikey_equivalence_unique": len(full_values) == 1,
        "literal_canonical_smiles_unique": len(canonical_values) == 1,
        "conflict_dimensions": conflicts,
        "identity_layer_counts": {
            "L1_heavy_atom_composition": len(composition_values),
            "L2_inchi_connectivity_block": len(connectivity_values),
            "L3_inchi_nonprotonation_key": len(nonprotonation_values),
            "full_standard_inchikey": len(full_values),
            "literal_canonical_smiles": len(canonical_values),
        },
        "level_definitions": {
            "L1": (
                "heavy-atom element composition; hydrogen, isotope, bond, "
                "charge, and stereochemistry insensitive"
            ),
            "L2": "standard InChIKey connectivity block",
            "L3": "first two standard InChIKey blocks; protonation layer excluded",
        },
        "deprecated_fields": {
            "automatic_selection_permitted": "unattended_selection_qualified",
            "invalid_candidate_routes": "inadmissible_candidate_routes",
        },
    }


def _result(
    status: str,
    code: str | None,
    reason: str | None,
    *,
    support_status: str,
    route_rows: list[dict[str, Any]] | None = None,
    input_evidence: dict[str, Any] | None = None,
    output_evidence: dict[str, Any] | None = None,
    output_smiles: str | None = None,
    output_inchikey: str | None = None,
    path_used: str | None = None,
    repair_codes: list[str] | None = None,
    evidence_dimensions: list[str] | None = None,
    warning_codes: list[str] | None = None,
    qualified_success: bool = False,
) -> StrictReconstructionResult:
    warnings = list(warning_codes or ([code] if code else []))
    rows = list(route_rows or [])
    evidence = dict(output_evidence or {})
    evidence["candidate_assessment"] = _candidate_assessment(
        rows,
        evidence.get("evidence_dimensions"),
        status=status,
        qualified_success=qualified_success,
        support_status=support_status,
        repair_codes=repair_codes,
        output_smiles=output_smiles,
        output_inchikey=output_inchikey,
        evidence_candidate_full_inchikey=evidence.get(
            "selected_full_inchikey"
        ),
    )
    return StrictReconstructionResult(
        status=status,
        support_status=support_status,
        output_smiles=output_smiles,
        output_inchikey=output_inchikey,
        rejection_reason=reason,
        warning_codes=warnings,
        path_used=path_used or "V6_EVIDENCE_DIMENSION_AUDIT",
        route_results=rows,
        evidence_families=list(evidence_dimensions or []),
        input_evidence=dict(input_evidence or {}),
        output_evidence=evidence,
        repair_codes=list(repair_codes or []),
        qualified_success=qualified_success,
    )


def _dimension(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **details}


def _local_monomer_bootstrap_failure_details(bootstrap: Any) -> dict[str, str]:
    reason_codes = {
        str(code)
        for result in bootstrap.inference_results
        for code in result.reason_codes
    }
    if "R3_PORT_REUSED_BY_MULTIPLE_PARTNERS" in reason_codes:
        return {
            "status": "rejected",
            "code": "V6_LOCAL_MONOMER_EVIDENCE_CONFLICT",
            "reason": "explicit local-monomer port evidence is contradictory",
            "support_status": "supported",
        }
    if "SOURCE_IDENTITY_MAPPING_UNRESOLVED" in reason_codes:
        return {
            "status": "rejected",
            "code": "SOURCE_IDENTITY_MAPPING_UNRESOLVED",
            "reason": (
                "the source-declared polymer sequence cannot be mapped to the "
                "unknown coordinate residue at a unique position"
            ),
            "support_status": "supported",
        }
    if "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE" in reason_codes:
        return {
            "status": "rejected",
            "code": "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE",
            "reason": (
                "a unique source-declared residue identity cannot account for "
                "the observed unknown-residue atoms"
            ),
            "support_status": "supported",
        }
    if "SOURCE_IDENTITY_CONSTRAINT_CONFLICT" in reason_codes:
        return {
            "status": "rejected",
            "code": "SOURCE_IDENTITY_CONSTRAINT_CONFLICT",
            "reason": (
                "the unknown-residue graph conflicts with its unique "
                "source-declared residue identity"
            ),
            "support_status": "supported",
        }
    if bootstrap.status == "rejected":
        return {
            "status": "rejected",
            "code": "V6_LOCAL_MONOMER_INPUT_REJECTED",
            "reason": (
                "one or more Unified-unknown components failed source-bound "
                "monomer validation"
            ),
            "support_status": "supported",
        }
    if "MULTIPLE_R3_PORTS_NOT_REPRESENTABLE" in reason_codes:
        return {
            "status": "not_supported",
            "code": "V6_MULTIPOINT_SCAFFOLD_NOT_SUPPORTED",
            "reason": (
                "the source contains a component with multiple independent "
                "external attachment ports, which the R1/R2/R3 monomer model "
                "cannot represent"
            ),
            "support_status": "not_supported",
        }
    return {
        "status": "not_supported",
        "code": "V6_LOCAL_MONOMER_INFERENCE_NOT_UNIQUE",
        "reason": (
            "one or more Unified-unknown residues did not yield a unique "
            "entity-local monomer graph"
        ),
        "support_status": "not_supported",
    }


def _route_key(row: dict[str, Any]) -> str | None:
    return _v5._route_key(row)


def _residue_mapping_ledger_audit(
    evidence: dict[str, Any], input_evidence: dict[str, Any]
) -> dict[str, Any]:
    """Check a path mapping ledger against input atoms and trusted templates."""
    from .paths.residue_template_factory import (
        get_residue_template,
        standard_pdb_atom_name_map,
    )

    inventory = list(input_evidence.get("residue_atom_inventory", []))
    inventory_by_position = {
        int(row["position"]): row for row in inventory
        if isinstance(row, dict) and isinstance(row.get("position"), int)
    }
    expected_source_bound_r3_serials: set[int] = set()
    expected_r3_serials_by_position: dict[int, set[int]] = {}
    expected_r3_atom_names_by_position: dict[int, set[str]] = {}
    source_bound_r3_resolution_errors = []
    for bond_index, bond in enumerate(input_evidence.get("cyclization_bonds", [])):
        if not isinstance(bond, dict):
            source_bound_r3_resolution_errors.append({
                "bond_index": bond_index,
                "reason": "malformed_cyclization_bond",
            })
            continue
        if str(bond.get("evidence_source", "")).strip().lower() not in _EXPLICIT_SOURCES:
            continue
        for suffix in ("1", "2"):
            if str(bond.get(f"rgroup_{suffix}", "")).strip().upper() != "R3":
                continue
            try:
                position = int(bond[f"position_{suffix}"])
                atom_name = str(bond[f"atom_{suffix}"]).strip().upper()
                source = inventory_by_position[position]
                matches = [
                    int(atom["serial"])
                    for atom in source.get("atoms", [])
                    if str(atom.get("atom_name", "")).strip().upper()
                    == atom_name
                ]
            except Exception as exc:
                source_bound_r3_resolution_errors.append({
                    "bond_index": bond_index,
                    "endpoint": suffix,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            if len(matches) != 1:
                source_bound_r3_resolution_errors.append({
                    "bond_index": bond_index,
                    "endpoint": suffix,
                    "position": position,
                    "atom_name": atom_name,
                    "match_count": len(matches),
                    "reason": "r3_endpoint_not_uniquely_resolved",
                })
                continue
            expected_source_bound_r3_serials.add(matches[0])
            expected_r3_serials_by_position.setdefault(position, set()).add(
                matches[0]
            )
            expected_r3_atom_names_by_position.setdefault(position, set()).add(
                atom_name
            )
    source_bound_r3_resolution_passed = not source_bound_r3_resolution_errors
    residues = evidence.get("residue_evidence")
    residues = residues if isinstance(residues, list) else []
    row_audits = []
    observed_positions = []
    for row in residues:
        if not isinstance(row, dict):
            row_audits.append({"passed": False, "reason": "malformed_residue_row"})
            continue
        try:
            position = int(row.get("residue_position"))
            source = inventory_by_position[position]
            template = get_residue_template(
                str(source.get("residue_name") or "")
            )
            source_atoms = list(source.get("atoms", []))
            observed_serials = {
                int(atom["serial"]) for atom in source_atoms
            }
            raw_serial_map = row.get("serial_to_template_atom_index", {})
            if not isinstance(raw_serial_map, dict):
                raise ValueError("malformed serial-to-template mapping")
            if any(
                type(serial) is not str or type(index) is not int
                for serial, index in raw_serial_map.items()
            ):
                raise ValueError("noncanonical serial-to-template mapping type")
            serial_map = {
                int(serial): int(index)
                for serial, index in raw_serial_map.items()
            }
            raw_deferred_oxt = row.get("deferred_terminal_oxt_serials", [])
            raw_deferred_r3 = row.get(
                "deferred_consumed_r3_template_atoms", []
            )
            raw_deferred_r3_anchors = row.get(
                "deferred_consumed_r3_anchor_serials", []
            )
            raw_source_bound_r3 = row.get(
                "source_bound_explicit_r3_attachment_serials", []
            )
            if not all(isinstance(value, list) for value in (
                raw_deferred_oxt,
                raw_deferred_r3,
                raw_deferred_r3_anchors,
                raw_source_bound_r3,
            )):
                raise ValueError("malformed deferred R3/OXT evidence")
            if any(
                not isinstance(item, dict)
                or set(item) != {"template_atom_index", "atom_name"}
                or type(item.get("template_atom_index")) is not int
                or type(item.get("atom_name")) is not str
                for item in raw_deferred_r3
            ):
                raise ValueError("malformed deferred R3 template atom row")
            if any(
                type(serial) is not int
                for values in (
                    raw_deferred_oxt,
                    raw_deferred_r3_anchors,
                    raw_source_bound_r3,
                )
                for serial in values
            ):
                raise ValueError("noncanonical deferred R3/OXT serial type")
            deferred = {int(serial) for serial in raw_deferred_oxt}
            deferred_r3_rows = [
                {
                    "template_atom_index": int(item["template_atom_index"]),
                    "atom_name": str(item["atom_name"]).strip().upper(),
                }
                for item in raw_deferred_r3
            ]
            deferred_r3 = {
                int(item["template_atom_index"]): str(item["atom_name"])
                for item in deferred_r3_rows
            }
            deferred_r3_anchor_serials = {
                int(serial) for serial in raw_deferred_r3_anchors
            }
            source_bound_r3_serials = {
                int(serial) for serial in raw_source_bound_r3
            }
            serial_map_canonical = raw_serial_map == {
                str(serial): index for serial, index in sorted(serial_map.items())
            }
            deferred_oxt_canonical = raw_deferred_oxt == sorted(deferred)
            deferred_r3_canonical = raw_deferred_r3 == [
                {"template_atom_index": index, "atom_name": atom_name}
                for index, atom_name in sorted(deferred_r3.items())
            ]
            deferred_r3_anchors_canonical = (
                raw_deferred_r3_anchors == sorted(deferred_r3_anchor_serials)
            )
            source_bound_r3_canonical = (
                raw_source_bound_r3 == sorted(source_bound_r3_serials)
            )
            mapped_indices = set(serial_map.values())
            template_atom_count = template.mol.GetNumAtoms()
            raw_unmapped_template_count = (
                template_atom_count - len(mapped_indices)
            )
            effective_unmapped_template_count = (
                raw_unmapped_template_count - len(deferred_r3)
            )
            raw_external_indices = row.get("external_attachment_indices", {})
            if not isinstance(raw_external_indices, dict) or any(
                type(serial) is not str or type(index) is not int
                for serial, index in raw_external_indices.items()
            ):
                raise ValueError("noncanonical external attachment mapping")
            external_indices = {
                int(serial): int(index)
                for serial, index in raw_external_indices.items()
            }
            external_indices_canonical = raw_external_indices == {
                str(serial): index
                for serial, index in sorted(external_indices.items())
            }
            unassigned = sorted(
                int(index)
                for index, assignment in Chem.FindMolChiralCenters(
                    template.mol,
                    includeUnassigned=True,
                    useLegacyImplementation=False,
                )
                if assignment == "?"
            )
            residue_key = list(row.get("residue_key", []))
            expected_graph_sha = hashlib.sha256(
                template.smiles.encode("utf-8")
            ).hexdigest()
            source_residue_name = str(
                source.get("residue_name", "")
            ).strip().upper()
            trusted_name_map = standard_pdb_atom_name_map(
                source_residue_name, template.smiles
            )
            expected_standard_serial_map = {
                int(atom["serial"]): int(trusted_name_map[atom_name])
                for atom in source_atoms
                if (
                    atom_name
                    := str(atom.get("atom_name", "")).strip().upper()
                ) in trusted_name_map
            }
            unrecognized_standard_atom_names = {
                str(atom.get("atom_name", "")).strip().upper()
                for atom in source_atoms
                if str(atom.get("atom_name", "")).strip().upper()
                not in trusted_name_map
                and str(atom.get("atom_name", "")).strip().upper() != "OXT"
            }
            standard_serial_map_bound = (
                not trusted_name_map
                or (
                    not unrecognized_standard_atom_names
                    and serial_map == expected_standard_serial_map
                )
            )
            deferred_r3_bound = not deferred_r3
            if deferred_r3:
                expected_leaving_name = {
                    "ASP": "OD2",
                    "GLU": "OE2",
                }.get(source_residue_name)
                expected_leaving_index = trusted_name_map.get(
                    expected_leaving_name or ""
                )
                anchor_index = template.r3_anchor_index
                trusted_anchor_names = {
                    name
                    for name, template_index in trusted_name_map.items()
                    if template_index == anchor_index
                }
                leaving_bond = (
                    template.mol.GetBondBetweenAtoms(
                        int(anchor_index), int(expected_leaving_index)
                    )
                    if anchor_index is not None
                    and expected_leaving_index is not None
                    else None
                )
                deferred_r3_bound = bool(
                    expected_leaving_name
                    and str(template.r3).strip().upper() == "OH"
                    and anchor_index is not None
                    and expected_leaving_index is not None
                    and deferred_r3
                    == {int(expected_leaving_index): expected_leaving_name}
                    and len(deferred_r3_anchor_serials) == 1
                    and deferred_r3_anchor_serials
                    == expected_r3_serials_by_position.get(position, set())
                    and expected_r3_atom_names_by_position.get(position, set())
                    == trusted_anchor_names
                    and all(
                        serial_map.get(serial) == int(anchor_index)
                        for serial in deferred_r3_anchor_serials
                    )
                    and leaving_bond is not None
                    and leaving_bond.GetBondType() == Chem.BondType.SINGLE
                )
            checks = {
                "position_bound": (
                    type(row.get("residue_position")) is int
                    and position == int(source["position"])
                ),
                "integer_field_types_canonical": all(
                    type(row.get(field)) is int
                    for field in (
                        "observed_heavy_atom_count",
                        "mapped_heavy_atom_count",
                        "template_heavy_atom_count",
                        "mapped_template_heavy_atom_count",
                        "unmapped_template_heavy_atom_count",
                        "effective_unmapped_template_heavy_atom_count",
                        "deferred_consumed_r3_count",
                        "deferred_terminal_oxt_count",
                        "mapping_candidate_count",
                    )
                ),
                "serial_map_canonical": serial_map_canonical,
                "deferred_oxt_canonical": deferred_oxt_canonical,
                "deferred_r3_canonical": deferred_r3_canonical,
                "deferred_r3_anchors_canonical": (
                    deferred_r3_anchors_canonical
                ),
                "source_bound_r3_canonical": source_bound_r3_canonical,
                "residue_key_bound": (
                    len(residue_key) >= 2
                    and residue_key[0] == source.get("residue_name")
                    and int(residue_key[1]) == int(source.get("residue_number"))
                ),
                "pdb_resname_bound": row.get("pdb_resname") == residue_key[0],
                "template_symbol_bound": row.get("unified_symbol") == template.symbol,
                "template_source_bound": row.get("unified_source") == template.source,
                "template_graph_bound": (
                    row.get("monomer_graph_sha256") == expected_graph_sha
                ),
                "free_graph_bound": (
                    row.get("free_monomer_graph_sha256")
                    == template.free_graph_sha256
                ),
                "rgroup_defaults_bound": row.get("rgroup_defaults") == {
                    "R1": template.r1, "R2": template.r2, "R3": template.r3,
                },
                "r3_anchor_bound": (
                    (
                        row.get("r3_anchor_template_atom_index") is None
                        and template.r3_anchor_index is None
                    )
                    or (
                        type(row.get("r3_anchor_template_atom_index")) is int
                        and row.get("r3_anchor_template_atom_index")
                        == template.r3_anchor_index
                    )
                ),
                "observed_atom_count_bound": (
                    row.get("observed_heavy_atom_count") == len(observed_serials)
                ),
                "observed_serials_complete": (
                    set(serial_map) | deferred == observed_serials
                    and not (set(serial_map) & deferred)
                ),
                "mapping_indices_in_range": all(
                    0 <= index < template_atom_count for index in mapped_indices
                ),
                "standard_serial_map_recomputed": standard_serial_map_bound,
                "deferred_r3_indices_in_range": all(
                    0 <= index < template_atom_count for index in deferred_r3
                ),
                "deferred_r3_indices_disjoint": not (
                    mapped_indices & set(deferred_r3)
                ),
                "deferred_r3_source_bound": deferred_r3_bound,
                "source_bound_r3_serials_recomputed": (
                    source_bound_r3_resolution_passed
                    and source_bound_r3_serials
                    == expected_source_bound_r3_serials
                ),
                "deferred_r3_count_bound": (
                    row.get("deferred_consumed_r3_count", 0)
                    == len(deferred_r3)
                ),
                "mapping_injective_recomputed": (
                    len(mapped_indices) == len(serial_map)
                ),
                "mapped_atom_count_bound": (
                    row.get("mapped_heavy_atom_count") == len(serial_map)
                ),
                "template_atom_count_bound": (
                    row.get("template_heavy_atom_count") == template_atom_count
                ),
                "mapped_template_count_bound": (
                    row.get("mapped_template_heavy_atom_count")
                    == len(mapped_indices)
                ),
                "unmapped_template_count_bound": (
                    row.get("unmapped_template_heavy_atom_count")
                    == raw_unmapped_template_count
                ),
                "effective_unmapped_template_count_bound": (
                    row.get(
                        "effective_unmapped_template_heavy_atom_count",
                        raw_unmapped_template_count,
                    )
                    == effective_unmapped_template_count
                ),
                "mapping_complete_recomputed": (
                    row.get("mapping_complete") is True
                    and set(serial_map) | deferred == observed_serials
                ),
                "mapping_unique_asserted": (
                    row.get("mapping_injective") is True
                    and row.get("mapping_unique") is True
                    and row.get("mapping_candidate_count") == 1
                    and row.get("external_attachment_mapping_unique") is True
                ),
                "template_mapping_complete_recomputed": (
                    row.get("template_mapping_complete") is True
                    and len(mapped_indices) + len(deferred_r3)
                    == template_atom_count
                ),
                "external_indices_bound": all(
                    serial_map.get(serial) == index
                    for serial, index in external_indices.items()
                ),
                "external_indices_canonical": external_indices_canonical,
                "template_stereo_bound": (
                    row.get("template_unassigned_stereocenter_indices")
                    == unassigned
                    and row.get("template_stereo_complete") is (not unassigned)
                ),
            }
            observed_positions.append(position)
            row_audits.append({
                "residue_position": position,
                "unified_symbol": template.symbol,
                "checks": checks,
                "passed": all(checks.values()),
            })
        except Exception as exc:
            row_audits.append({
                "residue_position": row.get("residue_position"),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    expected_positions = sorted(inventory_by_position)
    passed = (
        bool(inventory)
        and len(inventory_by_position) == len(inventory)
        and len(residues) == len(inventory)
        and sorted(observed_positions) == expected_positions
        and len(set(observed_positions)) == len(observed_positions)
        and all(row.get("passed") is True for row in row_audits)
    )
    return _dimension(
        passed,
        expected_residue_count=len(inventory),
        observed_residue_count=len(residues),
        expected_residue_positions=expected_positions,
        observed_residue_positions=sorted(observed_positions),
        expected_source_bound_r3_serials=sorted(
            expected_source_bound_r3_serials
        ),
        expected_r3_atom_names_by_position={
            str(position): sorted(atom_names)
            for position, atom_names in sorted(
                expected_r3_atom_names_by_position.items()
            )
        },
        source_bound_r3_resolution_errors=source_bound_r3_resolution_errors,
        row_audits=row_audits,
    )


def _terminal_r2_ledger_audit(
    evidence: dict[str, Any], input_evidence: dict[str, Any]
) -> dict[str, Any]:
    inventory = list(input_evidence.get("residue_atom_inventory", []))
    terminal = evidence.get("terminal_r2_materialization")
    if not inventory:
        return _dimension(False, reason="missing_residue_inventory")
    terminal_position = len(inventory)
    signatures = _input_closure_signatures(input_evidence)
    head_to_tail = _v5._connection_signature(
        1, "R1", terminal_position, "R2"
    ) in signatures
    occupied = sorted({
        (int(position), str(rgroup).upper())
        for bond in input_evidence.get("cyclization_bonds", [])
        for position, rgroup in (
            (bond.get("position_1"), bond.get("rgroup_1")),
            (bond.get("position_2"), bond.get("rgroup_2")),
        )
    })
    if head_to_tail:
        return _dimension(
            terminal is None,
            head_to_tail=True,
            expected_terminal_ledger=None,
            observed_terminal_ledger=terminal,
        )
    if not isinstance(terminal, dict):
        return _dimension(False, head_to_tail=False, reason="missing_terminal_ledger")
    status = terminal.get("status")
    occupied_bound = terminal.get("occupied_ports") == [
        list(port) for port in occupied
    ]
    position_bound = terminal.get("terminal_position") == terminal_position
    carbon_serials = [
        int(atom["serial"])
        for atom in inventory[-1].get("atoms", [])
        if str(atom.get("atom_name", "")).upper() == "C"
    ]
    terminal_name = str(inventory[-1].get("residue_name", "")).upper()
    terminal_residue_rows = [
        row for row in evidence.get("residue_evidence", [])
        if row.get("residue_position") == terminal_position
    ]
    expected_status = None
    expected_r2_default = None
    expected_cap_element = None
    status_derivation_error = None
    try:
        if (terminal_position, "R2") in occupied:
            expected_status = "not_materialized_port_occupied"
        elif terminal_name in {"NME", "NH2"}:
            expected_status = "explicit_terminal_cap_present"
        else:
            from .paths.residue_template_factory import get_residue_template

            if len(terminal_residue_rows) != 1 or len(carbon_serials) != 1:
                raise ValueError("terminal residue or carbonyl atom is not unique")
            residue_row = terminal_residue_rows[0]
            template = get_residue_template(
                str(residue_row.get("unified_symbol") or "")
            )
            expected_r2_default = str(template.r2).strip().upper()
            cap_atomic_number = {
                "OH": 8,
                "NH2": 7,
                "SH": 16,
            }.get(expected_r2_default)
            if (
                cap_atomic_number is None
                and expected_r2_default not in {"", "-", "H"}
            ):
                raise ValueError(
                    f"unsupported terminal R2 default {expected_r2_default!r}"
                )
            serial_to_template = {
                int(serial): int(index)
                for serial, index in residue_row.get(
                    "serial_to_template_atom_index", {}
                ).items()
            }
            carbon_template_index = serial_to_template.get(carbon_serials[0])
            if carbon_template_index is None:
                raise ValueError("terminal carbonyl serial is not mapped")
            mapped_single_caps = []
            for serial, template_index in serial_to_template.items():
                atom = template.mol.GetAtomWithIdx(template_index)
                bond = template.mol.GetBondBetweenAtoms(
                    carbon_template_index, template_index
                )
                if (
                    serial != carbon_serials[0]
                    and atom.GetAtomicNum() in {7, 8, 16}
                    and bond is not None
                    and bond.GetBondType() == Chem.BondType.SINGLE
                ):
                    mapped_single_caps.append(atom.GetSymbol())
            if len(mapped_single_caps) > 1:
                raise ValueError("multiple mapped terminal cap atoms")
            if mapped_single_caps:
                expected_status = "observed_or_template_present"
                expected_cap_element = mapped_single_caps[0]
            elif cap_atomic_number is None:
                expected_status = "no_materialized_cap_for_R2_default"
            else:
                observed_oxt = [
                    atom for atom in inventory[-1].get("atoms", [])
                    if str(atom.get("atom_name", "")).upper() == "OXT"
                ]
                expected_status = (
                    "materialized_from_observed_terminal_oxt"
                    if expected_r2_default == "OH" and len(observed_oxt) == 1
                    else f"materialized_from_R2_{expected_r2_default}_default"
                )
                expected_cap_element = Chem.GetPeriodicTable().GetElementSymbol(
                    cap_atomic_number
                )
    except Exception as exc:
        status_derivation_error = f"{type(exc).__name__}: {exc}"

    carbon_required = expected_status not in {
        "not_materialized_port_occupied",
        "explicit_terminal_cap_present",
        None,
    }
    carbon_bound = (
        len(carbon_serials) == 1
        and terminal.get("carbon_pdb_serial") == carbon_serials[0]
        if carbon_required else "carbon_pdb_serial" not in terminal
    )
    r2_default_bound = (
        terminal.get("r2_default") == expected_r2_default
        if expected_r2_default is not None else "r2_default" not in terminal
    )
    cap_element_bound = (
        terminal.get("cap_element") == expected_cap_element
        if expected_cap_element is not None else "cap_element" not in terminal
    )
    cap_residue_bound = (
        terminal.get("cap_residue") == terminal_name
        if expected_status == "explicit_terminal_cap_present"
        else "cap_residue" not in terminal
    )
    status_bound = status_derivation_error is None and status == expected_status
    passed = (
        status_bound
        and occupied_bound
        and position_bound
        and carbon_bound
        and r2_default_bound
        and cap_element_bound
        and cap_residue_bound
    )
    return _dimension(
        passed,
        head_to_tail=False,
        status=status,
        expected_status=expected_status,
        status_bound=status_bound,
        status_derivation_error=status_derivation_error,
        occupied_ports_bound=occupied_bound,
        terminal_position_bound=position_bound,
        carbon_serial_bound=carbon_bound,
        r2_default_bound=r2_default_bound,
        cap_element_bound=cap_element_bound,
        cap_residue_bound=cap_residue_bound,
    )


def _verified_path_mapping_source(
    route_rows: list[dict[str, Any]],
    path_evidence: dict[str, dict[str, Any]],
    supplied_path_evidence: dict[str, dict[str, Any]],
    input_evidence: dict[str, Any],
    selected_key: str,
    selected_canonical: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one A/E mapping ledger only after binding every successful row."""
    route_audits = []
    valid_sources = []
    successful_row_count = 0
    for route in ("a", "e"):
        rows = [
            row for row in route_rows
            if row.get("route") == route and row.get("status") == "success"
        ]
        evidence = path_evidence.get(route)
        supplied_evidence = supplied_path_evidence.get(route)
        if not rows:
            route_audits.append({
                "route": route,
                "required": False,
                "successful_route_row_count": 0,
                "non_success_evidence_ignored": isinstance(
                    supplied_evidence, dict
                ),
                "passed": True,
            })
            continue

        successful_row_count += len(rows)
        row = rows[0]
        evidence_row = evidence if isinstance(evidence, dict) else {}
        row_smiles = row.get("output_smiles")
        evidence_smiles = evidence_row.get("output_smiles")
        row_canonical, row_smiles_key = _v5._canonical_identity(
            row_smiles if isinstance(row_smiles, str) else None
        )
        evidence_canonical, evidence_smiles_key = _v5._canonical_identity(
            evidence_smiles if isinstance(evidence_smiles, str) else None
        )
        residue_ledger = _residue_mapping_ledger_audit(
            evidence_row, input_evidence
        )
        terminal_ledger = _terminal_r2_ledger_audit(
            evidence_row, input_evidence
        )
        checks = {
            "unique_successful_route_row": len(rows) == 1,
            "evidence_present": isinstance(evidence, dict),
            "evidence_route_bound": evidence_row.get("route") == route,
            "geometry_mode_bound": (
                evidence_row.get("geometry_inference_enabled") is (route == "e")
            ),
            "generation_provenance_bound": (
                evidence_row.get("generation_provenance") == {
                    "schema_version": "1.0.0-path-a-e-generation-provenance.1",
                    "generator": (
                        "cycpep_master.paths.path_a.generate_with_evidence"
                    ),
                    "requested_route": route,
                    "geometric_cyclization_argument": route == "e",
                    "geometry_stage_executed": route == "e",
                }
            ),
            "evidence_error_field_present": "error" in evidence_row,
            "evidence_error_free": evidence_row.get("error") is None,
            "row_full_inchikey_bound": _route_key(row) == selected_key,
            "row_smiles_identity_bound": (
                row_canonical == selected_canonical
                and row_smiles_key == selected_key
            ),
            "evidence_full_inchikey_bound": (
                evidence_row.get("output_inchikey") == selected_key
            ),
            "evidence_smiles_identity_bound": (
                evidence_canonical == selected_canonical
                and evidence_smiles_key == selected_key
            ),
            "row_evidence_smiles_identity_equal": (
                row_canonical is not None
                and row_canonical == evidence_canonical
            ),
            "residue_mapping_ledger_consistent": residue_ledger["passed"],
            "terminal_r2_ledger_consistent": terminal_ledger["passed"],
        }
        passed = len(rows) == 1 and all(checks.values())
        route_audits.append({
            "route": route,
            "required": True,
            "successful_route_row_count": len(rows),
            "row_canonical_smiles": row_canonical,
            "row_smiles_full_inchikey": row_smiles_key,
            "evidence_canonical_smiles": evidence_canonical,
            "evidence_smiles_full_inchikey": evidence_smiles_key,
            "residue_mapping_ledger": residue_ledger,
            "terminal_r2_ledger": terminal_ledger,
            "checks": checks,
            "passed": passed,
        })
        if passed:
            valid_sources.append((route, evidence_row))

    required_audits = [row for row in route_audits if row["required"]]
    passed = (
        successful_row_count > 0
        and bool(valid_sources)
        and all(row["passed"] for row in required_audits)
    )
    source_route, source = valid_sources[0] if passed else (None, {})
    binding = _dimension(
        passed,
        mapping_source_route=source_route,
        selected_full_inchikey=selected_key,
        selected_canonical_smiles=selected_canonical,
        successful_path_route_row_count=successful_row_count,
        valid_mapping_source_routes=[route for route, _ in valid_sources],
        route_audits=route_audits,
    )
    return source, binding


def _input_closure_signatures(input_evidence: dict[str, Any]) -> set[tuple]:
    return {
        _v5._connection_signature(
            bond["position_1"],
            bond["rgroup_1"],
            bond["position_2"],
            bond["rgroup_2"],
        )
        for bond in input_evidence.get("cyclization_bonds", [])
    }


def _symbolic_trace_passes(
    row: dict[str, Any],
    selected_key: str,
    input_evidence: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    evidence = row.get("explicit_only_evidence")
    if not isinstance(evidence, dict):
        return False, {"reason": "missing_explicit_only_evidence"}
    trace = evidence.get("topology_construction_trace")
    if not isinstance(trace, dict):
        return False, {"reason": "missing_topology_construction_trace"}
    observed = set()
    counterfactuals_complete = True
    for connection in trace.get("connections", []):
        raw = connection.get("signature")
        if not isinstance(raw, list) or len(raw) != 2:
            counterfactuals_complete = False
            continue
        observed.add(
            _v5._connection_signature(
                raw[0][0], raw[0][1], raw[1][0], raw[1][1]
            )
        )
        if not connection.get("identity_changes_when_removed"):
            counterfactuals_complete = False
    expected = _input_closure_signatures(input_evidence)
    passed = (
        evidence.get("status") == "success"
        and evidence.get("output_inchikey") == selected_key
        and evidence.get("geometry_inference_disabled") is True
        and trace.get("status") == "verified"
        and trace.get("selected_inchikey") == selected_key
        and trace.get("allow_geometric_inference") is False
        and trace.get("unexpected_connection_count") == 0
        and bool(expected)
        and observed == expected
        and counterfactuals_complete
    )
    return passed, {
        "route": row.get("route"),
        "expected_closure_count": len(expected),
        "traced_closure_count": len(observed),
        "counterfactuals_complete": counterfactuals_complete,
        "geometry_inference_disabled": evidence.get("geometry_inference_disabled"),
    }


def _path_a_trace_passes(
    evidence: dict[str, Any],
    selected_key: str,
    expected_route: str,
    input_evidence: dict[str, Any],
    *,
    allow_linear: bool = False,
) -> tuple[bool, dict[str, Any]]:
    closures = list(evidence.get("closure_evidence", []))
    expected_bonds = {
        _v5._connection_signature(
            bond["position_1"], bond["rgroup_1"],
            bond["position_2"], bond["rgroup_2"],
        ): bond
        for bond in input_evidence.get("cyclization_bonds", [])
    }
    serials_by_position_and_name = {
        (int(residue["position"]), str(atom["atom_name"]).upper()): int(
            atom["serial"]
        )
        for residue in input_evidence.get("residue_atom_inventory", [])
        for atom in residue.get("atoms", [])
    }
    inventory_by_position = {
        int(residue["position"]): residue
        for residue in input_evidence.get("residue_atom_inventory", [])
    }
    graph_indices_by_serial = {}
    graph_offset = 0
    try:
        from .paths.residue_template_factory import get_residue_template

        residue_rows = sorted(
            evidence.get("residue_evidence", []),
            key=lambda row: int(row.get("residue_position", -1)),
        )
        for residue_row in residue_rows:
            template = get_residue_template(
                str(residue_row.get("unified_symbol") or "")
            )
            template_atom_count = template.mol.GetNumAtoms()
            serial_map = {
                int(serial): int(template_index)
                for serial, template_index in residue_row.get(
                    "serial_to_template_atom_index", {}
                ).items()
            }
            mapped_indices = set(serial_map.values())
            if (
                len(mapped_indices) != len(serial_map)
                or any(
                    not 0 <= template_index < template_atom_count
                    for template_index in mapped_indices
                )
            ):
                raise ValueError("invalid residue template mapping")
            deferred_rows = list(residue_row.get(
                "deferred_consumed_r3_template_atoms", []
            ))
            declared_deferred = {
                int(deferred["template_atom_index"])
                for deferred in deferred_rows
                if isinstance(deferred, dict)
                and deferred.get("template_atom_index") is not None
            }
            removed_indices = (
                set(range(template_atom_count)) - mapped_indices
            )
            if (
                len(declared_deferred) != len(deferred_rows)
                or declared_deferred != removed_indices
            ):
                raise ValueError("unledgered residue template atom removal")
            for serial, template_index in serial_map.items():
                graph_indices_by_serial[serial] = graph_offset + template_index - sum(
                    removed_index < template_index
                    for removed_index in removed_indices
                )
            graph_offset += template_atom_count - len(removed_indices)
    except Exception:
        graph_indices_by_serial = {}
    closure_audits = []
    observed_signatures = []
    for closure in closures:
        try:
            endpoint_1 = closure["endpoint_1"]
            endpoint_2 = closure["endpoint_2"]
            signature = _v5._connection_signature(
                endpoint_1["residue_position"], endpoint_1["rgroup"],
                endpoint_2["residue_position"], endpoint_2["rgroup"],
            )
            expected = expected_bonds[signature]
            endpoint_signature = sorted((
                (
                    endpoint_1["residue_position"],
                    endpoint_1.get("atom_name") or endpoint_1.get("rgroup"),
                ),
                (
                    endpoint_2["residue_position"],
                    endpoint_2.get("atom_name") or endpoint_2.get("rgroup"),
                ),
            ))
            expected_closure_id = hashlib.sha256(
                repr(endpoint_signature).encode("utf-8")
            ).hexdigest()[:16]

            def endpoint_checks(endpoint: dict[str, Any], first: bool) -> bool:
                position = int(endpoint["residue_position"])
                atom_name = str(endpoint["atom_name"]).upper()
                source_residue = inventory_by_position[position]
                expected_position = int(
                    expected["position_1" if first else "position_2"]
                )
                expected_atom = str(
                    expected["atom_1" if first else "atom_2"]
                ).upper()
                expected_rgroup = str(
                    expected["rgroup_1" if first else "rgroup_2"]
                ).upper()
                serial = serials_by_position_and_name[(position, atom_name)]
                return all((
                    position == expected_position,
                    atom_name == expected_atom,
                    str(endpoint["rgroup"]).upper() == expected_rgroup,
                    endpoint.get("candidate_serials") == [serial],
                    endpoint.get("pdb_serial") == serial,
                    endpoint.get("pdb_resseq")
                    == source_residue.get("residue_number"),
                    endpoint.get("resname")
                    == source_residue.get("residue_name"),
                    endpoint.get("resolution_unique") is True,
                    endpoint.get("final_graph_atom_index")
                    == graph_indices_by_serial.get(serial),
                ))

            checks = {
                "closure_id_bound": closure.get("closure_id") == expected_closure_id,
                "bond_type_bound": closure.get("bond_type") == expected["bond_type"],
                "evidence_source_bound": (
                    closure.get("evidence_source") == expected["evidence_source"]
                ),
                "evidence_explicit": closure.get("evidence_is_explicit") is True,
                "endpoint_1_bound": endpoint_checks(endpoint_1, True),
                "endpoint_2_bound": endpoint_checks(endpoint_2, False),
                "distinct_graph_endpoints": (
                    endpoint_1.get("final_graph_atom_index")
                    != endpoint_2.get("final_graph_atom_index")
                ),
                "endpoints_resolved": (
                    closure.get("endpoints_resolved_uniquely") is True
                ),
                "materialized": closure.get("materialized") is True,
                "graph_bond_type_bound": closure.get("graph_bond_type") == "SINGLE",
                "baseline_identity_bound": (
                    closure.get("baseline_inchikey") == selected_key
                ),
                "counterfactual_complete": (
                    closure.get("counterfactual_status") == "success"
                    and closure.get("counterfactual_error") is None
                    and bool(closure.get("counterfactual_inchikey"))
                    and closure.get("counterfactual_inchikey") != selected_key
                    and closure.get("identity_changes_when_removed") is True
                ),
            }
            observed_signatures.append(signature)
            closure_audits.append({
                "closure_id": closure.get("closure_id"),
                "signature": [list(item) for item in signature],
                "checks": checks,
                "passed": all(checks.values()),
            })
        except Exception as exc:
            closure_audits.append({
                "closure_id": (
                    closure.get("closure_id") if isinstance(closure, dict) else None
                ),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    expected_signatures = set(expected_bonds)
    trace_ledger_passed = (
        len(closures) == len(expected_bonds)
        and set(observed_signatures) == expected_signatures
        and len(set(observed_signatures)) == len(observed_signatures)
        and all(row.get("passed") is True for row in closure_audits)
    )
    if allow_linear:
        # Linear topology asserts the complete absence of closures: the v5
        # input audit classified the selected chain as closure-less, so a
        # successful A/E route must have generated zero closures whose output
        # identity is still bound.  The all-closures flags are vacuous for an
        # empty closure set, so they are not consulted in this branch.
        passed = (
            evidence.get("route") == expected_route
            and evidence.get("geometry_inference_enabled") is (expected_route == "e")
            and evidence.get("output_inchikey") == selected_key
            and not closures
            and evidence.get("closure_count") == 0
            and evidence.get("all_closure_endpoints_resolved_uniquely") is True
            and evidence.get("all_closures_materialized") is True
            and trace_ledger_passed
        )
    else:
        passed = (
            evidence.get("route") == expected_route
            and evidence.get("geometry_inference_enabled") is (expected_route == "e")
            and evidence.get("output_inchikey") == selected_key
            and bool(closures)
            and evidence.get("closure_count") == len(closures)
            and evidence.get("all_closure_endpoints_resolved_uniquely") is True
            and evidence.get("all_closures_materialized") is True
            and evidence.get("all_closures_explicit") is True
            and evidence.get("all_closures_identity_determining") is True
            and trace_ledger_passed
        )
    return passed, {
        "route": evidence.get("route"),
        "expected_route": expected_route,
        "geometry_inference_enabled": evidence.get("geometry_inference_enabled"),
        "closure_count": len(closures),
        "expected_closure_count": len(expected_bonds),
        "trace_ledger_passed": trace_ledger_passed,
        "allow_linear": allow_linear,
        "closure_audits": closure_audits,
        "all_closure_endpoints_resolved_uniquely": evidence.get(
            "all_closure_endpoints_resolved_uniquely"
        ),
        "all_closures_materialized": evidence.get("all_closures_materialized"),
        "all_closures_explicit": evidence.get("all_closures_explicit"),
        "all_closures_identity_determining": evidence.get(
            "all_closures_identity_determining"
        ),
    }


def _chem_comp_stereo_fallback(
    evidence: dict[str, Any],
    input_evidence: dict[str, Any],
    mapping_source: dict[str, Any],
    molecule: Chem.Mol,
    output_unassigned: list[int],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Authorize CIF R/S only after serial-level coverage is closed."""
    audit = input_evidence.get("chem_comp_evidence_audit")
    base = {
        "enabled": bool(enabled),
        "authority": "_chem_comp_atom.pdbx_stereo_config",
        "passed": False,
        "required_serials": [],
        "declared_serials": [],
        "unassigned_output_centers": list(output_unassigned),
        "reason": None,
    }
    if not enabled:
        base["reason"] = "chem_comp_evidence_disabled"
        return base
    if not isinstance(audit, dict) or audit.get("status") != "validated":
        base["reason"] = "chem_comp_evidence_not_validated"
        return base
    if audit.get("stereo_coverage_complete") is not True:
        base["reason"] = audit.get("stereo_failure_reason") or "stereo_coverage_incomplete"
        return base

    required: set[int] = set()
    binding_errors: list[str] = []
    for residue in mapping_source.get("residue_evidence", []):
        if not isinstance(residue, dict):
            binding_errors.append("malformed_residue_evidence")
            continue
        try:
            from .paths.residue_template_factory import get_residue_template
            template = get_residue_template(str(residue.get("unified_symbol") or ""))
            serial_map = {
                int(serial): int(index)
                for serial, index in residue.get(
                    "serial_to_template_atom_index", {}
                ).items()
            }
            centers = {
                int(index)
                for index, _assignment in Chem.FindMolChiralCenters(
                    template.mol,
                    includeUnassigned=True,
                    useLegacyImplementation=False,
                )
            }
            for index in centers:
                matches = [serial for serial, mapped in serial_map.items() if mapped == index]
                if len(matches) != 1:
                    binding_errors.append(
                        f"template stereocenter {residue.get('residue_position')}:{index} is not uniquely mapped"
                    )
                else:
                    required.add(matches[0])
        except Exception as exc:
            binding_errors.append(f"{type(exc).__name__}: {exc}")

    declared = {
        int(serial): str(config).upper()
        for serial, config in (audit.get("_stereo_centers") or {}).items()
    }
    base["required_serials"] = sorted(required)
    base["declared_serials"] = sorted(declared)
    if binding_errors:
        base["binding_errors"] = binding_errors
        base["reason"] = "stereo_center_binding_failed"
        return base
    if required and set(declared) != required:
        base["reason"] = "stereo_center_coverage_incomplete"
        return base
    if not declared and output_unassigned:
        base["reason"] = "unassigned_output_stereocenters_without_authority"
        return base
    # The authority is accepted only as a complete, serial-bound fallback. It
    # never repairs graph, mapping, connectivity, or candidate conflicts.
    base["passed"] = True
    base["reason"] = None
    return base


def _mapping_dimensions(
    evidence: dict[str, Any],
    *,
    input_evidence: dict[str, Any] | None = None,
    chem_comp_evidence_enabled: bool = False,
    output_molecule: Chem.Mol | None = None,
    output_unassigned: list[int] | None = None,
) -> tuple[dict, dict, dict]:
    residues = list(evidence.get("residue_evidence", []))
    sources = sorted({str(row.get("unified_source", "")) for row in residues})
    library_passed = bool(residues) and evidence.get("library_chemistry_unique") is True
    mapping_passed = (
        bool(residues)
        and evidence.get("atom_mapping_complete") is True
        and evidence.get("atom_mapping_unique") is True
        and all(
            row.get(
                "effective_unmapped_template_heavy_atom_count",
                row.get("unmapped_template_heavy_atom_count"),
            )
            == 0
            for row in residues
        )
    )
    stereo_passed = bool(residues) and all(
        row.get("template_stereo_complete") is True
        and isinstance(row.get("coordinate_stereochemistry"), dict)
        and row["coordinate_stereochemistry"].get("passed") is True
        for row in residues
    ) and (
        isinstance(evidence.get("assembled_coordinate_stereochemistry"), dict)
        and evidence["assembled_coordinate_stereochemistry"].get("passed") is True
    ) and (
        isinstance(evidence.get("emergent_stereochemistry"), dict)
        and evidence["emergent_stereochemistry"].get("passed") is True
    )
    chem_comp_stereo_audit: dict[str, Any] = {
        "enabled": bool(chem_comp_evidence_enabled),
        "passed": False,
        "reason": "not_evaluated",
    }
    if (
        not stereo_passed
        and chem_comp_evidence_enabled
        and bool(residues)
        and output_molecule is not None
    ):
        chem_comp_stereo_audit = _chem_comp_stereo_fallback(
            evidence,
            input_evidence or {},
            evidence,
            output_molecule,
            list(output_unassigned or []),
            enabled=True,
        )
        if chem_comp_stereo_audit.get("passed") is True:
            stereo_passed = True
    return (
        _dimension(
            library_passed,
            residue_count=len(residues),
            unified_sources=sources,
            monomer_graph_sha256=[
                row.get("monomer_graph_sha256") for row in residues
            ],
        ),
        _dimension(
            mapping_passed,
            atom_mapping_complete=evidence.get("atom_mapping_complete"),
            atom_mapping_unique=evidence.get("atom_mapping_unique"),
            unmapped_template_heavy_atom_count=sum(
                int(row.get("unmapped_template_heavy_atom_count", 0))
                for row in residues
            ),
            deferred_consumed_r3_template_atom_count=sum(
                int(row.get("deferred_consumed_r3_count", 0))
                for row in residues
            ),
            effective_unmapped_template_heavy_atom_count=sum(
                int(row.get(
                    "effective_unmapped_template_heavy_atom_count",
                    row.get("unmapped_template_heavy_atom_count", 0),
                ))
                for row in residues
            ),
        ),
        _dimension(
            stereo_passed,
            template_unassigned_stereocenter_count=sum(
                len(row.get("template_unassigned_stereocenter_indices", []))
                for row in residues
            ),
            coordinate_stereochemistry=[
                row.get("coordinate_stereochemistry") for row in residues
            ],
            assembled_coordinate_stereochemistry=evidence.get(
                "assembled_coordinate_stereochemistry"
            ),
            emergent_stereochemistry=evidence.get("emergent_stereochemistry"),
            chem_comp_stereochemistry=chem_comp_stereo_audit,
        ),
    )


def _local_monomer_evidence_consistency_dimension(
    route_rows: list[dict[str, Any]],
    input_evidence: dict[str, Any],
    mapping_source: dict[str, Any],
    selected_key: str,
) -> dict[str, Any]:
    """Bind each inferred graph and optional R3 port to its unique PDB mapping."""
    bootstrap = input_evidence.get("local_monomer_bootstrap")
    if not isinstance(bootstrap, dict):
        return _dimension(True, required=False, reason="no_entity_local_monomer")
    symbolic = []
    for row in route_rows:
        if str(row.get("route")) not in {"b", "g"}:
            continue
        passed, details = _symbolic_trace_passes(row, selected_key, input_evidence)
        symbolic.append({**details, "passed": passed})

    residue_rows = list(mapping_source.get("residue_evidence", []))
    inference_checks = []
    for inference in bootstrap.get("inference_results", []):
        if not isinstance(inference, dict):
            inference_checks.append({
                "passed": False,
                "reason": "malformed_inference_result",
            })
            continue
        residue_key = list(inference.get("residue_key", []))
        matches = [
            row for row in residue_rows
            if list(row.get("residue_key", [])) == residue_key
        ]
        candidate = Chem.MolFromSmiles(str(inference.get("candidate_smiles") or ""))
        canonical = (
            Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
            if candidate is not None else ""
        )
        candidate_graph_sha256 = (
            hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if canonical else None
        )
        template_graph_sha256 = (
            matches[0].get("free_monomer_graph_sha256")
            if len(matches) == 1 else None
        )
        port = inference.get("r3_port")
        serial = str(port.get("pdb_serial")) if isinstance(port, dict) else None
        mapped_index = None
        anchor_index = None
        r3_passed = True
        if isinstance(port, dict):
            mapped_index = (
                matches[0].get("external_attachment_indices", {}).get(serial)
                if len(matches) == 1 else None
            )
            anchor_index = (
                matches[0].get("r3_anchor_template_atom_index")
                if len(matches) == 1 else None
            )
            r3_passed = (
                mapped_index is not None
                and anchor_index is not None
                and int(mapped_index) == int(anchor_index)
            )
        inference_evidence = inference.get("evidence")
        resolution_mode = (
            inference_evidence.get("resolution_mode")
            if isinstance(inference_evidence, dict) else None
        )
        direct_match_checks = {
            "required": True,
            "passed": False,
            "reason": "missing_or_unsupported_resolution_mode",
            "allowed_resolution_modes": [
                "embedded_mmcif_chem_comp",
                "locally_inferred",
                "unified_library_match",
            ],
        }
        if resolution_mode == "unified_library_match":
            ledger = inference_evidence.get("library_first_match")
            selected_match = (
                ledger.get("selected_match_evidence")
                if isinstance(ledger, dict) else None
            )
            selected_identity = (
                ledger.get("selected_identity")
                if isinstance(ledger, dict) else None
            )
            path_match = matches[0] if len(matches) == 1 else {}
            candidate_key = (
                Chem.MolToInchiKey(candidate) if candidate is not None else None
            )
            ledger_explicit = (
                ledger.get("connectivity", {}).get("explicit_edges")
                if isinstance(ledger, dict) else None
            )
            selected_explicit = (
                selected_match.get("explicit_intra_residue_edges")
                if isinstance(selected_match, dict) else None
            )
            checks = {
                "released": isinstance(ledger, dict)
                and ledger.get("released") is True,
                "one_identity": isinstance(ledger, dict)
                and ledger.get("strict_identity_count") == 1,
                "selected_identity_bound": isinstance(selected_identity, list)
                and len(selected_identity) == 2
                and selected_identity[0] == candidate_key
                and isinstance(selected_match, dict)
                and selected_identity[1] == selected_match.get("port_identity"),
                "selected_symbol_bound": isinstance(ledger, dict)
                and ledger.get("selected_symbol") == path_match.get("unified_symbol"),
                "selected_graph_bound": isinstance(selected_match, dict)
                and selected_match.get("full_inchikey") == candidate_key
                and selected_match.get("free_monomer_graph_sha256")
                == candidate_graph_sha256
                and selected_match.get("free_monomer_graph_sha256")
                == template_graph_sha256,
                "mapping_strict": isinstance(selected_match, dict)
                and all(selected_match.get(name) is True for name in (
                    "mapping_complete", "mapping_injective",
                    "template_mapping_complete", "mapping_unique",
                    "external_attachment_mapping_unique",
                )),
                "mapping_bound_to_path": isinstance(selected_match, dict)
                and selected_match.get("serial_to_template_atom_index")
                == path_match.get("serial_to_template_atom_index"),
                "bond_order_geometry": isinstance(selected_match, dict)
                and isinstance(selected_match.get("bond_order_geometry"), dict)
                and selected_match["bond_order_geometry"].get("passed") is True,
                "coordinate_stereochemistry": isinstance(selected_match, dict)
                and isinstance(
                    selected_match.get("coordinate_stereochemistry"), dict
                )
                and selected_match["coordinate_stereochemistry"].get("passed")
                is True,
                "explicit_edges_bound": ledger_explicit == selected_explicit,
            }
            direct_match_checks = {
                "required": True,
                "passed": all(checks.values()),
                "checks": checks,
                "selected_symbol": (
                    ledger.get("selected_symbol") if isinstance(ledger, dict)
                    else None
                ),
                "selected_identity": selected_identity,
            }
        elif resolution_mode == "embedded_mmcif_chem_comp":
            embedded = (
                inference_evidence.get("embedded_chem_comp_resolution")
                if isinstance(inference_evidence, dict) else None
            )
            embedded = embedded if isinstance(embedded, dict) else {}
            coordinate = input_evidence.get("coordinate_input")
            coordinate = coordinate if isinstance(coordinate, dict) else {}
            templates = coordinate.get("embedded_chem_comp_templates")
            templates = templates if isinstance(templates, dict) else {}
            component = templates.get(str(inference.get("pdb_resname") or ""))
            component = component if isinstance(component, dict) else {}
            manifests = bootstrap.get("derived_manifests", [])
            manifest_matches = [
                row for row in manifests
                if row.get("component_snapshot_sha256")
                == inference_evidence.get("component_snapshot_sha256")
            ]
            aliases = bootstrap.get("pdb_aliases", [])
            alias_matches = [
                row for row in aliases
                if row.get("pdb_resname") == inference.get("pdb_resname")
                and row.get("component_snapshot_sha256")
                == inference_evidence.get("component_snapshot_sha256")
            ]
            path_match = matches[0] if len(matches) == 1 else {}
            candidate_key = (
                Chem.MolToInchiKey(candidate) if candidate is not None else None
            )
            manifest_bound = len(manifest_matches) == 1 and (
                manifest_matches[0].get("component_source_input_sha256")
                == component.get("source_input_sha256")
                and manifest_matches[0].get("resolution_mode")
                == "embedded_mmcif_chem_comp"
            )
            alias_bound = len(alias_matches) == 1 and all((
                alias_matches[0].get("component_source_input_sha256")
                == component.get("source_input_sha256"),
                alias_matches[0].get("resolution_mode")
                == "embedded_mmcif_chem_comp",
                alias_matches[0].get("graph_sha256")
                == candidate_graph_sha256,
                alias_matches[0].get("full_inchikey") == candidate_key,
                alias_matches[0].get("target_symbol")
                == path_match.get("unified_symbol"),
            ))
            checks = {
                "released": inference_evidence.get("released") is True,
                "component_present_in_coordinate_audit": bool(component),
                "snapshot_bound": bool(component)
                and component.get("component_snapshot_sha256")
                == inference_evidence.get("component_snapshot_sha256")
                == embedded.get("component_snapshot_sha256"),
                "source_payload_bound": bool(component)
                and component.get("source_input_sha256")
                == inference_evidence.get("component_source_input_sha256")
                == embedded.get("source_input_sha256")
                == coordinate.get("embedded_chem_comp_source_payload_sha256"),
                "identity_bound": embedded.get("full_inchikey") == candidate_key,
                "observed_connectivity_exact": (
                    embedded.get("observed_connectivity_exact") is True
                ),
                "stereo_source_bound": embedded.get("stereochemistry_source")
                == "_chem_comp_atom.pdbx_stereo_config",
                "bond_order_source_bound": embedded.get("bond_order_source")
                == "_chem_comp_bond.value_order",
                "manifest_or_alias_bound": manifest_bound != alias_bound,
                "path_graph_bound": path_match.get("free_monomer_graph_sha256")
                == candidate_graph_sha256,
            }
            direct_match_checks = {
                "required": True,
                "passed": all(checks.values()),
                "checks": checks,
                "component_snapshot_sha256": inference_evidence.get(
                    "component_snapshot_sha256"
                ),
                "component_source_input_sha256": inference_evidence.get(
                    "component_source_input_sha256"
                ),
                "materialization_binding": (
                    "derived_manifest" if manifest_bound
                    else "unified_library_alias" if alias_bound
                    else None
                ),
            }
        elif resolution_mode == "locally_inferred":
            path_match = matches[0] if len(matches) == 1 else {}
            candidate_key = (
                Chem.MolToInchiKey(candidate) if candidate is not None else None
            )
            manifests = bootstrap.get("derived_manifests", [])
            manifest_matches = [
                row for row in manifests
                if row.get("graph_sha256") == candidate_graph_sha256
                and row.get("full_inchikey") == candidate_key
                and row.get("input_sha256") == inference_evidence.get("input_sha256")
                and row.get("resolution_mode") == "locally_inferred"
            ]
            aliases = bootstrap.get("pdb_aliases", [])
            alias_matches = [
                row for row in aliases
                if row.get("pdb_resname") == inference.get("pdb_resname")
                and row.get("graph_sha256") == candidate_graph_sha256
                and row.get("full_inchikey") == candidate_key
                and row.get("input_sha256") == inference_evidence.get("input_sha256")
                and row.get("resolution_mode") == "locally_inferred"
            ]
            manifest = manifest_matches[0] if len(manifest_matches) == 1 else {}
            alias = alias_matches[0] if len(alias_matches) == 1 else {}
            library_attempt = inference_evidence.get("library_first_match")
            library_attempt = (
                library_attempt if isinstance(library_attempt, dict) else {}
            )
            connectivity = inference_evidence.get("connectivity")
            connectivity = connectivity if isinstance(connectivity, dict) else {}
            checks = {
                "inference_version_bound": (
                    inference_evidence.get("inference_version")
                    == "local-monomer-inference-1"
                ),
                "candidate_identity_bound": (
                    inference_evidence.get("candidate_smiles") == [canonical]
                    and inference_evidence.get("candidate_graph_count") == 1
                    and inference_evidence.get("uncertain_candidate_count") == 0
                ),
                "connectivity_unique": (
                    connectivity.get("connectivity_candidate_count") == 1
                    and not connectivity.get("ambiguous_geometry_edges")
                ),
                "library_fallback_bound": (
                    library_attempt.get("resolution_mode")
                    == "unified_library_match"
                    and library_attempt.get("released") is False
                    and bool(library_attempt.get("fallback_reason_codes"))
                ),
                "one_manifest": len(manifest_matches) == 1,
                "one_alias": len(alias_matches) == 1,
                "manifest_alias_symbol_bound": (
                    bool(manifest)
                    and alias.get("target_symbol") == manifest.get("symbol")
                    == path_match.get("unified_symbol")
                ),
                "manifest_alias_source_bound": (
                    bool(manifest)
                    and manifest.get("source_entity_id")
                    == alias.get("source_entity_id")
                ),
                "path_graph_bound": (
                    path_match.get("free_monomer_graph_sha256")
                    == candidate_graph_sha256
                ),
            }
            direct_match_checks = {
                "required": True,
                "passed": all(checks.values()),
                "checks": checks,
                "manifest_symbol": manifest.get("symbol"),
                "alias_target_symbol": alias.get("target_symbol"),
                "input_sha256": inference_evidence.get("input_sha256"),
            }
        passed = (
            inference.get("status") == "unique"
            and inference.get("candidate_graph_count") == 1
            and candidate_graph_sha256 is not None
            and len(matches) == 1
            and candidate_graph_sha256 == template_graph_sha256
            and matches[0].get("mapping_complete") is True
            and matches[0].get("mapping_unique") is True
            and r3_passed
            and direct_match_checks["passed"]
        )
        inference_checks.append({
            "residue_key": residue_key,
            "pdb_resname": inference.get("pdb_resname"),
            "candidate_graph_sha256": candidate_graph_sha256,
            "template_graph_sha256": template_graph_sha256,
            "mapping_match_count": len(matches),
            "pdb_serial": port.get("pdb_serial") if isinstance(port, dict) else None,
            "mapped_template_atom_index": mapped_index,
            "r3_anchor_template_atom_index": anchor_index,
            "r3_required": isinstance(port, dict),
            "r3_passed": r3_passed,
            "resolution_mode": resolution_mode,
            "direct_match_checks": direct_match_checks,
            "passed": passed,
        })
    symbolic_passed = any(row["passed"] for row in symbolic)
    passed = bool(inference_checks) and all(
        row["passed"] for row in inference_checks
    )
    return _dimension(
        passed,
        required=True,
        expected_inchikey=selected_key,
        inference_checks=inference_checks,
        symbolic_reassemblies=symbolic,
        symbolic_cross_check_required=False,
        symbolic_cross_check_available=bool(symbolic),
        symbolic_cross_check_passed=symbolic_passed,
    )


def _audited_graph_context(
    canonical: str,
    input_evidence: dict[str, Any],
    mapping_source: dict[str, Any],
    minimum_macrocycle_ring_size: int,
) -> dict[str, Any]:
    """Authorize only a ledgered free-R2 hydroxyl beyond observed atoms."""
    selected_canonical, selected_key = _v5._canonical_identity(canonical)
    source_canonical, source_key = _v5._canonical_identity(
        mapping_source.get("output_smiles")
        if isinstance(mapping_source.get("output_smiles"), str) else None
    )
    source_binding = {
        "route": mapping_source.get("route"),
        "selected_canonical_smiles": selected_canonical,
        "selected_full_inchikey": selected_key,
        "source_canonical_smiles": source_canonical,
        "source_smiles_full_inchikey": source_key,
        "source_declared_full_inchikey": mapping_source.get("output_inchikey"),
    }
    source_binding["passed"] = all((
        mapping_source.get("route") in {"a", "e"},
        selected_canonical is not None,
        selected_canonical == source_canonical,
        selected_key is not None,
        selected_key == source_key,
        selected_key == mapping_source.get("output_inchikey"),
    ))
    if not source_binding["passed"]:
        raise ValueError("graph audit mapping source identity is not bound")
    try:
        context = _v5._output_context(
            canonical, input_evidence, minimum_macrocycle_ring_size
        )
        return {**context, "mapping_source_graph_binding": source_binding}
    except Exception:
        if not isinstance(input_evidence.get("local_monomer_bootstrap"), dict):
            raise
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError("selected output could not be reparsed")
    Chem.SanitizeMol(molecule)
    output_counts = _v5._atom_element_counts(molecule)
    input_counts = dict(input_evidence["heavy_element_counts"])
    delta = {
        element: output_counts.get(element, 0) - input_counts.get(element, 0)
        for element in sorted(set(output_counts) | set(input_counts))
    }
    if any(value < 0 for value in delta.values()):
        raise ValueError(f"selected output deletes observed heavy atoms: {delta}")
    terminal = mapping_source.get("terminal_r2_materialization")
    expected_delta = {"O": 1} if (
        isinstance(terminal, dict)
        and terminal.get("status") == "materialized_from_R2_OH_default"
    ) else {}
    positive_delta = {key: value for key, value in delta.items() if value > 0}
    if positive_delta != expected_delta:
        raise ValueError(
            f"output additions {positive_delta} do not equal ledgered R2 additions "
            f"{expected_delta}"
        )
    ring_sizes = sorted(
        (len(ring) for ring in Chem.GetSymmSSSR(molecule)), reverse=True
    )
    largest_ring = ring_sizes[0] if ring_sizes else 0
    if largest_ring < minimum_macrocycle_ring_size:
        if str(input_evidence.get("topology_class")) != "linear":
            raise ValueError(
                f"largest perceived output ring has {largest_ring} atoms; required "
                f">={minimum_macrocycle_ring_size}"
            )
        # Linear inputs are asserted by the closed input audit, so an acyclic
        # graph is not a defect; retain the field values (largest_ring=0, the
        # requested threshold) for auditors to inspect below.
    return {
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "heavy_element_counts": output_counts,
        "connected_component_count": len(Chem.GetMolFrags(molecule)),
        "ring_sizes": ring_sizes,
        "largest_ring_size": largest_ring,
        "minimum_required_ring_size": minimum_macrocycle_ring_size,
        "sanitized": True,
        "heavy_atom_conserved": not positive_delta,
        "template_completion_ledger": {
            "authority": "entity_local_R2_default_and_path_a_materialization",
            "added_element_counts": expected_delta,
            "carbon_pdb_serial": terminal.get("carbon_pdb_serial"),
            "ledger_closed": True,
        } if expected_delta else None,
        "mapping_source_graph_binding": source_binding,
    }
def _attach_path_evidence(
    route_rows: list[dict[str, Any]],
    execution_artifacts: Mapping[Any, Any],
    pdb_path: str | Path,
    chain_id: str,
) -> dict[str, dict[str, Any]]:
    by_route: dict[str, dict[str, Any]] = {}
    for route in ("a", "e"):
        evidence = execution_artifacts.get((route, "path_evidence"))
        if not isinstance(evidence, dict):
            smiles, error, evidence = generate_with_evidence(
                str(pdb_path),
                chain_id,
                geometric_cyclization=(route == "e"),
            )
            evidence = dict(evidence)
            evidence["output_smiles"] = smiles
            evidence["error"] = error
        evidence = dict(evidence)
        by_route[route] = evidence
        for row in route_rows:
            if row.get("route") == route:
                row["evidence_dimensions_input"] = evidence
                break
    return by_route


def _path_evidence_sha256(evidence: dict[str, Any]) -> str:
    payload = json.dumps(
        evidence,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fresh_path_evidence_replay(
    route_rows: list[dict[str, Any]],
    path_evidence: dict[str, dict[str, Any]],
    pdb_path: str | Path | None,
    chain_id: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Regenerate every successful A/E ledger before it can be trusted."""
    required_routes = sorted({
        str(row.get("route"))
        for row in route_rows
        if row.get("route") in {"a", "e"} and row.get("status") == "success"
    })
    if not required_routes:
        return {}, _dimension(True, required=False, route_audits=[])
    if pdb_path is None or not chain_id:
        return {}, _dimension(
            False,
            required=True,
            reason="missing_source_for_fresh_path_evidence_replay",
            required_routes=required_routes,
            route_audits=[],
        )

    fresh_by_route = {}
    route_audits = []
    for route in required_routes:
        smiles, error, evidence = generate_with_evidence(
            str(pdb_path),
            str(chain_id),
            geometric_cyclization=(route == "e"),
            _use_cache=False,
        )
        fresh = dict(evidence)
        fresh["output_smiles"] = smiles
        fresh["error"] = error
        supplied = path_evidence.get(route)
        supplied = supplied if isinstance(supplied, dict) else {}
        rows = [
            row for row in route_rows
            if row.get("route") == route and row.get("status") == "success"
        ]
        row = rows[0] if len(rows) == 1 else {}
        fresh_canonical, fresh_key = _v5._canonical_identity(smiles)
        row_canonical, row_key = _v5._canonical_identity(row.get("output_smiles"))
        supplied_sha256 = _path_evidence_sha256(supplied)
        fresh_sha256 = _path_evidence_sha256(fresh)
        checks = {
            "unique_successful_route_row": len(rows) == 1,
            "fresh_generation_succeeded": error is None and smiles is not None,
            "fresh_route_bound": fresh.get("route") == route,
            "fresh_geometry_mode_bound": (
                fresh.get("geometry_inference_enabled") is (route == "e")
            ),
            "fresh_generation_provenance_bound": (
                fresh.get("generation_provenance") == {
                    "schema_version": "1.0.0-path-a-e-generation-provenance.1",
                    "generator": (
                        "cycpep_master.paths.path_a.generate_with_evidence"
                    ),
                    "requested_route": route,
                    "geometric_cyclization_argument": route == "e",
                    "geometry_stage_executed": route == "e",
                }
            ),
            "supplied_evidence_exactly_replayed": (
                supplied == fresh and supplied_sha256 == fresh_sha256
            ),
            "row_identity_exactly_replayed": (
                row_canonical is not None
                and row_canonical == fresh_canonical
                and row_key == fresh_key
                and row.get("output_inchikey") == fresh_key
            ),
        }
        passed = all(checks.values())
        route_audits.append({
            "route": route,
            "checks": checks,
            "supplied_evidence_sha256": supplied_sha256,
            "fresh_evidence_sha256": fresh_sha256,
            "passed": passed,
        })
        if passed:
            fresh_by_route[route] = fresh
    return fresh_by_route, _dimension(
        len(route_audits) == len(required_routes)
        and all(row["passed"] for row in route_audits),
        required=True,
        required_routes=required_routes,
        route_audits=route_audits,
    )


def _adjudicate_evidence_dimensions(
    route_rows: list[dict[str, Any]],
    input_evidence: dict[str, Any],
    path_evidence: dict[str, dict[str, Any]],
    *,
    minimum_macrocycle_ring_size: int = 8,
    pdb_path: str | Path | None = None,
    chain_id: str | None = None,
) -> StrictReconstructionResult:
    candidates = [
        row for row in route_rows
        if row.get("route") in _CHEMICAL_ROUTES
        and row.get("status") == "success"
        and _route_key(row)
        and row.get("output_smiles")
    ]
    input_repair_codes = sorted(set(input_evidence.get("repair_codes", [])))
    linear_topology = str(input_evidence.get("topology_class")) == "linear"
    candidate_keys = sorted({_route_key(row) for row in candidates})
    consistency = _dimension(
        len(candidate_keys) == 1,
        successful_candidate_count=len(candidates),
        distinct_full_inchikeys=candidate_keys,
    )
    if not candidates:
        return _result(
            "rejected",
            "V6_NO_SUCCESSFUL_CHEMICAL_CANDIDATE",
            "no chemical reconstruction route produced an auditable molecular candidate",
            support_status="supported",
            route_rows=route_rows,
            input_evidence=input_evidence,
            output_evidence={"evidence_dimensions": {"candidate_consistency": consistency}},
            repair_codes=input_repair_codes,
        )
    if len(candidate_keys) != 1:
        return _result(
            "rejected",
            "V6_CONFLICTING_SUCCESSFUL_CANDIDATE_IDENTITIES",
            f"successful chemical candidates emitted {len(candidate_keys)} full InChIKeys",
            support_status="supported",
            route_rows=route_rows,
            input_evidence=input_evidence,
            output_evidence={"evidence_dimensions": {"candidate_consistency": consistency}},
            repair_codes=input_repair_codes,
        )
    selected_key = str(candidate_keys[0])
    family_keys: dict[str, set[str]] = {}
    for row in candidates:
        family = _v5.ROUTE_FAMILIES.get(str(row.get("route")))
        if family not in _v5.QUALIFYING_FAMILIES:
            continue
        family_keys.setdefault(family, set()).add(str(_route_key(row)))
    qualifying_families = sorted(family_keys)
    observed_family_consensus = bool(
        len(qualifying_families) >= 2
        and all(len(keys) == 1 for keys in family_keys.values())
        and len({
            next(iter(keys))
            for keys in family_keys.values()
            if len(keys) == 1
        }) == 1
    )
    independent_family_consensus = _dimension(
        observed_family_consensus or linear_topology,
        required=not linear_topology,
        observed_family_consensus=observed_family_consensus,
        required_family_count=0 if linear_topology else 2,
        successful_qualifying_families=qualifying_families,
        family_full_inchikeys={
            family: sorted(keys) for family, keys in family_keys.items()
        },
    )
    diagnostic_keys = sorted(_v5._raw_geometric_keys(route_rows))
    disagreeing_diagnostic_keys = [
        key for key in diagnostic_keys if key != selected_key
    ]
    diagnostic_identity_consistency = _dimension(
        not disagreeing_diagnostic_keys or linear_topology,
        required=bool(diagnostic_keys) and not linear_topology,
        observed_identity_consistency=not disagreeing_diagnostic_keys,
        selected_full_inchikey=selected_key,
        diagnostic_full_inchikeys=diagnostic_keys,
        disagreeing_full_inchikeys=disagreeing_diagnostic_keys,
    )
    selected_row = next(
        row for route in _ROUTE_ORDER for row in candidates
        if row.get("route") == route and _route_key(row) == selected_key
    )
    canonical, canonical_key = _v5._canonical_identity(selected_row["output_smiles"])
    if not canonical or canonical_key != selected_key:
        return _result(
            "rejected",
            "V6_SELECTED_IDENTITY_DRIFT",
            "canonicalization changed or removed the selected full InChIKey",
            support_status="supported",
            route_rows=route_rows,
            input_evidence=input_evidence,
            repair_codes=input_repair_codes,
        )

    fresh_path_evidence, path_replay = _fresh_path_evidence_replay(
        route_rows, path_evidence, pdb_path, chain_id
    )
    mapping_source, mapping_binding = _verified_path_mapping_source(
        route_rows,
        fresh_path_evidence,
        path_evidence,
        input_evidence,
        selected_key,
        canonical,
    )
    chem_comp_stereo_enabled = bool(input_evidence.get("chem_comp_evidence_enabled"))
    canonical_mol = (
        Chem.MolFromSmiles(canonical)
        if isinstance(canonical, str) and canonical
        else None
    )
    output_unassigned_centers = (
        [
            int(idx)
            for idx, assignment in Chem.FindMolChiralCenters(
                canonical_mol, includeUnassigned=True, useLegacyImplementation=False
            )
            if assignment == "?"
        ]
        if canonical_mol is not None
        else []
    )
    library, atom_mapping, template_stereo = _mapping_dimensions(
        mapping_source,
        input_evidence=input_evidence,
        chem_comp_evidence_enabled=chem_comp_stereo_enabled,
        output_molecule=canonical_mol,
        output_unassigned=output_unassigned_centers,
    )
    explicit_input = list(input_evidence.get("cyclization_bonds", []))
    if linear_topology:
        # For a linear input the v5 audit has already asserted that neither
        # explicit records nor the geometric scan produced any closure, so the
        # positive connectivity evidence is the complete absence of explicit
        # closure bonds without a coordinate-inferred connectivity repair.
        explicit_connectivity = _dimension(
            not explicit_input
            and "CONNECTIVITY_INFERRED_FROM_COORDINATES"
            not in input_evidence.get("repair_codes", []),
            closure_count=len(explicit_input),
            sources=sorted({
                str(bond.get("evidence_source", "")) for bond in explicit_input
            }),
            input_repair_codes=list(input_evidence.get("repair_codes", [])),
        )
    else:
        explicit_connectivity = _dimension(
            bool(explicit_input)
            and all(
                str(bond.get("evidence_source", "")).lower() in _EXPLICIT_SOURCES
                for bond in explicit_input
            )
            and "CONNECTIVITY_INFERRED_FROM_COORDINATES"
            not in input_evidence.get("repair_codes", []),
            closure_count=len(explicit_input),
            sources=sorted({
                str(bond.get("evidence_source", "")) for bond in explicit_input
            }),
            input_repair_codes=list(input_evidence.get("repair_codes", [])),
        )

    route_trace_results = []
    for row in candidates:
        route = str(row.get("route"))
        if route in {"a", "e"}:
            passed, details = _path_a_trace_passes(
                fresh_path_evidence.get(route, {}),
                selected_key,
                route,
                input_evidence,
                allow_linear=linear_topology,
            )
        elif route in {"b", "g"}:
            passed, details = _symbolic_trace_passes(
                row, selected_key, input_evidence
            )
        else:
            passed, details = False, {
                "route": route,
                "reason": "route_does_not_expose_closure_identity_trace",
            }
        details["passed"] = passed
        route_trace_results.append(details)
    required_path_traces = [
        row for row in route_trace_results if row.get("expected_route") in {"a", "e"}
    ]
    closure_trace = _dimension(
        any(row["passed"] for row in route_trace_results)
        and all(row["passed"] for row in required_path_traces),
        all_successful_path_traces_pass=all(
            row["passed"] for row in required_path_traces
        ),
        route_traces=route_trace_results,
    )

    try:
        graph_context = _audited_graph_context(
            canonical,
            input_evidence,
            mapping_source,
            minimum_macrocycle_ring_size,
        )
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            raise ValueError("selected output could not be reparsed")
        unassigned = [
            index for index, assignment in Chem.FindMolChiralCenters(
                molecule,
                includeUnassigned=True,
                useLegacyImplementation=False,
            )
            if assignment == "?"
        ]
        dummy_count = sum(
            atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()
        )
        graph_passed = (
            graph_context.get("sanitized") is True
            and (
                graph_context.get("heavy_atom_conserved") is True
                or (
                    isinstance(graph_context.get("template_completion_ledger"), dict)
                    and graph_context["template_completion_ledger"].get("ledger_closed")
                    is True
                )
            )
            and graph_context.get("connected_component_count") == 1
            and dummy_count == 0
        )
        chemical_graph = _dimension(
            graph_passed,
            **graph_context,
            dummy_atom_count=dummy_count,
            formal_charge=Chem.GetFormalCharge(molecule),
        )
        stereochemistry = _dimension(
            template_stereo["passed"] and not unassigned,
            template_stereo=template_stereo,
            coordinate_stereochemistry=template_stereo.get(
                "coordinate_stereochemistry", []
            ),
            output_unassigned_stereocenter_indices=unassigned,
        )
    except Exception as exc:
        chemical_graph = _dimension(False, error=f"{type(exc).__name__}: {exc}")
        stereochemistry = _dimension(False, error="chemical_graph_audit_failed")

    dimensions = {
        "library_chemistry": library,
        "atom_mapping": atom_mapping,
        "explicit_connectivity": explicit_connectivity,
        "closure_identity_trace": closure_trace,
        "chemical_graph_audit": chemical_graph,
        "stereochemistry": stereochemistry,
        "candidate_consistency": consistency,
        "independent_family_consensus": independent_family_consensus,
        "diagnostic_identity_consistency": diagnostic_identity_consistency,
        "mapping_evidence_binding": mapping_binding,
        "path_evidence_fresh_replay": path_replay,
        "local_monomer_evidence_consistency": (
            _local_monomer_evidence_consistency_dimension(
            route_rows, input_evidence, mapping_source, selected_key
            )
        ),
    }
    output_evidence = {
        "evidence_dimensions": dimensions,
        "selected_route": selected_row.get("route"),
        "selected_full_inchikey": selected_key,
        "mapping_source_route": mapping_binding.get("mapping_source_route"),
        "mapping_source_full_inchikey": (
            selected_key if mapping_binding["passed"] else None
        ),
        "mapping_source_canonical_smiles": (
            canonical if mapping_binding["passed"] else None
        ),
    }
    failed_dimensions = sorted(
        name for name, evidence in dimensions.items() if not evidence["passed"]
    )
    if failed_dimensions:
        return _result(
            "rejected",
            "V6_INSUFFICIENT_EVIDENCE_DIMENSIONS",
            f"required evidence dimensions failed: {failed_dimensions}",
            support_status="supported",
            route_rows=route_rows,
            input_evidence=input_evidence,
            output_evidence=output_evidence,
            repair_codes=input_repair_codes,
        )

    derived_sources = {
        source for source in library.get("unified_sources", [])
        if source == "local_structure_derived"
    }
    bootstrap = input_evidence.get("local_monomer_bootstrap")
    bootstrap_modes = sorted(set(
        str(value) for value in (
            bootstrap.get("resolution_modes", [])
            if isinstance(bootstrap, dict) else []
        )
    ))
    embedded_chem_comp_only = bootstrap_modes == ["embedded_mmcif_chem_comp"]
    acceptance_mode = "multi_family_consensus"
    if embedded_chem_comp_only:
        reconstruction_mode = "embedded_mmcif_chem_comp"
    elif derived_sources or (
        isinstance(bootstrap, dict) and bootstrap.get("derived_symbols")
    ):
        reconstruction_mode = "locally_inferred"
    elif isinstance(bootstrap, dict) and bootstrap.get("pdb_aliases"):
        reconstruction_mode = "unified_alias"
    else:
        reconstruction_mode = "curated_library"
    adjudication_class = (
        reconstruction_mode
        if reconstruction_mode == "locally_inferred"
        else acceptance_mode
    )
    output_evidence["adjudication_class"] = adjudication_class
    output_evidence["acceptance_mode"] = acceptance_mode
    output_evidence["reconstruction_mode"] = reconstruction_mode
    output_evidence["successful_chemical_route_count"] = len(candidates)
    repair_codes = sorted(set(input_evidence.get("repair_codes", [])))
    support_status = "qualified"
    if derived_sources and not embedded_chem_comp_only:
        repair_codes.append("LOCALLY_INFERRED_MONOMER")
        support_status = "repaired"
    if repair_codes:
        support_status = "repaired"
    repair_codes = sorted(set(repair_codes))
    selected_warnings = list(selected_row.get("warning_codes", []))
    if selected_warnings:
        return _result(
            "rejected",
            "V6_SELECTED_ROUTE_HAS_WARNINGS",
            f"selected route has unresolved warnings: {selected_warnings}",
            support_status="supported",
            route_rows=route_rows,
            input_evidence=input_evidence,
            output_evidence=output_evidence,
            repair_codes=input_repair_codes,
        )
    output_evidence["accepted_evidence_dimensions"] = sorted(dimensions)
    return _result(
        "success",
        None,
        None,
        support_status=support_status,
        route_rows=route_rows,
        input_evidence=input_evidence,
        output_evidence=output_evidence,
        output_smiles=canonical,
        output_inchikey=selected_key,
        path_used=f"V6_EVIDENCE_DIMENSIONS:{selected_row.get('route')}",
        repair_codes=repair_codes,
        evidence_dimensions=sorted(dimensions),
        warning_codes=repair_codes,
        qualified_success=not repair_codes,
    )


def reconstruct_pdb_fail_closed_v6(
    pdb_path: str | Path,
    chain_id: str = "L",
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    embedded_chem_comp_templates: dict[str, dict] | None = None,
    coordinate_input_evidence: dict[str, Any] | None = None,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> StrictReconstructionResult:
    """Reconstruct using evidence sufficiency, never implementation counts.

    ``allow_linear_topology`` opts into accepting a selected chain with no
    explicit or geometric cyclization evidence as ``topology_class="linear"``
    instead of rejecting it for lacking closure evidence.  It is keyword-only
    and defaults to ``False``; the cyclic contract is byte-identical when off.
    """
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
            resolved = reconstruct_pdb_fail_closed_v6(
                pdb_path,
                chain_id,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                embedded_chem_comp_templates=(
                    embedded_chem_comp_templates
                ),
                coordinate_input_evidence=coordinate_input_evidence,
                allow_linear_topology=allow_linear_topology,
                allow_chem_comp_evidence=allow_chem_comp_evidence,
                chem_comp_evidence=chem_comp_evidence,
            )
        if monomer_context is not None:
            resolved.input_evidence.setdefault(
                "monomer_resolution", dict(resolution_ledger)
            )
        return resolved
    try:
        from .paths._map_utils import persistent_overlay_audit

        persistent_audit = persistent_overlay_audit()
    except Exception as exc:
        return _result(
            "rejected",
            "V6_PERSISTENT_OVERLAY_AUDIT_FAILED",
            f"{type(exc).__name__}: {exc}",
            support_status="supported",
        )
    isolation_policy = {
        "require_empty_persistent_overlay": bool(
            require_empty_persistent_overlay
        ),
        "persistent_overlay_state_sha256": persistent_audit["state_sha256"],
        "activation_mode": "curated_registry_no_local_activation",
    }
    base_isolation_evidence = {
        "persistent_overlay_audit": persistent_audit,
        "entity_local_isolation": isolation_policy,
    }
    if minimum_macrocycle_ring_size < 3:
        return _result(
            "rejected",
            "V6_INVALID_MACROCYCLE_THRESHOLD",
            "minimum_macrocycle_ring_size must be at least 3",
            support_status="supported",
            input_evidence=base_isolation_evidence,
        )
    if (
        require_empty_persistent_overlay
        and persistent_audit["status"] != "empty"
    ):
        return _result(
            "rejected",
            "V6_PERSISTENT_OVERLAY_NOT_EMPTY",
            "formal entity-local reconstruction requires an empty persistent overlay",
            support_status="supported",
            input_evidence=base_isolation_evidence,
        )
    strict_chem_comp_evidence = (
        chem_comp_evidence if allow_chem_comp_evidence else None
    )
    validation = _v5.validate_pdb_reconstruction_input_v5(
        pdb_path,
        chain_id,
        allow_linear_topology=allow_linear_topology,
        chem_comp_evidence=strict_chem_comp_evidence,
    )
    if not validation.accepted:
        rejected_evidence = dict(validation.context)
        rejected_evidence.update(base_isolation_evidence)
        return _result(
            "rejected",
            validation.warning_codes[0] if validation.warning_codes else "V6_INPUT_AUDIT_FAILED",
            validation.reason or "input PDB audit failed",
            support_status="supported",
            input_evidence=rejected_evidence,
        )
    input_evidence = dict(validation.context)
    input_evidence.update(base_isolation_evidence)
    if allow_chem_comp_evidence:
        input_evidence["chem_comp_evidence_enabled"] = True
        input_evidence["chem_comp_evidence_audit"] = {
            key: value for key, value in (input_evidence.get("chem_comp_evidence_audit") or {}).items()
            if not str(key).startswith("_")
        }
    if coordinate_input_evidence is not None:
        input_evidence["coordinate_input"] = coordinate_input_evidence
    ambiguous_bond_orders = sorted({
        str(bond.get("bond_type", "unknown")).lower()
        for bond in input_evidence.get("cyclization_bonds", [])
        if str(bond.get("bond_type", "unknown")).lower() not in {
            "peptide", "disulfide", "isopeptide", "ester", "thioether",
            "staple_thioether",
        }
    })
    if ambiguous_bond_orders:
        return _result(
            "not_supported",
            "V6_EXPLICIT_CLOSURE_BOND_ORDER_AMBIGUOUS",
            "explicit closure type does not determine a unique bond order: "
            f"{ambiguous_bond_orders}",
            support_status="not_supported",
            input_evidence=input_evidence,
            repair_codes=sorted(set(input_evidence.get("repair_codes", []))),
        )
    if embedded_chem_comp_templates:
        from .core.local_monomer_inference import (
            audit_known_monomers_against_embedded_components,
        )

        coordinate = input_evidence.get("coordinate_input") or {}
        known_component_audit = audit_known_monomers_against_embedded_components(
            pdb_path,
            chain_id,
            embedded_chem_comp_templates,
            coordinate_source_sha256=coordinate.get(
                "embedded_chem_comp_source_payload_sha256"
            ),
        )
        input_evidence["known_residue_embedded_component_audit"] = (
            known_component_audit
        )
        if known_component_audit["status"] == "rejected":
            return _result(
                "rejected",
                "V6_LIBRARY_VS_EMBEDDED_COMPONENT_CONFLICT",
                "one or more known Unified residues conflict with source-bound "
                "embedded chemical-component evidence",
                support_status="supported",
                input_evidence=input_evidence,
                repair_codes=sorted(set(input_evidence.get("repair_codes", []))),
            )
        if known_component_audit["status"] == "not_supported":
            protonation_conflict = any(
                row["identity_relation"] == "protonation_or_charge_conflict"
                for row in known_component_audit["rows"]
            )
            return _result(
                "not_supported",
                (
                    "V6_LIBRARY_VS_EMBEDDED_PROTONATION_CONFLICT"
                    if protonation_conflict
                    else "V6_EMBEDDED_KNOWN_COMPONENT_NOT_RESOLVABLE"
                ),
                (
                    "known-residue protonation or charge differs between Unified "
                    "and source-bound embedded chemical-component evidence"
                    if protonation_conflict
                    else "source-bound embedded evidence for one or more known "
                    "residues cannot establish a unique full chemical identity"
                ),
                support_status="not_supported",
                input_evidence=input_evidence,
                repair_codes=sorted(set(input_evidence.get("repair_codes", []))),
            )
    support = _v5._family_support_context(list(input_evidence["residue_names"]))
    input_evidence["support_assessment"] = support
    if support["missing_residue_templates"]:
        from .core.local_monomer_inference import bootstrap_unknown_monomers
        from .paths._map_utils import isolated_monomer_registry

        bootstrap = bootstrap_unknown_monomers(
            pdb_path,
            chain_id,
            embedded_chem_comp_templates=embedded_chem_comp_templates,
            source_identity_audit=(
                input_evidence.get("coordinate_input", {}).get(
                    "source_sequence_identity_audit"
                )
                if isinstance(input_evidence.get("coordinate_input"), dict)
                else None
            ),
        )
        resolution_modes = sorted({
            str(result.evidence.get("resolution_mode") or "locally_inferred")
            for result in bootstrap.inference_results
        })
        input_evidence["local_monomer_bootstrap"] = {
            "status": bootstrap.status,
            "resolution_modes": resolution_modes,
            "derived_symbols": [
                str(row.get("symbol", "")) for row in bootstrap.derived_rows
            ],
            "derived_manifests": [
                dict(row) for row in bootstrap.manifest_entries
            ],
            "pdb_aliases": list(bootstrap.pdb_aliases),
            "quarantine_rows": list(bootstrap.quarantine_rows),
            "inference_results": [
                asdict(result) for result in bootstrap.inference_results
            ],
            "persistent_writes": 0,
        }
        if not bootstrap.ready:
            isolation_policy["activation_mode"] = "local_inference_not_activated"
            failure = _local_monomer_bootstrap_failure_details(bootstrap)
            return _result(
                failure["status"],
                failure["code"],
                failure["reason"],
                support_status=failure["support_status"],
                input_evidence=input_evidence,
                repair_codes=sorted(set(input_evidence.get("repair_codes", []))),
            )
        bootstrap_repair = None
        if resolution_modes == ["unified_library_match"]:
            bootstrap_repair = "UNIFIED_LIBRARY_STRUCTURE_ALIAS"
        elif resolution_modes != ["embedded_mmcif_chem_comp"]:
            bootstrap_repair = "LOCALLY_INFERRED_MONOMER_OR_ALIAS"
        if bootstrap_repair:
            input_evidence["repair_codes"] = sorted(set(
                input_evidence.get("repair_codes", []) + [bootstrap_repair]
            ))
        try:
            with isolated_monomer_registry(
                derived_rows=bootstrap.derived_rows,
                pdb_aliases=bootstrap.pdb_aliases,
                include_persistent_user=False,
                require_empty_persistent_derived=require_empty_persistent_overlay,
            ) as registry_audit:
                isolation_policy["activation_mode"] = "entity_local_registry"
                input_evidence["local_monomer_bootstrap"][
                    "registry_isolation"
                ] = registry_audit
                support = _v5._family_support_context(
                    list(input_evidence["residue_names"])
                )
                input_evidence["support_assessment"] = support
                if support["missing_residue_templates"]:
                    execution_result = _result(
                        "not_supported",
                        "V6_LOCAL_OVERLAY_DID_NOT_CLOSE_COVERAGE",
                        f"entity-local overlay still lacks templates: "
                        f"{support['missing_residue_templates']}",
                        support_status="not_supported",
                        input_evidence=input_evidence,
                        repair_codes=sorted(set(input_evidence.get("repair_codes", []))),
                    )
                else:
                    execution_result = _execute_v6(
                        pdb_path,
                        chain_id,
                        input_evidence,
                        minimum_macrocycle_ring_size,
                        validation.pdb_audit,
                    )
            execution_result.input_evidence["local_monomer_bootstrap"][
                "registry_isolation"
            ] = dict(registry_audit)
            return execution_result
        except ValueError as exc:
            return _result(
                "rejected",
                "V6_LOCAL_OVERLAY_ACTIVATION_REJECTED",
                str(exc),
                support_status="supported",
                input_evidence=input_evidence,
                repair_codes=sorted(set(input_evidence.get("repair_codes", []))),
            )
    return _execute_v6(
        pdb_path,
        chain_id,
        input_evidence,
        minimum_macrocycle_ring_size,
        validation.pdb_audit,
    )


def reconstruct_structure_fail_closed_v6(
    input_path: str | Path,
    chain_id: str = "L",
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> StrictReconstructionResult:
    """Reconstruct PDB, PDB.GZ, mmCIF, or mmCIF.GZ through one strict API.

    ``allow_linear_topology`` is forwarded to the prepared-structure audit so a
    closure-less selected chain can be accepted as linear topology.
    """
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
                input_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            resolved = reconstruct_structure_fail_closed_v6(
                input_path,
                chain_id,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                allow_linear_topology=allow_linear_topology,
                allow_chem_comp_evidence=allow_chem_comp_evidence,
                chem_comp_evidence=chem_comp_evidence,
            )
        if monomer_context is not None:
            resolved.input_evidence.setdefault(
                "monomer_resolution", dict(resolution_ledger)
            )
        return resolved
    from .core.structure_io import CoordinateInputError, prepare_coordinate_input

    try:
        with prepare_coordinate_input(input_path, chain_id) as prepared:
            return reconstruct_prepared_structure_fail_closed_v6(
                prepared,
                minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
                require_empty_persistent_overlay=require_empty_persistent_overlay,
                allow_linear_topology=allow_linear_topology,
                allow_chem_comp_evidence=allow_chem_comp_evidence,
                chem_comp_evidence=chem_comp_evidence,
            )
    except CoordinateInputError as exc:
        return coordinate_input_error_result_v6(
            input_path,
            chain_id,
            exc,
            require_empty_persistent_overlay=require_empty_persistent_overlay,
        )


def reconstruct_prepared_structure_fail_closed_v6(
    prepared: Any,
    *,
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    allow_linear_topology: bool = False,
    allow_chem_comp_evidence: bool = False,
    chem_comp_evidence: dict[str, Any] | None = None,
    monomer_context: Mapping[str, Any] | None = None,
) -> StrictReconstructionResult:
    """Run V6 on one audited ``PreparedCoordinateInput`` instance."""
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
                prepared.pdb_path, kind="coordinate"
            ),
        ) as resolution_ledger:
            resolved = reconstruct_prepared_structure_fail_closed_v6(
                prepared,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
                allow_linear_topology=allow_linear_topology,
                allow_chem_comp_evidence=allow_chem_comp_evidence,
                chem_comp_evidence=chem_comp_evidence,
            )
        if monomer_context is not None:
            resolved.input_evidence.setdefault(
                "monomer_resolution", dict(resolution_ledger)
            )
        return resolved
    result = reconstruct_pdb_fail_closed_v6(
        prepared.pdb_path,
        prepared.chain_id,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
        embedded_chem_comp_templates=prepared.audit.get(
            "embedded_chem_comp_templates"
        ),
        coordinate_input_evidence=prepared.audit,
        allow_linear_topology=allow_linear_topology,
        allow_chem_comp_evidence=allow_chem_comp_evidence,
        chem_comp_evidence=chem_comp_evidence,
    )
    result.input_evidence["coordinate_input"] = prepared.audit
    if prepared.source_format == "mmcif" and result.path_used:
        result.path_used = f"MMCIF_PROJECTED:{result.path_used}"
    return result


def coordinate_input_error_result_v6(
    input_path: str | Path,
    chain_id: str,
    error: Any,
    *,
    require_empty_persistent_overlay: bool = False,
) -> StrictReconstructionResult:
    """Map a coordinate preparation error to the public strict V6 contract."""
    from .paths._map_utils import persistent_overlay_audit

    try:
        persistent_audit = persistent_overlay_audit()
        isolation_evidence = {
            "persistent_overlay_audit": persistent_audit,
            "entity_local_isolation": {
                "require_empty_persistent_overlay": bool(
                    require_empty_persistent_overlay
                ),
                "persistent_overlay_state_sha256": persistent_audit[
                    "state_sha256"
                ],
                "activation_mode": "coordinate_input_rejected_before_activation",
            },
        }
    except Exception as audit_exc:
        isolation_evidence = {
            "persistent_overlay_audit_error": (
                f"{type(audit_exc).__name__}: {audit_exc}"
            )
        }
    return _result(
        "not_supported" if error.not_supported else "rejected",
        error.code,
        str(error),
        support_status="not_supported" if error.not_supported else "supported",
        input_evidence={
            **isolation_evidence,
            "coordinate_input": {
                "source_path": str(Path(input_path)),
                "source_chain_id": chain_id,
                "error_code": error.code,
            }
        },
    )


def _execute_v6(
    pdb_path: str | Path,
    chain_id: str,
    input_evidence: dict[str, Any],
    minimum_macrocycle_ring_size: int,
    pdb_audit=None,
) -> StrictReconstructionResult:
    """Execute one strict request with request-local identity memoization."""
    with identity_memo_context():
        return _execute_v6_impl(
            pdb_path,
            chain_id,
            input_evidence,
            minimum_macrocycle_ring_size,
            pdb_audit,
        )


def _execute_v6_impl(
    pdb_path: str | Path,
    chain_id: str,
    input_evidence: dict[str, Any],
    minimum_macrocycle_ring_size: int,
    pdb_audit=None,
) -> StrictReconstructionResult:
    """Execute candidate paths after registry coverage has been established."""
    try:
        execution_artifacts: dict[Any, Any] = {}
        previous = reconstruct_pdb_fail_closed(
            pdb_path,
            chain_id,
            _execution_artifacts=execution_artifacts,
            _pdb_audit=pdb_audit,
        )
        route_rows = [dict(row) for row in previous.route_results]
        _v5._attach_topology_construction_traces(
            route_rows,
            pdb_path,
            chain_id,
            input_evidence,
            execution_artifacts=execution_artifacts,
        )
        _v5._attach_explicit_only_monomer_evidence(
            route_rows, pdb_path, chain_id, input_evidence
        )
        path_evidence = _attach_path_evidence(
            route_rows,
            execution_artifacts,
            pdb_path,
            chain_id,
        )
        return _adjudicate_evidence_dimensions(
            route_rows,
            input_evidence,
            path_evidence,
            minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
            pdb_path=pdb_path,
            chain_id=chain_id,
        )
    except Exception as exc:
        return _result(
            "failed",
            "V6_INTERNAL_ERROR",
            f"{type(exc).__name__}: {exc}",
            support_status="unknown",
            input_evidence=input_evidence,
            repair_codes=sorted(set(input_evidence.get("repair_codes", []))),
        )


def remediation_version() -> str:
    return f"{__version__}+remediation.6"
