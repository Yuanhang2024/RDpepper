"""Independent semantic validation for V6 candidate-assessment records."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from rdkit import Chem


_ROUTE_ORDER = tuple("abceg")
_INADMISSIBLE_REASONS = {
    "output_smiles_missing",
    "identity_resolution_failed",
    "declared_full_inchikey_missing",
    "declared_recomputed_identity_mismatch",
}
_COUNT_BY_REASON = {
    "nonchemical_route": "nonchemical_route_rows",
    "route_status_not_success": "unsuccessful_chemical_route_rows",
    "output_smiles_missing": "missing_output_smiles_rows",
    "identity_resolution_failed": "identity_resolution_failed_rows",
    "declared_full_inchikey_missing": "missing_declared_identity_rows",
    "declared_recomputed_identity_mismatch": (
        "declared_identity_mismatch_rows"
    ),
    "admitted": "admitted_candidate_rows",
}


class CandidateAssessmentSemanticError(ValueError):
    """Raised when a schema-valid candidate assessment is contradictory."""


def _fail(message: str) -> None:
    raise CandidateAssessmentSemanticError(message)


def _identity(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        _fail("candidate contains an unparsable canonical SMILES")
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    key = Chem.MolToInchiKey(molecule)
    blocks = key.split("-")
    if len(blocks) != 3:
        _fail("candidate Standard InChIKey does not have three blocks")
    composition: Counter[str] = Counter(
        atom.GetSymbol().upper()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    )
    return {
        "canonical_smiles": canonical,
        "full_inchikey": key,
        "inchi_connectivity_block": blocks[0],
        "inchi_second_block": blocks[1],
        "inchi_nonprotonation_key": "-".join(blocks[:2]),
        "inchi_protonation_flag": blocks[2],
        "heavy_atom_composition": dict(sorted(composition.items())),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
    }


def _sorted_routes(routes: set[str]) -> list[str]:
    return sorted(routes, key=lambda route: _ROUTE_ORDER.index(route))


def _expected_conflicts(candidates: list[dict[str, Any]]) -> list[str]:
    compositions = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for candidate in candidates
        for value in candidate["heavy_atom_composition_variants"]
    }
    connectivity = {
        candidate["inchi_connectivity_block"] for candidate in candidates
    }
    full_keys = {candidate["full_inchikey"] for candidate in candidates}
    by_connectivity: dict[str, list[dict[str, Any]]] = {}
    by_nonprotonation: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_connectivity.setdefault(
            candidate["inchi_connectivity_block"], []
        ).append(candidate)
        by_nonprotonation.setdefault(
            candidate["inchi_nonprotonation_key"], []
        ).append(candidate)

    conflicts: list[str] = []
    if len(compositions) > 1:
        conflicts.append("heavy_atom_composition")
    if len(connectivity) > 1:
        conflicts.append("connectivity")
    if any(
        len({row["inchi_second_block"] for row in rows}) > 1
        for rows in by_connectivity.values()
    ):
        conflicts.append("nonprotonation_identity_within_connectivity")
    if any(
        len({row["inchi_protonation_flag"] for row in rows}) > 1
        or len({
            charge
            for row in rows
            for charge in row["formal_charge_values"]
        }) > 1
        for rows in by_nonprotonation.values()
    ):
        conflicts.append(
            "protonation_or_formal_charge_within_nonprotonation_identity"
        )
    if len(full_keys) > 1:
        conflicts.append("standard_inchikey_equivalence")
    return conflicts


def validate_candidate_assessment_semantics(
    assessment: dict[str, Any],
) -> None:
    """Validate cross-field and recomputed-chemistry invariants.

    Run JSON Schema validation first. This validator deliberately remains
    independent from the V6 producer and never receives benchmark truth.
    """
    candidates = assessment["candidates"]
    audits = assessment["route_identity_audits"]
    counts = assessment["route_row_counts"]
    mode = assessment["mode"]

    expected_mode = (
        "no_candidate"
        if not candidates
        else "candidate_unique" if len(candidates) == 1
        else "candidate_ensemble"
    )
    if mode != expected_mode:
        _fail("mode does not match the candidate array")
    if assessment["candidate_count"] != len(candidates):
        _fail("candidate_count does not match the candidate array")
    if candidates != sorted(candidates, key=lambda row: row["full_inchikey"]):
        _fail("candidates are not deterministically sorted")

    if assessment["automatic_selection_permitted"] != assessment[
        "unattended_selection_qualified"
    ]:
        _fail("automatic-selection compatibility alias differs")
    if assessment["exploratory_only"] == assessment[
        "unattended_selection_qualified"
    ]:
        _fail("exploratory_only is not the inverse unattended state")
    if assessment["invalid_candidate_routes"] != assessment[
        "inadmissible_candidate_routes"
    ]:
        _fail("invalid-route compatibility alias differs")

    context = assessment["result_context"]
    success = context["status"] == "success"
    if assessment["strict_output_emitted"] is not success:
        _fail("strict_output_emitted does not match success status")
    if context["qualified_success"] != (
        success and not context["repair_codes"]
    ):
        _fail("qualified_success does not match success/repair semantics")
    selected = assessment["selected_output_audit"]
    bound = selected["reason"] == "bound"
    if assessment["selected_output_identity_bound"] is not bound:
        _fail("selected-output binding flag differs from its reason")

    reason_counts = Counter(row["reason"] for row in audits)
    if counts["total_route_rows"] != len(audits):
        _fail("total_route_rows does not match route audits")
    if counts["chemical_route_rows"] + counts["nonchemical_route_rows"] != len(
        audits
    ):
        _fail("chemical/nonchemical counts do not partition route audits")
    if counts["chemical_route_rows"] != (
        counts["successful_chemical_route_rows"]
        + counts["unsuccessful_chemical_route_rows"]
    ):
        _fail("successful/unsuccessful counts do not partition chemical rows")
    successful_partition = sum(
        counts[field]
        for field in (
            "missing_output_smiles_rows",
            "identity_resolution_failed_rows",
            "missing_declared_identity_rows",
            "declared_identity_mismatch_rows",
            "admitted_candidate_rows",
        )
    )
    if successful_partition != counts["successful_chemical_route_rows"]:
        _fail("successful chemical rows are not completely partitioned")
    for reason, field in _COUNT_BY_REASON.items():
        if counts[field] != reason_counts[reason]:
            _fail(f"{field} does not match route-audit reasons")
    if counts["distinct_candidate_equivalence_classes"] != len(candidates):
        _fail("distinct candidate count does not match candidates")
    if assessment["successful_chemical_route_row_count"] != counts[
        "successful_chemical_route_rows"
    ]:
        _fail("successful route scalar differs from route_row_counts")
    if assessment["admitted_candidate_route_row_count"] != counts[
        "admitted_candidate_rows"
    ]:
        _fail("admitted route scalar differs from route_row_counts")

    for audit in audits:
        if audit["admitted"] != (audit["reason"] == "admitted"):
            _fail("route audit admitted flag differs from reason")
        if audit["admitted"]:
            if audit["row_status"] != "success":
                _fail("admitted route audit row_status is not success")
            if (
                audit["declared_full_inchikey"] is None
                or audit["declared_full_inchikey"]
                != audit["recomputed_full_inchikey"]
            ):
                _fail(
                    "admitted route audit declared identity differs from "
                    "recomputed identity"
                )
            audit_identity = _identity(audit["canonical_smiles"])
            if (
                audit_identity is None
                or audit_identity["canonical_smiles"]
                != audit["canonical_smiles"]
                or audit_identity["full_inchikey"]
                != audit["recomputed_full_inchikey"]
            ):
                _fail(
                    "admitted route audit canonical SMILES does not "
                    "recompute to its identity"
                )
    inadmissible = sorted({
        row["route"] for row in audits if row["reason"] in _INADMISSIBLE_REASONS
    })
    if assessment["inadmissible_candidate_routes"] != inadmissible:
        _fail("inadmissible route set differs from route audits")
    identity_failed = sorted({
        row["route"]
        for row in audits
        if row["reason"] == "identity_resolution_failed"
    })
    if assessment["identity_resolution_failed_routes"] != identity_failed:
        _fail("identity-resolution failure set differs from route audits")

    admitted_by_key: dict[str, list[dict[str, Any]]] = {}
    for audit in audits:
        if audit["admitted"]:
            admitted_by_key.setdefault(
                audit["recomputed_full_inchikey"], []
            ).append(audit)

    candidate_keys = [candidate["full_inchikey"] for candidate in candidates]
    if len(candidate_keys) != len(set(candidate_keys)):
        _fail("candidate full_inchikey values are not unique")
    if set(candidate_keys) != set(admitted_by_key):
        _fail("candidate identity set does not match admitted route audits")

    for candidate in candidates:
        variants = candidate["canonical_smiles_variants"]
        identities = [_identity(smiles) for smiles in variants]
        if variants != sorted(set(variants)):
            _fail("canonical SMILES variants are not sorted and unique")
        if candidate["canonical_smiles"] != variants[0]:
            _fail("representative canonical SMILES is not the first variant")
        if any(
            identity["canonical_smiles"] != smiles
            or identity["full_inchikey"] != candidate["full_inchikey"]
            for identity, smiles in zip(identities, variants)
        ):
            _fail("candidate SMILES does not recompute to its identity class")
        reference = identities[0]
        for field in (
            "inchi_connectivity_block",
            "inchi_second_block",
            "inchi_nonprotonation_key",
            "inchi_protonation_flag",
        ):
            if candidate[field] != reference[field]:
                _fail(f"candidate {field} differs from recomputed identity")
        compositions = sorted(
            {json.dumps(row["heavy_atom_composition"], sort_keys=True): row[
                "heavy_atom_composition"
            ] for row in identities}.values(),
            key=lambda value: json.dumps(value, sort_keys=True),
        )
        charges = sorted({row["formal_charge"] for row in identities})
        if candidate["heavy_atom_composition_variants"] != compositions:
            _fail("candidate composition variants differ from SMILES")
        if candidate["heavy_atom_composition"] != compositions[0]:
            _fail("candidate representative composition differs")
        if candidate["formal_charge_values"] != charges:
            _fail("candidate charge variants differ from SMILES")
        expected_charge = charges[0] if len(charges) == 1 else None
        if candidate["formal_charge"] != expected_charge:
            _fail("candidate representative charge differs")

        supporting = admitted_by_key.get(candidate["full_inchikey"], [])
        routes = {row["route"] for row in supporting}
        hashes = {row["identity_input_sha256"] for row in supporting}
        if candidate["routes"] != _sorted_routes(routes):
            _fail("candidate routes differ from admitted route audits")
        if candidate["supporting_route_count"] != len(routes):
            _fail("supporting_route_count does not match distinct routes")
        if candidate["admitted_route_row_count"] != len(supporting):
            _fail("admitted_route_row_count does not match route audits")
        if candidate["route_identity_input_sha256"] != sorted(hashes):
            _fail("candidate route hashes differ from route audits")

    compositions = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for candidate in candidates
        for value in candidate["heavy_atom_composition_variants"]
    }
    connectivity = {
        candidate["inchi_connectivity_block"] for candidate in candidates
    }
    nonprotonation = {
        candidate["inchi_nonprotonation_key"] for candidate in candidates
    }
    full_keys = {candidate["full_inchikey"] for candidate in candidates}
    canonical = {
        value
        for candidate in candidates
        for value in candidate["canonical_smiles_variants"]
    }
    expected_counts = {
        "L1_heavy_atom_composition": len(compositions),
        "L2_inchi_connectivity_block": len(connectivity),
        "L3_inchi_nonprotonation_key": len(nonprotonation),
        "full_standard_inchikey": len(full_keys),
        "literal_canonical_smiles": len(canonical),
    }
    if assessment["identity_layer_counts"] != expected_counts:
        _fail("identity-layer counts differ from candidates")
    if assessment["standard_inchikey_equivalence_unique"] != (
        len(full_keys) == 1
    ):
        _fail("Standard InChIKey uniqueness flag differs from candidates")
    if assessment["literal_canonical_smiles_unique"] != (len(canonical) == 1):
        _fail("canonical-SMILES uniqueness flag differs from candidates")
    shared = None
    if len(compositions) == 1:
        shared = "L1"
    if len(connectivity) == 1:
        shared = "L2"
    if len(nonprotonation) == 1:
        shared = "L3"
    if assessment["highest_shared_candidate_level"] != shared:
        _fail("highest shared candidate level differs from candidates")
    if assessment["conflict_dimensions"] != _expected_conflicts(candidates):
        _fail("conflict dimensions differ from candidates")

    evidence_key = assessment["evidence_candidate_full_inchikey"]
    evidence_bound = (
        len(candidates) == 1 and evidence_key == candidates[0]["full_inchikey"]
    )
    if assessment["evidence_candidate_identity_bound"] != evidence_bound:
        _fail("evidence-candidate binding differs from candidate identity")
    if assessment["highest_evidence_qualified_level"] is not None and not (
        evidence_bound and shared is not None
    ):
        _fail("evidence qualification is not bound to a unique candidate")
    if bound:
        if len(candidates) != 1:
            _fail("selected output is bound without one candidate")
        if selected["declared_full_inchikey"] != candidates[0]["full_inchikey"]:
            _fail("selected declared identity differs from candidate")
        selected_identity = _identity(selected["canonical_smiles"])
        if selected_identity["full_inchikey"] != candidates[0]["full_inchikey"]:
            _fail("selected canonical SMILES differs from candidate")

    unattended = bool(
        success
        and bound
        and context["qualified_success"]
        and len(candidates) == 1
    )
    if assessment["unattended_selection_qualified"] != unattended:
        _fail("unattended-selection state differs from strict result context")
