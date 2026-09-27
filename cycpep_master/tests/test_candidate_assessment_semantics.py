from __future__ import annotations

from copy import deepcopy

import pytest
from rdkit import Chem

from cycpep_master.candidate_assessment import (
    CandidateAssessmentSemanticError,
    validate_candidate_assessment_semantics,
)
from cycpep_master.remediation_v6 import _candidate_assessment


def _row(route: str, smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return {
        "route": route,
        "status": "success",
        "output_smiles": smiles,
        "output_inchikey": Chem.MolToInchiKey(molecule),
    }


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [_row("a", "N[C@@H](C)C(=O)O")],
        [
            _row("a", "N[C@@H](C)C(=O)O"),
            _row("b", "N[C@H](C)C(=O)O"),
        ],
        [
            _row("a", "O=c1cccc[nH]1"),
            _row("b", "Oc1ccccn1"),
        ],
    ],
)
def test_generated_candidate_assessments_are_semantically_valid(rows):
    assessment = _candidate_assessment(
        rows, {}, status="rejected", qualified_success=False
    )
    validate_candidate_assessment_semantics(assessment)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(candidate_count=2),
            "candidate_count",
        ),
        (
            lambda value: value["route_row_counts"].update(
                admitted_candidate_rows=0
            ),
            "successful chemical rows",
        ),
        (
            lambda value: value["candidates"][0].update(
                inchi_connectivity_block="A" * 14
            ),
            "inchi_connectivity_block",
        ),
        (
            lambda value: value.update(
                invalid_candidate_routes=["b"]
            ),
            "compatibility alias",
        ),
        (
            lambda value: value["route_identity_audits"][0].update(
                admitted=False
            ),
            "admitted flag",
        ),
    ],
)
def test_semantic_validator_rejects_cross_field_contradictions(mutate, message):
    original = _candidate_assessment(
        [_row("a", "N[C@@H](C)C(=O)O")],
        {},
        status="rejected",
        qualified_success=False,
    )
    assessment = deepcopy(original)
    mutate(assessment)

    with pytest.raises(CandidateAssessmentSemanticError, match=message):
        validate_candidate_assessment_semantics(assessment)


def test_semantic_validator_rejects_duplicate_candidate_identity():
    assessment = _candidate_assessment(
        [_row("a", "N[C@@H](C)C(=O)O")],
        {},
        status="rejected",
        qualified_success=False,
    )
    assessment["candidates"].append(deepcopy(assessment["candidates"][0]))
    assessment["candidate_count"] = 2
    assessment["mode"] = "candidate_ensemble"
    assessment["route_row_counts"][
        "distinct_candidate_equivalence_classes"
    ] = 2

    with pytest.raises(
        CandidateAssessmentSemanticError,
        match="full_inchikey values are not unique",
    ):
        validate_candidate_assessment_semantics(assessment)


def test_semantic_validator_rejects_missing_admitted_identity():
    assessment = _candidate_assessment(
        [
            _row("a", "N[C@@H](C)C(=O)O"),
            _row("b", "NCC(=O)O"),
        ],
        {},
        status="rejected",
        qualified_success=False,
    )
    assessment["candidates"] = assessment["candidates"][:1]
    assessment["candidate_count"] = 1
    assessment["mode"] = "candidate_unique"
    assessment["route_row_counts"][
        "distinct_candidate_equivalence_classes"
    ] = 1

    with pytest.raises(
        CandidateAssessmentSemanticError,
        match="candidate identity set does not match admitted route audits",
    ):
        validate_candidate_assessment_semantics(assessment)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("row_status", "failed", "row_status is not success"),
        (
            "declared_full_inchikey",
            None,
            "declared identity differs from recomputed identity",
        ),
        (
            "canonical_smiles",
            "CC(=O)O",
            "canonical SMILES does not recompute to its identity",
        ),
    ],
)
def test_semantic_validator_rejects_invalid_admitted_audit(
    field, value, message
):
    assessment = _candidate_assessment(
        [_row("a", "N[C@@H](C)C(=O)O")],
        {},
        status="rejected",
        qualified_success=False,
    )
    assessment["route_identity_audits"][0][field] = value

    with pytest.raises(CandidateAssessmentSemanticError, match=message):
        validate_candidate_assessment_semantics(assessment)
