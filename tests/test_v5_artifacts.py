from __future__ import annotations

import pytest

from cycpep_master.core.artifacts import (
    BudgetState,
    ChemicalLevel,
    CoordinateLevel,
    CoordinateOrigin,
    EvidenceBasis,
    EvidenceProfile,
    FlexibilityLevel,
    FormatLevel,
    InputArtifact,
    ArtifactStatus,
    ArtifactType,
    ClaimBoundary,
    artifact_payload_sha256,
    inherit_evidence,
    make_artifact_id,
    validate_artifact_identity,
    validate_evidence_profile,
    validate_inherited_evidence,
)


def test_artifact_ids_are_deterministic_and_parent_bound():
    payload = {"value": 1}
    first = make_artifact_id("ChemicalGraphArtifact", ("a" * 64,), payload)
    second = make_artifact_id("ChemicalGraphArtifact", ("a" * 64,), payload)
    changed = make_artifact_id("ChemicalGraphArtifact", ("b" * 64,), payload)

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_downstream_inheritance_cannot_promote_chemical_evidence():
    parent = EvidenceProfile(
        chemical_level=ChemicalLevel.C2,
        chemical_basis=EvidenceBasis.HYPOTHESIS,
    )
    child = EvidenceProfile(
        chemical_level=ChemicalLevel.C3,
        chemical_basis=EvidenceBasis.QUALIFIED,
    )

    with pytest.raises(ValueError, match="chemical"):
        validate_inherited_evidence(parent, child)


def test_stage_owned_dimensions_can_advance_without_cross_promotion():
    parent = EvidenceProfile(
        chemical_level=ChemicalLevel.C3,
        chemical_basis=EvidenceBasis.SPECIFIED,
    )
    conformer = inherit_evidence(
        parent,
        coordinate_origin=CoordinateOrigin.GENERATED,
        coordinate_level=CoordinateLevel.X1,
    )
    validated = inherit_evidence(
        conformer,
        format_level=FormatLevel.Q2,
    )
    flexible = inherit_evidence(
        validated,
        flexibility_level=FlexibilityLevel.F1,
        budget_state=BudgetState.UNSATISFIED,
    )

    validate_inherited_evidence(
        parent, conformer, allow_coordinate=True
    )
    validate_inherited_evidence(
        conformer, validated, allow_format=True
    )
    validate_inherited_evidence(
        validated, flexible, allow_flexibility=True
    )
    assert flexible.chemical_rigor == "C3:S"


def test_nonhex_artifact_ids_are_rejected():
    payload = {"input_kind": "test"}
    artifact = InputArtifact(
        artifact_type=ArtifactType.INPUT,
        artifact_id="g" * 64,
        parent_artifact_ids=(),
        status=ArtifactStatus.MATERIALIZED,
        payload_sha256=artifact_payload_sha256(payload),
        evidence=EvidenceProfile(),
        claim_boundary=ClaimBoundary(),
        input_kind="test",
        source_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="artifact_id"):
        validate_artifact_identity(artifact, payload=payload)


def test_budget_cannot_be_satisfied_at_f0():
    with pytest.raises(ValueError, match="torsion budget"):
        validate_evidence_profile(EvidenceProfile(
            flexibility_level=FlexibilityLevel.F0,
            budget_state=BudgetState.SATISFIED,
        ))


def test_nonowner_stage_cannot_change_evidence_even_by_downgrade():
    parent = EvidenceProfile(
        chemical_level=ChemicalLevel.C3,
        chemical_basis=EvidenceBasis.SPECIFIED,
        coordinate_origin=CoordinateOrigin.GENERATED,
        coordinate_level=CoordinateLevel.X1,
        format_level=FormatLevel.Q2,
    )
    child = inherit_evidence(
        parent,
        coordinate_origin=CoordinateOrigin.NONE,
        coordinate_level=CoordinateLevel.X0,
    )

    with pytest.raises(ValueError, match="coordinate"):
        validate_inherited_evidence(parent, child)
