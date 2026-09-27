from __future__ import annotations

import json
from pathlib import Path

import pytest

from cycpep_master import application
from cycpep_master import prepare_ligand_from_sequence


def test_package_root_exports_sequence_facade():
    assert prepare_ligand_from_sequence is (
        application.prepare_ligand_from_sequence
    )


def _schema_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "v5_artifact.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_full_sequence_pipeline_emits_all_six_typed_artifacts(
    tmp_path,
):
    pytest.importorskip("meeko")
    validator = _schema_validator()

    result = application.prepare_ligand_from_sequence(
        "ACDEFG",
        tmp_path / "prepared",
        cyclization="head-to-tail",
        conformer_count=2,
        generate_pdbqt=True,
        torsdof_limit=100,
        template_strategy="off",
        random_seed=42,
        num_threads=1,
    )

    assert result["status"] == "success"
    data = result["data"]
    assert data["requested_artifact_status"] == "PDBQT_COMPLETE"
    assert len(data["mol2_artifacts"]) == 2
    assert len(data["flexibility_artifacts"]) == 2
    assert len(data["pdbqt_artifacts"]) == 2
    artifacts = [
        data["input_artifact"],
        data["chemical_graph"],
        data["mol2_ensemble"],
        *data["mol2_artifacts"],
        *data["flexibility_artifacts"],
        *data["pdbqt_artifacts"],
    ]
    for artifact in artifacts:
        validator.validate(artifact)
    observed_types = {
        artifact["artifact_type"] for artifact in artifacts
    }
    assert observed_types == {
        "InputArtifact",
        "ChemicalGraphArtifact",
        "ConformerEnsembleArtifact",
        "ValidatedMol2Artifact",
        "FlexibilityAssessmentArtifact",
        "PdbqtArtifact",
    }
    for mol2_artifact, flexibility, pdbqt in zip(
        data["mol2_artifacts"],
        data["flexibility_artifacts"],
        data["pdbqt_artifacts"],
    ):
        assert flexibility["parent_artifact_ids"][0] == (
            mol2_artifact["artifact_id"]
        )
        assert pdbqt["parent_artifact_ids"] == [
            mol2_artifact["artifact_id"],
            flexibility["artifact_id"],
        ]
        assert pdbqt["evidence"]["chemical_rigor"] == (
            mol2_artifact["evidence"]["chemical_rigor"]
        )
        assert Path(pdbqt["path"]).is_file()


def test_graph_only_abstention_never_fabricates_mol2_or_pdbqt(
    tmp_path,
):
    result = application.prepare_ligand_from_sequence(
        "[NOT_A_MONOMER]AC",
        tmp_path / "abstain",
        cyclization="linear",
    )

    assert result["status"] == "success"
    data = result["data"]
    assert data["chemical_graph"]["status"] == "PARTIAL"
    assert data["chemical_graph"]["evidence"]["chemical_rigor"] == "C1:H"
    assert data["requested_artifact_status"] == "GRAPH_PARTIAL"
    assert data["mol2_artifacts"] == []
    assert data["pdbqt_artifacts"] == []


def test_logical_artifact_ids_and_manifests_are_output_location_independent(
    tmp_path,
):
    results = []
    for name in ("first", "second"):
        result = application.prepare_ligand_from_sequence(
            "ACDEFG",
            tmp_path / name,
            cyclization="head-to-tail",
            conformer_count=1,
            generate_pdbqt=False,
            template_strategy="off",
            random_seed=42,
            num_threads=1,
        )
        assert result["status"] == "success"
        assert result["data"]["requested_artifact_status"] == (
            "MOL2_COMPLETE"
        )
        results.append(result["data"])

    first, second = results
    assert first["input_artifact"]["artifact_id"] == (
        second["input_artifact"]["artifact_id"]
    )
    assert first["chemical_graph"]["artifact_id"] == (
        second["chemical_graph"]["artifact_id"]
    )
    assert first["mol2_ensemble"]["artifact_id"] == (
        second["mol2_ensemble"]["artifact_id"]
    )
    assert first["mol2_artifacts"][0]["artifact_id"] == (
        second["mol2_artifacts"][0]["artifact_id"]
    )
    assert first["mol2_ensemble"]["manifest_sha256"] == (
        second["mol2_ensemble"]["manifest_sha256"]
    )
