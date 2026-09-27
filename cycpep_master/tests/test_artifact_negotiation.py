from __future__ import annotations

import json
from pathlib import Path

from cycpep_master import application


def _candidate_payload() -> dict:
    return {
        "status": "success",
        "quality": "candidate",
        "smiles": None,
        "graph": None,
        "candidate_smiles": "CC",
        "candidate_graph": None,
        "candidate_rigor": "L2:H",
        "ambiguous": False,
        "warning_codes": ["INFERRED_CHEMISTRY_UNQUALIFIED"],
        "alternatives": [],
        "provenance": {},
        "strict_status": "rejected",
    }


def _raw_payload() -> dict:
    return {
        "status": "success",
        "quality": "raw",
        "smiles": None,
        "graph": {
            "atoms": [
                {
                    "serial": 1,
                    "element": "C",
                    "xyz": [0.0, 0.0, 0.0],
                }
            ],
            "bonds": [],
        },
        "candidate_smiles": None,
        "candidate_graph": None,
        "candidate_rigor": None,
        "ambiguous": False,
        "warning_codes": ["NO_BONDS_AVAILABLE"],
        "alternatives": [],
        "provenance": {},
        "strict_status": "rejected",
    }


def _fake_mol2(smiles, output_path=None, **_kwargs):
    path = Path(output_path)
    path.write_text(f"MOCK MOL2 {smiles}\n", encoding="ascii")
    return str(path), None


def _fake_receipt(mol2_path, **_kwargs):
    path = Path(str(mol2_path) + ".validation.json")
    path.write_text("{}\n", encoding="ascii")
    return path


def test_strict_export_keeps_candidate_handoff_disabled(tmp_path):
    result = application.export_structure(
        _candidate_payload(), tmp_path / "strict.mol2"
    )

    # Strict export does not promote candidate chemistry into a validated
    # MOL2; the request degrades to a typed diagnostic artifact instead of
    # being rejected.
    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] in {
        "metadata_only", "degraded_format",
    }
    assert not (tmp_path / "strict.mol2").exists()


def test_best_export_regenerates_candidate_mol2(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cycpep_master.export.smiles_to_mol2", _fake_mol2
    )
    monkeypatch.setattr(
        application, "_write_mol2_validation_receipt", _fake_receipt
    )

    result = application.export_best_available(
        _candidate_payload(), tmp_path / "candidate.mol2"
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "fulfilled"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "mol2"
    assert artifact["rigor"] == "L2:H"
    assert artifact["coordinate_mode"] == "regenerated"
    assert artifact["role"] == "candidate"
    assert Path(artifact["path"]).is_file()


def test_best_export_returns_graph_artifact_for_raw_result(tmp_path):
    output = tmp_path / "raw.mol2"
    result = application.export_best_available(_raw_payload(), output)

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "degraded_format"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "graph_json"
    assert artifact["rigor"] == "L0:C"
    assert not output.exists()
    graph_path = Path(artifact["path"])
    assert graph_path.name == "raw.mol2.graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["graph"]["atoms"][0]["element"] == "C"


def test_best_export_returns_metadata_when_no_structure_exists(tmp_path):
    result = application.export_best_available(
        {
            "status": "failed",
            "quality": None,
            "smiles": None,
            "graph": None,
            "warning_codes": ["NO_READABLE_STRUCTURE"],
        },
        tmp_path / "missing.sdf",
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "metadata_only"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "metadata"
    assert artifact["rigor"] == "L0:NONE"
    assert Path(artifact["path"]).is_file()


def test_coordinate_best_export_never_regenerates_candidate_coordinates(
    tmp_path, monkeypatch
):
    coordinate = tmp_path / "input.pdb"
    coordinate.write_text("END\n", encoding="ascii")
    monkeypatch.setattr(
        application,
        "_reconstruct_and_materialize_result_first_mol2",
        lambda *_args, **_kwargs: (
            _candidate_payload(), None, "source mapping unavailable"
        ),
    )

    def unexpected_regeneration(*_args, **_kwargs):
        raise AssertionError("coordinate MOL2 export must preserve PDB coordinates")

    monkeypatch.setattr(
        "cycpep_master.export.smiles_to_mol2", unexpected_regeneration
    )

    result = application.export_best_available(
        coordinate,
        tmp_path / "fallback.mol2",
        source_kind="coordinate",
        chain_id="L",
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "metadata_only"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "metadata"
    assert artifact["coordinate_mode"] == "none"
    assert "source mapping unavailable" in artifact["warnings"]
    assert not (tmp_path / "fallback.mol2").exists()


def test_best_pdbqt_does_not_bypass_missing_validated_mol2(
    tmp_path, monkeypatch
):
    coordinate = tmp_path / "input.pdb"
    coordinate.write_text("END\n", encoding="ascii")
    metadata = tmp_path / "candidate.pdbqt.metadata.json"
    metadata.write_text("{}\n", encoding="ascii")

    def fake_prepare(*_args, **_kwargs):
        return {
            "operation": "prepare_ligand_pdbqt_from_pdb",
            "status": "not_supported",
            "data": {
                "artifacts": [{
                    "format": "metadata",
                    "path": str(metadata),
                    "rigor": "L0:NONE",
                    "coordinate_mode": "none",
                }]
            },
            "error": "validated parent MOL2 unavailable",
        }

    monkeypatch.setattr(
        application, "prepare_ligand_pdbqt_from_pdb", fake_prepare
    )
    result = application.prepare_ligand_pdbqt_best_available(
        coordinate, tmp_path / "candidate.pdbqt"
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "metadata_only"
    artifact = result["data"]["artifacts"][0]
    assert artifact["format"] == "metadata"
    assert not (tmp_path / "candidate.pdbqt").exists()


def test_best_pdbqt_returns_graph_for_raw_result(tmp_path, monkeypatch):
    coordinate = tmp_path / "input.pdb"
    coordinate.write_text("END\n", encoding="ascii")
    graph = tmp_path / "raw.pdbqt.graph.json"
    graph.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        application,
        "prepare_ligand_pdbqt_from_pdb",
        lambda *_args, **_kwargs: {
            "operation": "prepare_ligand_pdbqt_from_pdb",
            "status": "not_supported",
            "data": {
                "artifacts": [{
                    "format": "graph_json",
                    "path": str(graph),
                    "rigor": "L0:C",
                    "coordinate_mode": "raw",
                }]
            },
            "error": "validated parent MOL2 unavailable",
        },
    )

    result = application.prepare_ligand_pdbqt_best_available(
        coordinate, tmp_path / "raw.pdbqt"
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "degraded_format"
    assert result["data"]["artifacts"][0]["format"] == "graph_json"
    assert not (tmp_path / "raw.pdbqt").exists()


def test_coordinate_best_export_reuses_selected_result(tmp_path, monkeypatch):
    coordinate = tmp_path / "input.pdb"
    coordinate.write_text("END\n", encoding="ascii")
    selected = _candidate_payload()
    calls = {"reconstruct": 0, "materialize": 0}

    def combined(_source, destination, **kwargs):
        calls["reconstruct"] += 1
        calls["materialize"] += 1
        assert kwargs["minimum_macrocycle_ring_size"] == 11
        assert kwargs["require_empty_persistent_overlay"] is True
        output = Path(destination)
        output.write_text(
            "@<TRIPOS>MOLECULE\nmock\n0 0 0 0 0\n"
            "SMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n",
            encoding="ascii",
        )
        return selected, str(output), None

    monkeypatch.setattr(
        application,
        "_reconstruct_and_materialize_result_first_mol2",
        combined,
    )
    monkeypatch.setattr(
        application, "_write_mol2_validation_receipt", _fake_receipt
    )

    result = application.export_best_available(
        coordinate,
        tmp_path / "candidate.mol2",
        source_kind="coordinate",
        minimum_macrocycle_ring_size=11,
        require_empty_persistent_overlay=True,
    )

    assert result["status"] == "success"
    assert calls == {"reconstruct": 1, "materialize": 1}


def test_best_export_preserves_child_degraded_success(tmp_path, monkeypatch):
    metadata = tmp_path / "child.metadata.json"
    metadata.write_text("{}\n", encoding="ascii")
    child = {
        "operation": "export",
        "status": "success",
        "data": {
            "requested_format": "mol2",
            "requested_format_status": "metadata_only",
            "artifacts": [{
                "format": "metadata",
                "path": str(metadata),
            }],
        },
        "error": None,
    }
    monkeypatch.setattr(application, "export_structure", lambda *_a, **_k: child)

    result = application.export_best_available(
        "CC", tmp_path / "requested.mol2", source_kind="smiles"
    )

    assert result["status"] == "success"
    assert result["operation"] == "export_best_available"
    assert result["data"]["artifacts"] == child["data"]["artifacts"]
    assert not (tmp_path / "requested.mol2.metadata.json").exists()


def test_best_pdbqt_normalizes_successful_degradation(tmp_path, monkeypatch):
    metadata = tmp_path / "parent.metadata.json"
    metadata.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        application,
        "prepare_ligand_pdbqt_from_pdb",
        lambda *_args, **_kwargs: {
            "operation": "prepare_ligand_pdbqt_from_pdb",
            "status": "success",
            "data": {
                "artifacts": [{
                    "format": "metadata",
                    "path": str(metadata),
                }],
            },
            "error": None,
        },
    )

    result = application.prepare_ligand_pdbqt_best_available(
        tmp_path / "input.pdb", tmp_path / "output.pdbqt"
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format"] == "pdbqt"
    assert result["data"]["requested_format_status"] == "metadata_only"
