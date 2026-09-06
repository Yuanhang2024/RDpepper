"""Tests for post-evaluation entity-deduplicated overlay publication."""
from __future__ import annotations

import hashlib
import json

import pytest

from cycpep_master.core.overlay_collector import (
    collect_v6_overlay_inputs,
    main,
    publish_from_sealed_results,
)


SMILES = "N[C@@H](CC#N)C(=O)O"


def _record(case_id, entity_id, *, input_hash, status="success", smiles=SMILES):
    inference = {
        "status": "unique",
        "pdb_resname": "ZZZ",
        "residue_key": ["ZZZ", 1, True],
        "candidate_smiles": smiles,
        "candidate_r3_mapped_smiles": None,
        "r3_port": None,
        "candidate_graph_count": 1,
        "reason_codes": [],
        "evidence": {"input_sha256": input_hash},
    }
    empty_state = {"aliases": [], "entries": []}
    bootstrap = {
        "status": "ready" if status == "success" else "quarantined",
        "persistent_writes": 0,
        "inference_results": [inference],
        "quarantine_rows": [],
    }
    if status == "success":
        bootstrap["registry_isolation"] = {
            "entity_local_overlay": True,
            "persistent_overlay_required_empty": True,
            "registry_restoration_verified": True,
            "persistent_overlay_at_start": {"status": "empty"},
        }
    state_sha256 = hashlib.sha256(json.dumps(
        empty_state, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "schema_version": "1.1.0",
        "case_id": case_id,
        "chemical_entity_id": entity_id,
        "tool": "cycpep_master_v6",
        "status": status,
        "support_status": "partially_supported",
        "chemical_audit_status": "passed" if status == "success" else "not_run",
        "provenance": {"fresh_process_per_job": True, "scores": {
            "input_evidence": {
                "persistent_overlay_audit": {
                    "status": "empty",
                    "disk_derived_row_count": 0,
                    "disk_pdb_alias_count": 0,
                    "disk_manifest_entry_count": 0,
                    "memory_derived_row_count": 0,
                    "memory_pdb_alias_count": 0,
                    "disk_memory_count_consistent": True,
                    "disk_memory_semantic_consistent": True,
                    "state": empty_state,
                    "state_sha256": state_sha256,
                },
                "entity_local_isolation": {
                    "require_empty_persistent_overlay": True,
                    "persistent_overlay_state_sha256": state_sha256,
                    "activation_mode": (
                        "entity_local_registry" if status == "success"
                        else "local_inference_not_activated"
                    ),
                },
                "local_monomer_bootstrap": bootstrap,
            },
        }},
    }


def _sealed_files(tmp_path, records):
    results = tmp_path / "raw_results.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({
        "status": "SEALED_RAW_RESULTS_PENDING_INDEPENDENT_SCORING",
        "raw_results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
        "result_count": len(records),
    }), encoding="utf-8")
    return results, manifest


def test_collector_deduplicates_paired_formats_by_entity():
    records = [
        _record("pdb", "entity-1", input_hash="a" * 64),
        _record("cif", "entity-1", input_hash="b" * 64),
    ]
    candidates, quarantine, summary = collect_v6_overlay_inputs(records)
    assert len(candidates) == 2
    assert {_record["source_entity_id"] for _record in candidates} == {"entity-1"}
    assert quarantine == []
    assert summary["unique_entity_count"] == 1
    assert summary["accepted_entity_count"] == 1


def test_collector_rejects_forged_empty_overlay_state_hash():
    record = _record("pdb", "entity-1", input_hash="a" * 64)
    record["provenance"]["scores"]["input_evidence"][
        "persistent_overlay_audit"
    ]["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="invalid empty-overlay state binding"):
        collect_v6_overlay_inputs([record])


def test_collector_rejects_entity_when_one_paired_record_is_unqualified():
    records = [
        _record("pdb", "entity-1", input_hash="a" * 64),
        _record(
            "cif", "entity-1", input_hash="b" * 64,
            status="not_supported",
        ),
    ]
    candidates, quarantine, summary = collect_v6_overlay_inputs(records)
    assert candidates == []
    assert quarantine[-1]["reason_codes"] == (
        "ENTITY_PAIRED_RECORD_NOT_UNIFORMLY_AUDITED"
    )
    assert summary["accepted_entity_count"] == 0


def test_collector_rejects_cross_format_graph_conflict():
    records = [
        _record("pdb", "entity-1", input_hash="a" * 64),
        _record(
            "cif", "entity-1", input_hash="b" * 64,
            smiles="N[C@@H](CCC#N)C(=O)O",
        ),
    ]
    candidates, quarantine, _summary = collect_v6_overlay_inputs(records)
    assert candidates == []
    assert quarantine[-1]["reason_codes"] == "PAIRED_RECORD_MONOMER_CONFLICT"


def test_collector_rejects_missing_bootstrap_on_failed_paired_record():
    failed = _record(
        "cif", "entity-1", input_hash="b" * 64, status="failed"
    )
    failed["provenance"]["scores"]["input_evidence"].pop(
        "local_monomer_bootstrap"
    )
    records = [
        _record("pdb", "entity-1", input_hash="a" * 64),
        failed,
    ]
    candidates, quarantine, _summary = collect_v6_overlay_inputs(records)
    assert candidates == []
    assert quarantine[-1]["reason_codes"] == (
        "ENTITY_PAIRED_RECORD_NOT_UNIFORMLY_AUDITED"
    )


def test_collector_rejects_different_complete_monomer_sets():
    second = _record("cif", "entity-1", input_hash="b" * 64)
    extra = dict(
        second["provenance"]["scores"]["input_evidence"]
        ["local_monomer_bootstrap"]["inference_results"][0]
    )
    extra["pdb_resname"] = "YYY"
    extra["candidate_smiles"] = "N[C@@H](CCC#N)C(=O)O"
    second["provenance"]["scores"]["input_evidence"][
        "local_monomer_bootstrap"
    ]["inference_results"].append(extra)
    candidates, quarantine, _summary = collect_v6_overlay_inputs([
        _record("pdb", "entity-1", input_hash="a" * 64), second,
    ])
    assert candidates == []
    assert quarantine[-1]["reason_codes"] == "PAIRED_RECORD_MONOMER_SET_CONFLICT"


def test_collector_quarantines_cross_entity_pdb_code_graph_conflict():
    records = [
        _record("one", "entity-1", input_hash="a" * 64),
        _record(
            "two", "entity-2", input_hash="b" * 64,
            smiles="N[C@@H](CCC#N)C(=O)O",
        ),
    ]
    candidates, quarantine, summary = collect_v6_overlay_inputs(records)
    assert candidates == []
    assert summary["accepted_entity_count"] == 0
    assert summary["cross_entity_pdb_resname_conflict_count"] == 1
    conflicts = [
        row for row in quarantine
        if row["reason_codes"] == "CROSS_ENTITY_PDB_RESNAME_GRAPH_CONFLICT"
    ]
    assert {row["record_id"] for row in conflicts} == {"entity-1", "entity-2"}


def test_sealed_publication_binds_inputs_and_command(tmp_path):
    records = [
        _record("pdb", "entity-1", input_hash="a" * 64),
        _record("cif", "entity-1", input_hash="b" * 64),
    ]
    results, run_manifest = _sealed_files(tmp_path, records)
    output = publish_from_sealed_results(
        results,
        run_manifest,
        tmp_path / "generation_001",
        command=["fixture-command"],
    )
    bundle = json.loads((output / "bundle_manifest.json").read_text())
    provenance = bundle["publication_provenance"]
    assert provenance["accepted_entity_count"] == 1
    assert provenance["candidate_observation_count"] == 2
    assert provenance["command"] == ["fixture-command"]
    assert provenance["raw_results_sha256"] == hashlib.sha256(
        results.read_bytes()
    ).hexdigest()


def test_sealed_publication_rejects_hash_drift(tmp_path):
    results, run_manifest = _sealed_files(
        tmp_path, [_record("pdb", "entity-1", input_hash="a" * 64)]
    )
    results.write_text(results.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        publish_from_sealed_results(
            results, run_manifest, tmp_path / "generation_001"
        )


def test_cli_requires_sealed_manifest_and_publishes(tmp_path):
    results, run_manifest = _sealed_files(
        tmp_path, [_record("pdb", "entity-1", input_hash="a" * 64)]
    )
    output = tmp_path / "generation_001"
    assert main([
        "--results", str(results),
        "--run-manifest", str(run_manifest),
        "--output", str(output),
    ]) == 0
    assert (output / "bundle_manifest.json").is_file()
