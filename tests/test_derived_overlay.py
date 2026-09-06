"""Isolation and schema tests for the append-only derived monomer layer."""
from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest
from rdkit import Chem

from cycpep_master.core.derived_monomers import (
    build_derived_batch,
    build_derived_row,
    publish_overlay_bundle,
    unified_schema,
    write_overlay_artifacts,
)
from cycpep_master.core import derived_monomers
from cycpep_master.paths import _map_utils as mu
from cycpep_master.paths import path_b, path_g
from cycpep_master.paths.residue_template_factory import get_residue_template


TEST_SMILES = "N[C@@H](CC#N)C(=O)O"


def _row():
    return build_derived_row("ZZZ", TEST_SMILES, monomer_id=20000)[0]


def test_tracked_overlay_has_exact_empty_235_column_schema():
    with open(mu._DERIVED_CSV, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        assert tuple(next(reader)) == unified_schema()
        assert len(unified_schema()) == 235
        assert list(reader) == []


def test_builder_is_deterministic_and_leaves_unknown_metadata_blank():
    first, first_manifest = build_derived_row("z z z", TEST_SMILES, monomer_id=20000)
    second, second_manifest = build_derived_row("z z z", TEST_SMILES, monomer_id=20000)
    assert first == second
    assert first_manifest == second_manifest
    assert tuple(first) == unified_schema()
    assert first["symbol"].startswith("LCL_Z_Z_Z_")
    assert first["source"] == "local_structure_derived"
    assert first["iupac_name"] == ""
    assert first["pubchem_cid"] == ""
    assert first["natural_analog"] == ""
    assert all(first[field] != "" for field in unified_schema()[20:-7])


def test_entity_local_overlay_is_visible_to_a_and_notation_then_restored():
    row = _row()
    symbol = row["symbol"]
    assert symbol not in mu._unified_by_symbol
    snapshots = {
        "unified": copy.deepcopy(mu._unified_by_symbol),
        "smiles": copy.deepcopy(mu.monomers2smi_dict),
        "map": copy.deepcopy(mu.map_to_helm_dict),
        "reverse": copy.deepcopy(mu._symbol_to_map),
    }
    with mu.isolated_monomer_registry(derived_rows=[row]) as ledger:
        assert ledger["derived_symbols"] == [symbol]
        assert ledger["pdb_aliases"] == {}
        assert ledger["persistent_user_enabled"] is False
        assert ledger["disk_writes"] == 0
        assert ledger["entity_local_overlay"] is True
        assert ledger["persistent_overlay_at_start"]["status"] == "empty"
        assert ledger["registry_restoration_verified"] is False
        assert get_residue_template(symbol).source == "local_structure_derived"
        assembled = mu.get_smi_from_map(f"G{{nnr:{symbol}}}A")
        assert Chem.MolFromSmiles(assembled) is not None
        assert symbol in mu._symbol_to_map
    assert mu._unified_by_symbol == snapshots["unified"]
    assert mu.monomers2smi_dict == snapshots["smiles"]
    assert mu.map_to_helm_dict == snapshots["map"]
    assert mu._symbol_to_map == snapshots["reverse"]
    assert symbol not in mu._unified_by_symbol
    assert ledger["registry_restoration_verified"] is True


def test_entity_local_overlay_rejects_nonempty_persistent_start(monkeypatch):
    monkeypatch.setattr(mu, "_persistent_derived_by_symbol", {"LEAK": _row()})
    with pytest.raises(ValueError, match="requires an empty persistent"):
        with mu.isolated_monomer_registry(
            derived_rows=[_row()], require_empty_persistent_derived=True
        ):
            pass


def test_nonformal_registry_mode_allows_published_persistent_overlay(monkeypatch):
    monkeypatch.setattr(mu, "_persistent_derived_by_symbol", {"LEAK": _row()})
    with mu.isolated_monomer_registry() as ledger:
        assert ledger["persistent_overlay_required_empty"] is False
        assert ledger["persistent_overlay_at_start"]["status"] == "nonempty"


def test_persistent_audit_detects_manifest_entry_without_loaded_row(
    tmp_path, monkeypatch
):
    row, entry = build_derived_row("ZZZ", TEST_SMILES, monomer_id=20000)
    derived = tmp_path / "derived.csv"
    manifest = tmp_path / "manifest.json"
    quarantine = tmp_path / "quarantine.csv"
    write_overlay_artifacts(
        [row], [entry], [],
        derived_path=derived, manifest_path=manifest, quarantine_path=quarantine,
    )
    with derived.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(unified_schema())
    monkeypatch.setattr(mu, "_DERIVED_CSV", str(derived))
    monkeypatch.setattr(mu, "_DERIVED_MANIFEST", str(manifest))
    audit = mu.persistent_overlay_audit()
    assert audit["status"] == "nonempty"
    assert audit["disk_manifest_entry_count"] == 1
    assert audit["disk_derived_row_count"] == 0
    assert audit["disk_memory_count_consistent"] is False
    assert audit["disk_memory_semantic_consistent"] is False


def test_entity_local_pdb_alias_reaches_a_b_and_g_then_restores():
    row = _row()
    symbol = row["symbol"]
    assert mu.resolve_pdb_alias("ZZZ") is None
    with mu.isolated_monomer_registry(
        derived_rows=[row],
        pdb_aliases=[{"pdb_resname": "ZZZ", "target_symbol": symbol}],
    ) as ledger:
        assert ledger["pdb_aliases"] == {"ZZZ": symbol}
        assert get_residue_template("ZZZ").symbol == symbol
        assert path_b._pdb_name_to_helm_symbol("ZZZ") == symbol
        assert path_g._residue_symbol("ZZZ") == symbol
    assert mu.resolve_pdb_alias("ZZZ") is None


def test_entity_local_pdb_alias_rejects_unknown_target():
    with pytest.raises(ValueError, match="targets unknown symbol"):
        with mu.isolated_monomer_registry(
            pdb_aliases=[{"pdb_resname": "ZZZ", "target_symbol": "MISSING"}]
        ):
            pass


def test_entity_local_pdb_alias_cannot_override_builtin_code():
    row = _row()
    with pytest.raises(ValueError, match="conflicts with built-in symbol"):
        with mu.isolated_monomer_registry(
            derived_rows=[row],
            pdb_aliases=[{
                "pdb_resname": "ALA", "target_symbol": row["symbol"],
            }],
        ):
            pass


def test_overlay_restores_all_registries_after_exception():
    row = _row()
    before = (
        copy.deepcopy(mu._unified_by_symbol),
        copy.deepcopy(mu.monomers2smi_dict),
        copy.deepcopy(mu.monomers2r_groups_dict),
        copy.deepcopy(mu.map_to_helm_dict),
        copy.deepcopy(mu._symbol_to_map),
    )
    with pytest.raises(RuntimeError, match="fixture"):
        with mu.isolated_monomer_registry(derived_rows=[row]):
            raise RuntimeError("fixture")
    after = (
        mu._unified_by_symbol,
        mu.monomers2smi_dict,
        mu.monomers2r_groups_dict,
        mu.map_to_helm_dict,
        mu._symbol_to_map,
    )
    assert after == before


def test_path_g_special_registration_survives_registry_rebuild():
    path_g._register_special_residues()
    code = next(
        code for code in path_g._sr.all_codes()
        if path_g._sr.get_symbol(code) in mu.monomers2smi_dict
    )
    symbol = path_g._sr.get_symbol(code)
    with mu.isolated_monomer_registry(include_persistent_user=False):
        mu.monomers2smi_dict.pop(symbol, None)
        mu.monomers2r_groups_dict.pop(symbol, None)
        path_g._register_special_residues()
        assert symbol in mu.monomers2smi_dict


def test_overlay_rejects_casefold_symbol_and_unified_id_collisions():
    row = _row()
    casefold_collision = dict(row)
    casefold_collision["symbol"] = "a"
    with pytest.raises(ValueError, match="case-insensitive"):
        with mu.isolated_monomer_registry(derived_rows=[casefold_collision]):
            pass

    id_collision = dict(row)
    with open(mu._UNIFIED_CSV, encoding="utf-8-sig", newline="") as handle:
        id_collision["monomer_id"] = next(csv.DictReader(handle))["monomer_id"]
    with pytest.raises(ValueError, match="conflicts with Unified"):
        with mu.isolated_monomer_registry(derived_rows=[id_collision]):
            pass


def test_overlay_rejects_symbol_override_and_duplicate_graph():
    row = _row()
    override = dict(row)
    override["symbol"] = "A"
    with pytest.raises(ValueError, match="override"):
        with mu.isolated_monomer_registry(derived_rows=[override]):
            pass

    duplicate = dict(row)
    duplicate["monomer_id"] = 20001
    duplicate["symbol"] = row["symbol"] + "_DUP"
    with pytest.raises(ValueError, match="duplicates derived graph"):
        with mu.isolated_monomer_registry(derived_rows=[row, duplicate]):
            pass


def test_batch_ids_are_graph_hash_sorted_and_duplicate_candidates_become_aliases():
    rows, manifests, aliases = build_derived_batch([
        {"pdb_resname": "SIL", "smiles": TEST_SMILES, "input_sha256": "a" * 64},
        {"pdb_resname": "BOR", "smiles": "B(O)(O)C[C@H](N)C(=O)O", "input_sha256": "b" * 64},
    ])
    assert aliases == []
    assert [entry["graph_sha256"] for entry in manifests] == sorted(
        entry["graph_sha256"] for entry in manifests
    )
    assert [int(row["monomer_id"]) for row in rows] == list(
        range(int(rows[0]["monomer_id"]), int(rows[0]["monomer_id"]) + 2)
    )
    duplicate_rows, duplicate_manifests, duplicate_aliases = build_derived_batch([
        {"pdb_resname": "ONE", "smiles": TEST_SMILES},
        {"pdb_resname": "TWO", "smiles": TEST_SMILES},
    ])
    assert len(duplicate_rows) == len(duplicate_manifests) == 1
    assert duplicate_aliases == [{
        "pdb_resname": "TWO",
        "target_symbol": duplicate_rows[0]["symbol"],
        "graph_sha256": duplicate_manifests[0]["graph_sha256"],
        "full_inchikey": duplicate_manifests[0]["full_inchikey"],
        "input_sha256": "",
        "source_entity_id": "",
    }]


def test_unified_duplicate_becomes_alias_without_a_derived_row():
    rows, manifests, aliases = build_derived_batch([
        {"pdb_resname": "ALA_ALIAS", "smiles": "C[C@H](N)C(=O)O"},
    ])
    assert rows == []
    assert manifests == []
    assert aliases[0]["pdb_resname"] == "ALA_ALIAS"
    assert aliases[0]["target_symbol"] == "A"


def test_artifact_writer_never_modifies_unified_and_keeps_quarantine_separate(tmp_path):
    unified_hash = __import__("hashlib").sha256(
        open(mu._UNIFIED_CSV, "rb").read()
    ).hexdigest()
    row, manifest = build_derived_row("ZZZ", TEST_SMILES, monomer_id=20000)
    write_overlay_artifacts(
        [row],
        [manifest],
        [{
            "record_id": "Q1", "pdb_resname": "BAD", "status": "not_supported",
            "reason_codes": "AMBIGUOUS_GRAPH", "input_sha256": "c" * 64,
            "candidate_graph_count": 2, "details_json": "{}",
        }],
        derived_path=tmp_path / "derived.csv",
        manifest_path=tmp_path / "manifest.json",
        quarantine_path=tmp_path / "quarantine.csv",
    )
    assert __import__("hashlib").sha256(
        open(mu._UNIFIED_CSV, "rb").read()
    ).hexdigest() == unified_hash
    assert len(list(csv.DictReader((tmp_path / "derived.csv").open()))) == 1
    assert len(list(csv.DictReader((tmp_path / "quarantine.csv").open()))) == 1


def test_loader_recomputes_manifest_entry_and_alias_identities(tmp_path, monkeypatch):
    row, entry = build_derived_row("ZZZ", TEST_SMILES, monomer_id=20000)
    _rows, _entries, aliases = build_derived_batch([{
        "pdb_resname": "ALA_ALIAS",
        "smiles": "C[C@H](N)C(=O)O",
    }])
    derived = tmp_path / "derived.csv"
    manifest = tmp_path / "manifest.json"
    quarantine = tmp_path / "quarantine.csv"
    write_overlay_artifacts(
        [row], [entry], [], pdb_aliases=aliases,
        derived_path=derived, manifest_path=manifest, quarantine_path=quarantine,
    )
    monkeypatch.setattr(mu, "_DERIVED_CSV", str(derived))
    monkeypatch.setattr(mu, "_DERIVED_MANIFEST", str(manifest))
    base = mu._load_unified_with_reconstruction_rows()
    loaded = mu._load_derived_rows(base)
    all_rows = {**base, **loaded}
    assert mu._load_derived_aliases(all_rows, loaded) == {"ALA_ALIAS": "A"}

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entries"][0]["graph_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest graph_sha256"):
        mu._load_derived_aliases(all_rows, loaded)

    payload["entries"][0]["graph_sha256"] = entry["graph_sha256"]
    payload["pdb_aliases"][0]["full_inchikey"] = "INVALID"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="alias 'ALA_ALIAS' full_inchikey"):
        mu._load_derived_aliases(all_rows, loaded)


def test_post_evaluation_bundle_is_atomic_immutable_and_append_only(tmp_path):
    first = publish_overlay_bundle(
        tmp_path / "generation_001",
        [{
            "pdb_resname": "ZZZ",
            "smiles": TEST_SMILES,
            "input_sha256": "a" * 64,
            "source_entity_id": "entity-1",
        }],
        [{
            "record_id": "Q1",
            "pdb_resname": "BAD",
            "status": "not_supported",
            "reason_codes": "AMBIGUOUS_GRAPH",
            "input_sha256": "b" * 64,
            "candidate_graph_count": 2,
            "details_json": "{}",
        }],
    )
    first_rows = list(csv.DictReader(
        (first / "derived_monomer_library.csv").open(encoding="utf-8")
    ))
    first_manifest = json.loads(
        (first / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest["status"] == "VALIDATED_APPEND_ONLY_OVERLAY_BUNDLE"
    assert first_manifest["derived_row_count"] == 1
    assert first_manifest["new_derived_row_count"] == 1

    second = publish_overlay_bundle(
        tmp_path / "generation_002",
        [
            {
                "pdb_resname": "ZZ2",
                "smiles": TEST_SMILES,
                "input_sha256": "c" * 64,
                "source_entity_id": "entity-2",
            },
            {
                "pdb_resname": "BOR",
                "smiles": "B(O)(O)C[C@H](N)C(=O)O",
                "input_sha256": "d" * 64,
                "source_entity_id": "entity-3",
            },
        ],
        existing_bundle=first,
    )
    second_rows = list(csv.DictReader(
        (second / "derived_monomer_library.csv").open(encoding="utf-8")
    ))
    assert second_rows[0] == first_rows[0]
    assert int(second_rows[1]["monomer_id"]) == int(first_rows[0]["monomer_id"]) + 1
    overlay_manifest = json.loads(
        (second / "derived_monomer_manifest.json").read_text(encoding="utf-8")
    )
    assert any(
        alias["pdb_resname"] == "ZZ2"
        and alias["target_symbol"] == first_rows[0]["symbol"]
        for alias in overlay_manifest["pdb_aliases"]
    )
    second_manifest = json.loads(
        (second / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert second_manifest["derived_row_count"] == 2
    assert second_manifest["new_derived_row_count"] == 1
    assert second_manifest["parent_bundle_manifest_sha256"]

    with pytest.raises(FileExistsError, match="already exists"):
        publish_overlay_bundle(second, [])


def test_post_evaluation_bundle_rejects_tampered_parent(tmp_path):
    first = publish_overlay_bundle(
        tmp_path / "generation_001",
        [{"pdb_resname": "ZZZ", "smiles": TEST_SMILES}],
    )
    with (first / "derived_monomer_library.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="artifact mismatch"):
        publish_overlay_bundle(
            tmp_path / "generation_002", [], existing_bundle=first
        )
    assert not (tmp_path / "generation_002").exists()


def test_post_evaluation_bundle_recomputes_self_consistent_parent_manifest(tmp_path):
    first = publish_overlay_bundle(
        tmp_path / "generation_001",
        [{"pdb_resname": "ZZZ", "smiles": TEST_SMILES}],
    )
    overlay_path = first / "derived_monomer_manifest.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    overlay["entries"][0]["graph_sha256"] = "0" * 64
    overlay_path.write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bundle_path = first / "bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["artifacts"]["derived_monomer_manifest.json"] = {
        "sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "size_bytes": overlay_path.stat().st_size,
    }
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest graph_sha256"):
        publish_overlay_bundle(
            tmp_path / "generation_002", [], existing_bundle=first
        )
    assert not (tmp_path / "generation_002").exists()


def test_post_evaluation_bundle_cleans_stage_on_failure(tmp_path, monkeypatch):
    before = set(tmp_path.iterdir())

    def fail_writer(*_args, **_kwargs):
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(derived_monomers, "write_overlay_artifacts", fail_writer)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        publish_overlay_bundle(
            tmp_path / "generation_001",
            [{"pdb_resname": "ZZZ", "smiles": TEST_SMILES}],
        )
    assert set(tmp_path.iterdir()) == before
    assert not (tmp_path / "generation_001").exists()
