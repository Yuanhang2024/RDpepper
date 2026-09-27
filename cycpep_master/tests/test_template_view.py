from __future__ import annotations

import hashlib
import json

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.docking import template_library
from cycpep_master.docking.template_view import (
    TemplateLibraryError,
    load_template_library_view,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_index(tmp_path, entries):
    index = tmp_path / "templates_index.json"
    index.write_text(json.dumps(entries), encoding="utf-8")
    return index


def test_view_filters_disallowed_sources_and_is_immutable(tmp_path):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "cpbind.pdb",
                "smiles": "CC",
                "source": "cpbind",
            },
            "4_N-C_002": {
                "pdb_path": "cpsea.pdb",
                "smiles": "CCC",
                "source": "cpsea",
            },
            "4_N-C_003": {
                "pdb_path": "synthetic.pdb",
                "smiles": "CCCC",
                "source": "synthetic",
            },
        },
    )
    view = load_template_library_view(index)
    assert list(view.entries) == ["4_N-C_001"]
    assert view.allowed_sources == frozenset({"cpbind", "scaffold"})
    assert view.formal is False
    with pytest.raises(TypeError):
        view.entries["new"] = {}
    with pytest.raises(TypeError):
        view.entries["4_N-C_001"]["source"] = "cpsea"


def test_view_rejects_path_escape_even_for_nonformal_index(tmp_path):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "../outside.pdb",
                "smiles": "CC",
                "source": "cpbind",
            }
        },
    )
    with pytest.raises(TemplateLibraryError, match="escapes library root"):
        load_template_library_view(index)


def test_formal_view_verifies_index_templates_and_build_artifacts(tmp_path):
    template = tmp_path / "4_N-C" / "centroid_001.pdb"
    template.parent.mkdir()
    template.write_text("MODEL\nENDMDL\n", encoding="ascii")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("{}\n", encoding="ascii")
    build_script = tmp_path / "build_clean.py"
    build_script.write_text("# frozen builder\n", encoding="ascii")
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "4_N-C/centroid_001.pdb",
                "smiles": "CC",
                "source": "cpbind",
            }
        },
    )
    manifest = tmp_path / "template_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "index_sha256": _sha(index),
                "allowed_sources": ["cpbind", "scaffold"],
                "entry_count": 1,
                "build_inputs": {"source_manifest.json": _sha(source_manifest)},
                "build_scripts": {"build_clean.py": _sha(build_script)},
                "template_files": {
                    "4_N-C/centroid_001.pdb": _sha(template),
                },
            }
        ),
        encoding="utf-8",
    )
    view = load_template_library_view(
        index, manifest_path=manifest, require_formal=True
    )
    assert view.formal is True
    assert view.template_path(view.entries["4_N-C_001"]) == template.resolve()

    template.write_text("changed\n", encoding="ascii")
    with pytest.raises(TemplateLibraryError, match="PDB SHA-256 mismatch"):
        load_template_library_view(index, manifest_path=manifest, require_formal=True)


def test_find_template_fails_closed_when_map_cannot_be_assembled(
    tmp_path, monkeypatch
):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "template.pdb",
                "smiles": "CC",
                "source": "cpbind",
            }
        },
    )
    view = load_template_library_view(index)
    monkeypatch.setattr(template_library, "_smiles_from_map", lambda _value: None)
    meta = {}
    assert template_library.find_template(
        "AAAA{cyc:N-C}", library_view=view, meta_out=meta
    ) is None
    assert meta == {"status": "not_assessable", "reason": "MAP_TO_SMILES_FAILED"}


def test_ranking_is_deterministic_and_shared_source_bonus_is_applied():
    query = Chem.MolFromSmiles("CC")
    entries = [
        ("4_N-C_002", {"smiles": "CC", "source": "scaffold"}),
        ("4_N-C_001", {"smiles": "CC", "source": "cpbind"}),
    ]
    ranked, invalid = template_library._rank_entries(query, entries)
    assert invalid == 0
    assert [item[2] for item in ranked] == ["4_N-C_001", "4_N-C_002"]


def test_generate_conformers_distinguishes_missing_template_from_fallback(
    tmp_path,
):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "missing.pdb",
                "smiles": "C1CCCCC1",
                "source": "cpbind",
            }
        },
    )
    view = load_template_library_view(index)
    meta = {}
    mol, conformers = template_library.generate_conformers(
        "C1CCCCC1",
        "AAAA{cyc:N-C}",
        n_conformers=1,
        meta_out=meta,
        library_view=view,
    )
    assert mol is not None
    assert conformers
    assert meta["status"] == "fallback_success"
    assert meta["template_status"] == "template_load_failed"
    assert meta["template_audit"]["missing_pdb_count"] == 1
    trace = meta["template_audit"]["attempt_trace"]
    assert len(trace) == 1
    assert trace[0]["attempt_index"] == 0
    assert trace[0]["template_key"] == "4_N-C_001"
    assert trace[0]["source"] == "cpbind"
    assert trace[0]["similarity"] == 1.0
    assert trace[0]["ranking_score"] == 1.03
    assert trace[0]["random_seed"] == 1042
    assert trace[0]["pdb_path"] == "missing.pdb"
    assert trace[0]["status"] == "missing_pdb"
    assert trace[0]["template_pdb_sha256"] is None
    assert trace[0]["elapsed_sec"] >= 0.0
    assert meta["fallback_audit"]["status"] == "success"
    assert meta["fallback_audit"]["selected_pool_conformer_ids"]
    assert meta["scene"] == "B"


def _embedded_hydrogenated_mol(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _single_working_template_view(tmp_path):
    template = tmp_path / "template.pdb"
    template.write_text("END\n", encoding="ascii")
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": template.name,
                "smiles": "C1CCCCC1",
                "source": "cpbind",
                "cyc_mode": "N-C",
                "n_res": 4,
            }
        },
    )
    return load_template_library_view(index)


def test_partial_template_success_is_supplemented_to_requested_count(
    tmp_path, monkeypatch
):
    view = _single_working_template_view(tmp_path)
    guided = _embedded_hydrogenated_mol("C1CCCCC1")

    def fake_borrow(*_args, meta_out=None, **_kwargs):
        meta_out.update(
            {
                "status": "success",
                "matched_residue_count": 4,
                "fixed_ca_atom_count": 4,
                "maximum_ca_drift_angstrom": 0.125,
                "optimization_status": "converged",
                "optimization_result": 0,
            }
        )
        return Chem.Mol(guided), "ok"

    monkeypatch.setattr(template_library, "borrow_residue_coords", fake_borrow)
    meta = {}
    mol, conformers = template_library.generate_conformers(
        "C1CCCCC1",
        "AAAA{cyc:N-C}",
        n_conformers=2,
        meta_out=meta,
        library_view=view,
        random_seed=31,
    )

    assert mol is not None
    assert len(conformers) == 2
    assert meta["status"] == "hybrid_success"
    assert meta["scene"] == "A+B"
    assert meta["requested_conformer_count"] == 2
    assert meta["produced_conformer_count"] == 2
    assert meta["guided_conformer_count"] == 1
    assert meta["fallback_conformer_count"] == 1
    trace = meta["template_audit"]["attempt_trace"]
    assert len(trace) == 1
    assert trace[0]["status"] == "success"
    assert trace[0]["random_seed"] == 1031
    assert trace[0]["borrow_audit"]["optimization_status"] == "converged"
    assert trace[0]["borrow_audit"]["maximum_ca_drift_angstrom"] == 0.125
    assert len(trace[0]["template_pdb_sha256"]) == 64
    assert trace[0]["elapsed_sec"] >= 0.0
    fallback = meta["fallback_audit"]
    assert fallback["status"] == "success"
    assert fallback["selected_pool_conformer_ids"]
    assert fallback["output_conformer_ids"] == conformers
    assert len(fallback["optimization_trace"]) == fallback["final_embed_count"]
    assert {row["status"] for row in fallback["optimization_trace"]} <= {
        "converged", "iteration_limit", "exception"
    }


def test_partial_template_result_is_retained_when_fallback_embedding_fails(
    tmp_path, monkeypatch
):
    view = _single_working_template_view(tmp_path)
    guided = _embedded_hydrogenated_mol("C1CCCCC1")

    def fake_borrow(*_args, meta_out=None, **_kwargs):
        meta_out.update({"status": "success", "optimization_status": "iteration_limit"})
        return Chem.Mol(guided), "ok"

    monkeypatch.setattr(template_library, "borrow_residue_coords", fake_borrow)
    monkeypatch.setattr(template_library.AllChem, "EmbedMultipleConfs", lambda *_a, **_k: [])
    meta = {}
    mol, conformers = template_library.generate_conformers(
        "C1CCCCC1",
        "AAAA{cyc:N-C}",
        n_conformers=2,
        meta_out=meta,
        library_view=view,
    )

    assert mol is not None
    assert len(conformers) == 1
    assert meta["status"] == "partial_template_success"
    assert meta["produced_conformer_count"] == 1
    assert meta["requested_conformer_count"] == 2
    assert meta["reason"] == "ETKDG_POOL_EMBED_FAILED_AFTER_PARTIAL_TEMPLATE_SUCCESS"
    assert meta["fallback_audit"]["status"] == "embed_failed"
    assert meta["fallback_audit"]["random_coordinate_retry_used"] is True


def test_borrow_residue_coords_records_mmff_iteration_limit(monkeypatch):
    smiles = "N[C@@H](C)C(=O)N[C@@H](C)C(=O)O"
    template = Chem.AddHs(Chem.MolFromSmiles(smiles))
    template.AddConformer(Chem.Conformer(template.GetNumAtoms()))

    from cycpep_master.docking import build_template_library

    monkeypatch.setattr(
        build_template_library,
        "_load_template_mol",
        lambda *_args, **_kwargs: (Chem.Mol(template), None),
    )

    def fake_embed(mol, _params):
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
        return 0

    monkeypatch.setattr(template_library.AllChem, "EmbedMolecule", fake_embed)
    monkeypatch.setattr(
        template_library.AllChem, "MMFFOptimizeMolecule", lambda *_a, **_k: 1
    )
    meta = {}
    mol, detail = template_library.borrow_residue_coords(
        smiles, "AA", "unused.pdb", smiles, meta_out=meta
    )

    assert mol is not None
    assert detail.startswith("ok")
    assert meta["status"] == "success"
    assert meta["matched_residue_count"] == 2
    assert meta["fixed_ca_atom_count"] == 2
    assert meta["optimization_status"] == "iteration_limit"
    assert meta["optimization_result"] == 1
    assert meta["maximum_ca_drift_angstrom"] == 0.0
    assert len(meta["backbone_atom_map"]) == 2


def test_borrow_residue_coords_rejects_prefix_only_backbone_mapping(
    monkeypatch,
):
    generated = "NCC(=O)NCC(=O)O"
    template_smiles = "NCC(=O)O"
    template = Chem.AddHs(Chem.MolFromSmiles(template_smiles))
    template.AddConformer(Chem.Conformer(template.GetNumAtoms()))

    from cycpep_master.docking import build_template_library

    monkeypatch.setattr(
        build_template_library,
        "_load_template_mol",
        lambda *_args, **_kwargs: (Chem.Mol(template), None),
    )
    meta = {}
    molecule, error = template_library.borrow_residue_coords(
        generated,
        "GG",
        "unused.pdb",
        template_smiles,
        meta_out=meta,
    )

    assert molecule is None
    assert "complete residue-level backbone mapping required" in error
    assert meta["reason"] == "BACKBONE_MAPPING_INCOMPLETE"
    assert meta["expected_residue_count"] == 2
    assert meta["generated_backbone_count"] == 2
    assert meta["template_backbone_count"] == 1


def test_explicit_template_strategies_have_distinct_selection_scopes(tmp_path):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "nc.pdb",
                "smiles": "CC",
                "source": "cpbind",
                "n_res": 4,
            },
            "4_SC_001": {
                "pdb_path": "sc.pdb",
                "smiles": "CCC",
                "source": "scaffold",
                "n_res": 4,
            },
            "5_N-C_001": {
                "pdb_path": "five.pdb",
                "smiles": "CCCC",
                "source": "cpbind",
                "n_res": 5,
            },
        },
    )
    view = load_template_library_view(index)
    query = Chem.MolFromSmiles("CCC")

    full, _, full_bucket, full_scope = template_library._strategy_candidates(
        "AAAA{cyc:N-C}", query, view, "full", 17
    )
    nearest, _, nearest_bucket, nearest_scope = (
        template_library._strategy_candidates(
            "AAAA{cyc:N-C}", query, view, "nearest_morgan", 17
        )
    )
    random_a, _, random_bucket, random_scope = (
        template_library._strategy_candidates(
            "AAAA{cyc:N-C}", query, view, "random_same_length", 17
        )
    )
    random_b, _, _, _ = template_library._strategy_candidates(
        "AAAA{cyc:N-C}", query, view, "random_same_length", 17
    )

    assert [row[2] for row in full] == ["4_N-C_001"]
    assert full_bucket == "4_N-C"
    assert full_scope == "residue_count_and_topology"
    assert [row[2] for row in nearest] == ["4_SC_001", "4_N-C_001"]
    assert nearest_bucket == "4_ANY_TOPOLOGY"
    assert nearest_scope == "same_length_pure_morgan"
    assert random_bucket == "4_ANY_TOPOLOGY"
    assert random_scope == "same_length_seeded_random"
    assert [row[2] for row in random_a] == [row[2] for row in random_b]
    assert {row[2] for row in random_a} == {"4_N-C_001", "4_SC_001"}


def test_template_off_is_a_first_class_deterministic_control(tmp_path):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "unused.pdb",
                "smiles": "C1CCCCC1",
                "source": "cpbind",
                "n_res": 4,
            }
        },
    )
    view = load_template_library_view(index)
    lookup_meta = {}
    assert template_library.find_template(
        "AAAA{cyc:N-C}",
        library_view=view,
        meta_out=lookup_meta,
        template_strategy="off",
        random_seed=91,
    ) is None
    assert lookup_meta == {
        "status": "template_disabled",
        "template_strategy": "off",
        "random_seed": 91,
        "template_count": 0,
    }

    generation_meta = {}
    molecule, conformers = template_library.generate_conformers(
        "C1CCCCC1",
        "AAAA{cyc:N-C}",
        n_conformers=1,
        meta_out=generation_meta,
        library_view=view,
        template_strategy="off",
        random_seed=91,
    )
    assert molecule is not None
    assert conformers
    assert generation_meta["status"] == "fallback_success"
    assert generation_meta["template_status"] == "disabled_by_control"
    assert generation_meta["template_strategy"] == "off"
    assert generation_meta["random_seed"] == 91
    assert generation_meta["template_audit"]["candidate_attempt_budget"] == 3
    assert generation_meta["template_audit"]["attempted_count"] == 0


def test_unknown_template_strategy_is_rejected(tmp_path):
    index = _write_index(tmp_path, {})
    view = load_template_library_view(index)
    with pytest.raises(ValueError, match="template_strategy"):
        template_library.generate_conformers(
            "C1CCCCC1",
            "AAAA{cyc:N-C}",
            library_view=view,
            template_strategy="unknown",
        )


def test_nonpositive_conformer_count_is_rejected(tmp_path):
    view = load_template_library_view(_write_index(tmp_path, {}))
    with pytest.raises(ValueError, match="at least one"):
        template_library.generate_conformers(
            "C1CCCCC1",
            "AAAA{cyc:N-C}",
            n_conformers=0,
            library_view=view,
        )


def test_large_molecule_without_template_skips_unguided_pool(
    tmp_path, monkeypatch
):
    view = load_template_library_view(_write_index(tmp_path, {}))

    def unexpected_embed(*_args, **_kwargs):
        raise AssertionError("large-molecule resource policy must skip ETKDG pool")

    monkeypatch.setattr(
        template_library.AllChem, "EmbedMultipleConfs", unexpected_embed
    )
    meta = {}
    molecule, conformers = template_library.generate_conformers(
        "C" * template_library.LARGE_MOLECULE_HEAVY_ATOMS,
        "AAAA{cyc:N-C}",
        n_conformers=5,
        meta_out=meta,
        library_view=view,
    )

    assert molecule is None
    assert conformers[0].startswith("resource_limit:")
    assert meta["status"] == "total_failed"
    assert meta["failure_class"] == "resource_limit"
    assert meta["reason"] == "RESOURCE_LIMIT_UNGUIDED_ETKDG_SKIPPED"
    assert meta["resource_policy"] == {
        "policy": "bounded_large_molecule",
        "heavy_atom_count": template_library.LARGE_MOLECULE_HEAVY_ATOMS,
        "large_molecule_threshold": template_library.LARGE_MOLECULE_HEAVY_ATOMS,
        "etkdg_timeout_seconds": template_library.ETKDG_TIMEOUT_SECONDS,
        "candidate_attempt_budget": 3,
        "unguided_fallback_allowed": False,
    }
    assert meta["fallback_audit"]["status"] == "resource_limited"


def test_large_partial_template_result_is_retained_without_fallback(
    tmp_path, monkeypatch
):
    large_smiles = "C" * template_library.LARGE_MOLECULE_HEAVY_ATOMS
    template = tmp_path / "template.pdb"
    template.write_text("END\n", encoding="ascii")
    view = load_template_library_view(
        _write_index(
            tmp_path,
            {
                "4_N-C_001": {
                    "pdb_path": template.name,
                    "smiles": large_smiles,
                    "source": "cpbind",
                    "cyc_mode": "N-C",
                    "n_res": 4,
                }
            },
        )
    )
    guided = Chem.AddHs(Chem.MolFromSmiles(large_smiles))
    guided.AddConformer(Chem.Conformer(guided.GetNumAtoms()))

    def fake_borrow(*_args, meta_out=None, **_kwargs):
        meta_out.update({"status": "success"})
        return Chem.Mol(guided), "ok"

    def unexpected_embed(*_args, **_kwargs):
        raise AssertionError("partial large-molecule result must not use fallback")

    monkeypatch.setattr(template_library, "borrow_residue_coords", fake_borrow)
    monkeypatch.setattr(
        template_library.AllChem, "EmbedMultipleConfs", unexpected_embed
    )
    meta = {}
    molecule, conformers = template_library.generate_conformers(
        large_smiles,
        "AAAA{cyc:N-C}",
        n_conformers=2,
        meta_out=meta,
        library_view=view,
    )

    assert molecule is not None
    assert len(conformers) == 1
    assert meta["status"] == "partial_template_success"
    assert meta["reason"] == "RESOURCE_LIMIT_FALLBACK_SKIPPED"
    assert meta["failure_class"] == "resource_limit"
    assert meta["guided_conformer_count"] == 1
    assert meta["fallback_conformer_count"] == 0
    assert meta["template_audit"]["candidate_attempt_budget"] == 3
    assert meta["fallback_audit"]["initial_embed_count"] == 0


def test_medium_molecule_does_not_repeat_failed_pool_embedding(
    tmp_path, monkeypatch
):
    view = load_template_library_view(_write_index(tmp_path, {}))
    calls = []

    def failed_embed(_mol, numConfs, params):
        calls.append((numConfs, params.timeout, params.useRandomCoords))
        return []

    monkeypatch.setattr(
        template_library.AllChem, "EmbedMultipleConfs", failed_embed
    )
    meta = {}
    molecule, conformers = template_library.generate_conformers(
        "C" * (template_library.RANDOM_RETRY_MAX_HEAVY_ATOMS + 1),
        "AAAA{cyc:N-C}",
        n_conformers=5,
        meta_out=meta,
        library_view=view,
    )

    assert molecule is None
    assert conformers == ["ETKDG pool embed failed (scene B)"]
    assert calls == [
        (
            10,
            template_library.ETKDG_TIMEOUT_SECONDS,
            False,
        )
    ]
    assert meta["fallback_audit"]["random_coordinate_retry_allowed"] is False
    assert meta["fallback_audit"]["random_coordinate_retry_used"] is False


def test_cyc_mode_recognizes_real_index_disulfide_bucket():
    assert template_library._cyc_mode("AAAA") == "none"
    assert template_library._cyc_mode("AAAA{cyc:N-C}") == "N-C"
    actual_index_map = "CHLQATDYGC{cyc:2:R3-11:R3}{nt:ACE}{ct:NME}"
    assert template_library._cyc_mode(actual_index_map) == "disulfide"
    assert template_library._cyc_mode(
        "PARCTIVSCN{cyc:4:R3-9:R3}{cyc:N-C}"
    ) == "mixed"


def test_bucket_nc_fallback_only_for_explicit_sc(tmp_path):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "nc.pdb",
                "smiles": "CC",
                "source": "cpbind",
                "n_res": 4,
                "cyc_mode": "N-C",
            }
        },
    )
    view = load_template_library_view(index)
    # linear/none must not borrow the N-C bucket.
    assert template_library._bucket_entries("AAAA", view) == ([], "")
    # A non-disulfide side-chain closure may borrow N-C when no SC bucket exists.
    sc_entries, sc_bucket = template_library._bucket_entries(
        "AKAA{cyc:2:R3-4:R3}", view
    )
    assert [key for key, _ in sc_entries] == ["4_N-C_001"]
    assert sc_bucket == "4_N-C"
    # mixed topologies must not borrow N-C.
    assert template_library._bucket_entries(
        "AKAA{cyc:2:R3-4:R3}{cyc:N-C}", view
    ) == ([], "")


def test_real_disulfide_map_selects_disulfide_bucket(tmp_path):
    actual_index_map = "CHLQATDYGC{cyc:2:R3-11:R3}{nt:ACE}{ct:NME}"
    index = _write_index(
        tmp_path,
        {
            "10_disulfide_001": {
                "pdb_path": "ss.pdb",
                "smiles": "CSSC",
                "source": "cpbind",
                "n_res": 10,
                "cyc_mode": "disulfide",
            },
            "10_SC_001": {
                "pdb_path": "sc.pdb",
                "smiles": "CCCC",
                "source": "cpbind",
                "n_res": 10,
                "cyc_mode": "SC",
            },
        },
    )
    view = load_template_library_view(index)

    entries, bucket = template_library._bucket_entries(actual_index_map, view)

    assert bucket == "10_disulfide"
    assert [key for key, _ in entries] == ["10_disulfide_001"]


def test_malformed_n_res_entries_are_skipped_and_counted_invalid(tmp_path):
    index = _write_index(
        tmp_path,
        {
            "4_N-C_001": {
                "pdb_path": "good.pdb",
                "smiles": "CC",
                "source": "cpbind",
                "n_res": "not-an-int",
            },
            "4_N-C_002": {
                "pdb_path": "good2.pdb",
                "smiles": "CCC",
                "source": "cpbind",
                "n_res": 4,
            },
            "4_N-C_003": {
                "pdb_path": "bad-smiles.pdb",
                "smiles": "not a smiles",
                "source": "cpbind",
                "n_res": 4,
            },
        },
    )
    view = load_template_library_view(index)
    query = Chem.MolFromSmiles("CCC")

    entries, bucket, invalid_count = template_library._same_length_entries(
        "AAAA{cyc:N-C}", view
    )
    assert invalid_count == 1
    # Bad n_res is skipped; the bad-SMILES row is still same-length and is
    # filtered later by _strategy_candidates (invalid_template_count).
    assert [key for key, _ in entries] == ["4_N-C_002", "4_N-C_003"]

    ranked, invalid, bucket, scope = template_library._strategy_candidates(
        "AAAA{cyc:N-C}", query, view, "nearest_morgan", 17
    )
    assert invalid == 2  # malformed n_res + unparseable template SMILES
    assert [row[2] for row in ranked] == ["4_N-C_002"]

    meta = {}
    template_library.find_template(
        "AAAA{cyc:N-C}",
        library_view=view,
        meta_out=meta,
        template_strategy="nearest_morgan",
    )
    assert meta["invalid_template_count"] == 2
