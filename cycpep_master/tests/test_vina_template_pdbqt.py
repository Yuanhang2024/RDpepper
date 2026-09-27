from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.docking import vina_wrapper
from cycpep_master.docking.template_library import generate_conformers

pytest.importorskip("meeko")
pytestmark = pytest.mark.skip(
    reason=(
        "V4 retired template-guided direct PDBQT; templates are no longer "
        "a ligand-flexibility runtime input"
    )
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES_INDEX = PACKAGE_ROOT / "data/templates/templates_index.json"


def _scene_a_entry() -> tuple[str, str]:
    """Return (smiles, map) of a packaged template whose borrow succeeds."""
    if not _TEMPLATES_INDEX.is_file():
        pytest.skip("packaged template index not available")
    index = json.loads(_TEMPLATES_INDEX.read_text(encoding="utf-8"))
    for key in ("5_SC_004", "5_none_001", "5_SC_007"):
        entry = index.get(key)
        if entry is None:
            continue
        pdb = PACKAGE_ROOT / "data/templates" / str(entry["pdb_path"]).replace("\\", "/")
        if pdb.is_file():
            return entry["smiles"], entry["map"]
    pytest.skip("no packaged scene-A template available")


def test_template_guided_pdbqt_uses_template_coordinates(tmp_path):
    """The template branch must install template coords, not silently fall back."""
    smiles, map_str = _scene_a_entry()
    output = tmp_path / "ligand.pdbqt"
    audit = {}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        smiles,
        str(output),
        generated_map=map_str,
        n_template_confs=1,
        num_confs=1,
        protonate=True,
        random_seed=42,
        conf_out=audit,
    )
    assert error is None
    assert audit["source"] == "template"
    assert audit["template_error"] is None
    assert audit["template_meta"]["scene"] == "A"
    assert output.exists() and output.stat().st_size > 0
    pdbqt = output.read_text(encoding="utf-8")
    assert "ROOT" in pdbqt and "TORSDOF" in pdbqt

    # Same seed, same molecule: the docked conformer must BE the template-guided
    # conformer (coordinate identity proves no silent ETKDG re-embed).
    reference, ref_cids = generate_conformers(
        vina_wrapper.protonate_ph74(smiles), map_str, n_conformers=1, meta_out={}
    )
    assert reference is not None and ref_cids
    ref_conf = reference.GetConformer(ref_cids[0])
    out_conf = audit["mol_3d"].GetConformer(0)
    assert out_conf.GetNumAtoms() == ref_conf.GetNumAtoms()
    max_diff = max(
        abs(out_conf.GetAtomPosition(i).x - ref_conf.GetAtomPosition(i).x)
        + abs(out_conf.GetAtomPosition(i).y - ref_conf.GetAtomPosition(i).y)
        + abs(out_conf.GetAtomPosition(i).z - ref_conf.GetAtomPosition(i).z)
        for i in range(out_conf.GetNumAtoms())
    )
    assert max_diff < 1e-3


def test_template_scene_b_reports_etkdg_with_audit(tmp_path):
    """A map with no compatible template must be reported as etkdg, not template."""
    output = tmp_path / "ligand.pdbqt"
    audit = {}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "C1CCCCC1",
        str(output),
        generated_map="AAAA{cyc:N-C}",
        n_template_confs=1,
        num_confs=1,
        protonate=False,
        random_seed=42,
        conf_out=audit,
    )
    assert error is None
    assert audit["source"] == "etkdg"
    assert audit["template_meta"]["scene"] == "B"
    assert audit["template_error"] is None


def test_hybrid_primary_pose_and_rigidity_sources_are_separate(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import template_library

    smiles = "CCO"
    ensemble = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            ensemble, numConfs=2, randomSeed=9, numThreads=1
        )
    )

    def fake_generate(
        _smiles,
        _map,
        n_conformers=2,
        meta_out=None,
        **_kwargs,
    ):
        assert n_conformers == 2
        meta_out.update({
            "status": "hybrid_success",
            "scene": "A+B",
            "guided_conformer_count": 1,
            "fallback_conformer_count": 1,
            "template_audit": {
                "selected_template_keys": ["template_0"]
            },
        })
        return ensemble, conformer_ids

    monkeypatch.setattr(
        template_library, "generate_conformers", fake_generate
    )
    output = tmp_path / "hybrid.pdbqt"
    audit = {}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        smiles,
        str(output),
        generated_map="AAA",
        n_template_confs=2,
        protonate=False,
        conf_out=audit,
    )

    assert error is None
    assert audit["source"] == "template"
    assert audit["primary_conformer_source"] == "template"
    assert audit["rigidity_reference_source"] == "not_evaluated"
    assert audit["rigidity_evidence_level"] == "not_evaluated"


def test_no_generated_map_reports_embed3d(tmp_path):
    output = tmp_path / "ligand.pdbqt"
    audit = {}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "C1CCCCC1",
        str(output),
        num_confs=1,
        random_seed=42,
        conf_out=audit,
    )
    assert error is None
    assert audit["source"] == "embed3d"
    assert audit["template_meta"] is None
    assert audit["template_error"] is None


def test_template_branch_exception_is_recorded_not_silent(tmp_path, monkeypatch):
    """A broken template conformer must fall back visibly, not silently."""
    from cycpep_master.docking import template_library

    mismatch_mol = Chem.AddHs(Chem.MolFromSmiles("CCC"))  # 10 atoms incl. H
    assert AllChem.EmbedMolecule(mismatch_mol, randomSeed=1) == 0

    def fake_generate(_smiles, _map, n_conformers=1, meta_out=None):
        meta_out.clear()
        meta_out.update({"status": "template_success", "scene": "A"})
        return mismatch_mol, [0]

    monkeypatch.setattr(template_library, "generate_conformers", fake_generate)

    output = tmp_path / "ligand.pdbqt"
    audit = {}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "CCO",  # 9 atoms incl. H -> conformer count mismatch
        str(output),
        generated_map="AAA{cyc:N-C}",
        n_template_confs=1,
        num_confs=1,
        protonate=False,
        random_seed=42,
        conf_out=audit,
    )
    assert error is None
    assert audit["source"] == "embed3d"  # fell back to ETKDG ...
    assert audit["template_error"] is not None  # ... but the reason is recorded
    assert "mismatch" in audit["template_error"]
    assert audit["template_meta"]["scene"] == "A"


def test_resource_limited_template_generation_is_not_retried_unguided(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import template_library
    from cycpep_master.export import conformer

    def fake_generate(_smiles, _map, n_conformers=1, meta_out=None):
        meta_out.clear()
        meta_out.update({
            "status": "total_failed",
            "failure_class": "resource_limit",
            "reason": "RESOURCE_LIMIT_UNGUIDED_ETKDG_SKIPPED",
        })
        return None, ["resource_limit"]

    def unexpected_embed(*_args, **_kwargs):
        raise AssertionError("resource-limited generation must not be retried")

    monkeypatch.setattr(template_library, "generate_conformers", fake_generate)
    monkeypatch.setattr(conformer, "_embed_3d", unexpected_embed)

    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "C1CCCCC1",
        str(tmp_path / "unused.pdbqt"),
        generated_map="AAAA{cyc:N-C}",
        protonate=False,
    )

    assert error == (
        "not_supported: conformer generation resource limit: "
        "RESOURCE_LIMIT_UNGUIDED_ETKDG_SKIPPED"
    )


def test_multi_resource_limit_is_not_retried_unguided(tmp_path, monkeypatch):
    from cycpep_master.docking import template_library
    from cycpep_master.export import conformer

    def fake_generate(_smiles, _map, n_conformers=1, meta_out=None):
        meta_out.update({
            "failure_class": "resource_limit",
            "reason": "RESOURCE_LIMIT_UNGUIDED_ETKDG_SKIPPED",
        })
        return None, ["resource_limit"]

    def unexpected_embed(*_args, **_kwargs):
        raise AssertionError("resource-limited generation must not be retried")

    monkeypatch.setattr(template_library, "generate_conformers", fake_generate)
    monkeypatch.setattr(conformer, "_embed_3d", unexpected_embed)

    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "C1CCCCC1",
        str(tmp_path),
        n_confs=2,
        generated_map="AAAA{cyc:N-C}",
    )

    assert paths == []
    assert error == (
        "not_supported: conformer generation resource limit: "
        "RESOURCE_LIMIT_UNGUIDED_ETKDG_SKIPPED"
    )


def test_multi_template_guided_pdbqt_writes_all_conformers(tmp_path):
    smiles, map_str = _scene_a_entry()
    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        smiles,
        str(tmp_path),
        n_confs=2,
        generated_map=map_str,
    )
    assert error is None
    assert len(paths) == 2
    for path in paths:
        assert Path(path).stat().st_size > 0
