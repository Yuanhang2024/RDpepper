from __future__ import annotations

import builtins
import re
from types import SimpleNamespace
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from cycpep_master import application
from cycpep_master.docking import vina_wrapper
from cycpep_master.docking.receptor_pdbqt import _clear_prior_pdbqt_output
from cycpep_master.paths._map_utils import get_smi_from_map


pytest.importorskip("meeko")

V3_MULTI_RETIRED = pytest.mark.skip(
    reason="V4 retired direct multi-output PDBQT preparation"
)
V3_DIRECT_FLEX_RETIRED = pytest.mark.skip(
    reason="V4 flexibility is evaluated only downstream of validated MOL2"
)


def _cyclic_lysine_smiles(length: int = 4) -> str:
    smiles = get_smi_from_map("K" * length + "{cyc:N-C}")
    assert smiles
    return smiles


def _write_minimal_pdb(path: Path) -> None:
    path.write_text(
        "HETATM    1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00          C  \n"
        "END\n",
        encoding="ascii",
    )


def _write_explicit_h_methane(path: Path) -> None:
    lines = [
        "HETATM    1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00          C  \n"
    ]
    coordinates = (
        (1.090, 0.000, 0.000),
        (-0.363, 1.027, 0.000),
        (-0.363, -0.513, 0.889),
        (-0.363, -0.513, -0.889),
    )
    for serial, (x, y, z) in enumerate(coordinates, start=2):
        lines.append(
            f"HETATM{serial:5d}  H{serial - 1:<2} LIG A   1       "
            f"{x:7.3f}   {y:7.3f}   {z:7.3f}  1.00  0.00          H  \n"
        )
    lines.extend(("CONECT    1    2    3    4    5\n", "END\n"))
    path.write_text("".join(lines), encoding="ascii")


def test_tree_validator_rejects_atom_outside_root_or_branch():
    malformed = "\n".join(
        (
            "ROOT",
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00     0.000 C",
            "ENDROOT",
            "ATOM      2  C   LIG A   1       1.000   0.000   0.000  0.00  0.00     0.000 C",
            "TORSDOF 0",
            "",
        )
    )
    with pytest.raises(RuntimeError, match="outside ROOT/BRANCH"):
        vina_wrapper._validate_pdbqt_torsion_tree(malformed)


def test_unsafe_string_branch_editor_is_disabled():
    with pytest.raises(RuntimeError, match="post-hoc BRANCH deletion is disabled"):
        vina_wrapper._freeze_lowest_sigma_bonds("", None, limit=1)


def test_invalid_limit_fails_closed_and_removes_stale_output(tmp_path):
    output = tmp_path / "ligand.pdbqt"
    output.write_text("stale", encoding="utf-8")
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "CC",
        str(output),
        torsdof_limit=-1,
    )
    assert error == "torsdof_limit must be non-negative"
    assert output.read_text(encoding="utf-8") == "stale"


def test_invalid_random_seed_returns_an_error_and_removes_stale_output(tmp_path):
    output = tmp_path / "ligand.pdbqt"
    output.write_text("stale", encoding="utf-8")
    audit = {"stale": True}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "CC",
        str(output),
        random_seed="bad",
        conf_out=audit,
    )
    assert error == "random_seed must be an integer"
    assert output.read_text(encoding="utf-8") == "stale"
    assert audit == {}


def test_smiles_ligand_pdbqt_rejects_unsupported_elements(tmp_path):
    output = tmp_path / "selenium.pdbqt"

    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "[Se]C",
        str(output),
        protonate=False,
    )

    assert error == "not_supported: unsupported ligand element(s): Se"
    assert not output.exists()


def test_pdb_ligand_pdbqt_rejects_unsupported_elements(tmp_path):
    source = tmp_path / "selenium.pdb"
    output = tmp_path / "selenium.pdbqt"
    source.write_text(
        "HETATM    1 SE   LIG A   1       0.000   0.000   0.000  1.00  0.00          Se  \n"
        "END\n",
        encoding="ascii",
    )

    error = vina_wrapper.pdb_to_pdbqt_meeko(str(source), str(output))

    assert "validated MOL2" in error
    assert not output.exists()


def test_pdb_ligand_facade_routes_through_validated_mol2(
    tmp_path, monkeypatch
):
    source = tmp_path / "ligand.pdb"
    output = tmp_path / "ligand.pdbqt"
    source.write_text(
        "ATOM      1  C   GLY A   1       0.000   0.000   "
        "0.000  1.00  0.00           C  \nEND\n",
        encoding="ascii",
    )
    observed = {}

    def fake_prepare(coordinate_path, output_path, **kwargs):
        observed.update({
            "coordinate_path": coordinate_path,
            "output_path": output_path,
            **kwargs,
        })
        Path(output_path).write_text(
            "ROOT_atoms_marker\n", encoding="ascii"
        )
        return {"status": "success", "data": {}}

    monkeypatch.setattr(
        application,
        "prepare_ligand_pdbqt_from_pdb",
        fake_prepare,
    )

    error = vina_wrapper.pdb_to_pdbqt_meeko(
        str(source), str(output)
    )

    assert error is None
    assert observed["coordinate_path"] == str(source)
    assert observed["output_path"] == str(output)
    assert observed["chain_id"] == "A"


@pytest.mark.parametrize("is_receptor", (False, True))
@pytest.mark.parametrize("alias_kind", ("same", "symlink", "hardlink"))
def test_pdb_conversion_never_clears_input_alias(tmp_path, is_receptor, alias_kind):
    source = tmp_path / "input.pdb"
    _write_minimal_pdb(source)
    original = source.read_bytes()
    output = source if alias_kind == "same" else tmp_path / "output.pdbqt"
    if alias_kind == "symlink":
        try:
            output.symlink_to(source)
        except OSError as exc:
            pytest.skip(f"symbolic links unavailable: {exc}")
    elif alias_kind == "hardlink":
        try:
            output.hardlink_to(source)
        except OSError as exc:
            pytest.skip(f"hard links unavailable: {exc}")

    error = vina_wrapper.pdb_to_pdbqt_meeko(
        str(source), str(output), is_receptor=is_receptor
    )

    assert error is not None
    if not is_receptor:
        assert "must not alias" in error
        assert source.read_bytes() == original
        return
    assert "cannot clear prior" in error
    assert source.exists()
    assert source.read_bytes() == original
    if alias_kind == "symlink":
        assert output.is_symlink()
    else:
        assert output.exists()


def test_pdbqt_cleanup_replaces_unrelated_output_symlink_without_touching_target(
    tmp_path,
):
    source = tmp_path / "input.pdb"
    target = tmp_path / "old-output.pdbqt"
    output = tmp_path / "output.pdbqt"
    _write_minimal_pdb(source)
    target.write_text("prior result", encoding="ascii")
    try:
        output.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")

    error = _clear_prior_pdbqt_output(
        str(source), output, error_prefix="cannot clear prior PDBQT output"
    )

    assert error is None
    assert not output.exists()
    assert not output.is_symlink()
    assert target.read_text(encoding="ascii") == "prior result"


def test_ligand_pdbqt_allows_omittable_explicit_hydrogens(tmp_path):
    source = tmp_path / "methane.pdb"
    output = tmp_path / "methane.pdbqt"
    _write_explicit_h_methane(source)

    error = vina_wrapper.pdb_to_pdbqt_meeko(str(source), str(output))

    assert error is None
    assert output.is_file()
    assert "TORSDOF" in output.read_text(encoding="ascii")


@V3_MULTI_RETIRED
def test_multi_ligand_pdbqt_rejects_unsupported_elements(tmp_path):
    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "[Se]C", str(tmp_path / "ensemble"), n_confs=2
    )

    assert paths == []
    assert error == "not_supported: unsupported ligand element(s): Se"


@V3_MULTI_RETIRED
def test_multi_ligand_pdbqt_preserves_prior_outputs_on_invalid_input(tmp_path):
    output_dir = tmp_path / "ensemble"
    output_dir.mkdir()
    stale = output_dir / "lig_0.pdbqt"
    stale.write_text("old result", encoding="utf-8")

    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "not a smiles", str(output_dir), n_confs=1
    )

    assert paths == []
    assert "invalid SMILES" in error
    assert stale.read_text(encoding="utf-8") == "old result"


@V3_MULTI_RETIRED
def test_multi_ligand_pdbqt_clears_stale_outputs_after_validation(tmp_path):
    pytest.importorskip("meeko")
    output_dir = tmp_path / "ensemble"
    output_dir.mkdir()
    stale = output_dir / "lig_99.pdbqt"
    stale.write_text("old result", encoding="utf-8")

    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "CC", str(output_dir), n_confs=1
    )

    assert error is None
    assert len(paths) == 1
    assert not stale.exists()
    assert (output_dir / "lig_0.pdbqt").exists()


@V3_MULTI_RETIRED
def test_multi_valid_request_clears_stale_before_missing_meeko(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "ensemble"
    output_dir.mkdir()
    stale = output_dir / "lig_0.pdbqt"
    stale.write_text("old result", encoding="utf-8")
    real_import = builtins.__import__

    def missing_meeko(name, *args, **kwargs):
        if name == "meeko":
            raise ImportError("simulated missing Meeko")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_meeko)

    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "CC", str(output_dir), n_confs=1
    )

    assert paths == []
    assert error == "meeko not installed"
    assert not stale.exists()


@V3_MULTI_RETIRED
def test_multi_valid_request_clears_stale_before_embedding_failure(
    tmp_path, monkeypatch
):
    from cycpep_master.export import conformer

    output_dir = tmp_path / "ensemble"
    output_dir.mkdir()
    stale = output_dir / "lig_0.pdbqt"
    stale.write_text("old result", encoding="utf-8")

    def fail_embed(*_args, **_kwargs):
        return None, "simulated embedding failure"

    monkeypatch.setattr(conformer, "_embed_3d", fail_embed)

    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "CC", str(output_dir), n_confs=1
    )

    assert paths == []
    assert error == "3D embed failed: simulated embedding failure"
    assert not stale.exists()


@V3_MULTI_RETIRED
def test_multi_valid_request_clears_stale_before_resource_limit(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import template_library
    from cycpep_master.export import conformer

    output_dir = tmp_path / "ensemble"
    output_dir.mkdir()
    stale = output_dir / "lig_0.pdbqt"
    stale.write_text("old result", encoding="utf-8")

    def resource_limited(_smiles, _map, n_conformers=1, meta_out=None):
        meta_out.update({
            "failure_class": "resource_limit",
            "reason": "simulated resource limit",
        })
        return None, []

    def unexpected_embed(*_args, **_kwargs):
        raise AssertionError("resource-limited generation must not be retried")

    monkeypatch.setattr(template_library, "generate_conformers", resource_limited)
    monkeypatch.setattr(conformer, "_embed_3d", unexpected_embed)

    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "C1CCCCC1",
        str(output_dir),
        n_confs=2,
        generated_map="AAAA{cyc:N-C}",
    )

    assert paths == []
    assert error == (
        "not_supported: conformer generation resource limit: "
        "simulated resource limit"
    )
    assert not stale.exists()


@V3_MULTI_RETIRED
def test_multi_valid_request_clears_stale_before_post_generation_failure(
    tmp_path, monkeypatch
):
    import meeko

    output_dir = tmp_path / "ensemble"
    output_dir.mkdir()
    stale = output_dir / "lig_0.pdbqt"
    stale.write_text("old result", encoding="utf-8")

    class NoSetupPreparation:
        def __init__(self, **_kwargs):
            pass

        def prepare(self, _molecule):
            return []

    monkeypatch.setattr(meeko, "MoleculePreparation", NoSetupPreparation)

    paths, error = vina_wrapper.smiles_to_ligand_pdbqt_multi(
        "CC", str(output_dir), n_confs=1
    )

    assert paths == []
    assert error == "all conformer PDBQT writes failed"
    assert not stale.exists()


def test_failed_stale_output_cleanup_also_clears_conf_out(tmp_path):
    output_directory = tmp_path / "ligand.pdbqt"
    output_directory.mkdir()
    audit = {"stale": True}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "CC",
        str(output_directory),
        torsdof_limit=1,
        conf_out=audit,
    )
    assert "cannot clear prior output" in error
    assert audit == {}


@V3_DIRECT_FLEX_RETIRED
def test_enabled_noop_budget_still_validates_ensemble_parameters(tmp_path):
    output = tmp_path / "ligand.pdbqt"
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "CC",
        str(output),
        torsdof_limit=57,
        torsion_ensemble_size=1,
    )
    assert "torsion_ensemble_size must be an integer >= 2" in error
    assert not output.exists()


@V3_DIRECT_FLEX_RETIRED
def test_adaptive_budget_rebuilds_a_valid_meeko_tree(tmp_path):
    output = tmp_path / "ligand.pdbqt"
    audit = {}
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        _cyclic_lysine_smiles(),
        str(output),
        num_confs=1,
        random_seed=42,
        rigid_macrocycles=True,
        protonate=False,
        torsdof_limit=10,
        torsion_ensemble_size=4,
        torsion_num_threads=1,
        conf_out=audit,
    )
    assert error is None
    text = output.read_text(encoding="utf-8")
    tree = vina_wrapper._validate_pdbqt_torsion_tree(text)
    assert tree["torsdof"] <= 10
    assert tree["branch_count"] == tree["torsdof"]
    assert int(re.findall(r"^TORSDOF\s+(\d+)", text, re.MULTILINE)[-1]) <= 10

    budget = audit["torsion_budget"]
    assert budget["status"] == "applied"
    assert budget["initial_torsdof"] > 10
    assert budget["final_torsdof"] == tree["torsdof"]
    assert budget["ensemble"]["embedded_conformer_count"] >= 2
    assert budget["frozen_bonds"]
    assert all(item["sigma_deg"] < float("inf") for item in budget["frozen_bonds"])
    assert len(budget["bond_sigma_deg"]) == budget["initial_rotatable_bond_count"]
    assert budget["final_rotatable_bond_count"] == tree["torsdof"]
    invariants = budget["setup_atom_invariants"]
    assert invariants["max_abs_coordinate_component_delta"] == 0.0
    assert invariants["max_abs_charge_delta"] == 0.0
    assert invariants["atom_type_changes"] == 0


@V3_DIRECT_FLEX_RETIRED
def test_budget_uses_multiple_matched_templates_without_reembedding(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import template_library, torsion_budget

    smiles = _cyclic_lysine_smiles()
    ensemble = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            ensemble, numConfs=4, randomSeed=19, numThreads=1
        )
    )
    assert len(conformer_ids) == 4
    requested_counts = []

    def fake_generate(
        generated_smiles,
        _generated_map,
        n_conformers=1,
        meta_out=None,
        **_kwargs,
    ):
        assert generated_smiles == smiles
        requested_counts.append(n_conformers)
        result = Chem.Mol(ensemble)
        result.RemoveAllConformers()
        selected = conformer_ids[:n_conformers]
        output_ids = [
            result.AddConformer(
                ensemble.GetConformer(conformer_id), assignId=True
            )
            for conformer_id in selected
        ]
        meta_out.update({
            "status": "template_success",
            "scene": "A",
            "guided_conformer_count": len(output_ids),
            "fallback_conformer_count": 0,
            "template_audit": {
                "selected_template_keys": [
                    f"template_{index}" for index in range(len(output_ids))
                ]
            },
        })
        return result, output_ids

    def unexpected_embed(*_args, **_kwargs):
        raise AssertionError(
            "matched-template rigidity must not create an ETKDG ensemble"
        )

    monkeypatch.setattr(
        template_library, "generate_conformers", fake_generate
    )
    monkeypatch.setattr(
        torsion_budget.AllChem, "EmbedMultipleConfs", unexpected_embed
    )
    audit = {}
    output = tmp_path / "template-ranked.pdbqt"
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        smiles,
        str(output),
        generated_map="KKKK{cyc:N-C}",
        n_template_confs=1,
        num_confs=1,
        random_seed=19,
        rigid_macrocycles=True,
        protonate=False,
        torsdof_limit=10,
        torsion_ensemble_size=4,
        torsion_num_threads=1,
        conf_out=audit,
    )

    assert error is None
    assert requested_counts == [1, 4]
    budget = audit["torsion_budget"]
    assert budget["status"] == "applied"
    assert budget["rigidity_reference_source"] == "matched_templates"
    assert budget["rigidity_evidence_level"] == "template_conditioned"
    assert budget["sigma_method"] == (
        "matched-template circular standard deviation"
    )
    assert budget["ensemble"]["template_keys"] == [
        "template_0",
        "template_1",
        "template_2",
        "template_3",
    ]
    assert budget["reference_selection"]["selected_conformer_count"] == 4
    assert output.is_file()


@V3_DIRECT_FLEX_RETIRED
def test_template_rigidity_fallback_is_labeled_model_generated(monkeypatch):
    from cycpep_master.docking import ligand_pdbqt, template_library

    smiles = "CCCC"
    prepared = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(prepared, randomSeed=7) == 0
    ensemble = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            ensemble, numConfs=3, randomSeed=11, numThreads=1
        )
    )
    assert len(conformer_ids) == 3

    def fake_generate(
        _smiles,
        _generated_map,
        n_conformers=3,
        meta_out=None,
        **_kwargs,
    ):
        assert n_conformers == 3
        meta_out.update({
            "status": "fallback_success",
            "scene": "B",
            "guided_conformer_count": 0,
            "fallback_conformer_count": 3,
            "template_audit": {"selected_template_keys": []},
        })
        return ensemble, conformer_ids

    monkeypatch.setattr(
        template_library, "generate_conformers", fake_generate
    )
    reference, audit, error = ligand_pdbqt._template_rigidity_reference(
        smiles,
        "AAAA{cyc:N-C}",
        molecule_3d=prepared,
        n_conformers=3,
        random_seed=11,
    )

    assert error is None
    assert reference.GetNumConformers() == 3
    assert reference.GetProp("CYCPEP_TORSION_ENSEMBLE_SOURCE") == (
        "etkdg_fallback"
    )
    assert audit["source"] == "etkdg_fallback"
    assert audit["evidence_level"] == "model_generated"
    assert audit["template_keys"] == []


@V3_DIRECT_FLEX_RETIRED
def test_template_rigidity_exception_uses_explicit_etkdg_fallback(
    monkeypatch,
):
    from cycpep_master.docking import ligand_pdbqt, template_library

    smiles = "CCCC"
    prepared = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(prepared, randomSeed=5) == 0
    fallback = Chem.AddHs(Chem.MolFromSmiles(smiles))
    fallback_ids = list(
        AllChem.EmbedMultipleConfs(
            fallback, numConfs=3, randomSeed=13, numThreads=1
        )
    )
    calls = []

    def fake_generate(
        _smiles,
        _generated_map,
        n_conformers=3,
        meta_out=None,
        template_strategy="full",
        **_kwargs,
    ):
        calls.append(template_strategy)
        if template_strategy != "off":
            raise RuntimeError("simulated template failure")
        meta_out.update({
            "status": "fallback_success",
            "scene": "B",
            "guided_conformer_count": 0,
            "fallback_conformer_count": len(fallback_ids),
        })
        return fallback, fallback_ids

    monkeypatch.setattr(
        template_library, "generate_conformers", fake_generate
    )
    reference, audit, error = ligand_pdbqt._template_rigidity_reference(
        smiles,
        "AAAA{cyc:N-C}",
        molecule_3d=prepared,
        n_conformers=3,
        random_seed=13,
    )

    assert error is None
    assert calls == ["full", "off"]
    assert reference.GetProp("CYCPEP_TORSION_ENSEMBLE_SOURCE") == (
        "etkdg_fallback"
    )
    assert audit["generation"]["template_attempt"]["status"] == "exception"
    assert audit["generation"]["fallback_attempt"]["status"] == (
        "fallback_success"
    )


def test_torsion_reference_rejects_nonbond_atom_pair():
    from cycpep_master.docking import torsion_budget

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            molecule, numConfs=2, randomSeed=23, numThreads=1
        )
    )
    assert len(conformer_ids) == 2
    molecule.SetProp(
        "CYCPEP_TORSION_ENSEMBLE_SOURCE", "matched_templates"
    )

    with pytest.raises(ValueError, match="not a molecular bond"):
        torsion_budget.bond_dihedral_sigma(
            molecule, [(0, 3)], metadata_out={}
        )


def test_ring_bond_is_not_assessable_for_selective_freezing():
    from cycpep_master.docking import torsion_budget

    molecule = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            molecule, numConfs=2, randomSeed=29, numThreads=1
        )
    )
    assert len(conformer_ids) == 2
    molecule.SetProp(
        "CYCPEP_TORSION_ENSEMBLE_SOURCE", "matched_templates"
    )
    metadata = {}
    sigma = torsion_budget.bond_dihedral_sigma(
        molecule, [(0, 1)], metadata_out=metadata
    )

    assert sigma[(0, 1)] == float("inf")
    assert metadata["bond_quartet_audit"][0]["reason"] == (
        "ring_bond_not_freezable"
    )


def test_all_quartets_handle_dihedral_wraparound_consistently():
    from cycpep_master.docking import torsion_budget

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=31) == 0
    original = Chem.Conformer(molecule.GetConformer(0))
    molecule.RemoveAllConformers()
    conformer_ids = [
        molecule.AddConformer(Chem.Conformer(original), assignId=True)
        for _ in range(2)
    ]
    for conformer_id, angle in zip(conformer_ids, (179.0, -179.0)):
        rdMolTransforms.SetDihedralDeg(
            molecule.GetConformer(conformer_id), 0, 1, 2, 3, angle
        )
    molecule.SetProp(
        "CYCPEP_TORSION_ENSEMBLE_SOURCE", "matched_templates"
    )
    metadata = {}
    sigma = torsion_budget.bond_dihedral_sigma(
        molecule, [(1, 2)], metadata_out=metadata
    )

    assert sigma[(1, 2)] < 2.0
    audit = metadata["bond_quartet_audit"][0]
    assert audit["quartet_count"] > 1
    assert audit["assessable"] is True
    assert audit["aggregation"] == "maximum_quartet_sigma"
    assert metadata["quartet_sigma_aggregation"] == (
        "maximum_quartet_sigma"
    )


@V3_DIRECT_FLEX_RETIRED
def test_unassessable_torsions_fail_closed_without_output(tmp_path, monkeypatch):
    output = tmp_path / "ligand.pdbqt"
    audit = {"stale": True}

    def all_unassessable(_mol, bonds, metadata_out=None, **_kwargs):
        if metadata_out is not None:
            metadata_out.update(
                {
                    "requested_conformer_count": 4,
                    "embedded_conformer_count": 4,
                    "random_seed": 42,
                    "num_threads": 1,
                }
            )
        return {bond: float("inf") for bond in bonds}

    monkeypatch.setattr(vina_wrapper, "_bond_dihedral_sigma", all_unassessable)
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        _cyclic_lysine_smiles(),
        str(output),
        num_confs=1,
        random_seed=42,
        rigid_macrocycles=True,
        protonate=False,
        torsdof_limit=10,
        torsion_ensemble_size=4,
        torsion_num_threads=1,
        conf_out=audit,
    )
    assert "insufficient assessable torsions" in error
    assert not output.exists()
    assert audit == {}


@V3_DIRECT_FLEX_RETIRED
def test_budgeted_pdbqt_is_deterministic_for_fixed_seeds(tmp_path):
    smiles = _cyclic_lysine_smiles()
    outputs = []
    for name in ("first.pdbqt", "second.pdbqt"):
        path = tmp_path / name
        error = vina_wrapper.smiles_to_ligand_pdbqt(
            smiles,
            str(path),
            num_confs=1,
            random_seed=17,
            rigid_macrocycles=True,
            protonate=False,
            torsdof_limit=10,
            torsion_ensemble_size=3,
            torsion_num_threads=1,
        )
        assert error is None
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]


def test_dock_peptide_requires_explicit_smiles_preparation_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        vina_wrapper,
        "pdb_to_pdbqt_meeko",
        lambda *_args, **_kwargs: calls.append("pdb") or None,
    )
    affinity, error = vina_wrapper.dock_peptide(
        "ligand.pdb",
        "receptor.pdb",
        (0.0, 0.0, 0.0),
        ligand_smiles="CC",
    )
    assert affinity is None
    assert "require ligand_preparation_mode='audited_smiles'" in error
    assert calls == []


def test_dock_peptide_propagates_adaptive_budget_for_provided_smiles(
    tmp_path, monkeypatch
):
    calls = {}

    def fake_smiles(smiles, pdbqt_path, **kwargs):
        calls["smiles"] = smiles
        calls["pdbqt_path"] = pdbqt_path
        calls["kwargs"] = kwargs
        kwargs["conf_out"].update({
            "source": "template",
            "torsion_budget": {"status": "applied", "final_torsdof": 12},
            "pdbqt_tree": {"torsdof": 12},
        })
        Path(pdbqt_path).write_text("mock ligand\n", encoding="utf-8")
        return None

    def fake_pdb(_source, _destination, is_receptor=False):
        assert is_receptor
        calls["receptor"] = True
        Path(_destination).write_text("mock receptor\n", encoding="utf-8")
        return None

    monkeypatch.setattr(vina_wrapper, "smiles_to_ligand_pdbqt", fake_smiles)
    monkeypatch.setattr(vina_wrapper, "pdb_to_pdbqt_meeko", fake_pdb)
    monkeypatch.setattr(
        vina_wrapper,
        "run_vina",
        lambda **kwargs: calls.setdefault("vina", kwargs) and (-7.25, None),
    )
    audit = {"stale": True}
    affinity, error = vina_wrapper.dock_peptide(
        "ligand.pdb",
        "receptor.pdb",
        (1.0, 2.0, 3.0),
        output_dir=str(tmp_path),
        cleanup=False,
        ligand_preparation_mode="audited_smiles",
        ligand_smiles="NCC(=O)O",
        generated_map="G",
        torsdof_limit=12,
        torsion_ensemble_size=6,
        torsion_num_threads=2,
        torsion_audit_out=audit,
    )
    assert error is None
    assert affinity == -7.25
    assert calls["smiles"] == "NCC(=O)O"
    assert calls["kwargs"]["generated_map"] == "G"
    assert calls["kwargs"]["torsdof_limit"] == 12
    assert calls["kwargs"]["torsion_ensemble_size"] == 6
    assert calls["kwargs"]["torsion_num_threads"] == 2
    assert calls["receptor"] is True
    assert audit == {
        "status": "success",
        "preparation_mode": "audited_smiles",
        "chemical_graph_source": "provided_smiles",
        "original_coordinates_preserved": False,
        "generated_map_supplied": True,
        "torsion_budget": {"status": "applied", "final_torsdof": 12},
        "pdbqt_tree": {"torsdof": 12},
        "conformer_source": "template",
    }


def test_dock_peptide_can_require_qualified_v6_graph(tmp_path, monkeypatch):
    reconstruction = SimpleNamespace(
        status="success",
        support_status="supported",
        qualified_success=True,
        output_smiles="C1CCCCC1",
        output_inchikey="FULL-INCHI-KEY",
        path_used="V6_EVIDENCE_DIMENSION_AUDIT",
        repair_codes=["EXPLICIT_CONNECTION_RECOVERED"],
        warning_codes=[],
        rejection_reason=None,
    )
    reconstructed = {}

    def fake_reconstruct(path, chain):
        reconstructed.update({"path": path, "chain": chain})
        return reconstruction

    def fake_smiles(smiles, _pdbqt_path, **kwargs):
        reconstructed["smiles"] = smiles
        kwargs["conf_out"].update({
            "source": "embed3d",
            "torsion_budget": {"status": "not_needed"},
            "pdbqt_tree": {"torsdof": 3},
        })
        Path(_pdbqt_path).write_text("mock ligand\n", encoding="utf-8")
        return None

    from cycpep_master import remediation_v6

    monkeypatch.setattr(
        remediation_v6, "reconstruct_structure_fail_closed_v6", fake_reconstruct
    )
    monkeypatch.setattr(vina_wrapper, "smiles_to_ligand_pdbqt", fake_smiles)
    def fake_pdb(_source, destination, **_kwargs):
        Path(destination).write_text("mock receptor\n", encoding="utf-8")
        return None

    monkeypatch.setattr(vina_wrapper, "pdb_to_pdbqt_meeko", fake_pdb)
    monkeypatch.setattr(vina_wrapper, "run_vina", lambda **_kwargs: (-4.0, None))
    audit = {}
    affinity, error = vina_wrapper.dock_peptide(
        "ligand.cif",
        "receptor.pdb",
        (0.0, 0.0, 0.0),
        output_dir=str(tmp_path),
        cleanup=False,
        ligand_preparation_mode="audited_smiles",
        peptide_chain_id="PEPTIDE_A",
        torsion_audit_out=audit,
    )
    assert (affinity, error) == (-4.0, None)
    assert reconstructed == {
        "path": "ligand.cif",
        "chain": "PEPTIDE_A",
        "smiles": "C1CCCCC1",
    }
    assert audit["chemical_graph_source"] == "v6_reconstruction"
    assert audit["v6_reconstruction"]["qualified_success"] is True
    assert audit["v6_reconstruction"]["output_inchikey"] == "FULL-INCHI-KEY"


def test_dock_peptide_rejects_unqualified_v6_before_pdbqt(tmp_path, monkeypatch):
    reconstruction = SimpleNamespace(
        status="rejected",
        support_status="supported",
        qualified_success=False,
        output_smiles=None,
        output_inchikey=None,
        path_used="V6_EVIDENCE_DIMENSION_AUDIT",
        repair_codes=[],
        warning_codes=["V6_TOPOLOGY_CONFLICT"],
        rejection_reason="explicit connections conflict",
    )
    from cycpep_master import remediation_v6

    monkeypatch.setattr(
        remediation_v6,
        "reconstruct_structure_fail_closed_v6",
        lambda *_args, **_kwargs: reconstruction,
    )

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("PDBQT preparation must not run after V6 rejection")

    monkeypatch.setattr(vina_wrapper, "smiles_to_ligand_pdbqt", should_not_run)
    monkeypatch.setattr(vina_wrapper, "pdb_to_pdbqt_meeko", should_not_run)
    audit = {}
    affinity, error = vina_wrapper.dock_peptide(
        "ligand.pdb",
        "receptor.pdb",
        (0.0, 0.0, 0.0),
        output_dir=str(tmp_path),
        cleanup=False,
        ligand_preparation_mode="audited_smiles",
        torsion_audit_out=audit,
    )
    assert affinity is None
    assert "was not a qualified success" in error
    assert "explicit connections conflict" in error
    assert audit["status"] == "rejected"
    assert audit["v6_reconstruction"]["warning_codes"] == [
        "V6_TOPOLOGY_CONFLICT"
    ]
