from __future__ import annotations

from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.docking.mol2_input import (
    Mol2ValidationError,
    default_receipt_path,
    load_validated_mol2,
    write_validation_receipt,
)
from cycpep_master.export.conformer import mol_to_mol2
from cycpep_master.docking.torsion_prior import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    TorsionPriorError,
    TorsionPriorMatch,
    build_query_keys,
    load_torsion_prior,
)
from cycpep_master.docking.mol2_pdbqt import (
    _prepare_parent_molecule,
    baseline_output_path,
    mol2_to_ligand_pdbqt,
)
from cycpep_master import application
from cycpep_master.docking.build_torsion_priors import (
    _circular_statistics,
    _compile_level,
)


def _validated_mol2(
    tmp_path: Path,
    smiles: str = "[NH3+]CC(=O)[O-]",
) -> tuple[Path, Path]:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=17) == 0
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetProp("_TriposAtomName", f"{atom.GetSymbol()}{index}")
        atom.SetProp("_TriposResidueName", "GLY")
        atom.SetProp("_TriposChainId", "L")
        atom.SetIntProp("_TriposResidueNumber", 1)
        atom.SetProp("_TriposInsertionCode", "")
    output = tmp_path / "parent.mol2"
    produced, error = mol_to_mol2(molecule, output_path=str(output))
    assert error is None and Path(produced) == output
    receipt = write_validation_receipt(
        output,
        coordinate_mode="source_bound",
        rigor="L2:Q",
        quality="exact",
        source_heavy_atom_mapping_complete=True,
        atom_provenance_complete=True,
        source_input_sha256="a" * 64,
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
        max_source_coordinate_delta_angstrom=0.0,
        evidence_manifest_sha256="b" * 64,
    )
    return output, receipt


def test_validated_mol2_restores_identity_coordinates_and_metadata(
    tmp_path,
):
    output, receipt = _validated_mol2(tmp_path)

    validated = load_validated_mol2(output)

    assert validated.receipt_path == receipt
    assert validated.coordinate_mode == "source_bound"
    assert validated.rigor == "L2:Q"
    assert validated.quality == "exact"
    assert validated.full_inchikey == Chem.MolToInchiKey(
        validated.molecule
    )
    assert validated.molecule.GetNumConformers() == 1
    assert validated.molecule.GetAtomWithIdx(0).GetProp(
        "_TriposAtomName"
    )
    assert validated.molecule.GetAtomWithIdx(0).GetProp(
        "_TriposChainId"
    ) == "L"


def test_parent_preparation_completes_partial_explicit_hydrogens(
    tmp_path,
):
    output, _receipt = _validated_mol2(tmp_path)
    validated = load_validated_mol2(output)
    partial = Chem.AddHs(
        Chem.MolFromSmiles("CCO"), onlyOnAtoms=[0]
    )
    assert AllChem.EmbedMolecule(partial, randomSeed=17) == 0
    parent = replace(
        validated,
        molecule=partial,
        full_inchikey=Chem.MolToInchiKey(partial),
        atom_count=partial.GetNumAtoms(),
        heavy_atom_count=partial.GetNumHeavyAtoms(),
        formal_charge=int(Chem.GetFormalCharge(partial)),
    )

    prepared, audit = _prepare_parent_molecule(parent)

    assert audit["generated_hydrogen_count"] == 3
    assert prepared.GetNumHeavyAtoms() == partial.GetNumHeavyAtoms()
    assert audit["heavy_atom_invariants_valid"] is True


def test_mol2_without_validation_receipt_cannot_be_bypassed(tmp_path):
    output, receipt = _validated_mol2(tmp_path)
    receipt.unlink()

    with pytest.raises(
        Mol2ValidationError, match="no SMILES/PDB bypass"
    ):
        load_validated_mol2(output)


def test_mol2_hash_drift_is_rejected(tmp_path):
    output, _receipt = _validated_mol2(tmp_path)
    output.write_text(
        output.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Mol2ValidationError, match="SHA-256"):
        load_validated_mol2(output)


def test_mol2_receipt_identity_drift_is_rejected(tmp_path):
    output, receipt = _validated_mol2(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["full_inchikey"] = "INVALID"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        Mol2ValidationError, match="full_inchikey"
    ):
        load_validated_mol2(output)


def test_default_receipt_path_is_adjacent_to_parent(tmp_path):
    parent = tmp_path / "ligand.mol2"
    assert default_receipt_path(parent) == Path(
        str(parent) + ".validation.json"
    )


def _prior_files(tmp_path, keys, *, exact=None):
    runtime = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "levels": {
            "exact": ({keys.exact: exact} if exact else {}),
            "residue_class_ring": {},
            "morgan": {},
            "generic": {},
        },
    }
    runtime_path = tmp_path / "torsion_priors_runtime.json"
    runtime_path.write_text(
        json.dumps(runtime, sort_keys=True), encoding="utf-8"
    )
    digest = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "torsion_prior_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "runtime_sha256": digest,
        }),
        encoding="utf-8",
    )
    return runtime_path, manifest_path


def test_torsion_prior_exact_match_can_freeze_high_confidence_bond(
    tmp_path,
):
    parent, _receipt = _validated_mol2(tmp_path)
    validated = load_validated_mol2(parent)
    bond = next(
        bond
        for bond in validated.molecule.GetBonds()
        if not bond.IsInRing()
        and bond.GetBeginAtom().GetAtomicNum() > 1
        and bond.GetEndAtom().GetAtomicNum() > 1
    )
    pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    keys = build_query_keys(
        validated.molecule,
        pair,
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
    )
    runtime, manifest = _prior_files(
        tmp_path,
        keys,
        exact={
            "confidence": "high",
            "rigidity_score": 0.91,
            "calibration_unit": "structure",
            "calibration_false_rigid_count": 1,
            "calibration_evaluable_count": 100,
            "calibration_false_rigid_ci_high": 0.04,
            "leave_one_entity_out_false_rigid_ci_high": 0.04,
        },
    )

    index = load_torsion_prior(runtime, manifest_path=manifest)
    match = index.query(keys, flexibility_mode="balanced")

    assert match.lookup_level == "exact"
    assert match.eligible_to_freeze is True
    assert match.rigidity_score == pytest.approx(0.91)


def test_torsion_prior_requires_false_rigid_calibration_to_freeze(
    tmp_path,
):
    parent, _receipt = _validated_mol2(tmp_path)
    validated = load_validated_mol2(parent)
    bond = next(iter(validated.molecule.GetBonds()))
    keys = build_query_keys(
        validated.molecule,
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
    )
    runtime, manifest = _prior_files(
        tmp_path,
        keys,
        exact={"confidence": "high", "rigidity_score": 1.0},
    )

    match = load_torsion_prior(
        runtime, manifest_path=manifest
    ).query(keys)

    assert match.status == "matched"
    assert match.eligible_to_freeze is False


def test_torsion_prior_zero_denominator_never_freezes(tmp_path):
    parent, _receipt = _validated_mol2(tmp_path)
    validated = load_validated_mol2(parent)
    bond = next(iter(validated.molecule.GetBonds()))
    keys = build_query_keys(
        validated.molecule,
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
    )
    runtime, manifest = _prior_files(
        tmp_path,
        keys,
        exact={
            "confidence": "high",
            "rigidity_score": 1.0,
            "calibration_unit": "structure",
            "calibration_false_rigid_count": 0,
            "calibration_evaluable_count": 0,
            "calibration_false_rigid_ci_high": 0.0,
        },
    )

    match = load_torsion_prior(
        runtime, manifest_path=manifest
    ).query(keys)

    assert match.status == "matched"
    assert match.eligible_to_freeze is False


def test_low_confidence_prior_never_freezes(tmp_path):
    parent, _receipt = _validated_mol2(tmp_path)
    validated = load_validated_mol2(parent)
    bond = next(iter(validated.molecule.GetBonds()))
    keys = build_query_keys(
        validated.molecule,
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
    )
    runtime, manifest = _prior_files(
        tmp_path,
        keys,
        exact={"confidence": "low", "rigidity_score": 1.0},
    )

    match = load_torsion_prior(
        runtime, manifest_path=manifest
    ).query(keys)

    assert match.status == "matched"
    assert match.eligible_to_freeze is False


def test_unknown_prior_keeps_bond_flexible(tmp_path):
    parent, _receipt = _validated_mol2(tmp_path)
    validated = load_validated_mol2(parent)
    bond = next(iter(validated.molecule.GetBonds()))
    keys = build_query_keys(
        validated.molecule,
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
    )
    runtime, manifest = _prior_files(tmp_path, keys)

    match = load_torsion_prior(
        runtime, manifest_path=manifest
    ).query(keys)

    assert match.status == "unavailable"
    assert match.eligible_to_freeze is False


def test_torsion_prior_hash_drift_is_rejected(tmp_path):
    parent, _receipt = _validated_mol2(tmp_path)
    validated = load_validated_mol2(parent)
    bond = next(iter(validated.molecule.GetBonds()))
    keys = build_query_keys(
        validated.molecule,
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
        topology_class="head_to_tail",
        macrocycle_ring_size=8,
    )
    runtime, manifest = _prior_files(tmp_path, keys)
    runtime.write_text(
        runtime.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TorsionPriorError, match="SHA-256"):
        load_torsion_prior(runtime, manifest_path=manifest)


def test_torsion_query_keys_ignore_explicit_hydrogen_index_expansion():
    molecule = Chem.MolFromSmiles("CCCC")
    with_hydrogens = Chem.AddHs(molecule)
    pair = (1, 2)

    heavy_keys = build_query_keys(
        molecule,
        pair,
        topology_class="linear",
        macrocycle_ring_size=None,
    )
    explicit_h_keys = build_query_keys(
        with_hydrogens,
        pair,
        topology_class="linear",
        macrocycle_ring_size=None,
    )

    assert explicit_h_keys == heavy_keys


def test_torsion_query_keys_are_invariant_to_atom_order():
    molecule = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")
    order = list(reversed(range(molecule.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(molecule, order)
    new_index = {
        old_index: index for index, old_index in enumerate(order)
    }

    original = build_query_keys(
        molecule,
        (0, 1),
        topology_class="linear",
        macrocycle_ring_size=None,
    )
    reordered = build_query_keys(
        renumbered,
        (new_index[0], new_index[1]),
        topology_class="linear",
        macrocycle_ring_size=None,
    )

    assert reordered == original


class _UnavailablePrior:
    runtime_sha256 = "a" * 64
    manifest_sha256 = "b" * 64

    def query(self, _keys, *, flexibility_mode="balanced"):
        return TorsionPriorMatch(
            status="unavailable",
            lookup_level=None,
            key=None,
            rigidity_score=None,
            confidence="unavailable",
            eligible_to_freeze=False,
            statistics=None,
            reason="test prior unavailable",
        )


class _RigidPrior:
    runtime_sha256 = "c" * 64
    manifest_sha256 = "d" * 64

    def query(self, _keys, *, flexibility_mode="balanced"):
        return TorsionPriorMatch(
            status="matched",
            lookup_level="generic",
            key="test",
            rigidity_score=0.99,
            confidence="high",
            eligible_to_freeze=True,
            statistics={"n_entities": 100},
            reason=None,
        )


def test_mol2_pdbqt_never_bypasses_missing_receipt(tmp_path):
    parent, receipt = _validated_mol2(tmp_path, "CCCC")
    receipt.unlink()
    output = tmp_path / "ligand.pdbqt"

    audit, error = mol2_to_ligand_pdbqt(parent, output)

    assert audit is None
    assert "validated MOL2 required" in error
    assert not output.exists()


def test_torsdof_below_limit_skips_prior_and_ensemble(tmp_path):
    pytest.importorskip("meeko")
    parent, _receipt = _validated_mol2(tmp_path, "CCCC")
    output = tmp_path / "ligand.pdbqt"
    calls = {"prior": 0, "embed": 0}

    def unexpected_prior(*_args, **_kwargs):
        calls["prior"] += 1
        raise AssertionError("prior must not be loaded")

    def unexpected_embed(*_args, **_kwargs):
        calls["embed"] += 1
        raise AssertionError("ensemble must not be generated")

    audit, error = mol2_to_ligand_pdbqt(
        parent,
        output,
        torsdof_limit=100,
        load_prior=unexpected_prior,
        embed_multiple=unexpected_embed,
    )

    assert error is None
    assert audit["budget_satisfied"] is True
    assert audit["ensemble_fallback_triggered"] is False
    assert calls == {"prior": 0, "embed": 0}
    assert output.is_file()


def test_lookup_sufficient_skips_ensemble(tmp_path):
    pytest.importorskip("meeko")
    parent, _receipt = _validated_mol2(tmp_path, "CCCCCC")
    output = tmp_path / "lookup.pdbqt"
    calls = {"embed": 0}

    def unexpected_embed(*_args, **_kwargs):
        calls["embed"] += 1
        raise AssertionError("lookup-sufficient path must not embed")

    audit, error = mol2_to_ligand_pdbqt(
        parent,
        output,
        torsdof_limit=0,
        flexibility_mode="balanced",
        load_prior=lambda _path: _RigidPrior(),
        embed_multiple=unexpected_embed,
    )

    assert error is None
    assert audit["budget_satisfied"] is True
    assert audit["lookup_covered_bond_count"] > 0
    assert audit["ensemble_fallback_triggered"] is False
    assert calls["embed"] == 0
    assert audit["final_torsdof"] == 0


def test_balanced_without_manifest_downgrades_without_embedding(
    tmp_path,
):
    pytest.importorskip("meeko")
    parent, _receipt = _validated_mol2(tmp_path, "CCCCCC")
    output = tmp_path / "fallback.pdbqt"
    calls = []
    def counted_embed(_molecule, *, numConfs, params):
        calls.append(numConfs)
        raise AssertionError("the PDBQT layer must not generate conformers")

    audit, error = mol2_to_ligand_pdbqt(
        parent,
        output,
        torsdof_limit=0,
        flexibility_mode="balanced",
        ensemble_size=4,
        load_prior=lambda _path: _UnavailablePrior(),
        embed_multiple=counted_embed,
    )

    assert error is None
    assert calls == []
    assert audit["requested_flexibility_mode"] == "balanced"
    assert audit["effective_flexibility_mode"] == "fast"
    assert audit["ensemble_fallback_triggered"] is False
    assert audit["ensemble"]["status"] == "unavailable"
    assert audit["embedding_performed"] is False
    assert (
        "BALANCED_DOWNGRADED_TO_FAST_NO_ENSEMBLE"
        in audit["flexibility_warning_codes"]
    )
    assert audit["budget_satisfied"] is False
    assert audit["output_role"] == "baseline_budget_unsatisfied"
    assert audit["atom_invariants_valid"] is True


def test_failed_flexibility_keeps_baseline_for_strict_budget(
    tmp_path,
):
    pytest.importorskip("meeko")
    parent, _receipt = _validated_mol2(tmp_path, "CCCCCC")
    output = tmp_path / "strict.pdbqt"
    calls = []

    def failed_embed(_molecule, *, numConfs, params):
        calls.append(numConfs)
        raise AssertionError("the PDBQT layer must not generate conformers")

    audit, error = mol2_to_ligand_pdbqt(
        parent,
        output,
        torsdof_limit=0,
        flexibility_mode="balanced",
        strict_budget=True,
        load_prior=lambda _path: _UnavailablePrior(),
        embed_multiple=failed_embed,
    )

    baseline = baseline_output_path(output)
    assert error.startswith("not_supported:")
    assert calls == []
    assert audit["effective_flexibility_mode"] == "fast"
    assert audit["embedding_performed"] is False
    assert audit["budget_satisfied"] is False
    assert baseline.is_file()
    assert audit["baseline_pdbqt_path"] == str(baseline)
    assert not output.exists()


def test_failed_flexibility_returns_baseline_when_not_strict(
    tmp_path,
):
    pytest.importorskip("meeko")
    parent, _receipt = _validated_mol2(tmp_path, "CCCCCC")
    output = tmp_path / "nonstrict.pdbqt"

    audit, error = mol2_to_ligand_pdbqt(
        parent,
        output,
        torsdof_limit=0,
        flexibility_mode="balanced",
        strict_budget=False,
        load_prior=lambda _path: _UnavailablePrior(),
        embed_multiple=lambda *_args, **_kwargs: [],
    )

    assert error is None
    assert audit["budget_satisfied"] is False
    assert audit["output_role"] == "baseline_budget_unsatisfied"
    assert output.is_file()
    assert baseline_output_path(output).is_file()


def test_application_mol2_entrypoint_loads_parent_once(
    tmp_path, monkeypatch
):
    pytest.importorskip("meeko")
    from cycpep_master.docking import mol2_input

    parent, receipt = _validated_mol2(tmp_path, "CCCC")
    original = mol2_input.load_validated_mol2
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(mol2_input, "load_validated_mol2", counted)

    result = application.prepare_ligand_pdbqt_from_mol2(
        parent,
        tmp_path / "single-load.pdbqt",
        receipt_path=receipt,
        torsdof_limit=100,
    )

    assert result["status"] == "success"
    assert len(calls) == 1


def test_application_mol2_entrypoint_inherits_parent_evidence(tmp_path):
    pytest.importorskip("meeko")
    parent, receipt = _validated_mol2(tmp_path, "CCCC")
    output = tmp_path / "application.pdbqt"

    result = application.prepare_ligand_pdbqt_from_mol2(
        parent,
        output,
        receipt_path=receipt,
        torsdof_limit=100,
    )

    assert result["status"] == "success"
    assert result["data"]["parent_mol2_sha256"]
    assert result["data"]["inherited_rigor"] == "L2:Q"
    assert result["data"]["inherited_quality"] == "exact"
    assert result["data"]["parent_coordinate_mode"] == "source_bound"
    assert result["data"]["requested_format_status"] == "fulfilled"
    assert result["data"]["artifacts"][0]["rigor"] == "L2:Q"


def test_application_mol2_entrypoint_reports_missing_receipt(tmp_path):
    parent, receipt = _validated_mol2(tmp_path, "CCCC")
    receipt.unlink()

    result = application.prepare_ligand_pdbqt_from_mol2(
        parent, tmp_path / "missing.pdbqt"
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format"] == "pdbqt"
    assert result["data"]["requested_format_status"] == "unavailable"
    assert result["data"]["qualification_status"] == "unqualified"
    assert "validated MOL2 required" in result["data"]["reason"]
    assert result["data"]["artifacts"][0]["format"] == "mol2"


def test_application_mol2_entrypoint_rejects_receipt_hash_drift(
    tmp_path,
):
    parent, receipt = _validated_mol2(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["mol2_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    result = application.prepare_ligand_pdbqt_from_mol2(
        parent, tmp_path / "ligand.pdbqt"
    )

    assert result["status"] == "success"
    assert result["data"]["requested_format_status"] == "unavailable"
    assert result["data"]["qualification_status"] == "unqualified"
    assert "SHA-256" in result["data"]["reason"]
    assert result["data"]["artifacts"][0]["format"] == "mol2"


def test_source_bound_receipt_rejects_incomplete_mapping(tmp_path):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=3) == 0
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetProp("_TriposAtomName", f"{atom.GetSymbol()}{index}")
        atom.SetProp("_TriposResidueName", "LIG")
        atom.SetProp("_TriposChainId", "L")
        atom.SetIntProp("_TriposResidueNumber", 1)
        atom.SetProp("_TriposInsertionCode", "")
    output = tmp_path / "incomplete.mol2"
    produced, error = mol_to_mol2(molecule, output_path=str(output))
    assert error is None and produced

    with pytest.raises(
        Mol2ValidationError, match="complete source heavy-atom mapping"
    ):
        write_validation_receipt(
            output,
            coordinate_mode="source_bound",
            rigor="L2:Q",
            quality="exact",
            source_heavy_atom_mapping_complete=False,
            atom_provenance_complete=True,
        )


def test_thorough_without_manifest_downgrades_to_fast(tmp_path):
    parent, _receipt = _validated_mol2(tmp_path, "CCCC")

    audit, error = mol2_to_ligand_pdbqt(
        parent,
        tmp_path / "thorough.pdbqt",
        torsdof_limit=0,
        flexibility_mode="thorough",
        ensemble_size=4,
    )

    assert error is None
    assert audit["requested_flexibility_mode"] == "thorough"
    assert audit["effective_flexibility_mode"] == "fast"
    assert audit["embedding_performed"] is False
    assert audit["ensemble_fallback_triggered"] is False
    assert (
        "THOROUGH_DOWNGRADED_TO_FAST_NO_ENSEMBLE"
        in audit["flexibility_warning_codes"]
    )


def test_mixed_lookup_without_manifest_does_not_evaluate_sigma(
    tmp_path, monkeypatch
):
    pytest.importorskip("meeko")
    from cycpep_master.docking import mol2_pdbqt as module

    parent, _receipt = _validated_mol2(tmp_path, "CCCCCC")
    output = tmp_path / "mixed.pdbqt"
    calls = {"query": 0, "sigma_bonds": None}

    class MixedPrior:
        runtime_sha256 = "1" * 64
        manifest_sha256 = "2" * 64

        def query(self, _keys, *, flexibility_mode="balanced"):
            calls["query"] += 1
            if calls["query"] == 1:
                return _RigidPrior().query(_keys)
            return _UnavailablePrior().query(_keys)

    def zero_sigma(_ensemble, bonds, **_kwargs):
        calls["sigma_bonds"] = list(bonds)
        return {bond: 0.0 for bond in bonds}

    monkeypatch.setattr(module, "bond_dihedral_sigma", zero_sigma)
    audit, error = mol2_to_ligand_pdbqt(
        parent,
        output,
        torsdof_limit=0,
        load_prior=lambda _path: MixedPrior(),
    )

    assert error is None
    assert audit["lookup_covered_bond_count"] == 1
    assert audit["lookup_unresolved_bond_count"] == calls["query"] - 1
    lookup_frozen = {
        tuple(row["atom_indices"])
        for row in audit["frozen_bonds"]
        if row["phase"] == "lookup"
    }
    assert lookup_frozen
    assert calls["sigma_bonds"] is None
    assert audit["effective_flexibility_mode"] == "fast"
    assert audit["embedding_performed"] is False


def test_v4_output_is_deterministic_for_fixed_parent_prior_and_seed(
    tmp_path,
):
    pytest.importorskip("meeko")
    parent, _receipt = _validated_mol2(tmp_path, "CCCCCC")
    outputs = []
    for name in ("first.pdbqt", "second.pdbqt"):
        output = tmp_path / name
        audit, error = mol2_to_ligand_pdbqt(
            parent,
            output,
            torsdof_limit=0,
            load_prior=lambda _path: _RigidPrior(),
            random_seed=42,
        )
        assert error is None
        assert audit["budget_satisfied"] is True
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]


def test_application_smiles_convenience_materializes_validated_parent(
    tmp_path, monkeypatch
):
    pytest.importorskip("meeko")
    from cycpep_master.docking import ligand_pdbqt

    def forbidden_direct_path(*_args, **_kwargs):
        raise AssertionError("V4 convenience path bypassed parent MOL2")

    monkeypatch.setattr(
        ligand_pdbqt,
        "smiles_to_ligand_pdbqt",
        forbidden_direct_path,
    )
    output = tmp_path / "smiles.pdbqt"
    result = application.prepare_ligand_pdbqt(
        "CCCC",
        output,
        num_confs=1,
        protonate=False,
        torsdof_limit=100,
    )

    parent = Path(str(output) + ".parent.mol2")
    receipt = default_receipt_path(parent)
    assert result["status"] == "success"
    assert parent.is_file()
    assert receipt.is_file()
    assert result["data"]["parent_mol2_path"] == str(parent)
    assert result["data"]["coordinate_materializer_contract"] == (
        "legacy_smiles_compatibility"
    )
    assert (
        "LEGACY_SMILES_MOL2_PATH_WITHOUT_V5_ENSEMBLE_QA"
        in result["data"]["warning_codes"]
    )
    validated = load_validated_mol2(parent)
    assert validated.coordinate_mode == "regenerated"
    assert validated.rigor == "L2:H"


def test_application_export_mol2_writes_validation_receipt(tmp_path):
    output = tmp_path / "exported.mol2"

    result = application.export_structure(
        "CCCC",
        output,
        source_kind="smiles",
        output_format="mol2",
        num_confs=1,
    )

    assert result["status"] == "success"
    receipt = Path(result["data"]["validation_receipt_path"])
    assert receipt.is_file()
    assert application.validate_mol2(
        output, receipt_path=receipt
    )["status"] == "success"


def test_application_pdb_convenience_routes_through_parent_mol2(
    tmp_path, monkeypatch
):
    coordinate = tmp_path / "input.pdb"
    coordinate.write_text("END\n", encoding="ascii")
    output = tmp_path / "coordinate.pdbqt"
    captured = {}

    def fake_export(_source, parent_path, **_kwargs):
        exported = application.export_structure(
            "CCCC",
            parent_path,
            source_kind="smiles",
            output_format="mol2",
            num_confs=1,
        )
        return {
            "operation": "export_best_available",
            "status": "success",
            "data": {
                **exported["data"],
                "requested_format_status": "fulfilled",
                "artifacts": [],
            },
        }

    def fake_prepare(parent_path, destination, **kwargs):
        captured.update({
            "parent_path": str(parent_path),
            "destination": str(destination),
            **kwargs,
        })
        Path(destination).write_text("PDBQT\n", encoding="ascii")
        return {
            "operation": "prepare_ligand_pdbqt_from_mol2",
            "status": "success",
            "data": {
                "output_path": str(destination),
                "parent_coordinate_mode": "source_bound",
                "artifacts": [],
            },
        }

    monkeypatch.setattr(application, "export_best_available", fake_export)
    monkeypatch.setattr(
        application, "prepare_ligand_pdbqt_from_mol2", fake_prepare
    )
    monkeypatch.setattr(
        application,
        "reconstruct_structure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("PDB convenience path reconstructed after MOL2")
        ),
    )

    result = application.prepare_ligand_pdbqt_from_pdb(
        coordinate, output
    )

    assert result["status"] == "success"
    assert captured["parent_path"] == str(
        Path(str(output) + ".parent.mol2")
    )
    assert captured["receipt_path"].endswith(".validation.json")


def test_generated_map_guides_parent_mol2_not_flexibility(
    tmp_path, monkeypatch
):
    pytest.importorskip("meeko")
    from cycpep_master.docking import template_library

    smiles = "CCCC"
    guided = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(guided, randomSeed=21) == 0

    def fake_templates(
        _smiles,
        _map,
        n_conformers=1,
        meta_out=None,
        **_kwargs,
    ):
        meta_out.update({
            "status": "template_success",
            "scene": "A",
            "guided_conformer_count": 1,
            "fallback_conformer_count": 0,
        })
        return guided, [0]

    captured = {}

    def fake_standard(parent_path, output_path, **kwargs):
        parent = load_validated_mol2(
            parent_path, receipt_path=kwargs["receipt_path"]
        )
        captured["coordinate_mode"] = parent.coordinate_mode
        captured["coordinate_level"] = parent.coordinate_level
        captured["kwargs"] = kwargs
        Path(output_path).write_text("PDBQT\n", encoding="ascii")
        return {
            "operation": "prepare_ligand_pdbqt_from_mol2",
            "status": "success",
            "data": {
                "output_path": str(output_path),
                "audit": {},
                "artifacts": [],
            },
        }

    monkeypatch.setattr(
        template_library, "generate_conformers", fake_templates
    )
    monkeypatch.setattr(
        application, "prepare_ligand_pdbqt_from_mol2", fake_standard
    )
    result = application.prepare_ligand_pdbqt(
        smiles,
        tmp_path / "guided.pdbqt",
        generated_map="AAAA",
        num_confs=1,
        protonate=False,
        torsdof_limit=10,
    )

    assert result["status"] == "success"
    # Template guidance is a generation method, not experimental source
    # coordinates: the parent receipt reports regenerated/X1 and must not
    # claim the X2 source-completion semantics.
    assert captured["coordinate_mode"] == "regenerated"
    assert captured["coordinate_level"] == "X1"
    assert "generated_map" not in captured["kwargs"]


def test_budget_serialization_failure_retains_committed_baseline(
    tmp_path, monkeypatch
):
    pytest.importorskip("meeko")
    from cycpep_master.docking import mol2_pdbqt as module

    parent, _receipt = _validated_mol2(tmp_path, "CCCCCC")
    output = tmp_path / "writer-failure.pdbqt"
    real_serialize = module._serialize_setup
    calls = {"count": 0}

    def fail_second(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated budget writer failure")
        return real_serialize(*args, **kwargs)

    monkeypatch.setattr(module, "_serialize_setup", fail_second)
    audit, error = mol2_to_ligand_pdbqt(
        parent,
        output,
        torsdof_limit=0,
        load_prior=lambda _path: _RigidPrior(),
    )

    assert error is None
    assert "simulated budget writer failure" in audit[
        "flexibility_error"
    ]
    assert calls["count"] == 2
    assert baseline_output_path(output).is_file()
    assert audit["baseline_pdbqt_path"] == str(
        baseline_output_path(output)
    )
    assert output.is_file()
    assert audit["output_role"] == "baseline_flexibility_failed"
    assert audit["budget_satisfied"] is False


def test_failed_reexport_removes_stale_validation_receipt(
    tmp_path, monkeypatch
):
    import cycpep_master.export as export_module

    output = tmp_path / "stale.mol2"
    first = application.export_structure(
        "CCCC",
        output,
        source_kind="smiles",
        output_format="mol2",
        num_confs=1,
    )
    receipt = Path(first["data"]["validation_receipt_path"])
    assert receipt.is_file()
    monkeypatch.setattr(
        export_module,
        "smiles_to_mol2",
        lambda *_args, **_kwargs: (
            None,
            "failed: simulated export failure",
        ),
    )

    second = application.export_structure(
        "CCCC",
        output,
        source_kind="smiles",
        output_format="mol2",
    )

    assert second["status"] == "failed"
    assert not receipt.exists()


def test_offline_circular_statistics_wrap_across_180_degrees():
    rows = [
        {
            "entity_key": "entity-1",
            "entity_angle_deg": 179.0,
            "source_class": "bound",
            "n_observations": 1,
            "n_structures": 1,
        },
        {
            "entity_key": "entity-2",
            "entity_angle_deg": -179.0,
            "source_class": "predicted_design",
            "n_observations": 1,
            "n_structures": 1,
        },
        {
            "entity_key": "entity-3",
            "entity_angle_deg": 178.0,
            "source_class": "predicted_complex",
            "n_observations": 1,
            "n_structures": 1,
        },
    ]

    statistics = _circular_statistics(rows)

    assert abs(abs(statistics["circular_mean_deg"]) - 180.0) < 1.0
    assert statistics["circular_std_deg"] < 2.0
    assert statistics["maximum_source_mean_delta_deg"] < 4.0


def test_offline_statistics_give_each_cross_source_entity_one_vote():
    rows = [
        {
            "entity_key": "shared",
            "entity_angle_deg": 0.0,
            "source_class": "bound",
            "n_observations": 10,
            "n_structures": 10,
        },
        {
            "entity_key": "shared",
            "entity_angle_deg": 0.0,
            "source_class": "predicted_complex",
            "n_observations": 100,
            "n_structures": 100,
        },
        {
            "entity_key": "other",
            "entity_angle_deg": 90.0,
            "source_class": "predicted_design",
            "n_observations": 1,
            "n_structures": 1,
        },
    ]

    statistics = _circular_statistics(rows)

    assert statistics["n_entities"] == 2
    assert statistics["circular_mean_deg"] == pytest.approx(45.0)


def _summary_row(
    *,
    entity: str,
    source: str,
    angles: list[float],
    structure: str | None = None,
):
    radians = [math.radians(angle) for angle in angles]
    histogram = [0] * 24
    for angle in angles:
        normalized = (angle + 180.0) % 360.0 - 180.0
        index = min(
            23,
            max(0, int(math.floor((normalized + 180.0) / 15.0))),
        )
        histogram[index] += 1
    return {
        "entity_key": entity,
        "source_class": source,
        "structure_key": structure,
        "cosine_mean": sum(math.cos(value) for value in radians)
        / len(radians),
        "sine_mean": sum(math.sin(value) for value in radians)
        / len(radians),
        "histogram_counts": histogram,
        "n_observations": len(angles),
        "n_structures": 1 if structure else len(angles),
    }


def test_offline_statistics_preserve_within_entity_multimodality():
    rows = [
        _summary_row(
            entity=f"entity-{index:02d}",
            source="predicted_complex",
            angles=[0.0, 40.0],
        )
        for index in range(20)
    ]

    statistics = _circular_statistics(rows)

    assert statistics["n_entities"] == 20
    assert statistics["circular_std_deg"] > 19.0
    assert statistics["normalized_entropy"] > 0.15
    assert statistics["rigidity_score"] < 0.70
    assert statistics["calibration_evaluable_count"] == 0
    assert statistics["calibration_false_rigid_ci_high"] is None
    assert statistics["confidence"] == "low"


def test_exact_statistics_use_structure_level_calibration():
    rows = [
        _summary_row(
            entity="same-full-inchikey",
            source=(
                "pdb_derived_complex"
                if index % 2
                else "predicted_complex"
            ),
            angles=[179.0 if index % 3 else -179.0],
            structure=f"structure-{index:02d}",
        )
        for index in range(40)
    ]

    statistics = _circular_statistics(
        rows, calibration_unit="structure"
    )

    assert statistics["n_entities"] == 1
    assert statistics["n_structures"] == 40
    assert statistics["calibration_unit"] == "structure"
    assert statistics["calibration_evaluable_count"] == 40
    assert statistics["calibration_false_rigid_count"] == 0
    assert statistics["calibration_false_rigid_ci_high"] < 0.10
    assert statistics["confidence"] == "high"


def test_compile_level_retains_calibrated_exact_and_rejects_bimodal():
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE observations (
            exact_key VARCHAR,
            residue_class_ring_key VARCHAR,
            morgan_key VARCHAR,
            generic_key VARCHAR,
            entity_key VARCHAR,
            source_class VARCHAR,
            file_sha256 VARCHAR,
            model_id INTEGER,
            angle_deg DOUBLE
        )
        """
    )
    exact_rows = [
        (
            "exact-rigid",
            "residue-rigid",
            "morgan-rigid",
            "generic-rigid",
            "same-entity",
            "source-a" if index % 2 else "source-b",
            f"{index:064x}",
            1,
            179.0 if index % 3 else -179.0,
        )
        for index in range(40)
    ]
    bimodal_rows = [
        (
            f"exact-flexible-{entity}",
            "residue-bimodal",
            f"morgan-flexible-{entity}",
            "generic-bimodal",
            f"entity-{entity}",
            "source-a",
            f"{1000 + entity:064x}",
            1,
            angle,
        )
        for entity in range(20)
        for angle in (0.0, 40.0)
    ]
    connection.executemany(
        "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [*exact_rows, *bimodal_rows],
    )

    exact = _compile_level(connection, "exact", "exact_key")
    residue = _compile_level(
        connection,
        "residue_class_ring",
        "residue_class_ring_key",
    )

    assert "exact-rigid" in exact
    assert exact["exact-rigid"]["calibration_unit"] == "structure"
    assert exact["exact-rigid"][
        "calibration_false_rigid_ci_high"
    ] < 0.10
    assert "residue-bimodal" not in residue
