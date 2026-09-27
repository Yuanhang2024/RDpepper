from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from cycpep_master.export import conformer_ensemble as module
from cycpep_master.sequence import build_molecule_from_sequence


def _chemical_graph():
    return build_molecule_from_sequence(
        "ACDEFG", cyclization="head-to-tail"
    )["chemical_graph"]


def _candidate(parent: Chem.Mol) -> Chem.Mol:
    molecule = Chem.Mol(parent)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(
            index,
            (
                float(index % 5),
                float((index // 5) % 5),
                float(index // 25),
            ),
        )
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def _install_fast_materializer_fakes(monkeypatch):
    monkeypatch.setattr(
        module,
        "find_coordinate_evidence",
        lambda *_args, **_kwargs: {
            "status": "disabled",
            "matches": [],
            "reason": None,
        },
    )
    monkeypatch.setattr(
        module,
        "_embed",
        lambda parent, **_kwargs: (
            _candidate(parent),
            {"status": "embedded"},
        ),
    )
    monkeypatch.setattr(
        module,
        "_optimize",
        lambda *_args, **_kwargs: {
            "status": "converged",
            "method": "fixture",
            "energy": 0.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_constrained_relaxation",
        lambda *_args, **_kwargs: {
            "status": "converged",
            "method": "fixture",
            "energy": 0.0,
            "mechanism": "fixture",
            "iterations": 1,
            "tolerance_deg": module.PRIOR_RELAX_TOLERANCE_DEG,
            "bonds": [],
            "prior_unsatisfied": [],
        },
    )

    def fake_mol2(_molecule, output_path=None):
        path = Path(str(output_path))
        path.write_text("fixture mol2\n", encoding="utf-8")
        return str(path), None

    monkeypatch.setattr(module, "mol_to_mol2", fake_mol2)


def _write_receipt(path, **_metadata):
    receipt = Path(str(path) + ".validation.json")
    receipt.write_text("{}\n", encoding="utf-8")
    return receipt


def test_receipt_failure_is_isolated_and_later_members_survive(
    tmp_path, monkeypatch
):
    _install_fast_materializer_fakes(monkeypatch)
    monkeypatch.setattr(
        module,
        "_qa",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        module, "_is_duplicate", lambda *_args, **_kwargs: False
    )
    calls = {"count": 0}

    def flaky_receipt(path, **metadata):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("fixture receipt failure")
        return _write_receipt(path, **metadata)

    monkeypatch.setattr(
        module, "write_validation_receipt", flaky_receipt
    )

    result = module.materialize_mol2_ensemble(
        _chemical_graph(),
        tmp_path / "ensemble",
        ensemble_size=2,
        template_strategy="off",
        torsion_prior_path=tmp_path / "missing.json",
    )

    assert result["ensemble"]["status"] == "MATERIALIZED"
    assert result["ensemble"]["produced_count"] == 2
    assert len(result["validated_mol2_artifacts"]) == 2
    assert any(
        row["status"] == "mol2_validation_failed"
        for row in result["attempts"]
    )


def test_qa_rejections_produce_failed_ensemble_without_artifacts(
    tmp_path, monkeypatch
):
    _install_fast_materializer_fakes(monkeypatch)
    monkeypatch.setattr(
        module,
        "_qa",
        lambda *_args, **_kwargs: {
            "passed": False,
            "reason": "fixture QA rejection",
        },
    )

    result = module.materialize_mol2_ensemble(
        _chemical_graph(),
        tmp_path / "failed",
        ensemble_size=2,
        template_strategy="off",
        torsion_prior_path=tmp_path / "missing.json",
    )

    assert result["ensemble"]["status"] == "FAILED"
    assert result["ensemble"]["produced_count"] == 0
    assert result["validated_mol2_artifacts"] == []
    assert {
        row["status"] for row in result["attempts"]
    } == {"qa_rejected"}


def test_rmsd_duplicate_gate_closes_as_partial_not_complete(
    tmp_path, monkeypatch
):
    _install_fast_materializer_fakes(monkeypatch)
    monkeypatch.setattr(
        module,
        "_qa",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        module,
        "_is_duplicate",
        lambda _candidate_molecule, accepted: bool(accepted),
    )
    monkeypatch.setattr(
        module, "write_validation_receipt", _write_receipt
    )

    result = module.materialize_mol2_ensemble(
        _chemical_graph(),
        tmp_path / "partial",
        ensemble_size=2,
        template_strategy="off",
        torsion_prior_path=tmp_path / "missing.json",
    )

    assert result["ensemble"]["status"] == "PARTIAL"
    assert result["ensemble"]["produced_count"] == 1
    assert result["ensemble"]["warnings"] == ["ENSEMBLE_PARTIAL"]
    assert any(
        row["status"] == "duplicate"
        for row in result["attempts"]
    )


def test_real_qa_rejects_broken_bond_geometry_and_missing_closure():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC"))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (float(index), 0.0, 0.0))
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (10.0, 0.0, 0.0))
    molecule.AddConformer(conformer)

    qa = module._qa(
        molecule,
        expected_inchikey=Chem.MolToInchiKey(molecule),
        optimization={"status": "converged", "energy": 0.0},
        topology_class="head_to_tail",
    )

    assert qa["passed"] is False
    assert qa["bond_geometry"]["invalid_bond_count"] > 0
    assert qa["closure_valid"] is False
