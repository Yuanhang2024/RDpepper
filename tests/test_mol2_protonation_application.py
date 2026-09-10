from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master import application
from cycpep_master.docking.mol2_input import (
    default_receipt_path, load_validated_mol2, write_validation_receipt,
)
from cycpep_master.export.conformer import mol_to_mol2


def make_parent(tmp_path, level="X3", interleaved=False):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CN(C)C[C@H](O)C(=O)O"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    if interleaved:
        heavy = [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() > 1]
        hydrogens = [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() == 1]
        order = []
        for i in range(max(len(heavy), len(hydrogens))):
            if i < len(heavy):
                order.append(heavy[i])
            if i < len(hydrogens):
                order.append(hydrogens[i])
        molecule = Chem.RenumberAtoms(molecule, order)
    path = tmp_path / "input.mol2"
    _, error = mol_to_mol2(molecule, str(path))
    assert error is None
    heavy = [a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum() > 1]
    mapped = heavy if level == "X3" else heavy[:3] if level == "X2" else []
    generated = [i for i in heavy if i not in mapped]
    write_validation_receipt(
        path, coordinate_mode={"X1": "regenerated", "X2": "template_completed", "X3": "source_bound"}[level],
        coordinate_level=level, rigor="L3:Q", quality="high",
        source_heavy_atom_mapping_complete=level == "X3", atom_provenance_complete=True,
        mapped_heavy_atom_indices=mapped, generated_heavy_atom_indices=generated,
        expected_full_inchikey=Chem.MolToInchiKey(molecule),
    )
    return path


@pytest.mark.parametrize("level,interleaved", [("X3", False), ("X3", True), ("X2", True), ("X1", False)])
def test_protonate_mol2_preserves_parent_and_coordinate_provenance(tmp_path, level, interleaved):
    source = make_parent(tmp_path, level, interleaved)
    original = source.read_bytes()
    old_receipt = default_receipt_path(source).read_bytes()
    before = load_validated_mol2(source)
    output = tmp_path / "microstate.mol2"
    result = application.protonate_mol2(source, output)
    assert result["status"] == "success", result
    after = load_validated_mol2(output)
    assert source.read_bytes() == original
    assert default_receipt_path(source).read_bytes() == old_receipt
    assert after.coordinate_level == level
    assert after.rigor == "L1:H"
    assert after.formal_charge == 0
    assert Chem.MolToSmiles(Chem.RemoveHs(after.molecule)) != Chem.MolToSmiles(Chem.RemoveHs(before.molecule))
    assert sorted(a.GetFormalCharge() for a in after.molecule.GetAtoms() if a.GetFormalCharge()) == [-1, 1]
    left = [a for a in before.molecule.GetAtoms() if a.GetAtomicNum() > 1]
    right = [a for a in after.molecule.GetAtoms() if a.GetAtomicNum() > 1]
    for a, b in zip(left, right):
        assert a.GetAtomicNum() == b.GetAtomicNum()
        assert tuple(before.molecule.GetConformer().GetAtomPosition(a.GetIdx())) == tuple(after.molecule.GetConformer().GetAtomPosition(b.GetIdx()))
        assert before.receipt["atom_coordinate_origins"][str(a.GetIdx())] == after.receipt["atom_coordinate_origins"][str(b.GetIdx())]
    report = json.loads(Path(result["data"]["microstate_report_path"]).read_text())
    assert report["parent_mol2_sha256"] == before.sha256
    assert report["output_full_inchikey"] == after.full_inchikey
    assert "not experimental" in report["claim_boundary"]
    assert after.receipt["evidence_manifest_sha256"] == application._evidence_digest(report)


def test_existing_outputs_and_alias_are_not_overwritten(tmp_path):
    source = make_parent(tmp_path)
    source_text = source.read_bytes()
    assert application.protonate_mol2(source, source)["status"] == "invalid_input"
    assert source.read_bytes() == source_text
    output = tmp_path / "existing.mol2"
    output.write_text("sentinel")
    assert application.protonate_mol2(source, output)["status"] == "invalid_input"
    assert output.read_text() == "sentinel"


def test_invalid_policy_and_receipt_do_not_emit_microstate(tmp_path):
    source = make_parent(tmp_path)
    out = tmp_path / "invalid.mol2"
    assert application.protonate_mol2(source, out, policy="force_neutral")["status"] == "invalid_input"
    assert not out.exists()
    receipt = default_receipt_path(source)
    payload = json.loads(receipt.read_text())
    payload["mol2_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    result = application.protonate_mol2(source, out)
    assert result["status"] != "success"
    assert not out.exists()


def test_protonate_smiles_does_not_report_failed_rule_as_success(monkeypatch):
    from cycpep_master.docking import protonation
    def fail(_mol):
        raise ValueError("invalid microstate")
    monkeypatch.setattr(protonation, "protonate_molecule_ph74", fail)
    result = application.protonate_smiles("CN(C)C")
    assert result["status"] != "success"
    assert "invalid microstate" in result["error"]


def test_sequence_physiological_policy_surfaces_failure(monkeypatch):
    from cycpep_master.docking import protonation
    from cycpep_master.sequence import build_molecule_from_sequence
    def fail(_mol):
        raise ValueError("invalid microstate")
    monkeypatch.setattr(protonation, "protonate_molecule_ph74", fail)
    result = build_molecule_from_sequence("ARRA", cyclization="head-to-tail", protonation="physiological")
    graph = result["chemical_graph"]
    assert graph["materializable"] is False
    assert "invalid microstate" in graph["provenance"]["error"]


def test_cli_forwards_validated_mol2_protonation(tmp_path, monkeypatch, capsys):
    from cycpep_master.cli import main as cli
    captured = {}
    def call(source, output, **kwargs):
        captured.update(source=source, output=output, **kwargs)
        return {"operation": "protonate_mol2", "status": "success", "data": {}}
    monkeypatch.setattr(cli.services, "protonate_mol2", call)
    monkeypatch.setattr("sys.argv", ["rdpepper", "protonate-mol2", "a.mol2", "b.mol2", "--receipt", "a.validation.json", "--compact"])
    cli.main()
    assert captured == {"source": "a.mol2", "output": "b.mol2", "receipt_path": "a.validation.json"}
    assert json.loads(capsys.readouterr().out)["status"] == "success"
