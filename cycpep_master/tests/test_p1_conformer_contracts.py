import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.export import conformer


def _mol_with_conformer(smiles="c1ccccc1", seed=31):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
    return Chem.RemoveHs(molecule)


def test_mol2_preserves_aromatic_sybyl_atom_types_after_kekulization():
    molecule = _mol_with_conformer("c1ccncc1")

    block, error = conformer.mol_to_mol2(molecule)

    assert error is None
    atom_lines = block.split("@<TRIPOS>ATOM\n", 1)[1].split(
        "@<TRIPOS>BOND\n", 1
    )[0]
    atom_types = [line.split()[5] for line in atom_lines.splitlines() if line.strip()]
    assert atom_types.count("C.ar") == 5
    assert atom_types.count("N.ar") == 1
    assert "C.2" not in atom_types
    assert "N.2" not in atom_types


def test_mmff_fallback_is_recorded_as_uff(monkeypatch):
    monkeypatch.setattr(
        conformer.AllChem,
        "MMFFGetMoleculeProperties",
        lambda _mol: None,
    )

    stats, error = conformer.compute_conformer_ensemble_stats(
        "CC(N)C(=O)NC(C)C(=O)O",
        num_confs=3,
        random_seed=7,
    )

    assert error is None
    assert stats["requested_force_field"] == "mmff"
    assert stats["force_field"] == "uff"
    assert stats["force_field_fallback"] is True
    assert stats["force_fields_used"] == ["uff"]


def test_mol2_serializes_actual_force_field_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        conformer.AllChem,
        "MMFFGetMoleculeProperties",
        lambda _mol: None,
    )
    output = tmp_path / "fallback.mol2"

    path, error = conformer.smiles_to_mol2(
        "CCO", output_path=str(output), num_confs=1
    )

    assert error is None
    assert path == str(output)
    text = output.read_text(encoding="utf-8")
    assert "CYCPEP_REQUESTED_FORCE_FIELD=mmff" in text
    assert "CYCPEP_FORCE_FIELD=uff" in text


def test_optimize_false_reports_not_run():
    stats, error = conformer.compute_conformer_ensemble_stats(
        "CC(N)C(=O)NC(C)C(=O)O",
        num_confs=3,
        random_seed=7,
        optimize=False,
    )

    assert error is None
    assert stats["optimization_status"] == "not_run"
    assert stats["optimization_converged"] is None
    assert stats["optimization_requested"] is False


@pytest.mark.parametrize("writer", ["mol2", "sdf"])
def test_optimization_failure_is_returned_while_tuple_shape_stays_two(
    tmp_path, monkeypatch, writer
):
    molecule = _mol_with_conformer("CCO", seed=7)
    failure = "optimization did not converge (status=1)"
    monkeypatch.setattr(
        conformer,
        "_embed_3d",
        lambda *_args, **_kwargs: (molecule, None),
    )
    monkeypatch.setattr(
        conformer,
        "_optimize",
        lambda *_args, **_kwargs: (molecule, failure),
    )
    output = tmp_path / (f"failed.{writer}")

    if writer == "mol2":
        result = conformer.smiles_to_mol2(
            "CCO", output_path=str(output), force_field="uff"
        )
    else:
        result = conformer.smiles_to_sdf(
            "CCO", output_path=str(output), force_field="uff"
        )

    assert isinstance(result, tuple)
    assert len(result) == 2
    path, error = result
    assert path == str(output)
    assert error == failure
    assert output.is_file()
