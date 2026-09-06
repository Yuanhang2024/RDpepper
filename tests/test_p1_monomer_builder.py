"""Focused regression tests for monomer build/validation helpers.

These tests use temporary paths and in-memory registry entries only. They do
not invoke either script's CSV-writing main function and never modify the
frozen unified monomer library.
"""

import pytest
from rdkit import Chem

from cycpep_master import build_monomer_library as builder
from cycpep_master import validate_monomer_library as validator


def test_builder_paths_do_not_add_nested_package_directory(tmp_path):
    package_dir = tmp_path / "cycpep_master"
    package_dir.mkdir()
    paths = builder.resolve_build_paths(str(package_dir))

    assert paths["output"] == str(package_dir / "unified_monomer_library.csv")
    assert "cycpep_master{}cycpep_master".format("\\") not in paths["output"]


def test_existing_source_candidate_is_preferred(tmp_path):
    package_dir = tmp_path / "cycpep_master"
    data_dir = package_dir / "data"
    data_dir.mkdir(parents=True)
    source = data_dir / "NNAA_10000.txt"
    source.write_text("CCO\tUT\n", encoding="utf-8")

    paths = builder.resolve_build_paths(str(package_dir))

    assert paths["nnaa"] == str(source)


def test_bcut_descriptors_are_computed_and_status_is_clean():
    values, status = builder.compute_descriptors_with_status("CCO")

    assert all(values[name] != 0.0 for name in builder._BCUT_FIELDS)
    assert not set(builder._BCUT_FIELDS).intersection(status["failed_descriptors"])
    assert status["parse_ok"] is True
    assert isinstance(builder.compute_descriptors("CCO"), dict)


def test_bcut_failure_is_observable(monkeypatch):
    def fail_bcut(_mol):
        raise RuntimeError("synthetic BCUT failure")

    monkeypatch.setattr(builder.rdMolDescriptors, "BCUT2D", fail_bcut)
    with pytest.warns(RuntimeWarning, match=r"BCUT2D.*CCO"):
        values, status = builder.compute_descriptors_with_status("CCO")

    assert all(values[name] == 0.0 for name in builder._BCUT_FIELDS)
    assert set(builder._BCUT_FIELDS).issubset(status["failed_descriptors"])
    assert status["parse_ok"] is True


def test_smiles_parse_failure_is_observable():
    with pytest.warns(RuntimeWarning, match=r"parse.*not-a-smiles"):
        values, status = builder.compute_descriptors_with_status("not-a-smiles")

    assert status["parse_ok"] is False
    assert all(values[name] == 0.0 for name in builder.DESCRIPTOR_FIELDS)


@pytest.mark.parametrize(
    "smiles, expected_charge",
    [
        ("[O-][N+](=O)c1ccccc1", (1, -1)),
        ("C[N+](C)(C)C", (1, 0)),
        ("c1cc[nH+]cc1", (1, 0)),
    ],
)
def test_neutralization_preserves_non_backbone_ionic_groups(smiles, expected_charge):
    neutralized = builder.neutralize_full(smiles)
    mol = Chem.MolFromSmiles(neutralized)

    assert mol is not None
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    assert expected_charge[0] in charges
    if expected_charge[1]:
        assert expected_charge[1] in charges


def test_backbone_zwitterion_is_neutralized_structurally():
    mol = Chem.MolFromSmiles(builder.neutralize_backbone("[NH3+]CC(=O)[O-]"))

    assert mol is not None
    assert all(atom.GetFormalCharge() == 0 for atom in mol.GetAtoms())


def test_fragment_pair_requires_a_sanitized_single_component():
    assert validator.test_fragment_pair(validator.ALA_SMI, validator.ALA_SMI)
    assert not validator.test_fragment_pair("C" * 20, validator.ALA_SMI)


def test_disulfide_check_does_not_use_string_length(monkeypatch):
    monkeypatch.setattr(validator, "get_linear_peptide", lambda _parts: "CC")
    monkeypatch.setattr(validator, "cyclize_linpep_from_map", lambda *_args: "CC")

    assert validator.test_disulfide_cyclization("A") is False


def test_port_validation_accepts_no_r3_and_rejects_fake_r3(monkeypatch):
    _, ok_without_r3, has_r3 = validator._validate_port_declarations("A")
    assert ok_without_r3 is True
    assert has_r3 is False
    assert validator.test_monomer("A")[1] is True

    symbol = "UTFakeR3"
    monkeypatch.setitem(validator.monomers2smi_dict, symbol, "C")
    monkeypatch.setitem(
        validator.monomers2r_groups_dict,
        symbol,
        {"R1": "H", "R2": "OH", "R3": "H"},
    )

    _, declarations_ok, has_r3 = validator._validate_port_declarations(symbol)
    assert declarations_ok is False
    assert has_r3 is True
    assert validator.test_monomer(symbol) == (False, False)


def test_disulfide_result_has_ring_and_sulfur_sulfur_bond():
    assert validator.test_disulfide_cyclization("C") is True
