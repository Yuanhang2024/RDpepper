"""Tests for the compatibility MOL2 reader (core.mol2_compat)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master import application
from cycpep_master.core.mol2_compat import COMPATIBILITY_MODES, load_mol2
from cycpep_master.docking.mol2_input import write_validation_receipt
from cycpep_master.export.conformer import mol_to_mol2

FIXTURES = Path(__file__).parent / "fixtures" / "mol2_compat" / "highdb"
UNITY_CHARGED_FIXTURES = (
    Path(__file__).parent / "fixtures" / "mol2_compat" / "highdb_unity"
)

# Read-only copies of real candidate033 HighDB canonical artifacts that
# declare UNITY formal charges on valence-4 nitrogens: stock RDKit cannot
# sanitize them natively, the charge-aware reader recovers the declared
# state exactly.
UNITY_CHARGED_EXPECTATIONS = {
    "HIGHDB-2562.candidate033.mol2": {
        "aware_total": 2,
        "applied": [
            {"atom_id": 18, "charge": 1},
            {"atom_id": 67, "charge": 1},
        ],
        "atom_count": 221,
        "symbols": {18: "N", 67: "N"},
        "full_inchikey": "RGBFUKSHASHYGU-CPMQLRASSA-P",
    },
    "HIGHDB-2565.candidate033.mol2": {
        "aware_total": 2,
        "applied": [
            {"atom_id": 94, "charge": 1},
            {"atom_id": 143, "charge": 1},
        ],
        "atom_count": 198,
        "symbols": {94: "N", 143: "N"},
        "full_inchikey": "CENSWJAZMLIGIB-GLADGFSSSA-P",
    },
}


def _write_mol2(tmp_path: Path, name: str, smiles: str) -> tuple[Path, Chem.Mol]:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    path = tmp_path / name
    _, error = mol_to_mol2(molecule, str(path))
    assert error is None
    return path, molecule


def _unity_block(path: Path) -> tuple[list[str], int, int]:
    """Return (unity lines, start index, end index) of the UNITY section."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("@<TRIPOS>UNITY_ATOM_ATTR")
    end = start + 1
    charges = int(lines[start + 1].split()[1])
    end = start + 2 + 2 * charges
    return lines, start, end


def test_charge_aware_restores_unity_charges_native_misses(tmp_path):
    path, molecule = _write_mol2(tmp_path, "acetate.mol2", "CC(=O)[O-]")
    expected = Chem.GetFormalCharge(molecule)

    native_mol, native_report = load_mol2(path)
    assert native_mol is not None
    assert native_report["reader_mode"] == "rdkit_native"
    assert native_report["applied_formal_charges"] == []
    assert native_report["total_formal_charge"] != expected
    assert any(
        "does not restore" in warning for warning in native_report["warnings"]
    )

    aware_mol, aware_report = load_mol2(
        path, compatibility="rdkit_charge_aware"
    )
    assert aware_report["reader_mode"] == "rdkit_charge_aware"
    assert aware_report["total_formal_charge"] == expected
    assert aware_report["applied_formal_charges"] == [
        {"atom_id": 4, "charge": -1}
    ]
    charged = [
        atom for atom in aware_mol.GetAtoms() if atom.GetFormalCharge()
    ]
    assert len(charged) == 1
    assert charged[0].GetSymbol() == "O"
    assert aware_report["coordinates_verified"] is True
    assert aware_report["sanitized"] is True


def test_neutral_molecule_modes_agree(tmp_path):
    path, molecule = _write_mol2(tmp_path, "ethanol.mol2", "CCO")
    native_mol, native_report = load_mol2(path)
    aware_mol, aware_report = load_mol2(
        path, compatibility="rdkit_charge_aware"
    )
    assert native_report["total_formal_charge"] == 0
    assert aware_report["total_formal_charge"] == 0
    assert aware_report["applied_formal_charges"] == []
    assert native_report["atom_count"] == aware_report["atom_count"]
    assert (
        native_report["canonical_smiles"] == aware_report["canonical_smiles"]
    )
    assert aware_report["full_inchikey"] == Chem.MolToInchiKey(molecule)


def test_noncontiguous_atom_ids_mapped_by_parse_order(tmp_path):
    path, _ = _write_mol2(tmp_path, "shifted.mol2", "CC(=O)[O-]")
    lines, start, end = _unity_block(path)
    atom_start = lines.index("@<TRIPOS>ATOM")
    ids = [5, 9, 14] + [100 + i for i in range(20)]
    id_of_row = {}
    rewritten = []
    section = ""
    unity_remaining = 0
    row = 0
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            section = line[len("@<TRIPOS>"):]
            unity_remaining = 0
            rewritten.append(line)
            continue
        fields = line.split()
        if not fields:
            rewritten.append(line)
            continue
        if section == "ATOM":
            source_id = int(fields[0])
            id_of_row[source_id] = ids[row]
            fields[0] = str(ids[row])
            row += 1
            rewritten.append(" ".join(fields))
        elif section == "BOND":
            fields[1] = str(ids[int(fields[1]) - 1])
            fields[2] = str(ids[int(fields[2]) - 1])
            rewritten.append(" ".join(fields))
        elif section == "UNITY_ATOM_ATTR":
            if unity_remaining > 0:
                unity_remaining -= 1
                rewritten.append(line)
            else:
                fields[0] = str(ids[int(fields[0]) - 1])
                unity_remaining = int(fields[1])
                rewritten.append(" ".join(fields))
        else:
            rewritten.append(line)
    shifted = tmp_path / "shifted_ids.mol2"
    shifted.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    aware_mol, aware_report = load_mol2(
        shifted, compatibility="rdkit_charge_aware"
    )
    assert aware_report["atom_ids_contiguous"] is False
    assert aware_report["total_formal_charge"] == -1
    assert aware_report["applied_formal_charges"] == [
        {"atom_id": 100, "charge": -1}
    ]
    charged = [
        atom for atom in aware_mol.GetAtoms() if atom.GetFormalCharge()
    ]
    assert charged[0].GetSymbol() == "O"
    # the restored charge must land on the atom whose coordinates match the
    # charged ATOM row, not on whatever index id-1 would imply
    position = aware_mol.GetConformer().GetAtomPosition(charged[0].GetIdx())
    charged_source_row = next(
        line.split()
        for line in lines[atom_start + 1 : start]
        if int(line.split()[0]) == 4
    )
    expected = tuple(float(value) for value in charged_source_row[2:5])
    assert (
        abs(position.x - expected[0]) <= 1e-4
        and abs(position.y - expected[1]) <= 1e-4
        and abs(position.z - expected[2]) <= 1e-4
    )
    with pytest.raises(ValueError, match="rdkit_native"):
        load_mol2(shifted)


def test_conflicting_duplicate_unity_charge_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "conflict.mol2", "CC(=O)[O-]")
    lines, start, end = _unity_block(path)
    lines[start + 1 : end] = [
        "4 1",
        "charge -1",
        "4 1",
        "charge 1",
    ]
    target = tmp_path / "conflict.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting UNITY"):
        load_mol2(target, compatibility="rdkit_charge_aware")


def test_dangling_unity_atom_id_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "dangling.mol2", "CC(=O)[O-]")
    lines, start, end = _unity_block(path)
    lines[start + 1] = "999 1"
    target = tmp_path / "dangling.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the"):
        load_mol2(target, compatibility="rdkit_charge_aware")
    with pytest.raises(ValueError, match="absent from the"):
        load_mol2(target)


def test_noninteger_unity_charge_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "fractional.mol2", "CC(=O)[O-]")
    lines, start, end = _unity_block(path)
    lines[start + 2] = "charge -1.5"
    target = tmp_path / "fractional.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an integer"):
        load_mol2(target, compatibility="rdkit_charge_aware")


def test_dangling_bond_reference_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "bond.mol2", "CCO")
    lines = path.read_text(encoding="utf-8").splitlines()
    bond_start = lines.index("@<TRIPOS>BOND")
    fields = lines[bond_start + 1].split()
    fields[2] = "777"
    lines[bond_start + 1] = " ".join(fields)
    target = tmp_path / "bond.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the"):
        load_mol2(target, compatibility="rdkit_charge_aware")


def test_coordinates_and_source_bytes_preserved(tmp_path):
    path, molecule = _write_mol2(tmp_path, "keep.mol2", "CC(=O)[O-]")
    original_bytes = path.read_bytes()
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_start = lines.index("@<TRIPOS>ATOM")
    rows = []
    for line in lines[atom_start + 1 :]:
        if line.startswith("@<TRIPOS>"):
            break
        rows.append(line.split())

    aware_mol, report = load_mol2(
        path, compatibility="rdkit_charge_aware"
    )
    assert len(rows) == aware_mol.GetNumAtoms()
    conformer = aware_mol.GetConformer()
    for index, fields in enumerate(rows):
        position = conformer.GetAtomPosition(index)
        expected = tuple(float(value) for value in fields[2:5])
        observed = (position.x, position.y, position.z)
        assert max(
            abs(observed[axis] - expected[axis]) for axis in range(3)
        ) <= 1e-4
    assert path.read_bytes() == original_bytes


def test_unknown_mode_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "mode.mol2", "CCO")
    with pytest.raises(ValueError, match="unknown MOL2 compatibility mode"):
        load_mol2(path, compatibility="rdkit_best_effort")
    result = application.read_mol2(path, compatibility="rdkit_best_effort")
    assert result["status"] == "invalid_input"
    assert "unknown MOL2 compatibility mode" in result["error"]
    assert COMPATIBILITY_MODES == ("rdkit_native", "rdkit_charge_aware")


def test_multi_molecule_rejected(tmp_path):
    first, _ = _write_mol2(tmp_path, "first.mol2", "CCO")
    second, _ = _write_mol2(tmp_path, "second.mol2", "CCC")
    combined = tmp_path / "combined.mol2"
    combined.write_text(
        first.read_text(encoding="utf-8")
        + "\n"
        + second.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one molecule"):
        load_mol2(combined, compatibility="rdkit_charge_aware")


def _receipt_for(path: Path, molecule: Chem.Mol) -> Path:
    heavy = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    return write_validation_receipt(
        path,
        coordinate_mode="source_bound",
        coordinate_level="X3",
        rigor="L3:Q",
        quality="high",
        source_heavy_atom_mapping_complete=True,
        atom_provenance_complete=True,
        mapped_heavy_atom_indices=heavy,
        generated_heavy_atom_indices=[],
        expected_full_inchikey=Chem.MolToInchiKey(molecule),
    )


def test_receipt_verified_and_invalid_receipt_rejected(tmp_path):
    path, molecule = _write_mol2(tmp_path, "receipt.mol2", "CC(=O)[O-]")
    receipt = _receipt_for(path, molecule)

    _, report = load_mol2(
        path,
        compatibility="rdkit_charge_aware",
        receipt_path=receipt,
    )
    assert report["receipt_verification"]["status"] == "verified"
    assert report["receipt_verification"]["full_inchikey"] == (
        report["full_inchikey"]
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["mol2_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.validation.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_mol2(path, compatibility="rdkit_charge_aware", receipt_path=tampered)

    missing = tmp_path / "missing.validation.json"
    with pytest.raises(ValueError):
        load_mol2(
            path, compatibility="rdkit_charge_aware", receipt_path=missing
        )
    result = application.read_mol2(
        path, compatibility="rdkit_charge_aware", receipt_path=tampered
    )
    assert result["status"] == "invalid_input"

    # third-party files carry no receipt by default and must not require one
    _, plain = load_mol2(path, compatibility="rdkit_charge_aware")
    assert plain["receipt_verification"] == {"status": "not_requested"}


def test_app_envelope_and_capabilities(tmp_path):
    import cycpep_master

    path, molecule = _write_mol2(tmp_path, "app.mol2", "CC(=O)[O-]")
    result = application.read_mol2(
        path, compatibility="rdkit_charge_aware"
    )
    assert result["operation"] == "read_mol2"
    assert result["status"] == "success"
    data = result["data"]
    assert data["reader_mode"] == "rdkit_charge_aware"
    assert data["total_formal_charge"] == -1
    assert "mol" not in data and "molecule" not in data
    assert json.dumps(result)  # fully JSON-serializable envelope

    capabilities = application.capabilities()
    assert "read_mol2" in capabilities["data"]["operations"]
    assert "read_mol2" in application.__all__
    assert callable(cycpep_master.load_mol2)
    assert callable(cycpep_master.read_mol2)

    try:
        import rdpepper
    except ModuleNotFoundError:
        # outside an installed distribution the facade file still resolves
        # through its package location
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rdpepper",
            Path(cycpep_master.__file__).parent / "rdpepper" / "__init__.py",
        )
        rdpepper = importlib.util.module_from_spec(spec)
        sys.modules["rdpepper"] = rdpepper
        spec.loader.exec_module(rdpepper)

    assert "load_mol2" in rdpepper.__all__ and "read_mol2" in rdpepper.__all__
    assert rdpepper.load_mol2 is cycpep_master.load_mol2
    assert rdpepper.read_mol2 is cycpep_master.read_mol2


def test_sdf_export_positive(tmp_path):
    path, molecule = _write_mol2(tmp_path, "export.mol2", "CC(=O)[O-]")
    original_bytes = path.read_bytes()
    target = tmp_path / "exported.sdf"
    result = application.read_mol2(
        path, compatibility="rdkit_charge_aware", export_sdf=target
    )
    assert result["status"] == "success", result
    artifact = result["data"]["export_sdf"]
    assert artifact["artifact_kind"] == "sdf_file"
    assert artifact["path"] == str(target)
    assert artifact["roundtrip_max_coordinate_delta"] <= 0.001
    assert artifact["roundtrip_identity_verified"] is True
    assert len(artifact["sha256"]) == 64

    from rdkit import Chem as _Chem

    supplier = _Chem.SDMolSupplier(str(target), removeHs=False)
    records = [record for record in supplier]
    assert len(records) == 1 and records[0] is not None
    roundtrip = records[0]
    assert roundtrip.GetNumAtoms() == molecule.GetNumAtoms()
    assert _Chem.GetFormalCharge(roundtrip) == -1
    source = load_mol2(path, compatibility="rdkit_charge_aware")[0]
    for index in range(source.GetNumAtoms()):
        left = source.GetConformer().GetAtomPosition(index)
        right = roundtrip.GetConformer().GetAtomPosition(index)
        assert max(
            abs(left.x - right.x),
            abs(left.y - right.y),
            abs(left.z - right.z),
        ) <= 0.001
    assert path.read_bytes() == original_bytes  # source untouched


def test_sdf_export_refuses_overwrite_and_aliases(tmp_path):
    path, _ = _write_mol2(tmp_path, "alias.mol2", "CCO")
    source_bytes = path.read_bytes()

    existing = tmp_path / "existing.sdf"
    existing.write_text("keep me", encoding="utf-8")
    result = application.read_mol2(
        path, compatibility="rdkit_charge_aware", export_sdf=existing
    )
    assert result["status"] == "invalid_input"
    assert "refusing to overwrite" in result["error"]
    assert existing.read_text(encoding="utf-8") == "keep me"

    for alias in (path, path.parent / ".." / path.parent.name / path.name):
        result = application.read_mol2(
            path, compatibility="rdkit_charge_aware", export_sdf=alias
        )
        assert result["status"] == "invalid_input"
        assert "aliases the MOL2 input" in result["error"]

    hardlink = tmp_path / "hardlink.sdf"
    try:
        os.link(path, hardlink)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable on this filesystem: {exc}")
    result = application.read_mol2(
        path, compatibility="rdkit_charge_aware", export_sdf=hardlink
    )
    assert result["status"] == "invalid_input"
    assert path.read_bytes() == source_bytes
    assert hardlink.read_bytes() == source_bytes


def test_real_highdb_fixture_reads_without_receipt():
    fixtures = sorted(FIXTURES.glob("*.mol2"))
    if not fixtures:
        pytest.skip("no real HighDB MOL2 fixtures staged")
    for fixture in fixtures:
        molecule, report = load_mol2(
            fixture, compatibility="rdkit_charge_aware"
        )
        assert report["reader_mode"] == "rdkit_charge_aware"
        assert report["atom_count"] == molecule.GetNumAtoms() > 0
        assert report["sanitized"] is True
        assert report["receipt_verification"] == {"status": "not_requested"}
        assert report["coordinates_verified"] is True
        assert report["full_inchikey"]


def test_real_highdb_unity_charged_native_fails_aware_recovers():
    fixtures = sorted(UNITY_CHARGED_FIXTURES.glob("*.mol2"))
    if not fixtures:
        pytest.skip("no real UNITY-charged HighDB fixtures staged")
    assert len(fixtures) == len(UNITY_CHARGED_EXPECTATIONS)
    for fixture in fixtures:
        expected = UNITY_CHARGED_EXPECTATIONS[fixture.name]

        # stock RDKit cannot sanitize these real artifacts: the declared
        # UNITY charges are missing, so valence-4 nitrogens stay neutral
        with pytest.raises(ValueError, match="rdkit_native"):
            load_mol2(fixture)

        aware_mol, aware_report = load_mol2(
            fixture, compatibility="rdkit_charge_aware"
        )
        assert aware_report["reader_mode"] == "rdkit_charge_aware"
        assert aware_report["sanitized"] is True
        assert aware_report["atom_count"] == expected["atom_count"]
        assert (
            aware_report["total_formal_charge"] == expected["aware_total"]
        )
        assert (
            aware_report["applied_formal_charges"] == expected["applied"]
        )
        assert aware_report["full_inchikey"] == expected["full_inchikey"]
        for entry in expected["applied"]:
            atom = aware_mol.GetAtomWithIdx(entry["atom_id"] - 1)
            assert atom.GetSymbol() == expected["symbols"][entry["atom_id"]]
            assert atom.GetFormalCharge() == entry["charge"]
        assert aware_report["receipt_verification"] == {
            "status": "not_requested"
        }


def test_negative_unity_attribute_count_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "negative.mol2", "CC(=O)[O-]")
    lines, start, end = _unity_block(path)
    lines[start + 1] = "4 -1"
    target = tmp_path / "negative.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative"):
        load_mol2(target, compatibility="rdkit_charge_aware")


def test_unity_charge_attribute_without_value_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "novalue.mol2", "CC(=O)[O-]")
    lines, start, end = _unity_block(path)
    lines[start + 2] = "charge"
    target = tmp_path / "novalue.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_mol2(target, compatibility="rdkit_charge_aware")


def test_duplicate_unity_sections_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "twice.mol2", "CC(=O)[O-]")
    lines, start, end = _unity_block(path)
    lines.extend(
        [
            "@<TRIPOS>UNITY_ATOM_ATTR",
            "4 1",
            "charge -1",
        ]
    )
    target = tmp_path / "twice.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="multiple .*UNITY_ATOM_ATTR"):
        load_mol2(target, compatibility="rdkit_charge_aware")
    with pytest.raises(ValueError, match="multiple .*UNITY_ATOM_ATTR"):
        load_mol2(target)


def test_nonpositive_atom_identifier_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "zero.mol2", "CCO")
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_start = lines.index("@<TRIPOS>ATOM")
    fields = lines[atom_start + 1].split()
    fields[0] = "0"
    lines[atom_start + 1] = " ".join(fields)
    target = tmp_path / "zero.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-positive"):
        load_mol2(target, compatibility="rdkit_charge_aware")


def test_short_bond_row_rejected(tmp_path):
    path, _ = _write_mol2(tmp_path, "short.mol2", "CCO")
    lines = path.read_text(encoding="utf-8").splitlines()
    bond_start = lines.index("@<TRIPOS>BOND")
    fields = lines[bond_start + 1].split()
    lines[bond_start + 1] = " ".join(fields[:3])
    target = tmp_path / "short.mol2"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed MOL2 BOND row"):
        load_mol2(target, compatibility="rdkit_charge_aware")
