from __future__ import annotations

import os
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.core.pdb_parser import standard_pdb_atom_name_map
from cycpep_master.export.conformer import (
    _mol2_roundtrip_full_inchikey,
    _mol2_unity_formal_charges,
    mol_to_mol2,
    pdb_to_mol2,
    smiles_to_mol2,
    smiles_to_sdf,
)
from cycpep_master.paths.residue_template_factory import get_residue_template
from cycpep_master.pipeline import run_batch


PDB_TEXT = """\
ATOM      1  N   CYS L   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  CYS L   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   CYS L   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   CYS L   1       3.000   0.000   0.000  1.00  0.00           O
ATOM      5  CB  CYS L   1       1.000   1.000   0.000  1.00  0.00           C
ATOM      6  SG  CYS L   1       1.000   2.000   0.000  1.00  0.00           S
ATOM      7  N   CYS L   2       4.000   0.000   0.000  1.00  0.00           N
ATOM      8  CA  CYS L   2       5.000   0.000   0.000  1.00  0.00           C
ATOM      9  C   CYS L   2       6.000   0.000   0.000  1.00  0.00           C
ATOM     10  O   CYS L   2       7.000   0.000   0.000  1.00  0.00           O
ATOM     11  CB  CYS L   2       5.000   1.000   0.000  1.00  0.00           C
ATOM     12  SG  CYS L   2       5.000   2.000   0.000  1.00  0.00           S
CONECT    6   12
CONECT   12    6
CONECT    1    9
CONECT    9    1
END
"""


def _write_source_identity_alias_pdb(tmp_path):
    template = get_residue_template("ALA")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=31) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    names = {
        index: name
        for name, index in standard_pdb_atom_name_map(
            "ALA", template.smiles
        ).items()
    }
    rows = []
    serial = 1
    for residue in (1, 2, 3):
        resname = "ZZZ" if residue == 2 else "ALA"
        record = "HETATM" if residue == 2 else "ATOM  "
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(
                f"{record}{serial:5d} {names[atom.GetIdx()]:>4s} {resname:>3s} "
                f"A{residue:4d}    {point.x + residue * 10:8.3f}{point.y:8.3f}"
                f"{point.z:8.3f}  1.00  0.00          {atom.GetSymbol():>2s}"
            )
            serial += 1
    link = list(" " * 80)
    link[0:4] = "LINK"
    link[12:16] = "   N"
    link[17:20] = "ALA"
    link[21] = "A"
    link[22:26] = f"{1:4d}"
    link[42:46] = "   C"
    link[47:50] = "ALA"
    link[51] = "A"
    link[52:56] = f"{3:4d}"
    modres = list(" " * 80)
    modres[0:6] = "MODRES"
    modres[7:10] = "  1"
    modres[12:15] = "ZZZ"
    modres[16] = "A"
    modres[18:22] = f"{2:4d}"
    modres[24:27] = "DPN"
    modres[29:40] = "source-bound"
    path = tmp_path / "source-identity-alias.pdb"
    path.write_text(
        "\n".join([
            "SEQRES   1 A    3  ALA DPN ALA",
            "".join(modres),
            "".join(link),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def _write_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "cyclic_cys.pdb"
    path.write_text(PDB_TEXT, encoding="ascii")
    return path


def _write_v6_pdb(tmp_path: Path) -> Path:
    template = get_residue_template("CYS")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=29) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    names = {
        index: name
        for name, index in standard_pdb_atom_name_map(
            "CYS", template.smiles
        ).items()
    }
    assert len(names) == molecule.GetNumAtoms()

    rows = []
    serials = {}
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0)):
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            name = names[atom.GetIdx()]
            rows.append(
                f"ATOM  {serial:5d} {name:>4s} CYS L{residue:4d}    "
                f"{point.x + offset:8.3f}{point.y:8.3f}{point.z:8.3f}"
                f"  1.00  0.00          {atom.GetSymbol():>2s}"
            )
            serials[(residue, name)] = serial
            serial += 1
    rows.extend((
        f"CONECT{serials[(1, 'SG')]:5d}{serials[(2, 'SG')]:5d}",
        f"CONECT{serials[(2, 'SG')]:5d}{serials[(1, 'SG')]:5d}",
        f"CONECT{serials[(1, 'N')]:5d}{serials[(2, 'C')]:5d}",
        f"CONECT{serials[(2, 'C')]:5d}{serials[(1, 'N')]:5d}",
        "END",
    ))
    path = tmp_path / "cyclic_cys_v6.pdb"
    path.write_text("\n".join(rows) + "\n", encoding="ascii")
    return path


def _parse_mol2(block: str):
    lines = block.splitlines()
    atom_start = lines.index("@<TRIPOS>ATOM") + 1
    atom_end = next(
        index
        for index in range(atom_start, len(lines))
        if lines[index].startswith("@<TRIPOS>")
    )
    bond_start = lines.index("@<TRIPOS>BOND")
    atoms = {}
    for line in lines[atom_start:atom_end]:
        fields = line.split()
        atoms[int(fields[0])] = tuple(float(value) for value in fields[2:5])
    bonds = set()
    bond_end = next(
        (
            index for index in range(bond_start + 1, len(lines))
            if lines[index].startswith("@<TRIPOS>")
        ),
        len(lines),
    )
    for line in lines[bond_start + 1 : bond_end]:
        fields = line.split()
        if len(fields) >= 4:
            bonds.add(frozenset((int(fields[1]), int(fields[2]))))
    return atoms, bonds


def _mol2_bond_types(block: str):
    lines = block.splitlines()
    start = lines.index("@<TRIPOS>BOND") + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("@<TRIPOS>")),
        len(lines),
    )
    return [line.split()[3] for line in lines[start:end] if line.split()]


def _parse_mol2_structured(block: str):
    lines = block.splitlines()
    atom_start = lines.index("@<TRIPOS>ATOM") + 1
    atom_end = next(
        index
        for index in range(atom_start, len(lines))
        if lines[index].startswith("@<TRIPOS>")
    )
    bond_start = lines.index("@<TRIPOS>BOND")
    substructure_start = lines.index("@<TRIPOS>SUBSTRUCTURE")
    atoms = {}
    for line in lines[atom_start:atom_end]:
        fields = line.split()
        atoms[int(fields[0])] = {
            "name": fields[1],
            "type": fields[5],
            "substructure_id": int(fields[6]),
            "substructure_name": fields[7],
            "charge": float(fields[8]),
        }
    bonds = {}
    for line in lines[bond_start + 1:substructure_start]:
        fields = line.split()
        bonds[int(fields[0])] = {
            "begin": int(fields[1]),
            "end": int(fields[2]),
            "type": fields[3],
        }
    substructures = {}
    for line in lines[substructure_start + 1:]:
        fields = line.split()
        substructures[int(fields[0])] = {
            "name": fields[1],
            "root_atom": int(fields[2]),
            "chain": fields[5],
        }
    return atoms, bonds, substructures


def _two_atom_molecule(bond_type):
    molecule = Chem.RWMol()
    molecule.AddAtom(Chem.Atom("N" if bond_type == Chem.BondType.DATIVE else "C"))
    molecule.AddAtom(Chem.Atom("Cu" if bond_type == Chem.BondType.DATIVE else "C"))
    molecule.AddBond(0, 1, bond_type)
    result = molecule.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
    conformer.Set3D(True)
    result.AddConformer(conformer)
    return result


def test_pdb_to_mol2_preserves_all_coordinates_and_crosslinks(tmp_path):
    pdb_path = _write_pdb(tmp_path)

    block, error = pdb_to_mol2(str(pdb_path), chain_id="L", path="a")

    assert error is None
    assert block.startswith("@<TRIPOS>MOLECULE\n")
    atoms, bonds = _parse_mol2(block)
    assert len(atoms) == 12
    assert set(atoms.values()) == {
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 2.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
        (7.0, 0.0, 0.0),
        (5.0, 1.0, 0.0),
        (5.0, 2.0, 0.0),
    }

    atom_by_coord = {coord: atom_id for atom_id, coord in atoms.items()}
    disulfide = frozenset(
        (atom_by_coord[(1.0, 2.0, 0.0)], atom_by_coord[(5.0, 2.0, 0.0)])
    )
    head_to_tail = frozenset(
        (atom_by_coord[(0.0, 0.0, 0.0)], atom_by_coord[(6.0, 0.0, 0.0)])
    )
    assert disulfide in bonds
    assert head_to_tail in bonds
    assert "@<TRIPOS>SUBSTRUCTURE" in block
    assert "CYS1" in block and "CYS2" in block
    assert " SG " in block


def test_v6_mol2_export_rejects_conflicting_h_diagnostic(tmp_path):
    pdb_path = _write_v6_pdb(tmp_path)

    block, error = pdb_to_mol2(
        str(pdb_path), chain_id="L", path="v6"
    )

    assert block is None
    assert "diagnostic_identity_consistency" in error


def test_v6_mol2_export_rejects_source_identity_incomplete_unknown_alias(tmp_path):
    pdb_path = _write_source_identity_alias_pdb(tmp_path)

    block, error = pdb_to_mol2(str(pdb_path), chain_id="A", path="v6")

    assert block is None
    assert "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE" in error


def test_nonmapping_path_never_silently_regenerates_coordinates(tmp_path):
    pdb_path = _write_pdb(tmp_path)
    block, error = pdb_to_mol2(str(pdb_path), chain_id="L", path="b")
    assert block is None
    assert "use smiles_to_mol2 explicitly" in error


def test_smiles_to_mol2_emits_tripos_not_mdl():
    block, error = smiles_to_mol2("CCO", random_seed=7)

    assert error is None
    first_record = next(line for line in block.splitlines() if line.strip() and not line.startswith("#"))
    assert first_record == "@<TRIPOS>MOLECULE"
    assert "@<TRIPOS>ATOM" in block
    assert "@<TRIPOS>BOND" in block
    assert "V2000" not in block


def test_mol2_writer_emits_concrete_kekule_bond_orders():
    molecule = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=31) == 0
    molecule = Chem.RemoveHs(molecule)

    block, error = mol_to_mol2(molecule)

    assert error is None
    bond_types = _mol2_bond_types(block)
    assert bond_types.count("1") == 3
    assert bond_types.count("2") == 3
    assert "ar" not in bond_types


def test_mol2_writer_keeps_amide_carbon_nitrogen_as_single_bond():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=37) == 0
    molecule = Chem.RemoveHs(molecule)

    block, error = mol_to_mol2(molecule)

    assert error is None
    atoms, bonds, _ = _parse_mol2_structured(block)
    carbonyl_carbon = next(
        atom.GetIdx() + 1
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "C"
        and any(
            neighbor.GetSymbol() == "O"
            and molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
            == Chem.BondType.DOUBLE
            for neighbor in atom.GetNeighbors()
        )
    )
    amide_nitrogen = next(
        neighbor.GetIdx() + 1
        for neighbor in molecule.GetAtomWithIdx(carbonyl_carbon - 1).GetNeighbors()
        if neighbor.GetSymbol() == "N"
    )
    amide_bond = next(
        row for row in bonds.values()
        if frozenset((row["begin"], row["end"]))
        == frozenset((carbonyl_carbon, amide_nitrogen))
    )
    assert amide_bond["type"] == "1"
    assert "am" not in {row["type"] for row in bonds.values()}
    assert atoms[amide_nitrogen]["type"] == "N.am"


def test_mol2_writer_roundtrips_triple_charge_bonds_and_residue_labels():
    molecule = Chem.AddHs(Chem.MolFromSmiles("[NH3+]CC#N"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=41) == 0
    molecule = Chem.RemoveHs(molecule)
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetProp("_TriposAtomName", f"{atom.GetSymbol()}{index}")
        atom.SetProp("_TriposResidueName", "LIG")
        atom.SetProp("_TriposChainId", "L")
        atom.SetIntProp("_TriposResidueNumber", 42)
        atom.SetProp("_TriposInsertionCode", "A")

    block, error = mol_to_mol2(molecule)

    assert error is None
    atoms, bonds, substructures = _parse_mol2_structured(block)
    assert atoms[1]["charge"] == 1.0
    assert sum(row["charge"] for row in atoms.values()) == 1.0
    assert {row["substructure_id"] for row in atoms.values()} == {1}
    assert {row["substructure_name"] for row in atoms.values()} == {"LIG42A"}
    expected_bonds = {
        (
            frozenset((bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1)),
            {
                Chem.BondType.SINGLE: "1",
                Chem.BondType.DOUBLE: "2",
                Chem.BondType.TRIPLE: "3",
            }[bond.GetBondType()],
        )
        for bond in molecule.GetBonds()
    }
    observed_bonds = {
        (frozenset((row["begin"], row["end"])), row["type"])
        for row in bonds.values()
    }
    assert observed_bonds == expected_bonds
    assert "3" in {row["type"] for row in bonds.values()}
    assert substructures == {
        1: {"name": "LIG42A", "root_atom": 1, "chain": "L"}
    }
    assert _mol2_unity_formal_charges(block) == {0: 1}


@pytest.mark.parametrize(
    "smiles",
    [
        "[NH3+]CC#N",
        "NC(=[NH2+])NCCC[C@H](N)C(=O)[O-]",
        "N[C@@H](Cc1c[nH]cn1)C(=O)[O-]",
        "[NH3+][C@@H](C)C(=O)[O-]",
        "[NH3+][C@@H](CCCNC(=[NH2+])N)C(=O)[O-]",
    ],
)
def test_mol2_writer_preserves_full_charged_identity_for_two_readers(smiles):
    pybel = pytest.importorskip("openbabel.pybel")
    if "mol2" not in pybel.informats:
        pytest.skip("Open Babel Python binding has no MOL2 reader plugin")
    expected = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=43) == 0

    block, error = mol_to_mol2(molecule)

    assert error is None
    rdkit_key, rdkit_error = _mol2_roundtrip_full_inchikey(block)
    assert rdkit_error is None
    assert rdkit_key == expected
    openbabel_smiles = pybel.readstring("mol2", block).write("can").strip()
    openbabel_molecule = Chem.MolFromSmiles(openbabel_smiles)
    assert openbabel_molecule is not None
    assert Chem.MolToInchiKey(openbabel_molecule) == expected


@pytest.mark.parametrize("bond_type", [Chem.BondType.DATIVE, Chem.BondType.ZERO])
def test_mol2_writer_rejects_unsupported_bond_types_without_output(
    tmp_path, bond_type
):
    output = tmp_path / f"unsupported-{bond_type.name}.mol2"

    result, error = mol_to_mol2(_two_atom_molecule(bond_type), output)

    assert result is None
    assert error == f"unsupported MOL2 bond type at bond 1: {bond_type.name}"
    assert not output.exists()


@pytest.mark.parametrize("bond_type", [Chem.BondType.DATIVE, Chem.BondType.ZERO])
def test_mol2_writer_removes_stale_output_on_unsupported_bond(
    tmp_path, bond_type
):
    output = tmp_path / f"preexisting-{bond_type.name}.mol2"
    original = b"preexisting validated artifact\n"
    output.write_bytes(original)

    result, error = mol_to_mol2(_two_atom_molecule(bond_type), output)

    assert result is None
    assert error == f"unsupported MOL2 bond type at bond 1: {bond_type.name}"
    assert not output.exists()


def test_pdb_to_mol2_default_path_remains_compatible_with_path_a(tmp_path):
    pdb_path = _write_pdb(tmp_path)

    default_block, default_error = pdb_to_mol2(str(pdb_path), chain_id="L")
    explicit_block, explicit_error = pdb_to_mol2(
        str(pdb_path), chain_id="L", path="a"
    )

    assert default_error is None
    assert explicit_error is None
    assert default_block == explicit_block


def test_pdb_to_mol2_rejects_unmapped_source_atoms(tmp_path):
    pdb_path = _write_pdb(tmp_path)
    text = pdb_path.read_text(encoding="ascii").replace(
        "END\n",
        "ATOM     13  OXT CYS L   2       8.000   0.000   0.000  1.00  0.00           O\nEND\n",
    )
    pdb_path.write_text(text, encoding="ascii")

    block, error = pdb_to_mol2(str(pdb_path), chain_id="L", path="a")

    assert block is None
    assert error == "incomplete PDB atom mapping: 1 source heavy atoms unmapped"


def test_pdb_to_mol2_rejects_same_input_and_output_without_deleting_input(
    tmp_path,
):
    pdb_path = _write_pdb(tmp_path)
    original = pdb_path.read_bytes()

    result, error = pdb_to_mol2(
        str(pdb_path), output_path=str(pdb_path), chain_id="L", path="a"
    )

    assert result is None
    assert "same file" in error
    assert pdb_path.read_bytes() == original


def test_pdb_to_mol2_rejects_hardlink_output_without_deleting_input(tmp_path):
    pdb_path = _write_pdb(tmp_path)
    alias_path = tmp_path / "alias.pdb"
    try:
        os.link(pdb_path, alias_path)
    except (AttributeError, OSError) as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    original = pdb_path.read_bytes()

    result, error = pdb_to_mol2(
        str(pdb_path), output_path=str(alias_path), chain_id="L", path="a"
    )

    assert result is None
    assert "same file" in error
    assert pdb_path.read_bytes() == original
    assert alias_path.read_bytes() == original


def test_pdb_to_mol2_normalizes_same_chain_ter_segments(tmp_path):
    pdb_path = _write_pdb(tmp_path)
    lines = pdb_path.read_text(encoding="ascii").splitlines()
    segmented = []
    inserted = False
    for line in lines:
        if (
            not inserted
            and line.startswith(("ATOM", "HETATM"))
            and int(line[22:26]) == 2
        ):
            segmented.append("TER       0      CYS L   1")
            inserted = True
        segmented.append(line)
    pdb_path.write_text("\n".join(segmented) + "\n", encoding="ascii")

    block, error = pdb_to_mol2(str(pdb_path), chain_id="L", path="a")

    assert error is None
    atoms, _ = _parse_mol2(block)
    assert len(atoms) == 12


def test_v6_mol2_uses_isolated_coordinate_entity_not_stale_seqres(tmp_path):
    pdb_path = _write_v6_pdb(tmp_path)
    pdb_path.write_text(
        "SEQRES   1 L    2  ALA ALA\n"
        + pdb_path.read_text(encoding="ascii"),
        encoding="ascii",
    )

    block, error = pdb_to_mol2(str(pdb_path), chain_id="L", path="v6")

    assert block is None
    assert "diagnostic_identity_consistency" in error


def test_pdb_to_mol2_rejects_conflicting_conect_before_path_a(tmp_path):
    pdb_path = _write_pdb(tmp_path)
    text = pdb_path.read_text(encoding="ascii").replace(
        "CONECT    6   12\n",
        "CONECT    6   12    7\n",
    )
    pdb_path.write_text(text, encoding="ascii")

    block, error = pdb_to_mol2(str(pdb_path), chain_id="L", path="a")

    assert block is None
    assert error.startswith("V5_EXPLICIT_CONNECTION_VALENCE_CONFLICT:")


def test_run_batch_forwards_policy_and_records_mol2_receipt(
    tmp_path, monkeypatch
):
    pdb_path = _write_pdb(tmp_path)
    captured = {}

    def fake_export(_source, output_path, **kwargs):
        captured.update(kwargs)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("mock mol2\n", encoding="ascii")
        receipt = Path(str(output) + ".validation.json")
        receipt.write_text("{}\n", encoding="ascii")
        return {
            "operation": "export",
            "status": "success",
            "data": {
                "requested_format_status": "fulfilled",
                "output_path": str(output),
                "coordinate_mode": "template_completed",
                "coordinate_level": "X2",
                "validation_receipt_path": str(receipt),
                "validation_receipt_sha256": "a" * 64,
            },
            "error": None,
        }

    monkeypatch.setattr(
        "cycpep_master.application.export_structure", fake_export
    )

    result = run_batch(
        [str(pdb_path)],
        path="a",
        export_dir=str(tmp_path / "exports"),
        chain_id="L",
        target_chain_id="R",
        fallback_policy="max_coverage",
    )[0]

    assert result["export_status"] == "success"
    assert captured["fallback_policy"] == "max_coverage"
    assert result["export_requested_format_status"] == "fulfilled"
    assert result["export_coordinate_mode"] == "template_completed"
    assert result["export_coordinate_level"] == "X2"
    assert result["export_validation_receipt_path"].endswith(
        ".validation.json"
    )
    assert result["export_validation_receipt_sha256"] == "a" * 64


def test_run_batch_real_mol2_export_writes_validation_receipt(tmp_path):
    pdb_path = _write_pdb(tmp_path)
    result = run_batch(
        [str(pdb_path)],
        path="a",
        export_dir=str(tmp_path / "exports"),
        chain_id="L",
        target_chain_id="R",
    )[0]

    assert result["export_status"] == "success"
    assert result["export_requested_format_status"] == "fulfilled"
    assert result["export_coordinate_level"] == "X3"
    assert result["export_coordinate_mode"] == "source_bound"
    receipt = Path(result["export_validation_receipt_path"])
    assert receipt.is_file()
    assert len(result["export_validation_receipt_sha256"]) == 64


def test_run_batch_routes_sdf_to_sdf_writer(tmp_path):
    pdb_path = _write_pdb(tmp_path)
    export_dir = tmp_path / "exports"

    result = run_batch(
        [str(pdb_path)],
        path="a",
        export_dir=str(export_dir),
        export_format="sdf",
        chain_id="L",
        target_chain_id="R",
    )[0]

    output = export_dir / "cyclic_cys.sdf"
    assert result["export_coordinate_mode"] == "regenerated"
    assert result["export_format"] == "sdf"
    assert result["export_path"] == str(output)
    text = output.read_text(encoding="utf-8")
    assert text.rstrip().endswith("$$$$")
    assert "@<TRIPOS>" not in text


def test_run_batch_v6_preserves_diagnostic_identity_veto(tmp_path):
    pdb_path = _write_v6_pdb(tmp_path)
    result = run_batch(
        [str(pdb_path)], chain_id="L", target_chain_id="R"
    )[0]

    assert result["status"] == "rejected"
    assert result["qualified_success"] is False
    assert "diagnostic_identity_consistency" in result["error"]
    dimension = result["output_evidence"]["evidence_dimensions"][
        "diagnostic_identity_consistency"
    ]
    assert dimension["passed"] is False


def test_mol2_writer_rejects_nonfinite_coordinates(tmp_path):
    molecule = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.SetAtomPosition(0, (float("nan"), 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
    molecule.AddConformer(conformer)
    output = tmp_path / "bad.mol2"
    output.write_text("stale", encoding="utf-8")

    block, error = mol_to_mol2(molecule, output_path=str(output))

    assert block is None
    assert "non-finite" in error
    assert not output.exists()


def test_smiles_to_mol2_clears_stale_output_before_input_validation(tmp_path):
    output = tmp_path / "stale.mol2"
    output.write_text("old result", encoding="utf-8")

    result, error = smiles_to_mol2(
        "not a smiles", output_path=str(output), force_field="bad"
    )

    assert result is None
    assert "force_field" in error
    assert not output.exists()


def test_smiles_to_sdf_clears_stale_output_before_input_validation(tmp_path):
    output = tmp_path / "stale.sdf"
    output.write_text("old result", encoding="utf-8")

    result, error = smiles_to_sdf(
        "not a smiles", output_path=str(output), force_field="bad"
    )

    assert result is None
    assert "force_field" in error
    assert not output.exists()


def test_pdb_to_mol2_rejects_unknown_fallback_policy(tmp_path):
    pdb_path = _write_pdb(tmp_path)

    result, error = pdb_to_mol2(
        pdb_path, chain_id="L", fallback_policy="unknown"
    )

    assert result is None
    assert error.startswith("invalid_input: invalid fallback policy")


def test_pdb_to_mol2_clears_stale_output_when_input_preparation_fails(tmp_path):
    output = tmp_path / "stale.mol2"
    output.write_text("old result", encoding="utf-8")

    result, error = pdb_to_mol2(
        str(tmp_path / "missing.pdb"), output_path=str(output), chain_id="L"
    )

    assert result is None
    assert error
    assert not output.exists()


def test_pdb_to_mol2_preserves_selected_result_through_normalization(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    from cycpep_master.paths._map_utils import registry_epoch

    selected = SimpleNamespace(
        status="failed",
        provenance={
            "failure_reason": "selected result sentinel",
            "request_binding": {
                "source_sha256": None,
                "normalized_sha256": None,
                "normalized_chain_id": "L",
                "minimum_macrocycle_ring_size": 8,
                "require_empty_persistent_overlay": True,
                "infer_bond_orders": True,
                "registry_epoch": registry_epoch(),
            },
        },
    )

    @contextmanager
    def fake_prepare(path, chain_id):
        yield SimpleNamespace(
            pdb_path=Path(path), chain_id=chain_id, audit={}
        )

    def unexpected_reconstruction(*_args, **_kwargs):
        raise AssertionError("selected result must not be recomputed")

    monkeypatch.setattr(
        "cycpep_master.core.structure_io.prepare_coordinate_input",
        fake_prepare,
    )
    monkeypatch.setattr(
        "cycpep_master.result_first.reconstruct_structure",
        unexpected_reconstruction,
    )

    result, error = pdb_to_mol2(
        source,
        chain_id="L",
        path="result_first",
        _result_first_result=selected,
    )

    assert result is None
    assert error == "result-first source binding mismatch: registry_epoch"


def test_batch_export_rejects_windows_drive_syntax_in_name(tmp_path):
    from cycpep_master.export.conformer import batch_export

    rows = batch_export([("C:evil", "CC")], str(tmp_path))

    assert rows == [
        ("C:evil", None, "invalid output name: output name must not contain ':'")
    ]
    assert not (tmp_path / "C:evil.mol2").exists()


@pytest.mark.parametrize("force_field", ["bad", "", None])
def test_export_rejects_unknown_force_field(force_field, tmp_path):
    output = tmp_path / "bad.sdf"

    result, error = smiles_to_mol2(
        "CC", output_path=str(tmp_path / "bad.mol2"), force_field=force_field
    )

    assert result is None
    assert "force_field" in error


def test_batch_export_removes_written_file_when_worker_reports_error(
    tmp_path, monkeypatch
):
    from cycpep_master.export import conformer

    def partial_output(_smiles, output_path=None, **_kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("partial\n", encoding="ascii")
        return str(output_path), "optimization did not converge"

    monkeypatch.setattr(conformer, "smiles_to_mol2", partial_output)

    rows = conformer.batch_export(
        [("partial", "CC")], str(tmp_path), format="mol2"
    )

    assert rows == [
        ("partial", None, "optimization did not converge")
    ]
    assert not (tmp_path / "partial.mol2").exists()


def test_batch_export_detects_case_insensitive_name_collision(tmp_path):
    from cycpep_master.export.conformer import batch_export

    rows = batch_export([("A", "CC"), ("a", "CCC")], str(tmp_path))

    assert all(path is None and "collision" in error for _, path, error in rows)


# ── Optimization honesty: non-convergence is never a clean success ───────


def test_smiles_to_mol2_warns_on_optimization_failure(tmp_path, monkeypatch):
    from cycpep_master.export import conformer

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    monkeypatch.setattr(conformer, "_embed_3d", lambda *_a, **_k: (mol, None))
    monkeypatch.setattr(
        conformer,
        "_optimize",
        lambda *_a, **_k: (mol, "optimization did not converge (status=1)"),
    )
    output = tmp_path / "warn.mol2"

    with pytest.warns(UserWarning, match="optimization"):
        path, error = conformer.smiles_to_mol2(
            "CCO", output_path=str(output), force_field="uff"
        )

    assert error == "optimization did not converge (status=1)"
    assert path == str(output)
    assert output.stat().st_size > 0


def test_smiles_to_sdf_warns_on_optimization_failure(tmp_path, monkeypatch):
    from cycpep_master.export import conformer

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    monkeypatch.setattr(conformer, "_embed_3d", lambda *_a, **_k: (mol, None))
    monkeypatch.setattr(
        conformer,
        "_optimize",
        lambda *_a, **_k: (mol, "optimization did not converge (status=1)"),
    )
    output = tmp_path / "warn.sdf"

    with pytest.warns(UserWarning, match="optimization"):
        path, error = conformer.smiles_to_sdf(
            "CCO", output_path=str(output), force_field="uff"
        )

    assert error == "optimization did not converge (status=1)"
    assert path == str(output)
    assert output.stat().st_size > 0


@pytest.mark.parametrize("window", [-1.0, float("nan"), float("inf")])
def test_compute_conformer_ensemble_stats_rejects_invalid_energy_window(window):
    from cycpep_master.export import conformer

    stats, error = conformer.compute_conformer_ensemble_stats(
        "AAA", num_confs=3, energy_window=window
    )

    assert stats is None
    assert "energy_window" in error
    assert "finite non-negative" in error


def test_compute_conformer_ensemble_stats_records_nonconvergence(monkeypatch):
    from cycpep_master.export import conformer

    class FakeForceField:
        def Initialize(self):
            return None

        def Minimize(self, maxIts=1000):
            return 1  # non-converged

        def CalcEnergy(self):
            return 1.0

    monkeypatch.setattr(
        conformer.AllChem, "MMFFGetMoleculeProperties", lambda _mol: object()
    )
    monkeypatch.setattr(
        conformer.AllChem,
        "MMFFGetMoleculeForceField",
        lambda *_args, **_kwargs: FakeForceField(),
    )

    stats, error = conformer.compute_conformer_ensemble_stats(
        "CC(N)C(=O)NC(C)C(=O)O", num_confs=3, random_seed=7
    )

    assert error == "optimization did not converge for 3/3 conformers"
    assert stats["status"] == "failed"
    assert stats["optimization_status"] == "failed"
    assert stats["optimization_converged"] is False
    assert stats["optimization_nonconverged_count"] == 3
