import math

from rdkit import Chem

from cycpep_master.paths import generate_f, generate_h


def _atom(serial, x, *, y=0.0, name="C", residue=1):
    return (
        f"ATOM  {serial:5d} {name:>4s} ALA A{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00           C  "
    )


def test_path_f_rejects_disconnected_fragments(tmp_path):
    path = tmp_path / "two-fragments-f.pdb"
    path.write_text(
        "\n".join([_atom(1, 0.0), _atom(2, 20.0, residue=2), "END", ""]),
        encoding="ascii",
    )

    smiles, error = generate_f(str(path), "A")

    assert smiles is None
    assert error == "multiple disconnected fragments: 2"


def test_path_h_rejects_disconnected_fragments(tmp_path):
    path = tmp_path / "two-fragments-h.pdb"
    path.write_text(
        "\n".join(
            [
                _atom(1, 0.0),
                _atom(2, 1.5),
                _atom(3, 20.0, residue=2),
                _atom(4, 21.5, residue=2),
                "CONECT    1    2",
                "CONECT    3    4",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )

    smiles, error = generate_h(str(path), "A")

    assert smiles is None
    assert error == "multiple disconnected fragments: 2"


def test_path_h_ignores_conect_inside_nonfirst_model(tmp_path):
    path = tmp_path / "models.pdb"
    path.write_text(
        "\n".join([
            "MODEL        1",
            _atom(1, 0.0),
            _atom(2, 1.5),
            "ENDMDL",
            "MODEL        2",
            "CONECT    1    2",
            "ENDMDL",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    smiles, error = generate_h(str(path), "A")

    assert smiles is None
    assert error == "no intra-chain CONECT records (use Path F)"


def test_path_h_rejects_duplicate_serial_in_first_model(tmp_path):
    path = tmp_path / "duplicate-serial.pdb"
    path.write_text(
        "\n".join([
            _atom(1, 0.0),
            _atom(1, 1.5, name="N"),
            "CONECT    1    2",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    smiles, error = generate_h(str(path), "A")

    assert smiles is None
    assert "duplicate atom serial 1" in error


def test_path_h_geometry_completes_omitted_macrocycle_closure(tmp_path):
    radius = 1.5 / (2.0 * math.sin(math.pi / 8.0))
    atoms = [
        _atom(
            index + 1,
            radius * math.cos(2.0 * math.pi * index / 8.0),
            y=radius * math.sin(2.0 * math.pi * index / 8.0),
        )
        for index in range(8)
    ]
    conect = [
        f"CONECT{index:5d}{index + 1:5d}"
        for index in range(1, 8)
    ]
    path = tmp_path / "omitted-closure.pdb"
    path.write_text("\n".join([*atoms, *conect, "END", ""]), encoding="ascii")

    smiles, error = generate_h(str(path), "A")

    assert error is None
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    assert molecule.GetRingInfo().NumRings() == 1
    assert len(molecule.GetRingInfo().AtomRings()[0]) == 8


def test_path_h_preserves_explicit_conect_double_bond(tmp_path):
    path = tmp_path / "explicit-double.pdb"
    path.write_text(
        "\n".join([
            _atom(1, 0.0),
            _atom(2, 1.34),
            "CONECT    1    2    2",
            "CONECT    2    1    1",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    smiles, error = generate_h(str(path), "A")

    assert error is None
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    assert molecule.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE


def test_path_h_rejects_conflicting_conect_multiplicity(tmp_path):
    path = tmp_path / "conflicting-order.pdb"
    path.write_text(
        "\n".join([
            _atom(1, 0.0),
            _atom(2, 1.34),
            "CONECT    1    2    2",
            "CONECT    2    1",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    smiles, error = generate_h(str(path), "A")

    assert smiles is None
    assert "conflicting CONECT multiplicity" in error
