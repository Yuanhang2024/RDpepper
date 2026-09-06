from cycpep_master.chemical_audit import audit_pdb_text
from cycpep_master.core import (
    cyclization,
    detect_cyclization,
    detect_cyclization_from_pdb,
    pdb_parser,
    read_conect,
    read_conect_pairs,
)
from cycpep_master.core.pdb_utils import pdb_atom_element
from cycpep_master.reconstruction import _auto_peptide_chain_ids


def _atom(serial, residue, x, *, name="CA", chain="A"):
    element = "N" if name == "N" else "C"
    return (
        f"ATOM  {serial:5d} {name:>4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          "
        f"{element:>2s}  "
    )


def test_legacy_pdb_parsers_use_only_first_model(tmp_path):
    path = tmp_path / "models.pdb"
    path.write_text(
        "\n".join([
            "MODEL        1",
            _atom(1, 1, 0.0),
            _atom(2, 2, 1.5),
            "ENDMDL",
            "MODEL        2",
            _atom(1, 1, 100.0),
            _atom(2, 2, 101.5),
            "CONECT    1    9",
            "ENDMDL",
            "CONECT    1    2",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    atoms = cyclization.read_atoms(str(path), "A")
    residues = pdb_parser.get_res_seq(str(path), "A")
    residue_atoms = pdb_parser.get_pdb_atoms(
        str(path), residues[0]["key"], "A"
    )

    assert atoms[1]["xyz"][0] == 0.0
    assert cyclization.read_conect(str(path)) == [(1, 2)]
    assert len(residues) == 2
    assert len(residue_atoms) == 1
    assert pdb_parser.read_conect(str(path)) == {1: [2]}


def test_core_package_preserves_legacy_exports_and_names_coordinate_parsers():
    assert detect_cyclization is pdb_parser.detect_cyclization
    assert read_conect is pdb_parser.read_conect
    assert detect_cyclization_from_pdb is cyclization.detect_cyclization
    assert read_conect_pairs is cyclization.read_conect


def test_first_model_is_selected_by_record_order_not_declared_number(tmp_path):
    path = tmp_path / "models-numbered-from-five.pdb"
    path.write_text(
        "\n".join([
            "MODEL        5",
            _atom(1, 1, 5.0),
            _atom(2, 2, 6.5),
            "ENDMDL",
            "MODEL        6",
            _atom(1, 1, 60.0),
            _atom(2, 2, 61.5),
            "ENDMDL",
            "CONECT    1    2",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    atoms = cyclization.read_atoms(str(path), "A")
    residues = pdb_parser.get_res_seq(str(path), "A")

    assert atoms[1]["xyz"][0] == 5.0
    assert len(residues) == 2
    assert cyclization.read_conect(str(path)) == [(1, 2)]


def test_connection_metadata_inside_later_model_is_ignored(tmp_path):
    link = list(" " * 80)
    link[0:6] = "LINK  "
    link[12:16] = f"{'SG':>4s}"
    link[17:20] = "CYS"
    link[21] = "A"
    link[22:26] = f"{1:4d}"
    link[42:46] = f"{'SG':>4s}"
    link[47:50] = "CYS"
    link[51] = "B"
    link[52:56] = f"{1:4d}"
    path = tmp_path / "later-model-connections.pdb"
    path.write_text(
        "\n".join([
            "MODEL        1",
            _atom(1, 1, 0.0, chain="A"),
            _atom(2, 1, 1.5, chain="B"),
            "ENDMDL",
            "MODEL        2",
            "SSBOND   1 CYS A    1    CYS B    1",
            "".join(link),
            "ENDMDL",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    assert cyclization.read_ssbond(str(path)) == []
    assert cyclization.read_link(str(path)) == []


def test_auto_chain_selection_uses_only_first_model(tmp_path):
    path = tmp_path / "model-chain-scope.pdb"
    first = [
        _atom(1, 1, 0.0, name="N", chain="A"),
        _atom(2, 1, 1.0, name="CA", chain="A"),
        _atom(3, 1, 2.0, name="C", chain="A"),
        _atom(4, 2, 3.0, name="N", chain="A"),
        _atom(5, 2, 4.0, name="CA", chain="A"),
        _atom(6, 2, 5.0, name="C", chain="A"),
    ]
    second = [
        line[:21] + "B" + line[22:]
        for line in first
    ]
    path.write_text(
        "\n".join([
            "MODEL        5", *first, "ENDMDL",
            "MODEL        6", *second, "ENDMDL", "END", "",
        ]),
        encoding="ascii",
    )

    assert _auto_peptide_chain_ids(str(path)) == ["A"]


def test_pdb_audit_rejects_nonfinite_coordinates():
    line = _atom(1, 1, 0.0)
    line = line[:30] + f"{float('nan'):8.3f}" + line[38:]
    result = audit_pdb_text(line + "\nEND\n")

    assert result.accepted is False
    assert "MALFORMED_PDB_ATOM_RECORD" in {
        issue.code for issue in result.issues
    }


def test_missing_element_column_respects_pdb_atom_name_alignment():
    def record(raw_name):
        return "ATOM      1 " + raw_name + "ALA A   1      0.000   0.000   0.000"

    assert pdb_atom_element(record(" CA ")) == "C"
    assert pdb_atom_element(record("SE  ")) == "Se"
    assert pdb_atom_element(record("CL  ")) == "Cl"


def test_pdb_audit_rejects_duplicate_self_and_unknown_conect():
    first = _atom(1, 1, 0.0)
    duplicate_identity = _atom(2, 1, 1.0)
    duplicate_serial = _atom(1, 2, 2.0)
    result = audit_pdb_text("\n".join([
        first,
        duplicate_identity,
        duplicate_serial,
        "CONECT    1    1",
        "CONECT    1  999",
        "CONECT bad",
        "END",
    ]))

    codes = {issue.code for issue in result.issues}
    assert "DUPLICATE_PDB_ATOM_SERIAL" in codes
    assert "DUPLICATE_PDB_ATOM_IDENTITY" in codes
    assert "PDB_SELF_CONNECTION" in codes
    assert "UNRESOLVED_PDB_CONECT" in codes
    assert "MALFORMED_PDB_CONECT_RECORD" in codes
