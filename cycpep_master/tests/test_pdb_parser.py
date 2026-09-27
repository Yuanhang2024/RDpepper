from cycpep_master.core.pdb_parser import get_pdb_atoms, get_res_seq, read_conect


def test_read_conect_merges_repeated_source_records(tmp_path):
    source = tmp_path / "repeated-source.pdb"
    source.write_text(
        "CONECT    1    2\n"
        "CONECT    1    3\n"
        "CONECT    1    2    4\n"
        "END\n",
        encoding="ascii",
    )

    assert read_conect(source) == {1: [2, 3, 4]}


def test_mixed_atom_hetatm_records_remain_one_residue(tmp_path):
    def atom(record, serial, name, residue, element):
        return (
            f"{record:<6}{serial:5d} {name:>4s} ALA A{residue:4d}    "
            f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
            f"          {element:>2s}"
        )

    source = tmp_path / "mixed-record-residue.pdb"
    source.write_text(
        "\n".join([
            atom("ATOM", 1, "N", 1, "N"),
            atom("ATOM", 2, "CA", 1, "C"),
            atom("ATOM", 3, "C", 1, "C"),
            atom("ATOM", 4, "O", 1, "O"),
            atom("HETATM", 5, "CB", 1, "C"),
            atom("ATOM", 6, "N", 2, "N"),
            atom("ATOM", 7, "CA", 2, "C"),
            atom("ATOM", 8, "C", 2, "C"),
            atom("ATOM", 9, "O", 2, "O"),
            atom("ATOM", 10, "CB", 2, "C"),
            "END",
            "",
        ]),
        encoding="ascii",
    )

    residues = get_res_seq(source, "A")

    assert len(residues) == 2
    assert residues[0]["record_types"] == ["ATOM", "HETATM"]
    assert residues[0]["het"] is False
    assert {atom["name"] for atom in get_pdb_atoms(source, residues[0]["key"], "A")} == {
        "N", "CA", "C", "O", "CB"
    }
