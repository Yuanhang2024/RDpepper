from cycpep_master.core.pdb_utils import extract_chain


def _atom(serial, chain, x):
    return (
        f"ATOM  {serial:5d}   CA ALA {chain}{1:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          C  "
    )


def _link(chain1, chain2):
    record = list(" " * 80)
    record[0:6] = "LINK  "
    record[12:16] = f"{'CA':>4s}"
    record[17:20] = "ALA"
    record[21] = chain1
    record[22:26] = f"{1:4d}"
    record[42:46] = f"{'CA':>4s}"
    record[47:50] = "ALA"
    record[51] = chain2
    record[52:56] = f"{1:4d}"
    return "".join(record)


def test_extract_chain_uses_first_model_and_preserves_only_internal_connections(
    tmp_path,
):
    source = tmp_path / "models.pdb"
    output = tmp_path / "chain-a.pdb"
    source.write_text(
        "\n".join(
            [
                "MODEL        1",
                _atom(1, "A", 1.0),
                _atom(2, "A", 2.0),
                _atom(3, "B", 3.0),
                "ENDMDL",
                "MODEL        2",
                _atom(1, "A", 101.0),
                _atom(2, "A", 102.0),
                "CONECT    1    9",
                "ENDMDL",
                _link("A", "A"),
                _link("A", "B"),
                "SSBOND   1 CYS A    1    CYS A    2",
                "SSBOND   2 CYS A    1    CYS B    1",
                "CONECT    1    2    3",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )

    result = extract_chain(str(source), "A", str(output))
    text = output.read_text(encoding="utf-8")

    assert result == str(output)
    assert "   1.000" in text
    assert " 101.000" not in text
    assert _link("A", "A") in text
    assert _link("A", "B") not in text
    assert "SSBOND   1" in text
    assert "SSBOND   2" not in text
    assert "CONECT    1    2" in text
    assert "CONECT    1    2    3" not in text
    assert " B   1" not in text
