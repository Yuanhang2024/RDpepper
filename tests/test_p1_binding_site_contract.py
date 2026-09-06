"""Binding-site residue identity and ambiguity contract tests."""

from __future__ import annotations

import pytest

from cycpep_master.docking.box import get_binding_site_center


def _atom(
    serial,
    name,
    residue,
    chain,
    resid,
    x,
    y=0.0,
    z=0.0,
    *,
    insertion_code="",
):
    return (
        f"ATOM  {serial:5d} {name:>4s} {residue:>3s} {chain:1s}"
        f"{resid:4d}{insertion_code:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          C\n"
    )


def test_chainless_selection_rejects_same_resnum_on_multiple_chains(tmp_path):
    source = tmp_path / "receptor.pdb"
    source.write_text(
        _atom(1, "CA", "ALA", "A", 10, 0.0)
        + _atom(2, "CA", "ALA", "B", 10, 100.0),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="ambiguous binding-site residue"):
        get_binding_site_center(str(source), [10])


def test_explicit_chain_resolves_same_resnum_on_other_chain(tmp_path):
    source = tmp_path / "receptor.pdb"
    source.write_text(
        _atom(1, "CA", "ALA", "A", 10, 0.0)
        + _atom(2, "CA", "ALA", "B", 10, 100.0),
        encoding="ascii",
    )

    assert get_binding_site_center(str(source), [10], chain_id="A") == (0.0, 0.0, 0.0)


def test_explicit_chain_rejects_multiple_insertion_codes_without_selector(
    tmp_path,
):
    source = tmp_path / "receptor.pdb"
    source.write_text(
        _atom(1, "CA", "ALA", "A", 10, 0.0)
        + _atom(2, "CA", "ALA", "A", 10, 20.0, insertion_code="A"),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="insertion-code"):
        get_binding_site_center(str(source), [10], chain_id="A")


def test_chainless_selection_keeps_distinct_residue_numbers(tmp_path):
    source = tmp_path / "receptor.pdb"
    source.write_text(
        _atom(1, "CA", "ALA", "A", 10, 0.0)
        + _atom(2, "CA", "ALA", "B", 11, 10.0),
        encoding="ascii",
    )

    assert get_binding_site_center(str(source), [10, 11]) == (5.0, 0.0, 0.0)
