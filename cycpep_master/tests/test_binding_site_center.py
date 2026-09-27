"""Binding-site center selection must never silently use the whole receptor."""
from __future__ import annotations

import pytest

from cycpep_master.docking.vina_wrapper import get_binding_site_center


def _atom(serial, name, altloc, residue, chain, resid, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:>4s}{altloc:1s}{residue:>3s} {chain:1s}"
        f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00          {element:>2s}\n"
    )


def _fixture(tmp_path):
    path = tmp_path / "receptor.pdb"
    path.write_text(
        "".join((
            _atom(1, "CA", "", "ALA", "A", 10, 0, 0, 0, "C"),
            _atom(2, "CB", "", "ALA", "A", 10, 2, 0, 0, "C"),
            _atom(3, "H", "", "ALA", "A", 10, 100, 0, 0, "H"),
            _atom(4, "CA", "A", "GLY", "A", 11, 10, 2, 0, "C"),
            _atom(5, "CA", "B", "GLY", "A", 11, 80, 80, 80, "C"),
            _atom(6, "CA", "", "SER", "B", 10, 50, 50, 50, "C"),
            "END\n",
        )),
        encoding="ascii",
    )
    return path


def test_binding_site_center_selects_requested_chain_residues_and_heavy_atoms(tmp_path):
    path = _fixture(tmp_path)
    center = get_binding_site_center(str(path), [10, 11], chain_id="A")
    assert center == pytest.approx((4.0, 2.0 / 3.0, 0.0))


def test_binding_site_center_rejects_ambiguous_residue_across_chains(tmp_path):
    path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="ambiguous binding-site residue"):
        get_binding_site_center(str(path), [10], include_hydrogens=True)


@pytest.mark.parametrize("residue_ids", [[], [True], ["not-an-integer"]])
def test_binding_site_center_rejects_invalid_residue_selection(tmp_path, residue_ids):
    path = _fixture(tmp_path)
    with pytest.raises(ValueError):
        get_binding_site_center(str(path), residue_ids)


def test_binding_site_center_rejects_missing_residue_instead_of_fallback(tmp_path):
    path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="No atoms found"):
        get_binding_site_center(str(path), [999], chain_id="A")


def test_binding_site_center_rejects_multicharacter_legacy_chain(tmp_path):
    path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="at most one character"):
        get_binding_site_center(str(path), [10], chain_id="AA")
