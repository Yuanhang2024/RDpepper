"""Focused regressions for residue-graph and explicit-connectivity contracts."""

from __future__ import annotations

import warnings

import pytest
from rdkit import Chem

from cycpep_master.paths.path_h import generate_h
from cycpep_master.reconstruction import (
    _explicit_chain_components,
    _profile_from_residue_graph,
)
from cycpep_master.paths._map_utils import get_smi_from_map


def _atom_line(serial, name, element, chain, residue=1):
    return (
        f"ATOM  {serial:5d} {name:>4s} CYS {chain}{residue:4d}    "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
        f"          {element:>2s}"
    )


def _link_record(chain1="A", chain2="B", residue1=1, residue2=1):
    record = list(" " * 80)
    record[0:6] = "LINK  "
    record[12:16] = f"{'SG':>4s}"
    record[17:20] = "CYS"
    record[21] = chain1
    record[22:26] = f"{residue1:4d}"
    record[42:46] = f"{'SG':>4s}"
    record[47:50] = "CYS"
    record[51] = chain2
    record[52:56] = f"{residue2:4d}"
    return "".join(record)


def _two_chain_file(tmp_path, *records):
    path = tmp_path / "explicit-multichain.pdb"
    lines = [
        _atom_line(1, "SG", "S", "A"),
        _atom_line(2, "SG", "S", "B"),
        *records,
        "END",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def test_residue_cycle_rank_includes_isolated_chain_vertices():
    profile = _profile_from_residue_graph(
        [["A", "A"], ["B"]],
        [((0, 1, 1), (0, 2, 2))],
    )

    assert profile["chain_count"] == 2
    assert profile["macrocycle_count"] == 1
    assert profile["macrocycle_count_metric"] == "cycle_rank"


@pytest.mark.parametrize(
    "record_kind",
    ["link", "conect"],
)
def test_auto_chain_components_use_explicit_link_or_conect(tmp_path, record_kind):
    if record_kind == "link":
        path = _two_chain_file(tmp_path, _link_record())
    else:
        path = _two_chain_file(tmp_path, "CONECT    1    2")

    groups, evidence = _explicit_chain_components(str(path), ["A", "B"])

    assert groups == [["A", "B"]]
    assert evidence[record_kind]
    assert evidence["inference"] == "explicit_records_only"


@pytest.mark.parametrize(
    "map_text",
    [
        "GILFVSKN{cyc:N-N}",
        "C{cyc:1-1}",
        "C{cyc:1:R1-1:R2}",
        "AA{cyc:1:R1-99:R2}",
    ],
)
def test_legacy_map_rejects_dropped_or_self_cyclization(map_text):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_smi_from_map(map_text)

    assert result is None
    assert caught
    assert any(
        any(token in str(item.message) for token in ("cyclization", "cyc", "MAP"))
        for item in caught
    )


def test_legacy_map_valid_macrocycle_still_assembles():
    smiles = get_smi_from_map("AAAAA{cyc:N-C}")
    molecule = Chem.MolFromSmiles(smiles)

    assert molecule is not None
    assert max(len(ring) for ring in molecule.GetRingInfo().AtomRings()) >= 8


def test_path_h_preserves_explicit_single_bond_against_carbonylic_heuristic(tmp_path):
    path = tmp_path / "explicit-single.pdb"
    path.write_text(
        "\n".join(
            [
                _atom_line(1, "C", "C", "A"),
                _atom_line(2, "O", "O", "A"),
                # One occurrence is an explicit single bond. The short C-O
                # geometry would otherwise trigger Path F's C=O heuristic.
                "CONECT    1    2",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )

    smiles, error = generate_h(str(path), "A")

    assert error is None
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    assert molecule.GetNumBonds() == 1
    assert molecule.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE
