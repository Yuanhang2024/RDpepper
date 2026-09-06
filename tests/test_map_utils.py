"""Regression tests for MAP -> SMILES assembly (paths/_map_utils.py).

Covers the two correctness bugs fixed in 2026-06:
  1. Free C-terminus (and free Asp/Glu side chains) must be carboxylic acids
     C(=O)O, NOT aldehydes C=O. An aldehyde is an ADMET toxicity alert and
     wrong chemistry.
  2. Cyclic peptides with a free side chain (e.g. a head-to-tail ring leaving
     an unbonded Asp R3) must cap that side chain to COOH, NOT leave a bare
     dummy '*'.

Plus the historical multi-disulfide assembly (3+ S-S bridges) and validation
guards.
"""
import warnings

import pytest

from cycpep_master.paths import _map_utils
from cycpep_master.paths._map_utils import get_smi_from_map
from .conftest import canonical, count_smarts, has_dummy

ALDEHYDE = "[CX3H1]=O"
CARBOXYL = "[CX3](=O)[OX2H1]"
DISULFIDE = "[#16X2]-[#16X2]"


def test_missing_reconstruction_slices_fall_back_to_unified(monkeypatch):
    expected = {"A": {"symbol": "A"}}
    monkeypatch.setattr(
        _map_utils, "_load_from_sublibraries", lambda _stems: None
    )
    monkeypatch.setattr(
        _map_utils, "_load_unified_single_file", lambda: expected
    )

    assert _map_utils._load_unified_with_reconstruction_rows() is expected


# ── Bug 1: free C-terminus / side chains are carboxylic acids ───────────────

@pytest.mark.parametrize("mp, n_acid", [
    ("KD", 2),    # free C-term + Asp R3 side chain
    ("KE", 2),    # free C-term + Glu R3 side chain
    ("AG", 1),    # free C-term only
    ("AAA", 1),
    ("GGGG", 1),
    ("{nt:ACE}AAA", 1),  # acetyl N-term, free C-term
])
def test_free_cterminus_is_acid_not_aldehyde(mp, n_acid):
    smi = get_smi_from_map(mp)
    assert smi, f"assembly failed for {mp!r}"
    assert count_smarts(smi, ALDEHYDE) == 0, f"{mp!r} produced an aldehyde: {smi}"
    assert count_smarts(smi, CARBOXYL) == n_acid, (
        f"{mp!r} expected {n_acid} COOH, got {count_smarts(smi, CARBOXYL)}: {smi}")


def test_kd_exact_smiles():
    """Pin the exact canonical SMILES so any regression is caught precisely."""
    smi = canonical(get_smi_from_map("KD"))
    assert smi == "N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CC(=O)O)C(=O)O"


def test_amide_cterminus_preserved():
    """An NH2-capped C-terminus stays an amide; only the free Asp R3 changes."""
    smi = get_smi_from_map("KD{ct:NH2}")
    assert smi
    assert count_smarts(smi, "[CX3](=O)[NX3]") >= 1  # at least the amide cap
    assert count_smarts(smi, ALDEHYDE) == 0


# ── Bug 2: cyclic peptides leave no bare dummy on free side chains ──────────

@pytest.mark.parametrize("mp", [
    "FKAGD{cyc:N-C}",        # head-to-tail; free Asp R3 must cap to COOH
    "KAAD{cyc:1:R1-4:R3}",   # isopeptide; free N-term + free C-term
])
def test_no_residual_dummy_in_cyclic(mp):
    smi = get_smi_from_map(mp)
    assert smi, f"assembly failed for {mp!r}"
    assert not has_dummy(smi), f"{mp!r} left a bare dummy '*': {smi}"
    assert count_smarts(smi, ALDEHYDE) == 0


# ── Multi-disulfide assembly (historical 3+ S-S failure) ────────────────────

@pytest.mark.parametrize("mp, n_ss", [
    ("ACCA{cyc:2:R3-3:R3}", 1),
    ("ACCACCA{cyc:2:R3-3:R3}{cyc:5:R3-6:R3}", 2),
    ("ACCACCACCA{cyc:2:R3-3:R3}{cyc:5:R3-6:R3}{cyc:8:R3-9:R3}", 3),
    ("ACCACCACCACCA{cyc:2:R3-3:R3}{cyc:5:R3-6:R3}{cyc:8:R3-9:R3}{cyc:11:R3-12:R3}", 4),
])
def test_multi_disulfide_bridges(mp, n_ss):
    smi = get_smi_from_map(mp)
    assert smi, f"assembly failed for {mp!r}"
    assert count_smarts(smi, DISULFIDE) == n_ss, (
        f"{mp!r} expected {n_ss} S-S, got {count_smarts(smi, DISULFIDE)}")
    assert not has_dummy(smi)
    assert count_smarts(smi, ALDEHYDE) == 0


# ── Real disulfide-rich peptide toxins ──────────────────────────────────────

TOXINS = {
    # name: (sequence, [(cys_pos1, cys_pos2), ...], head_to_tail)
    "omega-MVIIA": ("CKGKGAKCSRLMYDCCTGSCRSGKC", [(1, 16), (8, 20), (15, 25)], False),
    "alpha-GI": ("ECCNPACGRHYSC", [(2, 7), (3, 13)], False),
    "alpha-ImI": ("GCCSDPRCAWRC", [(2, 8), (3, 12)], False),
    "mu-GIIIA": ("RDCCTPPKKCKDRQCKPQRCCA", [(3, 15), (4, 20), (10, 21)], False),
    "apamin": ("CNCKAPETALCARRCQQH", [(1, 11), (3, 15)], False),
    "tachyplesin": ("KWCFRVCYRGICYRRCR", [(3, 16), (7, 12)], False),
    "SFTI-1": ("GRCTKSIPPICFPD", [(3, 11)], True),
}


def _toxin_map(seq, ss, head_to_tail):
    cyc = "".join(f"{{cyc:{i}:R3-{j}:R3}}" for i, j in ss)
    if head_to_tail:
        cyc = "{cyc:N-C}" + cyc
    return seq + cyc


@pytest.mark.parametrize("name", list(TOXINS))
def test_disulfide_toxins(name):
    seq, ss, h2t = TOXINS[name]
    smi = get_smi_from_map(_toxin_map(seq, ss, h2t))
    assert smi, f"{name}: assembly failed"
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"{name}: invalid SMILES"
    assert len(Chem.GetMolFrags(mol)) == 1, f"{name}: fragmented (rings not closed)"
    assert count_smarts(smi, DISULFIDE) == len(ss), f"{name}: wrong S-S count"
    assert not has_dummy(smi), f"{name}: residual dummy"
    assert count_smarts(smi, ALDEHYDE) == 0, f"{name}: aldehyde present"


# ── Validation guards (must reject impossible topologies with a warning) ────

def test_r3_on_residue_without_sidechain_is_rejected():
    """Ala has no R3; a {cyc:..:R3} bond on it must be refused (returns None)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_smi_from_map("ACAC{cyc:1:R3-3:R3}")  # residue 1 is Ala
    assert result is None
    assert any("R3" in str(w.message) for w in caught)


def test_same_residue_double_ring_is_rejected():
    """One attachment point used by two bonds is unsupported; refuse with warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_smi_from_map("CCCC{cyc:1:R3-2:R3}{cyc:1:R3-3:R3}")
    assert result is None
    assert len(caught) >= 1


# ── Terminal-modifier endpoint contract ───────────────────────────────────


@pytest.mark.parametrize(
    "mp", ["AC{nt:ACE}DE", "AB{ct:NME}C", "A{nt:ACE}B"],
)
def test_misplaced_map_terminal_modifier_rejected(mp):
    with pytest.raises(ValueError, match="misplaced MAP terminal modifier"):
        _map_utils.map_to_helm(mp)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _map_utils.get_smi_from_map(mp) is None
    assert any(
        "misplaced MAP terminal modifier" in str(w.message) for w in caught
    )


@pytest.mark.parametrize(
    "mp, code",
    [
        ("{nt:DKA}AAA", "DKA"),
        ("{nt:GOA}AAA", "GOA"),
        ("AAA{ct:PPD}", "PPD"),
        ("AAA{ct:MOR}", "MOR"),
    ],
)
def test_registered_but_unsupported_map_modifier_is_explicit(mp, code):
    with pytest.raises(ValueError, match="unsupported MAP .*terminal modifier"):
        _map_utils.map_to_helm(mp)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _map_utils.get_smi_from_map(mp) is None
    assert any(
        "unsupported" in str(w.message) and code in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize(
    "mp", ["{nt:ACE}AAA", "AAAAA{nt:ACE}", "KD{ct:NME}", "KD{ct:NH2}"],
)
def test_supported_map_terminal_modifiers_preserved(mp):
    assert _map_utils.get_smi_from_map(mp)
    _map_utils.map_to_helm(mp)  # must not raise
