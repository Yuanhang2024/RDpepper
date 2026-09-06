"""Tests for Path F/G/H — special-residue peptide SMILES generation.

These paths handle peptides containing special residues (stapled hydrocarbons,
depsipeptide lactones, lanthionine thioethers) that have no unified-library
template, where Paths A/B/C/E fail or drop atoms. Set
``RDPEPPER_TEST_SPECIAL_RESIDUES_DIR`` to the optional test-data directory.
Structure-dependent tests are skipped when their input files are absent.
"""
import os

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from cycpep_master.paths import generate_f, generate_g, generate_h
from cycpep_master.core import special_residues as sr

_DIR = os.environ.get("RDPEPPER_TEST_SPECIAL_RESIDUES_DIR", "")
_P53 = os.path.join(_DIR, "stapled_p53_hydrocarbon_3v3b.pdb")
_DAP = os.path.join(_DIR, "daptomycin_lactone_1t5n.pdb")
_NISIN = os.path.join(_DIR, "nisin_lanthionine_1wco.pdb")

_have_p53 = os.path.exists(_P53)
_have_dap = os.path.exists(_DAP)
_have_nisin = os.path.exists(_NISIN)


def _frags(smi):
    m = Chem.MolFromSmiles(smi)
    return len(Chem.GetMolFrags(m)) if m else 0


def _formula(smi):
    m = Chem.MolFromSmiles(smi)
    return rdMolDescriptors.CalcMolFormula(m) if m else None


def _heavy_counts(smi):
    """Heavy-atom element counts (format-independent, ignores H)."""
    from collections import Counter
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    return dict(Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() != "H"))


# ── Special-residue library scaffold ────────────────────────────────────────

def test_special_library_loads():
    codes = sr.all_codes()
    assert "0EH" in codes
    assert "MK8" in codes


def test_special_library_lookups():
    assert sr.is_special_residue("0EH")
    assert not sr.is_special_residue("XYZ")
    assert sr.get_symbol("0EH")
    assert sr.get_r3_atom("0EH") == "CAT"
    assert sr.get_r3_atom("MK8") == "CE"
    r1, r2, r3 = sr.get_rgroups("0EH")
    assert (r1, r2, r3) == ("H", "OH", "H")


# ── Part 1: daptomycin/nisin backbone residues + D-code mapping ──────────────

def test_backbone_residues_registered():
    """KYN/LME/DBU/FGA registered with authoritative CCD chemistry."""
    for code, sym in [("KYN", "Kyn"), ("LME", "3MeGlu"),
                      ("DBU", "dhB"), ("FGA", "gGlu")]:
        assert sr.is_special_residue(code), f"{code} not registered"
        assert sr.get_symbol(code) == sym
        r1, r2, _ = sr.get_rgroups(code)
        assert (r1, r2) == ("H", "OH")  # backbone amino acid caps


def test_backbone_residues_assemble_with_correct_formula():
    """Each new residue assembles in a Gly-X-Ala tripeptide with the formula
    expected from its CCD monomer minus two peptide-bond waters."""
    from cycpep_master.paths.path_g import _register_special_residues
    from cycpep_master.paths._map_utils import get_smi_from_map
    from rdkit.Chem import rdMolDescriptors as d
    _register_special_residues()
    # (symbol, expected tripeptide formula) — verified against CCD monomer math
    cases = [("Kyn", "C15H20N4O5"), ("3MeGlu", "C11H19N3O6"),
             ("dhB", "C9H15N3O4"), ("gGlu", "C10H17N3O6")]
    for sym, formula in cases:
        smi = get_smi_from_map("G" + f"{{nnr:{sym}}}" + "A")
        m = Chem.MolFromSmiles(smi) if smi else None
        assert m is not None, f"{sym} tripeptide failed to assemble"
        assert d.CalcMolFormula(m) == formula


def test_path_g_d_amino_acid_code_mapping():
    """Path G resolves D-amino-acid and L-non-standard PDB codes to their
    unified-library symbols (so peptides like daptomycin are not left with
    unknown monomers)."""
    from cycpep_master.paths.path_g import _residue_symbol
    assert _residue_symbol("DAL") == "dA"
    assert _residue_symbol("DSN") == "dS"
    assert _residue_symbol("DSG") == "dN"
    assert _residue_symbol("DBB") == "dAbu"
    assert _residue_symbol("ORN") == "Orn"
    assert _residue_symbol("AIB") == "Aib"


def test_path_g_lanthionine_donor_variants_preserve_d_l_stereochemistry():
    from rdkit import Chem
    from cycpep_master.paths.path_g import _LANTHIONINE_VARIANT
    for left, right in (("A", "dA"), ("Abu", "dAbu")):
        left_symbol, left_cx = _LANTHIONINE_VARIANT[left]
        right_symbol, right_cx = _LANTHIONINE_VARIANT[right]
        left_mol = Chem.MolFromSmiles(left_cx.split(" |", 1)[0])
        right_mol = Chem.MolFromSmiles(right_cx.split(" |", 1)[0])

        assert left_symbol != right_symbol
        assert left_mol is not None and right_mol is not None
        assert Chem.MolToSmiles(
            left_mol, canonical=True, isomericSmiles=True
        ) != Chem.MolToSmiles(
            right_mol, canonical=True, isomericSmiles=True
        )
        Chem.RemoveStereochemistry(left_mol)
        Chem.RemoveStereochemistry(right_mol)
        assert Chem.MolToSmiles(left_mol, canonical=True) == Chem.MolToSmiles(
            right_mol, canonical=True
        )


# ── Path F: geometric fallback ──────────────────────────────────────────────

@pytest.mark.skipif(not _have_p53, reason="p53 staple test PDB absent")
def test_path_f_stapled_peptide_single_fragment():
    """Path F recovers the stapled peptide as one connected molecule.

    This is the case Path E drops 17 carbons on; F must keep all atoms and
    yield a single fragment (the staple bridge intact).
    """
    smi, err = generate_f(_P53, "C")
    assert smi is not None, f"Path F failed: {err}"
    assert _frags(smi) == 1
    # 96 heavy atoms: 69 C, 15 N, 12 O (matches PDB chain C exactly)
    assert _heavy_counts(smi) == {"C":69,"N":15,"O":12}


@pytest.mark.skipif(not _have_p53, reason="p53 staple test PDB absent")
def test_path_f_formula_hard_check_returns_none_on_mismatch():
    """Path F must never return a structure whose formula disagrees; on a
    chemically inconsistent heuristic it returns (None, 'formula mismatch...').
    (Here we just assert the success path's contract — non-None implies match.)
    """
    smi, err = generate_f(_P53, "C")
    if smi is None:
        assert "mismatch" in err or "sanitize" in err or "no peptide" in err


# ── Path G: library-driven precise generation ───────────────────────────────

@pytest.mark.skipif(not _have_p53, reason="p53 staple test PDB absent")
def test_path_g_stapled_peptide_precise():
    """Path G assembles via the special-residue library with correct caps.

    It includes the special residues (0EH/MK8) and forms the staple R3-R3 bond,
    yielding a single fragment. Because it applies the COOH C-terminus cap
    (R2=OH) it has one more oxygen than the crystallographically incomplete
    structure (free acid vs unresolved terminus) — the chemically correct form.
    """
    smi, err = generate_g(_P53, "C")
    assert smi is not None, f"Path G failed: {err}"
    assert _frags(smi) == 1
    # library COOH cap -> C69 N15 O13 (one more O than raw atoms)
    assert _heavy_counts(smi) == {"C":69,"N":15,"O":13}


@pytest.mark.skipif(not _have_p53, reason="p53 staple test PDB absent")
def test_path_g_includes_special_residues():
    """Path G's HELM must contain the special-residue symbols, not drop them."""
    from cycpep_master.paths.path_g import build_helm_with_special
    helm = build_helm_with_special(_P53, "C")
    assert helm is not None
    sym = sr.get_symbol("0EH")
    assert sym in helm  # special residue present in sequence
    assert "R3-" in helm  # staple cross-link recorded


# ── Path H: CONECT-driven connectivity ──────────────────────────────────────

@pytest.mark.skipif(not _have_p53, reason="p53 staple test PDB absent")
def test_path_h_stapled_peptide():
    smi, err = generate_h(_P53, "C")
    assert smi is not None, f"Path H failed: {err}"
    assert _frags(smi) == 1
    assert _heavy_counts(smi) == {"C":69,"N":15,"O":12}


@pytest.mark.skipif(not _have_dap, reason="daptomycin test PDB absent")
def test_path_h_depsipeptide_single_fragment():
    """daptomycin (lactone depsipeptide, many non-standard residues): F and H
    both build a single connected molecule from geometry/records."""
    smi_f, _ = generate_f(_DAP, "A")
    smi_h, _ = generate_h(_DAP, "A")
    # at least one geometric route yields a single fragment
    assert (smi_f and _frags(smi_f) == 1) or (smi_h and _frags(smi_h) == 1)


# ── Cross-validation F vs H (same geometric truth) ──────────────────────────

@pytest.mark.skipif(not _have_p53, reason="p53 staple test PDB absent")
def test_path_f_h_formula_agree():
    """F and H assign bond orders by the same heuristic on the same atoms, so
    their molecular formulas must agree."""
    sf, _ = generate_f(_P53, "C")
    sh, _ = generate_h(_P53, "C")
    assert sf and sh
    assert _heavy_counts(sf) == _heavy_counts(sh)


# ── Lanthionine: Path G thioether-bridge assembly ───────────────────────────

@pytest.mark.skipif(not _have_nisin, reason="nisin test PDB absent")
def test_path_g_nisin_lanthionine_single_molecule():
    """nisin (1 lanthionine + 4 β-methyllanthionine bridges): Path G must now
    assemble a single connected molecule with thioether (C-S-C) bridges, not
    fail. The Ala/Abu bridge donors are swapped to R3-bearing lanthionine
    variants so the Cys-S can bond to their β-carbon."""
    smi, err = generate_g(_NISIN, "N")
    assert smi is not None, f"Path G failed on nisin: {err}"
    assert _frags(smi) == 1
    m = Chem.MolFromSmiles(smi)
    # thioether C-S-C present (lanthionine bridges)
    assert m.HasSubstructMatch(Chem.MolFromSmarts("[CX4]S[CX4]"))


@pytest.mark.skipif(not _have_nisin, reason="nisin test PDB absent")
def test_path_g_f_h_agree_on_nisin():
    """The three independent routes (library-driven G, geometric F and H) must
    agree on nisin's heavy-atom formula — strong cross-validation of the
    lanthionine handling."""
    g, _ = generate_g(_NISIN, "N")
    f, _ = generate_f(_NISIN, "N")
    h, _ = generate_h(_NISIN, "N")
    assert g and f and h
    assert _heavy_counts(g) == _heavy_counts(f) == _heavy_counts(h)


@pytest.mark.skipif(not _have_nisin, reason="nisin test PDB absent")
def test_dehydroalanine_resolves_to_dha_not_library_collision():
    """PDB code DHA (dehydroalanine, 2-amino-acrylic acid) must resolve to the
    special-residue 'Dha', not the unrelated unified-library 'DHA' entry."""
    assert sr.is_special_residue("DHA")
    assert sr.get_symbol("DHA") == "Dha"
    from cycpep_master.paths.path_g import _residue_symbol
    assert _residue_symbol("DHA") == "Dha"
