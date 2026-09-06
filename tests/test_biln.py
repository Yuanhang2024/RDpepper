"""Tests for BILN ↔ HELM ↔ MAP interconversion and stapled-peptide bond types."""
import pytest

from cycpep_master.paths._map_utils import (
    biln_to_helm, helm_to_biln, get_smi_from_biln, get_smi_from_map,
)
from cycpep_master.compare.smiles_compare import rdkit_canonical


# ── BILN → HELM ──────────────────────────────────────────────────────────

def test_biln_to_helm_linear():
    helm = biln_to_helm("P-E-P-T-I-D-E")
    assert "PEPTIDE1{P.E.P.T.I.D.E}" in helm


def test_biln_to_helm_disulfide():
    helm = biln_to_helm("C(1,3)-A-A-A-C(1,3)")
    assert "PEPTIDE1{C.A.A.A.C}" in helm
    assert "1:R3-5:R3" in helm


def test_biln_to_helm_multichain():
    helm = biln_to_helm("A-G.K(1,3)-E.G-L-E-E(1,3)")
    assert "PEPTIDE1{" in helm
    assert "PEPTIDE2{" in helm
    assert "1:R3-" in helm


def test_biln_to_helm_nnaa():
    helm = biln_to_helm("[meL]-A-A")
    assert "[meL]" in helm


# ── HELM → BILN ──────────────────────────────────────────────────────────

def test_helm_to_biln_linear():
    biln = helm_to_biln("PEPTIDE1{P.E.P.T.I.D.E}$$$$V2.0")
    assert biln == "P-E-P-T-I-D-E"


def test_helm_to_biln_disulfide():
    biln = helm_to_biln("PEPTIDE1{C.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3$$$")
    assert "C(1,3)" in biln
    assert "C(1,3)" in biln.split('-')[-1]  # last residue has bond annotation


def test_helm_to_biln_head_to_tail():
    biln = helm_to_biln("PEPTIDE1{A.A.A.A.A}$PEPTIDE1,PEPTIDE1,1:R1-5:R2$$$")
    assert "A(1,1)" in biln  # N-term R1
    assert "A(1,2)" in biln  # C-term R2


# ── Round-trip ───────────────────────────────────────────────────────────

def test_biln_helm_roundtrip_linear():
    biln = "P-E-P-T-I-D-E"
    helm = biln_to_helm(biln)
    biln2 = helm_to_biln(helm)
    assert biln == biln2


def test_biln_helm_roundtrip_disulfide():
    biln = "C(1,3)-A-A-A-C(1,3)"
    helm = biln_to_helm(biln)
    biln2 = helm_to_biln(helm)
    assert biln == biln2


# ── BILN → SMILES ────────────────────────────────────────────────────────

def test_biln_to_smiles_disulfide():
    smi = get_smi_from_biln("C(1,3)-A-A-A-C(1,3)")
    assert smi is not None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    # Should have 1 disulfide bond
    n_ss = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16]-[#16]")))
    assert n_ss == 1


def test_biln_to_smiles_head_to_tail():
    smi = get_smi_from_biln("A(1,1)-A-A-A-A(1,2)")
    assert smi is not None
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    assert rdMolDescriptors.CalcNumRings(mol) >= 1


def test_biln_to_smiles_matches_helm():
    """BILN→SMILES should give same result as HELM→SMILES for equivalent input."""
    biln_smi = get_smi_from_biln("C(1,3)-A-A-A-C(1,3)")
    # C-A-A-A-C with disulfide 1:R3-5:R3
    helm_smi = get_smi_from_map("CAAAC{cyc:1:R3-5:R3}")
    if biln_smi and helm_smi:
        assert rdkit_canonical(biln_smi) == rdkit_canonical(helm_smi)


# ── Stapled peptide bond classification ──────────────────────────────────

def test_staple_thioether_classification():
    from cycpep_master.core.cyclization import classify_link_type
    # Thioether: SG to side-chain C (not backbone)
    link = {'atom1': 'SG', 'atom2': 'CD'}
    assert classify_link_type(link) == 'staple_thioether'


def test_staple_alkyl_classification():
    from cycpep_master.core.cyclization import classify_link_type
    # Alkyl cross-link: side-chain C to side-chain C
    link = {'atom1': 'CG', 'atom2': 'CG'}
    assert classify_link_type(link) == 'staple_alkyl'


def test_staple_does_not_break_existing_types():
    from cycpep_master.core.cyclization import classify_link_type
    assert classify_link_type({'atom1': 'SG', 'atom2': 'SG'}) == 'disulfide'
    assert classify_link_type({'atom1': 'SG', 'atom2': 'CB'}) == 'thioether'
    assert classify_link_type({'atom1': 'NZ', 'atom2': 'CG'}) == 'isopeptide'
    assert classify_link_type({'atom1': 'N', 'atom2': 'C'}) == 'peptide'
