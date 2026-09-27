"""Lanthionine variant donor-stereo regression tests.

The runtime lanthionine bridge-donor variants in
``cycpep_master.paths.path_g._LANTHIONINE_VARIANT`` must encode the same
absolute donor alpha-carbon arrangement as their base residues.  With a
thioether sulfur grafted on CB, the CB carbon outranks the carbonyl carbon,
so an L-arranged donor alpha carbon is CIP **R** and a D-arranged donor is
CIP **S** (same ranking swap that makes L-cysteine R).  The historical
LanA/dLanA strings had the two tags mirrored (LanA carried the D arrangement
and dLanA the L arrangement), silently inverting every D-Ala-derived plain
lanthionine donor assembled by Path G.

All checks are synthetic and truth-free: the references are the known
proteinogenic amino acids and the correctly-shaped bMeLan/dBMeLan pair.
"""
from rdkit import Chem

from cycpep_master.paths import path_g
from cycpep_master.paths._map_utils import (
    get_smi_from_map,
    helm_to_map,
)
from cycpep_master.core.monomer_resolution import monomer_resolution_context

try:  # RDKit >= 2021 style accurate CIP assignment
    from rdkit.Chem import rdCIPLabeler

    def _assign_cip(mol):
        rdCIPLabeler.AssignCIPLabels(mol)
        return mol

except ImportError:  # pragma: no cover - legacy fallback
    def _assign_cip(mol):
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return mol


#: donor alpha-carbon CIP once a thioether-S sits on CB, per arrangement
LARRANGED, DARRANGED = "R", "S"


def _capped_variant_cip(cxsmiles):
    """Cap the variant monomer like a bridge donor and read C(alpha).

    Cap rules keep every local chiral tag untouched: the dummy on N gets a
    hydrogen, the dummy on the carboxyl carbon gets an oxygen, and every
    other dummy (the R3 attachment on CB) becomes sulfur.
    """
    mol = Chem.MolFromSmiles(cxsmiles)
    assert mol is not None, cxsmiles
    rw = Chem.RWMol(mol)
    for atom in list(rw.GetAtoms()):
        if atom.GetAtomicNum() != 0:
            continue
        neighbor = next((n for n in atom.GetNeighbors()), None)
        if neighbor is None:
            atom.SetAtomicNum(8)
        elif neighbor.GetAtomicNum() == 7:
            atom.SetAtomicNum(1)
        elif neighbor.GetAtomicNum() == 6 and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(neighbor).GetAtomicNum() == 8
            for bond in neighbor.GetBonds()
        ):
            atom.SetAtomicNum(8)
        else:
            atom.SetAtomicNum(16)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    _assign_cip(capped)
    return {
        atom.GetIdx(): atom.GetProp("_CIPCode")
        for atom in capped.GetAtoms()
        if atom.HasProp("_CIPCode")
    }


def _donor_alpha_cip(cxsmiles):
    """CIP label of the alpha carbon of a capped bridge-donor variant."""
    labels = _capped_variant_cip(cxsmiles)
    assert len(labels) == 1, (cxsmiles, labels)
    return next(iter(labels.values()))


def test_reference_cysteine_controls_define_thioether_semantics():
    # L-cysteine is the canonical S-ranking exception: L-arranged -> R.
    for smi, expected in (
        ("N[C@@H](CS)C(=O)O", "R"),  # L-Cys
        ("N[C@H](CS)C(=O)O", "S"),   # D-Cys
        ("C[C@H](N)C(=O)O", "S"),    # L-Ala
        ("C[C@@H](N)C(=O)O", "R"),   # D-Ala
    ):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        _assign_cip(mol)
        got = {
            a.GetProp("_CIPCode")
            for a in mol.GetAtoms()
            if a.HasProp("_CIPCode")
        }
        assert got == {expected}, (smi, got)


def test_grafting_sulfur_reorders_cip_without_flipping_geometry():
    # Grafting S onto the CB of the canonical residues changes only the CIP
    # ranking (S outranks the carbonyl), never the encoded arrangement: the
    # @ tag must not be "fixed" to preserve the old R/S label.
    for smi, expected_after_graft in (
        ("C[C@H](N)C(=O)O", "R"),   # L-Ala arrangement + S on CB -> R
        ("C[C@@H](N)C(=O)O", "S"),  # D-Ala arrangement + S on CB -> S
    ):
        mol = Chem.MolFromSmiles(smi)
        rw = Chem.RWMol(mol)
        alpha = next(
            a for a in rw.GetAtoms()
            if a.GetAtomicNum() == 6
            and any(n.GetAtomicNum() == 7 for n in a.GetNeighbors())
            and any(
                n.GetAtomicNum() == 6
                and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(n).GetAtomicNum() == 8
                    for b in n.GetBonds()
                )
                for n in a.GetNeighbors()
            )
        )
        cb = next(
            n for n in alpha.GetNeighbors()
            if n.GetAtomicNum() == 6
            and not any(
                b.GetBondType() == Chem.BondType.DOUBLE for b in n.GetBonds()
            )
        )
        sulfur = rw.AddAtom(Chem.Atom(16))
        rw.AddBond(cb.GetIdx(), sulfur, Chem.BondType.SINGLE)
        grafted = rw.GetMol()
        Chem.SanitizeMol(grafted)
        _assign_cip(grafted)
        got = {
            a.GetProp("_CIPCode")
            for a in grafted.GetAtoms()
            if a.HasProp("_CIPCode")
        }
        assert got == {expected_after_graft}, (smi, got)


def test_lanthionine_variant_templates_absolute_configuration():
    # The four runtime variants must match their base-residue arrangement;
    # bMeLan/dBMeLan were already correct and act as the in-table control.
    for base_symbol, (variant_symbol, cxsmiles) in path_g._LANTHIONINE_VARIANT.items():
        expected = (
            LARRANGED
            if not base_symbol.startswith("d")
            else DARRANGED
        )
        got = _donor_alpha_cip(cxsmiles)
        assert got == expected, (
            base_symbol,
            variant_symbol,
            cxsmiles,
            got,
            expected,
        )


def _assemble_bridge(donor_symbol):
    """Register the runtime variants and assemble donor-Cys thioether."""
    with monomer_resolution_context({"include_persistent_user": False}):
        path_g._register_lanthionine_variants()
        helm = (
            "PEPTIDE1{%s.C}$PEPTIDE1,PEPTIDE1,1:R3-2:R3$$$" % donor_symbol
        )
        map_payload = helm_to_map(helm)
        smiles = get_smi_from_map(map_payload)
    assert smiles, donor_symbol
    return smiles


def _thioether_alpha_atoms(mol):
    """Alpha carbons whose CB carries sulfur (donor and Cys-acceptor both)."""
    centers = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        has_n = any(n.GetAtomicNum() == 7 for n in atom.GetNeighbors())
        has_carbonyl = any(
            n.GetAtomicNum() == 6
            and any(
                b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()
            )
            for n in atom.GetNeighbors()
        )
        cb_bears_s = any(
            n.GetAtomicNum() == 6
            and any(m.GetAtomicNum() == 16 for m in n.GetNeighbors())
            and not any(
                b.GetBondType() == Chem.BondType.DOUBLE for b in n.GetBonds()
            )
            for n in atom.GetNeighbors()
        )
        if has_n and has_carbonyl and cb_bears_s:
            centers.append(atom.GetIdx())
    return centers


def test_map_engine_bridge_assembly_matches_variant_arrangement():
    # End to end through the same HELM->MAP->SMILES engine Path G uses.  The
    # assembled donor-Cys ring has exactly two thioether alpha carbons: the
    # donor under test and the fixed L-cysteine acceptor (L-Cys + thioether
    # = CIP R by the same sulfur-ranking rule).
    for donor_symbol, expected in (
        ("LanA", LARRANGED),
        ("dLanA", DARRANGED),
        ("bMeLan", LARRANGED),
        ("dBMeLan", DARRANGED),
    ):
        smiles = _assemble_bridge(donor_symbol)
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, (donor_symbol, smiles)
        _assign_cip(mol)
        centers = _thioether_alpha_atoms(mol)
        assert len(centers) == 2, (donor_symbol, smiles, centers)
        labels = sorted(
            mol.GetAtomWithIdx(idx).GetProp("_CIPCode") for idx in centers
        )
        assert labels == sorted([expected, "R"]), (
            donor_symbol, smiles, labels,
        )


def test_lan_a_vs_d_lan_a_assembly_differ_only_at_the_donor_center():
    l_smiles = _assemble_bridge("LanA")
    d_smiles = _assemble_bridge("dLanA")
    l_mol, d_mol = Chem.MolFromSmiles(l_smiles), Chem.MolFromSmiles(d_smiles)
    assert l_mol is not None and d_mol is not None
    _assign_cip(l_mol)
    _assign_cip(d_mol)

    def skeleton(m):
        s = Chem.Mol(m)
        Chem.RemoveStereochemistry(s)
        return s

    matches = skeleton(d_mol).GetSubstructMatches(skeleton(l_mol))
    assert matches, "LanA/dLanA assemblies must share one skeleton"
    mapping = matches[0]  # query l_idx -> target d_idx
    inverted = []
    for l_idx, d_idx in enumerate(mapping):
        d_atom, l_atom = d_mol.GetAtomWithIdx(d_idx), l_mol.GetAtomWithIdx(l_idx)
        d_cip = d_atom.GetProp("_CIPCode") if d_atom.HasProp("_CIPCode") else None
        l_cip = l_atom.GetProp("_CIPCode") if l_atom.HasProp("_CIPCode") else None
        if d_cip or l_cip:
            assert d_cip is not None and l_cip is not None, (l_idx, d_idx)
            if d_cip != l_cip:
                inverted.append(d_idx)
    assert len(inverted) == 1, (l_smiles, d_smiles, inverted)
    assert inverted[0] in _thioether_alpha_atoms(d_mol)


def test_plain_residues_outside_thioether_context_are_unchanged():
    # No-thioether control: the ordinary A/dA monomers keep the standard
    # proteinogenic identities (L -> S, D -> R by CIP without a CB sulfur).
    with monomer_resolution_context({"include_persistent_user": False}):
        path_g._register_lanthionine_variants()
        helm = "PEPTIDE1{A.dA}$$$$"
        smiles = get_smi_from_map(helm_to_map(helm))
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        _assign_cip(mol)
        labels = sorted(
            a.GetProp("_CIPCode")
            for a in mol.GetAtoms()
            if a.HasProp("_CIPCode")
        )
        assert labels == ["R", "S"], (smiles, labels)
