"""CXSMILES generation from a neutral amino-acid SMILES.

Given a neutral monomer SMILES (e.g. ``CC(N)C(=O)O`` for alanine), detect the
peptide backbone (N-Cα-C(=O)-OH) and emit a CXSMILES with R-group dummy atoms
(``_R1`` = backbone N-terminus, ``_R2`` = backbone C-terminus) plus an inferred
``R3`` side-chain leaving group. This is the engine that turns a user- or
literature-supplied SMILES into a library monomer entry.

Originally part of ``build_monomer_library.py`` (the offline library builder);
extracted here so it lives inside the installable ``cycpep_master`` package and
can be reused by the monomer-admin API without reaching outside the package.
``build_monomer_library`` re-imports these names for backward compatibility.

The public entry point is :func:`gen_cxsmiles`. Returns a dict with keys
``replaced_SMILES, CXSMILES, Monomer_Type, Polymer_Type, R1, R2, R3``.
"""
from rdkit import Chem

# Backbone SMARTS in priority order.
# Each matches: N, [bridge C atoms...], C(=O)XHn where X is O, N, or S.
# H counts relaxed to handle both side-chain-substituted and unsubstituted cases.
BACKBONE_PATTERNS = [
    ("alpha_aa", Chem.MolFromSmarts("[N;!R;H2]-[C;!R]-[C;!R](=[O;!R])-[O,N,S;!R;H1,H2]")),
    ("beta_aa", Chem.MolFromSmarts("[N;!R;H2]-[C;!R]-[C;!R]-[C;!R](=[O;!R])-[O,N,S;!R;H1,H2]")),
    ("gamma_aa", Chem.MolFromSmarts("[N;!R;H2]-[C;!R]-[C;!R]-[C;!R]-[C;!R](=[O;!R])-[O,N,S;!R;H1,H2]")),
    ("n_alkyl_aa", Chem.MolFromSmarts("[N;!R;H1]-[C;!R]-[C;!R](=[O;!R])-[O,N,S;!R;H1,H2]")),
]

# Side-chain functional groups for R3 inference.
# Semantics:
#   R3 = "OH" → side chain has an extra -COOH (leaves as H2O on connection)
#   R3 = "H"  → side chain has -NH2 / -OH / -SH (leaves as H2 on connection)
#   R3 = "-"  → no modifiable side-chain group
#
# Key: must distinguish "side chain" from "backbone variant":
#   - CONH2 at C-term is R2's variant (C(=O)NH2 vs C(=O)OH), NOT an R3 site
#   - Guanidino NH2 is not a standard attachment point
#   - Aromatic NH2 (aniline) is not a standard attachment point
#   - Only aliphatic NH2/OH/SH count for R3

# COOH anywhere not in backbone → R3=OH
SIDECHAIN_COOH = Chem.MolFromSmarts("[C;!R](=[O;!R])-[O;!R;H1]")

# Aliphatic primary amine: NH2–CH2– (connected to sp3 carbon, not amide/guanidine/aniline)
SIDECHAIN_ALIPH_NH2 = Chem.MolFromSmarts("[N;!R;H2]-[C;!R;H2]")

# Aliphatic hydroxyl: OH–CH– or OH–CH2– (not phenol, not carboxyl OH)
SIDECHAIN_ALIPH_OH = Chem.MolFromSmarts("[O;!R;H1]-[C;!R;H1,H2]")

# Aliphatic thiol: SH–CH– or SH–CH2– (not thiophenol)
SIDECHAIN_ALIPH_SH = Chem.MolFromSmarts("[S;!R;H1]-[C;!R;H1,H2]")


def _infer_r3(mol, backbone_atom_indices):
    """Infer R3 (side-chain attachment point leaving group) by scanning for
    functional groups NOT belonging to the backbone match.
    Returns one of: "OH" (extra COOH), "H" (extra NH2/OH/SH), "-" (none)."""
    bb_set = set(backbone_atom_indices)

    # Extra COOH not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_COOH):
        c_atom = match[0]
        if c_atom not in bb_set:
            return "OH"

    # Extra aliphatic NH2 not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_ALIPH_NH2):
        if match[0] not in bb_set:
            return "H"

    # Extra aliphatic OH not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_ALIPH_OH):
        if match[0] not in bb_set:
            return "H"

    # Extra aliphatic SH not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_ALIPH_SH):
        if match[0] not in bb_set:
            return "H"

    return "-"


def gen_cxsmiles(neutral_smi: str) -> dict:
    """Core CXSMILES generator for one neutral SMILES."""
    mol = Chem.MolFromSmiles(neutral_smi)
    if mol is None:
        return _helm_fallback(neutral_smi)

    for patt_name, patt in BACKBONE_PATTERNS:
        matches = mol.GetSubstructMatches(patt)
        if matches:
            return _build_cx(mol, matches[0], neutral_smi)

    return _fallback_cx(mol, neutral_smi)


def _build_cx(mol, match: tuple, neutral_smi: str) -> dict:
    """Build CXSMILES from a backbone SMARTS match.
    Backbone match layout (applies to all patterns):
      match[0] = N, match[-3] = carbonyl-C, match[-2] = =O,
      match[-1] = terminal O/N/S cap atom
    """
    n_idx = match[0]
    cap_idx = match[-1]
    cap_default = {8: "OH", 7: "NH2", 16: "SH"}.get(
        mol.GetAtomWithIdx(cap_idx).GetAtomicNum()
    )
    if cap_default is None:
        return _fallback_cx(mol, neutral_smi)

    mol_h = Chem.AddHs(mol)
    h_on_n = None
    for nb in mol_h.GetAtomWithIdx(n_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_n = nb.GetIdx()
            break
    h_on_cap = None
    for nb in mol_h.GetAtomWithIdx(cap_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_cap = nb.GetIdx()
            break

    if h_on_n is None or h_on_cap is None:
        return _fallback_cx(mol, neutral_smi)

    rw = Chem.RWMol(mol_h)
    rw.GetAtomWithIdx(h_on_n).SetAtomicNum(0)
    rw.RemoveAtom(h_on_cap)
    rw.GetAtomWithIdx(cap_idx).SetAtomicNum(0)

    final_mol = rw.GetMol()
    final_mol = Chem.RemoveHs(final_mol, updateExplicitCount=True)
    try:
        Chem.SanitizeMol(final_mol)
    except Exception:
        pass

    cx_raw = Chem.MolToSmiles(final_mol)
    cx_mol = Chem.MolFromSmiles(cx_raw)
    if cx_mol is None:
        return _helm_fallback(neutral_smi)

    star_atoms = [a.GetIdx() for a in cx_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(star_atoms) < 2:
        return _helm_fallback(neutral_smi)

    r1_idx = r2_idx = None
    for sa in star_atoms:
        for nb in cx_mol.GetAtomWithIdx(sa).GetNeighbors():
            if nb.GetAtomicNum() == 7:
                r1_idx = sa
            elif nb.GetAtomicNum() == 6:
                r2_idx = sa
    if r1_idx is None:
        r1_idx = star_atoms[0]
    if r2_idx is None:
        r2_idx = star_atoms[1]
    if r1_idx == r2_idx:
        others = [s for s in star_atoms if s != r1_idx]
        r2_idx = others[0] if others else r1_idx

    n_atoms = cx_mol.GetNumAtoms()
    labels = [""] * n_atoms
    labels[r1_idx] = "_R1"
    labels[r2_idx] = "_R2"

    r3 = _infer_r3(mol, match)

    return {
        "replaced_SMILES": neutral_smi,
        "CXSMILES": f"{cx_raw} |${';'.join(labels)}$|",
        "Monomer_Type": "Backbone",
        "Polymer_Type": "PEPTIDE",
        "R1": "H",
        "R2": cap_default,
        "R3": r3,
    }


def _fallback_cx(mol, neutral_smi: str) -> dict:
    """Find any amine + any carboxyl, mark as connection points."""
    amine_m = mol.GetSubstructMatches(SIDECHAIN_ALIPH_NH2)
    carboxyl_m = mol.GetSubstructMatches(SIDECHAIN_COOH)
    if not amine_m or not carboxyl_m:
        return _helm_fallback(neutral_smi)

    n_idx = amine_m[0][0]
    o_sgl_idx = carboxyl_m[0][2]

    mol_h = Chem.AddHs(mol)
    h_on_n = None
    for nb in mol_h.GetAtomWithIdx(n_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_n = nb.GetIdx()
            break
    h_on_oh = None
    for nb in mol_h.GetAtomWithIdx(o_sgl_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_oh = nb.GetIdx()
            break

    if h_on_n is None or h_on_oh is None:
        return _helm_fallback(neutral_smi)

    rw = Chem.RWMol(mol_h)
    rw.GetAtomWithIdx(h_on_n).SetAtomicNum(0)
    rw.RemoveAtom(h_on_oh)
    rw.GetAtomWithIdx(o_sgl_idx).SetAtomicNum(0)

    final_mol = rw.GetMol()
    final_mol = Chem.RemoveHs(final_mol, updateExplicitCount=True)
    try:
        Chem.SanitizeMol(final_mol)
    except Exception:
        pass

    cx_raw = Chem.MolToSmiles(final_mol)
    cx_mol = Chem.MolFromSmiles(cx_raw)
    if cx_mol is None:
        return _helm_fallback(neutral_smi)

    star_atoms = [a.GetIdx() for a in cx_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not star_atoms:
        return _helm_fallback(neutral_smi)

    n_atoms = cx_mol.GetNumAtoms()
    labels = [""] * n_atoms
    if len(star_atoms) >= 1:
        labels[star_atoms[0]] = "_R1"
    if len(star_atoms) >= 2:
        labels[star_atoms[1]] = "_R2"

    # Build backbone atom set for R3 inference
    bb_set = {amine_m[0][0], carboxyl_m[0][0], carboxyl_m[0][1], o_sgl_idx}
    r3 = _infer_r3(mol, tuple(bb_set))

    return {
        "replaced_SMILES": neutral_smi,
        "CXSMILES": f"{cx_raw} |${';'.join(labels)}$|",
        "Monomer_Type": "Backbone",
        "Polymer_Type": "PEPTIDE",
        "R1": "H",
        "R2": "OH",
        "R3": r3,
    }


def _helm_fallback(neutral_smi: str) -> dict:
    return {
        "replaced_SMILES": neutral_smi,
        "CXSMILES": "",
        "Monomer_Type": "Backbone",
        "Polymer_Type": "PEPTIDE",
        "R1": "H",
        "R2": "OH",
        "R3": "-",
    }
