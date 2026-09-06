"""Molecule building helpers: combo assembly, CONECT application, orphan removal."""
from rdkit import Chem


def _consume_explicit_attachment_hydrogen(atom):
    """Replace one explicit attachment-site hydrogen with a new heavy-atom bond."""
    explicit_h = atom.GetNumExplicitHs()
    if explicit_h > 0:
        atom.SetNumExplicitHs(explicit_h - 1)


def add_crosslink_bond(combo, begin_idx, end_idx):
    """Add a crosslink while keeping explicit-H valence bookkeeping valid."""
    _consume_explicit_attachment_hydrogen(combo.GetAtomWithIdx(begin_idx))
    _consume_explicit_attachment_hydrogen(combo.GetAtomWithIdx(end_idx))
    combo.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)


_DETERMINISTIC_SINGLE_BOND_CLOSURES = frozenset({
    "peptide",
    "disulfide",
    "isopeptide",
    "ester",
    "thioether",
    "staple_thioether",
})


def materialize_typed_crosslink(combo, begin_idx, end_idx, bond_type):
    """Materialize a chemically determined single-bond closure fail-closed."""
    normalized = str(bond_type or "unknown").strip().lower()
    if normalized not in _DETERMINISTIC_SINGLE_BOND_CLOSURES:
        raise ValueError(
            f"closure bond order is not uniquely determined for {normalized}"
        )
    existing = combo.GetBondBetweenAtoms(begin_idx, end_idx)
    if existing is not None:
        if existing.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError(
                f"existing {normalized} closure has incompatible bond order"
            )
        return False
    add_crosslink_bond(combo, begin_idx, end_idx)
    return True


def add_to_combo(combo, smi):
    """Append a SMILES molecule template to an RWMol combo, returning the offset."""
    mol = Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol)
    offset = combo.GetNumAtoms()
    for a in mol.GetAtoms():
        a2 = Chem.Atom(a.GetAtomicNum())
        a2.SetChiralTag(a.GetChiralTag())
        a2.SetFormalCharge(a.GetFormalCharge())
        a2.SetNumExplicitHs(a.GetNumExplicitHs())
        a2.SetNoImplicit(a.GetNoImplicit())
        a2.SetIsAromatic(a.GetIsAromatic())
        combo.AddAtom(a2)
    for b in mol.GetBonds():
        combo.AddBond(b.GetBeginAtomIdx() + offset,
                      b.GetEndAtomIdx() + offset,
                      b.GetBondType())
    return offset


def apply_conect(combo, pdb2g, pdb2r, conect):
    """Add inter-residue cross-link bonds (disulfide, isopeptide) from CONECT."""
    added = set()
    for src, targets in conect.items():
        if src not in pdb2g:
            continue
        si = pdb2g[src]
        ri = pdb2r.get(src)
        for tgt in targets:
            if tgt not in pdb2g:
                continue
            ti = pdb2g[tgt]
            rj = pdb2r.get(tgt)
            if ri is not None and rj is not None and ri == rj:
                continue
            if combo.GetBondBetweenAtoms(si, ti):
                continue
            key = (min(si, ti), max(si, ti))
            if key in added:
                continue
            added.add(key)
            add_crosslink_bond(combo, si, ti)


def apply_geometric_crosslinks(combo, pdb2g, pdb2r, pdb_path, chain_id, *,
                               radius_multiplier=None, distance_ceiling=None):
    """Path E injection: add cyclization bonds that explicit records omitted.

    Uses geometric covalent-radius detection (cyclization.geometric_crosslink_
    atom_pairs) to recover cross-residue bonds (disulfides, isopeptides,
    head-to-tail closures) in PDBs that lack explicit connectivity records
    (e.g. AlphaFold-relaxed structures).

    Existing mapped bonds retain priority one edge at a time. Geometry may add
    a different missing edge even when an unrelated or partial CONECT/LINK
    record exists. Returns the number of bonds added.
    """
    from .cyclization import geometric_crosslink_atom_pairs
    n_added = 0
    for sa, sb in geometric_crosslink_atom_pairs(
        pdb_path, chain_id,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    ):
        if sa not in pdb2g or sb not in pdb2g:
            continue
        si, ti = pdb2g[sa], pdb2g[sb]
        ri, rj = pdb2r.get(sa), pdb2r.get(sb)
        if ri is not None and rj is not None and ri == rj:
            continue
        if combo.GetBondBetweenAtoms(si, ti):
            continue
        add_crosslink_bond(combo, si, ti)
        n_added += 1
    return n_added


def remove_orphans(combo, assigned_globals, pdb2g):
    """Remove template atoms that were not assigned to any PDB atom."""
    orphans = sorted(
        [a.GetIdx() for a in combo.GetAtoms() if a.GetIdx() not in assigned_globals],
        reverse=True,
    )
    for oi in orphans:
        combo.RemoveAtom(oi)
        for pa in list(pdb2g):
            if pdb2g[pa] > oi:
                pdb2g[pa] -= 1


def finish_mol(combo):
    """Sanitize and convert RWMol to canonical SMILES."""
    try:
        mol = combo.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smiles, None
    except Exception as e:
        return None, f"Sanitize: {e}"
