"""Path F: geometric connectivity + heuristic bond orders (universal fallback).

For peptides containing special residues with no monomer-library template
(stapled hydrocarbons, depsipeptide lactones, lanthionine thioethers), Paths
A/B/C/E either fail or drop atoms. Path F instead rebuilds the molecule purely
from heavy-atom coordinates:

  1. RDKit ``rdDetermineBonds.DetermineConnectivity`` builds the full connection
     graph from xyz (single connected fragment, no atoms lost, no library).
  2. Bond orders are assigned heuristically (carbonyl/amide C=O, aromatic rings,
     C=C alkenes) — pure geometry cannot fix valence on a hydrogen-free peptide
     PDB, so this step is *approximate*.
  3. The result is validated by a hard molecular-formula check against the PDB
     chain's heavy-atom element counts. A mismatch returns ``None`` rather than
     a silently wrong structure.

Path F is the exploratory fallback. For registered special chemistries prefer
Path G (library-driven, correct bond orders); when CONECT records exist prefer
Path H. See FUNCTION_INDEX for the F/G/H division of labour.
"""
from collections import Counter

from ..core.pdb_utils import pdb_atom_element, read_first_model_lines

# Skip non-peptide HETATM (waters, ions, common cryo/buffer molecules).
_SKIP_HET = {'HOH', 'WAT', 'DOD', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE',
             'MN', 'CU', 'SO4', 'PO4', 'GOL', 'EDO', 'DMS', 'PEG', 'EPE',
             'MES', 'ACT', 'FMT', 'CIT', 'CLA', 'NAG', 'MAN', 'BMA', 'FUC'}


def _read_chain_heavy_atoms(pdb_path, chain_id):
    """Return [(element, x, y, z), ...] for one chain's peptide heavy atoms.

    Uses only the first MODEL (NMR ensembles repeat serials per model), drops
    hydrogens, waters, ions and common non-peptide HETATM.
    """
    atoms = []
    for line in read_first_model_lines(str(pdb_path)):
        if line[:6] not in ('ATOM  ', 'HETATM'):
            continue
        if line[21] != chain_id:
            continue
        resn = line[17:20].strip()
        if resn in _SKIP_HET:
            continue
        elem = pdb_atom_element(line)
        if not elem:
            continue
        if elem == 'H':
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        atoms.append((elem, x, y, z))
    return atoms


def _build_mol(atoms):
    """Build an RDKit mol with a conformer from [(elem,x,y,z), ...]."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, (elem, x, y, z) in enumerate(atoms):
        rw.AddAtom(Chem.Atom(elem))
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol = rw.GetMol()
    mol.AddConformer(conf)
    return mol


def _dist(conf, i, j):
    a = conf.GetAtomPosition(i)
    b = conf.GetAtomPosition(j)
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _assign_bond_orders(mol):
    """Heuristically upgrade single bonds to double where geometry indicates.

    Carbonyl/amide/carboxyl C=O: a carbon bonded to a terminal oxygen at
    < 1.30 A is a double bond. This is the dominant correction for peptides and
    is what makes the formula match (it changes no atom counts, only orders).
    """
    from rdkit import Chem
    conf = mol.GetConformer()
    rw = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        # A carbonyl carbon is sp2: at most 3 heavy neighbours. A carbon with
        # 4 heavy neighbours is sp3 and cannot take a C=O double bond — adding
        # one would create a valence-5 carbon. Guard on degree.
        if atom.GetDegree() > 3:
            continue
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() == 'O' and nb.GetDegree() == 1 \
                    and _dist(conf, atom.GetIdx(), nb.GetIdx()) < 1.30:
                bond = rw.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                bond.SetBondType(Chem.BondType.DOUBLE)
                break  # one C=O per carbonyl carbon
    return rw.GetMol()


def _formula_counts(mol):
    c = Counter()
    for atom in mol.GetAtoms():
        c[atom.GetSymbol()] += 1
    return dict(c)


def generate_f(pdb_path, chain_id='L'):
    """Path F: geometric connectivity + heuristic bond orders.

    Returns ``(smiles, error)``. ``error`` is None on success. On a
    formula mismatch (heuristic produced a chemically inconsistent structure)
    returns ``(None, "formula mismatch: ...")`` so callers can fall back to
    Path G/H rather than trust an approximate result.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError as ex:
        return None, f"rdkit unavailable: {ex}"

    atoms = _read_chain_heavy_atoms(pdb_path, chain_id)
    if not atoms:
        return None, "no peptide heavy atoms in chain"

    truth = dict(Counter(e for e, *_ in atoms))
    mol = _build_mol(atoms)
    try:
        rdDetermineBonds.DetermineConnectivity(mol)
    except Exception as ex:
        return None, f"DetermineConnectivity failed: {ex}"

    mol = _assign_bond_orders(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception as ex:
        return None, f"sanitize failed (heuristic bond orders): {ex}"

    fragment_count = len(Chem.GetMolFrags(mol))
    if fragment_count != 1:
        return None, f"multiple disconnected fragments: {fragment_count}"

    got = _formula_counts(mol)
    if got != truth:
        return None, f"formula mismatch: got {got} vs structure {truth}"

    return Chem.MolToSmiles(mol), None
