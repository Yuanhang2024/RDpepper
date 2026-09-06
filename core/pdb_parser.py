"""PDB residue sequence parser and structure analysis."""
from functools import lru_cache
import threading
from typing import Optional, Dict, List
from rdkit import Chem

from .pdb_utils import read_first_model_lines


_AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


@lru_cache(maxsize=128)
def standard_pdb_atom_name_map(residue_name, template_smiles):
    """Map canonical PDB heavy-atom names to indices in a standard-AA template.

    RDKit's sequence builder carries standard PDB atom names.  Removing its
    terminal OXT yields a residue graph that can be matched to the package
    template without relying on SMILES atom order.  Formal charges and
    implicit-hydrogen flags are neutralized only on disposable matching copies;
    the returned indices refer to the original, chemically intact template.
    """
    one_letter = _AA_3TO1.get(str(residue_name).upper())
    if not one_letter:
        return {}
    reference = Chem.MolFromSequence(one_letter)
    template = Chem.MolFromSmiles(template_smiles)
    if reference is None or template is None:
        return {}
    editable = Chem.RWMol(reference)
    oxt_indices = [
        atom.GetIdx()
        for atom in editable.GetAtoms()
        if atom.GetPDBResidueInfo()
        and atom.GetPDBResidueInfo().GetName().strip() == 'OXT'
    ]
    for index in sorted(oxt_indices, reverse=True):
        editable.RemoveAtom(index)
    reference = editable.GetMol()
    match_reference = Chem.Mol(reference)
    match_template = Chem.Mol(template)
    for molecule in (match_reference, match_template):
        for atom in molecule.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
    match = match_template.GetSubstructMatch(match_reference, useChirality=False)
    if len(match) != match_reference.GetNumAtoms():
        return {}
    mapping = {}
    for reference_index, template_index in enumerate(match):
        info = reference.GetAtomWithIdx(reference_index).GetPDBResidueInfo()
        if info is not None:
            mapping[info.GetName().strip()] = int(template_index)
    # RDKit's sequence residue template attaches CD1 to the atom labelled CG2
    # for isoleucine.  Canonical PDB naming attaches CD1 to CG1, so swap the
    # two gamma labels before this map is used for coordinates or CONECT edges.
    if str(residue_name).upper() == 'ILE' and {'CG1', 'CG2'} <= mapping.keys():
        mapping['CG1'], mapping['CG2'] = mapping['CG2'], mapping['CG1']
    if str(residue_name).upper() == 'ARG' and {
        'NE', 'CZ', 'NH1', 'NH2'
    } <= mapping.keys():
        cz_index = mapping['CZ']
        ne_index = mapping['NE']
        terminal_nitrogens = [
            atom
            for atom in template.GetAtomWithIdx(cz_index).GetNeighbors()
            if atom.GetSymbol() == 'N' and atom.GetIdx() != ne_index
        ]
        neutral_single = [
            atom.GetIdx()
            for atom in terminal_nitrogens
            if atom.GetFormalCharge() == 0
            and template.GetBondBetweenAtoms(
                cz_index, atom.GetIdx()
            ).GetBondType() == Chem.BondType.SINGLE
        ]
        cationic_double = [
            atom.GetIdx()
            for atom in terminal_nitrogens
            if atom.GetFormalCharge() > 0
            and template.GetBondBetweenAtoms(
                cz_index, atom.GetIdx()
            ).GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(neutral_single) == 1 and len(cationic_double) == 1:
            mapping['NH1'] = neutral_single[0]
            mapping['NH2'] = cationic_double[0]
    return mapping


def parse_backbone(mol):
    """Find backbone atom indices (N, CA, CB, C, O) in an RDKit molecule."""
    # Search the complete N-CA-C(=O) motif.  Selecting the first nitrogen and
    # the last carbonyl independently misidentifies acidic side-chain carbonyls.
    candidates = []
    for n_atom in mol.GetAtoms():
        if n_atom.GetSymbol() != 'N':
            continue
        for ca_atom in n_atom.GetNeighbors():
            if ca_atom.GetSymbol() != 'C':
                continue
            for c_atom in ca_atom.GetNeighbors():
                if c_atom.GetSymbol() != 'C' or c_atom.GetIdx() == n_atom.GetIdx():
                    continue
                double_oxygens = [
                    neighbor
                    for neighbor in c_atom.GetNeighbors()
                    if neighbor.GetSymbol() == 'O'
                    and mol.GetBondBetweenAtoms(
                        c_atom.GetIdx(), neighbor.GetIdx()
                    ).GetBondType() == Chem.BondType.DOUBLE
                ]
                if not double_oxygens:
                    continue
                candidates.append((n_atom, ca_atom, c_atom, double_oxygens[0]))
    if not candidates:
        return None, None, None, None, None

    # A monomer should expose one peptide backbone.  This deterministic order
    # is only a tie-breaker for malformed or unusual rows; their later mapping
    # and chemistry audits still fail closed when the assignment is ambiguous.
    n_atom, ca_atom, c_atom, o_atom = min(
        candidates,
        key=lambda item: tuple(atom.GetIdx() for atom in item),
    )
    backbone = {n_atom.GetIdx(), c_atom.GetIdx()}
    cb_candidates = [
        neighbor.GetIdx()
        for neighbor in ca_atom.GetNeighbors()
        if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() not in backbone
    ]
    cb_idx = min(cb_candidates) if cb_candidates else None
    return (
        n_atom.GetIdx(),
        ca_atom.GetIdx(),
        cb_idx,
        c_atom.GetIdx(),
        o_atom.GetIdx(),
    )


def get_res_seq(pdb, chain_id='L', include_het=True):
    """Extract ordered residue sequence from a PDB file.

    include_het=True (default) keeps all residues, including non-standard
    HETATM residues -- this is the historical behavior relied on by Path A/B/C
    and by cyclization detection (to match LINK endpoints). Water and common
    ions are always dropped. include_het=False restricts the list to standard
    amino-acid ATOM residues.
    """
    _SKIP = {'HOH', 'WAT', 'DOD', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'SO4', 'PO4'}
    by_identity = {}
    for line in read_first_model_lines(str(pdb)):
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        if line[21] != chain_id:
            continue
        rname = line[17:20].strip()
        rnum = int(line[22:26])
        icode = line[26:27].strip()
        het = line.startswith('HETATM')
        if rname in _SKIP:
            continue
        identity = (rname, rnum, icode)
        row = by_identity.setdefault(identity, {
            'name': rname,
            'num': rnum,
            'icode': icode,
            'has_atom': False,
            'has_hetatm': False,
        })
        row['has_hetatm' if het else 'has_atom'] = True
    res = []
    for row in by_identity.values():
        if not include_het and not row['has_atom']:
            continue
        het = not row['has_atom']
        key = (
            (row['name'], row['num'], het, row['icode'])
            if row['icode']
            else (row['name'], row['num'], het)
        )
        res.append({
            'key': key,
            'name': row['name'],
            'num': row['num'],
            'icode': row['icode'],
            'het': het,
            'record_types': [
                record_type
                for record_type, present in (
                    ('ATOM', row['has_atom']),
                    ('HETATM', row['has_hetatm']),
                )
                if present
            ],
        })
    res.sort(key=lambda x: (x['num'], x.get('icode', '')))
    return res


_RESIDUE_ATOM_INDEX_CACHE: dict = {}
_RESIDUE_ATOM_INDEX_ORDER: list = []
_RESIDUE_ATOM_INDEX_MAX = 16
_RESIDUE_ATOM_INDEX_LOCK = threading.RLock()


def _clear_residue_atom_index_cache() -> None:
    with _RESIDUE_ATOM_INDEX_LOCK:
        _RESIDUE_ATOM_INDEX_CACHE.clear()
        _RESIDUE_ATOM_INDEX_ORDER.clear()


def _residue_atom_index(pdb, chain_id):
    """Parse one chain's first-model heavy atoms grouped by residue key."""
    index: dict = {}
    for line in read_first_model_lines(str(pdb)):
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        if line[21] != chain_id:
            continue
        rname = line[17:20].strip()
        try:
            rnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26:27].strip()
        name = line[12:16].strip()
        elem = line[76:78].strip() if len(line) > 76 else ''
        if not elem:
            from .cyclization import _element_from_name
            elem = _element_from_name(line[12:16])
        elem = elem.upper()
        if elem == 'H':
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError):
            continue
        index.setdefault((rname, rnum, icode), []).append({
            'num': int(line[6:11].strip()),
            'name': name,
            'elem': elem,
            'xyz': xyz,
        })
    for atoms in index.values():
        atoms.sort(key=lambda x: x['num'])
    return index


def _cached_residue_atom_index(pdb, chain_id):
    import os

    try:
        stat = os.stat(pdb)
        key = (
            os.path.normcase(os.path.abspath(str(pdb))),
            chain_id,
            stat.st_mtime_ns,
            stat.st_size,
        )
    except OSError:
        return None, _residue_atom_index(pdb, chain_id)
    with _RESIDUE_ATOM_INDEX_LOCK:
        cached = _RESIDUE_ATOM_INDEX_CACHE.get(key)
        if cached is not None:
            _RESIDUE_ATOM_INDEX_ORDER.remove(key)
            _RESIDUE_ATOM_INDEX_ORDER.append(key)
            return key, cached
    index = _residue_atom_index(pdb, chain_id)
    with _RESIDUE_ATOM_INDEX_LOCK:
        cached = _RESIDUE_ATOM_INDEX_CACHE.get(key)
        if cached is None:
            _RESIDUE_ATOM_INDEX_CACHE[key] = index
            _RESIDUE_ATOM_INDEX_ORDER.append(key)
            while len(_RESIDUE_ATOM_INDEX_ORDER) > (
                _RESIDUE_ATOM_INDEX_MAX
            ):
                stale = _RESIDUE_ATOM_INDEX_ORDER.pop(0)
                _RESIDUE_ATOM_INDEX_CACHE.pop(stale, None)
            cached = _RESIDUE_ATOM_INDEX_CACHE[key]
        else:
            _RESIDUE_ATOM_INDEX_ORDER.remove(key)
            _RESIDUE_ATOM_INDEX_ORDER.append(key)
        return key, cached


def get_pdb_atoms(pdb, rkey, chain_id='L'):
    """Get all heavy atoms for a specific residue key from a PDB file.

    Route portfolios ask per-residue atom sets hundreds of times per entity
    (observed 1,451 full-file scans on one 77-residue structure); the chain
    is parsed once per file identity and served from a residue-keyed index.
    Fresh dict copies keep the historical mutate-your-result semantics.
    """
    if len(rkey) not in (3, 4):
        raise ValueError(f"invalid residue key: {rkey!r}")
    expected_name = str(rkey[0])
    expected_number = int(rkey[1])
    expected_icode = str(rkey[3]).strip() if len(rkey) == 4 else ''
    _, index = _cached_residue_atom_index(pdb, chain_id)
    atoms = index.get((expected_name, expected_number, expected_icode), [])
    return [dict(atom) for atom in atoms]


def read_conect(pdb):
    """Parse CONECT records from a PDB file."""
    conect = {}
    for line in read_first_model_lines(str(pdb)):
        if line.startswith('CONECT'):
            parts = line.strip().split()
            source = int(parts[1])
            targets = conect.setdefault(source, [])
            for token in parts[2:]:
                target = int(token)
                if target not in targets:
                    targets.append(target)
    return conect


def parse_chain_sequence(pdb_path, chain_id):
    """Extract ordered residue sequence from an arbitrary chain."""
    return get_res_seq(pdb_path, chain_id, include_het=True)


def detect_cyclization(residues):
    """Detect cyclization type from chain L residue composition."""
    names = [r['name'] for r in residues]
    n = len(residues)
    has_ace = 'ACE' in names
    has_nme = 'NME' in names
    cys_count = names.count('CYS')

    if n >= 2 and not has_ace and not has_nme:
        if cys_count >= 3:
            return 'head-to-tail + disulfide'
        return 'head-to-tail'
    elif has_ace and has_nme and n >= 4:
        if cys_count >= 2:
            return 'linear (capped) + disulfide'
        return 'linear (capped)'
    elif has_ace or has_nme:
        return 'linear (capped) — possibly cyclic via other'
    elif cys_count >= 2:
        return 'disulfide only'
    return 'linear'


# ── Ported from mirror: topology classifier + LINK parser (Lite was missing these) ─

def classify_topology_from_records(residues=None, conect: Optional[Dict[int, List[int]]] = None,
                      links: Optional[List[str]] = None, pdb_path: Optional[str] = None,
                      chain_id: str = 'L', target_chain: str = 'R') -> str:
    """Classify cyclization topology from pre-parsed CONECT/LINK records or residues.

    Low-level classifier returning a topology *string*. Distinct from
    cyclization.detect_cyclization, which is the high-level entry point taking
    only (pdb_path, chain_id) and returning a rich CyclizationInfo object. Use
    this when the caller has already parsed residues/conect/links (e.g.
    pipeline.run_batch) and only needs the topology label.

    Priority: CONECT cross-chain bonds > LINK records > residue heuristics.

    Args:
        residues: list of dicts with 'name', 'num', 'het' keys (chain L)
        conect: {atom_num: [connected_atom_nums]} from read_conect()
        links: list of LINK record strings from parse_link_record()
        pdb_path: path to PDB file (used to build atom→residue mapping if conect given)
        chain_id: peptide chain ID (default 'L')
        target_chain: target protein chain ID (default 'R')

    Returns:
        One of: 'lactam', 'isopeptide', 'disulfide', 'head_to_tail',
        'sidechain_to_tail', 'bicyclic', 'multicyclic', 'linear', 'linear_capped'
    """
    # ── Method 1: CONECT record analysis (most reliable) ──
    if conect and residues and pdb_path:
        result = _detect_from_conect(conect, residues, pdb_path, chain_id, target_chain)
        if result and result != 'linear':
            return result

    # ── Method 2: LINK record analysis ──
    if links:
        result = _detect_from_links(links, chain_id)
        if result and result != 'linear':
            return result

    # ── Method 3: residue composition heuristic (fallback) ──
    if residues:
        return _detect_from_residues(residues)

    return 'linear'


def _detect_from_conect(
    conect: Dict[int, List[int]],
    residues: List[Dict],
    pdb_path: str,
    chain_id: str = 'L',
    target_chain: str = 'R',
) -> str:
    """Detect cyclization from CONECT cross-chain/peptide-internal bonds.

    Strategy:
    1. Build atom→(chain_id, residue_key, atom_name) mapping for all chains
    2. Find CONECT pairs where both atoms are in the peptide chain (chain_id)
       AND the residue gap |i-j| ≥ 2 → cyclization bond
    3. Classify by atom names of the bonded pair
    """
    # Build atom→residue mapping
    atom2info = {}  # atom_num → (chain_id, rname, rnum, atom_name)

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            ch = line[21]
            if ch not in (chain_id, target_chain):
                continue
            atom_num = int(line[6:11].strip())
            rname = line[17:20].strip()
            rnum = int(line[22:26])
            aname = line[12:16].strip()
            atom2info[atom_num] = (ch, rname, rnum, aname)

    # Find peptide-internal CONECT bonds (skip backbone sequential bonds)
    pep_bonds = []  # (atom1, atom2, residue_gap)
    for src, targets in conect.items():
        if src not in atom2info:
            continue
        src_ch, src_name, src_num, src_aname = atom2info[src]
        if src_ch != chain_id:
            continue
        for tgt in targets:
            if tgt not in atom2info:
                continue
            tgt_ch, tgt_name, tgt_num, tgt_aname = atom2info[tgt]
            if tgt_ch != chain_id:
                continue
            gap = abs(src_num - tgt_num)
            if gap >= 2:  # not sequential backbone (gap 0-1 is same or adjacent residue)
                pep_bonds.append((src_aname, tgt_aname, gap, src_num, tgt_num))

    if not pep_bonds:
        return 'linear'

    # Classify bonds
    disulfide_bonds = []
    amide_bonds = []
    ester_bonds = []
    thioether_bonds = []

    for a1, a2, gap, n1, n2 in pep_bonds:
        a1u = a1.upper()
        a2u = a2.upper()
        # Disulfide: SG-SG
        if 'SG' in (a1u, a2u):
            disulfide_bonds.append((a1, a2, gap, n1, n2))
        # Ester: sidechain oxygen (OG/OD/OE) to backbone carbon (C)
        elif (a1u.startswith('O') and a2u == 'C') or (a2u.startswith('O') and a1u == 'C'):
            ester_bonds.append((a1, a2, gap, n1, n2))
        # Thioether: sidechain sulfur to carbon
        elif 'SD' in (a1u, a2u) or a1u.startswith('S') or a2u.startswith('S'):
            thioether_bonds.append((a1, a2, gap, n1, n2))
        # Amide/Isopeptide/Lactam: sidechain N (NZ,ND,NE) to C, or OG to C
        else:
            amide_bonds.append((a1, a2, gap, n1, n2))

    n_bonds = len(pep_bonds)
    n_disulfide = len(disulfide_bonds)

    # Multi-bond classification
    if n_bonds >= 3:
        return 'multicyclic'
    if n_bonds >= 2:
        if n_disulfide >= 1:
            return 'bicyclic'  # includes disulfide + other bridge
        return 'bicyclic'

    # Single bond classification
    a1, a2, gap, n1, n2 = pep_bonds[0]
    a1u, a2u = a1.upper(), a2.upper()

    # Simplified detection: large gap + backbone atom involvement → head-to-tail
    if gap >= len(residues) * 0.8:  # bond spans most of the peptide
        if a1u == 'C' or a2u == 'C':
            return 'head_to_tail'

    # Disulfide
    if n_disulfide >= 1:
        return 'disulfide'

    # Ester → lactone
    if ester_bonds:
        return 'lactone'

    # Thioether
    if thioether_bonds:
        return 'sidechain_to_tail'

    # Amide/Isopeptide/Lactam
    n_involved = any(a.upper().startswith('N') and a.upper() not in ('N',)
                     for a in (a1, a2))
    if n_involved or amide_bonds:
        return 'lactam'  # sidechain amide bridge

    return 'sidechain_to_tail'


def _detect_from_links(links: List[str], chain_id: str = 'L') -> str:
    """Detect cyclization from LINK records.

    Example: LINK N GLN A 1 C GLY A 7 1.23 → head-to-tail (N-term to C-term)
    """
    for link in links:
        parts = link.strip().split()
        if len(parts) < 10:
            continue
        try:
            atom1 = parts[1]
            res1 = parts[2]
            ch1 = parts[3]
            seq1 = int(parts[4])
            atom2 = parts[5]
            res2 = parts[6]
            ch2 = parts[7]
            seq2 = int(parts[8])

            # Accept any chain (not just chain_id) — Scaffold uses 'A', CPBind uses 'L'
            if ch1 != ch2:
                continue
            # N-terminal atom → C-terminal atom = head-to-tail
            if atom1.upper() == 'N' and atom2.upper() == 'C' and abs(seq2 - seq1) >= 2:
                return 'head_to_tail'
            # SG → SG = disulfide
            if atom1.upper() == 'SG' and atom2.upper() == 'SG':
                return 'disulfide'
            # Sidechain N → C = lactam/isopeptide
            if (atom1.upper() in ('NZ', 'ND', 'NE') and atom2.upper() == 'C'):
                return 'lactam'
            if (atom2.upper() in ('NZ', 'ND', 'NE') and atom1.upper() == 'C'):
                return 'lactam'
        except (ValueError, IndexError):
            continue

    return 'linear'


def _detect_from_residues(residues: List[Dict]) -> str:
    """Detect cyclization type from residue composition (heuristic fallback)."""
    names = [r['name'] for r in residues]
    n = len(residues)
    has_ace = 'ACE' in names
    has_nme = 'NME' in names
    cys_count = names.count('CYS')

    if n >= 2 and not has_ace and not has_nme:
        if cys_count >= 3:
            return 'head_to_tail + disulfide'
        return 'head_to-tail'
    elif has_ace and has_nme and n >= 4:
        if cys_count >= 2:
            return 'linear (capped) + disulfide'
        return 'linear_capped'
    elif has_ace or has_nme:
        return 'linear_capped'
    elif cys_count >= 2:
        return 'disulfide'
    return 'linear'


def parse_link_record(pdb_path: str) -> List[str]:
    """Extract LINK records from a PDB file.

    Returns:
        List of full LINK record lines
    """
    links = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('LINK'):
                links.append(line.rstrip())
    return links


def get_capping(residues):
    """Extract capping group info from chain L residues."""
    n_term = None
    c_term = None
    for r in residues:
        if r['name'] == 'ACE':
            n_term = 'ACE'
        if r['name'] == 'NME':
            c_term = 'NME'
    return n_term, c_term


def get_het_capping_by_conect(pdb_path, residues, chain_id='L'):
    """Detect HETATM capping groups connected via CONECT to standard residues."""
    conect = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('CONECT'):
                parts = line.strip().split()
                conect[int(parts[1])] = [int(x) for x in parts[2:]]

    atom2key = {}
    for r in residues:
        for line in open(pdb_path):
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            if line[21] != chain_id:
                continue
            rname_l = line[17:20].strip()
            rnum_l = int(line[22:26])
            het_l = line.startswith('HETATM')
            if (rname_l, rnum_l, het_l) == r['key']:
                atom2key[int(line[6:11].strip())] = r['key']

    n_term_het = None
    c_term_het = None
    for src, targets in conect.items():
        if src not in atom2key:
            continue
        src_key = atom2key[src]
        if not src_key[2]:
            continue
        for tgt in targets:
            if tgt not in atom2key:
                continue
            tgt_key = atom2key[tgt]
            if tgt_key[2]:
                continue
            het_res = [r for r in residues if r['key'] == src_key]
            if het_res:
                het_name = het_res[0]['name']
                if het_name == 'NME' or 'NME' in het_name.upper():
                    c_term_het = het_name
                elif het_name == 'ACE' or 'ACE' in het_name.upper():
                    n_term_het = het_name

    return n_term_het, c_term_het
