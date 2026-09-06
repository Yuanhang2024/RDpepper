"""PDB SSBOND and LINK record parsers.

Provides data-driven cyclization detection by reading cross-link records
from PDB files instead of heuristic-based guessing.
"""
import os
import re
import math
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .geometry_params import (
    DEFAULT_RADIUS_MULTIPLIER,
    DEFAULT_DISTANCE_CEILING,
    NONDEFAULT_GEOMETRY_PARAMS,
    is_nondefault_geometry,
    resolve_geometry_params,
)
from .pdb_utils import read_first_model_lines


# ══════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CyclizationBond:
    """Single cyclization bond in a peptide."""
    bond_type: str  # 'disulfide', 'thioether', 'ester', 'isopeptide', 'peptide', 'unknown'
    pos1: int  # 1-indexed residue position in chain
    pos2: int
    rgroup1: str  # 'R1', 'R2', 'R3'
    rgroup2: str
    atom1: Optional[str] = None  # PDB atom name (e.g., 'SG', 'NZ')
    atom2: Optional[str] = None
    res1: Optional[str] = None  # PDB residue name
    res2: Optional[str] = None
    evidence_source: str = 'unknown'  # ssbond, link, conect, geometry, or unknown


@dataclass
class CyclizationInfo:
    """Complete cyclization topology of a peptide."""
    topology: str  # 'linear', 'monocyclic', 'bicyclic', 'tricyclic', 'polycyclic'
    bonds: List[CyclizationBond]
    description: str  # Human-readable description
    geometry_radius_multiplier: Optional[float] = None
    geometry_distance_ceiling: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# PDB Record Parsers
# ══════════════════════════════════════════════════════════════════════════

def read_ssbond(pdb_path: str) -> List[Dict]:
    """Parse SSBOND records from PDB.

    SSBOND format:
        SSBOND   1 CYS A    3    CYS A   17                          1555   1555  2.03

    Returns list of dicts:
        [{'chain1': 'A', 'res1': 'CYS', 'num1': 3,
          'chain2': 'A', 'res2': 'CYS', 'num2': 17}, ...]
    """
    ssbonds = []
    for line in read_first_model_lines(str(pdb_path)):
        if not line.startswith('SSBOND'):
            continue
        try:
            res1 = line[11:14].strip()
            chain1 = line[15] if len(line) > 15 else ' '
            num1 = int(line[17:21].strip())
            icode1 = line[21:22].strip()
            res2 = line[25:28].strip()
            chain2 = line[29] if len(line) > 29 else ' '
            num2 = int(line[31:35].strip())
            icode2 = line[35:36].strip()
            ssbonds.append({
                'chain1': chain1,
                'res1': res1,
                'num1': num1,
                'icode1': icode1,
                'chain2': chain2,
                'res2': res2,
                'num2': num2,
                'icode2': icode2,
            })
        except (ValueError, IndexError):
            continue
    return ssbonds


def read_link(pdb_path: str) -> List[Dict]:
    """Parse LINK records from PDB.

    LINK format:
        LINK         SG  CYS A   4                 C   DAL A  10     1555   1555  1.78

    Returns list of dicts:
        [{'atom1': 'SG', 'res1': 'CYS', 'chain1': 'A', 'num1': 4,
          'atom2': 'C', 'res2': 'DAL', 'chain2': 'A', 'num2': 10}, ...]
    """
    links = []
    for line in read_first_model_lines(str(pdb_path)):
        if not line.startswith('LINK'):
            continue
        try:
            atom1 = line[12:16].strip()
            res1 = line[17:20].strip()
            chain1 = line[21] if len(line) > 21 else ' '
            num1 = int(line[22:26].strip())
            icode1 = line[26:27].strip()
            atom2 = line[42:46].strip()
            res2 = line[47:50].strip()
            chain2 = line[51] if len(line) > 51 else ' '
            num2 = int(line[52:56].strip())
            icode2 = line[56:57].strip()
            links.append({
                'atom1': atom1,
                'res1': res1,
                'chain1': chain1,
                'num1': num1,
                'icode1': icode1,
                'atom2': atom2,
                'res2': res2,
                'chain2': chain2,
                'num2': num2,
                'icode2': icode2,
            })
        except (ValueError, IndexError):
            continue
    return links


# ══════════════════════════════════════════════════════════════════════════
# Covalent-radius bond model (element-based, valence-aware)
# ══════════════════════════════════════════════════════════════════════════

# Covalent radii in Angstrom (Cordero et al. 2008, single-bond).
_COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
    'P': 1.07, 'SE': 1.20, 'F': 0.57, 'CL': 1.02, 'BR': 1.20,
    'I': 1.39, 'B': 0.84, 'SI': 1.11, 'ZN': 1.22, 'NA': 1.66,
}
# Tolerance multiplier: a bond exists if d <= (r1 + r2) * (1 + TOL).
# 0.30 absorbs PDB coordinate noise and relaxed geometry while staying well
# below non-bonded contact distances (which are ~ sum of *van der Waals* radii,
# roughly 2x covalent).
_BOND_TOL = DEFAULT_RADIUS_MULTIPLIER - 1.0
# Hard ceiling so two large atoms (e.g. S-S = 2.1*1.3 = 2.73) never run away.
_MAX_BOND_DIST = DEFAULT_DISTANCE_CEILING


def _element_from_name(atom_name: str) -> str:
    """Infer element symbol from a PDB atom name when cols 77-78 are blank.

    PDB atom names are right-justified with the element in the first 1-2 chars
    of the 4-col field; for ATOM records the element is usually the leading
    alpha character(s). Two-letter elements (CL, BR, SE, ZN, FE) are checked
    first, then single-letter.
    """
    field = atom_name[:4].ljust(4)
    n = field.strip().upper()
    if not n:
        return ''
    # A leading blank is the PDB convention for a one-letter element (e.g.
    # `` CA `` is alpha-carbon), while a two-letter element is left aligned
    # (e.g. ``CA  `` is calcium).  Leading digits in hydrogen names behave as
    # the one-letter form.
    one_letter_alignment = field[0].isspace() or field[0].isdigit()
    n = n.lstrip('0123456789')
    if not n:
        return ''
    if one_letter_alignment:
        return n[0]
    two = n[:2]
    if two in ('CL', 'BR', 'SE', 'ZN', 'FE', 'NA', 'SI', 'MG', 'MN', 'CA'):
        return two
    return n[0]


def _is_covalent_bond(
    a: Dict, b: Dict, *,
    radius_multiplier: float = DEFAULT_RADIUS_MULTIPLIER,
    distance_ceiling: float = DEFAULT_DISTANCE_CEILING,
) -> bool:
    """True if atoms a, b are within covalent-bond distance for their elements.

    Uses summed covalent radii with a tolerance, NOT a single global cutoff,
    so S-S (long) and C-N (short) are each judged on their own scale and
    hydrogen bonds / van der Waals contacts are excluded.
    """
    ea = str(a['elem']).strip().upper()
    eb = str(b['elem']).strip().upper()
    ra = _COVALENT_RADII.get(ea)
    rb = _COVALENT_RADII.get(eb)
    if ra is None or rb is None:
        return False  # unknown element: don't guess
    d = math.dist(a['xyz'], b['xyz'])
    if d < 0.4:
        return False  # overlapping/duplicate atoms, not a real bond
    cutoff = min((ra + rb) * radius_multiplier, distance_ceiling)
    return d <= cutoff


def _parse_atoms_for_chain(pdb_path: str, chain_id: str) -> Dict[int, Dict]:
    """Read ATOM/HETATM records for one chain.

    Returns {serial: {'name','resn','resseq','het','elem','xyz'}} so that CONECT
    serials and covalent-radius distance checks can be resolved to (residue,
    atom, element).
    """
    atoms: Dict[int, Dict] = {}
    for line in read_first_model_lines(str(pdb_path)):
        if line[:6] not in ('ATOM  ', 'HETATM'):
            continue
        if line[21] != chain_id:
            continue
        try:
            serial = int(line[6:11])
            name = line[12:16].strip()
            resn = line[17:20].strip()
            resseq = int(line[22:26])
            icode = line[26:27].strip()
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        elem = line[76:78].strip() if len(line) >= 78 else ''
        if not elem:
            elem = _element_from_name(line[12:16])
        atoms[serial] = {
            'name': name, 'resn': resn, 'resseq': resseq,
            'icode': icode, 'resid': (resseq, icode),
            'het': line.startswith('HETATM'), 'elem': elem.upper(),
            'xyz': (x, y, z),
        }
    return atoms


_ATOM_CACHE: Dict[tuple, Dict[int, Dict]] = {}
_ATOM_CACHE_ORDER: List[tuple] = []
_ATOM_CACHE_MAX = 32
_ATOM_CACHE_LOCK = threading.RLock()


def _copy_atoms(atoms: Dict[int, Dict]) -> Dict[int, Dict]:
    return {serial: dict(atom) for serial, atom in atoms.items()}


def _clear_atom_parse_cache() -> None:
    with _ATOM_CACHE_LOCK:
        _ATOM_CACHE.clear()
        _ATOM_CACHE_ORDER.clear()


def read_atoms(pdb_path: str, chain_id: str) -> Dict[int, Dict]:
    """Read ATOM/HETATM records for one chain (memoized per file identity).

    Cyclization and bond detection re-read the same normalized PDB many
    times per entity (routes, models, retries); the cache keys on path,
    chain, mtime, and size, so a rewritten file never serves stale atoms.
    Callers treat the returned records as read-only.
    """
    try:
        stat = os.stat(pdb_path)
        key = (os.path.normcase(os.path.abspath(pdb_path)), chain_id,
               stat.st_mtime_ns, stat.st_size)
    except OSError:
        return _parse_atoms_for_chain(pdb_path, chain_id)
    with _ATOM_CACHE_LOCK:
        cached = _ATOM_CACHE.get(key)
        if cached is not None:
            _ATOM_CACHE_ORDER.remove(key)
            _ATOM_CACHE_ORDER.append(key)
            return _copy_atoms(cached)
    atoms = _parse_atoms_for_chain(pdb_path, chain_id)
    with _ATOM_CACHE_LOCK:
        cached = _ATOM_CACHE.get(key)
        if cached is None:
            _ATOM_CACHE[key] = _copy_atoms(atoms)
            _ATOM_CACHE_ORDER.append(key)
            while len(_ATOM_CACHE_ORDER) > _ATOM_CACHE_MAX:
                stale = _ATOM_CACHE_ORDER.pop(0)
                _ATOM_CACHE.pop(stale, None)
            cached = _ATOM_CACHE[key]
        else:
            _ATOM_CACHE_ORDER.remove(key)
            _ATOM_CACHE_ORDER.append(key)
        return _copy_atoms(cached)


def _parse_conect(pdb_path: str) -> List[tuple]:
    """Parse CONECT records into a list of (serial_a, serial_b) atom-pair bonds.

    CONECT format: columns of 5-char atom serials, first is the source atom,
    the rest are bonded neighbours. Duplicate/undirected pairs are de-duped.
    """
    pairs = set()
    for line in read_first_model_lines(str(pdb_path)):
        if not line.startswith('CONECT'):
            continue
        body = line.rstrip('\n')
        nums = []
        for i in range(6, len(body), 5):
            tok = body[i:i + 5].strip()
            if tok:
                try:
                    nums.append(int(tok))
                except ValueError:
                    pass
        if len(nums) < 2:
            continue
        a = nums[0]
        for b in nums[1:]:
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


_CONECT_CACHE: Dict[tuple, List[tuple]] = {}
_CONECT_CACHE_ORDER: List[tuple] = []
_CONECT_CACHE_MAX = 32
_CONECT_CACHE_LOCK = threading.RLock()


def _clear_conect_parse_cache() -> None:
    with _CONECT_CACHE_LOCK:
        _CONECT_CACHE.clear()
        _CONECT_CACHE_ORDER.clear()


def read_conect(pdb_path: str) -> List[tuple]:
    """Parse CONECT records (memoized per file identity, fresh list per call)."""
    try:
        stat = os.stat(pdb_path)
        key = (os.path.normcase(os.path.abspath(pdb_path)),
               stat.st_mtime_ns, stat.st_size)
    except OSError:
        return _parse_conect(pdb_path)
    with _CONECT_CACHE_LOCK:
        cached = _CONECT_CACHE.get(key)
        if cached is not None:
            _CONECT_CACHE_ORDER.remove(key)
            _CONECT_CACHE_ORDER.append(key)
            return list(cached)
    pairs = _parse_conect(pdb_path)
    with _CONECT_CACHE_LOCK:
        cached = _CONECT_CACHE.get(key)
        if cached is None:
            _CONECT_CACHE[key] = list(pairs)
            _CONECT_CACHE_ORDER.append(key)
            while len(_CONECT_CACHE_ORDER) > _CONECT_CACHE_MAX:
                stale = _CONECT_CACHE_ORDER.pop(0)
                _CONECT_CACHE.pop(stale, None)
            cached = _CONECT_CACHE[key]
        else:
            _CONECT_CACHE_ORDER.remove(key)
            _CONECT_CACHE_ORDER.append(key)
        return list(cached)


# ══════════════════════════════════════════════════════════════════════════
# Geometric / CONECT bond inference (fallback when SSBOND/LINK are absent)
# ══════════════════════════════════════════════════════════════════════════

_CAPS = {'ACE', 'NME', 'NH2'}


# Backbone atom names (used to tell main-chain from side-chain attachment).
_BACKBONE_N = {'N'}
_BACKBONE_C = {'C'}            # carbonyl carbon
_BACKBONE_OTHER = {'CA', 'O', 'OXT', 'H', 'HA'}
# Heavy elements that can form cyclization crosslinks (exclude H).
_LINKABLE_ELEMS = {'C', 'N', 'O', 'S', 'P', 'SE'}


def _rgroup_for_atom(atom: Dict, pos: int, n_res: int) -> str:
    """Assign R-group for an atom participating in a crosslink.

    R1 = backbone alpha-amine (N-terminus side), R2 = backbone carbonyl
    (C-terminus side), R3 = any side-chain attachment.
    """
    name = atom['name']
    if name in _BACKBONE_N:
        return 'R1'
    if name in _BACKBONE_C:
        return 'R2'
    return 'R3'  # side-chain


def _bond_type_from_elems(a: Dict, b: Dict, rg1: str, rg2: str) -> str:
    """Name the bond chemically from the element pair and attachment context."""
    ea, eb = a['elem'], b['elem']
    pair = frozenset((ea, eb))
    backbone = (rg1 in ('R1', 'R2')) and (rg2 in ('R1', 'R2'))
    if pair == frozenset(('S', 'S')):
        return 'disulfide'
    if backbone and pair == frozenset(('C', 'N')):
        return 'peptide'           # head-to-tail main-chain amide
    if pair == frozenset(('C', 'N')):
        return 'isopeptide'        # side-chain amide (Lys-Asp/Glu etc.)
    if pair == frozenset(('C', 'O')):
        return 'ester'             # Ser/Thr-O to carbonyl
    if pair == frozenset(('C', 'S')) or pair == frozenset(('S', 'C')):
        return 'thioether'         # Cys-S to carbon
    if 'S' in pair:
        return 'thioether'
    return 'crosslink'             # generic but real covalent crosslink


def _classify_atom_pair(a: Dict, b: Dict, n_res: int,
                        pos_of: Dict[int, int]) -> Optional[CyclizationBond]:
    """Turn a bonded atom pair (a, b) into a CyclizationBond, or None if it is
    not a cyclization bond (same residue, adjacent backbone, cap, or H).

    Element/valence-based: any heavy-atom crosslink between two *different*
    residues is a cyclization bond, classified chemically by element pair and
    backbone-vs-side-chain attachment. This generalizes beyond a fixed atom-name
    table so non-standard NNAA side chains are covered.
    """
    # Must be a cross-residue bond
    if a['resid'] == b['resid']:
        return None
    # Skip hydrogens and unknown elements
    if a['elem'] not in _LINKABLE_ELEMS or b['elem'] not in _LINKABLE_ELEMS:
        return None
    # Skip cap residues (terminal modification, not cyclization)
    if a['resn'] in _CAPS or b['resn'] in _CAPS:
        return None
    # Both residues must be part of the parsed chain sequence
    def position(atom):
        exact = pos_of.get(atom['resid'])
        if exact is not None:
            return exact
        if not atom.get('icode'):
            return pos_of.get(atom['resseq'])
        return None

    pos1, pos2 = position(a), position(b)
    if pos1 is None or pos2 is None:
        return None
    # Ignore backbone-only "other" atoms (CA, O) that aren't attachment points
    if a['name'] in _BACKBONE_OTHER or b['name'] in _BACKBONE_OTHER:
        return None

    rg1 = _rgroup_for_atom(a, pos1, n_res)
    rg2 = _rgroup_for_atom(b, pos2, n_res)
    bt = _bond_type_from_elems(a, b, rg1, rg2)

    # Adjacent residues joined by the forward C(i)-N(i+1) bond are the normal
    # linear backbone.  The reverse N(1)-C(n) pair is a head-to-tail closure;
    # for a two-residue macrocycle both pairs are numerically adjacent, so an
    # absolute-position test would incorrectly discard the closure.
    if bt == 'peptide':
        forward = (
            (a['name'] == 'C' and b['name'] == 'N' and pos2 == pos1 + 1)
            or (b['name'] == 'C' and a['name'] == 'N' and pos1 == pos2 + 1)
        )
        if forward:
            return None

    return CyclizationBond(
        bond_type=bt, pos1=pos1, pos2=pos2, rgroup1=rg1, rgroup2=rg2,
        atom1=a['name'], atom2=b['name'], res1=a['resn'], res2=b['resn'],
    )


def _is_forward_backbone_pair(
    atom1: str, pos1: int, atom2: str, pos2: int
) -> bool:
    """True only for the ordinary C(i)-N(i+1) backbone direction."""
    return (
        atom1 == 'C' and atom2 == 'N' and pos2 == pos1 + 1
    ) or (
        atom2 == 'C' and atom1 == 'N' and pos1 == pos2 + 1
    )


def _spatial_candidate_pairs(
    atoms: List[Dict], distance_ceiling: float
) -> List[tuple]:
    """Return a deterministic superset of pairs within the distance ceiling."""
    cell = max(distance_ceiling, 1e-6)
    grid: Dict[tuple, List[int]] = {}
    for index, atom in enumerate(atoms):
        x, y, z = atom['xyz']
        key = (
            int(math.floor(x / cell)),
            int(math.floor(y / cell)),
            int(math.floor(z / cell)),
        )
        grid.setdefault(key, []).append(index)
    pairs = set()
    for (cx, cy, cz), indices in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    partners = grid.get((cx + dx, cy + dy, cz + dz))
                    if not partners:
                        continue
                    for left in indices:
                        for right in partners:
                            if left < right:
                                pairs.add((left, right))
    return sorted(pairs)


def detect_geometric_bonds(pdb_path: str, chain_id: str,
                           pos_of: Dict[int, int], n_res: int,
                           *, include_geometry: bool = True,
                           radius_multiplier: Optional[float] = None,
                           distance_ceiling: Optional[float] = None) -> List[CyclizationBond]:
    """Infer cyclization bonds from CONECT records, then covalent-radius distance.

    CONECT (explicit connectivity) is trusted first; covalent-radius distance
    pairing then fills in bonds CONECT did not record. Both pass through the
    same element/valence classifier, so hydrogen bonds and van der Waals
    contacts never inflate the count.
    """
    resolved_radius_multiplier, resolved_distance_ceiling = resolve_geometry_params(
        radius_multiplier, distance_ceiling
    )
    atoms = read_atoms(pdb_path, chain_id)
    if not atoms:
        return []

    bonds: List[CyclizationBond] = []
    seen_pairs = set()

    def _add(bond: CyclizationBond):
        # R3 is a port class, not an atom identity. Distinct side-chain atom
        # pairs on the same residues must remain distinct candidates so the
        # strict audit can detect conflicts instead of losing one here.
        key = _bond_pair_key(bond)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        bonds.append(bond)

    # ── 1. CONECT-based (explicit connectivity) ──
    for sa, sb in read_conect(pdb_path):
        a, b = atoms.get(sa), atoms.get(sb)
        if a is None or b is None:
            continue
        bond = _classify_atom_pair(a, b, n_res, pos_of)
        if bond is not None:
            bond.evidence_source = 'conect'
            _add(bond)

    if not include_geometry:
        return bonds

    # ── 2. Covalent-radius distance pairing over heavy attachment atoms ──
    # Consider backbone N/C and every side-chain heavy atom; the covalent-bond
    # test (element-pair specific) decides what is actually bonded.
    cand = [a for a in atoms.values()
            if a['elem'] in _LINKABLE_ELEMS and a['name'] not in _BACKBONE_OTHER]
    # The cell grid is a guaranteed superset because every element-specific
    # covalent cutoff is bounded by the global distance ceiling.
    for i, j in _spatial_candidate_pairs(
        cand, resolved_distance_ceiling
    ):
        a, b = cand[i], cand[j]
        if a['resid'] == b['resid']:
            continue
        if not _is_covalent_bond(
            a, b,
            radius_multiplier=resolved_radius_multiplier,
            distance_ceiling=resolved_distance_ceiling,
        ):
            continue
        bond = _classify_atom_pair(a, b, n_res, pos_of)
        if bond is not None:
            bond.evidence_source = 'geometry'
            _add(bond)

    return bonds


def geometric_crosslink_atom_pairs(pdb_path: str, chain_id: str,
                                   *, radius_multiplier: Optional[float] = None,
                                   distance_ceiling: Optional[float] = None) -> List[tuple]:
    """Return cross-residue covalent bonds as (serial_a, serial_b) atom pairs.

    Atom-level counterpart of detect_geometric_bonds: same covalent-radius
    geometry, but emits PDB atom serials (not residue positions) so callers
    that build a molecule from PDB atoms (Path A/C) can inject cyclization
    bonds the CONECT records omitted.

    Excludes: same-residue pairs, hydrogens/unknown elements, cap residues,
    and adjacent-residue backbone peptide bonds (the linear chain, not a ring).
    Backbone-other atoms (CA/O/OXT) are skipped as attachment points.
    """
    resolved_radius_multiplier, resolved_distance_ceiling = resolve_geometry_params(
        radius_multiplier, distance_ceiling
    )
    atoms = read_atoms(pdb_path, chain_id)
    if not atoms:
        return []
    position_of = {
        resid: position
        for position, resid in enumerate(
            sorted({atom['resid'] for atom in atoms.values()}), start=1
        )
    }

    # candidate heavy attachment atoms (backbone N/C + side-chain heavy atoms)
    cand = [(s, a) for s, a in atoms.items()
            if a['elem'] in _LINKABLE_ELEMS and a['name'] not in _BACKBONE_OTHER
            and a['resn'] not in _CAPS]

    pairs = []
    seen = set()
    atom_rows = [atom for _serial, atom in cand]
    for i, j in _spatial_candidate_pairs(
        atom_rows, resolved_distance_ceiling
    ):
        si, ai = cand[i]
        sj, aj = cand[j]
        if ai['resid'] == aj['resid']:
            continue
        if not _is_covalent_bond(
            ai, aj,
            radius_multiplier=resolved_radius_multiplier,
            distance_ceiling=resolved_distance_ceiling,
        ):
            continue
        # Exclude only forward C(i)-N(i+1). For a two-residue cycle the
        # reverse N(1)-C(2) closure is also numerically adjacent.
        if _is_forward_backbone_pair(
            ai['name'], position_of[ai['resid']],
            aj['name'], position_of[aj['resid']],
        ):
            continue
        key = (min(si, sj), max(si, sj))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def classify_link_type(link: Dict) -> str:
    """Classify LINK bond type based on atoms involved.

    Returns: 'disulfide', 'thioether', 'ester', 'isopeptide', 'peptide',
             'staple_rcm', 'staple_thioether', 'staple_alkyl', 'unknown'
    """
    a1, a2 = link['atom1'], link['atom2']

    # Disulfide: SG-SG (Cys-Cys)
    if a1 == 'SG' and a2 == 'SG':
        return 'disulfide'

    # Thioether (lanthionine/sactipeptide): SG-C
    if (a1 == 'SG' and a2 in ('CA', 'CB', 'C')) or (a2 == 'SG' and a1 in ('CA', 'CB', 'C')):
        return 'thioether'

    # Staple — thioether (FC01-type): side-chain S to side-chain C
    # distinguish from lanthionine by checking for non-backbone C
    # (handled above as 'thioether'; staple_thioether is a sub-case
    #  surfaced when both atoms are side-chain, not backbone CA/CB)
    if (a1 in ('SG',) and a2 not in ('CA', 'CB', 'C')) or \
       (a2 in ('SG',) and a1 not in ('CA', 'CB', 'C')):
        return 'staple_thioether'

    # Ester: O-C (Ser/Thr side chain O to carbonyl C)
    if (a1 in ('OG', 'OG1') and a2 == 'C') or (a2 in ('OG', 'OG1') and a1 == 'C'):
        return 'ester'

    # Isopeptide/lactam: Lys NZ to a side-chain or terminal carbonyl carbon.
    if (a1 == 'NZ' and a2 in ('C', 'CG', 'CD')) or \
       (a2 == 'NZ' and a1 in ('C', 'CG', 'CD')):
        return 'isopeptide'

    # Peptide: N-C (backbone amide bond, e.g., head-to-tail cyclization)
    if (a1 == 'N' and a2 == 'C') or (a2 == 'N' and a1 == 'C'):
        return 'peptide'

    # Staple — RCM (ring-closing metathesis): C=C double bond between
    # side-chain carbons (e.g. allylglycine CG or similar).  Detected
    # when both atoms are non-backbone carbons.
    if a1 not in ('N', 'CA', 'C', 'O', 'OXT') and a2 not in ('N', 'CA', 'C', 'O', 'OXT'):
        # Both side-chain atoms — could be RCM (C=C) or alkyl (C-C)
        # If we can check bond order, RCM is double; otherwise classify
        # by atom type: CG/CD/CE side-chain C's → staple
        if a1.startswith('C') and a2.startswith('C'):
            return 'staple_alkyl'  # default to alkyl; RCM needs bond-order info

    return 'unknown'


# ══════════════════════════════════════════════════════════════════════════
# Cyclization Detector
# ══════════════════════════════════════════════════════════════════════════

def _bond_endpoint_key(bond: CyclizationBond, first: bool) -> tuple:
    """Return an order-independent, atom-aware endpoint identity."""
    if first:
        return (bond.pos1, (bond.atom1 or bond.rgroup1 or '').upper())
    return (bond.pos2, (bond.atom2 or bond.rgroup2 or '').upper())


def _bond_pair_key(bond: CyclizationBond) -> tuple:
    return tuple(sorted((_bond_endpoint_key(bond, True),
                         _bond_endpoint_key(bond, False))))


def _merge_bond_candidates(*groups: List[CyclizationBond]) -> List[CyclizationBond]:
    """Merge exact endpoint pairs while retaining the strongest provenance.

    Record evidence takes precedence over coordinate inference for the same
    atom pair. Distinct pairs are retained so a partial SSBOND/LINK record does
    not hide a second closure represented by CONECT or geometry. Endpoint
    conflicts remain visible to the strict input audit instead of being
    silently resolved here.
    """
    precedence = {'ssbond': 4, 'link': 3, 'conect': 2, 'geometry': 1,
                  'unknown': 0}
    merged: Dict[tuple, CyclizationBond] = {}
    order: List[tuple] = []
    for group in groups:
        for bond in group:
            key = _bond_pair_key(bond)
            current = merged.get(key)
            if current is None:
                merged[key] = bond
                order.append(key)
                continue
            if precedence.get(bond.evidence_source, 0) > precedence.get(
                    current.evidence_source, 0):
                merged[key] = bond
    return [merged[key] for key in order]


def detect_cyclization(pdb_path: str, chain_id: str = 'L',
                       *, allow_geometric_inference: bool = True,
                       radius_multiplier: Optional[float] = None,
                       distance_ceiling: Optional[float] = None) -> CyclizationInfo:
    """Data-driven cyclization detection from SSBOND/LINK records.

    Replaces heuristic-based detection with ground-truth PDB data.
    """
    from .pdb_parser import get_res_seq

    resolved_radius_multiplier, resolved_distance_ceiling = resolve_geometry_params(
        radius_multiplier, distance_ceiling
    )

    residues = get_res_seq(pdb_path, chain_id, include_het=True)
    if not residues:
        return CyclizationInfo(
            topology='linear',
            bonds=[],
            description='no residues found'
        )

    n_res = len(residues)
    record_positions = {}
    ambiguous_record_ids = set()
    for position, residue in enumerate(residues, start=1):
        identity = (
            residue['name'], int(residue['num']), str(residue.get('icode', ''))
        )
        if identity in record_positions:
            ambiguous_record_ids.add(identity)
        else:
            record_positions[identity] = position
    for identity in ambiguous_record_ids:
        record_positions.pop(identity, None)

    def record_position(record, endpoint):
        return record_positions.get((
            str(record[f'res{endpoint}']),
            int(record[f'num{endpoint}']),
            str(record.get(f'icode{endpoint}', '')),
        ))

    ssbonds = read_ssbond(pdb_path)
    links = read_link(pdb_path)

    bonds = []

    # Parse SSBOND records
    for ss in ssbonds:
        if ss['chain1'] == chain_id and ss['chain2'] == chain_id:
            pos1 = record_position(ss, 1)
            pos2 = record_position(ss, 2)
            if pos1 is not None and pos2 is not None:
                bonds.append(CyclizationBond(
                    bond_type='disulfide',
                    pos1=pos1,
                    pos2=pos2,
                    rgroup1='R3',  # Cys side chain
                    rgroup2='R3',
                    atom1='SG',
                    atom2='SG',
                    res1=ss['res1'],
                    res2=ss['res2'],
                    evidence_source='ssbond',
                ))

    ssbond_pairs = {_bond_pair_key(bond) for bond in bonds}

    # Parse LINK records
    for lk in links:
        if lk['chain1'] == chain_id and lk['chain2'] == chain_id:
            pos1 = record_position(lk, 1)
            pos2 = record_position(lk, 2)
            if pos1 is not None and pos2 is not None:
                link_type = classify_link_type(lk)

                link_bond = CyclizationBond(
                    bond_type=link_type,
                    pos1=pos1,
                    pos2=pos2,
                    rgroup1='R1' if lk['atom1'] == 'N' else (
                        'R2' if lk['atom1'] == 'C' else 'R3'
                    ),
                    rgroup2='R1' if lk['atom2'] == 'N' else (
                        'R2' if lk['atom2'] == 'C' else 'R3'
                    ),
                    atom1=lk['atom1'],
                    atom2=lk['atom2'],
                    res1=lk['res1'],
                    res2=lk['res2'],
                    evidence_source='link',
                )

                # A LINK-only SG-SG edge is valid explicit evidence. Suppress
                # it only when the exact atom endpoints are already present as
                # an SSBOND record.
                if link_type == 'disulfide' and _bond_pair_key(link_bond) in ssbond_pairs:
                    continue

                # Skip LINKs to terminal cap residues (ACE/NME/NH2 amidation,
                # not a true cyclization bond)
                _CAPS = {'ACE', 'NME', 'NH2'}
                if lk['res1'] in _CAPS or lk['res2'] in _CAPS:
                    continue

                # Skip adjacent-residue peptide bonds (normal linear backbone,
                # not a ring).  This must check position adjacency, not just
                # the 'peptide' type, because non-standard residues (D-AAs,
                # NNAAs) also form linear peptide C(i)-N(i+1) bonds.
                if link_type == 'peptide' and _is_forward_backbone_pair(
                    lk['atom1'], pos1, lk['atom2'], pos2
                ):
                    continue

                bonds.append(link_bond)

    # Reconcile evidence per edge. CONECT is always considered; coordinate
    # inference may be disabled by strict ablation/evaluation callers.
    residue_id_counts = {}
    for residue in residues:
        identity = (int(residue['num']), str(residue.get('icode', '')))
        residue_id_counts[identity] = residue_id_counts.get(identity, 0) + 1
    pos_of = {
        (int(residue['num']), str(residue.get('icode', ''))): idx + 1
        for idx, residue in enumerate(residues)
        if residue_id_counts[
            (int(residue['num']), str(residue.get('icode', '')))
        ] == 1
    }
    conect_and_geometry = detect_geometric_bonds(
        pdb_path,
        chain_id,
        pos_of,
        n_res,
        include_geometry=allow_geometric_inference,
        radius_multiplier=resolved_radius_multiplier,
        distance_ceiling=resolved_distance_ceiling,
    )
    bonds = _merge_bond_candidates(bonds, conect_and_geometry)

    # Classify topology
    n_bonds = len(bonds)
    if n_bonds == 0:
        topology = 'linear'
    elif n_bonds == 1:
        topology = 'monocyclic'
    elif n_bonds == 2:
        topology = 'bicyclic'
    elif n_bonds == 3:
        topology = 'tricyclic'
    else:
        topology = 'polycyclic'

    # Generate human-readable description
    bond_types = [b.bond_type for b in bonds]
    type_counts = {}
    for bt in bond_types:
        type_counts[bt] = type_counts.get(bt, 0) + 1

    desc_parts = []
    for bt in sorted(type_counts.keys()):
        count = type_counts[bt]
        if bt == 'peptide':
            desc_parts.append('head-to-tail')
        elif bt == 'disulfide':
            s = 's' if count > 1 else ''
            desc_parts.append(f'{count} disulfide bridge{s}')
        elif bt == 'thioether':
            s = 's' if count > 1 else ''
            desc_parts.append(f'{count} thioether bridge{s}')
        elif bt == 'ester':
            s = 's' if count > 1 else ''
            desc_parts.append(f'{count} ester linkage{s}')
        elif bt == 'isopeptide':
            s = 's' if count > 1 else ''
            desc_parts.append(f'{count} isopeptide bond{s}')
        else:
            s = 's' if count > 1 else ''
            desc_parts.append(f'{count} {bt} bond{s}')

    description = ' + '.join(desc_parts) if desc_parts else 'linear'

    warnings: List[str] = []
    info = CyclizationInfo(
        topology=topology,
        bonds=bonds,
        description=description,
    )
    if (
        allow_geometric_inference
        and is_nondefault_geometry(
            resolved_radius_multiplier, resolved_distance_ceiling
        )
    ):
        info.geometry_radius_multiplier = resolved_radius_multiplier
        info.geometry_distance_ceiling = resolved_distance_ceiling
        warnings.append(NONDEFAULT_GEOMETRY_PARAMS)
    info.warnings = warnings
    return info
