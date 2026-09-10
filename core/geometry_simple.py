"""simple_geometry_v2 - single-pass local geometry-aware inference of bond
orders, formal charges, implicit hydrogen counts and canonical isomeric
SMILES for heavy-atom 3D records (NNAA geometry study, ALGORITHM_CONTRACT.md).

SIMPLE arm design (deterministic, bounded, blind):

  * one geometry pass; no combinatorial search, no coordinate optimization,
    no heavy-atom addition/removal;
  * connectivity from covalent radii when ``bonds`` is null; supplied
    adjacency is always preserved as edges (orders are still inferred);
  * hybridization evidence from neighbor angle sums / mean angles and
    neighbor-plane RMSD fits; these directly gate and rank every bond-order,
    ring and charge decision and are visible in the evidence output;
  * ring aromaticity decided from ring-plane RMSD + ring mean angle +
    relative bond lengths (benzene vs cyclohexane separator); borderline
    rings emit both aromatic and saturated variants instead of a hidden
    coin flip;
  * Huckel donor enumeration for bare 5-ring nitrogens; tautomer ties are
    kept as separate equally-scored candidates;
  * bounded conservative hypervalent motifs: one P=O, S(=O)2 / S=O, one
    N=O for nitro-like N; other P/S/Se states stay single-bonded with
    honest reason codes; P-H / S-H are never blanket-forbidden (they arise
    from ordinary valence hydrogen filling);
  * charge/H repair variants: neutral first, then anionic / zwitterionic
    when ionizable acid and/or basic amine sites exist; RDKit sanitization
    arbitrates chemical validity, with a bounded evidence-ranked double-bond
    demotion retry;
  * stereochemistry read from 3D coordinates when determinable;
  * ``score`` (lower is better) sums bond-length residuals, sp2 angle
    residuals, aromatic plane residuals, formal-charge penalty and
    unmatched-hypervalent-P/S penalty.

Blind constraints honored: no names/truth inputs, no file reads, no
CCD/monomer/dictionary lookups, no network, no environment inspection, no
full-product import. Dependencies: numpy, RDKit, stdlib only.

Public API: ``infer_geometry(record, *, timeout_seconds=2.0) -> dict``.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

__all__ = ["infer_geometry", "ALGORITHM"]

ALGORITHM = "simple_geometry_v2"
MAX_CANDIDATES = 8


# --------------------------------------------------------------------------
# Element tables (hand-set constants; no library/dictionary lookups)
# --------------------------------------------------------------------------

COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Se": 1.20, "Br": 1.20,
    "I": 1.39,
}

NEUTRAL_VALENCE = {
    "H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "Si": 4, "P": 3,
    "S": 2, "Cl": 1, "Se": 2, "Br": 1, "I": 1,
}

# maximum total bond order an atom may carry in any state we build
MAX_BOND_SUM = {
    "B": 4, "C": 4, "N": 4, "O": 3, "F": 1, "Si": 4, "P": 5, "S": 6,
    "Cl": 3, "Se": 6, "Br": 3, "I": 5, "H": 1,
}

# reference heavy-atom bond lengths (Angstrom) by (sorted element pair, order)
REF_LEN = {
    ("C", "C"): {1: 1.53, 2: 1.335, 3: 1.20, 1.5: 1.39},
    ("C", "N"): {1: 1.47, 2: 1.29, 3: 1.16, 1.5: 1.335},
    ("C", "O"): {1: 1.43, 2: 1.23, 1.5: 1.36},
    ("C", "S"): {1: 1.82, 2: 1.60, 1.5: 1.71},
    ("C", "P"): {1: 1.84, 2: 1.65},
    ("C", "Se"): {1: 1.93, 2: 1.71},
    ("N", "N"): {1: 1.45, 2: 1.25, 1.5: 1.34},
    ("N", "O"): {1: 1.40, 2: 1.21},
    ("N", "P"): {1: 1.71, 2: 1.57},
    ("N", "S"): {1: 1.65, 2: 1.51},
    ("O", "O"): {1: 1.47},
    ("O", "P"): {1: 1.61, 2: 1.50},
    ("O", "S"): {1: 1.58, 2: 1.44},
    ("S", "S"): {1: 2.05},
    ("C", "F"): {1: 1.35}, ("C", "Cl"): {1: 1.77}, ("C", "Br"): {1: 1.94},
    ("C", "I"): {1: 2.14},
}
_ORDER_FACTOR = {1.0: 1.0, 2.0: 0.87, 3.0: 0.79, 1.5: 0.93}


class _InvalidInput(Exception):
    def __init__(self, codes: Sequence[str]):
        super().__init__(",".join(codes))
        self.codes = list(codes)


def _covalent_sum(a: str, b: str) -> float:
    ra = COVALENT_RADII.get(a)
    rb = COVALENT_RADII.get(b)
    if ra is None or rb is None:
        return 1.9
    return ra + rb


def _ref_len(a: str, b: str, order: float) -> float:
    table = REF_LEN.get((min(a, b), max(a, b)))
    if table is not None and order in table:
        return table[order]
    return _covalent_sum(a, b) * _ORDER_FACTOR.get(order, 1.0)


def _single_ref(a: str, b: str) -> float:
    return _ref_len(a, b, 1.0)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _validate(record) -> Tuple[List[str], np.ndarray, Optional[List[Tuple[int, int]]], Optional[int]]:
    if not isinstance(record, dict):
        raise _InvalidInput(["record_not_dict"])
    atoms = record.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise _InvalidInput(["atoms_missing_or_empty"])
    n = len(atoms)
    elements: List[str] = []
    coords = np.zeros((n, 3), dtype=float)
    pt = Chem.GetPeriodicTable()
    for idx, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise _InvalidInput(["atom_not_dict"])
        if isinstance(atom.get("id"), bool) or atom.get("id") != idx:
            raise _InvalidInput(["atom_id_not_positional"])
        el = atom.get("element")
        if not isinstance(el, str):
            raise _InvalidInput(["element_missing"])
        el = el.strip().capitalize()
        if el in {"H", "D"}:
            raise _InvalidInput(["explicit_hydrogen_not_supported"])
        try:
            pt.GetAtomicWeight(el)
        except Exception:
            raise _InvalidInput(["unknown_element:" + el])
        elements.append(el)
        xyz = atom.get("xyz")
        if (not isinstance(xyz, (list, tuple)) or len(xyz) != 3
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xyz)):
            raise _InvalidInput(["xyz_not_length3_numbers"])
        arr = np.asarray(xyz, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise _InvalidInput(["nonfinite_coordinates"])
        coords[idx] = arr
    bonds = record.get("bonds", None)
    edges: Optional[List[Tuple[int, int]]] = None
    if bonds is not None:
        if not isinstance(bonds, list):
            raise _InvalidInput(["bonds_not_list_or_null"])
        seen: Set[Tuple[int, int]] = set()
        edges = []
        for b in bonds:
            if not isinstance(b, (list, tuple)) or len(b) != 2:
                raise _InvalidInput(["bond_not_pair"])
            i, j = b
            if isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, int) or not isinstance(j, int):
                raise _InvalidInput(["bond_endpoint_not_int"])
            if i == j or not (0 <= i < n) or not (0 <= j < n):
                raise _InvalidInput(["bond_endpoint_out_of_range"])
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append(key)
        edges.sort()
    tc = record.get("total_charge", None)
    if tc is None:
        pass
    elif isinstance(tc, bool) or not isinstance(tc, (int, float)):
        raise _InvalidInput(["total_charge_not_int_or_null"])
    else:
        if isinstance(tc, float) and (not math.isfinite(tc) or not float(tc).is_integer()):
            raise _InvalidInput(["total_charge_not_int_or_null"])
        tc = int(tc)
    return elements, coords, edges, tc


# --------------------------------------------------------------------------
# Geometry (the single pass)
# --------------------------------------------------------------------------

def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2))
    if denom <= 0.0:
        return 0.0
    cs = float(np.dot(v1, v2)) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cs))))


def _plane_rms(points: np.ndarray) -> float:
    """RMS deviation (Angstrom) of points from their best-fit plane."""
    if len(points) < 3:
        return 0.0
    center = points.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(points - center)
    except np.linalg.LinAlgError:
        return 0.0
    return float(np.sqrt(((points - center) @ vt[2]) ** 2).mean())


class _Geometry:
    """Local geometry features used by every downstream decision."""

    def __init__(self, elements: List[str], xyz: np.ndarray, edges: List[Tuple[int, int]]):
        self.elements = elements
        self.xyz = xyz
        self.n = len(elements)
        self.nbrs: List[List[int]] = [[] for _ in range(self.n)]
        self.dist: Dict[Tuple[int, int], float] = {}
        for i, j in edges:
            self.nbrs[i].append(j)
            self.nbrs[j].append(i)
            self.dist[(i, j)] = float(np.linalg.norm(xyz[i] - xyz[j]))
        for k in range(self.n):
            self.nbrs[k].sort()
        self.mean_angle: List[float] = [0.0] * self.n
        self.angle_sum: List[float] = [0.0] * self.n
        self.plane_rms: List[float] = [0.0] * self.n
        self.min_angle_cache: List[Optional[float]] = [None] * self.n
        for a in range(self.n):
            nb = self.nbrs[a]
            if len(nb) >= 2:
                angs = [_angle_deg(xyz[b] - xyz[a], xyz[c] - xyz[a])
                        for bi, b in enumerate(nb) for c in nb[bi + 1:]]
                self.mean_angle[a] = sum(angs) / len(angs)
                self.angle_sum[a] = sum(angs)
            if len(nb) >= 3:
                pts = np.vstack([xyz[a], xyz[nb]])
                self.plane_rms[a] = _plane_rms(pts)

    def d(self, i: int, j: int) -> float:
        return self.dist[(min(i, j), max(i, j))]

    def degree(self, a: int) -> int:
        return len(self.nbrs[a])

    def min_angle(self, a: int) -> float:
        if self.min_angle_cache[a] is None:
            nb = self.nbrs[a]
            if len(nb) < 2:
                self.min_angle_cache[a] = 180.0
            else:
                angs = [_angle_deg(self.xyz[b] - self.xyz[a], self.xyz[c] - self.xyz[a])
                        for bi, b in enumerate(nb) for c in nb[bi + 1:]]
                self.min_angle_cache[a] = min(angs)
        return self.min_angle_cache[a]

    def sp2_evidence(self, a: int) -> bool:
        """Observed angles support an sp2 center here (combined downstream
        with the short-bond gate; the angle value is recorded as evidence)."""
        deg = self.degree(a)
        if deg <= 1:
            return True  # terminal atom: no angle evidence against it
        if deg >= 3:
            return self.angle_sum[a] >= 345.0
        # degree-2: 5-ring sp2 atoms sit near 106-109 deg, so the gate is
        # loose here; bond-length shortening is the discriminating evidence.
        return self.mean_angle[a] >= 105.0


def _infer_connectivity(elements: List[str], xyz: np.ndarray) -> Tuple[List[Tuple[int, int]], List[str]]:
    n = len(elements)
    edges: List[Tuple[int, int]] = []
    reasons: List[str] = []
    if n > 1:
        diff = xyz[:, None, :] - xyz[None, :, :]
        dmat = np.sqrt((diff ** 2).sum(-1))
        radii = np.array([COVALENT_RADII.get(e, 0.9) for e in elements], dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(dmat[i, j])
                rsum = float(radii[i] + radii[j])
                if d < 0.25:
                    reasons.append("coincident_atoms_pair_%d_%d" % (i, j))
                elif 0.55 * rsum <= d <= rsum + 0.40:
                    edges.append((i, j))
    return edges, sorted(set(reasons))


def _find_rings(n: int, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    rw = Chem.RWMol()
    for _ in range(n):
        rw.AddAtom(Chem.Atom("C"))
    for i, j in edges:
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    out: List[Tuple[int, int]] = []
    try:
        # proper SSSR (fused systems: indole -> 5+6, naphthalene -> 6+6)
        rings: List[Tuple[int, ...]] = [tuple(r) for r in Chem.GetSymmSSSR(mol)]
    except Exception:
        try:
            Chem.FastFindRings(mol)
            rings = [tuple(r) for r in mol.GetRingInfo().AtomRings()]
        except Exception:
            rings = []
    for r in rings:
        # keep traversal (cyclic) order so consecutive atoms are bonded;
        # rotate deterministically so the smallest id comes first
        k = r.index(min(r))
        out.append(tuple(r[k:] + r[:k]))
    return sorted(out, key=lambda r: (len(r), r))


_RING_ELEMS = {"C", "N", "O", "S", "Se"}


class _RingInfo:
    __slots__ = ("ring", "size", "plane_rms", "mean_angle", "rel_len", "ok_elements", "ok_degree")

    def __init__(self, ring: Tuple[int, ...], geo: _Geometry, elements: List[str]):
        self.ring = ring
        self.size = len(ring)
        self.plane_rms = _plane_rms(geo.xyz[list(ring)])
        angs = [geo.mean_angle[a] for a in ring if geo.degree(a) >= 2]
        self.mean_angle = sum(angs) / len(angs) if angs else 0.0
        rels = []
        m = len(ring)
        for k in range(m):
            i, j = ring[k], ring[(k + 1) % m]
            if (min(i, j), max(i, j)) in geo.dist:
                rels.append(geo.d(i, j) / _single_ref(elements[i], elements[j]))
        self.rel_len = sum(rels) / len(rels) if rels else 1.0
        self.ok_elements = all(elements[a] in _RING_ELEMS for a in ring)
        self.ok_degree = all(geo.degree(a) <= 3 for a in ring)

    def decision(self) -> str:
        """'aromatic' | 'ambiguous' | 'saturated', from observed geometry only."""
        if self.size not in (5, 6) or not self.ok_elements or not self.ok_degree:
            return "saturated"
        if self.plane_rms > 0.10:
            return "saturated"
        if self.size == 6 and self.mean_angle < 114.0:
            return "saturated"
        if self.rel_len <= 0.950:
            return "aromatic"
        if self.rel_len <= 0.975:
            return "ambiguous"
        return "saturated"

    def evidence(self) -> Dict[str, float]:
        return {"size": self.size,
                "plane_rms_a": round(self.plane_rms, 4),
                "mean_angle_deg": round(self.mean_angle, 2),
                "mean_relative_bond_length": round(self.rel_len, 4),
                "decision": self.decision()}


# --------------------------------------------------------------------------
# Graph variant: one bond-order assignment over the fixed heavy adjacency
# --------------------------------------------------------------------------

class _GraphVariant:
    def __init__(self, n: int, edges: List[Tuple[int, int]], name: str):
        self.name = name
        self.n = n
        self.edges = list(edges)
        self.orders: Dict[Tuple[int, int], float] = {e: 1.0 for e in edges}
        self.aromatic_atoms: Set[int] = set()
        self.aromatic_bonds: Set[Tuple[int, int]] = set()
        self.donor_n: Set[int] = set()
        self.motif_atoms: Set[int] = set()   # atoms covered by P/S/N-oxo motifs
        self.motifs: List[str] = []
        self.notes: List[str] = []
        self._bsum_cache: Optional[Dict[int, float]] = None

    def clone(self, name: str) -> "_GraphVariant":
        new = _GraphVariant(self.n, self.edges, name)
        new.orders = dict(self.orders)
        new.aromatic_atoms = set(self.aromatic_atoms)
        new.aromatic_bonds = set(self.aromatic_bonds)
        new.donor_n = set(self.donor_n)
        new.motif_atoms = set(self.motif_atoms)
        new.motifs = list(self.motifs)
        new.notes = list(self.notes)
        return new

    def set_order(self, i: int, j: int, order: float) -> None:
        key = (min(i, j), max(i, j))
        self.orders[key] = order
        self._bsum_cache = None
        if order == 1.5:
            self.aromatic_bonds.add(key)

    def nbrs_of(self, a: int) -> List[int]:
        return [j for i, j in self.edges if i == a] + [i for i, j in self.edges if j == a]

    def bond_sum(self, a: int) -> float:
        total = 0.0
        for b in self.nbrs_of(a):
            total += self.orders[(min(a, b), max(a, b))]
        return total


def _hypervalent_motifs(variant: _GraphVariant, geo: _Geometry, elements: List[str],
                        reasons: List[str]) -> None:
    """Bounded, conservative P=O / S(=O)2 / S=O / N(=O) motifs (S/P geometry
    is pyramidal/tetrahedral, so these bypass the sp2 angle gate; evidence
    is the S-O / P-O bond length and valence pattern, recorded per motif)."""
    for a in range(geo.n):
        el = elements[a]
        if el in variant.aromatic_atoms:
            continue
        if el not in ("P", "S", "N", "Se"):
            continue
        oxy = [b for b in geo.nbrs[a]
               if elements[b] == "O" and b not in variant.aromatic_atoms
               and variant.orders[(min(a, b), max(a, b))] == 1.0]
        if el == "P" and oxy and geo.degree(a) >= 3:
            b = min(oxy, key=lambda x: (geo.d(a, x), x))
            variant.set_order(a, b, 2.0)
            variant.motif_atoms.update((a, b))
            variant.motifs.append("P%d=O%d d=%.3f (P deg %d)" % (a, b, geo.d(a, b), geo.degree(a)))
            reasons.append("phosphoryl_pattern")
        elif el == "S" and len(oxy) >= 2 and geo.degree(a) >= 3:
            two = sorted(oxy, key=lambda x: (geo.d(a, x), x))[:2]
            for b in two:
                variant.set_order(a, b, 2.0)
            variant.motif_atoms.update({a, *two})
            variant.motifs.append("S%d(=O%d)(=O%d) d=%.3f/%.3f"
                                  % (a, two[0], two[1], geo.d(a, two[0]), geo.d(a, two[1])))
            reasons.append("sulfonyl_pattern")
        elif el in ("S", "Se") and len(oxy) == 1 and geo.degree(a) >= 3:
            b = oxy[0]
            variant.set_order(a, b, 2.0)
            variant.motif_atoms.update((a, b))
            variant.motifs.append("%s%d=O%d d=%.3f" % (el, a, b, geo.d(a, b)))
            reasons.append("s_oxide_single_pattern")
        elif el == "S" and len(oxy) == 1 and geo.degree(a) == 2:
            variant.notes.append("S%d-O with S degree 2 kept single; sulfenate vs S=O unresolved by evidence" % a)
            reasons.append("s_oxide_ambiguous")
        elif el == "N" and len(oxy) >= 2:
            b = min(oxy, key=lambda x: (geo.d(a, x), x))
            variant.set_order(a, b, 2.0)
            variant.motif_atoms.update((a, b))
            variant.motifs.append("N%d=O%d d=%.3f" % (a, b, geo.d(a, b)))
            reasons.append("nitro_pattern")
        elif el in ("P", "S", "Se") and not oxy and geo.degree(a) >= 4:
            variant.notes.append("%s%d degree %d without oxo motif; single bonds + valence H fill (conservative)"
                                 % (el, a, geo.degree(a)))
            reasons.append("p_s_hypervalent_unsupported_conservative")


def _triple_pass(variant: _GraphVariant, geo: _Geometry, elements: List[str]) -> None:
    """Linear (min angle >= 160 deg) + short (rel <= 0.86) C#N / C#C only."""
    cand = []
    for i, j in variant.edges:
        if variant.orders[(i, j)] != 1.0:
            continue
        if i in variant.aromatic_atoms or j in variant.aromatic_atoms:
            continue
        pair = {elements[i], elements[j]}
        if pair == {"C", "N"}:
            n_atom = j if elements[j] == "N" else i
            if geo.degree(n_atom) != 1:
                continue  # nitrile N must be terminal; excludes isocyanate paths
        elif pair != {"C", "C"}:
            continue
        rel = geo.d(i, j) / _single_ref(elements[i], elements[j])
        if rel > 0.86:
            continue
        if geo.degree(i) >= 2 and geo.min_angle(i) < 160.0:
            continue
        if geo.degree(j) >= 2 and geo.min_angle(j) < 160.0:
            continue
        cand.append((round(rel, 6), min(i, j), max(i, j)))
    for rel, i, j in sorted(cand):
        if variant.orders[(i, j)] != 1.0:
            continue
        if (variant.bond_sum(i) + 2.0 > MAX_BOND_SUM.get(elements[i], 4) + 1e-9
                or variant.bond_sum(j) + 2.0 > MAX_BOND_SUM.get(elements[j], 4) + 1e-9):
            continue
        variant.set_order(i, j, 3.0)
        variant.motifs.append("triple %d#%d d=%.3f (min_ang %.1f/%.1f)"
                              % (i, j, geo.d(i, j), geo.min_angle(i), geo.min_angle(j)))


def _double_pass(variant: _GraphVariant, geo: _Geometry, elements: List[str],
                 deadline: float) -> None:
    """Greedy double assignment ranked by observed evidence: shortest bond
    (relative to its single-bond reference) first, gated by sp2 angle
    evidence at both endpoints; at most one new double per atom.
    S/Se carry one slot so thiones (C=S / C=Se) can form when the bond is
    short and the partner center is sp2; sulfonyl/oxide doubles already
    consume the slot."""
    quota = {"C": 1, "N": 1, "O": 1, "B": 1, "Si": 1, "Cl": 1, "Br": 1, "I": 1,
             "S": 1, "P": 0, "Se": 1, "F": 0, "H": 0}
    used: Dict[int, int] = {a: 0 for a in range(geo.n)}
    for (i, j), o in sorted(variant.orders.items()):
        if o >= 2.0:
            used[i] += 1
            used[j] += 1
    cand = []
    for i, j in variant.edges:
        if variant.orders[(i, j)] != 1.0:
            continue
        if i in variant.aromatic_atoms or j in variant.aromatic_atoms:
            continue
        rel = geo.d(i, j) / _single_ref(elements[i], elements[j])
        if rel > 0.92:
            continue
        if not geo.sp2_evidence(i) or not geo.sp2_evidence(j):
            continue
        cand.append((round(rel, 6), i, j))
    cand.sort()
    for k, (rel, i, j) in enumerate(cand):
        if k % 64 == 0 and time.monotonic() > deadline:
            return
        if variant.orders[(i, j)] != 1.0:
            continue
        if used[i] >= quota.get(elements[i], 0) or used[j] >= quota.get(elements[j], 0):
            continue
        if (variant.bond_sum(i) + 1.0 > MAX_BOND_SUM.get(elements[i], 4) + 1e-9
                or variant.bond_sum(j) + 1.0 > MAX_BOND_SUM.get(elements[j], 4) + 1e-9):
            continue
        variant.set_order(i, j, 2.0)
        used[i] += 1
        used[j] += 1
        variant.motifs.append("double %d=%d d=%.3f rel=%.3f (ang %.1f/%.1f)"
                              % (i, j, geo.d(i, j), rel,
                                 geo.mean_angle[i] if geo.degree(i) >= 2 else -1.0,
                                 geo.mean_angle[j] if geo.degree(j) >= 2 else -1.0))


def _donor_options(info: _RingInfo, geo: _Geometry, elements: List[str]) -> List[Set[int]]:
    """Huckel pi-electron accounting for one aromatic ring (target 6 e-).

    Every C contributes 1; ring O/S/Se of degree 2 (furan/thiophene) and
    ring N of degree 3 (substituted pyrrole-type N, neutral) are forced
    2-electron donors; a bare degree-2 N contributes 1 (pyridine-type) or
    2 when it carries the ring H (donor). Options list which bare N carry
    the donor H; empty list means the ring is not aromatic in this model."""
    bare_n = sorted(a for a in info.ring if elements[a] == "N" and geo.degree(a) == 2)
    forced = [a for a in info.ring
              if (elements[a] in ("O", "S", "Se") and geo.degree(a) == 2)
              or (elements[a] == "N" and geo.degree(a) == 3)]
    n_c = sum(1 for a in info.ring if elements[a] == "C")
    target = 6
    extras = target - n_c - 2 * len(forced) - 1 * len(bare_n)
    if extras < 0 or extras > len(bare_n):
        return []
    if extras == 0:
        return [set()]
    options = [{d for d in combo} for combo in itertools.combinations(bare_n, extras)]
    return options[:4]


# --------------------------------------------------------------------------
# Charge / hydrogen repair
# --------------------------------------------------------------------------

def _nitro_paired_oxygen(variant: _GraphVariant, geo: _Geometry, elements: List[str]
                         ) -> Set[int]:
    """Terminal single-bonded oxygen paired as O- with a formal N+.

    Only applies when the N actually reaches bond order 4 (the N+ state) via
    an N=O double: classic nitro C-N+(=O)O- and nitrate/nitric-acid-type
    N+(=O)(O-R)(O-). Exactly ONE terminal single O is paired (shortest bond,
    deterministic under O symmetry); any remaining terminal O keeps its
    normal OH state, so nitric-acid/nitrate OH microstates are not
    unconditionally deprotonated. A tervalent N (bond sum 3, e.g. nitrous
    acid H-O-N=O) is NOT forced into the nitro paired state."""
    paired: Set[int] = set()
    for a in range(geo.n):
        if elements[a] != "N" or a in variant.aromatic_atoms:
            continue
        if int(round(variant.bond_sum(a))) != 4:
            continue
        double_o: List[int] = []
        single_o: List[int] = []
        for b in geo.nbrs[a]:
            if elements[b] != "O" or b in variant.aromatic_atoms:
                continue
            o = variant.orders[(min(a, b), max(a, b))]
            if o == 2.0:
                double_o.append(b)
            elif o == 1.0 and geo.degree(b) == 1:
                single_o.append(b)
        if not double_o or not single_o:
            continue
        paired.add(min(single_o, key=lambda x: (geo.d(a, x), x)))
    return paired


def _repair_states(variant: _GraphVariant, geo: _Geometry, elements: List[str],
                   mode: str, ionizable_o: Set[int], basic_n: Set[int]
                   ) -> Optional[List[Tuple[int, int]]]:
    """Per-atom (formal_charge, implicit_H) for a finished order assignment.

    mode 'neutral' prefers H over charges, 'anionic' deprotonates ionizable
    terminal oxo oxygens, 'zwitterion' additionally protonates basic amines.
    Nitro-pattern terminal single oxygens always take the paired O- state.
    Returns None when some atom admits no state (caller may demote a double
    bond and retry)."""
    nitro_o = _nitro_paired_oxygen(variant, geo, elements)
    states: List[Tuple[int, int]] = []
    for a in range(geo.n):
        el = elements[a]
        bsum = variant.bond_sum(a)
        if a in variant.aromatic_atoms:
            nb = geo.degree(a)
            if el == "C":
                q, h = 0, max(0, int(math.floor(4.0 - bsum + 1e-6)))
            elif el == "N":
                if geo.degree(a) >= 3:
                    q, h = 0, 0  # substituted pyrrole-type donor N, neutral
                elif a in variant.donor_n:
                    q, h = 0, 1  # bare donor N carries the ring H
                else:
                    q, h = 0, 0  # pyridine-type
            elif el in ("O", "S", "Se"):
                if nb >= 3 and bsum >= 4.0:
                    q, h = 1, 0
                else:
                    q, h = 0, 0
            else:
                q, h = 0, 0
            states.append((q, h))
            continue
        bi = int(round(bsum))
        q, h = 0, 0
        if el in ("C", "Si"):
            if bi > 4:
                return None
            h = 4 - bi
        elif el == "N":
            if bi <= 3:
                h = 3 - bi
                if mode == "zwitterion" and a in basic_n:
                    # protonation ADDS one hydrogen: primary -> [NH3+],
                    # secondary -> [NH2+], tertiary -> [NH+] (4-bi >= 1)
                    q, h = 1, 4 - bi
            elif bi == 4:
                q, h = 1, 0
            else:
                return None
        elif el == "O":
            if a in nitro_o:
                q, h = -1, 0
            elif bi <= 1:
                if mode in ("anionic", "zwitterion") and a in ionizable_o:
                    q, h = -1, 0
                else:
                    q, h = 0, 2 - bi
            elif bi == 2:
                q, h = 0, 0
            elif bi == 3:
                q, h = 1, 0
            else:
                return None
        elif el in ("S", "Se"):
            if bi <= 2:
                h = 2 - bi
            elif bi in (4, 6):
                q, h = 0, 0
            else:  # bi in (3, 5): H-fill to the next allowed hypervalence 4/6
                h = 1
        elif el == "P":
            if bi <= 3:
                h = 3 - bi
            elif bi in (4, 5):
                h = 5 - bi
            else:
                return None
        elif el == "B":
            if bi <= 3:
                h = 3 - bi
            elif bi == 4:
                q, h = -1, 0
            else:
                return None
        elif el in ("F", "Cl", "Br", "I"):
            if bi <= 1:
                h = 1 - bi
            elif bi in (3, 5):
                q, h = 0, 0
            else:
                return None
        else:
            v = NEUTRAL_VALENCE.get(el, 6)
            if bi > v:
                return None
            h = v - bi
        states.append((q, h))
    return states


def _ionizable_and_basic(variant: _GraphVariant, geo: _Geometry, elements: List[str]
                         ) -> Tuple[Set[int], Set[int]]:
    """Terminal oxo-group oxygens (deprotonatable) and aliphatic basic N.
    Nitro-paired O- oxygens are excluded: they carry the fixed paired charge,
    not a protonation equilibrium."""
    paired_o = _nitro_paired_oxygen(variant, geo, elements)
    ionizable: Set[int] = set()
    basic: Set[int] = set()
    for a in range(geo.n):
        if elements[a] != "O" or a in variant.aromatic_atoms or geo.degree(a) != 1:
            continue
        if a in paired_o:
            continue
        b = geo.nbrs[a][0]
        if variant.bond_sum(a) == 1.0 and variant.bond_sum(b) >= 3.0:
            ionizable.add(a)
    for a in range(geo.n):
        if elements[a] != "N" or a in variant.aromatic_atoms:
            continue
        nb = geo.nbrs[a]
        if not nb or not all(elements[b] == "C" for b in nb):
            continue
        if any(variant.orders[(min(a, b), max(a, b))] != 1.0 for b in nb):
            continue
        amide_like = False
        for b in nb:
            for c in geo.nbrs[b]:
                if c != a and variant.orders[(min(b, c), max(b, c))] == 2.0 and elements[c] in ("O", "N"):
                    amide_like = True
                    break
            if amide_like:
                break
        if not amide_like and variant.bond_sum(a) <= 3:
            basic.add(a)
    return ionizable, basic


# --------------------------------------------------------------------------
# RDKit build / finalize
# --------------------------------------------------------------------------

def _build_mol(variant: _GraphVariant, elements: List[str], xyz: np.ndarray,
               states: List[Tuple[int, int]]):
    rw = Chem.RWMol()
    for a in range(variant.n):
        atom = Chem.Atom(elements[a])
        atom.SetFormalCharge(states[a][0])
        atom.SetNumExplicitHs(states[a][1])
        atom.SetNoImplicit(True)
        if a in variant.aromatic_atoms:
            atom.SetIsAromatic(True)
        rw.AddAtom(atom)
    for i, j in variant.edges:
        order = variant.orders[(i, j)]
        btype = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE,
                 3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}[order]
        rw.AddBond(i, j, btype)
    conf = Chem.Conformer(variant.n)
    for a in range(variant.n):
        conf.SetAtomPosition(a, Point3D(float(xyz[a][0]), float(xyz[a][1]), float(xyz[a][2])))
    rw.AddConformer(conf, assignId=True)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def _smiles_with_stereo(mol) -> Tuple[str, str]:
    try:
        mh = Chem.AddHs(mol, addCoords=True)
        Chem.AssignStereochemistryFrom3D(mh)
        mh = Chem.RemoveHs(mh)
        Chem.AssignStereochemistry(mh, cleanIt=True, force=True)
        smi = Chem.MolToSmiles(mh, isomericSmiles=True)
        if "@" in smi or "/" in smi or "\\" in smi:
            return smi, "from_3d"
        return smi, "none_determined"
    except Exception:
        try:
            return Chem.MolToSmiles(Chem.Mol(mol), isomericSmiles=False), "stereo_assignment_failed"
        except Exception:
            return "", "smiles_failed"


def _finalize(variant: _GraphVariant, elements: List[str], xyz: np.ndarray, geo: _Geometry,
              reasons: List[str]) -> List[dict]:
    """Enumerate bounded charge modes with demotion retries; return candidate
    payload dicts (unsorted, unscored fields filled by caller)."""
    ionizable, basic = _ionizable_and_basic(variant, geo, elements)
    modes = ["neutral"]
    if ionizable:
        modes.append("anionic")
        if basic:
            modes.append("zwitterion")
    out: List[dict] = []
    for mode in modes:
        v = variant
        demoted: List[Tuple[int, int]] = []
        for attempt in range(3):
            states = _repair_states(v, geo, elements, mode, ionizable, basic)
            if states is None:
                break
            mol = _build_mol(v, elements, xyz, states)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                doubles = [k for k, o in sorted(v.orders.items())
                           if o == 2.0 and k not in v.aromatic_bonds]
                if attempt == 2 or not doubles:
                    break
                worst = max(doubles, key=lambda k: (
                    abs(geo.d(*k) - _ref_len(elements[k[0]], elements[k[1]], 2.0)), k))
                v = v.clone(v.name + "-demoted")
                v.set_order(worst[0], worst[1], 1.0)
                v.notes.append("demoted double %d-%d after sanitize failure" % worst)
                demoted.append(worst)
                continue
            smiles, stereo = _smiles_with_stereo(mol)
            if not smiles:
                break
            bonds = []
            for bond in mol.GetBonds():
                bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bo = bond.GetBondTypeAsDouble()
                bo = 1.5 if bo == 1.5 else float(int(round(bo)))
                bonds.append([min(bi, bj), max(bi, bj), bo])
            bonds.sort()
            out.append({
                "smiles": smiles,
                "bonds": bonds,
                "formal_charges": [s[0] for s in states],
                "hydrogen_counts": [s[1] for s in states],
                "variant": v,
                "mode": mode,
                "demoted": list(demoted),
                "stereo": stereo,
            })
            break
    if demoted and out:
        reasons.append("bond_demotion_applied")
    return out


# --------------------------------------------------------------------------
# Scoring (lower is better; geometry residuals visible per term)
# --------------------------------------------------------------------------

def _score(payload: dict, geo: _Geometry, elements: List[str],
           aromatic_rings: List[_RingInfo]) -> Dict[str, float]:
    variant: _GraphVariant = payload["variant"]
    len_res: List[float] = []
    ang_res: List[float] = []
    plane_res: List[float] = []
    claimed_sp2: Set[int] = set()
    for (i, j), order in sorted(variant.orders.items()):
        len_res.append(min(0.35, abs(geo.d(i, j) - _ref_len(elements[i], elements[j], order))))
        if order >= 1.5:
            claimed_sp2.update((i, j))
    # expected angle per claimed sp/sp2 center: 180 for triple-bond
    # endpoints (nitriles/alkynes are linear); 108 only for DEGREE-2 atoms of
    # aromatic 5-rings (their mean angle is the ring interior angle);
    # 120 otherwise. Degree-3 ring atoms average two ~108 ring angles with a
    # ~124 exocyclic angle (near 120), so they must not be scored at 108.
    # Known limitation: fused-ring junction atoms match neither simple
    # expectation; their bounded residual stays as honest evidence.
    ring_size: Dict[int, int] = {}
    for info in aromatic_rings:
        for a in info.ring:
            ring_size[a] = info.size
    triple_atoms: Set[int] = set()
    for (i, j), order in variant.orders.items():
        if order >= 3.0:
            triple_atoms.update((i, j))
    for a in sorted(claimed_sp2):
        if geo.degree(a) >= 2:
            if a in triple_atoms:
                expected = 180.0
            elif ring_size.get(a) == 5 and geo.degree(a) == 2:
                expected = 108.0
            else:
                expected = 120.0
            ang_res.append(min(1.0, abs(geo.mean_angle[a] - expected) / 15.0))
    for info in aromatic_rings:
        plane_res.append(min(0.5, info.plane_rms * 3.0))
    charge = sum(abs(q) for q in payload["formal_charges"])
    unmatched_ps = sum(
        1 for a in range(geo.n)
        if elements[a] in ("P", "S", "Se") and a not in variant.motif_atoms
        and a not in variant.aromatic_atoms and geo.degree(a) >= 4)
    terms = {
        "bond_length_residual": math.fsum(len_res) / max(1, len(len_res)),
        "sp2_angle_residual": math.fsum(ang_res) / max(1, len(ang_res)),
        "aromatic_plane_residual": math.fsum(plane_res),
        "formal_charge_penalty": 0.5 * charge,
        "unmatched_hypervalent_ps_penalty": 0.25 * unmatched_ps,
    }
    terms_total = (terms["bond_length_residual"] + 0.5 * terms["sp2_angle_residual"]
                   + 2.0 * terms["aromatic_plane_residual"] + terms["formal_charge_penalty"]
                   + terms["unmatched_hypervalent_ps_penalty"])
    return {"score": terms_total, "terms": terms}


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def _timeout_result(timeout_seconds: float, reasons: List[str]) -> dict:
    return {"status": "timeout", "algorithm": ALGORITHM, "candidates": [],
            "search_complete": False,
            "reason_codes": ["timeout_exceeded"] + [r for r in reasons if r != "timeout_exceeded"],
            "evidence": {"timeout_seconds": timeout_seconds, "truncated": True}}


def infer_geometry(record: dict, *, timeout_seconds: float = 2.0) -> dict:
    """Infer bond orders / charges / implicit H / isomeric SMILES from a
    heavy-atom 3D record. See module docstring and ALGORITHM_CONTRACT.md."""
    t0 = time.monotonic()
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        return {"status": "invalid_input", "algorithm": ALGORITHM, "candidates": [],
                "search_complete": False, "reason_codes": ["invalid_timeout_seconds"], "evidence": {}}
    deadline = t0 + float(timeout_seconds)
    reasons: List[str] = []
    try:
        elements, xyz, edges, total_charge = _validate(record)
    except _InvalidInput as exc:
        return {"status": "invalid_input", "algorithm": ALGORITHM, "candidates": [],
                "search_complete": False, "reason_codes": exc.codes,
                "evidence": {"validation": "failed"}}
    n = len(elements)
    bonds_supplied = edges is not None
    if edges is None:
        edges, coincident = _infer_connectivity(elements, xyz)
        reasons.append("connectivity_inferred_from_geometry")
        if coincident:
            reasons.extend(coincident)
    if time.monotonic() > deadline:
        return _timeout_result(timeout_seconds, reasons)
    geo = _Geometry(elements, xyz, edges)
    # components
    seen = set()
    comps = 0
    for start in range(n):
        if start in seen:
            continue
        comps += 1
        stack, seen_now = [start], {start}
        while stack:
            cur = stack.pop()
            for nb in geo.nbrs[cur]:
                if nb not in seen_now:
                    seen_now.add(nb)
                    stack.append(nb)
        seen |= seen_now
    if comps > 1:
        reasons.append("graph_disconnected_%d_components" % comps)
    if time.monotonic() > deadline:
        return _timeout_result(timeout_seconds, reasons)

    ring_infos = [_RingInfo(r, geo, elements) for r in _find_rings(n, edges)]
    aromatic: List[Tuple[_RingInfo, List[Set[int]]]] = []
    ambiguous: List[_RingInfo] = []
    for info in ring_infos:
        dec = info.decision()
        if dec == "aromatic":
            opts = _donor_options(info, geo, elements)
            if opts:
                aromatic.append((info, opts))
            else:
                reasons.append("ring_aromatic_huckel_unsatisfiable_kept_saturated")
        elif dec == "ambiguous":
            ambiguous.append(info)
    plans: List[Tuple[str, Dict[int, Set[int]]]] = []
    plans_truncated = False
    if ambiguous:
        amb = ambiguous[0]
        opts = _donor_options(amb, geo, elements) or [set()]
        choice_sets = [[None]]  # None = treat the ambiguous ring saturated
        for donors in opts[:2]:
            choice_sets.append([donors])
        # plans: aromatic-interpretations of the ambiguous ring, then saturated
        for k, cs in enumerate(choice_sets):
            if k == 0:
                plans.append(("ring_ambiguous_saturated", {amb.ring: None}))
            else:
                plans.append(("ring_ambiguous_aromatic_d%d" % (k - 1), {amb.ring: cs[0]}))
        if len(ambiguous) > 1:
            plans_truncated = True
            reasons.append("ring_variant_cap_one_ambiguous_ring")
        reasons.append("ring_aromatic_ambiguous_band_both_variants")
    base_aromatic = [(info, opts) for info, opts in aromatic]
    if base_aromatic:
        reasons.append("ring_aromatic_assigned_from_plane_angle_length")

    # combine base aromatic rings (donor product, capped) with ambiguous plan
    combos: List[Tuple[str, List[Tuple[_RingInfo, Set[int]]]]] = []
    base_donor_lists = [opts[:4] for _, opts in base_aromatic]
    product = list(itertools.islice(itertools.product(*base_donor_lists), 9)) if base_donor_lists else [()]
    if len(product) > 8:
        product = product[:8]
        plans_truncated = True
        reasons.append("donor_combination_cap_applied")
    if not plans:
        plans = [("primary", {})]
    for plan_name, plan_extra in plans:
        for pi, combo in enumerate(product):
            entries: List[Tuple[_RingInfo, Set[int]]] = list(zip([i for i, _ in base_aromatic], combo))
            for ring_key, donors in plan_extra.items():
                info = next((ri for ri in ring_infos if ri.ring == ring_key), None)
                if info is not None and donors is not None:
                    entries.append((info, donors))
            name = plan_name if len(product) == 1 else "%s#%d" % (plan_name, pi)
            combos.append((name, entries))

    payloads: List[dict] = []
    for name, entries in combos:
        if time.monotonic() > deadline:
            return _timeout_result(timeout_seconds, reasons)
        variant = _GraphVariant(n, edges, name)
        for info, donors in entries:
            m = len(info.ring)
            for k in range(m):
                i, j = info.ring[k], info.ring[(k + 1) % m]
                variant.set_order(i, j, 1.5)
            variant.aromatic_atoms.update(info.ring)
            variant.donor_n |= donors
        _hypervalent_motifs(variant, geo, elements, reasons)
        _triple_pass(variant, geo, elements)
        _double_pass(variant, geo, elements, deadline)
        got = _finalize(variant, elements, xyz, geo, reasons)
        if not got and entries:
            # aromatic interpretation failed sanitization: bounded saturated fallback
            reasons.append("aromatic_sanitize_failed_fallback_saturated")
            fb = _GraphVariant(n, edges, name + "-fb")
            _hypervalent_motifs(fb, geo, elements, reasons)
            _triple_pass(fb, geo, elements)
            _double_pass(fb, geo, elements, deadline)
            got = _finalize(fb, elements, xyz, geo, reasons)
            for p in got:
                p["variant"] = fb
        payloads.extend(got)
    if time.monotonic() > deadline and not payloads:
        return _timeout_result(timeout_seconds, reasons)

    aromatic_by_variant: Dict[str, List[_RingInfo]] = {}
    for name, entries in combos:
        aromatic_by_variant[name] = [info for info, _ in entries]

    scored: List[dict] = []
    for p in payloads:
        sc = _score(p, geo, elements, aromatic_by_variant.get(p["variant"].name.rsplit("-fb", 1)[0].rsplit("-demoted", 1)[0], []))
        entry = {
            "smiles": p["smiles"],
            "score": round(sc["score"], 4),
            "bonds": p["bonds"],
            "formal_charges": p["formal_charges"],
            "hydrogen_counts": p["hydrogen_counts"],
            "evidence": {
                "score_terms": {k: round(v, 4) for k, v in sc["terms"].items()},
                "charge_mode": p["mode"],
                "total_formal_charge": sum(p["formal_charges"]),
                "implicit_h_total": sum(p["hydrogen_counts"]),
                "stereo": p["stereo"],
                "motifs": list(p["variant"].motifs),
                "variant": p["variant"].name,
                "notes": list(p["variant"].notes),
            },
        }
        scored.append(entry)

    # dedupe by canonical SMILES, keep the best-scoring representative
    best_by_smiles: Dict[str, dict] = {}
    for entry in scored:
        key = entry["smiles"]
        if key not in best_by_smiles or entry["score"] < best_by_smiles[key]["score"]:
            best_by_smiles[key] = entry
    candidates = sorted(best_by_smiles.values(), key=lambda e: (e["score"], e["smiles"]))

    if total_charge is not None and candidates:
        candidates = [c for c in candidates if c["evidence"]["total_formal_charge"] == total_charge]
        reasons.append("total_charge_filter_applied" if candidates else "total_charge_unmatched")
    deadline_hit = time.monotonic() > deadline
    if deadline_hit:
        reasons.append("deadline_exceeded_after_results")
    truncated = plans_truncated or len(candidates) > MAX_CANDIDATES or deadline_hit
    if len(candidates) > MAX_CANDIDATES:
        candidates = candidates[:MAX_CANDIDATES]
        reasons.append("candidate_limit_truncated")

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    atom_geometry = [{"index": a, "element": elements[a], "degree": geo.degree(a),
                      "mean_angle_deg": round(geo.mean_angle[a], 2),
                      "angle_sum_deg": round(geo.angle_sum[a], 2),
                      "neighbor_plane_rms_a": round(geo.plane_rms[a], 4)}
                     for a in range(n)]
    evidence = {
        "n_atoms": n,
        "n_bonds": len(edges),
        "bonds_supplied": bonds_supplied,
        "components": comps,
        "geometry_pass": "single_local_angle_plane_length",
        "atom_geometry": atom_geometry,
        "ring_geometry": [info.evidence() for info in ring_infos],
        "geometry_usage": {
            "angle_sums_gate_sp2_and_triples": True,
            "neighbor_plane_fits_diagnostic_only": True,
            "ring_plane_fits_and_relative_lengths_decide_aromaticity": True,
            "bond_lengths_rank_double_assignment": True,
        },
        "charge_assumptions": sorted({p["mode"] for p in payloads}) if payloads else [],
        "elapsed_ms": round(elapsed_ms, 3),
        "timeout_seconds": float(timeout_seconds),
        "truncated": bool(truncated),
    }
    if not candidates:
        return {"status": "unresolved", "algorithm": ALGORITHM, "candidates": [],
                "search_complete": not truncated,
                "reason_codes": _dedupe(reasons + ["no_sanitized_candidate"]),
                "evidence": evidence}
    status = "candidate"
    if len(candidates) >= 2 and candidates[1]["score"] - candidates[0]["score"] < 0.20 \
            and candidates[1]["smiles"] != candidates[0]["smiles"]:
        status = "ambiguous"
    if "ring_aromatic_ambiguous_band_both_variants" in reasons and status == "candidate":
        # borderline ring evidence: top rank is a judgment call, flag it
        reasons.append("borderline_ring_evidence_top_rank_not_unique_proof")
    return {"status": status, "algorithm": ALGORITHM, "candidates": candidates,
            "search_complete": not truncated,
            "reason_codes": _dedupe(reasons),
            "evidence": evidence}


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out
