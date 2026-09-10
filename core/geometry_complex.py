"""complex_geometry_v2: experimental geometry-aware NNAA heavy-graph inference.

Bounded joint bond-order / formal-charge / implicit-H assignment from heavy-atom
3D coordinates plus (optional) undirected adjacency, per the shared contract in
``.zcode_nnaa_geometry_v2/ALGORITHM_CONTRACT.md``.

Algorithm outline (one module, no framework):

1. Validate the record; coordinates and supplied adjacency are preserved exactly.
   ``bonds=None`` triggers a bounded covalent-radius greedy connectivity pass.
2. Precompute static geometry: distance matrix, per-atom inter-neighbour angles,
   SSSR rings (RDKit FastFindRings on the single-bond graph) with planarity /
   bond-length-equalization classification for aromatic-geometry priors.
3. Bounded beam search over per-edge bond orders {1,2,3}.  Candidate orders are
   scored against element-pair/order reference lengths; orders with hopeless
   length evidence are hard-dropped.  Valence caps from explicit per-element
   valence-state tables prune expansions.  A length-greedy constructive
   assignment (own bounded candidate generator, evaluated through the identical
   scoring path) seeds diversity; RDKit DetermineBondOrders is intentionally
   NOT called so every step has a deterministic bound.
4. For each complete order assignment, charge/H microstates are enumerated by
   DFS over per-atom states (explicit valence models: N/O cation/anion states,
   S and Se valences 2/4/6, P valences 3/5 plus phosphonium, P-H allowed, no
   sp3d narrative - integer valence models only; charged carbon states are
   excluded because heavy-atom geometry cannot justify them).  If total charge
   is unknown, net charge is explored over {-1,0,+1} and reported as an
   assumption, never collapsed to a fake unique answer.
5. Each (orders, charges, H) triple becomes an RDKit mol built from the input
   coordinates with NoImplicit+NumExplicitHs; candidates survive only if
   Chem.SanitizeMol succeeds (RDKit is the valence gate, not the generator).
   Scoring: bond lengths (aromatic rings use ring-averaged references),
   post-assignment hybridization vs observed heavy angles/planarity
   (sp/sp2/sp3/pyramidal/bent/hypervalent classes), aromatic-ring agreement
   between geometry and perceived aromaticity, conjugation bonuses, and
   state/H/net-charge priors.  Lower score is better.
6. Kekule/resonance-equivalent assignments coalesce via canonical SMILES.
   Top <=8 candidates are returned with per-candidate evidence.  Any cap or
   deadline hit sets ``search_complete=false`` and a reason code.

Deterministic caps (see constants): beam width, per-assignment microstate
budget, global mol-build budget, and a wall-clock deadline checked inside the
beam, microstate and build loops.  Unsatisfiable valence constraints return an
honest ``unresolved`` result.  Uses only RDKit, numpy and stdlib; no file or
network access, no CCD/monomer library lookups, no atom names or residue
identifiers, no benchmark truth.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D


ALGORITHM = "complex_geometry_v2"
DEFAULT_TIMEOUT_SECONDS = 2.0

# --- deterministic search caps ------------------------------------------------
MAX_BEAM = 128                     # beam width over edge-order assignments
MAX_COMBOS_PER_ASSIGNMENT = 64     # charge/H microstates enumerated per assignment
MAX_MOL_BUILDS = 384               # global RDKit build+sanitize budget
MAX_CANDIDATES = 8
AMBIGUITY_MARGIN = 0.5             # score gap below which top-2 counts ambiguous
HARD_LENGTH_PENALTY = 16.0         # order options beyond this are dropped
LENGTH_PENALTY_CAP = 25.0
PARTIAL_CHARGE_WINDOW = 2          # transient DFS charge bound
CHARGE_STATES_UNKNOWN = (-1, 0, 1) # net charge explored when total_charge is null

# --- explicit valence-state models: element -> [(formal charge, valence B, prior)]
# Charged carbon states are deliberately excluded: with only heavy-atom geometry
# as evidence they are never distinguishable from the neutral microstate, and
# they triple the microstate search space without observable benefit.
ATOM_STATES: Dict[str, List[Tuple[int, int, float]]] = {
    "C":  [(0, 4, 0.0)],
    "N":  [(0, 3, 0.0), (1, 4, 0.25), (-1, 2, 0.45)],
    "O":  [(0, 2, 0.0), (-1, 1, 0.2)],
    "S":  [(0, 2, 0.0), (0, 4, 0.1), (0, 6, 0.1), (-1, 1, 0.3)],
    "Se": [(0, 2, 0.0), (0, 4, 0.2), (0, 6, 0.2), (-1, 1, 0.35)],
    "P":  [(0, 3, 0.0), (0, 5, 0.1), (1, 4, 0.5), (-1, 2, 0.9)],
    "B":  [(0, 3, 0.1), (-1, 4, 0.45)],
    "Si": [(0, 4, 0.0)],
    "As": [(0, 3, 0.1), (0, 5, 0.25), (1, 4, 0.6)],
    "F":  [(0, 1, 0.0), (-1, 0, 0.3)],
    "Cl": [(0, 1, 0.0), (-1, 0, 0.3)],
    "Br": [(0, 1, 0.0), (-1, 0, 0.3)],
    "I":  [(0, 1, 0.0), (0, 3, 0.9), (0, 5, 1.3), (-1, 0, 0.3)],
}

HMAX: Dict[str, int] = {  # max implicit hydrogens per element (no blanket P-H ban)
    "C": 4, "N": 3, "O": 2, "S": 2, "Se": 2, "P": 3, "B": 4, "Si": 4, "As": 3,
    "F": 0, "Cl": 0, "Br": 0, "I": 1,
}

H_PENALTY: Dict[str, Dict[int, float]] = {
    "P": {1: 0.2, 2: 0.45, 3: 0.7},
    "S": {1: 0.05, 2: 0.35},
    "Se": {1: 0.1, 2: 0.4},
    "O": {2: 0.25},
    "N": {3: 0.1},
}

# degree caps used only for bonds=null connectivity inference
DEGREE_CAPS: Dict[str, int] = {
    "C": 4, "N": 4, "O": 3, "S": 4, "Se": 4, "P": 4, "B": 4, "Si": 4, "As": 4,
    "F": 1, "Cl": 1, "Br": 1, "I": 3,
}
DEFAULT_DEGREE_CAP = 6
CONNECTIVITY_TOLERANCE = 0.45  # Angstrom slack over covalent radii sum

# --- reference bond lengths (Angstrom) by sorted element pair and order ------
_BLEN: Dict[Tuple[str, str], Dict[int, float]] = {
    ("B", "C"): {1: 1.56}, ("B", "N"): {1: 1.56}, ("B", "O"): {1: 1.36},
    ("As", "O"): {1: 1.74}, ("As", "N"): {1: 1.84}, ("As", "C"): {1: 1.96},
    ("Br", "Br"): {1: 2.28}, ("Br", "C"): {1: 1.94}, ("Br", "O"): {1: 1.79},
    ("Br", "P"): {1: 2.26}, ("Br", "S"): {1: 2.21},
    ("C", "C"): {1: 1.54, 2: 1.34, 3: 1.20},
    ("C", "Cl"): {1: 1.77}, ("C", "F"): {1: 1.35}, ("C", "I"): {1: 2.14},
    ("C", "N"): {1: 1.47, 2: 1.29, 3: 1.16},
    ("C", "O"): {1: 1.43, 2: 1.23},
    ("C", "P"): {1: 1.84, 2: 1.66},
    ("C", "S"): {1: 1.82, 2: 1.60},
    ("C", "Se"): {1: 1.97, 2: 1.73},
    ("C", "Si"): {1: 1.89},
    ("Cl", "Cl"): {1: 1.99}, ("Cl", "N"): {1: 1.75}, ("Cl", "O"): {1: 1.68},
    ("Cl", "P"): {1: 2.04}, ("Cl", "S"): {1: 2.07},
    ("F", "N"): {1: 1.36}, ("F", "O"): {1: 1.42}, ("F", "P"): {1: 1.57},
    ("F", "S"): {1: 1.56},
    ("I", "I"): {1: 2.67}, ("I", "N"): {1: 2.10}, ("I", "O"): {1: 2.00},
    ("N", "N"): {1: 1.45, 2: 1.25, 3: 1.10},
    ("N", "O"): {1: 1.40, 2: 1.21},
    ("N", "P"): {1: 1.70}, ("N", "S"): {1: 1.65}, ("N", "Se"): {1: 1.86},
    ("O", "O"): {1: 1.47},
    ("O", "P"): {1: 1.61, 2: 1.48},
    ("O", "S"): {1: 1.58, 2: 1.44},
    ("O", "Se"): {1: 1.72, 2: 1.60},
    ("O", "Si"): {1: 1.64},
    ("P", "P"): {1: 2.21}, ("P", "S"): {1: 2.05, 2: 1.92},
    ("S", "S"): {1: 2.05}, ("Se", "Se"): {1: 2.34},
}

_LOOSE_SIGMA_ELEMENTS = {"P", "S", "Se", "As", "Si", "I", "Br"}
_AROMATIC_RING_ELEMENTS = {"C", "N", "O", "S", "Se"}
_BOND_TYPE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
              3: Chem.BondType.TRIPLE}

_PERIODIC = Chem.GetPeriodicTable()


# =============================================================================
# validation
# =============================================================================
def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate(record: Any) -> Tuple[Optional[List[Tuple[str, np.ndarray]]],
                                    Optional[List[Tuple[int, int]]],
                                    Optional[int], List[str]]:
    """Return (atoms, bonds or None, total_charge or None, errors)."""
    errors: List[str] = []
    if not isinstance(record, dict):
        return None, None, None, ["record_not_dict"]
    raw_atoms = record.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        return None, None, None, ["atoms_missing_or_empty"]
    atoms: List[Tuple[str, np.ndarray]] = []
    for pos, atom in enumerate(raw_atoms):
        if not isinstance(atom, dict):
            errors.append(f"atom_{pos}_not_dict")
            continue
        atom_id = atom.get("id")
        if not isinstance(atom_id, int) or isinstance(atom_id, bool):
            errors.append(f"atom_{pos}_bad_id")
            continue
        if atom_id != pos:
            errors.append(f"atom_id_{atom_id}_noncontiguous")
            continue
        element = atom.get("element")
        if not isinstance(element, str):
            errors.append(f"atom_{pos}_bad_element")
            continue
        element = element.strip().capitalize()
        try:
            _PERIODIC.GetAtomicNumber(element)
        except (RuntimeError, ValueError):
            errors.append(f"atom_{pos}_unknown_element_{element}")
            continue
        if element == "H":
            errors.append(f"atom_{pos}_explicit_hydrogen")
            continue
        xyz = atom.get("xyz")
        if (not isinstance(xyz, (list, tuple)) or len(xyz) != 3
                or not all(_finite(v) for v in xyz)):
            errors.append(f"atom_{pos}_bad_xyz")
            continue
        atoms.append((element, np.array([float(v) for v in xyz], dtype=float)))
    if errors or len(atoms) != len(raw_atoms):
        return None, None, None, errors

    total_charge: Optional[int] = None
    raw_q = record.get("total_charge", None)
    if raw_q is not None:
        if isinstance(raw_q, bool):
            errors.append("total_charge_not_integer")
        elif isinstance(raw_q, int):
            total_charge = raw_q
        elif isinstance(raw_q, float) and raw_q.is_integer():
            total_charge = int(raw_q)
        else:
            errors.append("total_charge_not_integer")
    if errors:
        return None, None, None, errors

    raw_bonds = record.get("bonds", None)
    if raw_bonds is None:
        return atoms, None, total_charge, []
    if not isinstance(raw_bonds, list):
        return None, None, None, ["bonds_not_list_or_null"]
    n = len(atoms)
    bonds: List[Tuple[int, int]] = []
    seen = set()
    for pos, bond in enumerate(raw_bonds):
        if (not isinstance(bond, (list, tuple)) or len(bond) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool)
                           for v in bond)):
            errors.append(f"bond_{pos}_malformed")
            continue
        i, j = int(bond[0]), int(bond[1])
        if i == j:
            errors.append(f"bond_{pos}_self_loop")
            continue
        if not (0 <= i < n and 0 <= j < n):
            errors.append(f"bond_{pos}_endpoint_out_of_range")
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue  # duplicate adjacency is tolerated
        seen.add(key)
        bonds.append(key)
    if errors:
        return None, None, None, errors
    return atoms, bonds, total_charge, []


# =============================================================================
# geometry precompute
# =============================================================================
class _Geometry:
    """Static per-record geometry: distances, angles, rings."""

    def __init__(self, elements: List[str], pos: np.ndarray,
                 adjacency: List[Tuple[int, int]]):
        self.elements = elements
        self.n = len(elements)
        self.pos = pos
        self.adjacency = adjacency
        self.neighbors: List[List[int]] = [[] for _ in range(self.n)]
        self.edge_index: Dict[Tuple[int, int], int] = {}
        for idx, (i, j) in enumerate(adjacency):
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
            self.edge_index[(i, j)] = idx
        diff = pos[:, None, :] - pos[None, :, :]
        self.dist = np.sqrt(np.maximum((diff * diff).sum(-1), 0.0))
        self._angles()
        self._rings()

    def _angles(self) -> None:
        self.angles: List[List[Tuple[float, int, int]]] = []
        for i in range(self.n):
            entries = []
            nbrs = sorted(self.neighbors[i])
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    j, k = nbrs[a], nbrs[b]
                    v1 = self.pos[j] - self.pos[i]
                    v2 = self.pos[k] - self.pos[i]
                    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                    if norm < 1e-9:
                        continue
                    cosv = float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))
                    entries.append((math.degrees(math.acos(cosv)), j, k))
            self.angles.append(entries)

    def _rings(self) -> None:
        self.rings: List[Dict[str, Any]] = []
        rw = Chem.RWMol()
        for el in self.elements:
            rw.AddAtom(Chem.Atom(el))
        for i, j in self.adjacency:
            rw.AddBond(i, j, Chem.BondType.SINGLE)
        rw.UpdatePropertyCache(False)
        Chem.FastFindRings(rw)
        info = rw.GetRingInfo()
        for atom_ring in info.AtomRings():
            if len(atom_ring) not in (5, 6):
                continue
            cycle = list(atom_ring)  # FastFindRings returns cycle order
            ring = sorted(cycle)
            if not all(self.elements[a] in _AROMATIC_RING_ELEMENTS
                       for a in ring):
                continue
            rset = set(ring)
            edges = []
            for a in ring:
                for b in self.neighbors[a]:
                    if b in rset and a < b:
                        edges.append((a, b))
            if len(edges) != len(ring):
                continue
            cycle_edges = []
            for k in range(len(cycle)):
                a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                if a == b:
                    cycle_edges = []
                    break
                cycle_edges.append((min(a, b), max(a, b)))
            if not cycle_edges:
                continue
            lengths = [float(self.dist[a, b]) for a, b in edges]
            spread = max(lengths) - min(lengths)
            mean_len = float(sum(lengths) / len(lengths))
            pts = self.pos[ring]
            centred = pts - pts.mean(axis=0)
            if np.linalg.norm(centred) < 1e-9:
                rmsd = 0.0
            else:
                # residual from best-fit plane = smallest singular value
                _, svals, _ = np.linalg.svd(centred, full_matrices=False)
                residual = float(svals[2]) if len(svals) >= 3 else 0.0
                rmsd = max(residual, 0.0) / math.sqrt(len(ring))
            aromatic_geom = (rmsd <= 0.10 and spread <= 0.12
                             and 1.30 <= mean_len <= 1.48)
            self.rings.append({
                "atoms": ring, "edges": edges, "cycle_edges": cycle_edges,
                "size": len(ring),
                "planar_rmsd": rmsd, "length_spread": spread,
                "mean_length": mean_len, "aromatic_geometry": aromatic_geom,
            })
        self.ring_edge_owner: Dict[Tuple[int, int], int] = {}
        for ridx, ring in enumerate(self.rings):
            for edge in ring["edges"]:
                self.ring_edge_owner.setdefault(edge, ridx)

    def ring_atoms_in_geom_aromatic(self) -> set:
        out = set()
        for ring in self.rings:
            if ring["aromatic_geometry"]:
                out.update(ring["atoms"])
        return out


def _element_pair_sigma(ref: float, e1: str, e2: str) -> float:
    sigma = 0.03 + 0.012 * ref
    if e1 in _LOOSE_SIGMA_ELEMENTS or e2 in _LOOSE_SIGMA_ELEMENTS:
        sigma *= 1.30
    return sigma


def _reference_length(e1: str, e2: str, order: int) -> float:
    table = _BLEN.get((min(e1, e2), max(e1, e2)))
    if table and order in table:
        return table[order]
    r1 = _PERIODIC.GetRcovalent(_PERIODIC.GetAtomicNumber(e1))
    r2 = _PERIODIC.GetRcovalent(_PERIODIC.GetAtomicNumber(e2))
    base = r1 + r2
    return base * {1: 1.0, 2: 0.88, 3: 0.78}[order]


def _length_penalty(dist: float, e1: str, e2: str, order: int) -> float:
    ref = _reference_length(e1, e2, order)
    sigma = _element_pair_sigma(ref, e1, e2)
    return min(((dist - ref) / sigma) ** 2, LENGTH_PENALTY_CAP)


# =============================================================================
# states
# =============================================================================
def _states_for(element: str) -> List[Tuple[int, int, float]]:
    if element in ATOM_STATES:
        return ATOM_STATES[element]
    z = _PERIODIC.GetAtomicNumber(element)
    try:
        vals = [v for v in _PERIODIC.GetValenceList(z) if v > 0]
    except (RuntimeError, ValueError):
        vals = []
    return [(0, v, 0.1) for v in vals] or [(0, 0, 0.3)]


def _infer_connectivity(elements: List[str], pos: np.ndarray,
                        dist: np.ndarray) -> List[Tuple[int, int]]:
    n = len(elements)
    caps = [DEGREE_CAPS.get(e, DEFAULT_DEGREE_CAP) for e in elements]
    radii = [_PERIODIC.GetRcovalent(_PERIODIC.GetAtomicNumber(e)) for e in elements]
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            if 0.4 <= d <= radii[i] + radii[j] + CONNECTIVITY_TOLERANCE:
                candidates.append((d, i, j))
    candidates.sort()
    degree = [0] * n
    edges: List[Tuple[int, int]] = []
    for d, i, j in candidates:
        if degree[i] < caps[i] and degree[j] < caps[j]:
            edges.append((i, j))
            degree[i] += 1
            degree[j] += 1
    edges.sort()
    return edges


# =============================================================================
# hybridization / geometry scoring post assignment
# =============================================================================
_HYPER_ELEMENTS = {"S", "Se", "P", "As", "I"}


def _atom_geom_class(element: str, has_triple: bool, n_doubles: int, degree: int,
                     bond_sum: int, neighbor_has_double: bool,
                     in_geom_aromatic_ring: bool) -> Tuple[str, float, float]:
    """Return (class label, target angle deg, sigma deg) for an atom.

    Hypervalent S/Se/P/As/I geometry follows the observed coordination number
    (4-coordinate sulfonyl/phosphoryl centres are tetrahedral), not the formal
    bond-order sum - no sp3d narrative, integer valence models only.
    """
    if element in _HYPER_ELEMENTS and degree >= 6:
        return "hypervalent_octa", -1.0, 15.0  # {90,180} nearest
    if element in _HYPER_ELEMENTS and (degree >= 4 or bond_sum >= 4):
        # 4-coordinate sulfonyl/phosphoryl centres: tetrahedral, loose sigma
        return "hypervalent_tetrahedral", 109.5, 13.0
    if has_triple or n_doubles >= 2:
        return "sp_linear", 180.0, 9.0
    if n_doubles == 1:
        return "sp2_planar", 120.0, 12.0
    if element == "O":
        return "bent", 111.0, 11.0
    if element in ("S", "Se"):
        return "bent", 98.0, 11.0
    if element in ("N", "P", "As") and 2 <= degree <= 3:
        if neighbor_has_double or in_geom_aromatic_ring:
            return "planar_conjugated", 118.0, 13.0
        return "pyramidal", 109.5, 14.0
    return "sp3", 109.5, 12.0


def _angle_score(geom: _Geometry, orders: List[int],
                 geom_aromatic_atoms: set) -> Tuple[float, Dict[int, str], int]:
    """Score observed heavy angles against post-assignment hybridization."""
    total = 0.0
    labels: Dict[int, str] = {}
    mismatches = 0
    for i in range(geom.n):
        entries = geom.angles[i]
        if not entries:
            labels[i] = "terminal_or_isolated"
            continue
        orders_here = [orders[geom.edge_index[(min(i, j), max(i, j))]]
                       for j in geom.neighbors[i]]
        has_triple = 3 in orders_here
        n_doubles = orders_here.count(2)
        bond_sum = sum(orders_here)
        neighbor_has_double = False
        for j in geom.neighbors[i]:
            for k in geom.neighbors[j]:
                if k == j or k == i:
                    continue
                ojk = orders[geom.edge_index[(min(j, k), max(j, k))]]
                if ojk >= 2:
                    neighbor_has_double = True
                    break
            if neighbor_has_double:
                break
        label, target, sigma = _atom_geom_class(
            geom.elements[i], has_triple, n_doubles,
            len(geom.neighbors[i]), bond_sum, neighbor_has_double,
            i in geom_aromatic_atoms)
        labels[i] = label
        atom_pen = 0.0
        for theta_deg, _j, _k in entries:
            if target < 0:  # octahedron-like: nearest of {90, 180}
                pen = min(((theta_deg - 90.0) / 15.0) ** 2,
                          ((theta_deg - 180.0) / 15.0) ** 2)
            else:
                pen = ((theta_deg - target) / sigma) ** 2
            atom_pen += min(pen, 8.0)
        if label == "sp2_planar" and len(entries) == 3:
            angle_sum = sum(e[0] for e in entries)
            if angle_sum < 350.0:
                atom_pen += min(0.6 * ((350.0 - angle_sum) / 10.0) ** 2, 8.0)
        atom_pen = min(atom_pen, 24.0)
        total += atom_pen
        if atom_pen > 4.0:
            mismatches += 1
    return total, labels, mismatches


def _ring_aromatizable(ring: Dict[str, Any], orders: Sequence[int],
                       edge_index: Dict[Tuple[int, int], int]) -> bool:
    """Cheap Kekule-pattern check used only to rank which full assignments
    deserve the mol-build budget first (final scoring uses RDKit's own
    aromaticity perception, which stays authoritative)."""
    seq = [orders[edge_index[e]] for e in ring["cycle_edges"]]
    if any(o not in (1, 2) for o in seq):
        return False
    k = len(seq)
    if k == 6:
        return all(seq[i] != seq[(i + 1) % k] for i in range(k))
    if k == 5:
        twos = [i for i, o in enumerate(seq) if o == 2]
        if len(twos) != 2:
            return False
        a, b = twos
        return (a - b) % k not in (1, k - 1)
    return False


def _final_estimate(geom: _Geometry, orders: Sequence[int],
                    base_pen: List[Dict[int, float]],
                    angle_cache: Dict[Tuple[int, ...],
                                      Tuple[float, Dict[int, str], int]],
                    geom_aromatic_atoms: set) -> float:
    """Build-budget ordering estimate: length + angle + ring agreement.

    Mirrors _score_candidate closely enough to rank assignments without
    building RDKit mols; the authoritative score is computed per candidate.
    """
    if orders not in angle_cache:
        angle_cache[orders] = _angle_score(geom, list(orders),
                                           geom_aromatic_atoms)
    angle_score = angle_cache[orders][0]
    length_score = 0.0
    ring_score = 0.0
    used_override = set()
    for ring in geom.rings:
        if ring["aromatic_geometry"] and \
                _ring_aromatizable(ring, orders, geom.edge_index):
            used_override.update(ring["edges"])
            ring_score -= 0.6 * ring["size"]
        elif ring["aromatic_geometry"] != _ring_aromatizable(
                ring, orders, geom.edge_index):
            ring_score += 1.2 * ring["size"]
    for eidx, (i, j) in enumerate(geom.adjacency):
        if (i, j) in used_override:
            ring = geom.rings[geom.ring_edge_owner[(i, j)]]
            ref = ring["mean_length"]
            sigma = _element_pair_sigma(ref, geom.elements[i], geom.elements[j])
            length_score += min(((float(geom.dist[i, j]) - ref) / sigma) ** 2,
                                LENGTH_PENALTY_CAP)
        else:
            length_score += base_pen[eidx][int(orders[eidx])]
    return length_score + angle_score + ring_score


def _conjugation_bonus(geom: _Geometry, orders: List[int]) -> float:
    bonus = 0.0
    lone_pair = {"N", "O", "S", "Se", "P"}
    for i in range(geom.n):
        if geom.elements[i] not in lone_pair:
            continue
        orders_here = [orders[geom.edge_index[(min(i, j), max(i, j))]]
                       for j in geom.neighbors[i]]
        if orders_here and all(o == 1 for o in orders_here):
            for j in geom.neighbors[i]:
                for k in geom.neighbors[j]:
                    if k == i:
                        continue
                    if orders[geom.edge_index[(min(j, k), max(j, k))]] >= 2:
                        bonus -= 0.25
                        break
                else:
                    continue
                break
    bonus = max(bonus, -1.5)
    diene = 0
    for i, j in geom.adjacency:
        e = geom.edge_index[(i, j)]
        if orders[e] != 1:
            continue

        def _other_double(a: int, b: int) -> bool:
            for k in geom.neighbors[a]:
                if k == b:
                    continue
                if orders[geom.edge_index[(min(a, k), max(a, k))]] >= 2:
                    return True
            return False

        if _other_double(i, j) and _other_double(j, i):
            diene += 1
    bonus -= min(0.15 * diene, 0.6)
    return bonus


# =============================================================================
# candidate build & score
# =============================================================================
def _build_mol(geom: _Geometry, orders: List[int], charges: List[int],
               hs: List[int]):
    rw = Chem.RWMol()
    for idx, element in enumerate(geom.elements):
        atom = Chem.Atom(element)
        atom.SetFormalCharge(int(charges[idx]))
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(hs[idx]))
        rw.AddAtom(atom)
    for eidx, (i, j) in enumerate(geom.adjacency):
        rw.AddBond(i, j, _BOND_TYPE[int(orders[eidx])])
    conf = Chem.Conformer(geom.n)
    for idx in range(geom.n):
        x, y, z = geom.pos[idx]
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    rw.AddConformer(conf, assignId=True)
    return rw.GetMol()


def _score_candidate(geom: _Geometry, orders: List[int], charges: List[int],
                     hs: List[int], mol, state_prior: float,
                     net_charge_known: bool,
                     geom_aromatic_atoms: set,
                     angle_cached: Tuple[float, Dict[int, str], int]
                     ) -> Dict[str, Any]:
    aromatic_edges = set()
    for b in mol.GetBonds():
        if b.GetIsAromatic():
            aromatic_edges.add((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())))

    length_score = 0.0
    for eidx, (i, j) in enumerate(geom.adjacency):
        e = (i, j)
        if e in aromatic_edges and e in geom.ring_edge_owner:
            ring = geom.rings[geom.ring_edge_owner[e]]
            ref = ring["mean_length"]
            sigma = _element_pair_sigma(ref, geom.elements[i], geom.elements[j])
            pen = min(((float(geom.dist[i, j]) - ref) / sigma) ** 2,
                      LENGTH_PENALTY_CAP)
        else:
            pen = _length_penalty(float(geom.dist[i, j]), geom.elements[i],
                                  geom.elements[j], int(orders[eidx]))
        length_score += pen

    ring_score = 0.0
    for ring in geom.rings:
        perceived = all(e in aromatic_edges for e in ring["edges"])
        if ring["aromatic_geometry"] and perceived:
            ring_score -= 0.6 * ring["size"]
        elif ring["aromatic_geometry"] != perceived:
            ring_score += 1.2 * ring["size"]

    angle_score, labels, mismatches = angle_cached
    conjugation = _conjugation_bonus(geom, orders)
    radicals = int(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    net_q = int(sum(charges))
    prior_score = state_prior + (0.0 if net_charge_known
                                 else 0.3 * abs(net_q)) + 4.0 * radicals
    total = (length_score + angle_score + ring_score + prior_score
             + conjugation)
    Chem.AssignStereochemistryFrom3D(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    smiles = Chem.MolToSmiles(mol)
    hypervalent = [i for i in range(geom.n)
                   if geom.elements[i] in _HYPER_ELEMENTS
                   and sum(orders[geom.edge_index[(min(i, j), max(i, j))]]
                           for j in geom.neighbors[i]) >= 4]
    return {
        "smiles": smiles,
        "score": round(float(total), 4),
        "bonds": [[i, j, 1.5 if (i, j) in aromatic_edges else int(orders[eidx])]
                  for eidx, (i, j) in enumerate(geom.adjacency)],
        "formal_charges": [int(q) for q in charges],
        "hydrogen_counts": [int(h) for h in hs],
        "evidence": {
            "score_length": round(float(length_score), 4),
            "score_angle": round(float(angle_score), 4),
            "score_ring": round(float(ring_score + conjugation), 4),
            "score_priors": round(float(prior_score), 4),
            "charge_total": net_q,
            "hydrogen_total": int(sum(hs)),
            "aromatic_ring_bonds": len(aromatic_edges),
            "radical_electrons": radicals,
            "hybridization": labels,
            "hybridization_mismatch_count": mismatches,
            "hypervalent_atoms": hypervalent,
        },
    }


# =============================================================================
# main entry
# =============================================================================
def infer_geometry(record: dict, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
                   ) -> dict:
    """Infer bond orders, formal charges and implicit H from heavy-atom geometry.

    See the module docstring and ALGORITHM_CONTRACT.md for the I/O schema.
    """
    started = time.perf_counter()
    base = {"algorithm": ALGORITHM}

    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not _finite(timeout_seconds) or timeout_seconds <= 0):
        return {**base, "status": "invalid_input", "candidates": [],
                "search_complete": False,
                "reason_codes": ["invalid_timeout_seconds"],
                "evidence": {}}
    deadline = started + float(timeout_seconds)

    atoms, bonds, total_charge, errors = _validate(record)
    if atoms is None:
        return {**base, "status": "invalid_input", "candidates": [],
                "search_complete": False, "reason_codes":
                    [f"invalid_input:{e}" for e in errors] or ["invalid_input"],
                "evidence": {"validation_errors": errors}}

    elements = [e for e, _ in atoms]
    pos = np.array([p for _, p in atoms], dtype=float)
    n = len(elements)

    def _exceeded() -> bool:
        return time.perf_counter() > deadline

    reason_codes: List[str] = []
    bonds_source = "input"
    if bonds is None:
        dist0 = np.sqrt(np.maximum(
            ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1), 0.0))
        bonds = _infer_connectivity(elements, pos, dist0)
        bonds_source = "inferred_geometry"
        reason_codes.append("connectivity_inferred")
    geom = _Geometry(elements, pos, bonds)
    m = len(geom.adjacency)

    # per-atom valence caps and states
    states = [_states_for(e) for e in elements]
    max_b = [max(s[1] for s in st) for st in states]

    # per-edge candidate orders with length penalties
    edge_options: List[List[Tuple[int, float]]] = []
    for i, j in geom.adjacency:
        pair_key = (min(elements[i], elements[j]),
                    max(elements[i], elements[j]))
        opts = []
        for order in (1, 2, 3):
            if pair_key not in _BLEN and order > 1:
                continue  # unknown element pair: single bonds only (fallback radii)
            pen = _length_penalty(float(geom.dist[i, j]), elements[i],
                                  elements[j], order)
            opts.append((order, pen))
        opts.sort(key=lambda t: (t[1], t[0]))
        kept = [(o, p) for o, p in opts if p <= HARD_LENGTH_PENALTY]
        if not kept:
            kept = [opts[0]]
        edge_options.append(kept)

    # most-constrained edges first (few options, then most informative spread)
    edge_order = sorted(
        range(m),
        key=lambda e: (len(edge_options[e]),
                       -(max(p for _, p in edge_options[e])
                         - min(p for _, p in edge_options[e])), e))

    beam_truncated = False
    deadline_hit = False
    finals: List[Tuple[float, Tuple[int, ...]]] = []

    if m == 0:
        finals = [(0.0, ())]
    else:
        beam: List[Tuple[float, Tuple[int, ...], Tuple[int, ...]]] = [
            (0.0, (), tuple([0] * n))]
        for step, e in enumerate(edge_order):
            if _exceeded():
                deadline_hit = True
                beam = []
                break
            i, j = geom.adjacency[e]
            children = []
            for score, orders, sums in beam:
                for order, pen in edge_options[e]:
                    if sums[i] + order > max_b[i] or sums[j] + order > max_b[j]:
                        continue
                    new_sums = list(sums)
                    new_sums[i] += order
                    new_sums[j] += order
                    children.append((score + pen, orders + (order,),
                                     tuple(new_sums)))
            if not children:
                beam = []
                break
            children.sort(key=lambda t: (t[0], t[1]))
            if len(children) > MAX_BEAM:
                children = children[:MAX_BEAM]
                beam_truncated = True
            beam = children
        if beam and not deadline_hit:
            # beam orders follow edge-processing order; remap to adjacency order
            finals = []
            for score, proc_orders, _ in beam:
                adj_orders = [0] * m
                for k, e in enumerate(edge_order):
                    adj_orders[e] = proc_orders[k]
                finals.append((score, tuple(adj_orders)))
        elif deadline_hit and beam:
            finals = []

    if not deadline_hit and finals:
        # length-greedy constructive seed (bounded own generator component)
        sums = [0] * n
        greedy_orders: List[int] = [0] * m
        feasible = True
        for e in range(m):
            i, j = geom.adjacency[e]
            pick = None
            for order, _pen in sorted(edge_options[e]):
                if sums[i] + order <= max_b[i] and sums[j] + order <= max_b[j]:
                    pick = order
                    break
            if pick is None:
                feasible = False
                break
            greedy_orders[e] = pick
            sums[i] += pick
            sums[j] += pick
        if feasible:
            greedy_score = sum(
                next(p for o, p in edge_options[e] if o == greedy_orders[e])
                for e in range(m))
            merged: Dict[Tuple[int, ...], float] = {}
            for score, orders in finals:
                if orders not in merged or score < merged[orders]:
                    merged[orders] = score
            merged[tuple(greedy_orders)] = min(
                merged.get(tuple(greedy_orders), greedy_score), greedy_score)
            finals = [(score, orders) for orders, score in merged.items()]

    # iterate complete order assignments, best geometry-aware estimate first
    base_pen: List[Dict[int, float]] = []
    for i, j in geom.adjacency:
        base_pen.append({
            order: _length_penalty(float(geom.dist[i, j]), elements[i],
                                   elements[j], order)
            for order in (1, 2, 3)
        })
    geom_aromatic_atoms = geom.ring_atoms_in_geom_aromatic()
    angle_cache: Dict[Tuple[int, ...], Tuple[float, Dict[int, str], int]] = {}
    ranked_finals: List[Tuple[float, Tuple[int, ...]]] = []
    for _prefix, orders in finals:
        est = _final_estimate(geom, orders, base_pen, angle_cache,
                              geom_aromatic_atoms)
        ranked_finals.append((est, orders))
    ranked_finals.sort(key=lambda t: (t[0], t[1]))
    finals = ranked_finals
    allowed_totals = (total_charge,) if total_charge is not None \
        else CHARGE_STATES_UNKNOWN

    combo_truncated = False
    build_truncated = False
    builds = 0
    sanitize_failures = 0
    infeasible_assignments = 0
    best_by_smiles: Dict[str, Dict[str, Any]] = {}

    for prefix_score, orders in finals:
        if _exceeded():
            deadline_hit = True
            break
        if builds >= MAX_MOL_BUILDS:
            build_truncated = True
            break
        angle_cached = angle_cache.get(orders) or _angle_score(
            geom, list(orders), geom_aromatic_atoms)
        sums = [0] * n
        for e, (i, j) in enumerate(geom.adjacency):
            sums[i] += orders[e]
            sums[j] += orders[e]
        # per-atom microstate options (q, h, prior)
        per_atom: List[List[Tuple[int, int, float]]] = []
        feasible = True
        for idx in range(n):
            opts = []
            for q, b_val, prior in states[idx]:
                h = b_val - sums[idx]
                if 0 <= h <= HMAX.get(elements[idx], 0):
                    opts.append((q, h, prior
                                 + H_PENALTY.get(elements[idx], {}).get(h, 0.0)))
            if not opts:
                feasible = False
                break
            per_atom.append(opts)
        if not feasible:
            infeasible_assignments += 1
            continue

        order_atoms = sorted(range(n), key=lambda i: (-len(per_atom[i]), i))
        min_q_remaining = [0] * (n + 1)
        max_q_remaining = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            mn = min(o[0] for o in per_atom[order_atoms[k]])
            mx = max(o[0] for o in per_atom[order_atoms[k]])
            min_q_remaining[k] = min_q_remaining[k + 1] + mn
            max_q_remaining[k] = max_q_remaining[k + 1] + mx
        charge_combo: List[int] = [0] * n
        h_combo: List[int] = [0] * n
        prior_combo: List[float] = [0.0] * n
        combos: List[Tuple[float, Tuple[int, ...], Tuple[int, ...]]] = []
        combo_budget = MAX_COMBOS_PER_ASSIGNMENT
        allowed_min, allowed_max = min(allowed_totals), max(allowed_totals)
        local_truncated = False

        def _dfs(k: int, partial_q: int) -> None:
            nonlocal combo_budget, local_truncated
            if combo_budget <= 0:
                local_truncated = True
                return
            if k == n:
                if partial_q in allowed_totals:
                    combo_budget -= 1
                    combos.append((sum(prior_combo),
                                   tuple(charge_combo), tuple(h_combo)))
                return
            if _exceeded():
                local_truncated = True
                return
            atom_idx = order_atoms[k]
            for q, h, prior in per_atom[atom_idx]:
                new_q = partial_q + q
                if new_q > PARTIAL_CHARGE_WINDOW or \
                        new_q < -PARTIAL_CHARGE_WINDOW:
                    local_truncated = True
                    continue
                if new_q + max_q_remaining[k + 1] < allowed_min or \
                        new_q + min_q_remaining[k + 1] > allowed_max:
                    continue
                charge_combo[atom_idx] = q
                h_combo[atom_idx] = h
                prior_combo[atom_idx] = prior
                _dfs(k + 1, new_q)
            charge_combo[atom_idx] = 0
            h_combo[atom_idx] = 0
            prior_combo[atom_idx] = 0.0

        _dfs(0, 0)
        if local_truncated:
            combo_truncated = True
            deadline_hit = deadline_hit or _exceeded()
        combos.sort(key=lambda t: (t[0], t[1], t[2]))
        for state_prior, charges, hs in combos:
            if builds >= MAX_MOL_BUILDS:
                build_truncated = True
                break
            if _exceeded():
                deadline_hit = True
                break
            mol = _build_mol(geom, list(orders), list(charges), list(hs))
            builds += 1
            try:
                Chem.SanitizeMol(mol)
            except Exception:  # noqa: BLE001 - RDKit valence/kekulize gate
                sanitize_failures += 1
                continue
            cand = _score_candidate(geom, list(orders), list(charges), list(hs),
                                    mol, state_prior,
                                    total_charge is not None,
                                    geom_aromatic_atoms, angle_cached)
            prev = best_by_smiles.get(cand["smiles"])
            if prev is None or cand["score"] < prev["score"]:
                best_by_smiles[cand["smiles"]] = cand
        if deadline_hit or build_truncated:
            break

    candidates = sorted(best_by_smiles.values(),
                        key=lambda c: (c["score"], c["smiles"]))[:MAX_CANDIDATES]

    if beam_truncated:
        reason_codes.append("beam_truncated")
    if combo_truncated:
        reason_codes.append("charge_enumeration_truncated")
    if build_truncated:
        reason_codes.append("mol_build_budget_exhausted")
    if deadline_hit:
        reason_codes.append("deadline_exceeded")

    output_truncated = len(best_by_smiles) > MAX_CANDIDATES
    if output_truncated:
        reason_codes.append("candidate_output_truncated")
    truncated = (beam_truncated or combo_truncated or build_truncated
                 or deadline_hit or output_truncated)

    if not candidates:
        if deadline_hit:
            status = "timeout"
        else:
            status = "unresolved"
            if infeasible_assignments and not finals:
                reason_codes.append("no_valence_consistent_assignment")
            elif sanitize_failures and finals:
                reason_codes.append("sanitize_failed_all")
            elif finals:
                reason_codes.append("no_charge_consistent_microstate")
            else:
                reason_codes.append("no_feasible_bond_assignment")
    elif (len(candidates) >= 2
          and candidates[1]["score"] - candidates[0]["score"] < AMBIGUITY_MARGIN
          and candidates[1]["smiles"] != candidates[0]["smiles"]):
        status = "ambiguous"
        reason_codes.append("degenerate_top_candidates")
    else:
        status = "candidate"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    evidence = {
        "n_atoms": n,
        "n_edges": m,
        "bonds_source": bonds_source,
        "geometry_usage": ["bond_lengths", "heavy_atom_angles",
                           "planarity_angle_sum", "ring_aromaticity_geometry",
                           "post_assignment_hybridization_check"],
        "charge_search": {
            "total_charge_input": total_charge,
            "net_charge_states_explored": list(allowed_totals),
            "assumption": "fixed_input_charge" if total_charge is not None
            else "bounded_net_charge_exploration_-1_0_+1",
        },
        "search": {
            "beam_width_cap": MAX_BEAM,
            "beam_final_assignments": len(finals),
            "beam_truncated": beam_truncated,
            "microstate_cap_per_assignment": MAX_COMBOS_PER_ASSIGNMENT,
            "microstate_enumeration_truncated": combo_truncated,
            "mol_builds": builds,
            "mol_build_cap": MAX_MOL_BUILDS,
            "mol_build_truncated": build_truncated,
            "deadline_exceeded": deadline_hit,
            "elapsed_ms": round(elapsed_ms, 3),
        },
        "sanitization_failures": sanitize_failures,
        "infeasible_assignments": infeasible_assignments,
        "rings": [{"size": r["size"], "atoms": r["atoms"],
                   "planar_rmsd": round(r["planar_rmsd"], 4),
                   "length_spread": round(r["length_spread"], 4),
                   "aromatic_geometry_prior": r["aromatic_geometry"]}
                  for r in geom.rings],
    }
    return {"status": status, "algorithm": ALGORITHM, "candidates": candidates,
            "search_complete": not truncated, "reason_codes": reason_codes,
            "evidence": evidence}


__all__ = ["infer_geometry", "ALGORITHM"]
