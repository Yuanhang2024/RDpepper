"""Path H: CONECT/record-driven connectivity + bond orders.

Path F builds the heavy-atom graph purely from interatomic distances, which can
over-bond at close contacts (e.g. a valence-5 carbon when two atoms happen to
sit within covalent range). When a PDB ships explicit CONECT records, those are
the authoritative connectivity and should be trusted instead of guessed.

Path H therefore:
  1. Reads heavy-atom coordinates AND the CONECT records of one chain.
  2. Builds the molecular graph from CONECT (intra-chain heavy-atom pairs),
     falling back to distance-based connectivity only for atoms left with no
     CONECT partner.
  3. Assigns bond orders heuristically (carbonyl C=O by geometry), same as
     Path F, then validates by molecular formula against the chain's heavy
     atoms.

For structures with complete CONECT records (most crystallographic PDBs and the
CPBind/Scaffold sets), Path H is the reliable geometric route; Path F remains
the fallback when records are absent.
"""
from collections import Counter, defaultdict, deque

from .path_f import (
    _read_chain_heavy_atoms, _build_mol, _assign_bond_orders,
    _formula_counts, _SKIP_HET,
)
from ..core.pdb_utils import pdb_atom_element, read_first_model_lines


def _read_chain_serials_and_conect(pdb_path, chain_id):
    """Return ``(serial->atom index, explicit pair->bond order)``.

    Only first MODEL; only heavy peptide atoms of the requested chain. The
    atom index aligns with the order produced by ``_read_chain_heavy_atoms``.
    Repeated neighbors within one CONECT record encode bond multiplicity;
    repeated identical records do not increase it.
    """
    serial_to_idx = {}
    directed_multiplicity = defaultdict(int)
    idx = 0
    for line in read_first_model_lines(str(pdb_path)):
        if line.startswith('CONECT'):
            fields = [
                line[index:index + 5].strip()
                for index in range(6, len(line.rstrip()), 5)
            ]
            if not fields or not fields[0]:
                raise ValueError(f"malformed CONECT record: {line.rstrip()!r}")
            try:
                source = int(fields[0])
                neighbors = [int(value) for value in fields[1:] if value]
            except ValueError as exc:
                raise ValueError(
                    f"malformed CONECT record: {line.rstrip()!r}"
                ) from exc
            counts = Counter(neighbors)
            for neighbor, multiplicity in counts.items():
                if neighbor == source:
                    raise ValueError(f"self CONECT edge {source}-{source}")
                directed_multiplicity[(source, neighbor)] = max(
                    directed_multiplicity[(source, neighbor)], multiplicity
                )
            continue
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
        if elem.capitalize() == 'H':
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            continue
        if serial in serial_to_idx:
            raise ValueError(
                f"duplicate atom serial {serial} in selected first model"
            )
        serial_to_idx[serial] = idx
        idx += 1
    bond_orders = {}
    undirected = {
        tuple(sorted(pair)) for pair in directed_multiplicity
    }
    for left, right in sorted(undirected):
        forward = directed_multiplicity.get((left, right), 0)
        reverse = directed_multiplicity.get((right, left), 0)
        if forward and reverse and forward != reverse:
            raise ValueError(
                "conflicting CONECT multiplicity for edge "
                f"{left}-{right}: {forward} vs {reverse}"
            )
        multiplicity = max(forward, reverse)
        if multiplicity not in (1, 2, 3):
            raise ValueError(
                f"unsupported CONECT multiplicity {multiplicity} for edge "
                f"{left}-{right}"
            )
        bond_orders[(left, right)] = multiplicity
    return serial_to_idx, bond_orders


def _shortest_path_length(adjacency, start, target, *, limit=None):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        if limit is not None and distance >= limit:
            continue
        for neighbor in adjacency.get(node, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def generate_h(pdb_path, chain_id='L'):
    """Path H: CONECT-record-driven graph + heuristic bond orders.

    Returns ``(smiles, error)``; error is None on success. Requires CONECT
    records covering the chain; if none apply, returns an error so callers fall
    back to Path F.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D
    except ImportError as ex:
        return None, f"rdkit unavailable: {ex}"

    atoms = _read_chain_heavy_atoms(pdb_path, chain_id)
    if not atoms:
        return None, "no peptide heavy atoms in chain"
    truth = dict(Counter(e for e, *_ in atoms))

    try:
        serial_to_idx, conect_bonds = _read_chain_serials_and_conect(
            pdb_path, chain_id
        )
    except (OSError, TypeError, ValueError) as exc:
        return None, f"invalid explicit-connectivity input: {exc}"
    intra = {
        (serial_to_idx[a], serial_to_idx[b]): order
        for (a, b), order in conect_bonds.items()
        if a in serial_to_idx and b in serial_to_idx
    }
    if not intra:
        return None, "no intra-chain CONECT records (use Path F)"

    # Build mol; add CONECT bonds, then distance-complete any unbonded atoms.
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, (elem, x, y, z) in enumerate(atoms):
        rw.AddAtom(Chem.Atom(elem))
        conf.SetAtomPosition(i, Point3D(x, y, z))
    seen = set()
    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    explicit_orders = {}
    for (i, j), order in intra.items():
        key = (min(i, j), max(i, j))
        if i != j and key not in seen:
            rw.AddBond(i, j, bond_types[order])
            seen.add(key)
            explicit_orders[key] = order
    mol = rw.GetMol()
    mol.AddConformer(conf)

    # Compare against geometry even when every atom has an explicit neighbor.
    # Missing component links and plausible macrocycle closures are added, but
    # short-cycle contact edges are not allowed to over-complete a partial graph.
    try:
        geometry_rw = Chem.RWMol()
        for elem, *_coordinates in atoms:
            geometry_rw.AddAtom(Chem.Atom(elem))
        geometry = geometry_rw.GetMol()
        geometry.AddConformer(Chem.Conformer(conf), assignId=True)
        rdDetermineBonds.DetermineConnectivity(geometry)
        adjacency = defaultdict(set)
        for left, right in seen:
            adjacency[left].add(right)
            adjacency[right].add(left)
        for bond in geometry.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            path_length = _shortest_path_length(
                adjacency, i, j, limit=6
            )
            should_add = (
                not adjacency.get(i)
                or not adjacency.get(j)
                or path_length is None
                or path_length >= 7
            )
            if not should_add:
                continue
            rw.AddBond(i, j, Chem.BondType.SINGLE)
            seen.add(key)
            adjacency[i].add(j)
            adjacency[j].add(i)
        mol = rw.GetMol()
        mol.AddConformer(Chem.Conformer(conf), assignId=True)
    except Exception as ex:
        return None, f"connectivity completion failed: {ex}"

    # Bond-order heuristics may inspect every edge, but they are not allowed to
    # rewrite an order supplied by an explicit CONECT record.  In particular,
    # the heuristic C=O upgrade must not turn an explicitly single bond into a
    # double bond.  Re-apply and verify all explicit orders after the heuristic
    # pass; a missing edge is a hard failure rather than a silently repaired
    # graph.
    mol = _assign_bond_orders(mol)
    for (left, right), order in explicit_orders.items():
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond is None:
            return None, (
                "explicit CONECT edge was lost during bond-order assignment: "
                f"{left}-{right}"
            )
        bond.SetBondType(bond_types[order])
    try:
        Chem.SanitizeMol(mol)
    except Exception as ex:
        return None, f"sanitize failed: {ex}"

    fragment_count = len(Chem.GetMolFrags(mol))
    if fragment_count != 1:
        return None, f"multiple disconnected fragments: {fragment_count}"

    got = _formula_counts(mol)
    if got != truth:
        return None, f"formula mismatch: got {got} vs structure {truth}"
    return Chem.MolToSmiles(mol), None
