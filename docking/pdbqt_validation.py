"""Independent validation and atomic serialization for ligand PDBQT."""

import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


_AUTODOCK4_TYPES = frozenset(
    {
        "C", "A", "N", "NA", "NS", "OA", "OS", "S", "SA", "HD", "HS", "H",
        "F", "Cl", "CL", "Br", "BR", "I", "P", "MG", "Mg",
        # Common uppercase metal/hetero spellings used by AD4 typer rules.
        "MN", "Mn", "ZN", "Zn", "FE", "Fe", "CA", "Ca", "CU", "Cu",
        "NI", "Ni", "CO", "Co", "SI", "Si", "B", "AL", "Al",
    }
)


def _atom_charge_and_type(line: str, tokens: list[str]) -> tuple[str, str]:
    """Extract the charge and AutoDock type from a PDBQT ATOM/HETATM line."""
    if len(tokens) >= 12:
        return tokens[-2], tokens[-1]
    charge = line[66:76].strip()
    atype = line[77:79].strip() if len(line) >= 78 else ""
    return charge, atype


def _connectivity_adjacency(connectivity) -> dict[int, set[int]] | None:
    if connectivity is None:
        return None
    adjacency: dict[int, set[int]] = defaultdict(set)
    if isinstance(connectivity, Mapping):
        pairs = (
            (left, right)
            for left, neighbors in connectivity.items()
            for right in neighbors
        )
    else:
        pairs = connectivity
    for left, right in pairs:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    return adjacency


def _connected_component(
    serials: set[int],
    adjacency: dict[int, set[int]],
) -> set[int]:
    if not serials:
        return set()
    visited: set[int] = set()
    stack = [min(serials)]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adjacency.get(current, ()):
            if neighbor in serials and neighbor not in visited:
                stack.append(neighbor)
    return visited


def validate_pdbqt_torsion_tree(pdbqt_string, *, connectivity=None):
    """Fail closed on malformed ROOT/BRANCH structure and TORSDOF drift.

    Rejects non-numeric/non-finite atom charges, missing or invalid AutoDock
    atom types, self-referential or duplicated BRANCH records, branches that
    re-enter known atoms (cycles), and branches attaching to unknown atoms
    (disconnected trees).  When ``connectivity`` is supplied (a mapping of
    atom serial -> neighbor serials, or an iterable of serial pairs), every
    BRANCH must correspond to a real bond and the full torsion graph must be
    connected. Chemical rings remain valid because torsion-tree acyclicity is
    enforced from BRANCH parent/child declarations, not from all molecular
    bonds. Returns an audit dict with ``atom_count``,
    ``branch_count`` and ``torsdof``.
    """
    atom_serials = set()
    root_serials = set()
    branch_stack = []
    branch_pairs = []
    branch_first_atoms = {}
    atom_groups = {}
    inside_root = False
    root_seen = False
    root_closed = False
    torsdof_values = []

    for line_number, line in enumerate(pdbqt_string.splitlines(), start=1):
        tag = line.split(maxsplit=1)[0] if line.strip() else ""
        if tag == "ROOT":
            if root_seen or branch_stack:
                raise RuntimeError(f"invalid ROOT at PDBQT line {line_number}")
            root_seen = True
            inside_root = True
        elif tag == "ENDROOT":
            if not inside_root or root_closed:
                raise RuntimeError(f"invalid ENDROOT at PDBQT line {line_number}")
            inside_root = False
            root_closed = True
        elif tag == "BRANCH":
            if inside_root or not root_closed:
                raise RuntimeError(
                    f"invalid BRANCH placement at PDBQT line {line_number}"
                )
            parts = line.split()
            if len(parts) != 3:
                raise RuntimeError(f"malformed BRANCH at PDBQT line {line_number}")
            try:
                begin, end = int(parts[1]), int(parts[2])
            except ValueError:
                raise RuntimeError(
                    f"malformed BRANCH at PDBQT line {line_number}"
                ) from None
            if begin == end:
                raise RuntimeError(
                    f"self-referential BRANCH at PDBQT line {line_number}"
                )
            pair = (begin, end)
            if pair in branch_pairs:
                raise RuntimeError(
                    f"duplicate BRANCH at PDBQT line {line_number}"
                )
            if begin not in atom_serials:
                raise RuntimeError(
                    f"BRANCH attaches to unknown atom serial {begin} "
                    f"at PDBQT line {line_number}"
                )
            if end in atom_serials:
                raise RuntimeError(
                    f"BRANCH re-enters known atom serial {end} "
                    f"at PDBQT line {line_number}"
                )
            branch_pairs.append(pair)
            branch_first_atoms[pair] = None
            branch_stack.append(pair)
        elif tag == "ENDBRANCH":
            parts = line.split()
            try:
                pair = (
                    (int(parts[1]), int(parts[2]))
                    if len(parts) == 3
                    else None
                )
            except ValueError:
                pair = None
            if not branch_stack or pair != branch_stack[-1]:
                raise RuntimeError(
                    f"unpaired ENDBRANCH at PDBQT line {line_number}"
                )
            if branch_first_atoms[pair] is None:
                raise RuntimeError(
                    f"BRANCH contains no child atom at PDBQT line {line_number}"
                )
            branch_stack.pop()
        elif tag in {"ATOM", "HETATM"}:
            if not inside_root and not branch_stack:
                raise RuntimeError(
                    f"atom outside ROOT/BRANCH tree at PDBQT line {line_number}"
                )
            tokens = line.split()
            try:
                serial = int(line[6:11])
            except (IndexError, ValueError):
                raise RuntimeError(
                    f"malformed atom serial at PDBQT line {line_number}"
                ) from None
            if serial in atom_serials:
                raise RuntimeError(f"duplicate PDBQT atom serial {serial}")
            if branch_stack and branch_first_atoms[branch_stack[-1]] is None:
                expected = branch_stack[-1][1]
                if serial != expected:
                    raise RuntimeError(
                        f"BRANCH child atom {serial} does not match declared "
                        f"serial {expected} at PDBQT line {line_number}"
                    )
                branch_first_atoms[branch_stack[-1]] = serial
            try:
                coordinates = tuple(float(line[start:end]) for start, end in (
                    (30, 38), (38, 46), (46, 54)
                ))
            except (IndexError, ValueError):
                raise RuntimeError(
                    f"malformed atom coordinates at PDBQT line {line_number}"
                ) from None
            if not all(math.isfinite(value) for value in coordinates):
                raise RuntimeError(
                    f"non-finite atom coordinates at PDBQT line {line_number}"
                )
            charge, atype = _atom_charge_and_type(line, tokens)
            if not charge:
                raise RuntimeError(
                    f"missing charge at PDBQT line {line_number}"
                )
            try:
                charge_value = float(charge)
            except ValueError:
                raise RuntimeError(
                    f"non-numeric charge {charge!r} at PDBQT line {line_number}"
                ) from None
            if not math.isfinite(charge_value):
                raise RuntimeError(
                    f"non-finite charge {charge!r} at PDBQT line {line_number}"
                )
            if not atype:
                raise RuntimeError(
                    f"missing AutoDock atom type at PDBQT line {line_number}"
                )
            if atype not in _AUTODOCK4_TYPES:
                raise RuntimeError(
                    f"invalid AutoDock atom type {atype!r} "
                    f"at PDBQT line {line_number}"
                )
            atom_serials.add(serial)
            atom_groups[serial] = branch_stack[-1] if branch_stack else ("ROOT",)
            if inside_root:
                root_serials.add(serial)
        elif tag == "TORSDOF":
            parts = line.split()
            if len(parts) != 2 or branch_stack or inside_root:
                raise RuntimeError(f"malformed TORSDOF at PDBQT line {line_number}")
            try:
                torsdof_values.append(int(parts[1]))
            except ValueError:
                raise RuntimeError(
                    f"malformed TORSDOF at PDBQT line {line_number}"
                ) from None

    if not root_seen or not root_closed or inside_root or branch_stack:
        raise RuntimeError("incomplete PDBQT torsion tree")
    if len(torsdof_values) != 1:
        raise RuntimeError("PDBQT must contain exactly one TORSDOF record")
    if torsdof_values[0] != len(branch_pairs):
        raise RuntimeError(
            f"TORSDOF/BRANCH mismatch: {torsdof_values[0]} != {len(branch_pairs)}"
        )
    for begin, end in branch_pairs:
        if begin not in atom_serials or end not in atom_serials:
            raise RuntimeError(
                f"BRANCH references missing atom serials: {begin}, {end}"
            )

    adjacency = _connectivity_adjacency(connectivity)
    connectivity_edges = set()
    if adjacency is not None:
        for begin, end in branch_pairs:
            if end not in adjacency.get(begin, ()):
                raise RuntimeError(
                    f"BRANCH {begin} {end} does not correspond to a real bond"
                )
        if _connected_component(root_serials, adjacency) != root_serials:
            raise RuntimeError("PDBQT ROOT atoms are disconnected")
        if _connected_component(atom_serials, adjacency) != atom_serials:
            raise RuntimeError("PDBQT torsion tree is disconnected")
        branch_edges = {frozenset(pair) for pair in branch_pairs}
        for left, neighbors in adjacency.items():
            if left not in atom_serials:
                continue
            for right in neighbors:
                if right not in atom_serials or left == right:
                    continue
                edge = frozenset((left, right))
                connectivity_edges.add(edge)
                if (
                    atom_groups.get(left) != atom_groups.get(right)
                    and edge not in branch_edges
                ):
                    raise RuntimeError(
                        "molecular bond is not represented by a rigid group or "
                        f"BRANCH: {min(edge)}, {max(edge)}"
                    )

    try:
        from meeko import PDBQTMolecule

        PDBQTMolecule(pdbqt_string, skip_typing=True)
    except Exception as exc:
        raise RuntimeError(f"Meeko rejected emitted PDBQT: {exc}") from exc
    audit = {
        "atom_count": len(atom_serials),
        "branch_count": len(branch_pairs),
        "torsdof": torsdof_values[0],
    }
    if adjacency is not None:
        audit.update({
            "connectivity_edge_count": len(connectivity_edges),
            "connectivity_preserved": True,
        })
    return audit


def atomic_write_text(path, text):
    """Write UTF-8 text atomically with the legacy newline contract."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise
