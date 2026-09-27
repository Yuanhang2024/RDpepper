"""Docking-box center calculations independent of Vina execution."""

import math
from pathlib import Path
from typing import List, Optional, Tuple

def get_protein_center(
    pdb_path: str, *, chain_id: Optional[str] = None
) -> Tuple[float, float, float]:
    """Get the heavy-ATOM center of the first protein model."""
    selected_chain = None if chain_id is None else str(chain_id).strip()
    if selected_chain is not None and len(selected_chain) > 1:
        raise ValueError("legacy PDB chain ID must contain at most one character")
    try:
        lines = Path(pdb_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read {pdb_path}: {exc}") from exc
    coordinates = []
    seen = set()
    saw_model = False
    active_model = False
    first_model_seen = False
    for line in lines:
        if line.startswith("MODEL"):
            saw_model = True
            active_model = not first_model_seen
            first_model_seen = True
            continue
        if line.startswith("ENDMDL"):
            active_model = False
            continue
        if saw_model and not active_model:
            continue
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        if selected_chain is not None and line[21].strip() != selected_chain:
            continue
        if line[16].strip() not in {"", "A"}:
            continue
        atom_name = line[12:16].strip()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = atom_name.lstrip("0123456789")[:1].upper()
        if element in {"H", "D"}:
            continue
        key = (line[21], line[22:27], atom_name)
        if key in seen:
            continue
        try:
            xyz = tuple(float(line[start:end]) for start, end in (
                (30, 38), (38, 46), (46, 54)
            ))
        except ValueError:
            raise ValueError("malformed protein atom coordinates") from None
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("non-finite protein atom coordinates")
        seen.add(key)
        coordinates.append(xyz)
    if not coordinates:
        raise ValueError("No protein ATOM coordinates found")
    return tuple(
        sum(row[axis] for row in coordinates) / len(coordinates)
        for axis in range(3)
    )


def get_binding_site_center(
    pdb_path: str,
    residue_ids: List[int],
    *,
    chain_id: Optional[str] = None,
    include_hydrogens: bool = False,
) -> Tuple[float, float, float]:
    """Get the geometric center of explicitly selected binding-site residues."""
    requested = set()
    for value in residue_ids:
        if isinstance(value, bool):
            raise ValueError("binding-site residue IDs must be integers")
        try:
            requested.add(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("binding-site residue IDs must be integers") from exc
    if not requested:
        raise ValueError("at least one binding-site residue ID is required")
    selected_chain = None if chain_id is None else str(chain_id).strip()
    if selected_chain is not None and len(selected_chain) > 1:
        raise ValueError("legacy PDB chain ID must contain at most one character")

    atoms = {}
    observed_residue_keys = set()
    residue_keys_by_number = {residue_id: set() for residue_id in requested}
    saw_model = False
    active_model = False
    first_model_seen = False
    path = Path(pdb_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read {pdb_path}: {exc}") from exc
    for line in lines:
        if line.startswith("MODEL"):
            saw_model = True
            active_model = not first_model_seen
            first_model_seen = True
            continue
        if line.startswith("ENDMDL"):
            active_model = False
            continue
        if saw_model and not active_model:
            continue
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        atom_chain = line[21].strip()
        if selected_chain is not None and atom_chain != selected_chain:
            continue
        try:
            residue_id = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        if residue_id not in requested:
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip()
        if not element:
            element = atom_name.lstrip("0123456789")[:1].upper()
        if not include_hydrogens and element in {"H", "D"}:
            continue
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError("non-finite binding-site atom coordinates")
        insertion_code = line[26].strip()
        residue_key = (atom_chain, residue_id, insertion_code)
        observed_residue_keys.add(residue_key)
        residue_keys_by_number.setdefault(residue_id, set()).add(residue_key)
        key = (atom_chain, residue_id, insertion_code, atom_name)
        if key not in atoms or altloc == "":
            atoms[key] = (x, y, z, altloc)
    if not atoms:
        scope = (
            f" in chain {selected_chain!r}" if selected_chain is not None else ""
        )
        raise ValueError(
            f"No atoms found for binding-site residues {sorted(requested)}{scope}"
        )
    ambiguous = {
        residue_id: sorted(keys)
        for residue_id, keys in residue_keys_by_number.items()
        if len(keys) > 1
    }
    if ambiguous:
        details = "; ".join(
            f"{residue_id}: {keys}"
            for residue_id, keys in sorted(ambiguous.items())
        )
        scope = (
            f" in chain {selected_chain!r}"
            if selected_chain is not None
            else " across all chains"
        )
        raise ValueError(
            "ambiguous binding-site residue identity"
            f"{scope}; each residue number must resolve to one "
            "chain+resnum+insertion-code key; "
            f"matches: {details}"
        )
    observed_residues = {residue_id for _, residue_id, _ in observed_residue_keys}
    missing = requested - observed_residues
    if missing:
        raise ValueError(
            f"No atoms found for requested binding-site residues {sorted(missing)}"
        )
    coordinates = list(atoms.values())
    count = len(coordinates)
    return tuple(
        sum(row[axis] for row in coordinates) / count for axis in range(3)
    )
