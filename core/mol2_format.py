"""Format-only MOL2 helpers with no conformer-generation dependencies."""

from __future__ import annotations


def mol2_unity_formal_charges(content: str) -> dict[int, int]:
    """Return zero-based atom indices and formal charges from a MOL2 block."""
    lines = content.splitlines()
    try:
        index = lines.index("@<TRIPOS>UNITY_ATOM_ATTR") + 1
    except ValueError:
        return {}

    charges = {}
    while index < len(lines) and not lines[index].startswith("@<TRIPOS>"):
        header = lines[index].split()
        index += 1
        if not header:
            continue
        if len(header) < 2:
            raise ValueError("malformed MOL2 UNITY_ATOM_ATTR header")
        atom_id, attribute_count = int(header[0]), int(header[1])
        for _ in range(attribute_count):
            if index >= len(lines):
                raise ValueError("truncated MOL2 UNITY_ATOM_ATTR record")
            fields = lines[index].split()
            index += 1
            if len(fields) >= 2 and fields[0] == "charge":
                charges[atom_id - 1] = int(fields[1])
    return charges


__all__ = ["mol2_unity_formal_charges"]

