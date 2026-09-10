"""Compatibility MOL2 reader: stock RDKit semantics or UNITY charge restore.

``rdkit_native`` reproduces the original generic ``Chem.MolFromMol2File``
parse (with ``removeHs=False``; the historical benchmark reads used
``removeHs=True``).  ``rdkit_charge_aware`` re-reads the same file with
``sanitize=False, removeHs=False``, restores formal charges declared in
``@<TRIPOS>UNITY_ATOM_ATTR`` on an internal copy, and then sanitizes
unconditionally.  The MOL2 partial-charge column is never applied as a
formal charge and a charge-aware failure never silently falls back to the
native reader.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from rdkit import Chem

from .mol2_format import mol2_unity_formal_charges

COMPATIBILITY_MODES = ("rdkit_native", "rdkit_charge_aware")
_COORDINATE_TOLERANCE = 1e-4


def _section_lines(content: str, name: str) -> list[list[str]]:
    marker = f"@<TRIPOS>{name}"
    lines = content.splitlines()
    start = lines.index(marker) + 1 if marker in lines else None
    if start is None:
        return []
    rows = []
    for line in lines[start:]:
        if line.startswith("@<TRIPOS>"):
            break
        if line.strip():
            rows.append(line.split())
    return rows


def _atom_rows(content: str) -> list[dict[str, Any]]:
    """Return ATOM rows in file order with strictly parsed identity fields."""
    if content.count("@<TRIPOS>MOLECULE") != 1:
        raise ValueError(
            "MOL2 compatibility reader requires exactly one molecule per file"
        )
    raw = _section_lines(content, "ATOM")
    if not raw:
        raise ValueError("MOL2 file has no @<TRIPOS>ATOM section")
    if content.count("@<TRIPOS>ATOM") > 1:
        raise ValueError("multiple @<TRIPOS>ATOM sections in one molecule")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for fields in raw:
        if len(fields) < 6:
            raise ValueError(f"malformed MOL2 ATOM row: {' '.join(fields)}")
        try:
            atom_id = int(fields[0])
            position = tuple(float(value) for value in fields[2:5])
        except ValueError as exc:
            raise ValueError(
                f"invalid MOL2 atom identifier or coordinate: {fields[0]}"
            ) from exc
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"non-finite MOL2 coordinate at atom {fields[0]}")
        if atom_id <= 0:
            raise ValueError(f"non-positive MOL2 atom identifier: {atom_id}")
        if atom_id in seen:
            raise ValueError(f"duplicate MOL2 atom identifier: {atom_id}")
        seen.add(atom_id)
        rows.append(
            {
                "atom_id": atom_id,
                "name": fields[1],
                "position": position,
                "sybyl": fields[5],
            }
        )
    return rows


def _validate_bond_references(
    content: str, valid_ids: set[int]
) -> None:
    for fields in _section_lines(content, "BOND"):
        if len(fields) < 4:
            raise ValueError("malformed MOL2 BOND row")
        try:
            left, right = int(fields[1]), int(fields[2])
        except ValueError as exc:
            raise ValueError("invalid MOL2 bond atom identifier") from exc
        if left not in valid_ids or right not in valid_ids:
            raise ValueError(
                "MOL2 bond references an atom identifier absent from the "
                "ATOM table"
            )


def _unity_charges(content: str, valid_ids: set[int]) -> dict[int, int]:
    """Strictly map UNITY-declared formal charges by real ATOM identifier."""
    lines = content.splitlines()
    if lines.count("@<TRIPOS>UNITY_ATOM_ATTR") > 1:
        raise ValueError("multiple @<TRIPOS>UNITY_ATOM_ATTR sections")
    try:
        index = lines.index("@<TRIPOS>UNITY_ATOM_ATTR") + 1
    except ValueError:
        return {}
    charges: dict[int, int] = {}
    while index < len(lines) and not lines[index].startswith("@<TRIPOS>"):
        header = lines[index].split()
        index += 1
        if not header:
            continue
        if len(header) < 2:
            raise ValueError("malformed MOL2 UNITY_ATOM_ATTR header")
        try:
            atom_id, attribute_count = int(header[0]), int(header[1])
        except ValueError as exc:
            raise ValueError(
                "invalid MOL2 UNITY_ATOM_ATTR header"
            ) from exc
        if attribute_count < 0:
            raise ValueError(
                "negative MOL2 UNITY_ATOM_ATTR attribute count: "
                f"{attribute_count}"
            )
        if atom_id not in valid_ids:
            raise ValueError(
                "UNITY attribute references atom identifier absent from the "
                f"ATOM table: {atom_id}"
            )
        for _ in range(attribute_count):
            if index >= len(lines) or lines[index].startswith("@<TRIPOS>"):
                raise ValueError("truncated MOL2 UNITY_ATOM_ATTR record")
            fields = lines[index].split()
            index += 1
            if fields and fields[0] == "charge":
                if len(fields) != 2:
                    raise ValueError(
                        "UNITY charge attribute must be exactly "
                        "'charge <integer>'"
                    )
                try:
                    charge = int(fields[1])
                except ValueError as exc:
                    raise ValueError(
                        "UNITY formal charge is not an integer: "
                        f"{fields[1]}"
                    ) from exc
                if atom_id in charges and charges[atom_id] != charge:
                    raise ValueError(
                        "conflicting UNITY formal charges for atom "
                        f"{atom_id}: {charges[atom_id]} != {charge}"
                    )
                charges[atom_id] = charge
    return charges


def _renumbered_content(content: str, rows: list[dict[str, Any]]) -> str:
    """Return an internal copy whose ATOM identifiers are 1..N in file order.

    BOND endpoints and UNITY headers are remapped through the same
    identifier table so the stock ``mol2_unity_formal_charges`` index
    convention (identifier - 1) matches the RDKit parse order.
    """
    id_to_index = {row["atom_id"]: i + 1 for i, row in enumerate(rows)}
    output: list[str] = []
    section = ""
    unity_attributes_remaining = 0
    for line in content.splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line[len("@<TRIPOS>"):]
            unity_attributes_remaining = 0
            output.append(line)
            continue
        fields = line.split()
        if section == "ATOM" and fields:
            fields[0] = str(id_to_index[int(fields[0])])
            output.append(" ".join(fields))
        elif section == "BOND" and fields:
            fields[1] = str(id_to_index[int(fields[1])])
            fields[2] = str(id_to_index[int(fields[2])])
            output.append(" ".join(fields))
        elif section == "UNITY_ATOM_ATTR" and fields:
            if unity_attributes_remaining > 0:
                unity_attributes_remaining -= 1
                output.append(line)
            else:
                fields[0] = str(id_to_index[int(fields[0])])
                unity_attributes_remaining = int(fields[1])
                output.append(" ".join(fields))
        else:
            output.append(line)
    return "\n".join(output) + "\n"


def _verify_against_atom_rows(
    molecule: Chem.Mol, rows: list[dict[str, Any]], warnings: list[str]
) -> None:
    if molecule.GetNumAtoms() != len(rows):
        raise ValueError(
            "MOL2 atom table differs from the parsed molecule: "
            f"{len(rows)} rows vs {molecule.GetNumAtoms()} atoms"
        )
    if molecule.GetNumConformers() != 1:
        raise ValueError(
            "MOL2 compatibility read requires exactly one conformer"
        )
    conformer = molecule.GetConformer()
    periodic = Chem.GetPeriodicTable()
    for index, row in enumerate(rows):
        symbol = str(row["sybyl"]).split(".", 1)[0]
        try:
            atomic_number = periodic.GetAtomicNumber(symbol)
        except Exception:
            warnings.append(
                f"unrecognized SYBYL type {row['sybyl']} for atom "
                f"{row['atom_id']}; element not verified"
            )
        else:
            if molecule.GetAtomWithIdx(index).GetAtomicNum() != atomic_number:
                raise ValueError(
                    "parsed element differs from the ATOM table at atom "
                    f"{row['atom_id']}"
                )
        position = conformer.GetAtomPosition(index)
        observed = (position.x, position.y, position.z)
        expected = row["position"]
        if max(
            abs(observed[axis] - expected[axis]) for axis in range(3)
        ) > _COORDINATE_TOLERANCE:
            raise ValueError(
                "parsed coordinates differ from the ATOM table at atom "
                f"{row['atom_id']}"
            )


def _identity_fields(
    molecule: Chem.Mol, warnings: list[str]
) -> tuple[str | None, str | None]:
    try:
        smiles = Chem.MolToSmiles(molecule)
    except Exception as exc:
        warnings.append(f"canonical SMILES unavailable: {exc}")
        smiles = None
    try:
        inchikey = Chem.MolToInchiKey(molecule)
    except Exception as exc:
        warnings.append(f"full InChIKey unavailable: {exc}")
        inchikey = None
    if not inchikey:
        inchikey = None
    return smiles, inchikey


def _verify_receipt(
    source: Path,
    receipt_path: str | Path,
    reader_inchikey: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    from ..docking.mol2_input import load_validated_mol2

    validated = load_validated_mol2(source, receipt_path=receipt_path)
    if reader_inchikey is not None:
        if reader_inchikey != validated.full_inchikey:
            raise ValueError(
                "reader result differs from the validated receipt identity: "
                f"{reader_inchikey} != {validated.full_inchikey}"
            )
    else:
        warnings.append(
            "reader InChIKey unavailable; receipt identity was not "
            "cross-checked against this reader's molecule"
        )
    return {
        "status": "verified",
        "receipt_path": str(validated.receipt_path),
        "receipt_sha256": validated.receipt_sha256,
        "source_binding": "mol2_sha256",
        "full_inchikey": validated.full_inchikey,
    }


def load_mol2(
    mol2_path: str | Path,
    *,
    compatibility: str = "rdkit_native",
    receipt_path: str | Path | None = None,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Read one MOL2 file and return the molecule plus a JSON-ready report.

    Raises ``ValueError`` (including ``Mol2ValidationError``) for malformed
    input, unknown compatibility modes, or a receipt that does not verify.
    """
    if compatibility not in COMPATIBILITY_MODES:
        raise ValueError(
            f"unknown MOL2 compatibility mode: {compatibility!r}; expected "
            f"one of {COMPATIBILITY_MODES}"
        )
    source = Path(mol2_path)
    if not source.is_file():
        raise ValueError(f"MOL2 file is missing or empty: {source}")
    raw = source.read_bytes()
    if not raw:
        raise ValueError(f"MOL2 file is missing or empty: {source}")
    content = raw.decode("utf-8")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    warnings: list[str] = []

    rows = _atom_rows(content)
    atom_ids = [row["atom_id"] for row in rows]
    contiguous = atom_ids == list(range(1, len(rows) + 1))
    unity_by_id = _unity_charges(content, set(atom_ids))

    if compatibility == "rdkit_native":
        molecule = Chem.MolFromMol2Block(
            content, sanitize=True, removeHs=False
        )
        if molecule is None:
            raise ValueError(
                "RDKit could not read the MOL2 file under rdkit_native "
                "compatibility"
            )
        if molecule.GetNumAtoms() != len(rows):
            warnings.append(
                "rdkit_native parsed atom count differs from the ATOM table: "
                f"{molecule.GetNumAtoms()} vs {len(rows)}"
            )
        if unity_by_id:
            warnings.append(
                "file declares UNITY formal charges for "
                f"{len(unity_by_id)} atom(s); rdkit_native does not restore "
                "them"
            )
        applied: list[dict[str, int]] = []
        coordinates_verified = None
        notes = [
            "rdkit_native mirrors Chem.MolFromMol2File(removeHs=False); "
            "historical benchmark reads used removeHs=True",
            "UNITY formal charges and the MOL2 partial-charge column are "
            "not applied in this mode",
        ]
    else:
        _validate_bond_references(content, set(atom_ids))
        normalized = content if contiguous else _renumbered_content(
            content, rows
        )
        molecule = Chem.MolFromMol2Block(
            normalized, sanitize=False, removeHs=False
        )
        if molecule is None:
            raise ValueError(
                "RDKit could not read the MOL2 file under "
                "rdkit_charge_aware compatibility"
            )
        _verify_against_atom_rows(molecule, rows, warnings)
        applied_by_index = mol2_unity_formal_charges(normalized)
        for index, charge in sorted(applied_by_index.items()):
            molecule.GetAtomWithIdx(index).SetFormalCharge(int(charge))
        applied = [
            {
                "atom_id": atom_ids[index],
                "charge": int(charge),
            }
            for index, charge in sorted(applied_by_index.items())
        ]
        try:
            Chem.SanitizeMol(molecule)
        except Exception as exc:
            raise ValueError(
                f"charge-aware MOL2 sanitization failed: {exc}"
            ) from exc
        coordinates_verified = True
        notes = [
            "UNITY formal charges restored from @<TRIPOS>UNITY_ATOM_ATTR; "
            "the MOL2 partial-charge column is never applied as formal "
            "charge",
            "sanitization is mandatory; no silent fallback to rdkit_native",
        ]

    smiles, inchikey = _identity_fields(molecule, warnings)
    receipt: dict[str, Any] = {"status": "not_requested"}
    if receipt_path is not None:
        receipt = _verify_receipt(source, receipt_path, inchikey, warnings)
    report = {
        "reader_mode": compatibility,
        "mol2_path": str(source),
        "mol2_sha256": source_sha256,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "total_formal_charge": int(Chem.GetFormalCharge(molecule)),
        "applied_formal_charges": applied,
        "canonical_smiles": smiles,
        "full_inchikey": inchikey,
        "sanitized": True,
        "coordinates_verified": coordinates_verified,
        "atom_ids_contiguous": contiguous,
        "unity_formal_charge_atoms": len(unity_by_id),
        "receipt_verification": receipt,
        "warnings": warnings,
        "notes": notes,
        "claim_boundary": (
            "this report describes one reader's parse of the file; it is "
            "not an independent chemistry truth"
        ),
    }
    return molecule, report


__all__ = ["COMPATIBILITY_MODES", "load_mol2"]
