"""Validated MOL2 loading for the V5 ligand-preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from rdkit import Chem

from ..core.mol2_format import mol2_unity_formal_charges


RECEIPT_SCHEMA_VERSION = "1.0.0-cycpep-mol2-validation.1"
RECEIPT_SUFFIX = ".validation.json"
VALID_COORDINATE_MODES = frozenset({
    "source_bound",
    "template_completed",
    "regenerated",
})


class Mol2ValidationError(ValueError):
    """A MOL2 file cannot satisfy its declared validation receipt."""


def validation_error_status(error: Exception | str) -> str:
    message = str(error)
    unavailable_markers = (
        "MOL2 file is missing or empty",
        "validated MOL2 receipt is required",
    )
    return (
        "not_supported"
        if message.startswith(unavailable_markers)
        else "rejected"
    )


@dataclass(frozen=True)
class ValidatedMol2:
    path: Path
    sha256: str
    receipt_path: Path
    receipt_sha256: str
    molecule: Chem.Mol
    full_inchikey: str
    coordinate_mode: str
    coordinate_level: str
    rigor: str
    quality: str | None
    formal_charge: int
    atom_count: int
    heavy_atom_count: int
    source_heavy_atom_mapping_complete: bool
    atom_provenance_complete: bool
    receipt: Mapping[str, Any]


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def default_receipt_path(mol2_path: str | Path) -> Path:
    return Path(str(Path(mol2_path)) + RECEIPT_SUFFIX)


def _section_lines(content: str, name: str) -> list[str]:
    marker = f"@<TRIPOS>{name}"
    lines = content.splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError:
        return []
    result = []
    for line in lines[start:]:
        if line.startswith("@<TRIPOS>"):
            break
        if line.strip():
            result.append(line)
    return result


def _restore_tripos_metadata(
    molecule: Chem.Mol,
    content: str,
) -> None:
    atom_lines = _section_lines(content, "ATOM")
    if len(atom_lines) != molecule.GetNumAtoms():
        raise Mol2ValidationError(
            "MOL2 atom table differs from the parsed molecule"
        )
    substructures = {}
    for line in _section_lines(content, "SUBSTRUCTURE"):
        fields = line.split()
        if len(fields) < 2:
            raise Mol2ValidationError("malformed MOL2 SUBSTRUCTURE row")
        try:
            substructure_id = int(fields[0])
        except ValueError as exc:
            raise Mol2ValidationError(
                "invalid MOL2 substructure identifier"
            ) from exc
        chain_id = (
            ""
            if len(fields) < 6 or fields[5] == "****"
            else fields[5]
        )
        substructures[substructure_id] = {
            "name": fields[1],
            "chain_id": chain_id,
        }
    for expected_index, line in enumerate(atom_lines, 1):
        fields = line.split()
        if len(fields) < 8:
            raise Mol2ValidationError("malformed MOL2 ATOM row")
        try:
            atom_id = int(fields[0])
            substructure_id = int(fields[6])
        except ValueError as exc:
            raise Mol2ValidationError(
                "invalid MOL2 atom or substructure identifier"
            ) from exc
        if atom_id != expected_index:
            raise Mol2ValidationError(
                "MOL2 atom identifiers are not contiguous and ordered"
            )
        atom = molecule.GetAtomWithIdx(expected_index - 1)
        metadata = substructures.get(substructure_id, {})
        residue_name = str(fields[7])
        residue_number = substructure_id
        atom.SetProp("_TriposAtomName", fields[1])
        atom.SetProp("_TriposAtomType", fields[5])
        atom.SetProp("_TriposResidueName", residue_name)
        atom.SetProp(
            "_TriposChainId", str(metadata.get("chain_id") or "")
        )
        atom.SetIntProp("_TriposResidueNumber", residue_number)
        atom.SetProp("_TriposSubstructureName", str(
            metadata.get("name") or residue_name
        ))


def read_mol2_molecule(path: str | Path) -> tuple[Chem.Mol, str]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise Mol2ValidationError(f"MOL2 file is missing or empty: {source}")
    content = source.read_text(encoding="utf-8")
    molecule = Chem.MolFromMol2Block(
        content,
        sanitize=False,
        removeHs=False,
    )
    if molecule is None:
        raise Mol2ValidationError("RDKit could not read the MOL2 file")
    for atom_index, charge in mol2_unity_formal_charges(
        content
    ).items():
        if atom_index < 0 or atom_index >= molecule.GetNumAtoms():
            raise Mol2ValidationError(
                "MOL2 formal-charge atom index is out of range"
            )
        molecule.GetAtomWithIdx(atom_index).SetFormalCharge(int(charge))
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise Mol2ValidationError(
            f"MOL2 graph sanitization failed: {exc}"
        ) from exc
    if molecule.GetNumConformers() != 1:
        raise Mol2ValidationError(
            "validated MOL2 requires exactly one 3D conformer"
        )
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise Mol2ValidationError("MOL2 conformer is not marked as 3D")
    for atom_index in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_index)
        if not all(math.isfinite(value) for value in (
            position.x,
            position.y,
            position.z,
        )):
            raise Mol2ValidationError(
                f"non-finite MOL2 coordinate at atom {atom_index + 1}"
            )
    _restore_tripos_metadata(molecule, content)
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey:
        raise Mol2ValidationError(
            "MOL2 did not produce a complete InChIKey"
        )
    return molecule, inchikey


def build_validation_receipt(
    mol2_path: str | Path,
    *,
    coordinate_mode: str,
    rigor: str,
    quality: str | None,
    source_heavy_atom_mapping_complete: bool,
    atom_provenance_complete: bool,
    source_input_sha256: str | None = None,
    topology_class: str | None = None,
    macrocycle_ring_size: int | None = None,
    max_source_coordinate_delta_angstrom: float | None = None,
    evidence_manifest_sha256: str | None = None,
    expected_full_inchikey: str | None = None,
    expected_connectivity_inchikey: str | None = None,
    coordinate_level: str | None = None,
    mapped_heavy_atom_indices: list[int] | None = None,
    generated_heavy_atom_indices: list[int] | None = None,
    atom_coordinate_origins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = Path(mol2_path).resolve()
    molecule, inchikey = read_mol2_molecule(source)
    identity_expectation: str | None = None
    if (
        expected_full_inchikey is not None
        and inchikey != str(expected_full_inchikey)
    ):
        raise Mol2ValidationError(
            "MOL2 full InChIKey differs from the expected parent identity: "
            f"{inchikey} != {expected_full_inchikey}"
        )
    if expected_full_inchikey is not None:
        identity_expectation = "full_inchikey"
    if expected_connectivity_inchikey is not None:
        expected_block = str(expected_connectivity_inchikey).split("-")[0]
        if inchikey.split("-")[0] != expected_block:
            raise Mol2ValidationError(
                "MOL2 connectivity block differs from the expected parent "
                f"identity: {inchikey.split('-')[0]} != {expected_block}"
            )
        identity_expectation = "connectivity_block"
    mode = str(coordinate_mode).strip()
    if mode not in VALID_COORDINATE_MODES:
        raise Mol2ValidationError(
            f"unsupported MOL2 coordinate mode: {coordinate_mode!r}"
        )
    if not str(rigor).strip():
        raise Mol2ValidationError("MOL2 validation requires a rigor label")
    if not atom_provenance_complete:
        raise Mol2ValidationError(
            "MOL2 atom provenance must be complete"
        )
    if evidence_manifest_sha256 is not None and (
        len(str(evidence_manifest_sha256)) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(evidence_manifest_sha256).lower()
        )
    ):
        raise Mol2ValidationError(
            "evidence_manifest_sha256 must be a SHA-256"
        )
    if mode == "source_bound" and not source_heavy_atom_mapping_complete:
        raise Mol2ValidationError(
            "source-bound MOL2 requires complete source heavy-atom mapping"
        )
    level = coordinate_level or {
        "source_bound": "X3",
        "template_completed": "X2",
        "regenerated": "X1",
    }[mode]
    if level not in {"X1", "X2", "X3"}:
        raise Mol2ValidationError(f"invalid coordinate evidence level: {level}")
    if mode == "source_bound" and level != "X3":
        raise Mol2ValidationError("source-bound mode requires X3")
    heavy_indices = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    }
    mapped_indices = sorted({
        int(value) for value in (mapped_heavy_atom_indices or [])
    })
    generated_indices = sorted({
        int(value) for value in (generated_heavy_atom_indices or [])
    })
    if level == "X3" and not mapped_indices:
        mapped_indices = sorted(heavy_indices)
    if level == "X1" and not generated_indices:
        generated_indices = sorted(heavy_indices)
    mapped_set = set(mapped_indices)
    generated_set = set(generated_indices)
    if mapped_set & generated_set:
        raise Mol2ValidationError("mapped and generated atom indices overlap")
    if not mapped_set <= heavy_indices or not generated_set <= heavy_indices:
        raise Mol2ValidationError(
            "coordinate provenance references a non-heavy or unknown atom"
        )
    if mapped_set | generated_set != heavy_indices:
        raise Mol2ValidationError(
            "coordinate provenance does not cover every heavy atom"
        )
    if level == "X2" and (not mapped_indices or not generated_indices):
        raise Mol2ValidationError(
            "X2 requires both mapped and generated atom indices"
        )
    if level == "X1" and (
        mapped_indices or generated_set != heavy_indices
    ):
        raise Mol2ValidationError(
            "X1 requires every heavy atom to be generated"
        )
    if level == "X3" and (
        generated_indices or mapped_set != heavy_indices
    ):
        raise Mol2ValidationError(
            "X3 requires every heavy atom to be source mapped"
        )
    origins = {
        str(key): str(value)
        for key, value in (atom_coordinate_origins or {}).items()
    }
    if not origins:
        origins = {
            **{str(index): "source" for index in mapped_indices},
            **{str(index): "generated" for index in generated_indices},
        }
    expected_origins = {
        **{str(index): "source" for index in mapped_indices},
        **{str(index): "generated" for index in generated_indices},
    }
    if origins != expected_origins:
        raise Mol2ValidationError(
            "atom coordinate origins differ from the mapped/generated ledger"
        )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "validated",
        "mol2_path": source.name,
        "mol2_locator": source.name,
        "mol2_sha256": sha256_path(source),
        "full_inchikey": inchikey,
        "identity_expectation": identity_expectation,
        "coordinate_mode": mode,
        "coordinate_level": level,
        "mapped_heavy_atom_indices": mapped_indices,
        "generated_heavy_atom_indices": generated_indices,
        "generated_heavy_atom_count": len(generated_indices),
        "atom_coordinate_origins": origins,
        "rigor": str(rigor),
        "quality": quality,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "source_heavy_atom_mapping_complete": bool(
            source_heavy_atom_mapping_complete
        ),
        "atom_provenance_complete": True,
        "source_input_sha256": source_input_sha256,
        "topology_class": topology_class,
        "macrocycle_ring_size": macrocycle_ring_size,
        "max_source_coordinate_delta_angstrom": (
            max_source_coordinate_delta_angstrom
        ),
        "evidence_manifest_sha256": (
            str(evidence_manifest_sha256).lower()
            if evidence_manifest_sha256 is not None
            else None
        ),
    }


def write_validation_receipt(
    mol2_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    **metadata: Any,
) -> Path:
    output = (
        Path(receipt_path)
        if receipt_path is not None
        else default_receipt_path(mol2_path)
    )
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite MOL2 validation receipt: {output}"
        )
    receipt = build_validation_receipt(mol2_path, **metadata)
    _atomic_write_json(output, receipt)
    return output


def load_validated_mol2(
    mol2_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> ValidatedMol2:
    source = Path(mol2_path).resolve()
    receipt_source = (
        Path(receipt_path).resolve()
        if receipt_path is not None
        else default_receipt_path(source).resolve()
    )
    if not receipt_source.is_file():
        raise Mol2ValidationError(
            "validated MOL2 receipt is required; no SMILES/PDB bypass is "
            f"allowed: {receipt_source}"
        )
    try:
        receipt = json.loads(
            receipt_source.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise Mol2ValidationError(
            f"cannot read MOL2 validation receipt: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise Mol2ValidationError(
            "MOL2 validation receipt must be a JSON object"
        )
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise Mol2ValidationError(
            "unsupported MOL2 validation receipt schema"
        )
    if receipt.get("status") != "validated":
        raise Mol2ValidationError("MOL2 receipt is not validated")
    observed_hash = sha256_path(source)
    if receipt.get("mol2_sha256") != observed_hash:
        raise Mol2ValidationError("MOL2 SHA-256 differs from its receipt")
    molecule, inchikey = read_mol2_molecule(source)
    checks = {
        "full_inchikey": inchikey,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
    }
    for field, observed in checks.items():
        if receipt.get(field) != observed:
            raise Mol2ValidationError(
                f"MOL2 receipt field differs from file: {field}"
            )
    coordinate_mode = str(receipt.get("coordinate_mode") or "")
    if coordinate_mode not in VALID_COORDINATE_MODES:
        raise Mol2ValidationError(
            "MOL2 receipt has an invalid coordinate mode"
        )
    if receipt.get("atom_provenance_complete") is not True:
        raise Mol2ValidationError(
            "MOL2 receipt lacks complete atom provenance"
        )
    coordinate_level = str(receipt.get("coordinate_level") or {
        "source_bound": "X3",
        "template_completed": "X2",
        "regenerated": "X1",
    }[coordinate_mode])
    expected_level = {
        "source_bound": "X3",
        "template_completed": "X2",
        "regenerated": "X1",
    }[coordinate_mode]
    if coordinate_level != expected_level:
        raise Mol2ValidationError(
            "MOL2 receipt coordinate mode and level disagree"
        )
    try:
        generated_indices = sorted({
            int(value)
            for value in (
                receipt.get("generated_heavy_atom_indices") or []
            )
        })
        mapped_indices = sorted({
            int(value)
            for value in (
                receipt.get("mapped_heavy_atom_indices") or []
            )
        })
    except (TypeError, ValueError) as exc:
        raise Mol2ValidationError(
            "MOL2 receipt contains invalid atom provenance indices"
        ) from exc
    heavy_indices = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    }
    mapped_set = set(mapped_indices)
    generated_set = set(generated_indices)
    if mapped_set & generated_set:
        raise Mol2ValidationError(
            "MOL2 receipt mapped/generated atom indices overlap"
        )
    if not mapped_set <= heavy_indices or not generated_set <= heavy_indices:
        raise Mol2ValidationError(
            "MOL2 receipt references a non-heavy or unknown atom"
        )
    if mapped_set | generated_set != heavy_indices:
        raise Mol2ValidationError(
            "MOL2 receipt provenance does not cover every heavy atom"
        )
    if coordinate_level == "X3" and (
        generated_indices or mapped_set != heavy_indices
    ):
        raise Mol2ValidationError("X3 receipt has an invalid source ledger")
    if coordinate_level == "X2" and (
        not mapped_indices or not generated_indices
    ):
        raise Mol2ValidationError(
            "X2 receipt lacks mapped/generated atom ledger"
        )
    if coordinate_level == "X1" and (
        mapped_indices or generated_set != heavy_indices
    ):
        raise Mol2ValidationError("X1 receipt has an invalid generated ledger")
    if receipt.get("generated_heavy_atom_count") != len(generated_indices):
        raise Mol2ValidationError(
            "MOL2 receipt generated atom count differs from its ledger"
        )
    origins = receipt.get("atom_coordinate_origins") or {}
    expected_origins = {
        **{str(index): "source" for index in mapped_indices},
        **{str(index): "generated" for index in generated_indices},
    }
    if origins != expected_origins:
        raise Mol2ValidationError(
            "MOL2 receipt atom origins differ from its ledger"
        )
    if (
        coordinate_mode == "source_bound"
        and receipt.get("source_heavy_atom_mapping_complete") is not True
    ):
        raise Mol2ValidationError(
            "source-bound MOL2 receipt lacks complete source mapping"
        )
    rigor = str(receipt.get("rigor") or "")
    if not rigor:
        raise Mol2ValidationError("MOL2 receipt lacks inherited rigor")
    return ValidatedMol2(
        path=source,
        sha256=observed_hash,
        receipt_path=receipt_source,
        receipt_sha256=sha256_path(receipt_source),
        molecule=molecule,
        full_inchikey=inchikey,
        coordinate_mode=coordinate_mode,
        coordinate_level=coordinate_level,
        rigor=rigor,
        quality=receipt.get("quality"),
        formal_charge=int(Chem.GetFormalCharge(molecule)),
        atom_count=molecule.GetNumAtoms(),
        heavy_atom_count=molecule.GetNumHeavyAtoms(),
        source_heavy_atom_mapping_complete=bool(
            receipt.get("source_heavy_atom_mapping_complete")
        ),
        atom_provenance_complete=True,
        receipt=receipt,
    )
