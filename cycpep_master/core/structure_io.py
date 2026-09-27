"""Auditable coordinate-input normalization for cyclic-peptide reconstruction."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import math
import os
import stat
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import gemmi

from .mmcif_chem_comp import (
    EmbeddedChemCompError,
    extract_embedded_chem_comp_templates,
)


_SKIP_RESIDUES = {
    "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "SO4", "PO4"
}


class CoordinateInputError(ValueError):
    def __init__(self, code: str, message: str, *, not_supported: bool = False):
        super().__init__(message)
        self.code = code
        self.not_supported = not_supported


@dataclass(frozen=True)
class PreparedCoordinateInput:
    pdb_path: Path
    chain_id: str
    source_format: str
    audit: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_coordinate_source(path: Path) -> None:
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_NOT_FOUND", f"coordinate input does not exist: {path}"
        ) from exc
    except OSError as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_READ_FAILED",
            f"coordinate input cannot be accessed: {type(exc).__name__}: {exc}",
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CoordinateInputError(
            "COORDINATE_INPUT_NOT_FILE", f"coordinate input is not a file: {path}"
        )


def _source_sha256(path: Path) -> str:
    try:
        return sha256(path)
    except OSError as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_READ_FAILED",
            f"coordinate input cannot be read: {type(exc).__name__}: {exc}",
        ) from exc


def _read_mmcif_structure(path: Path) -> gemmi.Structure:
    try:
        structure = gemmi.read_structure(str(path))
        if any(
            entity.entity_type == gemmi.EntityType.Polymer
            and bool(entity.full_sequence)
            for entity in structure.entities
        ):
            structure.setup_entities()
        return structure
    except OSError as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_READ_FAILED",
            f"mmCIF input cannot be read: {type(exc).__name__}: {exc}",
        ) from exc
    except (IndexError, RuntimeError, ValueError) as exc:
        raise CoordinateInputError(
            "MALFORMED_MMCIF_INPUT",
            f"mmCIF input cannot be parsed: {type(exc).__name__}: {exc}",
        ) from exc


def _mmcif_connection_model_scopes(
    path: Path,
) -> dict[str, tuple[int, int] | None]:
    """Return explicit endpoint model numbers keyed by ``_struct_conn.id``."""
    try:
        document = gemmi.cif.read_file(str(path))
        if len(document) == 0:
            raise ValueError("mmCIF document has no data blocks")
        table = document[0].find_mmcif_category("_struct_conn.")
    except (IndexError, OSError, RuntimeError, ValueError) as exc:
        raise CoordinateInputError(
            "MALFORMED_MMCIF_INPUT",
            f"mmCIF connection scope cannot be parsed: {type(exc).__name__}: {exc}",
        ) from exc
    if len(table) == 0:
        return {}
    tag_indices = {tag.lower(): index for index, tag in enumerate(table.tags)}
    id_index = tag_indices.get("_struct_conn.id")
    model_1_index = tag_indices.get("_struct_conn.pdbx_ptnr1_pdb_model_num")
    model_2_index = tag_indices.get("_struct_conn.pdbx_ptnr2_pdb_model_num")
    if id_index is None:
        raise CoordinateInputError(
            "MALFORMED_MMCIF_INPUT", "_struct_conn rows do not declare an id"
        )
    scopes: dict[str, tuple[int, int] | None] = {}
    for row in table:
        connection_id = str(row[id_index])
        if connection_id in scopes:
            raise CoordinateInputError(
                "MALFORMED_MMCIF_INPUT",
                f"duplicate _struct_conn.id: {connection_id}",
            )
        if model_1_index is None or model_2_index is None:
            scopes[connection_id] = None
            continue
        raw_models = (str(row[model_1_index]), str(row[model_2_index]))
        if any(value in {"", ".", "?"} for value in raw_models):
            scopes[connection_id] = None
            continue
        try:
            scopes[connection_id] = (int(raw_models[0]), int(raw_models[1]))
        except ValueError as exc:
            raise CoordinateInputError(
                "MALFORMED_MMCIF_INPUT",
                f"invalid _struct_conn model scope for {connection_id}: {raw_models}",
            ) from exc
    return scopes


def _is_pdb_end_record(line: bytes | str) -> bool:
    marker = b"END" if isinstance(line, bytes) else "END"
    return line[:6].strip() == marker


def _read_pdb_structure(path: Path) -> tuple[gemmi.Structure, int]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
        explicit_models = any(line.startswith(b"MODEL") for line in lines)
        current_model: int | None = None
        model_count = 0
        ignored_later_model_connections = 0
        scoped_lines = []
        for line in lines:
            if _is_pdb_end_record(line):
                scoped_lines.append(line)
                break
            if line.startswith(b"MODEL"):
                model_count += 1
                current_model = model_count
            elif line.startswith(b"ENDMDL"):
                current_model = None
            if (
                explicit_models
                and current_model not in {None, 1}
                and line.startswith((b"LINK  ", b"SSBOND"))
            ):
                ignored_later_model_connections += 1
                continue
            scoped_lines.append(line)
        structure = gemmi.read_pdb_string(b"".join(scoped_lines))
        if any(
            entity.entity_type == gemmi.EntityType.Polymer
            and bool(entity.full_sequence)
            for entity in structure.entities
        ):
            structure.setup_entities()
        return structure, ignored_later_model_connections
    except OSError as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_READ_FAILED",
            f"PDB input cannot be read: {type(exc).__name__}: {exc}",
        ) from exc
    except (IndexError, RuntimeError, ValueError) as exc:
        raise CoordinateInputError(
            "MALFORMED_PDB_INPUT",
            f"PDB input cannot be parsed: {type(exc).__name__}: {exc}",
        ) from exc


def _pdb_layout_audit(path: Path, chain_id: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_READ_FAILED",
            f"PDB input cannot be audited: {type(exc).__name__}: {exc}",
        ) from exc

    explicit_models = any(line.startswith("MODEL") for line in lines)
    in_first_model = not explicit_models
    first_model_seen = False
    selected_atom_count = 0
    selected_segment_count = 0
    selected_segment_open = False
    chain_ids: set[str] = set()
    for line in lines:
        if _is_pdb_end_record(line):
            break
        if line.startswith("MODEL"):
            if first_model_seen:
                in_first_model = False
            else:
                first_model_seen = True
                in_first_model = True
            continue
        if line.startswith("ENDMDL"):
            if in_first_model:
                break
            continue
        if not in_first_model:
            continue
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            atom_chain = line[21:22]
            chain_ids.add(atom_chain)
            if atom_chain == chain_id:
                selected_atom_count += 1
                if not selected_segment_open:
                    selected_segment_count += 1
                    selected_segment_open = True
            continue
        if line.startswith("TER") and len(line) > 21 and line[21:22] == chain_id:
            selected_segment_open = False
    return {
        "source_first_model_chain_ids": sorted(chain_ids),
        "source_selected_chain_atom_count": selected_atom_count,
        "source_selected_chain_segment_count": selected_segment_count,
    }


def _pdb_source_polymer_evidence(
    path: Path, chain_id: str
) -> dict[str, Any]:
    """Extract raw-record evidence needed to classify PDB HETATM residues.

    Gemmi may assign an HETATM block to a polymer entity solely because it is
    adjacent to SEQRES coordinates.  The raw record type and MODRES declaration
    are retained separately so source identity cannot rely on that inference.
    """
    record_types_by_key: dict[tuple[int, str], set[str]] = {}
    modres_by_key: dict[tuple[int, str], dict[str, str]] = {}
    try:
        lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_READ_FAILED",
            f"PDB input cannot be audited: {type(exc).__name__}: {exc}",
        ) from exc
    explicit_models = any(line.startswith("MODEL") for line in lines)
    in_first_model = not explicit_models
    first_model_seen = False
    for line in lines:
        if _is_pdb_end_record(line):
            break
        if line.startswith("MODEL"):
            if first_model_seen:
                in_first_model = False
            else:
                first_model_seen = True
                in_first_model = True
            continue
        if line.startswith("ENDMDL"):
            if in_first_model:
                break
            continue
        if line.startswith("MODRES") and len(line) >= 27:
            if line[16:17] != chain_id:
                continue
            try:
                key = (int(line[18:22]), line[22:23].strip())
            except ValueError:
                continue
            modres_by_key[key] = {
                "modified_name": line[12:15].strip().upper(),
                "standard_name": line[24:27].strip().upper(),
            }
            continue
        if not in_first_model:
            continue
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            if line[21:22] != chain_id:
                continue
            try:
                key = (int(line[22:26]), line[26:27].strip())
            except ValueError:
                continue
            record_types_by_key.setdefault(key, set()).add(line[:6].strip())
            continue
    return {
        "record_types_by_key": record_types_by_key,
        "modres_by_key": modres_by_key,
    }


def _mmcif_source_polymer_evidence(
    path: Path, chain_id: str
) -> dict[str, Any]:
    """Extract explicit mmCIF label-entity membership for HETATM rows."""
    try:
        document = gemmi.cif.read_file(str(path))
        block = document.sole_block()
    except (IndexError, OSError, RuntimeError, ValueError) as exc:
        raise CoordinateInputError(
            "MALFORMED_MMCIF_INPUT",
            f"mmCIF polymer evidence cannot be parsed: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    polymer_entity_ids: set[str] = set()
    entity_table = block.find_mmcif_category("_entity.")
    if len(entity_table):
        tags = list(entity_table.tags)
        id_tag = "_entity.id" if "_entity.id" in tags else None
        type_tag = "_entity.type" if "_entity.type" in tags else None
        if id_tag is not None and type_tag is not None:
            polymer_entity_ids.update(
                str(row[id_tag]).strip()
                for row in entity_table
                if str(row[type_tag]).strip().lower() == "polymer"
            )
    entity_poly_table = block.find_mmcif_category("_entity_poly.")
    if len(entity_poly_table):
        tags = list(entity_poly_table.tags)
        id_tag = "_entity_poly.entity_id" if "_entity_poly.entity_id" in tags else None
        if id_tag is not None:
            polymer_entity_ids.update(
                str(row[id_tag]).strip()
                for row in entity_poly_table
            )
    polymer_sequence_positions: dict[str, set[int]] = {}
    entity_poly_seq_table = block.find_mmcif_category("_entity_poly_seq.")
    if len(entity_poly_seq_table):
        tags = list(entity_poly_seq_table.tags)
        entity_tag = (
            "_entity_poly_seq.entity_id"
            if "_entity_poly_seq.entity_id" in tags else None
        )
        num_tag = (
            "_entity_poly_seq.num"
            if "_entity_poly_seq.num" in tags else None
        )
        if entity_tag is not None and num_tag is not None:
            for row in entity_poly_seq_table:
                entity_id = str(row[entity_tag]).strip()
                try:
                    sequence_position = int(str(row[num_tag]).strip())
                except (TypeError, ValueError):
                    continue
                if entity_id and sequence_position > 0:
                    polymer_sequence_positions.setdefault(
                        entity_id, set()
                    ).add(sequence_position)
    label_asym_entity: dict[str, str] = {}
    struct_asym_table = block.find_mmcif_category("_struct_asym.")
    if len(struct_asym_table):
        tags = list(struct_asym_table.tags)
        asym_tag = "_struct_asym.id" if "_struct_asym.id" in tags else None
        entity_tag = (
            "_struct_asym.entity_id" if "_struct_asym.entity_id" in tags else None
        )
        if asym_tag is not None and entity_tag is not None:
            label_asym_entity = {
                str(row[asym_tag]).strip(): str(row[entity_tag]).strip()
                for row in struct_asym_table
            }
    record_types_by_key: dict[tuple[int, str], set[str]] = {}
    polymer_membership_by_key: dict[tuple[int, str], bool] = {}
    unresolved_polymer_membership_claim_by_key: dict[tuple[int, str], bool] = {}
    polymer_binding_rows_by_key: dict[
        tuple[int, str], list[tuple[str, int | None, bool, bool]]
    ] = {}
    atom_table = block.find_mmcif_category("_atom_site.")
    if len(atom_table):
        tags = list(atom_table.tags)
        group_tag = "_atom_site.group_PDB" if "_atom_site.group_PDB" in tags else None
        auth_seq_tag = "_atom_site.auth_seq_id" if "_atom_site.auth_seq_id" in tags else None
        auth_asym_tag = "_atom_site.auth_asym_id" if "_atom_site.auth_asym_id" in tags else None
        ins_tag = (
            "_atom_site.pdbx_PDB_ins_code"
            if "_atom_site.pdbx_PDB_ins_code" in tags else None
        )
        label_entity_tag = (
            "_atom_site.label_entity_id"
            if "_atom_site.label_entity_id" in tags else None
        )
        label_asym_tag = (
            "_atom_site.label_asym_id"
            if "_atom_site.label_asym_id" in tags else None
        )
        label_seq_tag = (
            "_atom_site.label_seq_id"
            if "_atom_site.label_seq_id" in tags else None
        )
        if group_tag and auth_seq_tag and auth_asym_tag:
            for row in atom_table:
                try:
                    residue_number = int(str(row[auth_seq_tag]).strip())
                except (TypeError, ValueError):
                    continue
                if str(row[group_tag]).strip().upper() != "HETATM":
                    continue
                if str(row[auth_asym_tag]).strip() != chain_id:
                    continue
                insertion_code = (
                    str(row[ins_tag]).strip()
                    if ins_tag is not None else ""
                )
                if insertion_code in {".", "?"}:
                    insertion_code = ""
                key = (residue_number, insertion_code)
                record_types_by_key.setdefault(key, set()).add("HETATM")
                label_entity = (
                    str(row[label_entity_tag]).strip()
                    if label_entity_tag is not None else ""
                )
                label_asym = (
                    str(row[label_asym_tag]).strip()
                    if label_asym_tag is not None else ""
                )
                resolved_label_entity = (
                    label_entity
                    if label_entity not in {"", ".", "?"}
                    else label_asym_entity.get(label_asym, "")
                )
                try:
                    label_sequence_position = int(
                        str(row[label_seq_tag]).strip()
                    ) if label_seq_tag is not None else None
                except (TypeError, ValueError):
                    label_sequence_position = None
                label_position_is_bound = (
                    label_sequence_position is not None
                    and label_sequence_position > 0
                    and label_sequence_position in polymer_sequence_positions.get(
                        resolved_label_entity, set()
                    )
                )
                explicit = (
                    resolved_label_entity in polymer_entity_ids
                    and label_position_is_bound
                )
                polymer_binding_rows_by_key.setdefault(key, []).append((
                    resolved_label_entity,
                    label_sequence_position,
                    explicit,
                    resolved_label_entity in polymer_entity_ids,
                ))
    for key, binding_rows in polymer_binding_rows_by_key.items():
        binding_signatures = {
            (entity, sequence_position)
            for entity, sequence_position, _explicit, _polymer_claim
            in binding_rows
        }
        all_rows_bound = all(
            explicit for _entity, _position, explicit, _claim in binding_rows
        )
        same_binding = len(binding_signatures) == 1
        explicit = bool(binding_rows) and all_rows_bound and same_binding
        polymer_membership_by_key[key] = explicit
        if (
            any(claim for _entity, _position, _bound, claim in binding_rows)
            and not explicit
        ):
            unresolved_polymer_membership_claim_by_key[key] = True
    return {
        "record_types_by_key": record_types_by_key,
        "polymer_membership_by_key": polymer_membership_by_key,
        "unresolved_polymer_membership_claim_by_key": (
            unresolved_polymer_membership_claim_by_key
        ),
    }


def coordinate_format(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        return "mmcif"
    if name.endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz")):
        return "pdb"
    raise CoordinateInputError(
        "UNSUPPORTED_COORDINATE_FORMAT",
        f"unsupported coordinate filename: {Path(path).name}",
        not_supported=True,
    )


def _clean_altloc(value: str) -> str:
    return "" if value in {"\x00", " ", ""} else value.strip()


def _select_alternate_locations(
    chain: gemmi.Chain,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], str]]:
    groups = removed = removed_heavy = 0
    selected_counts: dict[str, int] = {}
    selected_altlocs: dict[tuple[int, str, str], str] = {}
    for residue in chain:
        for atom_name in sorted({atom.name.strip() for atom in residue}):
            atoms = [atom for atom in residue if atom.name.strip() == atom_name]
            key = (int(residue.seqid.num), residue.seqid.icode.strip(), atom_name)
            if len(atoms) == 1 and not _clean_altloc(atoms[0].altloc):
                selected_altlocs[key] = ""
                continue
            groups += 1
            selected = min(
                atoms,
                key=lambda atom: (
                    -float(atom.occ) if math.isfinite(float(atom.occ)) else 0.0,
                    0 if not _clean_altloc(atom.altloc) else 1,
                    0 if _clean_altloc(atom.altloc) == "A" else 1,
                    _clean_altloc(atom.altloc),
                    int(atom.serial),
                ),
            )
            selected_serial = int(selected.serial)
            selected_name = _clean_altloc(selected.altloc) or "blank"
            selected_altlocs[key] = _clean_altloc(selected.altloc)
            selected_counts[selected_name] = selected_counts.get(selected_name, 0) + 1
            removals = [
                (atom.name, atom.altloc, atom.element.name, atom.element.name != "H")
                for atom in atoms
                if int(atom.serial) != selected_serial
            ]
            for name, altloc, element, is_heavy in removals:
                residue.remove_atom(name, altloc, gemmi.Element(element))
                removed += 1
                removed_heavy += int(is_heavy)
            retained = [atom for atom in residue if int(atom.serial) == selected_serial]
            if len(retained) != 1:
                raise CoordinateInputError(
                    "ALTLOC_SELECTION_FAILED",
                    f"selected alternate location was not retained: serial {selected_serial}",
                )
            retained[0].altloc = "\x00"
    return (
        {
            "alternate_location_candidate_group_count": groups,
            "alternate_location_removed_atom_count": removed,
            "alternate_location_removed_heavy_atom_count": removed_heavy,
            "selected_altloc_counts": dict(sorted(selected_counts.items())),
        },
        selected_altlocs,
    )


def _retain_reconstructable_residues(chain: gemmi.Chain) -> dict[str, int]:
    removed_residues = removed_atoms = removed_heavy = 0
    for index in range(len(chain) - 1, -1, -1):
        residue = chain[index]
        if residue.name.strip().upper() not in _SKIP_RESIDUES:
            continue
        removed_residues += 1
        removed_atoms += len(residue)
        removed_heavy += sum(atom.element.name != "H" for atom in residue)
        del chain[index]
    if not chain:
        raise CoordinateInputError(
            "EMPTY_SELECTED_CHAIN", "selected chain has no reconstructable residues"
        )
    unrepresentable_names = sorted({
        residue.name.strip()
        for residue in chain
        if not residue.name.strip()
        or len(residue.name.strip()) > 3
        or not residue.name.strip().isascii()
        or not residue.name.strip().isprintable()
    })
    if unrepresentable_names:
        raise CoordinateInputError(
            "MMCIF_COMPONENT_ID_NOT_PDB_REPRESENTABLE",
            "component identifiers cannot be represented in the legacy PDB "
            "residue-name field: " + ",".join(unrepresentable_names),
            not_supported=True,
        )
    return {
        "excluded_solvent_or_ion_residue_count": removed_residues,
        "excluded_solvent_or_ion_atom_count": removed_atoms,
        "excluded_solvent_or_ion_heavy_atom_count": removed_heavy,
    }


def _retain_unique_polymer_subchain(chain: gemmi.Chain) -> dict[str, Any]:
    """Disambiguate label entities merged under one mmCIF auth chain.

    Gemmi groups ``_atom_site.auth_asym_id`` values into a Chain, so unrelated
    non-polymer label asym IDs can accompany the requested peptide. Select a
    subchain only when exactly one multi-residue group carries positive polymer
    evidence. Multiple polymer candidates remain an explicit ambiguity.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for index, residue in enumerate(chain):
        if residue.name.strip().upper() in _SKIP_RESIDUES:
            continue
        key = (residue.subchain.strip(), residue.entity_id.strip())
        groups.setdefault(key, []).append(index)
    empty_audit = {
        "source_reconstructable_label_entity_count": len(groups),
        "label_entity_selection_mode": "not_required",
        "selected_label_subchain_id": None,
        "selected_entity_id": None,
        "excluded_nonselected_entity_residue_count": 0,
        "excluded_nonselected_entity_atom_count": 0,
        "excluded_nonselected_entity_heavy_atom_count": 0,
    }
    if len(groups) <= 1:
        return empty_audit

    candidates = []
    for key, indices in groups.items():
        residues = [chain[index] for index in indices]
        if len(residues) < 2:
            continue
        has_polymer_evidence = any(
            residue.entity_type == gemmi.EntityType.Polymer
            or residue.het_flag == "A"
            for residue in residues
        )
        if has_polymer_evidence:
            candidates.append(key)
    if len(candidates) > 1:
        raise CoordinateInputError(
            "MMCIF_AUTH_CHAIN_ENTITY_AMBIGUOUS",
            "requested auth chain contains multiple reconstructable polymer "
            "label entities: " + ",".join(
                f"{subchain or '<blank>'}:{entity or '<blank>'}"
                for subchain, entity in sorted(candidates)
            ),
            not_supported=True,
        )
    if len(candidates) != 1:
        empty_audit["label_entity_selection_mode"] = (
            "no_unique_polymer_evidence_preserved_fail_closed"
        )
        return empty_audit

    selected_key = candidates[0]
    retained_indices = set(groups[selected_key])
    excluded_indices = [
        index for index in range(len(chain)) if index not in retained_indices
    ]
    excluded_residues = len(excluded_indices)
    excluded_atoms = sum(len(chain[index]) for index in excluded_indices)
    excluded_heavy = sum(
        atom.element.name != "H"
        for index in excluded_indices
        for atom in chain[index]
    )
    for index in reversed(excluded_indices):
        del chain[index]
    return {
        "source_reconstructable_label_entity_count": len(groups),
        "label_entity_selection_mode": "unique_polymer_label_entity",
        "selected_label_subchain_id": selected_key[0],
        "selected_entity_id": selected_key[1],
        "excluded_nonselected_entity_residue_count": excluded_residues,
        "excluded_nonselected_entity_atom_count": excluded_atoms,
        "excluded_nonselected_entity_heavy_atom_count": excluded_heavy,
    }


def _reconstructable_chain_indices(
    model: gemmi.Model, chain_id: str
) -> tuple[list[int], list[int]]:
    matching = [
        index for index, chain in enumerate(model) if chain.name == chain_id
    ]
    reconstructable = [
        index
        for index in matching
        if any(
            residue.name.strip().upper() not in _SKIP_RESIDUES
            for residue in model[index]
        )
    ]
    return matching, reconstructable


def _explicit_pdb_polymer_subchains(structure: gemmi.Structure) -> set[str]:
    return {
        subchain
        for entity in structure.entities
        if entity.entity_type == gemmi.EntityType.Polymer
        and bool(entity.full_sequence)
        for subchain in entity.subchains
    }


def _source_sequence_identity_audit(
    structure: gemmi.Structure,
    selected_chain: gemmi.Chain,
    chain_id: str,
    source_sha256: str,
    *,
    source_format: str = "pdb",
    source_layout_audit: dict[str, Any] | None = None,
    source_polymer_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture source-declared polymer identity before sequence metadata removal.

    The normalized projection intentionally drops SEQRES/entity sequences.  This
    audit preserves only the source-bound declaration needed to adjudicate an
    otherwise unknown coordinate residue.  Coordinate rows are ordered by a
    source-stable sequence key before they are paired with the declaration.  A
    row that cannot be assigned a unique source position remains explicitly
    ambiguous or unmapped; it is never silently paired by Gemmi block order.
    """
    source_layout_audit = source_layout_audit or {}
    source_polymer_evidence = source_polymer_evidence or {}
    coordinate_rows = []
    excluded_nonpolymer_count = 0
    for index, residue in enumerate(selected_chain, start=1):
        residue_name = residue.name.strip().upper()
        if residue_name in _SKIP_RESIDUES:
            continue
        # ``_retain_unique_polymer_subchain`` and ``_retain_reconstructable_residues``
        # run before this audit.  Keep only Gemmi's explicit polymer rows while
        # their source entity annotations are still intact; a later normalization
        # step intentionally flattens those annotations for the PDB projection.
        if residue.entity_type != gemmi.EntityType.Polymer:
            excluded_nonpolymer_count += 1
            continue
        label_seq = residue.label_seq
        try:
            label_seq = int(label_seq) if label_seq is not None else None
        except (TypeError, ValueError):
            label_seq = None
        coordinate_rows.append({
            "coordinate_index": index,
            "coordinate_resseq": int(residue.seqid.num),
            "coordinate_icode": residue.seqid.icode.strip(),
            "coordinate_label_seq": label_seq,
            "coordinate_het_flag": residue.het_flag.strip(),
            "coordinate_residue_name": residue_name,
            "subchain": residue.subchain.strip(),
            "entity_id": residue.entity_id.strip(),
        })

    mapping_basis = "pdb_auth_seqid"
    mapping_limitation = None
    mapping_reason = None
    canonical_keys: list[Any] = []
    if source_format == "mmcif":
        labels = [row["coordinate_label_seq"] for row in coordinate_rows]
        if labels and all(label is not None for label in labels):
            if len(set(labels)) == len(labels) and all(int(label) > 0 for label in labels):
                mapping_basis = "mmcif_label_seq_id"
                canonical_keys = [int(label) for label in labels]
            else:
                mapping_reason = "non_unique_or_invalid_label_seq_id"
        elif labels:
            mapping_reason = "partial_label_seq_id"
        else:
            mapping_reason = "label_seq_id_unavailable_from_gemmi"
        if not canonical_keys:
            mapping_basis = "mmcif_auth_seqid_fallback"
            mapping_limitation = (
                "Gemmi did not expose a complete unique _atom_site.label_seq_id; "
                "mapping uses unique auth sequence number/insertion code"
            )
            canonical_keys = [
                (row["coordinate_resseq"], row["coordinate_icode"])
                for row in coordinate_rows
            ]
    else:
        canonical_keys = [
            (row["coordinate_resseq"], row["coordinate_icode"])
            for row in coordinate_rows
        ]

    duplicate_keys = len(set(canonical_keys)) != len(canonical_keys)
    if duplicate_keys:
        mapping_reason = mapping_reason or "duplicate_coordinate_sequence_key"
    if source_format == "pdb" and int(
        source_layout_audit.get("source_selected_chain_segment_count") or 1
    ) > 1:
        mapping_reason = mapping_reason or "multiple_source_chain_segments"
    mapping_valid = not duplicate_keys and not (
        source_format == "pdb"
        and int(source_layout_audit.get("source_selected_chain_segment_count") or 1) > 1
    )
    if mapping_reason == "non_unique_or_invalid_label_seq_id":
        # A duplicate label sequence position cannot be made unique by sorting.
        mapping_valid = False
    if mapping_reason == "partial_label_seq_id":
        # Mixed availability is safe only with the explicit auth-key fallback.
        mapping_valid = not duplicate_keys
    ordered_rows = [
        row for _key, row in sorted(
            zip(canonical_keys, coordinate_rows),
            key=lambda item: item[0],
        )
    ]
    if not mapping_valid:
        # Preserve deterministic presentation for diagnostics, but never use the
        # resulting order as a unique source binding.
        ordered_rows = list(ordered_rows)
    selected_groups = {
        (row["subchain"], row["entity_id"])
        for row in ordered_rows
    }
    selected_subchains = {subchain for subchain, _entity in selected_groups}
    selected_entities = {entity for _subchain, entity in selected_groups}
    entity_rows = []
    for entity_index, entity in enumerate(structure.entities):
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        declared = [str(name).strip().upper() for name in entity.full_sequence]
        if not declared:
            continue
        entity_subchains = {str(value).strip() for value in entity.subchains}
        overlaps = entity_subchains & selected_subchains
        # PDB entities can carry an empty entity_id while the entity name is
        # the auth chain; mmCIF uses label entity IDs on the residues.
        if not overlaps and str(entity.name).strip() not in selected_entities:
            if not (str(entity.name).strip() == chain_id and len(structure.entities) == 1):
                continue
        entity_rows.append({
            "entity_index": entity_index,
            "entity_id": str(entity.name).strip(),
            "subchains": sorted(entity_subchains),
            "declared_names": declared,
        })

    base = {
        "source_sha256": source_sha256,
        "source_chain_id": chain_id,
        "source_metadata_present": bool(entity_rows),
        "status": "absent" if not entity_rows else "unique",
        "entity_count": len(entity_rows),
        "entities": [dict(row) for row in entity_rows],
        "rows": [],
        "mapping_basis": mapping_basis,
        "mapping_limitation": mapping_limitation,
        "mapping_reason": mapping_reason,
        "coordinate_row_count": len(ordered_rows),
        "excluded_nonpolymer_coordinate_residue_count": excluded_nonpolymer_count,
    }
    unresolved_membership_claims = source_polymer_evidence.get(
        "unresolved_polymer_membership_claim_by_key", {}
    )
    unresolved_claim_count = sum(
        bool(claim) for claim in unresolved_membership_claims.values()
    )
    if unresolved_claim_count:
        base["unresolved_polymer_membership_claim_count"] = (
            unresolved_claim_count
        )
    if not entity_rows:
        if unresolved_claim_count:
            base["status"] = "partial"
            base["mapping_reason"] = (
                "hetero_residue_lacks_polymer_membership_evidence"
            )
        return base
    if excluded_nonpolymer_count:
        # A source sequence cannot account for an omitted non-polymer coordinate
        # row.  Keep the declaration for diagnostics, but make unknown-residue
        # adjudication fail closed when that row is encountered downstream.
        base["status"] = "partial"

    if len(entity_rows) != 1:
        rows = []
        for entity_row in entity_rows:
            for position, declared_name in enumerate(
                entity_row["declared_names"], start=1
            ):
                rows.append({
                    "source_sha256": source_sha256,
                    "chain_id": chain_id,
                    "subchain": (
                        entity_row["subchains"][0]
                        if len(entity_row["subchains"]) == 1 else None
                    ),
                    "entity_id": entity_row["entity_id"],
                    "sequence_position": position,
                    "declared_name": declared_name,
                    "coordinate_resseq": None,
                    "coordinate_icode": None,
                    "coordinate_residue_name": None,
                    "mapping_state": "ambiguous",
                    "mapping_status": "ambiguous",
                })
        base["status"] = "ambiguous"
        base["rows"] = rows
        return base

    entity_row = entity_rows[0]
    declared_names = entity_row["declared_names"]

    if source_format in {"pdb", "mmcif"} and any(
        row["coordinate_het_flag"] == "H" for row in ordered_rows
    ):
        record_types_by_key = source_polymer_evidence.get(
            "record_types_by_key", {}
        )
        modres_by_key = source_polymer_evidence.get("modres_by_key", {})
        polymer_membership_by_key = source_polymer_evidence.get(
            "polymer_membership_by_key", {}
        )
        expected_names_by_key = {}
        if mapping_valid:
            for position, coordinate in enumerate(ordered_rows, start=1):
                if position <= len(declared_names):
                    key = (
                        coordinate["coordinate_resseq"],
                        coordinate["coordinate_icode"],
                    )
                    expected_names_by_key[key] = declared_names[position - 1]
        retained_rows = []
        for coordinate in ordered_rows:
            if coordinate["coordinate_het_flag"] != "H":
                retained_rows.append(coordinate)
                continue
            key = (
                coordinate["coordinate_resseq"],
                coordinate["coordinate_icode"],
            )
            modres = modres_by_key.get(key)
            modified_name = (
                str(modres.get("modified_name") or "").strip().upper()
                if isinstance(modres, dict) else ""
            )
            expected_name = expected_names_by_key.get(key)
            modres_parent = (
                str(modres.get("standard_name") or "").strip().upper()
                if isinstance(modres, dict) else ""
            )
            record_types = set(record_types_by_key.get(key, ()))
            modres_matches_source = (
                isinstance(modres, dict)
                and modified_name == coordinate["coordinate_residue_name"]
                and bool(expected_name)
                and modres_parent == expected_name
            )
            modres_conflicts_with_source = (
                isinstance(modres, dict)
                and not modres_matches_source
            )
            explicit_polymer_membership = (
                (
                    source_format == "pdb"
                    and "HETATM" in record_types
                    and modres_matches_source
                )
                or (
                    source_format == "mmcif"
                    and bool(polymer_membership_by_key.get(key))
                )
            )
            sequence_identity_match = (
                mapping_valid
                and expected_name == coordinate["coordinate_residue_name"]
                and not modres_conflicts_with_source
            )
            if explicit_polymer_membership or sequence_identity_match:
                retained_rows.append(coordinate)
                continue
            excluded_nonpolymer_count += 1
        if len(retained_rows) != len(ordered_rows):
            ordered_rows = retained_rows
            mapping_reason = mapping_reason or (
                "hetero_residue_lacks_polymer_membership_evidence"
            )
            base["status"] = "partial"
            base["mapping_reason"] = mapping_reason
            base["coordinate_row_count"] = len(ordered_rows)
            base["excluded_nonpolymer_coordinate_residue_count"] = (
                excluded_nonpolymer_count
            )
            canonical_keys = [
                (row["coordinate_resseq"], row["coordinate_icode"])
                for row in ordered_rows
            ]
            duplicate_keys = len(set(canonical_keys)) != len(canonical_keys)
            mapping_valid = mapping_valid and not duplicate_keys

    rows = []
    if mapping_valid and mapping_basis == "mmcif_label_seq_id":
        by_position = {
            int(row["coordinate_label_seq"]): row for row in ordered_rows
        }
        if len(declared_names) != len(ordered_rows) or any(
            position < 1 or position > len(declared_names)
            for position in by_position
        ):
            base["status"] = "partial"
        for position, declared_name in enumerate(declared_names, start=1):
            coordinate = by_position.get(position)
            if coordinate is None:
                base["status"] = "partial"
                rows.append({
                    "source_sha256": source_sha256,
                    "chain_id": chain_id,
                    "subchain": (
                        entity_row["subchains"][0]
                        if len(entity_row["subchains"]) == 1 else None
                    ),
                    "entity_id": entity_row["entity_id"],
                    "sequence_position": position,
                    "declared_name": declared_name,
                    "coordinate_resseq": None,
                    "coordinate_icode": None,
                    "coordinate_label_seq": None,
                    "coordinate_residue_name": None,
                    "mapping_state": "unmapped",
                    "mapping_status": "unmapped",
                    "mapping_reason": "missing_label_sequence_position",
                    "declared_name_matches_coordinate": False,
                })
                continue
            row = {
                "source_sha256": source_sha256,
                "chain_id": chain_id,
                "subchain": coordinate["subchain"],
                "entity_id": entity_row["entity_id"] or coordinate["entity_id"],
                "sequence_position": position,
                "declared_name": declared_name,
                "coordinate_resseq": coordinate["coordinate_resseq"],
                "coordinate_icode": coordinate["coordinate_icode"],
                "coordinate_label_seq": coordinate["coordinate_label_seq"],
                "coordinate_residue_name": coordinate["coordinate_residue_name"],
                "mapping_state": "unique",
                "mapping_status": "unique",
                "mapping_reason": "label_seq_id_projection",
                "declared_name_matches_coordinate": (
                    declared_name == coordinate["coordinate_residue_name"]
                ),
            }
            rows.append(row)
    elif len(declared_names) == len(ordered_rows) and mapping_valid:
        for position, (declared_name, coordinate) in enumerate(
            zip(declared_names, ordered_rows),
            start=1,
        ):
            row_state = "unique"
            row_status = "unique"
            row_reason = "canonical_coordinate_order_projection"
            if not mapping_valid:
                row_state = "ambiguous"
                row_status = "ambiguous"
                row_reason = mapping_reason or "non_unique_coordinate_order"
                base["status"] = "ambiguous"
            row = {
                "source_sha256": source_sha256,
                "chain_id": chain_id,
                "subchain": coordinate["subchain"],
                "entity_id": entity_row["entity_id"] or coordinate["entity_id"],
                "sequence_position": position,
                "declared_name": declared_name,
                "coordinate_resseq": coordinate["coordinate_resseq"],
                "coordinate_icode": coordinate["coordinate_icode"],
                "coordinate_label_seq": coordinate["coordinate_label_seq"],
                "coordinate_residue_name": coordinate["coordinate_residue_name"],
                "mapping_state": row_state,
                "mapping_status": row_status,
                "mapping_reason": row_reason,
                "declared_name_matches_coordinate": (
                    declared_name == coordinate["coordinate_residue_name"]
                ),
            }
            rows.append(row)
    elif len(declared_names) == len(ordered_rows):
        # Equal counts do not rescue duplicate sequence keys, multiple chain
        # segments, or otherwise non-unique source positions.
        base["status"] = "ambiguous"
        for position, (declared_name, coordinate) in enumerate(
            zip(declared_names, ordered_rows),
            start=1,
        ):
            rows.append({
                "source_sha256": source_sha256,
                "chain_id": chain_id,
                "subchain": coordinate["subchain"],
                "entity_id": entity_row["entity_id"] or coordinate["entity_id"],
                "sequence_position": position,
                "declared_name": declared_name,
                "coordinate_resseq": coordinate["coordinate_resseq"],
                "coordinate_icode": coordinate["coordinate_icode"],
                "coordinate_label_seq": coordinate["coordinate_label_seq"],
                "coordinate_residue_name": coordinate["coordinate_residue_name"],
                "mapping_state": "ambiguous",
                "mapping_status": "ambiguous",
                "mapping_reason": mapping_reason or "non_unique_coordinate_order",
                "declared_name_matches_coordinate": False,
            })
    else:
        # A count mismatch can be caused by missing coordinates or insertion
        # codes. Bind only an exact name occurring once on each side; all
        # aliases and repeated names remain explicit non-bindings.
        declared_name_positions: dict[str, list[int]] = {}
        coordinate_name_positions: dict[str, list[int]] = {}
        for position, name in enumerate(declared_names, start=1):
            declared_name_positions.setdefault(name, []).append(position)
        for index, coordinate in enumerate(ordered_rows, start=1):
            coordinate_name_positions.setdefault(
                coordinate["coordinate_residue_name"], []
            ).append(index)
        coordinate_by_index = {
            index: coordinate for index, coordinate in enumerate(ordered_rows, start=1)
        }
        mapped_coordinate_indices: set[int] = set()
        for position, declared_name in enumerate(declared_names, start=1):
            candidates = [
                index for index in coordinate_name_positions.get(declared_name, [])
                if index not in mapped_coordinate_indices
            ]
            state = "unmapped"
            coordinate = None
            if len(candidates) == 1 and len(declared_name_positions[declared_name]) == 1:
                coordinate_index = candidates[0]
                coordinate = coordinate_by_index[coordinate_index]
                mapped_coordinate_indices.add(coordinate_index)
                state = "unique"
            elif len(candidates) > 1 or len(declared_name_positions[declared_name]) > 1:
                state = "ambiguous"
            rows.append({
                "source_sha256": source_sha256,
                "chain_id": chain_id,
                "subchain": (
                    coordinate["subchain"] if coordinate is not None
                    else (entity_row["subchains"][0] if len(entity_row["subchains"]) == 1 else None)
                ),
                "entity_id": entity_row["entity_id"] or (
                    coordinate["entity_id"] if coordinate is not None else None
                ),
                "sequence_position": position,
                "declared_name": declared_name,
                "coordinate_resseq": (
                    coordinate["coordinate_resseq"] if coordinate is not None else None
                ),
                "coordinate_icode": (
                    coordinate["coordinate_icode"] if coordinate is not None else None
                ),
                "coordinate_label_seq": (
                    coordinate["coordinate_label_seq"] if coordinate is not None else None
                ),
                "coordinate_residue_name": (
                    coordinate["coordinate_residue_name"] if coordinate is not None else None
                ),
                "mapping_state": state,
                "mapping_status": state,
                "mapping_reason": "exact_name_unique" if state == "unique" else (
                    "repeated_name_or_multiple_candidates" if state == "ambiguous"
                    else "no_exact_coordinate_name"
                ),
                "declared_name_matches_coordinate": (
                    coordinate is not None and declared_name == coordinate["coordinate_residue_name"]
                ),
            })
        mapped_declared = {
            int(row["sequence_position"])
            for row in rows if row["mapping_state"] == "unique"
        }
        if len(mapped_declared) != len(declared_names) or not mapping_valid:
            base["status"] = "partial"
    base["rows"] = rows
    return base


def _pdb_chain_object_rows(
    model: gemmi.Model,
    matching: list[int],
    selected: int | None,
    explicit_polymer_subchains: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for index in matching:
        chain = model[index]
        residue_numbers = [int(residue.seqid.num) for residue in chain]
        atom_serials = [int(atom.serial) for residue in chain for atom in residue]
        rows.append({
            "source_chain_object_index": index,
            "residue_count": len(chain),
            "atom_count": sum(len(residue) for residue in chain),
            "heavy_atom_count": _heavy_atom_count(chain),
            "residue_number_range": (
                [min(residue_numbers), max(residue_numbers)]
                if residue_numbers
                else None
            ),
            "atom_serial_range": (
                [min(atom_serials), max(atom_serials)] if atom_serials else None
            ),
            "polymer_residue_count": sum(
                residue.entity_type == gemmi.EntityType.Polymer
                for residue in chain
            ),
            "seqres_polymer_residue_count": sum(
                residue.entity_type == gemmi.EntityType.Polymer
                and residue.subchain in explicit_polymer_subchains
                for residue in chain
            ),
            "selected": index == selected,
        })
    return rows


def _select_unique_pdb_polymer_chain_object(
    model: gemmi.Model,
    chain_id: str,
    matching: list[int],
    explicit_polymer_subchains: set[str],
) -> tuple[int, list[int], dict[str, Any]]:
    """Select one duplicate PDB chain object using explicit SEQRES evidence."""
    explicit_polymer_indices = [
        index
        for index in matching
        if any(
            residue.entity_type == gemmi.EntityType.Polymer
            and residue.subchain in explicit_polymer_subchains
            for residue in model[index]
        )
    ]
    polymer_indices = [
        index
        for index in matching
        if any(
            residue.entity_type == gemmi.EntityType.Polymer
            for residue in model[index]
        )
    ]
    selected = (
        explicit_polymer_indices[0]
        if len(explicit_polymer_indices) == 1
        else None
    )
    selected_has_unbound_polymer = bool(
        selected is not None
        and any(
            residue.entity_type == gemmi.EntityType.Polymer
            and residue.subchain not in explicit_polymer_subchains
            for residue in model[selected]
        )
    )
    if (
        selected is None
        or polymer_indices != [selected]
        or selected_has_unbound_polymer
    ):
        raise CoordinateInputError(
            "PDB_AUTH_CHAIN_POLYMER_OBJECT_NOT_UNIQUE",
            f"model 1 has {len(matching)} chains named {chain_id!r}, but "
            f"{len(explicit_polymer_indices)} have explicit SEQRES polymer "
            f"evidence, {len(polymer_indices)} contain Gemmi polymer residues, "
            f"and selected_unbound_polymer={selected_has_unbound_polymer}",
            not_supported=True,
        )
    excluded = [index for index in matching if index != selected]
    explicit_nonpolymer_indices = [
        index
        for index in excluded
        if any(
            residue.entity_type == gemmi.EntityType.NonPolymer
            for residue in model[index]
        )
    ]
    excluded_nonselected_residue_count = sum(
        len(model[index]) for index in excluded
    )
    excluded_nonselected_atom_count = sum(
        len(residue)
        for index in excluded
        for residue in model[index]
    )
    excluded_nonselected_heavy_atom_count = sum(
        atom.element.name != "H"
        for index in excluded
        for residue in model[index]
        for atom in residue
    )
    excluded_nonpolymer_residue_count = sum(
        len(model[index]) for index in explicit_nonpolymer_indices
    )
    excluded_nonpolymer_atom_count = sum(
        len(residue)
        for index in explicit_nonpolymer_indices
        for residue in model[index]
    )
    excluded_nonpolymer_heavy_atom_count = sum(
        atom.element.name != "H"
        for index in explicit_nonpolymer_indices
        for residue in model[index]
        for atom in residue
    )
    return selected, excluded, {
        "pdb_chain_object_selection_mode": "unique_seqres_polymer_object",
        "source_polymer_chain_object_count": 1,
        "source_gemmi_polymer_chain_object_count": 1,
        "explicit_seqres_polymer_subchains": sorted(explicit_polymer_subchains),
        "selected_source_chain_object_index": selected,
        "pdb_matching_chain_object_rows": _pdb_chain_object_rows(
            model, matching, selected, explicit_polymer_subchains
        ),
        "excluded_nonselected_chain_object_count": len(excluded),
        "excluded_nonselected_chain_residue_count": (
            excluded_nonselected_residue_count
        ),
        "excluded_nonselected_chain_atom_count": excluded_nonselected_atom_count,
        "excluded_nonselected_chain_heavy_atom_count": (
            excluded_nonselected_heavy_atom_count
        ),
        "excluded_nonpolymer_chain_object_count": len(
            explicit_nonpolymer_indices
        ),
        "excluded_nonpolymer_chain_residue_count": (
            excluded_nonpolymer_residue_count
        ),
        "excluded_nonpolymer_chain_atom_count": excluded_nonpolymer_atom_count,
        "excluded_nonpolymer_chain_heavy_atom_count": (
            excluded_nonpolymer_heavy_atom_count
        ),
        "duplicate_auth_chain_boundary_audit_applied": True,
        "cross_object_structured_connection_count": 0,
        "ambiguous_object_structured_connection_count": 0,
        "cross_object_pdb_conect_pair_count": 0,
        "unresolved_object_pdb_conect_pair_count": 0,
    }


def _source_address_chain_object_indices(
    model: gemmi.Model, address: gemmi.AtomAddress
) -> set[int]:
    indices: set[int] = set()
    requested_altloc = _clean_altloc(address.altloc)
    for index, chain in enumerate(model):
        if chain.name != address.chain_name:
            continue
        for residue in chain:
            if (
                int(residue.seqid.num) != int(address.res_id.seqid.num)
                or residue.seqid.icode.strip()
                != address.res_id.seqid.icode.strip()
                or residue.name.strip().upper()
                != address.res_id.name.strip().upper()
            ):
                continue
            if any(
                atom.name.strip() == address.atom_name.strip()
                and (
                    not requested_altloc
                    or _clean_altloc(atom.altloc) == requested_altloc
                )
                for atom in residue
            ):
                indices.add(index)
    return indices


def _validate_pdb_duplicate_chain_boundary(
    model: gemmi.Model,
    selected: int,
    excluded: list[int],
    source_connections: list[gemmi.Connection],
    pdb_conect_pairs: list[tuple[int, int]],
) -> dict[str, int]:
    """Reject explicit bonds crossing a uniquely selected PDB chain object."""
    boundary_indices = {selected, *excluded}
    excluded_indices = set(excluded)
    cross_structured = []
    ambiguous_structured = []
    for connection in source_connections:
        left = (
            _source_address_chain_object_indices(model, connection.partner1)
            & boundary_indices
        )
        right = (
            _source_address_chain_object_indices(model, connection.partner2)
            & boundary_indices
        )
        if len(left) != 1 or len(right) != 1:
            ambiguous_structured.append(connection.name)
            continue
        left_index = next(iter(left))
        right_index = next(iter(right))
        if (
            left_index == selected and right_index in excluded_indices
        ) or (
            right_index == selected and left_index in excluded_indices
        ):
            cross_structured.append(connection.name)

    serial_owners = {
        int(atom.serial): index
        for index, chain in enumerate(model)
        for residue in chain
        for atom in residue
    }
    cross_conect = []
    unresolved_conect = []
    for pair in pdb_conect_pairs:
        left_owner = serial_owners.get(pair[0])
        right_owner = serial_owners.get(pair[1])
        if (
            left_owner == selected and right_owner in excluded_indices
        ) or (
            right_owner == selected and left_owner in excluded_indices
        ):
            cross_conect.append(pair)
        elif (
            left_owner in boundary_indices or right_owner in boundary_indices
        ) and (left_owner is None or right_owner is None):
            unresolved_conect.append(pair)

    if (
        cross_structured
        or ambiguous_structured
        or cross_conect
        or unresolved_conect
    ):
        details = []
        if cross_structured:
            details.append(
                "cross-object LINK/SSBOND=" + ",".join(sorted(cross_structured))
            )
        if ambiguous_structured:
            details.append(
                "ambiguous LINK/SSBOND endpoints="
                + ",".join(sorted(ambiguous_structured))
            )
        if cross_conect:
            details.append(
                "cross-object CONECT="
                + ",".join(f"{left}-{right}" for left, right in cross_conect)
            )
        if unresolved_conect:
            details.append(
                "unresolved CONECT="
                + ",".join(f"{left}-{right}" for left, right in unresolved_conect)
            )
        raise CoordinateInputError(
            "PDB_AUTH_CHAIN_OBJECT_CONNECTION_NOT_SUPPORTED",
            "unique SEQRES polymer chain object cannot be isolated safely: "
            + "; ".join(details),
            not_supported=True,
        )
    return {
        "cross_object_structured_connection_count": 0,
        "ambiguous_object_structured_connection_count": 0,
        "cross_object_pdb_conect_pair_count": 0,
        "unresolved_object_pdb_conect_pair_count": 0,
    }


def _validate_legacy_pdb_representability(chain: gemmi.Chain) -> None:
    long_atom_names = sorted({
        atom.name.strip()
        for residue in chain
        for atom in residue
        if not atom.name.strip()
        or len(atom.name.strip()) > 4
        or not atom.name.strip().isascii()
        or not atom.name.strip().isprintable()
    })
    if long_atom_names:
        raise CoordinateInputError(
            "COORDINATE_ATOM_NAME_NOT_PDB_REPRESENTABLE",
            "atom identifiers cannot be represented in the legacy PDB atom-name "
            "field: " + ",".join(long_atom_names),
            not_supported=True,
        )
    invalid_coordinate_endpoints = sorted({
        (int(residue.seqid.num), residue.seqid.icode.strip(), atom.name.strip())
        for residue in chain
        for atom in residue
        if any(
            not math.isfinite(value) or not -999.999 <= value <= 9999.999
            for value in (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
        )
    })
    if invalid_coordinate_endpoints:
        raise CoordinateInputError(
            "COORDINATE_POSITION_NOT_PDB_REPRESENTABLE",
            "atom coordinates cannot be represented in legacy PDB 8.3 fields: "
            + ",".join(map(str, invalid_coordinate_endpoints)),
            not_supported=True,
        )
    invalid_sequence_ids = sorted({
        int(residue.seqid.num)
        for residue in chain
        if not -999 <= int(residue.seqid.num) <= 9999
    })
    if invalid_sequence_ids:
        raise CoordinateInputError(
            "COORDINATE_RESIDUE_SEQUENCE_NOT_PDB_REPRESENTABLE",
            "residue sequence numbers exceed the legacy PDB field: "
            + ",".join(map(str, invalid_sequence_ids)),
            not_supported=True,
        )


def _heavy_atom_count(chain: gemmi.Chain) -> int:
    return sum(atom.element.name != "H" for residue in chain for atom in residue)


def _normalize_selected_chain_segments(chain: gemmi.Chain) -> None:
    """Serialize the projected peptide chain as one contiguous PDB segment."""
    if not chain:
        return
    subchain = chain[0].subchain
    entity_id = chain[0].entity_id
    entity_type = chain[0].entity_type
    for residue in chain:
        residue.subchain = subchain
        residue.entity_id = entity_id
        residue.entity_type = entity_type


def _address_key(address: gemmi.AtomAddress) -> tuple[int, str, str]:
    return (
        int(address.res_id.seqid.num),
        address.res_id.seqid.icode.strip(),
        address.atom_name.strip(),
    )


def _source_address_key(
    address: gemmi.AtomAddress,
) -> tuple[int, str, str, str]:
    """Include component identity when assigning a source connection."""
    return (
        int(address.res_id.seqid.num),
        address.res_id.seqid.icode.strip(),
        address.res_id.name.strip().upper(),
        address.atom_name.strip(),
    )


def _atom_serials(chain: gemmi.Chain) -> dict[tuple[int, str, str], int]:
    serials: dict[tuple[int, str, str], int] = {}
    for residue in chain:
        for atom in residue:
            key = (int(residue.seqid.num), residue.seqid.icode.strip(), atom.name.strip())
            if key in serials:
                raise CoordinateInputError(
                    "DUPLICATE_NORMALIZED_ATOM_ENDPOINT", f"duplicate endpoint: {key}"
                )
            serials[key] = int(atom.serial)
    return serials


def _serial_addresses(chain: gemmi.Chain) -> dict[int, tuple[int, str, str]]:
    addresses: dict[int, tuple[int, str, str]] = {}
    for residue in chain:
        for atom in residue:
            serial = int(atom.serial)
            if serial in addresses:
                raise CoordinateInputError(
                    "DUPLICATE_SOURCE_ATOM_SERIAL", f"duplicate atom serial: {serial}"
                )
            addresses[serial] = (
                int(residue.seqid.num),
                residue.seqid.icode.strip(),
                atom.name.strip(),
            )
    return addresses


def _validate_model_atom_serial_uniqueness(model: gemmi.Model) -> None:
    """Reject ambiguous model-global serials before interpreting PDB CONECT."""
    seen: set[int] = set()
    duplicates: set[int] = set()
    for chain in model:
        for residue in chain:
            for atom in residue:
                serial = int(atom.serial)
                if serial in seen:
                    duplicates.add(serial)
                seen.add(serial)
    if duplicates:
        raise CoordinateInputError(
            "DUPLICATE_SOURCE_ATOM_SERIAL",
            "PDB CONECT requires model-global unique atom serials; duplicates: "
            + ",".join(map(str, sorted(duplicates))),
        )


def _pdb_conect_pairs(path: Path) -> tuple[list[tuple[int, int]], int]:
    pairs: set[tuple[int, int]] = set()
    all_pairs: set[tuple[int, int]] = set()
    try:
        lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CoordinateInputError(
            "COORDINATE_INPUT_READ_FAILED",
            f"PDB input cannot be audited: {type(exc).__name__}: {exc}",
        ) from exc
    explicit_models = any(line.startswith("MODEL") for line in lines)
    current_model: int | None = None
    model_count = 0
    for line in lines:
        if _is_pdb_end_record(line):
            break
        if line.startswith("MODEL"):
            model_count += 1
            current_model = model_count
            continue
        if line.startswith("ENDMDL"):
            current_model = None
            continue
        if not line.startswith("CONECT"):
            continue
        selected_model_scope = not explicit_models or current_model in {None, 1}
        try:
            serials = [
                int(line[offset : offset + 5])
                for offset in range(6, len(line), 5)
                if line[offset : offset + 5].strip()
            ]
        except ValueError as exc:
            raise CoordinateInputError(
                "MALFORMED_PDB_INPUT", f"invalid CONECT record: {line!r}"
            ) from exc
        if len(serials) < 2:
            continue
        source = serials[0]
        for target in serials[1:]:
            if source != target:
                pair = tuple(sorted((source, target)))
                all_pairs.add(pair)
                if selected_model_scope:
                    pairs.add(pair)
    return sorted(pairs), len(all_pairs)


def _is_immediate_residue_successor(
    predecessor: gemmi.AtomAddress, successor: gemmi.AtomAddress
) -> bool:
    """Return true only for an unambiguous next residue in PDB ordering."""
    if predecessor.chain_name != successor.chain_name:
        return False
    left_num = int(predecessor.res_id.seqid.num)
    right_num = int(successor.res_id.seqid.num)
    left_code = predecessor.res_id.seqid.icode.strip()
    right_code = successor.res_id.seqid.icode.strip()
    if right_num == left_num + 1:
        return not left_code and not right_code
    if right_num != left_num or len(right_code) != 1:
        return False
    if not left_code:
        return right_code == "A"
    return (
        len(left_code) == 1
        and left_code.isascii()
        and right_code.isascii()
        and ord(right_code) == ord(left_code) + 1
    )


def _requires_explicit_connection(connection: gemmi.Connection) -> bool:
    endpoints = {
        connection.partner1.atom_name.strip(): connection.partner1,
        connection.partner2.atom_name.strip(): connection.partner2,
    }
    if set(endpoints) != {"C", "N"}:
        return True
    # Only C(i)-N(i+1) is an implicit linear-backbone bond.  In particular,
    # N(1)-C(2) in a two-residue peptide is a head-to-tail closure and must be
    # materialized even though the residue numbers differ by one.
    return not _is_immediate_residue_successor(endpoints["C"], endpoints["N"])


def _model_endpoint_signature(
    model: gemmi.Model, address: gemmi.AtomAddress
) -> tuple[str, str] | None:
    chains = [chain for chain in model if chain.name == address.chain_name]
    if len(chains) != 1:
        return None
    residues = [
        residue
        for residue in chains[0]
        if int(residue.seqid.num) == int(address.res_id.seqid.num)
        and residue.seqid.icode.strip() == address.res_id.seqid.icode.strip()
        and residue.name == address.res_id.name
    ]
    if len(residues) != 1:
        return None
    atoms = [atom for atom in residues[0] if atom.name.strip() == address.atom_name.strip()]
    elements = {atom.element.name for atom in atoms}
    if len(atoms) == 0 or len(elements) != 1:
        return None
    return residues[0].name, next(iter(elements))


def _connection_is_model_invariant(
    structure: gemmi.Structure, connection: gemmi.Connection
) -> bool:
    signatures = [
        (
            _model_endpoint_signature(model, connection.partner1),
            _model_endpoint_signature(model, connection.partner2),
        )
        for model in structure
    ]
    return bool(
        signatures
        and signatures[0][0] is not None
        and signatures[0][1] is not None
        and all(signature == signatures[0] for signature in signatures[1:])
    )


def _connection_endpoint_retained(
    address: gemmi.AtomAddress,
    selected_altlocs: dict[tuple[int, str, str], str],
) -> bool:
    selected_altloc = selected_altlocs.get(_address_key(address))
    return selected_altloc is not None and selected_altloc == _clean_altloc(address.altloc)


def _append_conect(path: Path, pairs: list[tuple[int, int]]) -> None:
    if not pairs:
        return
    lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    end_index = next(
        (index for index, line in enumerate(lines) if line.startswith("END")),
        len(lines),
    )
    lines[end_index:end_index] = [f"CONECT{left:5d}{right:5d}" for left, right in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def mmcif_to_pdb(
    source: str | Path,
    output: str | Path,
    chain_id: str,
    *,
    source_format: str = "mmcif",
) -> dict[str, Any]:
    """Project one coordinate model and chain to PDB without chemical inference."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    _validate_coordinate_source(source)
    if source == output or (output.exists() and os.path.samefile(source, output)):
        raise CoordinateInputError(
            "COORDINATE_PROJECTION_OUTPUT_CONFLICT",
            "coordinate projection output must differ from the source path",
        )
    source_hash = _source_sha256(source)
    if source_format == "mmcif":
        structure = _read_mmcif_structure(source)
        source_layout_audit: dict[str, Any] = {}
        source_polymer_evidence = _mmcif_source_polymer_evidence(
            source, chain_id
        )
    elif source_format == "pdb":
        structure, ignored_later_model_connections = _read_pdb_structure(source)
        source_layout_audit = _pdb_layout_audit(source, chain_id)
        source_polymer_evidence = _pdb_source_polymer_evidence(source, chain_id)
        source_layout_audit["ignored_later_model_structured_connection_count"] = (
            ignored_later_model_connections
        )
    else:
        raise ValueError(f"unsupported projection source format: {source_format}")
    if len(structure) == 0:
        raise CoordinateInputError(
            "EMPTY_COORDINATE_INPUT", f"{source_format} input has no models"
        )
    source_model_count = len(structure)
    model = structure[0]
    explicit_pdb_polymer_subchains = (
        _explicit_pdb_polymer_subchains(structure)
        if source_format == "pdb"
        else set()
    )
    matching, reconstructable = _reconstructable_chain_indices(model, chain_id)
    solvent_only_indices: list[int] = []
    excluded_nonpolymer_indices: list[int] = []
    pdb_chain_object_audit: dict[str, Any] = {
        "pdb_chain_object_selection_mode": "not_applicable",
        "source_polymer_chain_object_count": None,
        "source_gemmi_polymer_chain_object_count": None,
        "explicit_seqres_polymer_subchains": None,
        "selected_source_chain_object_index": None,
        "pdb_matching_chain_object_rows": None,
        "excluded_nonselected_chain_object_count": 0,
        "excluded_nonselected_chain_residue_count": 0,
        "excluded_nonselected_chain_atom_count": 0,
        "excluded_nonselected_chain_heavy_atom_count": 0,
        "excluded_nonpolymer_chain_object_count": 0,
        "excluded_nonpolymer_chain_residue_count": 0,
        "excluded_nonpolymer_chain_atom_count": 0,
        "excluded_nonpolymer_chain_heavy_atom_count": 0,
        "duplicate_auth_chain_boundary_audit_applied": False,
        "cross_object_structured_connection_count": 0,
        "ambiguous_object_structured_connection_count": 0,
        "cross_object_pdb_conect_pair_count": 0,
        "unresolved_object_pdb_conect_pair_count": 0,
    }
    if len(matching) == 1:
        selected_chain_index = matching[0]
        if source_format == "pdb":
            source_polymer_count = sum(
                any(
                    residue.entity_type == gemmi.EntityType.Polymer
                    and residue.subchain in explicit_pdb_polymer_subchains
                    for residue in model[index]
                )
                for index in matching
            )
            source_gemmi_polymer_count = sum(
                any(
                    residue.entity_type == gemmi.EntityType.Polymer
                    for residue in model[index]
                )
                for index in matching
            )
            pdb_chain_object_audit.update({
                "pdb_chain_object_selection_mode": "single_matching_chain_object",
                "source_polymer_chain_object_count": source_polymer_count,
                "source_gemmi_polymer_chain_object_count": (
                    source_gemmi_polymer_count
                ),
                "explicit_seqres_polymer_subchains": sorted(
                    explicit_pdb_polymer_subchains
                ),
                "selected_source_chain_object_index": selected_chain_index,
                "pdb_matching_chain_object_rows": _pdb_chain_object_rows(
                    model,
                    matching,
                    selected_chain_index,
                    explicit_pdb_polymer_subchains,
                ),
            })
    elif source_format == "pdb":
        (
            selected_chain_index,
            excluded_nonpolymer_indices,
            pdb_chain_object_audit,
        ) = _select_unique_pdb_polymer_chain_object(
            model,
            chain_id,
            matching,
            explicit_pdb_polymer_subchains,
        )
        solvent_only_indices = [
            index
            for index in excluded_nonpolymer_indices
            if index not in reconstructable
        ]
    else:
        raise CoordinateInputError(
            "CHAIN_SELECTION_FAILED",
            f"model 1 has {len(matching)} chains named {chain_id!r} "
            f"({len(reconstructable)} contain reconstructable residues)",
        )
    excluded_segment_residues = sum(
        len(model[index]) for index in solvent_only_indices
    )
    excluded_segment_atoms = sum(
        len(residue)
        for index in solvent_only_indices
        for residue in model[index]
    )
    excluded_segment_heavy_atoms = sum(
        atom.element.name != "H"
        for index in solvent_only_indices
        for residue in model[index]
        for atom in residue
    )
    source_connections = [
        connection
        for connection in structure.connections
        if connection.partner1.chain_name == chain_id
        and connection.partner2.chain_name == chain_id
    ]
    structured_connection_count = len(source_connections)
    ignored_nonselected_model_connections = 0
    model_invariant_unscoped_connection_count = 0
    if source_format == "mmcif" and source_model_count > 1 and source_connections:
        connection_scopes = _mmcif_connection_model_scopes(source)
        unresolved_connections = [
            connection
            for connection in source_connections
            if connection_scopes.get(connection.name) is None
        ]
        invariant_unscoped = {
            connection.name
            for connection in unresolved_connections
            if _connection_is_model_invariant(structure, connection)
        }
        unresolved = sorted(
            connection.name
            for connection in unresolved_connections
            if connection.name not in invariant_unscoped
        )
        if unresolved:
            raise CoordinateInputError(
                "MMCIF_CONNECTION_MODEL_SCOPE_UNRESOLVED",
                "multi-model mmCIF connections require explicit endpoint model "
                "numbers: " + ",".join(unresolved),
                not_supported=True,
            )
        model_invariant_unscoped_connection_count = len(invariant_unscoped)
        cross_model = sorted({
            connection.name
            for connection in source_connections
            if connection_scopes.get(connection.name) is not None
            and connection_scopes[connection.name][0]
            != connection_scopes[connection.name][1]
        })
        if cross_model:
            raise CoordinateInputError(
                "MMCIF_CROSS_MODEL_CONNECTION_NOT_SUPPORTED",
                "cross-model mmCIF connections cannot be projected to one model: "
                + ",".join(cross_model),
                not_supported=True,
            )
        selected_model_number = int(model.num)
        selected_model_connections = [
            connection
            for connection in source_connections
            if connection.name in invariant_unscoped
            or connection_scopes[connection.name][0] == selected_model_number
        ]
        ignored_nonselected_model_connections = (
            len(source_connections) - len(selected_model_connections)
        )
        source_connections = selected_model_connections
    source_pdb_conect_pairs: list[tuple[int, int]] = []
    source_pdb_conect_pair_count = 0
    if source_format == "pdb":
        source_pdb_conect_pairs, source_pdb_conect_pair_count = _pdb_conect_pairs(
            source
        )
        if source_pdb_conect_pairs:
            _validate_model_atom_serial_uniqueness(model)
        if pdb_chain_object_audit["duplicate_auth_chain_boundary_audit_applied"]:
            pdb_chain_object_audit.update(
                _validate_pdb_duplicate_chain_boundary(
                    model,
                    selected_chain_index,
                    excluded_nonpolymer_indices,
                    source_connections,
                    source_pdb_conect_pairs,
                )
            )
    isolated = structure.clone()
    for index in range(len(isolated) - 1, 0, -1):
        del isolated[index]
    isolated_model = isolated[0]
    for index in range(len(isolated_model) - 1, -1, -1):
        if index != selected_chain_index:
            del isolated_model[index]
    selected = isolated_model[0]
    label_entity_audit = (
        _retain_unique_polymer_subchain(selected)
        if source_format == "mmcif"
        else {
            "source_reconstructable_label_entity_count": None,
            "label_entity_selection_mode": "unavailable_in_pdb",
            "selected_label_subchain_id": None,
            "selected_entity_id": None,
            "excluded_nonselected_entity_residue_count": 0,
            "excluded_nonselected_entity_atom_count": 0,
            "excluded_nonselected_entity_heavy_atom_count": 0,
        }
    )
    residue_audit = _retain_reconstructable_residues(selected)
    residue_audit["excluded_solvent_or_ion_residue_count"] += (
        excluded_segment_residues
    )
    residue_audit["excluded_solvent_or_ion_atom_count"] += excluded_segment_atoms
    residue_audit["excluded_solvent_or_ion_heavy_atom_count"] += (
        excluded_segment_heavy_atoms
    )
    source_sequence_identity_audit = _source_sequence_identity_audit(
        structure,
        selected,
        chain_id,
        source_hash,
        source_format=source_format,
        source_layout_audit=source_layout_audit,
        source_polymer_evidence=source_polymer_evidence,
    )
    _validate_legacy_pdb_representability(selected)
    altloc_audit, selected_altlocs = _select_alternate_locations(selected)
    _normalize_selected_chain_segments(selected)
    selected_source_serials = _serial_addresses(selected)
    pdb_conect_address_pairs = [
        (selected_source_serials[left], selected_source_serials[right])
        for left, right in source_pdb_conect_pairs
        if left in selected_source_serials and right in selected_source_serials
    ]
    retained_endpoint_ids = {
        (
            int(residue.seqid.num),
            residue.seqid.icode.strip(),
            residue.name.strip().upper(),
            atom.name.strip(),
        )
        for residue in selected
        for atom in residue
    }
    connection_endpoint_selected = [
        (
            _source_address_key(connection.partner1) in retained_endpoint_ids,
            _source_address_key(connection.partner2) in retained_endpoint_ids,
        )
        for connection in source_connections
    ]
    retained_connections = [
        connection
        for connection, endpoints_selected in zip(
            source_connections, connection_endpoint_selected
        )
        if all(endpoints_selected)
        and _connection_endpoint_retained(connection.partner1, selected_altlocs)
        and _connection_endpoint_retained(connection.partner2, selected_altlocs)
    ]
    excluded_nonselected_residue_connections = sum(
        not all(endpoints_selected)
        for endpoints_selected in connection_endpoint_selected
    )
    altloc_mismatch_count = sum(
        not _connection_endpoint_retained(connection.partner1, selected_altlocs)
        or not _connection_endpoint_retained(connection.partner2, selected_altlocs)
        for connection, endpoints_selected in zip(
            source_connections, connection_endpoint_selected
        )
        if all(endpoints_selected)
    )
    connections = [
        connection
        for connection in retained_connections
        if _requires_explicit_connection(connection)
    ]
    normalized_chain = chain_id if len(chain_id) == 1 and chain_id.isascii() else "A"
    selected.name = normalized_chain
    isolated.connections.clear()
    # The projection contract is coordinate-entity scoped. Source SEQRES or
    # mmCIF polymer sequences may describe aliases or residues absent from the
    # selected coordinates, so they must not be serialized as current evidence.
    isolated.clear_sequences()
    expected_heavy = _heavy_atom_count(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = gemmi.PdbWriteOptions()
    options.ssbond_records = False
    options.link_records = False
    options.conect_records = False
    options.preserve_serial = False
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".pdb",
    )
    os.close(descriptor)
    temporary_output = Path(temporary_name)
    try:
        isolated.write_pdb(str(temporary_output), options)
        roundtrip = gemmi.read_structure(str(temporary_output))
        if len(roundtrip) != 1 or len(roundtrip[0]) != 1:
            raise CoordinateInputError(
                "MMCIF_PROJECTION_ROUNDTRIP_FAILED",
                "normalized output is not single-model and single-chain",
            )
        normalized_chain_obj = roundtrip[0][0]
        if _heavy_atom_count(normalized_chain_obj) != expected_heavy:
            raise CoordinateInputError(
                "MMCIF_PROJECTION_ATOM_LOSS",
                "heavy-atom count changed during projection",
            )
        serials = _atom_serials(normalized_chain_obj)
        pairs: list[tuple[int, int]] = []
        connection_rows = []
        for connection in connections:
            left_key = _address_key(connection.partner1)
            right_key = _address_key(connection.partner2)
            left = serials.get(left_key)
            right = serials.get(right_key)
            if left is None or right is None:
                raise CoordinateInputError(
                    "MMCIF_CONNECTION_ENDPOINT_LOST",
                    f"connection endpoint absent after projection: "
                    f"{left_key}-{right_key}",
                )
            pair = tuple(sorted((left, right)))
            if pair not in pairs:
                pairs.append(pair)
            connection_rows.append(
                {
                    "name": connection.name,
                    "type": str(connection.type),
                    "partner_1": list(left_key),
                    "partner_2": list(right_key),
                    "normalized_serials": list(pair),
                }
            )
        for left_key, right_key in pdb_conect_address_pairs:
            left = serials.get(left_key)
            right = serials.get(right_key)
            if left is None or right is None:
                continue
            pair = tuple(sorted((left, right)))
            if pair not in pairs:
                pairs.append(pair)
            connection_rows.append(
                {
                    "name": "PDB_CONECT",
                    "type": "PDB_CONECT",
                    "partner_1": list(left_key),
                    "partner_2": list(right_key),
                    "normalized_serials": list(pair),
                }
            )
        _append_conect(temporary_output, sorted(pairs))
        normalized_hash = sha256(temporary_output)
        os.replace(temporary_output, output)
    except CoordinateInputError:
        raise
    except (IndexError, OSError, RuntimeError, UnicodeError, ValueError) as exc:
        raise CoordinateInputError(
            "COORDINATE_PROJECTION_WRITE_FAILED",
            f"coordinate projection could not be serialized: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    finally:
        try:
            temporary_output.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "source_format": source_format,
        "source_sha256": source_hash,
        "source_model_count": source_model_count,
        "selected_model_index": 1,
        "selected_auth_chain_id": chain_id,
        "source_matching_chain_object_count": len(matching),
        "source_reconstructable_chain_object_count": len(reconstructable),
        "excluded_solvent_only_chain_object_count": len(solvent_only_indices),
        "normalized_chain_id": normalized_chain,
        "normalized_heavy_atom_count": expected_heavy,
        "source_sequence_metadata_removed": True,
        "source_sequence_identity_audit": source_sequence_identity_audit,
        "structured_connection_count": structured_connection_count,
        "first_model_structured_connection_count": len(source_connections),
        "ignored_nonselected_model_structured_connection_count": (
            ignored_nonselected_model_connections
        ),
        "model_invariant_unscoped_connection_count": (
            model_invariant_unscoped_connection_count
        ),
        "structured_connection_altloc_mismatch_count": altloc_mismatch_count,
        "excluded_nonselected_residue_structured_connection_count": (
            excluded_nonselected_residue_connections
        ),
        "source_pdb_conect_pair_count": source_pdb_conect_pair_count,
        "first_model_pdb_conect_pair_count": len(source_pdb_conect_pairs),
        "selected_chain_pdb_conect_pair_count": len(pdb_conect_address_pairs),
        "materialized_explicit_connection_count": len(pairs),
        "connection_rows": connection_rows,
        "normalized_sha256": normalized_hash,
        "projection_applied": True,
        **source_layout_audit,
        **pdb_chain_object_audit,
        **label_entity_audit,
        **residue_audit,
        **altloc_audit,
    }


def pdb_to_pdb(
    source: str | Path,
    output: str | Path,
    chain_id: str,
) -> dict[str, Any]:
    """Project the first PDB model and selected chain to normalized PDB."""
    return mmcif_to_pdb(
        source,
        output,
        chain_id,
        source_format="pdb",
    )


@contextlib.contextmanager
def prepare_coordinate_input(
    source: str | Path, chain_id: str
) -> Iterator[PreparedCoordinateInput]:
    source = Path(source).resolve()
    fmt = coordinate_format(source)
    _validate_coordinate_source(source)
    compressed = source.name.lower().endswith(".gz")
    with contextlib.ExitStack() as stack:
        original_source_sha256 = _source_sha256(source)
        readable = source
        if compressed:
            suffix = ".cif" if fmt == "mmcif" else ".pdb"
            handle = stack.enter_context(
                tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            )
            temporary = Path(handle.name)
            handle.close()
            stack.callback(temporary.unlink, missing_ok=True)
            try:
                with gzip.open(source, "rb") as source_handle:
                    payload = source_handle.read()
            except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
                raise CoordinateInputError(
                    "INVALID_COMPRESSED_COORDINATE_INPUT",
                    f"compressed coordinate input is invalid: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            except OSError as exc:
                raise CoordinateInputError(
                    "COORDINATE_INPUT_READ_FAILED",
                    f"compressed coordinate input cannot be read: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            temporary.write_bytes(payload)
            readable = temporary
        else:
            suffix = ".cif" if fmt == "mmcif" else ".pdb"
            handle = stack.enter_context(
                tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            )
            snapshot = Path(handle.name)
            handle.close()
            stack.callback(snapshot.unlink, missing_ok=True)
            try:
                snapshot.write_bytes(source.read_bytes())
            except OSError as exc:
                raise CoordinateInputError(
                    "COORDINATE_INPUT_READ_FAILED",
                    f"coordinate input cannot be snapshotted: {type(exc).__name__}: {exc}",
                ) from exc
            if sha256(snapshot) != original_source_sha256:
                raise CoordinateInputError(
                    "COORDINATE_INPUT_CHANGED_DURING_READ",
                    "coordinate input changed while its immutable payload snapshot was created",
                )
            readable = snapshot
        if fmt == "pdb":
            directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="cycpep_pdb_"))
            )
            projected = directory / "normalized.pdb"
            audit = pdb_to_pdb(readable, projected, chain_id)
            if compressed:
                audit["decompressed_payload_sha256"] = audit["source_sha256"]
                audit["source_sha256"] = original_source_sha256
                source_identity = audit.get("source_sequence_identity_audit")
                if isinstance(source_identity, dict):
                    source_identity["source_sha256"] = original_source_sha256
                    for row in source_identity.get("rows", []):
                        if isinstance(row, dict):
                            row["source_sha256"] = original_source_sha256
            if _source_sha256(source) != original_source_sha256:
                raise CoordinateInputError(
                    "COORDINATE_INPUT_CHANGED_DURING_READ",
                    "coordinate input changed during normalization",
                )
            audit["compressed_source"] = compressed
            yield PreparedCoordinateInput(
                projected,
                audit["normalized_chain_id"],
                "pdb",
                audit,
            )
            return
        directory = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="cycpep_mmcif_"))
        )
        projected = directory / "normalized.pdb"
        audit = mmcif_to_pdb(readable, projected, chain_id)
        try:
            embedded_components = extract_embedded_chem_comp_templates(readable)
        except EmbeddedChemCompError as exc:
            raise CoordinateInputError(
                exc.code, str(exc), not_supported=not exc.rejected
            ) from exc
        audit["embedded_chem_comp_templates"] = embedded_components
        audit["embedded_chem_comp_template_count"] = len(embedded_components)
        audit["embedded_chem_comp_ids"] = sorted(embedded_components)
        source_hashes = {
            component.get("source_input_sha256")
            for component in embedded_components.values()
        }
        audit["embedded_chem_comp_source_payload_sha256"] = (
            next(iter(source_hashes))
            if len(source_hashes) == 1
            else sha256(readable)
        )
        if compressed:
            audit["decompressed_payload_sha256"] = audit["source_sha256"]
            audit["source_sha256"] = original_source_sha256
            source_identity = audit.get("source_sequence_identity_audit")
            if isinstance(source_identity, dict):
                source_identity["source_sha256"] = original_source_sha256
                for row in source_identity.get("rows", []):
                    if isinstance(row, dict):
                        row["source_sha256"] = original_source_sha256
        if _source_sha256(source) != original_source_sha256:
            raise CoordinateInputError(
                "COORDINATE_INPUT_CHANGED_DURING_READ",
                "coordinate input changed during normalization",
            )
        audit["compressed_source"] = compressed
        yield PreparedCoordinateInput(
            projected, audit["normalized_chain_id"], "mmcif", audit
        )
