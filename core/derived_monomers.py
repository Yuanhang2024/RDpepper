"""Fail-closed construction and persistence for structure-derived monomers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import Descriptors

from .cxsmiles_gen import gen_cxsmiles


_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_UNIFIED_CSV = _PACKAGE_DIR / "unified_monomer_library.csv"
_DERIVED_CSV = _PACKAGE_DIR / "derived_monomer_library.csv"
_MANIFEST_JSON = _PACKAGE_DIR / "derived_monomer_manifest.json"
_QUARANTINE_CSV = _PACKAGE_DIR / "derived_monomer_quarantine.csv"
_DESCRIPTOR_MAP = dict(Descriptors._descList)
_QUARANTINE_FIELDS = (
    "record_id", "pdb_resname", "status", "reason_codes", "input_sha256",
    "candidate_graph_count", "details_json",
)


class PortSemanticMismatchError(ValueError):
    """The free graph matches, but its evidenced attachment semantics do not."""


def unified_schema() -> tuple[str, ...]:
    with _UNIFIED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration as exc:
            raise ValueError("Unified monomer library has no header") from exc


def _canonical_smiles(smiles: str, *, stereo: bool) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"unparseable monomer SMILES: {smiles!r}")
    if not stereo:
        molecule = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=bool(stereo)
    )


def _graph_metadata(smiles: str) -> dict[str, str]:
    canonical = _canonical_smiles(smiles, stereo=True)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError("canonical monomer SMILES did not round-trip")
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey:
        raise ValueError("monomer did not yield a full InChIKey")
    return {
        "smiles_canonical": canonical,
        "smiles_canonical_nostereo": _canonical_smiles(canonical, stereo=False),
        "inchikey": inchikey,
        "graph_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _descriptor_values(smiles: str, fields: Iterable[str]) -> dict[str, float]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("descriptor input is not parseable")
    values = {}
    for field in fields:
        function = _DESCRIPTOR_MAP.get(field)
        if function is None:
            raise ValueError(f"RDKit descriptor {field!r} is unavailable")
        try:
            value = float(function(molecule))
        except Exception as exc:
            raise ValueError(f"RDKit descriptor {field!r} failed") from exc
        if not math.isfinite(value):
            raise ValueError(f"RDKit descriptor {field!r} is nonfinite")
        values[field] = value
    return values


def _safe_resname(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().upper()).strip("_")
    return cleaned or "UNK"


def _cxsmiles_with_explicit_r3(
    neutral_smiles: str,
    mapped_smiles: str,
    cap: str,
) -> tuple[dict[str, str], dict[str, object]]:
    """Generate CXSMILES whose R3 dummy is bound to the evidenced atom."""
    marked = Chem.MolFromSmiles(mapped_smiles)
    if marked is None:
        raise ValueError("R3-marked monomer SMILES is not parseable")
    marked_atoms = [atom for atom in marked.GetAtoms() if atom.GetAtomMapNum() == 9003]
    if len(marked_atoms) != 1:
        raise ValueError("R3-marked monomer must contain exactly one atom map 9003")
    unmarked = Chem.Mol(marked)
    for atom in unmarked.GetAtoms():
        atom.SetAtomMapNum(0)
    if Chem.MolToInchiKey(unmarked) != Chem.MolToInchiKey(
        Chem.MolFromSmiles(neutral_smiles)
    ):
        raise ValueError("R3 marker changes the neutral monomer identity")
    generated = gen_cxsmiles(mapped_smiles)
    cxsmiles = str(generated.get("CXSMILES", "")).strip()
    molecule = Chem.MolFromSmiles(cxsmiles) if cxsmiles else None
    if molecule is None:
        raise ValueError("R3-marked monomer backbone could not be serialized")
    editable = Chem.RWMol(molecule)
    targets = [atom for atom in editable.GetAtoms() if atom.GetAtomMapNum() == 9003]
    if len(targets) != 1:
        raise ValueError("R3 atom marker was not preserved through CXSMILES generation")
    target = targets[0]
    anchor_element = target.GetSymbol()
    cap = str(cap).strip().upper()
    if cap == "H":
        explicit_h = int(target.GetNumExplicitHs())
        if explicit_h:
            target.SetNumExplicitHs(explicit_h - 1)
        dummy = Chem.Atom(0)
        dummy.SetProp("atomLabel", "_R3")
        dummy_index = editable.AddAtom(dummy)
        editable.AddBond(target.GetIdx(), dummy_index, Chem.BondType.SINGLE)
    elif cap == "OH":
        candidates = [
            neighbor for neighbor in target.GetNeighbors()
            if neighbor.GetAtomicNum() == 8
            and editable.GetBondBetweenAtoms(
                target.GetIdx(), neighbor.GetIdx()
            ).GetBondType() == Chem.BondType.SINGLE
        ]
        if len(candidates) != 1:
            raise ValueError("R3=OH requires one unique side-chain hydroxyl oxygen")
        dummy = candidates[0]
        dummy.SetAtomicNum(0)
        dummy.SetNumExplicitHs(0)
        dummy.SetNoImplicit(True)
        dummy.SetProp("atomLabel", "_R3")
    else:
        raise ValueError(f"unsupported explicit R3 cap {cap!r}")
    target.SetAtomMapNum(0)
    molecule = editable.GetMol()
    molecule.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(molecule)
    cxsmiles = Chem.MolToCXSmiles(molecule)
    reparsed = Chem.MolFromSmiles(cxsmiles)
    labels = [
        atom.GetProp("atomLabel")
        for atom in reparsed.GetAtoms() if atom.HasProp("atomLabel")
    ]
    if labels.count("_R1") != 1 or labels.count("_R2") != 1 or labels.count("_R3") != 1:
        raise ValueError("explicit R3 CXSMILES does not contain unique R1/R2/R3 ports")
    r3_dummy = next(
        atom for atom in reparsed.GetAtoms()
        if atom.HasProp("atomLabel") and atom.GetProp("atomLabel") == "_R3"
    )
    neighbors = list(r3_dummy.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError("R3 dummy must have exactly one anchor")
    generated = dict(generated)
    generated["CXSMILES"] = cxsmiles
    generated["R3"] = cap
    return generated, {
        "r3_cap": cap,
        "r3_anchor_element": anchor_element,
        "r3_anchor_cx_index": int(neighbors[0].GetIdx()),
        "r3_cx_sha256": hashlib.sha256(cxsmiles.encode("utf-8")).hexdigest(),
    }


def build_derived_row(
    pdb_resname: str,
    smiles: str,
    *,
    monomer_id: int,
    r3_mapped_smiles: str | None = None,
    r3_port: dict | None = None,
    source_cxsmiles: str | None = None,
    source_r_groups: dict | None = None,
    source: str = "local_structure_derived",
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one exact 235-column row from already established chemistry."""
    schema = unified_schema()
    if len(schema) != 235:
        raise ValueError(f"expected 235 Unified fields, observed {len(schema)}")
    metadata = _graph_metadata(smiles)
    r3_metadata: dict[str, object] = {}
    if source_cxsmiles:
        if r3_mapped_smiles:
            raise ValueError(
                "source_cxsmiles and r3_mapped_smiles are mutually exclusive"
            )
        molecule = Chem.MolFromSmiles(str(source_cxsmiles))
        if molecule is None:
            raise ValueError("source CXSMILES is not parseable")
        labels = [
            atom.GetProp("atomLabel")
            for atom in molecule.GetAtoms()
            if atom.HasProp("atomLabel")
        ]
        if labels.count("_R1") != 1 or labels.count("_R2") != 1:
            raise ValueError(
                "source CXSMILES requires unique R1 and R2 ports"
            )
        generated = {
            "CXSMILES": str(source_cxsmiles),
            "R1": str((source_r_groups or {}).get("R1", "H")),
            "R2": str((source_r_groups or {}).get("R2", "OH")),
            "R3": str((source_r_groups or {}).get("R3", "-")),
        }
    elif r3_mapped_smiles:
        if not r3_port or not r3_port.get("cap"):
            raise ValueError("explicit R3 marker requires structured port evidence")
        generated, r3_metadata = _cxsmiles_with_explicit_r3(
            metadata["smiles_canonical"], r3_mapped_smiles, str(r3_port["cap"])
        )
    else:
        generated = gen_cxsmiles(metadata["smiles_canonical"])
    cxsmiles = str(generated.get("CXSMILES", "")).strip()
    if not cxsmiles or "_R1" not in cxsmiles or "_R2" not in cxsmiles:
        raise ValueError("monomer backbone/ports could not be determined uniquely")
    symbol = f"LCL_{_safe_resname(pdb_resname)}_{metadata['graph_sha256'][:8]}"
    row: dict[str, object] = {field: "" for field in schema}
    row.update({
        "monomer_id": int(monomer_id),
        "symbol": symbol,
        "source": str(source),
        "smiles_canonical": metadata["smiles_canonical"],
        "smiles_canonical_nostereo": metadata["smiles_canonical_nostereo"],
        "smiles_original": metadata["smiles_canonical"],
        "compound_name": f"Structure-derived {pdb_resname} monomer ({metadata['graph_sha256'][:8]})",
        "monomer_type": "NNAA",
        "polymer_type": "PEPTIDE",
        "version": "structure-derived-1",
        "replaced_SMILES": metadata["smiles_canonical"],
        "CXSMILES": cxsmiles,
        "Monomer_Type": "Backbone",
        "Polymer_Type": "PEPTIDE",
        "R1": str(generated.get("R1", "H")),
        "R2": str(generated.get("R2", "OH")),
        "R3": str(generated.get("R3", "-")),
    })
    row.update(_descriptor_values(metadata["smiles_canonical"], schema[20:-7]))
    manifest = {
        "monomer_id": int(monomer_id),
        "symbol": symbol,
        "pdb_resname": str(pdb_resname),
        "graph_sha256": metadata["graph_sha256"],
        "full_inchikey": metadata["inchikey"],
        "source": str(source),
        "inference_version": "structure-derived-1",
        **r3_metadata,
    }
    if r3_port:
        manifest["r3_port_evidence"] = dict(r3_port)
    return row, manifest


def next_derived_id() -> int:
    maximum = 0
    for path in (_UNIFIED_CSV, _DERIVED_CSV):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    maximum = max(maximum, int(str(row.get("monomer_id", "")).strip()))
                except ValueError:
                    continue
    return maximum + 1


def _ported_graph_identity(cxsmiles: str, cap: str) -> str:
    molecule = Chem.MolFromSmiles(str(cxsmiles))
    if molecule is None:
        raise ValueError("R3 ported graph identity could not be constructed")
    labels = sorted(
        atom.GetProp("atomLabel")
        for atom in molecule.GetAtoms() if atom.HasProp("atomLabel")
    )
    if labels.count("_R3") != 1:
        raise ValueError("R3 ported graph does not expose one R3 label")
    return (
        f"{str(cap).strip().upper()}|"
        f"{Chem.MolToCXSmiles(molecule, isomericSmiles=True)}"
    )


def _candidate_port_identity(candidate: dict) -> str | None:
    source_cxsmiles = str(
        candidate.get("source_cxsmiles") or ""
    ).strip()
    if source_cxsmiles and "_R3" in source_cxsmiles:
        return _ported_graph_identity(
            source_cxsmiles,
            str((candidate.get("source_r_groups") or {}).get("R3", "-")),
        )
    mapped_smiles = str(candidate.get("r3_mapped_smiles") or "").strip()
    port = candidate.get("r3_port")
    if not mapped_smiles and not port:
        return None
    if not mapped_smiles or not isinstance(port, dict) or not port.get("cap"):
        raise PortSemanticMismatchError("incomplete candidate R3 port semantics")
    try:
        generated, _metadata = _cxsmiles_with_explicit_r3(
            str(candidate.get("smiles", "")), mapped_smiles, str(port["cap"])
        )
        return _ported_graph_identity(generated["CXSMILES"], str(port["cap"]))
    except ValueError as exc:
        raise PortSemanticMismatchError(str(exc)) from exc


def _unified_port_identity(row: dict) -> str | None:
    cxsmiles = str(row.get("CXSMILES", "")).strip()
    if not cxsmiles or "_R3" not in cxsmiles:
        return None
    return _ported_graph_identity(cxsmiles, str(row.get("R3", "-")))


@lru_cache(maxsize=1)
def _unified_identities() -> dict[str, tuple[str, str | None]]:
    identities = {}
    with _UNIFIED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            smiles = str(row.get("smiles_canonical", "")).strip()
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                continue
            key = Chem.MolToInchiKey(molecule)
            symbol = str(row.get("symbol", "")).strip()
            if key and symbol:
                identities.setdefault(
                    key, (symbol, _unified_port_identity(row))
                )
    return identities


def _alias(candidate: dict, target_symbol: str, metadata: dict) -> dict:
    row = {
        "pdb_resname": str(candidate["pdb_resname"]),
        "target_symbol": target_symbol,
        "graph_sha256": metadata["graph_sha256"],
        "full_inchikey": metadata["inchikey"],
        "input_sha256": str(candidate.get("input_sha256", "")),
        "source_entity_id": str(candidate.get("source_entity_id", "")),
    }
    for field in (
        "component_snapshot_sha256",
        "component_source_input_sha256",
        "resolution_mode",
    ):
        if candidate.get(field):
            row[field] = str(candidate[field])
    return row


def build_derived_batch(
    candidates: Iterable[dict],
    *,
    starting_id: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build graph-sorted rows and collapse existing/duplicate graphs to aliases."""
    prepared = {}
    duplicate_aliases = []
    aliases = []
    unified = _unified_identities()
    for candidate in candidates:
        candidate = dict(candidate)
        metadata = _graph_metadata(str(candidate["smiles"]))
        graph = metadata["graph_sha256"]
        candidate_port = _candidate_port_identity(candidate)
        if metadata["inchikey"] in unified:
            target_symbol, target_port = unified[metadata["inchikey"]]
            if candidate_port is not None and candidate_port != target_port:
                if candidate.get("resolution_mode") != "embedded_mmcif_chem_comp":
                    raise PortSemanticMismatchError(
                        f"free graph matches Unified {target_symbol}, but R3 anchor/cap differs"
                    )
            else:
                aliases.append(_alias(candidate, target_symbol, metadata))
                continue
        if graph in prepared:
            existing_port = prepared[graph][1]
            if candidate_port != existing_port:
                raise PortSemanticMismatchError(
                    "duplicate free graphs carry different R3 anchor/cap semantics"
                )
            duplicate_aliases.append((candidate, graph, metadata))
            continue
        prepared[graph] = (candidate, candidate_port)
    rows, manifests = [], []
    start = next_derived_id() if starting_id is None else int(starting_id)
    if start < 1:
        raise ValueError("starting_id must be positive")
    symbol_by_graph = {}
    for offset, graph in enumerate(sorted(prepared)):
        candidate = prepared[graph][0]
        row, manifest = build_derived_row(
            str(candidate["pdb_resname"]),
            str(candidate["smiles"]),
            monomer_id=start + offset,
            r3_mapped_smiles=candidate.get("r3_mapped_smiles"),
            r3_port=candidate.get("r3_port"),
            source_cxsmiles=candidate.get("source_cxsmiles"),
            source_r_groups=candidate.get("source_r_groups"),
            source=str(
                candidate.get("source")
                or "local_structure_derived"
            ),
        )
        manifest["input_sha256"] = str(candidate.get("input_sha256", ""))
        manifest["source_entity_id"] = str(candidate.get("source_entity_id", ""))
        for field in (
            "component_snapshot_sha256",
            "component_source_input_sha256",
            "resolution_mode",
        ):
            if candidate.get(field):
                manifest[field] = str(candidate[field])
        rows.append(row)
        manifests.append(manifest)
        symbol_by_graph[graph] = str(row["symbol"])
    aliases.extend(
        _alias(candidate, symbol_by_graph[graph], metadata)
        for candidate, graph, metadata in duplicate_aliases
    )
    aliases.sort(key=lambda row: (
        row["pdb_resname"], row["target_symbol"], row["input_sha256"]
    ))
    return rows, manifests, aliases


def _atomic_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(handle.name, path)
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def write_overlay_artifacts(
    rows: list[dict],
    manifest_entries: list[dict],
    quarantine_rows: list[dict],
    *,
    pdb_aliases: list[dict] | None = None,
    derived_path: Path = _DERIVED_CSV,
    manifest_path: Path = _MANIFEST_JSON,
    quarantine_path: Path = _QUARANTINE_CSV,
) -> None:
    """Atomically write validated overlay artifacts; Unified is never touched."""
    from ..paths import _map_utils

    _map_utils._validate_derived_rows(
        rows, _map_utils._load_unified_with_reconstruction_rows(),
        require_exact_schema=True,
    )
    _atomic_csv(derived_path, unified_schema(), rows)
    _atomic_csv(quarantine_path, _QUARANTINE_FIELDS, quarantine_rows)
    manifest = {
        "schema_version": "1.0.0",
        "status": "VALIDATED_OVERLAY" if rows else "EMPTY_VALIDATED_OVERLAY",
        "source": "local_structure_derived",
        "unified_library_sha256": hashlib.sha256(_UNIFIED_CSV.read_bytes()).hexdigest(),
        "entries": manifest_entries,
        "pdb_aliases": list(pdb_aliases or []),
        "quarantine_count": len(quarantine_rows),
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def _read_csv_rows(path: Path, expected_fields: tuple[str, ...]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"existing overlay CSV schema is invalid: {path.name}")
        return list(reader)


def _file_record(path: Path) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _validate_loaded_overlay_semantics(rows: list[dict], manifest: dict) -> dict[str, dict]:
    """Recompute parent-overlay identities instead of trusting its manifests."""
    from ..paths import _map_utils

    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("existing overlay manifest schema_version is invalid")
    expected_status = "VALIDATED_OVERLAY" if rows else "EMPTY_VALIDATED_OVERLAY"
    if manifest.get("status") != expected_status:
        raise ValueError("existing overlay manifest status is invalid")
    if manifest.get("source") != "local_structure_derived":
        raise ValueError("existing overlay manifest source is invalid")
    entries = manifest.get("entries")
    aliases = manifest.get("pdb_aliases")
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise ValueError("existing overlay manifest entries are malformed")
    if not isinstance(aliases, list) or any(not isinstance(item, dict) for item in aliases):
        raise ValueError("existing overlay manifest aliases are malformed")

    base_rows = _map_utils._load_unified_with_reconstruction_rows()
    validated = _map_utils._validate_derived_rows(
        rows, base_rows, require_exact_schema=True
    )
    entries_by_symbol = {}
    for entry in entries:
        symbol = str(entry.get("symbol", "")).strip()
        if not symbol or symbol in entries_by_symbol:
            raise ValueError("existing overlay manifest symbols are missing or duplicated")
        entries_by_symbol[symbol] = entry
    if set(entries_by_symbol) != set(validated):
        raise ValueError("existing overlay manifest entries do not match CSV rows")

    for symbol, row in validated.items():
        entry = entries_by_symbol[symbol]
        metadata = _graph_metadata(str(row.get("smiles_canonical", "")))
        expected = {
            "symbol": symbol,
            "monomer_id": str(row.get("monomer_id", "")).strip(),
            "graph_sha256": metadata["graph_sha256"],
            "full_inchikey": metadata["inchikey"],
            "source": str(row.get("source", "")).strip(),
            "inference_version": str(row.get("version", "")).strip(),
        }
        for field, value in expected.items():
            if str(entry.get(field, "")).strip() != value:
                raise ValueError(
                    f"existing overlay manifest {field} does not match {symbol!r}"
                )
        cxsmiles = str(row.get("CXSMILES", "")).strip()
        r3_dummies = []
        molecule = Chem.MolFromSmiles(cxsmiles)
        if molecule is not None:
            r3_dummies = [
                atom for atom in molecule.GetAtoms()
                if atom.HasProp("atomLabel") and atom.GetProp("atomLabel") == "_R3"
            ]
        if r3_dummies:
            if len(r3_dummies) != 1 or len(r3_dummies[0].GetNeighbors()) != 1:
                raise ValueError(f"existing overlay R3 graph is ambiguous for {symbol!r}")
            anchor = r3_dummies[0].GetNeighbors()[0]
            expected_r3 = {
                "r3_cap": str(row.get("R3", "")).strip().upper(),
                "r3_anchor_element": anchor.GetSymbol(),
                "r3_anchor_cx_index": str(anchor.GetIdx()),
                "r3_cx_sha256": hashlib.sha256(cxsmiles.encode("utf-8")).hexdigest(),
            }
            for field, value in expected_r3.items():
                if str(entry.get(field, "")).strip() != value:
                    raise ValueError(
                        f"existing overlay manifest {field} does not match {symbol!r}"
                    )
        elif any(str(entry.get(field, "")).strip() for field in (
            "r3_cap", "r3_anchor_element", "r3_anchor_cx_index", "r3_cx_sha256"
        )):
            raise ValueError(f"existing overlay manifest has orphan R3 metadata for {symbol!r}")

    all_rows = {**base_rows, **validated}
    full_rows = _map_utils._load_unified_single_file()
    seen_aliases = set()
    for alias in aliases:
        pdb_resname = str(alias.get("pdb_resname", "")).strip().upper()
        target = str(alias.get("target_symbol", "")).strip()
        if not pdb_resname or not target or target not in all_rows:
            raise ValueError("existing overlay manifest alias target is invalid")
        alias_key = (pdb_resname, target)
        if alias_key in seen_aliases:
            raise ValueError("existing overlay manifest contains duplicate aliases")
        seen_aliases.add(alias_key)
        identity = _map_utils._manifest_graph_identity(
            all_rows[target], label=f"existing overlay alias {pdb_resname!r}",
            full_rows=full_rows,
        )
        for field, value in identity.items():
            if str(alias.get(field, "")).strip() != value:
                raise ValueError(
                    f"existing overlay alias {pdb_resname!r} {field} is invalid"
                )
    return validated


def _load_verified_bundle(root: Path) -> tuple[list[dict], dict, list[dict], str]:
    bundle_manifest_path = root / "bundle_manifest.json"
    if not bundle_manifest_path.is_file():
        raise ValueError(f"existing overlay bundle lacks bundle_manifest.json: {root}")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if bundle_manifest.get("status") != "VALIDATED_APPEND_ONLY_OVERLAY_BUNDLE":
        raise ValueError("existing overlay bundle status is not validated")
    if bundle_manifest.get("schema_version") != "1.0.0":
        raise ValueError("existing overlay bundle schema_version is invalid")
    artifact_records = bundle_manifest.get("artifacts")
    expected_names = {
        "derived_monomer_library.csv",
        "derived_monomer_manifest.json",
        "derived_monomer_quarantine.csv",
    }
    if not isinstance(artifact_records, dict) or set(artifact_records) != expected_names:
        raise ValueError("existing overlay bundle artifact set is invalid")
    for name in sorted(expected_names):
        path = root / name
        observed = _file_record(path)
        expected = artifact_records[name]
        if observed != expected:
            raise ValueError(f"existing overlay bundle artifact mismatch: {name}")
    overlay_manifest = json.loads(
        (root / "derived_monomer_manifest.json").read_text(encoding="utf-8")
    )
    unified_hash = hashlib.sha256(_UNIFIED_CSV.read_bytes()).hexdigest()
    if overlay_manifest.get("unified_library_sha256") != unified_hash:
        raise ValueError("existing overlay bundle targets a different Unified library")
    rows = _read_csv_rows(root / "derived_monomer_library.csv", unified_schema())
    quarantine = _read_csv_rows(
        root / "derived_monomer_quarantine.csv", _QUARANTINE_FIELDS
    )
    if len(rows) != len(overlay_manifest.get("entries", [])):
        raise ValueError("existing overlay rows and manifest entries differ in count")
    _validate_loaded_overlay_semantics(rows, overlay_manifest)
    counts = {
        "derived_row_count": len(rows),
        "alias_count": len(overlay_manifest.get("pdb_aliases", [])),
        "quarantine_count": len(quarantine),
    }
    for field, value in counts.items():
        if bundle_manifest.get(field) != value:
            raise ValueError(f"existing overlay bundle {field} is invalid")
    if overlay_manifest.get("quarantine_count") != len(quarantine):
        raise ValueError("existing overlay manifest quarantine_count is invalid")
    return (
        rows,
        overlay_manifest,
        quarantine,
        hashlib.sha256(bundle_manifest_path.read_bytes()).hexdigest(),
    )


def _deduplicate_dicts(rows: Iterable[dict], fields: Iterable[str]) -> list[dict]:
    field_order = tuple(fields)
    unique = {}
    for row in rows:
        normalized = {field: row.get(field, "") for field in field_order}
        key = tuple(str(normalized[field]) for field in field_order)
        unique[key] = normalized
    return [unique[key] for key in sorted(unique)]


def publish_overlay_bundle(
    output_dir: str | Path,
    candidates: Iterable[dict],
    quarantine_rows: Iterable[dict] = (),
    *,
    existing_bundle: str | Path | None = None,
    publication_provenance: dict | None = None,
) -> Path:
    """Publish one immutable post-evaluation overlay generation atomically.

    Entity-local inference results must be collected only after every formal
    entity has run in isolation. Existing rows and monomer IDs are preserved;
    new graphs are sorted within the new batch and appended. The destination
    must not exist, so a prior generation can never be overwritten in place.
    """
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"overlay bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    existing_entries: list[dict] = []
    existing_aliases: list[dict] = []
    existing_quarantine: list[dict] = []
    parent_manifest_sha256 = None
    if existing_bundle is not None:
        (
            existing_rows,
            existing_manifest,
            existing_quarantine,
            parent_manifest_sha256,
        ) = _load_verified_bundle(Path(existing_bundle).resolve())
        existing_entries = [dict(row) for row in existing_manifest.get("entries", [])]
        existing_aliases = [dict(row) for row in existing_manifest.get("pdb_aliases", [])]

    existing_by_key = {}
    for row, entry in zip(existing_rows, existing_entries):
        key = str(entry.get("full_inchikey", ""))
        if not key or key in existing_by_key:
            raise ValueError("existing overlay has missing or duplicate full InChIKey")
        existing_by_key[key] = (
            str(row.get("symbol", "")),
            _unified_port_identity(row),
            str(entry.get("graph_sha256", "")),
        )

    remaining_candidates = []
    new_aliases = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        metadata = _graph_metadata(str(candidate.get("smiles", "")))
        existing = existing_by_key.get(metadata["inchikey"])
        if existing is None:
            remaining_candidates.append(candidate)
            continue
        target_symbol, target_port, graph_sha256 = existing
        candidate_port = _candidate_port_identity(candidate)
        if metadata["graph_sha256"] != graph_sha256:
            raise ValueError("full InChIKey collision with a different canonical graph")
        if candidate_port is not None and candidate_port != target_port:
            raise PortSemanticMismatchError(
                f"free graph matches derived {target_symbol}, but R3 anchor/cap differs"
            )
        new_aliases.append(_alias(candidate, target_symbol, metadata))

    maximum_id = max(
        [int(str(row.get("monomer_id", "0"))) for row in existing_rows] or [
            next_derived_id() - 1
        ]
    )
    new_rows, new_entries, batch_aliases = build_derived_batch(
        remaining_candidates, starting_id=maximum_id + 1
    )
    all_rows = existing_rows + new_rows
    all_entries = existing_entries + new_entries
    alias_fields = (
        "pdb_resname", "target_symbol", "graph_sha256", "full_inchikey",
        "input_sha256", "source_entity_id",
    )
    all_aliases = _deduplicate_dicts(
        [*existing_aliases, *new_aliases, *batch_aliases], alias_fields
    )
    all_quarantine = _deduplicate_dicts(
        [*existing_quarantine, *map(dict, quarantine_rows)], _QUARANTINE_FIELDS
    )

    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        write_overlay_artifacts(
            all_rows,
            all_entries,
            all_quarantine,
            pdb_aliases=all_aliases,
            derived_path=stage / "derived_monomer_library.csv",
            manifest_path=stage / "derived_monomer_manifest.json",
            quarantine_path=stage / "derived_monomer_quarantine.csv",
        )
        artifact_names = (
            "derived_monomer_library.csv",
            "derived_monomer_manifest.json",
            "derived_monomer_quarantine.csv",
        )
        bundle_manifest = {
            "schema_version": "1.0.0",
            "status": "VALIDATED_APPEND_ONLY_OVERLAY_BUNDLE",
            "parent_bundle_manifest_sha256": parent_manifest_sha256,
            "unified_library_sha256": hashlib.sha256(_UNIFIED_CSV.read_bytes()).hexdigest(),
            "derived_row_count": len(all_rows),
            "new_derived_row_count": len(new_rows),
            "alias_count": len(all_aliases),
            "quarantine_count": len(all_quarantine),
            "publication_provenance": dict(publication_provenance or {}),
            "artifacts": {
                name: _file_record(stage / name) for name in artifact_names
            },
        }
        (stage / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination
