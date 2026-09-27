"""Immutable, source-scoped views over cyclic-peptide conformer templates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_TEMPLATE_SOURCES = frozenset({"cpbind", "scaffold"})


class TemplateLibraryError(ValueError):
    """Raised when a template library cannot satisfy its declared contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_child(root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise TemplateLibraryError(
            f"template path escapes library root: {relative!r}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class TemplateLibraryView:
    """Read-only entries plus the provenance needed to interpret them."""

    root: Path
    index_path: Path
    index_sha256: str
    allowed_sources: frozenset[str]
    entries: Mapping[str, Mapping[str, Any]]
    formal: bool
    manifest_path: Path | None = None

    def template_path(self, entry: Mapping[str, Any]) -> Path:
        return _resolved_child(self.root, str(entry.get("pdb_path", "")))


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TemplateLibraryError(f"cannot read template manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TemplateLibraryError("template manifest must be a JSON object")
    return payload


def load_template_library_view(
    index_path: str | Path,
    *,
    allowed_sources: frozenset[str] = DEFAULT_TEMPLATE_SOURCES,
    manifest_path: str | Path | None = None,
    require_formal: bool = False,
) -> TemplateLibraryView:
    """Load a source-scoped view and optionally verify a frozen manifest.

    A formal manifest binds the index hash, allowed sources, every selected PDB
    hash, and the build inputs/script. The latter two are required as non-empty
    mappings so an index assembled from an already-mixed centroid set cannot be
    relabelled as a clean rebuild.
    """
    index = Path(index_path).resolve()
    if not index.is_file():
        raise TemplateLibraryError(f"template index does not exist: {index}")
    root = index.parent
    normalized_sources = frozenset(str(source).lower() for source in allowed_sources)
    if not normalized_sources:
        raise TemplateLibraryError("allowed_sources must not be empty")
    try:
        raw = json.loads(index.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TemplateLibraryError(f"cannot read template index {index}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TemplateLibraryError("template index must be a JSON object")

    selected: dict[str, Mapping[str, Any]] = {}
    for key in sorted(raw):
        entry = raw[key]
        if not isinstance(entry, dict):
            raise TemplateLibraryError(f"template entry {key!r} is not an object")
        source = str(entry.get("source", "")).lower()
        if not source:
            raise TemplateLibraryError(f"template entry {key!r} has no source")
        if source not in normalized_sources:
            continue
        relative = str(entry.get("pdb_path", "")).replace("\\", "/")
        if not relative:
            raise TemplateLibraryError(f"template entry {key!r} has no pdb_path")
        _resolved_child(root, relative)
        copied = dict(entry)
        copied["source"] = source
        copied["pdb_path"] = relative
        selected[str(key)] = MappingProxyType(copied)

    digest = _sha256(index)
    formal = False
    resolved_manifest = Path(manifest_path).resolve() if manifest_path else None
    if resolved_manifest is not None:
        manifest = _load_manifest(resolved_manifest)
        if str(manifest.get("index_sha256", "")).lower() != digest:
            raise TemplateLibraryError("template manifest index SHA-256 mismatch")
        declared_sources = frozenset(
            str(source).lower() for source in manifest.get("allowed_sources", [])
        )
        if declared_sources != normalized_sources:
            raise TemplateLibraryError("template manifest allowed_sources mismatch")
        if int(manifest.get("entry_count", -1)) != len(selected):
            raise TemplateLibraryError("template manifest entry_count mismatch")
        build_inputs = manifest.get("build_inputs")
        build_scripts = manifest.get("build_scripts")
        if not isinstance(build_inputs, dict) or not build_inputs:
            raise TemplateLibraryError("formal template manifest lacks build_inputs")
        if not isinstance(build_scripts, dict) or not build_scripts:
            raise TemplateLibraryError("formal template manifest lacks build_scripts")
        for section_name, artifacts in (
            ("build_inputs", build_inputs),
            ("build_scripts", build_scripts),
        ):
            for relative, expected_sha256 in sorted(artifacts.items()):
                artifact = _resolved_child(root, str(relative))
                if not artifact.is_file():
                    raise TemplateLibraryError(
                        f"template manifest {section_name} file is missing: {relative}"
                    )
                if _sha256(artifact) != str(expected_sha256).lower():
                    raise TemplateLibraryError(
                        f"template manifest {section_name} SHA-256 mismatch: {relative}"
                    )
        files = manifest.get("template_files")
        if not isinstance(files, dict):
            raise TemplateLibraryError("formal template manifest lacks template_files")
        expected_paths = {str(entry["pdb_path"]) for entry in selected.values()}
        if set(files) != expected_paths:
            raise TemplateLibraryError("template manifest file set mismatch")
        for relative in sorted(expected_paths):
            template_path = _resolved_child(root, relative)
            if not template_path.is_file():
                raise TemplateLibraryError(f"template PDB is missing: {relative}")
            if _sha256(template_path) != str(files[relative]).lower():
                raise TemplateLibraryError(f"template PDB SHA-256 mismatch: {relative}")
        formal = True
    elif require_formal:
        raise TemplateLibraryError("formal template use requires a manifest")

    return TemplateLibraryView(
        root=root,
        index_path=index,
        index_sha256=digest,
        allowed_sources=normalized_sources,
        entries=MappingProxyType(selected),
        formal=formal,
        manifest_path=resolved_manifest,
    )
