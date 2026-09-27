"""Machine-readable applicability routing for V5 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "applicability_manifest.json"
)


def load_applicability_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != (
        "1.0.0-cycpep-v5-applicability.1"
    ):
        raise ValueError("unsupported applicability manifest schema")
    return manifest


def evaluate_sequence_applicability(
    *,
    residue_count: int,
    topology_class: str,
    contains_nnaa: bool,
) -> dict[str, Any]:
    manifest = load_applicability_manifest()
    selected = None
    for band in manifest["sequence_length_bands"]:
        maximum = band["maximum"]
        if residue_count >= int(band["minimum"]) and (
            maximum is None or residue_count <= int(maximum)
        ):
            selected = dict(band)
            break
    if selected is None:
        raise ValueError("residue_count is outside applicability manifest")
    topology_status = manifest["topology_classes"].get(
        topology_class, "out_of_domain"
    )
    warnings = []
    if selected["status"] != "supported":
        warnings.append(
            f"SEQUENCE_LENGTH_{selected['status'].upper()}"
        )
    if topology_status != "supported":
        warnings.append(
            f"TOPOLOGY_{str(topology_status).upper()}"
        )
    if contains_nnaa:
        warnings.append("NNAA_REGISTRY_DOMAIN")
    return {
        "length_band": selected,
        "topology_status": topology_status,
        "contains_nnaa": bool(contains_nnaa),
        "warnings": warnings,
        "hard_reject": False,
        "manifest_path": str(MANIFEST_PATH),
    }


__all__ = [
    "MANIFEST_PATH",
    "evaluate_sequence_applicability",
    "load_applicability_manifest",
]
