"""Opt-in geometry-only graph inference; existing reconstruction is unchanged."""

from __future__ import annotations

from collections.abc import Mapping
import math


def infer_monomer_geometry(
    record: Mapping,
    *,
    mode: str = "simple",
    timeout_seconds: float = 1.0,
) -> dict:
    """Infer candidate chemistry without residue names or library lookup.

    Atoms carry zero-based ids, elements and Angstrom coordinates. Optional
    bonds specify adjacency, not bond orders. Candidates are experimental
    hypotheses; successful execution is not independent chemistry validation.
    """
    if mode not in {"simple", "complex"}:
        raise ValueError("mode must be 'simple' or 'complex'")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")
    if mode == "simple":
        from .geometry_simple import infer_geometry
    else:
        from .geometry_complex import infer_geometry
    payload = {
        "atoms": record.get("atoms"),
        "bonds": record.get("bonds"),
        "total_charge": record.get("total_charge"),
    }
    return infer_geometry(payload, timeout_seconds=float(timeout_seconds))
