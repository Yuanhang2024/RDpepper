"""Public deterministic cyclic-peptide notation conversion API.

HELM, MAP, and BILN serialize the same monomer/connection graph. Conversion
to SMILES assembles that graph with the active audited monomer registry.
General SMILES-to-monomer decomposition is intentionally not exposed here.
"""
from __future__ import annotations

from typing import Any, Mapping

from .paths._map_utils import (
    biln_to_helm as _legacy_biln_to_helm,
    get_smi_from_biln,
    get_smi_from_map,
    helm_to_biln as _legacy_helm_to_biln,
    helm_to_map as _legacy_helm_to_map,
    map_to_helm as _legacy_map_to_helm,
)
from .exact_v1 import (
    biln_to_exact_v1,
    chemical_graph_equivalent,
    edge_v1_to_exact_v1,
    exact_v1_equivalent,
    exact_v1_from_v6_result,
    exact_v1_to_biln,
    exact_v1_to_edge_v1,
    exact_v1_to_helm,
    exact_v1_to_json,
    exact_v1_to_legacy_v5,
    exact_v1_to_map,
    exact_v1_to_smiles,
    helm_to_exact_v1,
    legacy_v5_to_exact_v1,
    map_to_exact_v1,
    model_projection_equivalent,
)


def _require_exact(kind: str, payload: str):
    document = {
        "map": map_to_exact_v1,
        "helm": helm_to_exact_v1,
        "biln": biln_to_exact_v1,
    }[kind](payload)
    if document["exactness_status"] != "EXACT":
        raise ValueError(
            "exact_v1 abstained: "
            + ", ".join(document["reason_codes"])
        )
    return document


def _resolution_scope(
    monomer_context: Mapping[str, Any] | None,
    *,
    kind: str,
    payload: Any,
):
    from .core.monomer_resolution import (
        monomer_resolution_context,
        monomer_symbol_hints,
    )

    return monomer_resolution_context(
        monomer_context,
        required_symbols=monomer_symbol_hints(payload, kind=kind),
    )


def map_to_helm(
    map_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Serialize validated MAP as HELM through the exact_v1 gate.

    The historical serializer is retained to preserve source ordering and
    byte-compatible public output.  Canonical output is available through
    :func:`exact_v1_to_helm`.
    """
    with _resolution_scope(monomer_context, kind="map", payload=map_text):
        _require_exact("map", map_text)
        return _legacy_map_to_helm(map_text)


def helm_to_map(
    helm_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Serialize validated HELM as MAP through the exact_v1 gate."""
    with _resolution_scope(monomer_context, kind="helm", payload=helm_text):
        _require_exact("helm", helm_text)
        return _legacy_helm_to_map(helm_text)


def biln_to_helm(
    biln_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Serialize validated BILN as HELM through the exact_v1 gate."""
    with _resolution_scope(monomer_context, kind="biln", payload=biln_text):
        _require_exact("biln", biln_text)
        return _legacy_biln_to_helm(biln_text)


def helm_to_biln(
    helm_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Serialize validated HELM as BILN through the exact_v1 gate."""
    with _resolution_scope(monomer_context, kind="helm", payload=helm_text):
        _require_exact("helm", helm_text)
        return _legacy_helm_to_biln(helm_text)


def map_to_biln(
    map_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Serialize one validated MAP monomer graph as BILN."""
    with _resolution_scope(monomer_context, kind="map", payload=map_text):
        return helm_to_biln(map_to_helm(map_text))


def biln_to_map(
    biln_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Serialize one validated BILN monomer graph as MAP."""
    with _resolution_scope(monomer_context, kind="biln", payload=biln_text):
        return helm_to_map(biln_to_helm(biln_text))


def map_to_smiles(
    map_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Assemble validated MAP into an isomeric molecular SMILES."""
    with _resolution_scope(monomer_context, kind="map", payload=map_text):
        return exact_v1_to_smiles(_require_exact("map", map_text))


def helm_to_smiles(
    helm_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Assemble validated HELM into an isomeric molecular SMILES."""
    with _resolution_scope(monomer_context, kind="helm", payload=helm_text):
        return exact_v1_to_smiles(_require_exact("helm", helm_text))


def biln_to_smiles(
    biln_text: str,
    *,
    monomer_context: Mapping[str, Any] | None = None,
) -> str:
    """Assemble validated BILN into an isomeric molecular SMILES."""
    with _resolution_scope(monomer_context, kind="biln", payload=biln_text):
        return exact_v1_to_smiles(_require_exact("biln", biln_text))


__all__ = [
    "biln_to_helm",
    "biln_to_exact_v1",
    "biln_to_map",
    "biln_to_smiles",
    "chemical_graph_equivalent",
    "edge_v1_to_exact_v1",
    "exact_v1_equivalent",
    "exact_v1_from_v6_result",
    "exact_v1_to_biln",
    "exact_v1_to_edge_v1",
    "exact_v1_to_helm",
    "exact_v1_to_json",
    "exact_v1_to_legacy_v5",
    "exact_v1_to_map",
    "exact_v1_to_smiles",
    "helm_to_biln",
    "helm_to_exact_v1",
    "helm_to_map",
    "helm_to_smiles",
    "legacy_v5_to_exact_v1",
    "map_to_biln",
    "map_to_exact_v1",
    "map_to_helm",
    "map_to_smiles",
    "model_projection_equivalent",
]
