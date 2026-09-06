"""Conservative mapping from reconstruction labels to one rigor level.

The mapper is deliberately additive: it reads a ReconstructionResult (or its
JSON-ready mapping) and never mutates or augments the result object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class RigorLevel:
    """Two orthogonal evidence dimensions rendered as ``Lx:Y``."""

    recovery_level: str
    provenance: str

    @property
    def label(self) -> str:
        """Return the compact, human-readable rigor label."""
        return f"{self.recovery_level}:{self.provenance}"


_QUALIFIED = "qualified"
_REPAIRED = {"repaired", "repair", "labelled", "labeled"}

# Warning codes that mark a result's chemical identity as diagnostic-only,
# divergent, or integrity-conflicted.  A result carrying any of these
# conditions must not inherit the "recovered" (R) provenance letter merely
# from a high/medium recovery quality: the recovery level (L dimension)
# stays what the quality ladder says, but the provenance letter drops to H.
# Mirrors ``result_first._RIGOR_DOWNGRADE_WARNING_CODES``; kept local so the
# mapper remains import-free and JSON-dict friendly.
_DIAGNOSTIC_INTEGRITY_WARNING_CODES = frozenset({
    "F_H_DIAGNOSTIC_ONLY",
    "F_H_IDENTITY_DIVERGENT",
    "REGISTRY_TEMPLATE_ASSISTED",
    "INFERENCE_ENGINES_DISAGREE",
    "COORDINATE_MAPPING_IDENTITY_DIVERGENT",
    "STRICT_FALLBACK_BLOCKED",
    "DEGRADED_MISSING_SIDECHAIN",
    "DEGRADED_MISSING_BACKBONE",
    "DEGRADED_TRUNCATED_RESIDUE",
    "DEGRADED_ABNORMAL_ATOM_NAME",
    "DEGRADED_ELEMENT_MISMATCH",
    "DEGRADED_UNKNOWN_RESIDUE",
    "DEGRADED_SEQUENCE_GAP",
    "DEGRADED_EXTERNAL_ENDPOINT_UNRESOLVED",
    "DEGRADED_TEMPLATE_PROJECTION",
    "DEGRADED_CANDIDATE_PROMOTION_BLOCKED",
})


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _strict_evidence(result: Any) -> Any:
    strict = _field(result, "strict_result")
    if strict is None:
        # Pipeline rows flatten StrictReconstructionResult with asdict.
        strict = result
    evidence = _field(strict, "output_evidence", {})
    return evidence if isinstance(evidence, Mapping) else {}


def _has_diagnostic_integrity_condition(result: Any) -> bool:
    """Report whether diagnostic/integrity conditions mark the chemistry.

    Recovery quality and chemical rigor are orthogonal: a ``high`` or
    ``medium`` recovery keeps its recovery level, but when the result
    carries integrity findings or diagnostic-only/divergence warning codes,
    the provenance letter is H (heuristic identity evidence), never R.
    """
    findings = _field(result, "integrity_findings")
    if isinstance(findings, (list, tuple)) and any(
        str(item) for item in findings
    ):
        return True
    codes = _field(result, "warning_codes")
    if isinstance(codes, (list, tuple)):
        return any(
            str(code) in _DIAGNOSTIC_INTEGRITY_WARNING_CODES for code in codes
        )
    return False


def _exact_level(result: Any, support: str) -> str:
    """Resolve exact recovery conservatively from strict evidence.

    Strict qualification establishes the L2 contract.  L3 is granted only by
    the explicit strict evidence field; an explicitly weaker L1 result is
    respected rather than silently promoted.
    """
    evidence = _strict_evidence(result)
    qualified_level = evidence.get("highest_evidence_qualified_level")
    if qualified_level == "L3":
        return "L3"
    if qualified_level == "L1":
        return "L1"
    # Missing/legacy evidence is treated as no stereo claim, not as failure.
    return "L2"


def rigor_from_result(result: Any) -> RigorLevel:
    """Map a ReconstructionResult or JSON-ready dict to a RigorLevel.

    The mapping never upgrades evidence.  Unsuccessful or structurally empty
    results use the serializable sentinel ``L0:NONE``.
    """
    if result is None:
        return RigorLevel("L0", "NONE")

    # Accept an application envelope as a convenience while preserving the
    # documented direct-result and JSON-dict inputs.
    if isinstance(result, Mapping):
        nested = result.get("data")
        if result.get("operation") == "reconstruct_result_first" and isinstance(
            nested, Mapping
        ):
            result = nested

    status = str(_field(result, "status", "") or "").strip().lower()
    quality = _field(result, "quality")
    quality = str(quality).strip().lower() if quality is not None else None
    if status != "success" or quality is None:
        return RigorLevel("L0", "NONE")

    if quality == "exact":
        strict = _field(result, "strict_result")
        support = _field(strict, "support_status") if strict is not None else None
        if support is None:
            support = _field(result, "support_status")
        support = str(support or "").strip().lower()
        if support == _QUALIFIED:
            return RigorLevel(_exact_level(result, support), "Q")
        # An exact result without a qualified strict contract is never Q.
        return RigorLevel("L2", "R")

    if quality in {"high", "medium"}:
        if _has_diagnostic_integrity_condition(result):
            # Diagnostic/integrity conditions cannot inherit the recovered
            # (R) provenance merely from high/medium recovery quality.
            return RigorLevel("L2", "H")
        return RigorLevel("L2", "R")
    if quality == "candidate":
        return RigorLevel("L2", "H")
    if quality == "hypothesis":
        return RigorLevel("L1", "H")
    if quality == "topology":
        return RigorLevel("L1", "H")
    if quality == "partial":
        return RigorLevel("L1", "R")
    if quality == "raw":
        return RigorLevel("L0", "C")
    return RigorLevel("L0", "NONE")


__all__ = ["RigorLevel", "rigor_from_result"]
