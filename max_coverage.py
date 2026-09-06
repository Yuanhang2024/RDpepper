"""Max-coverage fallback primitives (V7 proposal, reference implementation).

Every function here is pure and side-effect free so it can be unit-tested and
bolted onto the existing pipeline without touching V6.0.0 behavior. The
package default remains ``strict_v6``; callers opt in per operation with
``fallback_policy="max_coverage"`` (see V7_MAX_COVERAGE_FALLBACK_DESIGN.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

STRICT_V6 = "strict_v6"
MAX_COVERAGE = "max_coverage"
_POLICIES = (STRICT_V6, MAX_COVERAGE)


def coerce_policy(value: str | None) -> str:
    if value is None:
        return STRICT_V6
    if value not in _POLICIES:
        raise ValueError(f"unknown fallback policy {value!r}; expected one of {_POLICIES}")
    return value


# --------------------------------------------------------------------------
# Class A: data-content input failures -> diagnosis instead of bare reject
# --------------------------------------------------------------------------

@dataclass
class InputDiagnosis:
    status: str = "degraded_input"
    error_kind: str = ""
    message: str = ""
    salvageable_prefix: str = ""
    failure_position: int | None = None


def diagnose_input(payload: str, *, parser: Callable[[str], object]) -> InputDiagnosis:
    """Best-effort diagnosis for unparseable data content.

    ``parser`` must raise ``ValueError`` for content failures. On failure the
    longest prefix that parses standalone is reported so callers can retain
    the salvageable fragment. Purely parameter-level errors (empty payload,
    wrong type) stay hard failures - they are caller bugs, not content.
    """
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("payload must be a nonempty string")
    try:
        parser(payload)
        return InputDiagnosis(error_kind="none", message="parsed", salvageable_prefix=payload)
    except ValueError as exc:
        message = str(exc)
        prefix = ""
        position = None
        for cut in range(len(payload), 0, -1):
            candidate = payload[:cut].rstrip(" -|,\n")
            if not candidate:
                continue
            try:
                parser(candidate)
            except ValueError:
                continue
            prefix = candidate
            position = cut
            break
        return InputDiagnosis(
            error_kind="content_parse_error",
            message=message,
            salvageable_prefix=prefix,
            failure_position=position,
        )


# --------------------------------------------------------------------------
# Class B: notation hard-validation -> placeholder salvage
# --------------------------------------------------------------------------

@dataclass
class NotationSalvage:
    status: str = "notation_partial"
    tokens: list[str] = field(default_factory=list)
    placeholders: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def salvaged_monomer_count(self) -> int:
        return sum(1 for t in self.tokens if not t.startswith("\u0000"))


def salvage_notation_prefix(
    notation: str,
    *,
    tokenizer: Callable[[str], Sequence[str]],
    parse_token: Callable[[str], object],
    placeholder: str = "\u0000UNK",
) -> NotationSalvage:
    """Replace failing tokens with placeholders and keep parsing.

    The caller supplies the notation-specific tokenization and per-token
    validator, so MAP/BILN/HELM all route through one salvage routine. The
    result feeds a C1:H symbolic partial whose content now includes the
    salvaged prefix, failing positions, and raw tokens.
    """
    result = NotationSalvage()
    for index, token in enumerate(tokenizer(notation)):
        try:
            parse_token(token)
            result.tokens.append(token)
        except ValueError as exc:
            result.tokens.append(placeholder)
            result.placeholders.append({
                "position": index,
                "raw_token": token,
                "error": str(exc),
            })
            result.errors.append(f"token {index} ({token!r}): {exc}")
    if not result.placeholders:
        result.status = "notation_complete"
    return result


# --------------------------------------------------------------------------
# Class D: MOL2 coordinate-mapping hard reject -> X-tiered output
# --------------------------------------------------------------------------

@dataclass
class CoordinateTier:
    tier: str            # "X3" | "X2" | "X1"
    mapped_atom_count: int
    generated_atom_count: int
    generated_atom_indices: list[int] = field(default_factory=list)
    max_displacement_angstrom: float = 0.0


def apply_coordinate_tier(
    conformer,
    atom_count: int,
    source_coordinates: dict[int, tuple[float, float, float]] | None,
    *,
    apply: Callable[[int, int, float, float, float], None] | None = None,
) -> CoordinateTier:
    """Decide and apply the X tier for one embedded conformer.

    ``source_coordinates`` maps atom index -> (x, y, z). When it covers every
    heavy atom the conformer is overwritten with source coordinates (X3, zero
    displacement by construction). Partial coverage keeps embedded coordinates
    for the remainder (X2). No coverage keeps everything embedded (X1). The
    MOL2/PDBQT writer then runs unconditionally and copies this receipt into
    its audit trail.
    """
    mapping = source_coordinates or {}
    covered = {i for i in mapping if 0 <= i < atom_count}
    apply = apply or (lambda i, j, x, y, z: conformer.SetAtomPosition(i, (x, y, z)))
    if atom_count and len(covered) == atom_count:
        for i, (x, y, z) in mapping.items():
            apply(i, 0, x, y, z)
        return CoordinateTier("X3", mapped_atom_count=atom_count,
                              generated_atom_count=0)
    generated = [i for i in range(atom_count) if i not in covered]
    if covered:
        for i in covered:
            x, y, z = mapping[i]
            apply(i, 0, x, y, z)
        return CoordinateTier("X2", mapped_atom_count=len(covered),
                              generated_atom_count=len(generated),
                              generated_atom_indices=generated)
    return CoordinateTier("X1", mapped_atom_count=0,
                          generated_atom_count=atom_count,
                          generated_atom_indices=list(range(atom_count)))


# --------------------------------------------------------------------------
# Class E: fail-closed states -> attach deepest salvage payload
# --------------------------------------------------------------------------

def salvage_payload(state: dict, *, partial: dict | None) -> dict:
    """Attach the deepest parsed fragment to a fail-closed state."""
    if partial:
        state = dict(state)
        state["salvage"] = partial
    return state
