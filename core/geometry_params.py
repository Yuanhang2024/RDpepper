"""Single source of truth for the geometric distance-tolerance parameters.

These constants tune covalent-bond distance judgement in cyclization
detection (``core/cyclization.py``).  All public entry points that expose
them default to ``None`` (resolving to the defaults below), so behaviour is
identical to before unless a caller explicitly opts in.
"""
import math
from typing import Tuple

DEFAULT_RADIUS_MULTIPLIER = 1.3   # 键判定: d <= (r1+r2)*multiplier
DEFAULT_DISTANCE_CEILING = 3.0    # 硬距离上限 (Å)
# 可被用户改动的两个几何参数名枚举：既供警告标记用（append 到 warnings 列表），
# 也作为非默认参数的单一枚举源（gui/params_panel 依赖它列出字段）。
NONDEFAULT_GEOMETRY_PARAMS = frozenset(
    {"radius_multiplier", "distance_ceiling"}
)


def resolve_geometry_params(
    radius_multiplier=None, distance_ceiling=None
) -> Tuple[float, float]:
    """None→默认值；非 None 校验为正有限 float，否则 ValueError（消息说明哪个参数非法）。"""
    if radius_multiplier is None:
        radius_multiplier = DEFAULT_RADIUS_MULTIPLIER
    else:
        try:
            radius_multiplier = float(radius_multiplier)
        except (TypeError, ValueError):
            raise ValueError(
                "radius_multiplier must be a finite positive float, "
                f"got {radius_multiplier!r}"
            ) from None
        if not (radius_multiplier > 0.0) or not math.isfinite(radius_multiplier):
            raise ValueError(
                "radius_multiplier must be a finite positive float, "
                f"got {radius_multiplier!r}"
            )
    if distance_ceiling is None:
        distance_ceiling = DEFAULT_DISTANCE_CEILING
    else:
        try:
            distance_ceiling = float(distance_ceiling)
        except (TypeError, ValueError):
            raise ValueError(
                "distance_ceiling must be a finite positive float, "
                f"got {distance_ceiling!r}"
            ) from None
        if not (distance_ceiling > 0.0) or not math.isfinite(distance_ceiling):
            raise ValueError(
                "distance_ceiling must be a finite positive float, "
                f"got {distance_ceiling!r}"
            )
    return radius_multiplier, distance_ceiling


def is_nondefault_geometry(radius_multiplier, distance_ceiling) -> bool:
    """与默认值比较（abs diff > 1e-9 视为不同）。"""
    resolved_radius_multiplier, resolved_distance_ceiling = resolve_geometry_params(
        radius_multiplier, distance_ceiling
    )
    return (
        abs(resolved_radius_multiplier - DEFAULT_RADIUS_MULTIPLIER) > 1e-9
        or abs(resolved_distance_ceiling - DEFAULT_DISTANCE_CEILING) > 1e-9
    )