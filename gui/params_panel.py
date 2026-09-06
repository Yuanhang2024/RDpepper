"""Geometry-parameter panel for the reconstruction workspace.

Renders the two tunable covalent-geometry parameters shipped by
:mod:`cycpep_master.core.geometry_params` (covalent radius multiplier and
distance ceiling) as drop-in rows with per-row default hints and reset
buttons, plus a global "restore all defaults" action.
"""
from __future__ import annotations

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

try:  # 后端提供的单一默认值源（另一个代理正在落盘）
    from ..core.geometry_params import (
        DEFAULT_DISTANCE_CEILING,
        DEFAULT_RADIUS_MULTIPLIER,
        NONDEFAULT_GEOMETRY_PARAMS,
    )
except ImportError:
    # 后端模块缺失时的临时兜底
    DEFAULT_DISTANCE_CEILING = 3.0
    DEFAULT_RADIUS_MULTIPLIER = 1.3
    NONDEFAULT_GEOMETRY_PARAMS = frozenset(
        {"radius_multiplier", "distance_ceiling"}
    )

#: keyword 参数名（按后端 NONDEFAULT_GEOMETRY_PARAMS 单一枚举）
GEOMETRY_PARAM_FIELDS = tuple(
    name for name in NONDEFAULT_GEOMETRY_PARAMS
    if name in ("radius_multiplier", "distance_ceiling")
)

_NONDEFAULT_STYLE = "color: #c9650a; font-weight: bold;"


class ParamRow(QtWidgets.QWidget):
    """One row: name label, spin box, default hint, single-row reset."""

    valueChanged = pyqtSignal()

    def __init__(
        self,
        label: str,
        default: float,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._label = label
        self._default = float(default)

        self.name_label = QtWidgets.QLabel(label)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(self._default)
        self.default_label = QtWidgets.QLabel(f"默认: {self._default}")
        self.reset_button = QtWidgets.QPushButton("↺")
        self.reset_button.setToolTip(f"恢复默认值 {self._default}")
        self.reset_button.setFixedWidth(32)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.name_label)
        layout.addWidget(self.spin)
        layout.addSpacing(6)
        layout.addWidget(self.default_label)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)

        self.spin.valueChanged.connect(self.valueChanged)
        self.spin.valueChanged.connect(self._update_marker)
        self.reset_button.clicked.connect(self.set_to_default)
        self._update_marker()

    def value(self) -> float:
        return self.spin.value()

    def set_to_default(self):
        self.spin.setValue(self._default)
        self._update_marker()

    def is_default(self) -> bool:
        return abs(self.value() - self._default) < 1e-9

    def _update_marker(self):
        if self.is_default():
            self.name_label.setText(self._label)
            self.name_label.setStyleSheet("")
        else:
            self.name_label.setText(f"* {self._label}")
            self.name_label.setStyleSheet(_NONDEFAULT_STYLE)


class GeometryParamsPanel(QtWidgets.QWidget):
    """Editors for the covalent-geometry tolerances used at reconstruction."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.radius_row = ParamRow(
            "共价半径倍数",
            DEFAULT_RADIUS_MULTIPLIER,
            minimum=0.5,
            maximum=2.0,
            step=0.05,
            decimals=2,
        )
        self.distance_row = ParamRow(
            "距离上限 (Å)",
            DEFAULT_DISTANCE_CEILING,
            minimum=1.5,
            maximum=6.0,
            step=0.1,
            decimals=1,
        )
        self.rows = (self.radius_row, self.distance_row)

        self.reset_all_button = QtWidgets.QPushButton("全部恢复默认")
        self.reset_all_button.clicked.connect(self.reset_all)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.radius_row)
        layout.addWidget(self.distance_row)
        layout.addSpacing(4)
        layout.addWidget(self.reset_all_button)
        layout.addStretch(1)

    def values(self) -> dict:
        """Current geometry parameter values keyed by field name."""
        return {
            "radius_multiplier": self.radius_row.value(),
            "distance_ceiling": self.distance_row.value(),
        }

    def reset_all(self):
        for row in self.rows:
            row.set_to_default()