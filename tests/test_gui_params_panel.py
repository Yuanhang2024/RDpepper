"""Tests for the geometry-parameter panel and its main-window wiring.

Runs headless via the `minimal` QPA plugin so the suite never needs a
display. The backend :mod:`cycpep_master.core.geometry_params` may not be
landed yet; assertions use the panel's own default constants, which fall
back to the documented temporary values (1.3 / 3.0) in that case.
"""
from __future__ import annotations

import os

# 必须在 import PyQt5 之前设置，否则无显示环境下 QApplication 构造失败
os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets

from cycpep_master.gui.params_panel import (
    DEFAULT_DISTANCE_CEILING,
    DEFAULT_RADIUS_MULTIPLIER,
    GEOMETRY_PARAM_FIELDS,
    GeometryParamsPanel,
    ParamRow,
)


@pytest.fixture(scope="module")
def qt_app():
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception as exc:  # pragma: no cover - only on exotic platforms
        pytest.skip(f"Qt unavailable ({exc})")
    yield app


@pytest.fixture
def panel(qt_app):
    instance = GeometryParamsPanel()
    yield instance
    instance.setParent(None)
    instance.deleteLater()
    qt_app.processEvents()


def test_default_values(qt_app, panel):
    assert panel.values() == {
        "radius_multiplier": DEFAULT_RADIUS_MULTIPLIER,
        "distance_ceiling": DEFAULT_DISTANCE_CEILING,
    }


def test_geometry_fields_cover_both_params(qt_app, panel):
    assert set(GEOMETRY_PARAM_FIELDS) == {"radius_multiplier", "distance_ceiling"}


def test_marker_appears_and_single_row_reset(qt_app, panel):
    row = panel.radius_row
    assert row.is_default()
    assert "*" not in row.name_label.text()

    row.spin.setValue(1.45)
    qt_app.processEvents()
    assert not row.is_default()
    assert row.name_label.text().startswith("*")
    assert row.name_label.styleSheet()

    row.reset_button.click()
    qt_app.processEvents()
    assert row.is_default()
    assert "*" not in row.name_label.text()
    # 另一行未改动，保持默认
    assert panel.distance_row.is_default()


def test_value_changed_signal(qt_app, panel):
    emissions = []
    panel.radius_row.valueChanged.connect(lambda: emissions.append(True))
    panel.radius_row.spin.setValue(1.7)
    panel.radius_row.reset_button.click()
    assert len(emissions) >= 2


def test_reset_all(qt_app, panel):
    panel.radius_row.spin.setValue(1.55)
    panel.distance_row.spin.setValue(4.2)
    qt_app.processEvents()
    assert not panel.radius_row.is_default()
    assert not panel.distance_row.is_default()

    panel.reset_all()
    qt_app.processEvents()
    assert panel.radius_row.is_default()
    assert panel.distance_row.is_default()
    assert panel.values() == {
        "radius_multiplier": DEFAULT_RADIUS_MULTIPLIER,
        "distance_ceiling": DEFAULT_DISTANCE_CEILING,
    }


def test_param_row_value_boundaries(qt_app):
    row = ParamRow("探针", 1.3, 0.5, 2.0, 0.05, 2)
    row.spin.setValue(0.5)
    assert row.value() == 0.5
    row.spin.setValue(2.0)
    assert row.value() == 2.0
