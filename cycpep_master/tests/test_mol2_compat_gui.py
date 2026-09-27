"""GUI coverage for the Read MOL2 charge-aware compatibility group.

The ``read_mol2`` application service itself is implemented separately; these
tests monkeypatch it (``raising=False`` until integration) and verify the
window wiring: button dispatch with the current selections, native default,
empty optional paths becoming ``None``, and report/error display through the
shared ServiceWorker/result panel.
"""
from __future__ import annotations

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from cycpep_master import application  # noqa: E402
from cycpep_master.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qt_app, monkeypatch):
    monkeypatch.setattr(MainWindow, "_refresh_capabilities", lambda self: None)
    monkeypatch.setattr(
        application,
        "read_mol2",
        lambda *_args, **_kwargs: {
            "operation": "read_mol2",
            "status": "success",
            "data": {},
        },
        raising=False,
    )
    instance = MainWindow()
    yield instance
    for worker in list(instance._workers):
        worker.wait(5000)
    instance.close()
    qt_app.processEvents()


def test_read_mol2_controls_exist_and_default_to_native(window):
    assert window.mol2_read_compatibility.currentData() == "rdkit_native"
    charge_aware_label = window.mol2_read_compatibility.itemText(1)
    assert "UNITY formal charges" in charge_aware_label
    assert "not protonation" in charge_aware_label
    # The group extends the existing utilities page; no new docking sub-tab.
    assert window.workspace_tabs.widget(3).count() == 9


def test_read_mol2_button_dispatches_service_with_selections(
    window, tmp_path, monkeypatch
):
    calls = []

    def fake_read_mol2(
        mol2_path, *, compatibility, receipt_path, export_sdf
    ):
        calls.append((mol2_path, compatibility, receipt_path, export_sdf))
        return {
            "operation": "read_mol2",
            "status": "success",
            "data": {},
        }

    monkeypatch.setattr(
        application, "read_mol2", fake_read_mol2, raising=False
    )
    window._run_service = (
        lambda function, *args, **kwargs: calls.append((function, args, kwargs))
    )
    source = tmp_path / "input.mol2"
    source.write_text("@<TRIPOS>MOLECULE\nligand\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    sdf = tmp_path / "artifact.sdf"
    window.mol2_read_input.setText(str(source))
    window.mol2_read_receipt.setText(str(receipt))
    window.mol2_read_export_sdf.setText(str(sdf))
    window.mol2_read_compatibility.setCurrentIndex(1)

    window.mol2_read_button.click()

    assert calls
    function, args, kwargs = calls[0]
    assert function is application.read_mol2
    assert args == (str(source),)
    assert kwargs == {
        "compatibility": "rdkit_charge_aware",
        "receipt_path": str(receipt),
        "export_sdf": str(sdf),
    }


def test_read_mol2_button_normalizes_empty_optional_paths_to_none(
    window, tmp_path
):
    calls = []
    window._run_service = (
        lambda function, *args, **kwargs: calls.append((function, args, kwargs))
    )
    source = tmp_path / "input.mol2"
    source.write_text("@<TRIPOS>MOLECULE\nligand\n", encoding="utf-8")
    window.mol2_read_input.setText(str(source))

    window.mol2_read_button.click()

    assert calls
    _function, args, kwargs = calls[0]
    assert args == (str(source),)
    assert kwargs == {
        "compatibility": "rdkit_native",
        "receipt_path": None,
        "export_sdf": None,
    }


def test_read_mol2_worker_displays_report_then_error(
    window, qt_app, tmp_path, monkeypatch
):
    report = {
        "operation": "read_mol2",
        "status": "success",
        "data": {
            "reader_mode": "rdkit_native",
            "applied_formal_charges": {"4": 1},
            "atom_counts": {"atoms": 31, "heavy_atoms": 17},
            "total_formal_charge": 0,
            "canonical_smiles": "C1CC1",
            "full_inchikey": "C1CC1-INCHIKEY",
            "warnings": [],
            "receipt_status": "not_requested",
        },
    }
    attempts = []

    def flaky_read_mol2(mol2_path, *, compatibility, receipt_path, export_sdf):
        attempts.append((mol2_path, compatibility, receipt_path, export_sdf))
        if len(attempts) == 1:
            return report
        raise RuntimeError("unsupported charge model")

    monkeypatch.setattr(
        application, "read_mol2", flaky_read_mol2, raising=False
    )
    source = tmp_path / "input.mol2"
    source.write_text("@<TRIPOS>MOLECULE\nligand\n", encoding="utf-8")
    window.mol2_read_input.setText(str(source))

    def run_and_collect():
        window.mol2_read_button.click()
        deadline = time.monotonic() + 5
        while window._workers and time.monotonic() < deadline:
            qt_app.processEvents()
        qt_app.processEvents()
        return json.loads(window.result_json.toPlainText())

    payload = run_and_collect()
    assert attempts[0] == (str(source), "rdkit_native", None, None)
    assert payload["status"] == "success"
    assert payload["data"]["reader_mode"] == "rdkit_native"
    assert window.result_status.text() == "success"
    assert window.result_title.text() == "Read Mol2"

    payload = run_and_collect()
    assert payload["status"] == "failed"
    assert "unsupported charge model" in payload["error"]
    assert window.result_status.text() == "failed"
