"""GUI conversion must share the CLI's fail-closed V6 default semantics."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")

from cycpep_master.gui import workers
from cycpep_master.gui.main_window import _PATHS


class _SignalCapture:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


def _worker(path="v6"):
    worker = workers.ConvertWorker("fixture.cif", path, "I", False)
    capture = _SignalCapture()
    worker.done = capture
    return worker, capture


def test_gui_lists_v6_first_and_labels_candidates_diagnostic():
    assert _PATHS[0][0] == "v6"
    assert "fail-closed" in _PATHS[0][1]
    assert all("diagnostic" in label for _key, label in _PATHS[1:])


def test_gui_v6_emits_only_successful_qualified_graph(monkeypatch):
    from cycpep_master import application

    monkeypatch.setattr(
        application,
        "reconstruct_structure",
        lambda *args, **kwargs: {
            "operation": "reconstruct_structure",
            "status": "success",
            "data": {"smiles": "C1CC1", "warning_codes": []},
        },
    )
    worker, capture = _worker()
    worker.run()
    assert capture.values == [("C1CC1", "")]


def test_gui_v6_surfaces_typed_rejection_without_smiles(monkeypatch):
    from cycpep_master import application

    monkeypatch.setattr(
        application,
        "reconstruct_structure",
        lambda *args, **kwargs: {
            "operation": "reconstruct_structure",
            "status": "rejected",
            "data": {"warning_codes": ["V6_TOPOLOGY_CONFLICT"]},
            "error": "rejected: explicit connections conflict",
        },
    )
    worker, capture = _worker()
    worker.run()
    assert capture.values == [(
        "",
        "rejected: explicit connections conflict [V6_TOPOLOGY_CONFLICT]",
    )]


def test_gui_unknown_diagnostic_path_does_not_fallback_to_a():
    worker, capture = _worker("unknown")
    worker.run()
    assert capture.values[0][0] == ""
    assert "unknown diagnostic path" in capture.values[0][1]


def test_gui_diagnostic_worker_uses_application_service(monkeypatch):
    from cycpep_master import application

    observed = {}

    def fake(inputs, **kwargs):
        observed.update(inputs=inputs, **kwargs)
        return {
            "operation": "reconstruct",
            "status": "success",
            "data": {"results": [{"smiles": "CC"}]},
        }

    monkeypatch.setattr(application, "reconstruct_coordinates", fake)
    worker, capture = _worker("a")
    worker.run()

    assert capture.values == [("CC", "")]
    assert observed["inputs"] == ["fixture.cif"]
    assert observed["path"] == "a"
    assert observed["chain_id"] == "I"


def test_admet_worker_uses_application_service(monkeypatch):
    from cycpep_master import application

    monkeypatch.setattr(
        application,
        "predict_admet",
        lambda values: {
            "operation": "admet",
            "status": "success",
            "data": {
                "predictions": [
                    {"input_smiles": values[0], "QED": 0.5}
                ]
            },
        },
    )
    worker = workers.AdmetWorker("CC")
    capture = _SignalCapture()
    worker.done = capture
    worker.run()

    assert capture.values == [(
        {"input_smiles": "CC", "QED": 0.5},
        "",
    )]
