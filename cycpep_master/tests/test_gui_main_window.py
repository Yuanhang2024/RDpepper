from __future__ import annotations

import json
import threading
import time

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtGui, QtWidgets

from cycpep_master import application
from cycpep_master.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qt_app, monkeypatch):
    monkeypatch.setattr(MainWindow, "_refresh_capabilities", lambda self: None)
    instance = MainWindow()
    yield instance
    for worker in list(instance._workers):
        worker.wait(5000)
    instance.close()
    qt_app.processEvents()


def test_gui_has_controls_for_all_extended_services(window):
    assert window.windowTitle() == "RDpepper"
    expected = (
        "batch_export_manifest",
        "template_map",
        "template_output",
        "sequence_input",
        "sequence_output_dir",
        "sequence_cyclization",
        "sequence_conformers",
        "rec_monomer_context",
        "ligand_pdb_input",
        "prepared_vina_ligand",
        "batch_dock_directory",
    )
    assert all(hasattr(window, name) for name in expected)
    assert window.workspace_tabs.widget(2).count() == 5
    assert window.workspace_tabs.widget(3).count() == 9


def test_gui_marks_degraded_format_success_as_partial(window):
    window._show_result({
        "operation": "export",
        "status": "success",
        "data": {
            "requested_format_status": "metadata_only",
            "artifacts": [{"format": "metadata"}],
        },
        "error": None,
    })

    assert window.result_status.text() == "partial"


def test_export_gui_defaults_to_max_coverage_policy(window, tmp_path):
    calls = []
    window._run_service = (
        lambda function, *args, **kwargs: calls.append(
            (function, args, kwargs)
        )
    )
    window.export_smiles.setPlainText("CC")
    window.export_output.setText(str(tmp_path / "ligand.mol2"))

    window._run_export()

    assert calls
    assert calls[0][0] is application.export_structure
    assert calls[0][2]["fallback_policy"] == "max_coverage"


def test_reconstruction_gui_forwards_monomer_context(
    window, tmp_path
):
    calls = []
    window._run_service = (
        lambda function, *args, **kwargs: calls.append(
            (function, args, kwargs)
        )
    )
    source = tmp_path / "custom.pdb"
    source.write_text("END\n", encoding="ascii")
    window.rec_file.setText(str(source))
    window.rec_monomer_context.setText(
        '{"component_ids":{"XAA":"XAA"},"allow_network":false}'
    )

    window._run_reconstruction()

    assert calls
    assert calls[0][2]["monomer_context"] == {
        "component_ids": {"XAA": "XAA"},
        "allow_network": False,
    }


def test_sequence_gui_forwards_monomer_context(window, tmp_path):
    calls = []
    window._run_service = (
        lambda function, *args, **kwargs: calls.append(
            (function, args, kwargs)
        )
    )
    window.sequence_input.setPlainText("[GUI_AA]AC")
    window.sequence_output_dir.setText(str(tmp_path / "sequence"))
    window.sequence_monomer_context.setText(
        '{"definitions":[{"symbol":"GUI_AA",'
        '"smiles":"N[C@@H](CCl)C(=O)O"}]}'
    )
    window.sequence_generate_pdbqt.setChecked(False)

    window._run_prepare_sequence()

    assert calls
    assert calls[0][2]["monomer_context"]["definitions"][0][
        "symbol"
    ] == "GUI_AA"


def test_gui_extended_actions_dispatch_to_application_services(
    window, tmp_path, monkeypatch
):
    calls = []
    window._run_service = (
        lambda function, *args, **kwargs: calls.append((function, args, kwargs))
    )

    manifest = tmp_path / "smiles.json"
    manifest.write_text(json.dumps({"one": "C"}), encoding="utf-8")
    window.batch_export_manifest.setText(str(manifest))
    window.batch_export_output_dir.setText(str(tmp_path / "exports"))
    window._run_batch_export()

    window.template_map.setPlainText("PEPTIDE1{A.C}$$$$")
    window.template_smiles.setPlainText("CC")
    window.template_output.setText(str(tmp_path / "templates.sdf"))
    window._run_template_lookup()
    window._run_template_conformers()

    window.sequence_input.setPlainText("ACDEFG")
    window.sequence_output_dir.setText(str(tmp_path / "sequence"))
    window.sequence_stereochemistry.setText('{"2":"D"}')
    window.sequence_terminal_modifications.setText('{"N":"ACE"}')
    window.sequence_conformers.setValue(3)
    window.sequence_generate_pdbqt.setChecked(False)
    window._run_prepare_sequence()

    window.ligand_pdb_input.setText(str(tmp_path / "ligand.pdb"))
    window.ligand_pdb_output.setText(str(tmp_path / "ligand.pdbqt"))
    window._run_ligand_pdbqt_from_pdb()

    window.prepared_vina_ligand.setText(str(tmp_path / "ligand.pdbqt"))
    window.prepared_vina_receptor.setText(str(tmp_path / "receptor.pdbqt"))
    window.prepared_vina_output.setText(str(tmp_path / "out.pdbqt"))
    window._run_prepared_vina()

    monkeypatch.setattr(
        application,
        "discover_docking_coordinate_files",
        lambda _directory: [str(tmp_path / "peptide.pdb")],
    )
    window.batch_dock_directory.setText(str(tmp_path))
    window.batch_dock_receptor.setText(str(tmp_path / "receptor.pdb"))
    window._run_batch_docking()

    assert [call[0] for call in calls] == [
        application.batch_export_structures,
        application.find_conformer_template,
        application.generate_template_conformers,
        application.prepare_ligand_from_sequence,
        application.prepare_ligand_pdbqt_from_pdb,
        application.run_prepared_vina,
        application.batch_dock_structures,
    ]
    assert calls[0][1][0] == [("one", "C")]
    sequence_call = calls[3]
    assert sequence_call[1] == (
        "ACDEFG",
        str(tmp_path / "sequence"),
    )
    assert sequence_call[2]["stereochemistry"] == {"2": "D"}
    assert sequence_call[2]["terminal_modifications"] == {"N": "ACE"}
    assert sequence_call[2]["conformer_count"] == 3
    assert sequence_call[2]["generate_pdbqt"] is False


def test_gui_does_not_expose_retired_direct_pdbqt_ensemble(window):
    assert not hasattr(window, "ligand_ensemble_smiles")
    assert not hasattr(window, "_run_ligand_pdbqt_ensemble")


def test_gui_forwards_torsion_controls(window):
    calls = []
    window._run_service = (
        lambda function, *args, **kwargs: calls.append((function, args, kwargs))
    )
    window.ligand_torsion_ensemble_size.setValue(11)
    window.ligand_torsion_threads.setValue(3)
    window._run_ligand_pdbqt()
    window.dock_center_mode.setCurrentIndex(2)
    window.dock_torsdof.setValue(14)
    window.dock_torsion_ensemble_size.setValue(12)
    window.dock_torsion_threads.setValue(4)
    window._run_docking()

    assert calls[0][2]["torsion_ensemble_size"] == 11
    assert calls[0][2]["torsion_num_threads"] == 3
    assert calls[1][2]["torsdof_limit"] == 14
    assert calls[1][2]["torsion_ensemble_size"] == 12
    assert calls[1][2]["torsion_num_threads"] == 4


def test_gui_pdbqt_validation_accepts_file_or_text(window, tmp_path):
    calls = []
    window._run_service = (
        lambda function, *args, **kwargs: calls.append((function, args, kwargs))
    )

    window.pdbqt_validate_payload.setPlainText("ROOT\nENDROOT\nTORSDOF 0")
    window._run_pdbqt_validation()
    source = tmp_path / "ligand.pdbqt"
    window.pdbqt_validate_input.setText(str(source))
    window._run_pdbqt_validation()

    assert calls[0] == (
        application.validate_pdbqt,
        (),
        {"payload": "ROOT\nENDROOT\nTORSDOF 0"},
    )
    assert calls[1] == (
        application.validate_pdbqt,
        (),
        {"input_path": str(source)},
    )


def test_close_waits_for_active_service_worker(window, qt_app):
    release = threading.Event()

    def slow_service():
        release.wait(5)
        return {"operation": "slow", "status": "success", "data": {}}

    window._run_service(slow_service)
    deadline = time.monotonic() + 2
    while not any(worker.isRunning() for worker in window._workers):
        qt_app.processEvents()
        assert time.monotonic() < deadline

    close_event = QtGui.QCloseEvent()
    window.closeEvent(close_event)
    assert not close_event.isAccepted()
    assert window._closing is True

    release.set()
    deadline = time.monotonic() + 5
    while window._workers and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    assert not window._workers


def test_immediate_close_blocks_deferred_capability_worker(qt_app, monkeypatch):
    calls = []
    monkeypatch.setattr(
        application,
        "capabilities",
        lambda: calls.append(True)
        or {"operation": "capabilities", "status": "success", "data": {}},
    )
    instance = MainWindow()
    instance.close()
    qt_app.processEvents()

    assert instance._closing is True
    assert not instance._workers
    assert calls == []
