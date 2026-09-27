"""Diagnostic Vina modes: score_only and local_only alongside default docking."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from cycpep_master import application
from cycpep_master.docking import vina


ATOM = "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
LIGAND = "ROOT\n" + ATOM + "ENDROOT\nTORSDOF 0\n"
DOCKING_STDOUT = "mode | affinity | dist from best mode\n1 -5.5 0 0\n"
SCORE_ONLY_STDOUT = (
    "Estimated Free Energy of Binding   : -9.361 (kcal/mol) [=(1)+(2)+(3)-(4)]\n"
    "(1) Final Intermolecular Energy    : -14.834 (kcal/mol)\n"
)


def _stage_inputs(tmp_path, pose=LIGAND):
    ligand = tmp_path / "ligand.pdbqt"
    receptor = tmp_path / "receptor.pdbqt"
    executable = tmp_path / "vina.exe"
    output = tmp_path / "output.pdbqt"
    ligand.write_text(pose)
    receptor.write_text(ATOM)
    executable.write_bytes(b"")
    return ligand, receptor, executable, output


def _process_run(stdout, output=None, payload=LIGAND):
    def run(command, **kwargs):
        if output is not None:
            output.write_text(payload)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run


def test_default_mode_is_plain_docking(tmp_path):
    ligand, receptor, executable, output = _stage_inputs(tmp_path)
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        output.write_text(LIGAND)
        return SimpleNamespace(returncode=0, stdout=DOCKING_STDOUT, stderr="")

    affinity, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        find_executable=lambda: str(executable), run_process=run,
    )
    assert (affinity, error) == (-5.5, None)
    assert "--score_only" not in observed["command"]
    assert "--local_only" not in observed["command"]
    # Explicit docking mode must produce the identical command shape.
    _, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(tmp_path / "o2.pdbqt"),
        mode="docking", find_executable=lambda: str(executable),
        run_process=_process_run(DOCKING_STDOUT, tmp_path / "o2.pdbqt"),
    )
    assert error is None


def test_score_only_sends_flag_and_requires_no_output(tmp_path):
    ligand, receptor, executable, output = _stage_inputs(tmp_path)
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        # Vina writes no output file in score_only mode.
        return SimpleNamespace(returncode=0, stdout=SCORE_ONLY_STDOUT, stderr="")

    affinity, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="score_only", find_executable=lambda: str(executable), run_process=run,
    )
    assert (affinity, error) == (-9.361, None)
    assert "--score_only" in observed["command"]
    assert "--local_only" not in observed["command"]
    assert "--out" not in observed["command"]
    assert not output.exists()


def test_score_only_never_touches_existing_output(tmp_path):
    ligand, receptor, executable, output = _stage_inputs(tmp_path)
    output.write_text("caller sentinel")

    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(returncode=0, stdout=SCORE_ONLY_STDOUT, stderr="")

    affinity, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="score_only", find_executable=lambda: str(executable), run_process=run,
    )
    assert (affinity, error) == (-9.361, None)
    assert "--out" not in observed["command"]
    assert output.read_text() == "caller sentinel"

    def failing(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    _, failure_error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="score_only", find_executable=lambda: str(executable),
        run_process=failing,
    )
    assert failure_error is not None
    assert output.read_text() == "caller sentinel"

    def unparseable(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="no energy here", stderr="")

    _, parse_error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="score_only", find_executable=lambda: str(executable),
        run_process=unparseable,
    )
    assert parse_error is not None
    assert output.read_text() == "caller sentinel"


def test_score_only_rejects_unparseable_stdout(tmp_path):
    ligand, receptor, executable, output = _stage_inputs(tmp_path)
    _, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="score_only", find_executable=lambda: str(executable),
        run_process=_process_run("Estimated Free Energy of Binding\n"),
    )
    assert error == "Failed to parse affinity from Vina output"


def test_local_only_sends_flag_and_verifies_pose(tmp_path):
    ligand, receptor, executable, output = _stage_inputs(tmp_path)
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        output.write_text(LIGAND)
        return SimpleNamespace(returncode=0, stdout=SCORE_ONLY_STDOUT, stderr="")

    affinity, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="local_only", find_executable=lambda: str(executable), run_process=run,
    )
    assert (affinity, error) == (-9.361, None)
    assert "--local_only" in observed["command"]
    assert "--score_only" not in observed["command"]


def test_local_only_rejects_missing_or_altered_output(tmp_path):
    ligand, receptor, executable, output = _stage_inputs(tmp_path)
    _, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="local_only", find_executable=lambda: str(executable),
        run_process=_process_run(SCORE_ONLY_STDOUT),  # no file written
    )
    assert error is not None and "Vina produced no output file" in error

    extra_atom = ATOM + "ATOM      2  O   LIG A   1       1.000   0.000   0.000  1.00  0.00     0.000 O\n"
    shrunk = "ROOT\n" + extra_atom + "ENDROOT\nTORSDOF 0\n"
    _, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode="local_only", find_executable=lambda: str(executable),
        run_process=_process_run(SCORE_ONLY_STDOUT, output, payload=shrunk),
    )
    assert error is not None and "atom-count mismatch" in error
    assert not output.exists()


@pytest.mark.parametrize("bad_mode", ["minimize", "score-only", "", None, 1])
def test_invalid_mode_rejected(tmp_path, bad_mode):
    ligand, receptor, _, output = _stage_inputs(tmp_path)
    _, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        mode=bad_mode,
    )
    assert error is not None and error.startswith("Invalid Vina mode")

    result = application.run_prepared_vina(
        str(ligand), str(receptor), str(output), center=(0, 0, 0), mode=bad_mode,
    )
    assert result["status"] == "invalid_input"


def test_run_prepared_vina_forwards_mode(monkeypatch, tmp_path):
    observed = {}

    def run(*args, **kwargs):
        observed.clear()
        observed.update(kwargs)
        return -6.4, None

    monkeypatch.setattr(vina, "run_vina", run)
    result = application.run_prepared_vina(
        "ligand", "receptor", tmp_path / "out", center=(0, 0, 0),
        mode="local_only",
    )
    assert result["status"] == "success"
    assert observed["mode"] == "local_only"
    assert result["data"]["mode"] == "local_only"
    assert result["data"]["output_pdbqt_written"] is True
    assert result["data"]["output_pdbqt"] == str(tmp_path / "out")

    scored = application.run_prepared_vina(
        "ligand", "receptor", tmp_path / "out2", center=(0, 0, 0),
        mode="score_only",
    )
    assert scored["status"] == "success"
    assert observed["mode"] == "score_only"
    assert scored["data"]["output_pdbqt_written"] is False
    assert scored["data"]["output_pdbqt"] is None

    # Default docking keeps the pre-mode call shape: no mode kwarg at all.
    default = application.run_prepared_vina(
        "ligand", "receptor", tmp_path / "out3", center=(0, 0, 0),
    )
    assert default["status"] == "success"
    assert "mode" not in observed
    assert default["data"]["mode"] == "docking"
    assert default["data"]["output_pdbqt_written"] is True


def test_cli_vina_forwards_mode(monkeypatch, capsys):
    from cycpep_master.cli import main as cli

    observed = {}

    def execute(*args, **kwargs):
        observed.update(kwargs)
        return {"operation": "run_prepared_vina", "status": "success", "data": {}}

    monkeypatch.setattr(cli.services, "run_prepared_vina", execute)
    assert cli.main([
        "vina", "ligand", "receptor", "output", "--center", "0", "0", "0",
        "--mode", "score_only",
    ]) == 0
    assert observed["mode"] == "score_only"
    assert cli.main([
        "vina", "ligand", "receptor", "output", "--center", "0", "0", "0",
    ]) == 0
    assert observed["mode"] == "docking"
    capsys.readouterr()


def test_parse_vina_score_only_energy_formats():
    assert vina.parse_vina_score_only_energy(SCORE_ONLY_STDOUT) == -9.361
    assert vina.parse_vina_score_only_energy(
        "Estimated Free Energy of Binding   : +0.000 (kcal/mol)\n"
    ) == 0.0
    assert vina.parse_vina_score_only_energy(DOCKING_STDOUT) is None
    assert vina.parse_vina_score_only_energy("") is None
