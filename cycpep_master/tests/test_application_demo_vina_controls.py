from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from cycpep_master import application
from cycpep_master.docking import vina


ATOM = "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
LIGAND = "ROOT\n" + ATOM + "ENDROOT\nTORSDOF 0\n"


def test_vina_execution_controls_reach_process(tmp_path):
    ligand = tmp_path / "ligand.pdbqt"
    receptor = tmp_path / "receptor.pdbqt"
    executable = tmp_path / "vina.exe"
    output = tmp_path / "output.pdbqt"
    ligand.write_text(LIGAND)
    receptor.write_text(ATOM)
    executable.write_bytes(b"")
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        output.write_text("MODEL 1\nREMARK VINA RESULT: -5.5 0 0\n" + LIGAND + "ENDMDL\n")
        return SimpleNamespace(returncode=0, stdout="mode | affinity | dist from best mode\n1 -5.5 0 0\n", stderr="")

    affinity, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(output),
        seed=17, cpu=2, max_evals=10000, timeout_seconds=120,
        find_executable=lambda: str(executable), run_process=run,
    )
    assert (affinity, error) == (-5.5, None)
    for option, value in (("seed", "17"), ("cpu", "2"), ("max_evals", "10000")):
        assert observed["command"][observed["command"].index("--" + option) + 1] == value
    assert observed["kwargs"]["timeout"] == 120


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"cpu": -1}, {"max_evals": 1.5}, {"timeout_seconds": float("nan")}, {"timeout_seconds": 0}])
def test_invalid_controls_do_not_remove_existing_output(tmp_path, kwargs):
    output = tmp_path / "output.pdbqt"
    output.write_text("keep")
    _, error = vina.run_vina("absent", "absent", (0, 0, 0), (24, 24, 24), str(output), **kwargs)
    assert error.startswith("Invalid Vina execution controls")
    assert output.read_text() == "keep"
    result = application.run_prepared_vina("absent", "absent", output, center=(0, 0, 0), **kwargs)
    assert result["status"] == "invalid_input"


def test_public_api_forwards_and_records_controls(monkeypatch, tmp_path):
    observed = {}

    def run(*args, **kwargs):
        observed.update(kwargs)
        return -6.2, None

    monkeypatch.setattr(vina, "run_vina", run)
    result = application.run_prepared_vina(
        "ligand", "receptor", tmp_path / "out", center=(0, 0, 0),
        seed=29, cpu=2, max_evals=10000, timeout_seconds=120,
    )
    assert result["status"] == "success"
    assert observed["seed"] == 29
    assert observed["max_evals"] == 10000
    assert result["data"]["cpu"] == 2
    assert result["data"]["timeout_seconds"] == 120
    assert result["data"]["seed_mode"] == "fixed"
    automatic = application.run_prepared_vina(
        "ligand", "receptor", tmp_path / "auto", center=(0, 0, 0), seed=0, cpu=0,
    )
    assert automatic["data"]["seed_mode"] == "automatic"
    assert automatic["data"]["cpu_mode"] == "automatic"


def test_cli_forwards_search_controls(monkeypatch, capsys):
    from cycpep_master.cli import main as cli

    observed = {}

    def execute(*args, **kwargs):
        observed.update(kwargs)
        return {"operation": "run_prepared_vina", "status": "success", "data": {}}

    monkeypatch.setattr(cli.services, "run_prepared_vina", execute)
    assert cli.main([
        "vina", "ligand", "receptor", "output", "--center", "0", "0", "0",
        "--seed", "17", "--cpu", "2", "--max-evals", "10000",
        "--timeout-seconds", "120",
    ]) == 0
    assert {key: observed[key] for key in ("seed", "cpu", "max_evals", "timeout_seconds")} == {
        "seed": 17, "cpu": 2, "max_evals": 10000, "timeout_seconds": 120,
    }
    capsys.readouterr()


def test_custom_timeout_message(tmp_path):
    ligand = tmp_path / "ligand.pdbqt"
    receptor = tmp_path / "receptor.pdbqt"
    executable = tmp_path / "vina.exe"
    for path in (ligand, receptor, executable):
        path.write_text(LIGAND)

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    _, error = vina.run_vina(
        str(ligand), str(receptor), (0, 0, 0), (24, 24, 24), str(tmp_path / "out"),
        timeout_seconds=45, find_executable=lambda: str(executable), run_process=run,
    )
    assert error == "Vina timed out (>45 s)"
