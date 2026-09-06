from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cycpep_master.docking import vina_wrapper


pytest.importorskip("meeko")

GOLDEN = (
    Path(__file__).parent
    / "golden"
    / "vina_wrapper"
    / "cyclohexane_seed42.pdbqt"
)


@pytest.mark.skip(
    reason="Release-021 direct-SMILES golden is superseded by V4 MOL2 parent"
)
def test_fixed_seed_ligand_pdbqt_matches_golden_bytes_and_audit(tmp_path):
    output = tmp_path / "ligand.pdbqt"
    audit = {}

    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "C1CCCCC1",
        str(output),
        num_confs=1,
        random_seed=42,
        protonate=False,
        rigid_macrocycles=True,
        conf_out=audit,
    )

    assert error is None
    assert output.read_bytes() == GOLDEN.read_bytes()
    assert audit["source"] == "embed3d"
    assert audit["template_meta"] is None
    assert audit["template_error"] is None
    assert audit["torsion_budget"]["status"] == "disabled"
    assert audit["torsion_budget"]["initial_torsdof"] == 0
    assert audit["torsion_budget"]["final_torsdof"] == 0
    assert audit["pdbqt_tree"] == {
        "atom_count": 6,
        "branch_count": 0,
        "torsdof": 0,
        "connectivity_edge_count": 6,
        "connectivity_preserved": True,
    }


def test_run_vina_preserves_command_order_timeout_and_working_directory(
    tmp_path, monkeypatch
):
    executable = tmp_path / "bin" / "vina.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"")
    observed = {}
    monkeypatch.chdir(tmp_path)
    ligand = tmp_path / "ligand.pdbqt"
    receptor = tmp_path / "receptor.pdbqt"
    ligand.write_text(
        "ROOT\n"
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        "ENDROOT\nTORSDOF 0\n",
        encoding="utf-8",
    )
    receptor.write_text("RECEPTOR\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        destination = Path(command[command.index("--out") + 1])
        if not destination.is_absolute():
            destination = Path(kwargs["cwd"]) / destination
        destination.write_text(
            "MODEL 1\n"
            "REMARK VINA RESULT 1 -7.25 0 0\n"
            "ROOT\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
            "ENDROOT\nTORSDOF 0\n"
            "ENDMDL\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout="mode | affinity | dist from best mode\n-----+----------\n1 -7.25 0.0 0.0\n",
            stderr="",
        )

    monkeypatch.setattr(vina_wrapper, "_find_vina", lambda: str(executable))
    monkeypatch.setattr(vina_wrapper.subprocess, "run", fake_run)

    affinity, error = vina_wrapper.run_vina(
        ligand_pdbqt="ligand.pdbqt",
        receptor_pdbqt="receptor.pdbqt",
        center=(1.0, 2.0, 3.0),
        box_size=(20.0, 21.0, 22.0),
        output_pdbqt="output.pdbqt",
        exhaustiveness=17,
        num_modes=5,
    )

    assert (affinity, error) == (-7.25, None)
    assert observed["command"] == [
        str(executable),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--center_x",
        "1.0",
        "--center_y",
        "2.0",
        "--center_z",
        "3.0",
        "--size_x",
        "20.0",
        "--size_y",
        "21.0",
        "--size_z",
        "22.0",
        "--out",
        str(tmp_path / "output.pdbqt"),
        "--exhaustiveness",
        "17",
        "--num_modes",
        "5",
    ]
    assert observed["kwargs"] == {
        "capture_output": True,
        "text": True,
        "timeout": 600,
        "cwd": str(executable.parent),
    }


def test_run_vina_timeout_contract_is_stable(tmp_path, monkeypatch):
    executable = tmp_path / "vina"
    executable.write_bytes(b"")
    (tmp_path / "ligand.pdbqt").write_text("LIGAND\n", encoding="utf-8")
    (tmp_path / "receptor.pdbqt").write_text("RECEPTOR\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vina_wrapper, "_find_vina", lambda: str(executable))

    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 600)

    monkeypatch.setattr(vina_wrapper.subprocess, "run", timeout)

    affinity, error = vina_wrapper.run_vina(
        "ligand.pdbqt",
        "receptor.pdbqt",
        (0.0, 0.0, 0.0),
        (25.0, 25.0, 25.0),
        "output.pdbqt",
    )

    assert affinity is None
    assert error == "Vina timed out (>10 min)"
