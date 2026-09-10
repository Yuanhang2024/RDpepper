"""CLI coverage for the optional MOL2 charge-aware compatibility reader.

The ``read_mol2`` application service itself is implemented separately; these
tests monkeypatch it (``raising=False`` until integration) and verify only the
``read-mol2`` command wiring: global allowlist routing, argument dispatch,
option forwarding, help, and exit codes.
"""
from __future__ import annotations

import json

import pytest

from cycpep_master.cli import main as cli


def _success(data=None):
    return {
        "operation": "read_mol2",
        "status": "success",
        "data": data if data is not None else {},
    }


def _write_mol2(tmp_path):
    source = tmp_path / "input.mol2"
    source.write_text("@<TRIPOS>MOLECULE\nligand\n", encoding="utf-8")
    return source


def test_read_mol2_is_in_the_global_command_allowlist():
    assert "read-mol2" in cli._COMMANDS


def test_read_mol2_cli_defaults_to_native_reader(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def fake_read_mol2(
        mol2_path, *, compatibility, receipt_path, export_sdf
    ):
        calls.append((mol2_path, compatibility, receipt_path, export_sdf))
        return _success({"reader_mode": compatibility})

    monkeypatch.setattr(
        cli.services, "read_mol2", fake_read_mol2, raising=False
    )

    assert cli.main(["read-mol2", str(_write_mol2(tmp_path))]) == 0

    assert calls == [(str(tmp_path / "input.mol2"), "rdkit_native", None, None)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["data"]["reader_mode"] == "rdkit_native"


def test_read_mol2_cli_forwards_charge_aware_selection_and_paths(
    tmp_path, monkeypatch
):
    calls = []

    def fake_read_mol2(
        mol2_path, *, compatibility, receipt_path, export_sdf
    ):
        calls.append((mol2_path, compatibility, receipt_path, export_sdf))
        return _success()

    monkeypatch.setattr(
        cli.services, "read_mol2", fake_read_mol2, raising=False
    )
    source = _write_mol2(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    sdf = tmp_path / "artifact.sdf"

    exit_code = cli.main(
        [
            "read-mol2",
            str(source),
            "--compatibility",
            "rdkit_charge_aware",
            "--receipt",
            str(receipt),
            "--export-sdf",
            str(sdf),
        ]
    )

    assert exit_code == 0
    assert calls == [
        (str(source), "rdkit_charge_aware", str(receipt), str(sdf))
    ]


def test_read_mol2_cli_rejects_unknown_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli.services, "read_mol2", lambda *_args, **_kwargs: _success(),
        raising=False,
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "read-mol2",
                str(_write_mol2(tmp_path)),
                "--compatibility",
                "openeye_magic",
            ]
        )
    assert excinfo.value.code == 2


def test_read_mol2_cli_help_lists_command_and_options(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["read-mol2", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "read-mol2" in out
    assert "--compatibility" in out
    assert "rdkit_charge_aware" in out
    assert "--receipt" in out
    assert "--export-sdf" in out


def test_read_mol2_cli_failure_envelope_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.services,
        "read_mol2",
        lambda *_args, **_kwargs: {
            "operation": "read_mol2",
            "status": "failed",
            "data": {},
            "error": "cannot parse MOL2",
        },
        raising=False,
    )
    assert cli.main(["read-mol2", "missing.mol2"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
