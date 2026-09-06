from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import pytest

from cycpep_master.cli import main as cli
from cycpep_master.pipeline import _build_csv_fields


def _success(operation):
    return {"operation": operation, "status": "success", "data": {}}


def _minimal_cyclic_pdb() -> str:
    rows = [
        "LINK           N   ALA A   1                 C   ALA A   3",
    ]
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0), (3, 20.0)):
        for name, x, element in (
            ("N", 0.0, "N"), ("CA", 1.4, "C"), ("C", 2.8, "C"),
            ("O", 3.8, "O"), ("CB", 1.4, "C"),
        ):
            rows.append(
                f"ATOM  {serial:5d} {name:>4s} ALA A{residue:4d}    "
                f"{x + offset:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
                f"          {element:>2s}"
            )
            serial += 1
    return "\n".join([*rows, "END", ""])


def test_cli_defaults_to_fail_closed_v6(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    captured = {}

    def fake_run_batch(paths, **kwargs):
        captured.update({"paths": paths, **kwargs})
        return [{
            "status": "success",
            "support_status": "supported",
            "qualified_success": True,
            "repair_codes": [],
            "smiles": "C1CC1",
        }]

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    monkeypatch.setattr(
        sys, "argv", ["cycpep", "--pdb", str(source), "--no-csv"]
    )
    cli.main()
    assert captured["path"] == "v6"
    assert captured["require_empty_persistent_overlay"] is False
    assert "qualified=True" in capsys.readouterr().out


def test_cli_forwards_formal_empty_overlay_policy(tmp_path, monkeypatch):
    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    captured = {}

    def fake_run_batch(_paths, **kwargs):
        captured.update(kwargs)
        return [{
            "status": "success",
            "support_status": "qualified",
            "qualified_success": True,
            "repair_codes": [],
            "smiles": "C1CC1",
        }]

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", [
        "cycpep", "--pdb", str(source), "--no-csv",
        "--require-empty-persistent-overlay",
    ])

    cli.main()

    assert captured["path"] == "v6"
    assert captured["require_empty_persistent_overlay"] is True


def test_cli_exposes_every_diagnostic_path(tmp_path, monkeypatch):
    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    observed = []

    def fake_run_batch(paths, **kwargs):
        observed.append(kwargs["path"])
        return [{"status": "failed", "error": "development fixture"}]

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    for path in "abcefgh":
        monkeypatch.setattr(
            sys,
            "argv",
            ["cycpep", "--pdb", str(source), "--no-csv", "--path", path],
        )
        cli.main()
    assert observed == list("abcefgh")


def test_directory_discovery_includes_pdb_mmcif_and_gzip(tmp_path):
    expected = {
        "a.pdb", "b.ent", "c.cif", "d.mmcif", "e.pdb.gz", "f.cif.gz"
    }
    for name in [*expected, "ignored.sdf"]:
        (tmp_path / name).write_text("", encoding="ascii")
    observed = {path.split("\\")[-1].split("/")[-1] for path in cli._find_coordinate_files(str(tmp_path))}
    assert observed == expected


def test_default_directory_export_suffix_matches_requested_format(tmp_path):
    args = type("Args", (), {
        "export_dir": None,
        "pdb": None,
        "dir": str(tmp_path / "inputs"),
        "export_format": "sdf",
    })()

    assert cli._derive_export_dir(args).endswith("inputs_sdf")


def test_json_output_write_failure_is_structured(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "result.json"
    monkeypatch.setattr(Path, "write_text", lambda *_a, **_k: (_ for _ in ()).throw(
        OSError("disk full")
    ))
    args = type("Args", (), {"json_out": str(destination), "compact": True})()

    exit_code = cli._emit_json(_success("capabilities"), args)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "disk full" in payload["error"]


def test_legacy_cli_converts_batch_write_exception_to_nonzero(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    monkeypatch.setattr(
        cli, "run_batch", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))
    )

    exit_code = cli.main(["--pdb", str(source), "--csv", str(tmp_path / "x.csv")])

    assert exit_code == 1
    assert "disk full" in capsys.readouterr().err


def test_flexibility_csv_schema_exposes_components_not_only_composite():
    fields = _build_csv_fields(
        compute_rmsd=True, has_target=False, do_compare=False
    )
    for field in (
        "flexibility_status",
        "flexibility_proxy",
        "rmsd",
        "backbone_rmsd_mean",
        "sidechain_rmsd_mean",
        "energy_std",
        "flexibility_num_confs",
        "flexibility_num_kept",
        "flexibility_pair_count",
        "flexibility_error",
    ):
        assert field in fields


def test_cli_prints_flexibility_proxy_components(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")

    def fake_run_batch(_paths, **_kwargs):
        return [{
            "status": "success",
            "support_status": "supported",
            "qualified_success": True,
            "repair_codes": [],
            "smiles": "C1CC1",
            "flexibility_proxy": {
                "status": "success",
                "flexibility_proxy": 1.25,
                "rmsd_mean": 0.8,
                "backbone_rmsd_mean": 0.6,
                "sidechain_rmsd_mean": 1.1,
                "energy_std": 4.5,
            },
        }]

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cycpep", "--pdb", str(source), "--no-csv", "--flexibility-proxy"],
    )
    cli.main()
    output = capsys.readouterr().out
    assert "FLEXIBILITY_PROXY: status=success score=1.25" in output
    assert "backbone_rmsd=0.6" in output
    assert "sidechain_rmsd=1.1" in output


def test_cli_batch_reports_repaired_success_as_unqualified(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")

    def fake_run_batch(_paths, **_kwargs):
        return [{
            "file": source.name,
            "status": "success",
            "support_status": "repaired",
            "qualified_success": False,
            "repair_codes": ["LOCALLY_INFERRED_MONOMER"],
            "smiles": "C1CC1",
            "error": None,
        }]

    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["cycpep", "--dir", str(tmp_path)])

    cli.main()

    output = capsys.readouterr().out
    assert "[UNQUALIFIED]" in output
    assert "Done: 0 qualified, 1 unqualified, 0 errors, 1 total" in output


def test_cli_real_v6_reports_unsupported_coordinate_code(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "input.sdf"
    source.write_text("unsupported\n", encoding="ascii")
    monkeypatch.setattr(
        sys, "argv", ["cycpep", "--pdb", str(source), "--no-csv"]
    )

    exit_code = cli.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "STATUS: not_supported support=not_supported qualified=False" in output
    assert "CODES: UNSUPPORTED_COORDINATE_FORMAT" in output
    assert "unsupported coordinate filename" in output


def test_cli_real_v6_uses_strict_coordinate_pipeline(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.pdb"
    source.write_text(_minimal_cyclic_pdb(), encoding="ascii")
    monkeypatch.setattr(
        sys,
        "argv",
        ["cycpep", "--pdb", str(source), "--chain", "A", "--no-csv"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "STATUS:" in output or "ERROR:" in output


def test_legacy_batch_returns_nonzero_when_any_input_fails(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    monkeypatch.setattr(
        cli,
        "run_batch",
        lambda *_args, **_kwargs: [
            {"file": "input.pdb", "status": "failed", "error": "fixture"}
        ],
    )

    exit_code = cli.main(["--dir", str(tmp_path), "--no-csv"])

    assert exit_code == 1
    assert "1 errors" in capsys.readouterr().out


def test_cli_dispatches_extended_service_commands(tmp_path, monkeypatch, capsys):
    calls = []

    def capture(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return _success(name)

        return fake

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"one": "C"}), encoding="utf-8")
    command_cases = [
        (
            "batch_export_structures",
            ["batch-export", str(manifest), str(tmp_path / "exports")],
        ),
        ("find_conformer_template", ["template", "lookup", "PEPTIDE1{A}$$$$"]),
        (
            "generate_template_conformers",
            ["template", "generate", "C", "PEPTIDE1{A}$$$$", str(tmp_path / "x.sdf")],
        ),
        (
            "prepare_ligand_pdbqt_best_available",
            ["pdbqt", "ligand-pdb", "in.pdb", "out.pdbqt"],
        ),
        (
            "prepare_ligand_pdbqt_from_mol2",
            [
                "pdbqt",
                "ligand-mol2",
                "in.mol2",
                "out.pdbqt",
                "--torsdof-limit",
                "10",
                "--flexibility-mode",
                "balanced",
            ],
        ),
        (
            "prepare_ligand_from_sequence",
            [
                "prepare-sequence",
                "ACDEFG",
                str(tmp_path / "sequence"),
                "--cyclization",
                "head-to-tail",
                "--no-pdbqt",
            ],
        ),
        (
            "prepare_ligand_pdbqt_from_pdb",
            [
                "pdbqt",
                "ligand-pdb",
                "in.pdb",
                "out.pdbqt",
                "--strict-format",
            ],
        ),
        (
            "prepare_ligand_pdbqt_ensemble",
            ["pdbqt", "ensemble", "C", str(tmp_path / "ensemble")],
        ),
        (
            "run_prepared_vina",
            [
                "vina",
                "ligand.pdbqt",
                "receptor.pdbqt",
                "out.pdbqt",
                "--center",
                "1",
                "2",
                "3",
            ],
        ),
        (
            "batch_dock_structures",
            [
                "batch-dock",
                "receptor.pdb",
                "peptide.pdb",
                "--center",
                "1",
                "2",
                "3",
            ],
        ),
    ]
    for service_name, argv in command_cases:
        monkeypatch.setattr(cli.services, service_name, capture(service_name))
        assert cli.main(argv) == 0

    assert [name for name, _args, _kwargs in calls] == [
        name for name, _argv in command_cases
    ]
    pdbqt_calls = [
        (name, kwargs)
        for name, _args, kwargs in calls
        if name in {
            "prepare_ligand_pdbqt_best_available",
            "prepare_ligand_pdbqt_from_pdb",
        }
    ]
    assert [name for name, _kwargs in pdbqt_calls] == [
        "prepare_ligand_pdbqt_best_available",
        "prepare_ligand_pdbqt_from_pdb",
    ]
    assert pdbqt_calls[0][1]["fallback_policy"] == "max_coverage"
    assert pdbqt_calls[1][1]["fallback_policy"] == "strict_v6"
    capsys.readouterr()


def test_cli_strict_format_returns_nonzero_when_format_is_unfulfilled(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        cli.services,
        "export_structure",
        lambda *_args, **_kwargs: {
            "operation": "export",
            "status": "success",
            "data": {
                "requested_format": "mol2",
                "requested_format_status": "metadata_only",
                "artifacts": [{"format": "metadata"}],
            },
            "error": None,
        },
    )

    exit_code = cli.main([
        "export", "CC", "strict.mol2", "--strict-format"
    ])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["data"]["requested_format_status"] == "metadata_only"


def test_cli_export_policy_defaults_follow_strict_format(monkeypatch, capsys):
    calls = []

    def capture(*_args, **kwargs):
        calls.append(kwargs)
        return _success("export")

    monkeypatch.setattr(cli.services, "export_best_available", capture)
    assert cli.main(["export", "CC", "best.mol2"]) == 0
    monkeypatch.setattr(cli.services, "export_structure", capture)
    assert cli.main([
        "export", "CC", "strict.mol2", "--strict-format"
    ]) == 0

    assert calls[0]["fallback_policy"] == "max_coverage"
    assert calls[1]["fallback_policy"] == "strict_v6"
    capsys.readouterr()


def test_multichain_cli_recovers_input_after_chain_list(monkeypatch, capsys):
    captured = {}

    def fake_multichain(path, **kwargs):
        captured.update({"path": path, **kwargs})
        return _success("reconstruct_multichain")

    monkeypatch.setattr(
        cli.services, "reconstruct_multichain", fake_multichain
    )

    assert cli.main([
        "reconstruct",
        "--path",
        "v6",
        "--multichain",
        "--chains",
        "A",
        "B",
        "insulin.pdb",
    ]) == 0

    assert captured == {"path": "insulin.pdb", "chain_ids": ["A", "B"]}
    capsys.readouterr()


def test_reconstruct_cli_defaults_to_unified_auto(monkeypatch, capsys):
    captured = {}

    def fake_unified(source, chain_id=None, mode="auto"):
        captured.update({"source": source, "chain_id": chain_id, "mode": mode})
        return _success("reconstruct_structure")

    monkeypatch.setattr(cli.services, "reconstruct_structure", fake_unified)

    assert cli.main(["reconstruct", "peptide.pdb"]) == 0

    assert captured == {
        "source": "peptide.pdb",
        "chain_id": None,
        "mode": "auto",
    }
    capsys.readouterr()


def test_reconstruct_cli_forwards_unified_mode_and_chain(monkeypatch, capsys):
    captured = {}

    def fake_unified(source, chain_id=None, mode="auto"):
        captured.update({"source": source, "chain_id": chain_id, "mode": mode})
        return _success("reconstruct_structure")

    monkeypatch.setattr(cli.services, "reconstruct_structure", fake_unified)

    assert cli.main([
        "reconstruct",
        "PEPTIDE1{A.A}$$$$V2.0",
        "--mode",
        "best-effort",
        "--chain",
        "A",
    ]) == 0

    assert captured == {
        "source": "PEPTIDE1{A.A}$$$$V2.0",
        "chain_id": "A",
        "mode": "best_effort",
    }
    capsys.readouterr()


def test_reconstruct_cli_forwards_unified_chain_list(monkeypatch, capsys):
    captured = {}

    def fake_unified(source, chain_id=None, mode="auto"):
        captured.update({"source": source, "chain_id": chain_id, "mode": mode})
        return _success("reconstruct_structure")

    monkeypatch.setattr(cli.services, "reconstruct_structure", fake_unified)

    assert cli.main([
        "reconstruct", "peptide.pdb", "--chains", "A", "B"
    ]) == 0
    assert captured == {
        "source": "peptide.pdb",
        "chain_id": ["A", "B"],
        "mode": "auto",
    }
    capsys.readouterr()


def test_reconstruct_cli_strict_mode_calls_unified_service(monkeypatch, capsys):
    captured = {}

    def fake_unified(source, chain_id=None, mode="auto"):
        captured.update({"source": source, "chain_id": chain_id, "mode": mode})
        return _success("reconstruct_structure")

    monkeypatch.setattr(cli.services, "reconstruct_structure", fake_unified)

    assert cli.main([
        "reconstruct", "peptide.pdb", "--mode", "strict"
    ]) == 0

    assert captured == {
        "source": "peptide.pdb",
        "chain_id": None,
        "mode": "strict",
    }
    capsys.readouterr()


def test_reconstruct_cli_explicit_path_uses_legacy_pipeline(monkeypatch, capsys):
    captured = {}

    def fake_legacy(inputs, **kwargs):
        captured.update({"inputs": inputs, **kwargs})
        return _success("reconstruct")

    monkeypatch.setattr(cli.services, "reconstruct_coordinates", fake_legacy)

    assert cli.main(["reconstruct", "peptide.pdb", "--path", "v6"]) == 0

    assert captured["inputs"] == ["peptide.pdb"]
    assert captured["path"] == "v6"
    assert captured["chain_id"] == "L"
    assert captured["target_chain_id"] == "R"
    capsys.readouterr()


def test_reconstruct_cli_explicit_path_rejects_non_strict_mode(monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["reconstruct", "peptide.pdb", "--path", "v6", "--mode", "best-effort"])

    assert error.value.code == 2
    assert "cannot be combined with an explicit --path" in capsys.readouterr().err


def test_reconstruct_cli_unified_rejects_multiple_sources(monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["reconstruct", "a.pdb", "b.pdb"])

    assert error.value.code == 2
    assert "explicit --path" in capsys.readouterr().err


def test_reconstruct_cli_unified_rejects_legacy_batch_flags(monkeypatch, capsys):
    for argv in (
        ["reconstruct", "a.pdb", "--dir", "inputs"],
        ["reconstruct", "a.pdb", "--admet"],
        ["reconstruct", "a.pdb", "--export-dir", "out"],
        ["reconstruct", "a.pdb", "--multichain"],
        ["reconstruct", "a.pdb", "--target-chain", "R"],
        ["reconstruct", "a.pdb", "--export-format", "mol2"],
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 2
        assert "explicit --path" in capsys.readouterr().err


def test_reconstruct_cli_legacy_path_rejects_chains_without_multichain(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([
            "reconstruct", "peptide.pdb", "--path", "v6", "--chains", "A", "B"
        ])

    assert error.value.code == 2
    assert "--chains requires --multichain" in capsys.readouterr().err


def test_reconstruct_unified_cli_forwards_best_effort_spelling(
    monkeypatch, capsys
):
    captured = {}

    def fake_unified(source, chain_id=None, mode="auto"):
        captured.update({"source": source, "chain_id": chain_id, "mode": mode})
        return _success("reconstruct_unified")

    monkeypatch.setattr(cli.services, "reconstruct_unified", fake_unified)

    assert cli.main([
        "reconstruct-unified",
        "PEPTIDE1{A.A}$$$$V2.0",
        "--mode",
        "best-effort",
        "--chain",
        "A",
        "B",
    ]) == 0

    assert captured == {
        "source": "PEPTIDE1{A.A}$$$$V2.0",
        "chain_id": ["A", "B"],
        "mode": "best_effort",
    }
    capsys.readouterr()


def test_reconstruct_unified_cli_real_sequence_success(monkeypatch, capsys):
    assert cli.main(["reconstruct-unified", "AAAAA", "--compact"]) == 0

    output = capsys.readouterr().out
    assert '"operation": "reconstruct_unified"' in output
    assert '"status": "success"' in output
    assert '"source_kind": "sequence"' in output


def test_reconstruct_exact_cli_uses_dedicated_coordinate_service(
    monkeypatch, capsys
):
    captured = {}

    def fake_exact(source, **kwargs):
        captured.update({"source": source, **kwargs})
        return _success("reconstruct_exact_v1")

    monkeypatch.setattr(
        cli.services, "reconstruct_exact_v1", fake_exact
    )

    assert cli.main([
        "reconstruct-exact",
        "peptide.cif",
        "--chain",
        "LIG",
        "--minimum-macrocycle-ring-size",
        "6",
        "--allow-linear-topology",
    ]) == 0
    assert captured == {
        "source": "peptide.cif",
        "chain_id": "LIG",
        "minimum_macrocycle_ring_size": 6,
        "require_empty_persistent_overlay": True,
        "allow_linear_topology": True,
    }
    capsys.readouterr()


def test_prepare_sequence_cli_forwards_the_public_contract(
    tmp_path, monkeypatch, capsys
):
    captured = {}

    def fake(sequence, output_dir, **kwargs):
        captured.update({
            "sequence": sequence,
            "output_dir": output_dir,
            **kwargs,
        })
        return _success("prepare_ligand_from_sequence")

    monkeypatch.setattr(
        cli.services, "prepare_ligand_from_sequence", fake
    )
    output = tmp_path / "prepared"

    assert cli.main([
        "prepare-sequence",
        "AC[dA]EF",
        str(output),
        "--cyclization",
        "infer",
        "--stereochemistry-json",
        '{"2":"D"}',
        "--terminal-modifications-json",
        '{"N":"ACE","C":"NME"}',
        "--protonation",
        "physiological",
        "--conformers",
        "3",
        "--torsdof-limit",
        "10",
        "--flexibility-mode",
        "thorough",
        "--torsion-prior",
        "prior.json",
        "--template-strategy",
        "off",
        "--seed",
        "7",
        "--threads",
        "2",
        "--no-pdbqt",
    ]) == 0

    assert captured == {
        "sequence": "AC[dA]EF",
        "output_dir": str(output),
        "cyclization": "infer",
        "stereochemistry": {"2": "D"},
        "terminal_modifications": {"N": "ACE", "C": "NME"},
        "protonation": "physiological",
        "conformer_count": 3,
        "generate_pdbqt": False,
        "torsdof_limit": 10,
        "flexibility_mode": "thorough",
        "torsion_prior_path": "prior.json",
        "template_strategy": "off",
        "random_seed": 7,
        "num_threads": 2,
    }


def test_cli_monomer_resolve_materializes_request_local_definition(capsys):
    context = json.dumps({
        "definitions": [{
            "symbol": "CLI_CTX",
            "smiles": "N[C@@H](CCl)C(=O)O",
        }],
        "include_persistent_user": False,
    })

    assert cli.main([
        "monomer",
        "resolve",
        "CLI_CTX",
        "--monomer-context-json",
        context,
        "--compact",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "success"
    assert payload["data"]["status"] == "resolved"
    assert payload["data"]["chemical_rigor"] == "C3:S"
    assert payload["data"]["persistent_writes"] == 0


def test_cli_monomer_resolve_unknown_is_successful_partial(capsys):
    assert cli.main([
        "monomer",
        "resolve",
        "NO_SUCH_COMPONENT",
        "--monomer-context-json",
        '{"allow_network":false}',
        "--compact",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "success"
    assert payload["data"]["status"] == "partial"
    assert payload["data"]["chemical_rigor"] == "C1:H"
    assert payload["data"]["unresolved"][0]["code"] == (
        "MONOMER_UNRESOLVED"
    )
    capsys.readouterr()


def test_pdbqt_help_hides_retired_direct_ensemble(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["pdbqt", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "pdbqt ensemble" not in output
    assert "{ligand,ligand-mol2,ligand-pdb,receptor,validate}" in output


def test_convert_cli_forwards_edge_projection_capacity(
    monkeypatch, capsys
):
    captured = {}

    def fake_convert(source, target, payload, **kwargs):
        captured.update({
            "source": source,
            "target": target,
            "payload": payload,
            **kwargs,
        })
        return _success("convert")

    monkeypatch.setattr(
        cli.services, "convert_representation", fake_convert
    )

    assert cli.main([
        "convert",
        "--from",
        "map",
        "--to",
        "edge_v1",
        "CAAC{cyc:1:R3-4:R3}",
        "--edge-max-rings",
        "5",
        "--edge-max-position",
        "64",
    ]) == 0
    assert captured == {
        "source": "map",
        "target": "edge_v1",
        "payload": "CAAC{cyc:1:R3-4:R3}",
        "edge_max_rings": 5,
        "edge_max_position": 64,
    }
    capsys.readouterr()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--edge-max-rings", "0"),
        ("--edge-max-position", "-1"),
    ],
)
def test_convert_cli_rejects_nonpositive_edge_capacity(
    capsys, option, value
):
    with pytest.raises(SystemExit) as error:
        cli.main([
            "convert",
            "--from",
            "map",
            "--to",
            "edge_v1",
            "ACD",
            option,
            value,
        ])

    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_pdbqt_validation_cli_accepts_direct_payload(monkeypatch, capsys):
    captured = {}

    def fake_validate(**kwargs):
        captured.update(kwargs)
        return _success("validate_pdbqt")

    monkeypatch.setattr(cli.services, "validate_pdbqt", fake_validate)

    assert cli.main(["pdbqt", "validate", "--payload", "ROOT\nENDROOT\nTORSDOF 0"]) == 0

    assert captured == {"payload": "ROOT\nENDROOT\nTORSDOF 0"}
    capsys.readouterr()


@pytest.mark.parametrize(
    "argv",
    [
        ("convert", "--from", "smiles", "--to", "smiles", "CC", "--input-file"),
        ("audit", "--kind", "smiles", "CC", "--input-file"),
    ],
)
def test_cli_rejects_conflicting_payload_and_input_file(tmp_path, capsys, argv):
    source = tmp_path / "payload.txt"
    source.write_text("CC", encoding="ascii")

    assert cli.main([*argv, str(source), "--compact"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid_input"
    assert "mutually exclusive" in payload["error"]


def test_pdbqt_validation_cli_rejects_conflicting_sources(tmp_path, capsys):
    source = tmp_path / "payload.pdbqt"
    source.write_text("ROOT\nENDROOT\nTORSDOF 0\n", encoding="ascii")

    assert cli.main([
        "pdbqt", "validate", str(source), "--payload",
        "ROOT\nENDROOT\nTORSDOF 0", "--compact",
    ]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid_input"
    assert "mutually exclusive" in payload["error"]


def test_run_py_propagates_cli_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda: 7)
    run_path = Path(__file__).resolve().parents[1] / "run.py"

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(run_path), run_name="__main__")

    assert error.value.code == 7
