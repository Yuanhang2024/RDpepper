from __future__ import annotations

import importlib
from pathlib import Path
import tomllib

import cycpep_master
import rdpepper
from rdpepper import application as public_application
from rdpepper import cli as public_cli


def test_rdpepper_facade_preserves_public_object_identity():
    assert rdpepper.__version__ == cycpep_master.__version__ == "7.1.0"
    assert (
        rdpepper.reconstruct_structure
        is cycpep_master.reconstruct_structure
    )
    assert rdpepper.CyclicPeptideGraph is cycpep_master.CyclicPeptideGraph
    assert public_application is importlib.import_module(
        "cycpep_master.application"
    )


def test_rdpepper_application_module_forwards_attributes():
    module = importlib.import_module("rdpepper.application")
    assert (
        module.prepare_ligand_from_sequence
        is public_application.prepare_ligand_from_sequence
    )


def test_rdpepper_cli_uses_public_brand(capsys):
    try:
        public_cli.main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert capsys.readouterr().out.strip() == "rdpepper 7.1.0"


def test_legacy_cli_keeps_legacy_brand(capsys):
    from cycpep_master.cli.main import main

    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert capsys.readouterr().out.strip() == "cycpep-master 7.1.0"


def test_distribution_declares_new_and_compatibility_entry_points():
    config = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert config["project"]["name"] == "rdpepper"
    assert config["project"]["version"] == "7.1.0"
    scripts = config["project"]["scripts"]
    assert scripts["rdpepper"] == "rdpepper.cli:main"
    assert scripts["cycpep"] == "cycpep_master.cli.main:main"
    packages = config["tool"]["setuptools"]["packages"]
    assert "rdpepper" in packages
    assert "cycpep_master" in packages
