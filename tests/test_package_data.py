from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib

import cycpep_master


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _package_data_patterns() -> tuple[str, ...]:
    config = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return tuple(config["tool"]["setuptools"]["package-data"]["cycpep_master"])


def test_runtime_and_project_versions_are_v710():
    config = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert config["project"]["version"] == "7.1.0"
    assert cycpep_master.__version__ == "7.1.0"
    docking = config["project"]["optional-dependencies"]["docking"]
    assert any(value.startswith("meeko") for value in docking)
    assert any(value.startswith("scipy") for value in docking)


def _is_packaged(relative_path: str, patterns: tuple[str, ...]) -> bool:
    candidate = PurePosixPath(relative_path)
    return any(candidate.match(pattern) for pattern in patterns)


def test_runtime_chemistry_resources_are_declared_as_package_data():
    patterns = _package_data_patterns()
    required = (
        "THIRD_PARTY_DATA.md",
        "EXACT_V1.md",
        "EXACT_V1_MIGRATION.md",
        "V5_ARTIFACT_CONTRACT.md",
        "V5_REPRODUCIBILITY.md",
        "unified_monomer_library.csv",
        "special_residue_library.csv",
        "derived_monomer_library.csv",
        "derived_monomer_manifest.json",
        "derived_monomer_quarantine.csv",
        "libraries/manifest.json",
        "data/templates/templates_index.json",
        "data/applicability_manifest.json",
        "data/v5_evidence_dossier.json",
        "data/torsion_priors/torsion_priors_runtime.json",
        "data/torsion_priors/torsion_prior_manifest_runtime.json",
        "data/torsion_priors/torsion_definitions.json",
        "schemas/candidate_assessment.schema.json",
        "schemas/exact_v1.schema.json",
        "schemas/v5_artifact.schema.json",
    )
    for relative in required:
        assert (PACKAGE_ROOT / relative).is_file(), relative
        assert _is_packaged(relative, patterns), relative


def test_default_template_view_files_are_declared_and_present():
    patterns = _package_data_patterns()
    index = json.loads(
        (PACKAGE_ROOT / "data/templates/templates_index.json").read_text(
            encoding="utf-8"
        )
    )
    selected = {
        str(entry["pdb_path"]).replace("\\", "/")
        for entry in index.values()
        if str(entry.get("source", "")).lower() in {"cpbind", "scaffold"}
    }
    assert selected
    for relative in sorted(selected):
        package_relative = f"data/templates/{relative}"
        assert (PACKAGE_ROOT / package_relative).is_file(), package_relative
        assert _is_packaged(package_relative, patterns), package_relative


def test_default_torsion_prior_is_hash_verified_and_loadable():
    from cycpep_master import application
    from cycpep_master.docking.torsion_prior import load_torsion_prior

    root = PACKAGE_ROOT / "data" / "torsion_priors"
    runtime = root / "torsion_priors_runtime.json"
    manifest_path = root / "torsion_prior_manifest_runtime.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    frozen_manifest = root / "torsion_prior_manifest.json"
    assert manifest["frozen_source_manifest_sha256"] == hashlib.sha256(
        frozen_manifest.read_bytes()
    ).hexdigest()
    assert not _is_packaged(
        "data/torsion_priors/torsion_prior_manifest.json",
        _package_data_patterns(),
    )
    observed = hashlib.sha256(runtime.read_bytes()).hexdigest()

    assert observed == manifest["runtime_sha256"]
    index = load_torsion_prior(runtime)
    assert index.runtime_sha256 == observed
    assert {
        level: len(entries)
        for level, entries in index.levels.items()
    } == manifest["level_entry_counts"]
    capabilities = application.capabilities()["data"]
    assert capabilities["availability"]["torsion_prior"] is True
    assert capabilities["torsion_prior"]["runtime_sha256"] == observed
    assert capabilities["availability"]["v5_resources"] is True
    for resource in capabilities["v5_resources"].values():
        assert Path(resource["path"]).is_file()
        assert len(resource["sha256"]) == 64


def test_publishable_sources_contain_no_machine_local_paths():
    publishable = (
        PACKAGE_ROOT / "README.md",
        PACKAGE_ROOT / "V5_ARTIFACT_CONTRACT.md",
        PACKAGE_ROOT / "V5_REPRODUCIBILITY.md",
        PACKAGE_ROOT / "docking" / "build_template_library.py",
        PACKAGE_ROOT / "docking" / "compress_template_library.py",
        PACKAGE_ROOT
        / "data"
        / "torsion_priors"
        / "torsion_prior_manifest_runtime.json",
    )
    pattern = re.compile(
        r"(?:\b[A-Za-z]:[\\/]|/home/|/Users/)",
        re.IGNORECASE,
    )
    findings = {
        str(path.relative_to(PACKAGE_ROOT)): pattern.findall(
            path.read_text(encoding="utf-8")
        )
        for path in publishable
        if pattern.search(path.read_text(encoding="utf-8"))
    }
    assert findings == {}
