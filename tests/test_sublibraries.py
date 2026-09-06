"""Tests for the sub-library + metadata layer (build_sublibraries + manifest).

The unified monomer library is physically sliced by source into thin
sub-libraries under ``libraries/`` and loaded via ``manifest.json``. These
tests assert the foundational contract:

  1. The default manifest reproduces the original monomer set exactly (the
     zero-regression anchor) — loading from sub-libraries changes nothing.
  2. ``set_active_libraries`` actually shrinks/swaps the active vocabulary.
  3. The derived metadata (prop_status / position_class) matches the known
     per-source counts.

``set_active_libraries`` mutates module-global dicts, so the swap test restores
the default selection in a finally block to avoid polluting other tests.
"""
import csv
import json
import os

import pytest

from cycpep_master.paths import _map_utils as mu
from cycpep_master import build_sublibraries as builder

_LIB_DIR = mu._LIBRARIES_DIR
# The derived sub-library CSVs are git-ignored (regenerable via
# build_sublibraries.py); skip this suite when they are absent (fresh checkout),
# since the engine then falls back to the single unified CSV. The fallback
# itself is exercised by the rest of the test suite.
_SUBLIB_STEMS = (
    "caps", "core", "curated_cycpep", "nnaa_diverse", "helm_gpt", "special"
)
_have_libs = all(
    os.path.exists(os.path.join(_LIB_DIR, s + ".csv")) for s in _SUBLIB_STEMS)

pytestmark = pytest.mark.skipif(
    not _have_libs, reason="libraries/*.csv not generated (run build_sublibraries)")


def _read_sublib(stem):
    with open(os.path.join(_LIB_DIR, stem + ".csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _manifest():
    with open(os.path.join(_LIB_DIR, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


# ── Conservation: default load == full unified set ──────────────────────────

def test_sublibrary_row_conservation():
    """The three source slices must sum to the unified library row count."""
    n = sum(len(_read_sublib(s))
            for s in ("curated_cycpep", "nnaa_diverse", "helm_gpt"))
    with open(mu._UNIFIED_CSV, encoding="utf-8-sig") as f:
        n_unified = sum(1 for _ in csv.DictReader(f))
    assert n == n_unified == 13152


def test_default_manifest_preserves_unified_and_adds_declared_layers():
    """The source-scoped default preserves Unified and adds only declared
    caps/special symbols.

    Rebuilds both views fresh from their sources rather than reading the live
    dict, which other tests (e.g. Path G) may have injected extra symbols into.
    Restores the default selection afterwards.
    """
    saved = mu._unified_by_symbol
    try:
        # Manifest path: the exact source-scoped production selection.
        mu._unified_by_symbol = mu._load_from_sublibraries(_manifest()["load"])
        mu._build_monomer_dicts()
        manifest_keys = set(mu.monomers2smi_dict)
        # Legacy path: the single unified CSV.
        mu._unified_by_symbol = mu._load_unified_single_file()
        mu._build_monomer_dicts()
        legacy_keys = set(mu.monomers2smi_dict)
    finally:
        mu._unified_by_symbol = saved
        mu._build_monomer_dicts()
    declared_extra = set()
    for stem in ("caps", "core", "special"):
        declared_extra.update(mu._load_from_sublibraries([stem]))
    assert legacy_keys <= manifest_keys
    assert manifest_keys - legacy_keys == declared_extra - legacy_keys
    assert len(legacy_keys) == 13152


# ── set_active_libraries: selection actually takes effect ───────────────────

def test_set_active_libraries_shrinks_vocab():
    full = len(mu.monomers2smi_dict)
    try:
        n = mu.set_active_libraries(["core", "curated_cycpep"])
        assert n < full
        assert "A" in mu.monomers2smi_dict          # core AA present
        assert "dA" in mu.monomers2smi_dict          # curated present
        # an NNAA-only symbol must be gone
        assert "4EP" not in mu.monomers2smi_dict
    finally:
        mu.set_active_libraries(_manifest()["load"])
    assert len(mu.monomers2smi_dict) == full


def test_set_active_libraries_missing_raises():
    with pytest.raises(FileNotFoundError):
        mu.set_active_libraries(["does_not_exist"])


# ── Derived metadata matches known per-source counts ────────────────────────

def test_prop_status_counts():
    assert all(r["prop_status"] == "full" for r in _read_sublib("curated_cycpep"))
    assert all(r["prop_status"] == "struct_only" for r in _read_sublib("nnaa_diverse"))
    assert all(r["prop_status"] == "none" for r in _read_sublib("helm_gpt"))
    assert len(_read_sublib("curated_cycpep")) == 384
    assert len(_read_sublib("nnaa_diverse")) == 9998
    assert len(_read_sublib("helm_gpt")) == 2770


def test_position_class_values_valid():
    valid = {"flexible", "N_terminal_only", "C_terminal_only", "unknown"}
    for stem in ("curated_cycpep", "nnaa_diverse", "helm_gpt", "core", "special"):
        for r in _read_sublib(stem):
            assert r["position_class"] in valid


def test_manifest_default_loads_declared_layers():
    manifest = _manifest()
    assert manifest["load"] == [
        "caps",
        "core",
        "curated_cycpep",
        "nnaa_diverse",
        "helm_gpt",
        "special",
    ]
    assert manifest["training_vocab"] == ["core", "curated_cycpep"]


def test_builder_manifest_matches_checked_in_runtime_manifest():
    assert builder._DEFAULT_MANIFEST == _manifest()


def test_builder_reads_authoritative_core_and_caps_slices():
    core = builder._build_core()
    caps = builder._build_caps()

    assert len(core) == 20
    assert len({row["symbol"] for row in core}) == 20
    assert {row["symbol"] for row in caps} == {"ac", "nme", "nh2"}


def test_builder_regenerates_complete_manifest_in_isolated_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(builder, "_OUT_DIR", str(tmp_path))

    counts = builder.main()

    generated_manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert generated_manifest == builder._DEFAULT_MANIFEST
    assert counts["core"] == 20
    assert counts["caps"] == 3
    assert all(
        (tmp_path / f"{stem}.csv").is_file()
        for stem in generated_manifest["load"]
    )


def test_builder_validates_all_slices_before_creating_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "not-created"
    monkeypatch.setattr(builder, "_OUT_DIR", str(output))

    def invalid_core():
        raise ValueError("directed invalid core")

    monkeypatch.setattr(builder, "_build_core", invalid_core)
    with pytest.raises(ValueError, match="directed invalid core"):
        builder.main()
    assert not output.exists()
