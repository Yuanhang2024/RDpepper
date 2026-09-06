from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pytest

from cycpep_master import application
from cycpep_master.docking.torsion_prior import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    TorsionPriorError,
    clear_torsion_prior_cache,
    load_torsion_prior,
    torsion_prior_cache_info,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    application.clear_caches()
    yield
    application.clear_caches()


def _prior_files(root: Path) -> tuple[Path, Path]:
    runtime = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "levels": {
            "exact": {
                "fixture": {
                    "rigidity_score": 0.9,
                    "confidence": "high",
                }
            },
            "residue_class_ring": {},
            "morgan": {},
            "generic": {},
        },
    }
    runtime_path = root / "torsion_priors_runtime.json"
    runtime_path.write_text(
        json.dumps(runtime, sort_keys=True), encoding="utf-8"
    )
    manifest_path = root / "torsion_prior_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "runtime_sha256": hashlib.sha256(
                runtime_path.read_bytes()
            ).hexdigest(),
        }, sort_keys=True),
        encoding="utf-8",
    )
    return runtime_path, manifest_path


def test_torsion_prior_warm_cache_is_single_flight_and_immutable(
    tmp_path,
):
    runtime, manifest = _prior_files(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        indexes = list(pool.map(
            lambda _index: load_torsion_prior(
                runtime, manifest_path=manifest
            ),
            range(8),
        ))

    assert len({id(index) for index in indexes}) == 1
    assert torsion_prior_cache_info() == {
        "entries": 1,
        "hits": 7,
        "misses": 1,
    }
    index = indexes[0]
    with pytest.raises(TypeError):
        index.levels["exact"] = {}
    with pytest.raises(TypeError):
        index.levels["exact"]["fixture"]["rigidity_score"] = 0.0


def test_cache_invalidation_rechecks_changed_resource(tmp_path):
    runtime, manifest = _prior_files(tmp_path)
    first = load_torsion_prior(runtime, manifest_path=manifest)
    runtime.write_text(
        runtime.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TorsionPriorError, match="SHA-256"):
        load_torsion_prior(runtime, manifest_path=manifest)

    assert first.runtime_sha256 != hashlib.sha256(
        runtime.read_bytes()
    ).hexdigest()


def test_uncached_strict_load_does_not_reuse_warm_entry(tmp_path):
    runtime, manifest = _prior_files(tmp_path)
    cached = load_torsion_prior(runtime, manifest_path=manifest)
    strict = load_torsion_prior(
        runtime,
        manifest_path=manifest,
        use_cache=False,
    )

    assert strict is not cached
    assert strict.runtime_sha256 == cached.runtime_sha256
    assert torsion_prior_cache_info() == {
        "entries": 1,
        "hits": 0,
        "misses": 1,
    }


def test_repeated_capabilities_reuses_prior_and_resource_hashes(
    monkeypatch,
):
    original = application._sha256_file
    calls = []

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(application, "_sha256_file", counted)
    first = application.capabilities()
    first_call_count = len(calls)
    second = application.capabilities()

    assert first["data"]["version"] == "7.1.0"
    assert second["data"]["version"] == "7.1.0"
    assert first_call_count >= 5
    assert len(calls) == first_call_count
    assert torsion_prior_cache_info() == {
        "entries": 1,
        "hits": 1,
        "misses": 1,
    }

