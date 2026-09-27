"""Tests for the user-tunable geometric distance-tolerance parameters.

Covers ``core/geometry_params`` (resolution + validation), the threaded
tunables through cyclization detection, Path A/E generation, and the public
application reconstruction entries.  All new parameters are keyword-only and
default to ``None`` (= the built-in defaults), so existing behaviour is
preserved.
"""
import pytest

from cycpep_master.core.cyclization import (
    detect_cyclization,
    geometric_crosslink_atom_pairs,
)
from cycpep_master.core.geometry_params import (
    DEFAULT_DISTANCE_CEILING,
    DEFAULT_RADIUS_MULTIPLIER,
    NONDEFAULT_GEOMETRY_PARAMS,
    is_nondefault_geometry,
    resolve_geometry_params,
)
from cycpep_master.paths import path_a


# ── resolve / validate ──

def test_resolve_defaults():
    assert resolve_geometry_params(None, None) == (
        DEFAULT_RADIUS_MULTIPLIER, DEFAULT_DISTANCE_CEILING
    )
    assert resolve_geometry_params() == (1.3, 3.0)


def test_resolve_explicit_values():
    assert resolve_geometry_params(1.5, 3.2) == (1.5, 3.2)
    assert resolve_geometry_params(None, 2.5) == (1.3, 2.5)
    assert isinstance(resolve_geometry_params(), tuple)


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_resolve_rejects_invalid_radius_multiplier(bad):
    with pytest.raises(ValueError):
        resolve_geometry_params(bad, None)


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_resolve_rejects_invalid_distance_ceiling(bad):
    with pytest.raises(ValueError):
        resolve_geometry_params(None, bad)


def test_is_nondefault_geometry():
    assert not is_nondefault_geometry(None, None)
    assert not is_nondefault_geometry(1.3, 3.0)
    assert is_nondefault_geometry(1.31, 3.0)
    assert is_nondefault_geometry(1.3, 3.01)


# ── cyclization behaviour ──

def _bond_signature(info):
    return sorted(
        (b.bond_type, b.pos1, b.pos2, b.rgroup1, b.rgroup2,
         b.atom1, b.atom2, b.res1, b.res2)
        for b in info.bonds
    )


def test_default_detection_matches_explicit_defaults(pdb_files):
    for pdb in pdb_files:
        default = detect_cyclization(pdb, "L")
        explicit = detect_cyclization(
            pdb, "L", radius_multiplier=1.3, distance_ceiling=3.0
        )
        assert default.topology == explicit.topology
        assert _bond_signature(default) == _bond_signature(explicit)
        assert default.warnings == []
        assert default.geometry_radius_multiplier is None
        assert default.geometry_distance_ceiling is None
        assert explicit.warnings == []


def test_geometry_evidence_monotone_in_radius_multiplier(pdb_files):
    counts = {}
    for multiplier in (1.05, 1.3, 1.5):
        counts[multiplier] = sum(
            len(geometric_crosslink_atom_pairs(pdb, "L", radius_multiplier=multiplier))
            for pdb in pdb_files
        )
    assert counts[1.5] >= counts[1.3] >= counts[1.05]
    # The parameter actually bites: a tight multiplier admits fewer bonds.
    tight = sum(
        len(geometric_crosslink_atom_pairs(pdb, "L", radius_multiplier=0.6))
        for pdb in pdb_files
    )
    assert tight < counts[1.3]


def test_nondefault_warning_and_fields(pdb_files):
    info = detect_cyclization(
        pdb_files[0], "L", radius_multiplier=1.5, distance_ceiling=3.2
    )
    assert info.warnings == [NONDEFAULT_GEOMETRY_PARAMS]
    assert info.geometry_radius_multiplier == 1.5
    assert info.geometry_distance_ceiling == 3.2


# ── Path A/E generation ──

def test_path_a_generate_with_radius_multiplier(pdb_files):
    result = path_a.generate(
        pdb_files[0], "L", geometric_cyclization=True, radius_multiplier=1.5
    )
    assert isinstance(result, tuple) and len(result) == 2

    _smiles, _error, evidence = path_a.generate_with_evidence(
        pdb_files[0], "L", geometric_cyclization=True, radius_multiplier=1.5
    )
    assert evidence["geometry_warnings"] == [NONDEFAULT_GEOMETRY_PARAMS]
    assert evidence["geometry_radius_multiplier"] == 1.5
    assert evidence["geometry_distance_ceiling"] == DEFAULT_DISTANCE_CEILING

    _smiles, _error, default_evidence = path_a.generate_with_evidence(
        pdb_files[0], "L", geometric_cyclization=True
    )
    assert default_evidence["geometry_warnings"] == []
    assert default_evidence["geometry_radius_multiplier"] == DEFAULT_RADIUS_MULTIPLIER


# ── application-layer penetration ──

def test_application_reconstruction_accepts_geometry_params(pdb_files):
    from cycpep_master import application

    result = application.reconstruct_structure(
        str(pdb_files[0]), radius_multiplier=1.4, distance_ceiling=3.2
    )
    assert isinstance(result, dict)
    assert "status" in result