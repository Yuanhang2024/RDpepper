"""Tests for the conformer-ensemble size guard (export/conformer.py).

These do NOT exercise full 3D embedding (slow); they only verify that the
heavy-atom guard refuses oversized inputs cleanly instead of hanging, and that
the override parameter works.
"""
import pytest

from cycpep_master.export.conformer import (
    compute_conformer_ensemble_stats, STABILITY_MAX_HEAVY_ATOMS,
)


def test_oversized_molecule_is_refused():
    """A molecule above the heavy-atom ceiling returns (None, error), no hang."""
    big = "C" * (STABILITY_MAX_HEAVY_ATOMS + 50)  # long alkane, far over the limit
    stats, err = compute_conformer_ensemble_stats(big, num_confs=2)
    assert stats is None
    assert err is not None
    assert "too large" in err


def test_guard_override_lets_small_molecule_through():
    """Lowering max_heavy_atoms below a small molecule's size triggers the guard."""
    stats, err = compute_conformer_ensemble_stats(
        "CC(N)C(=O)O", num_confs=2, max_heavy_atoms=2)
    assert stats is None
    assert "too large" in err


def test_invalid_smiles_reports_error():
    stats, err = compute_conformer_ensemble_stats("not_a_smiles", num_confs=2)
    assert stats is None
    assert err is not None
