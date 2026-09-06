"""Tests for the user-monomer admin API and interactive add hook (Part 2).

Covers:
  - add_monomer registers a monomer from a plain SMILES, makes it immediately
    usable in get_smi_from_map, and round-trips through user_monomer_library.csv;
  - the CXSMILES engine migrated into the package (core.cxsmiles_gen) matches
    the build_monomer_library back-compat alias;
  - the interactive add hook is triple-gated OFF by default (no input() under
    pytest / non-TTY), and when forced via a mocked TTY it registers and retries.

The tests redirect the user library to a temp path so the real
user_monomer_library.csv is never written.
"""
import builtins
import io
import os
import sys
import tempfile
import warnings

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd

from cycpep_master.core import monomer_admin as ma
from cycpep_master.core.cxsmiles_gen import gen_cxsmiles
from cycpep_master.paths import _map_utils as mu


@pytest.fixture
def temp_user_csv(tmp_path):
    """Point both the admin module and the loader at a throwaway CSV."""
    p = str(tmp_path / "user_monomers.csv")
    saved = ma._USER_CSV
    ma._USER_CSV = p
    mu._USER_CSV = p
    try:
        yield p
    finally:
        ma._USER_CSV = saved
        mu._USER_CSV = saved


def _formula(smi):
    return rdmd.CalcMolFormula(Chem.MolFromSmiles(smi)) if smi else None


# ── CXSMILES engine migration ───────────────────────────────────────────────

def test_cxsmiles_engine_in_package():
    """gen_cxsmiles auto-detects the backbone and emits R-groups."""
    r = gen_cxsmiles("CC(N)C(=O)O")  # alanine
    assert r["CXSMILES"]
    assert "_R1" in r["CXSMILES"] and "_R2" in r["CXSMILES"]
    assert (r["R1"], r["R2"]) == ("H", "OH")


def test_cxsmiles_no_backbone_returns_empty():
    """A molecule with no peptide backbone yields empty CXSMILES."""
    assert gen_cxsmiles("c1ccccc1")["CXSMILES"] == ""


# ── add_monomer API ─────────────────────────────────────────────────────────

def test_add_monomer_runtime_and_assembly(temp_user_csv):
    sym = "UTNva"  # norvaline chemistry, not in the library
    assert sym not in mu.monomers2smi_dict
    rec = ma.add_monomer(sym, "CCC[C@H](N)C(=O)O", persist=False)
    assert rec["symbol"] == sym
    assert sym in mu.monomers2smi_dict
    # usable immediately: Gly-UTNva-Ala = C10H19N3O4 (3 residues - 2 H2O)
    smi = mu.get_smi_from_map("G" + f"{{nnr:{sym}}}" + "A")
    assert _formula(smi) == "C10H19N3O4"


def test_add_monomer_persist_roundtrip(temp_user_csv):
    ma.add_monomer("UTLeu", "CC(C)C[C@H](N)C(=O)O", persist=True)
    assert os.path.exists(temp_user_csv)
    rows = ma._read_user_library()
    assert any(r["symbol"] == "UTLeu" and r["CXSMILES"] for r in rows)


def test_add_monomer_duplicate_raises(temp_user_csv):
    ma.add_monomer("UTDup", "CCC[C@H](N)C(=O)O", persist=False)
    with pytest.raises(KeyError):
        ma.add_monomer("UTDup", "CCC[C@H](N)C(=O)O", persist=False)
    # overwrite=True succeeds
    ma.add_monomer("UTDup", "CC(C)C[C@H](N)C(=O)O", persist=False, overwrite=True)


def test_add_monomer_no_backbone_raises(temp_user_csv):
    with pytest.raises(ValueError):
        ma.add_monomer("UTBad", "c1ccccc1", persist=False)


def test_add_monomer_rejects_unknown_rgroup_default_without_registration(
    temp_user_csv,
):
    symbol = "UTBadR2"
    with pytest.raises(ValueError, match="R2 default must be one of"):
        ma.add_monomer(
            symbol,
            "CCC[C@H](N)C(=O)O",
            r2="foo",
            persist=False,
        )
    assert symbol not in mu.monomers2smi_dict


def test_add_monomer_rejects_declared_port_absent_from_cxsmiles(temp_user_csv):
    symbol = "UTMissingR3"
    with pytest.raises(ValueError, match="generated CXSMILES has no R3 port"):
        ma.add_monomer(
            symbol,
            "CCC[C@H](N)C(=O)O",
            r3="OH",
            persist=False,
        )
    assert symbol not in mu.monomers2smi_dict


def test_add_monomer_persistence_failure_does_not_mutate_runtime(
    temp_user_csv, monkeypatch
):
    symbol = "UTPersistFailure"

    def fail_persistence(_row):
        raise OSError("read-only destination")

    monkeypatch.setattr(ma, "_append_user_library", fail_persistence)
    with pytest.raises(OSError, match="read-only destination"):
        ma.add_monomer(
            symbol,
            "CCC[C@H](N)C(=O)O",
            persist=True,
        )
    assert symbol not in mu.monomers2smi_dict


def test_inferred_unmaterialized_r3_is_warned_and_marked_unavailable(
    temp_user_csv,
):
    symbol = "UTCysLike"
    with pytest.warns(RuntimeWarning, match="contains no R3 port"):
        record = ma.add_monomer(
            symbol,
            "N[C@H](CS)C(=O)O",
            persist=False,
        )
    assert record["R3"] == "-"
    assert "R3" not in mu.monomers2r_groups_dict[symbol]


def test_direct_record_registration_rejects_invalid_port_without_mutation(
    temp_user_csv,
):
    symbol = "UTBypass"
    record = ma.add_monomer(
        symbol,
        "CCC[C@H](N)C(=O)O",
        persist=False,
    )
    record["R2"] = "foo"
    with pytest.raises(ValueError, match="R2 default must be one of"):
        mu.register_user_monomer_record(record, overwrite=True)
    assert mu.monomers2r_groups_dict[symbol]["R2"] == "OH"


def test_case_insensitive_symbol_collision_is_rejected(temp_user_csv):
    ma.add_monomer("UTCase", "CCC[C@H](N)C(=O)O", persist=False)
    with pytest.raises(ValueError, match="case-insensitive identity"):
        ma.add_monomer("utcase", "CC(C)C[C@H](N)C(=O)O", persist=False)


def test_runtime_registration_failure_rolls_back_persisted_row(
    temp_user_csv, monkeypatch
):
    symbol = "UTRuntimeFailure"

    def fail_registration(*_args, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(mu, "register_user_monomer_record", fail_registration)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        ma.add_monomer(
            symbol,
            "CCC[C@H](N)C(=O)O",
            persist=True,
        )
    assert not os.path.exists(temp_user_csv)


def test_valid_persistent_user_row_reloads_without_name_error(temp_user_csv):
    symbol = "UTReload"
    ma.add_monomer(
        symbol,
        "CCC[C@H](N)C(=O)O",
        persist=True,
    )
    mu.monomers2smi_dict.pop(symbol)
    mu.monomers2r_groups_dict.pop(symbol)
    mu._active_user_rows.pop(symbol)

    mu._load_user_monomers()

    assert symbol in mu.monomers2smi_dict
    assert mu._active_user_rows[symbol]["CXSMILES"]


# ── Interactive hook gating (default OFF) ───────────────────────────────────

def test_interactive_default_off_no_prompt():
    """Default get_smi_from_map must warn+None on unknown monomers and never
    call input() (so batch/pytest are unaffected)."""
    def _boom(*a, **k):
        raise AssertionError("input() must not be called by default")
    saved = builtins.input
    builtins.input = _boom
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = mu.get_smi_from_map("G{nnr:UTUnknownZ}A")
        assert r is None
        assert any("unknown monomer" in str(x.message) for x in w)
    finally:
        builtins.input = saved


def test_interactive_disabled_when_not_tty():
    """Even interactive=True is suppressed when stdin is not a TTY."""
    class _NoTTY(io.StringIO):
        def isatty(self):
            return False
    saved = sys.stdin
    sys.stdin = _NoTTY("")
    try:
        assert mu._interactive_enabled(True) is False
    finally:
        sys.stdin = saved


def test_interactive_add_and_retry(temp_user_csv):
    """With a mocked TTY and scripted input, the unknown monomer is registered
    under its bare symbol and assembly is retried successfully."""
    class _TTY(io.StringIO):
        def isatty(self):
            return True
    saved_stdin, saved_input = sys.stdin, builtins.input
    answers = iter(["CCC[C@H](N)C(=O)O"])
    sys.stdin = _TTY("")
    builtins.input = lambda *a, **k: next(answers)
    try:
        smi = mu.get_smi_from_map("G{nnr:UTIxn}A", interactive=True)
        assert _formula(smi) == "C10H19N3O4"
        assert "UTIxn" in mu.monomers2smi_dict          # bare symbol
        assert "{nnr:UTIxn}" not in mu.monomers2smi_dict  # not the wrapped form
    finally:
        sys.stdin, builtins.input = saved_stdin, saved_input
