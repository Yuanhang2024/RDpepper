"""User-facing monomer registration.

``add_monomer`` lets a user register a new monomer from a plain neutral SMILES
(e.g. a residue not covered by the unified library). The SMILES is run through
the package's CXSMILES engine (:mod:`cycpep_master.core.cxsmiles_gen`) to derive
the CXSMILES + R-group attachment points, injected into the live runtime
dictionaries so it is usable immediately, and (by default) persisted to
``user_monomer_library.csv`` next to the unified library so it survives restarts.

This is the programmatic counterpart to the interactive add-monomer prompt in
``get_smi_from_map`` — both ultimately call :func:`add_monomer`.
"""
import csv
import os
import warnings

from .cxsmiles_gen import gen_cxsmiles
from ..paths import _map_utils as _mu

# Single source of truth for the user library path (also read at module load
# time by _map_utils._load_user_monomers).
_USER_CSV = _mu._USER_CSV
_USER_COLS = ["symbol", "CXSMILES", "R1", "R2", "R3", "smiles_original"]
_ALLOWED_R_GROUP_DEFAULTS = {
    "R1": frozenset({"H"}),
    "R2": frozenset({"H", "OH", "NH2"}),
    "R3": frozenset({"-", "H", "OH", "SH"}),
}


def _read_user_library():
    """Return list of row dicts from user_monomer_library.csv (empty if none)."""
    if not os.path.exists(_USER_CSV):
        return []
    with open(_USER_CSV, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _append_user_library(row):
    """Append one row to user_monomer_library.csv, writing a header if new."""
    new_file = not os.path.exists(_USER_CSV)
    with open(_USER_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_USER_COLS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _add_monomer_unlocked(
    symbol,
    smiles,
    *,
    r1=None,
    r2=None,
    r3=None,
    overwrite=False,
    persist=True,
):
    """Register a new monomer from a neutral SMILES.

    symbol    : the monomer symbol (model token / HELM element). Must be unique
                unless overwrite=True.
    smiles    : neutral SMILES of the free monomer (e.g. 'CC(N)C(=O)O').
    r1/r2/r3  : optionally override the auto-inferred R-group leaving groups
                ('H' / 'OH' / '-').
    overwrite : replace an existing symbol instead of raising.
    persist   : also append to user_monomer_library.csv (default True). When
                False the monomer is runtime-only and lost on restart.

    Returns the registered record dict
    {symbol, CXSMILES, R1, R2, R3, smiles_original}.
    Raises ValueError if the SMILES yields no usable CXSMILES (no detectable
    backbone), or KeyError if the symbol exists and overwrite is False.
    """
    if not symbol or not str(symbol).strip():
        raise ValueError("symbol must be a non-empty string")
    symbol = str(symbol).strip()

    gen = gen_cxsmiles(smiles)
    cx = gen.get("CXSMILES", "")
    if not cx:
        raise ValueError(
            f"could not derive CXSMILES from SMILES {smiles!r} "
            f"(no peptide backbone detected); supply a monomer with an "
            f"N-Cα-C(=O)-OH backbone")

    # R-group values: explicit overrides win over the inferred ones.
    rg = {"R1": gen.get("R1", "H"), "R2": gen.get("R2", "OH"),
          "R3": gen.get("R3", "-")}
    if r1 is not None:
        rg["R1"] = r1
    if r2 is not None:
        rg["R2"] = r2
    if r3 is not None:
        rg["R3"] = r3

    for port, allowed in _ALLOWED_R_GROUP_DEFAULTS.items():
        value = str(rg[port]).strip().upper()
        if value not in allowed:
            raise ValueError(
                f"{port} default must be one of {sorted(allowed)}, got {rg[port]!r}"
            )
        rg[port] = value
        has_port = f"_{port}" in cx
        explicitly_requested = {"R1": r1, "R2": r2, "R3": r3}[port] is not None
        if value != "-" and not has_port:
            if explicitly_requested:
                raise ValueError(
                    f"{port} was supplied but the generated CXSMILES has no {port} port"
                )
            warnings.warn(
                f"generated chemistry suggested {port}={value}, but CXSMILES "
                f"contains no {port} port; registering {port} as unavailable",
                RuntimeWarning,
            )
            rg[port] = "-"

    record = {
        "symbol": symbol, "CXSMILES": cx,
        "R1": rg["R1"], "R2": rg["R2"], "R3": rg["R3"],
        "smiles_original": smiles,
    }
    casefold_collision = next(
        (
            existing for existing in _mu.monomers2smi_dict
            if existing.casefold() == symbol.casefold() and existing != symbol
        ),
        None,
    )
    if casefold_collision is not None:
        raise ValueError(
            f"monomer {symbol!r} conflicts with {casefold_collision!r} "
            "by case-insensitive identity"
        )
    if not overwrite and symbol in _mu.monomers2smi_dict:
        raise KeyError(
            f"monomer {symbol!r} already registered "
            "(pass overwrite=True to replace)"
        )
    runtime_smiles = _mu.get_smi_from_cxsmiles(cx)
    if not isinstance(runtime_smiles, str) or not runtime_smiles.strip():
        raise ValueError("generated CXSMILES has no usable runtime representation")
    previous_file = None
    file_existed = os.path.exists(_USER_CSV)
    if persist and file_existed:
        with open(_USER_CSV, "rb") as handle:
            previous_file = handle.read()
    if persist:
        _append_user_library(record)
    try:
        _mu.register_user_monomer_record(record, overwrite=overwrite)
    except Exception:
        if persist:
            if file_existed:
                with open(_USER_CSV, "wb") as handle:
                    handle.write(previous_file)
            elif os.path.exists(_USER_CSV):
                os.remove(_USER_CSV)
        raise
    return record


def add_monomer(symbol, smiles, *, r1=None, r2=None, r3=None,
                overwrite=False, persist=True):
    """Register and optionally persist one monomer as a serialized write."""
    with _mu._REGISTRY_LOCK:
        return _add_monomer_unlocked(
            symbol,
            smiles,
            r1=r1,
            r2=r2,
            r3=r3,
            overwrite=overwrite,
            persist=persist,
        )
