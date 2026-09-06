"""Special residue library: PDB-code -> monomer mapping for special chemistries.

The unified monomer library (10k+ entries, indexed by HELM symbol) does not
cover special residues that appear in PDB structures under three-letter PDB
chemical-component codes (e.g. stapled-peptide residues 0EH/MK8, lanthionine
precursors DHA/DBB, depsipeptide residues KYN/DAL). This module loads an
independent table keyed by PDB code and exposes the lookups Paths G/H need:

  - pdb_code  -> library symbol (alias)
  - pdb_code  -> CXSMILES monomer fragment (with [*]+_R1/_R2/_R3 labels)
  - pdb_code  -> R-group caps (H / OH / -) and the R3 cross-link anchor atom

It does NOT modify the unified library; callers consult it as a fallback when
a residue is absent there. R-group convention matches the unified library:
R1 = backbone N-terminus, R2 = backbone C-terminus, R3 = side-chain cross-link
point (anchored to the real PDB atom name in ``r3_atom``).
"""
import csv
import os

_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'special_residue_library.csv')

# pdb_code -> dict(symbol, cxsmiles, r1, r2, r3, r3_atom, natural_analog,
#                  chemistry, source)
_DB = {}
_LOADED = False


def _load():
    global _LOADED
    if _LOADED:
        return
    if os.path.exists(_CSV):
        with open(_CSV, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                code = (row.get('pdb_code') or '').strip()
                if code:
                    _DB[code] = {k: (v or '').strip() for k, v in row.items()}
    _LOADED = True


def is_special_residue(pdb_code):
    """True if this PDB three-letter code is a registered special residue."""
    _load()
    return pdb_code in _DB


def get_symbol(pdb_code):
    """Library symbol (alias) for a PDB code, or None."""
    _load()
    rec = _DB.get(pdb_code)
    return rec['symbol'] if rec else None


def get_cxsmiles(pdb_code):
    """CXSMILES monomer fragment for a PDB code, or None."""
    _load()
    rec = _DB.get(pdb_code)
    return rec['cxsmiles'] if rec else None


def get_rgroups(pdb_code):
    """(r1, r2, r3) cap values for a PDB code, or None."""
    _load()
    rec = _DB.get(pdb_code)
    if not rec:
        return None
    return rec['r1'], rec['r2'], rec['r3']


def get_r3_atom(pdb_code):
    """Side-chain cross-link anchor atom name (e.g. 'CAT'), or None."""
    _load()
    rec = _DB.get(pdb_code)
    return rec.get('r3_atom') if rec else None


def get_record(pdb_code):
    """Full record dict for a PDB code, or None."""
    _load()
    return _DB.get(pdb_code)


def all_codes():
    """Set of all registered PDB codes."""
    _load()
    return set(_DB)
