"""Path B helper: HELM ↔ MAP ↔ SMILES conversion.

Uses unified_monomer_library.csv as the single source of truth
for all monomer data (CXSMILES, R1/R2/R3 attachment points).
"""

import os
import re
import copy
import csv as _csv
import json as _json
import hashlib
import math
import threading
import warnings
from contextlib import contextmanager
from functools import lru_cache
from rdkit import Chem, RDLogger


@contextmanager
def _suppress_rdkit_warnings():
    """Silence RDKit's C++ logger only within this block.

    The monomer library and CXSMILES round-trips legitimately produce many
    benign RDKit warnings (e.g. dummy-atom valence). Suppress them locally
    instead of globally, so real warnings from calling code are preserved.
    """
    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


# ── Data paths ────────────────────────────────────────────────────────────

_PATHS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PATHS_DIR)
_UNIFIED_CSV = os.path.join(_PROJECT_DIR, "unified_monomer_library.csv")
_DERIVED_CSV = os.path.join(_PROJECT_DIR, "derived_monomer_library.csv")
_DERIVED_MANIFEST = os.path.join(_PROJECT_DIR, "derived_monomer_manifest.json")
_FULL_UNIFIED_REFERENCE_ROWS = None


_REGISTRY_EPOCH = 0


def _invalidate_exact_v1_registry_cache():
    """Invalidate exact_v1 lazily without creating an import cycle.

    This is the shared hub called by every registry-state mutation site
    (registration, rebuild, user records, PDB aliases, snapshot restore),
    so it also bumps the registry epoch that memoized path-assembly caches
    (combo/tokenizer) embed in their keys.
    """
    global _REGISTRY_EPOCH
    _REGISTRY_EPOCH += 1
    import sys

    module = sys.modules.get("cycpep_master.exact_v1")
    reset = (
        getattr(module, "reset_exact_v1_registry_cache", None)
        if module is not None
        else None
    )
    if callable(reset):
        reset()


# ── Monomer database (unified library, optionally sliced into sub-libraries) ─

_LIBRARIES_DIR = os.path.join(_PROJECT_DIR, "libraries")
_MANIFEST = os.path.join(_LIBRARIES_DIR, "manifest.json")


def _load_unified_single_file():
    """Fallback loader: read the whole unified library CSV (legacy path)."""
    by_symbol = {}
    with open(_UNIFIED_CSV, "r", encoding="utf-8-sig") as _f:
        for _row in _csv.DictReader(_f):
            _sym = str(_row.get("symbol", "")).strip()
            if _sym:
                by_symbol[_sym] = _row
    return by_symbol


def _full_unified_reference_rows():
    """Return an immutable-by-convention cache of the complete Unified file."""
    global _FULL_UNIFIED_REFERENCE_ROWS
    if _FULL_UNIFIED_REFERENCE_ROWS is None:
        _FULL_UNIFIED_REFERENCE_ROWS = _load_unified_single_file()
    return _FULL_UNIFIED_REFERENCE_ROWS


def _unified_fieldnames():
    """Return the exact, case-sensitive 235-column Unified schema."""
    with open(_UNIFIED_CSV, "r", encoding="utf-8-sig", newline="") as _f:
        reader = _csv.reader(_f)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise ValueError("Unified monomer library has no header") from exc


def _row_graph_identity(row):
    """Return a strict molecular identity for duplicate/override checks."""
    smiles = next(
        (
            str(row.get(name, "")).strip()
            for name in (
                "smiles_canonical", "smiles_original", "replaced_SMILES"
            )
            if str(row.get(name, "")).strip()
        ),
        "",
    )
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        raise ValueError(
            f"derived monomer {row.get('symbol')!r} lacks a parseable full-monomer SMILES"
        )
    return Chem.MolToInchiKey(molecule)


def _validate_derived_rows(rows, base_rows, *, require_exact_schema=True):
    """Validate append-only derived rows without mutating the live registry."""
    schema = _unified_fieldnames()
    full_unified_rows = _full_unified_reference_rows()
    base_symbols = set(full_unified_rows) | set(base_rows)
    base_symbols_folded = {
        str(symbol).casefold(): symbol for symbol in base_symbols
    }
    base_ids = {
        int(str(row.get("monomer_id", "")).strip())
        for row in [*full_unified_rows.values(), *base_rows.values()]
        if str(row.get("monomer_id", "")).strip().lstrip("+-").isdigit()
    }
    base_graphs = {
        _row_graph_identity(row): symbol
        for symbol, row in [
            *full_unified_rows.items(), *base_rows.items()
        ]
        if any(str(row.get(name, "")).strip() for name in (
            "smiles_canonical", "smiles_original", "replaced_SMILES"
        ))
    }
    validated = {}
    ids = set()
    folded_symbols = {}
    graphs = {}
    for row_number, raw in enumerate(rows, start=2):
        row = dict(raw)
        if require_exact_schema and tuple(row) != schema:
            raise ValueError(
                f"derived monomer row {row_number} does not use the exact Unified schema"
            )
        symbol = str(row.get("symbol", "")).strip()
        monomer_id = str(row.get("monomer_id", "")).strip()
        if not symbol or not monomer_id:
            raise ValueError(f"derived monomer row {row_number} lacks symbol or monomer_id")
        folded = symbol.casefold()
        if folded in base_symbols_folded:
            existing = base_symbols_folded[folded]
            raise ValueError(
                f"derived monomer symbol {symbol!r} would override existing "
                f"symbol {existing!r} by case-insensitive identity"
            )
        if folded in folded_symbols:
            raise ValueError(
                f"derived monomer symbol {symbol!r} duplicates "
                f"{folded_symbols[folded]!r} by case-insensitive identity"
            )
        if symbol in base_symbols or symbol in validated:
            raise ValueError(f"derived monomer symbol {symbol!r} would override an existing row")
        try:
            numeric_id = int(monomer_id)
        except ValueError as exc:
            raise ValueError(f"derived monomer_id {monomer_id!r} is not an integer") from exc
        if numeric_id in base_ids:
            raise ValueError(
                f"derived monomer_id {monomer_id!r} conflicts with Unified"
            )
        if numeric_id in ids:
            raise ValueError(f"duplicate derived monomer_id {monomer_id!r}")
        if str(row.get("source", "")).strip() != "local_structure_derived":
            raise ValueError(
                f"derived monomer {symbol!r} has non-derived source {row.get('source')!r}"
            )
        cxsmiles = str(row.get("CXSMILES", "")).strip()
        if not cxsmiles or Chem.MolFromSmiles(cxsmiles) is None:
            raise ValueError(f"derived monomer {symbol!r} has invalid CXSMILES")
        if "_R1" not in cxsmiles or "_R2" not in cxsmiles:
            raise ValueError(f"derived monomer {symbol!r} lacks explicit R1/R2 ports")
        if not str(row.get("R1", "")).strip() or not str(row.get("R2", "")).strip():
            raise ValueError(f"derived monomer {symbol!r} lacks R1/R2 defaults")
        graph = _row_graph_identity(row)
        canonical = Chem.MolToSmiles(
            Chem.MolFromSmiles(str(row["smiles_canonical"])),
            canonical=True,
            isomericSmiles=True,
        )
        if canonical != str(row["smiles_canonical"]).strip():
            raise ValueError(f"derived monomer {symbol!r} has noncanonical SMILES")
        nonstereo = Chem.MolToSmiles(
            Chem.MolFromSmiles(str(row["smiles_canonical"])),
            canonical=True,
            isomericSmiles=False,
        )
        if nonstereo != str(row.get("smiles_canonical_nostereo", "")).strip():
            raise ValueError(
                f"derived monomer {symbol!r} has inconsistent nonstereo SMILES"
            )
        for field in schema[20:-7]:
            value = str(row.get(field, "")).strip()
            try:
                finite = math.isfinite(float(value))
            except ValueError:
                finite = False
            if not finite:
                raise ValueError(
                    f"derived monomer {symbol!r} descriptor {field!r} is missing or nonfinite"
                )
        if graph in base_graphs:
            raise ValueError(
                f"derived monomer {symbol!r} duplicates Unified graph {base_graphs[graph]!r}; "
                "record a PDB alias in the derived manifest instead"
            )
        if graph in graphs:
            raise ValueError(
                f"derived monomer {symbol!r} duplicates derived graph {graphs[graph]!r}"
            )
        ids.add(numeric_id)
        folded_symbols[folded] = symbol
        graphs[graph] = symbol
        validated[symbol] = row
    return validated


def _load_derived_rows(base_rows):
    """Load the strict append-only derived layer, if present."""
    if not os.path.exists(_DERIVED_CSV):
        return {}
    with open(_DERIVED_CSV, "r", encoding="utf-8-sig", newline="") as _f:
        reader = _csv.DictReader(_f)
        if tuple(reader.fieldnames or ()) != _unified_fieldnames():
            raise ValueError("derived_monomer_library.csv schema differs from Unified")
        rows = list(reader)
        if not rows:
            return {}
        return _validate_derived_rows(rows, base_rows, require_exact_schema=True)


def _manifest_graph_identity(row, *, label, full_rows=None):
    """Recompute the identity bound by an overlay manifest record."""
    smiles = str(row.get("smiles_canonical", "")).strip()
    if not smiles and full_rows is not None:
        symbol = str(row.get("symbol", "")).strip()
        full_row = full_rows.get(symbol, {})
        smiles = str(full_row.get("smiles_canonical", "")).strip()
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        raise ValueError(f"{label} has no valid canonical SMILES")
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey:
        raise ValueError(f"{label} has no computable full InChIKey")
    return {
        "graph_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "full_inchikey": inchikey,
    }


def _load_derived_aliases(all_rows, derived_rows):
    """Load hash-bound PDB aliases from the derived manifest."""
    if not os.path.exists(_DERIVED_MANIFEST):
        if derived_rows:
            raise ValueError("derived rows exist without derived_monomer_manifest.json")
        return {}
    with open(_DERIVED_MANIFEST, "r", encoding="utf-8") as handle:
        manifest = _json.load(handle)
    expected_hash = str(manifest.get("unified_library_sha256", "")).lower()
    with open(_UNIFIED_CSV, "rb") as handle:
        observed_hash = hashlib.sha256(handle.read()).hexdigest()
    if expected_hash != observed_hash:
        raise ValueError("derived monomer manifest does not bind the active Unified library")
    entries = manifest.get("entries", [])
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise ValueError("derived monomer manifest entries are malformed")
    manifest_symbols = {
        str(row.get("symbol", "")).strip()
        for row in entries
        if str(row.get("symbol", "")).strip()
    }
    if len(manifest_symbols) != len(entries):
        raise ValueError("derived monomer manifest symbols are missing or duplicated")
    if manifest_symbols != set(derived_rows):
        raise ValueError("derived monomer manifest entries do not match derived rows")
    for entry in entries:
        symbol = str(entry["symbol"]).strip()
        observed = _manifest_graph_identity(
            derived_rows[symbol], label=f"derived monomer {symbol!r}"
        )
        for field, value in observed.items():
            if str(entry.get(field, "")).strip() != value:
                raise ValueError(
                    f"derived monomer manifest {field} does not match {symbol!r}"
                )
    aliases = {}
    full_rows = None
    raw_aliases = manifest.get("pdb_aliases", [])
    if not isinstance(raw_aliases, list) or any(
        not isinstance(row, dict) for row in raw_aliases
    ):
        raise ValueError("derived PDB aliases are malformed")
    for index, raw in enumerate(raw_aliases, start=1):
        pdb_resname = str(raw.get("pdb_resname", "")).strip().upper()
        target = str(raw.get("target_symbol", "")).strip()
        if not pdb_resname or not target:
            raise ValueError(f"derived PDB alias {index} lacks pdb_resname or target_symbol")
        if target not in all_rows:
            raise ValueError(
                f"derived PDB alias {pdb_resname!r} targets unknown symbol {target!r}"
            )
        if not str(all_rows[target].get("smiles_canonical", "")).strip():
            full_rows = full_rows or _load_unified_single_file()
        observed = _manifest_graph_identity(
            all_rows[target],
            label=f"derived PDB alias target {target!r}",
            full_rows=full_rows,
        )
        for field, value in observed.items():
            if str(raw.get(field, "")).strip() != value:
                raise ValueError(
                    f"derived PDB alias {pdb_resname!r} {field} does not match "
                    f"target {target!r}"
                )
        previous = aliases.get(pdb_resname)
        if previous is not None and previous != target:
            raise ValueError(f"conflicting derived PDB alias {pdb_resname!r}")
        aliases[pdb_resname] = target
    return aliases


def _load_unified_with_reconstruction_rows():
    """Load the legacy Unified file with mandatory A-family rows overlaid.

    Generated source slices are optional in a clean checkout.  The tracked
    core and caps slices remain authoritative for standard residues and caps,
    while the single Unified file supplies the rest of the vocabulary.
    """
    by_symbol = _load_from_sublibraries(["caps", "core"])
    if by_symbol is None:
        return _load_unified_single_file()
    for symbol, row in _load_unified_single_file().items():
        by_symbol.setdefault(symbol, row)
    return by_symbol


def _load_from_sublibraries(stems):
    """Load the named sub-library CSVs under libraries/ into a symbol dict.

    Earlier stems win on symbol collisions (so the manifest order sets
    precedence). Returns {} if a stem file is missing so the caller can fall
    back to the single-file reader.
    """
    by_symbol = {}
    for _stem in stems:
        _path = os.path.join(_LIBRARIES_DIR, _stem + ".csv")
        if not os.path.exists(_path):
            return None
        with open(_path, "r", encoding="utf-8-sig") as _f:
            for _row in _csv.DictReader(_f):
                _sym = str(_row.get("symbol", "")).strip()
                if _sym and _sym not in by_symbol:
                    by_symbol[_sym] = _row
    return by_symbol


def _read_manifest():
    """Return the manifest dict, or None if absent/unreadable."""
    if not os.path.exists(_MANIFEST):
        return None
    try:
        with open(_MANIFEST, "r", encoding="utf-8") as _f:
            return _json.load(_f)
    except Exception:
        return None


# Build _unified_by_symbol: prefer the manifest-selected sub-libraries, fall
# back to the single unified CSV when libraries/ is absent (old checkouts).
_manifest = _read_manifest()
_unified_by_symbol: dict = {}
if _manifest and _manifest.get("load"):
    _sliced = _load_from_sublibraries(_manifest["load"])
    _unified_by_symbol = _sliced if _sliced is not None \
        else _load_unified_with_reconstruction_rows()
else:
    _unified_by_symbol = _load_unified_with_reconstruction_rows()

# The derived layer is append-only and validated before it can affect any path.
_persistent_derived_by_symbol = _load_derived_rows(_unified_by_symbol)
_unified_by_symbol.update(_persistent_derived_by_symbol)
_persistent_pdb_aliases = _load_derived_aliases(
    _unified_by_symbol, _persistent_derived_by_symbol
)
_active_pdb_aliases = dict(_persistent_pdb_aliases)


def persistent_overlay_audit():
    """Measure the active and on-disk persistent overlay without caller claims."""
    disk_rows = 0
    if os.path.exists(_DERIVED_CSV):
        with open(_DERIVED_CSV, "r", encoding="utf-8-sig", newline="") as handle:
            reader = _csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _unified_fieldnames():
                raise ValueError("derived_monomer_library.csv schema differs from Unified")
            disk_rows = sum(1 for _ in reader)
    disk_aliases = 0
    disk_entries = 0
    disk_entry_symbols = []
    disk_alias_state = []
    if os.path.exists(_DERIVED_MANIFEST):
        with open(_DERIVED_MANIFEST, "r", encoding="utf-8") as handle:
            manifest = _json.load(handle)
        aliases = manifest.get("pdb_aliases", [])
        entries = manifest.get("entries", [])
        if (
            not isinstance(aliases, list)
            or any(not isinstance(row, dict) for row in aliases)
            or not isinstance(entries, list)
            or any(not isinstance(row, dict) for row in entries)
        ):
            raise ValueError("derived PDB aliases are malformed")
        disk_aliases = len(aliases)
        disk_entries = len(entries)
        disk_entry_symbols = sorted(
            str(row.get("symbol", "")).strip() for row in entries
        )
        disk_alias_state = sorted(
            ({
                "pdb_resname": str(row.get("pdb_resname", "")).strip().upper(),
                "target_symbol": str(row.get("target_symbol", "")).strip(),
            } for row in aliases),
            key=lambda row: (row["pdb_resname"], row["target_symbol"]),
        )
    memory_rows = len(_persistent_derived_by_symbol)
    memory_aliases = len(_persistent_pdb_aliases)
    memory_alias_state = [
        {"pdb_resname": name, "target_symbol": target}
        for name, target in sorted(_persistent_pdb_aliases.items())
    ]
    state = {
        "entries": [
            {
                "symbol": symbol,
                **_manifest_graph_identity(
                    row, label=f"persistent derived monomer {symbol!r}"
                ),
                "row_sha256": hashlib.sha256(_json.dumps(
                    {key: str(value) for key, value in sorted(row.items())},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
            }
            for symbol, row in sorted(_persistent_derived_by_symbol.items())
        ],
        "aliases": memory_alias_state,
    }
    state_payload = _json.dumps(
        state, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    empty = not any((
        disk_rows, disk_entries, disk_aliases, memory_rows, memory_aliases
    ))
    artifacts = {}
    for name, path in (
        ("derived_monomer_library.csv", _DERIVED_CSV),
        ("derived_monomer_manifest.json", _DERIVED_MANIFEST),
    ):
        if os.path.exists(path):
            with open(path, "rb") as handle:
                payload = handle.read()
            artifacts[name] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
    return {
        "status": "empty" if empty else "nonempty",
        "disk_derived_row_count": disk_rows,
        "disk_pdb_alias_count": disk_aliases,
        "disk_manifest_entry_count": disk_entries,
        "memory_derived_row_count": memory_rows,
        "memory_pdb_alias_count": memory_aliases,
        "disk_memory_count_consistent": (
            disk_rows == memory_rows
            and disk_entries == memory_rows
            and disk_aliases == memory_aliases
        ),
        "disk_memory_semantic_consistent": (
            disk_entry_symbols == sorted(_persistent_derived_by_symbol)
            and disk_alias_state == memory_alias_state
        ),
        "state_sha256": hashlib.sha256(state_payload).hexdigest(),
        "state": state,
        "artifacts": artifacts,
    }


def resolve_pdb_alias(pdb_resname: str) -> str | None:
    """Return an active entity-local or persistent PDB-code alias."""
    return _active_pdb_aliases.get(str(pdb_resname).strip().upper())


def _auto_map_denotion(symbol: str) -> str:
    """Generate MAP notation for any monomer symbol.

    Standard AA -> single letter.
    All others -> {nnr:<symbol>}.
    """
    s = str(symbol).strip()
    if len(s) == 1:
        return s
    return f"{{nnr:{s}}}"


# ════════════════════════════════════════════════════════════════════════════
# CXSMILES / RGroups utilities
# (code under MIT licence Copyright (c) 2021-2024 Charles Xu and others)
# ════════════════════════════════════════════════════════════════════════════


def relabel_rgroup2index(smi):
    """Input: SMILES with _R groups -> SMILES with numeric :1, :2 groups."""
    r_group_name = re.findall(r'\[\*\:(_R\d)\]', smi)
    r_group_name = list(r_group_name)

    checked_rgroup = []
    for name in r_group_name:
        if name in checked_rgroup:
            smi = smi.replace(name, f'{int(name[2:])+1}', 1)
            name = f'_R{int(name[2:])+1}'
        else:
            smi = smi.replace(name, f'{name[2:]}', 1)
        checked_rgroup.append(name)
    return smi


def relabel_rgroup2label(smi):
    """Input: SMILES with numeric groups -> SMILES with _R groups."""
    r_group_name = re.findall(r'\[(\*\:\d)\]', smi)
    r_group_name = list(r_group_name)
    for name in r_group_name:
        smi = smi.replace(name, f'*:_R{name[2:]}')
    return smi


def get_smi_from_cxsmiles(cxsmiles):
    """Get SMILES from CXSMILES format."""
    smi_list = cxsmiles.split('|')
    smi, pos = smi_list[0], smi_list[1]
    labels = pos.split('$')[1].split(';')

    smi = re.sub(r'(?<!\[)\*(?!\])', '[*]', smi)

    for label in labels:
        if len(label) == 0:
            continue
        index = smi.index('[*]')
        smi = smi[:index] + f'[*:{label}]' + smi[index+3:]

    return smi.strip()


def get_cxsmiles_from_smi(smi):
    """Get CXSMILES from SMILES format."""
    cxsmiles = smi
    labels = re.findall(r'\[\*\:(.*?)\]', smi)
    r_groups = []
    for label in labels:
        cxsmiles = cxsmiles.replace(f'[*:{label}]', '[*]')
        r_groups.append(f'{label}')

    pos = list()
    r_group_idx = 0
    for i in range(len(cxsmiles)):
        if cxsmiles[i] in ('H', '@', '[', ']', '(', ')', '=', '-', '#', ':', '+',
                           '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '/', '\\',
                           'l', 'r', '.'):
            # '.' (fragment separator) is not an atom — must be skipped so the
            # per-atom label positions stay aligned in multi-fragment SMILES
            # (e.g. independent chains joined only by a side-chain crosslink).
            continue
        elif cxsmiles[i] == '*':
            if r_group_idx >= len(r_groups):
                # Bare [*] without matching R-group label — use 'DU' placeholder
                pos.append("DU")
            else:
                pos.append(f"{r_groups[r_group_idx]}")
                r_group_idx += 1
        else:
            pos.append('')
    pos = '|$' + ';'.join(pos) + '$|'
    return f'{cxsmiles} {pos}'


def replace_unused_r_groups(mol_smi, r_groups: dict, used_r_groups: list):
    """Cap unused R-group dummy atoms; preserve used ones.

    For used R3 groups that lack an explicit dummy atom (e.g. the
    side-chain COOH is part of the monomer SMILES), inject a
    dummy atom so that downstream cyclization can locate it.
    """
    # ── Inject missing R3 dummies before index‑based map lookup ──
    mol_smi_idx = relabel_rgroup2index(mol_smi)
    for r_group in r_groups.keys():
        if r_group not in used_r_groups:
            continue
        if r_group in ('R1', 'R2'):
            continue
        map_num = int(r_group[1:])
        if f'[*:{map_num}]' in mol_smi_idx:
            continue
        cap = r_groups[r_group]
        mol_smi_idx = _inject_sidechain_dummy(mol_smi_idx, map_num, cap)

    # ── Cap unused dummies by their library cap value ──
    # cap=='OH' (free backbone C-terminus R2, or free side-chain carboxyl R3 of
    # Asp/Glu): the dummy must become a hydroxyl O so the group is a carboxylic
    # acid C(=O)O — NOT deleted, which would leave an aldehyde C=O (an ADMET
    # toxicity alert and wrong chemistry). cap=='H' (free N-terminus R1, free
    # amine/thiol R3): delete the dummy, leaving the implicit H.
    mol = Chem.MolFromSmiles(mol_smi_idx)
    if mol is None:
        return mol_smi
    rw = Chem.RWMol(mol)
    to_delete = []
    for r_group in r_groups.keys():
        if r_group in used_r_groups:
            continue
        map_num = int(r_group[1:])
        cap = r_groups[r_group]
        for atom in rw.GetAtoms():
            if atom.GetAtomMapNum() == map_num:
                if cap in {'OH', 'NH2', 'SH'}:
                    atom.SetAtomicNum({'OH': 8, 'NH2': 7, 'SH': 16}[cap])
                    atom.SetAtomMapNum(0)
                    atom.SetNoImplicit(False)
                else:
                    neighbors = list(atom.GetNeighbors())
                    if len(neighbors) == 1:
                        neighbor = neighbors[0]
                        if neighbor.GetNoImplicit() and neighbor.GetNumExplicitHs() > 0:
                            neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
                    to_delete.append(atom.GetIdx())
                break
    for idx in sorted(to_delete, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass
    return relabel_rgroup2label(Chem.MolToSmiles(rw))


# ── Sidechain dummy injection ────────────────────────────────────────────

def _inject_sidechain_dummy(mol_smi: str, map_num: int, cap: str) -> str:
    """Inject an explicit dummy atom at the side-chain functional group.

    cap == 'OH' → locate COOH and replace the hydroxyl OH with [*:map_num]
    cap == 'H'  → locate aliphatic NH2/OH/SH and replace one H with [*:map_num]
    """
    mol = Chem.MolFromSmiles(mol_smi)
    if mol is None:
        return mol_smi

    if cap == 'OH':
        patt = Chem.MolFromSmarts('[C;!R](=[O;!R])-[O;!R]')
        matches = mol.GetSubstructMatches(patt)
        if matches:
            o_idx = matches[0][2]
            # Only replace if this O has an implicit H (hydroxyl, not carboxylate)
            if mol.GetAtomWithIdx(o_idx).GetTotalNumHs() != 1:
                pass
            else:
                rw = Chem.RWMol(mol)
                rw.ReplaceAtom(o_idx, Chem.Atom(0))
                rw.GetAtomWithIdx(o_idx).SetAtomMapNum(map_num)
                return Chem.MolToSmiles(rw)

    elif cap == 'H':
        for smarts in ['[N;!R;H2]-[C;!R]', '[O;!R;H1]-[C;!R]', '[S;!R;H1]-[C;!R]']:
            patt = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(patt)
            if matches:
                n_idx = matches[0][0]
                atom = mol.GetAtomWithIdx(n_idx)
                n_h = atom.GetTotalNumHs()
                if n_h < 1:
                    continue
                # Replace one implicit H with a dummy — add explicit H first
                rw = Chem.RWMol(mol)
                # Set explicit H count to n_h (makes them explicit), then replace one
                atom_rw = rw.GetAtomWithIdx(n_idx)
                atom_rw.SetNumExplicitHs(n_h)
                # Now we need to find an explicit H neighbor...
                # Simpler: just replace the N/O/S itself? No, that loses the atom.
                # Better: keep mol as-is, add dummy at map_num bound to this atom
                # by using the implicit H as the attachment point.
                # The simplest correct approach: create a [*:map_num] and bond it
                new_dummy = Chem.Atom(0)
                new_dummy.SetAtomMapNum(map_num)
                dummy_idx = rw.AddAtom(new_dummy)
                rw.AddBond(n_idx, dummy_idx, Chem.BondType.SINGLE)
                # Reduce explicit Hs by 1
                atom_rw.SetNumExplicitHs(n_h - 1)
                return Chem.MolToSmiles(rw)

        # Loose fallback: any non-aromatic NH2/OH/SH
        for smarts in ['[N;H2]', '[O;H1]', '[S;H1]']:
            patt = Chem.MolFromSmarts(smarts)
            for m2 in mol.GetSubstructMatches(patt):
                n_idx = m2[0]
                atom = mol.GetAtomWithIdx(n_idx)
                if atom.GetIsAromatic():
                    continue
                n_h = atom.GetTotalNumHs()
                if n_h < 1:
                    continue
                if atom.GetAtomicNum() == 8 and n_h != 1:
                    continue
                rw = Chem.RWMol(mol)
                atom_rw = rw.GetAtomWithIdx(n_idx)
                atom_rw.SetNumExplicitHs(n_h)
                new_dummy = Chem.Atom(0)
                new_dummy.SetAtomMapNum(map_num)
                dummy_idx = rw.AddAtom(new_dummy)
                rw.AddBond(n_idx, dummy_idx, Chem.BondType.SINGLE)
                atom_rw.SetNumExplicitHs(n_h - 1)
                return Chem.MolToSmiles(rw)

    return mol_smi


def clean_dummy_labels_in_cxsmiles(smi):
    """Clean dummy labels in CXSMILES."""
    smi_parts = smi.split('|')
    smi = smi_parts[0].replace('*', '[*]') + '|$' + smi_parts[1].split('$')[1] + '$|'
    return smi


def combine_fragments_rwmol(smi1, smi2):
    """Combine two fragments using direct RWMol bond formation.

    Path 2: avoids molzip/CXSMILES round-trip. Merges fragment 2
    into fragment 1 by bonding the :2 dummy neighbor to the :1 dummy neighbor,
    then removing both dummy atoms.
    """
    idx1 = relabel_rgroup2index(smi1)
    idx2 = relabel_rgroup2index(smi2)

    m1 = Chem.RWMol(Chem.MolFromSmiles(idx1))
    m2 = Chem.RWMol(Chem.MolFromSmiles(idx2))

    # Find dummy atoms and their neighbors
    d1_idx = None; n1_idx = None  # :2 dummy in fragment 1
    d2_idx = None; n2_idx = None  # :1 dummy in fragment 2
    for atom in m1.GetAtoms():
        if atom.GetAtomMapNum() == 2:
            d1_idx = atom.GetIdx()
            n1_idx = atom.GetNeighbors()[0].GetIdx()
            break
    for atom in m2.GetAtoms():
        if atom.GetAtomMapNum() == 1:
            d2_idx = atom.GetIdx()
            n2_idx = atom.GetNeighbors()[0].GetIdx()
            break
    if d1_idx is None or d2_idx is None:
        raise ValueError("Cannot find dummy atoms for bonding")

    offset = m1.GetNumAtoms()

    # Add all atoms from m2 to m1 (preserving all properties)
    for atom in m2.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_atom.SetIsotope(atom.GetIsotope())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            new_atom.SetChiralTag(atom.GetChiralTag())
        m1.AddAtom(new_atom)

    # Add all bonds from m2 to m1 (with offset)
    for bond in m2.GetBonds():
        i = offset + bond.GetBeginAtomIdx()
        j = offset + bond.GetEndAtomIdx()
        m1.AddBond(i, j, bond.GetBondType())

    # Bond the two neighbors together
    m1.AddBond(n1_idx, offset + n2_idx, Chem.BondType.SINGLE)

    # Remove dummy atoms (reverse order so indices stay valid)
    d2_offset_idx = offset + d2_idx
    mi = max(d1_idx, d2_offset_idx)
    mn = min(d1_idx, d2_offset_idx)
    m1.RemoveAtom(mi)
    m1.RemoveAtom(mn)

    return relabel_rgroup2label(Chem.MolToSmiles(m1))


def _combine_fragments_impl(smi1, smi2):
    """CXSMILES molzip-based fragment combination (default).

    Falls back to RWMol path when molzip fails (e.g., bare [*] atoms after
    unused R-group replacement).
    """
    # Path 1: CXSMILES molzip (preserves atom labels for chain assembly)
    try:
        m1 = Chem.MolFromSmiles(get_cxsmiles_from_smi(smi1))
        m2 = Chem.MolFromSmiles(get_cxsmiles_from_smi(smi2))
    except Exception:
        return combine_fragments_rwmol(smi1, smi2)

    for atm in m1.GetAtoms():
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R2":
            atm.SetAtomMapNum(10)
    for atm in m2.GetAtoms():
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R1":
            atm.SetAtomMapNum(10)
    try:
        mol = Chem.molzip(m1, m2)
    except Exception:
        return combine_fragments_rwmol(smi1, smi2)

    smi = Chem.MolToCXSmiles(mol)
    if '|' in smi:
        smi = get_smi_from_cxsmiles(clean_dummy_labels_in_cxsmiles(smi))
    return smi


@lru_cache(maxsize=16384)
def _combine_fragments_cached(smi1, smi2):
    return _combine_fragments_impl(smi1, smi2)


def combine_fragments(smi1, smi2):
    """Pure deterministic fragment combination; memoized for route retries.

    Route portfolios (a/b/c/e/g) and multi-model retries recombine identical
    monomer pairs thousands of times per entity; the cache preserves exact
    output semantics while removing the redundant RDKit round-trips.
    """
    try:
        return _combine_fragments_cached(smi1, smi2)
    except TypeError:
        return _combine_fragments_impl(smi1, smi2)


def _get_linear_peptide_cached(monomer_smis):
    smi = None
    for monomer in monomer_smis:
        smi = monomer if smi is None else combine_fragments(smi, monomer)
    return smi


@lru_cache(maxsize=4096)
def _get_linear_peptide_tuple(monomer_smis):
    return _get_linear_peptide_cached(monomer_smis)


def get_linear_peptide(monomer_smis):
    """Get linear peptide from monomers (memoized fold)."""
    try:
        return _get_linear_peptide_tuple(tuple(monomer_smis))
    except TypeError:
        return _get_linear_peptide_cached(monomer_smis)


def get_links_between_monomers(monomer_smis, cyclic_link=None):
    monomer_links = {}

    def add_link(source_idx, source_r_group, target_idx, target_r_group):
        if source_idx not in monomer_links:
            monomer_links[source_idx] = {source_r_group: (target_idx, target_r_group)}
        else:
            monomer_links[source_idx][source_r_group] = (target_idx, target_r_group)
        if target_idx not in monomer_links:
            monomer_links[target_idx] = {target_r_group: None}
        else:
            monomer_links[target_idx][target_r_group] = None

    for idx in range(len(monomer_smis) - 1):
        add_link(idx, '_R2', idx + 1, '_R1')

    if cyclic_link:
        def get_idx_rgroup(node):
            idx, r_group = node.split(':')
            return int(idx) - 1, f'_{r_group}'

        source, target = cyclic_link.split('-')
        source_idx, source_r_group = get_idx_rgroup(source)
        target_idx, target_r_group = get_idx_rgroup(target)
        add_link(source_idx, source_r_group, target_idx, target_r_group)

    return monomer_links


def restore_unused_rgroup(monomer_smis, monomer_r_groups, monomer_links):
    for idx, mol_cxsmi in enumerate(monomer_smis):
        r_groups = monomer_r_groups[idx]
        used_r_groups = [
            r_group[1:]
            for r_group in monomer_links.get(idx, {}).keys()
        ]
        mol_smi = replace_unused_r_groups(mol_cxsmi, r_groups, used_r_groups)
        monomer_smis[idx] = mol_smi
    return monomer_smis

# code under MIT licence Copyright (c) 2021-2024 Charles Xu and others, ends here


# ════════════════════════════════════════════════════════════════════════════
# Monomer SMILES & R-groups dictionaries (from unified library only)
# ════════════════════════════════════════════════════════════════════════════

monomers2smi_dict = {}
monomers2r_groups_dict = {}

# Minimal full-row records for the optional highest-precedence user layer.
# Unlike the strict 235-column derived layer, these rows are intentionally
# runtime-only projections of user_monomer_library.csv.  Keeping them separate
# lets formal isolated evaluation disable local user state without changing the
# manifest-selected Unified/Derived registry.
_active_user_rows = {}

def _build_monomer_dicts(*, include_persistent_user=True):
    """(Re)build monomers2smi_dict / monomers2r_groups_dict from current
    _unified_by_symbol, in place.

    All built-in chemistry, including the standard 20 amino acids and caps,
    comes from the manifest-selected unified library slices. Mutates the dicts
    in place so references held elsewhere stay valid.
    """
    monomers2smi_dict.clear()
    monomers2r_groups_dict.clear()
    _active_user_rows.clear()
    n_skipped = 0
    with _suppress_rdkit_warnings():
        for _sym, _row in _unified_by_symbol.items():
            _cx = str(_row.get("CXSMILES", "")).strip()
            if not _cx:
                continue
            try:
                _smi = get_smi_from_cxsmiles(_cx)
                monomers2smi_dict[_sym] = _smi
                _rg_map = {}
                for _rg in ("R1", "R2", "R3"):
                    _val = str(_row.get(_rg, "-")).strip()
                    if _val and _val != "-":
                        _rg_map[_rg] = _val
                monomers2r_groups_dict[_sym] = _rg_map

                # Register the CycPeptMPDB symbol as an alias when this NNAA
                # entry matched a CycPeptMPDB monomer (build_monomer_library
                # records it in best_cycpep_match_symbol). Without this, HELM
                # references using the CycPeptMPDB name (e.g. {nnr:Ser(tBu)})
                # would be unresolvable even though the chemistry is present
                # under the NNAA symbol (e.g. 4EP).
                _alias = str(_row.get("best_cycpep_match_symbol", "")).strip()
                if _alias and _alias != _sym and _alias not in monomers2smi_dict:
                    monomers2smi_dict[_alias] = _smi
                    monomers2r_groups_dict[_alias] = dict(_rg_map)
            except Exception:
                n_skipped += 1

    # Hard-coded aliases for CycPeptMPDB monomers whose numbering is a gap in
    # Monomer_All (no definition), empirically established as structural aliases
    # of defined monomers (verified: peptides containing them match literature
    # SMILES only under this mapping). build_monomer_library cannot infer these.
    for _alias, _real in _MONOMER_ALIASES.items():
        if _alias not in monomers2smi_dict and _real in monomers2smi_dict:
            monomers2smi_dict[_alias] = monomers2smi_dict[_real]
            monomers2r_groups_dict[_alias] = dict(monomers2r_groups_dict[_real])

    # User-registered monomers are an explicit highest-precedence layer. Formal
    # isolated evaluation disables this layer to avoid hidden local state.
    if include_persistent_user:
        _load_user_monomers()

    if n_skipped:
        warnings.warn(
            f"_map_utils: {n_skipped} monomer(s) skipped due to unparseable "
            f"CXSMILES while loading monomer library")


_USER_CSV = os.path.join(_PROJECT_DIR, "user_monomer_library.csv")
_USER_ALLOWED_R_GROUP_DEFAULTS = {
    "R1": frozenset({"H"}),
    "R2": frozenset({"H", "OH", "NH2"}),
    "R3": frozenset({"-", "H", "OH", "SH"}),
}


def _validate_user_monomer_record(record):
    """Validate an untrusted user-library row without mutating registries."""
    row = dict(record)
    symbol = str(row.get("symbol", "")).strip()
    cxsmiles = str(row.get("CXSMILES", "")).strip()
    if not symbol or not cxsmiles:
        raise ValueError("user monomer record requires symbol and CXSMILES")
    runtime_smiles = get_smi_from_cxsmiles(cxsmiles)
    if not isinstance(runtime_smiles, str) or not runtime_smiles.strip():
        raise ValueError("user monomer CXSMILES has no usable runtime representation")
    r_groups = {}
    for name, allowed in _USER_ALLOWED_R_GROUP_DEFAULTS.items():
        value = (str(row.get(name, "-")).strip() or "-").upper()
        if value not in allowed:
            raise ValueError(
                f"{name} default must be one of {sorted(allowed)}, got {value!r}"
            )
        has_port = f"_{name}" in cxsmiles
        if value != "-" and not has_port:
            raise ValueError(
                f"{name} has a leaving group but CXSMILES has no {name} port"
            )
        if value == "-" and has_port:
            raise ValueError(
                f"CXSMILES contains {name} but its leaving group is unavailable"
            )
        row[name] = value
        if value != "-":
            r_groups[name] = value
    row["symbol"] = symbol
    row["CXSMILES"] = cxsmiles
    return row, runtime_smiles, r_groups


def _load_user_monomers():
    """Merge user_monomer_library.csv into the live dicts (overrides allowed).

    Read inline (not via core.monomer_admin) to avoid an import cycle. Silent
    when the file is absent; skips rows whose CXSMILES fails to parse.
    """
    if not os.path.exists(_USER_CSV):
        return
    with _suppress_rdkit_warnings():
        with open(_USER_CSV, "r", encoding="utf-8-sig") as _f:
            for _row in _csv.DictReader(_f):
                try:
                    _row, _smi, _rg = _validate_user_monomer_record(_row)
                    _sym = _row["symbol"]
                    collision = next(
                        (
                            existing for existing in monomers2smi_dict
                            if existing.casefold() == _sym.casefold()
                            and existing != _sym
                        ),
                        None,
                    )
                    if collision is not None:
                        raise ValueError(
                            f"user monomer {_sym!r} conflicts with {collision!r} "
                            "by case-insensitive identity"
                        )
                except Exception as exc:
                    warnings.warn(f"skipping invalid user monomer row: {exc}")
                    continue
                monomers2smi_dict[_sym] = _smi
                monomers2r_groups_dict[_sym] = _rg
                _active_user_rows[_sym] = {
                    "symbol": _sym,
                    "CXSMILES": _row["CXSMILES"],
                    "R1": str(_row.get("R1", "-")).strip() or "-",
                    "R2": str(_row.get("R2", "-")).strip() or "-",
                    "R3": str(_row.get("R3", "-")).strip() or "-",
                    "smiles_original": str(
                        _row.get("smiles_original", "")
                    ).strip(),
                    "source": "user_registered",
                }


# Hard-coded aliases (see _build_monomer_dicts for rationale).
_MONOMER_ALIASES = {
    "Mono7": "Bal(3-Me)",
    "Mono8": "Bal(d3-CF3)",
}

_build_monomer_dicts()


def _set_active_libraries_unlocked(names):
    """Reload the monomer dictionaries from a chosen set of sub-libraries.

    names: list of sub-library stems under libraries/ (e.g.
    ["core", "curated_cycpep"]). Rebuilds _unified_by_symbol from those CSVs
    and repopulates monomers2smi_dict / monomers2r_groups_dict. Core amino acids
    and caps are always loaded from their library slices.

    Note: "core" (standard AA) is also always injected from code, so listing it
    is harmless. Returns the new monomer count.
    """
    global _unified_by_symbol
    requested = [name for name in names if name not in {"core", "caps"}]
    required = ["caps", "core", *requested]
    loaded = _load_from_sublibraries(required)
    if loaded is None:
        raise FileNotFoundError(
            f"one or more sub-libraries not found under {_LIBRARIES_DIR}: {names}")
    revalidated_derived = _validate_derived_rows(
        list(_persistent_derived_by_symbol.values()),
        loaded,
        require_exact_schema=True,
    )
    loaded.update(revalidated_derived)
    _unified_by_symbol = loaded
    _build_monomer_dicts()
    _rebuild_notation_indexes()
    _invalidate_exact_v1_registry_cache()
    return len(monomers2smi_dict)


def set_active_libraries(names):
    """Reload library slices as one serialized registry write."""
    with _REGISTRY_LOCK:
        return _set_active_libraries_unlocked(names)


def _connect_unique_dummies(pep_smi, bond_label_pairs):
    """Bond pairs of uniquely-labelled dummy atoms in one RDKit edit pass.

    pep_smi carries side-chain dummies tagged with UNIQUE atomLabels
    (e.g. '_RU2_R3', '_RU17_R3') for the ring bonds. Chain termini (R1/R2) are
    already capped during linear assembly, so only these unique ring dummies
    remain to be processed. For each (labelA, labelB) pair we bond the two
    dummies' heavy neighbours and delete the dummies. All ring bonds are formed
    on the SAME molecule (no SMILES re-parse / index drift), which is what
    fixes the historical 3+ disulfide failure.

    Returns canonical SMILES, or None on failure.
    """
    mol = Chem.MolFromSmiles(get_cxsmiles_from_smi(pep_smi))
    if mol is None:
        mol = Chem.MolFromSmiles(pep_smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)

    # Map each unique label -> (dummy_idx, neighbour_idx)
    label_to_dummy = {}
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0 or not atom.HasProp("atomLabel"):
            continue
        nbrs = atom.GetNeighbors()
        if len(nbrs) != 1:
            continue
        label_to_dummy[atom.GetProp("atomLabel")] = (atom.GetIdx(), nbrs[0].GetIdx())

    dummies_to_remove = []
    rw.BeginBatchEdit()
    for la, lb in bond_label_pairs:
        if la not in label_to_dummy or lb not in label_to_dummy:
            rw.CommitBatchEdit()
            return None  # a required attachment point is missing
        da, na = label_to_dummy[la]
        db, nb = label_to_dummy[lb]
        rw.AddBond(na, nb, Chem.BondType.SINGLE)
        dummies_to_remove.extend([da, db])
    for idx in dummies_to_remove:
        rw.RemoveAtom(idx)
    rw.CommitBatchEdit()

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def cyclize_linpep_from_map(monomer_list, cyclic_link, chain_breaks=None):
    """Build and cyclize peptide from monomer list.

    If cyclic_link is a string, handles single cyclization (backward compatible).
    If cyclic_link is a list, handles multi-cyclization (multi-disulfide,
    head-to-tail + disulfide, bicyclic, ...).

    chain_breaks: optional set/iterable of 0-based positions ``i`` meaning "no
    automatic backbone bond between monomer i and i+1" (a multi-chain boundary).
    Independent chains (e.g. insulin A/B, joined only by disulfides) are
    assembled as one monomer list with breaks at the chain boundaries; only the
    explicit cyclic_link records then connect the chains. Default None = every
    consecutive pair is backbone-bonded (single-chain, unchanged behaviour).

    Multi-cyclization correctness: each side-chain attachment point that
    participates in a ring is given a GLOBALLY UNIQUE dummy label
    (_RU{position}) before chain assembly, so N rings (e.g. 3 disulfides) can
    each be bonded to the exact intended atoms. All ring bonds are then formed
    in a single RDKit edit pass (no stepwise SMILES re-parse / index drift),
    which fixes the historical failure at 3+ disulfides.
    """
    if cyclic_link is None:
        return linpep_from_map(monomer_list)

    # Normalize to list for uniform handling
    if isinstance(cyclic_link, str):
        linker_list = [cyclic_link]
    else:
        linker_list = cyclic_link

    breaks = set(chain_breaks) if chain_breaks else set()

    monomer_smis = [copy.deepcopy(monomers2smi_dict[monomer]) for monomer in monomer_list]
    monomer_r_groups = [copy.deepcopy(monomers2r_groups_dict[monomer]) for monomer in monomer_list]

    # Build the multi-link dict (backbone R2->R1 + every ring link), skipping the
    # auto backbone bond across any chain-break boundary.
    monomer_links = {}
    for idx in range(len(monomer_smis) - 1):
        if idx in breaks:
            continue
        _add_link(monomer_links, idx, '_R2', idx + 1, '_R1')

    # Assign a globally-unique dummy label to each ring attachment point and
    # record the (labelA, labelB) pair we must bond. Uniqueness = _RU{pos}{rg}.
    bond_label_pairs = []
    for link in linker_list:
        source, target = link.split('-')
        source_idx, source_r_group = _get_idx_rgroup(source)   # 0-based, '_R3'
        target_idx, target_r_group = _get_idx_rgroup(target)
        _add_link(monomer_links, source_idx, source_r_group, target_idx, target_r_group)
        bond_label_pairs.append((source_idx, source_r_group, target_idx, target_r_group))

    # Cap truly-free R-groups FIRST, while every dummy still carries a parseable
    # numeric map label. Ring attachment points are listed in monomer_links so
    # restore_unused_rgroup preserves them; free side chains (e.g. a free Asp R3
    # in a head-to-tail cyclic peptide) are correctly capped to COOH here rather
    # than left as a bare dummy '*'. Renaming to non-numeric _RU labels must come
    # AFTER this step, since replace_unused_r_groups round-trips through RDKit
    # and cannot parse '[*:_RU..]'.
    monomer_smis = restore_unused_rgroup(monomer_smis, monomer_r_groups, monomer_links)

    # Now rename each ring attachment point's preserved dummy to its unique label.
    ring_pairs = []
    for source_idx, source_r_group, target_idx, target_r_group in bond_label_pairs:
        la = f'_RU{source_idx}{source_r_group}'   # e.g. _RU2_R3
        lb = f'_RU{target_idx}{target_r_group}'
        monomer_smis[source_idx] = monomer_smis[source_idx].replace(
            f'[*:{source_r_group}]', f'[*:{la}]', 1)
        monomer_smis[target_idx] = monomer_smis[target_idx].replace(
            f'[*:{target_r_group}]', f'[*:{lb}]', 1)
        ring_pairs.append((la, lb))

    pep_smi = get_linear_peptide(monomer_smis)

    # Form ALL ring bonds at once using the unique labels.
    return _connect_unique_dummies(pep_smi, ring_pairs)


def _add_link(monomer_links, source_idx, source_r_group, target_idx, target_r_group):
    if source_idx not in monomer_links:
        monomer_links[source_idx] = {source_r_group: (target_idx, target_r_group)}
    else:
        monomer_links[source_idx][source_r_group] = (target_idx, target_r_group)
    if target_idx not in monomer_links:
        monomer_links[target_idx] = {target_r_group: None}
    else:
        monomer_links[target_idx][target_r_group] = None


def _get_idx_rgroup(node):
    idx, r_group = node.split(':')
    return int(idx) - 1, f'_{r_group}'


def linpep_from_map(monomer_list):
    monomer_smis = [copy.deepcopy(monomers2smi_dict[monomer]) for monomer in monomer_list]
    monomer_r_groups = [copy.deepcopy(monomers2r_groups_dict[monomer]) for monomer in monomer_list]
    monomer_links = get_links_between_monomers(monomer_smis, cyclic_link=None)
    monomer_smis = restore_unused_rgroup(monomer_smis, monomer_r_groups, monomer_links)
    pep_smi = get_linear_peptide(monomer_smis)
    return pep_smi


# ════════════════════════════════════════════════════════════════════════════
# MAP ↔ HELM conversion (auto-generated from unified library)
# ════════════════════════════════════════════════════════════════════════════

# MAP_denotion -> Symbol (longest-first for greedy matching)
map_to_helm_dict: dict = {}
for _sym in monomers2smi_dict:
    _map_d = _auto_map_denotion(_sym)
    if _map_d not in map_to_helm_dict:
        map_to_helm_dict[_map_d] = _sym

# Synthesized MAP denotions for caps
map_to_helm_dict["{nt:ACE}"] = "ac"
map_to_helm_dict["{ct:NME}"] = "nme"
map_to_helm_dict["{ct:NH2}"] = "nh2"

# Also add cap symbols themselves so map_to_helm() can look them up
map_to_helm_dict["ac"] = "ac"
map_to_helm_dict["nme"] = "nme"
map_to_helm_dict["nh2"] = "nh2"

# Sort longest-first for greedy tokenization
map_to_helm_dict = dict(sorted(map_to_helm_dict.items(),
                               key=lambda kv: (len(kv[0]), kv[0]), reverse=True))

# Symbol -> MAP_denotion reverse lookup
_symbol_to_map: dict = {}
for _k, _v in map_to_helm_dict.items():
    _symbol_to_map[_v] = _k


def registry_epoch() -> int:
    """Monotonic counter bumped whenever monomer registry state changes.

    Cache keys in dependent modules (path assembly, tokenizers) embed this
    epoch so memoized results can never outlive the registry state that
    produced them.
    """
    return _REGISTRY_EPOCH


def _invalidate_sequence_token_cache():
    """Drop memoized tokenizations whenever the notation index changes."""
    _monomer_list_cached.cache_clear()


def _rebuild_notation_indexes():
    """Rebuild both notation indexes in place from the active monomer registry."""
    rebuilt = {}
    for symbol in monomers2smi_dict:
        denotion = _auto_map_denotion(symbol)
        rebuilt.setdefault(denotion, symbol)
    rebuilt.update({
        "{nt:ACE}": "ac", "{ct:NME}": "nme", "{ct:NH2}": "nh2",
        "ac": "ac", "nme": "nme", "nh2": "nh2",
    })
    map_to_helm_dict.clear()
    map_to_helm_dict.update(dict(sorted(
        rebuilt.items(), key=lambda item: (len(item[0]), item[0]), reverse=True
    )))
    _symbol_to_map.clear()
    for denotion, symbol in map_to_helm_dict.items():
        _symbol_to_map[symbol] = denotion
    _invalidate_sequence_token_cache()


def _register_monomer_unlocked(
    symbol, smiles, r_groups, *, overwrite=False
):
    """Inject a monomer into the live runtime dictionaries so it is immediately
    resolvable by get_smi_from_map / HELM tokenization.

    symbol    : monomer symbol (the model token / HELM element)
    smiles    : SMILES with _R group dummy labels (as produced by
                get_smi_from_cxsmiles)
    r_groups  : dict like {"R1": "H", "R2": "OH"} (only present R-groups)
    overwrite : if False and the symbol already exists, raise KeyError.

    Updates monomers2smi_dict, monomers2r_groups_dict, map_to_helm_dict (kept
    longest-first), and the _symbol_to_map reverse lookup. Does not persist to
    disk — callers that want persistence write their own CSV.
    """
    if not overwrite and symbol in monomers2smi_dict:
        raise KeyError(f"monomer {symbol!r} already registered "
                       f"(pass overwrite=True to replace)")
    monomers2smi_dict[symbol] = smiles
    monomers2r_groups_dict[symbol] = dict(r_groups)
    denot = _auto_map_denotion(symbol)
    map_to_helm_dict[denot] = symbol
    ordered = dict(sorted(map_to_helm_dict.items(),
                          key=lambda kv: (len(kv[0]), kv[0]),
                          reverse=True))
    map_to_helm_dict.clear()
    map_to_helm_dict.update(ordered)
    _symbol_to_map[symbol] = denot
    _invalidate_exact_v1_registry_cache()
    _invalidate_sequence_token_cache()


def register_monomer(symbol, smiles, r_groups, *, overwrite=False):
    with _REGISTRY_LOCK:
        return _register_monomer_unlocked(
            symbol, smiles, r_groups, overwrite=overwrite
        )


def _register_user_monomer_record_unlocked(
    record, *, overwrite=False
):
    """Register one user row for notation assembly and dynamic PDB templates.

    User records remain a distinct highest-precedence runtime layer.  They are
    excluded by ``isolated_monomer_registry`` unless explicitly enabled.
    Persistence remains the responsibility of ``core.monomer_admin``.
    """
    row, runtime_smiles, r_groups = _validate_user_monomer_record(record)
    symbol = row["symbol"]
    collision = next(
        (
            existing for existing in monomers2smi_dict
            if existing.casefold() == symbol.casefold() and existing != symbol
        ),
        None,
    )
    if collision is not None:
        raise ValueError(
            f"user monomer {symbol!r} conflicts with {collision!r} "
            "by case-insensitive identity"
        )
    register_monomer(
        symbol, runtime_smiles, r_groups, overwrite=overwrite
    )
    row["symbol"] = symbol
    row["source"] = "user_registered"
    _active_user_rows[symbol] = row
    return dict(row)


def register_user_monomer_record(record, *, overwrite=False):
    with _REGISTRY_LOCK:
        return _register_user_monomer_record_unlocked(
            record, overwrite=overwrite
        )


def _register_pdb_alias_unlocked(
    pdb_resname,
    target_symbol,
    *,
    overwrite=False,
):
    """Register one runtime PDB/PRD residue alias through the shared layer."""
    name = str(pdb_resname or "").strip().upper()
    target = str(target_symbol or "").strip()
    if not name or not target:
        raise ValueError("pdb_resname and target_symbol are required")
    if target not in monomers2smi_dict:
        raise KeyError(f"alias target is not registered: {target!r}")
    built_in = resolve_pdb_alias(name)
    if built_in is not None and built_in != target and not overwrite:
        raise KeyError(
            f"PDB alias {name!r} already resolves to {built_in!r}"
        )
    existing = _active_pdb_aliases.get(name)
    if existing is not None and existing != target and not overwrite:
        raise KeyError(
            f"PDB alias {name!r} already targets {existing!r}"
        )
    if existing == target:
        return {"pdb_resname": name, "target_symbol": target}
    _active_pdb_aliases[name] = target
    _invalidate_exact_v1_registry_cache()
    return {"pdb_resname": name, "target_symbol": target}


def register_pdb_alias(
    pdb_resname,
    target_symbol,
    *,
    overwrite=False,
):
    with _REGISTRY_LOCK:
        return _register_pdb_alias_unlocked(
            pdb_resname,
            target_symbol,
            overwrite=overwrite,
        )


_REGISTRY_LOCK = threading.RLock()


@contextmanager
def isolated_monomer_registry(
    *,
    derived_rows=(),
    pdb_aliases=(),
    include_persistent_user=False,
    require_empty_persistent_derived=False,
):
    """Activate entity-local derived rows and restore all registries on exit.

    The context is process-global and therefore serialized by a re-entrant
    lock. Formal parallel evaluation should still use one fresh process per
    entity. No derived or user CSV is written by this API.
    """
    with _REGISTRY_LOCK:
        persistent_audit = persistent_overlay_audit()
        if (
            require_empty_persistent_derived
            and persistent_audit["status"] != "empty"
        ):
            raise ValueError(
                "entity-local reconstruction requires an empty persistent "
                "derived overlay"
            )
        snapshots = {
            "unified": copy.deepcopy(_unified_by_symbol),
            "user_rows": copy.deepcopy(_active_user_rows),
            "smiles": copy.deepcopy(monomers2smi_dict),
            "rgroups": copy.deepcopy(monomers2r_groups_dict),
            "map": copy.deepcopy(map_to_helm_dict),
            "reverse": copy.deepcopy(_symbol_to_map),
            "pdb_aliases": copy.deepcopy(_active_pdb_aliases),
        }
        try:
            additions = _validate_derived_rows(
                list(derived_rows),
                _unified_by_symbol,
                require_exact_schema=True,
            )
            _unified_by_symbol.update(additions)
            activated_aliases = {}
            for index, raw in enumerate(pdb_aliases, start=1):
                pdb_resname = str(raw.get("pdb_resname", "")).strip().upper()
                target = str(raw.get("target_symbol", "")).strip()
                if not pdb_resname or not target:
                    raise ValueError(
                        f"entity-local PDB alias {index} lacks pdb_resname or target_symbol"
                    )
                if target not in _unified_by_symbol:
                    raise ValueError(
                        f"entity-local PDB alias {pdb_resname!r} targets unknown symbol {target!r}"
                    )
                from .residue_template_factory import _PDB_TO_UNIFIED_SYMBOL
                built_in = _PDB_TO_UNIFIED_SYMBOL.get(pdb_resname)
                if built_in is not None and built_in != target:
                    raise ValueError(
                        f"entity-local PDB alias {pdb_resname!r} conflicts with "
                        f"built-in symbol {built_in!r}"
                    )
                existing = _active_pdb_aliases.get(pdb_resname)
                if existing is not None and existing != target:
                    raise ValueError(
                        f"entity-local PDB alias {pdb_resname!r} conflicts with {existing!r}"
                    )
                _active_pdb_aliases[pdb_resname] = target
                activated_aliases[pdb_resname] = target
            _build_monomer_dicts(
                include_persistent_user=bool(include_persistent_user)
            )
            for pdb_resname, target in activated_aliases.items():
                if (
                    pdb_resname not in monomers2smi_dict
                    and target in monomers2smi_dict
                ):
                    monomers2smi_dict[pdb_resname] = copy.deepcopy(
                        monomers2smi_dict[target]
                    )
                    monomers2r_groups_dict[pdb_resname] = copy.deepcopy(
                        monomers2r_groups_dict.get(target, {})
                    )
            _rebuild_notation_indexes()
            _invalidate_exact_v1_registry_cache()
            ledger = {
                "derived_symbols": sorted(additions),
                "pdb_aliases": dict(sorted(activated_aliases.items())),
                "persistent_user_enabled": bool(include_persistent_user),
                "disk_writes": 0,
                "entity_local_overlay": True,
                "persistent_overlay_at_start": persistent_audit,
                "persistent_overlay_required_empty": bool(
                    require_empty_persistent_derived
                ),
                "registry_restoration_verified": False,
            }
            yield ledger
        finally:
            _unified_by_symbol.clear()
            _unified_by_symbol.update(snapshots["unified"])
            _active_user_rows.clear()
            _active_user_rows.update(snapshots["user_rows"])
            monomers2smi_dict.clear()
            monomers2smi_dict.update(snapshots["smiles"])
            monomers2r_groups_dict.clear()
            monomers2r_groups_dict.update(snapshots["rgroups"])
            map_to_helm_dict.clear()
            map_to_helm_dict.update(snapshots["map"])
            _symbol_to_map.clear()
            _symbol_to_map.update(snapshots["reverse"])
            _active_pdb_aliases.clear()
            _active_pdb_aliases.update(snapshots["pdb_aliases"])
            _invalidate_exact_v1_registry_cache()
            _invalidate_sequence_token_cache()
            if "ledger" in locals():
                ledger["registry_restoration_verified"] = all((
                    _unified_by_symbol == snapshots["unified"],
                    _active_user_rows == snapshots["user_rows"],
                    monomers2smi_dict == snapshots["smiles"],
                    monomers2r_groups_dict == snapshots["rgroups"],
                    map_to_helm_dict == snapshots["map"],
                    _symbol_to_map == snapshots["reverse"],
                    _active_pdb_aliases == snapshots["pdb_aliases"],
                ))


def extract_data(input_string):
    """Extract ALL cyclization tags and terminal mods from MAP string.

    Returns (linear_sequence, linker_list).
      - linker_list is [] for linear peptides
      - linker_list is ['{cyc:...}', ...] for cyclic peptides (any number of bonds)

    Handles simple ({cyc:2-5}) and detailed ({cyc:2:R3-5:R3}) formats.
    """
    cyc_pattern = r'\{cyc:\s*([N\d]+):?(R\d)?-([C\d\w]+):?(R\d)?\}'
    nterm_pattern = r'\{nt:[^}]+\}'

    # Find ALL cyclization tags (not just first one)
    cyc_matches = re.findall(cyc_pattern, input_string)
    result = re.sub(cyc_pattern, '', input_string).strip()

    linker_infos = []
    for start_pos, start_r, end_pos, end_r in cyc_matches:
        if end_pos == 'C':
            linker_info = f'{{cyc:{start_pos}:R1-{end_pos}:R2}}'
        elif start_r and end_r:
            linker_info = f'{{cyc:{start_pos}:{start_r}-{end_pos}:{end_r}}}'
        else:
            srg = 'R1' if (start_pos == 'N' or int(start_pos) == 1) else 'R3'
            erg = 'R3' if end_pos != 'C' else 'R2'
            linker_info = f'{{cyc:{start_pos}:{srg}-{end_pos}:{erg}}}'
        linker_infos.append(linker_info)

    # Move terminal modifications to the start
    nterm_modifications = re.findall(nterm_pattern, result)
    if nterm_modifications:
        result = re.sub(nterm_pattern, '', result)
        result = ''.join(nterm_modifications) + result

    return result, linker_infos


def _monomer_list_from_linear_seq_impl(linear_seq):
    """Tokenize MAP linear sequence into monomer symbols.

    An unknown ``{nnr:SYM}`` block (SYM not in the monomer library) is kept as
    a single token rather than shattered into one-character fragments, so the
    failure surfaces as a clear 'unknown monomer' warning upstream instead of
    a spurious KeyError deep in assembly.
    """
    tokens = []
    i = 0
    n = len(linear_seq)
    while i < n:
        matched = False
        for key in map_to_helm_dict.keys():
            if linear_seq[i:].startswith(key):
                tokens.append(map_to_helm_dict[key])
                i += len(key)
                matched = True
                break
        if not matched:
            # Consume an unknown {nnr:...} block whole, not character-by-character.
            if linear_seq[i:i + 5] == '{nnr:':
                end = linear_seq.find('}', i)
                if end != -1:
                    tokens.append(linear_seq[i:end + 1])
                    i = end + 1
                    continue
            tokens.append(linear_seq[i])
            i += 1
    return tokens


@lru_cache(maxsize=4096)
def _monomer_list_cached(linear_seq):
    return tuple(_monomer_list_from_linear_seq_impl(linear_seq))


def monomer_list_from_linear_seq(linear_seq):
    """Tokenize MAP linear sequence (memoized; fresh list per call).

    Route portfolios tokenize the identical sequence dozens of times per
    entity; the cache returns a defensive copy so callers keep the historical
    mutable-list semantics.
    """
    return list(_monomer_list_cached(linear_seq))


def _detect_conflicting_linkers(cyclic_linkers):
    """Detect an (position, R-group) used by more than one cyclization bond.

    get_smi_from_map / cyclize_linpep_from_map cannot form a second bond on an
    attachment point that is already consumed (e.g. residue 3 R3 used twice).
    Returns the conflicting "pos:Rg" token, or None if all attachment points
    are unique.
    """
    seen = set()
    for cl in cyclic_linkers:
        # cl looks like "1:R3-5:R3"
        m = re.match(r'(\d+):(R\d)-(\d+):(R\d)', cl)
        if not m:
            continue
        for pos, rg in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
            token = f'{pos}:{rg}'
            if token in seen:
                return token
            seen.add(token)
    return None


_LEGACY_CYC_TAG = re.compile(
    r"\{cyc:\s*(?:N|[1-9][0-9]*)(?::R[1-3])?\s*-\s*"
    r"(?:C|[1-9][0-9]*)(?::R[1-3])?\s*\}"
)


def _validate_legacy_cyc_tags(map_str):
    """Reject cyc annotations that the legacy assembler would drop.

    ``extract_data`` intentionally supports the historical shorthand syntax,
    but its permissive regular expression used to consume malformed endpoints
    such as ``{cyc:N-N}`` and then silently omit the resulting linker. Keep
    this guard local to the legacy entry point so strict representation parsers
    retain their existing contracts.
    """
    raw_tags = re.findall(r"\{cyc:[^{}]*\}", map_str)
    if "{cyc:" in map_str and not raw_tags:
        raise ValueError("malformed or unclosed MAP cyclization annotation")
    for raw_tag in raw_tags:
        if _LEGACY_CYC_TAG.fullmatch(raw_tag) is None:
            raise ValueError(f"malformed MAP cyclization annotation: {raw_tag}")


def _validate_assembled_cyc_linkers(
    cyclic_linkers, first_peptide_position, last_peptide_position
):
    """Validate resolved cyc endpoints before passing them to the assembler."""
    endpoint_pattern = re.compile(r"([1-9][0-9]*):(R[1-3])-([1-9][0-9]*):(R[1-3])")
    for linker in cyclic_linkers:
        match = endpoint_pattern.fullmatch(linker)
        if match is None:
            return f"unrepresentable cyclization endpoint: {linker}"
        left_pos, left_rg, right_pos, right_rg = match.groups()
        left_pos, right_pos = int(left_pos), int(right_pos)
        for position, rgroup in (
            (left_pos, left_rg),
            (right_pos, right_rg),
        ):
            if not first_peptide_position <= position <= last_peptide_position:
                return (
                    f"cyclization position {position} out of peptide range "
                    f"{first_peptide_position}..{last_peptide_position}"
                )
            if rgroup == "R1" and position != first_peptide_position:
                return f"R1 cyclization endpoint must be at position {first_peptide_position}"
            if rgroup == "R2" and position != last_peptide_position:
                return f"R2 cyclization endpoint must be at position {last_peptide_position}"
        if left_pos == right_pos:
            return f"self cyclization on residue position {left_pos} is unsupported"
    return None


def _validate_r3_available(cyclic_linkers, monomer_list):
    """Check every R3 attachment in the linkers points to a residue that has a
    side-chain (R3) connection in the unified library.

    A {cyc:i:R3-...} bond on a residue whose monomer has no R3 (e.g. Ala) is an
    invalid input that the assembler cannot satisfy. Returns an explanatory
    string for the first such violation, or None if all R3 points are valid.
    """
    for cl in cyclic_linkers:
        m = re.match(r'(\d+):(R\d)-(\d+):(R\d)', cl)
        if not m:
            continue
        for pos_s, rg in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
            if rg != 'R3':
                continue
            pos = int(pos_s)
            if pos < 1 or pos > len(monomer_list):
                return f'position {pos} out of range (1..{len(monomer_list)})'
            symbol = monomer_list[pos - 1]
            row = _unified_by_symbol.get(symbol)
            if row is None:
                # Unknown monomer symbol; let downstream handle, don't block here
                continue
            r3 = str(row.get('R3', '')).strip()
            if r3 in ('', '-'):
                return (f'residue {pos} ({symbol!r}) has no R3 side-chain '
                        f'attachment point')
    return None


def _interactive_enabled(interactive):
    """Triple-gate the interactive add-monomer prompt (all must hold):

      1. caller passed interactive=True (default False), OR the
         CYCPEP_INTERACTIVE environment variable is set to a truthy value;
      2. AND stdin is a real TTY (never prompt under pytest / pipes / cron).

    This keeps batch processing and the test suite at the historical behaviour
    (warn + return None) with zero chance of a blocking input() call.
    """
    import sys
    env = os.environ.get("CYCPEP_INTERACTIVE", "").strip().lower()
    want = bool(interactive) or env in ("1", "true", "yes", "on")
    if not want:
        return False
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _bare_symbol(token):
    """Strip a MAP denotion wrapper to the bare monomer symbol.

    '{nnr:IxNva}' -> 'IxNva'; a bare token is returned unchanged. New monomers
    must be registered under the bare symbol (as the unified-library NNAAs are),
    so that after registration the tokenizer unwraps the denotion to it.
    """
    m = re.match(r'^\{[a-z]+:(.+)\}$', token)
    return m.group(1) if m else token


def _prompt_and_add_monomers(unknown_tokens):
    """Prompt the user for a SMILES for each unknown monomer and register it.

    Returns the set of bare symbols successfully added. Empty input skips a
    symbol. Import of add_monomer is deferred to avoid an import cycle.
    """
    added = set()
    try:
        from ..core.monomer_admin import add_monomer
    except Exception:
        return added
    for token in unknown_tokens:
        sym = _bare_symbol(token)
        try:
            smiles = input(f"[cycpep] unknown monomer {sym!r} — enter a SMILES "
                           f"to register it (blank to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not smiles:
            continue
        try:
            add_monomer(sym, smiles)
            added.add(sym)
            print(f"[cycpep] registered {sym!r}.")
        except Exception as ex:
            print(f"[cycpep] could not register {sym!r}: {ex}")
    return added


def get_smi_from_map(map_str, interactive=False):
    """Convert MAP notation to SMILES.

    Supports multiple cyclization tags:
      - Single: {cyc:N-C}, {cyc:2:R3-5:R3}
      - Multiple: {cyc:3:R3-35:R3}{cyc:12:R3-28:R3}{cyc:17:R3-32:R3}

    Returns the SMILES string, or None on failure. Failures emit a warning
    explaining the cause (unsupported topology vs. internal error) instead of
    failing silently, so callers rebuilding data can log/skip with a reason.

    If interactive=True (or CYCPEP_INTERACTIVE is set) AND stdin is a TTY, the
    user is prompted to supply a SMILES for any unknown monomer and assembly is
    retried once. Default is non-interactive (warn + None), so batch/pytest
    behaviour is unchanged.
    """
    try:
        _validate_legacy_cyc_tags(map_str)
        _validate_map_terminal_modifier_positions(map_str)
    except (TypeError, ValueError) as exc:
        warnings.warn(f"get_smi_from_map: {exc}; skipping {map_str!r}")
        return None
    for modifier_match in _MODIFIER_CODE_RE.finditer(map_str):
        side_code, modifier_code = modifier_match.group(1), modifier_match.group(2)
        supported = (
            _NTERM_MODIFIER_MAP if side_code == "nt" else _CTERM_MODIFIER_MAP
        )
        if modifier_code not in supported:
            warnings.warn(
                "get_smi_from_map: "
                + _unsupported_modifier_message(
                    "N-terminal" if side_code == "nt" else "C-terminal",
                    modifier_code,
                )
                + f"; skipping {map_str!r}"
            )
            return None

    linear_seq, linkers = extract_data(map_str)

    # Multi-chain support: {br} markers split the sequence into independent
    # chains (no automatic backbone bond across a boundary). Tokenize each
    # segment and record the 0-based break positions (index of the last monomer
    # of each chain except the last). Single-chain MAP (no {br}) tokenizes
    # exactly as before — chain_breaks stays empty and behaviour is unchanged.
    if '{br}' in linear_seq:
        monomer_list = []
        chain_breaks = set()
        segments = [s for s in linear_seq.split('{br}')]
        for seg in segments:
            seg_tokens = monomer_list_from_linear_seq(seg) if seg else []
            monomer_list.extend(seg_tokens)
            if monomer_list:
                chain_breaks.add(len(monomer_list) - 1)
        # The last segment's end is not a real break; drop it.
        if monomer_list:
            chain_breaks.discard(len(monomer_list) - 1)
    else:
        monomer_list = monomer_list_from_linear_seq(linear_seq)
        chain_breaks = set()

    # Surface unknown monomers (not in the unified library) as an explicit,
    # actionable warning rather than letting them fail opaquely inside assembly.
    unknown = [m for m in monomer_list if m not in monomers2smi_dict]
    if unknown:
        # Optionally let an interactive user register the missing monomers and
        # retry once. Triple-gated so non-interactive use is unaffected.
        if _interactive_enabled(interactive):
            added = _prompt_and_add_monomers(sorted(set(unknown)))
            if added:
                # Re-tokenize: newly registered symbols are now in
                # map_to_helm_dict, so denotions like {nnr:X} unwrap to X.
                # Re-run the same {br}-aware tokenization as above.
                if '{br}' in linear_seq:
                    monomer_list = []
                    for seg in linear_seq.split('{br}'):
                        monomer_list.extend(
                            monomer_list_from_linear_seq(seg) if seg else [])
                else:
                    monomer_list = monomer_list_from_linear_seq(linear_seq)
                unknown = [m for m in monomer_list
                           if m not in monomers2smi_dict]
        if unknown:
            from collections import Counter
            missing = ", ".join(f"{s}×{c}" for s, c in Counter(unknown).items())
            warnings.warn(f"get_smi_from_map: unknown monomer(s) not in library "
                          f"({missing}); add to unified_monomer_library.csv. "
                          f"Skipping {map_str!r}")
            return None

    if not linkers:
        if chain_breaks:
            # Multi-chain but no explicit connections: assemble independent
            # chains as disconnected fragments (backbone bonds skipped at
            # breaks). cyclize_linpep_from_map with an empty link list still
            # honours chain_breaks.
            try:
                return cyclize_linpep_from_map(monomer_list, [],
                                               chain_breaks=chain_breaks)
            except Exception as ex:
                warnings.warn(f"get_smi_from_map: multi-chain assembly failed "
                              f"for {map_str!r}: {ex}")
                return None
        try:
            return linpep_from_map(monomer_list)
        except Exception as ex:
            warnings.warn(f"get_smi_from_map: linear assembly failed for "
                          f"{map_str!r}: {ex}")
            return None

    # Build linker strings for cyclize_linpep_from_map
    cyclic_linkers = []
    leading_cap_count = int(bool(monomer_list and monomer_list[0] == 'ac'))
    trailing_cap_count = int(bool(
        monomer_list and monomer_list[-1] in {'nme', 'nh2'}
    ))
    first_peptide_position = 1 + leading_cap_count
    last_peptide_position = len(monomer_list) - trailing_cap_count
    for linker in linkers:
        # Detailed format: {cyc:1:R1-8:R2}
        dt_m = re.search(r'\{cyc:\s*([N\d]+):(R\d)-([C\d]+):(R\d)\}', linker)
        if dt_m:
            start_raw = dt_m.group(1); start_rg = dt_m.group(2)
            end_raw = dt_m.group(3); end_rg = dt_m.group(4)
            start_pos = (
                first_peptide_position
                if start_raw == 'N'
                else int(start_raw) + leading_cap_count
            )
            end_pos = (
                last_peptide_position
                if end_raw == 'C'
                else int(end_raw) + leading_cap_count
            )
            cyclic_linkers.append(f'{start_pos}:{start_rg}-{end_pos}:{end_rg}')
        else:
            # Simple format: {cyc:N-C} or {cyc:2-5}
            sm = re.search(r'\{cyc:\s*(N|\d+)-([C]|\d+)\}', linker)
            if not sm:
                continue
            start_part, end_part = sm.group(1), sm.group(2)
            cl = ''
            if start_part == '1' or start_part == 'N':
                cl += f'{first_peptide_position}:R1-'
            else:
                cl += f'{int(start_part) + leading_cap_count}:R3-'
            semantic_length = last_peptide_position - leading_cap_count
            if end_part == 'C' or end_part == str(semantic_length):
                cl += f'{last_peptide_position}:R2'
            else:
                cl += f'{int(end_part) + leading_cap_count}:R3'
            cyclic_linkers.append(cl)

    endpoint_error = _validate_assembled_cyc_linkers(
        cyclic_linkers,
        first_peptide_position,
        last_peptide_position,
    )
    if endpoint_error is not None:
        warnings.warn(
            f"get_smi_from_map: invalid cyclization endpoint for {map_str!r}: "
            f"{endpoint_error}"
        )
        return None

    # Reject topologies the assembler cannot build: a single attachment point
    # (pos, R-group) used by two bonds (e.g. one residue bridging two rings).
    conflict = _detect_conflicting_linkers(cyclic_linkers)
    if conflict is not None:
        warnings.warn(f"get_smi_from_map: unsupported topology for {map_str!r}: "
                      f"attachment point {conflict} is used by more than one "
                      f"cyclization bond (same-residue multi-ring not supported)")
        return None

    # Reject R3 bonds on residues that have no side-chain attachment point
    # (e.g. {cyc:1:R3-...} where residue 1 is Ala). This catches position-index
    # mistakes that would otherwise fail opaquely inside the assembler.
    bad_r3 = _validate_r3_available(cyclic_linkers, monomer_list)
    if bad_r3 is not None:
        warnings.warn(f"get_smi_from_map: invalid R3 cyclization for {map_str!r}: "
                      f"{bad_r3}")
        return None

    try:
        return cyclize_linpep_from_map(monomer_list, cyclic_linkers,
                                       chain_breaks=chain_breaks)
    except Exception as ex:
        warnings.warn(f"get_smi_from_map: cyclization assembly failed for "
                      f"{map_str!r}: {ex}")
        return None


_STRICT_HELM_BLOCK = re.compile(r'(PEPTIDE[1-9][0-9]*)\{([^{}]*)\}')
_STRICT_HELM_CONNECTION = re.compile(
    r'(PEPTIDE[1-9][0-9]*),(PEPTIDE[1-9][0-9]*),'
    r'([1-9][0-9]*):R([1-3])-([1-9][0-9]*):R([1-3])'
)
_STRICT_BILN_TOKEN = re.compile(
    r'([A-Z]|\[[^\[\]]+\])((?:\([1-9][0-9]*,[1-3]\))*)'
)
_STRICT_BILN_ANNOTATION = re.compile(r'\(([1-9][0-9]*),([1-3])\)')
_STRICT_MAP_EDGE = re.compile(
    r'\{cyc:([1-9][0-9]*):R([1-3])-([1-9][0-9]*):R([1-3])\}'
)


def _monomer_has_rgroup(symbol, rgroup):
    live = monomers2r_groups_dict.get(symbol)
    if live is not None:
        value = str(live.get(f'R{rgroup}', '')).strip()
        return value not in ('', '-')
    row = _unified_by_symbol.get(symbol)
    if row is None:
        return False
    value = str(row.get(f'R{rgroup}', '')).strip()
    return value not in ('', '-')


def _validate_notation_edges(chains, edges):
    """Validate range, port availability, and single use for notation edges."""
    used = set()
    for left, right in edges:
        for chain_idx, residue_idx, rgroup in (left, right):
            if chain_idx < 0 or chain_idx >= len(chains):
                raise ValueError('connection references an unknown chain')
            chain = chains[chain_idx]
            if residue_idx < 1 or residue_idx > len(chain):
                raise ValueError('connection endpoint is out of range')
            if not _monomer_has_rgroup(chain[residue_idx - 1], rgroup):
                raise ValueError(
                    f'R{rgroup} is unavailable on the referenced monomer'
                )
            if rgroup == 1 and residue_idx != 1:
                raise ValueError('R1 is occupied by an implicit backbone bond')
            if rgroup == 2 and residue_idx != len(chain):
                raise ValueError('R2 is occupied by an implicit backbone bond')
            endpoint = (chain_idx, residue_idx, rgroup)
            if endpoint in used:
                raise ValueError('attachment endpoint is reused')
            used.add(endpoint)


def _strip_helm_token(token):
    if token.startswith('[') or token.endswith(']'):
        if not (token.startswith('[') and token.endswith(']')):
            raise ValueError('unbalanced HELM monomer brackets')
        token = token[1:-1]
    if not token:
        raise ValueError('empty HELM monomer')
    return token


def _parse_helm_strict(helm):
    if not isinstance(helm, str):
        raise ValueError('HELM must be a string')
    sections = helm.split('$')
    if len(sections) != 5 or sections[2] or sections[3]:
        raise ValueError('HELM must contain all five sections')
    if sections[4] and not re.fullmatch(r'V[0-9]+(?:\.[0-9]+)?', sections[4]):
        raise ValueError('invalid HELM version section')
    raw_blocks = sections[0].split('|') if sections[0] else []
    if not raw_blocks:
        raise ValueError('HELM contains no polymer')
    identifiers = []
    chains = []
    for raw in raw_blocks:
        match = _STRICT_HELM_BLOCK.fullmatch(raw)
        if match is None:
            raise ValueError('invalid HELM polymer block')
        identifier, payload = match.groups()
        if identifier in identifiers:
            raise ValueError('duplicate HELM polymer identifier')
        tokens = payload.split('.') if payload else []
        if not tokens or any(not token for token in tokens):
            raise ValueError('empty HELM polymer or monomer')
        chain = [_strip_helm_token(token) for token in tokens]
        if 'ac' in chain and (chain.count('ac') != 1 or chain[0] != 'ac'):
            raise ValueError('HELM N-terminal cap must occur exactly at chain start')
        for cap in ('nme', 'nh2'):
            if cap in chain and (chain.count(cap) != 1 or chain[-1] != cap):
                raise ValueError('HELM C-terminal cap must occur exactly at chain end')
        if 'nme' in chain and 'nh2' in chain:
            raise ValueError('HELM chain cannot contain multiple C-terminal caps')
        identifiers.append(identifier)
        chains.append(chain)
    chain_index = {identifier: index for index, identifier in enumerate(identifiers)}
    edges = []
    if sections[1]:
        for raw in sections[1].split('|'):
            match = _STRICT_HELM_CONNECTION.fullmatch(raw)
            if match is None:
                raise ValueError('invalid HELM connection')
            left_id, right_id, left_pos, left_rg, right_pos, right_rg = match.groups()
            if left_id not in chain_index or right_id not in chain_index:
                raise ValueError('HELM connection references an unknown polymer')
            edges.append((
                (chain_index[left_id], int(left_pos), int(left_rg)),
                (chain_index[right_id], int(right_pos), int(right_rg)),
            ))
    _validate_notation_edges(chains, edges)
    return chains, edges


def _map_tokenize_strict(segment):
    tokens = []
    position = 0
    while position < len(segment):
        matched = False
        for denotion, symbol in map_to_helm_dict.items():
            if denotion.startswith('{nt:') or denotion.startswith('{ct:'):
                continue
            if segment.startswith(denotion, position):
                tokens.append(symbol)
                position += len(denotion)
                matched = True
                break
        if not matched:
            raise ValueError(f'unrecognized or malformed MAP token at offset {position}')
    if not tokens:
        raise ValueError('empty MAP chain segment')
    return tokens


def _global_map_endpoint(chains, position, rgroup):
    offset = 0
    for chain_index, chain in enumerate(chains):
        if position <= offset + len(chain):
            return chain_index, position - offset, rgroup
        offset += len(chain)
    raise ValueError('MAP connection endpoint is out of range')


_NTERM_MODIFIER_MAP = {"ACE": "ac"}
_CTERM_MODIFIER_MAP = {"NME": "nme", "NH2": "nh2"}
_MODIFIER_TOKEN_RE = re.compile(r"\{(?:nt|ct):[^{}]+\}")
_MODIFIER_CODE_RE = re.compile(r"\{((?:nt|ct)):([^{}]+)\}")
# Registered in MAP_momomers_library_new.csv but not supported by assembly.
_UNSUPPORTED_REGISTERED_MODIFIERS = frozenset({"DKA", "GOA", "PPD", "MOR"})


def _validate_map_terminal_modifier_positions(map_str):
    """Reject terminal-modifier tokens embedded between chain content.

    ``{nt:...}``/``{ct:...}`` markers are legal only at a chain endpoint:
    there must not be non-modifier content on both sides of the token.
    """
    for match in _MODIFIER_TOKEN_RE.finditer(map_str):
        before = _MODIFIER_TOKEN_RE.sub("", map_str[:match.start()])
        after = _MODIFIER_TOKEN_RE.sub("", map_str[match.end():])
        if before.strip() and after.strip():
            raise ValueError(
                f"misplaced MAP terminal modifier {match.group(0)} "
                "(must be at a chain endpoint)"
            )


def _unsupported_modifier_message(side, code):
    note = (
        " (registered in the MAP monomer library but not supported)"
        if code in _UNSUPPORTED_REGISTERED_MODIFIERS
        else ""
    )
    return f"unsupported MAP {side} modifier {code!r}{note}"


def _parse_map_strict(map_str):
    if not isinstance(map_str, str) or not map_str:
        raise ValueError('MAP must be a nonempty string')
    _validate_map_terminal_modifier_positions(map_str)
    nterm_matches = re.findall(r'\{nt:([^{}]+)\}', map_str)
    cterm_matches = re.findall(r'\{ct:([^{}]+)\}', map_str)
    if len(nterm_matches) > 1 or len(cterm_matches) > 1:
        raise ValueError('duplicate MAP terminal modifier')
    working = re.sub(r'\{nt:[^{}]+\}', '', map_str)
    working = re.sub(r'\{ct:[^{}]+\}', '', working)
    edge_specs = []
    for match in _STRICT_MAP_EDGE.finditer(working):
        edge_specs.append(tuple(map(int, match.groups())))
    working = _STRICT_MAP_EDGE.sub('', working)
    head_to_tail_count = working.count('{cyc:N-C}')
    if head_to_tail_count > 1:
        raise ValueError('duplicate MAP head-to-tail connection')
    working = working.replace('{cyc:N-C}', '')
    if '{cyc:' in working:
        raise ValueError('malformed or unsupported MAP annotation')
    segments = working.split('{br}')
    if any(not segment for segment in segments):
        raise ValueError('empty MAP chain segment')
    chains = [_map_tokenize_strict(segment) for segment in segments]
    if (nterm_matches or cterm_matches) and len(chains) != 1:
        raise ValueError('MAP terminal modifiers are ambiguous for multiple chains')
    edges = []
    if head_to_tail_count:
        if len(chains) != 1 or nterm_matches or cterm_matches:
            raise ValueError('MAP head-to-tail connection requires one uncapped chain')
        edges.append(((0, 1, 1), (0, len(chains[0]), 2)))
    for left_pos, left_rg, right_pos, right_rg in edge_specs:
        edges.append((
            _global_map_endpoint(chains, left_pos, left_rg),
            _global_map_endpoint(chains, right_pos, right_rg),
        ))
    if nterm_matches:
        cap = _NTERM_MODIFIER_MAP.get(nterm_matches[0])
        if cap is None:
            raise ValueError(
                _unsupported_modifier_message('N-terminal', nterm_matches[0])
            )
        chains[0].insert(0, cap)
        edges = [
            (
                (left[0], left[1] + (left[0] == 0), left[2]),
                (right[0], right[1] + (right[0] == 0), right[2]),
            )
            for left, right in edges
        ]
    if cterm_matches:
        cap = _CTERM_MODIFIER_MAP.get(cterm_matches[0])
        if cap is None:
            raise ValueError(
                _unsupported_modifier_message('C-terminal', cterm_matches[0])
            )
        chains[0].append(cap)
    _validate_notation_edges(chains, edges)
    return chains, edges


def _split_biln_top_level(value, separator):
    """Split BILN only outside bracketed monomer names."""
    parts = []
    start = 0
    bracketed = False
    for index, character in enumerate(value):
        if character == '[':
            if bracketed:
                raise ValueError('nested BILN monomer brackets are invalid')
            bracketed = True
        elif character == ']':
            if not bracketed:
                raise ValueError('unbalanced BILN monomer brackets')
            bracketed = False
        elif character == separator and not bracketed:
            parts.append(value[start:index])
            start = index + 1
    if bracketed:
        raise ValueError('unbalanced BILN monomer brackets')
    parts.append(value[start:])
    return parts


def _parse_biln_strict(biln_str):
    if not isinstance(biln_str, str) or not biln_str:
        raise ValueError('BILN must be a nonempty string')
    chains = []
    bond_endpoints = {}
    for chain_index, segment in enumerate(_split_biln_top_level(biln_str, '.')):
        if not segment:
            raise ValueError('empty BILN chain segment')
        chain = []
        annotations = []
        for residue in _split_biln_top_level(segment, '-'):
            match = _STRICT_BILN_TOKEN.fullmatch(residue)
            if match is None:
                raise ValueError('invalid BILN residue or annotation')
            token, suffix = match.groups()
            symbol = token[1:-1] if token.startswith('[') else token
            chain.append(symbol)
            annotations.append([
                (int(bond_id), int(rgroup))
                for bond_id, rgroup in _STRICT_BILN_ANNOTATION.findall(suffix)
            ])
        chains.append(chain)
        for residue_index, residue_annotations in enumerate(annotations, 1):
            for bond_id, rgroup in residue_annotations:
                bond_endpoints.setdefault(bond_id, []).append(
                    (chain_index, residue_index, rgroup)
                )
    edges = []
    for bond_id in sorted(bond_endpoints):
        endpoints = bond_endpoints[bond_id]
        if len(endpoints) != 2:
            raise ValueError('BILN bond identifier must occur exactly twice')
        edges.append((endpoints[0], endpoints[1]))
    _validate_notation_edges(chains, edges)
    return chains, edges


def _format_helm(chains, edges, version='V2.0'):
    blocks = []
    for chain_index, chain in enumerate(chains, 1):
        sequence = '.'.join(f'[{symbol}]' if len(symbol) > 1 else symbol for symbol in chain)
        blocks.append(f'PEPTIDE{chain_index}{{{sequence}}}')
    connections = []
    for left, right in edges:
        connections.append(
            f'PEPTIDE{left[0] + 1},PEPTIDE{right[0] + 1},'
            f'{left[1]}:R{left[2]}-{right[1]}:R{right[2]}'
        )
    return f"{'|'.join(blocks)}${'|'.join(connections)}$$${version}"


def helm_to_map(helm):
    """Convert HELM string to MAP notation.  Inverse of map_to_helm().

    Supports multiple connections separated by | (e.g. head-to-tail + disulfides).
    """
    chains, edges = _parse_helm_strict(helm)
    cap_present = any(
        chain and (chain[0] == 'ac' or chain[-1] in {'nme', 'nh2'})
        for chain in chains
    )
    if cap_present and len(chains) != 1:
        raise ValueError('terminal caps are ambiguous for multiple HELM polymers')
    fragments = [_elements_to_map(chain) for chain in chains]
    leading_cap = [bool(chain and chain[0] == 'ac') for chain in chains]
    semantic_lengths = [
        len(chain)
        - int(leading_cap[index])
        - int(bool(chain and chain[-1] in {'nme', 'nh2'}))
        for index, chain in enumerate(chains)
    ]
    offsets = []
    running = 0
    for length in semantic_lengths:
        offsets.append(running)
        running += length
    tags = []
    for left, right in edges:
        if (
            len(chains) == 1
            and left == (0, 1, 1)
            and right == (0, len(chains[0]), 2)
        ):
            tags.append('{cyc:N-C}')
            continue
        left_global = offsets[left[0]] + left[1] - int(leading_cap[left[0]])
        right_global = offsets[right[0]] + right[1] - int(leading_cap[right[0]])
        tags.append(
            f'{{cyc:{left_global}:R{left[2]}-{right_global}:R{right[2]}}}'
        )
    map_format = '{br}'.join(fragments)
    nterm_mods = re.findall(r'\{nt:[^}]+\}', map_format)
    cterm_mods = re.findall(r'\{ct:[^}]+\}', map_format)
    map_format = re.sub(r'\{nt:[^}]+\}|\{ct:[^}]+\}', '', map_format)
    return map_format + ''.join(tags) + ''.join(nterm_mods) + ''.join(cterm_mods)


def _elements_to_map(elements):
    """Convert a list of HELM monomer symbols to a MAP fragment string."""
    out = ''
    for element in elements:
        if element == 'ac':
            out += '{nt:ACE}'
        elif element == 'nme':
            out += '{ct:NME}'
        elif element == 'nh2':
            out += '{ct:NH2}'
        elif element in _symbol_to_map:
            out += _symbol_to_map[element]
        else:
            out += _auto_map_denotion(element)
    return out


def _helm_to_map_single(helm):
    """Single-polymer HELM -> MAP (the original behaviour, unchanged)."""
    start = helm.index('{') + 1
    end = helm.index('}')
    helm_sequence = helm[start:end]
    elements = [elem.strip('[]') for elem in helm_sequence.split('.')]
    num_elements = len(elements)
    map_format = _elements_to_map(elements)

    dollar_split = helm.split('$')
    cyc_parts = []
    if len(dollar_split) > 2 and dollar_split[1]:
        # Parse ALL connections (may be |-separated)
        conn_strings = dollar_split[1].split('|')
        for conn_str in conn_strings:
            if not conn_str:
                continue
            last_element = conn_str.split(',')[-1]
            start_full = last_element.split('-')[0]
            end_full = last_element.split('-')[1]
            start_pos = start_full.split(':')[0]
            start_rgroup = start_full.split(':')[1]
            end_pos = end_full.split(':')[0]
            end_rgroup = end_full.split(':')[1]

            if (int(start_pos) == 1 and int(end_pos) == num_elements
                    and start_rgroup == 'R1' and end_rgroup == 'R2'):
                cyc_parts.append('{cyc:N-C}')
            else:
                cyc_parts.append(f'{{cyc:{start_pos}:{start_rgroup}-{end_pos}:{end_rgroup}}}')

    # Reorder: sequence first, then cyc tags, then terminal mods at end
    nterm_pattern = r'\{nt:[^}]+\}'
    cterm_pattern = r'\{ct:[^}]+\}'
    nterm_mods = re.findall(nterm_pattern, map_format)
    map_format = re.sub(nterm_pattern, '', map_format)
    cterm_mods = re.findall(cterm_pattern, map_format)
    map_format = re.sub(cterm_pattern, '', map_format)

    return map_format + ''.join(cyc_parts) + ''.join(nterm_mods) + ''.join(cterm_mods)


def _helm_to_map_multi(helm, blocks):
    """Multi-polymer HELM -> single MAP with {br} chain breaks and global
    cyclization positions.

    blocks: list of (polymer_id, sequence) from the HELM list part, e.g.
    [('PEPTIDE1', 'G.I...'), ('PEPTIDE2', 'F.V...')]. Connections in the
    HELM '$...$' section reference (polymer_id, position), translated here to
    global positions (chain offset + local position).
    """
    # Per-chain element lists, global offsets, and combined sequence.
    chain_elements = {}
    chain_offset = {}
    seq_fragments = []
    running = 0
    for pid, seq in blocks:
        elements = [e.strip('[]') for e in seq.split('.')] if seq else []
        chain_elements[pid] = elements
        chain_offset[pid] = running          # 0-based offset of this chain
        running += len(elements)
        seq_fragments.append(_elements_to_map(elements))
    total = running

    # Join chains with the {br} break marker (no auto backbone bond across it).
    map_format = '{br}'.join(seq_fragments)

    # Translate connections to global positions.
    cyc_parts = []
    dollar_split = helm.split('$')
    if len(dollar_split) > 2 and dollar_split[1]:
        for conn_str in dollar_split[1].split('|'):
            if not conn_str:
                continue
            parts = conn_str.split(',')
            # Expected: PEPTIDEx,PEPTIDEy,i:Rg-j:Rg
            if len(parts) < 3:
                continue
            src_pid, tgt_pid, rg_part = parts[0], parts[1], parts[-1]
            start_full, end_full = rg_part.split('-')
            s_pos, s_rg = start_full.split(':')
            e_pos, e_rg = end_full.split(':')
            g_start = chain_offset.get(src_pid, 0) + int(s_pos)
            g_end = chain_offset.get(tgt_pid, 0) + int(e_pos)
            if (g_start == 1 and g_end == total
                    and s_rg == 'R1' and e_rg == 'R2'):
                cyc_parts.append('{cyc:N-C}')
            else:
                cyc_parts.append(f'{{cyc:{g_start}:{s_rg}-{g_end}:{e_rg}}}')

    # Reorder terminal mods to the end, like the single-chain path.
    nterm_pattern = r'\{nt:[^}]+\}'
    cterm_pattern = r'\{ct:[^}]+\}'
    nterm_mods = re.findall(nterm_pattern, map_format)
    map_format = re.sub(nterm_pattern, '', map_format)
    cterm_mods = re.findall(cterm_pattern, map_format)
    map_format = re.sub(cterm_pattern, '', map_format)

    return map_format + ''.join(cyc_parts) + ''.join(nterm_mods) + ''.join(cterm_mods)


def map_to_helm(map_str):
    """Convert MAP notation to HELM string.  Inverse of helm_to_map().

    Handles multiple {cyc:...} tags (multi-disulfide, head-to-tail+disulfide, etc.)
    and {nt:ACE}/{ct:NME}/{ct:NH2} terminal modifiers (inserted as sequence monomers).
    """
    chains, edges = _parse_map_strict(map_str)
    return _format_helm(chains, edges)


# ════════════════════════════════════════════════════════════════════════════
# BILN (Boehringer Ingelheim Line Notation) interconversion
# ════════════════════════════════════════════════════════════════════════════

def biln_to_helm(biln_str: str) -> str:
    """Convert a BILN string to HELM notation.

    BILN format:
      - Monomers separated by ``-`` within a chain
      - Chains separated by ``.``
      - Cross-links annotated as ``residue(bond_id, R_group)`` where
        bond_id is a shared integer linking two residues and R_group is
        1 (N-term), 2 (C-term), or 3 (side-chain)

    Examples::

        biln_to_helm("P-E-P-T-I-D-E")
        # -> "PEPTIDE1{P.E.P.T.I.D.E}$$$$V2.0"

        biln_to_helm("C(1,3)-A-A-A-C(1,3)")
        # -> "PEPTIDE1{C.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3$$$V2.0"

        biln_to_helm("A-G.K(1,3)-E.G-L-E-E(1,3)")
        # -> two chains with a cross-link
    """
    chains, edges = _parse_biln_strict(biln_str)
    return _format_helm(chains, edges)


def helm_to_biln(helm_str: str) -> str:
    """Convert a HELM string to BILN notation.

    Inverse of :func:`biln_to_helm`. Cross-link information from HELM
    connection records is embedded into monomer annotations as
    ``residue(bond_id, R_group)``.
    """
    chains, edges = _parse_helm_strict(helm_str)
    chain_copies = [
        [f'[{symbol}]' if len(symbol) > 1 else symbol for symbol in chain]
        for chain in chains
    ]
    for bond_id, (left, right) in enumerate(edges, 1):
        for chain_index, residue_index, rgroup in (left, right):
            value = chain_copies[chain_index][residue_index - 1]
            chain_copies[chain_index][residue_index - 1] = (
                f'{value}({bond_id},{rgroup})'
            )
    return '.'.join('-'.join(chain) for chain in chain_copies)


def get_smi_from_biln(biln_str: str):
    """Convert a BILN string to SMILES via HELM → MAP → SMILES.

    Convenience wrapper: ``biln_to_helm`` → ``helm_to_map`` →
    ``get_smi_from_map``.
    """
    helm = biln_to_helm(biln_str)
    mp = helm_to_map(helm)
    if not mp or mp.startswith('ERROR'):
        return None
    return get_smi_from_map(mp)
