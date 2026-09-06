"""PDB utilities shared across cycpep_master (and downstream consumers).

Currently provides chain extraction — pulling a single chain (ATOM/HETATM
records of one chain id) from a multi-chain PDB into a standalone PDB. Used
by the multi-source template library (CPBind/CPSea complex PDBs: extract
chain L = peptide) and reusable by any caller that needs chain isolation.
"""
import gzip
import os
import threading
import tempfile
from typing import Iterable, Iterator, Optional


_PDB_TEXT_CACHE: dict[tuple, tuple[str, ...]] = {}
_PDB_TEXT_CACHE_ORDER: list[tuple] = []
_PDB_TEXT_CACHE_MAX = 32
_PDB_TEXT_CACHE_LOCK = threading.RLock()


def _pdb_file_key(path: str) -> tuple | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (
        os.path.normcase(os.path.abspath(str(path))),
        stat.st_mtime_ns,
        stat.st_size,
    )


def read_pdb_lines(path: str) -> tuple[str, ...]:
    """Read one PDB payload once per file identity as immutable lines."""
    key = _pdb_file_key(path)
    if key is not None:
        with _PDB_TEXT_CACHE_LOCK:
            cached = _PDB_TEXT_CACHE.get(key)
            if cached is not None:
                _PDB_TEXT_CACHE_ORDER.remove(key)
                _PDB_TEXT_CACHE_ORDER.append(key)
                return cached
    opener = gzip.open if str(path).endswith(".gz") else open
    kwargs = {"encoding": "utf-8", "errors": "replace"}
    with opener(path, "rt", **kwargs) as handle:
        lines = tuple(handle)
    if key is not None:
        with _PDB_TEXT_CACHE_LOCK:
            if key not in _PDB_TEXT_CACHE:
                _PDB_TEXT_CACHE[key] = lines
                _PDB_TEXT_CACHE_ORDER.append(key)
                while len(_PDB_TEXT_CACHE_ORDER) > _PDB_TEXT_CACHE_MAX:
                    stale = _PDB_TEXT_CACHE_ORDER.pop(0)
                    _PDB_TEXT_CACHE.pop(stale, None)
            else:
                lines = _PDB_TEXT_CACHE[key]
    return lines


def read_first_model_lines(path: str) -> tuple[str, ...]:
    return tuple(first_model_records(read_pdb_lines(path)))


def _clear_pdb_text_cache() -> None:
    with _PDB_TEXT_CACHE_LOCK:
        _PDB_TEXT_CACHE.clear()
        _PDB_TEXT_CACHE_ORDER.clear()


def first_model_records(lines: Iterable[str]) -> Iterator[str]:
    """Yield coordinates from the first MODEL block and global records.

    PDB model identifiers are labels and need not start at 1. Records outside
    MODEL blocks remain global; model-local CONECT records are retained only
    for the first block.
    """
    saw_model = False
    inside_model = False
    selected_model = False
    for line in lines:
        if line.startswith("MODEL"):
            selected_model = not saw_model
            saw_model = True
            inside_model = True
            continue
        if line.startswith("ENDMDL"):
            inside_model = False
            selected_model = False
            continue
        if (
            saw_model
            and line[:6] in ("ATOM  ", "HETATM")
            and not selected_model
        ):
            continue
        if (
            line.startswith(("CONECT", "LINK  ", "SSBOND"))
            and inside_model
            and not selected_model
        ):
            continue
        yield line


def pdb_atom_element(line: str) -> str:
    """Return a PDB atom element, respecting atom-name field alignment."""
    declared = line[76:78].strip() if len(line) >= 78 else ""
    if declared:
        return declared.capitalize()
    raw_name = line[12:16] if len(line) >= 16 else ""
    if not raw_name.strip():
        return ""
    # One-letter elements are right-justified in the four-column atom field;
    # two-letter elements are left-justified. Leading digits denote isotopic
    # hydrogen-style names, so use the next alphabetic character.
    if raw_name[0].isalpha():
        token = "".join(char for char in raw_name[:2] if char.isalpha())
        return token.capitalize()
    token = "".join(char for char in raw_name if char.isalpha())
    return token[:1].capitalize()


def _selected_conect_line(line: str, serials: set[int]) -> Optional[str]:
    """Return one CONECT record restricted to selected atom serials."""
    values = []
    for offset in range(6, len(line.rstrip("\r\n")), 5):
        token = line[offset:offset + 5].strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            return None
    if len(values) < 2 or values[0] not in serials:
        return None
    targets = [value for value in values[1:] if value in serials]
    if not targets:
        return None
    return "CONECT" + "".join(f"{value:5d}" for value in [values[0], *targets]) + "\n"


def extract_chain(pdb_path: str, chain_id: str, out_path: Optional[str] = None) -> Optional[str]:
    """Extract the first-model records for one PDB chain.

    Internal SSBOND, LINK and CONECT evidence is retained. Records that touch
    another chain are omitted because their other endpoint is absent from the
    extracted structure.

    Returns the output PDB path, or None if no matching-chain atoms found.
    """
    if out_path is None:
        out_path = os.path.join(tempfile.mkdtemp(prefix=f"chain_{chain_id}_"), f"chain_{chain_id}.pdb")
    try:
        with open(pdb_path, encoding="utf-8", errors="replace") as f:
            records = list(first_model_records(f))
    except Exception:
        return None

    atom_lines = [
        line for line in records
        if line[:6] in ("ATOM  ", "HETATM")
        and len(line) > 21
        and line[21] == chain_id
    ]
    if not atom_lines:
        return None

    serials = set()
    for line in atom_lines:
        try:
            serials.add(int(line[6:11]))
        except ValueError:
            continue

    connection_lines = []
    for line in records:
        if line.startswith("SSBOND"):
            if len(line) > 29 and line[15] == chain_id and line[29] == chain_id:
                connection_lines.append(line)
        elif line.startswith("LINK  "):
            if len(line) > 51 and line[21] == chain_id and line[51] == chain_id:
                connection_lines.append(line)
        elif line.startswith("CONECT"):
            selected = _selected_conect_line(line, serials)
            if selected is not None:
                connection_lines.append(selected)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.writelines([*connection_lines, *atom_lines, "END\n"])
    return out_path
