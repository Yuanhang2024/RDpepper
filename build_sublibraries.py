"""Derive per-source monomer sub-libraries from the unified library.

The unified library (``unified_monomer_library.csv``, 13152 monomers, 235
columns) is the source for the three large source slices. The tracked
``core.csv`` and ``caps.csv`` files are the authoritative reconstruction rows
for standard residues and terminal caps; this script validates and preserves
them while regenerating the source and special-residue slices. It is
idempotent: rerun to regenerate.

Each sub-library row keeps only the columns assembly needs
(``symbol, CXSMILES, R1, R2, R3, source``) plus three derived metadata columns:

  position_class : flexible / N_terminal_only / C_terminal_only / unknown
                   (from which backbone R-groups are open)
  has_sidechain  : True/False (R3 != '-')
  prop_status    : full / struct_only / none (property-label completeness)

Property *values* (logP/TPSA/qed/contain_perme) are NOT copied here; join back
to the unified library by ``symbol`` when needed. See FUNCTION_INDEX.
"""
import csv
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_UNIFIED = os.path.join(_DIR, "unified_monomer_library.csv")
_SPECIAL = os.path.join(_DIR, "special_residue_library.csv")
_OUT_DIR = os.path.join(_DIR, "libraries")
_STATIC_DIR = _OUT_DIR

# Sub-library column schema (order matters for stable diffs).
_COLS = ["symbol", "CXSMILES", "R1", "R2", "R3", "source",
         "position_class", "has_sidechain", "prop_status"]

# source value -> sub-library file stem
_SOURCE_TO_STEM = {
    "CycPeptMPDB": "curated_cycpep",
    "NNAA": "nnaa_diverse",
    "HELM-GPT": "helm_gpt",
}

_DEFAULT_MANIFEST = {
    "load": [
        "caps",
        "core",
        "curated_cycpep",
        "nnaa_diverse",
        "helm_gpt",
        "special",
    ],
    # Metadata only: which slices form the model's training vocabulary vs.
    # inference-time expansion candidates (rare/diverse NNAAs). Not wired into
    # any generation or training logic in this layer.
    "training_vocab": ["core", "curated_cycpep"],
}


def _position_class(r1, r2):
    has1, has2 = r1 not in ("", "-"), r2 not in ("", "-")
    if has1 and has2:
        return "flexible"
    if has2:
        return "N_terminal_only"
    if has1:
        return "C_terminal_only"
    return "unknown"


def _prop_status(row):
    """full = has membrane label; struct_only = has RDKit descriptors only."""
    if str(row.get("contain_perme", "")).strip() not in ("", "nan"):
        return "full"
    if str(row.get("qed", "")).strip() not in ("", "nan"):
        return "struct_only"
    return "none"


def _meta_row(symbol, cx, r1, r2, r3, source, prop_status):
    r1 = (r1 or "-").strip() or "-"
    r2 = (r2 or "-").strip() or "-"
    r3 = (r3 or "-").strip() or "-"
    return {
        "symbol": symbol, "CXSMILES": cx, "R1": r1, "R2": r2, "R3": r3,
        "source": source,
        "position_class": _position_class(r1, r2),
        "has_sidechain": str(r3 != "-"),
        "prop_status": prop_status,
    }


def _write(stem, rows):
    path = os.path.join(_OUT_DIR, stem + ".csv")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(temporary, path)
    return len(rows)


def _build_unified_slices():
    """Slice the unified library by source into three sub-libraries."""
    buckets = {stem: [] for stem in _SOURCE_TO_STEM.values()}
    with open(_UNIFIED, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = str(row.get("symbol", "")).strip()
            src = str(row.get("source", "")).strip()
            if not sym or src not in _SOURCE_TO_STEM:
                continue
            buckets[_SOURCE_TO_STEM[src]].append(_meta_row(
                sym, str(row.get("CXSMILES", "")).strip(),
                str(row.get("R1", "")), str(row.get("R2", "")),
                str(row.get("R3", "")), src, _prop_status(row)))
    return buckets


def _read_static_slice(stem):
    """Read one tracked authoritative reconstruction slice."""
    path = os.path.join(_STATIC_DIR, stem + ".csv")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _COLS:
            raise ValueError(f"{path} does not use the sub-library schema")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no monomer rows")
    return rows


def _build_core():
    """Preserve the tracked standard-amino-acid reconstruction rows."""
    rows = _read_static_slice("core")
    if len(rows) != 20 or len({row["symbol"] for row in rows}) != 20:
        raise ValueError("core.csv must contain 20 unique standard residues")
    return rows


def _build_caps():
    """Preserve the tracked terminal-cap reconstruction rows."""
    rows = _read_static_slice("caps")
    if {row["symbol"] for row in rows} != {"ac", "nme", "nh2"}:
        raise ValueError("caps.csv must contain ac, nme, and nh2")
    return rows


def _build_special():
    """Export the special-residue library to the sub-library schema."""
    if not os.path.exists(_SPECIAL):
        return []
    rows = []
    with open(_SPECIAL, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            rows.append(_meta_row(
                sym, str(row.get("cxsmiles", "")).strip(),
                str(row.get("r1", "")), str(row.get("r2", "")),
                str(row.get("r3", "")), "special", "none"))
    return rows


def main():
    # Materialize and validate every slice before writing any output. A broken
    # static layer must not leave a half-regenerated library directory behind.
    slices = _build_unified_slices()
    slices["caps"] = _build_caps()
    slices["core"] = _build_core()
    slices["special"] = _build_special()

    unified_total = sum(
        len(slices[stem]) for stem in _SOURCE_TO_STEM.values()
    )
    with open(_UNIFIED, "r", encoding="utf-8-sig") as f:
        n_unified = sum(1 for _ in csv.DictReader(f))
    if unified_total != n_unified:
        raise ValueError(
            "unified source slices do not conserve rows: "
            f"slices={unified_total}, unified={n_unified}"
        )

    os.makedirs(_OUT_DIR, exist_ok=True)
    counts = {}
    for stem in _DEFAULT_MANIFEST["load"]:
        counts[stem] = _write(stem, slices[stem])

    manifest_path = os.path.join(_OUT_DIR, "manifest.json")
    manifest_temporary = manifest_path + ".tmp"
    with open(manifest_temporary, "w", encoding="utf-8") as f:
        json.dump(_DEFAULT_MANIFEST, f, ensure_ascii=False, indent=2)
    os.replace(manifest_temporary, manifest_path)

    print("sub-library counts:", counts)
    print(f"unified slices sum = {unified_total}, unified rows = {n_unified} OK")
    return counts


if __name__ == "__main__":
    main()
