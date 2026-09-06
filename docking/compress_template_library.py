"""Compress an existing template library by re-clustering centroids.

After a full-scale build (e.g. CPBind 136k centers), the FP@0.7 first pass may
leave too many centroids (observed: 9033, 178MB). This script re-clusters the
existing centroids within each (n_res, cyc_mode) bucket at a stricter FP
Tanimoto threshold (default 0.85), keeping only the new centroids — preferring
binding-state sources (cpsea > cpbind > scaffold) as representatives. Deletes
the now-redundant PDB files. Minutes, not hours (operates on centroids only).

Usage:
    python -m cycpep_master.docking.compress_template_library \
        --templates-dir cycpep_master/data/templates \
        --threshold 0.85 --max-per-bucket 15
"""
import argparse
import json
import os
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem

SOURCE_PRIORITY = {"cpsea": 0, "cpbind": 1, "scaffold": 2}


def _fp(mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol)


def _tanimoto(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    inter = float((a * b).sum()); union = float(a.sum() + b.sum() - inter)
    return inter / union if union > 0 else 0.0


def compress(templates_dir: str, threshold: float = 0.85, max_per_bucket: int = 15):
    idx_path = os.path.join(templates_dir, "templates_index.json")
    with open(idx_path, encoding="utf-8") as f:
        index = json.load(f)

    # group entries by bucket (n_res_cyc)
    buckets = defaultdict(list)
    for key, entry in index.items():
        bucket = key.rsplit("_", 1)[0]  # "16_N-C" from "16_N-C_007"
        entry["_key"] = key
        entry["_orig_pdb_path"] = entry.get("pdb_path", "")
        buckets[bucket].append(entry)

    kept = {}
    removed_pdb = []
    for bucket, entries in sorted(buckets.items()):
        kept_in_bucket = {}  # new_key -> entry, populated as we rename
        # compute FP for each centroid
        fps = []
        for e in entries:
            mol = Chem.MolFromSmiles(e["smiles"])
            fps.append(_fp(mol) if mol is not None else None)
        # sort by source priority so binding-state centroids are seeded first
        order = sorted(range(len(entries)),
                       key=lambda i: (SOURCE_PRIORITY.get(entries[i].get("source", ""), 99), i))
        centroids = []  # indices into entries
        for i in order:
            if fps[i] is None:
                continue
            if any(_tanimoto(fps[i], fps[c]) >= threshold for c in centroids):
                continue
            centroids.append(i)
            if len(centroids) >= max_per_bucket:
                break
        # re-key centroids 001..NNN — two-phase rename to avoid clobbering:
        # phase 1 moves each kept PDB to a unique temp name, phase 2 renames
        # temp→final. Direct old→new rename can overwrite a still-kept file
        # when renumbering shifts centroids (e.g. old_006→new_001 overwrites
        # old_001). Temp names are unique so no collision.
        rename_plan = []  # (temp_path, final_path, new_pdb_rel)
        for k, i in enumerate(centroids, 1):
            e = entries[i]
            new_pdb_rel = f"{bucket}/centroid_{k:03d}.pdb"
            old_pdb = os.path.join(templates_dir, e["pdb_path"])
            final_pdb = os.path.join(templates_dir, new_pdb_rel)
            temp_pdb = os.path.join(templates_dir, bucket, f"_tmp_{k:03d}.pdb")
            rename_plan.append((old_pdb, temp_pdb, final_pdb, new_pdb_rel, e))
        # phase 1: old -> temp
        for old_pdb, temp_pdb, _, _, _ in rename_plan:
            if os.path.exists(old_pdb):
                os.replace(old_pdb, temp_pdb)
        # phase 2: temp -> final, update entry pdb_path
        for k, (_, temp_pdb, final_pdb, new_pdb_rel, e) in enumerate(rename_plan, 1):
            if os.path.exists(temp_pdb):
                os.replace(temp_pdb, final_pdb)
            e["pdb_path"] = new_pdb_rel
            e.pop("_key", None)
            e.pop("_orig_pdb_path", None)
            kept[f"{bucket}_{k:03d}"] = e
        # mark redundant PDBs for deletion (use ORIGINAL pdb_path, pre-rename)
        for i, e in enumerate(entries):
            if i not in centroids:
                old_pdb = os.path.join(templates_dir, e["_orig_pdb_path"])
                if os.path.exists(old_pdb):
                    removed_pdb.append(old_pdb)
                e.pop("_orig_pdb_path", None)

    for p in removed_pdb:
        try:
            os.remove(p)
        except OSError:
            pass
    # clean empty bucket dirs
    for bucket in buckets:
        d = os.path.join(templates_dir, bucket)
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)

    with open(idx_path, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"compressed: {len(index)} -> {len(kept)} centroids "
          f"(threshold={threshold}, max/bucket={max_per_bucket}), "
          f"removed {len(removed_pdb)} PDBs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-per-bucket", type=int, default=15)
    args = ap.parse_args()
    compress(args.templates_dir, args.threshold, args.max_per_bucket)


if __name__ == "__main__":
    main()
