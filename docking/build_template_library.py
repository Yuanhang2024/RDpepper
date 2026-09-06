"""Build a compact cyclic-peptide 3D template library from the Scaffold PDB set.

The Scaffold library (~20k theoretical-model PDBs, almost all N-C head-to-tail
standard-AA cyclic peptides) is too large to ship and too redundant to query
directly. This script clusters it by (residue_count, cyclization_mode) and
keeps only cluster centroids — a few MB of representative 3D conformers that
downstream `template_library.find_template` can borrow coordinates from.

Run once (or when Scaffold grows):
    python -m cycpep_master.docking.build_template_library \
        --scaffold-csv /path/to/intermediate_scaffold.csv \
        --scaffold-pdb-dir /path/to/AfCycDesign_Scaffold \
        --out-dir ./data/templates

Output:
    cycpep_master/data/templates/<res>_<cyc>/centroid_NNN.pdb
    cycpep_master/data/templates/templates_index.json
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign


RMSD_THRESHOLD = 1.5  # Å — same-shape cluster cutoff on heavy-atom backbone
FP_THRESHOLD = 0.7  # Morgan-FP Tanimoto — clusters chemically-similar peptides
                    # (raised from 0.6: stricter dedup so full-scale CPBind 136k
                    # centers converge to a manageable centroid count without a
                    # hard per-bucket cap. 0.7 keeps near-identical peptides merged.)
MAX_CENTROIDS_PER_BUCKET = 100000  # effectively no cap — let threshold converge


def _cyc_mode(map_str: str) -> str:
    """Extract cyclization mode tag from a MAP string ('N-C', 'SC', 'mixed')."""
    cycs = re.findall(r"\{cyc:([^}]+)\}", map_str)
    if not cycs:
        return "none"
    tags = set()
    for c in cycs:
        if "R3" in c:
            tags.add("SC")
        elif "-" in c and "R3" not in c:
            tags.add("N-C")
        else:
            tags.add("other")
    return "mixed" if len(tags) > 1 else next(iter(tags))


def _residue_count(map_str: str) -> int:
    # Count only residue symbols: bare standard-AA letters + {nnr:...} tokens.
    # Remove topology/cap annotations first; otherwise {cyc:N-C} contributes
    # spurious N/C residues (+2), which corrupts bucket keys.
    cleaned = re.sub(r"\{(?!nnr:)[^}]+\}", "", map_str)
    return len(re.findall(r"[A-Z]|\{nnr:[^}]+\}", cleaned))


def _load_template_mol(pdb_path: str, smiles: str, chain_id: Optional[str] = None):
    """Load a template PDB as a sanitized heavy-atom Mol WITH coords.

    If `chain_id` is given (CPBind/CPSea complex PDBs), first extract that
    chain (peptide chain L) via `core.pdb_utils.extract_chain` into a
    standalone PDB, then load it. If None (Scaffold standalone peptide PDBs),
    load the whole file.

    PDB bond orders are incomplete; we sanitize via the template SMILES mol
    (same molecule) — `AssignBondOrdersFromTemplate` reconciles them. Returns
    (mol_with_coords, error).

    The PDB and the declared SMILES must describe the same heavy-atom graph:
    an unrelated PDB (different atom count or connectivity) is rejected even
    when bond-order assignment happens to fail.
    """
    if chain_id is not None:
        from cycpep_master.core.pdb_utils import extract_chain
        extracted = extract_chain(pdb_path, chain_id)
        if extracted is None:
            return None, f"chain {chain_id} not found in {pdb_path}"
        pdb_path = extracted
    pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    if pdb_mol is None:
        return None, "pdb parse fail"
    smi_mol = Chem.MolFromSmiles(smiles)
    if smi_mol is None:
        return None, "smiles parse fail"
    if pdb_mol.GetNumHeavyAtoms() != smi_mol.GetNumHeavyAtoms():
        return (
            None,
            f"template PDB heavy-atom count {pdb_mol.GetNumHeavyAtoms()} "
            f"does not match declared SMILES count {smi_mol.GetNumHeavyAtoms()}",
        )
    if not _same_heavy_atom_graph(pdb_mol, smi_mol):
        return (
            None,
            "template PDB heavy-atom graph does not match the declared SMILES graph",
        )
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    except Exception:
        # Fallback: sanitize without property check (some PDBs have odd
        # valences), but only when the heavy-atom graph provably corresponds
        # to the declared SMILES; never substitute an unrelated PDB graph.
        try:
            Chem.SanitizeMol(
                pdb_mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
            mol = pdb_mol
        except Exception as e:
            return None, f"assign bond orders fail: {e}"
        if not _same_heavy_atom_graph(pdb_mol, smi_mol):
            return (
                None,
                "template PDB heavy-atom graph does not match the declared SMILES graph",
            )
    # Ensure RingInfo is initialized (the fallback path may skip it; Morgan FP
    # requires it). Safe no-op if already done.
    try:
        Chem.FastFindRings(mol)
    except Exception:
        pass
    if mol.GetNumConformers() == 0:
        return None, "no coords"
    return mol, None


def _skeleton_mol(mol) -> Chem.Mol:
    """Return a connectivity-only copy (elements, single bonds, no stereo)."""
    skeleton = Chem.RWMol(Chem.RemoveHs(mol))
    for atom in skeleton.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
    for bond in skeleton.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return skeleton.GetMol()


def _same_heavy_atom_graph(first: Chem.Mol, second: Chem.Mol) -> bool:
    """True when both mols have the same heavy-atom element/connectivity graph."""
    if first.GetNumHeavyAtoms() != second.GetNumHeavyAtoms():
        return False
    return Chem.MolToSmiles(
        _skeleton_mol(first), canonical=True, isomericSmiles=False
    ) == Chem.MolToSmiles(
        _skeleton_mol(second), canonical=True, isomericSmiles=False
    )


def _fp_tanimoto(fp_a, fp_b) -> float:
    import numpy as np
    a = np.asarray(fp_a, dtype=np.float32)
    b = np.asarray(fp_b, dtype=np.float32)
    inter = float((a * b).sum())
    union = float(a.sum() + b.sum() - inter)
    return inter / union if union > 0 else 0.0


def _greedy_cluster_fp(fps, threshold=FP_THRESHOLD, max_centroids=MAX_CENTROIDS_PER_BUCKET):
    """Greedy Morgan-FP-Tanimoto clustering. Returns list of centroid indices.
    FP similarity (not 3D RMSD) because RDKit GetBestRMS needs a substructure
    match, which fails for non-identical cyclic peptides. FP is a sound shape
    proxy for cyclic peptides sharing the same backbone.
    """
    centroids = []
    for i, fp_i in enumerate(fps):
        if fp_i is None:
            continue
        assigned = any(_fp_tanimoto(fp_i, fps[ci]) >= threshold for ci in centroids)
        if not assigned:
            centroids.append(i)
            if len(centroids) >= max_centroids:
                break
    return centroids


SOURCE_PRIORITY = {"cpsea": 0, "cpbind": 1, "scaffold": 2}


def _cyc_mode_from_cpbind(cyclization_type: str) -> str:
    """Map CPBind's cyclization_type vocabulary to template-library buckets."""
    t = (cyclization_type or "").lower()
    if "head-to-tail" in t and "disulfide" in t:
        return "mixed"
    if "head-to-tail" in t or "headtail" in t:
        return "N-C"
    if "disulfide" in t:
        return "disulfide"
    if "isopeptide" in t or "side" in t:
        return "SC"
    if "linear" in t:
        return "none"
    return "other"


def _normalize_scaffold(scaffold_csv: str, scaffold_pdb_dir: str, limit: int = 0):
    entries = []
    with open(scaffold_csv, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            map_str = row.get("map", "")
            if not map_str:
                continue
            fn = row["filename"]
            pdb_path = None
            for d in ("paper_set", "14-16_paper_set"):
                p = os.path.join(scaffold_pdb_dir, d, fn)
                if os.path.exists(p):
                    pdb_path = p
                    break
            if pdb_path is None:
                continue
            entries.append({
                "filename": fn, "smiles": row.get("smiles", ""), "map": map_str,
                "pdb_path": pdb_path, "source": "scaffold", "chain_id": None,
                "cyc_mode": _cyc_mode(map_str), "n_res": _residue_count(map_str),
            })
    return entries


def _normalize_cpbind(cpbind_root: str, cpbind_csv: Optional[str] = None,
                      limit: int = 0):
    """Normalize CPBind cluster centers into template entries (full-scale).

    Reads ALL cluster centers from CPBind_Cluster.tsv (~136666) — not a
    precomputed-sampled CSV. For each center:
      - if `cpbind_csv` (intermediate_cpbind.csv) has a precomputed map/smiles
        for that filename, reuse it (fast);
      - else derive map via extract_chain(L) -> build_helm_from_pdb ->
        helm_to_map -> get_smi_from_map (slower, ~0.02s each).
    cyclization_type is unavailable for derived entries, so cyc_mode comes
    from the derived MAP string via _cyc_mode.
    """
    from cycpep_master.core.pdb_utils import extract_chain
    from cycpep_master.paths.path_b import build_helm_from_pdb
    from cycpep_master.paths._map_utils import helm_to_map, get_smi_from_map

    cluster_tsv = os.path.join(cpbind_root, "CPBind_properties", "CPBind_Cluster.tsv")
    pdb_dir = os.path.join(cpbind_root, "CPBind_pdb")

    # optional precomputed map cache from intermediate CSV (filename -> {map, smiles, cyc})
    precomputed = {}
    if cpbind_csv and os.path.exists(cpbind_csv):
        with open(cpbind_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename", "")
                if fn:
                    precomputed[fn] = {
                        "map": row.get("map", ""), "smiles": row.get("smiles", ""),
                        "cyc": row.get("cyclization_type", ""),
                    }

    centers = []
    with open(cluster_tsv, encoding="utf-8") as f:
        next(f)
        seen = set()
        for line in f:
            c = line.split("\t")[0].strip()
            if c and c not in seen:
                seen.add(c); centers.append(c)
            if limit and len(centers) >= limit:
                break

    entries = []
    derived = 0
    for cid in centers:
        fn = f"{cid}.pdb"
        pdb_path = os.path.join(pdb_dir, fn)
        if not os.path.exists(pdb_path):
            continue
        pre = precomputed.get(fn)
        if pre and pre["map"] and pre["smiles"]:
            map_str, smi, cyc_type = pre["map"], pre["smiles"], pre["cyc"]
            cyc = _cyc_mode_from_cpbind(cyc_type) or _cyc_mode(map_str)
        else:
            # derive from chain L PDB
            try:
                chain_pdb = extract_chain(pdb_path, "L")
                if chain_pdb is None:
                    continue
                helm = build_helm_from_pdb(chain_pdb, "L")
                map_str = helm_to_map(helm)
                smi = get_smi_from_map(map_str)
                if not smi:
                    continue
                cyc = _cyc_mode(map_str)
                derived += 1
            except Exception:
                continue
        entries.append({
            "filename": fn, "smiles": smi, "map": map_str,
            "pdb_path": pdb_path, "source": "cpbind", "chain_id": "L",
            "cyc_mode": cyc, "n_res": _residue_count(map_str),
        })
    print(f"  [cpbind] {len(entries)} entries ({derived} derived, "
          f"{len(entries)-derived} from precomputed CSV)", flush=True)
    return entries


def _normalize_cpsea(cpsea_root: str, cpsea_limit: int = 0):
    """Normalize CPSea cluster centers into template entries.

    CPSea has no precomputed map/smiles CSV; derive from chain L PDB via
    build_helm_from_pdb -> helm_to_map -> get_smi_from_map. Uses Cluster.tsv
    centers only (~5902) to avoid redundant templates.
    """
    from cycpep_master.core.pdb_utils import extract_chain
    from cycpep_master.paths.path_b import build_helm_from_pdb
    from cycpep_master.paths._map_utils import helm_to_map, get_smi_from_map

    cluster_tsv = os.path.join(cpsea_root, "CPSea_PDB_properties", "CPSea_PDB_Cluster.tsv")
    pdb_dir = os.path.join(cpsea_root, "CPSea_PDB_pdb")
    centers = []
    with open(cluster_tsv, encoding="utf-8") as f:
        next(f)
        seen = set()
        for line in f:
            c = line.split("\t")[0].strip()
            if c and c not in seen:
                seen.add(c); centers.append(c)
            if cpsea_limit and len(centers) >= cpsea_limit:
                break
    entries = []
    for cid in centers:
        pdb_path = os.path.join(pdb_dir, f"{cid}.pdb")
        if not os.path.exists(pdb_path):
            continue
        try:
            chain_pdb = extract_chain(pdb_path, "L")
            if chain_pdb is None:
                continue
            helm = build_helm_from_pdb(chain_pdb, "L")
            map_str = helm_to_map(helm)
            smi = get_smi_from_map(map_str)
            if not smi:
                continue
            entries.append({
                "filename": f"{cid}.pdb", "smiles": smi, "map": map_str,
                "pdb_path": pdb_path, "source": "cpsea", "chain_id": "L",
                "cyc_mode": _cyc_mode(map_str), "n_res": _residue_count(map_str),
            })
        except Exception:
            continue
    return entries


def build(scaffold_csv: str, scaffold_pdb_dir: str, out_dir: str, limit: int = 0,
          cpbind_root: Optional[str] = None, cpbind_csv: Optional[str] = None,
          cpsea_root: Optional[str] = None, cpsea_limit: int = 0):
    # Normalize all sources into a single entry stream
    all_entries = []
    all_entries.extend(_normalize_scaffold(scaffold_csv, scaffold_pdb_dir, limit))
    if cpbind_root:
        all_entries.extend(_normalize_cpbind(cpbind_root, cpbind_csv=cpbind_csv, limit=limit))
    if cpsea_root:
        all_entries.extend(_normalize_cpsea(cpsea_root, cpsea_limit))
    print(f"normalized entries: {len(all_entries)} "
          f"(scaffold/cpbind/cpsea may include parse failures later)", flush=True)

    # Bucket by (residue_count, cyc_mode)
    buckets = defaultdict(list)
    for e in all_entries:
        buckets[(e["n_res"], e["cyc_mode"])].append(e)

    os.makedirs(out_dir, exist_ok=True)
    index = {}
    total_centroids = 0
    for (nres, cyc), entries in sorted(buckets.items()):
        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        mols, fps = [], []
        for e in entries:
            mol, _ = _load_template_mol(e["pdb_path"], e["smiles"], e["chain_id"])
            mols.append(mol)
            fps.append(fpgen.GetFingerprintAsNumPy(mol) if mol is not None else None)
        valid = [i for i, m in enumerate(mols) if m is not None]
        if not valid:
            continue
        fps_valid = [fps[i] for i in valid]
        centroid_idxs = _greedy_cluster_fp(fps_valid)
        bucket_dir = os.path.join(out_dir, f"{nres}_{cyc}")
        os.makedirs(bucket_dir, exist_ok=True)
        for k, vi in enumerate(centroid_idxs, 1):
            # choose the best source among the cluster-like neighborhood around vi:
            # cpsea > cpbind > scaffold, then vi itself. This preserves FP clustering
            # while preferring binding-state templates as centroids.
            i0 = valid[vi]
            fp0 = fps[i0]
            neighborhood = [i for i in valid
                            if fps[i] is not None and _fp_tanimoto(fp0, fps[i]) >= FP_THRESHOLD]
            i = min(neighborhood or [i0], key=lambda idx: SOURCE_PRIORITY.get(entries[idx]["source"], 99))
            e = entries[i]
            out_pdb = os.path.join(bucket_dir, f"centroid_{k:03d}.pdb")
            Chem.MolToPDBFile(mols[i], out_pdb)
            key = f"{nres}_{cyc}_{k:03d}"
            index[key] = {
                "pdb_path": os.path.relpath(out_pdb, out_dir),
                "smiles": e["smiles"],
                "map": e["map"],
                "n_res": nres,
                "cyc_mode": cyc,
                "source": e["source"],
                "source_filename": e["filename"],
            }
            total_centroids += 1
        src_counts = defaultdict(int)
        for e in entries:
            src_counts[e["source"]] += 1
        print(f"  bucket {nres}_{cyc}: {len(entries)} entries {dict(src_counts)} "
              f"-> {len(centroid_idxs)} centroids", flush=True)

    with open(os.path.join(out_dir, "templates_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nBuilt {total_centroids} centroids across {len(index)} buckets -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaffold-csv", required=True)
    ap.add_argument("--scaffold-pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap CSV rows per source for testing (0=all)")
    ap.add_argument("--cpbind-root",
                    help="CPBind root; required unless --no-cpbind")
    ap.add_argument("--cpbind-csv",
                    help="optional precomputed map/smiles cache for a CPBind subset")
    ap.add_argument("--cpsea-root",
                    help="CPSea_PDB root; required unless --no-cpsea")
    ap.add_argument("--cpsea-limit", type=int, default=0, help="cap CPSea cluster centers (0=all ~5902)")
    ap.add_argument("--no-cpbind", action="store_true")
    ap.add_argument("--no-cpsea", action="store_true")
    args = ap.parse_args()
    if not args.no_cpbind and not args.cpbind_root:
        ap.error("--cpbind-root is required unless --no-cpbind")
    if not args.no_cpsea and not args.cpsea_root:
        ap.error("--cpsea-root is required unless --no-cpsea")
    build(args.scaffold_csv, args.scaffold_pdb_dir, args.out_dir, args.limit,
          cpbind_root=None if args.no_cpbind else args.cpbind_root,
          cpbind_csv=None if args.no_cpbind else args.cpbind_csv,
          cpsea_root=None if args.no_cpsea else args.cpsea_root,
          cpsea_limit=args.cpsea_limit)


if __name__ == "__main__":
    main()
