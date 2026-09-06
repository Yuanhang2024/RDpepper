"""Scaffold template library lookup + residue-level coordinate borrowing.

Provides the runtime half of the Scaffold-template coordinate-guidance
feature (the build half is `build_template_library.py`). Given a generated
cyclic peptide, `find_template` locates the closest packaged template by
(residue_count, cyclization_mode) bucket + Morgan-FP similarity, and
`borrow_residue_coords` transfers the template's 3D backbone coordinates onto
the generated peptide — residue by residue, since RDKit's molecule-level
`ConstrainedEmbed` cannot align non-identical cyclic peptides (verified: a
generated SC-cyclic peptide is not a substructure of an N-C template, so
GetBestRMS/ConstrainedEmbed fail). Sidechains and residues absent from the
template (NNAAs, SC cyclization bonds) are filled by ETKDG with the borrowed
backbone atoms fixed via coordMap.
"""
import json
import hashlib
import os
import random
import re
import time
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from ..core.monomer_resolution import needs_monomer_resolution_scope

from .template_view import (
    DEFAULT_TEMPLATE_SOURCES,
    TemplateLibraryView,
    load_template_library_view,
)

_INDEX = None  # compatibility cache for the default source-scoped view
_DEFAULT_VIEW = None
_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "templates",
)
SOURCE_BONUS = {"cpsea": 0.05, "cpbind": 0.03, "scaffold": 0.0, "synthetic": -0.02}
TEMPLATE_STRATEGIES = frozenset({
    "full", "off", "random_same_length", "nearest_morgan",
})
ETKDG_TIMEOUT_SECONDS = 5
LARGE_MOLECULE_HEAVY_ATOMS = 90
LARGE_MOLECULE_TEMPLATE_ATTEMPTS = 3
RANDOM_RETRY_MAX_HEAVY_ATOMS = 50
FALLBACK_POOL_MULTIPLIER = 2
MIN_FALLBACK_POOL_SIZE = 6


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_view() -> Optional[TemplateLibraryView]:
    global _DEFAULT_VIEW
    if _DEFAULT_VIEW is not None:
        return _DEFAULT_VIEW
    idx_path = os.path.join(_TEMPLATES_DIR, "templates_index.json")
    if not os.path.exists(idx_path):
        return None
    _DEFAULT_VIEW = load_template_library_view(
        idx_path, allowed_sources=DEFAULT_TEMPLATE_SOURCES
    )
    return _DEFAULT_VIEW


def _load_index() -> dict:
    """Compatibility accessor for the default CPBind/Scaffold-only view."""
    global _INDEX
    if _INDEX is None:
        view = _default_view()
        _INDEX = dict(view.entries) if view is not None else {}
    return _INDEX


# SMARTS for a residue backbone: N - Cα - C(=O) - O
# Matches each peptide backbone unit. A 14-residue cyclic peptide yields 14.
_BACKBONE_SMARTS = Chem.MolFromSmarts(
    "[N;X3,X4][C;X4][C;X3](=[O;X1])"
)


def _cyc_mode(map_str: str) -> str:
    cycs = re.findall(r"\{cyc:([^}]+)\}", map_str)
    if not cycs:
        return "none"
    has_head_to_tail = any(tag == "N-C" for tag in cycs)
    sidechain_tags = [tag for tag in cycs if tag != "N-C"]
    if has_head_to_tail and sidechain_tags:
        return "mixed"
    if has_head_to_tail:
        return "N-C"
    if sidechain_tags and _map_has_disulfide(map_str):
        return "disulfide"
    if all("R3" in tag or re.fullmatch(r"[0-9]+-[0-9]+", tag)
           for tag in sidechain_tags):
        return "SC"
    return "other"


def _map_has_disulfide(map_str: str) -> bool:
    """Classify explicit side-chain closures by their assembled chemistry."""
    cleaned = re.sub(r"\{(?!nnr:)[^}]+\}", "", map_str)
    residue_tokens = re.findall(r"\{nnr:[^}]+\}|[A-Z]", cleaned)
    position_offset = 1 if re.search(r"\{nt:[^}]+\}", map_str) else 0
    resolved_explicit_edge = False
    for left_text, right_text in re.findall(
        r"\{cyc:(\d+):R3-(\d+):R3\}", map_str
    ):
        left = int(left_text) - position_offset - 1
        right = int(right_text) - position_offset - 1
        if not (0 <= left < len(residue_tokens) and 0 <= right < len(residue_tokens)):
            continue
        resolved_explicit_edge = True
        if residue_tokens[left] == "C" and residue_tokens[right] == "C":
            return True
    if resolved_explicit_edge:
        return False

    try:
        smiles = _smiles_from_map(map_str)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    except Exception:
        return False
    if mol is None:
        return False
    return any(
        bond.GetBeginAtom().GetAtomicNum() == 16
        and bond.GetEndAtom().GetAtomicNum() == 16
        for bond in mol.GetBonds()
    )


def _residue_count(map_str: str) -> int:
    # Count only residue symbols: bare standard-AA letters + {nnr:...} tokens.
    # Remove topology/cap annotations first; otherwise {cyc:N-C} contributes
    # spurious N/C residues (+2), corrupting bucket keys.
    cleaned = re.sub(r"\{(?!nnr:)[^}]+\}", "", map_str)
    return len(re.findall(r"[A-Z]|\{nnr:[^}]+\}", cleaned))


def _fp(mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol)


def _fp_tanimoto(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    inter = float((a * b).sum())
    union = float(a.sum() + b.sum() - inter)
    return inter / union if union > 0 else 0.0


def _view_or_default(
    library_view: Optional[TemplateLibraryView],
) -> Optional[TemplateLibraryView]:
    return library_view if library_view is not None else _default_view()


def _bucket_entries(
    generated_map: str,
    view: TemplateLibraryView,
) -> tuple[list[tuple[str, dict]], str]:
    n_res = _residue_count(generated_map)
    cyc = _cyc_mode(generated_map)
    prefixes = [f"{n_res}_{cyc}_"]
    # Only an explicit SC topology may borrow the N-C bucket of the same
    # residue count (its backbone is compatible; the SC bond is rebuilt
    # locally). linear/none, mixed, and other must never borrow N-C.
    if cyc == "SC":
        prefixes.append(f"{n_res}_N-C_")
    for prefix in prefixes:
        entries = [
            (key, dict(entry))
            for key, entry in view.entries.items()
            if key.startswith(prefix)
        ]
        if entries:
            return entries, prefix.rstrip("_")
    return [], ""


def _rank_entries(
    query_mol: Chem.Mol,
    entries: list[tuple[str, dict]],
) -> tuple[list[tuple[float, float, str, dict]], int]:
    query_fp = _fp(query_mol)
    ranked = []
    invalid_count = 0
    for key, entry in entries:
        template_mol = Chem.MolFromSmiles(str(entry.get("smiles", "")))
        if template_mol is None:
            invalid_count += 1
            continue
        similarity = _fp_tanimoto(query_fp, _fp(template_mol))
        score = similarity + SOURCE_BONUS.get(str(entry["source"]).lower(), 0.0)
        ranked.append((score, similarity, key, entry))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return ranked, invalid_count


def _same_length_entries(
    generated_map: str,
    view: TemplateLibraryView,
) -> tuple[list[tuple[str, dict]], str, int]:
    n_res = _residue_count(generated_map)
    entries = []
    invalid_count = 0
    for key, entry in view.entries.items():
        try:
            entry_n_res = int(entry.get("n_res", -1))
        except (TypeError, ValueError):
            # Malformed index entries are skipped and surfaced through the
            # invalid metadata instead of aborting the whole lookup.
            invalid_count += 1
            continue
        if entry_n_res == n_res:
            entries.append((key, dict(entry)))
    entries.sort(key=lambda item: item[0])
    return entries, f"{n_res}_ANY_TOPOLOGY", invalid_count


def _strategy_candidates(
    generated_map: str,
    query_mol: Chem.Mol,
    view: TemplateLibraryView,
    strategy: str,
    random_seed: int,
) -> tuple[list[tuple[float, float, str, dict]], int, str, str]:
    """Return ordered candidates and an explicit selection-scope ledger."""
    if strategy not in TEMPLATE_STRATEGIES:
        raise ValueError(
            f"template_strategy must be one of {sorted(TEMPLATE_STRATEGIES)}"
        )
    if strategy == "off":
        return [], 0, "DISABLED", "none"
    if strategy == "full":
        entries, bucket = _bucket_entries(generated_map, view)
        ranked, invalid_count = _rank_entries(query_mol, entries)
        return ranked, invalid_count, bucket, "residue_count_and_topology"

    entries, bucket, length_invalid_count = _same_length_entries(generated_map, view)
    query_fp = _fp(query_mol)
    candidates = []
    invalid_count = length_invalid_count
    for key, entry in entries:
        template_mol = Chem.MolFromSmiles(str(entry.get("smiles", "")))
        if template_mol is None:
            invalid_count += 1
            continue
        similarity = _fp_tanimoto(query_fp, _fp(template_mol))
        candidates.append((similarity, similarity, key, entry))
    if strategy == "nearest_morgan":
        candidates.sort(key=lambda item: (-item[1], item[2]))
        return candidates, invalid_count, bucket, "same_length_pure_morgan"

    candidates.sort(key=lambda item: item[2])
    random.Random(int(random_seed)).shuffle(candidates)
    return candidates, invalid_count, bucket, "same_length_seeded_random"


def find_template(
    generated_map: str,
    *,
    library_view: Optional[TemplateLibraryView] = None,
    meta_out: Optional[dict] = None,
    template_strategy: str = "full",
    random_seed: int = 42,
    monomer_context=None,
) -> Optional[dict]:
    """Find the closest packaged template for a generated peptide's MAP.

    Returns the index entry (with pdb_path, smiles, map) or None if no bucket
    matches. Bucket key = (residue_count, cyc_mode). For SC-cyclic generated
    peptides, if no SC bucket exists, falls back to the N-C bucket of the same
    residue count (borrow its backbone, rebuild the SC bond locally).
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                generated_map, kind="map"
            ),
        ):
            return find_template(
                generated_map,
                library_view=library_view,
                meta_out=meta_out,
                template_strategy=template_strategy,
                random_seed=random_seed,
            )
    if meta_out is not None:
        meta_out.clear()
    if template_strategy not in TEMPLATE_STRATEGIES:
        raise ValueError(
            f"template_strategy must be one of {sorted(TEMPLATE_STRATEGIES)}"
        )
    if template_strategy == "off":
        if meta_out is not None:
            meta_out.update({
                "status": "template_disabled",
                "template_strategy": template_strategy,
                "random_seed": int(random_seed),
                "template_count": 0,
            })
        return None
    view = _view_or_default(library_view)
    if view is None or not view.entries:
        if meta_out is not None:
            meta_out.update({"status": "no_library", "template_count": 0})
        return None
    query_smiles = _smiles_from_map(generated_map)
    query_mol = Chem.MolFromSmiles(query_smiles) if query_smiles else None
    if query_mol is None:
        if meta_out is not None:
            meta_out.update({"status": "not_assessable", "reason": "MAP_TO_SMILES_FAILED"})
        return None
    ranked, invalid_count, bucket, selection_scope = _strategy_candidates(
        generated_map, query_mol, view, template_strategy, random_seed
    )
    if not ranked:
        if meta_out is not None:
            meta_out.update({
                "status": "no_compatible_template",
                "bucket": bucket or None,
                "template_strategy": template_strategy,
                "selection_scope": selection_scope,
                "random_seed": int(random_seed),
                "invalid_template_count": invalid_count,
            })
        return None
    score, similarity, key, entry = ranked[0]
    if meta_out is not None:
        meta_out.update({
            "status": "template_success",
            "bucket": bucket,
            "template_key": key,
            "similarity": similarity,
            "score": score,
            "invalid_template_count": invalid_count,
            "template_strategy": template_strategy,
            "selection_scope": selection_scope,
            "random_seed": int(random_seed),
            "library_formal": view.formal,
            "library_index_sha256": view.index_sha256,
            "allowed_sources": sorted(view.allowed_sources),
        })
    return entry


def _template_ca_coordinates(path):
    coordinates = []
    with open(path, encoding="ascii", errors="replace") as handle:
        model_seen = False
        for line in handle:
            if line.startswith("MODEL"):
                if model_seen:
                    break
                model_seen = True
                continue
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip().upper() != "CA":
                continue
            try:
                coordinates.append((
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ))
            except (IndexError, ValueError):
                return []
    return coordinates


def find_coordinate_evidence(
    generated_map: str,
    generated_smiles: str,
    *,
    max_matches: int = 2,
    template_strategy: str = "full",
    random_seed: int = 42,
    library_view: Optional[TemplateLibraryView] = None,
    monomer_context=None,
):
    """Return immutable template coordinate evidence without embedding.

    This is the V5 runtime boundary: template lookup may expose compatible
    C-alpha coordinates and residue mapping, but never returns a generated
    conformer or writes a structure.
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                generated_map, kind="map"
            ),
        ):
            return find_coordinate_evidence(
                generated_map,
                generated_smiles,
                max_matches=max_matches,
                template_strategy=template_strategy,
                random_seed=random_seed,
                library_view=library_view,
            )
    if type(max_matches) is not int or max_matches < 1:
        raise ValueError("max_matches must be a positive integer")
    if template_strategy not in TEMPLATE_STRATEGIES:
        raise ValueError(
            f"template_strategy must be one of {sorted(TEMPLATE_STRATEGIES)}"
        )
    query = Chem.MolFromSmiles(str(generated_smiles))
    if query is None:
        return {
            "status": "invalid_query",
            "matches": [],
            "reason": "generated SMILES is not parseable",
        }
    view = _view_or_default(library_view)
    if template_strategy == "off" or view is None:
        return {
            "status": (
                "disabled" if template_strategy == "off" else "no_library"
            ),
            "matches": [],
            "reason": None,
        }
    ranked, invalid_count, bucket, selection_scope = _strategy_candidates(
        generated_map,
        query,
        view,
        template_strategy,
        int(random_seed),
    )
    expected_residues = _residue_count(generated_map)
    matches = []
    for score, similarity, key, entry in ranked:
        path = view.template_path(entry)
        if not path.is_file():
            continue
        coordinates = _template_ca_coordinates(path)
        if len(coordinates) != expected_residues:
            continue
        matches.append({
            "template_id": key,
            "template_source": str(entry.get("source", "")),
            "similarity": float(similarity),
            "ranking_score": float(score),
            "template_pdb_path": str(path.resolve()),
            "template_pdb_sha256": _sha256_file(path),
            "template_map": str(entry.get("map", "")),
            "template_full_inchikey": (
                Chem.MolToInchiKey(
                    Chem.MolFromSmiles(str(entry.get("smiles", "")))
                )
                if Chem.MolFromSmiles(str(entry.get("smiles", "")))
                is not None
                else None
            ),
            "residue_mapping": [
                {
                    "query_position": index,
                    "template_position": index,
                }
                for index in range(1, expected_residues + 1)
            ],
            "ca_coordinate_map": [
                {
                    "query_position": index,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                }
                for index, xyz in enumerate(coordinates, 1)
            ],
            "compatible_atom_count": expected_residues,
        })
        if len(matches) >= max_matches:
            break
    return {
        "status": "matched" if matches else "no_compatible_template",
        "matches": matches,
        "bucket": bucket or None,
        "selection_scope": selection_scope,
        "candidate_count": len(ranked),
        "invalid_template_count": invalid_count,
        "library_index_sha256": view.index_sha256,
        "allowed_sources": sorted(view.allowed_sources),
        "template_strategy": template_strategy,
    }


def _smiles_from_map(map_str: str) -> Optional[str]:
    """Best-effort: turn a MAP string into a SMILES via cycpep_master. Returns
    None if assembly fails (caller treats as no-FP path)."""
    try:
        from cycpep_master.paths._map_utils import get_smi_from_map
        smi = get_smi_from_map(map_str)
        return smi if smi and not smi.startswith("ERROR") else None
    except Exception:
        return None


def _backbone_atoms(mol):
    """Return list of (N_idx, Ca_idx, C_idx, O_idx) for each residue backbone.
    Order follows RDKit's substructure match order, which for a linear scan of
    N-Cα-C(=O) matches residues in sequence. (N-C cyclic peptides ring-close,
    so the last residue's C bonds to the first residue's N — the SMARTS still
    matches each backbone unit once.)
    """
    return list(mol.GetSubstructMatches(_BACKBONE_SMARTS))


def borrow_residue_coords(
    generated_smiles: str,
    generated_map: str,
    template_pdb_path: str,
    template_smiles: Optional[str] = None,
    random_seed: int = 42,
    meta_out: Optional[dict] = None,
) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Transfer a template's backbone 3D coords onto a generated peptide.

    1. Load template PDB → sanitized heavy-atom Mol WITH coords (via its own
       SMILES mol reconciling bond orders).
    2. Identify backbone atoms (N-Cα-C=O per residue) in both the generated
       SMILES mol and the template mol via the shared SMARTS.
    3. Match residues by sequence index, copy template backbone coords onto
       the generated mol's backbone atoms.
    4. ETKDG-embed the generated mol with those backbone atoms FIXED
       (coordMap) — fills sidechains / NNAAs / SC-cyclization bonds.

    Returns (mol_with_3d, error). Mol has heavy atoms only (AddHs caller's job).
    """
    if meta_out is not None:
        meta_out.clear()
        meta_out.update({
            "status": "started",
            "random_seed": int(random_seed),
            "template_pdb_path": str(template_pdb_path),
        })
    gen_mol = Chem.MolFromSmiles(generated_smiles)
    if gen_mol is None:
        if meta_out is not None:
            meta_out.update({"status": "failed", "reason": "INVALID_GENERATED_SMILES"})
        return None, "invalid generated SMILES"

    # Load template mol with coords (reuse the build-side loader — single
    # source for PDB→sanitized-mol-with-coords, incl. bond-order fallback).
    from cycpep_master.docking.build_template_library import _load_template_mol
    tpl_mol, load_err = _load_template_mol(template_pdb_path, template_smiles)
    if tpl_mol is None:
        if meta_out is not None:
            meta_out.update({
                "status": "failed", "reason": "TEMPLATE_LOAD_FAILED",
                "detail": str(load_err),
            })
        return None, f"template load fail: {load_err}"

    gen_bb = _backbone_atoms(gen_mol)
    tpl_bb = _backbone_atoms(tpl_mol)
    if not gen_bb or not tpl_bb:
        if meta_out is not None:
            meta_out.update({"status": "failed", "reason": "BACKBONE_NOT_IDENTIFIED"})
        return None, "backbone SMARTS matched no residues"
    expected_residue_count = _residue_count(generated_map)
    if (
        len(gen_bb) != expected_residue_count
        or len(tpl_bb) != expected_residue_count
    ):
        if meta_out is not None:
            meta_out.update({
                "status": "failed",
                "reason": "BACKBONE_MAPPING_INCOMPLETE",
                "expected_residue_count": expected_residue_count,
                "generated_backbone_count": len(gen_bb),
                "template_backbone_count": len(tpl_bb),
            })
        return None, (
            "complete residue-level backbone mapping required: "
            f"expected={expected_residue_count}, generated={len(gen_bb)}, "
            f"template={len(tpl_bb)}"
        )

    # Match residues by index; copy template coords to generated backbone atoms.
    from rdkit.Geometry import Point3D
    tpl_conf = tpl_mol.GetConformer()
    n_match = expected_residue_count
    # Fix ONLY the Cα atoms (one per matched residue). Fixing the full backbone
    # (N+Cα+C+O = 4×residues) over-constrains ETKDG and embed returns -1 on
    # macrocycles; Cα alone is enough to pin the ring shape while leaving
    # backbone torsions + sidechains free for ETKDG to fill.
    coord_map = {}  # gen_atom_idx -> Point3D
    for i in range(n_match):
        gCa, tCa = gen_bb[i][1], tpl_bb[i][1]
        pos = tpl_conf.GetAtomPosition(tCa)
        coord_map[gCa] = Point3D(pos.x, pos.y, pos.z)

    gen_h = Chem.AddHs(gen_mol)
    # Bounded-iteration constrained embed. The default EmbedMolecule retries
    # unboundedly when the borrowed-Cα distance geometry is unsatisfiable (a
    # mismatched template pins Cα coords the generated peptide's chemistry can't
    # meet), spinning in C++ for tens of seconds before giving up. Diagnostic on
    # 12 such peptides: all spun to the 40s process cap here, yet ALL embed in
    # 3-13s via unconstrained _embed_3d. Capping maxIterations makes the
    # constrained attempt fail fast so parent-MOL2 generation can fall back to
    # _embed_3d instead of returning no coordinate artifact.
    # Single-shot constrained embed. Measured on the borrowed-Cα macrocycles
    # that used to time out: each ETKDG iteration costs ~2.5s (full distance
    # geometry over ~140 atoms) and, when the template is a poor chemical match,
    # every iteration returns status=-1 — more iterations only burn time on a
    # geometrically unsatisfiable pin (maxIterations=10 -> 26s, 30 -> 72s, all
    # still -1). So try exactly once: a good template embeds on iter 1, a bad one
    # fails in ~2.5s and the caller falls back to unconstrained _embed_3d
    # (validated: 12/12 prior timeouts embed in 3-13s that way).
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.useRandomCoords = True
        params.useMacrocycleTorsions = True
        params.maxIterations = 1
        params.timeout = ETKDG_TIMEOUT_SECONDS
        params.SetCoordMap(coord_map)
        status = AllChem.EmbedMolecule(gen_h, params)
    except Exception as e:
        if meta_out is not None:
            meta_out.update({
                "status": "failed", "reason": "CONSTRAINED_EMBED_EXCEPTION",
                "detail": f"{type(e).__name__}: {e}",
            })
        return None, f"embed with coordMap failed: {e}"
    if status != 0:
        if meta_out is not None:
            meta_out.update({
                "status": "failed", "reason": "CONSTRAINED_EMBED_FAILED",
                "embed_status": int(status),
            })
        return None, "ETKDG could not embed with fixed Cα"

    # Light MMFF cleanup only. maxIters was 500, but on a borrowed-Cα macrocycle
    # whose pinned geometry is strained, MMFF converges slowly and spins for tens
    # of seconds (py-spy: this line, not the embed, is the real hang) — the
    # dominant cost in the 40s conformer timeouts. The embed already gives a
    # usable pose; 50 iters relaxes clashes without the long tail. Peptides where
    # even this is too strained fall back to unconstrained _embed_3d upstream
    # (validated: 12/12 prior timeouts embed in 3-13s that way).
    optimization_status = "not_attempted"
    optimization_result = None
    try:
        optimization_result = int(AllChem.MMFFOptimizeMolecule(gen_h, maxIters=50))
        optimization_status = (
            "converged" if optimization_result == 0 else "iteration_limit"
        )
    except Exception as exc:
        optimization_status = f"exception:{type(exc).__name__}"

    # Verify Cα atoms stayed pinned (sanity)
    import numpy as np
    conf = gen_h.GetConformer()
    max_drift = 0.0
    for g_idx, tgt in coord_map.items():
        p = conf.GetAtomPosition(g_idx)
        d = float(np.linalg.norm([p.x - tgt.x, p.y - tgt.y, p.z - tgt.z]))
        max_drift = max(max_drift, d)

    if meta_out is not None:
        meta_out.update({
            "status": "success" if max_drift < 1.0 else "success_with_drift_warning",
            "matched_residue_count": n_match,
            "fixed_ca_atom_count": len(coord_map),
            "maximum_ca_drift_angstrom": round(max_drift, 6),
            "optimization_status": optimization_status,
            "optimization_result": optimization_result,
            "embed_status": int(status),
            "backbone_atom_map": [
                {
                    "residue_index": index,
                    "generated_atom_indices": list(gen_bb[index]),
                    "template_atom_indices": list(tpl_bb[index]),
                    "fixed_generated_ca_index": int(gen_bb[index][1]),
                    "source_template_ca_index": int(tpl_bb[index][1]),
                }
                for index in range(n_match)
            ],
        })
    return gen_h, (f"ok, {n_match} Cα borrowed, drift={max_drift:.2f}Å"
                   if max_drift < 1.0 else f"warn: Cα drift={max_drift:.2f}Å")


def _etkdg_conformer(gen_mol, random_seed=42):
    """Single ETKDGv3 conformer (multi-seed internal) for a heavy-atom Mol.
    Returns (heavy_mol_with_Hs_and_1_conf, error)."""
    from rdkit import Chem
    mol = Chem.AddHs(gen_mol)
    p = AllChem.ETKDGv3(); p.numThreads = 0; p.randomSeed = random_seed
    p.timeout = ETKDG_TIMEOUT_SECONDS
    st = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=p)
    if len(st) == 0:
        retry = AllChem.ETKDGv3()
        retry.numThreads = 0
        retry.randomSeed = random_seed
        retry.useRandomCoords = True
        retry.timeout = ETKDG_TIMEOUT_SECONDS
        st = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=retry)
    if len(st) == 0:
        return None, "ETKDG embed failed"
    AllChem.MMFFOptimizeMolecule(mol, confId=st[0], maxIters=100)
    return mol, None


def _rmsd_pair(mol, cid_a, cid_b):
    from rdkit.Chem import rdMolAlign
    try:
        return float(rdMolAlign.GetBestRMS(mol, mol, cid_a, cid_b))
    except Exception:
        return None


def _maxmin_select(mol, conf_ids, k):
    """Max-min diversity selection: greedily pick k confs maximizing the min
    RMSD to already-selected confs. Returns list of conf ids (subset)."""
    if len(conf_ids) <= k:
        return list(conf_ids)
    import collections
    selected = [conf_ids[0]]
    remaining = set(conf_ids[1:])
    while len(selected) < k and remaining:
        best, best_score = None, -1.0
        for cid in remaining:
            scores = [_rmsd_pair(mol, cid, s) for s in selected]
            scores = [s for s in scores if s is not None]
            if not scores:
                continue
            min_d = min(scores)
            if min_d > best_score:
                best_score, best = min_d, cid
        if best is None:
            break
        selected.append(best); remaining.discard(best)
    return selected


def generate_conformers(
    generated_smiles: str,
    generated_map: str,
    n_conformers: int = 5,
    scaffold_csv: Optional[str] = None,
    meta_out: Optional[dict] = None,
    *,
    library_view: Optional[TemplateLibraryView] = None,
    template_strategy: str = "full",
    random_seed: int = 42,
    monomer_context=None,
) -> Tuple[Optional[Chem.Mol], list]:
    """User-facing: produce `n_conformers` diverse 3D conformers for a peptide.

    Scene A (find_template hits): borrow coords from the top-n_conformers most
    similar templates — each template is 1 conformer (its Cα pinned via coordMap,
    ETKDG fills the rest). Different templates → naturally diverse shapes.

    Scene B (find_template misses — residue count / topology outside the
    packaged library): ETKDG embeds a candidate pool of 3×n_conformers, then
    max-min selects the n_conformers most dissimilar ones (greedy diversity).

    If `meta_out` (a dict) is given, it is populated with `scene` ("A" or "B")
    so callers can tell whether a template was actually matched (Scene A) or the
    peptide fell through to unguided ETKDG (Scene B) — used by the RFT reward to
    harvest de-novo conformers of peptides the template library doesn't cover.

    Returns (mol_with_confids, list_of_confids). mol None on total failure.
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                generated_map, kind="map"
            ),
        ):
            return generate_conformers(
                generated_smiles,
                generated_map,
                n_conformers=n_conformers,
                scaffold_csv=scaffold_csv,
                meta_out=meta_out,
                library_view=library_view,
                template_strategy=template_strategy,
                random_seed=random_seed,
            )
    from rdkit import Chem

    if meta_out is not None:
        meta_out.clear()
    if template_strategy not in TEMPLATE_STRATEGIES:
        raise ValueError(
            f"template_strategy must be one of {sorted(TEMPLATE_STRATEGIES)}"
        )
    if int(n_conformers) < 1:
        raise ValueError("n_conformers must be at least one")
    gen_mol = Chem.MolFromSmiles(generated_smiles)
    if gen_mol is None:
        if meta_out is not None:
            meta_out.update({"status": "total_failed", "reason": "INVALID_SMILES"})
        return None, [f"invalid SMILES"]

    heavy_atom_count = int(gen_mol.GetNumHeavyAtoms())
    large_molecule = heavy_atom_count >= LARGE_MOLECULE_HEAVY_ATOMS
    candidate_attempt_budget = (
        LARGE_MOLECULE_TEMPLATE_ATTEMPTS
        if large_molecule else int(n_conformers) * 3
    )
    resource_policy = {
        "policy": "bounded_large_molecule" if large_molecule else "standard",
        "heavy_atom_count": heavy_atom_count,
        "large_molecule_threshold": LARGE_MOLECULE_HEAVY_ATOMS,
        "etkdg_timeout_seconds": ETKDG_TIMEOUT_SECONDS,
        "candidate_attempt_budget": candidate_attempt_budget,
        "unguided_fallback_allowed": not large_molecule,
    }

    # Scene A: top-K templates
    view = library_view if template_strategy == "off" else _view_or_default(
        library_view
    )
    worked = []
    template_status = "no_library"
    template_audit = {
        "attempted_count": 0,
        "missing_pdb_count": 0,
        "load_or_embed_failed_count": 0,
        "invalid_template_count": 0,
        "selected_template_keys": [],
        "template_strategy": template_strategy,
        "random_seed": int(random_seed),
        "candidate_attempt_budget": candidate_attempt_budget,
        "resource_policy": resource_policy,
        "attempt_trace": [],
    }
    if template_strategy == "off":
        template_status = "disabled_by_control"
        template_audit.update({
            "bucket": "DISABLED",
            "selection_scope": "none",
            "candidate_count": 0,
        })
    elif view is not None and view.entries:
        ranked, invalid_count, bucket, selection_scope = _strategy_candidates(
            generated_map, gen_mol, view, template_strategy, random_seed
        )
        template_audit["bucket"] = bucket or None
        template_audit["selection_scope"] = selection_scope
        template_audit["candidate_count"] = len(ranked)
        template_audit["invalid_template_count"] = invalid_count
        if not ranked:
            template_status = "no_compatible_template"
        else:
            template_status = "template_candidates"
            # Try a bounded candidate set; some template PDBs may be
            # missing (compress rename edge cases) or fail to load; skip them.
            for rank_i, (score, sim, key, entry) in enumerate(
                ranked[:candidate_attempt_budget]
            ):
                attempt_started = time.perf_counter()
                template_audit["attempted_count"] += 1
                tpl_pdb = view.template_path(entry)
                trace = {
                    "attempt_index": rank_i,
                    "template_key": key,
                    "source": str(entry.get("source", "")),
                    "cyclization_mode": str(entry.get("cyc_mode", "")),
                    "similarity": round(float(sim), 8),
                    "ranking_score": round(float(score), 8),
                    "random_seed": int(random_seed) + 1000 + rank_i,
                    "pdb_path": str(entry.get("pdb_path", "")),
                }
                if not tpl_pdb.is_file():
                    template_audit["missing_pdb_count"] += 1
                    trace["status"] = "missing_pdb"
                    trace["template_pdb_sha256"] = None
                    trace["elapsed_sec"] = round(
                        time.perf_counter() - attempt_started, 6
                    )
                    template_audit["attempt_trace"].append(trace)
                    continue
                trace["template_pdb_sha256"] = _sha256_file(tpl_pdb)
                borrow_audit = {}
                mol, err = borrow_residue_coords(
                    generated_smiles, generated_map, str(tpl_pdb), entry["smiles"],
                    random_seed=int(random_seed) + 1000 + rank_i,
                    meta_out=borrow_audit,
                )
                trace["borrow_audit"] = borrow_audit
                trace["detail"] = err
                if mol is not None:
                    worked.append((mol, sim, err, key))
                    template_audit["selected_template_keys"].append(key)
                    trace["status"] = "success"
                    trace["elapsed_sec"] = round(
                        time.perf_counter() - attempt_started, 6
                    )
                    template_audit["attempt_trace"].append(trace)
                    if len(worked) >= n_conformers:
                        break
                else:
                    template_audit["load_or_embed_failed_count"] += 1
                    trace["status"] = "load_or_embed_failed"
                    trace["elapsed_sec"] = round(
                        time.perf_counter() - attempt_started, 6
                    )
                    template_audit["attempt_trace"].append(trace)
            if len(worked) >= n_conformers:
                # merge all conformers onto the first working mol
                base = Chem.AddHs(Chem.MolFromSmiles(generated_smiles))
                merged_cids = []
                for m, _sim, _err, key in worked:
                    cid = base.AddConformer(m.GetConformer(0), assignId=True)
                    merged_cids.append(cid)
                if meta_out is not None:
                    meta_out.update({
                        "status": "template_success",
                        "scene": "A",
                        "template_status": "template_success",
                        "template_audit": template_audit,
                        "library_formal": view.formal,
                        "library_index_sha256": view.index_sha256,
                        "allowed_sources": sorted(view.allowed_sources),
                        "template_strategy": template_strategy,
                        "random_seed": int(random_seed),
                        "requested_conformer_count": int(n_conformers),
                        "produced_conformer_count": len(merged_cids),
                        "guided_conformer_count": len(merged_cids),
                        "fallback_conformer_count": 0,
                        "conformer_provenance": [
                            {
                                "conformer_id": int(conformer_id),
                                "source": "matched_template",
                                "template_key": worked[index][3],
                            }
                            for index, conformer_id in enumerate(
                                merged_cids
                            )
                        ],
                    })
                return base, merged_cids
            if worked:
                template_status = "partial_template_success_requires_fallback"
            if ranked:
                template_status = template_status if worked else (
                    "template_load_failed"
                    if template_audit["missing_pdb_count"] == template_audit["attempted_count"]
                    else "constrained_embed_failed"
                )
    if large_molecule:
        fallback_audit = {
            "status": "resource_limited",
            "reason": "UNGUIDED_ETKDG_DISABLED_FOR_LARGE_MOLECULE",
            "requested_pool_size": max(
                int(n_conformers) * FALLBACK_POOL_MULTIPLIER,
                MIN_FALLBACK_POOL_SIZE,
            ),
            "initial_embed_count": 0,
            "final_embed_count": 0,
            "random_coordinate_retry_used": False,
            "optimization_trace": [],
            "selected_pool_conformer_ids": [],
            "resource_policy": resource_policy,
        }
        if worked:
            base = Chem.AddHs(Chem.MolFromSmiles(generated_smiles))
            partial_ids = [
                base.AddConformer(m.GetConformer(0), assignId=True)
                for m, _sim, _err, _key in worked
            ]
            if meta_out is not None:
                meta_out.update({
                    "status": "partial_template_success",
                    "scene": "A",
                    "template_status": template_status,
                    "template_audit": template_audit,
                    "reason": "RESOURCE_LIMIT_FALLBACK_SKIPPED",
                    "failure_class": "resource_limit",
                    "library_formal": bool(view and view.formal),
                    "library_index_sha256": view.index_sha256 if view else None,
                    "allowed_sources": sorted(view.allowed_sources) if view else [],
                    "template_strategy": template_strategy,
                    "random_seed": int(random_seed),
                    "requested_conformer_count": int(n_conformers),
                    "produced_conformer_count": len(partial_ids),
                    "guided_conformer_count": len(partial_ids),
                    "fallback_conformer_count": 0,
                    "conformer_provenance": [
                        {
                            "conformer_id": int(conformer_id),
                            "source": "matched_template",
                            "template_key": worked[index][3],
                        }
                        for index, conformer_id in enumerate(partial_ids)
                    ],
                    "resource_policy": resource_policy,
                    "fallback_audit": fallback_audit,
                })
            return base, partial_ids
        if meta_out is not None:
            meta_out.update({
                "status": "total_failed",
                "scene": "B",
                "template_status": template_status,
                "template_audit": template_audit,
                "reason": "RESOURCE_LIMIT_UNGUIDED_ETKDG_SKIPPED",
                "failure_class": "resource_limit",
                "template_strategy": template_strategy,
                "random_seed": int(random_seed),
                "requested_conformer_count": int(n_conformers),
                "produced_conformer_count": 0,
                "resource_policy": resource_policy,
                "fallback_audit": fallback_audit,
            })
        return None, [
            "resource_limit: unguided ETKDG disabled for molecule with "
            f"{heavy_atom_count} heavy atoms"
        ]

    # Scene B: ETKDG pool + max-min
    pool_n = max(
        int(n_conformers) * FALLBACK_POOL_MULTIPLIER,
        MIN_FALLBACK_POOL_SIZE,
    )
    fallback_started = time.perf_counter()
    fallback_audit = {
        "requested_pool_size": int(pool_n),
        "random_coordinate_retry_used": False,
        "random_coordinate_retry_allowed": (
            heavy_atom_count <= RANDOM_RETRY_MAX_HEAVY_ATOMS
        ),
        "optimization_trace": [],
        "selected_pool_conformer_ids": [],
    }
    mol_h = Chem.AddHs(gen_mol)
    p = AllChem.ETKDGv3(); p.numThreads = 0; p.randomSeed = int(random_seed)
    p.timeout = ETKDG_TIMEOUT_SECONDS
    st = AllChem.EmbedMultipleConfs(mol_h, numConfs=pool_n, params=p)
    fallback_audit["initial_embed_count"] = len(st)
    if (
        len(st) == 0
        and fallback_audit["random_coordinate_retry_allowed"]
    ):
        fallback_audit["random_coordinate_retry_used"] = True
        retry = AllChem.ETKDGv3()
        retry.numThreads = 0
        retry.randomSeed = int(random_seed)
        retry.useRandomCoords = True
        retry.timeout = ETKDG_TIMEOUT_SECONDS
        st = AllChem.EmbedMultipleConfs(mol_h, numConfs=pool_n, params=retry)
    fallback_audit["final_embed_count"] = len(st)
    if len(st) == 0:
        fallback_audit["status"] = "embed_failed"
        fallback_audit["elapsed_sec"] = round(
            time.perf_counter() - fallback_started, 6
        )
        if worked:
            base = Chem.AddHs(Chem.MolFromSmiles(generated_smiles))
            partial_ids = [
                base.AddConformer(m.GetConformer(0), assignId=True)
                for m, _sim, _err, _key in worked
            ]
            if meta_out is not None:
                meta_out.update({
                    "status": "partial_template_success",
                    "scene": "A",
                    "template_status": template_status,
                    "template_audit": template_audit,
                    "reason": "ETKDG_POOL_EMBED_FAILED_AFTER_PARTIAL_TEMPLATE_SUCCESS",
                    "library_formal": bool(view and view.formal),
                    "library_index_sha256": view.index_sha256 if view else None,
                    "allowed_sources": sorted(view.allowed_sources) if view else [],
                    "template_strategy": template_strategy,
                    "random_seed": int(random_seed),
                    "requested_conformer_count": int(n_conformers),
                    "produced_conformer_count": len(partial_ids),
                    "guided_conformer_count": len(partial_ids),
                    "fallback_conformer_count": 0,
                    "conformer_provenance": [
                        {
                            "conformer_id": int(conformer_id),
                            "source": "matched_template",
                            "template_key": worked[index][3],
                        }
                        for index, conformer_id in enumerate(partial_ids)
                    ],
                    "fallback_audit": fallback_audit,
                })
            return base, partial_ids
        if meta_out is not None:
            meta_out.update({
                "status": "total_failed",
                "scene": "B",
                "template_status": template_status,
                "template_audit": template_audit,
                "reason": "ETKDG_POOL_EMBED_FAILED",
                "template_strategy": template_strategy,
                "random_seed": int(random_seed),
                "fallback_audit": fallback_audit,
            })
        return None, ["ETKDG pool embed failed (scene B)"]
    for cid in st:
        try:
            result = int(
                AllChem.MMFFOptimizeMolecule(mol_h, confId=cid, maxIters=50)
            )
            optimization_status = "converged" if result == 0 else "iteration_limit"
            fallback_audit["optimization_trace"].append({
                "pool_conformer_id": int(cid),
                "status": optimization_status,
                "result": result,
            })
        except Exception as exc:
            fallback_audit["optimization_trace"].append({
                "pool_conformer_id": int(cid),
                "status": "exception",
                "exception_type": type(exc).__name__,
            })
    picked = _maxmin_select(mol_h, list(st), n_conformers)
    fallback_audit["selected_pool_conformer_ids"] = [int(cid) for cid in picked]
    fallback_audit["status"] = "success"
    fallback_audit["elapsed_sec"] = round(
        time.perf_counter() - fallback_started, 6
    )
    if worked:
        needed = max(0, int(n_conformers) - len(worked))
        picked = picked[:needed]
        base = Chem.AddHs(Chem.MolFromSmiles(generated_smiles))
        merged_ids = [
            base.AddConformer(m.GetConformer(0), assignId=True)
            for m, _sim, _err, _key in worked
        ]
        merged_ids.extend(
            base.AddConformer(mol_h.GetConformer(cid), assignId=True)
            for cid in picked
        )
        fallback_audit["selected_pool_conformer_ids"] = [int(cid) for cid in picked]
        fallback_audit["output_conformer_ids"] = [int(cid) for cid in merged_ids]
        if meta_out is not None:
            meta_out.update({
                "status": "hybrid_success",
                "scene": "A+B",
                "template_status": template_status,
                "template_audit": template_audit,
                "library_formal": bool(view and view.formal),
                "library_index_sha256": view.index_sha256 if view else None,
                "allowed_sources": sorted(view.allowed_sources) if view else [],
                "template_strategy": template_strategy,
                "random_seed": int(random_seed),
                "requested_conformer_count": int(n_conformers),
                "produced_conformer_count": len(merged_ids),
                "guided_conformer_count": len(worked),
                "fallback_conformer_count": len(picked),
                "conformer_provenance": [
                    *[
                        {
                            "conformer_id": int(conformer_id),
                            "source": "matched_template",
                            "template_key": worked[index][3],
                        }
                        for index, conformer_id in enumerate(
                            merged_ids[:len(worked)]
                        )
                    ],
                    *[
                        {
                            "conformer_id": int(conformer_id),
                            "source": "etkdg_fallback",
                            "template_key": None,
                        }
                        for conformer_id in merged_ids[len(worked):]
                    ],
                ],
                "fallback_audit": fallback_audit,
            })
        return base, merged_ids
    if meta_out is not None:
        meta_out.update({
            "status": "fallback_success",
            "scene": "B",
            "template_status": template_status,
            "template_audit": template_audit,
            "library_formal": bool(view and view.formal),
            "library_index_sha256": view.index_sha256 if view else None,
            "allowed_sources": sorted(view.allowed_sources) if view else [],
            "template_strategy": template_strategy,
            "random_seed": int(random_seed),
            "requested_conformer_count": int(n_conformers),
            "produced_conformer_count": len(picked),
            "guided_conformer_count": 0,
            "fallback_conformer_count": len(picked),
            "conformer_provenance": [
                {
                    "conformer_id": int(conformer_id),
                    "source": "etkdg_fallback",
                    "template_key": None,
                }
                for conformer_id in picked
            ],
            "fallback_audit": fallback_audit,
        })
    return mol_h, picked


def add_synthetic_template(generated_smiles: str, generated_map: str,
                           mol_3d: "Chem.Mol") -> Optional[str]:
    """Self-augment the template library with a de-novo docked conformer.

    The packaged library (CPBind/CPSea/Scaffold) covers model-generated cyclic
    peptides poorly (~10% Scene-A hit rate), so most RFT peptides fall through to
    unguided ETKDG. When such a peptide docks well, its 3D pose IS a valid bound
    conformation for its residue-count/topology bucket — feeding it back lets the
    library grow to cover the generative distribution as RFT proceeds.

    Writes `mol_3d` (heavy atoms + Hs, one conformer) as a standalone PDB under
    the peptide's bucket dir and appends an index entry with source="synthetic".
    The index write is atomic (temp + os.replace) and this runs only in the main
    process after the serial GPU dock, so there is no concurrent-writer race.

    Returns the new template key, or None on failure (never raises — a failed
    harvest must not break the reward path).
    """
    # Runtime self-augmentation is development-only. Publication benchmarks
    # must use an immutable, pre-frozen template library.
    if os.environ.get("CYCPEP_MASTER_ALLOW_SYNTHETIC_TEMPLATE") != "1":
        return None

    try:
        n_res = _residue_count(generated_map)
        cyc = _cyc_mode(generated_map)
        bucket = f"{n_res}_{cyc}"
        bucket_dir = os.path.join(_TEMPLATES_DIR, bucket)
        os.makedirs(bucket_dir, exist_ok=True)

        idx_path = os.path.join(_TEMPLATES_DIR, "templates_index.json")
        index = {}
        if os.path.exists(idx_path):
            with open(idx_path, encoding="utf-8") as f:
                index = json.load(f)

        # unique key: {bucket}_syn_{N:03d}, N = next free synthetic slot
        existing = [k for k in index if k.startswith(f"{bucket}_syn_")]
        n = len(existing) + 1
        while f"{bucket}_syn_{n:03d}" in index:
            n += 1
        key = f"{bucket}_syn_{n:03d}"

        pdb_rel = f"{bucket}/synthetic_{n:03d}.pdb"
        pdb_abs = os.path.join(_TEMPLATES_DIR, pdb_rel)
        pdb_block = Chem.MolToPDBBlock(mol_3d)
        if not pdb_block or "ATOM" not in pdb_block and "HETATM" not in pdb_block:
            return None
        with open(pdb_abs, "w") as f:
            f.write(pdb_block)

        index[key] = {
            "pdb_path": pdb_rel,
            "smiles": generated_smiles,
            "map": generated_map,
            "n_res": n_res,
            "cyc_mode": cyc,
            "source": "synthetic",
            "source_filename": key,
        }
        # atomic index write so a crash can't truncate the shared index
        tmp = idx_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
        os.replace(tmp, idx_path)

        # invalidate the module cache so the new template is visible next call
        global _INDEX, _DEFAULT_VIEW
        _INDEX = None
        _DEFAULT_VIEW = None
        return key
    except Exception:
        return None
