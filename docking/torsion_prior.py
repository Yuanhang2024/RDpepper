"""Hash-verified, cached hierarchical torsion priors for V5 preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

RUNTIME_SCHEMA_VERSION = "1.0.0-cycpep-torsion-runtime.2"
MANIFEST_SCHEMA_VERSION = "1.0.0-cycpep-torsion-manifest.1"
LOOKUP_LEVELS = (
    "exact",
    "residue_class_ring",
    "morgan",
    "generic",
)
STANDARD_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
})
RESIDUE_CLASSES = {
    "ALA": "hydrophobic",
    "VAL": "hydrophobic",
    "LEU": "hydrophobic",
    "ILE": "hydrophobic",
    "MET": "hydrophobic",
    "PHE": "aromatic",
    "TRP": "aromatic",
    "TYR": "aromatic",
    "SER": "polar",
    "THR": "polar",
    "ASN": "polar",
    "GLN": "polar",
    "CYS": "sulfur",
    "GLY": "special",
    "PRO": "special",
    "ASP": "acidic",
    "GLU": "acidic",
    "LYS": "basic",
    "ARG": "basic",
    "HIS": "basic_aromatic",
}
MODE_RIGIDITY_THRESHOLDS = {
    "fast": 0.85,
    "balanced": 0.75,
    "thorough": 0.65,
}
_TORSION_PRIOR_CACHE_LOCK = threading.RLock()
_TORSION_PRIOR_CACHE: dict[tuple[Any, ...], "TorsionPriorIndex"] = {}
_TORSION_PRIOR_CACHE_HITS = 0
_TORSION_PRIOR_CACHE_MISSES = 0
_TORSION_PRIOR_CACHE_MAX_ENTRIES = 8


class TorsionPriorError(ValueError):
    """A torsion-prior resource or query violates its contract."""


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({
            str(key): _freeze_json(item)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _resource_fingerprint(path: Path) -> tuple[Any, ...]:
    stat = path.stat()
    return (
        str(path),
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def clear_torsion_prior_cache() -> None:
    """Clear process-local immutable prior resources (primarily for tests)."""
    global _TORSION_PRIOR_CACHE_HITS, _TORSION_PRIOR_CACHE_MISSES
    with _TORSION_PRIOR_CACHE_LOCK:
        _TORSION_PRIOR_CACHE.clear()
        _TORSION_PRIOR_CACHE_HITS = 0
        _TORSION_PRIOR_CACHE_MISSES = 0


def torsion_prior_cache_info() -> dict[str, int]:
    with _TORSION_PRIOR_CACHE_LOCK:
        return {
            "entries": len(_TORSION_PRIOR_CACHE),
            "hits": _TORSION_PRIOR_CACHE_HITS,
            "misses": _TORSION_PRIOR_CACHE_MISSES,
        }


@dataclass(frozen=True)
class TorsionQueryKeys:
    exact: str
    residue_class_ring: str
    morgan: str
    generic: str
    torsion_kind: str


@dataclass(frozen=True)
class TorsionPriorMatch:
    status: str
    lookup_level: str | None
    key: str | None
    rigidity_score: float | None
    confidence: str
    eligible_to_freeze: bool
    statistics: Mapping[str, Any] | None
    reason: str | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atom_name(atom: Chem.Atom) -> str:
    if atom.HasProp("_TriposAtomName"):
        return atom.GetProp("_TriposAtomName").strip().upper()
    info = atom.GetPDBResidueInfo()
    if info is not None:
        return info.GetName().strip().upper()
    return atom.GetSymbol().upper()


def _residue_name(atom: Chem.Atom) -> str:
    if atom.HasProp("_TriposResidueName"):
        value = atom.GetProp("_TriposResidueName").strip().upper()
    else:
        info = atom.GetPDBResidueInfo()
        value = (
            info.GetResidueName().strip().upper()
            if info is not None
            else "UNK"
        )
    match = re.match(r"([A-Z]{3})", value)
    normalized = match.group(1) if match else value
    return normalized if normalized in STANDARD_RESIDUES else normalized


def _residue_number(atom: Chem.Atom) -> int | None:
    if atom.HasProp("_TriposResidueNumber"):
        return int(atom.GetIntProp("_TriposResidueNumber"))
    info = atom.GetPDBResidueInfo()
    return int(info.GetResidueNumber()) if info is not None else None


def _ring_bin(size: int | None) -> str:
    if size is None or size <= 0:
        return "unknown"
    if size <= 6:
        return "4-6"
    if size <= 9:
        return "7-9"
    if size <= 13:
        return "10-13"
    if size <= 16:
        return "14-16"
    return "17+"


def _is_amide_bond(bond: Chem.Bond) -> bool:
    left = bond.GetBeginAtom()
    right = bond.GetEndAtom()
    atoms = (
        (left, right)
        if left.GetSymbol() == "C"
        else (right, left)
    )
    carbon, nitrogen = atoms
    if carbon.GetSymbol() != "C" or nitrogen.GetSymbol() != "N":
        return False
    return any(
        neighbor.GetSymbol() == "O"
        and carbon.GetOwningMol().GetBondBetweenAtoms(
            carbon.GetIdx(), neighbor.GetIdx()
        ).GetBondType() == Chem.BondType.DOUBLE
        for neighbor in carbon.GetNeighbors()
        if neighbor.GetIdx() != nitrogen.GetIdx()
    )


def _torsion_kind(left: Chem.Atom, right: Chem.Atom) -> str:
    names = {_atom_name(left), _atom_name(right)}
    same_residue = _residue_number(left) == _residue_number(right)
    if names == {"N", "CA"} and same_residue:
        return "phi"
    if names == {"CA", "C"} and same_residue:
        return "psi"
    if names == {"C", "N"} and not same_residue:
        return "omega"
    if names == {"CA", "CB"} and same_residue:
        return "chi1"
    return "sidechain_or_closure"


def _endpoint_descriptor(atom: Chem.Atom) -> dict[str, Any]:
    neighbor_residues = sorted({
        _residue_name(neighbor)
        for neighbor in atom.GetNeighbors()
        if neighbor.GetAtomicNum() > 1
        and _residue_number(neighbor) != _residue_number(atom)
    })
    residue = _residue_name(atom)
    return {
        "residue": residue,
        "residue_class": RESIDUE_CLASSES.get(residue, "nonstandard"),
        "atom": _atom_name(atom),
        "element": atom.GetSymbol(),
        "formal_charge": atom.GetFormalCharge(),
        "neighbor_residues": neighbor_residues,
    }


def _morgan_bond_environment(
    molecule: Chem.Mol,
    left: int,
    right: int,
) -> str:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fingerprint = generator.GetSparseCountFingerprint(
        molecule,
        fromAtoms=[int(left), int(right)],
    )
    payload = {
        "features": sorted(
            (int(key), int(value))
            for key, value in fingerprint.GetNonzeroElements().items()
        ),
        "bond_type": str(
            molecule.GetBondBetweenAtoms(left, right).GetBondType()
        ),
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("ascii")
    ).hexdigest()


def _graph_identity(molecule: Chem.Mol) -> tuple[str, str]:
    heavy = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
    smiles = Chem.MolToSmiles(
        heavy,
        canonical=True,
        isomericSmiles=True,
    )
    graph_sha256 = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    canonical_ranks = list(
        Chem.CanonicalRankAtoms(
            heavy,
            breakTies=True,
            includeChirality=True,
            includeIsotopes=True,
        )
    )
    protonation = _canonical_json({
        "formal_charge": int(Chem.GetFormalCharge(heavy)),
        "charged_atoms": sorted(
            (
                int(canonical_ranks[atom.GetIdx()]),
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
            )
            for atom in heavy.GetAtoms()
            if atom.GetFormalCharge()
        ),
    })
    return graph_sha256, protonation


def _canonical_bond_uid(
    molecule: Chem.Mol,
    left: int,
    right: int,
) -> str:
    ranks = list(
        Chem.CanonicalRankAtoms(
            molecule,
            breakTies=True,
            includeChirality=True,
            includeIsotopes=True,
        )
    )
    bond = molecule.GetBondBetweenAtoms(left, right)
    payload = {
        "endpoint_ranks": sorted((int(ranks[left]), int(ranks[right]))),
        "bond_type": str(bond.GetBondType()),
        "stereo": str(bond.GetStereo()),
        "aromatic": bool(bond.GetIsAromatic()),
    }
    return _canonical_json(payload)


def build_query_keys(
    molecule: Chem.Mol,
    bond_atom_pair: tuple[int, int],
    *,
    topology_class: str | None,
    macrocycle_ring_size: int | None,
) -> TorsionQueryKeys:
    original_left, original_right = sorted(
        (int(bond_atom_pair[0]), int(bond_atom_pair[1]))
    )
    if (
        original_left < 0
        or original_right >= molecule.GetNumAtoms()
        or original_left == original_right
    ):
        raise TorsionPriorError("invalid torsion-prior atom pair")
    heavy_indices = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if (
        original_left not in heavy_indices
        or original_right not in heavy_indices
    ):
        raise TorsionPriorError(
            "torsion-prior central bond must join heavy atoms"
        )
    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(heavy_indices)
    }
    working = (
        Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
        if len(heavy_indices) != molecule.GetNumAtoms()
        else Chem.Mol(molecule)
    )
    left_index = old_to_new[original_left]
    right_index = old_to_new[original_right]
    bond = working.GetBondBetweenAtoms(left_index, right_index)
    if bond is None:
        raise TorsionPriorError(
            "torsion-prior atom pair is not a molecular bond"
        )
    left = working.GetAtomWithIdx(left_index)
    right = working.GetAtomWithIdx(right_index)
    endpoints = sorted(
        (_endpoint_descriptor(left), _endpoint_descriptor(right)),
        key=_canonical_json,
    )
    kind = _torsion_kind(left, right)
    ring_bin = _ring_bin(macrocycle_ring_size)
    graph_sha256, protonation_signature = _graph_identity(working)
    exact = _canonical_json({
        "graph_sha256": graph_sha256,
        "protonation_signature": protonation_signature,
        "canonical_bond_uid": _canonical_bond_uid(
            working, left_index, right_index
        ),
        "torsion_kind": kind,
        "topology_class": topology_class or "unknown",
        "macrocycle_ring_size": macrocycle_ring_size,
        "endpoints": endpoints,
    })
    residue_class_ring = _canonical_json({
        "torsion_kind": kind,
        "topology_class": topology_class or "unknown",
        "ring_bin": ring_bin,
        "endpoint_classes": sorted(
            endpoint["residue_class"] for endpoint in endpoints
        ),
        "endpoint_atoms": sorted(
            endpoint["atom"] for endpoint in endpoints
        ),
    })
    generic = _canonical_json({
        "elements": sorted((left.GetSymbol(), right.GetSymbol())),
        "hybridizations": sorted((
            str(left.GetHybridization()),
            str(right.GetHybridization()),
        )),
        "degrees": sorted((left.GetDegree(), right.GetDegree())),
        "formal_charges": sorted((
            left.GetFormalCharge(),
            right.GetFormalCharge(),
        )),
        "bond_type": str(bond.GetBondType()),
        "amide": _is_amide_bond(bond),
        "ring": bool(bond.IsInRing()),
    })
    return TorsionQueryKeys(
        exact=exact,
        residue_class_ring=residue_class_ring,
        morgan=_morgan_bond_environment(
            working, left_index, right_index
        ),
        generic=generic,
        torsion_kind=kind,
    )


def classify_macrocycle_topology(
    *,
    residue_count: int,
    head_to_tail: bool,
    disulfide: bool,
    other: bool,
    closure_residue_spans: Sequence[int] = (),
) -> tuple[str, int | None]:
    """Classify macrocycle topology in the shared table vocabulary.

    Single classification core shared by the observation builder
    (docking/torsion_observations.py) and runtime prior queries so that
    table keys and lookup keys cannot drift apart.  ``closure_residue_spans``
    are ``|position difference| + 1`` values per closure bond.
    """
    if head_to_tail and (disulfide or other):
        topology = "mixed"
    elif head_to_tail:
        topology = "head_to_tail"
    elif disulfide and other:
        topology = "mixed_sidechain"
    elif disulfide:
        topology = "disulfide"
    elif other:
        topology = "sidechain"
    else:
        topology = "linear"
    ring_size = (
        residue_count
        if head_to_tail
        else max(closure_residue_spans, default=None)
    )
    return topology, ring_size


def _chain_of(atom: Chem.Atom) -> str | None:
    if atom.HasProp("_TriposChainId"):
        return atom.GetProp("_TriposChainId").strip().upper()
    info = atom.GetPDBResidueInfo()
    return (
        info.GetChainId().strip().upper() if info is not None else None
    )


def _residue_anchor(
    molecule: Chem.Mol,
    *,
    chain: str | None,
    residue_number: int | None,
    name: str,
) -> int | None:
    if residue_number is None:
        return None
    for atom in molecule.GetAtoms():
        if (
            _atom_name(atom) == name
            and _residue_number(atom) == residue_number
            and _chain_of(atom) == chain
        ):
            return atom.GetIdx()
    return None


def _chain_residue_bounds(
    molecule: Chem.Mol,
    chain: str | None,
) -> tuple[int | None, int | None]:
    numbers = [
        number
        for number in (
            _residue_number(atom)
            for atom in molecule.GetAtoms()
            if _chain_of(atom) == chain
        )
        if number is not None
    ]
    return (min(numbers), max(numbers)) if numbers else (None, None)


def _first_heavy_neighbor(
    atom: Chem.Atom,
    excluded: int,
) -> int | None:
    neighbors = sorted(
        neighbor.GetIdx()
        for neighbor in atom.GetNeighbors()
        if neighbor.GetIdx() != excluded
        and neighbor.GetAtomicNum() > 1
    )
    return neighbors[0] if neighbors else None


def prior_guidance_quartet(
    molecule: Chem.Mol,
    bond_atom_pair: tuple[int, int],
    *,
    cyclic_backbone: bool = False,
) -> tuple[int, int, int, int] | None:
    """Deterministic heavy-atom quartet for applying table dihedral means.

    Mirrors the observation-side conventions in
    docking/torsion_observations.py: phi/psi/omega anchor on the canonical
    N/CA/C atoms of the adjacent residues (backbone wrap-around for cyclic
    peptides), chi1 anchors on the backbone N, and every other bond takes
    the index-first heavy neighbour on each side.  Hydrogens are never
    selected.
    """
    left_index, right_index = sorted(
        (int(bond_atom_pair[0]), int(bond_atom_pair[1]))
    )
    left = molecule.GetAtomWithIdx(left_index)
    right = molecule.GetAtomWithIdx(right_index)
    kind = _torsion_kind(left, right)
    names = {_atom_name(left), _atom_name(right)}

    def endpoint_by_name(name: str) -> Chem.Atom:
        return left if _atom_name(left) == name else right

    if kind == "phi" and names == {"N", "CA"}:
        n_atom, ca_atom = endpoint_by_name("N"), endpoint_by_name("CA")
        residue = _residue_number(n_atom)
        chain = _chain_of(n_atom)
        chain_min, chain_max = _chain_residue_bounds(molecule, chain)
        previous = residue - 1
        if previous < (chain_min if chain_min is not None else previous):
            previous = chain_max if cyclic_backbone else None
        return _ordered_quartet(
            molecule,
            _residue_anchor(
                molecule,
                chain=chain,
                residue_number=previous,
                name="C",
            ),
            n_atom.GetIdx(),
            ca_atom.GetIdx(),
            _residue_anchor(
                molecule,
                chain=chain,
                residue_number=residue,
                name="C",
            ),
        )
    if kind == "psi" and names == {"CA", "C"}:
        ca_atom, c_atom = endpoint_by_name("CA"), endpoint_by_name("C")
        residue = _residue_number(ca_atom)
        chain = _chain_of(ca_atom)
        chain_min, chain_max = _chain_residue_bounds(molecule, chain)
        following = residue + 1
        if chain_max is not None and following > chain_max:
            following = chain_min if cyclic_backbone else None
        return _ordered_quartet(
            molecule,
            _residue_anchor(
                molecule,
                chain=chain,
                residue_number=residue,
                name="N",
            ),
            ca_atom.GetIdx(),
            c_atom.GetIdx(),
            _residue_anchor(
                molecule,
                chain=chain,
                residue_number=following,
                name="N",
            ),
        )
    if kind == "omega" and names == {"C", "N"}:
        c_atom, n_atom = endpoint_by_name("C"), endpoint_by_name("N")
        c_chain, n_chain = _chain_of(c_atom), _chain_of(n_atom)
        return _ordered_quartet(
            molecule,
            _residue_anchor(
                molecule,
                chain=c_chain,
                residue_number=_residue_number(c_atom),
                name="CA",
            ),
            c_atom.GetIdx(),
            n_atom.GetIdx(),
            _residue_anchor(
                molecule,
                chain=n_chain,
                residue_number=_residue_number(n_atom),
                name="CA",
            ),
        )
    if kind == "chi1" and names == {"CA", "CB"}:
        ca_atom, cb_atom = endpoint_by_name("CA"), endpoint_by_name("CB")
        return _ordered_quartet(
            molecule,
            _residue_anchor(
                molecule,
                chain=_chain_of(ca_atom),
                residue_number=_residue_number(ca_atom),
                name="N",
            ),
            ca_atom.GetIdx(),
            cb_atom.GetIdx(),
            _first_heavy_neighbor(cb_atom, ca_atom.GetIdx()),
        )
    return _ordered_quartet(
        molecule,
        _first_heavy_neighbor(left, right_index),
        left_index,
        right_index,
        _first_heavy_neighbor(right, left_index),
    )


def _ordered_quartet(
    molecule: Chem.Mol,
    first: int | None,
    second: int | None,
    third: int | None,
    fourth: int | None,
) -> tuple[int, int, int, int] | None:
    if None in (first, second, third, fourth):
        return None
    quartet = (int(first), int(second), int(third), int(fourth))
    if len(set(quartet)) != 4:
        return None
    if molecule.GetBondBetweenAtoms(quartet[1], quartet[2]) is None:
        return None
    return quartet


class TorsionPriorIndex:
    def __init__(
        self,
        runtime: Mapping[str, Any],
        *,
        runtime_sha256: str,
        manifest: Mapping[str, Any],
        manifest_sha256: str,
    ):
        self.runtime = _freeze_json(dict(runtime))
        self.runtime_sha256 = runtime_sha256
        self.manifest = _freeze_json(dict(manifest))
        self.manifest_sha256 = manifest_sha256
        self.levels = self.runtime["levels"]

    def query(
        self,
        keys: TorsionQueryKeys,
        *,
        flexibility_mode: str = "balanced",
    ) -> TorsionPriorMatch:
        if flexibility_mode not in MODE_RIGIDITY_THRESHOLDS:
            raise TorsionPriorError(
                "flexibility_mode must be fast, balanced, or thorough"
            )
        for level in LOOKUP_LEVELS:
            key = getattr(keys, level)
            entry = self.levels.get(level, {}).get(key)
            if not isinstance(entry, Mapping):
                continue
            confidence = str(entry.get("confidence") or "low")
            score = float(entry.get("rigidity_score") or 0.0)
            calibration_unit = str(
                entry.get("calibration_unit") or ""
            )
            false_rigid_count = entry.get(
                "calibration_false_rigid_count"
            )
            evaluable_count = entry.get(
                "calibration_evaluable_count"
            )
            false_rigid_upper = entry.get(
                "calibration_false_rigid_ci_high"
            )
            if false_rigid_upper is None:
                calibration_unit = str(
                    entry.get("calibration_unit") or "entity"
                )
                false_rigid_upper = entry.get(
                    (
                        "leave_one_structure_out_false_rigid_ci_high"
                        if calibration_unit == "structure"
                        else "leave_one_entity_out_false_rigid_ci_high"
                    )
                )
            expected_unit = (
                "structure" if level == "exact" else "entity"
            )
            valid_counts = bool(
                isinstance(false_rigid_count, int)
                and not isinstance(false_rigid_count, bool)
                and isinstance(evaluable_count, int)
                and not isinstance(evaluable_count, bool)
                and evaluable_count > 0
                and 0 <= false_rigid_count <= evaluable_count
            )
            calibration_passed = False
            if false_rigid_upper is not None:
                try:
                    upper = float(false_rigid_upper)
                    calibration_passed = bool(
                        calibration_unit == expected_unit
                        and valid_counts
                        and 0.0 <= upper <= 0.10
                    )
                except (TypeError, ValueError, OverflowError):
                    calibration_passed = False
            eligible = bool(
                confidence in {"high", "medium"}
                and score
                >= MODE_RIGIDITY_THRESHOLDS[flexibility_mode]
                and calibration_passed
            )
            return TorsionPriorMatch(
                status="matched",
                lookup_level=level,
                key=key,
                rigidity_score=score,
                confidence=confidence,
                eligible_to_freeze=eligible,
                statistics=entry,
                reason=(
                    None
                    if eligible
                    else "matched prior is not a high-confidence rigid candidate"
                ),
            )
        return TorsionPriorMatch(
            status="unavailable",
            lookup_level=None,
            key=None,
            rigidity_score=None,
            confidence="unavailable",
            eligible_to_freeze=False,
            statistics=None,
            reason="no torsion prior matched; bond remains flexible",
        )


def load_torsion_prior(
    runtime_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    use_cache: bool = True,
) -> TorsionPriorIndex:
    global _TORSION_PRIOR_CACHE_HITS, _TORSION_PRIOR_CACHE_MISSES
    runtime_source = Path(runtime_path).resolve()
    manifest_source = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else (
            runtime_source.with_name(
                "torsion_prior_manifest_runtime.json"
            )
            if runtime_source.with_name(
                "torsion_prior_manifest_runtime.json"
            ).is_file()
            else runtime_source.with_name(
                "torsion_prior_manifest.json"
            )
        )
    )
    if not runtime_source.is_file():
        raise TorsionPriorError(
            f"torsion runtime index is missing: {runtime_source}"
        )
    if not manifest_source.is_file():
        raise TorsionPriorError(
            f"torsion prior manifest is missing: {manifest_source}"
        )
    cache_key = (
        "torsion-prior-cache-v1",
        RUNTIME_SCHEMA_VERSION,
        MANIFEST_SCHEMA_VERSION,
        _resource_fingerprint(runtime_source),
        _resource_fingerprint(manifest_source),
    )
    with _TORSION_PRIOR_CACHE_LOCK:
        if use_cache:
            cached = _TORSION_PRIOR_CACHE.get(cache_key)
            if cached is not None:
                _TORSION_PRIOR_CACHE_HITS += 1
                return cached
            _TORSION_PRIOR_CACHE_MISSES += 1
        try:
            runtime_bytes = runtime_source.read_bytes()
            manifest_bytes = manifest_source.read_bytes()
            runtime = json.loads(runtime_bytes.decode("utf-8"))
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            raise TorsionPriorError(
                f"cannot read torsion prior resource: {exc}"
            ) from exc
        if runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise TorsionPriorError("unsupported torsion runtime schema")
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise TorsionPriorError("unsupported torsion manifest schema")
        runtime_hash = hashlib.sha256(runtime_bytes).hexdigest()
        if manifest.get("runtime_sha256") != runtime_hash:
            raise TorsionPriorError(
                "torsion runtime SHA-256 differs from its manifest"
            )
        levels = runtime.get("levels")
        if not isinstance(levels, dict):
            raise TorsionPriorError("torsion runtime lacks lookup levels")
        for level in LOOKUP_LEVELS:
            if not isinstance(levels.get(level), dict):
                raise TorsionPriorError(
                    f"torsion runtime lacks lookup level: {level}"
                )
        if cache_key != (
            "torsion-prior-cache-v1",
            RUNTIME_SCHEMA_VERSION,
            MANIFEST_SCHEMA_VERSION,
            _resource_fingerprint(runtime_source),
            _resource_fingerprint(manifest_source),
        ):
            raise TorsionPriorError(
                "torsion prior resource changed while it was being loaded"
            )
        index = TorsionPriorIndex(
            runtime,
            runtime_sha256=runtime_hash,
            manifest=manifest,
            manifest_sha256=hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        )
        if use_cache:
            _TORSION_PRIOR_CACHE[cache_key] = index
            while len(_TORSION_PRIOR_CACHE) > (
                _TORSION_PRIOR_CACHE_MAX_ENTRIES
            ):
                first_key = next(iter(_TORSION_PRIOR_CACHE))
                _TORSION_PRIOR_CACHE.pop(first_key)
        return index
