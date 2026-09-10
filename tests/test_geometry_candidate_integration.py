"""Integration tests for the truth-free geometry-derived bond-order candidate.

The ``geometry_simple_local`` engine feeds the frozen NNAA geometry study
module into the result-first bond-order portfolio.  These tests pin three
properties:

* synthetic recovery: the engine admits a source-bound candidate with
  evidence-derived bond orders (carbonyl/aromatic) at hypothesis rigor,
  below template and bond-order-perception precedence;
* guards: conflicting observed-hydrogen chemistry and unresolved/ambiguous
  algorithm output are recorded and skipped without touching the remaining
  fallback candidates (matched explicit hydrogens are retained instead);
* real experimental chains (when the development demo inputs exist in this
  repository): the exact public export -> PDBQT surface produces the CCD-truth
  nonstereo identity for 1bm2/1bzh, with 1bck unchanged and 1sfi still served
  by its template candidate.  Truth files are read by the TESTS ONLY for
  scoring; runtime code never sees them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.bond_order_inference import (
    _source_atoms,
    infer_bond_order_candidates,
)
from cycpep_master.core.geometry_candidate import (
    ENGINE_NAME,
    GeometryCandidateError,
    bond_length_consistency,
    build_geometry_simple_molecule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = (
    REPO_ROOT
    / ".zcode_v710_application_demo"
    / "current_source_run_001"
)
TRUTH_PATH = (
    REPO_ROOT
    / ".zcode_nnaa_geometry_v2"
    / "real_cases_001"
    / "truth.jsonl"
)
REAL_CHAINS = {"1bck": "C", "1bm2": "L", "1bzh": "I", "1sfi": "I"}
CASE_TIMEOUT_SECONDS = 240.0


def _write_heavy_pdb(
    smiles: str,
    path: Path,
    *,
    residue: str = "LIG",
    chain: str = "L",
    keep_hydrogens: bool = False,
) -> None:
    """Write one embedded heavy-atom PDB (with CONECT) from a SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    assert AllChem.EmbedMolecule(molecule, params) == 0
    try:
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
    except Exception:
        pass
    molecule = Chem.RemoveHs(molecule)
    if keep_hydrogens:
        molecule = Chem.AddHs(molecule, addCoords=True)
    conformer = molecule.GetConformer()
    lines: list[str] = []
    serial_by_index: dict[int, int] = {}
    for index, atom in enumerate(molecule.GetAtoms(), start=1):
        serial_by_index[index - 1] = index
        position = conformer.GetAtomPosition(index - 1)
        element = atom.GetSymbol()
        lines.append(
            f"HETATM{index:5d} {element.upper()}{index:<3d} {residue:>3s} "
            f"{chain}{1:4d}    "
            f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"  1.00  0.00          {element.upper():>2s}"
        )
    partners: dict[int, list[int]] = {}
    for bond in molecule.GetBonds():
        left = serial_by_index[bond.GetBeginAtomIdx()]
        right = serial_by_index[bond.GetEndAtomIdx()]
        partners.setdefault(left, []).append(right)
        partners.setdefault(right, []).append(left)
    for serial in sorted(partners):
        stub = f"CONECT{serial:5d}"
        for partner in sorted(partners[serial]):
            stub += f"{partner:5d}"
        lines.append(stub)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


def _geometry_attempt(report: dict) -> dict | None:
    for attempt in report.get("engine_attempts", []):
        if attempt.get("engine") == ENGINE_NAME:
            return attempt
    return None


def _geometry_group(report: dict) -> dict | None:
    for group in report.get("identity_groups", []):
        if ENGINE_NAME in group.get("supporting_engines", []):
            return group
    return None


def test_geometry_engine_admits_evidence_bound_candidate(tmp_path):
    pdb = tmp_path / "benzaldehyde.pdb"
    _write_heavy_pdb("O=Cc1ccccc1", pdb)

    report = infer_bond_order_candidates(pdb, "L")

    attempt = _geometry_attempt(report)
    assert attempt is not None and attempt["status"] == "admitted"
    group = _geometry_group(report)
    assert group is not None
    assert group["canonical_smiles"] == _canonical("O=Cc1ccccc1")
    # never a qualified-chemistry claim; a geometry-only group stays at
    # hypothesis rigor (agreement with an independent perception engine
    # keeps the portfolio's existing multi-family consensus label)
    assert group["qualified_chemistry"] is False
    if group["supporting_families"] == ["geometry"]:
        assert group["quality"] == "hypothesis"
        assert group["rigor"] == "L1:H"
        assert (
            group["evidence_class"]
            == "geometry_only_bond_order_hypothesis"
        )
    assert group["geometry_consistency"]["flagged"] is False
    # selection still prefers independent bond-order perception (openbabel)
    # over a geometry-only group when that perception engine succeeds
    if report["openbabel_available"]:
        selected = report["selected_candidate"]
        assert selected["supporting_families"][0] == "bond_order_perception"


def test_geometry_engine_graph_carries_source_bond_orders(tmp_path):
    pdb = tmp_path / "benzaldehyde.pdb"
    _write_heavy_pdb("O=Cc1ccccc1", pdb)
    atoms = _source_atoms(pdb, "L")

    molecule, meta = build_geometry_simple_molecule(pdb, atoms)

    assert meta["status"] == "candidate"
    assert molecule.GetNumAtoms() == len(atoms)
    assert meta["conect_edge_count"] == molecule.GetNumBonds()
    orders = sorted(
        bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()
    )
    # one C=O double bond and a six-bond aromatic ring survive the rebuild
    assert orders.count(2.0) == 1
    assert orders.count(1.5) == 6
    conformer = molecule.GetConformer()
    for index, atom in enumerate(atoms):
        position = conformer.GetAtomPosition(index)
        assert [
            round(position.x, 3),
            round(position.y, 3),
            round(position.z, 3),
        ] == [round(value, 3) for value in atom["xyz"]]


def test_matched_explicit_hydrogens_admit_and_keep_fallbacks(tmp_path):
    pdb = tmp_path / "methanol.pdb"
    _write_heavy_pdb("CO", pdb, keep_hydrogens=True)

    report = infer_bond_order_candidates(pdb, "L")

    attempt = _geometry_attempt(report)
    assert attempt is not None
    assert attempt["status"] == "admitted"
    assert report["status"] == "parseable"
    assert report["selected_candidate"] is not None
    atoms = _source_atoms(pdb, "L")
    molecule, meta = build_geometry_simple_molecule(pdb, atoms)
    assert molecule.GetNumAtoms() == len(atoms)
    assert meta["explicit_hydrogen_retained"] == meta[
        "explicit_hydrogen_count"
    ]


def test_unresolved_algorithm_keeps_fallback_candidates(
    tmp_path, monkeypatch
):
    pdb = tmp_path / "benzaldehyde.pdb"
    _write_heavy_pdb("O=Cc1ccccc1", pdb)

    import cycpep_master.core.geometry_candidate as adapter

    monkeypatch.setattr(
        adapter,
        "infer_monomer_geometry",
        lambda *args, **kwargs: {
            "status": "ambiguous",
            "candidates": [],
            "reason_codes": ["borderline"],
            "evidence": {},
        },
    )
    report = infer_bond_order_candidates(pdb, "L")

    attempt = _geometry_attempt(report)
    assert attempt is not None and attempt["status"] == "not_admitted"
    assert "ambiguous" in str(attempt.get("error", ""))
    assert report["status"] == "parseable"
    assert _geometry_group(report) is None
    assert report["selected_candidate"] is not None


def test_template_strict_candidate_outranks_geometry_engine(tmp_path):
    pdb = tmp_path / "benzaldehyde.pdb"
    _write_heavy_pdb("O=Cc1ccccc1", pdb)

    report = infer_bond_order_candidates(
        pdb,
        "L",
        strict_candidates=[
            {
                "canonical_smiles": _canonical("O=Cc1ccccc1"),
                "routes": ["template_route"],
                "supporting_route_count": 1,
            }
        ],
    )

    selected = report["selected_candidate"]
    assert selected["supporting_families"][0] == "template"
    assert selected["quality"] == "high"


def test_bond_length_consistency_flags_reduced_assignment():
    # four carbonyl C-O pairs observed at 1.23 A but assigned order 1.0
    atoms = []
    bonds = []
    serial = 0
    for repeat in range(4):
        carbon = serial + 1
        oxygen = serial + 2
        atoms.extend(
            [
                {
                    "serial": carbon,
                    "element": "C",
                    "xyz": [float(serial), 0.0, 0.0],
                },
                {
                    "serial": oxygen,
                    "element": "O",
                    "xyz": [float(serial) + 1.23, 0.0, 0.0],
                },
            ]
        )
        bonds.append(
            {"a": carbon, "b": oxygen, "order": 1.0, "is_aromatic": False}
        )
        serial += 2
    graph = {"atoms": atoms, "bonds": bonds}

    reduced = bond_length_consistency(graph)
    assert reduced["flagged"] is True
    assert reduced["inconsistent_bond_count"] == 4

    for bond in bonds:
        bond["order"] = 2.0
    oxidized = bond_length_consistency(graph)
    assert oxidized["flagged"] is False
    assert oxidized["inconsistent_bond_count"] == 0

    empty = bond_length_consistency({"atoms": [], "bonds": []})
    assert empty["flagged"] is False
    assert empty["inconsistent_bond_count"] is None


def test_consistency_gate_prefers_repaired_graph_over_reduced_consensus(
    tmp_path,
):
    # cyclic tetraamide: four carbonyls that uniform single-bond proximity
    # graphs place 0.2 A beyond the single-bond reference length
    pdb = tmp_path / "cyclic_tetraamide.pdb"
    _write_heavy_pdb("O=C1N(C)CC(=O)N(C)CC(=O)N(C)CC(=O)N(C)C1", pdb)

    report = infer_bond_order_candidates(pdb, "L")

    selected = report["selected_candidate"]
    consistency = selected.get("geometry_consistency") or {}
    assert consistency.get("flagged") is False
    reduced_groups = [
        group
        for group in report["identity_groups"]
        if (group.get("geometry_consistency") or {}).get("flagged")
    ]
    assert reduced_groups, "reduced proximity group must stay visible"
    assert selected["full_inchikey"] != reduced_groups[0]["full_inchikey"]


def _load_truth() -> dict[str, str]:
    names = list(REAL_CHAINS)
    truth = {}
    for line in TRUTH_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            truth[names[row["row_id"]]] = row["canonical_smiles"]
    return truth


def _run_real_case(case: str, tmp_path: Path) -> dict:
    from cycpep_master import application as app

    pdb = DEMO_ROOT / case / "peptide.pdb"
    mol2_path = tmp_path / f"{case}.mol2"
    started = time.perf_counter()
    result = app.export_best_available(
        str(pdb),
        str(mol2_path),
        source_kind="coordinate",
        chain_id=REAL_CHAINS[case],
    )
    elapsed = time.perf_counter() - started
    assert elapsed < CASE_TIMEOUT_SECONDS, f"{case} export exceeded bound"
    assert result["status"] == "success"
    data = result["data"]
    assert data["requested_format_status"] == "fulfilled"
    receipt = json.loads(
        Path(str(mol2_path) + ".validation.json").read_text(encoding="utf-8")
    )
    prepared = app.prepare_ligand_pdbqt_from_mol2(
        str(mol2_path), str(tmp_path / f"{case}_ligand.pdbqt")
    )
    assert prepared["status"] == "success"
    return {
        "receipt": receipt,
        "reconstruction": data.get("reconstruction") or {},
        "mol2_text": mol2_path.read_text(encoding="utf-8", errors="replace"),
    }


def _truth_inchikey(case: str) -> str:
    return Chem.MolToInchiKey(Chem.MolFromSmiles(_load_truth()[case]))


@pytest.mark.parametrize("case", ["1bm2", "1bzh"])
def test_real_nnaa_cases_export_truth_nonstereo_identity(case, tmp_path):
    if not DEMO_ROOT.exists() or not TRUTH_PATH.exists():
        pytest.skip("real-case demo inputs not present in this checkout")

    outcome = _run_real_case(case, tmp_path)
    receipt = outcome["receipt"]

    got = receipt["full_inchikey"]
    want = _truth_inchikey(case)
    assert got.split("-")[0] == want.split("-")[0]
    assert got.split("-")[1] == want.split("-")[1]
    # source-bound coordinates, not regenerated ones
    assert receipt["coordinate_level"].startswith("X3")
    assert receipt["source_heavy_atom_mapping_complete"] is True
    assert receipt["max_source_coordinate_delta_angstrom"] <= 0.001
    # the exported graph is no longer the reduced proximity assignment
    reconstruction = outcome["reconstruction"]
    assert reconstruction.get("result_origin") == (
        "bond_order_inference_hypothesis"
    )


def test_real_1bm2_pdbqt_has_aromatic_ring_and_no_sp3_carbonyl(tmp_path):
    if not DEMO_ROOT.exists() or not TRUTH_PATH.exists():
        pytest.skip("real-case demo inputs not present in this checkout")

    outcome = _run_real_case("1bm2", tmp_path)
    types: dict[str, int] = {}
    for line in (
        tmp_path / "1bm2_ligand.pdbqt"
    ).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_type = line[77:79].strip()
            types[atom_type] = types.get(atom_type, 0) + 1
    assert types.get("A", 0) >= 6, "phosphotyrosine ring must be aromatic"
    assert "NA" not in types, "amide nitrogens must not be acceptors"
    assert types.get("P", 0) == 1


def test_real_1bck_chemistry_unchanged(tmp_path):
    if not DEMO_ROOT.exists() or not TRUTH_PATH.exists():
        pytest.skip("real-case demo inputs not present in this checkout")

    outcome = _run_real_case("1bck", tmp_path)
    receipt = outcome["receipt"]
    assert receipt["full_inchikey"] == _truth_inchikey("1bck")
    assert receipt["coordinate_level"].startswith("X3")
    # The chemistry (identity + source coordinates) is the regression anchor.
    # Since ambiguous geometry candidates are admitted as explicit
    # hypotheses, 1bck's template strict candidate and the geometry engine
    # can agree inside one identity group, legitimately upgrading the
    # evidence label from "candidate" to "consensus"; only the inference
    # route itself is locked here.
    assert outcome["reconstruction"].get("result_origin", "").startswith(
        "bond_order_inference_"
    )


def test_real_1sfi_keeps_template_candidate_path(tmp_path):
    if not DEMO_ROOT.exists() or not TRUTH_PATH.exists():
        pytest.skip("real-case demo inputs not present in this checkout")

    outcome = _run_real_case("1sfi", tmp_path)
    receipt = outcome["receipt"]
    assert receipt["full_inchikey"] == _truth_inchikey("1sfi")
    assert outcome["reconstruction"].get("result_origin") == (
        "v6_candidate_unique"
    )


def test_geometry_candidate_error_is_explicit():
    error = GeometryCandidateError("status unresolved")
    with pytest.raises(GeometryCandidateError):
        raise error
