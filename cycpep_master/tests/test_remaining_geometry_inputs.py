"""Remaining geometry-input coverage: cap/linker hints and explicit hydrogens.

Contract under test:

* ``cycpep_master.core.monomer_resolution``: HETATM caps/linkers reach the
  coordinate symbol hints only with actual peptide-connection evidence
  (LINK/CONECT to a peptide residue) or a recognized cap/linker signature
  (ACE/NME names, N1+C1+C5 atoms as in AEA) sequence-adjacent in the same
  chain.  Unrelated ligands, waters, and ions are never hinted.
* ``cycpep_master.core.geometry_candidate``: observed explicit hydrogens are
  projected to a heavy-only graph for the frozen algorithm, then re-attached
  so the rebuilt molecule keeps the FULL source atom order; conflicting
  observed hydrogen chemistry rejects the geometry candidate and the
  portfolio keeps its fallbacks; heavy-only inputs keep their exact
  historical output shape.
* ``cycpep_master.export.conformer``: a verified source handoff whose MOL2
  writer fails without an IO error falls back to the honest X1 SMILES route
  with the concrete blocker recorded; IO errors still propagate.

Real run_002 demo inputs are read by these TESTS ONLY for light component
extraction (hints + adapter).  No full exports or docking runs here.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

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
    build_geometry_simple_molecule,
)
from cycpep_master.core.monomer_resolution import monomer_symbol_hints
from cycpep_master.export import conformer as conformer_module
from cycpep_master.export.conformer import _result_first_candidate_to_mol2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = (
    REPO_ROOT / ".zcode_v710_application_demo" / "current_source_run_002"
)
REAL_CHAINS = {"1bck": "C", "1bm2": "L", "1bzh": "I", "1sfi": "I"}


# ---------------------------------------------------------------------------
# Synthetic PDB builders
# ---------------------------------------------------------------------------

def _atom(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    xyz: tuple[float, float, float],
    element: str,
    *,
    hetatm: bool = False,
) -> str:
    record = "HETATM" if hetatm else "ATOM  "
    return (
        f"{record}{serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
        f"          {element:>2s}"
    )


def _link(
    name_a: str,
    res_a: str,
    chain_a: str,
    seq_a: int,
    name_b: str,
    res_b: str,
    chain_b: str,
    seq_b: int,
) -> str:
    return (
        f"LINK         {name_a:<4s}{res_a:>3s} {chain_a}{seq_a:4d}                "
        f"{name_b:<4s}{res_b:>3s} {chain_b}{seq_b:4d}      1555   1555  1.33"
    )


def _peptide_backbone(
    start_serial: int,
    resname: str,
    chain: str,
    resseq: int,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    hetatm: bool = False,
) -> tuple[list[str], int]:
    """Three-point backbone N/CA/C with realistic 1.33-1.52 A spacing."""
    rows = [
        _atom(
            start_serial, "N", resname, chain, resseq,
            origin, "N", hetatm=hetatm,
        ),
        _atom(
            start_serial + 1, "CA", resname, chain, resseq,
            (origin[0] + 1.45, origin[1], origin[2]), "C", hetatm=hetatm,
        ),
        _atom(
            start_serial + 2, "C", resname, chain, resseq,
            (origin[0] + 2.94, origin[1], origin[2]), "C", hetatm=hetatm,
        ),
    ]
    return rows, start_serial + 3


def _write_heavy_pdb(
    smiles: str,
    path: Path,
    *,
    keep_hydrogens: bool = False,
) -> None:
    """Write one embedded PDB (chain L, residue LIG 1, with CONECT)."""
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
    partners: dict[int, list[int]] = {}
    for index, atom in enumerate(molecule.GetAtoms(), start=1):
        position = conformer.GetAtomPosition(index - 1)
        element = atom.GetSymbol()
        lines.append(
            f"HETATM{index:5d} {element.upper()}{index:<3d} LIG L   1    "
            f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"  1.00  0.00          {element.upper():>2s}"
        )
        partners.setdefault(index, [])
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1
        partners.setdefault(left, []).append(right)
        partners.setdefault(right, []).append(left)
    for serial in sorted(partners):
        stub = f"CONECT{serial:5d}"
        for partner in sorted(partners[serial]):
            stub += f"{partner:5d}"
        lines.append(stub)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _relocate_one_carbon_hydrogen_to_oxygen(path: Path) -> None:
    """Move one C-riding hydrogen to 0.98 A from O (2.4 A from its CONECT C)."""
    lines = path.read_text(encoding="ascii").splitlines()
    heavy = {}
    for line in lines:
        if line.startswith("HETATM") and line[76:78].strip() in {"C", "O"}:
            heavy[line[76:78].strip()] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
    oxygen, carbon = heavy["O"], heavy["C"]
    for index, line in enumerate(lines):
        if not (line.startswith("HETATM") and line[76:78].strip() == "H"):
            continue
        position = (
            float(line[30:38]), float(line[38:46]), float(line[46:54])
        )
        carbon_distance = math.dist(position, carbon)
        oxygen_distance = math.dist(position, oxygen)
        if carbon_distance < 1.2 and oxygen_distance > 1.5:
            direction = [
                (o - c) / math.dist(oxygen, carbon)
                for o, c in zip(oxygen, carbon)
            ]
            moved = [o + 0.98 * d for o, d in zip(oxygen, direction)]
            lines[index] = (
                line[:30]
                + f"{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}"
                + line[54:]
            )
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            return
    raise AssertionError("no carbon-bound hydrogen found to relocate")


def _geometry_attempt(report: dict) -> dict | None:
    for attempt in report.get("engine_attempts", []):
        if attempt.get("engine") == ENGINE_NAME:
            return attempt
    return None


def _header_fields(block_or_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block_or_text.splitlines():
        if not line.startswith("#"):
            break
        for token in line.lstrip("#").split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields.setdefault(key, value)
    return fields


def _graphless_result(smiles: str, *, ambiguous: bool = False):
    return SimpleNamespace(
        candidate_smiles=None,
        candidate_graph=None,
        smiles=smiles,
        ambiguous=ambiguous,
    )


# Three-atom chain C-C-O at single-bond lengths: the ethanol candidate maps
# onto it as a full element/bond-order isomorphism, so a certified handoff is
# available for the writer-failure tests.
CCO_CHAIN_PDB = """\
HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG L   1       1.520   0.000   0.000  1.00  0.00           C
HETATM    3  O1  LIG L   1       3.040   0.000   0.000  1.00  0.00           O
CONECT    1    2
CONECT    2    1    3
CONECT    3    2
END
"""


# ---------------------------------------------------------------------------
# Cap/linker hint discovery
# ---------------------------------------------------------------------------

def test_ace_cap_linked_to_peptide_is_hinted(tmp_path):
    pdb = tmp_path / "ace_linked.pdb"
    peptide, next_serial = _peptide_backbone(1, "ALA", "L", 2)
    cap = [
        _atom(next_serial, "CH3", "ACE", "L", 1, (0.0, 1.33, 0.0), "C", hetatm=True),
        _atom(next_serial + 1, "C", "ACE", "L", 1, (1.52, 1.33, 0.0), "C", hetatm=True),
        _atom(next_serial + 2, "O", "ACE", "L", 1, (2.53, 1.33, 0.0), "O", hetatm=True),
    ]
    lines = [
        *cap,
        *peptide,
        _link("C", "ACE", "L", 1, "N", "ALA", "L", 2),
        "END",
    ]
    pdb.write_text("\n".join(lines) + "\n", encoding="ascii")

    hints = monomer_symbol_hints(pdb, kind="coordinate")

    assert "ALA" in hints
    assert "ACE" in hints


def test_aea_linker_connected_via_conect_is_hinted(tmp_path):
    pdb = tmp_path / "aea_conect.pdb"
    peptide, next_serial = _peptide_backbone(1, "LEU", "I", 406)
    linker = [
        _atom(next_serial, "N1", "AEA", "I", 407, (4.4, 0.0, 0.0), "N", hetatm=True),
        _atom(next_serial + 1, "C1", "AEA", "I", 407, (5.8, 0.0, 0.0), "C", hetatm=True),
        _atom(next_serial + 2, "C5", "AEA", "I", 407, (4.4, 1.5, 0.0), "C", hetatm=True),
    ]
    lines = [
        *peptide,
        *linker,
        # CONECT: peptide C (serial 3) bonded to AEA N1
        f"CONECT    3{next_serial:5d}",
        f"CONECT{next_serial:5d}    3{next_serial + 1:5d}{next_serial + 2:5d}",
        "END",
    ]
    pdb.write_text("\n".join(lines) + "\n", encoding="ascii")

    hints = monomer_symbol_hints(pdb, kind="coordinate")

    assert "LEU" in hints
    assert "AEA" in hints


def test_unrelated_ligand_water_ion_not_hinted(tmp_path):
    pdb = tmp_path / "unrelated.pdb"
    peptide, next_serial = _peptide_backbone(1, "GLY", "A", 5)
    entities = [
        _atom(next_serial, "O", "HOH", "A", 90, (20.0, 0.0, 0.0), "O", hetatm=True),
        _atom(next_serial + 1, "S", "SO4", "A", 91, (22.0, 0.0, 0.0), "S", hetatm=True),
        _atom(next_serial + 2, "O1", "SO4", "A", 91, (23.5, 0.0, 0.0), "O", hetatm=True),
        # unrelated ligand in another chain without any LINK/CONECT
        _atom(next_serial + 3, "C1", "LIG", "B", 1, (30.0, 0.0, 0.0), "C", hetatm=True),
        _atom(next_serial + 4, "N1", "LIG", "B", 1, (31.5, 0.0, 0.0), "N", hetatm=True),
    ]
    pdb.write_text("\n".join([*peptide, *entities, "END"]) + "\n", encoding="ascii")

    hints = monomer_symbol_hints(pdb, kind="coordinate")

    assert hints == ("GLY",)


def test_ace_adjacent_without_link_records_is_hinted(tmp_path):
    pdb = tmp_path / "ace_adjacent.pdb"
    peptide, next_serial = _peptide_backbone(1, "VAL", "L", 2)
    cap = [
        _atom(next_serial, "CH3", "ACE", "L", 1, (0.0, 1.33, 0.0), "C", hetatm=True),
        _atom(next_serial + 1, "C", "ACE", "L", 1, (1.52, 1.33, 0.0), "C", hetatm=True),
        _atom(next_serial + 2, "O", "ACE", "L", 1, (2.53, 1.33, 0.0), "O", hetatm=True),
        # water adjacent on the other side must NOT be hinted
        _atom(next_serial + 3, "O", "HOH", "L", 3, (0.0, -1.4, 0.0), "O", hetatm=True),
    ]
    pdb.write_text("\n".join([*peptide, *cap, "END"]) + "\n", encoding="ascii")

    hints = monomer_symbol_hints(pdb, kind="coordinate")

    assert "ACE" in hints
    assert "HOH" not in hints


# ---------------------------------------------------------------------------
# Explicit-hydrogen geometry adapter
# ---------------------------------------------------------------------------

def test_explicit_hydrogens_retained_and_matched(tmp_path):
    pdb = tmp_path / "methanol.pdb"
    _write_heavy_pdb("CO", pdb, keep_hydrogens=True)
    atoms = _source_atoms(pdb, "L")

    molecule, meta = build_geometry_simple_molecule(pdb, atoms)

    # full source order/count preserved, observed hydrogens re-attached
    assert molecule.GetNumAtoms() == len(atoms)
    assert meta["explicit_hydrogen_count"] == 4
    assert meta["explicit_hydrogen_retained"] == 4
    assert meta["explicit_hydrogen_validation"] == (
        "observed_hydrogens_retained_within_inferred_chemistry"
    )
    conformer = molecule.GetConformer()
    for index, atom in enumerate(atoms):
        position = conformer.GetAtomPosition(index)
        assert [
            round(position.x, 3), round(position.y, 3), round(position.z, 3)
        ] == [round(value, 3) for value in atom["xyz"]]
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "H":
            neighbors = list(atom.GetNeighbors())
            assert len(neighbors) == 1
            assert neighbors[0].GetAtomicNum() > 1
    assert Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule))) == "CO"

    # portfolio level: the geometry engine now admits this input
    report = infer_bond_order_candidates(pdb, "L")
    attempt = _geometry_attempt(report)
    assert attempt is not None and attempt["status"] == "admitted"
    assert report["status"] == "parseable"
    assert report["selected_candidate"] is not None


def test_conflicting_observed_hydrogens_reject_geometry_candidate(tmp_path):
    pdb = tmp_path / "conflict.pdb"
    _write_heavy_pdb("CO", pdb, keep_hydrogens=True)
    # the oxygen now rides two hydrogens while the algorithm infers one
    _relocate_one_carbon_hydrogen_to_oxygen(pdb)
    # strip CONECT so parents resolve by proximity: the relocated hydrogen
    # (0.98 A from O) binds to O and isolates the count-conflict gate
    pdb.write_text(
        "\n".join(
            line
            for line in pdb.read_text(encoding="ascii").splitlines()
            if not line.startswith("CONECT")
        )
        + "\n",
        encoding="ascii",
    )
    atoms = _source_atoms(pdb, "L")

    with pytest.raises(GeometryCandidateError) as raised:
        build_geometry_simple_molecule(pdb, atoms)
    message = str(raised.value).lower()
    assert "hydrogen" in message
    assert "exceed" in message

    report = infer_bond_order_candidates(pdb, "L")
    attempt = _geometry_attempt(report)
    assert attempt is not None and attempt["status"] == "not_admitted"
    assert "hydrogen" in str(attempt.get("error", "")).lower()
    assert report["status"] == "parseable"
    assert report["selected_candidate"] is not None


def _write_partial_hydrogen_pdb(
    path: Path, *, hydrogen_element: str = "H"
) -> None:
    """Methanol where only the O-bound hydrogen is observed (partial H).

    The carbon hydrogens are omitted exactly as X-ray peptides omit most
    riding hydrogens; ``hydrogen_element`` writes the observed atom's element
    column (H or D).
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    assert AllChem.EmbedMolecule(molecule, params) == 0
    conformer = molecule.GetConformer()
    kept = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() > 1:
            kept.append(atom.GetIdx())
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = min(
            (neighbor for neighbor in atom.GetNeighbors()),
            key=lambda neighbor: conformer.GetAtomPosition(
                atom.GetIdx()
            ).Distance(conformer.GetAtomPosition(neighbor.GetIdx())),
        )
        if parent.GetSymbol() == "O":
            kept.append(atom.GetIdx())
    kept_set = set(kept)
    lines = []
    serial_of_index = {}
    for serial, index in enumerate(kept, start=1):
        serial_of_index[index] = serial
        atom = molecule.GetAtomWithIdx(index)
        position = conformer.GetAtomPosition(index)
        element = (
            hydrogen_element
            if atom.GetAtomicNum() == 1
            else atom.GetSymbol().upper()
        )
        lines.append(
            f"HETATM{serial:5d} {element}{serial:<3d} LIG L   1    "
            f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"  1.00  0.00          {element:>2s}"
        )
    for index in kept:
        partners = [
            serial_of_index[neighbor.GetIdx()]
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            if neighbor.GetIdx() in serial_of_index
        ]
        if partners:
            lines.append(
                f"CONECT{serial_of_index[index]:5d}"
                + "".join(f"{partner:5d}" for partner in sorted(partners))
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_partial_hydrogens_are_lower_bound_and_retained(tmp_path):
    pdb = tmp_path / "partial.pdb"
    _write_partial_hydrogen_pdb(pdb)
    atoms = _source_atoms(pdb, "L")

    assert sum(
        1 for atom in atoms if atom["element"].upper() in {"H", "D"}
    ) == 1

    molecule, meta = build_geometry_simple_molecule(pdb, atoms)

    # the single observed O-H is retained as an explicit atom; the carbon's
    # unobserved hydrogens stay implicit for later export
    assert molecule.GetNumAtoms() == len(atoms) == 3
    assert meta["explicit_hydrogen_count"] == 1
    assert meta["explicit_hydrogen_retained"] == 1
    assert meta["explicit_hydrogen_validation"] == (
        "observed_hydrogens_retained_within_inferred_chemistry"
    )
    hydrogens = [
        atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "H"
    ]
    assert len(hydrogens) == 1
    neighbor = hydrogens[0].GetNeighbors()[0]
    assert neighbor.GetSymbol() == "O"
    carbon = next(
        atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "C"
    )
    assert carbon.GetNoImplicit() is True
    assert carbon.GetNumExplicitHs() == 3
    assert Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule))) == "CO"

    report = infer_bond_order_candidates(pdb, "L")
    attempt = _geometry_attempt(report)
    assert attempt is not None and attempt["status"] == "admitted"
    assert report["status"] == "parseable"


def test_deuterium_observed_is_preserved_as_isotope_two(tmp_path):
    pdb = tmp_path / "deuterated.pdb"
    _write_partial_hydrogen_pdb(pdb, hydrogen_element="D")
    atoms = _source_atoms(pdb, "L")

    assert [atom["element"] for atom in atoms if atom["element"] in {"H", "D"}] == ["D"]

    molecule, meta = build_geometry_simple_molecule(pdb, atoms)

    # D is materialized as isotope-2 hydrogen (never silently converted to
    # plain protium, never dropped)
    assert molecule.GetNumAtoms() == len(atoms) == 3
    assert meta["explicit_hydrogen_count"] == 1
    assert meta["explicit_hydrogen_retained"] == 1
    hydrogens = [
        atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 1
    ]
    assert len(hydrogens) == 1
    assert hydrogens[0].GetIsotope() == 2
    assert hydrogens[0].GetNeighbors()[0].GetSymbol() == "O"


def test_heavy_only_outputs_parity(tmp_path):
    pdb = tmp_path / "benzaldehyde.pdb"
    _write_heavy_pdb("O=Cc1ccccc1", pdb)
    atoms = _source_atoms(pdb, "L")

    molecule, meta = build_geometry_simple_molecule(pdb, atoms)

    assert molecule.GetNumAtoms() == len(atoms) == 8
    assert meta["explicit_hydrogen_count"] == 0
    assert meta["explicit_hydrogen_validation"] == "none_observed"
    orders = sorted(
        bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()
    )
    assert orders.count(2.0) == 1
    assert orders.count(1.5) == 6
    assert meta["conect_edge_count"] == molecule.GetNumBonds()
    report = infer_bond_order_candidates(pdb, "L")
    attempt = _geometry_attempt(report)
    assert attempt is not None and attempt["status"] == "admitted"


# ---------------------------------------------------------------------------
# Verified-handoff writer fallback
# ---------------------------------------------------------------------------

def test_handoff_writer_failure_falls_back_to_x1(tmp_path, monkeypatch):
    path = tmp_path / "chain.pdb"
    path.write_text(CCO_CHAIN_PDB)
    real_mol_to_mol2 = conformer_module.mol_to_mol2
    calls = {"count": 0}

    def flaky_writer(mol, output_path=None):
        calls["count"] += 1
        if calls["count"] == 1:
            # writer rejection (not an IO error) on the handoff molecule
            return None, "MOL2 graph normalization failed: injected"
        return real_mol_to_mol2(mol, output_path)

    monkeypatch.setattr(conformer_module, "mol_to_mol2", flaky_writer)

    block, error = _result_first_candidate_to_mol2(
        _graphless_result("CCO"),
        None,
        source_pdb_path=path,
        source_chain_id="L",
    )

    assert error is None and block is not None
    fields = _header_fields(block)
    assert fields["coordinate_tier"] == "X1"
    assert fields["fallback_origin"] == "smiles_only_no_source_graph"
    assert fields["source_handoff_blocked"] == "true"
    assert "verified_handoff_MOL2_writer_failed" in fields[
        "source_handoff_block_reason"
    ]
    assert calls["count"] >= 2


def test_handoff_io_failure_propagates(tmp_path, monkeypatch):
    path = tmp_path / "chain.pdb"
    path.write_text(CCO_CHAIN_PDB)

    def io_writer(mol, output_path=None):
        return None, "MOL2 write failed: injected OSError"

    monkeypatch.setattr(conformer_module, "mol_to_mol2", io_writer)

    block, error = _result_first_candidate_to_mol2(
        _graphless_result("CCO"),
        None,
        source_pdb_path=path,
        source_chain_id="L",
    )

    assert block is None
    assert error is not None
    assert error.startswith("MOL2 write failed")


# ---------------------------------------------------------------------------
# Real run_002 inputs: light extraction only (no exports, no docking)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(REAL_CHAINS))
def test_real_run002_hints_include_connected_caps_and_linkers(case):
    pdb = DEMO_ROOT / case / "peptide.pdb"
    if not pdb.is_file():
        pytest.skip("run_002 demo input not present in this checkout")

    hints = monomer_symbol_hints(pdb, kind="coordinate")

    if case == "1bm2":
        assert "ACE" in hints
        assert {"SLZ", "PTR", "VAL", "ASN", "PRO"} <= set(hints)
    elif case == "1bzh":
        assert "AEA" in hints
        assert {"ASP", "ALA", "GLU", "FLT", "LEU"} <= set(hints)
    elif case == "1bck":
        assert {"DAL", "MLE", "MVA", "BMT", "THR", "SAR", "VAL", "ALA"} == set(
            hints
        )
    else:
        assert "GLY" in hints and "ARG" in hints
    assert not {"HOH", "DOD", "WAT"} & set(hints)


@pytest.mark.parametrize("case", sorted(REAL_CHAINS))
def test_real_run002_adapter_preserves_source_atoms(case):
    pdb = DEMO_ROOT / case / "peptide.pdb"
    if not pdb.is_file():
        pytest.skip("run_002 demo input not present in this checkout")

    atoms = _source_atoms(pdb, REAL_CHAINS[case])
    try:
        molecule, meta = build_geometry_simple_molecule(pdb, atoms)
    except GeometryCandidateError as exc:
        # an honest rejection must name the concrete hydrogen evidence
        assert "hydrogen" in str(exc).lower()
        return

    assert molecule.GetNumAtoms() == len(atoms)
    observed = sum(
        1 for atom in atoms if atom["element"].upper() in {"H", "D"}
    )
    assert meta["explicit_hydrogen_count"] == observed
    assert meta["explicit_hydrogen_retained"] == observed
    if observed:
        assert meta["explicit_hydrogen_validation"] == (
            "observed_hydrogens_retained_within_inferred_chemistry"
        )
    conformer = molecule.GetConformer()
    for index, atom in enumerate(atoms):
        position = conformer.GetAtomPosition(index)
        assert [
            round(position.x, 3), round(position.y, 3), round(position.z, 3)
        ] == [round(value, 3) for value in atom["xyz"]]
