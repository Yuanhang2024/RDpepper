"""Application-demo regressions for receptor PDBQT preparation.

Covers the two defects found while wiring the structure -> MOL2 -> PDBQT ->
Vina demo flow on the 1BCK/1BM2/1BZH receptors:

1. ``autodock_atom_type`` emitted the AutoDock4-only ``HS`` type for thiol
   hydrogens; Vina 1.2.7 rejects it ("Atom type HS is not valid").  Every
   hydrogen bonded to N/O/S -- including thiol S-H -- must be ``HD``.
2. the receptor converter assigned charges without completing the missing
   polar hydrogens.  The obvious remedy, ``Chem.AddHs(...,
   addResidueInfo=True)``, does not terminate on receptors that already
   carry explicit hydrogens (observed on the 1BCK demo receptor), so the
   converter now adds hydrogens without residue info and restores their
   residue labels from the parent heavy atom.
"""

import json
import math
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from cycpep_master.docking.pdbqt_validation import _AUTODOCK4_TYPES
from cycpep_master.docking.receptor_pdbqt import (
    autodock_atom_type,
    pdb_to_receptor_pdbqt,
)

DEMO_CASES = (
    Path(__file__).resolve().parents[2]
    / ".zcode_v710_application_demo"
    / "cases"
)
DEMO_RECEPTORS = {
    case: DEMO_CASES / case / "receptor.pdb"
    for case in ("1bck", "1bm2", "1bzh")
    if (DEMO_CASES / case / "receptor.pdb").is_file()
}
VINA_EXECUTABLE = (
    Path(__file__).resolve().parents[1] / "vina" / "vina_1.2.7_win.exe"
)


def _pdb_atom(serial, name, residue, chain, resnum, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:>4s} {residue:>3s} {chain:1s}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


# A hydrogen-free GLY-CYS fragment: backbone atoms plus a free thiol sulfur
# that still needs its S-H hydrogen.
HEAVY_FRAGMENT_PDB = "".join(
    (
        _pdb_atom(1, "N", "GLY", "A", 1, 0.0, 0.0, 0.0, "N"),
        _pdb_atom(2, "CA", "GLY", "A", 1, 1.4, 0.0, 0.0, "C"),
        _pdb_atom(3, "C", "GLY", "A", 1, 2.1, 1.2, 0.0, "C"),
        _pdb_atom(4, "O", "GLY", "A", 1, 1.6, 2.3, 0.0, "O"),
        _pdb_atom(5, "N", "CYS", "A", 2, 3.4, 1.1, 0.0, "N"),
        _pdb_atom(6, "CA", "CYS", "A", 2, 4.1, 2.3, 0.4, "C"),
        _pdb_atom(7, "C", "CYS", "A", 2, 5.6, 2.2, 0.3, "C"),
        _pdb_atom(8, "O", "CYS", "A", 2, 6.1, 1.1, 0.2, "O"),
        _pdb_atom(9, "CB", "CYS", "A", 2, 4.2, 3.0, 1.7, "C"),
        _pdb_atom(10, "SG", "CYS", "A", 2, 4.3, 4.7, 1.9, "S"),
        "TER\nEND\n",
    )
)

# The same fragment carrying two explicit polar hydrogens (an amide N-H and
# the thiol S-H), mirroring the partially protonated 1BCK demo receptor.
PARTIAL_H_FRAGMENT_PDB = HEAVY_FRAGMENT_PDB.replace(
    "TER\n",
    _pdb_atom(11, "H", "GLY", "A", 1, -0.3, -0.7, 0.6, "H")
    + _pdb_atom(12, "H", "CYS", "A", 2, 4.36, 5.64, 2.01, "H")
    + "TER\n",
)


def _pdbqt_rows(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rows.append(
            {
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21],
                "resnum": int(line[22:26]),
                "xyz": (
                    round(float(line[30:38]), 3),
                    round(float(line[38:46]), 3),
                    round(float(line[46:54]), 3),
                ),
                "charge": float(line[70:76]),
                "type": line[77:79].strip(),
            }
        )
    return rows


def _heavy_coordinates(pdb_text):
    """Map (chain, resnum, atom name) -> rounded input heavy coordinates."""
    rows = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        if element == "H":
            continue
        rows[(line[21], int(line[22:26]), line[12:16].strip())] = (
            round(float(line[30:38]), 3),
            round(float(line[38:46]), 3),
            round(float(line[46:54]), 3),
        )
    return rows


def _input_hydrogen_coordinates(pdb_text):
    return {
        (
            line[21],
            int(line[22:26]),
            (
                round(float(line[30:38]), 3),
                round(float(line[38:46]), 3),
                round(float(line[46:54]), 3),
            ),
        )
        for line in pdb_text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
        and (line[76:78].strip() or line[12:16].strip()[:1]) == "H"
    }


def _assert_common_pdbqt_contract(text, input_pdb_text):
    rows = _pdbqt_rows(text)
    assert rows, "receptor PDBQT has no atom rows"
    types = {row["type"] for row in rows}
    assert types <= set(_AUTODOCK4_TYPES)
    assert "HS" not in types, "Vina 1.2.7 rejects the HS type"
    assert "HD" in types, "no polar hydrogen was emitted"
    assert all(
        math.isfinite(row["charge"]) for row in rows
    ), "non-finite charge emitted"
    assert any(
        abs(row["charge"]) > 1e-6 for row in rows
    ), "all charges are zero"
    # Heavy atoms keep their exact input coordinates.
    heavy_out = {
        (row["chain"], row["resnum"], row["name"]): row["xyz"]
        for row in rows
        if row["type"] not in {"HD", "H"}
    }
    heavy_in = _heavy_coordinates(input_pdb_text)
    assert not (set(heavy_in) - set(heavy_out)), "input heavy atom dropped"
    assert not {
        key
        for key in heavy_in
        if key in heavy_out and heavy_in[key] != heavy_out[key]
    }, "input heavy atom coordinates moved"
    # Nonpolar hydrogens are merged, never emitted.
    assert len(heavy_out) == len(heavy_in)
    # Hydrogen handling is documented in a REMARK for auditability.
    assert any(
        line.startswith("REMARK") and "hydrogens" in line
        for line in text.splitlines()
    )
    return rows


def test_thiol_hydrogen_uses_vina_polar_h_type_not_hs():
    thiol = Chem.AddHs(Chem.MolFromSmiles("CCS"))
    sulfur = thiol.GetAtomWithIdx(2)
    thiol_h = next(
        neighbor
        for neighbor in sulfur.GetNeighbors()
        if neighbor.GetAtomicNum() == 1
    )
    assert autodock_atom_type(thiol_h) == "HD"
    assert autodock_atom_type(sulfur) == "S"

    methanol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    oxygen = methanol.GetAtomWithIdx(1)
    carbon = methanol.GetAtomWithIdx(0)
    oxygen_h = next(
        n for n in oxygen.GetNeighbors() if n.GetAtomicNum() == 1
    )
    carbon_h = next(
        n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 1
    )
    assert autodock_atom_type(oxygen_h) == "HD"
    # Nonpolar hydrogens merge into their parent's charge (empty type).
    assert autodock_atom_type(carbon_h) == ""


def test_converter_adds_polar_hydrogens_and_preserves_heavy_atoms(tmp_path):
    pdb = tmp_path / "receptor.pdb"
    pdbqt = tmp_path / "receptor.pdbqt"
    pdb.write_text(HEAVY_FRAGMENT_PDB, encoding="ascii")

    assert pdb_to_receptor_pdbqt(str(pdb), str(pdbqt)) is None
    text = pdbqt.read_text(encoding="utf-8")
    rows = _assert_common_pdbqt_contract(text, HEAVY_FRAGMENT_PDB)

    hd_rows = [row for row in rows if row["type"] == "HD"]
    assert {row["resnum"] for row in hd_rows} == {1, 2}
    # The free thiol is completed: S-H emitted as HD and the sulfur typed S.
    assert any(row["name"] == "SG" and row["type"] == "S" for row in rows)
    # Hydrogen rows carry their parent residue labels.
    heavy_residues = {
        (row["chain"], row["resnum"], row["resname"])
        for row in rows
        if row["type"] not in {"HD", "H"}
    }
    assert all(
        (row["chain"], row["resnum"], row["resname"]) in heavy_residues
        for row in hd_rows
    )


def test_converter_keeps_existing_polar_hydrogen_coordinates(tmp_path):
    pdb = tmp_path / "receptor.pdb"
    pdbqt = tmp_path / "receptor.pdbqt"
    pdb.write_text(PARTIAL_H_FRAGMENT_PDB, encoding="ascii")

    assert pdb_to_receptor_pdbqt(str(pdb), str(pdbqt)) is None
    text = pdbqt.read_text(encoding="utf-8")
    rows = _assert_common_pdbqt_contract(text, PARTIAL_H_FRAGMENT_PDB)

    hd_rows = [row for row in rows if row["type"] == "HD"]
    emitted = {
        (row["chain"], row["resnum"], row["xyz"]) for row in hd_rows
    }
    # Every input hydrogen keeps its exact coordinates and residue label.
    assert _input_hydrogen_coordinates(PARTIAL_H_FRAGMENT_PDB) <= emitted
    # Missing polar hydrogens are still added beyond the input set.
    assert len(hd_rows) >= 3
    # The thiol sulfur keeps exactly its input hydrogen (no duplicate S-H).
    sg_thiol_h = [
        row
        for row in hd_rows
        if row["resnum"] == 2
        and math.dist(row["xyz"], (4.3, 4.7, 1.9)) < 1.5
    ]
    assert len(sg_thiol_h) == 1


def test_converter_rejects_unsupported_element(tmp_path):
    pdb = tmp_path / "receptor.pdb"
    pdb.write_text(
        _pdb_atom(1, "N", "GLY", "A", 1, 0.0, 0.0, 0.0, "N")
        + _pdb_atom(2, "CA", "GLY", "A", 1, 1.4, 0.0, 0.0, "C")
        + _pdb_atom(3, "C", "GLY", "A", 1, 2.1, 1.2, 0.0, "C")
        + _pdb_atom(4, "O", "GLY", "A", 1, 1.6, 2.3, 0.0, "O")
        + _pdb_atom(5, "SE", "MSE", "A", 2, 3.4, 1.1, 0.0, "SE")
        + "TER\nEND\n",
        encoding="ascii",
    )
    error = pdb_to_receptor_pdbqt(
        str(pdb), str(tmp_path / "receptor.pdbqt")
    )
    assert error is not None
    assert error.startswith("not_supported: receptor atom type is unsupported")


def test_converter_rejects_missing_input(tmp_path):
    error = pdb_to_receptor_pdbqt(
        str(tmp_path / "does-not-exist.pdb"),
        str(tmp_path / "receptor.pdbqt"),
    )
    assert error is not None
    # RDKit raises "Bad input file" for a missing path; the converter must
    # surface it as a typed error instead of emitting an output file.
    assert "failed" in error.lower()
    assert "does-not-exist" in error
    assert not (tmp_path / "receptor.pdbqt").exists()


@pytest.mark.skipif(
    len(DEMO_RECEPTORS) < 3, reason="application-demo receptor inputs absent"
)
@pytest.mark.parametrize("case", ["1bck", "1bm2", "1bzh"])
def test_prepare_receptor_pdbqt_succeeds_for_demo_receptors(tmp_path, case):
    from cycpep_master.application import prepare_receptor_pdbqt

    receptor_pdb = DEMO_RECEPTORS[case]
    output = tmp_path / f"{case}_receptor.pdbqt"
    result = prepare_receptor_pdbqt(str(receptor_pdb), str(output))

    assert result["status"] == "success", result.get("error")
    assert any("pH not assigned" in warning for warning in result["data"]["warnings"])
    text = output.read_text(encoding="utf-8")
    _assert_common_pdbqt_contract(text, receptor_pdb.read_text(encoding="utf-8"))
    # The 1BCK input carries 285 polar hydrogens; the other two carry none,
    # so polar-H counts differ per case but every receptor must gain some.
    rows = _pdbqt_rows(text)
    minimum_polar_h = {"1bck": 300, "1bm2": 150, "1bzh": 500}
    assert (
        sum(1 for row in rows if row["type"] == "HD") >= minimum_polar_h[case]
    )


def _vina_score_only(receptor, ligand, box):
    return subprocess.run(
        [
            str(VINA_EXECUTABLE),
            "--score_only",
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand),
            "--center_x",
            str(box["center_A"][0]),
            "--center_y",
            str(box["center_A"][1]),
            "--center_z",
            str(box["center_A"][2]),
            "--size_x",
            str(box["size_A"][0]),
            "--size_y",
            str(box["size_A"][1]),
            "--size_z",
            str(box["size_A"][2]),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.skipif(
    not VINA_EXECUTABLE.is_file(), reason="bundled Vina executable absent"
)
@pytest.mark.parametrize("case", ["1bck", "1bzh"])
def test_vina_score_only_accepts_demo_receptor_pdbqt(tmp_path, case):
    if case not in DEMO_RECEPTORS:
        pytest.skip(f"{case} demo receptor absent")
    receptor_pdb = DEMO_RECEPTORS[case]
    ligand = DEMO_CASES / case / "ligand_displaced.pdbqt"
    setup = DEMO_CASES / case / "docking_setup.json"
    if not (ligand.is_file() and setup.is_file()):
        pytest.skip(f"{case} demo case has no ligand/box setup")
    box = json.loads(setup.read_text(encoding="utf-8"))

    output = tmp_path / f"{case}_receptor.pdbqt"
    assert pdb_to_receptor_pdbqt(str(receptor_pdb), str(output)) is None
    result = _vina_score_only(output, ligand, box)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Estimated Free Energy of Binding" in result.stdout
    assert "not valid" not in (result.stdout + result.stderr).lower()
