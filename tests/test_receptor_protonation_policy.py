"""Explicit-pH receptor protonation policy for ``pdb_to_receptor_pdbqt``.

``ph=None`` must stay byte-compatible with the historical converter
(RDKit template valence hydrogens + the "pH not assigned" warning).
A numeric ``ph`` applies the residue-state template policy:

* pH 8: Asp/Glu (and C/OXT) carboxylates are anionic -- no hydroxyl HD;
* pH 8: Lys NZ carries three HD (ammonium) and Arg keeps the full
  guanidinium donor set (NE 1H, NH1 2H, NH2 2H);
* His is never auto-assigned: explicit input hydrogens are preserved as
  the user state, and a hydrogen-free His is reported as an
  UNDETERMINED microstate instead of a claimed tautomer;
* every input heavy atom keeps its exact coordinates and identity;
* a pH request fails closed when the policy backend is unavailable --
  it must never silently emit the unprotonated neutral default.
"""

import math
from pathlib import Path

import pytest
from rdkit import Chem

from cycpep_master.docking import receptor_pdbqt
from cycpep_master.docking.pdbqt_validation import _AUTODOCK4_TYPES
from cycpep_master.docking.receptor_pdbqt import (
    _apply_residue_template_protonation,
    _enforce_protonation_state,
    pdb_to_receptor_pdbqt,
)

DEMO_RECEPTORS = {
    case: (
        Path(__file__).resolve().parents[2]
        / ".zcode_v710_application_demo"
        / "current_source_run_002"
        / case
        / "receptor.pdb"
    )
    for case in ("1bm2", "1bck")
}

# Complete, realistic residues lifted verbatim from the demo receptors.
# Fragment key -> (case, residue number, PDB residue name).
FRAGMENT_RESIDUES = {
    "ASP": ("1bm2", 80, "ASP"),
    "GLU": ("1bm2", 71, "GLU"),
    "LYS": ("1bm2", 56, "LYS"),
    "ARG": ("1bm2", 67, "ARG"),
    "HIS": ("1bm2", 58, "HIS"),
    "CTERM_GLU": ("1bck", 165, "GLU"),  # carries OXT
}


def _residue_lines(case, residue_number, residue_name):
    path = DEMO_RECEPTORS[case]
    if not path.is_file():
        pytest.skip(f"{case} demo receptor absent")
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith(("ATOM", "HETATM"))
        and line[21] == "A"
        and int(line[22:26]) == residue_number
        and line[17:20].strip() == residue_name
    ]
    if not lines:
        pytest.skip(f"residue {residue_name}{residue_number} absent")
    return lines


def _fragment_pdb(residues):
    lines = []
    serial = 0
    for fragment_key in residues:
        case, number, residue_name = FRAGMENT_RESIDUES[fragment_key]
        for line in _residue_lines(case, number, residue_name):
            serial += 1
            lines.append(
                f"ATOM  {serial:5d} {line[12:16]} {line[17:20]} "
                f"{line[21]}{line[22:26]}    {line[30:54]}  1.00 20.00"
                f"          {line[76:78]:>2s}\n"
            )
    return "".join(lines) + "TER\nEND\n"


def _pdbqt_rows(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rows.append(
            {
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
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


def _input_heavy(pdb_text):
    rows = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if (line[76:78].strip() or line[12:16].strip()[:1]) == "H":
            continue
        rows[(int(line[22:26]), line[12:16].strip())] = (
            round(float(line[30:38]), 3),
            round(float(line[38:46]), 3),
            round(float(line[46:54]), 3),
        )
    return rows


def _hd_counts_by_parent(rows, cutoff=1.3):
    """Map (resnum, heavy atom name) -> number of HD rows bonded to it."""
    heavy = [row for row in rows if row["type"] not in {"HD", "H"}]
    counts = {}
    for row in rows:
        if row["type"] != "HD":
            continue
        nearest = min(
            heavy, key=lambda other: math.dist(row["xyz"], other["xyz"])
        )
        if math.dist(row["xyz"], nearest["xyz"]) <= cutoff:
            key = (nearest["resnum"], nearest["name"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def _convert(tmp_path, pdb_text, name, *, ph=...):
    pdb = tmp_path / f"{name}.pdb"
    pdbqt = tmp_path / f"{name}.pdbqt"
    pdb.write_text(pdb_text, encoding="ascii")
    kwargs = {} if ph is ... else {"ph": ph}
    error = pdb_to_receptor_pdbqt(str(pdb), str(pdbqt), **kwargs)
    assert error is None, error
    rows = _pdbqt_rows(pdbqt.read_text(encoding="utf-8"))
    remarks = [
        line for line in pdbqt.read_text(encoding="utf-8").splitlines()
        if line.startswith("REMARK")
    ]
    return rows, remarks


def test_default_mode_keeps_historical_protonation_and_warning(tmp_path):
    rows, remarks = _convert(tmp_path, _fragment_pdb(["ASP", "GLU"]), "acid")
    counts = _hd_counts_by_parent(rows)
    # Historical neutral template states: one hydroxyl HD per carboxylate.
    assert counts.get((80, "OD2")) == 1
    assert counts.get((71, "OE2")) == 1
    assert any("pH not assigned" in line for line in remarks)
    assert not any("protonation policy:" in line for line in remarks)


def test_ph8_deprotonates_asp_and_glu_carboxylates(tmp_path):
    rows, remarks = _convert(
        tmp_path, _fragment_pdb(["ASP", "GLU"]), "acid", ph=8.0
    )
    counts = _hd_counts_by_parent(rows)
    for key in ((80, "OD1"), (80, "OD2"), (71, "OE1"), (71, "OE2")):
        assert counts.get(key, 0) == 0, f"carboxylate {key} still protonated"
    # Both oxygens remain acceptors.
    for row in rows:
        if row["type"] not in {"HD", "H"} and row["name"] in {
            "OD1", "OD2", "OE1", "OE2"
        }:
            assert row["type"] == "OA"
    assert any("protonation policy: residue_template" in line for line in remarks)
    assert any("per-residue pKa NOT computed" in line for line in remarks)
    assert any("2 Asp/Glu carboxylate(s) anionic" in line for line in remarks)
    assert not any("pH not assigned" in line for line in remarks)
    assert all(math.isfinite(row["charge"]) for row in rows)


def test_ph8_deprotonates_terminal_carboxylate(tmp_path):
    rows, _ = _convert(
        tmp_path, _fragment_pdb(["CTERM_GLU"]), "cterm", ph=8.0
    )
    counts = _hd_counts_by_parent(rows)
    assert counts.get((165, "O"), 0) == 0
    assert counts.get((165, "OXT"), 0) == 0
    # The side-chain carboxylate of the same residue is also anionic.
    assert counts.get((165, "OE2"), 0) == 0


def test_ph8_keeps_lys_and_arg_cationic(tmp_path):
    rows, _ = _convert(tmp_path, _fragment_pdb(["LYS", "ARG"]), "base", ph=8.0)
    counts = _hd_counts_by_parent(rows)
    assert counts.get((56, "NZ")) == 3, "Lys ammonium must carry 3 polar H"
    assert counts.get((67, "NE")) == 1
    assert counts.get((67, "NH1")) == 2
    assert counts.get((67, "NH2")) == 2


def test_default_mode_leaves_lys_and_arg_neutral(tmp_path):
    rows, _ = _convert(tmp_path, _fragment_pdb(["LYS", "ARG"]), "base")
    counts = _hd_counts_by_parent(rows)
    # Documents the historical default the pH policy corrects.
    assert counts.get((56, "NZ")) == 2
    assert counts.get((67, "NH2")) == 1


def test_arg_guanidinium_net_formal_charge_is_plus_one(tmp_path):
    """Regression: the +1 lives only on the C=N nitrogen, never +3."""
    pdb = tmp_path / "arg.pdb"
    pdb.write_text(_fragment_pdb(["ARG"]), encoding="ascii")
    molecule = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        pass
    edited, _remarks, sites = _apply_residue_template_protonation(
        molecule, 8.0
    )

    def side_chain(mol):
        return {
            atom.GetPDBResidueInfo().GetName().strip(): atom
            for atom in mol.GetAtoms()
            if atom.GetPDBResidueInfo() is not None
            and atom.GetPDBResidueInfo().GetName().strip()
            in {"NE", "NH1", "NH2", "CZ"}
        }

    def net_charge(mol):
        return sum(atom.GetFormalCharge() for atom in side_chain(mol).values())

    def polar_hydrogens(mol):
        return {
            name: sum(
                1
                for neighbor in atom.GetNeighbors()
                if neighbor.GetAtomicNum() == 1
            )
            for name, atom in side_chain(mol).items()
            if name != "CZ"
        }

    assert net_charge(edited) == 1, "guanidinium must be net +1, not +3"
    final = _enforce_protonation_state(
        Chem.AddHs(edited, addCoords=True), sites
    )
    assert net_charge(final) == 1
    assert polar_hydrogens(final) == {"NE": 1, "NH1": 2, "NH2": 2}
    # Gasteiger conserves charge over the whole molecule: the emitted
    # partial charges sum to the molecule's formal charge (guanidinium +1
    # plus whatever template termini the isolated fragment carries).
    expected_total = sum(atom.GetFormalCharge() for atom in final.GetAtoms())
    rows, _remarks = _convert(tmp_path, _fragment_pdb(["ARG"]), "arg_q", ph=8.0)
    emitted = sum(row["charge"] for row in rows)
    assert emitted == pytest.approx(expected_total, abs=0.02), (
        expected_total,
        emitted,
    )


def test_his_without_input_hydrogens_is_marked_undetermined(tmp_path):
    rows, remarks = _convert(
        tmp_path, _fragment_pdb(["HIS"]), "his", ph=8.0
    )
    assert any(
        "protonation HIS" in line and "UNDETERMINED" in line
        for line in remarks
    )
    counts = _hd_counts_by_parent(rows)
    # The deterministic RDKit template tautomer (NE2-H) is retained but
    # explicitly not claimed as the determined microstate.
    assert counts.get((58, "NE2")) == 1
    assert counts.get((58, "ND1"), 0) == 0


def test_his_explicit_input_hydrogens_are_user_state(tmp_path):
    fragment = _fragment_pdb(["HIS"])
    lines = fragment.splitlines(keepends=True)
    positions = {}
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            positions[line[12:16].strip()] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )

    def hydrogen_line(serial, name, nitrogen, neighbors):
        nx, ny, nz = positions[nitrogen]
        total = [0.0, 0.0, 0.0]
        for other in neighbors:
            ox, oy, oz = positions[other]
            vx, vy, vz = ox - nx, oy - ny, oz - nz
            length = math.sqrt(vx * vx + vy * vy + vz * vz)
            for i, component in enumerate((vx, vy, vz)):
                total[i] += component / length
        magnitude = math.sqrt(sum(c * c for c in total))
        x, y, z = (
            nx - total[0] / magnitude,
            ny - total[1] / magnitude,
            nz - total[2] / magnitude,
        )
        return (
            f"ATOM  {serial:5d} {name:>4s} HIS A  58    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           H\n"
        )

    user = "".join(lines[:-2])  # drop TER/END
    user += hydrogen_line(101, "HD1", "ND1", ("CG", "CE1"))
    user += hydrogen_line(102, "HE2", "NE2", ("CD2", "CE1"))
    user += "TER\nEND\n"

    rows, remarks = _convert(tmp_path, user, "his_user", ph=8.0)
    counts = _hd_counts_by_parent(rows)
    assert counts.get((58, "ND1")) == 1
    assert counts.get((58, "NE2")) == 1
    # The exact user-state coordinates are preserved, not regenerated.
    hd_rows = [row for row in rows if row["type"] == "HD"]
    for name in ("ND1", "NE2"):
        nitrogen = next(row for row in rows if row["name"] == name)
        closest = min(
            hd_rows, key=lambda h: math.dist(h["xyz"], nitrogen["xyz"])
        )
        distance = math.dist(closest["xyz"], nitrogen["xyz"])
        assert distance < 1.3
    assert any(
        "protonation HIS" in line and "user state" in line
        for line in remarks
    )


@pytest.mark.skipif(
    not DEMO_RECEPTORS["1bm2"].is_file(), reason="1bm2 demo receptor absent"
)
def test_ph8_preserves_every_heavy_atom_exactly(tmp_path):
    pdb_text = DEMO_RECEPTORS["1bm2"].read_text(encoding="utf-8")
    default_rows, _ = _convert(tmp_path, pdb_text, "full_default")
    rows, _ = _convert(tmp_path, pdb_text, "full_ph8", ph=8.0)

    def heavy(rows):
        return [
            (row["resnum"], row["resname"], row["name"], row["xyz"])
            for row in rows
            if row["type"] not in {"HD", "H"}
        ]

    heavy_default, heavy_ph8 = heavy(default_rows), heavy(rows)
    assert heavy_ph8 == heavy_default, "heavy-atom identity or order drifted"
    expected = _input_heavy(pdb_text)
    assert len(heavy_ph8) == len(expected)
    for resnum, _resname, name, xyz in heavy_ph8:
        assert (resnum, name) in expected
        assert xyz == expected[(resnum, name)], f"{name}{resnum} moved"
    types = {row["type"] for row in rows}
    assert types <= set(_AUTODOCK4_TYPES)
    assert "HS" not in types
    assert all(math.isfinite(row["charge"]) for row in rows)


def test_ph_request_fails_closed_when_backend_unavailable(
    tmp_path, monkeypatch
):
    def unavailable():
        raise ImportError("simulated backend outage")

    monkeypatch.setattr(
        receptor_pdbqt, "_load_protonation_policy_backend", unavailable
    )
    pdb = tmp_path / "receptor.pdb"
    pdb.write_text(_fragment_pdb(["ASP"]), encoding="ascii")
    ph8_output = tmp_path / "ph8.pdbqt"
    error = pdb_to_receptor_pdbqt(
        str(pdb), str(ph8_output), ph=8.0
    )
    assert error is not None
    assert "unavailable" in error
    assert "refusing" in error
    assert not ph8_output.exists()
    # The default path never consults the backend and keeps working.
    assert (
        pdb_to_receptor_pdbqt(str(pdb), str(tmp_path / "default.pdbqt"))
        is None
    )


@pytest.mark.parametrize("bad_ph", [float("nan"), 0.0, -1.0, 14.5])
def test_invalid_ph_is_rejected(tmp_path, bad_ph):
    pdb = tmp_path / "receptor.pdb"
    pdb.write_text(_fragment_pdb(["ASP"]), encoding="ascii")
    output = tmp_path / "out.pdbqt"
    error = pdb_to_receptor_pdbqt(str(pdb), str(output), ph=bad_ph)
    assert error is not None
    assert error.startswith("invalid pH")
    assert not output.exists()


@pytest.mark.parametrize("high_ph", [10.6, 12.0, 14.0])
def test_ph_above_supported_range_fails_closed(tmp_path, high_ph):
    """No pretending: neutral Lys/Arg transitions are not assigned."""
    pdb = tmp_path / "receptor.pdb"
    pdb.write_text(_fragment_pdb(["LYS", "ARG"]), encoding="ascii")
    output = tmp_path / "out.pdbqt"
    error = pdb_to_receptor_pdbqt(str(pdb), str(output), ph=high_ph)
    assert error is not None
    assert error.startswith("not_supported: pH")
    assert "10.5" in error
    assert not output.exists()
