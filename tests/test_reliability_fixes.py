"""Focused reliability-fix regressions.

Covers: notation identity audits (BILN/HELM/MAP), template PDB/SMILES graph
correspondence, Vina output freshness, and PDBQT torsion-tree hardening.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.chemical_audit import audit_biln, audit_helm, audit_map
from cycpep_master.docking import build_template_library
from cycpep_master.docking import pdbqt_validation
from cycpep_master.docking import vina


# ── 1. BILN / HELM / MAP identity audits ───────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "C(1,3)-A-A-A-C(1,3)",
        "P-E-P-T-I-D-E",
        "[meL]-A-A",
        "A-G.K(1,3)-E.G-L-E-E(1,3)",
    ],
)
def test_audit_biln_accepts_valid_syntax(value):
    assert audit_biln(value).accepted


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ("A--C", "MALFORMED_BILN_SEQUENCE"),
        ("A-C.", "MALFORMED_BILN_SEQUENCE"),
        ("X-Y-Z", "UNKNOWN_MONOMER"),
        ("A(1,3)-C", "UNPAIRED_RING_CLOSURE"),
        ("A(1,4)-C(1,3)", "MALFORMED_BILN_CROSSLINK"),
        ("A(1,3)-C(1,3)-C(2,3)-C(2,3)", "INVALID_BILN_PORT"),
    ],
)
def test_audit_biln_rejects_invalid_identity(value, expected_code):
    result = audit_biln(value)
    assert not result.accepted
    assert expected_code in result.warning_codes


@pytest.mark.parametrize(
    "value",
    [
        "PEPTIDE1{A.G.V}$$$$V2.0",
        "PEPTIDE1{[meL].A.A}$$$$V2.0",
        "PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$V2.0",
        "PEPTIDE1{A.G.V}|PEPTIDE2{A.G.V}$$$$V2.0",
    ],
)
def test_audit_helm_accepts_valid_syntax(value):
    assert audit_helm(value).accepted


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ("PEPTIDE1{}$$$$V2.0", "MALFORMED_HELM_POLYMER"),
        ("PEPTIDE1{A.G.V}$PEPTIDE9,PEPTIDE1,1:R1-3:R2$$$V2.0", "HELM_UNKNOWN_POLYMER"),
        ("PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,1:R1-99:R2$$$V2.0", "HELM_ENDPOINT_OUT_OF_RANGE"),
        ("PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,3:R1-1:R2$$$V2.0", "HELM_ENDPOINT_OUT_OF_RANGE"),
        ("PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,1:R4-3:R2$$$V2.0", "MALFORMED_RING_CLOSURE"),
        ("PEPTIDE1{X.Y}$$$$V2.0", "UNKNOWN_MONOMER"),
        ("PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,1:R1-$$$V2.0", "MALFORMED_RING_CLOSURE"),
        ("PEPTIDE1{A.G.V}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$C$V2.0", "MALFORMED_HELM_SECTIONS"),
    ],
)
def test_audit_helm_rejects_invalid_identity(value, expected_code):
    result = audit_helm(value)
    assert not result.accepted
    assert expected_code in result.warning_codes


@pytest.mark.parametrize(
    "value",
    [
        "AGV{cyc:1:R1-3:R2}",
        "AC{br}CA{cyc:2:R3-3:R3}",
        "A{nnr:meL}V{cyc:1:R1-3:R2}",
        "{nt:ACE}AG{ct:NH2}",
    ],
)
def test_audit_map_accepts_valid_syntax(value):
    assert audit_map(value).accepted


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ("A{nnr:DefinitelyUnknown}G", "UNKNOWN_MONOMER"),
        ("AGV{cyc:1:R1-", "MALFORMED_RING_CLOSURE"),
        ("AGV{cyc:1:R1-99:R2}", "INVALID_MAP_SEMANTICS"),
        ("AGV{cyc:1:R1-3:R2}{cyc:1:R1-3:R2}", "DUPLICATE_ATTACHMENT_ENDPOINT"),
        ("AGV{cyc:1:R1-3:R2", "UNBALANCED_MAP_BRACES"),
    ],
)
def test_audit_map_rejects_invalid_identity(value, expected_code):
    result = audit_map(value)
    assert not result.accepted
    assert expected_code in result.warning_codes


# ── 2. Template PDB / SMILES graph correspondence ──────────────────────────


def _write_template_pdb(tmp_path, name, smiles, seed=1):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    path = tmp_path / name
    Chem.MolToPDBFile(mol, str(path))
    return path


def test_load_template_mol_accepts_matching_pdb_and_smiles(tmp_path):
    path = _write_template_pdb(tmp_path, "match.pdb", "N[C@@H](C)C(=O)O")
    mol, error = build_template_library._load_template_mol(
        str(path), "N[C@@H](C)C(=O)O"
    )
    assert error is None
    assert mol is not None
    assert mol.GetNumHeavyAtoms() == 6
    assert mol.GetNumConformers() == 1


def test_load_template_mol_rejects_heavy_atom_count_mismatch(tmp_path):
    path = _write_template_pdb(tmp_path, "butane.pdb", "CCCC")
    mol, error = build_template_library._load_template_mol(str(path), "CC")
    assert mol is None
    assert "heavy-atom count" in error


def test_load_template_mol_rejects_unrelated_graph_with_same_count(tmp_path):
    path = _write_template_pdb(tmp_path, "propane.pdb", "CCC")
    mol, error = build_template_library._load_template_mol(str(path), "CCO")
    assert mol is None
    assert "graph does not match" in error


def test_load_template_mol_rejects_ring_chain_mismatch(tmp_path):
    path = _write_template_pdb(tmp_path, "cyclopropane.pdb", "C1CC1")
    mol, error = build_template_library._load_template_mol(str(path), "CCC")
    assert mol is None
    assert "graph does not match" in error


def test_load_template_mol_fallback_requires_proven_correspondence(
    tmp_path, monkeypatch
):
    matching = _write_template_pdb(tmp_path, "match.pdb", "N[C@@H](C)C(=O)O")
    unrelated = _write_template_pdb(tmp_path, "unrelated.pdb", "CCC")

    def broken(*_args, **_kwargs):
        raise RuntimeError("forced bond-order failure")

    monkeypatch.setattr(
        build_template_library.AllChem, "AssignBondOrdersFromTemplate", broken
    )
    mol, error = build_template_library._load_template_mol(
        str(matching), "N[C@@H](C)C(=O)O"
    )
    assert error is None
    assert mol is not None
    mol, error = build_template_library._load_template_mol(str(unrelated), "CCO")
    assert mol is None
    assert "graph does not match" in error


# ── 3. Vina output freshness and parseability ──────────────────────────────


_VINA_STDOUT = (
    "mode | affinity | dist from best mode\n"
    "-----+----------+--------------------\n"
    "1 -7.25 0.0 0.0\n"
)
_VINA_OUTPUT = (
    "MODEL 1\n"
    "REMARK VINA RESULT 1 -7.25 0 0\n"
    "ROOT\n"
    "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
    "ENDROOT\nTORSDOF 0\n"
    "ENDMDL\n"
)
_VINA_LIGAND = (
    "ROOT\n"
    "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
    "ENDROOT\nTORSDOF 0\n"
)


def _write_vina_inputs(directory):
    ligand = directory / "ligand.pdbqt"
    receptor = directory / "receptor.pdbqt"
    ligand.write_text(_VINA_LIGAND, encoding="utf-8")
    receptor.write_text(_VINA_LIGAND, encoding="utf-8")
    return ligand, receptor


def _fake_vina_run(writer):
    def run(_command, **_kwargs):
        writer()
        return SimpleNamespace(returncode=0, stdout=_VINA_STDOUT, stderr="")

    return run


def _run_vina(output_path, run_process):
    ligand, receptor = _write_vina_inputs(output_path.parent)
    return vina.run_vina(
        ligand_pdbqt=str(ligand),
        receptor_pdbqt=str(receptor),
        center=(0.0, 0.0, 0.0),
        box_size=(20.0, 20.0, 20.0),
        output_pdbqt=str(output_path),
        find_executable=lambda: "vina.exe",
        run_process=run_process,
    )


def test_run_vina_accepts_new_parseable_output(tmp_path):
    output = tmp_path / "out.pdbqt"
    affinity, error = _run_vina(
        output,
        _fake_vina_run(lambda: output.write_text(_VINA_OUTPUT, encoding="utf-8")),
    )
    assert (affinity, error) == (-7.25, None)


def test_run_vina_rejects_missing_output(tmp_path):
    output = tmp_path / "out.pdbqt"
    affinity, error = _run_vina(output, _fake_vina_run(lambda: None))
    assert affinity is None
    assert "produced no output file" in error


def test_run_vina_rejects_stale_output(tmp_path):
    output = tmp_path / "out.pdbqt"
    output.write_text(_VINA_OUTPUT, encoding="utf-8")
    affinity, error = _run_vina(output, _fake_vina_run(lambda: None))
    assert affinity is None
    assert "produced no output file" in error


def test_run_vina_clears_stale_output_when_executable_is_missing(tmp_path):
    ligand, receptor = _write_vina_inputs(tmp_path)
    output = tmp_path / "out.pdbqt"
    output.write_text(_VINA_OUTPUT, encoding="utf-8")

    affinity, error = vina.run_vina(
        str(ligand),
        str(receptor),
        (0.0, 0.0, 0.0),
        (20.0, 20.0, 20.0),
        str(output),
        find_executable=lambda: None,
    )

    assert affinity is None
    assert "executable not found" in error
    assert not output.exists()


def test_run_vina_rejects_output_alias_without_deleting_ligand(tmp_path):
    ligand, receptor = _write_vina_inputs(tmp_path)
    original = ligand.read_bytes()

    affinity, error = vina.run_vina(
        str(ligand),
        str(receptor),
        (0.0, 0.0, 0.0),
        (20.0, 20.0, 20.0),
        str(ligand),
        find_executable=lambda: pytest.fail("Vina discovery must not run"),
    )

    assert affinity is None
    assert "alias the ligand input" in error
    assert ligand.read_bytes() == original


def test_run_vina_rejects_hardlink_output_without_deleting_ligand(tmp_path):
    ligand, receptor = _write_vina_inputs(tmp_path)
    output = tmp_path / "output-alias.pdbqt"
    try:
        os.link(ligand, output)
    except (AttributeError, OSError) as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    original = ligand.read_bytes()

    affinity, error = vina.run_vina(
        str(ligand),
        str(receptor),
        (0.0, 0.0, 0.0),
        (20.0, 20.0, 20.0),
        str(output),
        find_executable=lambda: pytest.fail("Vina discovery must not run"),
    )

    assert affinity is None
    assert "alias the ligand input" in error
    assert ligand.read_bytes() == original
    assert output.read_bytes() == original


def test_run_vina_rejects_empty_output(tmp_path):
    output = tmp_path / "out.pdbqt"
    affinity, error = _run_vina(
        output, _fake_vina_run(lambda: output.write_text("", encoding="utf-8"))
    )
    assert affinity is None
    assert "empty" in error


def test_run_vina_rejects_unparseable_output(tmp_path):
    output = tmp_path / "out.pdbqt"
    affinity, error = _run_vina(
        output,
        _fake_vina_run(
            lambda: output.write_text("MODEL 1\nENDMDL\n", encoding="utf-8")
        ),
    )
    assert affinity is None
    assert "not minimally parseable" in error
    assert not output.exists()


def test_run_vina_removes_output_after_ligand_invariant_failure(tmp_path):
    output = tmp_path / "out.pdbqt"
    changed = _VINA_OUTPUT.replace("0.000 C\n", "0.125 N\n")
    affinity, error = _run_vina(
        output,
        _fake_vina_run(lambda: output.write_text(changed, encoding="utf-8")),
    )

    assert affinity is None
    assert "changed ligand atom invariant" in error
    assert not output.exists()


@pytest.mark.parametrize("affinity_value", [float("nan"), float("inf"), float("-inf")])
def test_run_vina_rejects_nonfinite_affinity(tmp_path, affinity_value):
    output = tmp_path / "out.pdbqt"
    ligand, receptor = _write_vina_inputs(tmp_path)
    affinity, error = vina.run_vina(
        ligand_pdbqt=str(ligand),
        receptor_pdbqt=str(receptor),
        center=(0.0, 0.0, 0.0),
        box_size=(20.0, 20.0, 20.0),
        output_pdbqt=str(output),
        find_executable=lambda: "vina.exe",
        run_process=_fake_vina_run(
            lambda: output.write_text(_VINA_OUTPUT, encoding="utf-8")
        ),
        parse_affinity=lambda _stdout: affinity_value,
    )
    assert affinity is None
    assert "affinity" in error


def test_run_vina_rejects_truncated_ligand_output(tmp_path):
    ligand, receptor = _write_vina_inputs(tmp_path)
    ligand.write_text(
        _VINA_LIGAND.replace(
            "ENDROOT",
            "ATOM      2  N   LIG A   1       1.000   0.000   0.000  1.00  0.00    -0.100 NA\nENDROOT",
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.pdbqt"
    affinity, error = vina.run_vina(
        str(ligand),
        str(receptor),
        (0.0, 0.0, 0.0),
        (20.0, 20.0, 20.0),
        str(output),
        find_executable=lambda: "vina.exe",
        run_process=_fake_vina_run(
            lambda: output.write_text(_VINA_OUTPUT, encoding="utf-8")
        ),
    )
    assert affinity is None
    assert "atom-count mismatch" in error


def test_run_vina_rejects_ligand_type_or_charge_drift(tmp_path):
    ligand, receptor = _write_vina_inputs(tmp_path)
    output = tmp_path / "out.pdbqt"
    changed = _VINA_OUTPUT.replace("0.000 C\n", "0.125 N\n")
    affinity, error = vina.run_vina(
        str(ligand),
        str(receptor),
        (0.0, 0.0, 0.0),
        (20.0, 20.0, 20.0),
        str(output),
        find_executable=lambda: "vina.exe",
        run_process=_fake_vina_run(
            lambda: output.write_text(changed, encoding="utf-8")
        ),
    )
    assert affinity is None
    assert "changed ligand atom invariant" in error


@pytest.mark.parametrize(
    ("center", "box_size", "exhaustiveness", "num_modes", "message"),
    [
        ((float("nan"), 0, 0), (20, 20, 20), 8, 1, "center"),
        ((0, 0, 0), (20, 0, 20), 8, 1, "box size"),
        ((0, 0, 0), (20, 20, float("inf")), 8, 1, "box size"),
        ((0, 0, 0), (20, 20, 20), 0, 1, "search parameters"),
        ((0, 0, 0), (20, 20, 20), 8, 0, "search parameters"),
    ],
)
def test_run_vina_rejects_nonphysical_parameters(
    tmp_path, center, box_size, exhaustiveness, num_modes, message
):
    output = tmp_path / "out.pdbqt"
    output.write_text(_VINA_OUTPUT, encoding="utf-8")
    affinity, error = vina.run_vina(
        ligand_pdbqt="ligand.pdbqt",
        receptor_pdbqt="receptor.pdbqt",
        center=center,
        box_size=box_size,
        output_pdbqt=str(output),
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        find_executable=lambda: "vina.exe",
        run_process=lambda *_args, **_kwargs: pytest.fail("Vina must not run"),
    )
    assert affinity is None
    assert message in error
    assert not output.exists()


# ── 4. PDBQT torsion-tree hardening ────────────────────────────────────────


@pytest.fixture(scope="module")
def ethanol_pdbqt(tmp_path_factory):
    pytest.importorskip("meeko")
    from cycpep_master.docking import vina_wrapper

    output = tmp_path_factory.mktemp("pdbqt") / "ethanol.pdbqt"
    error = vina_wrapper.smiles_to_ligand_pdbqt(
        "CCO",
        str(output),
        num_confs=1,
        random_seed=42,
        protonate=False,
    )
    assert error is None
    return output.read_text(encoding="utf-8")


def _oxygen_atom_line(pdbqt_text):
    return next(
        line
        for line in pdbqt_text.splitlines()
        if line.startswith("ATOM") and line.split()[-1] == "OA"
    )


def test_validate_pdbqt_accepts_valid_meeko_output(ethanol_pdbqt):
    audit = pdbqt_validation.validate_pdbqt_torsion_tree(ethanol_pdbqt)
    assert audit == {"atom_count": 4, "branch_count": 1, "torsdof": 1}


@pytest.mark.parametrize(
    ("charge", "message"),
    [
        ("nan", "non-finite charge"),
        ("inf", "non-finite charge"),
        ("abc", "non-numeric charge"),
    ],
)
def test_validate_pdbqt_rejects_bad_charges(ethanol_pdbqt, charge, message):
    line = _oxygen_atom_line(ethanol_pdbqt)
    charge_token = line.split()[-2]
    mutated = ethanol_pdbqt.replace(charge_token, charge)
    with pytest.raises(RuntimeError, match=message):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


@pytest.mark.parametrize(
    ("type_token", "message"),
    [
        ("Zz", "invalid AutoDock atom type"),
        ("0.000", "invalid AutoDock atom type"),
    ],
)
def test_validate_pdbqt_rejects_invalid_types(ethanol_pdbqt, type_token, message):
    line = _oxygen_atom_line(ethanol_pdbqt)
    mutated = ethanol_pdbqt.replace(line, line.replace(" OA", f" {type_token}"))
    with pytest.raises(RuntimeError, match=message):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


def test_validate_pdbqt_rejects_missing_type(ethanol_pdbqt):
    line = _oxygen_atom_line(ethanol_pdbqt)
    mutated = ethanol_pdbqt.replace(line, line.replace(" OA", ""))
    with pytest.raises(RuntimeError, match="missing AutoDock atom type"):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("BRANCH   2   2", "self-referential BRANCH"),
        ("BRANCH   2   3\nBRANCH   2   3", "duplicate BRANCH"),
        ("BRANCH   9   3", "attaches to unknown atom serial"),
    ],
)
def test_validate_pdbqt_rejects_bad_branch_records(
    ethanol_pdbqt, mutation, message
):
    mutated = ethanol_pdbqt.replace("BRANCH   2   3", mutation, 1)
    with pytest.raises(RuntimeError, match=message):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


def test_validate_pdbqt_rejects_non_numeric_branch(ethanol_pdbqt):
    mutated = ethanol_pdbqt.replace("BRANCH   2   3", "BRANCH foo 3", 1)
    with pytest.raises(RuntimeError, match="malformed BRANCH"):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


def test_validate_pdbqt_rejects_wrong_first_branch_atom(ethanol_pdbqt):
    mutated = ethanol_pdbqt.replace("ATOM      3", "ATOM      9", 1)
    with pytest.raises(RuntimeError, match="does not match declared serial"):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


def test_validate_pdbqt_rejects_non_finite_coordinate(ethanol_pdbqt):
    line = _oxygen_atom_line(ethanol_pdbqt)
    mutated_line = line[:30] + f"{'nan':>8}" + line[38:]
    with pytest.raises(RuntimeError, match="non-finite atom coordinates"):
        pdbqt_validation.validate_pdbqt_torsion_tree(
            ethanol_pdbqt.replace(line, mutated_line)
        )


def test_validate_pdbqt_rejects_non_numeric_torsdof(ethanol_pdbqt):
    mutated = ethanol_pdbqt.replace("TORSDOF 1", "TORSDOF x")
    with pytest.raises(RuntimeError, match="malformed TORSDOF"):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


def test_validate_pdbqt_rejects_cyclic_branch(ethanol_pdbqt):
    mutated = ethanol_pdbqt.replace(
        "ATOM      4  H", "BRANCH   3   2\nATOM      4  H"
    )
    with pytest.raises(RuntimeError, match="re-enters known atom serial"):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)


def test_validate_pdbqt_connectivity_accepts_real_bonds(ethanol_pdbqt):
    connectivity = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
    audit = pdbqt_validation.validate_pdbqt_torsion_tree(
        ethanol_pdbqt, connectivity=connectivity
    )
    assert audit["branch_count"] == 1


def test_validate_pdbqt_rejects_branch_without_real_bond(ethanol_pdbqt):
    connectivity = {1: {2}, 2: {1, 5}, 5: {2}}
    with pytest.raises(RuntimeError, match="does not correspond to a real bond"):
        pdbqt_validation.validate_pdbqt_torsion_tree(
            ethanol_pdbqt, connectivity=connectivity
        )


def test_validate_pdbqt_rejects_disconnected_torsion_tree(ethanol_pdbqt):
    connectivity = {1: {2}, 2: {1, 3}, 3: {2}, 4: {}}
    with pytest.raises(RuntimeError, match="disconnected"):
        pdbqt_validation.validate_pdbqt_torsion_tree(
            ethanol_pdbqt, connectivity=connectivity
        )


def test_validate_pdbqt_rejects_unrepresented_ring_closure(ethanol_pdbqt):
    connectivity = {1: {2, 3}, 2: {1, 3}, 3: {1, 2, 4}, 4: {3}}
    with pytest.raises(RuntimeError, match="molecular bond is not represented"):
        pdbqt_validation.validate_pdbqt_torsion_tree(
            ethanol_pdbqt, connectivity=connectivity
        )


def test_validate_pdbqt_rejects_se_type(ethanol_pdbqt):
    line = _oxygen_atom_line(ethanol_pdbqt)
    mutated = ethanol_pdbqt.replace(line, line.replace(" OA", " SE"))
    with pytest.raises(RuntimeError, match="invalid AutoDock atom type"):
        pdbqt_validation.validate_pdbqt_torsion_tree(mutated)
