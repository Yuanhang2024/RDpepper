"""Engineering equivalence checks for the decomposed docking modules.

The fixed expectations in this file were captured from the verified, monolithic
release candidate ``cycpep_master_release_candidate_20260811_021``.  The tests
do not read that release at runtime and do not execute a scientific benchmark.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.docking import vina_wrapper
from cycpep_master.docking import box as docking_box
from cycpep_master.docking import ligand_pdbqt, protonation, receptor_pdbqt, vina
from cycpep_master.docking import workflow


RELEASE_021_SIGNATURE_SHAPES = {
    "_find_vina": "()",
    "pdb_to_pdbqt_meeko": "(pdb_path, pdbqt_path, is_receptor=False)",
    "_autodock_atom_type": "(atom)",
    "_pdb_to_pdbqt_simple_receptor": "(pdb_path, pdbqt_path)",
    "protonate_ph74": "(smiles)",
    "_bond_dihedral_sigma": (
        "(mol_3d, bond_atom_pairs, n_confs=8, random_seed=42, "
        "num_threads=1, metadata_out=None)"
    ),
    "_actual_rotatable_bonds": "(setup)",
    "_setup_atom_snapshot": "(setup)",
    "_validate_setup_atom_invariants": "(before, setup, tolerance=1e-12)",
    "_torsdof_from_setup": "(setup)",
    "_apply_adaptive_torsion_budget": (
        "(setup, mol_3d, preparator, limit=57, n_confs=8, "
        "random_seed=42, num_threads=1)"
    ),
    "_validate_pdbqt_torsion_tree": "(pdbqt_string)",
    "_atomic_write_text": "(path, text)",
    "_freeze_lowest_sigma_bonds": "(*_args, **_kwargs)",
    "smiles_to_ligand_pdbqt": (
        "(smiles, pdbqt_path, num_confs=10, random_seed=42, "
        "rigid_macrocycles=True, generated_map=None, n_template_confs=1, "
        "conf_out=None, protonate=True, torsdof_limit=None, "
        "torsion_ensemble_size=8, torsion_num_threads=1)"
    ),
    "smiles_to_ligand_pdbqt_multi": (
        "(smiles, out_dir, n_confs=3, generated_map=None, "
        "rigid_macrocycles=True)"
    ),
    "run_vina": (
        "(ligand_pdbqt, receptor_pdbqt, center, box_size, output_pdbqt, "
        "exhaustiveness=32, num_modes=9)"
    ),
    "_parse_vina_affinity": "(vina_stdout)",
    "dock_peptide": (
        "(peptide_pdb, receptor_pdb, center, box_size=(25.0, 25.0, 25.0), "
        "output_dir=None, cleanup=True, *, "
        "ligand_preparation_mode='original_pdb', ligand_smiles=None, "
        "peptide_chain_id='L', receptor_chain_id=None, generated_map=None, "
        "torsdof_limit=None, "
        "torsion_ensemble_size=8, torsion_num_threads=1, "
        "torsion_audit_out=None)"
    ),
    "batch_dock_peptides": (
        "(peptide_pdb_list, receptor_pdb, center, "
        "box_size=(25.0, 25.0, 25.0))"
    ),
    "get_protein_center": "(pdb_path)",
    "get_binding_site_center": (
        "(pdb_path, residue_ids, *, chain_id=None, include_hydrogens=False)"
    ),
}


def _signature_shape(function):
    signature = inspect.signature(function)
    parameters = [
        parameter.replace(annotation=inspect.Signature.empty)
        for parameter in signature.parameters.values()
    ]
    return str(
        signature.replace(
            parameters=parameters,
            return_annotation=inspect.Signature.empty,
        )
    )


def _atom(serial, name, residue, chain, resid, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:>4s} {residue:>3s} {chain:1s}"
        f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00          {element:>2s}\n"
    )


def _write_receptor_fixture(path):
    path.write_text(
        "".join(
            (
                _atom(1, "N", "GLY", "A", 1, 0.0, 0.0, 0.0, "N"),
                _atom(2, "CA", "GLY", "A", 1, 1.4, 0.0, 0.0, "C"),
                _atom(3, "C", "GLY", "A", 1, 2.1, 1.2, 0.0, "C"),
                _atom(4, "O", "GLY", "A", 1, 1.6, 2.3, 0.0, "O"),
                "TER\nEND\n",
            )
        ),
        encoding="ascii",
    )


def test_release_021_facade_function_signatures_are_preserved():
    local_functions = {
        name: value
        for name, value in vars(vina_wrapper).items()
        if inspect.isfunction(value) and value.__module__ == vina_wrapper.__name__
    }
    assert set(local_functions) == set(RELEASE_021_SIGNATURE_SHAPES)
    assert {
        name: _signature_shape(function)
        for name, function in local_functions.items()
    } == RELEASE_021_SIGNATURE_SHAPES


def test_decomposed_workflow_defaults_to_result_first_auto_preparation():
    implementation = inspect.signature(workflow._dock_peptide_impl)
    public = inspect.signature(workflow.dock_peptide)
    assert implementation.parameters["ligand_preparation_mode"].default == "auto"
    assert public.parameters["ligand_preparation_mode"].default == "auto"
    # The historical vina_wrapper facade remains byte-compatible for callers
    # that explicitly depend on the release-021 default.
    assert (
        inspect.signature(vina_wrapper.dock_peptide)
        .parameters["ligand_preparation_mode"]
        .default
        == "original_pdb"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("CC(=O)O", "CC(=O)[O-]"),
        ("NCC(=O)O", "[NH3+]CC(=O)[O-]"),
        ("NCCCCN", "[NH3+]CCCC[NH3+]"),
        ("NC(=N)N", "NC(N)=[NH2+]"),
        ("CS(=O)(=O)O", "CS(=O)(=O)[O-]"),
        ("OP(=O)(O)O", "O=P([O-])([O-])O"),
        ("c1ncc[nH]1", "c1c[nH]cn1"),
        ("not-smiles", "not-smiles"),
    ),
)
def test_protonation_matches_release_021_and_direct_module(source, expected):
    assert vina_wrapper.protonate_ph74(source) == expected
    assert protonation.protonate_ph74(source) == expected


def test_receptor_pdbqt_bytes_match_facade_direct_module_and_release_021(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    facade_output = tmp_path / "facade.pdbqt"
    direct_output = tmp_path / "direct.pdbqt"
    _write_receptor_fixture(receptor)

    assert (
        vina_wrapper.pdb_to_pdbqt_meeko(
            str(receptor), str(facade_output), is_receptor=True
        )
        is None
    )
    assert (
        receptor_pdbqt.pdb_to_receptor_pdbqt(
            str(receptor), str(direct_output)
        )
        is None
    )

    expected_text = "\n".join(
        (
            "REMARK  rigid receptor PDBQT (cycpep_master)",
            "ATOM      1 N    GLY A   1       0.000   0.000   0.000  1.00  0.00    -0.328 N ",
            "ATOM      2 CA   GLY A   1       1.400   0.000   0.000  1.00  0.00     0.016 C ",
            "ATOM      3 C    GLY A   1       2.100   1.200   0.000  1.00  0.00     0.055 C ",
            "ATOM      4 O    GLY A   1       1.600   2.300   0.000  1.00  0.00    -0.395 OA",
            "TER",
            "",
        )
    )
    assert facade_output.read_bytes() == direct_output.read_bytes()
    assert facade_output.read_text(encoding="utf-8") == expected_text


def test_box_centers_match_facade_direct_module_and_release_021(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    _write_receptor_fixture(receptor)

    expected = pytest.approx((1.275, 0.875, 0.0))
    assert vina_wrapper.get_protein_center(str(receptor)) == expected
    assert docking_box.get_protein_center(str(receptor)) == expected
    assert vina_wrapper.get_binding_site_center(
        str(receptor), [1], chain_id="A"
    ) == expected
    assert docking_box.get_binding_site_center(
        str(receptor), [1], chain_id="A"
    ) == expected


@pytest.mark.parametrize(
    ("stdout", "expected"),
    (
        (
            "mode | affinity | dist from best mode\n"
            "-----+----------+--------------------\n"
            "1 -7.25 0.0 0.0\n",
            -7.25,
        ),
        ("no result table\n", None),
        ("mode affinity\n----\n1 not-a-number\n2 -4.5\n", -4.5),
    ),
)
def test_vina_parser_matches_facade_direct_module_and_release_021(stdout, expected):
    assert vina_wrapper._parse_vina_affinity(stdout) == expected
    assert vina.parse_vina_affinity(stdout) == expected


def test_vina_command_contract_matches_facade_and_direct_module(tmp_path, monkeypatch):
    executable = tmp_path / "bin" / "vina.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"")
    observations = []
    monkeypatch.chdir(tmp_path)
    ligand = tmp_path / "ligand.pdbqt"
    receptor = tmp_path / "receptor.pdbqt"
    ligand.write_text(
        "ROOT\n"
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        "ENDROOT\nTORSDOF 0\n",
        encoding="utf-8",
    )
    receptor.write_text("RECEPTOR\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        observations.append((command, kwargs))
        destination = Path(command[command.index("--out") + 1])
        if not destination.is_absolute():
            destination = Path(kwargs["cwd"]) / destination
        destination.write_text(
            "MODEL 1\n"
            "REMARK VINA RESULT 1 -6.75 0 0\n"
            "ROOT\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
            "ENDROOT\nTORSDOF 0\n"
            "ENDMDL\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout="mode | affinity\n-----+---------\n1 -6.75 0 0\n",
            stderr="",
        )

    arguments = (
        "ligand.pdbqt",
        "receptor.pdbqt",
        (1.0, 2.0, 3.0),
        (20.0, 21.0, 22.0),
        "output.pdbqt",
        17,
        5,
    )
    monkeypatch.setattr(vina_wrapper, "_find_vina", lambda: str(executable))
    monkeypatch.setattr(vina_wrapper.subprocess, "run", fake_run)
    facade_result = vina_wrapper.run_vina(*arguments)
    direct_result = vina.run_vina(
        *arguments,
        find_executable=lambda: str(executable),
        run_process=fake_run,
    )

    assert facade_result == direct_result == (-6.75, None)
    assert observations[0] == observations[1]
    assert observations[0][0] == [
        str(executable),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--center_x",
        "1.0",
        "--center_y",
        "2.0",
        "--center_z",
        "3.0",
        "--size_x",
        "20.0",
        "--size_y",
        "21.0",
        "--size_z",
        "22.0",
        "--out",
        str(tmp_path / "output.pdbqt"),
        "--exhaustiveness",
        "17",
        "--num_modes",
        "5",
    ]
    assert observations[0][1] == {
        "capture_output": True,
        "text": True,
        "timeout": 600,
        "cwd": str(executable.parent),
    }


def test_vina_timeout_contract_matches_facade_and_direct_module(tmp_path, monkeypatch):
    executable = tmp_path / "vina"
    executable.write_bytes(b"")
    (tmp_path / "ligand.pdbqt").write_text("LIGAND\n", encoding="utf-8")
    (tmp_path / "receptor.pdbqt").write_text("RECEPTOR\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 600)

    monkeypatch.setattr(vina_wrapper, "_find_vina", lambda: str(executable))
    monkeypatch.setattr(vina_wrapper.subprocess, "run", timeout)
    arguments = (
        "ligand.pdbqt",
        "receptor.pdbqt",
        (0.0, 0.0, 0.0),
        (25.0, 25.0, 25.0),
        "output.pdbqt",
    )
    assert vina_wrapper.run_vina(*arguments) == (
        None,
        "Vina timed out (>10 min)",
    )
    assert vina.run_vina(
        *arguments,
        find_executable=lambda: str(executable),
        run_process=timeout,
    ) == (None, "Vina timed out (>10 min)")


def _workflow_dependencies(calls):
    def convert(source, destination, *, is_receptor=False):
        calls.append(
            ("convert", Path(source).name, Path(destination).name, is_receptor)
        )
        Path(destination).write_text("PDBQT\n", encoding="utf-8")
        return None

    def prepare(smiles, destination, **kwargs):
        calls.append(("prepare", smiles, Path(destination).name))
        kwargs["conf_out"].update(
            {
                "source": "template",
                "torsion_budget": {"status": "applied", "final_torsdof": 12},
                "pdbqt_tree": {"atom_count": 20, "branch_count": 12, "torsdof": 12},
            }
        )
        Path(destination).write_text("PDBQT\n", encoding="utf-8")
        return None

    def execute_vina(**kwargs):
        calls.append(
            (
                "vina",
                Path(kwargs["ligand_pdbqt"]).name,
                Path(kwargs["receptor_pdbqt"]).name,
                Path(kwargs["output_pdbqt"]).name,
                kwargs["center"],
                kwargs["box_size"],
            )
        )
        return -7.25, None

    return convert, prepare, execute_vina


def test_workflow_result_calls_and_audit_match_facade_and_direct_module(
    tmp_path, monkeypatch
):
    facade_calls = []
    direct_calls = []
    facade_dependencies = _workflow_dependencies(facade_calls)
    direct_dependencies = _workflow_dependencies(direct_calls)
    monkeypatch.setattr(vina_wrapper, "pdb_to_pdbqt_meeko", facade_dependencies[0])
    monkeypatch.setattr(vina_wrapper, "smiles_to_ligand_pdbqt", facade_dependencies[1])
    monkeypatch.setattr(vina_wrapper, "run_vina", facade_dependencies[2])
    facade_audit = {}
    direct_audit = {}
    common = {
        "peptide_pdb": "ligand.pdb",
        "receptor_pdb": "receptor.pdb",
        "center": (1.0, 2.0, 3.0),
        "box_size": (20.0, 21.0, 22.0),
        "cleanup": False,
        "ligand_preparation_mode": "audited_smiles",
        "ligand_smiles": "NCC(=O)O",
        "generated_map": "G",
        "torsdof_limit": 12,
        "torsion_ensemble_size": 6,
        "torsion_num_threads": 2,
    }

    facade_result = vina_wrapper.dock_peptide(
        **common,
        output_dir=str(tmp_path / "facade"),
        torsion_audit_out=facade_audit,
    )
    direct_result = workflow._dock_peptide_impl(
        **common,
        output_dir=str(tmp_path / "direct"),
        torsion_audit_out=direct_audit,
        convert_pdb=direct_dependencies[0],
        prepare_smiles=direct_dependencies[1],
        execute_vina=direct_dependencies[2],
    )

    expected_audit = {
        "status": "success",
        "preparation_mode": "audited_smiles",
        "chemical_graph_source": "provided_smiles",
        "original_coordinates_preserved": False,
        "generated_map_supplied": True,
        "torsion_budget": {"status": "applied", "final_torsdof": 12},
        "pdbqt_tree": {"atom_count": 20, "branch_count": 12, "torsdof": 12},
        "conformer_source": "template",
    }
    assert facade_result == direct_result == (-7.25, None)
    assert facade_audit == direct_audit == expected_audit
    assert facade_calls == direct_calls


def test_workflow_not_supported_status_matches_facade_and_direct_module(
    tmp_path, monkeypatch
):
    def convert(_source, _destination, *, is_receptor=False):
        if not is_receptor:
            return "not_supported: ligand graph unavailable"
        raise AssertionError("receptor conversion must not run")

    def should_not_run(**_kwargs):
        raise AssertionError("Vina must not run after ligand rejection")

    monkeypatch.setattr(vina_wrapper, "pdb_to_pdbqt_meeko", convert)
    monkeypatch.setattr(vina_wrapper, "run_vina", should_not_run)
    facade_audit = {}
    direct_audit = {}
    ligand_pdb = tmp_path / "ligand.pdb"
    ligand_pdb.write_text(
        "ATOM      1  C   ALA L   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n",
        encoding="ascii",
    )
    arguments = (str(ligand_pdb), "receptor.pdb", (0.0, 0.0, 0.0))
    facade_result = vina_wrapper.dock_peptide(
        *arguments,
        output_dir=str(tmp_path / "facade"),
        cleanup=False,
        torsion_audit_out=facade_audit,
    )
    direct_result = workflow._dock_peptide_impl(
        *arguments,
        output_dir=str(tmp_path / "direct"),
        cleanup=False,
        ligand_preparation_mode="original_pdb",
        torsion_audit_out=direct_audit,
        convert_pdb=convert,
        execute_vina=should_not_run,
    )
    expected_audit = {
        "status": "not_supported",
        "preparation_mode": "original_pdb",
        "coordinate_source": "input_pdb",
        "original_coordinates_preserved": True,
    }
    assert facade_result == direct_result == (
        None,
        "Ligand conversion failed: not_supported: ligand graph unavailable",
    )
    assert facade_audit == direct_audit == expected_audit


def test_facade_forwards_receptor_chain_id(monkeypatch):
    observed = {}

    def fake_impl(*args, **kwargs):
        observed.update(kwargs)
        return -5.0, None

    monkeypatch.setattr(workflow, "_dock_peptide_impl", fake_impl)

    result = vina_wrapper.dock_peptide(
        "ligand.pdb",
        "receptor.pdb",
        (0.0, 0.0, 0.0),
        receptor_chain_id="R",
    )

    assert result == (-5.0, None)
    assert observed["receptor_chain_id"] == "R"


def test_batch_docking_isolates_one_job_exception():
    calls = []

    def fake_dock(peptide, *_args):
        calls.append(peptide)
        if peptide == "bad.pdb":
            raise RuntimeError("worker crashed")
        return -6.0, None

    rows = workflow.batch_dock_peptides(
        ["good.pdb", "bad.pdb", "later.pdb"],
        "receptor.pdb",
        (0.0, 0.0, 0.0),
        dock=fake_dock,
    )

    assert calls == ["good.pdb", "bad.pdb", "later.pdb"]
    assert rows[0] == ("good", -6.0, None)
    assert rows[1][0:2] == ("bad", None)
    assert "worker crashed" in rows[1][2]
    assert rows[2] == ("later", -6.0, None)


def test_chain_selection_uses_first_model_and_rejects_duplicate_serials(tmp_path):
    source = tmp_path / "models.pdb"
    source.write_text(
        "MODEL        1\n"
        "ATOM      1  C   GLY L   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      1  C   GLY L   1       9.000   0.000   0.000  1.00  0.00           C\n"
        "ENDMDL\nEND\n",
        encoding="ascii",
    )
    selected = tmp_path / "selected.pdb"

    assert workflow._select_pdb_chain(source, selected, "L") is None
    text = selected.read_text(encoding="ascii")
    assert "   0.000" in text
    assert "   9.000" not in text

    source.write_text(
        "ATOM      1  C   GLY L   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      1  O   GLY L   1       1.000   0.000   0.000  1.00  0.00           O\nEND\n",
        encoding="ascii",
    )
    assert "duplicate atom serials" in workflow._select_pdb_chain(
        source, selected, "L"
    )


def test_workflow_missing_ligand_output_marks_audit_failed(tmp_path):
    audit = {}

    result = workflow._dock_peptide_impl(
        "unused.pdb",
        "unused-receptor.pdb",
        (0.0, 0.0, 0.0),
        output_dir=str(tmp_path),
        cleanup=False,
        ligand_preparation_mode="audited_smiles",
        ligand_smiles="CC",
        torsion_audit_out=audit,
        prepare_smiles=lambda *_a, **_k: None,
    )

    assert result[0] is None
    assert "no nonempty" in result[1]
    assert audit["status"] == "failed"


def test_receptor_atom_types_distinguish_donor_nitrogen_and_thiol_sulfur():
    amine = Chem.AddHs(Chem.MolFromSmiles("N"))
    thiol = Chem.AddHs(Chem.MolFromSmiles("CS"))

    assert receptor_pdbqt.autodock_atom_type(amine.GetAtomWithIdx(0)) == "N"
    sulfur = next(atom for atom in thiol.GetAtoms() if atom.GetSymbol() == "S")
    assert receptor_pdbqt.autodock_atom_type(sulfur) == "S"
    quaternary = Chem.MolFromSmiles("C[N+](C)(C)C")
    nitrogen = next(atom for atom in quaternary.GetAtoms() if atom.GetSymbol() == "N")
    assert receptor_pdbqt.autodock_atom_type(nitrogen) == "N"


def test_receptor_atom_type_marks_tertiary_amide_nitrogen_nonaccepting():
    molecule = Chem.MolFromSmiles("CC(=O)N(C)C")
    assert molecule is not None
    nitrogen = next(atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "N")

    assert receptor_pdbqt.autodock_atom_type(nitrogen) == "N"


def test_receptor_omitted_nonpolar_hydrogen_charge_is_merged(tmp_path):
    molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(molecule, randomSeed=7)
    source = tmp_path / "methane.pdb"
    output = tmp_path / "methane.pdbqt"
    Chem.MolToPDBFile(molecule, str(source))

    error = receptor_pdbqt.pdb_to_receptor_pdbqt(str(source), str(output))

    assert error is None
    atom_lines = [
        line for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    ]
    assert len(atom_lines) == 1
    assert abs(float(atom_lines[0].split()[-2])) <= 0.001


def test_box_uses_first_model_even_when_model_number_is_not_one(tmp_path):
    source = tmp_path / "model5.pdb"
    source.write_text(
        "MODEL        5\n"
        "ATOM      1  CA  GLY A   7       1.000   2.000   3.000  1.00  0.00           C\n"
        "ENDMDL\nEND\n",
        encoding="ascii",
    )

    assert docking_box.get_protein_center(str(source)) == (1.0, 2.0, 3.0)
    assert docking_box.get_binding_site_center(str(source), [7]) == (1.0, 2.0, 3.0)


def test_vina_affinity_parser_ignores_non_mode_diagnostic_rows():
    stdout = "mode | affinity\n-----+---------\nwarning -99.0\n1 -6.5 0 0\n"

    assert vina.parse_vina_affinity(stdout) == -6.5


@pytest.mark.skip(
    reason="Release-021 direct-SMILES golden is superseded by V4 MOL2 parent"
)
def test_fixed_seed_ligand_pdbqt_matches_facade_direct_module_and_release_021(
    tmp_path,
):
    pytest.importorskip("meeko")
    golden = (
        Path(__file__).parent
        / "golden"
        / "vina_wrapper"
        / "cyclohexane_seed42.pdbqt"
    ).read_bytes()
    facade_output = tmp_path / "facade.pdbqt"
    direct_output = tmp_path / "direct.pdbqt"
    facade_audit = {}
    direct_audit = {}
    kwargs = {
        "num_confs": 1,
        "random_seed": 42,
        "protonate": False,
        "rigid_macrocycles": True,
    }

    assert (
        vina_wrapper.smiles_to_ligand_pdbqt(
            "C1CCCCC1", str(facade_output), conf_out=facade_audit, **kwargs
        )
        is None
    )
    assert (
        ligand_pdbqt.smiles_to_ligand_pdbqt(
            "C1CCCCC1", str(direct_output), conf_out=direct_audit, **kwargs
        )
        is None
    )
    assert facade_output.read_bytes() == direct_output.read_bytes() == golden
    facade_molecule = facade_audit.pop("mol_3d")
    direct_molecule = direct_audit.pop("mol_3d")
    assert facade_audit == direct_audit
    assert Chem.MolToSmiles(facade_molecule) == Chem.MolToSmiles(direct_molecule)
    assert Chem.MolToMolBlock(facade_molecule) == Chem.MolToMolBlock(direct_molecule)
