from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master import remediation_v5 as v5
from cycpep_master.core.pdb_parser import standard_pdb_atom_name_map
from cycpep_master.core.pdb_parser import parse_backbone
from cycpep_master.core.derived_monomers import build_derived_row
from cycpep_master.core.structure_io import prepare_coordinate_input
from cycpep_master.paths import _map_utils
from cycpep_master.paths.path_a import generate_with_evidence
from cycpep_master.paths.residue_template_factory import get_residue_template
from cycpep_master.remediation_v6 import (
    _adjudicate_evidence_dimensions,
    _candidate_assessment,
    _local_monomer_bootstrap_failure_details,
    _local_monomer_evidence_consistency_dimension,
    _mapping_dimensions,
    _residue_mapping_ledger_audit,
    _terminal_r2_ledger_audit,
    reconstruct_pdb_fail_closed_v6,
)
from cycpep_master.core.local_monomer_inference import (
    _source_identity_row_for_residue,
)


def _passed_dimension():
    return {"passed": True}


def _candidate_row(route, smiles, **updates):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    row = {
        "route": route,
        "status": "success",
        "output_smiles": smiles,
        "output_inchikey": Chem.MolToInchiKey(molecule),
    }
    row.update(updates)
    return row


def test_v6_candidate_assessment_exposes_unique_unqualified_candidate():
    rows = [_candidate_row("a", "N[C@@H](C)C(=O)O")]
    dimensions = {
        name: _passed_dimension()
        for name in (
            "chemical_graph_audit",
            "atom_mapping",
            "explicit_connectivity",
            "closure_identity_trace",
            "mapping_evidence_binding",
            "path_evidence_fresh_replay",
        )
    }
    dimensions["stereochemistry"] = {"passed": False}

    assessment = _candidate_assessment(
        rows,
        dimensions,
        status="rejected",
        qualified_success=False,
        evidence_candidate_full_inchikey=rows[0]["output_inchikey"],
    )

    assert assessment["mode"] == "candidate_unique"
    assert assessment["candidate_count"] == 1
    assert assessment["highest_shared_candidate_level"] == "L3"
    assert assessment["highest_evidence_qualified_level"] == "L2"
    assert assessment["unattended_selection_qualified"] is False
    assert assessment["strict_output_emitted"] is False
    assert assessment["exploratory_only"] is True


def test_v6_candidate_assessment_separates_stereo_conflict_from_connectivity():
    rows = [
        _candidate_row(route, smiles)
        for route, smiles in (
            ("a", "N[C@@H](C)C(=O)O"),
            ("b", "N[C@H](C)C(=O)O"),
        )
    ]

    assessment = _candidate_assessment(
        rows,
        {},
        status="rejected",
        qualified_success=False,
    )

    assert assessment["mode"] == "candidate_ensemble"
    assert assessment["candidate_count"] == 2
    assert assessment["highest_shared_candidate_level"] == "L2"
    assert assessment["highest_evidence_qualified_level"] is None
    assert "connectivity" not in assessment["conflict_dimensions"]
    assert (
        "nonprotonation_identity_within_connectivity"
        in assessment["conflict_dimensions"]
    )
    assert (
        "standard_inchikey_equivalence"
        in assessment["conflict_dimensions"]
    )
    assert assessment["unattended_selection_qualified"] is False


def test_v6_candidate_assessment_rejects_declared_identity_drift():
    row = _candidate_row(
        "a", "N[C@@H](C)C(=O)O", output_inchikey="BAD-KEY"
    )
    assessment = _candidate_assessment(
        [row], {}, status="rejected", qualified_success=False
    )

    assert assessment["mode"] == "no_candidate"
    assert assessment["candidate_count"] == 0
    assert assessment["inadmissible_candidate_routes"] == ["a"]
    assert assessment["route_row_counts"][
        "declared_identity_mismatch_rows"
    ] == 1
    assert assessment["route_identity_audits"][0]["reason"] == (
        "declared_recomputed_identity_mismatch"
    )


def test_v6_candidate_assessment_is_stable_for_same_inchikey_variants():
    rows = [
        _candidate_row("a", "O=c1cccc[nH]1"),
        _candidate_row("b", "Oc1ccccn1"),
    ]
    forward = _candidate_assessment(
        rows, {}, status="rejected", qualified_success=False
    )
    reverse = _candidate_assessment(
        list(reversed(rows)), {}, status="rejected", qualified_success=False
    )

    assert forward == reverse
    assert forward["mode"] == "candidate_unique"
    assert forward["standard_inchikey_equivalence_unique"] is True
    assert forward["literal_canonical_smiles_unique"] is False
    assert len(forward["candidates"][0]["canonical_smiles_variants"]) == 2
    assert json.loads(json.dumps(forward, sort_keys=True)) == forward


def test_v6_candidate_assessment_does_not_label_connectivity_as_stereo():
    assessment = _candidate_assessment(
        [
            _candidate_row("a", "C1CCCCC1"),
            _candidate_row("b", "CCCCCC"),
        ],
        {},
        status="rejected",
        qualified_success=False,
    )

    assert "connectivity" in assessment["conflict_dimensions"]
    assert (
        "nonprotonation_identity_within_connectivity"
        not in assessment["conflict_dimensions"]
    )


def test_v6_candidate_assessment_does_not_qualify_an_ensemble():
    rows = [
        _candidate_row("a", "N[C@@H](C)C(=O)O"),
        _candidate_row("b", "N[C@H](C)C(=O)O"),
    ]
    dimensions = {
        name: _passed_dimension()
        for name in (
            "chemical_graph_audit",
            "atom_mapping",
            "explicit_connectivity",
            "closure_identity_trace",
            "mapping_evidence_binding",
            "path_evidence_fresh_replay",
            "stereochemistry",
        )
    }
    assessment = _candidate_assessment(
        rows,
        dimensions,
        status="rejected",
        qualified_success=False,
        evidence_candidate_full_inchikey=rows[0]["output_inchikey"],
    )

    assert assessment["mode"] == "candidate_ensemble"
    assert assessment["evidence_candidate_identity_bound"] is False
    assert assessment["highest_evidence_qualified_level"] is None


def test_v6_candidate_assessment_empty_input_is_typed_and_serializable():
    assessment = _candidate_assessment(
        [], {}, status="rejected", qualified_success=False
    )

    assert assessment["mode"] == "no_candidate"
    assert assessment["candidate_count"] == 0
    assert assessment["candidates"] == []
    assert assessment["standard_inchikey_equivalence_unique"] is False
    assert assessment["literal_canonical_smiles_unique"] is False
    assert json.loads(json.dumps(assessment, sort_keys=True)) == assessment


def test_v6_classifies_multiport_scaffold_as_capability_boundary():
    bootstrap = SimpleNamespace(
        status="quarantined",
        inference_results=[SimpleNamespace(
            reason_codes=["MULTIPLE_R3_PORTS_NOT_REPRESENTABLE"]
        )],
    )
    failure = _local_monomer_bootstrap_failure_details(bootstrap)
    assert failure["status"] == "not_supported"
    assert failure["code"] == "V6_MULTIPOINT_SCAFFOLD_NOT_SUPPORTED"
    assert failure["support_status"] == "not_supported"


def test_v6_does_not_misclassify_general_monomer_rejection_as_port_conflict():
    bootstrap = SimpleNamespace(
        status="rejected",
        inference_results=[SimpleNamespace(
            reason_codes=["MMCIF_CHEM_COMP_PEPTIDE_BACKBONE_INVALID"]
        )],
    )
    failure = _local_monomer_bootstrap_failure_details(bootstrap)
    assert failure["status"] == "rejected"
    assert failure["code"] == "V6_LOCAL_MONOMER_INPUT_REJECTED"
    assert "port" not in failure["reason"]


def test_v6_keeps_reused_port_as_explicit_evidence_conflict():
    bootstrap = SimpleNamespace(
        status="rejected",
        inference_results=[SimpleNamespace(
            reason_codes=["R3_PORT_REUSED_BY_MULTIPLE_PARTNERS"]
        )],
    )
    failure = _local_monomer_bootstrap_failure_details(bootstrap)
    assert failure["status"] == "rejected"
    assert failure["code"] == "V6_LOCAL_MONOMER_EVIDENCE_CONFLICT"


def _pdb_atom(serial, name, residue, x, y, z, *, residue_name="ALA", element=None):
    element = element or name[0]
    return (
        f"ATOM  {serial:5d} {name:>4s} {residue_name:>3s} A{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def _link_record(atom_1, residue_1, number_1, atom_2, residue_2, number_2):
    record = list(" " * 80)
    record[0:4] = "LINK"
    record[12:16] = f"{atom_1:>4s}"
    record[17:20] = f"{residue_1:>3s}"
    record[21] = "A"
    record[22:26] = f"{number_1:4d}"
    record[42:46] = f"{atom_2:>4s}"
    record[47:50] = f"{residue_2:>3s}"
    record[51] = "A"
    record[52:56] = f"{number_2:4d}"
    return "".join(record)


def _write_explicit_trialanine(tmp_path):
    path = tmp_path / "explicit-trialanine.pdb"
    template = get_residue_template("ALA")
    molecule, conformer, names = _embedded_named_atoms(template, seed=23)
    rows = []
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0), (3, 20.0)):
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(_pdb_atom(
                serial,
                names[atom.GetIdx()],
                residue,
                point.x + offset,
                point.y,
                point.z,
                element=atom.GetSymbol(),
            ))
            serial += 1
    path.write_text(
        "\n".join([
            _link_record("N", "ALA", 1, "C", "ALA", 3),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def test_v5_rejects_atom_hetatm_duplicate_identity(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    lines = path.read_text(encoding="ascii").splitlines()
    atom = next(line for line in lines if line.startswith("ATOM"))
    duplicate = f"HETATM{9999:5d}{atom[11:]}"
    path.write_text("\n".join([duplicate, *lines, ""]), encoding="ascii")

    with pytest.raises(v5._StrictInputError) as raised:
        v5._selected_chain_context(path, "A")

    assert raised.value.code == "V5_DUPLICATE_ATOM_IDENTITY"


def test_v5_rejects_conect_self_connection(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    lines = path.read_text(encoding="ascii").splitlines()
    serial = int(next(line for line in lines if line.startswith("ATOM"))[6:11])
    path.write_text(
        "\n".join([f"CONECT{serial:5d}{serial:5d}", *lines, ""]),
        encoding="ascii",
    )

    with pytest.raises(v5._StrictInputError) as raised:
        v5._selected_chain_context(path, "A")

    assert raised.value.code == "V5_SELF_CONNECTION"


def _write_explicit_r3_carboxyl_macrocycle(
    tmp_path, residue_name, *, acid_first=False
):
    leaving_name = {"ASP": "OD2", "GLU": "OE2"}[residue_name]
    anchor_name = {"ASP": "CG", "GLU": "CD"}[residue_name]
    order = (
        ((1, residue_name, 101), (2, "ALA", 103), (3, "LYS", 107))
        if acid_first
        else ((1, "GLY", 101), (2, "ALA", 103), (3, residue_name, 107))
    )
    order_label = "first" if acid_first else "last"
    path = tmp_path / (
        f"explicit-r3-{residue_name.lower()}-{order_label}.pdb"
    )
    rows = []
    serials = {}
    serial = 1
    for residue, current_name, seed in order:
        template = get_residue_template(current_name)
        molecule = Chem.AddHs(template.mol)
        assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
        AllChem.UFFOptimizeMolecule(molecule)
        molecule = Chem.RemoveHs(molecule)
        conformer = molecule.GetConformer()
        names = {
            index: name for name, index in standard_pdb_atom_name_map(
                current_name, template.smiles
            ).items()
        }
        for atom in molecule.GetAtoms():
            atom_name = names[atom.GetIdx()]
            if current_name == residue_name and atom_name == leaving_name:
                continue
            point = conformer.GetAtomPosition(atom.GetIdx())
            serials[(residue, atom_name)] = serial
            rows.append(_pdb_atom(
                serial,
                atom_name,
                residue,
                point.x + residue * 10.0,
                point.y,
                point.z,
                residue_name=current_name,
                element=atom.GetSymbol(),
            ))
            serial += 1
    path.write_text(
        "\n".join([
            (
                f"CONECT{serials[(1, anchor_name)]:5d}"
                f"{serials[(3, 'NZ')]:5d}"
                if acid_first
                else (
                    f"CONECT{serials[(1, 'N')]:5d}"
                    f"{serials[(3, anchor_name)]:5d}"
                )
            ),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def _write_explicit_serine_lactone(tmp_path):
    path = tmp_path / "explicit-serine-lactone.pdb"
    rows = []
    serials = {}
    serial = 1
    for residue, residue_name, seed in (
        (1, "SER", 109), (2, "ALA", 113), (3, "ALA", 127)
    ):
        template = get_residue_template(residue_name)
        molecule = Chem.AddHs(template.mol)
        assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
        AllChem.UFFOptimizeMolecule(molecule)
        molecule = Chem.RemoveHs(molecule)
        conformer = molecule.GetConformer()
        names = {
            index: name for name, index in standard_pdb_atom_name_map(
                residue_name, template.smiles
            ).items()
        }
        for atom in molecule.GetAtoms():
            atom_name = names[atom.GetIdx()]
            point = conformer.GetAtomPosition(atom.GetIdx())
            serials[(residue, atom_name)] = serial
            rows.append(_pdb_atom(
                serial,
                atom_name,
                residue,
                point.x + residue * 10.0,
                point.y,
                point.z,
                residue_name=residue_name,
                element=atom.GetSymbol(),
            ))
            serial += 1
    path.write_text(
        "\n".join([
            f"CONECT{serials[(1, 'OG')]:5d}{serials[(3, 'C')]:5d}",
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def _write_explicit_disulfide_tripeptide(tmp_path):
    path = tmp_path / "explicit-cys-ala-cys.pdb"
    rows = []
    serial = 1
    for residue, residue_name, offset, seed in (
        (1, "CYS", 0.0, 61),
        (2, "ALA", 10.0, 67),
        (3, "CYS", 20.0, 71),
    ):
        template = get_residue_template(residue_name)
        molecule = Chem.AddHs(template.mol)
        assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
        AllChem.UFFOptimizeMolecule(molecule)
        molecule = Chem.RemoveHs(molecule)
        conformer = molecule.GetConformer()
        names = {
            index: name for name, index in standard_pdb_atom_name_map(
                residue_name, template.smiles
            ).items()
        }
        assert len(names) == molecule.GetNumAtoms()
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(_pdb_atom(
                serial,
                names[atom.GetIdx()],
                residue,
                point.x + offset,
                point.y,
                point.z,
                residue_name=residue_name,
                element=atom.GetSymbol(),
            ))
            serial += 1
    path.write_text(
        "\n".join([
            _link_record("SG", "CYS", 1, "SG", "CYS", 3),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def _mirror_sidechain_across_backbone_plane(path, residue):
    lines = path.read_text(encoding="ascii").splitlines()
    selected = {}
    for index, line in enumerate(lines):
        if line[:6] not in {"ATOM  ", "HETATM"}:
            continue
        if int(line[22:26]) != residue:
            continue
        name = line[12:16].strip()
        if name in {"N", "CA", "C", "CB"}:
            selected[name] = (
                index,
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            )
    assert set(selected) == {"N", "CA", "C", "CB"}
    origin = selected["CA"][1]
    first = tuple(selected["N"][1][i] - origin[i] for i in range(3))
    second = tuple(selected["C"][1][i] - origin[i] for i in range(3))
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    norm2 = sum(value * value for value in normal)
    assert norm2 > 1e-8
    cb = selected["CB"][1]
    displacement = tuple(cb[i] - origin[i] for i in range(3))
    scale = 2.0 * sum(displacement[i] * normal[i] for i in range(3)) / norm2
    mirrored = tuple(cb[i] - scale * normal[i] for i in range(3))
    line_index = selected["CB"][0]
    line = lines[line_index]
    lines[line_index] = (
        line[:30]
        + f"{mirrored[0]:8.3f}{mirrored[1]:8.3f}{mirrored[2]:8.3f}"
        + line[54:]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _modres_record(modified_name, residue_number, standard_name, chain="A"):
    record = list(" " * 80)
    record[0:6] = "MODRES"
    record[7:10] = "  1"
    record[12:15] = f"{modified_name:>3s}"
    record[16] = chain
    record[18:22] = f"{residue_number:4d}"
    record[24:27] = f"{standard_name:>3s}"
    record[29:40] = "source-bound"
    return "".join(record)


def _write_trialanine_with_unknown_alias(
    tmp_path, *, include_modres=False, modres_standard_name="ALA"
):
    template = get_residue_template("ALA")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=31) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    names = {
        index: name for name, index in standard_pdb_atom_name_map(
            "ALA", template.smiles
        ).items()
    }
    rows = []
    serial = 1
    for residue in (1, 2, 3):
        resname = "ZZA" if residue == 2 else "ALA"
        record = "HETATM" if residue == 2 else "ATOM  "
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(
                f"{record}{serial:5d} {names[atom.GetIdx()]:>4s} {resname:>3s} "
                f"A{residue:4d}    {point.x + residue * 10:8.3f}{point.y:8.3f}"
                f"{point.z:8.3f}  1.00  0.00          {atom.GetSymbol():>2s}"
            )
            serial += 1
    path = tmp_path / "trialanine-unknown-alias.pdb"
    header = (
        [_modres_record("ZZA", 2, modres_standard_name)]
        if include_modres else []
    )
    path.write_text(
        "\n".join([
            *header,
            _link_record("N", "ALA", 1, "C", "ALA", 3),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def _embedded_named_atoms(template, *, seed, generic_prefix="X"):
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    n_idx, ca_idx, cb_idx, c_idx, o_idx = parse_backbone(molecule)
    names = {
        n_idx: "N", ca_idx: "CA", c_idx: "C", o_idx: "O",
    }
    if cb_idx is not None:
        names[cb_idx] = "CB"
    return molecule, conformer, {
        index: names.get(index, f"{generic_prefix}{index}")
        for index in range(molecule.GetNumAtoms())
    }


def _write_cyclic_peptide_with_novel_nnaa(tmp_path):
    ala = get_residue_template("ALA")
    novel_row = build_derived_row(
        "FIX", "N[C@@H](CCBr)C(=O)O", monomer_id=20000
    )[0]
    with _map_utils.isolated_monomer_registry(derived_rows=[novel_row]):
        novel = get_residue_template(novel_row["symbol"])
        novel_mol, novel_conf, novel_names = _embedded_named_atoms(
            novel, seed=37, generic_prefix="Q"
        )
    ala_mol, ala_conf, ala_names = _embedded_named_atoms(ala, seed=41)
    rows = []
    serial = 1
    for residue, resname, record, molecule, conformer, names in (
        (1, "ALA", "ATOM  ", ala_mol, ala_conf, ala_names),
        (2, "ZZQ", "HETATM", novel_mol, novel_conf, novel_names),
        (3, "ALA", "ATOM  ", ala_mol, ala_conf, ala_names),
    ):
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(
                f"{record}{serial:5d} {names[atom.GetIdx()]:>4s} {resname:>3s} "
                f"A{residue:4d}    {point.x + residue * 10:8.3f}{point.y:8.3f}"
                f"{point.z:8.3f}  1.00  0.00          {atom.GetSymbol():>2s}"
            )
            serial += 1
    path = tmp_path / "cyclic-novel-nnaa.pdb"
    path.write_text(
        "\n".join([
            _link_record("N", "ALA", 1, "C", "ALA", 3),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def _write_disulfide_peptide_with_novel_thiol(tmp_path):
    ala = get_residue_template("ALA")
    cys = get_residue_template("CYS")
    novel_row = build_derived_row(
        "FIX", "N[C@@H](C(C#N)CS)C(=O)O", monomer_id=20001
    )[0]
    with _map_utils.isolated_monomer_registry(derived_rows=[novel_row]):
        novel = get_residue_template(novel_row["symbol"])
        novel_mol, novel_conf, novel_names = _embedded_named_atoms(
            novel, seed=43, generic_prefix="Q"
        )
    sulfur_index = next(
        atom.GetIdx() for atom in novel_mol.GetAtoms() if atom.GetSymbol() == "S"
    )
    novel_names[sulfur_index] = "SG"
    ala_mol, ala_conf, ala_names = _embedded_named_atoms(ala, seed=47)
    cys_mol = Chem.AddHs(cys.mol)
    assert AllChem.EmbedMolecule(cys_mol, randomSeed=53) == 0
    AllChem.UFFOptimizeMolecule(cys_mol)
    cys_mol = Chem.RemoveHs(cys_mol)
    cys_conf = cys_mol.GetConformer()
    cys_names = {
        index: name for name, index in standard_pdb_atom_name_map(
            "CYS", cys.smiles
        ).items()
    }
    rows = []
    serial = 1
    for residue, resname, record, molecule, conformer, names in (
        (1, "ALA", "ATOM  ", ala_mol, ala_conf, ala_names),
        (2, "ZZS", "HETATM", novel_mol, novel_conf, novel_names),
        (3, "CYS", "ATOM  ", cys_mol, cys_conf, cys_names),
    ):
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(
                f"{record}{serial:5d} {names[atom.GetIdx()]:>4s} {resname:>3s} "
                f"A{residue:4d}    {point.x + residue * 10:8.3f}{point.y:8.3f}"
                f"{point.z:8.3f}  1.00  0.00          {atom.GetSymbol():>2s}"
            )
            serial += 1
    path = tmp_path / "disulfide-novel-thiol.pdb"
    path.write_text(
        "\n".join([
            _link_record("SG", "ZZS", 2, "SG", "CYS", 3),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def _single_a_inputs(path):
    validation = v5.validate_pdb_reconstruction_input_v5(path, "A")
    assert validation.accepted
    smiles, error, evidence = generate_with_evidence(str(path), "A")
    assert error is None
    evidence = dict(evidence)
    evidence["output_smiles"] = smiles
    evidence["error"] = error
    row = {
        "route": "a",
        "status": "success",
        "output_smiles": smiles,
        "output_inchikey": evidence["output_inchikey"],
        "warning_codes": [],
    }
    return row, dict(validation.context), evidence


def _single_e_inputs(path):
    validation = v5.validate_pdb_reconstruction_input_v5(path, "A")
    assert validation.accepted
    smiles, error, evidence = generate_with_evidence(
        str(path), "A", geometric_cyclization=True
    )
    assert error is None
    evidence = dict(evidence)
    evidence["output_smiles"] = smiles
    evidence["error"] = error
    row = {
        "route": "e",
        "status": "success",
        "output_smiles": smiles,
        "output_inchikey": evidence["output_inchikey"],
        "warning_codes": [],
    }
    return row, dict(validation.context), evidence


def test_v6_request_reuses_one_pdb_audit(tmp_path, monkeypatch):
    path = _write_explicit_trialanine(tmp_path)
    original = v5.audit_pdb_file
    calls = []

    def counted(*args, **kwargs):
        calls.append(str(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(v5, "audit_pdb_file", counted)
    result = reconstruct_pdb_fail_closed_v6(path, "A")

    assert result.status in {"success", "rejected", "not_supported"}
    assert calls == [str(path)]


def test_v6_end_to_end_uses_evidence_dimensions(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    result = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )
    assert result.status == "success", result.rejection_reason
    assert result.support_status == "qualified"
    assert result.qualified_success
    assert result.repair_codes == []
    persistent = result.input_evidence["persistent_overlay_audit"]
    policy = result.input_evidence["entity_local_isolation"]
    assert persistent["status"] == "empty"
    assert persistent["disk_memory_count_consistent"] is True
    assert policy["require_empty_persistent_overlay"] is True
    assert policy["persistent_overlay_state_sha256"] == persistent["state_sha256"]
    assert set(result.output_evidence["accepted_evidence_dimensions"]) == {
        "library_chemistry",
        "atom_mapping",
        "explicit_connectivity",
        "closure_identity_trace",
        "chemical_graph_audit",
        "stereochemistry",
        "candidate_consistency",
        "independent_family_consensus",
        "diagnostic_identity_consistency",
        "mapping_evidence_binding",
        "path_evidence_fresh_replay",
        "local_monomer_evidence_consistency",
    }
    assert result.output_evidence["adjudication_class"] == (
        "multi_family_consensus"
    )
    assert result.output_evidence["acceptance_mode"] == (
        "multi_family_consensus"
    )
    assert result.output_evidence["reconstruction_mode"] == "curated_library"
    assert result.output_evidence["successful_chemical_route_count"] >= 1
    assessment = result.output_evidence["candidate_assessment"]
    assert assessment["mode"] == "candidate_unique"
    assert assessment["highest_shared_candidate_level"] == "L3"
    assert assessment["highest_evidence_qualified_level"] == "L3"
    assert assessment["evidence_candidate_identity_bound"] is True
    assert assessment["unattended_selection_qualified"] is True
    assert assessment["strict_output_emitted"] is True
    assert assessment["selected_output_identity_bound"] is True
    assert assessment["automatic_selection_permitted"] is True
    assert assessment["exploratory_only"] is False
    assert assessment["candidates"][0]["full_inchikey"] == result.output_inchikey


def test_v6_rejects_standard_atom_element_swaps_with_preserved_counts(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    lines = path.read_text(encoding="ascii").splitlines()
    swapped = []
    changed = {"N": False, "O": False}
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            atom_name = line[12:16].strip()
            if atom_name in changed and not changed[atom_name]:
                replacement = "O" if atom_name == "N" else "N"
                line = f"{line[:76]}{replacement:>2s}{line[78:]}"
                changed[atom_name] = True
        swapped.append(line)
    assert all(changed.values())
    path.write_text("\n".join(swapped) + "\n", encoding="ascii")

    result = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )

    assert result.status == "rejected"
    assert result.qualified_success is False
    assert "V5_ATOM_NAME_ELEMENT_CONFLICT" in result.warning_codes


def test_v6_rejects_terminal_oxt_declared_as_hydrogen(tmp_path):
    path = _write_explicit_disulfide_tripeptide(tmp_path)
    lines = path.read_text(encoding="ascii").splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    serial = max(int(line[6:11]) for line in atom_lines) + 1
    oxt = _pdb_atom(
        serial,
        "OXT",
        3,
        24.0,
        0.0,
        0.0,
        residue_name="CYS",
        element="H",
    )
    path.write_text(
        "\n".join([line for line in lines if line != "END"] + [oxt, "END"]) + "\n",
        encoding="ascii",
    )

    result = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )

    assert result.status == "rejected"
    assert result.qualified_success is False
    assert "V5_ATOM_NAME_ELEMENT_CONFLICT" in result.warning_codes


def test_oxt_element_is_unambiguous_for_unknown_residue_names():
    assert v5._expected_standard_atom_element("ZZZ", "OXT") == "O"
    assert v5._expected_standard_atom_element("ZZZ", "CA") is None


def test_unified_strict_facade_matches_legacy_trialanine_golden(tmp_path):
    from cycpep_master.reconstruction import reconstruct_structure

    path = _write_explicit_trialanine(tmp_path)
    legacy = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )
    unified = reconstruct_structure(path, chain_id="A", mode="strict")

    expected_smiles = "C[C@@H]1NC(=O)[C@H](C)NC(=O)[C@H](C)NC1=O"
    expected_inchikey = "BTYOLWLBIVNFKQ-ZLUOBGJFSA-N"
    assert legacy.status == "success"
    assert legacy.qualified_success is True
    assert legacy.output_smiles == expected_smiles
    assert legacy.output_inchikey == expected_inchikey
    assert legacy.path_used == "V6_EVIDENCE_DIMENSIONS:a"
    assert legacy.warning_codes == []

    assert unified.status == "success"
    assert unified.quality == "exact"
    assert unified.result_origin == "strict_v6"
    assert unified.smiles == legacy.output_smiles
    assert unified.inchikey == legacy.output_inchikey
    assert unified.warning_codes == legacy.warning_codes
    assert unified.result is unified.strict_result
    assert unified.strict_result.output_smiles == legacy.output_smiles
    assert unified.strict_result.output_inchikey == legacy.output_inchikey


@pytest.mark.parametrize(
    ("residue_name", "leaving_name"),
    [("ASP", "OD2"), ("GLU", "OE2")],
)
def test_v6_accepts_explicit_r3_carboxyl_closure_with_consumed_leaving_atom(
    tmp_path, residue_name, leaving_name
):
    path = _write_explicit_r3_carboxyl_macrocycle(tmp_path, residue_name)

    result = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )

    assert result.status == "success", result.rejection_reason
    assert result.qualified_success is True
    assert result.repair_codes == []
    route_a = next(
        row for row in result.route_results
        if row.get("route") == "a" and row.get("status") == "success"
    )
    residue = next(
        row for row in route_a["evidence_dimensions_input"]["residue_evidence"]
        if row["residue_position"] == 3
    )
    assert residue["mapping_method"] == (
        "standard_pdb_atom_names_with_consumed_r3_leaving_atom"
    )
    assert residue["deferred_consumed_r3_template_atoms"][0][
        "atom_name"
    ] == leaving_name
    assert residue["effective_unmapped_template_heavy_atom_count"] == 0


def test_v6_trace_reindexes_endpoints_after_earlier_r3_atom_removal(tmp_path):
    path = _write_explicit_r3_carboxyl_macrocycle(
        tmp_path, "GLU", acid_first=True
    )

    result = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )

    assert result.status == "success", result.rejection_reason
    closure_trace = result.output_evidence["evidence_dimensions"][
        "closure_identity_trace"
    ]
    route_a = next(
        row for row in closure_trace["route_traces"]
        if row.get("expected_route") == "a"
    )
    assert route_a["passed"] is True
    assert route_a["closure_audits"][0]["checks"]["endpoint_2_bound"] is True


def test_v6_r3_source_audit_preserves_serine_og_lactone(tmp_path):
    path = _write_explicit_serine_lactone(tmp_path)

    result = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )

    assert result.status == "rejected"
    assert result.qualified_success is False
    dimensions = result.output_evidence["evidence_dimensions"]
    assert dimensions["independent_family_consensus"]["passed"] is False
    binding = dimensions["mapping_evidence_binding"]
    assert binding["passed"] is True


@pytest.mark.parametrize(
    "tamper",
    [
        "template_index",
        "source_bound_serial",
        "duplicate_source_bound",
        "duplicate_deferred_row",
        "template_identity",
    ],
)
def test_v6_mapping_ledger_rejects_tampered_consumed_r3_evidence(
    tmp_path, tamper
):
    path = _write_explicit_r3_carboxyl_macrocycle(tmp_path, "GLU")
    _row, input_evidence, evidence = _single_a_inputs(path)
    assert _residue_mapping_ledger_audit(evidence, input_evidence)["passed"]
    tampered = deepcopy(evidence)
    residue = next(
        row for row in tampered["residue_evidence"]
        if row["residue_position"] == 3
    )
    if tamper == "template_index":
        residue["deferred_consumed_r3_template_atoms"][0][
            "template_atom_index"
        ] = 0
    elif tamper == "source_bound_serial":
        residue["source_bound_explicit_r3_attachment_serials"] = [999999]
    elif tamper == "duplicate_source_bound":
        residue["source_bound_explicit_r3_attachment_serials"] *= 2
    elif tamper == "duplicate_deferred_row":
        residue["deferred_consumed_r3_template_atoms"] *= 2
    else:
        forged = get_residue_template("dE")
        residue.update({
            "unified_symbol": forged.symbol,
            "unified_source": forged.source,
            "monomer_graph_sha256": hashlib.sha256(
                forged.smiles.encode("utf-8")
            ).hexdigest(),
            "free_monomer_graph_sha256": forged.free_graph_sha256,
            "rgroup_defaults": {
                "R1": forged.r1, "R2": forged.r2, "R3": forged.r3,
            },
            "r3_anchor_template_atom_index": forged.r3_anchor_index,
        })

    audit = _residue_mapping_ledger_audit(tampered, input_evidence)

    assert audit["passed"] is False
    row_audit = next(
        row for row in audit["row_audits"] if row.get("residue_position") == 3
    )
    failed_checks = {
        name for name, passed in row_audit["checks"].items() if not passed
    }
    assert failed_checks & {
        "deferred_r3_indices_disjoint",
        "deferred_r3_source_bound",
        "deferred_r3_canonical",
        "source_bound_r3_canonical",
        "source_bound_r3_serials_recomputed",
        "template_symbol_bound",
    }


@pytest.mark.parametrize("tamper", ["boolean_count", "float_index"])
def test_v6_mapping_ledger_rejects_noncanonical_numeric_types(
    tmp_path, tamper
):
    path = _write_explicit_r3_carboxyl_macrocycle(tmp_path, "GLU")
    _row, input_evidence, evidence = _single_a_inputs(path)
    tampered = deepcopy(evidence)
    residue = tampered["residue_evidence"][0]
    if tamper == "boolean_count":
        residue["deferred_consumed_r3_count"] = False
    else:
        serial = next(iter(residue["serial_to_template_atom_index"]))
        residue["serial_to_template_atom_index"][serial] = float(
            residue["serial_to_template_atom_index"][serial]
        )

    audit = _residue_mapping_ledger_audit(tampered, input_evidence)

    assert audit["passed"] is False


def test_v6_end_to_end_accepts_link_endpoint_with_insertion_code(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    lines = path.read_text(encoding="ascii").splitlines()
    rewritten = []
    for line in lines:
        chars = list(line.ljust(80))
        if line.startswith("LINK"):
            chars[56] = "A"
        elif line.startswith(("ATOM", "HETATM")) and int(line[22:26]) == 3:
            chars[26] = "A"
        rewritten.append("".join(chars).rstrip())
    path.write_text("\n".join(rewritten) + "\n", encoding="ascii")

    result = reconstruct_pdb_fail_closed_v6(path, "A")

    assert result.status == "success", result.rejection_reason
    assert result.support_status == "qualified"
    assert result.qualified_success
    assert result.repair_codes == []
    assert result.input_evidence["residue_atom_inventory"][-1]["insertion_code"] == "A"
    assert "FAMILY" not in result.path_used


def test_v6_single_a_route_is_audited_but_cannot_form_family_consensus(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": evidence}, pdb_path=path, chain_id="A"
    )
    assert result.status == "rejected"
    dimensions = result.output_evidence["evidence_dimensions"]
    assert dimensions["candidate_consistency"][
        "successful_candidate_count"
    ] == 1
    assert dimensions["independent_family_consensus"]["passed"] is False
    assert "independent_family_consensus" in result.rejection_reason


def test_v6_a_and_e_same_family_cannot_form_consensus(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    a_row, input_evidence, a_evidence = _single_a_inputs(path)
    e_row, _unused, e_evidence = _single_e_inputs(path)

    result = _adjudicate_evidence_dimensions(
        [a_row, e_row],
        input_evidence,
        {"a": a_evidence, "e": e_evidence},
        pdb_path=path,
        chain_id="A",
    )

    assert result.status == "rejected"
    dimension = result.output_evidence["evidence_dimensions"][
        "independent_family_consensus"
    ]
    assert dimension["passed"] is False
    assert dimension["successful_qualifying_families"] == [
        "residue_template"
    ]


def test_v6_single_e_family_cannot_replace_independent_consensus(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    e_row, input_evidence, e_evidence = _single_e_inputs(path)
    failed_a_row = {
        "route": "a",
        "status": "rejected",
        "output_smiles": None,
        "output_inchikey": None,
        "warning_codes": ["FORCED_A_FAILURE"],
    }
    failed_a_evidence = {
        "route": "a",
        "output_smiles": None,
        "output_inchikey": None,
        "error": "forced A failure",
        "residue_evidence": [],
    }

    result = _adjudicate_evidence_dimensions(
        [failed_a_row, e_row],
        input_evidence,
        {"a": failed_a_evidence, "e": e_evidence},
        pdb_path=path,
        chain_id="A",
    )

    assert result.status == "rejected"
    dimensions = result.output_evidence["evidence_dimensions"]
    assert dimensions["independent_family_consensus"]["passed"] is False
    binding = dimensions["mapping_evidence_binding"]
    assert binding["passed"] is True
    assert binding["valid_mapping_source_routes"] == ["e"]
    assert binding["route_audits"][0]["non_success_evidence_ignored"] is True


def test_v6_f_h_diagnostic_identity_disagreement_is_a_veto(tmp_path):
    from cycpep_master.remediation_v3 import _identity

    path = _write_explicit_trialanine(tmp_path)
    a_row, input_evidence, a_evidence = _single_a_inputs(path)
    b_row = {
        "route": "b",
        "status": "success",
        "output_smiles": a_row["output_smiles"],
        "output_inchikey": a_row["output_inchikey"],
        "warning_codes": [],
    }
    diagnostic = {
        "route": "f",
        "status": "rejected",
        "output_smiles": None,
        "output_inchikey": None,
        "candidate_identity": _identity("CC"),
        "warning_codes": [],
    }

    result = _adjudicate_evidence_dimensions(
        [a_row, b_row, diagnostic],
        input_evidence,
        {"a": a_evidence},
        pdb_path=path,
        chain_id="A",
    )

    assert result.status == "rejected"
    dimension = result.output_evidence["evidence_dimensions"][
        "diagnostic_identity_consistency"
    ]
    assert dimension["passed"] is False
    assert dimension["disagreeing_full_inchikeys"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_inchikey", "XDTMQSROBMDMFD-UHFFFAOYSA-N"),
        ("output_smiles", "C1CCCCC1"),
        ("route", "a"),
        ("geometry_inference_enabled", False),
    ],
)
def test_v6_rejects_e_mapping_evidence_identity_drift(
    tmp_path, field, value
):
    path = _write_explicit_trialanine(tmp_path)
    e_row, input_evidence, evidence = _single_e_inputs(path)
    e_evidence = {**deepcopy(evidence), field: value}

    result = _adjudicate_evidence_dimensions(
        [e_row], input_evidence, {"e": e_evidence},
        pdb_path=path, chain_id="A",
    )

    assert result.status == "rejected"
    binding = result.output_evidence["evidence_dimensions"][
        "mapping_evidence_binding"
    ]
    assert binding["passed"] is False
    assert binding["mapping_source_route"] is None
    assert binding["route_audits"][1]["passed"] is False


def test_v6_rejects_drift_in_any_successful_path_evidence(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    a_row, input_evidence, a_evidence = _single_a_inputs(path)
    e_row, _e_input, e_evidence = _single_e_inputs(path)
    e_evidence = {
        **deepcopy(e_evidence),
        "output_inchikey": "XDTMQSROBMDMFD-UHFFFAOYSA-N",
    }

    result = _adjudicate_evidence_dimensions(
        [a_row, e_row],
        input_evidence,
        {"a": a_evidence, "e": e_evidence},
        pdb_path=path,
        chain_id="A",
    )

    assert result.status == "rejected"
    binding = result.output_evidence["evidence_dimensions"][
        "mapping_evidence_binding"
    ]
    assert binding["passed"] is False
    assert binding["valid_mapping_source_routes"] == ["a"]
    assert binding["route_audits"][0]["passed"] is True
    assert binding["route_audits"][1]["passed"] is False


def test_v6_entity_local_alias_is_repaired_and_never_leaks(tmp_path):
    path = _write_trialanine_with_unknown_alias(tmp_path)
    assert _map_utils.resolve_pdb_alias("ZZA") is None

    result = reconstruct_pdb_fail_closed_v6(path, "A")
    assert result.status == "success", result.rejection_reason
    assert result.support_status == "repaired"
    assert not result.qualified_success
    assert "UNIFIED_LIBRARY_STRUCTURE_ALIAS" in result.repair_codes
    assert result.input_evidence["local_monomer_bootstrap"]["persistent_writes"] == 0
    assert result.input_evidence["local_monomer_bootstrap"]["pdb_aliases"][0][
        "target_symbol"
    ] == "A"
    assert result.output_evidence["reconstruction_mode"] == "unified_alias"
    assessment = result.output_evidence["candidate_assessment"]
    assert assessment["strict_output_emitted"] is True
    assert assessment["selected_output_identity_bound"] is True
    assert assessment["unattended_selection_qualified"] is False
    assert assessment["result_context"]["support_status"] == "repaired"
    assert assessment["result_context"]["repair_codes"] == [
        "UNIFIED_LIBRARY_STRUCTURE_ALIAS"
    ]
    assert _map_utils.resolve_pdb_alias("ZZA") is None

    tampered_input = deepcopy(result.input_evidence)
    tampered_input["local_monomer_bootstrap"]["inference_results"][0][
        "evidence"
    ]["library_first_match"]["released"] = False
    path_evidence = {
        row["route"]: row["evidence_dimensions_input"]
        for row in result.route_results
        if row.get("route") in {"a", "e"}
        and isinstance(row.get("evidence_dimensions_input"), dict)
    }
    dimension = _local_monomer_evidence_consistency_dimension(
        result.route_results,
        tampered_input,
        path_evidence["a"],
        result.output_inchikey,
    )
    assert dimension["passed"] is False
    assert dimension["inference_checks"][0]["passed"] is False
    assert dimension["inference_checks"][0]["direct_match_checks"]["checks"][
        "released"
    ] is False

    tampered_port = deepcopy(result.input_evidence)
    tampered_port["local_monomer_bootstrap"]["inference_results"][0][
        "evidence"
    ]["library_first_match"]["selected_identity"][1] = "FORGED:R3"
    port_dimension = _local_monomer_evidence_consistency_dimension(
        result.route_results,
        tampered_port,
        path_evidence["a"],
        result.output_inchikey,
    )
    assert port_dimension["passed"] is False
    assert port_dimension["inference_checks"][0]["direct_match_checks"][
        "checks"
    ]["selected_identity_bound"] is False


def test_v6_partial_unknown_without_seqres_never_qualifies(tmp_path):
    """A coordinate-only partial alias remains explicitly non-qualified."""
    path = _write_cyclic_peptide_with_novel_nnaa(tmp_path)
    lines = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[22:26].strip() == "2":
            if line[12:16].strip() not in {"N", "CA", "C", "O", "CB"}:
                continue
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

    result = reconstruct_pdb_fail_closed_v6(path, "A")

    assert result.status == "success", result.rejection_reason
    assert result.support_status == "repaired"
    assert result.qualified_success is False
    assert result.output_smiles
    assert "UNIFIED_LIBRARY_STRUCTURE_ALIAS" in result.warning_codes
    inference = result.input_evidence["local_monomer_bootstrap"][
        "inference_results"
    ][0]
    assert inference["evidence"]["source_identity_constraint"]["status"] == (
        "absent"
    )
    assert inference["evidence"]["observed_heavy_atom_count"] == 5


def test_v6_source_identity_constraint_rejects_truncated_unknown_alias(tmp_path):
    path = _write_trialanine_with_unknown_alias(
        tmp_path, include_modres=True, modres_standard_name="DPN"
    )
    path.write_text(
        "SEQRES   1 A    3  ALA DPN ALA\n"
        + path.read_text(encoding="ascii").replace("ZZA", "ZZZ"),
        encoding="ascii",
    )

    with prepare_coordinate_input(path, "A") as prepared:
        result = reconstruct_pdb_fail_closed_v6(
            prepared.pdb_path,
            prepared.chain_id,
            coordinate_input_evidence=prepared.audit,
        )

    assert result.status == "rejected"
    assert result.qualified_success is False
    assert "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE" in result.warning_codes
    bootstrap = result.input_evidence["local_monomer_bootstrap"]
    assert bootstrap["inference_results"][0]["reason_codes"] == [
        "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"
    ]


def test_v6_source_identity_constraint_rejects_unavailable_declared_template(
    tmp_path,
):
    path = _write_trialanine_with_unknown_alias(
        tmp_path, include_modres=True, modres_standard_name="QQQ"
    )
    path.write_text(
        "SEQRES   1 A    3  ALA QQQ ALA\n"
        + path.read_text(encoding="ascii"),
        encoding="ascii",
    )

    with prepare_coordinate_input(path, "A") as prepared:
        result = reconstruct_pdb_fail_closed_v6(
            prepared.pdb_path,
            prepared.chain_id,
            coordinate_input_evidence=prepared.audit,
        )

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"]
    inference = result.input_evidence["local_monomer_bootstrap"][
        "inference_results"
    ][0]
    assert inference["reason_codes"] == [
        "SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"
    ]
    assert inference["evidence"]["source_identity_constraint"]["status"] == (
        "declared_template_unavailable"
    )


def test_v6_malformed_source_identity_coordinate_key_is_typed_rejection(tmp_path):
    path = _write_trialanine_with_unknown_alias(tmp_path)
    with prepare_coordinate_input(path, "A") as prepared:
        evidence = deepcopy(prepared.audit)
        evidence["source_sequence_identity_audit"] = {
            "source_metadata_present": True,
            "status": "unique",
            "rows": [{
                "mapping_state": "unique",
                "coordinate_resseq": "not-an-integer",
                "coordinate_icode": "",
                "coordinate_residue_name": "ZZA",
                "declared_name": "ALA",
            }],
        }
        result = reconstruct_pdb_fail_closed_v6(
            prepared.pdb_path,
            prepared.chain_id,
            coordinate_input_evidence=evidence,
        )

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_MAPPING_UNRESOLVED"]


def test_source_identity_row_malformed_coordinate_key_is_unmapped():
    row, status = _source_identity_row_for_residue(
        {
            "status": "unique",
            "rows": [{
                "mapping_state": "unique",
                "coordinate_resseq": "not-an-integer",
                "coordinate_icode": "",
                "coordinate_residue_name": "ZZA",
            }],
        },
        ("ZZA", 2, True),
    )

    assert row is None
    assert status == "invalid"


def test_source_identity_partial_audit_cannot_use_unique_row_constraint():
    row, status = _source_identity_row_for_residue(
        {
            "status": "partial",
            "rows": [{
                "mapping_state": "unique",
                "coordinate_resseq": 2,
                "coordinate_icode": "",
                "coordinate_residue_name": "ZZA",
                "declared_name": "ALA",
            }],
        },
        ("ZZA", 2, True),
    )

    assert row is None
    assert status == "partial"


def test_v6_source_identity_constraint_keeps_complete_unknown_alias_successful(
    tmp_path,
):
    path = _write_trialanine_with_unknown_alias(tmp_path, include_modres=True)
    path.write_text(
        "SEQRES   1 A    3  ALA ALA ALA\n" + path.read_text(encoding="ascii"),
        encoding="ascii",
    )

    with prepare_coordinate_input(path, "A") as prepared:
        result = reconstruct_pdb_fail_closed_v6(
            prepared.pdb_path,
            prepared.chain_id,
            coordinate_input_evidence=prepared.audit,
        )

    assert result.status == "success", result.rejection_reason
    assert result.support_status == "repaired"
    assert result.qualified_success is False
    assert "UNIFIED_LIBRARY_STRUCTURE_ALIAS" in result.repair_codes
    source_constraint = result.input_evidence["local_monomer_bootstrap"][
        "inference_results"
    ][0]["evidence"]["source_identity_constraint"]
    assert source_constraint["declared_name"] == "ALA"
    assert source_constraint["status"] == "constrained"


def test_v6_known_coordinate_residue_ignores_stale_source_sequence(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    path.write_text(
        "SEQRES   1 A    3  CYS CYS CYS\n" + path.read_text(encoding="ascii"),
        encoding="ascii",
    )

    with prepare_coordinate_input(path, "A") as prepared:
        result = reconstruct_pdb_fail_closed_v6(
            prepared.pdb_path,
            prepared.chain_id,
            coordinate_input_evidence=prepared.audit,
        )

    assert result.status == "success", result.rejection_reason
    assert result.repair_codes == []


def test_v6_rejects_mirrored_known_residue_coordinates(tmp_path):
    path = _write_trialanine_with_unknown_alias(tmp_path)
    _mirror_sidechain_across_backbone_plane(path, residue=1)
    result = reconstruct_pdb_fail_closed_v6(path, "A")
    assert result.status == "rejected"
    stereo = result.output_evidence["evidence_dimensions"]["stereochemistry"]
    assert stereo["passed"] is False
    assert any(
        evidence.get("passed") is False
        for evidence in stereo["coordinate_stereochemistry"]
    )


def test_v6_stereochemistry_dimension_requires_assembled_coordinate_audit():
    evidence = {
        "library_chemistry_unique": True,
        "atom_mapping_complete": True,
        "atom_mapping_unique": True,
        "residue_evidence": [{
            "unified_source": "core",
            "monomer_graph_sha256": "0" * 64,
            "unmapped_template_heavy_atom_count": 0,
            "template_stereo_complete": True,
            "template_unassigned_stereocenter_indices": [],
            "coordinate_stereochemistry": {"passed": True},
        }],
        "assembled_coordinate_stereochemistry": {
            "passed": False,
            "reason": "assembled_coordinate_stereochemistry_mismatch",
            "mismatch_graph_atom_indices": [4],
        },
        "emergent_stereochemistry": {
            "passed": True,
            "emergent_unassigned_center_count": 0,
        },
    }

    _library, _mapping, stereochemistry = _mapping_dimensions(evidence)

    assert stereochemistry["passed"] is False
    assert stereochemistry["assembled_coordinate_stereochemistry"] == (
        evidence["assembled_coordinate_stereochemistry"]
    )


def test_v6_stereochemistry_dimension_rejects_coordinate_only_emergent_center():
    evidence = {
        "library_chemistry_unique": True,
        "atom_mapping_complete": True,
        "atom_mapping_unique": True,
        "residue_evidence": [{
            "unified_source": "core",
            "monomer_graph_sha256": "0" * 64,
            "unmapped_template_heavy_atom_count": 0,
            "template_stereo_complete": True,
            "template_unassigned_stereocenter_indices": [],
            "coordinate_stereochemistry": {"passed": True},
        }],
        "assembled_coordinate_stereochemistry": {"passed": True},
        "emergent_stereochemistry": {
            "passed": False,
            "reason": "emergent_stereochemistry_requires_noncoordinate_authority",
            "emergent_unassigned_graph_atom_indices": [7],
            "coordinate_assignment_applied": False,
        },
    }

    _library, _mapping, stereochemistry = _mapping_dimensions(evidence)

    assert stereochemistry["passed"] is False
    assert stereochemistry["emergent_stereochemistry"] == (
        evidence["emergent_stereochemistry"]
    )


def test_v6_novel_local_monomer_reconstructs_only_as_repaired(tmp_path):
    path = _write_cyclic_peptide_with_novel_nnaa(tmp_path)
    result = reconstruct_pdb_fail_closed_v6(path, "A")
    assert result.status == "success", (
        result.rejection_reason,
        result.warning_codes,
        result.input_evidence.get("local_monomer_bootstrap"),
    )
    assert result.support_status == "repaired"
    assert not result.qualified_success
    assert "LOCALLY_INFERRED_MONOMER" in result.repair_codes
    bootstrap = result.input_evidence["local_monomer_bootstrap"]
    assert len(bootstrap["derived_symbols"]) == 1
    assert bootstrap["pdb_aliases"][0]["pdb_resname"] == "ZZQ"
    assert bootstrap["persistent_writes"] == 0
    assert result.output_evidence["adjudication_class"] == "locally_inferred"
    assert result.output_evidence["reconstruction_mode"] == "locally_inferred"
    assert _map_utils.resolve_pdb_alias("ZZQ") is None


def test_local_monomer_evidence_does_not_require_symbolic_second_path(tmp_path):
    path = _write_cyclic_peptide_with_novel_nnaa(tmp_path)
    result = reconstruct_pdb_fail_closed_v6(path, "A")

    assert result.status == "success", result.rejection_reason
    path_evidence = {
        route["route"]: route["evidence_dimensions_input"]
        for route in result.route_results
        if route.get("route") in {"a", "e"}
        and isinstance(route.get("evidence_dimensions_input"), dict)
    }
    dimension = _local_monomer_evidence_consistency_dimension(
        [], result.input_evidence, path_evidence["a"], result.output_inchikey
    )
    assert dimension["passed"] is True
    assert dimension["symbolic_reassemblies"] == []
    assert dimension["symbolic_cross_check_required"] is False


def test_local_monomer_evidence_rejects_candidate_template_graph_drift():
    dimension = _local_monomer_evidence_consistency_dimension(
        [],
        {"local_monomer_bootstrap": {"inference_results": [{
            "status": "unique",
            "pdb_resname": "ZZN",
            "residue_key": ["ZZN", 2, True],
            "candidate_smiles": "N[C@@H](CC#N)C(=O)O",
            "candidate_graph_count": 1,
            "r3_port": None,
        }]}},
        {"residue_evidence": [{
            "residue_key": ["ZZN", 2, True],
            "free_monomer_graph_sha256": "0" * 64,
            "mapping_complete": True,
            "mapping_unique": True,
        }]},
        "QUALIFIED-WHOLE-PEPTIDE-INCHIKEY",
    )
    assert dimension["passed"] is False


def test_embedded_ccd_evidence_requires_bound_coordinate_audit_without_crashing():
    smiles = "N[C@@H](CO)C(=O)O"
    molecule = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    graph_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    full_key = Chem.MolToInchiKey(molecule)
    snapshot_hash = "1" * 64
    source_hash = "2" * 64
    inference = {
        "status": "unique",
        "pdb_resname": "ZZC",
        "residue_key": ["ZZC", 2, True],
        "candidate_smiles": smiles,
        "candidate_graph_count": 1,
        "r3_port": None,
        "evidence": {
            "resolution_mode": "embedded_mmcif_chem_comp",
            "released": True,
            "component_snapshot_sha256": snapshot_hash,
            "component_source_input_sha256": source_hash,
            "embedded_chem_comp_resolution": {
                "component_snapshot_sha256": snapshot_hash,
                "source_input_sha256": source_hash,
                "full_inchikey": full_key,
                "observed_connectivity_exact": True,
                "stereochemistry_source": "_chem_comp_atom.pdbx_stereo_config",
                "bond_order_source": "_chem_comp_bond.value_order",
            },
        },
    }
    bootstrap = {
        "inference_results": [inference],
        "derived_manifests": [{
            "component_snapshot_sha256": snapshot_hash,
            "component_source_input_sha256": source_hash,
            "resolution_mode": "embedded_mmcif_chem_comp",
        }],
    }
    path_evidence = {"a": {"residue_evidence": [{
        "residue_key": ["ZZC", 2, True],
        "free_monomer_graph_sha256": graph_hash,
        "mapping_complete": True,
        "mapping_unique": True,
    }]}}

    missing = _local_monomer_evidence_consistency_dimension(
        [], {"local_monomer_bootstrap": bootstrap}, path_evidence["a"], full_key
    )
    assert missing["passed"] is False
    assert missing["inference_checks"][0]["direct_match_checks"]["checks"][
        "component_present_in_coordinate_audit"
    ] is False

    bound_input = {
        "local_monomer_bootstrap": bootstrap,
        "coordinate_input": {
            "embedded_chem_comp_source_payload_sha256": source_hash,
            "embedded_chem_comp_templates": {"ZZC": {
                "component_snapshot_sha256": snapshot_hash,
                "source_input_sha256": source_hash,
            }},
        },
    }
    bound = _local_monomer_evidence_consistency_dimension(
        [], bound_input, path_evidence["a"], full_key
    )
    assert bound["passed"] is True


def test_v6_novel_r3_requires_exact_anchor_without_second_path_count(tmp_path):
    path = _write_disulfide_peptide_with_novel_thiol(tmp_path)
    result = reconstruct_pdb_fail_closed_v6(path, "A")
    assert result.status == "success", (
        result.rejection_reason,
        result.warning_codes,
        result.input_evidence.get("local_monomer_bootstrap"),
        result.output_evidence,
    )
    assert result.support_status == "repaired"
    assert not result.qualified_success
    dimension = result.output_evidence["evidence_dimensions"][
        "local_monomer_evidence_consistency"
    ]
    assert dimension["passed"]
    assert dimension["symbolic_cross_check_required"] is False
    assert dimension["inference_checks"] == [
        {
            **dimension["inference_checks"][0],
            "passed": True,
        }
    ]
    assert dimension["inference_checks"][0][
        "mapped_template_atom_index"
    ] == dimension["inference_checks"][0]["r3_anchor_template_atom_index"]
    assert result.output_evidence["adjudication_class"] == "locally_inferred"
    assert _map_utils.resolve_pdb_alias("ZZS") is None


def test_v6_rejects_ambiguous_atom_mapping(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    evidence = deepcopy(evidence)
    evidence["atom_mapping_unique"] = False
    evidence["residue_evidence"][0]["mapping_unique"] = False
    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": evidence}, pdb_path=path, chain_id="A"
    )
    assert result.status == "rejected"
    assert result.warning_codes == ["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"]
    assert not result.output_evidence["evidence_dimensions"]["atom_mapping"]["passed"]


def test_v6_rejects_unproven_closure_counterfactual(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    evidence = deepcopy(evidence)
    evidence["all_closures_identity_determining"] = False
    evidence["closure_evidence"][0]["identity_changes_when_removed"] = False
    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": evidence}, pdb_path=path, chain_id="A"
    )
    assert result.status == "rejected"
    assert not result.output_evidence["evidence_dimensions"][
        "closure_identity_trace"
    ]["passed"]


@pytest.mark.parametrize(
    "tamper",
    [
        "monomer_graph_sha256",
        "unified_source",
        "serial_to_template_atom_index",
        "cropped_residue_evidence",
    ],
)
def test_v6_rejects_residue_mapping_ledger_tampering(tmp_path, tamper):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    evidence = deepcopy(evidence)

    if tamper == "cropped_residue_evidence":
        evidence["residue_evidence"] = evidence["residue_evidence"][:-1]
    else:
        residue = evidence["residue_evidence"][0]
        if tamper == "monomer_graph_sha256":
            residue[tamper] = "0" * 64
        elif tamper == "unified_source":
            residue[tamper] = "forged"
        else:
            serial = next(iter(residue[tamper]))
            residue[tamper][serial] = 999999

    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": evidence}, pdb_path=path, chain_id="A"
    )

    assert result.status == "rejected"
    binding = result.output_evidence["evidence_dimensions"][
        "mapping_evidence_binding"
    ]
    assert binding["passed"] is False
    assert binding["route_audits"][0]["residue_mapping_ledger"]["passed"] is False


def test_v6_rejects_forged_terminal_r2_ledger_for_head_to_tail_input(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    evidence = deepcopy(evidence)
    evidence["terminal_r2_materialization"] = {
        "status": "materialized_from_R2_OH_default",
        "occupied_ports": [],
    }
    terminal = _terminal_r2_ledger_audit(evidence, input_evidence)
    assert terminal["head_to_tail"] is True
    assert terminal["passed"] is False

    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": evidence}, pdb_path=path, chain_id="A"
    )

    assert result.status == "rejected"
    assert result.output_evidence["evidence_dimensions"][
        "path_evidence_fresh_replay"
    ]["passed"] is False


@pytest.mark.parametrize(
    "tamper",
    ["status", "missing_carbon_pdb_serial", "missing_terminal_position"],
)
def test_v6_rejects_free_terminal_r2_ledger_tampering(tmp_path, tamper):
    path = _write_explicit_disulfide_tripeptide(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    evidence = deepcopy(evidence)
    terminal = evidence["terminal_r2_materialization"]
    assert terminal["status"] == "materialized_from_R2_OH_default"

    if tamper == "status":
        terminal["status"] = "explicit_terminal_cap_present"
    elif tamper == "missing_carbon_pdb_serial":
        terminal.pop("carbon_pdb_serial")
    else:
        terminal.pop("terminal_position")

    audit = _terminal_r2_ledger_audit(evidence, input_evidence)
    assert audit["passed"] is False
    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": evidence}, pdb_path=path, chain_id="A"
    )
    assert result.status == "rejected"
    assert result.output_evidence["evidence_dimensions"][
        "path_evidence_fresh_replay"
    ]["passed"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "endpoint_serial",
        "closure_id",
        "final_graph_atom_index",
        "pdb_resseq",
        "resname",
        "graph_bond_type",
    ],
)
def test_v6_rejects_closure_endpoint_or_identity_tampering(tmp_path, tamper):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    evidence = deepcopy(evidence)
    closure = evidence["closure_evidence"][0]
    if tamper == "endpoint_serial":
        closure["endpoint_1"]["pdb_serial"] += 1000
    elif tamper == "closure_id":
        closure["closure_id"] = "forged-closure-id"
    elif tamper == "final_graph_atom_index":
        closure["endpoint_1"][tamper] += 1000
    elif tamper == "pdb_resseq":
        closure["endpoint_1"][tamper] += 1000
    elif tamper == "resname":
        closure["endpoint_1"][tamper] = "BAD"
    else:
        closure[tamper] = "DOUBLE"

    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": evidence}, pdb_path=path, chain_id="A"
    )

    assert result.status == "rejected"
    trace = result.output_evidence["evidence_dimensions"][
        "closure_identity_trace"
    ]
    assert trace["passed"] is False
    assert trace["route_traces"][0]["trace_ledger_passed"] is False


@pytest.mark.parametrize("mode", ["forged", None])
def test_v6_rejects_unknown_or_missing_local_resolution_mode(tmp_path, mode):
    path = _write_cyclic_peptide_with_novel_nnaa(tmp_path)
    result = reconstruct_pdb_fail_closed_v6(path, "A")
    assert result.status == "success", result.rejection_reason
    tampered_input = deepcopy(result.input_evidence)
    evidence = tampered_input["local_monomer_bootstrap"]["inference_results"][0][
        "evidence"
    ]
    if mode is None:
        evidence.pop("resolution_mode", None)
    else:
        evidence["resolution_mode"] = mode
    path_evidence = {
        route["route"]: route["evidence_dimensions_input"]
        for route in result.route_results
        if route.get("route") in {"a", "e"}
        and isinstance(route.get("evidence_dimensions_input"), dict)
    }

    dimension = _local_monomer_evidence_consistency_dimension(
        result.route_results,
        tampered_input,
        path_evidence["a"],
        result.output_inchikey,
    )

    assert dimension["passed"] is False
    direct = dimension["inference_checks"][0]["direct_match_checks"]
    assert direct["passed"] is False
    assert direct["reason"] == "missing_or_unsupported_resolution_mode"


@pytest.mark.parametrize(("source_route", "target_route"), [("a", "e"), ("e", "a")])
def test_v6_rejects_path_evidence_relabelled_between_a_and_e(
    tmp_path, source_route, target_route
):
    path = _write_explicit_trialanine(tmp_path)
    helper = _single_a_inputs if source_route == "a" else _single_e_inputs
    row, input_evidence, evidence = helper(path)
    forged_row = {**row, "route": target_route}
    forged_evidence = {
        **deepcopy(evidence),
        "route": target_route,
        "geometry_inference_enabled": target_route == "e",
    }

    result = _adjudicate_evidence_dimensions(
        [forged_row], input_evidence, {target_route: forged_evidence},
        pdb_path=path, chain_id="A",
    )

    assert result.status == "rejected"
    replay = result.output_evidence["evidence_dimensions"][
        "path_evidence_fresh_replay"
    ]
    assert replay["passed"] is False
    assert replay["route_audits"][0]["checks"][
        "supplied_evidence_exactly_replayed"
    ] is False


@pytest.mark.parametrize("tamper", ["boolean_count", "float_index"])
def test_v6_fresh_replay_distinguishes_json_numeric_types(tmp_path, tamper):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    forged_evidence = deepcopy(evidence)
    residue = forged_evidence["residue_evidence"][0]
    if tamper == "boolean_count":
        residue["deferred_consumed_r3_count"] = False
    else:
        serial = next(iter(residue["serial_to_template_atom_index"]))
        residue["serial_to_template_atom_index"][serial] = float(
            residue["serial_to_template_atom_index"][serial]
        )

    result = _adjudicate_evidence_dimensions(
        [row], input_evidence, {"a": forged_evidence},
        pdb_path=path, chain_id="A",
    )

    assert result.status == "rejected"
    replay = result.output_evidence["evidence_dimensions"][
        "path_evidence_fresh_replay"
    ]
    assert replay["passed"] is False
    assert replay["route_audits"][0]["checks"][
        "supplied_evidence_exactly_replayed"
    ] is False


def test_v6_rejects_conflicting_successful_chemical_candidates(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    conflict = {
        "route": "b",
        "status": "success",
        "output_smiles": "C1CCCCC1",
        "output_inchikey": "XDTMQSROBMDMFD-UHFFFAOYSA-N",
        "warning_codes": [],
    }
    result = _adjudicate_evidence_dimensions(
        [row, conflict], input_evidence, {"a": evidence},
        pdb_path=path, chain_id="A",
    )
    assert result.status == "rejected"
    assert result.warning_codes == [
        "V6_CONFLICTING_SUCCESSFUL_CANDIDATE_IDENTITIES"
    ]
    assessment = result.output_evidence["candidate_assessment"]
    assert assessment["mode"] == "candidate_ensemble"
    assert assessment["candidate_count"] == 2
    assert assessment["unattended_selection_qualified"] is False
    assert assessment["exploratory_only"] is True
    assert (
        "standard_inchikey_equivalence"
        in assessment["conflict_dimensions"]
    )


def test_v6_c3_0016_preserves_input_repairs_on_candidate_conflict(tmp_path):
    """A strict rejection must retain repair provenance from input auditing."""
    path = _write_explicit_trialanine(tmp_path)
    row, input_evidence, evidence = _single_a_inputs(path)
    input_evidence["repair_codes"] = [
        "CONNECTIVITY_INFERRED_FROM_COORDINATES",
        "CONNECTIVITY_INFERRED_FROM_COORDINATES",
    ]
    conflict = {
        "route": "b",
        "status": "success",
        "output_smiles": "C1CCCCC1",
        "output_inchikey": "XDTMQSROBMDMFD-UHFFFAOYSA-N",
        "warning_codes": [],
    }

    result = _adjudicate_evidence_dimensions(
        [row, conflict], input_evidence, {"a": evidence},
        pdb_path=path, chain_id="A",
    )

    assert result.status == "rejected"
    assert result.repair_codes == ["CONNECTIVITY_INFERRED_FROM_COORDINATES"]
    assert result.warning_codes == [
        "V6_CONFLICTING_SUCCESSFUL_CANDIDATE_IDENTITIES"
    ]


def test_v6_successful_f_and_h_rows_are_diagnostic_only(tmp_path):
    path = _write_explicit_trialanine(tmp_path)
    _row, input_evidence, _evidence = _single_a_inputs(path)
    diagnostic = [{
        "route": route,
        "status": "success",
        "output_smiles": "C1CCCCC1",
        "output_inchikey": "XDTMQSROBMDMFD-UHFFFAOYSA-N",
        "warning_codes": [],
    } for route in ("f", "h")]
    result = _adjudicate_evidence_dimensions(diagnostic, input_evidence, {})
    assert result.status == "rejected"
    assert result.warning_codes == ["V6_NO_SUCCESSFUL_CHEMICAL_CANDIDATE"]


def test_v6_rejects_nonempty_persistent_overlay_before_reconstruction(
    tmp_path, monkeypatch
):
    path = _write_explicit_trialanine(tmp_path)
    monkeypatch.setattr(
        _map_utils,
        "_persistent_derived_by_symbol",
        {"LEAK": {"symbol": "LEAK", "smiles_canonical": "CC"}},
    )
    result = reconstruct_pdb_fail_closed_v6(
        path, "A", require_empty_persistent_overlay=True
    )
    assert result.status == "rejected"
    assert result.warning_codes == ["V6_PERSISTENT_OVERLAY_NOT_EMPTY"]
    assert result.input_evidence["persistent_overlay_audit"]["status"] == "nonempty"


def test_v6_geometry_only_closure_cannot_authorize_success(tmp_path):
    path = tmp_path / "geometry-disulfide.pdb"
    rows = []
    serial = 1
    for residue, offset, sulfur_x in ((1, 0.0, 0.0), (2, 10.0, 2.0)):
        for name, element, xyz in (
            ("N", "N", (offset, 0.0, 0.0)),
            ("CA", "C", (offset + 1.4, 0.0, 0.0)),
            ("CB", "C", (offset + 1.4, 1.5, 0.0)),
            ("C", "C", (offset + 2.8, 0.0, 0.0)),
            ("O", "O", (offset + 3.9, 0.0, 0.0)),
            ("SG", "S", (sulfur_x, 10.0, 0.0)),
        ):
            rows.append(_pdb_atom(
                serial, name, residue, *xyz, residue_name="CYS", element=element
            ))
            serial += 1
    path.write_text("\n".join([*rows, "END"]) + "\n", encoding="ascii")
    result = reconstruct_pdb_fail_closed_v6(path, "A")
    assert result.status == "rejected"
    assert not result.qualified_success


def test_v6_ambiguous_explicit_carbon_closure_is_not_supported(tmp_path):
    path = tmp_path / "ambiguous-carbon-link.pdb"
    rows = []
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0), (3, 20.0), (4, 30.0)):
        for name, element, delta in (
            ("N", "N", 0.0), ("CA", "C", 1.4), ("CB", "C", 1.8),
            ("C", "C", 2.8), ("O", "O", 3.9),
        ):
            rows.append(_pdb_atom(
                serial, name, residue, offset + delta, 0.0, 0.0,
                element=element,
            ))
            serial += 1
    path.write_text(
        "\n".join([
            _link_record("CB", "ALA", 1, "CB", "ALA", 4),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    result = reconstruct_pdb_fail_closed_v6(path, "A")
    assert result.status == "not_supported"
    assert result.warning_codes == ["V6_EXPLICIT_CLOSURE_BOND_ORDER_AMBIGUOUS"]


def test_v6_outer_early_rejection_preserves_input_repairs(tmp_path, monkeypatch):
    path = tmp_path / "ambiguous-carbon-link-with-repair.pdb"
    rows = []
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0), (3, 20.0), (4, 30.0)):
        for name, element, delta in (
            ("N", "N", 0.0), ("CA", "C", 1.4), ("CB", "C", 1.8),
            ("C", "C", 2.8), ("O", "O", 3.9),
        ):
            rows.append(_pdb_atom(
                serial, name, residue, offset + delta, 0.0, 0.0,
                element=element,
            ))
            serial += 1
    path.write_text(
        "\n".join([
            _link_record("CB", "ALA", 1, "CB", "ALA", 4),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )

    validation = v5.validate_pdb_reconstruction_input_v5(path, "A")
    assert validation.accepted
    validation.context["repair_codes"] = [
        "CONNECTIVITY_INFERRED_FROM_COORDINATES",
        "CONNECTIVITY_INFERRED_FROM_COORDINATES",
    ]
    monkeypatch.setattr(
        v5,
        "validate_pdb_reconstruction_input_v5",
        lambda *_args, **_kwargs: validation,
    )

    result = reconstruct_pdb_fail_closed_v6(path, "A")

    assert result.status == "not_supported"
    assert result.warning_codes == ["V6_EXPLICIT_CLOSURE_BOND_ORDER_AMBIGUOUS"]
    assert result.repair_codes == ["CONNECTIVITY_INFERRED_FROM_COORDINATES"]
    assert result.output_evidence["candidate_assessment"]["result_context"][
        "repair_codes"
    ] == ["CONNECTIVITY_INFERRED_FROM_COORDINATES"]
