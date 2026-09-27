from __future__ import annotations

import gzip
import hashlib
import csv
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import gemmi
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import cycpep_master.core.structure_io as structure_io
import cycpep_master.pipeline as pipeline
from cycpep_master.core.structure_io import (
    CoordinateInputError,
    mmcif_to_pdb,
    prepare_coordinate_input,
)
from cycpep_master.core.pdb_parser import standard_pdb_atom_name_map
from cycpep_master.core.cyclization import detect_cyclization
from cycpep_master.export.conformer import pdb_to_mol2
from cycpep_master.paths.residue_template_factory import get_residue_template
from cycpep_master.pipeline import run_batch
from cycpep_master.remediation_v6 import reconstruct_structure_fail_closed_v6


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _candidate_assessment_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (PACKAGE_ROOT / "schemas" / "candidate_assessment.schema.json")
        .read_text(encoding="utf-8")
    )
    return jsonschema.Draft202012Validator(schema)


def _atom(serial, name, residue, x, element, y=0.0, z=0.0):
    return (
        f"ATOM  {serial:5d} {name:>4s} ALA A{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def _link(*, atom1="N", residue1=1, atom2="C", residue2=3):
    record = list(" " * 80)
    record[0:4] = "LINK"
    record[12:16] = f"{atom1:>4s}"
    record[17:20] = "ALA"
    record[21] = "A"
    record[22:26] = f"{residue1:4d}"
    record[42:46] = f"{atom2:>4s}"
    record[47:50] = "ALA"
    record[51] = "A"
    record[52:56] = f"{residue2:4d}"
    return "".join(record)


def _source_link(
    *,
    atom1: str,
    residue_name1: str,
    residue1: int,
    atom2: str,
    residue_name2: str,
    residue2: int,
    chain: str = "A",
) -> str:
    record = list(" " * 80)
    record[0:4] = "LINK"
    record[12:16] = f"{atom1:>4s}"
    record[17:20] = f"{residue_name1:>3s}"
    record[21] = chain
    record[22:26] = f"{residue1:4d}"
    record[42:46] = f"{atom2:>4s}"
    record[47:50] = f"{residue_name2:>3s}"
    record[51] = chain
    record[52:56] = f"{residue2:4d}"
    return "".join(record)


def _source_ssbond(
    *,
    residue1: int,
    residue2: int,
    chain: str = "A",
) -> str:
    record = list(" " * 80)
    record[0:6] = "SSBOND"
    record[7:10] = f"{1:3d}"
    record[11:14] = "CYS"
    record[15] = chain
    record[17:21] = f"{residue1:4d}"
    record[25:28] = "CYS"
    record[29] = chain
    record[31:35] = f"{residue2:4d}"
    return "".join(record)


def _source_modres(
    *, modified_name: str, residue_number: int, standard_name: str, chain: str = "A"
) -> str:
    record = list(" " * 80)
    record[0:6] = "MODRES"
    record[7:10] = "  1"
    record[12:15] = f"{modified_name:>3s}"
    record[16] = chain
    record[18:22] = f"{residue_number:4d}"
    record[24:27] = f"{standard_name:>3s}"
    record[29:40] = "source-bound"
    return "".join(record)


def _write_trialanine(path: Path) -> Path:
    template = get_residue_template("ALA")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=23) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    names = {
        index: name
        for name, index in standard_pdb_atom_name_map(
            "ALA", template.smiles
        ).items()
    }
    assert len(names) == molecule.GetNumAtoms()

    rows = [_link()]
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0), (3, 20.0)):
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(_atom(
                serial,
                names[atom.GetIdx()],
                residue,
                point.x + offset,
                atom.GetSymbol(),
                point.y,
                point.z,
            ))
            serial += 1
    path.write_text("\n".join([*rows, "END"]) + "\n", encoding="ascii")
    return path


def _to_mmcif(pdb: Path, cif: Path, *, chain_name: str = "A") -> Path:
    structure = gemmi.read_structure(str(pdb))
    structure[0][0].name = chain_name
    for connection in structure.connections:
        connection.partner1.chain_name = chain_name
        connection.partner2.chain_name = chain_name
    structure.make_mmcif_document().write_file(str(cif))
    return cif


def _write_shuffled_unknown_source(
    path: Path, *, include_modres: bool = False, modres_standard_name: str = "ALA"
) -> Path:
    source = _write_trialanine(path)
    lines = source.read_text(encoding="ascii").splitlines()
    blocks = {1: [], 2: [], 3: []}
    prefix = []
    for line in lines:
        if line.startswith("ATOM") and int(line[22:26]) == 2:
            line = "HETATM" + line[6:17] + "ZZZ" + line[20:]
        if line.startswith(("ATOM  ", "HETATM")):
            blocks[int(line[22:26])].append(line)
        elif line != "END":
            prefix.append(line)
    source.write_text(
        "\n".join(
            [
                "SEQRES   1 A    3  ALA DPN ALA",
                *(
                    [_source_modres(
                        modified_name="ZZZ",
                        residue_number=2,
                        standard_name=modres_standard_name,
                    )]
                    if include_modres else []
                ),
                *prefix,
                *(line for residue in (3, 1, 2) for line in blocks[residue]),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    return source


def _shuffle_mmcif_atom_site_rows(
    path: Path, *, explicit_polymer_membership: bool = False
) -> Path:
    document = gemmi.cif.read_file(str(path))
    table = document.sole_block().find_mmcif_category("_atom_site.")
    assert table.loop is not None
    tags = list(table.tags)
    sequence_index = tags.index("_atom_site.auth_seq_id")
    label_seq_index = tags.index("_atom_site.label_seq_id")
    width = table.loop.width()
    rows = [
        table.loop.values[index:index + width]
        for index in range(0, len(table.loop.values), width)
    ]
    if explicit_polymer_membership:
        label_entity_index = tags.index("_atom_site.label_entity_id")
        label_asym_index = tags.index("_atom_site.label_asym_id")
        for row in rows:
            row[label_entity_index] = "A"
            row[label_asym_index] = "A"
            row[label_seq_index] = row[sequence_index]
    ordered = sorted(rows, key=lambda row: (int(row[sequence_index]),))
    # Move all rows as residue blocks, retaining the intentionally shuffled
    # order from the PDB fixture rather than relying on Gemmi iteration order.
    rank = {3: 0, 1: 1, 2: 2}
    ordered = sorted(ordered, key=lambda row: rank[int(row[sequence_index])])
    for index in range(len(rows) - 1, -1, -1):
        table.remove_row(index)
    for row in ordered:
        table.loop.add_row(row)
    document.write_file(str(path))
    return path


def _to_mmcif_with_cross_chain_same_auth_seq(pdb: Path, cif: Path) -> Path:
    _to_mmcif(pdb, cif)
    document = gemmi.cif.read_file(str(cif))
    block = document.sole_block()

    entity_table = block.find_mmcif_category("_entity.")
    entity_tags = list(entity_table.tags)
    entity_id_index = entity_tags.index("_entity.id")
    entity_type_index = entity_tags.index("_entity.type")
    if not any(str(row[entity_id_index]).strip() == "B" for row in entity_table):
        entity_table.loop.add_row(
            [
                "B" if index == entity_id_index else (
                    "polymer" if index == entity_type_index else "?"
                )
                for index in range(len(entity_tags))
            ]
        )

    entity_poly_seq = block.find_mmcif_category("_entity_poly_seq.")
    poly_tags = list(entity_poly_seq.tags)
    poly_entity_index = poly_tags.index("_entity_poly_seq.entity_id")
    for row in list(entity_poly_seq):
        if str(row[poly_entity_index]).strip() != "A":
            continue
        clone = list(row)
        clone[poly_entity_index] = "B"
        entity_poly_seq.loop.add_row(clone)

    struct_asym_table = block.find_mmcif_category("_struct_asym.")
    if len(struct_asym_table) == 0:
        struct_loop = block.init_loop("_struct_asym.", ["id", "entity_id"])
        struct_tags = list(struct_loop.tags)
        struct_rows = []
    else:
        struct_loop = struct_asym_table.loop
        struct_tags = list(struct_asym_table.tags)
        struct_rows = list(struct_asym_table)
    struct_id_index = struct_tags.index("_struct_asym.id")
    struct_entity_index = struct_tags.index("_struct_asym.entity_id")
    if not any(str(row[struct_id_index]).strip() == "B" for row in struct_rows):
        struct_loop.add_row(
            [
                "B" if index == struct_id_index else (
                    "B" if index == struct_entity_index else "?"
                )
                for index in range(len(struct_tags))
            ]
        )

    atom_table = block.find_mmcif_category("_atom_site.")
    atom_tags = list(atom_table.tags)
    atom_id_index = atom_tags.index("_atom_site.id")
    label_asym_index = atom_tags.index("_atom_site.label_asym_id")
    label_entity_index = atom_tags.index("_atom_site.label_entity_id")
    label_seq_index = atom_tags.index("_atom_site.label_seq_id")
    auth_seq_index = atom_tags.index("_atom_site.auth_seq_id")
    auth_asym_index = atom_tags.index("_atom_site.auth_asym_id")
    rows = [list(row) for row in atom_table]
    serials = []
    for row in rows:
        try:
            serials.append(int(str(row[atom_id_index]).strip()))
        except (TypeError, ValueError):
            pass
    next_serial = max(serials, default=0) + 1000
    for offset, row in enumerate(rows):
        clone = list(row)
        clone[atom_id_index] = str(next_serial + offset)
        clone[label_asym_index] = "B"
        clone[label_entity_index] = "B"
        clone[label_seq_index] = str(row[auth_seq_index]).strip()
        clone[auth_asym_index] = "B"
        atom_table.loop.add_row(clone)
    document.write_file(str(cif))
    return cif


def _write_unknown_source_with_extra_ligand(path: Path) -> Path:
    source = _write_trialanine(path)
    lines = source.read_text(encoding="ascii").splitlines()
    blocks = {1: [], 2: [], 3: []}
    prefix = []
    for line in lines:
        if line.startswith("ATOM"):
            blocks[int(line[22:26])].append(line)
        elif line != "END":
            prefix.append(line)
    ligand = []
    for serial, line in enumerate(blocks[3], start=16):
        line = (
            "HETATM"
            + f"{serial:5d}"
            + line[11:17]
            + "ZZZ"
            + line[20:22]
            + "   4"
            + line[26:]
        )
        ligand.append(line)
    source.write_text(
        "\n".join(
            [
                "SEQRES   1 A    3  ALA DPN ALA",
                *prefix,
                *blocks[1],
                *blocks[2],
                *blocks[3],
                *ligand,
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    return source


def _with_embedded_alanine_component(cif: Path, component_id="ZZA") -> Path:
    document = gemmi.cif.read_file(str(cif))
    block = document.sole_block()
    component_table = block.find_mmcif_category("_chem_comp.")
    assert component_table.loop is not None
    component_tags = list(component_table.tags)
    id_column = component_tags.index("_chem_comp.id")
    type_column = component_tags.index("_chem_comp.type")
    matching_rows = [
        row for row in component_table
        if str(row[id_column]).upper() == component_id
    ]
    if matching_rows:
        assert len(matching_rows) == 1
        matching_rows[0][type_column] = "'L-peptide linking'"
    else:
        component_row = ["?"] * len(component_tags)
        component_row[id_column] = component_id
        component_row[type_column] = "'L-peptide linking'"
        component_table.loop.add_row(component_row)

    atom_loop = block.init_loop("_chem_comp_atom.", [
        "comp_id",
        "atom_id",
        "type_symbol",
        "pdbx_aromatic_flag",
        "pdbx_stereo_config",
    ])
    for name, element, stereo in (
        ("N", "N", "N"),
        ("H", "H", "N"),
        ("H2", "H", "N"),
        ("CA", "C", "S"),
        ("HA", "H", "N"),
        ("C", "C", "N"),
        ("O", "O", "N"),
        ("OXT", "O", "N"),
        ("HXT", "H", "N"),
        ("CB", "C", "N"),
        ("HB1", "H", "N"),
        ("HB2", "H", "N"),
        ("HB3", "H", "N"),
    ):
        atom_loop.add_row([component_id, name, element, "N", stereo])

    bond_loop = block.init_loop("_chem_comp_bond.", [
        "comp_id",
        "atom_id_1",
        "atom_id_2",
        "value_order",
        "pdbx_aromatic_flag",
    ])
    for left, right, order in (
        ("N", "H", "sing"),
        ("N", "H2", "sing"),
        ("N", "CA", "sing"),
        ("CA", "HA", "sing"),
        ("CA", "C", "sing"),
        ("CA", "CB", "sing"),
        ("C", "O", "doub"),
        ("C", "OXT", "sing"),
        ("OXT", "HXT", "sing"),
        ("CB", "HB1", "sing"),
        ("CB", "HB2", "sing"),
        ("CB", "HB3", "sing"),
    ):
        bond_loop.add_row([component_id, left, right, order, "N"])
    document.write_file(str(cif))
    return cif


def _to_mmcif_with_unknown_middle_alanine(pdb: Path, cif: Path) -> Path:
    lines = []
    for line in pdb.read_text(encoding="ascii").splitlines():
        if line.startswith("ATOM") and int(line[22:26]) == 2:
            line = "HETATM" + line[6:17] + "ZZA" + line[20:]
        lines.append(line)
    pdb.write_text("\n".join(lines) + "\n", encoding="ascii")
    return _with_embedded_alanine_component(_to_mmcif(pdb, cif))


def _to_multimodel_mmcif(
    pdb: Path,
    cif: Path,
    *,
    connection_models: tuple[int, int] | None,
) -> Path:
    structure = gemmi.read_structure(str(pdb))
    second_model = structure[0].clone()
    second_model.num = 2
    structure.add_model(second_model)
    document = structure.make_mmcif_document()
    table = document.sole_block().find_mmcif_category("_struct_conn.")
    assert len(table) == 1
    if connection_models is not None:
        assert table.loop is not None
        table.loop.add_columns(
            ["_struct_conn.pdbx_ptnr1_PDB_model_num"],
            str(connection_models[0]),
        )
        table.loop.add_columns(
            ["_struct_conn.pdbx_ptnr2_PDB_model_num"],
            str(connection_models[1]),
        )
    document.write_file(str(cif))
    return cif


def _to_mmcif_with_same_auth_entity_groups(
    pdb: Path,
    cif: Path,
    *,
    second_group_polymer: bool,
) -> Path:
    structure = gemmi.read_structure(str(pdb))
    chain = structure[0][0]
    for residue in chain:
        residue.subchain = "P"
        residue.entity_id = "1"
        residue.het_flag = "A"
    clones = [chain[0].clone()]
    if second_group_polymer:
        clones.append(chain[1].clone())
    for offset, residue in enumerate(clones, start=50):
        residue.seqid = gemmi.SeqId(offset, " ")
        residue.subchain = "X"
        residue.entity_id = "2"
        residue.het_flag = "A" if second_group_polymer else "H"
        chain.add_residue(residue)
    structure.make_mmcif_document().write_file(str(cif))
    return cif


def test_mmcif_projection_preserves_explicit_cycle_and_v6_identity(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif(pdb, tmp_path / "trialanine.cif")
    projected = tmp_path / "projected.pdb"
    audit = mmcif_to_pdb(cif, projected, "A")
    assert audit["materialized_explicit_connection_count"] == 1
    assert any(line.startswith("CONECT") for line in projected.read_text().splitlines())

    pdb_result = reconstruct_structure_fail_closed_v6(pdb, "A")
    cif_result = reconstruct_structure_fail_closed_v6(cif, "A")
    assert pdb_result.status == "success", pdb_result.rejection_reason
    assert cif_result.status == "success", cif_result.rejection_reason
    assert cif_result.output_inchikey == pdb_result.output_inchikey
    assert cif_result.path_used.startswith("MMCIF_PROJECTED:")
    assert cif_result.input_evidence["coordinate_input"]["source_format"] == "mmcif"


def test_known_embedded_component_exact_identity_gate_passes(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _with_embedded_alanine_component(
        _to_mmcif(pdb, tmp_path / "trialanine-with-ala-ccd.cif"),
        component_id="ALA",
    )

    result = reconstruct_structure_fail_closed_v6(cif, "A")

    assert result.status == "success", result.rejection_reason
    audit = result.input_evidence["known_residue_embedded_component_audit"]
    assert audit["status"] == "pass"
    assert audit["audited_residue_count"] == 3
    assert audit["exact_full_inchikey_count"] == 3
    assert audit["override_performed"] is False
    assert all(row["source_payload_bound"] for row in audit["rows"])


def test_known_embedded_component_stereo_conflict_rejects(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _with_embedded_alanine_component(
        _to_mmcif(pdb, tmp_path / "trialanine-conflicting-ala-ccd.cif"),
        component_id="ALA",
    )
    document = gemmi.cif.read_file(str(cif))
    atoms = document.sole_block().find_mmcif_category("_chem_comp_atom.")
    tags = list(atoms.tags)
    id_column = tags.index("_chem_comp_atom.atom_id")
    stereo_column = tags.index("_chem_comp_atom.pdbx_stereo_config")
    for row in atoms:
        if str(row[id_column]) == "CA":
            row[stereo_column] = "R"
    document.write_file(str(cif))

    result = reconstruct_structure_fail_closed_v6(cif, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["V6_LIBRARY_VS_EMBEDDED_COMPONENT_CONFLICT"]
    audit = result.input_evidence["known_residue_embedded_component_audit"]
    assert audit["status"] == "rejected"
    assert audit["conflict_count"] == 3
    assert {
        row["identity_relation"] for row in audit["rows"]
    } == {"stereochemistry_or_isotope_conflict"}


def test_mmcif_projection_selects_unique_polymer_label_entity(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif_with_same_auth_entity_groups(
        pdb,
        tmp_path / "same-auth-nonpolymer.cif",
        second_group_polymer=False,
    )
    with prepare_coordinate_input(cif, "A") as prepared:
        projected = gemmi.read_structure(str(prepared.pdb_path))
        assert [residue.seqid.num for residue in projected[0][0]] == [1, 2, 3]
        assert prepared.audit["label_entity_selection_mode"] == (
            "unique_polymer_label_entity"
        )
        assert prepared.audit["selected_label_subchain_id"] == "P"
        assert prepared.audit["selected_entity_id"] == "1"
        assert prepared.audit["excluded_nonselected_entity_residue_count"] == 1
        assert prepared.audit["excluded_nonselected_entity_heavy_atom_count"] > 0


def test_mmcif_projection_rejects_multiple_polymer_label_entities(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif_with_same_auth_entity_groups(
        pdb,
        tmp_path / "same-auth-two-polymers.cif",
        second_group_polymer=True,
    )
    with pytest.raises(
        CoordinateInputError,
        match="multiple reconstructable polymer label entities",
    ) as error:
        with prepare_coordinate_input(cif, "A"):
            pass
    assert error.value.code == "MMCIF_AUTH_CHAIN_ENTITY_AMBIGUOUS"
    assert error.value.not_supported is True


def test_mmcif_to_mol2_preserves_not_supported_coordinate_error(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif_with_same_auth_entity_groups(
        pdb,
        tmp_path / "same-auth-two-polymers.cif",
        second_group_polymer=True,
    )

    block, error = pdb_to_mol2(cif, chain_id="A", path="v6")

    assert block is None
    assert error is not None
    assert "not supported" in error
    assert "MMCIF_AUTH_CHAIN_ENTITY_AMBIGUOUS" in error


def test_mmcif_to_mol2_retains_embedded_component_authority(tmp_path):
    pdb = _write_trialanine(tmp_path / "unknown-middle.pdb")
    cif = _to_mmcif_with_unknown_middle_alanine(
        pdb, tmp_path / "unknown-middle.cif"
    )

    result = reconstruct_structure_fail_closed_v6(cif, "A")
    block, error = pdb_to_mol2(cif, chain_id="A", path="v6")

    assert result.status == "success", result.rejection_reason
    assert result.input_evidence["local_monomer_bootstrap"][
        "resolution_modes"
    ] == ["embedded_mmcif_chem_comp"]
    consistency = result.output_evidence["evidence_dimensions"][
        "local_monomer_evidence_consistency"
    ]
    assert consistency["passed"] is True
    assert consistency["inference_checks"][0]["direct_match_checks"][
        "materialization_binding"
    ] == "unified_library_alias"
    assert error is None
    assert block.startswith("@<TRIPOS>MOLECULE\n")
    assert "ZZA2" in block
    lines = block.splitlines()
    atom_start = lines.index("@<TRIPOS>ATOM") + 1
    atom_end = next(
        index
        for index in range(atom_start, len(lines))
        if lines[index].startswith("@<TRIPOS>")
    )
    atom_rows = [line.split() for line in lines[atom_start:atom_end]]
    assert sum(row[5] != "H" for row in atom_rows) == 15
    assert any(row[5] == "H" for row in atom_rows)


def test_two_residue_reverse_peptide_link_survives_projection(tmp_path):
    source = _write_trialanine(tmp_path / "source.pdb")
    atom_lines = [
        line
        for line in source.read_text(encoding="ascii").splitlines()
        if line.startswith(("ATOM", "HETATM")) and int(line[22:26]) <= 2
    ]
    source.write_text(
        "\n".join([
            _link(atom1="N", residue1=1, atom2="C", residue2=2),
            *atom_lines,
            "END",
        ]) + "\n",
        encoding="ascii",
    )

    raw = detect_cyclization(str(source), "A", allow_geometric_inference=False)
    assert [(bond.bond_type, bond.pos1, bond.pos2) for bond in raw.bonds] == [
        ("peptide", 1, 2)
    ]

    with prepare_coordinate_input(source, "A") as prepared:
        assert prepared.audit["materialized_explicit_connection_count"] == 1
        assert any(
            line.startswith("CONECT")
            for line in prepared.pdb_path.read_text(encoding="ascii").splitlines()
        )
        projected = detect_cyclization(
            str(prepared.pdb_path), "A", allow_geometric_inference=False
        )
        assert [(bond.bond_type, bond.pos1, bond.pos2) for bond in projected.bonds] == [
            ("peptide", 1, 2)
        ]


def test_mmcif_multichar_chain_is_projected_without_chain_loss(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif(pdb, tmp_path / "multichain-id.cif", chain_name="PEP")
    result = reconstruct_structure_fail_closed_v6(cif, "PEP")
    assert result.status == "success", result.rejection_reason
    coordinate = result.input_evidence["coordinate_input"]
    assert coordinate["selected_auth_chain_id"] == "PEP"
    assert coordinate["normalized_chain_id"] == "A"


def test_multimodel_mmcif_materializes_only_first_model_connection(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_multimodel_mmcif(
        pdb,
        tmp_path / "first-model-connection.cif",
        connection_models=(1, 1),
    )
    output = tmp_path / "projected.pdb"

    audit = mmcif_to_pdb(cif, output, "A")

    assert audit["source_model_count"] == 2
    assert audit["structured_connection_count"] == 1
    assert audit["first_model_structured_connection_count"] == 1
    assert audit["ignored_nonselected_model_structured_connection_count"] == 0
    assert audit["materialized_explicit_connection_count"] == 1


def test_multimodel_mmcif_ignores_connection_scoped_to_later_model(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_multimodel_mmcif(
        pdb,
        tmp_path / "later-model-connection.cif",
        connection_models=(2, 2),
    )
    output = tmp_path / "projected.pdb"

    audit = mmcif_to_pdb(cif, output, "A")

    assert audit["source_model_count"] == 2
    assert audit["structured_connection_count"] == 1
    assert audit["first_model_structured_connection_count"] == 0
    assert audit["ignored_nonselected_model_structured_connection_count"] == 1
    assert audit["materialized_explicit_connection_count"] == 0
    assert not any(
        line.startswith("CONECT")
        for line in output.read_text(encoding="ascii").splitlines()
    )


@pytest.mark.parametrize(
    ("connection_models", "error_code"),
    [((1, 2), "MMCIF_CROSS_MODEL_CONNECTION_NOT_SUPPORTED")],
)
def test_multimodel_mmcif_rejects_unsafe_connection_scope(
    tmp_path, connection_models, error_code
):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_multimodel_mmcif(
        pdb,
        tmp_path / "unsafe-connection-scope.cif",
        connection_models=connection_models,
    )
    output = tmp_path / "projected.pdb"

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == error_code
    assert exc_info.value.not_supported is True
    assert not output.exists()


def test_multimodel_mmcif_accepts_model_invariant_unscoped_connection(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_multimodel_mmcif(
        pdb,
        tmp_path / "model-invariant-unscoped.cif",
        connection_models=None,
    )
    output = tmp_path / "projected.pdb"

    audit = mmcif_to_pdb(cif, output, "A")

    assert audit["source_model_count"] == 2
    assert audit["model_invariant_unscoped_connection_count"] == 1
    assert audit["first_model_structured_connection_count"] == 1
    assert audit["materialized_explicit_connection_count"] == 1


def test_multimodel_mmcif_rejects_unscoped_connection_missing_in_later_model(
    tmp_path,
):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_multimodel_mmcif(
        pdb,
        tmp_path / "unscoped-missing-endpoint.cif",
        connection_models=None,
    )
    structure = gemmi.read_structure(str(cif))
    residue = structure[1]["A"][0]
    endpoint_index = next(
        index for index, atom in enumerate(residue) if atom.name.strip() == "N"
    )
    del residue[endpoint_index]
    structure.make_mmcif_document().write_file(str(cif))
    output = tmp_path / "projected.pdb"

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == "MMCIF_CONNECTION_MODEL_SCOPE_UNRESOLVED"
    assert exc_info.value.not_supported is True
    assert not output.exists()


def test_projection_rejects_source_output_alias_without_overwrite(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    source = _to_mmcif(pdb, tmp_path / "trialanine.cif")
    original = source.read_bytes()

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(source, source, "A")

    assert exc_info.value.code == "COORDINATE_PROJECTION_OUTPUT_CONFLICT"
    assert source.read_bytes() == original


def test_projection_rejects_hardlink_output_alias_without_overwrite(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    source = _to_mmcif(pdb, tmp_path / "trialanine.cif")
    output = tmp_path / "output-alias.cif"
    os.link(source, output)
    original = source.read_bytes()

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(source, output, "A")

    assert exc_info.value.code == "COORDINATE_PROJECTION_OUTPUT_CONFLICT"
    assert source.read_bytes() == original
    assert output.read_bytes() == original


def test_mmcif_gzip_uses_same_strict_entrypoint(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif(pdb, tmp_path / "trialanine.cif")
    compressed = tmp_path / "trialanine.cif.gz"
    with gzip.open(compressed, "wb") as handle:
        handle.write(cif.read_bytes())
    result = reconstruct_structure_fail_closed_v6(compressed, "A")
    assert result.status == "success", result.rejection_reason
    coordinate = result.input_evidence["coordinate_input"]
    assert coordinate["compressed_source"] is True
    assert coordinate["source_sha256"] == hashlib.sha256(
        compressed.read_bytes()
    ).hexdigest()
    assert coordinate["decompressed_payload_sha256"] == hashlib.sha256(
        cif.read_bytes()
    ).hexdigest()


def test_mmcif_to_mol2_preserves_projected_coordinates_and_residues(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif(pdb, tmp_path / "trialanine.cif")
    block, error = pdb_to_mol2(cif, chain_id="A", path="v6")
    assert error is None
    assert block.startswith("@<TRIPOS>MOLECULE\n")
    assert "@<TRIPOS>SUBSTRUCTURE" in block
    assert all(name in block for name in ("ALA1", "ALA2", "ALA3"))


def test_run_batch_accepts_mmcif_through_v6_default(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif(pdb, tmp_path / "trialanine.cif")
    result = run_batch([str(cif)], chain_id="A", target_chain_id="R")[0]
    assert result["input_format"] == "mmcif"
    assert result["status"] == "success", result.get("error")
    assert result["qualified_success"] is True


def _assert_batch_matches_strict(batch: dict, strict) -> None:
    def without_runtime(value):
        if isinstance(value, str):
            return re.sub(r"tmp[^\\/:]+\.(?:cif|pdb)", "<snapshot>", value)
        if isinstance(value, dict):
            return {
                key: without_runtime(item)
                for key, item in value.items()
                if key != "runtime_sec"
            }
        if isinstance(value, list):
            return [without_runtime(item) for item in value]
        return value

    for key, value in asdict(strict).items():
        assert without_runtime(batch[key]) == without_runtime(value), key
    assert batch["smiles"] == strict.output_smiles


def test_run_batch_v6_matches_strict_pdb(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")

    strict = reconstruct_structure_fail_closed_v6(pdb, "A")
    batch = run_batch([str(pdb)], chain_id="A", target_chain_id="R")[0]

    _assert_batch_matches_strict(batch, strict)


def test_run_batch_v6_preserves_schema_valid_candidate_assessment(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    unsupported = tmp_path / "unsupported.sdf"
    unsupported.write_text("unsupported\n", encoding="ascii")

    results = run_batch(
        [str(pdb), str(unsupported)],
        chain_id="A",
        target_chain_id="R",
    )
    validator = _candidate_assessment_validator()

    assert [row["status"] for row in results] == ["success", "not_supported"]
    for row in results:
        assessment = row["output_evidence"]["candidate_assessment"]
        validator.validate(assessment)
        assert assessment["result_context"]["status"] == row["status"]
        assert assessment["result_context"]["qualified_success"] is row[
            "qualified_success"
        ]


def test_run_batch_v6_forwards_empty_overlay_policy(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")

    strict = reconstruct_structure_fail_closed_v6(
        pdb, "A", require_empty_persistent_overlay=True
    )
    batch = run_batch(
        [str(pdb)],
        chain_id="A",
        target_chain_id="R",
        require_empty_persistent_overlay=True,
    )[0]

    _assert_batch_matches_strict(batch, strict)
    assert batch["input_evidence"]["entity_local_isolation"][
        "require_empty_persistent_overlay"
    ] is True


def test_run_batch_v6_matches_strict_mmcif_embedded_exact_pass(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _with_embedded_alanine_component(
        _to_mmcif(pdb, tmp_path / "trialanine-with-ala-ccd.cif"),
        component_id="ALA",
    )

    strict = reconstruct_structure_fail_closed_v6(cif, "A")
    batch = run_batch([str(cif)], chain_id="A", target_chain_id="R")[0]

    _assert_batch_matches_strict(batch, strict)
    assert batch["input_evidence"][
        "known_residue_embedded_component_audit"
    ]["status"] == "pass"


def test_run_batch_v6_matches_strict_mmcif_embedded_stereo_conflict(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _with_embedded_alanine_component(
        _to_mmcif(pdb, tmp_path / "trialanine-conflicting-ala-ccd.cif"),
        component_id="ALA",
    )
    document = gemmi.cif.read_file(str(cif))
    atoms = document.sole_block().find_mmcif_category("_chem_comp_atom.")
    tags = list(atoms.tags)
    id_column = tags.index("_chem_comp_atom.atom_id")
    stereo_column = tags.index("_chem_comp_atom.pdbx_stereo_config")
    for row in atoms:
        if str(row[id_column]) == "CA":
            row[stereo_column] = "R"
    document.write_file(str(cif))

    strict = reconstruct_structure_fail_closed_v6(cif, "A")
    batch = run_batch([str(cif)], chain_id="A", target_chain_id="R")[0]

    _assert_batch_matches_strict(batch, strict)
    assert batch["status"] == "rejected"
    assert batch["warning_codes"] == [
        "V6_LIBRARY_VS_EMBEDDED_COMPONENT_CONFLICT"
    ]


def test_run_batch_v6_matches_strict_mmcif_embedded_unknown_resolution(tmp_path):
    pdb = _write_trialanine(tmp_path / "unknown-middle.pdb")
    cif = _to_mmcif_with_unknown_middle_alanine(
        pdb, tmp_path / "unknown-middle.cif"
    )

    strict = reconstruct_structure_fail_closed_v6(cif, "A")
    batch = run_batch([str(cif)], chain_id="A", target_chain_id="R")[0]

    _assert_batch_matches_strict(batch, strict)
    assert batch["input_evidence"]["local_monomer_bootstrap"][
        "resolution_modes"
    ] == ["embedded_mmcif_chem_comp"]


def test_run_batch_v6_matches_strict_unsupported_coordinate_error(tmp_path):
    source = tmp_path / "unsupported.sdf"
    source.write_text("unsupported\n", encoding="ascii")

    strict = reconstruct_structure_fail_closed_v6(source, "A")
    batch = run_batch([str(source)], chain_id="A", target_chain_id="R")[0]

    _assert_batch_matches_strict(batch, strict)
    assert batch["status"] == "not_supported"
    assert batch["warning_codes"] == ["UNSUPPORTED_COORDINATE_FORMAT"]


def test_run_batch_v6_matches_strict_malformed_coordinate_error(tmp_path):
    source = tmp_path / "malformed.cif"
    source.write_text("not cif\n", encoding="ascii")

    strict = reconstruct_structure_fail_closed_v6(source, "A")
    batch = run_batch([str(source)], chain_id="A", target_chain_id="R")[0]

    _assert_batch_matches_strict(batch, strict)
    assert batch["status"] == "rejected"
    assert batch["warning_codes"] == ["MALFORMED_MMCIF_INPUT"]


def test_run_batch_csv_retains_success_and_failed_input_rows(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    unsupported = tmp_path / "unsupported.sdf"
    unsupported.write_text("unsupported\n", encoding="ascii")
    output = tmp_path / "results.csv"

    results = run_batch(
        [str(pdb), str(unsupported)],
        chain_id="A",
        target_chain_id="R",
        csv_output=str(output),
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(results) == len(rows) == 2
    assert [row["status"] for row in rows] == ["success", "not_supported"]
    assert rows[1]["warning_codes"] == "UNSUPPORTED_COORDINATE_FORMAT"
    assert "unsupported coordinate filename" in rows[1]["rejection_reason"]


def test_run_batch_uses_separate_target_projection_for_docking(
    tmp_path, monkeypatch
):
    pdb = _write_trialanine(tmp_path / "complex.pdb")
    lines = pdb.read_text(encoding="ascii").splitlines()
    target_lines = []
    for serial, (name, x, element) in enumerate((
        ("N", 0.0, "N"),
        ("CA", 1.4, "C"),
        ("C", 2.8, "C"),
        ("O", 3.8, "O"),
    ), start=900):
        line = _atom(serial, name, 1, x, element, y=30.0)
        target_lines.append(f"{line[:21]}R{line[22:]}")
    pdb.write_text(
        "\n".join([*lines[:-1], *target_lines, "END", ""]),
        encoding="ascii",
    )
    observed = {}

    def fake_dock(ligand_pdb, receptor_pdb):
        ligand = Path(ligand_pdb).read_text(encoding="ascii")
        receptor = Path(receptor_pdb).read_text(encoding="ascii")
        observed["ligand_chains"] = {
            line[21] for line in ligand.splitlines() if line.startswith("ATOM")
        }
        observed["receptor_chains"] = {
            line[21] for line in receptor.splitlines() if line.startswith("ATOM")
        }
        return -7.25

    monkeypatch.setattr(pipeline, "_dock_prepared_structures", fake_dock)
    result = run_batch(
        [str(pdb)],
        chain_id="A",
        target_chain_id="R",
        run_docking=True,
    )[0]

    assert result["status"] == "success"
    assert result["_row"]["target_sequence"] == "A"
    assert result["docking_score"] == -7.25
    assert observed == {"ligand_chains": {"A"}, "receptor_chains": {"R"}}


def test_unrepresentable_long_component_id_fails_closed(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    structure = gemmi.read_structure(str(pdb))
    structure[0][0][1].name = "LONG"
    cif = tmp_path / "long-component.cif"
    structure.make_mmcif_document().write_file(str(cif))
    result = reconstruct_structure_fail_closed_v6(cif, "A")
    assert result.status == "not_supported"
    assert result.warning_codes == ["MMCIF_COMPONENT_ID_NOT_PDB_REPRESENTABLE"]


def test_unrepresentable_long_atom_name_fails_without_overwriting_output(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    structure = gemmi.read_structure(str(pdb))
    structure[0][0][0][0].name = "LONGATOM"
    cif = tmp_path / "long-atom-name.cif"
    structure.make_mmcif_document().write_file(str(cif))
    output = tmp_path / "projected.pdb"
    output.write_bytes(b"existing output\n")

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == "COORDINATE_ATOM_NAME_NOT_PDB_REPRESENTABLE"
    assert exc_info.value.not_supported is True
    assert output.read_bytes() == b"existing output\n"


def test_unrepresentable_non_ascii_atom_name_fails_without_overwriting_output(
    tmp_path,
):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    structure = gemmi.read_structure(str(pdb))
    structure[0][0][0][0].name = "alpha-\N{GREEK SMALL LETTER ALPHA}"
    cif = tmp_path / "non-ascii-atom-name.cif"
    structure.make_mmcif_document().write_file(str(cif))
    output = tmp_path / "projected.pdb"
    output.write_bytes(b"existing output\n")

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == "COORDINATE_ATOM_NAME_NOT_PDB_REPRESENTABLE"
    assert exc_info.value.not_supported is True
    assert output.read_bytes() == b"existing output\n"


def test_unrepresentable_non_ascii_component_fails_without_overwriting_output(
    tmp_path,
):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    structure = gemmi.read_structure(str(pdb))
    structure[0][0][0].name = "\N{LATIN CAPITAL LETTER A WITH ACUTE}LA"
    cif = tmp_path / "non-ascii-component.cif"
    structure.make_mmcif_document().write_file(str(cif))
    output = tmp_path / "projected.pdb"
    output.write_bytes(b"existing output\n")

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == "MMCIF_COMPONENT_ID_NOT_PDB_REPRESENTABLE"
    assert exc_info.value.not_supported is True
    assert output.read_bytes() == b"existing output\n"


def test_unrepresentable_residue_sequence_fails_without_creating_output(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    structure = gemmi.read_structure(str(pdb))
    structure[0][0][0].seqid.num = 10000
    cif = tmp_path / "large-residue-sequence.cif"
    structure.make_mmcif_document().write_file(str(cif))
    output = tmp_path / "projected.pdb"

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == (
        "COORDINATE_RESIDUE_SEQUENCE_NOT_PDB_REPRESENTABLE"
    )
    assert exc_info.value.not_supported is True
    assert not output.exists()


@pytest.mark.parametrize("coordinate", [10000.0, float("nan")])
def test_unrepresentable_coordinate_fails_without_overwriting_output(
    tmp_path, coordinate
):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    structure = gemmi.read_structure(str(pdb))
    structure[0][0][0][0].pos.x = coordinate
    cif = tmp_path / "unrepresentable-coordinate.cif"
    structure.make_mmcif_document().write_file(str(cif))
    output = tmp_path / "projected.pdb"
    output.write_bytes(b"existing output\n")

    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == "COORDINATE_POSITION_NOT_PDB_REPRESENTABLE"
    assert exc_info.value.not_supported is True
    assert output.read_bytes() == b"existing output\n"


def test_projection_serialization_failure_preserves_existing_output(
    tmp_path, monkeypatch
):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    cif = _to_mmcif(pdb, tmp_path / "trialanine.cif")
    output = tmp_path / "projected.pdb"
    output.write_bytes(b"existing output\n")

    def fail_append(*_args, **_kwargs):
        raise UnicodeError("injected serialization failure")

    monkeypatch.setattr(structure_io, "_append_conect", fail_append)
    with pytest.raises(CoordinateInputError) as exc_info:
        mmcif_to_pdb(cif, output, "A")

    assert exc_info.value.code == "COORDINATE_PROJECTION_WRITE_FAILED"
    assert output.read_bytes() == b"existing output\n"
    assert not list(tmp_path.glob(".projected.pdb.*.pdb"))


def test_prepare_pdb_uses_audited_single_model_chain_projection(tmp_path):
    pdb = _write_trialanine(tmp_path / "trialanine.pdb")
    with prepare_coordinate_input(pdb, "A") as prepared:
        assert prepared.pdb_path != pdb.resolve()
        assert prepared.pdb_path.is_file()
        assert prepared.chain_id == "A"
        assert prepared.audit["projection_applied"] is True
        assert prepared.audit["source_model_count"] == 1
        assert prepared.audit["selected_model_index"] == 1
        assert prepared.audit["source_first_model_chain_ids"] == ["A"]
        assert prepared.audit["source_selected_chain_segment_count"] == 1


def test_prepare_pdb_preserves_source_sequence_identity_but_normalizes_sequence_free(
    tmp_path,
):
    source = _write_trialanine(tmp_path / "source-seqres.pdb")
    source.write_text(
        "SEQRES   1 A    3  ALA ALA ALA\n" + source.read_text(encoding="ascii"),
        encoding="ascii",
    )

    with prepare_coordinate_input(source, "A") as prepared:
        identity = prepared.audit["source_sequence_identity_audit"]
        assert identity["source_metadata_present"] is True
        assert identity["status"] == "unique"
        assert len(identity["rows"]) == 3
        assert all(row["mapping_state"] == "unique" for row in identity["rows"])
        assert all(row["source_sha256"] == prepared.audit["source_sha256"] for row in identity["rows"])
        normalized = gemmi.read_structure(str(prepared.pdb_path))
        assert all(not entity.full_sequence for entity in normalized.entities)


@pytest.mark.parametrize("source_kind", ["pdb", "mmcif"])
def test_source_identity_audit_binds_shuffled_residue_blocks_by_position(
    tmp_path, source_kind
):
    pdb = _write_shuffled_unknown_source(
        tmp_path / "shuffled-unknown.pdb",
        include_modres=True,
        modres_standard_name="DPN",
    )
    source = pdb
    if source_kind == "mmcif":
        source = _shuffle_mmcif_atom_site_rows(
            _to_mmcif(pdb, tmp_path / "shuffled-unknown.cif"),
            explicit_polymer_membership=True,
        )

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_CONSTRAINT_INCOMPLETE"]
    audit = result.input_evidence["coordinate_input"][
        "source_sequence_identity_audit"
    ]
    assert audit["rows"][1]["sequence_position"] == 2
    assert audit["rows"][1]["declared_name"] == "DPN"
    assert audit["rows"][1]["coordinate_resseq"] == 2
    assert audit["rows"][1]["coordinate_residue_name"] == "ZZZ"
    if source_kind == "mmcif":
        assert audit["mapping_basis"] == "mmcif_label_seq_id"


def test_mmcif_internal_het_without_label_membership_is_not_polymer_evidence(
    tmp_path,
):
    pdb = _write_shuffled_unknown_source(
        tmp_path / "internal-het-without-label-membership.pdb"
    )
    source = _shuffle_mmcif_atom_site_rows(
        _to_mmcif(pdb, tmp_path / "internal-het-without-label-membership.cif"),
        explicit_polymer_membership=False,
    )

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_MAPPING_UNRESOLVED"]
    audit = result.input_evidence["coordinate_input"][
        "source_sequence_identity_audit"
    ]
    assert audit["status"] == "partial"
    assert audit["mapping_basis"] == "mmcif_auth_seqid_fallback"


def test_mmcif_internal_het_without_entity_poly_seq_is_not_polymer_evidence(
    tmp_path,
):
    pdb = _write_shuffled_unknown_source(
        tmp_path / "internal-het-without-entity-poly-seq.pdb"
    )
    source = _shuffle_mmcif_atom_site_rows(
        _to_mmcif(pdb, tmp_path / "internal-het-without-entity-poly-seq.cif"),
        explicit_polymer_membership=True,
    )
    document = gemmi.cif.read_file(str(source))
    table = document.sole_block().find_mmcif_category("_entity_poly_seq.")
    for index in range(len(table) - 1, -1, -1):
        table.remove_row(index)
    document.write_file(str(source))

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_MAPPING_UNRESOLVED"]
    audit = result.input_evidence["coordinate_input"][
        "source_sequence_identity_audit"
    ]
    assert audit["status"] == "partial"
    assert audit["mapping_reason"] == (
        "hetero_residue_lacks_polymer_membership_evidence"
    )


def test_mmcif_internal_het_mixed_atom_binding_is_not_polymer_evidence(
    tmp_path,
):
    pdb = _write_shuffled_unknown_source(
        tmp_path / "internal-het-mixed-label-binding.pdb"
    )
    source = _shuffle_mmcif_atom_site_rows(
        _to_mmcif(pdb, tmp_path / "internal-het-mixed-label-binding.cif"),
        explicit_polymer_membership=True,
    )
    document = gemmi.cif.read_file(str(source))
    table = document.sole_block().find_mmcif_category("_atom_site.")
    tags = list(table.tags)
    group_index = tags.index("_atom_site.group_PDB")
    auth_seq_index = tags.index("_atom_site.auth_seq_id")
    label_seq_index = tags.index("_atom_site.label_seq_id")
    mutated = False
    for row in table:
        if (
            str(row[group_index]).strip().upper() == "HETATM"
            and str(row[auth_seq_index]).strip() == "2"
        ):
            row[label_seq_index] = "."
            mutated = True
            break
    assert mutated
    document.write_file(str(source))

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_MAPPING_UNRESOLVED"]
    audit = result.input_evidence["coordinate_input"][
        "source_sequence_identity_audit"
    ]
    assert audit["status"] == "partial"
    assert audit["mapping_reason"] == "partial_label_seq_id"


def test_mmcif_source_identity_does_not_borrow_same_auth_seq_from_other_chain(
    tmp_path,
):
    pdb = _write_shuffled_unknown_source(
        tmp_path / "cross-chain-source.pdb"
    )
    source = _to_mmcif_with_cross_chain_same_auth_seq(
        pdb, tmp_path / "cross-chain-source.cif"
    )

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_MAPPING_UNRESOLVED"]
    audit = result.input_evidence["coordinate_input"][
        "source_sequence_identity_audit"
    ]
    assert audit["status"] == "partial"


def test_pdb_internal_het_without_modres_is_not_polymer_evidence(tmp_path):
    source = _write_shuffled_unknown_source(
        tmp_path / "internal-het-without-modres.pdb"
    )

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_MAPPING_UNRESOLVED"]
    audit = result.input_evidence["coordinate_input"][
        "source_sequence_identity_audit"
    ]
    assert audit["status"] == "partial"
    assert audit["mapping_reason"] == (
        "hetero_residue_lacks_polymer_membership_evidence"
    )


def test_source_identity_audit_excludes_extra_nonpolymer_ligand(tmp_path):
    source = _write_unknown_source_with_extra_ligand(
        tmp_path / "extra-nonpolymer-ligand.pdb"
    )

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.warning_codes == ["SOURCE_IDENTITY_MAPPING_UNRESOLVED"]
    audit = result.input_evidence["coordinate_input"][
        "source_sequence_identity_audit"
    ]
    assert audit["status"] == "partial"
    assert audit["excluded_nonpolymer_coordinate_residue_count"] == 1
    assert all(row["coordinate_resseq"] != 4 for row in audit["rows"])


def test_prepare_pdb_selects_first_model_with_audit(tmp_path):
    source = _write_trialanine(tmp_path / "single.pdb")
    lines = source.read_text(encoding="ascii").splitlines()
    header = [line for line in lines if line.startswith(("LINK", "SSBOND"))]
    atoms = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    multi = tmp_path / "multi.pdb"
    multi.write_text(
        "\n".join(
            [
                *header,
                "MODEL        1",
                *atoms,
                "ENDMDL",
                "MODEL        2",
                *atoms,
                "ENDMDL",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with prepare_coordinate_input(multi, "A") as prepared:
        normalized = gemmi.read_structure(str(prepared.pdb_path))
        assert len(normalized) == 1
        assert len(normalized[0]) == 1
        assert prepared.audit["source_model_count"] == 2
        assert prepared.audit["selected_model_index"] == 1
        assert prepared.audit["source_selected_chain_atom_count"] == len(atoms)


def test_pdb_projection_preserves_and_audits_selected_chain_conect(tmp_path):
    source = _write_trialanine(tmp_path / "conect.pdb")
    lines = source.read_text(encoding="ascii").splitlines()
    atoms = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    serials = {
        (int(line[22:26]), line[12:16].strip()): int(line[6:11])
        for line in atoms
    }
    left = serials[(1, "N")]
    right = serials[(3, "C")]
    source.write_text(
        "\n".join(
            [
                *atoms,
                f"CONECT{left:5d}{right:5d}",
                f"CONECT{right:5d}{left:5d}",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with prepare_coordinate_input(source, "A") as prepared:
        assert prepared.audit["source_pdb_conect_pair_count"] == 1
        assert prepared.audit["materialized_explicit_connection_count"] == 1
        assert prepared.audit["connection_rows"] == [
            {
                "name": "PDB_CONECT",
                "type": "PDB_CONECT",
                "partner_1": [1, "", "N"],
                "partner_2": [3, "", "C"],
                "normalized_serials": sorted((left, right)),
            }
        ]
        conect = [
            line
            for line in prepared.pdb_path.read_text(encoding="ascii").splitlines()
            if line.startswith("CONECT")
        ]
        assert conect == [f"CONECT{left:5d}{right:5d}"]


def test_pdb_projection_drops_conect_to_removed_altloc(tmp_path):
    source = _write_trialanine(tmp_path / "altloc-conect.pdb")
    atom_lines = [
        line
        for line in source.read_text(encoding="ascii").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    selected = list(atom_lines[0])
    selected[16] = "A"
    selected[54:60] = f"{0.80:6.2f}"
    removed = selected.copy()
    removed[6:11] = f"{99:5d}"
    removed[16] = "B"
    removed[54:60] = f"{0.20:6.2f}"
    other_serial = int(atom_lines[1][6:11])
    source.write_text(
        "\n".join(
            [
                "".join(selected),
                "".join(removed),
                *atom_lines[1:],
                f"CONECT{99:5d}{other_serial:5d}",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with prepare_coordinate_input(source, "A") as prepared:
        assert prepared.audit["source_pdb_conect_pair_count"] == 1
        assert prepared.audit["first_model_pdb_conect_pair_count"] == 1
        assert prepared.audit["selected_chain_pdb_conect_pair_count"] == 0
        assert prepared.audit["materialized_explicit_connection_count"] == 0
        assert not any(
            line.startswith("CONECT")
            for line in prepared.pdb_path.read_text(encoding="ascii").splitlines()
        )


def test_pdb_projection_drops_structured_link_to_removed_altloc(tmp_path):
    source = _write_trialanine(tmp_path / "altloc-link.pdb")
    atom_lines = [
        line
        for line in source.read_text(encoding="ascii").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    atom_index = next(
        index
        for index, line in enumerate(atom_lines)
        if int(line[22:26]) == 1 and line[12:16].strip() == "C"
    )
    selected = list(atom_lines[atom_index])
    selected[16] = "A"
    selected[54:60] = f"{0.80:6.2f}"
    removed = selected.copy()
    removed[6:11] = f"{99:5d}"
    removed[16] = "B"
    removed[54:60] = f"{0.20:6.2f}"
    atom_lines[atom_index] = "".join(selected)
    atom_lines.insert(atom_index + 1, "".join(removed))
    link = list(_link())
    link[12:16] = f"{'C':>4s}"
    link[16] = "B"
    source.write_text(
        "\n".join(["".join(link), *atom_lines, "END"]) + "\n",
        encoding="ascii",
    )

    with prepare_coordinate_input(source, "A") as prepared:
        assert prepared.audit["structured_connection_count"] == 1
        assert prepared.audit["structured_connection_altloc_mismatch_count"] == 1
        assert prepared.audit["materialized_explicit_connection_count"] == 0
        assert not any(
            line.startswith("CONECT")
            for line in prepared.pdb_path.read_text(encoding="ascii").splitlines()
        )


def test_pdb_projection_ignores_conect_inside_later_model(tmp_path):
    source = _write_trialanine(tmp_path / "single.pdb")
    atom_lines = [
        line
        for line in source.read_text(encoding="ascii").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    left = int(atom_lines[0][6:11])
    right = int(atom_lines[1][6:11])
    multi = tmp_path / "later-model-conect.pdb"
    multi.write_text(
        "\n".join(
            [
                "MODEL        1",
                *atom_lines,
                "ENDMDL",
                "MODEL        2",
                *atom_lines,
                f"CONECT{left:5d}{right:5d}",
                "ENDMDL",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with prepare_coordinate_input(multi, "A") as prepared:
        assert prepared.audit["source_model_count"] == 2
        assert prepared.audit["source_pdb_conect_pair_count"] == 1
        assert prepared.audit["first_model_pdb_conect_pair_count"] == 0
        assert prepared.audit["selected_chain_pdb_conect_pair_count"] == 0
        assert prepared.audit["materialized_explicit_connection_count"] == 0


def test_pdb_projection_ignores_conect_after_end_record(tmp_path):
    source = _write_trialanine(tmp_path / "trailing-conect.pdb")
    lines = source.read_text(encoding="ascii").splitlines()
    atom_lines = [
        line for line in lines if line.startswith(("ATOM", "HETATM"))
    ]
    left = int(atom_lines[0][6:11])
    right = int(atom_lines[1][6:11])
    source.write_text(
        "\n".join([*atom_lines, "END", f"CONECT{left:5d}{right:5d}"]) + "\n",
        encoding="ascii",
    )

    with prepare_coordinate_input(source, "A") as prepared:
        assert prepared.audit["source_pdb_conect_pair_count"] == 0
        assert prepared.audit["first_model_pdb_conect_pair_count"] == 0
        assert prepared.audit["materialized_explicit_connection_count"] == 0
        assert not any(
            line.startswith("CONECT")
            for line in prepared.pdb_path.read_text(encoding="ascii").splitlines()
        )


def test_pdb_projection_ignores_link_inside_later_model(tmp_path):
    source = _write_trialanine(tmp_path / "single.pdb")
    atom_lines = [
        line
        for line in source.read_text(encoding="ascii").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    multi = tmp_path / "later-model-link.pdb"
    multi.write_text(
        "\n".join(
            [
                "MODEL        1",
                *atom_lines,
                "ENDMDL",
                "MODEL        2",
                _link(),
                *atom_lines,
                "ENDMDL",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with prepare_coordinate_input(multi, "A") as prepared:
        assert prepared.audit["source_model_count"] == 2
        assert prepared.audit["ignored_later_model_structured_connection_count"] == 1
        assert prepared.audit["structured_connection_count"] == 0
        assert prepared.audit["materialized_explicit_connection_count"] == 0


def test_pdb_same_chain_ter_segments_are_normalized_and_audited(tmp_path):
    source = _write_trialanine(tmp_path / "segmented.pdb")
    lines = source.read_text(encoding="ascii").splitlines()
    projected_lines = []
    inserted = False
    for line in lines:
        if (
            not inserted
            and line.startswith(("ATOM", "HETATM"))
            and int(line[22:26]) == 2
        ):
            projected_lines.append("TER       0      ALA A   1")
            inserted = True
        projected_lines.append(line)
    source.write_text("\n".join(projected_lines) + "\n", encoding="ascii")

    with prepare_coordinate_input(source, "A") as prepared:
        assert prepared.audit["source_selected_chain_segment_count"] == 2
        normalized_lines = prepared.pdb_path.read_text(encoding="ascii").splitlines()
        selected_ter_indices = [
            index
            for index, line in enumerate(normalized_lines)
            if line.startswith("TER") and len(line) > 21 and line[21:22] == "A"
        ]
        selected_atom_indices = [
            index
            for index, line in enumerate(normalized_lines)
            if line.startswith(("ATOM", "HETATM")) and line[21:22] == "A"
        ]
        assert selected_ter_indices
        assert max(selected_atom_indices) < min(selected_ter_indices)

    result = reconstruct_structure_fail_closed_v6(source, "A")
    assert result.status == "success", result.rejection_reason
    assert result.input_evidence["coordinate_input"][
        "source_selected_chain_segment_count"
    ] == 2


def _water(serial: int, chain: str, residue: int) -> str:
    return (
        f"HETATM{serial:5d}  O   HOH {chain}{residue:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           O"
    )


def _hetero_carbon(
    serial: int, residue_name: str, chain: str, residue: int
) -> str:
    return (
        f"HETATM{serial:5d}  C1  {residue_name:>3s} {chain}{residue:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
    )


def _trialanine_polymer_headers(chain: str) -> list[str]:
    return [
        "HEADER    PEPTIDE",
        "COMPND    MOL_ID: 1;",
        "COMPND   2 MOLECULE: TRIALANINE;",
        f"COMPND   3 CHAIN: {chain};",
        f"SEQRES   1 {chain}    3  ALA ALA ALA",
    ]


def _write_duplicate_chain_polymer_fixture(
    path: Path,
    *,
    chain: str,
    nonpolymer_names: tuple[str, ...] = (),
    water_count: int = 0,
) -> Path:
    source = _write_trialanine(path)
    peptide_lines = []
    for line in source.read_text(encoding="ascii").splitlines():
        if line == "END":
            continue
        row = list(line.ljust(80))
        if line.startswith(("ATOM  ", "HETATM")):
            row[21] = chain
        elif line.startswith("LINK"):
            row[21] = chain
            row[51] = chain
        peptide_lines.append("".join(row).rstrip())

    rows = [
        *_trialanine_polymer_headers(chain),
        *peptide_lines,
        f"TER       0      ALA {chain}   3",
    ]
    spacer_chains = [candidate for candidate in ("X", "Y") if candidate != chain]
    rows.extend([
        _water(90, spacer_chains[0], 1),
        f"TER      90      HOH {spacer_chains[0]}   1",
    ])
    next_serial = 100
    if nonpolymer_names:
        for offset, name in enumerate(nonpolymer_names, start=101):
            rows.append(_hetero_carbon(next_serial, name, chain, offset))
            next_serial += 1
        rows.append(
            f"TER     {next_serial:5d}      {nonpolymer_names[-1]:>3s} "
            f"{chain}{100 + len(nonpolymer_names):4d}"
        )
    if water_count:
        if nonpolymer_names:
            rows.extend([
                _water(91, spacer_chains[1], 1),
                f"TER      91      HOH {spacer_chains[1]}   1",
            ])
        for offset in range(water_count):
            rows.append(_water(next_serial, chain, 201 + offset))
            next_serial += 1
    source.write_text("\n".join([*rows, "END"]) + "\n", encoding="ascii")
    return source


@pytest.mark.parametrize(
    ("case_id", "chain", "nonpolymer_names", "water_count"),
    [
        ("BIRD-005", "A", ("EEE", "EEE", "EEE", "EEE", "EEE", "MOH"), 0),
        ("BIRD-007", "C", ("WMH",), 2),
        ("BIRD-008", "A", ("2PO", "2PO", "P1W", "P1W", "P1W"), 0),
        ("BIRD-012", "A", ("BNZ", "BNZ", "BNZ", "BNZ"), 1),
        ("BIRD-018", "A", ("EEE", "EEE", "EEE", "EEE"), 0),
    ],
)
def test_pdb_duplicate_chain_id_selects_unique_seqres_polymer_object(
    tmp_path, case_id, chain, nonpolymer_names, water_count
):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / f"{case_id}.pdb",
        chain=chain,
        nonpolymer_names=nonpolymer_names,
        water_count=water_count,
    )
    excluded_object_count = int(bool(nonpolymer_names)) + int(bool(water_count))

    with prepare_coordinate_input(source, chain) as prepared:
        audit = prepared.audit
        normalized = prepared.pdb_path.read_text(encoding="ascii")
        assert audit["source_matching_chain_object_count"] == (
            1 + excluded_object_count
        )
        assert audit["source_polymer_chain_object_count"] == 1
        assert audit["source_gemmi_polymer_chain_object_count"] == 1
        assert audit["selected_source_chain_object_index"] == 0
        assert audit["pdb_chain_object_selection_mode"] == (
            "unique_seqres_polymer_object"
        )
        assert audit["duplicate_auth_chain_boundary_audit_applied"] is True
        assert audit["excluded_nonselected_chain_object_count"] == (
            excluded_object_count
        )
        assert audit["excluded_nonselected_chain_residue_count"] == (
            len(nonpolymer_names) + water_count
        )
        assert audit["excluded_nonpolymer_chain_object_count"] == (
            int(bool(nonpolymer_names))
        )
        assert audit["excluded_nonpolymer_chain_residue_count"] == (
            len(nonpolymer_names)
        )
        assert audit["excluded_solvent_only_chain_object_count"] == int(
            bool(water_count)
        )
        assert audit["excluded_solvent_or_ion_residue_count"] == water_count
        assert audit["cross_object_structured_connection_count"] == 0
        assert audit["cross_object_pdb_conect_pair_count"] == 0
        object_rows = audit["pdb_matching_chain_object_rows"]
        assert len(object_rows) == 1 + excluded_object_count
        selected_row = next(row for row in object_rows if row["selected"])
        assert selected_row["source_chain_object_index"] == 0
        assert selected_row["residue_number_range"] == [1, 3]
        assert selected_row["atom_serial_range"] is not None
        assert selected_row["polymer_residue_count"] == 3
        assert selected_row["seqres_polymer_residue_count"] == 3
        assert all(
            row["polymer_residue_count"] == 0
            for row in object_rows
            if not row["selected"]
        )
        assert "HETATM" not in normalized

    result = reconstruct_structure_fail_closed_v6(source, chain)
    assert result.status == "success", result.rejection_reason


def test_pdb_unique_polymer_selection_rejects_cross_object_link(tmp_path):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / "peptide-water-cross-link.pdb",
        chain="A",
        water_count=1,
    )
    lines = [
        line for line in source.read_text(encoding="ascii").splitlines()
        if line != "END"
    ]
    cross_object_link = _source_link(
        atom1="O",
        residue_name1="HOH",
        residue1=201,
        atom2="N",
        residue_name2="ALA",
        residue2=1,
    )
    source.write_text(
        "\n".join([*lines, cross_object_link, "END"]) + "\n",
        encoding="ascii",
    )

    with pytest.raises(CoordinateInputError) as exc_info:
        with prepare_coordinate_input(source, "A"):
            pass
    assert exc_info.value.code == "PDB_AUTH_CHAIN_OBJECT_CONNECTION_NOT_SUPPORTED"
    assert exc_info.value.not_supported is True
    assert "cross-object LINK/SSBOND" in str(exc_info.value)


def test_pdb_unique_polymer_selection_rejects_cross_object_ssbond(tmp_path):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / "peptide-ligand-cross-ssbond.pdb",
        chain="A",
        nonpolymer_names=("CYS",),
    )
    lines = []
    for line in source.read_text(encoding="ascii").splitlines():
        if line == "END" or line.startswith("LINK"):
            continue
        row = list(line.ljust(80))
        if line.startswith("SEQRES"):
            row[19:22] = "CYS"
        elif line.startswith(("ATOM  ", "HETATM")):
            residue_number = int(line[22:26])
            if residue_number == 1:
                row[17:20] = "CYS"
                if line[12:16].strip() == "CB":
                    row[12:16] = f"{'SG':>4s}"
                    row[76:78] = f"{'S':>2s}"
            elif residue_number == 101:
                row[12:16] = f"{'SG':>4s}"
                row[76:78] = f"{'S':>2s}"
        lines.append("".join(row).rstrip())
    source.write_text(
        "\n".join([*lines, _source_ssbond(residue1=1, residue2=101), "END"])
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(CoordinateInputError) as exc_info:
        with prepare_coordinate_input(source, "A"):
            pass
    assert exc_info.value.code == "PDB_AUTH_CHAIN_OBJECT_CONNECTION_NOT_SUPPORTED"
    assert exc_info.value.not_supported is True
    assert "cross-object LINK/SSBOND" in str(exc_info.value)
    assert "ambiguous LINK/SSBOND" not in str(exc_info.value)


def test_pdb_unique_polymer_selection_rejects_ambiguous_link_endpoint(
    tmp_path,
):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / "ambiguous-link-endpoint.pdb",
        chain="A",
        nonpolymer_names=("EEE",),
    )
    lines = [
        line
        for line in source.read_text(encoding="ascii").splitlines()
        if line != "END"
    ]
    lines.extend([
        _water(91, "Y", 1),
        "TER      91      HOH Y   1",
        _hetero_carbon(200, "EEE", "A", 101),
        "TER     201      EEE A 101",
        _source_link(
            atom1="C1",
            residue_name1="EEE",
            residue1=101,
            atom2="N",
            residue_name2="ALA",
            residue2=1,
        ),
    ])
    source.write_text("\n".join([*lines, "END"]) + "\n", encoding="ascii")

    with pytest.raises(CoordinateInputError) as exc_info:
        with prepare_coordinate_input(source, "A"):
            pass
    assert exc_info.value.code == "PDB_AUTH_CHAIN_OBJECT_CONNECTION_NOT_SUPPORTED"
    assert exc_info.value.not_supported is True
    assert "ambiguous LINK/SSBOND endpoints" in str(exc_info.value)


def test_pdb_unique_polymer_selection_rejects_cross_object_conect(tmp_path):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / "peptide-water-cross-conect.pdb",
        chain="A",
        water_count=1,
    )
    lines = [
        line for line in source.read_text(encoding="ascii").splitlines()
        if line != "END"
    ]
    source.write_text(
        "\n".join([*lines, f"CONECT{1:5d}{100:5d}", "END"]) + "\n",
        encoding="ascii",
    )

    with pytest.raises(CoordinateInputError) as exc_info:
        with prepare_coordinate_input(source, "A"):
            pass
    assert exc_info.value.code == "PDB_AUTH_CHAIN_OBJECT_CONNECTION_NOT_SUPPORTED"
    assert exc_info.value.not_supported is True
    assert "cross-object CONECT" in str(exc_info.value)


def test_pdb_conect_rejects_model_global_duplicate_atom_serial(tmp_path):
    source = _write_trialanine(tmp_path / "duplicate-serial-solvent.pdb")
    lines = [
        line for line in source.read_text(encoding="ascii").splitlines()
        if line != "END"
    ]
    source.write_text(
        "\n".join([
            *_trialanine_polymer_headers("A"),
            *lines,
            "TER       0      ALA A   3",
            _water(100, "B", 201),
            "TER     101      HOH B 201",
            _water(3, "A", 101),
            "TER       3      HOH A 101",
            "CONECT    3    1",
            "END",
        ]) + "\n",
        encoding="ascii",
    )

    with pytest.raises(
        CoordinateInputError,
        match="model-global unique atom serials; duplicates: 3",
    ) as exc_info:
        with prepare_coordinate_input(source, "A"):
            pass
    assert exc_info.value.code == "DUPLICATE_SOURCE_ATOM_SERIAL"


def test_pdb_duplicate_chain_id_rejects_without_polymer_evidence(tmp_path):
    source = _write_trialanine(tmp_path / "two-peptide-duplicates.pdb")
    lines = [
        line for line in source.read_text(encoding="ascii").splitlines()
        if line != "END"
    ]
    duplicate = _atom(102, "CA", 101, 100.0, "C")
    source.write_text(
        "\n".join([
            *lines,
            "TER       0      ALA A   3",
            _water(100, "B", 101),
            "TER     101      HOH B 101",
            duplicate,
            "END",
        ]) + "\n",
        encoding="ascii",
    )

    with pytest.raises(CoordinateInputError) as exc_info:
        with prepare_coordinate_input(source, "A"):
            pass
    assert exc_info.value.code == "PDB_AUTH_CHAIN_POLYMER_OBJECT_NOT_UNIQUE"
    assert exc_info.value.not_supported is True


def test_pdb_duplicate_chain_id_rejects_multiple_polymer_objects(tmp_path):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / "two-polymer-objects.pdb",
        chain="A",
        water_count=1,
    )
    structure, _ = structure_io._read_pdb_structure(source)
    model = structure[0]
    polymer = next(
        chain
        for chain in model
        if any(
            residue.entity_type == gemmi.EntityType.Polymer
            for residue in chain
        )
    )
    model.add_chain(polymer.clone())
    matching = [
        index for index, chain in enumerate(model) if chain.name == "A"
    ]
    explicit_polymer_subchains = structure_io._explicit_pdb_polymer_subchains(
        structure
    )

    with pytest.raises(CoordinateInputError) as exc_info:
        structure_io._select_unique_pdb_polymer_chain_object(
            model, "A", matching, explicit_polymer_subchains
        )
    assert exc_info.value.code == "PDB_AUTH_CHAIN_POLYMER_OBJECT_NOT_UNIQUE"
    assert exc_info.value.not_supported is True


def test_pdb_duplicate_chain_id_rejects_excluded_unbound_polymer_object(
    tmp_path,
):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / "excluded-unbound-polymer.pdb",
        chain="A",
        water_count=1,
    )
    structure, _ = structure_io._read_pdb_structure(source)
    model = structure[0]
    selected = next(
        chain
        for chain in model
        if any(
            residue.entity_type == gemmi.EntityType.Polymer
            for residue in chain
        )
    )
    unbound_polymer = selected.clone()
    for residue in unbound_polymer:
        residue.subchain = "Qx"
        residue.entity_id = "Q"
        residue.entity_type = gemmi.EntityType.Polymer
    model.add_chain(unbound_polymer)
    matching = [
        index for index, chain in enumerate(model) if chain.name == "A"
    ]
    explicit_polymer_subchains = structure_io._explicit_pdb_polymer_subchains(
        structure
    )

    with pytest.raises(CoordinateInputError) as exc_info:
        structure_io._select_unique_pdb_polymer_chain_object(
            model, "A", matching, explicit_polymer_subchains
        )
    assert exc_info.value.code == "PDB_AUTH_CHAIN_POLYMER_OBJECT_NOT_UNIQUE"
    assert exc_info.value.not_supported is True


def test_pdb_duplicate_chain_id_rejects_selected_unbound_polymer_residue(
    tmp_path,
):
    source = _write_duplicate_chain_polymer_fixture(
        tmp_path / "selected-unbound-polymer.pdb",
        chain="A",
        water_count=1,
    )
    structure, _ = structure_io._read_pdb_structure(source)
    model = structure[0]
    selected = next(
        chain
        for chain in model
        if any(
            residue.entity_type == gemmi.EntityType.Polymer
            for residue in chain
        )
    )
    selected[0].subchain = "Qx"
    selected[0].entity_id = "Q"
    selected[0].entity_type = gemmi.EntityType.Polymer
    matching = [
        index for index, chain in enumerate(model) if chain.name == "A"
    ]
    explicit_polymer_subchains = structure_io._explicit_pdb_polymer_subchains(
        structure
    )

    with pytest.raises(CoordinateInputError) as exc_info:
        structure_io._select_unique_pdb_polymer_chain_object(
            model, "A", matching, explicit_polymer_subchains
        )
    assert exc_info.value.code == "PDB_AUTH_CHAIN_POLYMER_OBJECT_NOT_UNIQUE"
    assert exc_info.value.not_supported is True


@pytest.mark.parametrize(
    ("filename", "payload", "error_code"),
    [
        ("malformed.cif", b"this is not mmCIF", "MALFORMED_MMCIF_INPUT"),
        ("empty.cif", b"", "MALFORMED_MMCIF_INPUT"),
        ("corrupt.cif.gz", b"not a gzip stream", "INVALID_COMPRESSED_COORDINATE_INPUT"),
    ],
)
def test_strict_structure_api_rejects_malformed_mmcif_without_raising(
    tmp_path, filename, payload, error_code
):
    source = tmp_path / filename
    source.write_bytes(payload)

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.qualified_success is False
    assert result.warning_codes == [error_code]
    assert result.input_evidence["coordinate_input"]["error_code"] == error_code


@pytest.mark.parametrize("filename", ["missing.pdb", "missing.cif"])
def test_strict_structure_api_rejects_missing_input_without_raising(
    tmp_path, filename
):
    source = tmp_path / filename

    result = reconstruct_structure_fail_closed_v6(source, "A")

    assert result.status == "rejected"
    assert result.qualified_success is False
    assert result.warning_codes == ["COORDINATE_INPUT_NOT_FOUND"]
    assert result.input_evidence["coordinate_input"]["error_code"] == (
        "COORDINATE_INPUT_NOT_FOUND"
    )
