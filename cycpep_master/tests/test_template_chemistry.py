import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from cycpep_master.core.data import AA_SMILES
from cycpep_master.core.derived_monomers import build_derived_row
from cycpep_master.core.molecule import add_to_combo
from cycpep_master.core.pdb_parser import parse_backbone, standard_pdb_atom_name_map
from cycpep_master.paths import _map_utils, path_a, path_c
from cycpep_master.paths.residue_template_factory import (
    compose_capped_template,
    get_residue_template,
    get_residue_template_for_symbol,
    map_pdb_atoms,
    map_pdb_atoms_with_evidence,
)


EXPECTED_CHARGES = {"ARG": 1, "LYS": 1, "HIS": 1}
ONE_LETTER = {"ARG": "R", "LYS": "K", "HIS": "H"}


def test_lysine_r3_h_materializes_as_tetravalent_ammonium():
    template = get_residue_template("LYS")
    for molecule in (template.mol, Chem.MolFromSmiles(template.free_smiles)):
        assert molecule is not None
        positive = [
            atom for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1
        ]
        assert len(positive) == 1
        assert positive[0].GetTotalValence() == 4
        assert positive[0].GetTotalNumHs() == 3


def test_terminal_amide_derived_template_retains_observed_r2_cap():
    source = "NC(=O)[C@@H](N)CS"
    row, _manifest = build_derived_row(
        "CY3",
        source,
        monomer_id=23101,
        r3_mapped_smiles="NC(=O)[C@@H](N)C[SH:9003]",
        r3_port={"cap": "H"},
    )
    with _map_utils.isolated_monomer_registry(derived_rows=[row]):
        template = get_residue_template_for_symbol(row["symbol"], pdb_resname="CY3")
    assert template.r2 == "NH2"
    assert sum(atom.GetAtomicNum() > 1 for atom in template.mol.GetAtoms()) == 7
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(template.free_smiles)) == (
        Chem.MolToInchiKey(Chem.MolFromSmiles(source))
    )
    carbonyl_caps = [
        neighbor.GetSymbol()
        for atom in template.mol.GetAtoms()
        if atom.GetSymbol() == "C" and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(atom).GetSymbol() == "O"
            for bond in atom.GetBonds()
        )
        for neighbor in atom.GetNeighbors()
        if template.mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
        == Chem.BondType.SINGLE
        and neighbor.GetSymbol() in {"N", "O", "S"}
    ]
    assert carbonyl_caps == ["N"]


def test_ccd_convention_charge_is_shared_by_template_families():
    for residue_name, expected_charge in EXPECTED_CHARGES.items():
        template = Chem.MolFromSmiles(AA_SMILES[residue_name])
        assert template is not None
        assert Chem.GetFormalCharge(template) == expected_charge

        symbol = ONE_LETTER[residue_name]
        cyclic = _map_utils.get_smi_from_map(f"{symbol}{symbol}{{cyc:N-C}}")
        assembled = Chem.MolFromSmiles(cyclic)
        assert assembled is not None
        assert Chem.GetFormalCharge(assembled) == 2 * expected_charge


def test_combo_copy_preserves_formal_charge_and_named_template_mapping():
    for residue_name, expected_charge in EXPECTED_CHARGES.items():
        combo = Chem.RWMol()
        add_to_combo(combo, AA_SMILES[residue_name])
        assert sum(atom.GetFormalCharge() for atom in combo.GetAtoms()) == expected_charge
        mapping = standard_pdb_atom_name_map(
            residue_name, AA_SMILES[residue_name]
        )
        assert len(mapping) == combo.GetNumAtoms()
        assert {"N", "CA", "C", "O"}.issubset(mapping)


def test_path_a_sidechain_mapping_uses_atom_names_not_pdb_serial_order():
    residue_name = "ARG"
    template = get_residue_template(residue_name)
    name_map = standard_pdb_atom_name_map(residue_name, template.smiles)
    atoms = [
        {
            "num": 1000 - index,
            "name": name,
            "elem": template.mol.GetAtomWithIdx(template_index).GetSymbol(),
        }
        for index, (name, template_index) in enumerate(name_map.items())
    ]
    residue = {"key": (residue_name, 1, False), "name": residue_name}
    with (
        patch.object(path_a, "get_res_seq", return_value=[residue]),
        patch.object(path_a, "get_pdb_atoms", return_value=atoms),
        patch.object(path_a, "read_conect", return_value={}),
        patch.object(path_a, "_has_head_to_tail_evidence", return_value=False),
        patch(
            "cycpep_master.core.cyclization.detect_cyclization",
            return_value=SimpleNamespace(bonds=[]),
        ),
    ):
        combo, pdb_to_graph = path_a._build_combo("unused.pdb", "L")

    serial_by_name = {atom["name"]: atom["num"] for atom in atoms}
    assert combo.GetNumAtoms() == len(name_map) + 1
    assert sum(atom.GetSymbol() == "O" for atom in combo.GetAtoms()) == (
        sum(atom.GetSymbol() == "O" for atom in template.mol.GetAtoms()) + 1
    )
    assert {
        name: pdb_to_graph[serial_by_name[name]] for name in name_map
    } == name_map


def test_all_standard_path_a_templates_are_complete_unified_rows():
    for residue_name in (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
        "SER THR TRP TYR VAL"
    ).split():
        template = get_residue_template(residue_name)
        mapping = standard_pdb_atom_name_map(residue_name, template.smiles)
        assert template.source == "core"
        assert len(mapping) == template.mol.GetNumAtoms()
        assert {"N", "CA", "C", "O"}.issubset(mapping)


def test_unified_r3_defaults_are_materialized_without_dummy_atoms():
    aspartate = get_residue_template("ASP")
    cysteine = get_residue_template("CYS")
    assert "*" not in aspartate.smiles
    assert "*" not in cysteine.smiles
    assert rdMolDescriptors.CalcMolFormula(aspartate.mol) == "C4H7NO3"
    assert rdMolDescriptors.CalcMolFormula(cysteine.mol) == "C3H7NOS"


def test_caps_and_capped_templates_are_unified_file_backed():
    for resname in ("ACE", "NME", "NH2"):
        assert get_residue_template(resname).source == "core_cap"
    for cap in ("ACE", "NME", "NH2"):
        capped = compose_capped_template("ASP", cap)
        assert Chem.MolFromSmiles(capped.smiles) is not None


def test_path_c_retains_noncap_hetatm_for_unified_mapping():
    orn = {"key": ("ORN", 1, True), "name": "ORN", "het": True}
    ala = {"key": ("ALA", 2, False), "name": "ALA", "het": False}
    ace = {"key": ("ACE", 3, True), "name": "ACE", "het": True}
    atoms = {
        orn["key"]: [{"num": 1}],
        ala["key"]: [{"num": 2}],
        ace["key"]: [{"num": 3}],
    }
    with patch.object(path_c, "get_pdb_atoms", side_effect=lambda _p, key, _c: atoms[key]):
        extended = path_c._build_ext_residues(
            [orn, ala, ace], "unused.pdb", {ala["key"]: ace["key"]}
        )

    assert [row[0]["name"] for row in extended] == ["ORN", "ALA"]
    assert extended[0][2] is None
    assert extended[1][2] == "ACE"


def test_path_c_cap_detection_is_independent_of_record_type():
    orn_key = ("ORN", 1, True)
    nh2_key = ("NH2", 2, True)
    atom_to_key = {10: orn_key, 20: nh2_key}
    assert path_c._map_het_to_std({10: [20]}, atom_to_key) == {
        nh2_key: orn_key
    }


def test_path_c_unknown_hetatm_fails_closed(tmp_path):
    pdb_path = tmp_path / "unknown_hetatm.pdb"
    pdb_path.write_text(
        "HETATM    1  C1  ZZZ L   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "END\n",
        encoding="ascii",
    )

    smiles, error = path_c.generate(str(pdb_path), "L")
    assert smiles is None
    assert "No Unified symbol mapping for PDB residue ZZZ" in error


def test_dynamic_orn_template_maps_complete_named_atoms():
    template = get_residue_template("ORN")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=7) == 0
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    backbone = dict(zip(("N", "CA", "CB", "C", "O"), parse_backbone(molecule)))
    name_by_index = {
        index: name for name, index in backbone.items() if index is not None
    }
    atoms = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "num": atom.GetIdx() + 1,
            "name": name_by_index.get(atom.GetIdx(), f"{atom.GetSymbol()}{atom.GetIdx()}"),
            "elem": atom.GetSymbol(),
            "xyz": (position.x, position.y, position.z),
        })
    mapping = map_pdb_atoms(template, atoms)
    assert template.symbol == "Orn"
    assert len(mapping) == template.mol.GetNumAtoms()


def test_legacy_mapping_api_rejects_nonunique_mapping():
    template = get_residue_template("ALA")
    evidence = {
        "mapping_complete": True,
        "mapping_injective": True,
        "template_mapping_complete": True,
        "mapping_unique": False,
    }
    with patch(
        "cycpep_master.paths.residue_template_factory."
        "map_pdb_atoms_with_evidence",
        return_value=({1: 0}, evidence),
    ):
        with pytest.raises(ValueError, match="ambiguous mapping"):
            map_pdb_atoms(template, [{"num": 1, "name": "CA", "elem": "C"}])


def test_audited_connectivity_drives_mapping_when_geometry_has_false_contact():
    template = get_residue_template("ORN")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=71) == 0
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    backbone = dict(zip(("N", "CA", "CB", "C", "O"), parse_backbone(molecule)))
    name_by_index = {
        index: name for name, index in backbone.items() if index is not None
    }
    atoms = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "num": atom.GetIdx() + 1,
            "name": name_by_index.get(atom.GetIdx(), f"X{atom.GetIdx()}"),
            "elem": atom.GetSymbol(),
            "xyz": (position.x, position.y, position.z),
        })
    observed_edges = {
        tuple(sorted((bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1)))
        for bond in molecule.GetBonds()
    }
    false_pair = next(
        (left, right)
        for left in range(molecule.GetNumAtoms())
        for right in range(left + 1, molecule.GetNumAtoms())
        if molecule.GetBondBetweenAtoms(left, right) is None
        and molecule.GetAtomWithIdx(left).GetSymbol() == "C"
        and molecule.GetAtomWithIdx(right).GetSymbol() == "C"
    )
    left_xyz = atoms[false_pair[0]]["xyz"]
    atoms[false_pair[1]]["xyz"] = (left_xyz[0] + 1.2, left_xyz[1], left_xyz[2])

    with pytest.raises(ValueError, match="No complete atom mapping"):
        map_pdb_atoms_with_evidence(template, atoms)

    mapping, evidence = map_pdb_atoms_with_evidence(
        template, atoms, observed_edges=observed_edges
    )
    assert len(mapping) == molecule.GetNumAtoms()
    assert evidence["mapping_unique"] is True
    assert evidence["mapping_method"] == "element_adjacency_audited_connectivity"


def test_mapping_evidence_binds_unified_graph_ports_and_uniqueness():
    template = get_residue_template("ARG")
    name_map = standard_pdb_atom_name_map("ARG", template.smiles)
    atoms = [
        {
            "num": index + 1,
            "name": name,
            "elem": template.mol.GetAtomWithIdx(template_index).GetSymbol(),
        }
        for index, (name, template_index) in enumerate(name_map.items())
    ]
    mapping, evidence = map_pdb_atoms_with_evidence(template, atoms)
    assert len(mapping) == len(atoms)
    assert evidence["unified_symbol"] == "R"
    assert evidence["unified_source"] == "core"
    assert len(evidence["monomer_graph_sha256"]) == 64
    expected_free = Chem.MolToSmiles(
        Chem.MolFromSmiles("N[C@@H](CCCNC(=[NH2+])N)C(=O)O"),
        canonical=True,
        isomericSmiles=True,
    )
    assert evidence["free_monomer_graph_sha256"] == hashlib.sha256(
        expected_free.encode("utf-8")
    ).hexdigest()
    assert evidence["rgroup_defaults"] == {
        "R1": template.r1, "R2": template.r2, "R3": template.r3,
    }
    assert evidence["mapping_method"] == "standard_pdb_atom_names"
    assert evidence["mapping_complete"]
    assert evidence["mapping_injective"]
    assert evidence["template_mapping_complete"]
    assert evidence["unmapped_template_heavy_atom_count"] == 0
    assert evidence["mapping_unique"]
    assert evidence["external_attachment_mapping_unique"]


@pytest.mark.parametrize(
    ("residue_name", "leaving_name", "anchor_name"),
    [("ASP", "OD2", "CG"), ("GLU", "OE2", "CD")],
)
def test_explicit_r3_consumption_defers_only_the_carboxyl_leaving_oxygen(
    residue_name, leaving_name, anchor_name
):
    template = get_residue_template(residue_name)
    name_map = standard_pdb_atom_name_map(residue_name, template.smiles)
    serial_by_name = {
        name: serial for serial, name in enumerate(name_map, start=1)
    }
    atoms = [
        {
            "num": serial_by_name[name],
            "name": name,
            "elem": template.mol.GetAtomWithIdx(template_index).GetSymbol(),
        }
        for name, template_index in name_map.items()
        if name != leaving_name
    ]

    mapping, evidence = map_pdb_atoms_with_evidence(
        template,
        atoms,
        consumed_r3_serials={serial_by_name[anchor_name]},
    )
    strict_mapping = map_pdb_atoms(
        template,
        atoms,
        consumed_r3_serials={serial_by_name[anchor_name]},
    )

    assert len(mapping) == len(atoms)
    assert strict_mapping == mapping
    assert evidence["mapping_method"] == (
        "standard_pdb_atom_names_with_consumed_r3_leaving_atom"
    )
    assert evidence["template_mapping_complete"] is True
    assert evidence["unmapped_template_heavy_atom_count"] == 1
    assert evidence["effective_unmapped_template_heavy_atom_count"] == 0
    assert evidence["deferred_consumed_r3_template_atoms"] == [{
        "template_atom_index": name_map[leaving_name],
        "atom_name": leaving_name,
    }]
    assert evidence["deferred_consumed_r3_anchor_serials"] == [
        serial_by_name[anchor_name]
    ]


@pytest.mark.parametrize(
    ("missing_names", "source_bound_name"),
    [
        ({"OE2"}, None),
        ({"OE2"}, "N"),
        ({"OE1"}, "CD"),
        ({"OE2", "CB"}, "CD"),
    ],
)
def test_consumed_glutamate_r3_does_not_authorize_other_missing_atoms(
    missing_names, source_bound_name
):
    template = get_residue_template("GLU")
    name_map = standard_pdb_atom_name_map("GLU", template.smiles)
    serial_by_name = {
        name: serial for serial, name in enumerate(name_map, start=1)
    }
    atoms = [
        {
            "num": serial_by_name[name],
            "name": name,
            "elem": template.mol.GetAtomWithIdx(template_index).GetSymbol(),
        }
        for name, template_index in name_map.items()
        if name not in missing_names
    ]
    source_bound = (
        {serial_by_name[source_bound_name]} if source_bound_name else set()
    )

    _mapping, evidence = map_pdb_atoms_with_evidence(
        template, atoms, consumed_r3_serials=source_bound
    )

    assert evidence["template_mapping_complete"] is False
    assert evidence["effective_unmapped_template_heavy_atom_count"] > 0
    assert evidence["deferred_consumed_r3_template_atoms"] == []
    with pytest.raises(ValueError, match="Incomplete atom mapping"):
        map_pdb_atoms(
            template, atoms, consumed_r3_serials=source_bound
        )


@pytest.mark.parametrize("root_atom", [0, 1, 3, 7])
def test_arg_pdb_terminal_nitrogen_names_follow_template_chemistry(root_atom):
    template = get_residue_template("ARG")
    reordered_smiles = Chem.MolToSmiles(
        template.mol,
        canonical=False,
        isomericSmiles=True,
        rootedAtAtom=root_atom,
    )

    name_map = standard_pdb_atom_name_map("ARG", reordered_smiles)
    molecule = Chem.MolFromSmiles(reordered_smiles)
    cz_index = name_map["CZ"]
    nh1_index = name_map["NH1"]
    nh2_index = name_map["NH2"]

    assert molecule.GetAtomWithIdx(nh1_index).GetFormalCharge() == 0
    assert molecule.GetBondBetweenAtoms(
        cz_index, nh1_index
    ).GetBondType() == Chem.BondType.SINGLE
    assert molecule.GetAtomWithIdx(nh2_index).GetFormalCharge() == 1
    assert molecule.GetBondBetweenAtoms(
        cz_index, nh2_index
    ).GetBondType() == Chem.BondType.DOUBLE


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


def _write_embedded_cyclic_trialanine(tmp_path):
    template = get_residue_template("ALA")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=83) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    names = {
        index: name
        for name, index in standard_pdb_atom_name_map(
            "ALA", template.smiles
        ).items()
    }
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
    path = tmp_path / "embedded-cyclic-trialanine.pdb"
    path.write_text(
        "\n".join([
            _link_record("N", "ALA", 1, "C", "ALA", 3),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    return path


def test_assembled_coordinate_stereochemistry_passes_for_explicit_cycle(tmp_path):
    path = _write_embedded_cyclic_trialanine(tmp_path)

    _smiles, error, evidence = path_a.generate_with_evidence(str(path), "A")

    assert error is None
    assert evidence["assembled_coordinate_stereochemistry"]["passed"] is True


def test_assembled_coordinate_stereochemistry_reports_inverted_graph_atom(tmp_path):
    path = _write_embedded_cyclic_trialanine(tmp_path)
    combo, pdb_to_graph = path_a._build_combo(str(path), "A")
    molecule = combo.GetMol()
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    graph_index = next(
        index
        for index, assignment in Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
        if assignment != "?"
    )
    combo.GetAtomWithIdx(graph_index).InvertChirality()
    residues = path_a.get_res_seq(str(path), "A")
    atoms_by_serial = {
        int(atom["num"]): atom
        for residue in residues
        for atom in path_a.get_pdb_atoms(str(path), residue["key"], "A")
    }

    evidence = path_a._assembled_coordinate_stereochemistry_evidence(
        combo, pdb_to_graph, atoms_by_serial
    )

    assert evidence["passed"] is False
    assert evidence["mismatch_graph_atom_indices"] == [graph_index]


def test_emergent_stereochemistry_is_audited_without_coordinate_assignment():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC(O)C(=O)O"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=101) == 0
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    combo = Chem.RWMol(molecule)
    pdb_to_graph = {index + 1: index for index in range(molecule.GetNumAtoms())}
    atoms_by_serial = {}
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        atoms_by_serial[atom.GetIdx() + 1] = {
            "name": f"{atom.GetSymbol()}{atom.GetIdx()}",
            "xyz": (point.x, point.y, point.z),
        }
    before = combo.GetAtomWithIdx(1).GetChiralTag()

    evidence = path_a._audit_emergent_stereochemistry_from_coordinates(
        combo, pdb_to_graph, atoms_by_serial, preexisting_unassigned=()
    )

    assert evidence["passed"] is False
    assert evidence["emergent_unassigned_graph_atom_indices"] == [1]
    assert evidence["coordinate_observations"][0][
        "coordinate_observed_cip"
    ] in {"R", "S"}
    assert evidence["coordinate_assignment_applied"] is False
    assert combo.GetAtomWithIdx(1).GetChiralTag() == before


def _alanine_atoms(*, omit_last_carbon=False):
    rows = []
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0), (3, 20.0)):
        for name, element, delta in (
            ("N", "N", 0.0),
            ("CA", "C", 1.4),
            ("CB", "C", 1.8),
            ("C", "C", 2.8),
            ("O", "O", 3.9),
        ):
            if omit_last_carbon and residue == 3 and name == "C":
                serial += 1
                continue
            rows.append(_pdb_atom(serial, name, residue, offset + delta, 0.0, 0.0, element=element))
            serial += 1
    return rows


def _named_residue_atoms(residue_name, residue_number, serial_start):
    template = get_residue_template(residue_name)
    name_map = standard_pdb_atom_name_map(residue_name, template.smiles)
    rows = []
    serial_by_name = {}
    for offset, (name, template_index) in enumerate(name_map.items()):
        serial = serial_start + offset
        atom = template.mol.GetAtomWithIdx(template_index)
        rows.append(_pdb_atom(
            serial,
            name,
            residue_number,
            residue_number * 10.0 + offset,
            0.0,
            0.0,
            residue_name=residue_name,
            element=atom.GetSymbol(),
        ))
        serial_by_name[name] = serial
    return rows, serial_by_name, serial_start + len(name_map)


def test_public_path_a_rejects_unledgered_standard_residue_atom_loss(tmp_path):
    rows, _serials, _next_serial = _named_residue_atoms("ALA", 1, 1)
    rows = [row for row in rows if row[12:16].strip() != "CB"]
    path = tmp_path / "alanine-missing-cb.pdb"
    path.write_text("\n".join([*rows, "END"]) + "\n", encoding="ascii")

    smiles, error = path_a.generate(str(path), "A")

    assert smiles is None
    assert "Incomplete atom mapping for Unified monomer A" in error


def _write_typed_r3_to_terminal_r2(tmp_path, first_residue, atom_name, mode):
    rows = []
    serial = 1
    serial_maps = []
    for position, residue_name in enumerate((first_residue, "ALA", "ALA"), 1):
        residue_rows, serial_map, serial = _named_residue_atoms(
            residue_name, position, serial
        )
        rows.extend(residue_rows)
        serial_maps.append(serial_map)
    if mode == "link":
        record = _link_record(atom_name, first_residue, 1, "C", "ALA", 3)
    else:
        record = (
            f"CONECT{serial_maps[0][atom_name]:5d}{serial_maps[2]['C']:5d}"
        )
    path = tmp_path / f"{first_residue.lower()}-{mode}.pdb"
    path.write_text("\n".join([record, *rows, "END"]) + "\n", encoding="ascii")
    return path


def test_path_a_explicit_closure_is_traced_into_identity(tmp_path):
    path = tmp_path / "explicit-cycle.pdb"
    path.write_text(
        "\n".join([
            _link_record("N", "ALA", 1, "C", "ALA", 3),
            *_alanine_atoms(),
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    smiles, error, evidence = path_a.generate_with_evidence(str(path), "A")
    assert error is None
    assert Chem.MolFromSmiles(smiles) is not None
    assert evidence["closure_count"] == 1
    assert evidence["all_closure_endpoints_resolved_uniquely"]
    assert evidence["all_closures_materialized"]
    assert evidence["all_closures_explicit"]
    assert evidence["all_closures_identity_determining"]
    closure = evidence["closure_evidence"][0]
    assert closure["evidence_source"] == "link"
    assert closure["counterfactual_status"] == "success"
    assert closure["baseline_inchikey"] == evidence["output_inchikey"]
    assert closure["counterfactual_inchikey"] != evidence["output_inchikey"]


def test_link_only_lactam_consumes_attachment_cap_and_occupies_r2(tmp_path):
    path = _write_typed_r3_to_terminal_r2(tmp_path, "LYS", "NZ", "link")
    smiles, error, evidence = path_a.generate_with_evidence(str(path), "A")
    assert error is None
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    positive_nitrogens = [
        atom for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1
    ]
    assert len(positive_nitrogens) == 1
    assert positive_nitrogens[0].GetTotalValence() == 4
    assert evidence["terminal_r2_materialization"]["status"] == (
        "not_materialized_port_occupied"
    )
    assert [3, "R2"] in evidence["terminal_r2_materialization"][
        "occupied_ports"
    ]


def test_r3_r2_lactone_does_not_receive_free_terminal_hydroxyl(tmp_path):
    path = _write_typed_r3_to_terminal_r2(tmp_path, "SER", "OG", "link")
    smiles, error, evidence = path_a.generate_with_evidence(str(path), "A")
    assert error is None
    assert Chem.MolFromSmiles(smiles) is not None
    assert evidence["terminal_r2_materialization"]["status"] == (
        "not_materialized_port_occupied"
    )


def test_link_and_conect_lactam_materialization_are_identity_equivalent(tmp_path):
    link_path = _write_typed_r3_to_terminal_r2(tmp_path, "LYS", "NZ", "link")
    conect_path = _write_typed_r3_to_terminal_r2(
        tmp_path, "LYS", "NZ", "conect"
    )
    link_smiles, link_error = path_a.generate(str(link_path), "A")
    conect_smiles, conect_error = path_a.generate(str(conect_path), "A")
    assert link_error is None
    assert conect_error is None
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(link_smiles)) == (
        Chem.MolToInchiKey(Chem.MolFromSmiles(conect_smiles))
    )


def test_ambiguous_carbon_carbon_closure_bond_order_fails_closed(tmp_path):
    rows = []
    serial = 1
    for position in range(1, 5):
        residue_rows, _serial_map, serial = _named_residue_atoms(
            "ALA", position, serial
        )
        rows.extend(residue_rows)
    path = tmp_path / "ambiguous-carbon-closure.pdb"
    path.write_text(
        "\n".join([
            _link_record("CB", "ALA", 1, "CB", "ALA", 4),
            *rows,
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    smiles, error = path_a.generate(str(path), "A")
    assert smiles is None
    assert "bond order is not uniquely determined" in error


def test_path_c_resolves_link_only_cap_attachment(tmp_path):
    ala = {"key": ("ALA", 1, False), "name": "ALA", "num": 1}
    ace = {"key": ("ACE", 2, True), "name": "ACE", "num": 2}
    path = tmp_path / "link-cap.pdb"
    path.write_text(
        _link_record("C", "ACE", 2, "N", "ALA", 1) + "\nEND\n",
        encoding="ascii",
    )
    assert path_c._resolve_cap_attachments(
        str(path), [ala, ace], {}, {}, "A"
    ) == {ace["key"]: ala["key"]}


def test_path_c_rejects_link_conect_cap_disagreement(tmp_path):
    ala_1 = {"key": ("ALA", 1, False), "name": "ALA", "num": 1}
    ace = {"key": ("ACE", 2, True), "name": "ACE", "num": 2}
    ala_3 = {"key": ("ALA", 3, False), "name": "ALA", "num": 3}
    path = tmp_path / "conflicting-cap.pdb"
    path.write_text(
        _link_record("C", "ACE", 2, "N", "ALA", 1) + "\nEND\n",
        encoding="ascii",
    )
    atom2key = {20: ace["key"], 30: ala_3["key"]}
    import pytest
    with pytest.raises(ValueError, match="LINK/CONECT cap attachment disagreement"):
        path_c._resolve_cap_attachments(
            str(path), [ala_1, ace, ala_3], {20: [30]}, atom2key, "A"
        )


def test_path_a_missing_closure_endpoint_is_not_qualified(tmp_path):
    path = tmp_path / "missing-endpoint.pdb"
    path.write_text(
        "\n".join([
            _link_record("N", "ALA", 1, "C", "ALA", 3),
            *_alanine_atoms(omit_last_carbon=True),
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    _smiles, _error, evidence = path_a.generate_with_evidence(str(path), "A")
    assert not evidence["atom_mapping_complete"]
    assert not evidence["all_closure_endpoints_resolved_uniquely"]
    assert not evidence["all_closures_materialized"]
    assert not evidence["all_closures_identity_determining"]


def test_path_e_geometry_closure_remains_diagnostic(tmp_path):
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
    smiles, error, evidence = path_a.generate_with_evidence(
        str(path), "A", geometric_cyclization=True
    )
    assert error is None
    assert Chem.MolFromSmiles(smiles) is not None
    assert evidence["closure_count"] == 1
    assert evidence["all_closures_materialized"]
    assert evidence["all_closures_identity_determining"]
    assert not evidence["all_closures_explicit"]
    assert evidence["closure_evidence"][0]["evidence_source"] == "geometry"


def test_unknown_residue_fails_closed():
    import pytest

    with pytest.raises(ValueError, match="No Unified symbol mapping"):
        get_residue_template("ZZZ")


def test_path_a_family_has_no_hard_coded_chemistry_dependency():
    root = Path(path_a.__file__).parent
    forbidden = ("ALL_SMILES", "AA_SMILES", "AC_AA_SMILES", "AA_NHME_SMILES")
    for filename in ("path_a.py", "path_c.py"):
        source = (root / filename).read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden)

    remediation_source = (root.parent / "remediation_v5.py").read_text(
        encoding="utf-8"
    )
    assert "core.data" not in remediation_source
