from __future__ import annotations

import copy
import hashlib
import json

import pytest
from rdkit import Chem

from cycpep_master.core.mmcif_chem_comp import (
    EmbeddedChemCompError,
    extract_embedded_chem_comp_templates,
    materialize_standalone_peptide_component,
    resolve_embedded_peptide_component,
)
from cycpep_master.sequence import build_molecule_from_sequence
from cycpep_master.core.monomer_resolution import (
    monomer_resolution_context,
)
from cycpep_master.exact_v1 import map_to_exact_v1


def _atom(name: str, element: str, *, aromatic: bool = False, stereo: str = "N"):
    return {
        "_chem_comp_atom.comp_id": "XTY",
        "_chem_comp_atom.atom_id": name,
        "_chem_comp_atom.type_symbol": element,
        "_chem_comp_atom.pdbx_aromatic_flag": "Y" if aromatic else "N",
        "_chem_comp_atom.pdbx_stereo_config": stereo,
    }


def _bond(left: str, right: str, order: str = "sing", *, aromatic: bool = False):
    return {
        "_chem_comp_bond.comp_id": "XTY",
        "_chem_comp_bond.atom_id_1": left,
        "_chem_comp_bond.atom_id_2": right,
        "_chem_comp_bond.value_order": order,
        "_chem_comp_bond.pdbx_aromatic_flag": "Y" if aromatic else "N",
    }


def _tyrosine_like_component(*, ca_stereo: str = "S") -> dict:
    atoms = [
        _atom("N", "N"), _atom("H", "H"), _atom("H2", "H"),
        _atom("CA", "C", stereo=ca_stereo), _atom("HA", "H"),
        _atom("C", "C"), _atom("O", "O"), _atom("OXT", "O"),
        _atom("HXT", "H"), _atom("CB", "C"), _atom("HB2", "H"),
        _atom("HB3", "H"),
        *[
            _atom(name, "C", aromatic=True)
            for name in ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")
        ],
        _atom("OH", "O"), _atom("HO", "H"),
    ]
    bonds = [
        _bond("N", "H"), _bond("N", "H2"), _bond("N", "CA"),
        _bond("CA", "HA"), _bond("CA", "C"), _bond("CA", "CB"),
        _bond("C", "O", "doub"), _bond("C", "OXT"),
        _bond("OXT", "HXT"), _bond("CB", "HB2"), _bond("CB", "HB3"),
        _bond("CB", "CG"),
        _bond("CG", "CD1", aromatic=True),
        _bond("CD1", "CE1", aromatic=True),
        _bond("CE1", "CZ", aromatic=True),
        _bond("CZ", "CE2", aromatic=True),
        _bond("CE2", "CD2", aromatic=True),
        _bond("CD2", "CG", aromatic=True),
        _bond("CZ", "OH"), _bond("OH", "HO"),
    ]
    component = {
        "component_id": "XTY",
        "source_input_sha256": "0" * 64,
        "component_type": "L-peptide linking",
        "atom_rows": atoms,
        "bond_rows": bonds,
    }
    return _seal(component)


def _alkene_component(*, bond_stereo: str = "E") -> dict:
    atoms = [
        _atom("N", "N"), _atom("H", "H"), _atom("H2", "H"),
        _atom("CA", "C", stereo="S"), _atom("HA", "H"),
        _atom("C", "C"), _atom("O", "O"), _atom("OXT", "O"),
        _atom("HXT", "H"), _atom("CB", "C"), _atom("HB", "H"),
        _atom("CG", "C"), _atom("HG", "H"), _atom("CD", "C"),
        _atom("HD1", "H"), _atom("HD2", "H"), _atom("HD3", "H"),
    ]
    bonds = [
        _bond("N", "H"), _bond("N", "H2"), _bond("N", "CA"),
        _bond("CA", "HA"), _bond("CA", "C"), _bond("CA", "CB"),
        _bond("C", "O", "doub"), _bond("C", "OXT"),
        _bond("OXT", "HXT"), _bond("CB", "HB"),
        _bond("CB", "CG", "doub"), _bond("CG", "HG"),
        _bond("CG", "CD"), _bond("CD", "HD1"), _bond("CD", "HD2"),
        _bond("CD", "HD3"),
    ]
    bonds[10]["_chem_comp_bond.pdbx_stereo_config"] = bond_stereo
    return _seal({
        "component_id": "XTY",
        "source_input_sha256": "1" * 64,
        "component_type": "L-peptide linking",
        "atom_rows": atoms,
        "bond_rows": bonds,
    })


def _seal(component: dict) -> dict:
    component.pop("component_snapshot_sha256", None)
    payload = json.dumps(
        component, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    component["component_snapshot_sha256"] = hashlib.sha256(payload).hexdigest()
    return component


def _observed(component: dict):
    heavy = [
        row for row in component["atom_rows"]
        if row["_chem_comp_atom.type_symbol"] != "H"
        and row["_chem_comp_atom.atom_id"] != "OXT"
    ]
    atoms = [
        {
            "num": index + 1,
            "name": row["_chem_comp_atom.atom_id"],
            "elem": row["_chem_comp_atom.type_symbol"],
        }
        for index, row in enumerate(heavy)
    ]
    serial = {atom["name"]: atom["num"] for atom in atoms}
    names = set(serial)
    edges = [
        (serial[row["_chem_comp_bond.atom_id_1"]], serial[row["_chem_comp_bond.atom_id_2"]])
        for row in component["bond_rows"]
        if row["_chem_comp_bond.atom_id_1"] in names
        and row["_chem_comp_bond.atom_id_2"] in names
    ]
    return atoms, edges


def test_extracts_hashable_embedded_component_rows(tmp_path):
    source = tmp_path / "component.cif"
    source.write_text(
        """data_fixture
loop_
_chem_comp.id
_chem_comp.type
_chem_comp.name
_chem_comp.formula
XAA 'L-peptide linking' 'example amino acid' 'C6 H15 N4 O2 1'
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_stereo_config
XAA N N N N
XAA CA C N S
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
XAA N CA sing N
""",
        encoding="ascii",
    )

    first = extract_embedded_chem_comp_templates(source)
    second = extract_embedded_chem_comp_templates(source)

    assert list(first) == ["XAA"]
    assert first == second
    assert first["XAA"]["component_type"] == "L-peptide linking"
    assert first["XAA"]["component_formal_charge"] == "1"
    assert first["XAA"]["component_formal_charge_source"] == (
        "terminal_formula_token"
    )
    assert len(first["XAA"]["component_snapshot_sha256"]) == 64


def test_resolves_aromatic_peptide_with_terminal_leaving_group_and_r3_h():
    component = _tyrosine_like_component()
    atoms, edges = _observed(component)

    result = resolve_embedded_peptide_component(
        component, atoms, edges, r3_atom_name="OH", r3_cap="H"
    )

    molecule = Chem.MolFromSmiles(result["free_smiles"])
    mapped = Chem.MolFromSmiles(result["r3_mapped_smiles"])
    assert molecule is not None and mapped is not None
    assert result["polymer_leaving_heavy_atom_names"] == ["OXT"]
    assert result["observed_connectivity_exact"] is True
    assert sum(bond.GetIsAromatic() for bond in molecule.GetBonds()) == 6
    assert Chem.FindMolChiralCenters(
        molecule, includeUnassigned=True, useLegacyImplementation=False
    ) == [(1, "S")]
    assert [atom.GetSymbol() for atom in mapped.GetAtoms() if atom.GetAtomMapNum() == 9003] == ["O"]


def test_materializes_standalone_component_with_source_ports():
    component = _tyrosine_like_component()

    result = materialize_standalone_peptide_component(
        component,
        r3_atom_name="OH",
        r3_cap="H",
    )
    molecule = Chem.MolFromSmiles(result["ported_cxsmiles"])
    labels = [
        atom.GetProp("atomLabel")
        for atom in molecule.GetAtoms()
        if atom.HasProp("atomLabel")
    ]

    assert labels.count("_R1") == 1
    assert labels.count("_R2") == 1
    assert labels.count("_R3") == 1
    assert result["r_groups"] == {
        "R1": "H", "R2": "OH", "R3": "H"
    }
    assert result["resolution_mode"] == "standalone_ccd_component"


def test_standalone_component_preserves_declared_formal_charge():
    component = _tyrosine_like_component()
    component["atom_rows"].append(_atom("H3", "H"))
    component["bond_rows"].append(_bond("N", "H3"))
    component["component_formal_charge"] = "1"
    component["component_formal_charge_source"] = (
        "terminal_formula_token"
    )
    _seal(component)

    result = materialize_standalone_peptide_component(component)
    free = Chem.MolFromSmiles(result["free_smiles"])
    ported = Chem.MolFromSmiles(result["ported_cxsmiles"])

    assert Chem.GetFormalCharge(free) == 1
    assert ported is not None


def test_standalone_component_preserves_unambiguous_e_alkene():
    result = materialize_standalone_peptide_component(
        _alkene_component(bond_stereo="E")
    )
    molecule = Chem.MolFromSmiles(result["free_smiles"])
    stereo_bonds = [
        bond.GetStereo()
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]

    assert stereo_bonds == [Chem.BondStereo.STEREOE]


def test_charged_component_template_reaches_exact_sequence_graph():
    component = _tyrosine_like_component()
    component["atom_rows"].append(_atom("H3", "H"))
    component["bond_rows"].append(_bond("N", "H3"))
    component["component_formal_charge"] = "1"
    component["component_formal_charge_source"] = (
        "terminal_formula_token"
    )
    _seal(component)
    context = {
        "component_templates": {"CHARGED_XTY": component},
        "component_ids": {"CHARGED_XTY": "XTY"},
        "include_persistent_user": False,
    }

    result = build_molecule_from_sequence(
        "[CHARGED_XTY]A",
        cyclization="linear",
        monomer_context=context,
    )
    graph = result["chemical_graph"]

    assert graph["status"] == "MATERIALIZED"
    assert graph["evidence"]["chemical_rigor"] == "C3:Q"
    assert graph["formal_charge"] == 1
    assert graph["exact_v1"]["exactness_status"] == "EXACT"
    assert graph["monomer_resolution"]["persistent_writes"] == 0


def test_known_ccd_identity_becomes_exact_alias_not_override():
    component = _tyrosine_like_component()
    context = {
        "component_templates": {"XTY": component},
        "component_ids": {"TYR": "XTY"},
        "include_persistent_user": False,
    }

    with monomer_resolution_context(
        context, required_symbols=["TYR"]
    ) as ledger:
        document = map_to_exact_v1("{nnr:TYR}A")

    assert ledger["status"] == "resolved"
    assert ledger["resolved"]["TYR"]["resolved_symbol"] == "Y"
    assert any(
        row.get("resolution_mode") == "standalone_ccd_exact_alias"
        for row in ledger["pdb_aliases"]
    )
    assert document["exactness_status"] == "EXACT"


def test_rejects_embedded_component_connectivity_conflict():
    component = _tyrosine_like_component()
    atoms, edges = _observed(component)

    with pytest.raises(EmbeddedChemCompError) as raised:
        resolve_embedded_peptide_component(
            component, atoms, edges[:-1], r3_atom_name="OH", r3_cap="H"
        )

    assert raised.value.code == "MMCIF_CHEM_COMP_OBSERVED_CONNECTIVITY_MISMATCH"
    assert raised.value.rejected is True


def test_fail_closed_when_embedded_stereocenter_is_undeclared():
    component = _tyrosine_like_component(ca_stereo="N")
    atoms, edges = _observed(component)

    with pytest.raises(EmbeddedChemCompError) as raised:
        resolve_embedded_peptide_component(
            component, atoms, edges, r3_atom_name="OH", r3_cap="H"
        )

    assert raised.value.code == "MMCIF_CHEM_COMP_STEREOCHEMISTRY_UNRESOLVED"
    assert raised.value.rejected is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda component: component.__setitem__("component_type", "non-polymer"),
        lambda component: component["atom_rows"].clear(),
        lambda component: component["bond_rows"].clear(),
    ],
)
def test_fail_closed_for_ineligible_or_incomplete_component(mutation):
    component = _tyrosine_like_component()
    atoms, edges = _observed(component)
    mutation(component)
    _seal(component)

    with pytest.raises(EmbeddedChemCompError):
        resolve_embedded_peptide_component(component, atoms, edges)


def test_rejects_tampered_snapshot_and_invalid_source_hash():
    component = _tyrosine_like_component()
    atoms, edges = _observed(component)
    component["component_type"] = "D-peptide linking"

    with pytest.raises(EmbeddedChemCompError) as tampered:
        resolve_embedded_peptide_component(component, atoms, edges)
    assert tampered.value.code == "MMCIF_CHEM_COMP_SOURCE_BINDING_INVALID"
    assert tampered.value.rejected is True

    component = _tyrosine_like_component()
    component["source_input_sha256"] = "z" * 64
    _seal(component)
    with pytest.raises(EmbeddedChemCompError) as malformed:
        resolve_embedded_peptide_component(component, atoms, edges)
    assert malformed.value.code == "MMCIF_CHEM_COMP_SOURCE_BINDING_INVALID"


def test_rejects_invalid_backbone_and_r3_cap_semantics():
    component = _tyrosine_like_component()
    atoms, edges = _observed(component)
    component["bond_rows"] = [
        row for row in component["bond_rows"]
        if {row["_chem_comp_bond.atom_id_1"], row["_chem_comp_bond.atom_id_2"]}
        != {"OXT", "HXT"}
    ]
    component["atom_rows"] = [
        row for row in component["atom_rows"]
        if row["_chem_comp_atom.atom_id"] != "HXT"
    ]
    _seal(component)

    with pytest.raises(EmbeddedChemCompError) as backbone:
        resolve_embedded_peptide_component(component, atoms, edges)
    assert backbone.value.code == "MMCIF_CHEM_COMP_PEPTIDE_BACKBONE_INVALID"
    assert backbone.value.rejected is True

    component = _tyrosine_like_component()
    with pytest.raises(EmbeddedChemCompError) as r3:
        resolve_embedded_peptide_component(
            component, atoms, edges, r3_atom_name="OH", r3_cap="OH"
        )
    assert r3.value.code == "MMCIF_CHEM_COMP_PORT_SEMANTICS_MISMATCH"


def test_rejects_charge_bond_stereo_and_duplicate_observed_graph():
    component = _tyrosine_like_component()
    atoms, edges = _observed(component)
    component["component_formal_charge"] = "1"
    _seal(component)
    with pytest.raises(EmbeddedChemCompError) as charge:
        resolve_embedded_peptide_component(component, atoms, edges)
    assert charge.value.code == "MMCIF_CHEM_COMP_FORMAL_CHARGE_MISMATCH"

    component = _tyrosine_like_component()
    component["bond_rows"][0]["_chem_comp_bond.pdbx_stereo_config"] = "E"
    _seal(component)
    with pytest.raises(EmbeddedChemCompError) as stereo:
        resolve_embedded_peptide_component(component, atoms, edges)
    assert stereo.value.code == "MMCIF_CHEM_COMP_BOND_STEREOCHEMISTRY_NOT_SUPPORTED"

    component = _tyrosine_like_component()
    duplicate_atoms = copy.deepcopy(atoms)
    duplicate_atoms[1]["num"] = duplicate_atoms[0]["num"]
    with pytest.raises(EmbeddedChemCompError) as serial:
        resolve_embedded_peptide_component(component, duplicate_atoms, edges)
    assert serial.value.code == "MMCIF_CHEM_COMP_OBSERVED_ATOM_MISMATCH"

    with pytest.raises(EmbeddedChemCompError) as edge:
        resolve_embedded_peptide_component(component, atoms, [*edges, edges[0]])
    assert edge.value.code == "MMCIF_CHEM_COMP_OBSERVED_CONNECTIVITY_MISMATCH"


def test_assigns_unique_valence_implied_positive_charge():
    component = _tyrosine_like_component()
    component["atom_rows"].append(_atom("H3", "H"))
    component["bond_rows"].append(_bond("N", "H3"))
    component["component_formal_charge"] = "1"
    component["component_formal_charge_source"] = "terminal_formula_token"
    _seal(component)
    atoms, edges = _observed(component)

    result = resolve_embedded_peptide_component(component, atoms, edges)
    molecule = Chem.MolFromSmiles(result["free_smiles"])

    assert molecule is not None
    assert Chem.GetFormalCharge(molecule) == 1


def test_supports_case_normalized_halogen_and_terminal_amide_r2_cap():
    component = _tyrosine_like_component()
    component["atom_rows"].append(_atom("CL1", "CL"))
    component["bond_rows"].append(_bond("CD1", "CL1"))
    _seal(component)
    atoms, edges = _observed(component)
    result = resolve_embedded_peptide_component(component, atoms, edges)
    assert "Cl" in result["free_smiles"]

    component = _tyrosine_like_component()
    for row in component["atom_rows"]:
        if row["_chem_comp_atom.atom_id"] == "OXT":
            row["_chem_comp_atom.atom_id"] = "N1"
            row["_chem_comp_atom.type_symbol"] = "N"
        elif row["_chem_comp_atom.atom_id"] == "HXT":
            row["_chem_comp_atom.atom_id"] = "HN11"
    for row in component["bond_rows"]:
        for field in ("_chem_comp_bond.atom_id_1", "_chem_comp_bond.atom_id_2"):
            if row[field] == "OXT":
                row[field] = "N1"
            elif row[field] == "HXT":
                row[field] = "HN11"
    component["atom_rows"].append(_atom("HN12", "H"))
    component["bond_rows"].append(_bond("N1", "HN12"))
    _seal(component)
    atoms, edges = _observed(component)
    result = resolve_embedded_peptide_component(component, atoms, edges)
    assert result["r2_cap_atom_name"] == "N1"
    assert result["r2_cap_element"] == "N"
    assert result["r2_cap_explicit_hydrogen_count"] == 2
    assert result["polymer_leaving_heavy_atom_names"] == []


def test_rejects_disconnected_embedded_component():
    component = _tyrosine_like_component()
    component["atom_rows"].append(_atom("F1", "F"))
    _seal(component)
    atoms, edges = _observed(component)

    with pytest.raises(EmbeddedChemCompError) as raised:
        resolve_embedded_peptide_component(component, atoms, edges)
    assert raised.value.code == "MMCIF_CHEM_COMP_DISCONNECTED"
    assert raised.value.rejected is True


@pytest.mark.parametrize(
    "payload",
    [
        """data_one\n_chem_comp.id XAA\n_chem_comp.type 'L-peptide linking'\n"
        "data_two\n_chem_comp.id XBB\n_chem_comp.type 'L-peptide linking'\n""",
        """data_fixture\nloop_\n_chem_comp.id\n_chem_comp.type\n"
        "XAA 'L-peptide linking'\nXAA 'L-peptide linking'\n""",
        """data_fixture\nloop_\n_chem_comp_atom.atom_id\n"
        "_chem_comp_atom.type_symbol\nN N\n""",
        """data_fixture\nloop_\n_chem_comp_bond.atom_id_1\n"
        "_chem_comp_bond.atom_id_2\n_chem_comp_bond.value_order\nN CA sing\n""",
    ],
)
def test_rejects_ambiguous_or_unbound_mmcif_categories(tmp_path, payload):
    source = tmp_path / "invalid.cif"
    source.write_text(payload, encoding="ascii")

    with pytest.raises(EmbeddedChemCompError) as raised:
        extract_embedded_chem_comp_templates(source)
    assert raised.value.rejected is True
