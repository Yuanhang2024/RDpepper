"""Focused tests for the native mmCIF graph adapter."""

from __future__ import annotations

from pathlib import Path

from cycpep_master.core.native_mmcif_graph import (
    list_peptide_chains,
    read_native_mmcif,
)


_HEADER = """data_native_test
#
loop_
_entity.id
_entity.type
1 polymer
2 polymer
#
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
2 'polypeptide(L)'
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
#
loop_
_chem_comp.id
_chem_comp.type
LIG 'L-PEPTIDE LINKING'
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
LIG C1 C
LIG N1 N
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
LIG C1 N1 sing
#
"""

_ATOM_TAGS = """loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_formal_charge
"""


def _atom(
    serial: int,
    name: str,
    element: str,
    comp: str,
    chain: str,
    entity: int,
    seq: int,
    x: float,
    *,
    charge: str = "?",
    y: float = 0.0,
    z: float = 0.0,
) -> str:
    return (
        f"ATOM {serial} {element} {name} {comp} {chain} {entity} {seq} "
        f"{name} {comp} {chain} {seq} {x:.3f} {y:.3f} {z:.3f} "
        f"1.00 20.00 1 ? {charge}\n"
    )


def _conn(
    identifier: str,
    chain_1: str,
    seq_1: int,
    atom_1: str,
    chain_2: str,
    seq_2: int,
    atom_2: str,
    *,
    model_1: str = "?",
    model_2: str = "?",
) -> str:
    return (
        f"{identifier} covale {chain_1} {seq_1} LIG {atom_1} "
        f"{chain_2} {seq_2} LIG {atom_2} "
        f"{chain_1} {seq_1} LIG {atom_1} {chain_2} {seq_2} LIG {atom_2} "
        f"sing {model_1} {model_2}\n"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    include_connection: bool = True,
    partial: bool = False,
) -> Path:
    path = tmp_path / "native.cif"
    text = _HEADER + _ATOM_TAGS
    text += _atom(1, "C1", "C", "LIG", "A", 1, 1, 0.0, y=0.0)
    text += _atom(2, "N1", "N", "LIG", "A", 1, 1, 1.3, y=0.0)
    text += _atom(3, "C1", "C", "LIG", "B", 2, 1, 0.0, y=4.0)
    text += _atom(4, "N1", "N", "LIG", "B", 2, 1, 1.3, y=4.0)
    if partial:
        # Keep the row and its source ID, but expose incomplete source fields.
        text += _atom(5, "O1", "?", "LIG", "A", 1, 2, 2.0, y=0.0)
        text = text.replace(
            "ATOM 5 ? O1 LIG A 1 2 O1 LIG A 2 2.000 0.000 0.000 1.00 20.00 1 ? ?",
            "ATOM 5 ? O1 LIG A 1 2 O1 LIG A 2 ? 0.000 0.000 1.00 20.00 1 ? ?",
        )
    text += "#\n"
    if include_connection:
        text += """loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_atom_id
_struct_conn.pdbx_value_order
_struct_conn.pdbx_ptnr1_PDB_model_num
_struct_conn.pdbx_ptnr2_PDB_model_num
"""
        text += _conn("c1", "A", 1, "N1", "B", 1, "C1")
        text += "#\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_single_chain_keeps_elements_components_and_mapping(tmp_path):
    path = _write_fixture(tmp_path, include_connection=False)

    chains = list_peptide_chains(path)
    assert [chain.label_asym_id for chain in chains] == ["A", "B"]
    assert all(chain.peptide_bearing for chain in chains)
    assert chains[0].entity_id == "1"
    assert chains[0].auth_asym_id == "A"

    graph = read_native_mmcif(path, chain_ids="A")
    assert graph.selected_chain_ids == ("A",)
    assert [atom.element for atom in graph.atoms] == ["C", "N"]
    assert graph.atoms[0].coordinates == (0.0, 0.0, 0.0)
    assert graph.atoms[0].group_pdb == "ATOM"
    assert graph.atoms[0].b_iso == 20.0
    assert [bond.source for bond in graph.bonds] == ["chem_comp_bond"]
    assert graph.bonds[0].resolved is True
    assert graph.bonds[0].within_selection is True


def test_multichain_struct_conn_resolves_native_label_addresses(tmp_path):
    path = _write_fixture(tmp_path)
    graph = read_native_mmcif(path, chain_ids=["A", "B"])

    explicit = [bond for bond in graph.bonds if bond.source == "struct_conn"]
    assert len(explicit) == 1
    assert explicit[0].atom_id_1 == "2"
    assert explicit[0].atom_id_2 == "3"
    assert explicit[0].connection_type == "covale"
    assert explicit[0].within_selection is True
    assert explicit[0].endpoint_1.label_asym_id == "A"
    assert explicit[0].endpoint_2.label_asym_id == "B"


def test_partial_atom_fields_are_retained_and_warned(tmp_path):
    path = _write_fixture(tmp_path, include_connection=False, partial=True)
    graph = read_native_mmcif(path, chain_ids="A")

    assert len(graph.atoms) == 3
    partial = next(atom for atom in graph.atoms if atom.atom_id == "5")
    assert partial.element is None
    assert partial.coordinates is None
    assert any(item.startswith("atom_element_missing:5") for item in graph.warnings)
    assert any(item.startswith("atom_coordinate_incomplete:5") for item in graph.warnings)


def test_unresolved_explicit_connection_is_preserved_as_warning(tmp_path):
    path = _write_fixture(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace("c1 covale A 1 LIG N1 B 1 LIG C1", "c1 covale A 99 LIG N1 B 1 LIG C1")
    path.write_text(text, encoding="utf-8")

    graph = read_native_mmcif(path, chain_ids=["A", "B"])
    explicit = [bond for bond in graph.bonds if bond.source == "struct_conn"]
    assert len(explicit) == 1
    assert explicit[0].resolved is False
    assert explicit[0].atom_id_1 is None
    assert explicit[0].atom_id_2 == "3"
    assert "struct_conn_endpoint_unresolved:c1:1" in graph.warnings


def test_native_graph_projection_is_deterministic(tmp_path):
    path = _write_fixture(tmp_path)
    first = read_native_mmcif(path, chain_ids=["B", "A"]).to_dict()
    second = read_native_mmcif(path, chain_ids=["A", "B"]).to_dict()
    assert first == second


def test_selected_graph_excludes_unselected_chain_records(tmp_path):
    path = _write_fixture(tmp_path)
    graph = read_native_mmcif(path, chain_ids="A")

    assert [chain.label_asym_id for chain in graph.chains] == ["A"]
    assert [entity.entity_id for entity in graph.entities] == ["1"]
    assert graph.as_graph()["entities"][0]["entity_id"] == "1"
    assert graph.as_graph()["provenance"]["selected_chain_ids"] == ["A"]


def test_invalid_element_and_nonfinite_coordinate_are_retained_as_partial_atom(tmp_path):
    path = _write_fixture(tmp_path, include_connection=False)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "ATOM 2 N N1 LIG A 1 1 N1 LIG A 1 1.300 0.000 0.000 1.00 20.00 1 ? ?",
        "ATOM 2 XX N1 LIG A 1 1 N1 LIG A 1 nan 0.000 0.000 1.00 20.00 1 ? ?",
    )
    path.write_text(text, encoding="utf-8")

    graph = read_native_mmcif(path, chain_ids="A")
    atom = next(atom for atom in graph.atoms if atom.atom_id == "2")

    assert atom.element is None
    assert atom.element_raw == "XX"
    assert atom.coordinates is None
    assert "atom_element_unrecognized:2:XX" in graph.warnings
    assert any(item.startswith("atom_coordinate_invalid:2:x") for item in graph.warnings)
    assert any(item.startswith("atom_coordinate_incomplete:2") for item in graph.warnings)


def test_peptide_only_does_not_silently_include_nonpeptide_chains(tmp_path):
    path = _write_fixture(tmp_path, include_connection=False)
    text = path.read_text(encoding="utf-8")
    text = text.replace("'polypeptide(L)'", "'polydeoxyribonucleotide'")
    text = text.replace("'L-PEPTIDE LINKING'", "'NON-POLYMER'")
    path.write_text(text, encoding="utf-8")

    graph = read_native_mmcif(path, peptide_only=True)

    assert graph.selected_chain_ids == ()
    assert graph.atoms == ()
    assert "no_peptide_bearing_chain_found" in graph.warnings


_GLU_HEADER = """data_native_alt_test
#
loop_
_entity.id
_entity.type
1 polymer
2 polymer
#
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
2 'polypeptide(L)'
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
#
loop_
_chem_comp.id
_chem_comp.type
GLU 'L-PEPTIDE LINKING'
GLY 'L-PEPTIDE LINKING'
LIG 'L-PEPTIDE LINKING'
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
GLU N N
GLU CA C
GLU C C
GLU O O
GLU CB C
GLU CG C
GLU CD C
GLU OE1 O
GLU OE2 O
GLY N N
GLY CA C
GLY C C
GLY O O
LIG C1 C
LIG N1 N
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
GLU N CA sing
GLU CA C sing
GLU C O sing
GLU CA CB sing
GLU CB CG sing
GLU CG CD sing
GLU CD OE1 sing
GLU CD OE2 sing
GLY N CA sing
GLY CA C sing
GLY C O sing
LIG C1 N1 sing
#
"""

_ALT_ATOM_TAGS = """loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.label_alt_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_formal_charge
"""


def _alt_atom(
    serial: int,
    name: str,
    element: str,
    comp: str,
    chain: str,
    entity: int,
    seq: int,
    x: float,
    *,
    alt: str = "?",
    comp_label: str | None = None,
    y: float = 0.0,
    z: float = 0.0,
) -> str:
    # Label and author comp ids share one value by default; ``comp_label``
    # overrides the label field (e.g. '.' for a missing label) so the test can
    # exercise the label->auth fallback without touching auth_comp_id.
    label_comp = comp if comp_label is None else comp_label
    return (
        f"ATOM {serial} {element} {name} {label_comp} {chain} {entity} {seq} "
        f"{name} {comp} {chain} {seq} {alt} {x:.3f} {y:.3f} {z:.3f} "
        f"1.00 20.00 1 ? ?\n"
    )


def _glu_sidechain(chain: str, entity: int, seq: int, start_serial: int) -> list[str]:
    """GLU backbone plus A/B alternate conformers on every side-chain atom.

    Mirrors the real PDB pattern that left alternate side-chain copies
    unbonded (1B2A-B13 / 1B2G-B10 family).
    """
    rows: list[str] = []
    serial = start_serial
    for name, element in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
        rows.append(_alt_atom(serial, name, element, "GLU", chain, entity,
                              seq, float(serial - start_serial + 1)))
        serial += 1
    for name, element in (("CB", "C"), ("CG", "C"), ("CD", "C"),
                          ("OE1", "O"), ("OE2", "O")):
        rows.append(_alt_atom(serial, name, element, "GLU", chain, entity,
                              seq, float(serial - start_serial + 1), alt="A"))
        serial += 1
        rows.append(_alt_atom(serial, name, element, "GLU", chain, entity,
                              seq, float(serial - start_serial + 1), alt="B"))
        serial += 1
    return rows


def _write_alt_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "alt_glu.cif"
    text = _GLU_HEADER + _ALT_ATOM_TAGS
    # chain A residue 1 = single-conformer GLU (the unchanged normal path).
    for serial, (name, element) in enumerate(
        (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"),
         ("CB", "C"), ("CG", "C"), ("CD", "C"), ("OE1", "O"), ("OE2", "O")),
        start=1,
    ):
        text += _alt_atom(serial, name, element, "GLU", "A", 1, 1, float(serial))
    # chain B residue 1 = GLU with A/B alternate conformers on the side chain.
    for row in _glu_sidechain("B", 2, 1, start_serial=21):
        text += row
    # chain B residue 2 = GLY with a *missing* label_comp_id ('.') but a
    # present auth_comp_id (the label->auth fallback keeps its bonds intact).
    for serial, (name, element) in enumerate(
        (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")), start=100,
    ):
        text += _alt_atom(serial, name, element, "GLY", "B", 2, 2,
                          float(serial - 99), comp_label=".", alt=".")
    text += "#\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_alt_conformer_copies_all_receive_intra_residue_bonds(tmp_path):
    """Every alternate-conformer atom copy must carry the component's bonds.

    Regression for the 1B2A-B13 / 1B2G-B10 family: one atom name held several
    alternate atom_site rows, but bond building kept only the first copy, so
    the remaining copies were isolated vertices that disconnected the graph.
    """
    path = _write_alt_fixture(tmp_path)
    graph = read_native_mmcif(path, chain_ids=["B"])

    bonded = {edge_atom for edge in graph.bonds for edge_atom in (edge.a, edge.b)}
    atom_ids = {atom.atom_id for atom in graph.atoms}

    # every observed atom copy (including every alternate A/B copy) is bonded.
    assert atom_ids == bonded

    # Same-conformer pairing: CB-A is bonded to CG-A (not to CG-B).  Cross-
    # conformer pairs are deliberately not guessed.
    edges = {(bond.a, bond.b) for bond in graph.bonds}
    cb_a = next(a for a in graph.atoms if a.label_atom_id == "CB" and a.alt_id == "A")
    cb_b = next(a for a in graph.atoms if a.label_atom_id == "CB" and a.alt_id == "B")
    cg_a = next(a for a in graph.atoms if a.label_atom_id == "CG" and a.alt_id == "A")
    cg_b = next(a for a in graph.atoms if a.label_atom_id == "CG" and a.alt_id == "B")
    assert (cb_a.atom_id, cg_a.atom_id) in edges
    assert (cb_b.atom_id, cg_b.atom_id) in edges
    assert (cb_a.atom_id, cg_b.atom_id) not in edges
    assert (cb_b.atom_id, cg_a.atom_id) not in edges


def test_label_comp_id_missing_still_bonds_via_auth_fallback(tmp_path):
    """A residue with label_comp_id='.' but auth_comp_id present keeps bonds.

    The native graph looks up component bonds by ``label_comp_id or
    auth_comp_id``; a resolved author comp id is a sufficient fallback and the
    residue must remain fully connected.
    """
    path = _write_alt_fixture(tmp_path)
    graph = read_native_mmcif(path, chain_ids=["B"])

    gly_atoms = [a for a in graph.atoms if a.auth_comp_id == "GLY"]
    assert gly_atoms and all(a.label_comp_id is None for a in gly_atoms)
    bonded = {edge_atom for edge in graph.bonds for edge_atom in (edge.a, edge.b)}
    assert {a.atom_id for a in gly_atoms} <= bonded
    assert any(bond.source == "chem_comp_bond" for bond in graph.bonds)
