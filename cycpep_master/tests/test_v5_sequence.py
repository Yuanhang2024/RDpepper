from __future__ import annotations

from cycpep_master.sequence import build_molecule_from_sequence


def test_explicit_head_to_tail_sequence_builds_c3_specified_graph():
    result = build_molecule_from_sequence(
        "ACDEFG",
        cyclization="head-to-tail",
    )
    graph = result["chemical_graph"]

    assert graph["status"] == "MATERIALIZED"
    assert graph["evidence"]["chemical_rigor"] == "C3:S"
    assert graph["exact_v1"]["exactness_status"] == "EXACT"
    assert graph["topology_class"] == "head_to_tail"
    assert graph["full_inchikey"]


def test_inferred_cyclization_is_materialized_as_hypothesis():
    result = build_molecule_from_sequence(
        "ACDEFG",
        cyclization="infer",
    )
    graph = result["chemical_graph"]

    assert graph["status"] == "MATERIALIZED"
    assert graph["evidence"]["chemical_rigor"] == "C2:H"
    assert "CYCLIZATION_HYPOTHESIS_HEAD_TO_TAIL" in (
        result["warnings"]
    )


def test_stereo_override_and_terminal_caps_are_explicit():
    result = build_molecule_from_sequence(
        "ACD",
        cyclization="linear",
        stereochemistry={1: "D"},
        terminal_modifications={"N": "ACE", "C": "NME"},
    )
    graph = result["chemical_graph"]

    assert graph["status"] == "MATERIALIZED"
    assert graph["exact_v1"]["monomers"][0][
        "monomer_symbol"
    ] == "dA"
    assert len(graph["exact_v1"]["caps"]) == 2


def test_explicit_disulfide_uses_port_graph_not_name_guessing():
    result = build_molecule_from_sequence(
        "CAAC",
        cyclization={
            "bond_type": "SS",
            "src": {"position": 1, "port": "R3"},
            "dst": {"position": 4, "port": "R3"},
        },
    )
    graph = result["chemical_graph"]

    assert graph["status"] == "MATERIALIZED"
    assert "SS" in {
        row["bond_type"]
        for row in graph["exact_v1"]["bonds"]
    }


def test_unknown_monomer_returns_not_materializable_artifact():
    result = build_molecule_from_sequence(
        "[NOT_A_MONOMER]AC",
        cyclization="linear",
    )
    graph = result["chemical_graph"]

    assert graph["status"] == "PARTIAL"
    assert graph["evidence"]["chemical_rigor"] == "C1:H"
    assert graph["exact_v1"]["exactness_status"] == "ABSTAIN"
    assert graph["smiles"] is None
    assert result["alternatives"][0]["artifact_type"] == (
        "SymbolicChemicalGraphCandidate"
    )


def test_long_sequence_is_warning_not_hard_rejection():
    result = build_molecule_from_sequence(
        "A" * 31,
        cyclization="head-to-tail",
    )

    assert result["chemical_graph"]["status"] == "MATERIALIZED"
    assert "SEQUENCE_LENGTH_EXTERNAL_STRUCTURE_RECOMMENDED" in (
        result["warnings"]
    )


def test_physiological_microstate_is_explicitly_marked():
    result = build_molecule_from_sequence(
        "ARRA",
        cyclization="head-to-tail",
        protonation="physiological",
    )

    assert result["chemical_graph"]["status"] == "MATERIALIZED"
    assert "PHYSIOLOGICAL_MICROSTATE_HEURISTIC" in result["warnings"]
    assert result["chemical_graph"]["microstate_policy"] == (
        "physiological"
    )
    assert result["chemical_graph"]["evidence"][
        "chemical_rigor"
    ] == "C2:H"
    assert result["chemical_graph"]["parent_full_inchikey"]


def test_explicit_bond_type_contradiction_abstains():
    result = build_molecule_from_sequence(
        "KAD",
        cyclization={
            "bond_type": "SS",
            "src": {"position": 1, "port": "R3"},
            "dst": {"position": 3, "port": "R3"},
        },
    )

    graph = result["chemical_graph"]
    assert graph["status"] == "PARTIAL"
    assert graph["evidence"]["chemical_rigor"] == "C1:H"
    assert graph["exact_v1"]["exactness_status"] == "ABSTAIN"
    assert "EXPLICIT_BOND_TYPE_MISMATCH" in (
        graph["exact_v1"]["reason_codes"]
    )
