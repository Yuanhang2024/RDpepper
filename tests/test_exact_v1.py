from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from rdkit import Chem

from cycpep_master import application
from cycpep_master.core.cyclic_peptide_graph import (
    CyclicPeptideGraph,
    CyclicPeptideGraphError,
    PortBond,
    canonical_exact_v1_bytes,
)
from cycpep_master.exact_v1 import (
    ABSTAIN,
    EXACT,
    ExactV1Error,
    biln_to_exact_v1,
    chemical_graph_equivalent,
    edge_v1_to_exact_v1,
    exact_v1_equivalent,
    exact_v1_from_v6_result,
    exact_v1_to_biln,
    exact_v1_to_edge_v1,
    exact_v1_to_helm,
    exact_v1_to_json,
    exact_v1_to_legacy_v5,
    exact_v1_to_map,
    exact_v1_to_smiles,
    helm_to_exact_v1,
    legacy_v5_to_exact_v1,
    map_to_exact_v1,
    model_projection_equivalent,
    validate_exact_v1,
)


SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "exact_v1.schema.json"
    ).read_text(encoding="utf-8")
)


def _assert_exact(document):
    jsonschema.validate(document, SCHEMA)
    assert document["exactness_status"] == EXACT
    assert document["reason_codes"] == []
    assert document["graph_sha256"] == hashlib.sha256(
        canonical_exact_v1_bytes(document)
    ).hexdigest()


def test_head_to_tail_rotation_is_exact_equivalent():
    first = map_to_exact_v1("ACDEFG{cyc:N-C}")
    rotated = map_to_exact_v1("CDEFGA{cyc:N-C}")

    _assert_exact(first)
    _assert_exact(rotated)
    assert exact_v1_equivalent(first, rotated)
    assert canonical_exact_v1_bytes(first) == (
        canonical_exact_v1_bytes(rotated)
    )
    assert chemical_graph_equivalent(first, rotated)


def test_two_residue_head_to_tail_keeps_one_backbone_and_one_ht_bond():
    document = map_to_exact_v1("AC{cyc:N-C}")

    _assert_exact(document)
    assert [row["bond_type"] for row in document["bonds"]] == [
        "HT",
        "PEPTIDE",
    ]
    assert exact_v1_equivalent(document, document)
    assert exact_v1_equivalent(
        document, map_to_exact_v1(exact_v1_to_map(document))
    )


def test_non_symmetric_reversal_is_not_exact_equivalent():
    forward = map_to_exact_v1("ACDEFG{cyc:N-C}")
    reverse = map_to_exact_v1("GFEDCA{cyc:N-C}")

    assert not exact_v1_equivalent(forward, reverse)


@pytest.mark.parametrize(
    "left, right",
    [
        ("ACG{cyc:N-C}", "{nnr:dA}CG{cyc:N-C}"),
        ("ACD", "{nt:ACE}ACD"),
        ("KAD{cyc:1:R3-3:R3}", "KAD{cyc:N-C}"),
        ("CCAC{cyc:1:R3-4:R3}", "CCAC{cyc:2:R3-4:R3}"),
    ],
)
def test_stereo_caps_ports_and_topology_are_not_merged(left, right):
    left_document = map_to_exact_v1(left)
    right_document = map_to_exact_v1(right)

    _assert_exact(left_document)
    _assert_exact(right_document)
    assert not exact_v1_equivalent(
        left_document, right_document
    )


@pytest.mark.parametrize(
    "map_text, expected_count",
    [
        ("CAAC{cyc:1:R3-4:R3}", 1),
        (
            "CCCC{cyc:1:R3-2:R3}{cyc:3:R3-4:R3}",
            2,
        ),
        (
            "CCCCCC"
            "{cyc:1:R3-2:R3}"
            "{cyc:3:R3-4:R3}"
            "{cyc:5:R3-6:R3}",
            3,
        ),
    ],
)
def test_single_double_and_triple_disulfides(map_text, expected_count):
    document = map_to_exact_v1(map_text)

    _assert_exact(document)
    assert sum(
        row["bond_type"] == "SS" for row in document["bonds"]
    ) == expected_count


@pytest.mark.parametrize(
    "map_text, expected_type",
    [
        ("KAD{cyc:1:R3-3:R3}", "ISOPEPTIDE"),
        ("TAD{cyc:1:R3-3:R3}", "ESTER"),
        ("CAD{cyc:1:R3-3:R3}", "THIOETHER"),
        ("KAAA{cyc:1:R3-4:R2}", "SIDECHAIN_TO_TAIL"),
    ],
)
def test_port_chemistry_classifies_non_ht_closures(
    map_text, expected_type
):
    document = map_to_exact_v1(map_text)

    _assert_exact(document)
    assert expected_type in {
        row["bond_type"] for row in document["bonds"]
    }


@pytest.mark.parametrize(
    "map_text",
    [
        "{nt:ACE}ACD",
        "ACD{ct:NME}",
        "ACD{ct:NH2}",
        "{nt:ACE}ACD{ct:NME}",
    ],
)
def test_terminal_caps_are_hashed_and_roundtrip(map_text):
    document = map_to_exact_v1(map_text)

    _assert_exact(document)
    reparsed = map_to_exact_v1(exact_v1_to_map(document))
    assert exact_v1_equivalent(document, reparsed)
    assert exact_v1_to_edge_v1(document)["status"] == (
        "UNPROJECTABLE"
    )


def test_legacy_cap_inclusive_positions_are_audited_and_normalized():
    legacy = (
        "KGDGKGDFPD"
        "{cyc:2:R3-11:R3}"
        "{nt:ACE}{ct:NME}"
    )
    document = map_to_exact_v1(legacy)

    _assert_exact(document)
    assert document["source_kind"] == "map_legacy_cap_offset"
    assert document["normalization_codes"] == [
        "LEGACY_CAP_INCLUSIVE_POSITION_NORMALIZED"
    ]
    canonical_map = exact_v1_to_map(document)
    assert "{cyc:1:R3-10:R3}" in canonical_map
    assert exact_v1_equivalent(
        document, map_to_exact_v1(canonical_map)
    )
    application_result = application.convert_representation(
        "map", "exact_v1", legacy
    )
    assert application_result["status"] == "success"
    assert application_result["data"]["value"][
        "normalization_codes"
    ] == ["LEGACY_CAP_INCLUSIVE_POSITION_NORMALIZED"]


def test_nnaa_alias_normalizes_to_one_stable_identity():
    alias = map_to_exact_v1("{nnr:Mono7}A")
    canonical = map_to_exact_v1("{nnr:Bal(3-Me)}A")

    _assert_exact(alias)
    _assert_exact(canonical)
    assert exact_v1_equivalent(alias, canonical)


def test_multichain_canonicalization_is_chain_order_independent():
    first = map_to_exact_v1(
        "AC{br}AAC{cyc:2:R3-5:R3}"
    )
    swapped = map_to_exact_v1(
        "AAC{br}AC{cyc:3:R3-5:R3}"
    )

    _assert_exact(first)
    _assert_exact(swapped)
    assert exact_v1_equivalent(first, swapped)
    assert len(first["chain_breaks"]) == 1


def test_multichain_caps_remain_exact_in_helm_and_biln():
    helm = (
        "PEPTIDE1{[ac].A.C.[nme]}|"
        "PEPTIDE2{G.G.[nh2]}$$$$V2.0"
    )
    document = helm_to_exact_v1(helm)

    _assert_exact(document)
    assert len(document["caps"]) == 3
    assert exact_v1_equivalent(
        document, helm_to_exact_v1(exact_v1_to_helm(document))
    )
    assert exact_v1_equivalent(
        document, biln_to_exact_v1(exact_v1_to_biln(document))
    )
    with pytest.raises(ExactV1Error, match="MULTICHAIN_CAPS"):
        exact_v1_to_map(document)


def test_original_and_canonical_positions_are_bidirectionally_traced():
    document = map_to_exact_v1("ACDEFG{cyc:N-C}")

    _assert_exact(document)
    assert set(document["canonical_order"]) == set(
        range(1, len(document["monomers"]) + 1)
    )
    for source, canonical in document[
        "original_to_canonical"
    ].items():
        assert document["canonical_to_original"][str(canonical)] == (
            source
        )


def test_reused_sidechain_port_abstains():
    document = map_to_exact_v1(
        "CCCC"
        "{cyc:1:R3-2:R3}"
        "{cyc:1:R3-4:R3}"
    )

    jsonschema.validate(document, SCHEMA)
    assert document["exactness_status"] == ABSTAIN
    assert document["graph_sha256"] is None


@pytest.mark.parametrize(
    "source",
    [
        "CAAC{cyc:1:R3-4:R3}",
        "KAD{cyc:1:R3-3:R3}",
        "ACDEFG{cyc:N-C}",
        "AC{br}CE{cyc:2:R3-3:R3}",
        "{nt:ACE}ACD{ct:NME}",
    ],
)
def test_map_helm_biln_roundtrip_canonical_bytes(source):
    map_document = map_to_exact_v1(source)
    helm_document = helm_to_exact_v1(
        exact_v1_to_helm(map_document)
    )
    biln_document = biln_to_exact_v1(
        exact_v1_to_biln(map_document)
    )

    _assert_exact(map_document)
    assert canonical_exact_v1_bytes(map_document) == (
        canonical_exact_v1_bytes(helm_document)
    )
    assert canonical_exact_v1_bytes(map_document) == (
        canonical_exact_v1_bytes(biln_document)
    )


def test_exact_to_smiles_preserves_full_chemical_graph():
    document = map_to_exact_v1(
        "{nnr:dA}CDE{cyc:N-C}"
    )
    smiles = exact_v1_to_smiles(document)
    molecule = Chem.MolFromSmiles(smiles)

    _assert_exact(document)
    assert molecule is not None
    assert Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    ) == smiles
    assert chemical_graph_equivalent(
        document, map_to_exact_v1(exact_v1_to_map(document))
    )


def test_edge_projection_rejects_ring_and_position_overflow():
    four_rings = map_to_exact_v1(
        "CCCCCCCC"
        "{cyc:1:R3-2:R3}"
        "{cyc:3:R3-4:R3}"
        "{cyc:5:R3-6:R3}"
        "{cyc:7:R3-8:R3}"
    )
    long_linear = map_to_exact_v1("A" * 33)

    ring_projection = exact_v1_to_edge_v1(four_rings)
    position_projection = exact_v1_to_edge_v1(long_linear)
    assert ring_projection["status"] == "UNPROJECTABLE"
    assert ring_projection["reason_codes"] == [
        "EDGE_V1_RING_CAPACITY_EXCEEDED"
    ]
    assert position_projection["status"] == "UNPROJECTABLE"
    assert position_projection["reason_codes"] == [
        "EDGE_V1_POSITION_CAPACITY_EXCEEDED"
    ]
    long_edge = (
        "<LINEAR><NO_SRC><NO_DST>"
        "<NO_BOND><NO_SRC><NO_DST>"
        "<NO_BOND><NO_SRC><NO_DST>"
        + "<ALA>" * 33
    )
    parsed = edge_v1_to_exact_v1(long_edge)
    assert parsed["exactness_status"] == ABSTAIN
    assert "EDGE_V1_POSITION_CAPACITY_EXCEEDED" in (
        parsed["reason_codes"][0]
    )


def test_edge_v1_and_legacy_migrate_through_exact_v1():
    edge = (
        "<BOND_SS><SRC_POS_1><DST_POS_4>"
        "<NO_BOND><NO_SRC><NO_DST>"
        "<NO_BOND><NO_SRC><NO_DST>"
        "<CYS><ALA><ALA><CYS>"
    )
    document = edge_v1_to_exact_v1(edge)

    _assert_exact(document)
    assert exact_v1_to_edge_v1(document)["value"] == edge
    legacy = exact_v1_to_legacy_v5(document)
    assert legacy["status"] == "PROJECTED"
    assert exact_v1_equivalent(
        document, legacy_v5_to_exact_v1(legacy["value"])
    )


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (
            "edge",
            "<NO_BOND><NO_SRC><NO_DST>"
            "<LINEAR><NO_SRC><NO_DST>"
            "<NO_BOND><NO_SRC><NO_DST><ALA>",
        ),
        (
            "legacy",
            "<cyc1_SS><cyc1_HT>CAAC<cyc1_1_4>",
        ),
        (
            "legacy",
            "<linear><cyc1_HT>CAAC<cyc1_1_4>",
        ),
        (
            "legacy",
            "CAAC<cyc1_1_4>",
        ),
    ],
)
def test_model_representation_contradictions_abstain(kind, payload):
    document = (
        edge_v1_to_exact_v1(payload)
        if kind == "edge"
        else legacy_v5_to_exact_v1(payload)
    )

    assert document["exactness_status"] == ABSTAIN


@pytest.mark.parametrize(
    "legacy",
    [
        "<linear>ACD",
        "<cyc1_HT>ACD<cyc1_1_3>",
        "<cyc1_SS>CACC<cyc1_1_4>",
        "<cyc1_SC>KAD<cyc1_1_3>",
        "<cyc1_EST>TAD<cyc1_1_3>",
        "<cyc1_HSC>AAK<cyc1_1_3>",
        "<cyc1_THIO>CAD<cyc1_1_3>",
        "<cyc1_ALK>DAD<cyc1_1_3>",
        (
            "<cyc1_SS><cyc2_HT>CACC"
            "<cyc1_1_4><cyc2_1_4>"
        ),
    ],
)
def test_edge_projection_matches_frozen_model_contract(legacy):
    model = pytest.importorskip(
        "chimera_encoder_decoder.cola_representation"
    )
    expected = model.legacy_v5_to_edge_v1(legacy)
    document = legacy_v5_to_exact_v1(legacy)
    projection = exact_v1_to_edge_v1(
        document, preserve_source_order=True
    )

    _assert_exact(document)
    assert projection["status"] == "PROJECTED"
    assert projection["value"] == expected


def test_model_projection_equivalence_does_not_prove_exact_equivalence():
    sidechain = map_to_exact_v1("KAD{cyc:1:R3-3:R3}")
    sidechain_to_tail = map_to_exact_v1(
        "KAD{cyc:1:R3-3:R2}"
    )

    _assert_exact(sidechain)
    _assert_exact(sidechain_to_tail)
    assert not exact_v1_equivalent(
        sidechain, sidechain_to_tail
    )
    assert not chemical_graph_equivalent(
        sidechain, sidechain_to_tail
    )
    assert model_projection_equivalent(
        sidechain, sidechain_to_tail
    )


def test_notation_serializers_reject_unencoded_modifications():
    source = map_to_exact_v1("ACD")
    graph = CyclicPeptideGraph.from_exact_document(source)
    nodes = list(graph.monomers)
    nodes[0] = replace(nodes[0], modifications=("CUSTOM",))
    modified = replace(graph, monomers=tuple(nodes)).canonicalize()

    _assert_exact(modified)
    with pytest.raises(ExactV1Error, match="modifications"):
        exact_v1_to_map(modified)


def test_exact_json_hash_tampering_is_rejected():
    document = map_to_exact_v1("ACDEFG{cyc:N-C}")
    tampered = json.loads(exact_v1_to_json(document))
    tampered["monomers"][0]["monomer_id"] += "-tampered"

    with pytest.raises(CyclicPeptideGraphError):
        CyclicPeptideGraph.from_exact_document(tampered)


def test_registry_binding_rejects_rehashed_endpoint_atom_tampering():
    document = map_to_exact_v1("CAAC{cyc:1:R3-4:R3}")
    tampered = json.loads(exact_v1_to_json(document))
    tampered["bonds"][-1]["src"]["atom"] = "C"
    tampered["graph_sha256"] = hashlib.sha256(
        canonical_exact_v1_bytes(tampered)
    ).hexdigest()

    with pytest.raises(ExactV1Error, match="registry"):
        validate_exact_v1(tampered)


def test_hash_invalid_documents_fail_closed_in_equivalence_and_projection():
    document = map_to_exact_v1("ACDEFG{cyc:N-C}")
    tampered = json.loads(exact_v1_to_json(document))
    tampered["graph_sha256"] = "0" * 64

    assert not exact_v1_equivalent(document, tampered)
    projection = exact_v1_to_edge_v1(tampered)
    assert projection["status"] == "UNPROJECTABLE"


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_runtime_validator_enforces_schema_without_jsonschema(mutation):
    document = map_to_exact_v1("ACDEFG{cyc:N-C}")
    if mutation == "missing":
        document.pop("normalization_codes")
    else:
        document["unexpected"] = True

    with pytest.raises(ExactV1Error, match="fields"):
        validate_exact_v1(document)


def test_unknown_monomer_abstains_without_guessing():
    document = map_to_exact_v1("{nnr:NOT_A_MONOMER}A")

    assert document["exactness_status"] == ABSTAIN
    assert document["reason_codes"]


def test_monomer_resolution_is_case_sensitive():
    exact = map_to_exact_v1("{nnr:dA}A")
    wrong_case = map_to_exact_v1("{nnr:Da}A")

    _assert_exact(exact)
    assert wrong_case["exactness_status"] == ABSTAIN


def test_runtime_user_monomer_has_content_addressed_exact_identity():
    from cycpep_master.paths._map_utils import (
        isolated_monomer_registry,
    )

    with isolated_monomer_registry():
        added = application.add_monomer(
            "ExactRuntimeOnly999",
            "C[C@H](N)C(=O)O",
            persist=False,
        )
        assert added["status"] == "success"
        document = map_to_exact_v1(
            "{nnr:ExactRuntimeOnly999}G"
        )
        _assert_exact(document)
        assert document["monomers"][0]["monomer_id"].startswith(
            "library:user_registered:ExactRuntimeOnly999:"
        )


def test_v6_adapter_requires_l3_qualified_unrepaired_identity():
    expected = map_to_exact_v1("ACG{cyc:N-C}")
    smiles = exact_v1_to_smiles(expected)
    inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    route = {
        "status": "success",
        "output_inchikey": inchikey,
        "evidence_dimensions_input": {
            "residue_evidence": [
                {"residue_position": 1, "unified_symbol": "A"},
                {"residue_position": 2, "unified_symbol": "C"},
                {"residue_position": 3, "unified_symbol": "G"},
            ]
        },
    }
    strict = SimpleNamespace(
        status="success",
        support_status="qualified",
        qualified_success=True,
        repair_codes=[],
        warning_codes=[],
        output_inchikey=inchikey,
        output_evidence={
            "evidence_dimensions": {
                name: {"passed": True}
                for name in (
                    "library_chemistry",
                    "atom_mapping",
                    "chemical_graph_audit",
                    "stereochemistry",
                    "mapping_evidence_binding",
                )
            }
        },
        input_evidence={
            "cyclization_bonds": [{
                "bond_type": "peptide",
                "position_1": 1,
                "position_2": 3,
                "rgroup_1": "R1",
                "rgroup_2": "R2",
            }]
        },
        route_results=[route],
    )

    emitted = exact_v1_from_v6_result(strict)
    assert exact_v1_equivalent(expected, emitted)

    strict.repair_codes = ["CONNECTIVITY_INFERRED_FROM_COORDINATES"]
    abstained = exact_v1_from_v6_result(strict)
    assert abstained["exactness_status"] == ABSTAIN
    assert abstained["reason_codes"] == [
        "V6_RESULT_NOT_QUALIFIED_EXACT"
    ]

    strict.repair_codes = []
    strict.input_evidence["cyclization_bonds"] = [{
        "position_1": "malformed"
    }]
    malformed = exact_v1_from_v6_result(strict)
    assert malformed["exactness_status"] == ABSTAIN
    assert malformed["reason_codes"] == [
        "V6_CYCLIZATION_EVIDENCE_MALFORMED"
    ]


def test_pdb_and_mmcif_v6_emit_identical_exact_v1(tmp_path):
    gemmi = pytest.importorskip("gemmi")
    from cycpep_master.tests.test_remediation_v6 import (
        _write_explicit_trialanine,
    )

    pdb = _write_explicit_trialanine(tmp_path)
    structure = gemmi.read_structure(str(pdb))
    mmcif = tmp_path / "trialanine.cif"
    structure.make_mmcif_document().write_file(str(mmcif))

    pdb_result = application.reconstruct_exact_v1(
        pdb,
        chain_id="A",
        require_empty_persistent_overlay=True,
    )
    mmcif_result = application.reconstruct_exact_v1(
        mmcif,
        chain_id="A",
        require_empty_persistent_overlay=True,
    )

    assert pdb_result["status"] == "success"
    assert mmcif_result["status"] == "success"
    assert exact_v1_equivalent(
        pdb_result["data"]["exact_v1"],
        mmcif_result["data"]["exact_v1"],
    )
    assert (
        pdb_result["data"]["graph_sha256"]
        == mmcif_result["data"]["graph_sha256"]
    )
    assert exact_v1_equivalent(
        pdb_result["data"]["exact_v1"],
        map_to_exact_v1("AAA{cyc:N-C}"),
    )


def test_reconstruct_exact_preserves_v6_failure_status(tmp_path):
    result = application.reconstruct_exact_v1(
        tmp_path / "missing.pdb"
    )

    assert result["status"] == "success"
    assert result["data"]["v6"]["status"] == "rejected"
    assert result["data"]["exactness_status"] == ABSTAIN
    assert result["data"]["artifact_status"] == "PARTIAL"


def test_application_conversion_exposes_exact_and_bounded_projection():
    exact_result = application.convert_representation(
        "map", "exact_v1", "CAAC{cyc:1:R3-4:R3}"
    )
    assert exact_result["status"] == "success"
    document = exact_result["data"]["value"]
    _assert_exact(document)

    edge_result = application.convert_representation(
        "exact_v1",
        "edge_v1",
        document,
    )
    assert edge_result["status"] == "success"
    assert edge_result["data"]["value"].startswith("<BOND_SS>")


def test_application_preserves_exact_v1_abstention():
    abstained = map_to_exact_v1("{nnr:NOT_A_MONOMER}A")

    result = application.convert_representation(
        "exact_v1", "map", json.dumps(abstained)
    )

    assert result["status"] == "success"
    assert result["data"]["exactness_status"] == ABSTAIN
    assert result["data"]["exact_v1"] == abstained
    assert result["data"]["artifact_status"] == "PARTIAL"


def test_application_refuses_general_smiles_to_exact_v1():
    result = application.convert_representation(
        "smiles", "exact_v1", "CCO"
    )

    assert result["status"] == "success"
    assert "SMILES-to-monomer" in result["error"]
    assert result["data"]["exactness_status"] == ABSTAIN
    assert result["data"]["reason_codes"] == [
        "GENERAL_SMILES_DECOMPOSITION_UNSUPPORTED"
    ]
    assert result["data"]["artifact_status"] == "PARTIAL"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edge_max_rings": 0},
        {"edge_max_position": -1},
        {"edge_max_rings": True},
    ],
)
def test_application_rejects_invalid_edge_capacity(kwargs):
    result = application.convert_representation(
        "map", "edge_v1", "ACD", **kwargs
    )

    assert result["status"] == "invalid_input"
