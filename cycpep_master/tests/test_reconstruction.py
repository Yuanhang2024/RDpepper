"""Tests for the unified structure-reconstruction orchestration layer.

Directed parametrized cases: 6 linear, 6 branched, 4 multi-chain coordinates,
8 RDKit-fallback dispatch cases, 6 corrupt inputs (30 total).  Interface, CLI,
and strict-invariance assertions are non-parametrized and use small synthetic
inputs plus monkeypatching; no full benchmark is executed.
"""

from __future__ import annotations

import math
import warnings

import pytest

import cycpep_master
from cycpep_master import application
from cycpep_master.reconstruction import (
    MODES,
    STATUS_FAILED,
    STATUS_SUCCESS,
    UnifiedReconstructionResult,
    detect_source_kind,
    reconstruct_structure,
)


REQUIRED_FIELDS = (
    "status",
    "quality",
    "result_origin",
    "source_kind",
    "mode",
    "result",
    "smiles",
    "inchi",
    "inchikey",
    "graph",
    "ambiguous",
    "warnings",
    "warning_codes",
    "alternatives",
    "structure_profile",
    "provenance",
    "strict_result",
)

REQUIRED_PROFILE_FIELDS = (
    "chain_count",
    "backbone_layout",
    "macrocycle_count",
    "macrocycle_count_metric",
    "crosslink_types",
    "branch_point_count",
    "is_multichain",
    "display_label",
)


# --------------------------------------------------------------------------
# Helpers: small synthetic inputs
# --------------------------------------------------------------------------


def _atom_line(serial, name, x, y, z, chain, resn, resi, element):
    return (
        f"ATOM  {serial:5d} {name:>4s} {resn} {chain}{resi:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def _ring_pdb(n=8, chain="L", bond=1.5):
    lines = ["HEADER    RING"]
    radius = bond / (2 * math.sin(math.pi / n))
    for index in range(n):
        angle = 2 * math.pi * index / n
        lines.append(
            _atom_line(
                index + 1,
                f"C{index + 1}",
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.0,
                chain,
                "LIG",
                index + 1,
                "C",
            )
        )
    for index in range(n):
        partner = (index + 1) % n
        lines.append(f"CONECT{index + 1:5d}{partner + 1:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _two_chain_pdb(*, with_ssbond):
    rows = ["HEADER    MULTI"]
    serial = 1
    for chain in ("A", "B"):
        for residue in range(1, 4):
            for name, element, x in (
                ("N", "N", 0.0),
                ("CA", "C", 1.4),
                ("C", "C", 2.8),
                ("SG", "S", 1.0),
            ):
                rows.append(
                    _atom_line(
                        serial,
                        name,
                        x,
                        residue * 2.0,
                        chain == "B" and 2.0 or 0.0,
                        chain,
                        "CYS",
                        residue,
                        element,
                    )
                )
                serial += 1
    if with_ssbond:
        rows.append("SSBOND   1 CYS A    1    CYS B    1")
    rows.append("END")
    return "\n".join(rows) + "\n"


def _write_two_chain(tmp_path, name, *, with_ssbond):
    path = tmp_path / name
    path.write_text(_two_chain_pdb(with_ssbond=with_ssbond), encoding="ascii")
    return path


def _mmcif_with_chains(tmp_path, filename, chain_ids):
    import gemmi

    structure = gemmi.Structure()
    structure.name = "tiny"
    model = gemmi.Model("1")
    for chain_index, chain_id in enumerate(chain_ids):
        chain = gemmi.Chain(chain_id)
        for residue_index in range(2):
            residue = gemmi.Residue()
            residue.name = "ALA"
            residue.seqid = gemmi.SeqId(str(residue_index + 1))
            for name, element, x in (
                ("N", "N", 0.0),
                ("CA", "C", 1.4),
                ("C", "C", 2.8),
                ("O", "O", 3.8),
            ):
                atom = gemmi.Atom()
                atom.name = name
                atom.element = gemmi.Element(element)
                atom.pos = gemmi.Position(chain_index * 10.0 + x, residue_index * 2.0, 0.0)
                atom.occ = 1.0
                atom.b_iso = 20.0
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    path = tmp_path / filename
    structure.make_mmcif_document().write_file(str(path))
    return path


def _fake_result_first(
    *,
    status=STATUS_SUCCESS,
    quality="raw",
    smiles=None,
    graph=None,
    ambiguous=False,
    warning_codes=None,
    provenance=None,
    strict_result=None,
):
    from cycpep_master.result_first import ReconstructionResult

    graph = graph or {"atoms": [], "bonds": []}
    return ReconstructionResult(
        status=status,
        quality=quality,
        source="test_fixture",
        result=smiles if smiles is not None else graph,
        smiles=smiles,
        graph=graph,
        ambiguous=ambiguous,
        warnings=[],
        warning_codes=list(warning_codes or []),
        alternatives=[],
        provenance=dict(provenance or {}),
        strict_status=None,
        strict_result=strict_result,
    )


def _fake_strict(**kwargs):
    from cycpep_master.remediation_v5 import StrictReconstructionResult

    return StrictReconstructionResult(**kwargs)


# --------------------------------------------------------------------------
# Directed cases: 6 linear
# --------------------------------------------------------------------------


LINEAR_CASES = [
    ("AAAAA", "sequence", "linear"),
    ("sequence:ACDEF", "sequence", "linear"),
    ("PEPTIDE1{A.A.A.A}$$$$V2.0", "helm", "linear"),
    ("map:AAAA", "map", "linear"),
    ("P-E-P-T-I-D-E", "biln", "linear"),
    ("AAAAA{nt:ACE}", "map", "linear"),
]


@pytest.mark.parametrize(
    "source,expected_kind,expected_layout", LINEAR_CASES
)
def test_directed_linear_representation_cases(source, expected_kind, expected_layout):
    result = reconstruct_structure(source)

    assert result.status == STATUS_SUCCESS
    assert result.quality == "high"
    assert result.result_origin == "representation_assembly"
    assert result.source_kind == expected_kind
    assert result.smiles is not None
    assert result.inchi is not None
    assert result.inchikey is not None
    assert result.structure_profile["backbone_layout"] == expected_layout


# --------------------------------------------------------------------------
# Directed cases: 6 branched
# --------------------------------------------------------------------------


BRANCHED_CASES = [
    "AC{br}CG{cyc:2:R3-3:R3}",
    "AC{br}CA{cyc:2:R3-3:R3}",
    "C{br}AC{cyc:1:R3-3:R3}",
    "CC{br}CC{cyc:2:R3-3:R3}{cyc:1:R3-4:R3}",
    "CCAA{br}CC{cyc:1:R3-5:R3}{cyc:2:R3-6:R3}",
    "PEPTIDE1{C.A}|PEPTIDE2{C.A}$PEPTIDE1,PEPTIDE2,1:R3-1:R3$$$V2.0",
]


@pytest.mark.parametrize("source", BRANCHED_CASES)
def test_directed_branched_representation_cases(source):
    result = reconstruct_structure(source)

    assert result.status == STATUS_SUCCESS, result.provenance.get("failure_reason")
    assert result.quality == "high"
    assert result.structure_profile["chain_count"] >= 2
    assert result.structure_profile["is_multichain"] is True
    assert result.structure_profile["crosslink_types"] == ["R3-R3"]
    assert result.graph is not None
    assert result.graph["residue_count"] >= 2


# --------------------------------------------------------------------------
# Directed cases: 4 multi-chain coordinates
# --------------------------------------------------------------------------


def test_directed_multichain_explicit_chain_list(tmp_path):
    path = _write_two_chain(tmp_path, "insulin_like.pdb", with_ssbond=True)

    result = reconstruct_structure(path, chain_id=["A", "B"], mode="auto")

    assert result.status == STATUS_SUCCESS
    assert result.quality == "high"
    assert result.result_origin == "multichain_assembly"
    assert result.ambiguous is False
    assert "UNIFIED_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED" in result.warning_codes
    assert result.structure_profile["chain_count"] == 2
    assert result.structure_profile["is_multichain"] is True
    assert result.structure_profile["crosslink_types"] == ["R3-R3"]


def test_directed_multichain_auto_single_connected_component(tmp_path):
    path = _write_two_chain(tmp_path, "linked.pdb", with_ssbond=True)

    result = reconstruct_structure(path, chain_id=None, mode="auto")

    assert result.status == STATUS_SUCCESS
    assert result.result_origin == "multichain_assembly"
    assert result.structure_profile["chain_count"] == 2
    assert result.ambiguous is False


def test_directed_multichain_disconnected_auto_requires_explicit_chains(tmp_path):
    path = _write_two_chain(tmp_path, "disconnected.pdb", with_ssbond=False)

    result = reconstruct_structure(path, chain_id=None, mode="auto")

    assert result.status == STATUS_SUCCESS
    assert result.ambiguous is True
    assert "UNIFIED_CHAIN_AMBIGUOUS" in result.warning_codes
    assert result.alternatives
    assert all(
        {"status", "quality", "result_origin", "smiles", "graph_summary"}
        <= set(item)
        for item in result.alternatives
    )
    assert result.provenance["chain_resolution"]["ssbond_components"]


def test_directed_multichain_disconnected_best_effort_primary_candidate(tmp_path):
    path = _write_two_chain(tmp_path, "disconnected2.pdb", with_ssbond=False)

    result = reconstruct_structure(path, chain_id=None, mode="best_effort")

    assert result.status == STATUS_SUCCESS
    assert result.ambiguous is True
    assert "UNIFIED_CHAIN_AMBIGUOUS" in result.warning_codes
    assert result.alternatives, "best_effort must expose unselected chains"
    assert result.structure_profile["chain_count"] == 1


# --------------------------------------------------------------------------
# Directed cases: 8 RDKit-fallback dispatch (monkeypatched)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("low_grade", ["topology", "partial", "raw"])
def test_directed_registry_assembly_requires_coordinate_graph(
    tmp_path, monkeypatch, low_grade
):
    path = tmp_path / "input.pdb"
    path.write_text("END\n", encoding="ascii")
    strict = _fake_strict(
        status="rejected",
        warning_codes=["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"],
    )
    fake_rf = _fake_result_first(
        quality=low_grade,
        graph={"atoms": [], "bonds": []},
        warning_codes=[
            "V6_INSUFFICIENT_EVIDENCE_DIMENSIONS",
            "BOND_ORDERS_INFERRED",
        ],
        provenance={"ladder": "directed_low_grade"},
        strict_result=strict,
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first",
        lambda _path, _chain, **_kwargs: fake_rf,
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._try_registry_assembly",
        lambda _path, _chain: ("CC", None),
    )

    result = reconstruct_structure(path, chain_id="L", mode="auto")

    assert result.status == STATUS_SUCCESS
    assert result.quality == low_grade
    assert result.ambiguous is False
    assert result.alternatives == []
    assert result.result_origin == "result_first"
    assert result.smiles is None
    assert "UNIFIED_REGISTRY_ASSEMBLY_NOT_QUALIFIED" in result.warning_codes
    assert "V6_INSUFFICIENT_EVIDENCE_DIMENSIONS" in result.warning_codes
    assert result.provenance["registry_assembly"]["reason"] == (
        "LOW_GRADE_GRAPH_UNAVAILABLE"
    )
    assert result.provenance["underlying"] == fake_rf.provenance


def test_directed_registry_assembly_requires_matching_graph(
    tmp_path, monkeypatch
):
    path = tmp_path / "matching.pdb"
    path.write_text("END\n", encoding="ascii")
    strict = _fake_strict(
        status="rejected",
        warning_codes=["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"],
    )
    fake_rf = _fake_result_first(
        quality="partial",
        graph={
            "atoms": [
                {"serial": 1, "element": "C"},
                {"serial": 2, "element": "C"},
            ],
            "bonds": [{"a": 1, "b": 2, "order": None}],
        },
        warning_codes=["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"],
        strict_result=strict,
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first",
        lambda _path, _chain, **_kwargs: fake_rf,
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._try_registry_assembly",
        lambda _path, _chain: ("CC", None),
    )

    result = reconstruct_structure(path, chain_id="L", mode="auto")

    assert result.status == STATUS_SUCCESS
    assert result.quality == "medium"
    assert result.ambiguous is False
    assert result.result_origin == "coordinate_assembly"
    assert result.smiles == "CC"
    assert result.alternatives
    assert result.provenance["underlying"]["result_first"]["quality"] == (
        "partial"
    )
    assert result.provenance["underlying"]["result_first"]["provenance"] == {}
    assert result.provenance["underlying"]["registry_assembly"][
        "qualified"
    ] is True


def test_registry_topology_identity_rejects_non_isomorphic_same_summary_graph():
    from types import SimpleNamespace
    from cycpep_master.reconstruction import (
        _graph_topology_signature,
        _qualify_registry_assembly,
        _smiles_topology_signature,
    )

    atoms = [{"serial": serial, "element": "C"} for serial in range(1, 11)]
    cycle_edges = [
        {"a": serial, "b": serial % 8 + 1, "order": None}
        for serial in range(1, 9)
    ]
    coordinate_graph = {
        "atoms": atoms,
        "bonds": cycle_edges + [
            {"a": 1, "b": 9, "order": None},
            {"a": 1, "b": 10, "order": None},
        ],
    }
    registry_signature = _smiles_topology_signature("C1CCCCCCC1CC")
    coordinate_signature = _graph_topology_signature(coordinate_graph)

    assert registry_signature["heavy_atom_count"] == coordinate_signature[
        "heavy_atom_count"
    ]
    assert registry_signature["heavy_edge_count"] == coordinate_signature[
        "heavy_edge_count"
    ]
    assert registry_signature["heavy_element_counts"] == coordinate_signature[
        "heavy_element_counts"
    ]
    assert registry_signature["canonical_topology"] != coordinate_signature[
        "canonical_topology"
    ]
    audit = _qualify_registry_assembly(
        "C1CCCCCCC1CC", SimpleNamespace(graph=coordinate_graph)
    )
    assert audit["qualified"] is False
    assert audit["mismatches"] == ["canonical_topology"]


def test_directed_low_grade_with_fallback_block_not_replaced(tmp_path, monkeypatch):
    path = tmp_path / "blocked.pdb"
    path.write_text("END\n", encoding="ascii")
    fake_rf = _fake_result_first(
        quality="topology",
        graph={"atoms": [], "bonds": []},
        provenance={"fallback_block": {"blocked": True}},
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first",
        lambda _path, _chain, **_kwargs: fake_rf,
    )

    def _forbidden(_path, _chain):
        raise AssertionError("registry assembly must not bypass fallback_block")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._try_registry_assembly", _forbidden
    )

    result = reconstruct_structure(path, chain_id="L", mode="auto")

    assert result.status == STATUS_SUCCESS
    assert result.quality == "topology"
    assert result.result_origin == "result_first"
    assert "UNIFIED_COORDINATE_ASSEMBLY_NOT_V6_QUALIFIED" not in result.warning_codes


def test_directed_best_effort_after_auto_failure_without_block(tmp_path, monkeypatch):
    path = tmp_path / "failed.pdb"
    path.write_text("END\n", encoding="ascii")
    fake_rf = _fake_result_first(
        status=STATUS_FAILED,
        provenance={"failure_reason": "no readable structure"},
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first",
        lambda _path, _chain, **_kwargs: fake_rf,
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._try_registry_assembly",
        lambda _path, _chain: ("CC", None),
    )

    auto_result = reconstruct_structure(path, chain_id="L", mode="auto")
    best_result = reconstruct_structure(path, chain_id="L", mode="best_effort")

    assert auto_result.status == STATUS_FAILED
    assert best_result.status == STATUS_FAILED
    assert best_result.quality is None
    assert best_result.ambiguous is False
    assert "UNIFIED_REGISTRY_ASSEMBLY_NOT_QUALIFIED" in best_result.warning_codes
    assert best_result.provenance["underlying"]["registry_assembly"][
        "reason"
    ] == "LOW_GRADE_GRAPH_UNAVAILABLE"


def test_directed_best_effort_does_not_bypass_fallback_block(tmp_path, monkeypatch):
    path = tmp_path / "blocked2.pdb"
    path.write_text("END\n", encoding="ascii")
    fake_rf = _fake_result_first(
        status=STATUS_FAILED,
        provenance={"fallback_block": {"blocked": True}},
    )
    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first",
        lambda _path, _chain, **_kwargs: fake_rf,
    )

    def _forbidden(_path, _chain):
        raise AssertionError("best_effort must not bypass fallback_block")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._try_registry_assembly", _forbidden
    )

    result = reconstruct_structure(path, chain_id="L", mode="best_effort")

    assert result.status == STATUS_FAILED


@pytest.mark.parametrize(
    "quality", ["exact", "medium"]
)
def test_directed_exact_or_medium_shortcircuits_assembly(
    tmp_path, monkeypatch, quality
):
    path = tmp_path / "qualified.pdb"
    path.write_text("END\n", encoding="ascii")
    fake_rf = _fake_result_first(quality=quality, smiles="C1CC1")
    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first",
        lambda _path, _chain, **_kwargs: fake_rf,
    )

    def _forbidden(_path, _chain):
        raise AssertionError("assembly must not run when result-first qualifies")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._try_registry_assembly", _forbidden
    )

    result = reconstruct_structure(path, chain_id="L", mode="auto")

    assert result.status == STATUS_SUCCESS
    assert result.quality == quality
    assert result.result_origin == (
        "strict_v6" if quality == "exact" else "result_first"
    )
    assert result.smiles == "C1CC1"


# --------------------------------------------------------------------------
# Directed cases: 6 corrupt inputs
# --------------------------------------------------------------------------


def test_directed_corrupt_map_port(tmp_path):
    result = reconstruct_structure("AA{cyc:1:R4-2:R2}")

    assert result.status == STATUS_FAILED
    assert result.source_kind == "map"
    assert result.smiles is None
    assert "UNIFIED_REPRESENTATION_ASSEMBLY_FAILED" in result.warning_codes


def test_directed_corrupt_helm_version():
    result = reconstruct_structure("PEPTIDE1{A}$$$$V2.0 trailing")

    assert result.status == STATUS_FAILED
    assert result.source_kind == "helm"
    assert "UNIFIED_REPRESENTATION_ASSEMBLY_FAILED" in result.warning_codes


def test_directed_corrupt_biln():
    result = reconstruct_structure("A--C")

    assert result.status == STATUS_FAILED
    assert result.source_kind == "biln"
    assert "UNIFIED_REPRESENTATION_ASSEMBLY_FAILED" in result.warning_codes


def test_directed_corrupt_missing_coordinate_path():
    result = reconstruct_structure("D:/definitely_missing_cycpep.pdb")

    assert result.status == STATUS_FAILED
    assert result.source_kind == "coordinate"
    assert "UNIFIED_COORDINATE_INPUT_ERROR" in result.warning_codes
    assert "does not exist" in result.provenance["failure_reason"]


def test_directed_corrupt_unsupported_coordinate_file(tmp_path):
    path = tmp_path / "input.sdf"
    path.write_text("unsupported\n", encoding="ascii")

    result = reconstruct_structure(path)

    assert result.status == STATUS_FAILED
    assert result.source_kind == "coordinate"
    assert "UNIFIED_COORDINATE_INPUT_ERROR" in result.warning_codes
    assert "UNSUPPORTED_COORDINATE_FORMAT" in result.provenance["failure_reason"]


def test_directed_corrupt_undetermined_text_stays_failed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto_result = reconstruct_structure("not a valid anything !!!", mode="auto")
        best_result = reconstruct_structure(
            "not a valid anything !!!", mode="best_effort"
        )

    assert auto_result.status == STATUS_FAILED
    assert "UNIFIED_SOURCE_UNDETERMINED" in auto_result.warning_codes
    # best_effort must not fabricate a structure from garbage text.
    assert best_result.status == STATUS_FAILED
    assert best_result.smiles is None


def test_auto_chain_selection_rejects_ligand_only_ring(tmp_path):
    path = tmp_path / "ligand_only.pdb"
    path.write_text(_ring_pdb(), encoding="ascii")

    auto_result = reconstruct_structure(path, mode="auto")
    best_result = reconstruct_structure(path, mode="best_effort")

    for result in (auto_result, best_result):
        assert result.status == STATUS_FAILED
        assert "UNIFIED_NO_PEPTIDE_CHAIN" in result.warning_codes
        assert result.smiles is None


def test_auto_chain_selection_rejects_single_ligand_with_backbone_atom_names(
    tmp_path
):
    path = tmp_path / "ligand_named_like_backbone.pdb"
    rows = ["HEADER    LIGAND"]
    for serial, (name, element, x) in enumerate(
        (("N", "N", 0.0), ("CA", "C", 1.4), ("C", "C", 2.8)),
        start=1,
    ):
        rows.append(_atom_line(serial, name, x, 0.0, 0.0, "L", "LIG", 1, element))
    path.write_text("\n".join([*rows, "END", ""]), encoding="ascii")

    result = reconstruct_structure(path, mode="auto")

    assert result.status == STATUS_FAILED
    assert "UNIFIED_NO_PEPTIDE_CHAIN" in result.warning_codes


def test_best_effort_never_reinterprets_damaged_coordinate_as_text(tmp_path):
    path = tmp_path / "damaged.cif"
    path.write_text("this is not a coordinate file at all", encoding="ascii")

    auto_result = reconstruct_structure(path, mode="auto")
    best_result = reconstruct_structure(path, mode="best_effort")

    # A damaged coordinate input must stay a coordinate failure in every mode;
    # best_effort must not re-classify it as notation text.
    assert auto_result.status == STATUS_FAILED
    assert auto_result.source_kind == "coordinate"
    assert best_result.status == STATUS_FAILED
    assert best_result.source_kind == "coordinate"
    assert best_result.smiles is None
    assert best_result.result is None


# --------------------------------------------------------------------------
# Interface contract assertions
# --------------------------------------------------------------------------


def test_unified_dataclass_declares_required_fields():
    result = reconstruct_structure("AAAAA")

    assert isinstance(result, UnifiedReconstructionResult)
    for field in REQUIRED_FIELDS:
        assert hasattr(result, field), f"missing field: {field}"


def test_structure_profile_declares_required_fields():
    result = reconstruct_structure("AAAAA{cyc:N-C}")

    assert result.structure_profile is not None
    for field in REQUIRED_PROFILE_FIELDS:
        assert field in result.structure_profile, f"missing profile field: {field}"
    assert result.structure_profile["backbone_layout"] in {"linear", "cyclic"}


def test_status_is_success_only_when_structure_exists(tmp_path):
    path = tmp_path / "ok.pdb"
    path.write_text("END\n", encoding="ascii")
    success = reconstruct_structure("AAAAA")
    failed = reconstruct_structure("AA{cyc:1:R4-2:R2}")

    assert success.status == STATUS_SUCCESS
    assert success.smiles is not None
    assert failed.status == STATUS_FAILED
    assert failed.smiles is None
    assert failed.result is None


def test_plain_linear_sequence_is_never_claimed_as_map():
    assert detect_source_kind("AAAAA") == "sequence"
    assert detect_source_kind("ACDEF") == "sequence"
    assert detect_source_kind("AAAAA{cyc:N-C}") == "map"

    result = reconstruct_structure("AAAAA")
    assert result.source_kind == "sequence"
    assert result.structure_profile["macrocycle_count"] == 0
    assert result.structure_profile["backbone_layout"] == "linear"


def test_macrocycle_count_metric_is_cycle_rank_for_residue_graphs():
    result = reconstruct_structure("AAAAA{cyc:N-C}")

    profile = result.structure_profile
    assert profile["macrocycle_count"] == 1
    # Graph-derived macrocycle_count is cycle rank, not an independent
    # macrocycle tally; the metric must be explicit.
    assert profile["macrocycle_count_metric"] == "cycle_rank"


def test_reversed_head_to_tail_edge_is_still_cyclic():
    result = reconstruct_structure("AAAAA{cyc:5:R2-1:R1}")

    assert result.status == STATUS_SUCCESS
    assert result.structure_profile["backbone_layout"] == "cyclic"
    assert result.structure_profile["macrocycle_count"] == 1


def test_atom_graph_small_ring_is_not_counted_as_macrocycle():
    from cycpep_master.reconstruction import _profile_from_graph

    graph = {
        "atoms": [
            {"serial": serial, "element": "C"}
            for serial in range(1, 7)
        ],
        "bonds": [
            {"a": serial, "b": serial % 6 + 1, "order": None}
            for serial in range(1, 7)
        ],
    }
    profile = _profile_from_graph(graph)

    assert profile["macrocycle_count"] == 0
    assert profile["macrocycle_count_metric"] == "large_ring_sssr_count"


def test_profile_uses_final_smiles_over_strict_evidence(tmp_path, monkeypatch):
    path = tmp_path / "strict_cyclic.pdb"
    path.write_text("END\n", encoding="ascii")
    strict = _fake_strict(
        status=STATUS_SUCCESS,
        output_smiles="C1CC1",
        path_used="V6_TEST",
        qualified_success=True,
        input_evidence={
            "topology": "monocyclic",
            "cyclization_bonds": [
                {
                    "bond_type": "non_peptide",
                    "position_1": 1,
                    "position_2": 5,
                }
            ],
            "residue_count": 5,
        },
    )
    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_structure_fail_closed_v6",
        lambda _path, chain_id="L", **kwargs: strict,
    )

    result = reconstruct_structure(path, chain_id="L", mode="strict")

    assert result.status == STATUS_SUCCESS
    assert result.quality == "exact"
    profile = result.structure_profile
    assert profile["backbone_layout"] == "linear"
    assert profile["macrocycle_count"] == 0
    assert profile["macrocycle_count_metric"] == "large_ring_count"


def test_low_grade_profile_uses_returned_graph_over_strict_evidence():
    from types import SimpleNamespace
    from cycpep_master.reconstruction import _profile_for_coordinate

    strict = SimpleNamespace(
        input_evidence={
            "topology": "monocyclic",
            "cyclization_bonds": [],
            "residue_count": 3,
        }
    )
    graph = {
        "atoms": [
            {"serial": 1, "element": "C", "chain": "L", "residue_number": 1},
            {"serial": 2, "element": "C", "chain": "L", "residue_number": 2},
            {"serial": 3, "element": "C", "chain": "L", "residue_number": 3},
        ],
        "bonds": [
            {"a": 1, "b": 2, "order": None},
            {"a": 2, "b": 3, "order": None},
        ],
    }

    profile = _profile_for_coordinate(None, graph, strict)

    assert profile["backbone_layout"] == "linear"
    assert profile["macrocycle_count"] == 0
    assert profile["profile_evidence"] == "atom_graph_large_ring_analysis"


def test_inchi_failure_is_never_fabricated(tmp_path, monkeypatch):
    import cycpep_master.reconstruction as reconstruction

    def _boom(_molecule):
        raise RuntimeError("inchi engine unavailable")

    monkeypatch.setattr(reconstruction.rdkit_inchi, "MolToInchiKey", _boom)

    result = reconstruct_structure("AAAAA")

    assert result.status == STATUS_SUCCESS
    assert result.inchi is None
    assert result.inchikey is None
    assert "UNIFIED_INCHI_UNAVAILABLE" in result.warning_codes


def test_inchi_is_generated_only_from_cleanable_smiles():
    from cycpep_master.reconstruction import _compute_inchi

    inchi, key, error = _compute_inchi("not smiles !!!")
    assert inchi is None
    assert key is None
    assert error is not None


def test_strict_mode_coordinate_only_calls_strict_v6(tmp_path, monkeypatch):
    path = tmp_path / "strict.pdb"
    path.write_text("END\n", encoding="ascii")
    strict = _fake_strict(
        status=STATUS_SUCCESS,
        output_smiles="C1CC1",
        path_used="V6_TEST",
        qualified_success=True,
    )
    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_structure_fail_closed_v6",
        lambda _path, chain_id="L", **kwargs: strict,
    )

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("strict mode must not run result-first")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first", _forbidden
    )

    result = reconstruct_structure(path, chain_id="L", mode="strict")

    assert result.status == STATUS_SUCCESS
    assert result.quality == "exact"
    assert result.result_origin == "strict_v6"
    assert result.smiles == "C1CC1"
    assert result.strict_result is strict


def test_coordinate_strict_forwards_reconstruction_policy_kwargs(
    tmp_path, monkeypatch
):
    path = tmp_path / "strict_policy.pdb"
    path.write_text("END\n", encoding="ascii")
    strict = _fake_strict(
        status=STATUS_SUCCESS,
        output_smiles="C1CC1",
        path_used="V6_TEST",
        qualified_success=True,
    )
    received = {}

    def _strict(_path, *, chain_id, **kwargs):
        received["chain_id"] = chain_id
        received.update(kwargs)
        return strict

    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_structure_fail_closed_v6",
        _strict,
    )

    result = reconstruct_structure(
        path,
        chain_id="L",
        mode="strict",
        minimum_macrocycle_ring_size=11,
        require_empty_persistent_overlay=True,
    )

    assert result.status == STATUS_SUCCESS
    assert received == {
        "chain_id": "L",
        "minimum_macrocycle_ring_size": 11,
        "require_empty_persistent_overlay": True,
        "allow_linear_topology": False,
    }

    # The opt-in switch itself must reach the strict V6 entry untouched.
    result = reconstruct_structure(
        path,
        chain_id="L",
        mode="strict",
        allow_linear_topology=True,
    )
    assert result.status == STATUS_SUCCESS
    assert received["allow_linear_topology"] is True


def test_coordinate_auto_forwards_reconstruction_policy_kwargs(
    tmp_path, monkeypatch
):
    path = tmp_path / "auto_policy.pdb"
    path.write_text("END\n", encoding="ascii")
    received = {}

    def _result_first(_path, chain, **kwargs):
        received["chain"] = chain
        received.update(kwargs)
        return _fake_result_first(quality="exact", smiles="C1CC1")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._run_result_first", _result_first
    )

    result = reconstruct_structure(
        path,
        chain_id="L",
        mode="auto",
        minimum_macrocycle_ring_size=12,
        require_empty_persistent_overlay=True,
    )

    assert result.status == STATUS_SUCCESS
    assert received == {
        "chain": "L",
        "minimum_macrocycle_ring_size": 12,
        "require_empty_persistent_overlay": True,
        "radius_multiplier": None,
        "distance_ceiling": None,
    }


def test_strict_mode_strict_failure_is_typed_failed(tmp_path, monkeypatch):
    path = tmp_path / "rejected.pdb"
    path.write_text("END\n", encoding="ascii")
    strict = _fake_strict(
        status="rejected",
        rejection_reason="V6 insufficient evidence",
        warning_codes=["V6_INSUFFICIENT_EVIDENCE_DIMENSIONS"],
    )
    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_structure_fail_closed_v6",
        lambda _path, chain_id="L", **kwargs: strict,
    )

    result = reconstruct_structure(path, chain_id="L", mode="strict")

    assert result.status == STATUS_FAILED
    assert result.quality is None
    assert result.strict_result is strict
    assert "V6_INSUFFICIENT_EVIDENCE_DIMENSIONS" in result.warning_codes


def test_strict_mode_unqualified_success_fails_and_preserves_strict_result(
    tmp_path, monkeypatch
):
    path = tmp_path / "repaired.pdb"
    path.write_text("END\n", encoding="ascii")
    strict = _fake_strict(
        status=STATUS_SUCCESS,
        output_smiles="C1CC1",
        path_used="V5_EXPLICIT_MONOMER_FAMILY_RECOVERY",
        repair_codes=["EXPLICIT_MONOMER_FAMILY_RECOVERY"],
        qualified_success=False,
        warning_codes=["EXPLICIT_MONOMER_FAMILY_RECOVERY"],
    )
    monkeypatch.setattr(
        "cycpep_master.remediation_v6.reconstruct_structure_fail_closed_v6",
        lambda _path, chain_id="L", **kwargs: strict,
    )

    result = reconstruct_structure(path, chain_id="L", mode="strict")

    # Repaired/unqualified strict output must never be labelled exact.
    assert result.status == STATUS_FAILED
    assert result.quality is None
    assert result.smiles is None
    assert result.result is None
    assert result.strict_result is strict
    assert "UNIFIED_STRICT_NOT_QUALIFIED" in result.warning_codes
    assert result.provenance["underlying"]["qualified_success"] is False


def test_strict_mode_representation_uses_deterministic_assembly_only():
    result = reconstruct_structure("AAAAA{cyc:N-C}", mode="strict")

    assert result.status == STATUS_SUCCESS
    assert result.quality == "high"
    assert result.warning_codes == []
    assert result.strict_result is None


def test_mmcif_native_chain_selection_and_graph_fallback(tmp_path):
    single = _mmcif_with_chains(tmp_path, "single.cif", ("A",))
    multi = _mmcif_with_chains(tmp_path, "multi.cif", ("A", "B"))

    explicit = reconstruct_structure(single, chain_id="A", mode="auto")
    no_chain = reconstruct_structure(single, mode="auto")
    multi_chain = reconstruct_structure(multi, chain_id=["A", "B"], mode="auto")

    # Single-chain mmCIF with an explicit chain is a supported coordinate path.
    assert explicit.status in {STATUS_SUCCESS, STATUS_FAILED}
    assert explicit.source_kind == "coordinate"
    assert explicit.provenance["detection"]["coordinate_format"] == "mmcif"
    # Native adapter selection makes an unambiguous single-chain auto request
    # usable while retaining the adapter provenance in the detection record.
    assert no_chain.status == STATUS_SUCCESS
    assert no_chain.provenance["detection"]["native_mmcif"]["selected_chain_ids"]
    # When the legacy multi-chain assembler cannot consume the projected mmCIF,
    # the native graph is returned for all explicitly selected chains.
    assert multi_chain.status == STATUS_SUCCESS
    assert multi_chain.result_origin == "native_mmcif_graph"
    assert multi_chain.graph["selected_chain_ids"]
    assert len(multi_chain.graph["selected_chain_ids"]) == 2
    assert multi_chain.structure_profile["chain_count"] == 2
    assert multi_chain.ambiguous is False

    auto_multi = reconstruct_structure(multi, mode="auto")
    assert auto_multi.status == STATUS_SUCCESS
    assert auto_multi.result_origin == "native_mmcif_graph"
    assert auto_multi.ambiguous is True
    assert "UNIFIED_NATIVE_MMCIF_DISCONNECTED" in auto_multi.warning_codes
    assert auto_multi.provenance["chain_resolution"]["ssbond_components"]


def test_chain_id_string_selects_single_chain_and_list_selects_multichain(tmp_path):
    path = _write_two_chain(tmp_path, "both.pdb", with_ssbond=True)

    single = reconstruct_structure(path, chain_id="A", mode="auto")
    multi = reconstruct_structure(path, chain_id=("A", "B"), mode="auto")

    assert single.status in {STATUS_SUCCESS, STATUS_FAILED}
    assert single.provenance["dispatch"]["chain_id"] == "A"
    assert multi.status == STATUS_SUCCESS
    assert multi.provenance["dispatch"]["chain_id"] == ["A", "B"]
    assert multi.structure_profile["chain_count"] == 2


def test_explicit_disconnected_chain_list_is_never_silently_reduced(
    tmp_path, monkeypatch
):
    path = _write_two_chain(tmp_path, "explicit_disconnected.pdb", with_ssbond=False)
    captured = {}

    def fake_multichain(path_text, chain_ids, source_kind, mode, detection):
        from cycpep_master.reconstruction import _typed_failed

        captured["chain_ids"] = list(chain_ids)
        return _typed_failed(
            source=path_text,
            chain_id=chain_ids,
            mode=mode,
            source_kind=source_kind,
            detection=detection,
            branch="multichain_assembly",
            failure_reason="directed fixture failure",
            warning_codes=["UNIFIED_REPRESENTATION_ASSEMBLY_FAILED"],
        )

    monkeypatch.setattr(
        "cycpep_master.reconstruction._multichain_assembly", fake_multichain
    )
    result = reconstruct_structure(
        path, chain_id=["A", "B"], mode="best_effort"
    )

    assert captured["chain_ids"] == ["A", "B"]
    assert result.status == STATUS_SUCCESS
    assert result.ambiguous is True
    assert result.alternatives
    assert result.provenance["chain_ambiguity"]["resolved"] is True


def test_explicit_multichain_missing_member_is_rejected_before_assembly(
    tmp_path, monkeypatch
):
    path = _write_two_chain(tmp_path, "missing_chain.pdb", with_ssbond=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("an incomplete chain selection must not be assembled")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._multichain_assembly", forbidden
    )
    result = reconstruct_structure(
        path, chain_id=["A", "Z"], mode="best_effort"
    )

    assert result.status == STATUS_FAILED
    assert "UNIFIED_REQUESTED_CHAIN_NOT_AVAILABLE" in result.warning_codes
    resolution = result.provenance["underlying"]["chain_resolution"]
    assert resolution["requested_chain_ids"] == ["A", "Z"]
    assert resolution["missing_chain_ids"] == ["Z"]
    assert resolution["available_peptide_chain_ids"] == ["A", "B"]


def test_explicit_multichain_duplicate_member_is_rejected_before_assembly(
    tmp_path, monkeypatch
):
    path = _write_two_chain(tmp_path, "duplicate_chain.pdb", with_ssbond=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("duplicate chains must not reach assembly")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._multichain_assembly", forbidden
    )
    result = reconstruct_structure(path, chain_id=["A", "A"], mode="auto")

    assert result.status == STATUS_FAILED
    assert "UNIFIED_DUPLICATE_CHAIN_SELECTION" in result.warning_codes
    resolution = result.provenance["underlying"]["chain_resolution"]
    assert resolution["duplicate_chain_ids"] == ["A"]


def test_explicit_multichain_empty_member_is_rejected_before_assembly(
    tmp_path, monkeypatch
):
    path = _write_two_chain(tmp_path, "empty_chain.pdb", with_ssbond=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("empty chain identifiers must not reach assembly")

    monkeypatch.setattr(
        "cycpep_master.reconstruction._multichain_assembly", forbidden
    )
    result = reconstruct_structure(path, chain_id=["A", " "], mode="auto")

    assert result.status == STATUS_FAILED
    assert "UNIFIED_EMPTY_CHAIN_IDENTIFIER" in result.warning_codes
    resolution = result.provenance["underlying"]["chain_resolution"]
    assert resolution["empty_chain_id_positions"] == [1]


def test_explicit_prefixes_are_honored_and_mismatches_fail_closed(tmp_path):
    path = _mmcif_with_chains(tmp_path, "prefix.cif", ("A",))
    pdb_path = tmp_path / "prefix.pdb"
    pdb_path.write_text("END\n", encoding="ascii")

    sequence = reconstruct_structure("sequence:AAAAA")
    helm = reconstruct_structure("helm:PEPTIDE1{A.A}$$$$V2.0")
    map_prefixed = reconstruct_structure("map:AAAAA{cyc:N-C}")
    biln = reconstruct_structure("biln:C(1,3)-A-A-A-C(1,3)")
    coordinate = reconstruct_structure(f"coordinate:{path}", chain_id="A")
    pdb_prefixed = reconstruct_structure(f"pdb:{pdb_path}", chain_id="A")
    mismatch = reconstruct_structure(f"mmcif:{pdb_path}", chain_id="A")
    bad_sequence = reconstruct_structure("sequence:AAAAA{cyc:N-C}")

    assert sequence.source_kind == "sequence" and sequence.status == STATUS_SUCCESS
    assert helm.source_kind == "helm" and helm.status == STATUS_SUCCESS
    assert map_prefixed.source_kind == "map" and map_prefixed.status == STATUS_SUCCESS
    assert biln.source_kind == "biln" and biln.status == STATUS_SUCCESS
    assert coordinate.source_kind == "coordinate"
    assert coordinate.provenance["detection"]["coordinate_format"] == "mmcif"
    assert pdb_prefixed.source_kind == "pdb"
    assert mismatch.status == STATUS_FAILED
    assert mismatch.source_kind == "mmcif"
    assert "UNIFIED_PREFIX_KIND_MISMATCH" in mismatch.warning_codes
    assert bad_sequence.status == STATUS_FAILED
    assert "UNIFIED_PREFIX_KIND_MISMATCH" in bad_sequence.warning_codes


@pytest.mark.parametrize(
    "source,expected_kind",
    [
        ("map:PEPTIDE1{A.A}$$$$V2.0", "map"),
        ("helm:AAAAA", "helm"),
        ("biln:AAAAA", "biln"),
    ],
)
def test_best_effort_never_reinterprets_an_explicit_notation_prefix(
    source, expected_kind
):
    result = reconstruct_structure(source, mode="best_effort")

    assert result.status == STATUS_FAILED
    assert result.source_kind == expected_kind
    assert result.provenance["detection"]["explicit_prefix"] == expected_kind
    assert result.alternatives == []


def test_invalid_mode_is_typed_failed():
    result = reconstruct_structure("AAAAA", mode="aggressive")

    assert result.status == STATUS_FAILED
    assert "UNIFIED_INVALID_MODE" in result.warning_codes
    assert MODES == ("strict", "auto", "best_effort")


def test_best_effort_spelling_normalized_in_api():
    result = reconstruct_structure("AAAAA", mode="best-effort")

    assert result.status == STATUS_SUCCESS
    assert result.mode == "best_effort"


def test_legacy_entry_points_remain_intact():
    from cycpep_master import result_first
    from cycpep_master import reconstruction

    assert callable(result_first.reconstruct_structure)
    assert callable(application.reconstruct_result_first)
    assert callable(application.reconstruct_coordinates)
    assert callable(application.reconstruct_multichain)
    # The package-level lazy export resolves to the new orchestration API.
    assert cycpep_master.reconstruct_structure is reconstruction.reconstruct_structure
    assert cycpep_master.UnifiedReconstructionResult is UnifiedReconstructionResult


def test_application_service_returns_operation_contract():
    result = application.reconstruct_unified("AAAAA", mode="strict")

    assert result["operation"] == "reconstruct_unified"
    assert result["status"] == "success"
    assert result["data"]["status"] == "success"
    assert result["data"]["quality"] == "high"
    assert result["data"]["source_kind"] == "sequence"
    assert result["data"]["inchikey"] is not None


def test_application_service_failed_payload_is_json_ready(tmp_path):
    result = application.reconstruct_unified(
        "AA{cyc:1:R4-2:R2}", mode="best_effort"
    )

    assert result["status"] == "failed"
    assert result["error"]
    assert result["data"]["status"] == "failed"
    assert result["data"]["warning_codes"]
