from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import threading

import cycpep_master
from cycpep_master import application
from cycpep_master.core.monomer_resolution import (
    monomer_symbol_hints,
    monomer_resolution_context,
)
from cycpep_master.exact_v1 import (
    exact_v1_equivalent,
    exact_v1_to_edge_v1,
    map_to_exact_v1,
    validate_exact_v1,
)
from cycpep_master.paths import path_g
from cycpep_master.sequence import build_molecule_from_sequence


CUSTOM = {
    "definitions": [{
        "symbol": "ZZZ_TEST",
        "smiles": "N[C@@H](CCCl)C(=O)O",
    }],
    "include_persistent_user": False,
}


def _process_resolution_probe(own, other, smiles):
    context = {
        "definitions": [{"symbol": own, "smiles": smiles}],
        "include_persistent_user": False,
    }
    with monomer_resolution_context(context):
        return (
            map_to_exact_v1(f"{{nnr:{own}}}A")["exactness_status"],
            map_to_exact_v1(f"{{nnr:{other}}}A")["exactness_status"],
        )


def test_unknown_sequence_is_partial_not_rejected():
    result = build_molecule_from_sequence(
        "[UNKNOWN_REALTIME]AC",
        cyclization="linear",
    )
    graph = result["chemical_graph"]

    assert graph["status"] == "PARTIAL"
    assert graph["evidence"]["chemical_rigor"] == "C1:H"
    assert graph["materializable"] is False
    assert result["alternatives"]


def test_explicit_context_materializes_exact_and_restores_registry():
    before = map_to_exact_v1("{nnr:ZZZ_TEST}A")
    result = build_molecule_from_sequence(
        "[ZZZ_TEST]A",
        cyclization="linear",
        monomer_context=CUSTOM,
    )
    graph = result["chemical_graph"]
    after = map_to_exact_v1("{nnr:ZZZ_TEST}A")

    assert before["exactness_status"] == "ABSTAIN"
    assert graph["status"] == "MATERIALIZED"
    assert graph["evidence"]["chemical_rigor"] == "C3:S"
    assert graph["exact_v1"]["exactness_status"] == "EXACT"
    assert graph["monomer_resolution"]["persistent_writes"] == 0
    assert after["exactness_status"] == "ABSTAIN"


def test_representation_conversion_consumes_same_context():
    result = application.convert_representation(
        "map",
        "smiles",
        "{nnr:ZZZ_TEST}A",
        monomer_context=CUSTOM,
    )

    assert result["status"] == "success"
    assert result["data"]["artifact_status"] == "MATERIALIZED"
    assert result["data"]["chemical_rigor"] == "C3:S"
    assert result["data"]["value"]


def test_representation_miss_returns_lower_rigor_envelope():
    result = application.convert_representation(
        "map",
        "smiles",
        "{nnr:UNKNOWN_REALTIME}A",
    )

    assert result["status"] == "success"
    assert result["data"]["artifact_status"] == "PARTIAL"
    assert result["data"]["chemical_rigor"] == "C1:H"
    assert result["data"]["requested_artifact_status"] == (
        "NOT_MATERIALIZABLE"
    )


def test_public_resolver_reports_entity_local_extension():
    result = application.resolve_monomers(
        ["ZZZ_TEST"],
        monomer_context=CUSTOM,
    )

    assert result["status"] == "success"
    assert result["data"]["status"] == "resolved"
    assert result["data"]["chemical_rigor"] == "C3:S"
    assert result["data"]["resolved"]["ZZZ_TEST"]["ports"] == {
        "R1": "N",
        "R2": "C",
    }
    assert result["data"]["persistent_writes"] == 0
    assert result["data"]["registry_isolation"][
        "registry_restoration_verified"
    ] is True


def test_unified_public_api_accepts_context_and_unknown_is_partial():
    exact = cycpep_master.reconstruct_structure(
        "[ZZZ_TEST]A",
        monomer_context=CUSTOM,
    )
    partial = cycpep_master.reconstruct_structure(
        "[UNKNOWN_REALTIME]A"
    )

    assert exact.status == "success"
    assert exact.quality == "high"
    assert exact.smiles
    assert exact.provenance["monomer_resolution"]["status"] == "resolved"
    assert partial.status == "success"
    assert partial.quality == "partial"
    assert partial.smiles is None
    assert partial.graph["chemical_rigor"] == "C1:H"


def test_unknown_audit_returns_symbolic_layer_not_hard_rejection():
    result = application.audit_chemistry(
        "map", payload="{nnr:UNKNOWN_REALTIME}A"
    )

    assert result["status"] == "success"
    assert result["data"]["artifact_status"] == "PARTIAL"
    assert result["data"]["chemical_rigor"] == "C1:H"


def test_concurrent_entity_contexts_do_not_cross_resolve():
    contexts = {
        "CTX_ONE": {
            "definitions": [{
                "symbol": "CTX_ONE",
                "smiles": "N[C@@H](CBr)C(=O)O",
            }],
            "include_persistent_user": False,
        },
        "CTX_TWO": {
            "definitions": [{
                "symbol": "CTX_TWO",
                "smiles": "N[C@@H](CI)C(=O)O",
            }],
            "include_persistent_user": False,
        },
    }

    def run(own, other):
        with monomer_resolution_context(contexts[own]):
            own_result = map_to_exact_v1(f"{{nnr:{own}}}A")
            other_result = map_to_exact_v1(f"{{nnr:{other}}}A")
        return (
            own_result["exactness_status"],
            other_result["exactness_status"],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda pair: run(*pair),
            (("CTX_ONE", "CTX_TWO"), ("CTX_TWO", "CTX_ONE")),
        ))

    assert results == [("EXACT", "ABSTAIN"), ("EXACT", "ABSTAIN")]
    assert map_to_exact_v1(
        "{nnr:CTX_ONE}A"
    )["exactness_status"] == "ABSTAIN"
    assert map_to_exact_v1(
        "{nnr:CTX_TWO}A"
    )["exactness_status"] == "ABSTAIN"


def test_unscoped_reader_waits_for_entity_context_restoration():
    entered = threading.Event()
    release = threading.Event()

    def scoped_writer():
        with monomer_resolution_context(CUSTOM):
            entered.set()
            assert release.wait(60)
            return map_to_exact_v1(
                "{nnr:ZZZ_TEST}A"
            )["exactness_status"]

    def unscoped_reader():
        assert entered.wait(60)
        return map_to_exact_v1(
            "{nnr:ZZZ_TEST}A"
        )["exactness_status"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(scoped_writer)
        reader = pool.submit(unscoped_reader)
        assert entered.wait(60)
        assert not reader.done()
        release.set()

    assert writer.result() == "EXACT"
    assert reader.result() == "ABSTAIN"


def test_path_g_runtime_variant_is_visible_to_exact_registry():
    before = map_to_exact_v1("{nnr:LanA}A")
    with monomer_resolution_context({
        "include_persistent_user": False
    }):
        path_g._register_lanthionine_variants()
        during = map_to_exact_v1("{nnr:LanA}A")
    after = map_to_exact_v1("{nnr:LanA}A")

    assert before["exactness_status"] == "ABSTAIN"
    assert during["exactness_status"] == "EXACT"
    assert after["exactness_status"] == "ABSTAIN"


def test_standard_pdb_alias_does_not_trigger_network_fetch(monkeypatch):
    from cycpep_master.core import monomer_resolution as module

    calls = []
    monkeypatch.setattr(
        module,
        "_download_component",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with monomer_resolution_context(
        {"allow_network": True},
        required_symbols=["ALA", "GLY"],
    ) as ledger:
        assert ledger["status"] == "resolved"

    assert calls == []
    assert set(ledger["resolved"]) == {"ALA", "GLY"}


def test_context_exception_restores_registry():
    try:
        with monomer_resolution_context(CUSTOM):
            assert map_to_exact_v1(
                "{nnr:ZZZ_TEST}A"
            )["exactness_status"] == "EXACT"
            raise RuntimeError("fixture abort")
    except RuntimeError:
        pass

    assert map_to_exact_v1(
        "{nnr:ZZZ_TEST}A"
    )["exactness_status"] == "ABSTAIN"


def test_process_entity_contexts_are_independent():
    jobs = (
        ("PROC_ONE", "PROC_TWO", "N[C@@H](CBr)C(=O)O"),
        ("PROC_TWO", "PROC_ONE", "N[C@@H](CI)C(=O)O"),
    )
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            _process_resolution_probe,
            (row[0] for row in jobs),
            (row[1] for row in jobs),
            (row[2] for row in jobs),
        ))

    assert results == [("EXACT", "ABSTAIN"), ("EXACT", "ABSTAIN")]


def test_symbol_hints_cover_notations_and_peptide_pdb(tmp_path):
    assert monomer_symbol_hints(
        "{nnr:XAA}AC", kind="map"
    ) == ("XAA",)
    assert monomer_symbol_hints(
        "{nnr:XAA}" + ("A" * 500), kind="map"
    ) == ("XAA",)
    assert monomer_symbol_hints(
        "PEPTIDE1{A.[XBB]}$$$$", kind="helm"
    ) == ("XBB", "A")
    assert monomer_symbol_hints(
        "[XCC]-A", kind="biln"
    ) == ("XCC", "A")
    pdb = tmp_path / "hints.pdb"
    pdb.write_text(
        "\n".join([
            "HETATM    1    N XDD A   1       0.000   0.000   0.000  1.00  0.00           N",
            "HETATM    2   CA XDD A   1       1.400   0.000   0.000  1.00  0.00           C",
            "HETATM    3    C XDD A   1       2.800   0.000   0.000  1.00  0.00           C",
            "HETATM    4   C1 LIG B   1       5.000   0.000   0.000  1.00  0.00           C",
            "END",
            "",
        ]),
        encoding="ascii",
    )

    assert monomer_symbol_hints(
        pdb, kind="coordinate"
    ) == ("XDD",)
    compressed = tmp_path / "hints.pdb.gz"
    with gzip.open(compressed, "wt", encoding="ascii") as handle:
        handle.write(pdb.read_text(encoding="ascii"))
    assert monomer_symbol_hints(
        compressed, kind="coordinate"
    ) == ("XDD",)


def test_exact_validation_equivalence_and_projection_accept_context():
    document = map_to_exact_v1(
        "{nnr:ZZZ_TEST}A",
        monomer_context=CUSTOM,
    )

    validated = validate_exact_v1(
        document, monomer_context=CUSTOM
    )
    projected = exact_v1_to_edge_v1(
        document, monomer_context=CUSTOM
    )

    assert validated["exactness_status"] == "EXACT"
    assert exact_v1_equivalent(
        document, validated, monomer_context=CUSTOM
    )
    assert projected["status"] == "PROJECTED"


def test_template_lookup_consumes_custom_context(monkeypatch):
    from cycpep_master.docking import template_library

    observed = {}

    def fake_find(generated_map, *, meta_out=None, **_kwargs):
        observed["exactness_status"] = map_to_exact_v1(
            generated_map
        )["exactness_status"]
        if meta_out is not None:
            meta_out.update({
                "status": "no_compatible_template",
                "reason": "fixture",
            })
        return None

    monkeypatch.setattr(
        template_library, "find_template", fake_find
    )
    result = application.find_conformer_template(
        "{nnr:ZZZ_TEST}A",
        monomer_context=CUSTOM,
    )

    assert observed["exactness_status"] == "EXACT"
    assert result["status"] == "success"
    assert result["data"]["artifact_status"] == "PARTIAL"


def test_pdbqt_compatibility_path_keeps_smiles_when_context_export_fails(
    tmp_path, monkeypatch
):
    from cycpep_master.docking import template_library
    from cycpep_master.export import conformer

    observed = {}

    def fake_generate(smiles, generated_map, **_kwargs):
        observed["exactness_status"] = map_to_exact_v1(
            generated_map
        )["exactness_status"]
        return None, []

    monkeypatch.setattr(
        template_library, "generate_conformers", fake_generate
    )
    monkeypatch.setattr(
        conformer,
        "smiles_to_mol2",
        lambda *_args, **_kwargs: (None, "fixture export failure"),
    )

    result = application.prepare_ligand_pdbqt(
        "N[C@@H](CCl)C(=O)O",
        tmp_path / "ligand.pdbqt",
        generated_map="{nnr:ZZZ_TEST}A",
        protonate=False,
        monomer_context=CUSTOM,
    )

    assert observed["exactness_status"] == "EXACT"
    assert result["status"] == "success"
    assert result["data"]["artifact_status"] == "PARTIAL"
    assert result["data"]["requested_artifact_status"] == (
        "PDBQT_UNAVAILABLE"
    )
    assert result["data"]["alternatives"][0]["kind"] == "smiles"


def test_path_g_coordinate_route_consumes_custom_context(tmp_path):
    rows = []
    serial = 1
    for residue_number, residue_name, offset in (
        (1, "ZZA", 0.0),
        (2, "ALA", 4.0),
    ):
        for atom_name, x, element in (
            ("N", 0.0, "N"),
            ("CA", 1.4, "C"),
            ("C", 2.8, "C"),
        ):
            rows.append(
                f"HETATM{serial:5d} {atom_name:>4s} "
                f"{residue_name:>3s} L{residue_number:4d}    "
                f"{x + offset:8.3f}{0.0:8.3f}{0.0:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
            serial += 1
    source = tmp_path / "custom_path_g.pdb"
    source.write_text(
        "\n".join([*rows, "END", ""]), encoding="ascii"
    )
    context = {
        "definitions": [{
            "symbol": "ZZA",
            "smiles": "N[C@@H](CCl)C(=O)O",
        }],
        "include_persistent_user": False,
    }

    smiles, error = path_g.generate_g(
        source,
        chain_id="L",
        monomer_context=context,
    )

    assert error is None
    assert smiles
