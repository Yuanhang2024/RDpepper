from types import SimpleNamespace

import cycpep_master.pipeline as pipeline
from cycpep_master.compare import compare


def _prepared_input(tmp_path):
    source = tmp_path / "example.pdb"
    source.write_text("placeholder\n", encoding="ascii")
    prepared = SimpleNamespace(
        pdb_path=source,
        chain_id="L",
        source_format="pdb",
    )
    return [(str(source), prepared, None, None, None, None)]


def _stub_pipeline_metadata(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "parse_chain_sequence",
        lambda *_args: [{"name": "ALA"}],
    )
    monkeypatch.setattr(
        pipeline,
        "detect_cyclization",
        lambda *_args: SimpleNamespace(topology="linear"),
    )
    monkeypatch.setattr(
        pipeline,
        "build_helm_from_pdb",
        lambda *_args: "",
    )


def test_explicit_route_failure_is_not_replaced_by_reference_route(
    tmp_path, monkeypatch
):
    _stub_pipeline_metadata(monkeypatch)
    monkeypatch.setitem(
        pipeline.PATH_MAP,
        "b",
        lambda *_args: (None, "path b failed"),
    )
    monkeypatch.setitem(
        pipeline.PATH_MAP,
        "c",
        lambda *_args: ("CC", None),
    )

    result = pipeline._run_batch_prepared(
        _prepared_input(tmp_path), path="b"
    )[0]

    assert result["status"] == "failed"
    assert result["smiles"] is None
    assert result["primary_path"] == "b"
    assert result["primary_path_status"] == "failed"
    assert result["primary_path_error"] == "path b failed"
    assert result["fallback_status"] == "available"
    assert result["fallback_source"] == "c"
    assert result["fallback_smiles"] == "CC"
    assert result["fallback_warning"] == (
        "PRIMARY_PATH_FAILED_FALLBACK_AVAILABLE"
    )


def test_route_comparison_columns_name_the_actual_pair(tmp_path, monkeypatch):
    _stub_pipeline_metadata(monkeypatch)
    monkeypatch.setitem(
        pipeline.PATH_MAP,
        "b",
        lambda *_args: ("CCC", None),
    )
    monkeypatch.setitem(
        pipeline.PATH_MAP,
        "c",
        lambda *_args: ("CC", None),
    )
    output = tmp_path / "results.csv"

    result = pipeline._run_batch_prepared(
        _prepared_input(tmp_path), path="b", csv_output=str(output)
    )[0]

    assert result["status"] == "success"
    assert result["smiles"] == "CCC"
    assert result["_row"]["compare_b_vs_c"] == "DIFF"
    assert "compare_a_vs_c" not in result["_row"]
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert "compare_b_vs_c" in header
    assert "compare_a_vs_c" not in header


def test_c_route_comparison_is_reported_as_c_vs_a(tmp_path, monkeypatch):
    _stub_pipeline_metadata(monkeypatch)
    monkeypatch.setitem(
        pipeline.PATH_MAP,
        "c",
        lambda *_args: ("CC", None),
    )
    monkeypatch.setitem(
        pipeline.PATH_MAP,
        "a",
        lambda *_args: ("CC", None),
    )

    result = pipeline._run_batch_prepared(
        _prepared_input(tmp_path), path="c"
    )[0]

    assert result["_row"]["compare_c_vs_a"] == "PASS"
    assert "compare_a_vs_c" not in result["_row"]


def test_permissive_compare_rejects_different_composition_before_fingerprint():
    matched, detail = compare("C" * 28, "C" * 29)

    assert matched is False
    assert "composition mismatch" in detail
