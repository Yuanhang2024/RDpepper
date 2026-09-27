from types import SimpleNamespace

import cycpep_master.pipeline as pipeline
from cycpep_master.core.rigor import RigorLevel, rigor_from_result


def _result(quality, *, status="success", support=None, evidence=None):
    strict = None
    if support is not None or evidence is not None:
        strict = SimpleNamespace(
            support_status=support or "unknown",
            output_evidence=evidence or {},
        )
    return SimpleNamespace(
        status=status,
        quality=quality,
        strict_result=strict,
    )


def test_rigor_level_label_is_serializable():
    level = RigorLevel("L2", "Q")
    assert level.label == "L2:Q"


def test_exact_qualified_without_stereo_is_l2_q():
    result = _result(
        "exact",
        support="qualified",
        evidence={"highest_evidence_qualified_level": "L2"},
    )
    assert rigor_from_result(result).label == "L2:Q"


def test_exact_qualified_with_stereo_is_l3_q():
    result = _result(
        "exact",
        support="qualified",
        evidence={"highest_evidence_qualified_level": "L3"},
    )
    assert rigor_from_result(result).label == "L3:Q"


def test_exact_repaired_is_l2_r():
    assert rigor_from_result(_result("exact", support="repaired")).label == "L2:R"


def test_candidate_qualities_are_l2_r():
    for quality in ("high", "medium"):
        assert rigor_from_result(_result(quality)).label == "L2:R"


def test_fallback_qualities_are_conservative():
    assert rigor_from_result(_result("topology")).label == "L1:H"
    assert rigor_from_result(_result("partial")).label == "L1:R"
    assert rigor_from_result(_result("raw")).label == "L0:C"


def test_failed_result_is_none_provenance():
    assert rigor_from_result(_result(None, status="failed")).label == "L0:NONE"
    assert rigor_from_result(None).label == "L0:NONE"


def test_json_ready_dict_input_uses_strict_evidence():
    payload = {
        "status": "success",
        "quality": "exact",
        "strict_result": {
            "support_status": "qualified",
            "output_evidence": {"highest_evidence_qualified_level": "L3"},
        },
    }
    assert rigor_from_result(payload).label == "L3:Q"


def test_run_batch_csv_contains_rigor_columns(tmp_path, monkeypatch):
    source = tmp_path / "example.pdb"
    source.write_text("placeholder\n", encoding="ascii")
    prepared = SimpleNamespace(pdb_path=source, chain_id="L", source_format="pdb")
    monkeypatch.setattr(pipeline, "parse_chain_sequence", lambda *_args: [{"name": "ALA"}])
    monkeypatch.setattr(
        pipeline,
        "detect_cyclization",
        lambda *_args: SimpleNamespace(topology="linear"),
    )
    monkeypatch.setattr(pipeline, "build_helm_from_pdb", lambda *_args: "")
    monkeypatch.setitem(pipeline.PATH_MAP, "b", lambda *_args: ("CCC", None))
    monkeypatch.setitem(pipeline.PATH_MAP, "c", lambda *_args: ("CCC", None))
    output = tmp_path / "results.csv"

    rows = pipeline._run_batch_prepared(
        [(str(source), prepared, None, None, None, None)],
        path="b",
        csv_output=str(output),
    )

    assert rows[0]["rigor"] == "L2:R"
    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {"rigor_level", "rigor_provenance", "rigor"}.issubset(header)
