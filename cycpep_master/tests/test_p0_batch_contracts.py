"""Focused regression tests for batch worker identity contracts."""

from cycpep_master import application


def test_reconstruct_multichain_rejects_explicit_empty_chain_selection():
    result = application.reconstruct_multichain("unused.pdb", chain_ids=[])

    assert result["status"] == "invalid_input"
    assert "empty" in result["error"]


def test_reconstruct_rejects_equal_length_duplicate_rows_and_keeps_raw_rows(
    monkeypatch,
):
    import cycpep_master.pipeline as pipeline_module

    raw_rows = [
        {"source_path": "a.pdb", "status": "success", "smiles": "CC"},
        {"source_path": "a.pdb", "status": "success", "smiles": "CCC"},
    ]
    monkeypatch.setattr(pipeline_module, "run_batch", lambda *_a, **_k: raw_rows)

    result = application.reconstruct_coordinates(["a.pdb", "b.pdb"])

    assert result["status"] == "partial"
    assert result["data"]["identity_mismatch"] is True
    assert result["data"]["duplicate_result_ids"] == ["a.pdb"]
    assert result["data"]["raw_rows"] == raw_rows
    assert "duplicate result identities" in result["error"]


def test_reconstruct_rejects_equal_length_out_of_order_rows(monkeypatch):
    import cycpep_master.pipeline as pipeline_module

    raw_rows = [
        {"source_path": "b.pdb", "status": "success"},
        {"source_path": "a.pdb", "status": "success"},
    ]
    monkeypatch.setattr(pipeline_module, "run_batch", lambda *_a, **_k: raw_rows)

    result = application.reconstruct_coordinates(["a.pdb", "b.pdb"])

    assert result["status"] == "partial"
    assert result["data"]["order_mismatch"] is True
    assert result["data"]["actual_ids"] == ["b.pdb", "a.pdb"]
    assert "order" in result["error"]


def test_reconstruct_reports_legacy_file_when_source_path_is_none(monkeypatch):
    import cycpep_master.pipeline as pipeline_module

    raw_rows = [{"source_path": None, "file": "a.pdb", "status": "success"}]
    monkeypatch.setattr(pipeline_module, "run_batch", lambda *_a, **_k: raw_rows)

    result = application.reconstruct_coordinates(["a.pdb"])

    assert result["status"] == "success"
    assert result["data"]["identity_mismatch"] is False
    assert result["data"]["actual_ids"] == ["a.pdb"]


def test_reconstruct_malformed_worker_row_is_structured_and_preserved(monkeypatch):
    import cycpep_master.pipeline as pipeline_module

    raw_rows = [{"source_path": "a.pdb", "status": "success"}, "malformed"]
    monkeypatch.setattr(pipeline_module, "run_batch", lambda *_a, **_k: raw_rows)

    result = application.reconstruct_coordinates(["a.pdb", "b.pdb"])

    assert result["status"] == "partial"
    assert result["data"]["malformed_row_indices"] == [1]
    assert result["data"]["results"][1]["status"] == "failed"
    assert result["data"]["raw_rows"] == raw_rows
    assert "malformed worker rows" in result["error"]


def test_batch_export_rejects_equal_length_duplicate_rows_and_keeps_raw_rows(
    tmp_path, monkeypatch
):
    from cycpep_master.export import conformer

    raw_rows = [
        ("first", str(tmp_path / "first.mol2"), None),
        ("first", str(tmp_path / "first-2.mol2"), None),
    ]
    monkeypatch.setattr(conformer, "batch_export", lambda *_a, **_k: raw_rows)

    result = application.batch_export_structures(
        {"first": "C", "second": "CC"}, tmp_path
    )

    assert result["status"] == "partial"
    assert result["data"]["identity_mismatch"] is True
    assert result["data"]["duplicate_result_ids"] == ["first"]
    assert result["data"]["raw_rows"] == [list(row) for row in raw_rows]
    assert "duplicate result identities" in result["error"]


def test_batch_export_malformed_worker_row_is_structured(tmp_path, monkeypatch):
    from cycpep_master.export import conformer

    raw_rows = [("first", str(tmp_path / "first.mol2"))]
    monkeypatch.setattr(conformer, "batch_export", lambda *_a, **_k: raw_rows)

    result = application.batch_export_structures({"first": "C"}, tmp_path)

    assert result["status"] == "failed"
    assert result["data"]["malformed_row_indices"] == [0]
    assert result["data"]["results"][0]["status"] == "failed"
    assert result["data"]["raw_rows"] == [list(raw_rows[0])]
    assert "malformed worker rows" in result["error"]


def test_batch_dock_rejects_equal_length_duplicate_rows_and_keeps_raw_rows(
    monkeypatch,
):
    import cycpep_master.docking.workflow as workflow_module

    raw_rows = [("a", -5.0, None), ("a", -6.0, None)]
    monkeypatch.setattr(
        workflow_module, "batch_dock_peptides", lambda *_a, **_k: raw_rows
    )

    result = application.batch_dock_structures(
        ["a.pdb", "b.pdb"], "receptor.pdb", center=(0, 0, 0)
    )

    assert result["status"] == "partial"
    assert result["data"]["identity_mismatch"] is True
    assert result["data"]["duplicate_result_ids"] == ["a"]
    assert result["data"]["raw_rows"] == [list(row) for row in raw_rows]
    assert "duplicate result identities" in result["error"]


def test_batch_dock_malformed_worker_row_is_structured(monkeypatch):
    import cycpep_master.docking.workflow as workflow_module

    raw_rows = [("a", -5.0)]
    monkeypatch.setattr(
        workflow_module, "batch_dock_peptides", lambda *_a, **_k: raw_rows
    )

    result = application.batch_dock_structures(
        ["a.pdb"], "receptor.pdb", center=(0, 0, 0)
    )

    assert result["status"] == "failed"
    assert result["data"]["malformed_row_indices"] == [0]
    assert result["data"]["results"][0]["status"] == "failed"
    assert result["data"]["raw_rows"] == [list(raw_rows[0])]
    assert "malformed worker rows" in result["error"]
