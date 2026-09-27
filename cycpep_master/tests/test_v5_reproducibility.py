from __future__ import annotations

from pathlib import Path

import pytest

from cycpep_master.v5_reproducibility import reproduce_v5


def test_reproduction_receipt_is_deterministic_across_directories(
    tmp_path,
):
    outcomes = [
        reproduce_v5(
            tmp_path / name,
            conformer_count=1,
            generate_pdbqt=False,
            random_seed=42,
            num_threads=1,
        )
        for name in ("first", "second")
    ]

    first, second = outcomes
    assert first["result"]["status"] == "success"
    assert second["result"]["status"] == "success"
    assert first["receipt"]["reproduction_digest"] == (
        second["receipt"]["reproduction_digest"]
    )
    assert Path(first["receipt_path"]).is_file()
    assert Path(second["receipt_path"]).is_file()


def test_reproduction_refuses_nonempty_destination(tmp_path):
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "user.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="nonempty"):
        reproduce_v5(
            destination,
            conformer_count=1,
            generate_pdbqt=False,
        )

    assert (destination / "user.txt").read_text(
        encoding="utf-8"
    ) == "keep"
