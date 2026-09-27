from __future__ import annotations

import ast
from pathlib import Path

from rdkit import Chem


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EMBED_CALLS = frozenset({
    "ETKDGv3",
    "EmbedMolecule",
    "EmbedMultipleConfs",
})
EXPECTED_INVENTORY = {
    ("docking/template_library.py", "_etkdg_conformer", "ETKDGv3"),
    (
        "docking/template_library.py",
        "_etkdg_conformer",
        "EmbedMultipleConfs",
    ),
    (
        "docking/template_library.py",
        "borrow_residue_coords",
        "ETKDGv3",
    ),
    (
        "docking/template_library.py",
        "borrow_residue_coords",
        "EmbedMolecule",
    ),
    (
        "docking/template_library.py",
        "generate_conformers",
        "ETKDGv3",
    ),
    (
        "docking/template_library.py",
        "generate_conformers",
        "EmbedMultipleConfs",
    ),
    ("export/conformer.py", "_embed_3d", "ETKDGv3"),
    ("export/conformer.py", "_embed_3d", "EmbedMultipleConfs"),
    (
        "export/conformer.py",
        "compute_conformer_ensemble_stats",
        "ETKDGv3",
    ),
    (
        "export/conformer.py",
        "compute_conformer_ensemble_stats",
        "EmbedMultipleConfs",
    ),
    ("export/conformer_ensemble.py", "_embed", "ETKDGv3"),
    ("export/conformer_ensemble.py", "_embed", "EmbedMolecule"),
    # 6.1.0 max-coverage fallback: ETKDG is the last-resort branch after
    # local bond-geometry placement fails to fill a coordinate gap.
    ("export/conformer.py", "pdb_to_mol2", "EmbedMolecule"),
}


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str):
        self.relative_path = relative_path
        self.function_name = "<module>"
        self.calls: set[tuple[str, str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous = self.function_name
        self.function_name = node.name
        self.generic_visit(node)
        self.function_name = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr in EMBED_CALLS
        ):
            self.calls.add((
                self.relative_path,
                self.function_name,
                function.attr,
            ))
        self.generic_visit(node)


def _production_python_files():
    for relative_root in (
        "admet",
        "cli",
        "compare",
        "core",
        "docking",
        "export",
        "gui",
        "paths",
    ):
        yield from (PACKAGE_ROOT / relative_root).rglob("*.py")
    yield from (
        path
        for path in PACKAGE_ROOT.glob("*.py")
        if not path.name.startswith(("test_", "build_"))
    )


def test_all_production_embedding_call_sites_are_owned_and_reviewed():
    observed = set()
    for path in _production_python_files():
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        visitor = _CallVisitor(relative)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        observed.update(visitor.calls)

    assert observed == EXPECTED_INVENTORY


def test_pdbqt_and_torsion_budget_never_own_embedding():
    forbidden_modules = {
        "docking/mol2_pdbqt.py",
        "docking/torsion_budget.py",
    }
    assert not {
        row for row in EXPECTED_INVENTORY if row[0] in forbidden_modules
    }
    mol2_input = (
        PACKAGE_ROOT / "docking" / "mol2_input.py"
    ).read_text(encoding="utf-8")
    assert "export.conformer" not in mol2_input


def test_legacy_primary_and_retry_embeddings_are_bounded(
    monkeypatch,
):
    from cycpep_master.export import conformer

    observed = []

    def failed(_molecule, *, numConfs, params):
        observed.append((numConfs, int(params.timeout)))
        return []

    monkeypatch.setattr(
        conformer.AllChem, "EmbedMultipleConfs", failed
    )
    molecule, error = conformer._embed_3d(
        Chem.MolFromSmiles("CC"), num_confs=2
    )

    assert molecule is None
    assert "embedding failed" in error
    assert observed == [
        (2, conformer.ETKDG_ATTEMPT_TIMEOUT_SECONDS),
        (2, conformer.ETKDG_ATTEMPT_TIMEOUT_SECONDS),
    ]


def test_v5_materializer_embedding_is_bounded(monkeypatch):
    from cycpep_master.export import conformer_ensemble

    observed = []

    def failed(_molecule, parameters):
        observed.append(int(parameters.timeout))
        return -1

    monkeypatch.setattr(
        conformer_ensemble.AllChem, "EmbedMolecule", failed
    )
    molecule, audit = conformer_ensemble._embed(
        Chem.AddHs(Chem.MolFromSmiles("CC")),
        random_seed=42,
        num_threads=1,
    )

    assert molecule is None
    assert audit["status"] == "embed_failed"
    assert observed == [
        conformer_ensemble.V5_ETKDG_ATTEMPT_TIMEOUT_SECONDS
    ]
