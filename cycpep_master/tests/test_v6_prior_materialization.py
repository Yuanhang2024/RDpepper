"""V6.0.0 prior-guided conformer materialization tests."""

from __future__ import annotations

from types import SimpleNamespace

from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski

from cycpep_master.export import conformer_ensemble as module
from cycpep_master.sequence import build_molecule_from_sequence


def _graph(sequence: str = "ACDEFG", cyclization: str = "head-to-tail"):
    return build_molecule_from_sequence(
        sequence, cyclization=cyclization
    )["chemical_graph"]


def _annotated_parent(graph):
    molecule = Chem.AddHs(Chem.MolFromSmiles(str(graph["smiles"])))
    module._annotate_residues(molecule, dict(graph["exact_v1"]))
    return molecule


def _bond_by_names(molecule, left_name: str, right_name: str):
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        names = {
            begin.GetProp("_TriposAtomName").strip().upper(),
            end.GetProp("_TriposAtomName").strip().upper(),
        }
        if names == {left_name, right_name}:
            return sorted((begin.GetIdx(), end.GetIdx()))
    return None


def test_derived_topology_head_to_tail():
    graph = _graph()
    derived = module._derived_topology(dict(graph["exact_v1"]))
    assert derived["topology_class"] == "head_to_tail"
    assert derived["macrocycle_ring_size"] == 6
    assert derived["cyclic_backbone"] is True


def test_derived_topology_linear():
    graph = _graph("AG", cyclization="linear")
    derived = module._derived_topology(dict(graph["exact_v1"]))
    assert derived["topology_class"] == "linear"
    assert derived["macrocycle_ring_size"] is None
    assert derived["cyclic_backbone"] is False


def test_derived_topology_disulfide_and_mixed():
    monomers = [
        {"node_id": index, "chain_id": "A", "canonical_position": index}
        for index in (1, 2, 3)
    ]

    def bond(bond_type, src, dst):
        return {
            "bond_type": bond_type,
            "src": {"node_id": src},
            "dst": {"node_id": dst},
        }

    disulfide = module._derived_topology({
        "monomers": monomers,
        "bonds": [
            bond("PEPTIDE", 1, 2),
            bond("PEPTIDE", 2, 3),
            bond("SS", 1, 3),
        ],
    })
    assert disulfide["topology_class"] == "disulfide"
    assert disulfide["macrocycle_ring_size"] == 3

    mixed = module._derived_topology({
        "monomers": monomers,
        "bonds": [
            bond("PEPTIDE", 1, 2),
            bond("PEPTIDE", 2, 3),
            bond("SS", 1, 3),
            bond("HT", 1, 3),
        ],
    })
    assert mixed["topology_class"] == "mixed"
    assert mixed["macrocycle_ring_size"] == 3

    empty = module._derived_topology({})
    assert empty["topology_class"] == "unknown"
    assert empty["macrocycle_ring_size"] is None


def test_prior_guidance_quartet_uses_heavy_atoms_only():
    graph = _graph()
    parent = _annotated_parent(graph)
    n_ca = _bond_by_names(parent, "N", "CA")
    assert n_ca is not None
    quartet = __import__(
        "cycpep_master.docking.torsion_prior",
        fromlist=["prior_guidance_quartet"],
    ).prior_guidance_quartet(parent, n_ca, cyclic_backbone=True)
    assert quartet is not None
    assert len(set(quartet)) == 4
    assert all(
        parent.GetAtomWithIdx(index).GetAtomicNum() > 1
        for index in quartet
    )
    assert parent.GetBondBetweenAtoms(quartet[1], quartet[2]) is not None
    # Phi wraps to the last residue's carbonyl carbon on a cyclic backbone.
    residue_numbers = [
        parent.GetAtomWithIdx(index).GetIntProp(
            "_TriposResidueNumber"
        )
        for index in quartet
    ]
    assert residue_numbers[0] == 6


def _stub_prior(status="matched", confidence="high", std=10.0):
    def query(_keys, **_kwargs):
        return SimpleNamespace(
            status=status,
            lookup_level="exact",
            key="fixture-key",
            rigidity_score=0.9,
            confidence=confidence,
            eligible_to_freeze=True,
            statistics={
                "circular_mean_deg": 60.0,
                "circular_std_deg": std,
            },
            reason=None,
        )

    return SimpleNamespace(
        query=query,
        runtime_sha256="fixture-runtime",
        manifest_sha256="fixture-manifest",
    )


def test_prior_constraints_filters_and_exclusions():
    graph = _graph("AG", cyclization="linear")
    parent = _annotated_parent(graph)

    constraints, audit = module._prior_constraints(
        parent,
        _stub_prior(),
        topology_class="linear",
        macrocycle_ring_size=None,
        cyclic_backbone=False,
    )
    assert audit["status"] == "applied"
    assert constraints
    for row in constraints:
        assert 0.0 <= row["target_deg"] <= 360.0
        assert row["confidence"] in {"high", "medium"}

    wide, wide_audit = module._prior_constraints(
        parent,
        _stub_prior(std=45.0),
        topology_class="linear",
        macrocycle_ring_size=None,
        cyclic_backbone=False,
    )
    assert wide == []
    assert wide_audit["status"] == "no_applicable_prior"

    low, _ = module._prior_constraints(
        parent,
        _stub_prior(confidence="low"),
        topology_class="linear",
        macrocycle_ring_size=None,
        cyclic_backbone=False,
    )
    assert low == []

    none_constraints, none_audit = module._prior_constraints(
        parent,
        None,
        topology_class="linear",
        macrocycle_ring_size=None,
        cyclic_backbone=False,
    )
    assert none_constraints == []
    assert none_audit["status"] == "unavailable"

    excluded = frozenset(
        index
        for row in constraints
        for index in row["atom_indices"]
    )
    if excluded:
        remaining, _ = module._prior_constraints(
            parent,
            _stub_prior(),
            topology_class="linear",
            macrocycle_ring_size=None,
            cyclic_backbone=False,
            excluded_atoms=excluded,
        )
        assert remaining == []


def test_constrained_relaxation_native_constraint(tmp_path):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    quartet = (0, 1, 2, 3)
    constraints = [{
        "atom_indices": [0, 1],
        "quartet": list(quartet),
        "lookup_level": "exact",
        "confidence": "high",
        "target_deg": 60.0,
        "circular_std_deg": 5.0,
    }]
    audit = module._constrained_relaxation(molecule, constraints)
    assert audit["mechanism"] in {
        "rdkit_torsion_constraint",
        "project_relax",
    }
    assert audit["status"] in {"converged", "iteration_limit"}
    bond_audit = audit["bonds"][0]
    assert bond_audit["final_deg"] is not None
    assert abs(bond_audit["delta_deg"]) <= module.PRIOR_RELAX_TOLERANCE_DEG
    assert bond_audit["satisfied"] is True
    assert audit["prior_unsatisfied"] == []


def test_constrained_relaxation_projection_only(monkeypatch):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    monkeypatch.setattr(
        module, "_force_field_for", lambda _molecule: (None, None)
    )
    constraints = [{
        "atom_indices": [0, 1],
        "quartet": [0, 1, 2, 3],
        "lookup_level": "exact",
        "confidence": "high",
        "target_deg": 120.0,
        "circular_std_deg": 5.0,
    }]
    audit = module._constrained_relaxation(molecule, constraints)
    assert audit["mechanism"] == "projection_only"
    assert audit["bonds"][0]["final_deg"] is not None
    assert abs(audit["bonds"][0]["final_deg"] - 120.0) < 1e-6


def test_materialize_integrates_prior_guidance_audit(tmp_path, monkeypatch):
    graph = _graph("AG", cyclization="linear")
    monkeypatch.setattr(
        module, "load_torsion_prior", lambda _path: _stub_prior()
    )
    result = module.materialize_mol2_ensemble(
        graph,
        tmp_path / "mol2",
        ensemble_size=1,
        template_strategy="off",
        random_seed=42,
        num_threads=1,
    )
    assert result["ensemble"]["status"] in {"MATERIALIZED", "PARTIAL"}
    attempts = result["attempts"]
    assert attempts[0]["strategy"] == "torsion_prior_guided"
    guidance = attempts[0]["qa"]["prior_guidance"]
    assert guidance["status"] == "applied"
    assert guidance["applied_count"] >= 1
    constraint = guidance["constraint"]
    assert constraint["mechanism"] in {
        "rdkit_torsion_constraint",
        "project_relax",
        "projection_only",
    }
    assert len(constraint["bonds"]) == guidance["applied_count"]
    receipt = result["validated_mol2_artifacts"][0]["receipt_path"]
    import json

    payload = json.loads(
        __import__("pathlib").Path(str(receipt)).read_text(
            encoding="utf-8"
        )
    )
    assert payload["topology_class"] == "linear"
    assert payload["macrocycle_ring_size"] is None


def test_materialize_receipt_carries_derived_topology(tmp_path, monkeypatch):
    graph = _graph()
    result = module.materialize_mol2_ensemble(
        graph,
        tmp_path / "mol2",
        ensemble_size=1,
        template_strategy="off",
        random_seed=42,
        num_threads=1,
    )
    assert result["ensemble"]["status"] in {"MATERIALIZED", "PARTIAL"}
    import json
    from pathlib import Path

    receipt = result["validated_mol2_artifacts"][0]["receipt_path"]
    payload = json.loads(Path(str(receipt)).read_text(encoding="utf-8"))
    assert payload["topology_class"] == "head_to_tail"
    assert payload["macrocycle_ring_size"] == 6
