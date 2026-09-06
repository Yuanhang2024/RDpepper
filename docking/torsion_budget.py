"""Adaptive Meeko torsion budgeting with explicit invariants and audit data."""

import math

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def _circular_sigma_degrees(angles):
    cosine = sum(math.cos(angle) for angle in angles) / len(angles)
    sine = sum(math.sin(angle) for angle in angles) / len(angles)
    resultant = min(1.0, math.hypot(cosine, sine))
    return (
        math.degrees(math.sqrt(-2.0 * math.log(resultant)))
        if resultant > 1e-9
        else float("inf")
    )


def bond_dihedral_sigma(
    mol_3d,
    bond_atom_pairs,
    n_confs=8,
    random_seed=42,
    num_threads=1,
    metadata_out=None,
):
    """Rank RDKit-indexed bonds by circular dihedral dispersion."""
    if not isinstance(n_confs, int) or isinstance(n_confs, bool) or n_confs < 2:
        raise ValueError("n_confs must be an integer >= 2")
    if (
        not isinstance(num_threads, int)
        or isinstance(num_threads, bool)
        or num_threads < 1
    ):
        raise ValueError("num_threads must be an integer >= 1")

    probe = Chem.Mol(mol_3d)
    supplied_source = (
        probe.GetProp("CYCPEP_TORSION_ENSEMBLE_SOURCE")
        if probe.HasProp("CYCPEP_TORSION_ENSEMBLE_SOURCE")
        else None
    )
    if supplied_source:
        conformer_ids = [
            conformer.GetId() for conformer in probe.GetConformers()
        ]
        if len(conformer_ids) < 2:
            raise RuntimeError(
                "supplied torsion reference ensemble has fewer than two "
                "conformers"
            )
        evidence_level = (
            probe.GetProp("CYCPEP_TORSION_EVIDENCE_LEVEL")
            if probe.HasProp("CYCPEP_TORSION_EVIDENCE_LEVEL")
            else "unclassified"
        )
        sigma_method = (
            "matched-template circular standard deviation"
            if supplied_source == "matched_templates"
            else "provided-ensemble circular standard deviation"
        )
    else:
        conformer_ids = [
            conformer.GetId() for conformer in probe.GetConformers()
        ]
        if len(conformer_ids) < 2:
            raise RuntimeError(
                "validated torsion reference ensemble with at least two "
                "conformers is required; torsion_budget does not embed"
            )
        supplied_source = "provided_unclassified_ensemble"
        evidence_level = "unclassified"
        sigma_method = "provided-ensemble circular standard deviation"
    if metadata_out is not None:
        metadata_out.update({
            "requested_conformer_count": n_confs,
            "provided_conformer_count": len(conformer_ids),
            "embedded_conformer_count": len(conformer_ids),
            "embedding_performed": False,
            "random_seed": int(random_seed),
            "num_threads": num_threads,
            "source": supplied_source,
            "evidence_level": evidence_level,
            "sigma_method": sigma_method,
        })
        if probe.HasProp("CYCPEP_TORSION_TEMPLATE_KEYS"):
            metadata_out["template_keys"] = [
                value
                for value in probe.GetProp(
                    "CYCPEP_TORSION_TEMPLATE_KEYS"
                ).split("|")
                if value
            ]

    sigma = {}
    quartet_audit = []
    for left, right in bond_atom_pairs:
        bond_audit = {
            "atom_indices": [left, right],
            "quartet_count": 0,
            "quartets": [],
            "assessable": False,
            "reason": None,
        }
        if (
            not isinstance(left, int)
            or not isinstance(right, int)
            or isinstance(left, bool)
            or isinstance(right, bool)
            or left == right
            or left < 0
            or right < 0
            or left >= probe.GetNumAtoms()
            or right >= probe.GetNumAtoms()
        ):
            bond_audit["reason"] = "invalid_atom_index_pair"
            quartet_audit.append(bond_audit)
            if metadata_out is not None:
                metadata_out["bond_quartet_audit"] = quartet_audit
            raise ValueError(
                "rejected: torsion reference contains an invalid atom "
                f"index pair {left}-{right}"
            )
        molecular_bond = probe.GetBondBetweenAtoms(left, right)
        if molecular_bond is None:
            bond_audit["reason"] = "not_a_molecular_bond"
            quartet_audit.append(bond_audit)
            if metadata_out is not None:
                metadata_out["bond_quartet_audit"] = quartet_audit
            raise ValueError(
                "rejected: Meeko torsion pair is not a molecular bond: "
                f"{left}-{right}"
            )
        if molecular_bond.IsInRing():
            sigma[(left, right)] = float("inf")
            bond_audit["reason"] = "ring_bond_not_freezable"
            quartet_audit.append(bond_audit)
            continue
        left_neighbors = sorted(
            (
                atom.GetIdx()
                for atom in probe.GetAtomWithIdx(left).GetNeighbors()
                if atom.GetIdx() != right
            ),
            key=lambda index: (
                probe.GetAtomWithIdx(index).GetAtomicNum() == 1,
                index,
            ),
        )
        right_neighbors = sorted(
            (
                atom.GetIdx()
                for atom in probe.GetAtomWithIdx(right).GetNeighbors()
                if atom.GetIdx() != left
            ),
            key=lambda index: (
                probe.GetAtomWithIdx(index).GetAtomicNum() == 1,
                index,
            ),
        )
        bond = (left, right)
        quartets = [
            (first, left, right, fourth)
            for first in left_neighbors
            for fourth in right_neighbors
            if first != fourth
        ]
        bond_audit["quartet_count"] = len(quartets)
        if not quartets or len(conformer_ids) < 2:
            sigma[bond] = float("inf")
            bond_audit["reason"] = (
                "no_legal_dihedral_quartet"
                if not quartets
                else "fewer_than_two_conformers"
            )
            quartet_audit.append(bond_audit)
            continue

        quartet_sigmas = []
        all_evaluable = True
        for quartet in quartets:
            angles = []
            for conformer_id in conformer_ids:
                conformer = probe.GetConformer(conformer_id)
                try:
                    angle = float(
                        rdMolTransforms.GetDihedralDeg(
                            conformer, *quartet
                        )
                    )
                except Exception:
                    all_evaluable = False
                    break
                if not math.isfinite(angle):
                    all_evaluable = False
                    break
                angles.append(math.radians(angle))
            if not all_evaluable or len(angles) != len(conformer_ids):
                break
            quartet_sigma = _circular_sigma_degrees(angles)
            quartet_sigmas.append(quartet_sigma)
            bond_audit["quartets"].append({
                "atom_indices": list(quartet),
                "sigma_deg": (
                    quartet_sigma
                    if math.isfinite(quartet_sigma)
                    else None
                ),
            })
        if (
            not all_evaluable
            or len(quartet_sigmas) != len(quartets)
            or not all(math.isfinite(value) for value in quartet_sigmas)
        ):
            sigma[bond] = float("inf")
            bond_audit["reason"] = "incomplete_quartet_evidence"
            quartet_audit.append(bond_audit)
            continue
        spread = max(quartet_sigmas) - min(quartet_sigmas)
        bond_audit["sigma_spread_deg"] = spread
        bond_audit["aggregation"] = "maximum_quartet_sigma"
        sigma[bond] = max(quartet_sigmas)
        bond_audit["assessable"] = True
        bond_audit["reason"] = None
        bond_audit["sigma_deg"] = sigma[bond]
        quartet_audit.append(bond_audit)
    if metadata_out is not None:
        metadata_out["bond_quartet_audit"] = quartet_audit
        metadata_out["quartet_sigma_aggregation"] = (
            "maximum_quartet_sigma"
        )
    return sigma


def actual_rotatable_bonds(setup):
    """Return unique RDKit atom-index bonds represented by Meeko tree edges."""
    connectivity = setup.flexibility_model.get("rigid_body_connectivity", {})
    return sorted({tuple(sorted(pair)) for pair in connectivity.values()})


def setup_atom_snapshot(setup):
    """Capture atom fields that flexibility rebuilding must not change."""
    return {
        int(atom.index): {
            "atomic_num": int(atom.atomic_num),
            "atom_type": str(atom.atom_type),
            "charge": float(atom.charge),
            "coord": tuple(float(value) for value in atom.coord),
            "is_ignore": bool(atom.is_ignore),
        }
        for atom in setup.atoms
    }


def validate_setup_atom_invariants(
    before,
    setup,
    tolerance=1e-12,
    *,
    snapshot=setup_atom_snapshot,
):
    """Fail closed if flexibility rebuilding changes atom properties."""
    after = snapshot(setup)
    if set(before) != set(after):
        raise RuntimeError("torsion budgeting changed the Meeko atom index set")

    max_coordinate_delta = 0.0
    max_charge_delta = 0.0
    for atom_index in sorted(before):
        prior = before[atom_index]
        current = after[atom_index]
        for field in ("atomic_num", "atom_type", "is_ignore"):
            if prior[field] != current[field]:
                raise RuntimeError(
                    f"torsion budgeting changed atom {atom_index} field {field}"
                )
        charge_delta = abs(prior["charge"] - current["charge"])
        coordinate_delta = max(
            abs(left - right)
            for left, right in zip(prior["coord"], current["coord"])
        )
        if not math.isfinite(charge_delta) or not math.isfinite(
            coordinate_delta
        ):
            raise RuntimeError(
                "torsion budgeting produced a non-finite atom invariant at "
                f"{atom_index}"
            )
        max_charge_delta = max(max_charge_delta, charge_delta)
        max_coordinate_delta = max(max_coordinate_delta, coordinate_delta)
        if charge_delta > tolerance or coordinate_delta > tolerance:
            raise RuntimeError(
                f"torsion budgeting changed atom {atom_index} coordinates or charge"
            )
    return {
        "atom_count": len(after),
        "coordinate_tolerance": tolerance,
        "max_abs_coordinate_component_delta": max_coordinate_delta,
        "max_abs_charge_delta": max_charge_delta,
        "atom_type_changes": 0,
        "atomic_number_changes": 0,
        "ignore_flag_changes": 0,
    }


def torsdof_from_setup(setup):
    graph = setup.flexibility_model.get("rigid_body_graph", {})
    return max(0, len(graph) - 1)


def apply_adaptive_torsion_budget(
    setup,
    mol_3d,
    preparator,
    limit=57,
    n_confs=8,
    random_seed=42,
    num_threads=1,
    *,
    list_bonds=actual_rotatable_bonds,
    count_torsdof=torsdof_from_setup,
    calculate_sigma=bond_dihedral_sigma,
):
    """Freeze low-dispersion Meeko bonds and rebuild a valid torsion tree."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("torsdof_limit must be a non-negative integer")
    if not isinstance(n_confs, int) or isinstance(n_confs, bool) or n_confs < 2:
        raise ValueError("torsion_ensemble_size must be an integer >= 2")
    if (
        not isinstance(num_threads, int)
        or isinstance(num_threads, bool)
        or num_threads < 1
    ):
        raise ValueError("torsion_num_threads must be an integer >= 1")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")

    initial_bonds = list_bonds(setup)
    initial_torsdof = count_torsdof(setup)
    if initial_torsdof != len(initial_bonds):
        raise RuntimeError(
            "Meeko torsion model is internally inconsistent before budgeting: "
            f"torsdof={initial_torsdof}, bonds={len(initial_bonds)}"
        )
    audit = {
        "status": "not_needed" if initial_torsdof <= limit else "pending",
        "limit": limit,
        "initial_torsdof": initial_torsdof,
        "final_torsdof": initial_torsdof,
        "initial_rotatable_bond_count": len(initial_bonds),
        "final_rotatable_bond_count": len(initial_bonds),
        "frozen_bonds": [],
        "implicitly_rigidified_bonds": [],
        "retained_bonds": [list(bond) for bond in initial_bonds],
        "bond_sigma_deg": [],
        "sigma_method": "not_evaluated",
        "rigidity_reference_source": "not_evaluated",
        "rigidity_evidence_level": "not_evaluated",
    }
    if initial_torsdof <= limit:
        return setup, audit

    ensemble = {}
    sigma = calculate_sigma(
        mol_3d,
        initial_bonds,
        n_confs=n_confs,
        random_seed=random_seed,
        num_threads=num_threads,
        metadata_out=ensemble,
    )
    finite_ranked = sorted(
        (
            bond
            for bond in initial_bonds
            if math.isfinite(sigma.get(bond, float("inf")))
        ),
        key=lambda bond: (sigma[bond], bond),
    )
    audit["ensemble"] = ensemble
    audit["sigma_method"] = str(
        ensemble.get(
            "sigma_method", "provided-ensemble circular standard deviation"
        )
    )
    audit["rigidity_reference_source"] = str(
        ensemble.get("source", "provided_unclassified_ensemble")
    )
    audit["rigidity_evidence_level"] = str(
        ensemble.get("evidence_level", "unclassified")
    )
    audit["finite_sigma_bond_count"] = len(finite_ranked)
    audit["not_assessable_bond_count"] = (
        len(initial_bonds) - len(finite_ranked)
    )
    audit["bond_sigma_deg"] = [
        {
            "atom_indices": list(bond),
            "assessable": math.isfinite(
                sigma.get(bond, float("inf"))
            ),
            "sigma_deg": (
                sigma[bond]
                if math.isfinite(sigma.get(bond, float("inf")))
                else None
            ),
        }
        for bond in initial_bonds
    ]
    if ensemble.get("embedded_conformer_count", 0) < 2:
        raise RuntimeError("torsion ensemble produced fewer than two conformers")

    current = initial_torsdof
    for bond in finite_ranked:
        if current <= limit:
            break
        bond_info = setup.bond_info.get(bond)
        if bond_info is None or not bond_info.rotatable:
            continue
        before = current
        bond_info.rotatable = False
        preparator.calc_flex(setup)
        current = count_torsdof(setup)
        audit["frozen_bonds"].append({
            "atom_indices": list(bond),
            "sigma_deg": sigma[bond],
            "torsdof_before": before,
            "torsdof_after": current,
        })

    if current > limit:
        raise RuntimeError(
            "insufficient assessable torsions to satisfy the requested budget: "
            f"initial={initial_torsdof}, final={current}, limit={limit}, "
            f"finite_sigma={len(finite_ranked)}"
        )
    final_bonds = set(list_bonds(setup))
    if current != len(final_bonds):
        raise RuntimeError(
            "Meeko torsion model is internally inconsistent after budgeting: "
            f"torsdof={current}, bonds={len(final_bonds)}"
        )
    explicitly_frozen = {
        tuple(item["atom_indices"]) for item in audit["frozen_bonds"]
    }
    all_rigidified = set(initial_bonds) - final_bonds
    audit["status"] = "applied"
    audit["final_torsdof"] = current
    audit["final_rotatable_bond_count"] = len(final_bonds)
    audit["retained_bonds"] = [list(bond) for bond in sorted(final_bonds)]
    audit["implicitly_rigidified_bonds"] = [
        list(bond)
        for bond in sorted(all_rigidified - explicitly_frozen)
    ]
    return setup, audit


def freeze_lowest_sigma_bonds(*_args, **_kwargs):
    """Retain the disabled legacy string editor as an explicit failure."""
    raise RuntimeError(
        "post-hoc BRANCH deletion is disabled; apply the budget to a Meeko setup"
    )
