"""V5 validated-MOL2 to audited ligand-PDBQT preparation."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from rdkit import Chem

from .ligand_pdbqt import (
    _pdbqt_connectivity_from_molecule,
    _unsupported_ligand_error,
    _writer_text,
)
from .mol2_input import (
    Mol2ValidationError,
    ValidatedMol2,
    load_validated_mol2,
    sha256_path,
    validation_error_status,
)
from .pdbqt_validation import (
    atomic_write_text,
    validate_pdbqt_torsion_tree,
)
from .flexibility import load_validated_flexibility_ensemble
from .torsion_budget import (
    actual_rotatable_bonds,
    bond_dihedral_sigma,
    setup_atom_snapshot,
    torsdof_from_setup,
    validate_setup_atom_invariants,
)
from .torsion_prior import (
    TorsionPriorError,
    build_query_keys,
    load_torsion_prior,
)


FLEXIBILITY_MODES = frozenset({"fast", "balanced", "thorough"})

#: Parent MOL2 qualities whose chemistry is settled enough for a validated
#: ligand PDBQT.  ``specified`` is caller-supplied complete chemistry
#: (C3:S); graph-only evidence tiers (topology/partial/raw/opaque) lack
#: bond orders and formal charges, and a medium bundle carries no single
#: selected identity -- emitting a PDBQT from those would fabricate atom
#: types, partial charges, and rotatable bonds.
PDBQT_ELIGIBLE_PARENT_QUALITIES = frozenset({
    "exact", "high", "candidate", "hypothesis", "specified",
})


def baseline_output_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return Path(str(output) + ".baseline.pdbqt")


def _default_prior_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "torsion_priors"
        / "torsion_priors_runtime.json"
    )


def _clear_output(path: Path) -> str | None:
    try:
        path.unlink(missing_ok=True)
        return None
    except OSError as exc:
        return f"cannot clear prior output {path}: {exc}"


def _heavy_graph_snapshot(molecule: Chem.Mol) -> dict[str, Any]:
    conformer = molecule.GetConformer()
    heavy_indices = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    return {
        "atom_indices": heavy_indices,
        "atomic_numbers": [
            molecule.GetAtomWithIdx(index).GetAtomicNum()
            for index in heavy_indices
        ],
        "formal_charges": [
            molecule.GetAtomWithIdx(index).GetFormalCharge()
            for index in heavy_indices
        ],
        "coordinates": [
            (
                float(conformer.GetAtomPosition(index).x),
                float(conformer.GetAtomPosition(index).y),
                float(conformer.GetAtomPosition(index).z),
            )
            for index in heavy_indices
        ],
        "bonds": sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                str(bond.GetBondType()),
                bool(bond.GetIsAromatic()),
            )
            for bond in molecule.GetBonds()
            if bond.GetBeginAtom().GetAtomicNum() > 1
            and bond.GetEndAtom().GetAtomicNum() > 1
        ),
    }


def _prepare_parent_molecule(
    parent: ValidatedMol2,
) -> tuple[Chem.Mol, dict[str, Any]]:
    molecule = Chem.Mol(parent.molecule)
    before = _heavy_graph_snapshot(molecule)
    hydrogen_count_before = sum(
        atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()
    )
    molecule = Chem.RemoveHs(molecule, sanitize=True)
    molecule = Chem.AddHs(molecule, addCoords=True)
    hydrogen_count_after = sum(
        atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()
    )
    generated_hydrogens = max(
        0, hydrogen_count_after - hydrogen_count_before
    )
    after = _heavy_graph_snapshot(molecule)
    if before != after:
        raise RuntimeError(
            "rejected: hydrogen completion changed the parent MOL2 heavy "
            "graph or coordinates"
        )
    completed_inchikey = Chem.MolToInchiKey(molecule)
    if completed_inchikey != parent.full_inchikey:
        raise RuntimeError(
            "rejected: hydrogen completion changed the parent MOL2 "
            "complete InChIKey"
        )
    return molecule, {
        "protonation_applied": False,
        "protonation_policy": "preserve_validated_parent_formal_state",
        "formal_charge_before": parent.formal_charge,
        "formal_charge_after": int(Chem.GetFormalCharge(molecule)),
        "explicit_hydrogen_count_before": hydrogen_count_before,
        "explicit_hydrogen_count_after": hydrogen_count_after,
        "removed_hydrogen_count": max(
            0, hydrogen_count_before - hydrogen_count_after
        ),
        "generated_hydrogen_count": generated_hydrogens,
        "full_inchikey_before": parent.full_inchikey,
        "full_inchikey_after": completed_inchikey,
        "heavy_atom_invariants_valid": True,
    }


def _serialize_setup(
    setup,
    molecule: Chem.Mol,
    *,
    writer,
    validate_tree,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    text = _writer_text(writer.write_string(setup, add_index_map=True))
    connectivity = _pdbqt_connectivity_from_molecule(molecule, text)
    tree = validate_tree(text, connectivity=connectivity)
    expected_atom_count = sum(
        not atom.is_ignore for atom in setup.atoms
    )
    if tree["atom_count"] != expected_atom_count:
        raise RuntimeError(
            "rejected: PDBQT atom count differs from the Meeko setup"
        )
    mapping = {}
    for line in text.splitlines():
        if not line.startswith("REMARK INDEX MAP "):
            continue
        values = line.removeprefix("REMARK INDEX MAP ").split()
        if len(values) % 2:
            raise RuntimeError("rejected: malformed PDBQT INDEX MAP")
        for index in range(0, len(values), 2):
            source_index = int(values[index]) - 1
            serial = int(values[index + 1])
            if source_index in mapping or serial in mapping.values():
                raise RuntimeError(
                    "rejected: PDBQT INDEX MAP is not one-to-one"
                )
            mapping[source_index] = serial
    pdbqt_coordinates = {}
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        serial = int(line[6:11])
        pdbqt_coordinates[serial] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    parent_conformer = molecule.GetConformer()
    heavy_indices = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    }
    if not heavy_indices.issubset(mapping):
        raise RuntimeError(
            "rejected: PDBQT INDEX MAP omits parent MOL2 heavy atoms"
        )
    maximum_delta = 0.0
    for atom_index in sorted(heavy_indices):
        serial = mapping[atom_index]
        if serial not in pdbqt_coordinates:
            raise RuntimeError(
                "rejected: mapped parent MOL2 atom is absent from PDBQT"
            )
        source = parent_conformer.GetAtomPosition(atom_index)
        emitted = pdbqt_coordinates[serial]
        delta = max(
            abs(float(source.x) - emitted[0]),
            abs(float(source.y) - emitted[1]),
            abs(float(source.z) - emitted[2]),
        )
        maximum_delta = max(maximum_delta, delta)
    if maximum_delta > 0.0011:
        raise RuntimeError(
            "rejected: PDBQT heavy-atom coordinates differ from parent MOL2"
        )
    setup_by_index = {
        int(atom.index): atom
        for atom in setup.atoms
        if not atom.is_ignore
    }
    if not heavy_indices.issubset(setup_by_index):
        raise RuntimeError(
            "rejected: Meeko setup omits parent MOL2 heavy atoms"
        )
    for atom_index in heavy_indices:
        setup_atom = setup_by_index[atom_index]
        parent_atom = molecule.GetAtomWithIdx(atom_index)
        if int(setup_atom.atomic_num) != parent_atom.GetAtomicNum():
            raise RuntimeError(
                "rejected: Meeko atomic number differs from parent MOL2"
            )
        if (
            not math.isfinite(float(setup_atom.charge))
            or not str(setup_atom.atom_type)
        ):
            raise RuntimeError(
                "rejected: Meeko produced invalid charge or atom type"
            )
    parent_audit = {
        "index_map_complete_for_heavy_atoms": True,
        "parent_heavy_atom_count": len(heavy_indices),
        "maximum_parent_coordinate_component_delta_angstrom": (
            maximum_delta
        ),
        "meeko_atom_types_valid": True,
        "meeko_partial_charges_finite": True,
        "meeko_total_partial_charge": sum(
            float(atom.charge)
            for atom in setup.atoms
            if not atom.is_ignore
        ),
    }
    return text, tree, parent_audit


def _freeze_ranked_bonds(
    setup,
    preparator,
    ranked_bonds: list[tuple[tuple[int, int], dict[str, Any]]],
    *,
    limit: int,
    phase: str,
) -> list[dict[str, Any]]:
    frozen = []
    for bond, evidence in ranked_bonds:
        current = torsdof_from_setup(setup)
        if current <= limit:
            break
        info = setup.bond_info.get(bond)
        if info is None:
            info = setup.bond_info.get((bond[1], bond[0]))
        if info is None or not info.rotatable:
            continue
        before = current
        info.rotatable = False
        preparator.calc_flex(setup)
        after = torsdof_from_setup(setup)
        frozen.append({
            "atom_indices": list(bond),
            "phase": phase,
            "torsdof_before": before,
            "torsdof_after": after,
            **evidence,
        })
    return frozen


def mol2_to_ligand_pdbqt(
    mol2_path: str | Path | ValidatedMol2,
    output_path: str | Path,
    *,
    torsdof_limit: int | None = None,
    flexibility_mode: str = "balanced",
    torsion_prior_path: str | Path | None = None,
    ensemble_manifest_path: str | Path | None = None,
    ensemble_manifest_sha256: str | None = None,
    ensemble_size: int = 4,
    random_seed: int = 42,
    num_threads: int = 1,
    strict_budget: bool = False,
    receipt_path: str | Path | None = None,
    load_parent=load_validated_mol2,
    load_prior=load_torsion_prior,
    embed_multiple=None,
) -> tuple[dict[str, Any] | None, str | None]:
    output = Path(output_path)
    baseline_path = baseline_output_path(output)
    if flexibility_mode not in FLEXIBILITY_MODES:
        return None, (
            "invalid_input: flexibility_mode must be fast, balanced, or "
            "thorough"
        )
    if (
        torsdof_limit is not None
        and (
            not isinstance(torsdof_limit, int)
            or isinstance(torsdof_limit, bool)
            or torsdof_limit < 0
        )
    ):
        return None, "invalid_input: torsdof_limit must be non-negative"
    if (
        not isinstance(num_threads, int)
        or isinstance(num_threads, bool)
        or num_threads < 1
    ):
        return None, "invalid_input: num_threads must be at least one"
    audit: dict[str, Any] | None = None
    baseline_text: str | None = None
    baseline_tree: dict[str, Any] | None = None
    try:
        parent = (
            mol2_path
            if isinstance(mol2_path, ValidatedMol2)
            else load_parent(mol2_path, receipt_path=receipt_path)
        )
    except Mol2ValidationError as exc:
        status = validation_error_status(exc)
        return None, f"{status}: validated MOL2 required: {exc}"
    except Exception as exc:
        return None, f"failed: validated MOL2 loader failed: {exc}"
    if parent.quality not in PDBQT_ELIGIBLE_PARENT_QUALITIES:
        # PDBQT fixes atom types, partial charges, and rotatable bonds:
        # generating it from unresolved chemistry would fabricate exactly
        # the evidence the parent lacks.  The format is honestly
        # unavailable; the caller degrades to the parent artifact.
        return None, (
            "not_supported: PDBQT requires settled chemistry; parent MOL2 "
            f"quality {parent.quality!r} is below the eligibility floor "
            f"{sorted(PDBQT_ELIGIBLE_PARENT_QUALITIES)}"
        )
    if output.resolve() in {
        parent.path.resolve(),
        parent.receipt_path.resolve(),
    }:
        return None, "invalid_input: PDBQT output aliases parent MOL2 input"
    for stale in (output, baseline_path):
        error = _clear_output(stale)
        if error:
            return None, f"failed: {error}"
    try:
        requested_mode = flexibility_mode
        effective_mode = requested_mode
        validated_ensemble = None
        ensemble_audit = None
        if requested_mode in {"balanced", "thorough"}:
            if ensemble_manifest_path is not None:
                validated_ensemble, ensemble_audit = (
                    load_validated_flexibility_ensemble(
                        parent,
                        ensemble_manifest_path,
                        expected_manifest_sha256=(
                            ensemble_manifest_sha256
                        ),
                    )
                )
                if (
                    requested_mode == "thorough"
                    and validated_ensemble is not None
                    and ensemble_audit.get("produced_count")
                    != ensemble_audit.get("requested_count")
                ):
                    validated_ensemble = None
                    ensemble_audit = {
                        **ensemble_audit,
                        "status": "insufficient",
                        "reason": (
                            "thorough mode requires a complete validated "
                            "ensemble"
                        ),
                    }
            else:
                ensemble_audit = {
                    "status": "unavailable",
                    "reason": "no ensemble manifest supplied",
                }
            if validated_ensemble is None:
                effective_mode = "fast"
        downgrade_code = None
        if (
            requested_mode in {"balanced", "thorough"}
            and effective_mode == "fast"
        ):
            prefix = requested_mode.upper()
            downgrade_code = (
                f"{prefix}_DOWNGRADED_TO_FAST_NO_ENSEMBLE"
                if ensemble_manifest_path is None
                else f"{prefix}_DOWNGRADED_TO_FAST_INVALID_ENSEMBLE"
            )
        molecule, parent_audit = _prepare_parent_molecule(parent)
        unsupported = _unsupported_ligand_error(molecule)
        if unsupported:
            return None, unsupported
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        preparator = MoleculePreparation(rigid_macrocycles=True)
        setups = preparator.prepare(molecule)
        if not setups:
            return None, "not_supported: Meeko produced no ligand setup"
        baseline_setup = setups[0]
        baseline_snapshot = setup_atom_snapshot(baseline_setup)
        initial_torsdof = torsdof_from_setup(baseline_setup)
        baseline_text, baseline_tree, parent_setup_audit = _serialize_setup(
            baseline_setup,
            molecule,
            writer=PDBQTWriterLegacy,
            validate_tree=validate_pdbqt_torsion_tree,
        )
        audit = {
            "parent_mol2_path": str(parent.path),
            "parent_mol2_sha256": parent.sha256,
            "parent_receipt_path": str(parent.receipt_path),
            "parent_receipt_sha256": parent.receipt_sha256,
            "parent_full_inchikey": parent.full_inchikey,
            "parent_coordinate_mode": parent.coordinate_mode,
            "parent_coordinate_level": parent.coordinate_level,
            "parent_mapped_heavy_atom_indices": list(
                parent.receipt.get("mapped_heavy_atom_indices") or []
            ),
            "parent_generated_heavy_atom_indices": list(
                parent.receipt.get("generated_heavy_atom_indices") or []
            ),
            "parent_atom_coordinate_origins": dict(
                parent.receipt.get("atom_coordinate_origins") or {}
            ),
            "inherited_rigor": parent.rigor,
            "inherited_quality": parent.quality,
            **parent_audit,
            **parent_setup_audit,
            "initial_torsdof": initial_torsdof,
            "flexibility_mode": requested_mode,
            "requested_flexibility_mode": requested_mode,
            "effective_flexibility_mode": effective_mode,
            "torsdof_limit": torsdof_limit,
            "lookup_covered_bond_count": 0,
            "lookup_unresolved_bond_count": 0,
            "lookup_matches": [],
            "torsion_prior_path": None,
            "torsion_prior_runtime_sha256": None,
            "torsion_prior_manifest_sha256": None,
            "ensemble_fallback_triggered": False,
            "ensemble_size": (
                validated_ensemble.GetNumConformers()
                if validated_ensemble is not None
                else 1
            ),
            "ensemble": ensemble_audit,
            "ensemble_manifest_path": (
                str(Path(ensemble_manifest_path).resolve())
                if ensemble_manifest_path is not None
                else None
            ),
            "ensemble_manifest_sha256_expected": (
                ensemble_manifest_sha256
            ),
            "embedding_performed": False,
            "flexibility_warning_codes": (
                [downgrade_code] if downgrade_code else []
            ),
            "legacy_generation_parameters_ignored": {
                "ensemble_size": ensemble_size,
                "random_seed": random_seed,
                "num_threads": num_threads,
                "embed_multiple_supplied": embed_multiple is not None,
                "reason": "V5 PDBQT does not generate conformers",
            },
            "frozen_bonds": [],
            "final_torsdof": initial_torsdof,
            "budget_satisfied": (
                torsdof_limit is None
                or initial_torsdof <= torsdof_limit
            ),
            "baseline_pdbqt_path": None,
            "baseline_output_sha256": None,
            "pdbqt_tree_valid": True,
            "atom_invariants_valid": True,
            "baseline_tree": baseline_tree,
            "output_path": None,
            "output_sha256": None,
            "output_role": "baseline",
        }
        if torsdof_limit is None or initial_torsdof <= torsdof_limit:
            atomic_write_text(output, baseline_text)
            audit["output_path"] = str(output)
            audit["output_sha256"] = sha256_path(output)
            return audit, None

        atomic_write_text(baseline_path, baseline_text)
        audit["baseline_pdbqt_path"] = str(baseline_path)
        audit["baseline_output_sha256"] = sha256_path(baseline_path)
        budget_setup = copy.deepcopy(baseline_setup)
        initial_bonds = actual_rotatable_bonds(budget_setup)

        prior = None
        prior_error = None
        runtime_path = (
            Path(torsion_prior_path)
            if torsion_prior_path is not None
            else _default_prior_path()
        )
        try:
            prior = load_prior(runtime_path)
            audit["torsion_prior_path"] = str(runtime_path)
            audit["torsion_prior_runtime_sha256"] = getattr(
                prior, "runtime_sha256", None
            )
            audit["torsion_prior_manifest_sha256"] = getattr(
                prior, "manifest_sha256", None
            )
        except Exception as exc:
            prior_error = f"{type(exc).__name__}: {exc}"
        ranked_lookup = []
        unresolved = []
        for bond in initial_bonds:
            if prior is None:
                unresolved.append(bond)
                audit["lookup_matches"].append({
                    "atom_indices": list(bond),
                    "status": "unavailable",
                    "reason": prior_error or "torsion prior unavailable",
                })
                continue
            try:
                keys = build_query_keys(
                    molecule,
                    bond,
                    topology_class=parent.receipt.get(
                        "topology_class"
                    ),
                    macrocycle_ring_size=parent.receipt.get(
                        "macrocycle_ring_size"
                    ),
                )
                match = prior.query(
                    keys, flexibility_mode=effective_mode
                )
            except (TorsionPriorError, ValueError) as exc:
                unresolved.append(bond)
                audit["lookup_matches"].append({
                    "atom_indices": list(bond),
                    "status": "unavailable",
                    "reason": str(exc),
                })
                continue
            match_row = {
                "atom_indices": list(bond),
                "status": match.status,
                "lookup_level": match.lookup_level,
                "lookup_key": match.key,
                "rigidity_score": match.rigidity_score,
                "confidence": match.confidence,
                "eligible_to_freeze": match.eligible_to_freeze,
                "reason": match.reason,
                "calibration_unit": (
                    (match.statistics or {}).get("calibration_unit")
                ),
                "calibration_false_rigid_count": (
                    (match.statistics or {}).get(
                        "calibration_false_rigid_count"
                    )
                ),
                "calibration_evaluable_count": (
                    (match.statistics or {}).get(
                        "calibration_evaluable_count"
                    )
                ),
                "calibration_false_rigid_ci_high": (
                    (match.statistics or {}).get(
                        "calibration_false_rigid_ci_high"
                    )
                ),
                "leave_one_source_out_false_rigid_count": (
                    (match.statistics or {}).get(
                        "leave_one_source_out_false_rigid_count"
                    )
                ),
                "leave_one_source_out_evaluable_count": (
                    (match.statistics or {}).get(
                        "leave_one_source_out_evaluable_count"
                    )
                ),
                "leave_one_source_out_false_rigid_ci_high": (
                    (match.statistics or {}).get(
                        "leave_one_source_out_false_rigid_ci_high"
                    )
                ),
            }
            audit["lookup_matches"].append(match_row)
            if match.eligible_to_freeze:
                ranked_lookup.append((
                    bond,
                    {
                        "lookup_level": match.lookup_level,
                        "lookup_key": match.key,
                        "rigidity_score": match.rigidity_score,
                        "confidence": match.confidence,
                        "calibration_unit": match_row[
                            "calibration_unit"
                        ],
                        "calibration_false_rigid_count": match_row[
                            "calibration_false_rigid_count"
                        ],
                        "calibration_evaluable_count": match_row[
                            "calibration_evaluable_count"
                        ],
                        "calibration_false_rigid_ci_high": match_row[
                            "calibration_false_rigid_ci_high"
                        ],
                        "leave_one_source_out_false_rigid_count": (
                            match_row[
                                "leave_one_source_out_false_rigid_count"
                            ]
                        ),
                        "leave_one_source_out_evaluable_count": (
                            match_row[
                                "leave_one_source_out_evaluable_count"
                            ]
                        ),
                        "leave_one_source_out_false_rigid_ci_high": (
                            match_row[
                                "leave_one_source_out_false_rigid_ci_high"
                            ]
                        ),
                    },
                ))
            else:
                unresolved.append(bond)
        audit["lookup_covered_bond_count"] = len(ranked_lookup)
        audit["lookup_unresolved_bond_count"] = len(unresolved)
        ranked_lookup.sort(
            key=lambda item: (
                -float(item[1]["rigidity_score"] or 0.0),
                0 if item[1]["confidence"] == "high" else 1,
                item[0],
            )
        )
        audit["frozen_bonds"].extend(
            _freeze_ranked_bonds(
                budget_setup,
                preparator,
                ranked_lookup,
                limit=torsdof_limit,
                phase="lookup",
            )
        )

        current_torsdof = torsdof_from_setup(budget_setup)
        unresolved_current = [
            bond
            for bond in actual_rotatable_bonds(budget_setup)
            if bond in set(unresolved)
        ]
        if (
            current_torsdof > torsdof_limit
            and unresolved_current
            and effective_mode in {"balanced", "thorough"}
            and validated_ensemble is not None
        ):
            unresolved_current = [
                bond
                for bond in actual_rotatable_bonds(budget_setup)
                if bond in set(unresolved)
            ]
            audit["ensemble_fallback_triggered"] = True
            if unresolved_current:
                conformer_count = (
                    validated_ensemble.GetNumConformers()
                )
                sigma = bond_dihedral_sigma(
                    validated_ensemble,
                    unresolved_current,
                    n_confs=conformer_count,
                    random_seed=random_seed,
                    num_threads=num_threads,
                    metadata_out=ensemble_audit,
                )
                ranked_fallback = [
                    (
                        bond,
                        {
                            "sigma_deg": float(sigma[bond]),
                            "confidence": "validated_ensemble",
                        },
                    )
                    for bond in unresolved_current
                    if math.isfinite(sigma.get(bond, float("inf")))
                ]
                ranked_fallback.sort(
                    key=lambda item: (item[1]["sigma_deg"], item[0])
                )
                audit["frozen_bonds"].extend(
                    _freeze_ranked_bonds(
                        budget_setup,
                        preparator,
                        ranked_fallback,
                        limit=torsdof_limit,
                        phase="validated_ensemble_fallback",
                    )
                )

        final_torsdof = torsdof_from_setup(budget_setup)
        audit["final_torsdof"] = final_torsdof
        audit["budget_satisfied"] = final_torsdof <= torsdof_limit
        invariants = validate_setup_atom_invariants(
            baseline_snapshot, budget_setup
        )
        audit["setup_atom_invariants"] = invariants
        audit["atom_invariants_valid"] = True
        if audit["budget_satisfied"]:
            budgeted_text, tree, budgeted_parent_audit = _serialize_setup(
                budget_setup,
                molecule,
                writer=PDBQTWriterLegacy,
                validate_tree=validate_pdbqt_torsion_tree,
            )
            if int(tree["torsdof"]) != final_torsdof:
                raise RuntimeError(
                    "rejected: PDBQT TORSDOF differs from the budget audit"
                )
            atomic_write_text(output, budgeted_text)
            audit["pdbqt_tree"] = tree
            audit["budgeted_parent_invariants"] = budgeted_parent_audit
            audit["output_path"] = str(output)
            audit["output_sha256"] = sha256_path(output)
            audit["output_role"] = "budgeted"
            return audit, None

        if not strict_budget:
            atomic_write_text(output, baseline_text)
            audit["pdbqt_tree"] = baseline_tree
            audit["output_path"] = str(output)
            audit["output_sha256"] = sha256_path(output)
            audit["output_role"] = "baseline_budget_unsatisfied"
            return audit, None
        return audit, (
            "not_supported: strict torsion budget could not be satisfied; "
            "baseline PDBQT was retained"
        )
    except Exception as exc:
        message = str(exc)
        classified = (
            message
            if message.startswith((
                "invalid_input:",
                "not_supported:",
                "rejected:",
                "timeout:",
            ))
            else (
                "failed: MOL2-to-PDBQT preparation failed: "
                + message
            )
        )
        if (
            audit is not None
            and baseline_text is not None
            and baseline_tree is not None
            and audit.get("baseline_pdbqt_path")
            and Path(str(audit["baseline_pdbqt_path"])).is_file()
        ):
            audit["budget_satisfied"] = False
            audit["flexibility_error"] = classified
            if not strict_budget:
                try:
                    atomic_write_text(output, baseline_text)
                    audit["pdbqt_tree"] = baseline_tree
                    audit["output_path"] = str(output)
                    audit["output_sha256"] = sha256_path(output)
                    audit["output_role"] = (
                        "baseline_flexibility_failed"
                    )
                    return audit, None
                except Exception as write_exc:
                    return audit, (
                        "failed: could not commit validated baseline "
                        f"PDBQT after flexibility failure: {write_exc}"
                    )
            return audit, classified
        return audit, classified
