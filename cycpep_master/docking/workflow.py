"""High-level peptide docking workflow independent of preparation details."""

import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .ligand_pdbqt import (
    pdb_to_ligand_pdbqt,
    smiles_to_ligand_pdbqt,
)
from .receptor_pdbqt import pdb_to_receptor_pdbqt
from .vina import run_vina


def _select_pdb_chain(source, destination, chain_id):
    """Write one PDB chain while retaining only internal explicit bonds."""
    selected = str(chain_id).strip()
    if len(selected) != 1:
        return "chain ID must contain exactly one character"
    try:
        lines = Path(source).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        return f"cannot read PDB input: {exc}"
    model_starts = [index for index, line in enumerate(lines) if line.startswith("MODEL")]
    if model_starts:
        first_start = model_starts[0]
        try:
            first_end = next(
                index for index in range(first_start + 1, len(lines))
                if lines[index].startswith("ENDMDL")
            )
        except StopIteration:
            return "selected PDB contains an unterminated first MODEL"
        last_end = max(
            (index for index, line in enumerate(lines) if line.startswith("ENDMDL")),
            default=first_end,
        )
        scoped_lines = lines[:first_start] + lines[first_start + 1:first_end] + lines[last_end + 1:]
    else:
        scoped_lines = lines
    atom_lines = [
        line for line in scoped_lines
        if line.startswith(("ATOM  ", "HETATM"))
        and len(line) > 21
        and line[21] == selected
    ]
    if not atom_lines:
        return f"selected chain {selected!r} contains no atoms"
    try:
        serial_values = [int(line[6:11]) for line in atom_lines]
    except ValueError:
        return "selected chain contains a malformed atom serial"
    if len(serial_values) != len(set(serial_values)):
        return "selected chain contains duplicate atom serials"
    identities = [
        (line[21], line[22:26], line[26:27], line[12:16].strip())
        for line in atom_lines
    ]
    if len(identities) != len(set(identities)):
        return "selected chain contains duplicate atom identities"
    serials = set(serial_values)
    output = []
    for line in scoped_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            if len(line) > 21 and line[21] == selected:
                output.append(line)
            continue
        if line.startswith("CONECT"):
            try:
                numbers = [int(token) for token in line.split()[1:]]
            except ValueError:
                return "selected PDB contains a malformed CONECT record"
            if len(numbers) < 2:
                return "selected PDB contains a malformed CONECT record"
            source_selected = numbers[0] in serials
            selected_targets = [value for value in numbers[1:] if value in serials]
            if source_selected and len(selected_targets) != len(numbers) - 1:
                return (
                    "not_supported: selected chain participates in an "
                    "inter-chain CONECT bond"
                )
            if not source_selected and selected_targets:
                return (
                    "not_supported: selected chain participates in an "
                    "inter-chain CONECT bond"
                )
            if source_selected:
                output.append(
                    f"CONECT{numbers[0]:5d}"
                    + "".join(f"{value:5d}" for value in selected_targets)
                )
            continue
        if line.startswith("LINK") and len(line) >= 52:
            endpoints = {line[21:22], line[51:52]}
            if selected in endpoints and endpoints != {selected}:
                return "not_supported: selected chain participates in an inter-chain LINK"
            if endpoints == {selected}:
                output.append(line)
            continue
        if line.startswith("SSBOND") and len(line) >= 30:
            endpoints = {line[15:16], line[29:30]}
            if selected in endpoints and endpoints != {selected}:
                return "not_supported: selected chain participates in an inter-chain SSBOND"
            if endpoints == {selected}:
                output.append(line)
    output.extend(("TER", "END"))
    try:
        Path(destination).write_text(
            "\n".join(output) + "\n", encoding="ascii", errors="strict"
        )
    except (OSError, UnicodeError) as exc:
        return f"cannot write selected-chain PDB: {exc}"
    return None


def _nonempty_file(path):
    candidate = Path(path)
    return candidate.is_file() and candidate.stat().st_size > 0


def _default_convert_pdb(source, destination, *, is_receptor=False):
    if is_receptor:
        return pdb_to_receptor_pdbqt(source, destination)
    return pdb_to_ligand_pdbqt(source, destination)


def _reconstruction_audit(result):
    """Return a serializable audit subset for a unified reconstruction result."""
    if is_dataclass(result) and not isinstance(result, type):
        payload = asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {
            name: getattr(result, name, None)
            for name in (
                "status", "quality", "result_origin", "source_kind", "mode",
                "smiles", "graph", "ambiguous", "warnings", "warning_codes",
                "alternatives", "structure_profile", "provenance",
                "strict_result",
            )
        }
    return {
        key: payload.get(key)
        for key in (
            "status", "quality", "result_origin", "source_kind", "mode",
            "smiles", "graph", "ambiguous", "warnings", "warning_codes",
            "alternatives", "structure_profile", "provenance",
            "strict_result",
        )
        if key in payload
    }


def _dock_peptide_impl(
    peptide_pdb: str,
    receptor_pdb: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    output_dir: Optional[str] = None,
    cleanup: bool = True,
    *,
    ligand_preparation_mode: str = "auto",
    ligand_smiles: Optional[str] = None,
    peptide_chain_id: str = "L",
    receptor_chain_id: Optional[str] = None,
    generated_map: Optional[str] = None,
    torsdof_limit: Optional[int] = None,
    torsion_ensemble_size: int = 8,
    torsion_num_threads: int = 1,
    torsion_audit_out: Optional[dict] = None,
    reconstruction_mode: str = "auto",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    convert_pdb: Callable = _default_convert_pdb,
    prepare_smiles: Callable = smiles_to_ligand_pdbqt,
    execute_vina: Callable = run_vina,
    reconstruct: Optional[Callable] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Execute docking with explicit, injectable component dependencies."""
    if torsion_audit_out is not None:
        torsion_audit_out.clear()
    allowed_modes = {"original_pdb", "audited_smiles", "auto"}
    if ligand_preparation_mode not in allowed_modes:
        return None, (
            "Invalid ligand_preparation_mode: expected one of "
            f"{sorted(allowed_modes)}, got {ligand_preparation_mode!r}"
        )
    if ligand_preparation_mode == "original_pdb" and any(
        value is not None
        for value in (ligand_smiles, generated_map, torsdof_limit)
    ):
        return None, (
            "SMILES, MAP, and torsion-budget options require "
            "ligand_preparation_mode='audited_smiles'"
        )

    reconstruction_audit = None
    if ligand_preparation_mode == "auto":
        if ligand_smiles is not None:
            ligand_preparation_mode = "audited_smiles"
        else:
            try:
                from ..reconstruction import reconstruct_structure

                reconstruction = reconstruct_structure(
                    peptide_pdb,
                    chain_id=peptide_chain_id,
                    mode=reconstruction_mode,
                    minimum_macrocycle_ring_size=int(
                        minimum_macrocycle_ring_size
                    ),
                    require_empty_persistent_overlay=bool(
                        require_empty_persistent_overlay
                    ),
                )
                reconstruction_audit = _reconstruction_audit(reconstruction)
            except Exception as exc:
                reconstruction_audit = {
                    "status": "failed",
                    "quality": None,
                    "result_origin": "unified_reconstruction",
                    "warning_codes": ["UNIFIED_INTERNAL_ERROR"],
                    "warnings": [],
                    "provenance": {
                        "error": f"{type(exc).__name__}: {exc}"
                    },
                }
                reconstruction = None
            prepared_smiles = (
                getattr(reconstruction, "smiles", None)
                if reconstruction is not None
                else None
            )
            if (
                reconstruction is None
                or getattr(reconstruction, "status", None) != "success"
                or not prepared_smiles
            ):
                if torsion_audit_out is not None:
                    torsion_audit_out.update({
                        "status": "not_supported",
                        "preparation_mode": "auto",
                        "chemical_graph_source": "unified_reconstruction",
                        "reconstruction": reconstruction_audit,
                        "error": (
                            "unified reconstruction did not provide a "
                            "serializable SMILES payload"
                        ),
                    })
                return None, (
                    "not_supported: unified reconstruction did not provide a "
                    "serializable SMILES payload"
                )
            ligand_smiles = prepared_smiles
            ligand_preparation_mode = "audited_smiles"

    use_temp = output_dir is None
    if use_temp:
        output_dir = tempfile.mkdtemp(prefix="vina_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    ligand_pdbqt = os.path.join(output_dir, "ligand.pdbqt")
    receptor_pdbqt = os.path.join(output_dir, "receptor.pdbqt")
    output_pdbqt = os.path.join(output_dir, "output.pdbqt")

    try:
        if ligand_preparation_mode == "original_pdb":
            selected_ligand = os.path.join(output_dir, "ligand.selected.pdb")
            error = _select_pdb_chain(
                peptide_pdb, selected_ligand, peptide_chain_id
            )
            if error:
                return None, f"Ligand chain selection failed: {error}"
            error = convert_pdb(
                selected_ligand, ligand_pdbqt, is_receptor=False
            )
            if torsion_audit_out is not None:
                preparation_status = "success"
                if error:
                    preparation_status = (
                        "not_supported"
                        if str(error).startswith("not_supported:")
                        else "failed"
                    )
                torsion_audit_out.update({
                    "status": preparation_status,
                    "preparation_mode": "original_pdb",
                    "coordinate_source": "input_pdb",
                    "original_coordinates_preserved": True,
                })
            if error:
                return None, f"Ligand conversion failed: {error}"
            if not _nonempty_file(ligand_pdbqt):
                if torsion_audit_out is not None:
                    torsion_audit_out.update({
                        "status": "failed",
                        "error": "no nonempty PDBQT output",
                    })
                return None, "Ligand conversion failed: no nonempty PDBQT output"
        else:
            chemical_graph_source = (
                "unified_reconstruction"
                if reconstruction_audit is not None
                else "provided_smiles"
            )
            prepared_smiles = ligand_smiles
            if prepared_smiles is None:
                if reconstruct is None:
                    from ..remediation_v6 import (
                        reconstruct_structure_fail_closed_v6,
                    )

                    reconstruct = reconstruct_structure_fail_closed_v6
                reconstruction = reconstruct(peptide_pdb, peptide_chain_id)
                reconstruction_audit = {
                    "status": reconstruction.status,
                    "support_status": reconstruction.support_status,
                    "qualified_success": reconstruction.qualified_success,
                    "output_inchikey": reconstruction.output_inchikey,
                    "path_used": reconstruction.path_used,
                    "repair_codes": list(reconstruction.repair_codes),
                    "warning_codes": list(reconstruction.warning_codes),
                    "rejection_reason": reconstruction.rejection_reason,
                }
                if (
                    not reconstruction.qualified_success
                    or not reconstruction.output_smiles
                ):
                    if torsion_audit_out is not None:
                        reconstruction_status = str(reconstruction.status)
                        if reconstruction_status not in {
                            "rejected", "failed", "not_supported", "timeout"
                        }:
                            reconstruction_status = "rejected"
                        torsion_audit_out.update({
                            "status": reconstruction_status,
                            "preparation_mode": "audited_smiles",
                            "chemical_graph_source": "v6_reconstruction",
                            "v6_reconstruction": reconstruction_audit,
                        })
                    return None, (
                        "Ligand V6 reconstruction was not a qualified success: "
                        f"status={reconstruction.status}; "
                        "reason="
                        f"{reconstruction.rejection_reason or 'unspecified'}"
                    )
                prepared_smiles = reconstruction.output_smiles
                chemical_graph_source = "v6_reconstruction"

            preparation_audit = {}
            error = prepare_smiles(
                prepared_smiles,
                ligand_pdbqt,
                generated_map=generated_map,
                conf_out=preparation_audit,
                torsdof_limit=torsdof_limit,
                torsion_ensemble_size=torsion_ensemble_size,
                torsion_num_threads=torsion_num_threads,
            )
            if torsion_audit_out is not None:
                preparation_status = "success"
                if error:
                    preparation_status = (
                        "not_supported"
                        if "not_supported:" in str(error).lower()
                        else "failed"
                    )
                torsion_audit_out.update({
                    "status": preparation_status,
                    "preparation_mode": "audited_smiles",
                    "chemical_graph_source": chemical_graph_source,
                    "original_coordinates_preserved": False,
                    "generated_map_supplied": generated_map is not None,
                    "torsion_budget": preparation_audit.get("torsion_budget"),
                    "pdbqt_tree": preparation_audit.get("pdbqt_tree"),
                    "conformer_source": preparation_audit.get("source"),
                })
                if reconstruction_audit is not None:
                    audit_key = (
                        "v6_reconstruction"
                        if "qualified_success" in reconstruction_audit
                        else "reconstruction"
                    )
                    torsion_audit_out[audit_key] = reconstruction_audit
                if error:
                    torsion_audit_out["error"] = error
            if error:
                return None, (
                    f"Ligand audited-SMILES conversion failed: {error}"
                )
            if not _nonempty_file(ligand_pdbqt):
                if torsion_audit_out is not None:
                    torsion_audit_out.update({
                        "status": "failed",
                        "error": "no nonempty PDBQT output",
                    })
                return None, (
                    "Ligand audited-SMILES conversion failed: "
                    "no nonempty PDBQT output"
                )

        receptor_input = receptor_pdb
        if receptor_chain_id is not None:
            receptor_input = os.path.join(output_dir, "receptor.selected.pdb")
            error = _select_pdb_chain(
                receptor_pdb, receptor_input, receptor_chain_id
            )
            if error:
                return None, f"Receptor chain selection failed: {error}"
        error = convert_pdb(
            receptor_input, receptor_pdbqt, is_receptor=True
        )
        if error:
            return None, f"Receptor conversion failed: {error}"
        if not _nonempty_file(receptor_pdbqt):
            return None, "Receptor conversion failed: no nonempty PDBQT output"

        affinity, error = execute_vina(
            ligand_pdbqt=ligand_pdbqt,
            receptor_pdbqt=receptor_pdbqt,
            center=center,
            box_size=box_size,
            output_pdbqt=output_pdbqt,
        )
        if error:
            return None, error
        if affinity is None:
            return None, "Vina returned no affinity without an error"
        return affinity, None
    finally:
        if cleanup and use_temp:
            shutil.rmtree(output_dir, ignore_errors=True)


def dock_peptide(
    peptide_pdb: str,
    receptor_pdb: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    output_dir: Optional[str] = None,
    cleanup: bool = True,
    *,
    ligand_preparation_mode: str = "auto",
    ligand_smiles: Optional[str] = None,
    peptide_chain_id: str = "L",
    receptor_chain_id: Optional[str] = None,
    generated_map: Optional[str] = None,
    torsdof_limit: Optional[int] = None,
    torsion_ensemble_size: int = 8,
    torsion_num_threads: int = 1,
    torsion_audit_out: Optional[dict] = None,
    reconstruction_mode: str = "auto",
    minimum_macrocycle_ring_size: int = 8,
    require_empty_persistent_overlay: bool = False,
    monomer_context=None,
) -> Tuple[Optional[float], Optional[str]]:
    """Dock a peptide through the decomposed canonical component stack."""
    from ..core.monomer_resolution import needs_monomer_resolution_scope

    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            (
                monomer_context
                if monomer_context is not None
                else {"include_persistent_user": True}
            ),
            required_symbols=monomer_symbol_hints(
                peptide_pdb, kind="coordinate"
            ),
        ):
            return dock_peptide(
                peptide_pdb,
                receptor_pdb,
                center,
                box_size,
                output_dir,
                cleanup,
                ligand_preparation_mode=ligand_preparation_mode,
                ligand_smiles=ligand_smiles,
                peptide_chain_id=peptide_chain_id,
                receptor_chain_id=receptor_chain_id,
                generated_map=generated_map,
                torsdof_limit=torsdof_limit,
                torsion_ensemble_size=torsion_ensemble_size,
                torsion_num_threads=torsion_num_threads,
                torsion_audit_out=torsion_audit_out,
                reconstruction_mode=reconstruction_mode,
                minimum_macrocycle_ring_size=(
                    minimum_macrocycle_ring_size
                ),
                require_empty_persistent_overlay=(
                    require_empty_persistent_overlay
                ),
            )
    return _dock_peptide_impl(
        peptide_pdb,
        receptor_pdb,
        center,
        box_size,
        output_dir,
        cleanup,
        ligand_preparation_mode=ligand_preparation_mode,
        ligand_smiles=ligand_smiles,
        peptide_chain_id=peptide_chain_id,
        receptor_chain_id=receptor_chain_id,
        generated_map=generated_map,
        torsdof_limit=torsdof_limit,
        torsion_ensemble_size=torsion_ensemble_size,
        torsion_num_threads=torsion_num_threads,
        torsion_audit_out=torsion_audit_out,
        reconstruction_mode=reconstruction_mode,
        minimum_macrocycle_ring_size=minimum_macrocycle_ring_size,
        require_empty_persistent_overlay=require_empty_persistent_overlay,
    )


def batch_dock_peptides(
    peptide_pdb_list: List[str],
    receptor_pdb: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    *,
    dock: Callable = dock_peptide,
    monomer_context=None,
) -> List[Tuple[str, Optional[float], Optional[str]]]:
    """Dock multiple peptides to one receptor in deterministic list order."""
    from ..core.monomer_resolution import needs_monomer_resolution_scope

    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            (
                monomer_context
                if monomer_context is not None
                else {"include_persistent_user": True}
            ),
            required_symbols=monomer_symbol_hints(
                peptide_pdb_list, kind="coordinate"
            ),
        ):
            return batch_dock_peptides(
                peptide_pdb_list,
                receptor_pdb,
                center,
                box_size,
                dock=dock,
            )
    results = []
    for peptide_pdb in peptide_pdb_list:
        name = Path(peptide_pdb).stem
        try:
            affinity, error = dock(
                peptide_pdb, receptor_pdb, center, box_size
            )
        except Exception as exc:
            affinity = None
            error = f"failed: {type(exc).__name__}: {exc}"
        results.append((name, affinity, error))
    return results
