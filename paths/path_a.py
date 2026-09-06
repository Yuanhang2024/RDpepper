"""Path A/E: reconstruct residues from Unified-library-backed templates."""
import copy
import hashlib
import os
import threading

from rdkit import Chem
from rdkit.Geometry import Point3D

from ..core.pdb_parser import (
    get_res_seq,
    get_pdb_atoms,
    parse_backbone,
    read_conect,
)
from ..core.geometry_params import (
    NONDEFAULT_GEOMETRY_PARAMS,
    is_nondefault_geometry,
    resolve_geometry_params,
)
from ..core.monomer_resolution import needs_monomer_resolution_scope
from ..core.molecule import (
    add_to_combo,
    apply_conect,
    apply_geometric_crosslinks,
    materialize_typed_crosslink,
    remove_orphans,
    finish_mol,
)
from ._map_utils import registry_epoch
from .residue_template_factory import (
    cap_atom_name_map,
    get_residue_template,
    map_pdb_atoms,
    map_pdb_atoms_with_evidence,
)


def _has_head_to_tail_evidence(pdb_path, chain_id, n_res, *, allow_geometry,
                               radius_multiplier=None, distance_ceiling=None):
    from ..core.cyclization import detect_cyclization

    info = detect_cyclization(
        pdb_path,
        chain_id,
        allow_geometric_inference=allow_geometry,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    required = {(1, 'R1'), (n_res, 'R2')}
    for bond in info.bonds:
        endpoints = {(bond.pos1, bond.rgroup1), (bond.pos2, bond.rgroup2)}
        if bond.bond_type == 'peptide' and endpoints == required:
            return True
    return False


_EXPLICIT_CLOSURE_SOURCES = frozenset({"ssbond", "link", "conect"})


def _generation_provenance(geometric_cyclization):
    route = "e" if geometric_cyclization else "a"
    # NB: this dict shape is a frozen remediation_v6 contract (checked by
    # exact equality in `_fresh_path_evidence_replay`); it must NOT grow new
    # keys. Resolved geometry parameters are exposed as top-level evidence
    # keys instead (geometry_radius_multiplier / geometry_distance_ceiling /
    # geometry_warnings).
    return {
        "schema_version": "1.0.0-path-a-e-generation-provenance.1",
        "generator": "cycpep_master.paths.path_a.generate_with_evidence",
        "requested_route": route,
        "geometric_cyclization_argument": bool(geometric_cyclization),
        "geometry_stage_executed": bool(geometric_cyclization),
    }


def _geometry_warnings(geometric_cyclization, radius_multiplier=None,
                       distance_ceiling=None):
    if (
        geometric_cyclization
        and is_nondefault_geometry(radius_multiplier, distance_ceiling)
    ):
        return [NONDEFAULT_GEOMETRY_PARAMS]
    return []


def _materialize_explicit_topology_bonds(
    combo,
    pdb2g,
    residues,
    serials_by_position_and_name,
    pdb_path,
    chain_id,
    *,
    radius_multiplier=None,
    distance_ceiling=None,
):
    """Apply uniquely resolved, chemically typed explicit closures."""
    from ..core.cyclization import detect_cyclization

    topology = detect_cyclization(
        pdb_path,
        chain_id,
        allow_geometric_inference=False,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    materialized = []
    for bond in topology.bonds:
        endpoints = []
        for position, atom_name, rgroup in (
            (int(bond.pos1), str(bond.atom1 or "").strip().upper(), bond.rgroup1),
            (int(bond.pos2), str(bond.atom2 or "").strip().upper(), bond.rgroup2),
        ):
            if not atom_name:
                atom_name = {"R1": "N", "R2": "C"}.get(str(rgroup).upper(), "")
            serials = serials_by_position_and_name.get((position, atom_name), [])
            if len(serials) != 1:
                raise ValueError(
                    f"explicit closure endpoint {position}:{atom_name} is not unique"
                )
            serial = serials[0]
            if serial not in pdb2g:
                raise ValueError(
                    f"explicit closure endpoint serial {serial} is not mapped"
                )
            endpoints.append((serial, pdb2g[serial]))
        left, right = endpoints[0][1], endpoints[1][1]
        if left == right:
            raise ValueError("explicit closure resolves both endpoints to one atom")
        added = materialize_typed_crosslink(
            combo, left, right, str(bond.bond_type)
        )
        materialized.append({
            "source": str(bond.evidence_source),
            "bond_type": str(bond.bond_type),
            "pdb_serials": [endpoints[0][0], endpoints[1][0]],
            "graph_indices": [left, right],
            "rgroups": [str(bond.rgroup1), str(bond.rgroup2)],
            "added": added,
        })
    return materialized


def _materialize_free_c_terminal_hydroxyl(
    combo,
    pdb2g,
    residues,
    pdb_path,
    chain_id,
    *,
    has_head_to_tail,
    occupied_ports=(),
):
    """Restore the template-declared R2 cap for a free peptide C terminus."""
    if has_head_to_tail or not residues:
        return None
    terminal_name = str(residues[-1].get("name", "")).strip().upper()
    terminal_position = len(residues)
    occupied = sorted({
        (int(position), str(rgroup).upper())
        for position, rgroup in occupied_ports
    })
    if (terminal_position, "R2") in occupied:
        return {
            "status": "not_materialized_port_occupied",
            "terminal_position": terminal_position,
            "occupied_ports": [list(port) for port in occupied],
        }
    if terminal_name in {"NME", "NH2"}:
        return {
            "status": "explicit_terminal_cap_present",
            "terminal_position": terminal_position,
            "cap_residue": terminal_name,
            "occupied_ports": [list(port) for port in occupied],
        }
    terminal_template = get_residue_template(terminal_name)
    r2_default = str(terminal_template.r2).strip().upper()
    cap_atomic_number = {"OH": 8, "NH2": 7, "SH": 16}.get(r2_default)
    if cap_atomic_number is None and r2_default not in {"", "-", "H"}:
        raise ValueError(f"unsupported terminal R2 default {r2_default!r}")
    terminal_atoms = get_pdb_atoms(pdb_path, residues[-1]["key"], chain_id)
    carbonyl = [
        atom for atom in terminal_atoms
        if str(atom["name"]).strip().upper() == "C" and atom["num"] in pdb2g
    ]
    if len(carbonyl) != 1:
        raise ValueError("free C-terminal carbonyl atom is not uniquely mapped")
    carbon_index = pdb2g[carbonyl[0]["num"]]
    carbon = combo.GetAtomWithIdx(carbon_index)
    single_caps = [
        neighbor for neighbor in carbon.GetNeighbors()
        if neighbor.GetAtomicNum() in {7, 8, 16}
        and combo.GetBondBetweenAtoms(
            carbon_index, neighbor.GetIdx()
        ).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(single_caps) > 1:
        raise ValueError("free C terminus has multiple heteroatom cap candidates")
    if single_caps:
        cap = single_caps[0]
        if cap_atomic_number is not None and cap.GetAtomicNum() != cap_atomic_number:
            raise ValueError(
                f"terminal R2 cap element {cap.GetSymbol()} conflicts with "
                f"template default {r2_default}"
            )
        evidence = {
            "status": "observed_or_template_present",
            "terminal_position": terminal_position,
            "carbon_pdb_serial": carbonyl[0]["num"],
            "carbon_graph_index": carbon_index,
            "cap_graph_index": cap.GetIdx(),
            "cap_element": cap.GetSymbol(),
            "r2_default": r2_default,
            "occupied_ports": [list(port) for port in occupied],
        }
        if cap.GetAtomicNum() == 8:
            evidence["oxygen_graph_index"] = cap.GetIdx()
        return evidence
    if cap_atomic_number is None:
        return {
            "status": "no_materialized_cap_for_R2_default",
            "terminal_position": terminal_position,
            "carbon_pdb_serial": carbonyl[0]["num"],
            "carbon_graph_index": carbon_index,
            "r2_default": r2_default,
            "occupied_ports": [list(port) for port in occupied],
        }
    cap_index = combo.AddAtom(Chem.Atom(cap_atomic_number))
    combo.AddBond(carbon_index, cap_index, Chem.BondType.SINGLE)
    observed_oxt = [
        atom for atom in terminal_atoms
        if str(atom["name"]).strip().upper() == "OXT"
    ]
    evidence = {
        "status": (
            "materialized_from_observed_terminal_oxt"
            if r2_default == "OH" and len(observed_oxt) == 1
            else f"materialized_from_R2_{r2_default}_default"
        ),
        "terminal_position": terminal_position,
        "carbon_pdb_serial": carbonyl[0]["num"],
        "carbon_graph_index": carbon_index,
        "cap_graph_index": cap_index,
        "cap_element": Chem.GetPeriodicTable().GetElementSymbol(cap_atomic_number),
        "r2_default": r2_default,
        "oxygen_pdb_serial": (
            observed_oxt[0]["num"]
            if r2_default == "OH" and len(observed_oxt) == 1 else None
        ),
        "occupied_ports": [list(port) for port in occupied],
    }
    if r2_default == "OH":
        evidence["oxygen_graph_index"] = cap_index
    return evidence


def _molecule_inchikey(molecule):
    candidate = Chem.Mol(molecule)
    candidate.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(candidate)
    key = Chem.MolToInchiKey(candidate)
    if not key:
        raise ValueError("molecule did not yield a full InChIKey")
    return key


def _audit_emergent_stereochemistry_from_coordinates(
    combo, pdb2g, atoms_by_serial, *, preexisting_unassigned=()
):
    """Report newly ambiguous centers without assigning stereo from geometry."""
    assembled = combo.GetMol()
    assembled.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistry(assembled, cleanIt=True, force=True)
    preexisting = {int(index) for index in preexisting_unassigned}
    unassigned = [
        int(index)
        for index, assignment in Chem.FindMolChiralCenters(
            assembled,
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
        if assignment == "?" and int(index) not in preexisting
    ]
    if not unassigned:
        return {
            "passed": True,
            "reason": None,
            "emergent_unassigned_center_count": 0,
            "emergent_unassigned_graph_atom_indices": [],
            "coordinate_observations": [],
            "coordinate_assignment_applied": False,
            "policy": "coordinate_geometry_cannot_assign_stereochemistry",
        }

    serial_by_graph = {int(graph): int(serial) for serial, graph in pdb2g.items()}
    observed = Chem.Mol(assembled)
    Chem.RemoveStereochemistry(observed)
    conformer = Chem.Conformer(observed.GetNumAtoms())
    for graph_index in range(observed.GetNumAtoms()):
        serial = serial_by_graph.get(graph_index)
        source = atoms_by_serial.get(serial) if serial is not None else None
        if source is None or "xyz" not in source:
            continue
        conformer.SetAtomPosition(graph_index, Point3D(*map(float, source["xyz"])))
    observed.RemoveAllConformers()
    observed.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(
        observed, confId=0, replaceExistingTags=True
    )
    Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
    observed_centers = dict(Chem.FindMolChiralCenters(
        observed,
        includeUnassigned=True,
        useLegacyImplementation=False,
    ))

    ledger = []
    for graph_index in unassigned:
        atom = assembled.GetAtomWithIdx(graph_index)
        required = {graph_index, *[neighbor.GetIdx() for neighbor in atom.GetNeighbors()]}
        missing = sorted(index for index in required if index not in serial_by_graph)
        assignment = observed_centers.get(graph_index)
        serial = serial_by_graph.get(graph_index)
        ledger.append({
            "graph_atom_index": graph_index,
            "pdb_serial": serial,
            "pdb_atom_name": (
                str(atoms_by_serial[serial]["name"])
                if serial in atoms_by_serial else None
            ),
            "coordinate_observed_cip": assignment,
            "coordinate_neighborhood_complete": not missing,
            "missing_coordinate_graph_atom_indices": missing,
            "coordinate_neighbor_graph_indices": sorted(required - {graph_index}),
            "evidence_source": "observed_3d_after_explicit_assembly",
            "assignment_applied": False,
        })
    return {
        "passed": False,
        "reason": "emergent_stereochemistry_requires_noncoordinate_authority",
        "emergent_unassigned_center_count": len(unassigned),
        "emergent_unassigned_graph_atom_indices": unassigned,
        "coordinate_observations": ledger,
        "coordinate_assignment_applied": False,
        "policy": "coordinate_geometry_cannot_assign_stereochemistry",
    }


def _assembled_coordinate_stereochemistry_evidence(
    combo,
    pdb2g,
    atoms_by_serial,
):
    """Verify that explicit template stereo survives complete graph assembly."""
    assembled = combo.GetMol()
    assembled.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(assembled)
    Chem.AssignStereochemistry(assembled, cleanIt=True, force=True)
    expected_centers = dict(Chem.FindMolChiralCenters(
        assembled,
        includeUnassigned=True,
        useLegacyImplementation=False,
    ))
    specified_centers = {
        int(index): str(assignment)
        for index, assignment in expected_centers.items()
        if assignment != "?"
    }
    if not specified_centers:
        return {
            "passed": True,
            "reason": None,
            "specified_center_count": 0,
            "matched_center_count": 0,
            "mismatch_graph_atom_indices": [],
            "evidence_source": "observed_3d_after_complete_assembly",
        }

    serial_by_graph = {}
    duplicate_graph_indices = set()
    for serial, graph_index in pdb2g.items():
        graph_index = int(graph_index)
        if graph_index in serial_by_graph:
            duplicate_graph_indices.add(graph_index)
        serial_by_graph[graph_index] = int(serial)
    if duplicate_graph_indices:
        return {
            "passed": False,
            "reason": "non_unique_graph_to_coordinate_mapping",
            "duplicate_graph_atom_indices": sorted(duplicate_graph_indices),
            "specified_center_count": len(specified_centers),
            "matched_center_count": 0,
            "mismatch_graph_atom_indices": sorted(specified_centers),
            "evidence_source": "observed_3d_after_complete_assembly",
        }

    missing = {}
    for graph_index in specified_centers:
        atom = assembled.GetAtomWithIdx(graph_index)
        required = {
            graph_index,
            *(int(neighbor.GetIdx()) for neighbor in atom.GetNeighbors()),
        }
        absent = sorted(index for index in required if index not in serial_by_graph)
        if absent:
            missing[str(graph_index)] = absent
    if missing:
        return {
            "passed": False,
            "reason": "incomplete_coordinate_neighborhood_for_specified_stereo",
            "missing_coordinate_graph_atom_indices": missing,
            "specified_center_count": len(specified_centers),
            "matched_center_count": 0,
            "mismatch_graph_atom_indices": sorted(specified_centers),
            "evidence_source": "observed_3d_after_complete_assembly",
        }

    observed = Chem.Mol(assembled)
    Chem.RemoveStereochemistry(observed)
    conformer = Chem.Conformer(observed.GetNumAtoms())
    for graph_index, serial in serial_by_graph.items():
        source = atoms_by_serial.get(serial)
        xyz = source.get("xyz") if isinstance(source, dict) else None
        if xyz is not None and len(xyz) == 3:
            conformer.SetAtomPosition(
                graph_index, Point3D(*map(float, xyz))
            )
    observed.RemoveAllConformers()
    observed.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(
        observed, confId=0, replaceExistingTags=True
    )
    Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
    observed_centers = dict(Chem.FindMolChiralCenters(
        observed,
        includeUnassigned=True,
        useLegacyImplementation=False,
    ))
    mismatches = sorted(
        index
        for index, assignment in specified_centers.items()
        if observed_centers.get(index) != assignment
    )
    matched = len(specified_centers) - len(mismatches)
    return {
        "passed": not mismatches,
        "reason": None if not mismatches else "assembled_coordinate_stereochemistry_mismatch",
        "specified_center_count": len(specified_centers),
        "matched_center_count": matched,
        "mismatch_graph_atom_indices": mismatches,
        "expected_centers": {
            str(index): value for index, value in sorted(specified_centers.items())
        },
        "observed_centers": {
            str(index): str(observed_centers.get(index, "?"))
            for index in sorted(specified_centers)
        },
        "evidence_source": "observed_3d_after_complete_assembly",
    }


def _closure_endpoint(
    bond,
    *,
    first,
    residues,
    serials_by_position_and_name,
    pdb2g,
):
    position = int(bond.pos1 if first else bond.pos2)
    atom_name = str((bond.atom1 if first else bond.atom2) or "").strip().upper()
    rgroup = str((bond.rgroup1 if first else bond.rgroup2) or "").strip().upper()
    if not atom_name:
        atom_name = {"R1": "N", "R2": "C"}.get(rgroup, "")
    serials = list(serials_by_position_and_name.get((position, atom_name), ()))
    serial = serials[0] if len(serials) == 1 else None
    graph_index = pdb2g.get(serial) if serial is not None else None
    residue = residues[position - 1] if 1 <= position <= len(residues) else None
    return {
        "residue_position": position,
        "pdb_resseq": residue.get("num") if residue else None,
        "resname": residue.get("name") if residue else None,
        "atom_name": atom_name or None,
        "rgroup": rgroup or None,
        "candidate_serials": serials,
        "pdb_serial": serial,
        "final_graph_atom_index": graph_index,
        "resolution_unique": len(serials) == 1 and graph_index is not None,
    }


def _closure_evidence(
    combo,
    *,
    pdb_path,
    chain_id,
    residues,
    serials_by_position_and_name,
    pdb2g,
    allow_geometry,
    radius_multiplier=None,
    distance_ceiling=None,
):
    from ..core.cyclization import detect_cyclization

    molecule = combo.GetMol()
    baseline_key = _molecule_inchikey(molecule)
    detected = detect_cyclization(
        pdb_path,
        chain_id,
        allow_geometric_inference=allow_geometry,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    rows = []
    for bond in detected.bonds:
        endpoint_1 = _closure_endpoint(
            bond,
            first=True,
            residues=residues,
            serials_by_position_and_name=serials_by_position_and_name,
            pdb2g=pdb2g,
        )
        endpoint_2 = _closure_endpoint(
            bond,
            first=False,
            residues=residues,
            serials_by_position_and_name=serials_by_position_and_name,
            pdb2g=pdb2g,
        )
        endpoint_signature = sorted(
            (
                (
                    endpoint_1["residue_position"],
                    endpoint_1["atom_name"] or endpoint_1["rgroup"],
                ),
                (
                    endpoint_2["residue_position"],
                    endpoint_2["atom_name"] or endpoint_2["rgroup"],
                ),
            )
        )
        closure_id = hashlib.sha256(
            repr(endpoint_signature).encode("utf-8")
        ).hexdigest()[:16]
        indices = (
            endpoint_1["final_graph_atom_index"],
            endpoint_2["final_graph_atom_index"],
        )
        graph_bond = None
        if all(index is not None for index in indices) and indices[0] != indices[1]:
            graph_bond = molecule.GetBondBetweenAtoms(int(indices[0]), int(indices[1]))
        materialized = graph_bond is not None
        counterfactual_status = "not_materialized"
        counterfactual_key = None
        counterfactual_error = None
        identity_changes = False
        if materialized:
            try:
                edited = Chem.RWMol(molecule)
                edited.RemoveBond(int(indices[0]), int(indices[1]))
                counterfactual_key = _molecule_inchikey(edited.GetMol())
                counterfactual_status = "success"
                identity_changes = counterfactual_key != baseline_key
            except Exception as exc:
                counterfactual_status = "failed"
                counterfactual_error = str(exc)
        source = str(bond.evidence_source or "unknown").lower()
        rows.append({
            "closure_id": closure_id,
            "bond_type": str(bond.bond_type),
            "evidence_source": source,
            "evidence_is_explicit": source in _EXPLICIT_CLOSURE_SOURCES,
            "endpoint_1": endpoint_1,
            "endpoint_2": endpoint_2,
            "endpoints_resolved_uniquely": (
                endpoint_1["resolution_unique"] and endpoint_2["resolution_unique"]
            ),
            "materialized": materialized,
            "graph_bond_type": str(graph_bond.GetBondType()) if graph_bond else None,
            "baseline_inchikey": baseline_key,
            "counterfactual_status": counterfactual_status,
            "counterfactual_inchikey": counterfactual_key,
            "counterfactual_error": counterfactual_error,
            "identity_changes_when_removed": identity_changes,
        })
    return baseline_key, rows


_COMBO_CACHE: "dict[tuple, tuple]" = {}
_COMBO_CACHE_ORDER: "list[tuple]" = []
_COMBO_CACHE_MAX = 16
_COMBO_CACHE_LOCK = threading.RLock()


def _freeze_cache_value(value):
    if isinstance(value, dict):
        items = [
            (_freeze_cache_value(key), _freeze_cache_value(item))
            for key, item in value.items()
        ]
        return tuple(sorted(items, key=lambda item: repr(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(
            (_freeze_cache_value(item) for item in value), key=repr
        ))
    return value


def _combo_cache_key(pdb_path, chain_id, geometric_cyclization, *,
                     collect_evidence, radius_multiplier, distance_ceiling,
                     mapping_overrides) -> tuple | None:
    try:
        stat = os.stat(pdb_path)
        stat_token = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    if mapping_overrides is None:
        overrides_token = None
    else:
        overrides_token = hashlib.sha256(
            repr(_freeze_cache_value(mapping_overrides)).encode("utf-8")
        ).hexdigest()
    return (
        os.path.normcase(os.path.abspath(str(pdb_path))),
        chain_id,
        bool(geometric_cyclization),
        bool(collect_evidence),
        radius_multiplier,
        distance_ceiling,
        overrides_token,
        registry_epoch(),
        stat_token,
    )


def _build_combo(
    pdb_path,
    chain_id='L',
    geometric_cyclization=False,
    *,
    collect_evidence=False,
    radius_multiplier=None,
    distance_ceiling=None,
    mapping_overrides=None,
    _use_cache=True,
):
    """Build RWMol combo + atom mapping from PDB, without final sanitization.

    Returns (combo, pdb2g) or raises ValueError on failure.  Route
    portfolios and export retries rebuild the identical assembly dozens of
    times per entity; results are memoized against the file identity and
    the monomer-registry epoch, and every hit returns fresh mutable copies
    so callers keep build-from-scratch semantics.

    geometric_cyclization (Path E mode): after CONECT-based crosslinks, also
    add cross-residue bonds detected by covalent-radius geometry, recovering
    cyclization in PDBs that lack CONECT records.
    """
    key = (
        _combo_cache_key(
            pdb_path, chain_id, geometric_cyclization,
            collect_evidence=collect_evidence,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
            mapping_overrides=mapping_overrides,
        )
        if _use_cache
        else None
    )
    if key is not None:
        with _COMBO_CACHE_LOCK:
            cached = _COMBO_CACHE.get(key)
            if cached is not None:
                _COMBO_CACHE_ORDER.remove(key)
                _COMBO_CACHE_ORDER.append(key)
                if collect_evidence:
                    combo, pdb2g, evidence = cached
                    return (
                        Chem.RWMol(combo),
                        dict(pdb2g),
                        copy.deepcopy(evidence),
                    )
                combo, pdb2g = cached
                return Chem.RWMol(combo), dict(pdb2g)
    result = _build_combo_impl(
        pdb_path,
        chain_id,
        geometric_cyclization,
        collect_evidence=collect_evidence,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
        mapping_overrides=mapping_overrides,
    )
    if key is not None:
        with _COMBO_CACHE_LOCK:
            if key not in _COMBO_CACHE:
                if collect_evidence:
                    combo, pdb2g, evidence = result
                    _COMBO_CACHE[key] = (
                        Chem.RWMol(combo),
                        dict(pdb2g),
                        copy.deepcopy(evidence),
                    )
                else:
                    combo, pdb2g = result
                    _COMBO_CACHE[key] = (
                        Chem.RWMol(combo), dict(pdb2g)
                    )
                _COMBO_CACHE_ORDER.append(key)
                while len(_COMBO_CACHE_ORDER) > _COMBO_CACHE_MAX:
                    stale = _COMBO_CACHE_ORDER.pop(0)
                    _COMBO_CACHE.pop(stale, None)
    return result


def _build_combo_impl(
    pdb_path,
    chain_id='L',
    geometric_cyclization=False,
    *,
    collect_evidence=False,
    radius_multiplier=None,
    distance_ceiling=None,
    mapping_overrides=None,
):
    """Uncached combo assembly (see :func:`_build_combo`)."""
    residues = get_res_seq(pdb_path, chain_id)
    if not residues:
        raise ValueError(f"no {chain_id}-chain residues in {pdb_path}")
    templates = [get_residue_template(r['name']) for r in residues]

    combo = Chem.RWMol()
    offs = []
    anchors = []

    for ri, (r, template) in enumerate(zip(residues, templates)):
        smi = template.smiles
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        n_idx, ca_idx, cb_idx, c_idx, o_idx = parse_backbone(mol)
        cap_map = cap_atom_name_map(template)
        if template.symbol == "ac":
            c_idx = cap_map.get("C")
        elif template.symbol in {"nme", "nh2"}:
            n_idx = cap_map.get("N")
        anchors.append((n_idx, c_idx))

        off = add_to_combo(combo, smi)
        offs.append(off)

    # Peptide backbone bonds
    for i in range(len(residues) - 1):
        c_local = anchors[i][1]
        n_local = anchors[i + 1][0]
        if c_local is not None and n_local is not None:
            ci = offs[i] + c_local
            ni = offs[i + 1] + n_local
            if not combo.GetBondBetweenAtoms(ci, ni):
                combo.AddBond(ci, ni, Chem.BondType.SINGLE)

    # Cyclic backbone (head-to-tail) only when supported by an explicit
    # connection for Path A, or by explicit/coordinate evidence for Path E.
    rnames = [r['name'] for r in residues]
    has_head_to_tail = len(residues) >= 2 and _has_head_to_tail_evidence(
        pdb_path,
        chain_id,
        len(residues),
        allow_geometry=geometric_cyclization,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    if ('ACE' not in rnames and 'NME' not in rnames and 'NH2' not in rnames
            and has_head_to_tail):
        f_n = anchors[0][0]
        l_c = anchors[-1][1]
        if f_n is not None and l_c is not None:
            ni = offs[0] + f_n
            ci = offs[-1] + l_c
            if not combo.GetBondBetweenAtoms(ni, ci):
                combo.AddBond(ni, ci, Chem.BondType.SINGLE)

    # Atom mapping. Template chemistry comes only from the Unified registry;
    # PDB names/coordinates select a complete mapping into that graph.
    pdb2g = {}
    pdb2r = {}
    assigned_globals = set()
    residue_evidence = []
    atoms_by_serial = {}
    serials_by_position_and_name = {}
    conect = read_conect(pdb_path)
    connected_serials = set(conect)
    connected_serials.update(target for targets in conect.values() for target in targets)
    from ..core.cyclization import detect_cyclization
    explicit_topology = detect_cyclization(
        pdb_path,
        chain_id,
        allow_geometric_inference=False,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    consumed_r3_serials = set()
    for bond in explicit_topology.bonds:
        for position, atom_name, rgroup in (
            (
                int(bond.pos1),
                str(bond.atom1 or "").strip().upper(),
                str(bond.rgroup1 or "").strip().upper(),
            ),
            (
                int(bond.pos2),
                str(bond.atom2 or "").strip().upper(),
                str(bond.rgroup2 or "").strip().upper(),
            ),
        ):
            if not atom_name or not (1 <= position <= len(residues)):
                continue
            matches = [
                atom["num"]
                for atom in get_pdb_atoms(
                    pdb_path, residues[position - 1]["key"], chain_id
                )
                if str(atom["name"]).strip().upper() == atom_name
            ]
            if len(matches) == 1:
                connected_serials.add(matches[0])
                if rgroup == "R3":
                    consumed_r3_serials.add(matches[0])

    for ri, (r, template) in enumerate(zip(residues, templates)):
        off = offs[ri]
        pats = get_pdb_atoms(pdb_path, r['key'], chain_id)
        atoms_by_serial.update({int(atom["num"]): atom for atom in pats})
        for atom in pats:
            key = (ri + 1, str(atom["name"]).strip().upper())
            serials_by_position_and_name.setdefault(key, []).append(atom["num"])
        if collect_evidence:
            override = (
                mapping_overrides.get(ri + 1)
                if isinstance(mapping_overrides, dict) else None
            )
            if override is not None:
                mapping = {
                    int(serial): int(template_index)
                    for serial, template_index in override.items()
                }
                _mapping, mapping_evidence = map_pdb_atoms_with_evidence(
                    template,
                    pats,
                    connected_serials,
                    consumed_r3_serials=consumed_r3_serials,
                )
                mapping_evidence["serial_to_template_atom_index"] = {
                    str(serial): int(index)
                    for serial, index in sorted(mapping.items())
                }
                mapping_evidence["mapping_method"] = "mapping_aware_candidate_override"
                mapping_evidence["mapping_unique"] = False
            else:
                mapping, mapping_evidence = map_pdb_atoms_with_evidence(
                    template,
                    pats,
                    connected_serials,
                    consumed_r3_serials=consumed_r3_serials,
                )
            mapping_evidence["residue_position"] = ri + 1
            mapping_evidence["residue_key"] = list(r["key"])
            residue_evidence.append(mapping_evidence)
        else:
            mapping = map_pdb_atoms(
                template,
                pats,
                connected_serials,
                consumed_r3_serials=consumed_r3_serials,
            )
        for serial, template_index in mapping.items():
            global_index = off + template_index
            pdb2g[serial] = global_index
            pdb2r[serial] = ri
            assigned_globals.add(global_index)

    remove_orphans(combo, assigned_globals, pdb2g)
    preexisting_unassigned = [
        int(index)
        for index, assignment in Chem.FindMolChiralCenters(
            combo.GetMol(),
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
        if assignment == "?"
    ]
    apply_conect(combo, pdb2g, pdb2r, conect)
    explicit_materialization = _materialize_explicit_topology_bonds(
        combo,
        pdb2g,
        residues,
        serials_by_position_and_name,
        pdb_path,
        chain_id,
        radius_multiplier=radius_multiplier,
        distance_ceiling=distance_ceiling,
    )
    occupied_ports = {
        (int(position), str(rgroup).upper())
        for bond in explicit_topology.bonds
        for position, rgroup in (
            (bond.pos1, bond.rgroup1), (bond.pos2, bond.rgroup2)
        )
    }
    terminal_r2_materialization = _materialize_free_c_terminal_hydroxyl(
        combo,
        pdb2g,
        residues,
        pdb_path,
        chain_id,
        has_head_to_tail=has_head_to_tail,
        occupied_ports=occupied_ports,
    )
    if (
        isinstance(terminal_r2_materialization, dict)
        and terminal_r2_materialization.get("oxygen_pdb_serial") is not None
        and terminal_r2_materialization.get("oxygen_graph_index") is not None
    ):
        pdb2g[int(terminal_r2_materialization["oxygen_pdb_serial"])] = int(
            terminal_r2_materialization["oxygen_graph_index"]
        )
    emergent_stereochemistry = _audit_emergent_stereochemistry_from_coordinates(
        combo,
        pdb2g,
        atoms_by_serial,
        preexisting_unassigned=preexisting_unassigned,
    )
    if geometric_cyclization:
        apply_geometric_crosslinks(
            combo, pdb2g, pdb2r, pdb_path, chain_id,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
        )
    assembled_coordinate_stereochemistry = (
        _assembled_coordinate_stereochemistry_evidence(
            combo, pdb2g, atoms_by_serial
        )
    )
    if collect_evidence:
        output_inchikey, closure_rows = _closure_evidence(
            combo,
            pdb_path=pdb_path,
            chain_id=chain_id,
            residues=residues,
            serials_by_position_and_name=serials_by_position_and_name,
            pdb2g=pdb2g,
            allow_geometry=geometric_cyclization,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
        )
        resolved_radius_multiplier, resolved_distance_ceiling = (
            resolve_geometry_params(radius_multiplier, distance_ceiling)
        )
        return combo, pdb2g, {
            "route": "e" if geometric_cyclization else "a",
            "generation_provenance": _generation_provenance(
                geometric_cyclization
            ),
            "residue_evidence": residue_evidence,
            "library_chemistry_unique": all(
                row["unified_symbol"] and row["monomer_graph_sha256"]
                for row in residue_evidence
            ),
            "atom_mapping_complete": all(
                row["mapping_complete"]
                and row["mapping_injective"]
                and row["template_mapping_complete"]
                for row in residue_evidence
            ),
            "atom_mapping_unique": all(
                row["mapping_unique"]
                and row["external_attachment_mapping_unique"]
                for row in residue_evidence
            ),
            "geometry_inference_enabled": bool(geometric_cyclization),
            "geometry_radius_multiplier": resolved_radius_multiplier,
            "geometry_distance_ceiling": resolved_distance_ceiling,
            "geometry_warnings": _geometry_warnings(
                geometric_cyclization, radius_multiplier, distance_ceiling
            ),
            "output_inchikey": output_inchikey,
            "explicit_topology_materialization": explicit_materialization,
            "terminal_r2_materialization": terminal_r2_materialization,
            "emergent_stereochemistry": emergent_stereochemistry,
            "assembled_coordinate_stereochemistry": (
                assembled_coordinate_stereochemistry
            ),
            "closure_evidence": closure_rows,
            "closure_count": len(closure_rows),
            "all_closure_endpoints_resolved_uniquely": all(
                row["endpoints_resolved_uniquely"] for row in closure_rows
            ),
            "all_closures_materialized": all(
                row["materialized"] for row in closure_rows
            ),
            "all_closures_explicit": bool(closure_rows) and all(
                row["evidence_is_explicit"] for row in closure_rows
            ),
            "all_closures_identity_determining": bool(closure_rows) and all(
                row["counterfactual_status"] == "success"
                and row["identity_changes_when_removed"]
                for row in closure_rows
            ),
        }
    return combo, pdb2g


def generate(pdb_path, chain_id='L', geometric_cyclization=False, *,
             radius_multiplier=None, distance_ceiling=None,
             monomer_context=None):
    """Generate SMILES from PDB (public API).

    geometric_cyclization=True enables Path E: recover missing cyclization
    edges via covalent-radius geometry while retaining explicit edges.
    """
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                pdb_path, kind="coordinate"
            ),
        ):
            return generate(
                pdb_path,
                chain_id,
                geometric_cyclization,
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
            )
    try:
        combo, _pdb2g = _build_combo(
            pdb_path, chain_id, geometric_cyclization,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
        )
        return finish_mol(combo)
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def generate_with_evidence(pdb_path, chain_id='L', geometric_cyclization=False, *,
                           radius_multiplier=None, distance_ceiling=None,
                           mapping_overrides=None, monomer_context=None,
                           _use_cache=True):
    """Generate a candidate plus a structured library/mapping evidence ledger."""
    if needs_monomer_resolution_scope(monomer_context):
        from ..core.monomer_resolution import (
            monomer_resolution_context,
            monomer_symbol_hints,
        )

        with monomer_resolution_context(
            monomer_context,
            required_symbols=monomer_symbol_hints(
                pdb_path, kind="coordinate"
            ),
        ):
            return generate_with_evidence(
                pdb_path,
                chain_id,
                geometric_cyclization,
                radius_multiplier=radius_multiplier,
                distance_ceiling=distance_ceiling,
                mapping_overrides=mapping_overrides,
                _use_cache=_use_cache,
            )
    try:
        combo, _pdb2g, evidence = _build_combo(
            pdb_path,
            chain_id,
            geometric_cyclization,
            collect_evidence=True,
            radius_multiplier=radius_multiplier,
            distance_ceiling=distance_ceiling,
            mapping_overrides=mapping_overrides,
            _use_cache=_use_cache,
        )
        smiles, error = finish_mol(combo)
        return smiles, error, evidence
    except Exception as exc:
        resolved_radius_multiplier, resolved_distance_ceiling = (
            resolve_geometry_params(radius_multiplier, distance_ceiling)
        )
        return None, str(exc), {
            "route": "e" if geometric_cyclization else "a",
            "generation_provenance": _generation_provenance(
                geometric_cyclization
            ),
            "library_chemistry_unique": False,
            "atom_mapping_complete": False,
            "atom_mapping_unique": False,
            "geometry_inference_enabled": bool(geometric_cyclization),
            "geometry_radius_multiplier": resolved_radius_multiplier,
            "geometry_distance_ceiling": resolved_distance_ceiling,
            "geometry_warnings": _geometry_warnings(
                geometric_cyclization, radius_multiplier, distance_ceiling
            ),
            "output_inchikey": None,
            "closure_evidence": [],
            "closure_count": 0,
            "all_closure_endpoints_resolved_uniquely": False,
            "all_closures_materialized": False,
            "all_closures_explicit": False,
            "all_closures_identity_determining": False,
            "residue_evidence": [],
        }
