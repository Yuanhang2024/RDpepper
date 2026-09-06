"""Prospective specifications for family-aware v5 reconstruction."""

from __future__ import annotations

from collections import Counter

from rdkit import Chem

from cycpep_master.core.cyclization import detect_cyclization, detect_geometric_bonds
from cycpep_master.core.data import AA_SMILES
from cycpep_master.core.molecule import apply_geometric_crosslinks
from cycpep_master.core.pdb_parser import standard_pdb_atom_name_map
from cycpep_master.paths import generate_a
from cycpep_master.remediation_v5 import (
    _adjudicate_route_rows,
    _family_support_context,
    _not_supported,
    _selected_chain_context,
    _standard_residue_geometry_conflicts,
    _standard_residue_completion_ledger,
    _trace_topology_construction,
    reconstruct_pdb_fail_closed_v5,
)


MACROCYCLE = "C1CCCCCCC1"
OTHER_MACROCYCLE = "C1CCCCCCCC1"
LINEAR = "CCCCCCCC"


def _identity(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        Chem.MolToInchiKey(mol),
    )


def _context(smiles: str = MACROCYCLE, *, repaired: bool = False) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    counts = Counter(atom.GetSymbol().upper() for atom in mol.GetAtoms())
    repairs = ["CONNECTIVITY_INFERRED_FROM_COORDINATES"] if repaired else []
    return {
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "heavy_element_counts": dict(sorted(counts.items())),
        "repair_codes": repairs,
    }


def _row(route: str, smiles: str, *, raw: str | None = None) -> dict:
    canonical, key = _identity(smiles)
    raw_identity = None
    if raw:
        raw_smiles, raw_key = _identity(raw)
        raw_identity = {
            "output_smiles": raw_smiles,
            "output_inchikey": raw_key,
        }
    return {
        "route": route,
        "status": "success",
        "output_smiles": canonical,
        "output_inchikey": key,
        "candidate_identity": raw_identity,
        "warning_codes": [],
    }


def _recovery_context(*, evidence_source: str = "link") -> dict:
    context = _context()
    context["support_assessment"] = {"strict_consensus_supported": True}
    context["cyclization_bonds"] = [
        {
            "bond_type": "peptide",
            "position_1": 1,
            "position_2": 4,
            "atom_1": "N",
            "atom_2": "C",
            "rgroup_1": "R1",
            "rgroup_2": "R2",
            "evidence_source": evidence_source,
        }
    ]
    return context


def _standard_recovery_fixture(*, evidence_source: str = "link") -> tuple[dict, str]:
    from cycpep_master.paths._map_utils import get_smi_from_map

    smiles = get_smi_from_map("AAAA{cyc:N-C}")
    assert smiles is not None
    output = Chem.MolFromSmiles(smiles)
    reference = Chem.MolFromSequence("AAAA")
    assert output is not None and reference is not None
    inventory = []
    serial = 1
    for position in range(1, 5):
        atoms = []
        for atom in reference.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None or info.GetResidueNumber() != position:
                continue
            name = info.GetName().strip()
            if name == "OXT":
                continue
            atoms.append(
                {
                    "serial": serial,
                    "atom_name": name,
                    "element": atom.GetSymbol().upper(),
                }
            )
            serial += 1
        inventory.append(
            {
                "position": position,
                "residue_number": position,
                "residue_name": "ALA",
                "atoms": atoms,
            }
        )
    context = {
        "heavy_atom_count": output.GetNumHeavyAtoms(),
        "heavy_element_counts": dict(
            sorted(Counter(atom.GetSymbol().upper() for atom in output.GetAtoms()).items())
        ),
        "repair_codes": [],
        "residue_atom_inventory": inventory,
        "support_assessment": {"strict_consensus_supported": True},
        "cyclization_bonds": [
            {
                "bond_type": "peptide",
                "position_1": 1,
                "position_2": 4,
                "atom_1": "N",
                "atom_2": "C",
                "rgroup_1": "R1",
                "rgroup_2": "R2",
                "evidence_source": evidence_source,
            }
        ],
    }
    return context, smiles


def _with_explicit_only_evidence(row: dict, context: dict) -> dict:
    signature = [[1, "R1"], [4, "R2"]]
    row["explicit_only_evidence"] = {
        "status": "success",
        "output_smiles": row["output_smiles"],
        "output_inchikey": row["output_inchikey"],
        "geometry_inference_disabled": True,
        "topology_construction_trace": {
            "status": "verified",
            "route": row["route"],
            "selected_inchikey": row["output_inchikey"],
            "connection_count": 1,
            "unexpected_connection_count": 0,
            "allow_geometric_inference": False,
            "connections": [
                {
                    "signature": signature,
                    "identity_changes_when_removed": True,
                }
            ],
        },
    }
    return row


def test_family_support_preflight_uses_structured_library_membership():
    supported = _family_support_context(["ALA", "GLY"])
    assert supported["strict_consensus_supported"]
    assert supported["supported_qualifying_families"] == [
        "residue_template",
        "monomer_library",
    ]

    unsupported = _family_support_context(["ALA", "ZZZ"])
    assert not unsupported["strict_consensus_supported"]
    assert unsupported["missing_residue_templates"] == ["ZZZ"]
    assert unsupported["missing_monomer_symbols"] == ["ZZZ"]


def test_family_support_preflight_accepts_unified_dynamic_template():
    support = _family_support_context(["ORN"])
    assert support["missing_residue_templates"] == []
    assert support["missing_monomer_symbols"] == []
    assert support["strict_consensus_supported"]


def test_explicit_family_coverage_failure_is_not_supported():
    result = _not_supported(
        "V5_QUALIFYING_FAMILY_MONOMER_COVERAGE_UNSUPPORTED",
        "fixture",
        input_evidence={"repair_codes": []},
    )
    assert result.status == "not_supported"
    assert result.support_status == "not_supported"
    assert result.warning_codes == [
        "V5_QUALIFYING_FAMILY_MONOMER_COVERAGE_UNSUPPORTED"
    ]


def test_two_routes_from_one_implementation_family_are_insufficient():
    result = _adjudicate_route_rows(
        [_row("a", MACROCYCLE), _row("c", MACROCYCLE)],
        _context(),
    )
    assert result.status == "rejected"
    assert result.warning_codes == ["V5_INSUFFICIENT_EVIDENCE_FAMILIES"]


def test_two_agreeing_implementation_families_succeed():
    result = _adjudicate_route_rows(
        [_row("a", MACROCYCLE), _row("b", MACROCYCLE)],
        _context(),
    )
    assert result.status == "success"
    assert result.qualified_success
    assert result.evidence_families == ["monomer_library", "residue_template"]
    assert result.output_evidence["heavy_atom_conserved"]
    assert result.output_evidence["largest_ring_size"] == 8


def test_common_f_h_wrong_candidate_vetoes_family_consensus():
    rows = [
        _row("a", MACROCYCLE),
        _row("b", MACROCYCLE),
        _row("f", MACROCYCLE, raw=OTHER_MACROCYCLE),
        _row("h", MACROCYCLE, raw=OTHER_MACROCYCLE),
    ]
    result = _adjudicate_route_rows(rows, _context())
    assert result.status == "rejected"
    assert result.warning_codes == [
        "V5_GEOMETRIC_RAW_IDENTITY_CONFLICT"
    ]


def test_distinct_f_h_wrong_candidates_veto_family_consensus():
    rows = [
        _row("a", MACROCYCLE),
        _row("b", MACROCYCLE),
        _row("f", MACROCYCLE, raw=OTHER_MACROCYCLE),
        _row("h", MACROCYCLE, raw="C1CCCCCCCCC1"),
    ]
    result = _adjudicate_route_rows(rows, _context())
    assert result.status == "rejected"
    assert result.warning_codes == [
        "V5_GEOMETRIC_RAW_IDENTITY_CONFLICT"
    ]


def test_coordinate_diagnostic_family_cannot_replace_monomer_family():
    result = _adjudicate_route_rows(
        [_row("a", MACROCYCLE), _row("f", MACROCYCLE)],
        _context(),
    )
    assert result.status == "rejected"
    assert result.warning_codes == ["V5_INSUFFICIENT_EVIDENCE_FAMILIES"]


def test_heavy_atom_nonconservation_is_rejected():
    result = _adjudicate_route_rows(
        [_row("a", MACROCYCLE), _row("b", MACROCYCLE)],
        _context(OTHER_MACROCYCLE),
    )
    assert result.status == "rejected"
    assert result.warning_codes == ["V5_HEAVY_ATOM_DELETION_REJECTED"]


def test_explicit_b_g_recovery_bypasses_residue_family_conflict_as_repaired():
    context, selected = _standard_recovery_fixture()
    rows = [
        _row("a", MACROCYCLE),
        _with_explicit_only_evidence(_row("b", selected), context),
        _with_explicit_only_evidence(_row("g", selected), context),
    ]
    result = _adjudicate_route_rows(rows, context)
    assert result.status == "success"
    assert not result.qualified_success
    assert result.path_used == "V5_EXPLICIT_MONOMER_FAMILY_RECOVERY"
    assert result.evidence_families == ["monomer_library"]
    assert "EXPLICIT_MONOMER_FAMILY_RECOVERY" in result.repair_codes
    assert "RESIDUE_FAMILY_CONFLICT_BYPASSED" in result.repair_codes
    assert result.output_evidence["closure_provenance_complete"]
    assert result.output_evidence["observed_atom_provenance_ledger"]["ledger_closed"]


def test_explicit_b_g_recovery_requires_both_routes():
    context, selected = _standard_recovery_fixture()
    rows = [
        _row("a", MACROCYCLE),
        _with_explicit_only_evidence(_row("g", selected), context),
    ]
    result = _adjudicate_route_rows(rows, context)
    assert result.status == "rejected"
    assert result.warning_codes == ["V5_CROSS_FAMILY_IDENTITY_CONFLICT"]


def test_geometry_derived_closure_cannot_enter_explicit_b_g_recovery():
    context, selected = _standard_recovery_fixture(evidence_source="geometry")
    rows = [
        _row("a", MACROCYCLE),
        _with_explicit_only_evidence(_row("b", selected), context),
        _with_explicit_only_evidence(_row("g", selected), context),
    ]
    result = _adjudicate_route_rows(rows, context)
    assert result.status == "rejected"
    assert result.warning_codes == ["V5_CROSS_FAMILY_IDENTITY_CONFLICT"]


def test_standard_completion_ledger_accounts_for_missing_terminal_oxygen():
    reference = Chem.MolFromSequence("AA")
    assert reference is not None
    output_counts = Counter(atom.GetSymbol().upper() for atom in reference.GetAtoms())
    inventory = []
    input_counts = Counter(output_counts)
    input_counts["O"] -= 1
    for position in (1, 2):
        atoms = []
        for atom in reference.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None or info.GetResidueNumber() != position:
                continue
            name = info.GetName().strip()
            if name == "OXT":
                continue
            atoms.append(
                {
                    "serial": len(atoms) + 1,
                    "atom_name": name,
                    "element": atom.GetSymbol().upper(),
                }
            )
        inventory.append(
            {
                "position": position,
                "residue_number": position,
                "residue_name": "ALA",
                "atoms": atoms,
            }
        )
    ledger = _standard_residue_completion_ledger(
        {
            "heavy_element_counts": dict(input_counts),
            "residue_atom_inventory": inventory,
            "cyclization_bonds": [],
        },
        dict(output_counts),
    )
    assert ledger["ledger_closed"]
    assert ledger["added_element_counts"] == {"O": 1}
    assert ledger["added_atoms"] == [
        {"position": 2, "atom_name": "OXT", "element": "O"}
    ]


def test_linear_output_is_not_a_cyclic_peptide_success():
    result = _adjudicate_route_rows(
        [_row("a", LINEAR), _row("b", LINEAR)],
        _context(LINEAR),
    )
    assert result.status == "rejected"
    assert result.warning_codes == ["V5_MACROCYCLE_NOT_DEMONSTRATED"]


def test_coordinate_repair_is_explicit_and_not_unqualified_success():
    result = _adjudicate_route_rows(
        [_row("a", MACROCYCLE), _row("b", MACROCYCLE)],
        _context(repaired=True),
    )
    assert result.status == "success"
    assert not result.qualified_success
    assert result.warning_codes == ["CONNECTIVITY_INFERRED_FROM_COORDINATES"]
    assert result.repair_codes == result.warning_codes


def test_route_label_alone_cannot_claim_edge_level_closure_provenance():
    context = _context()
    context["cyclization_bonds"] = [
        {
            "bond_type": "peptide",
            "position_1": 1,
            "position_2": 4,
            "atom_1": "N",
            "atom_2": "C",
            "rgroup_1": "R1",
            "rgroup_2": "R2",
            "evidence_source": "link",
        }
    ]
    result = _adjudicate_route_rows(
        [_row("a", MACROCYCLE), _row("b", MACROCYCLE)],
        context,
    )
    assert result.status == "success"
    assert not result.qualified_success
    provenance = result.output_evidence["accepted_closure_provenance"]
    assert not provenance[0]["traceable_to_selected_identity"]
    assert provenance[0]["selected_identity_supporters"] == []


def test_untraceable_selected_closure_is_explicitly_unqualified():
    context = _context()
    context["cyclization_bonds"] = [
        {
            "bond_type": "peptide",
            "position_1": 1,
            "position_2": 4,
            "atom_1": "N",
            "atom_2": "C",
            "rgroup_1": "R1",
            "rgroup_2": "R2",
            "evidence_source": "link",
        }
    ]
    result = _adjudicate_route_rows(
        [_row("a", MACROCYCLE), _row("b", MACROCYCLE)],
        context,
    )
    assert result.status == "success"
    assert not result.qualified_success
    assert "ACCEPTED_CLOSURE_PROVENANCE_UNRESOLVED" in result.repair_codes


def test_actual_b_route_traces_each_declared_closure_by_counterfactual(tmp_path):
    from cycpep_master.paths.path_b import generate as generate_b

    path = tmp_path / "traceable-head-to-tail.pdb"
    path.write_text(
        "\n".join(
            [
                _link_record("N", "ALA", 1, "C", "ALA", 3),
                _pdb_atom(1, 1, 0.0, atom_name="N", element="N"),
                _pdb_atom(2, 2, 5.0, atom_name="CA", element="C"),
                _pdb_atom(3, 3, 1.3, atom_name="C", element="C"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    smiles, error = generate_b(str(path), "A")
    assert error is None
    _, key = _identity(smiles)
    context = {
        "cyclization_bonds": [
            {
                "position_1": 1,
                "position_2": 3,
                "rgroup_1": "R1",
                "rgroup_2": "R2",
            }
        ]
    }
    trace = _trace_topology_construction("b", path, "A", key, context)
    assert trace["status"] == "verified"
    assert trace["connection_count"] == 1
    assert trace["connections"][0]["identity_changes_when_removed"]


def _pdb_atom(
    serial: int,
    residue: int,
    x: float,
    *,
    y: float = 0.0,
    z: float = 0.0,
    atom_name: str = "C",
    element: str = "C",
    altloc: str = " ",
    insertion_code: str = " ",
    residue_name: str = "ALA",
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:>4s}{altloc}{residue_name:>3s} A{residue:4d}"
        f"{insertion_code:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          "
        f"{element:>2s}  "
    )


def _same_chain_link(chain_id: str = "A") -> str:
    line = list(_link_record("C", "ALA", 1, "N", "ALA", 3))
    line[21] = chain_id
    line[51] = chain_id
    return "".join(line)


def _link_record(
    atom1: str,
    res1: str,
    num1: int,
    atom2: str,
    res2: str,
    num2: int,
    *,
    icode1: str = " ",
    icode2: str = " ",
) -> str:
    line = [" "] * 80
    line[0:6] = "LINK  "
    line[12:16] = f"{atom1:>4s}"
    line[17:20] = f"{res1:>3s}"
    line[21] = "A"
    line[22:26] = f"{num1:4d}"
    line[26] = icode1
    line[42:46] = f"{atom2:>4s}"
    line[47:50] = f"{res2:>3s}"
    line[51] = "A"
    line[52:56] = f"{num2:4d}"
    line[56] = icode2
    return "".join(line)


def _seqres_record(serial: int, chain: str, total: int, names: list[str]) -> str:
    return f"SEQRES {serial:3d} {chain}{total:5d}  {' '.join(names)}"


def _ssbond_record(
    num1: int,
    num2: int,
    *,
    icode1: str = " ",
    icode2: str = " ",
) -> str:
    line = [" "] * 80
    line[0:6] = "SSBOND"
    line[7:10] = f"{1:3d}"
    line[11:14] = "CYS"
    line[15] = "A"
    line[17:21] = f"{num1:4d}"
    line[21] = icode1
    line[25:28] = "CYS"
    line[29] = "A"
    line[31:35] = f"{num2:4d}"
    line[35] = icode2
    return "".join(line)


def test_selected_chain_context_resolves_link_insertion_code(tmp_path):
    path = tmp_path / "link-insertion-code.pdb"
    path.write_text(
        "\n".join(
            [
                _link_record("C", "ALA", 1, "N", "ALA", 3, icode2="A"),
                _pdb_atom(1, 1, 0.0, atom_name="C"),
                _pdb_atom(2, 2, 10.0, atom_name="CA"),
                _pdb_atom(
                    3, 3, 20.0, atom_name="N", element="N", insertion_code="A"
                ),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    context = _selected_chain_context(path, "A")

    assert context["same_chain_record_counts"]["link"] == 1
    assert context["explicit_connection_audit"]["selected_closure_edge_count"] == 1
    assert context["residue_atom_inventory"][-1]["insertion_code"] == "A"
    assert context["repair_codes"] == []


def test_selected_chain_context_resolves_ssbond_insertion_code(tmp_path):
    path = tmp_path / "ssbond-insertion-code.pdb"
    path.write_text(
        "\n".join(
            [
                _ssbond_record(1, 3, icode2="A"),
                _pdb_atom(
                    1, 1, 0.0, atom_name="SG", element="S", residue_name="CYS"
                ),
                _pdb_atom(2, 2, 10.0, atom_name="CA"),
                _pdb_atom(
                    3,
                    3,
                    20.0,
                    atom_name="SG",
                    element="S",
                    insertion_code="A",
                    residue_name="CYS",
                ),
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    context = _selected_chain_context(path, "A")

    assert context["same_chain_record_counts"]["ssbond"] == 1
    assert context["explicit_connection_audit"]["selected_closure_edge_count"] == 1
    assert context["accepted_cyclization_evidence_sources"] == ["ssbond"]
    assert context["repair_codes"] == []


def test_isoleucine_pdb_atom_map_attaches_cd1_to_cg1():
    mapping = standard_pdb_atom_name_map("ILE", AA_SMILES["ILE"])
    template = Chem.MolFromSmiles(AA_SMILES["ILE"])
    assert template is not None
    assert template.GetBondBetweenAtoms(mapping["CG1"], mapping["CD1"]) is not None
    assert template.GetBondBetweenAtoms(mapping["CG2"], mapping["CD1"]) is None


def test_standard_residue_geometry_detects_equal_formula_alias_conflict():
    valid = {
        "position": 1,
        "residue_number": 1,
        "residue_name": "LEU",
        "atoms": [
            {"atom_name": "CB", "element": "C", "xyz": (0.0, 0.0, 0.0)},
            {"atom_name": "CG", "element": "C", "xyz": (1.52, 0.0, 0.0)},
            {"atom_name": "CD2", "element": "C", "xyz": (2.28, 1.32, 0.0)},
        ],
    }
    assert _standard_residue_geometry_conflicts([valid]) == []
    invalid = {**valid, "atoms": [dict(atom) for atom in valid["atoms"]]}
    invalid["atoms"][2]["xyz"] = (0.0, 1.52, 0.0)
    conflicts = _standard_residue_geometry_conflicts([invalid])
    assert [(row["atom_1"], row["atom_2"]) for row in conflicts] == [("CG", "CD2")]


def test_selected_chain_context_rejects_standard_residue_geometry_conflict(tmp_path):
    path = tmp_path / "equal-formula-alias-conflict.pdb"
    path.write_text(
        "\n".join(
            [
                _seqres_record(1, "A", 2, ["LEU", "ALA"]),
                _link_record("N", "LEU", 1, "C", "ALA", 2),
                _pdb_atom(1, 1, -1.5, atom_name="N", element="N", residue_name="LEU"),
                _pdb_atom(2, 1, 0.0, atom_name="CB", element="C", residue_name="LEU"),
                _pdb_atom(3, 1, 1.52, atom_name="CG", element="C", residue_name="LEU"),
                _pdb_atom(
                    4,
                    1,
                    0.0,
                    y=1.52,
                    atom_name="CD2",
                    element="C",
                    residue_name="LEU",
                ),
                _pdb_atom(5, 2, 3.0, atom_name="C", element="C"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_STANDARD_RESIDUE_BOND_GEOMETRY_CONFLICT"
    else:
        raise AssertionError("contradictory standard-residue geometry must be rejected")


def test_selected_chain_context_rejects_seqres_coordinate_truncation(tmp_path):
    path = tmp_path / "seqres-truncated.pdb"
    path.write_text(
        "\n".join(
            [
                _seqres_record(1, "A", 3, ["ALA", "ALA", "ALA"]),
                _pdb_atom(1, 1, 0.0, atom_name="N", element="N"),
                _pdb_atom(2, 1, 1.4, atom_name="C", element="C"),
                _pdb_atom(3, 2, 3.0, atom_name="N", element="N"),
                _pdb_atom(4, 2, 4.4, atom_name="C", element="C"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_SEQRES_COORDINATE_MISMATCH"
    else:
        raise AssertionError("SEQRES/coordinate truncation must be rejected")


def test_selected_chain_context_rejects_same_chain_atoms_after_ter(tmp_path):
    ter = list(" " * 80)
    ter[0:6] = "TER   "
    ter[6:11] = f"{3:5d}"
    ter[17:20] = "ALA"
    ter[21] = "A"
    ter[22:26] = f"{1:4d}"
    path = tmp_path / "same-chain-after-ter.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, atom_name="N", element="N"),
                _pdb_atom(2, 1, 1.4, atom_name="C", element="C"),
                "".join(ter),
                _pdb_atom(4, 2, 3.0, atom_name="N", element="N"),
                _pdb_atom(5, 2, 4.4, atom_name="C", element="C"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_MULTISEGMENT_CHAIN_REJECTED"
    else:
        raise AssertionError("selected-chain atoms after TER must be rejected")


def test_selected_chain_context_rejects_altloc_before_reconstruction(tmp_path):
    path = tmp_path / "altloc.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, altloc="A"),
                _pdb_atom(2, 2, 1.5),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_ALTLOC_INPUT_REJECTED"
    else:
        raise AssertionError("alternate locations must be rejected")


def test_adjacent_conect_does_not_hide_coordinate_repair(tmp_path, monkeypatch):
    path = tmp_path / "adjacent-conect.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, atom_name="C", element="C"),
                _pdb_atom(2, 2, 1.5, atom_name="N", element="N"),
                _pdb_atom(3, 2, 3.0, atom_name="C", element="C"),
                _pdb_atom(4, 3, 4.5, atom_name="N", element="N"),
                "CONECT    1    2",
                "CONECT    3    4",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _Topology:
        topology = "monocyclic"
        bonds = [
            type(
                "Bond",
                (),
                {
                    "bond_type": "head_to_tail",
                    "pos1": 1,
                    "pos2": 3,
                    "atom1": "C",
                    "atom2": "N",
                    "rgroup1": "R2",
                    "rgroup2": "R1",
                },
            )()
        ]

    monkeypatch.setattr(
        "cycpep_master.remediation_v5.detect_cyclization",
        lambda *_: _Topology(),
    )
    context = _selected_chain_context(path, "A")
    assert context["same_chain_record_counts"]["conect_closure_edges"] == 0
    assert context["repair_codes"] == ["CONNECTIVITY_INFERRED_FROM_COORDINATES"]


def test_ignored_link_does_not_hide_coordinate_repair(tmp_path, monkeypatch):
    path = tmp_path / "ignored-link.pdb"
    path.write_text(
        "\n".join(
            [
                _same_chain_link(),
                _pdb_atom(1, 1, 0.0, atom_name="C", element="C"),
                _pdb_atom(2, 2, 1.5),
                _pdb_atom(3, 3, 3.0, atom_name="N", element="N"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _Topology:
        topology = "monocyclic"
        bonds = [
            type(
                "Bond",
                (),
                {
                    "bond_type": "head_to_tail",
                    "pos1": 1,
                    "pos2": 3,
                    "atom1": "C",
                    "atom2": "N",
                    "rgroup1": "R2",
                    "rgroup2": "R1",
                    "evidence_source": "geometry",
                },
            )()
        ]

    monkeypatch.setattr(
        "cycpep_master.remediation_v5.detect_cyclization",
        lambda *_: _Topology(),
    )
    context = _selected_chain_context(path, "A")
    assert context["same_chain_record_counts"]["link"] == 1
    assert context["accepted_cyclization_evidence_sources"] == ["geometry"]
    assert context["repair_codes"] == ["CONNECTIVITY_INFERRED_FROM_COORDINATES"]


def test_accepted_link_is_not_mislabeled_as_coordinate_repair(tmp_path, monkeypatch):
    path = tmp_path / "accepted-link.pdb"
    path.write_text(
        "\n".join(
            [
                _same_chain_link(),
                _pdb_atom(1, 1, 0.0, atom_name="C", element="C"),
                _pdb_atom(2, 2, 1.5),
                _pdb_atom(3, 3, 3.0, atom_name="N", element="N"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _Topology:
        topology = "monocyclic"
        bonds = [
            type(
                "Bond",
                (),
                {
                    "bond_type": "isopeptide",
                    "pos1": 1,
                    "pos2": 3,
                    "atom1": "NZ",
                    "atom2": "CG",
                    "rgroup1": "R3",
                    "rgroup2": "R3",
                    "evidence_source": "link",
                },
            )()
        ]

    monkeypatch.setattr(
        "cycpep_master.remediation_v5.detect_cyclization",
        lambda *_: _Topology(),
    )
    context = _selected_chain_context(path, "A")
    assert context["accepted_cyclization_evidence_sources"] == ["link"]
    assert context["repair_codes"] == []


def test_selected_chain_context_rejects_conflicting_closure_partners(
    tmp_path, monkeypatch
):
    path = tmp_path / "conflicting-endpoint.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, atom_name="SG", element="S", residue_name="CYS"),
                _pdb_atom(2, 2, 2.0, atom_name="SG", element="S", residue_name="CYS"),
                _pdb_atom(3, 3, 4.0, atom_name="SG", element="S", residue_name="CYS"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _Topology:
        topology = "bicyclic"
        bonds = [
            type(
                "Bond",
                (),
                {
                    "bond_type": "disulfide",
                    "pos1": 1,
                    "pos2": 2,
                    "atom1": "SG",
                    "atom2": "SG",
                    "rgroup1": "R3",
                    "rgroup2": "R3",
                    "evidence_source": "ssbond",
                },
            )(),
            type(
                "Bond",
                (),
                {
                    "bond_type": "disulfide",
                    "pos1": 1,
                    "pos2": 3,
                    "atom1": "SG",
                    "atom2": "SG",
                    "rgroup1": "R3",
                    "rgroup2": "R3",
                    "evidence_source": "conect",
                },
            )(),
        ]

    monkeypatch.setattr(
        "cycpep_master.remediation_v5.detect_cyclization",
        lambda *_: _Topology(),
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_CONFLICTING_CYCLIZATION_ENDPOINT"
    else:
        raise AssertionError("conflicting closure partners must be rejected")


def test_selected_chain_context_rejects_truncated_link_endpoint(tmp_path):
    path = tmp_path / "truncated-link-endpoint.pdb"
    path.write_text(
        "\n".join(
            [
                _link_record("C", "ALA", 1, "N", "ALA", 3),
                _pdb_atom(1, 1, 0.0, atom_name="C", element="C"),
                _pdb_atom(2, 2, 3.0, atom_name="CA", element="C"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_TRUNCATED_CONNECTION_ENDPOINT"
    else:
        raise AssertionError("a missing LINK endpoint must be rejected")


def test_selected_chain_context_rejects_malformed_conect(tmp_path):
    path = tmp_path / "malformed-conect.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0),
                _pdb_atom(2, 2, 3.0),
                "CONECT    1 XXXX",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_MALFORMED_CONECT_RECORD"
    else:
        raise AssertionError("a malformed CONECT record must be rejected")


def test_selected_chain_context_rejects_explicit_multi_partner_atom(tmp_path):
    path = tmp_path / "multi-partner-conect.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, atom_name="SG", element="S", residue_name="CYS"),
                _pdb_atom(2, 2, 2.0, atom_name="SG", element="S", residue_name="CYS"),
                _pdb_atom(3, 3, 4.0, atom_name="SG", element="S", residue_name="CYS"),
                "CONECT    1    2    3",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_EXPLICIT_CONNECTION_VALENCE_CONFLICT"
    else:
        raise AssertionError("one closure atom cannot have two explicit partners")


def test_selected_chain_context_rejects_explicit_geometry_partner_conflict(tmp_path):
    path = tmp_path / "record-geometry-conflict.pdb"
    path.write_text(
        "\n".join(
            [
                _link_record("NZ", "LYS", 1, "C", "ALA", 2),
                _pdb_atom(1, 1, 0.0, atom_name="NZ", element="N", residue_name="LYS"),
                _pdb_atom(2, 2, 1.3, atom_name="C", element="C"),
                _pdb_atom(3, 3, -1.3, atom_name="C", element="C"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _selected_chain_context(path, "A")
    except ValueError as exc:
        assert getattr(exc, "code", None) == "V5_CONFLICTING_CYCLIZATION_ENDPOINT"
    else:
        raise AssertionError("record/geometry partner conflict must be rejected")


def test_geometric_detector_records_distance_evidence_source(tmp_path):
    path = tmp_path / "distance-source.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, atom_name="C", element="C"),
                _pdb_atom(2, 2, 20.0, atom_name="C", element="C"),
                _pdb_atom(3, 3, 1.3, atom_name="N", element="N"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bonds = detect_geometric_bonds(str(path), "A", {1: 1, 2: 2, 3: 3}, 3)
    assert len(bonds) == 1
    assert bonds[0].evidence_source == "geometry"


def test_geometric_detector_preserves_conect_evidence_source(tmp_path):
    path = tmp_path / "conect-source.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, atom_name="C", element="C"),
                _pdb_atom(2, 2, 20.0, atom_name="C", element="C"),
                _pdb_atom(3, 3, 1.3, atom_name="N", element="N"),
                "CONECT    1    3",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bonds = detect_geometric_bonds(str(path), "A", {1: 1, 2: 2, 3: 3}, 3)
    assert len(bonds) == 1
    assert bonds[0].evidence_source == "conect"


def test_conect_keeps_distinct_r3_atom_pairs_on_same_residue_pair(tmp_path):
    path = tmp_path / "two-r3-pairs.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0, atom_name="CB", element="C"),
                _pdb_atom(2, 1, 5.0, atom_name="CG", element="C"),
                _pdb_atom(3, 3, 10.0, atom_name="CD", element="C"),
                _pdb_atom(4, 3, 15.0, atom_name="CE", element="C"),
                "CONECT    1    3",
                "CONECT    2    4",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bonds = detect_geometric_bonds(
        str(path), "A", {1: 1, 3: 2}, 2, include_geometry=False
    )
    assert len(bonds) == 2
    assert {
        frozenset((bond.atom1, bond.atom2)) for bond in bonds
    } == {frozenset(("CB", "CD")), frozenset(("CG", "CE"))}


def test_partial_link_does_not_suppress_second_geometric_closure(tmp_path):
    path = tmp_path / "partial-link.pdb"
    path.write_text(
        "\n".join(
            [
                _link_record("NZ", "LYS", 1, "C", "ALA", 3),
                _pdb_atom(1, 1, 0.0, atom_name="NZ", element="N", residue_name="LYS"),
                _pdb_atom(2, 2, 10.0, atom_name="SG", element="S", residue_name="CYS"),
                _pdb_atom(3, 3, 1.3, atom_name="C", element="C"),
                _pdb_atom(4, 4, 12.0, atom_name="SG", element="S", residue_name="CYS"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    info = detect_cyclization(str(path), "A")
    assert len(info.bonds) == 2
    assert {bond.evidence_source for bond in info.bonds} == {"link", "geometry"}


def test_link_ports_follow_actual_sidechain_and_backbone_atoms(tmp_path):
    path = tmp_path / "sidechain-backbone-link.pdb"
    path.write_text(
        "\n".join(
            [
                _link_record("NZ", "LYS", 1, "C", "ALA", 3),
                _pdb_atom(1, 1, 0.0, atom_name="NZ", element="N", residue_name="LYS"),
                _pdb_atom(2, 2, 10.0, atom_name="CA", element="C"),
                _pdb_atom(3, 3, 1.3, atom_name="C", element="C"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    info = detect_cyclization(
        str(path), "A", allow_geometric_inference=False
    )
    assert len(info.bonds) == 1
    assert (info.bonds[0].rgroup1, info.bonds[0].rgroup2) == ("R3", "R2")


def test_link_only_disulfide_is_retained_as_explicit_evidence(tmp_path):
    path = tmp_path / "link-only-disulfide.pdb"
    path.write_text(
        "\n".join(
            [
                _link_record("SG", "CYS", 1, "SG", "CYS", 3),
                _pdb_atom(1, 1, 0.0, atom_name="SG", element="S", residue_name="CYS"),
                _pdb_atom(2, 2, 10.0, atom_name="CA", element="C"),
                _pdb_atom(3, 3, 20.0, atom_name="SG", element="S", residue_name="CYS"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    info = detect_cyclization(str(path), "A", allow_geometric_inference=False)
    assert len(info.bonds) == 1
    assert info.bonds[0].bond_type == "disulfide"
    assert info.bonds[0].evidence_source == "link"


def test_exact_ssbond_and_link_disulfide_are_deduplicated(tmp_path):
    ssbond = list(" " * 80)
    ssbond[0:6] = "SSBOND"
    ssbond[7:10] = f"{1:3d}"
    ssbond[11:14] = "CYS"
    ssbond[15] = "A"
    ssbond[17:21] = f"{1:4d}"
    ssbond[25:28] = "CYS"
    ssbond[29] = "A"
    ssbond[31:35] = f"{3:4d}"
    path = tmp_path / "ssbond-and-link-disulfide.pdb"
    path.write_text(
        "\n".join(
            [
                "".join(ssbond),
                _link_record("SG", "CYS", 1, "SG", "CYS", 3),
                _pdb_atom(1, 1, 0.0, atom_name="SG", element="S", residue_name="CYS"),
                _pdb_atom(2, 2, 10.0, atom_name="CA", element="C"),
                _pdb_atom(3, 3, 20.0, atom_name="SG", element="S", residue_name="CYS"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    info = detect_cyclization(str(path), "A", allow_geometric_inference=False)
    assert len(info.bonds) == 1
    assert info.bonds[0].evidence_source == "ssbond"


def test_path_a_does_not_auto_close_unrecorded_linear_chain(tmp_path):
    path = tmp_path / "linear-dipeptide.pdb"
    atoms = []
    serial = 1
    for residue, offset in ((1, 0.0), (2, 5.0)):
        for name, element, delta in (
            ("N", "N", 0.0),
            ("CA", "C", 1.4),
            ("CB", "C", 1.4),
            ("C", "C", 2.8),
            ("O", "O", 3.9),
        ):
            atoms.append(
                _pdb_atom(
                    serial,
                    residue,
                    offset + delta,
                    atom_name=name,
                    element=element,
                )
            )
            serial += 1
    path.write_text("\n".join([*atoms, "END"]) + "\n", encoding="utf-8")
    smiles, error = generate_a(str(path), "A")
    assert error is None
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert not Chem.GetSymmSSSR(mol)


def test_path_e_geometry_is_not_globally_masked_by_unrelated_conect(tmp_path):
    path = tmp_path / "partial-conect.pdb"
    path.write_text(
        "\n".join(
            [
                _pdb_atom(1, 1, 0.0),
                _pdb_atom(2, 2, 1.3),
                _pdb_atom(3, 3, 20.0),
                _pdb_atom(4, 4, 21.3),
                "CONECT    1    2",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    combo = Chem.RWMol()
    for _ in range(4):
        combo.AddAtom(Chem.Atom("C"))
    combo.AddBond(0, 1, Chem.BondType.SINGLE)
    added = apply_geometric_crosslinks(
        combo,
        {1: 0, 2: 1, 3: 2, 4: 3},
        {1: 0, 2: 1, 3: 2, 4: 3},
        str(path),
        "A",
    )
    assert added == 1
    assert combo.GetBondBetweenAtoms(2, 3) is not None


def test_v5_end_to_end_uses_verified_production_topology_trace(tmp_path):
    path = tmp_path / "cyclic_trialanine.pdb"
    atoms = []
    serial = 1
    for residue, offset in ((1, 0.0), (2, 10.0), (3, 20.0)):
        for name, element, delta in (
            ("N", "N", 0.0),
            ("CA", "C", 1.4),
            ("CB", "C", 1.8),
            ("C", "C", 2.8),
            ("O", "O", 3.9),
        ):
            atoms.append(
                _pdb_atom(
                    serial,
                    residue,
                    offset + delta,
                    atom_name=name,
                    element=element,
                )
            )
            serial += 1
    path.write_text(
        "\n".join(
            [
                _link_record("N", "ALA", 1, "C", "ALA", 3),
                *atoms,
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = reconstruct_pdb_fail_closed_v5(path, "A")
    assert result.status == "success", result.rejection_reason
    assert result.qualified_success
    assert result.repair_codes == []
    assert result.output_evidence["closure_provenance_complete"]
    assert result.output_evidence["accepted_closure_provenance"][0][
        "traceable_to_selected_identity"
    ]
    traced = [
        row
        for row in result.route_results
        if row.get("topology_construction_trace", {}).get("status") == "verified"
    ]
    assert traced
