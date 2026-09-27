from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.core.derived_monomers import (
    PortSemanticMismatchError,
    build_derived_batch,
    build_derived_row,
)
from cycpep_master.core.local_monomer_inference import (
    bootstrap_unknown_monomers,
    infer_residue_monomer,
)
from cycpep_master.paths import _map_utils
from cycpep_master.paths.residue_template_factory import resolve_symbol
from cycpep_master.core.pdb_parser import (
    get_pdb_atoms,
    parse_backbone,
    standard_pdb_atom_name_map,
)
from cycpep_master.paths.residue_template_factory import get_residue_template
from cycpep_master.core.cyclization import (
    _element_from_name,
    _is_covalent_bond,
    read_atoms,
)


def _pdb_line(serial, name, resname, xyz, element, *, het=True, resseq=1):
    record = "HETATM" if het else "ATOM  "
    return (
        f"{record}{serial:5d} {name:>4s} {resname:>3s} A{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          "
        f"{element:>2s}"
    )


def test_covalent_geometry_normalizes_two_letter_element_case():
    carbon = {"elem": "C", "xyz": (0.0, 0.0, 0.0)}
    bromine = {"elem": "Br", "xyz": (1.94, 0.0, 0.0)}
    assert _is_covalent_bond(carbon, bromine)


def test_blank_element_inference_respects_pdb_atom_name_alignment(tmp_path):
    assert _element_from_name(" CA ") == "C"
    assert _element_from_name("CA  ") == "CA"
    path, _key = _write_disguised_alanine(tmp_path)
    path.write_text(
        "\n".join(
            line[:76] + "  " + line[78:]
            if line.startswith(("ATOM", "HETATM")) else line
            for line in path.read_text(encoding="ascii").splitlines()
        ) + "\n",
        encoding="ascii",
    )
    parsed = read_atoms(str(path), "A")
    alpha_carbon = next(atom for atom in parsed.values() if atom["name"] == "CA")
    assert alpha_carbon["elem"] == "C"


def _link_record(atom1, res1, num1, atom2, res2, num2):
    line = [" "] * 80
    line[0:6] = "LINK  "
    line[12:16] = f"{atom1:>4s}"
    line[17:20] = f"{res1:>3s}"
    line[21] = "A"
    line[22:26] = f"{num1:4d}"
    line[42:46] = f"{atom2:>4s}"
    line[47:50] = f"{res2:>3s}"
    line[51] = "A"
    line[52:56] = f"{num2:4d}"
    return "".join(line)


def _write_unknown_cysteine_crosslink(
    tmp_path, *, links=1, second_local_atom=None, geometry_only=False,
    record_type="link",
):
    template = get_residue_template("CYS")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=31) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    names = {
        index: name
        for name, index in standard_pdb_atom_name_map(
            "CYS", template.smiles
        ).items()
    }
    conformer = molecule.GetConformer()
    lines = []
    sg_xyz = None
    serial_by_name = {}
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        name = names[atom.GetIdx()]
        xyz = (point.x, point.y, point.z)
        serial = atom.GetIdx() + 1
        serial_by_name[name] = serial
        if name == "SG":
            sg_xyz = xyz
        lines.append(_pdb_line(serial, name, "ZZS", xyz, atom.GetSymbol()))
    assert sg_xyz is not None
    next_serial = molecule.GetNumAtoms() + 1
    partner_positions = [3, 5][:links]
    for offset, resseq in enumerate(partner_positions):
        xyz = (
            (sg_xyz[0] + 2.05, sg_xyz[1], sg_xyz[2])
            if geometry_only and offset == 0
            else (20.0 + offset * 10.0, 0.0, 0.0)
        )
        lines.append(_pdb_line(
            next_serial + offset, "SG", "CYS", xyz, "S", resseq=resseq
        ))
    records = []
    if not geometry_only:
        if record_type == "link":
            records.append(_link_record("SG", "ZZS", 1, "SG", "CYS", 3))
            if links == 2:
                records.append(_link_record(
                    second_local_atom or "SG", "ZZS", 1, "SG", "CYS", 5
                ))
        elif record_type == "conect":
            records.append(
                f"CONECT{serial_by_name['SG']:5d}{next_serial:5d}"
            )
        elif record_type == "ssbond":
            ssbond = list(" " * 80)
            ssbond[0:6] = "SSBOND"
            ssbond[7:10] = f"{1:3d}"
            ssbond[11:14] = "ZZS"
            ssbond[15] = "A"
            ssbond[17:21] = f"{1:4d}"
            ssbond[25:28] = "CYS"
            ssbond[29] = "A"
            ssbond[31:35] = f"{3:4d}"
            records.append("".join(ssbond))
        else:
            raise ValueError(record_type)
    path = tmp_path / "unknown-cysteine-crosslink.pdb"
    path.write_text(
        "\n".join([*records, *lines, "END"]) + "\n", encoding="ascii"
    )
    return path, ("ZZS", 1, True), serial_by_name


def _write_disguised_alanine(tmp_path, *, resname="ZZA"):
    template = get_residue_template("ALA")
    molecule = Chem.AddHs(template.mol)
    assert AllChem.EmbedMolecule(molecule, randomSeed=19) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)
    names = {
        index: name
        for name, index in standard_pdb_atom_name_map(
            "ALA", template.smiles
        ).items()
    }
    conformer = molecule.GetConformer()
    lines = []
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(_pdb_line(
            atom.GetIdx() + 1,
            names[atom.GetIdx()],
            resname,
            (point.x, point.y, point.z),
            atom.GetSymbol(),
        ))
    path = tmp_path / "disguised-alanine.pdb"
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="ascii")
    return path, (resname, 1, True)


def _write_nitrile_nnaa(tmp_path, *, resname="ZZN"):
    full = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](CC#N)C(=O)O"))
    assert AllChem.EmbedMolecule(full, randomSeed=23) == 0
    AllChem.UFFOptimizeMolecule(full)
    full = Chem.RemoveHs(full)
    conformer = full.GetConformer()
    carbonyl_c = next(
        atom for atom in full.GetAtoms()
        if atom.GetSymbol() == "C" and any(
            neighbor.GetSymbol() == "O" for neighbor in atom.GetNeighbors()
        )
    )
    oxygens = [
        neighbor for neighbor in carbonyl_c.GetNeighbors()
        if neighbor.GetSymbol() == "O"
    ]
    carbonyl_o = next(
        oxygen for oxygen in oxygens
        if full.GetBondBetweenAtoms(
            carbonyl_c.GetIdx(), oxygen.GetIdx()
        ).GetBondType() == Chem.BondType.DOUBLE
    )
    hydroxyl_o = next(oxygen for oxygen in oxygens if oxygen is not carbonyl_o)
    n_index, ca_index, _cb_index, c_index, o_index = parse_backbone(full)
    backbone_n = full.GetAtomWithIdx(n_index)
    alpha_c = full.GetAtomWithIdx(ca_index)
    carbonyl_c = full.GetAtomWithIdx(c_index)
    carbonyl_o = full.GetAtomWithIdx(o_index)
    sidechain = next(
        atom for atom in alpha_c.GetNeighbors()
        if atom.GetIdx() not in {backbone_n.GetIdx(), carbonyl_c.GetIdx()}
    )
    nitrile_c = next(
        atom for atom in sidechain.GetNeighbors() if atom.GetIdx() != alpha_c.GetIdx()
    )
    nitrile_n = next(
        atom for atom in nitrile_c.GetNeighbors() if atom.GetIdx() != sidechain.GetIdx()
    )
    names = {
        backbone_n.GetIdx(): "N",
        alpha_c.GetIdx(): "CA",
        carbonyl_c.GetIdx(): "C",
        carbonyl_o.GetIdx(): "O",
        sidechain.GetIdx(): "CB",
        nitrile_c.GetIdx(): "CG",
        nitrile_n.GetIdx(): "NZ",
    }
    lines = []
    serial = 1
    for atom in full.GetAtoms():
        if atom.GetIdx() == hydroxyl_o.GetIdx():
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(_pdb_line(
            serial,
            names[atom.GetIdx()],
            resname,
            (point.x, point.y, point.z),
            atom.GetSymbol(),
        ))
        serial += 1
    path = tmp_path / "nitrile-nnaa.pdb"
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="ascii")
    return path, (resname, 1, True)


def _write_unknown_from_smiles(tmp_path, smiles, *, resname="ZZI", seed=59):
    row = build_derived_row("FIX", smiles, monomer_id=23000)[0]
    with _map_utils.isolated_monomer_registry(derived_rows=[row]):
        template = get_residue_template(row["symbol"])
        molecule = Chem.AddHs(template.mol)
        assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
        AllChem.UFFOptimizeMolecule(molecule)
        molecule = Chem.RemoveHs(molecule)
    n_idx, ca_idx, _cb_idx, c_idx, o_idx = parse_backbone(molecule)
    fixed_names = {n_idx: "N", ca_idx: "CA", c_idx: "C", o_idx: "O"}
    lines = []
    conformer = molecule.GetConformer()
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        name = fixed_names.get(atom.GetIdx(), f"X{atom.GetIdx()}")
        lines.append(_pdb_line(
            atom.GetIdx() + 1,
            name,
            resname,
            (point.x, point.y, point.z),
            atom.GetSymbol(),
        ))
    path = tmp_path / f"{resname.lower()}-unknown.pdb"
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="ascii")
    return path, (resname, 1, True)


def _write_same_code_two_graphs(tmp_path):
    lines = []
    serial = 1
    for resseq, standard, seed in ((1, "ALA", 61), (2, "VAL", 67)):
        template = get_residue_template(standard)
        molecule = Chem.AddHs(template.mol)
        assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
        AllChem.UFFOptimizeMolecule(molecule)
        molecule = Chem.RemoveHs(molecule)
        names = {
            index: name for name, index in standard_pdb_atom_name_map(
                standard, template.smiles
            ).items()
        }
        conformer = molecule.GetConformer()
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            lines.append(_pdb_line(
                serial,
                names[atom.GetIdx()],
                "ZZX",
                (point.x + resseq * 20.0, point.y, point.z),
                atom.GetSymbol(),
                resseq=resseq,
            ))
            serial += 1
    path = tmp_path / "same-code-two-graphs.pdb"
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="ascii")
    return path


def test_unique_local_graph_collapses_to_existing_unified_alias(tmp_path):
    path, key = _write_disguised_alanine(tmp_path)
    result = infer_residue_monomer(path, "A", key)
    assert result.unique, result.reason_codes
    assert result.candidate_graph_count == 1
    inferred = Chem.MolFromSmiles(result.candidate_smiles)
    reference = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    assert Chem.MolToInchiKey(inferred) == Chem.MolToInchiKey(reference)
    rows, manifests, aliases = build_derived_batch([{
        "pdb_resname": result.pdb_resname,
        "smiles": result.candidate_smiles,
        "input_sha256": result.evidence["input_sha256"],
    }])
    assert rows == manifests == []
    assert aliases[0]["target_symbol"] == "A"


def test_pdb_bootstrap_activates_existing_graph_as_isolated_alias(tmp_path):
    path, _key = _write_disguised_alanine(tmp_path)
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert bootstrap.ready
    assert bootstrap.derived_rows == []
    assert bootstrap.pdb_aliases[0]["pdb_resname"] == "ZZA"
    assert bootstrap.pdb_aliases[0]["target_symbol"] == "A"
    match = bootstrap.inference_results[0].evidence["library_first_match"]
    assert bootstrap.inference_results[0].evidence["resolution_mode"] == (
        "unified_library_match"
    )
    assert match["released"] is True
    assert match["selected_symbol"] == "A"
    assert "dA" not in match["equivalent_symbols"]
    assert match["selected_match_evidence"]["coordinate_stereochemistry"][
        "passed"
    ] is True
    with _map_utils.isolated_monomer_registry(
        derived_rows=bootstrap.derived_rows,
        pdb_aliases=bootstrap.pdb_aliases,
    ):
        assert resolve_symbol("ZZA") == "A"
    assert _map_utils.resolve_pdb_alias("ZZA") is None


def test_library_first_matches_existing_noncanonical_unified_graph(tmp_path):
    path, key = _write_nitrile_nnaa(tmp_path)
    result = infer_residue_monomer(path, "A", key)
    assert result.unique, (result.reason_codes, result.evidence)
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(result.candidate_smiles)) == (
        Chem.MolToInchiKey(Chem.MolFromSmiles("N[C@@H](CC#N)C(=O)O"))
    )
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert bootstrap.ready
    assert bootstrap.derived_rows == []
    assert bootstrap.inference_results[0].evidence["resolution_mode"] == (
        "unified_library_match"
    )
    assert bootstrap.inference_results[0].evidence["library_first_match"][
        "released"
    ] is True
    attempt = bootstrap.inference_results[0].evidence["library_first_match"]
    assert attempt["audited_connectivity_used_for_mapping"] is True
    assert attempt["selected_match_evidence"]["mapping_method"] == (
        "element_adjacency_audited_connectivity"
    )


def test_observed_oxt_is_reused_as_terminal_hydroxyl(tmp_path):
    path, key = _write_nitrile_nnaa(tmp_path, resname="ZZO")
    control = infer_residue_monomer(path, "A", key)
    assert control.unique, (control.reason_codes, control.evidence)

    lines = path.read_text(encoding="ascii").splitlines()
    atoms = {
        line[12:16].strip(): line
        for line in lines if line.startswith(("ATOM", "HETATM"))
    }
    carbon = tuple(float(atoms["C"][start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
    oxygen = tuple(float(atoms["O"][start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
    alpha_carbon = tuple(
        float(atoms["CA"][start:end])
        for start, end in ((30, 38), (38, 46), (46, 54))
    )
    c_to_o = tuple(oxygen[index] - carbon[index] for index in range(3))
    c_to_ca = tuple(alpha_carbon[index] - carbon[index] for index in range(3))
    o_length = sum(value * value for value in c_to_o) ** 0.5
    ca_length = sum(value * value for value in c_to_ca) ** 0.5
    vector = tuple(
        -(c_to_o[index] / o_length + c_to_ca[index] / ca_length)
        for index in range(3)
    )
    length = sum(value * value for value in vector) ** 0.5
    oxt = tuple(
        carbon[index] + 1.34 * vector[index] / length for index in range(3)
    )
    serial = max(
        int(line[6:11]) for line in lines if line.startswith(("ATOM", "HETATM"))
    ) + 1
    path.write_text(
        "\n".join([
            *[line for line in lines if line != "END"],
            _pdb_line(serial, "OXT", "ZZO", oxt, "O"),
            "END",
        ]) + "\n",
        encoding="ascii",
    )

    observed = infer_residue_monomer(path, "A", key)
    assert observed.unique, (observed.reason_codes, observed.evidence)
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(observed.candidate_smiles)) == (
        Chem.MolToInchiKey(Chem.MolFromSmiles(control.candidate_smiles))
    )
    ledgers = observed.evidence["chemistry_candidates"]
    assert ledgers and all(not row["terminal_hydroxyl_added"] for row in ledgers)
    assert all(row["terminal_hydroxyl_source"] == "observed_OXT" for row in ledgers)


def test_neutral_sidechain_amide_is_not_ionization_ambiguous(tmp_path):
    source = "N[C@@H](CC(=O)NCC)C(=O)O"
    path, key = _write_unknown_from_smiles(
        tmp_path, source, resname="ZZM", seed=103
    )
    result = infer_residue_monomer(path, "A", key)
    assert result.unique, (result.reason_codes, result.evidence)
    assert "IONIZATION_STATE_UNRESOLVED" not in result.reason_codes
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(result.candidate_smiles)) == (
        Chem.MolToInchiKey(Chem.MolFromSmiles(source))
    )


def test_blank_element_two_letter_halogen_recovers_exact_graph(tmp_path):
    source = "N[C@@H](CCBr)C(=O)O"
    path, key = _write_unknown_from_smiles(
        tmp_path, source, resname="ZZR", seed=107
    )
    rewritten = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[76:78].strip().upper() == "BR":
            line = line[:12] + f"{'BR':<4s}" + line[16:76] + "  " + line[78:]
        rewritten.append(line)
    path.write_text("\n".join(rewritten) + "\n", encoding="ascii")

    parsed = get_pdb_atoms(str(path), key, "A")
    bromine = next(atom for atom in parsed if atom["name"] == "BR")
    assert bromine["elem"] == "BR"
    result = infer_residue_monomer(path, "A", key)
    assert result.unique, (result.reason_codes, result.evidence)
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(result.candidate_smiles)) == (
        Chem.MolToInchiKey(Chem.MolFromSmiles(source))
    )


def test_truly_novel_unique_graph_builds_entity_local_235_column_row(tmp_path):
    smiles = "N[C@@H](CCBr)C(=O)O"
    path, _key = _write_unknown_from_smiles(
        tmp_path, smiles, resname="ZZQ", seed=101
    )
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert bootstrap.ready
    assert bootstrap.inference_results[0].evidence.get("resolution_mode") != (
        "unified_library_match"
    )
    assert bootstrap.inference_results[0].evidence["library_first_match"][
        "released"
    ] is False
    assert len(bootstrap.derived_rows) == 1
    assert bootstrap.derived_rows[0]["source"] == "local_structure_derived"
    assert len(bootstrap.derived_rows[0]) == 235
    symbol = bootstrap.derived_rows[0]["symbol"]
    assert bootstrap.pdb_aliases[0]["target_symbol"] == symbol
    with _map_utils.isolated_monomer_registry(
        derived_rows=bootstrap.derived_rows,
        pdb_aliases=bootstrap.pdb_aliases,
    ):
        assert resolve_symbol("ZZQ") == symbol


def test_missing_backbone_is_quarantined_with_structured_row(tmp_path):
    path, key = _write_disguised_alanine(tmp_path)
    text = path.read_text(encoding="ascii")
    path.write_text(
        "\n".join(line for line in text.splitlines() if line[12:16].strip() != "O")
        + "\n",
        encoding="ascii",
    )
    result = infer_residue_monomer(path, "A", key)
    assert not result.unique
    assert result.status == "quarantined"
    assert result.reason_codes == ["INCOMPLETE_OR_AMBIGUOUS_PEPTIDE_BACKBONE"]
    row = result.quarantine_row()
    assert row["status"] == "not_supported"
    assert row["candidate_graph_count"] == "0"
    assert row["input_sha256"] == result.evidence["input_sha256"]
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert not bootstrap.ready
    assert bootstrap.derived_rows == []
    assert bootstrap.pdb_aliases == []
    assert bootstrap.quarantine_rows[0]["reason_codes"] == (
        "INCOMPLETE_OR_AMBIGUOUS_PEPTIDE_BACKBONE"
    )


def test_far_explicit_edge_is_quarantined_as_conflict(tmp_path):
    path, key = _write_disguised_alanine(tmp_path)
    path.write_text(
        path.read_text(encoding="ascii").replace(
            "END\n", "CONECT    1    4\nEND\n"
        ),
        encoding="ascii",
    )
    result = infer_residue_monomer(path, "A", key)
    assert not result.unique
    assert result.reason_codes == ["EXPLICIT_GEOMETRY_CONFLICT"]
    assert result.evidence["connectivity"]["explicit_geometry_conflicts"]
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert not bootstrap.ready
    attempt = bootstrap.inference_results[0].evidence["library_first_match"]
    assert attempt["released"] is False
    assert attempt["fallback_reason_codes"] == ["EXPLICIT_GEOMETRY_CONFLICT"]


def test_ambiguous_sidechain_bond_order_is_never_registered(tmp_path):
    atoms = [
        (1, "N", (0.00, 0.00, 0.00), "N"),
        (2, "CA", (1.45, 0.00, 0.00), "C"),
        (3, "C", (2.20, 1.31, 0.00), "C"),
        (4, "O", (1.70, 2.43, 0.00), "O"),
        (5, "CB", (2.00, -0.80, 1.15), "C"),
        (6, "CG", (3.35, -0.65, 1.55), "C"),
    ]
    path = tmp_path / "ambiguous-order.pdb"
    path.write_text(
        "\n".join([
            *[
                _pdb_line(serial, name, "ZZB", xyz, element)
                for serial, name, xyz, element in atoms
            ],
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    result = infer_residue_monomer(path, "A", ("ZZB", 1, True))
    assert not result.unique
    assert result.status == "quarantined"
    assert result.reason_codes == [
        "AMBIGUOUS_CONSTITUTIONAL_OR_STEREOCHEMICAL_GRAPH"
    ]
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert not bootstrap.ready
    assert bootstrap.derived_rows == []


def test_explicit_disulfide_infers_exact_r3_dummy(tmp_path):
    path, key, serials = _write_unknown_cysteine_crosslink(tmp_path)
    result = infer_residue_monomer(path, "A", key)
    assert result.unique, (result.reason_codes, result.evidence)
    assert result.r3_port["pdb_serial"] == serials["SG"]
    assert result.r3_port["atom_name"] == "SG"
    assert result.r3_port["cap"] == "H"
    assert result.r3_port["evidence_source"] == "link"
    assert ":9003" in result.candidate_r3_mapped_smiles
    row, manifest = build_derived_row(
        "ZZS",
        result.candidate_smiles,
        monomer_id=22000,
        r3_mapped_smiles=result.candidate_r3_mapped_smiles,
        r3_port=result.r3_port,
    )
    molecule = Chem.MolFromSmiles(row["CXSMILES"])
    labels = {
        atom.GetProp("atomLabel"): atom
        for atom in molecule.GetAtoms() if atom.HasProp("atomLabel")
    }
    assert set(labels) == {"_R1", "_R2", "_R3"}
    assert labels["_R3"].GetNeighbors()[0].GetSymbol() == "S"
    assert row["R3"] == "H"
    assert manifest["r3_anchor_element"] == "S"
    assert manifest["r3_port_evidence"]["pdb_serial"] == serials["SG"]


def test_explicit_r3_can_use_unique_library_first_alias(tmp_path):
    path, _key, serials = _write_unknown_cysteine_crosslink(tmp_path)
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert bootstrap.ready
    assert bootstrap.derived_rows == []
    assert bootstrap.pdb_aliases[0]["target_symbol"] == "C"
    result = bootstrap.inference_results[0]
    assert result.evidence["resolution_mode"] == "unified_library_match"
    match = result.evidence["library_first_match"]
    assert match["released"] is True
    assert match["selected_match_evidence"]["external_attachment_indices"][
        str(serials["SG"])
    ] == match["selected_match_evidence"]["r3_anchor_template_atom_index"]
    assert result.r3_port["cap"] == "H"
    assert ":9003" in result.candidate_r3_mapped_smiles


def test_geometry_only_r3_is_not_authorized(tmp_path):
    path, key, _serials = _write_unknown_cysteine_crosslink(
        tmp_path, geometry_only=True
    )
    result = infer_residue_monomer(path, "A", key)
    assert not result.unique
    assert result.status == "not_supported"
    assert result.reason_codes == ["GEOMETRY_ONLY_R3_NOT_AUTHORIZED"]


def test_conect_only_disulfide_infers_exact_r3(tmp_path):
    path, key, serials = _write_unknown_cysteine_crosslink(
        tmp_path, record_type="conect"
    )
    result = infer_residue_monomer(path, "A", key)
    assert result.unique, (result.reason_codes, result.evidence)
    assert result.r3_port["pdb_serial"] == serials["SG"]
    assert result.r3_port["evidence_source"] == "conect"
    assert result.r3_port["cap"] == "H"


def test_ssbond_disulfide_infers_exact_r3(tmp_path):
    path, key, serials = _write_unknown_cysteine_crosslink(
        tmp_path, record_type="ssbond"
    )
    result = infer_residue_monomer(path, "A", key)
    assert result.unique, (result.reason_codes, result.evidence)
    assert result.r3_port["pdb_serial"] == serials["SG"]
    assert result.r3_port["evidence_source"] == "ssbond"
    assert result.r3_port["cap"] == "H"


def test_reused_r3_port_is_rejected(tmp_path):
    path, key, _serials = _write_unknown_cysteine_crosslink(tmp_path, links=2)
    result = infer_residue_monomer(path, "A", key)
    assert not result.unique
    assert result.status == "rejected"
    assert result.reason_codes == ["R3_PORT_REUSED_BY_MULTIPLE_PARTNERS"]
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert bootstrap.status == "rejected"
    assert bootstrap.quarantine_rows[0]["status"] == "rejected"


def test_two_distinct_r3_atoms_are_not_supported(tmp_path):
    path, key, _serials = _write_unknown_cysteine_crosslink(
        tmp_path, links=2, second_local_atom="CB"
    )
    result = infer_residue_monomer(path, "A", key)
    assert not result.unique
    assert result.status == "not_supported"
    assert result.reason_codes == ["MULTIPLE_R3_PORTS_NOT_REPRESENTABLE"]
    observed = result.evidence["ports"]["R3"]
    assert observed["representation"] == "multiple_independent_r3_ports"
    assert len(observed["observed_ports"]) == 2
    assert len({row["pdb_serial"] for row in observed["observed_ports"]}) == 2


def test_flat_stereocenter_is_quarantined(tmp_path):
    atoms = [
        (1, "N", (0.00, 0.00, 0.00), "N"),
        (2, "CA", (1.45, 0.00, 0.00), "C"),
        (3, "C", (2.20, 1.31, 0.00), "C"),
        (4, "O", (1.70, 2.43, 0.00), "O"),
        (5, "CB", (2.00, -1.30, 0.00), "C"),
    ]
    path = tmp_path / "flat-stereocenter.pdb"
    path.write_text(
        "\n".join([
            *[
                _pdb_line(serial, name, "ZZF", xyz, element)
                for serial, name, xyz, element in atoms
            ],
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    result = infer_residue_monomer(path, "A", ("ZZF", 1, True))
    assert not result.unique
    assert "UNRESOLVED_STEREOCHEMISTRY" in result.reason_codes
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert not bootstrap.ready
    attempt = bootstrap.inference_results[0].evidence["library_first_match"]
    assert attempt["released"] is False


def test_sidechain_amine_ionization_ambiguity_is_quarantined(tmp_path):
    path, key = _write_unknown_from_smiles(
        tmp_path, "N[C@@H](CCCCCN)C(=O)O"
    )
    result = infer_residue_monomer(path, "A", key)
    assert not result.unique
    assert result.status == "quarantined"
    assert result.reason_codes == ["IONIZATION_STATE_UNRESOLVED"]


def test_same_pdb_code_with_two_local_graphs_is_quarantined(tmp_path):
    path = _write_same_code_two_graphs(tmp_path)
    bootstrap = bootstrap_unknown_monomers(path, "A")
    assert not bootstrap.ready
    assert bootstrap.status == "quarantined"
    assert bootstrap.derived_rows == []
    assert bootstrap.pdb_aliases == []
    assert {row["reason_codes"] for row in bootstrap.quarantine_rows} == {
        "PDB_RESNAME_MAPS_TO_MULTIPLE_LOCAL_GRAPHS"
    }


def test_derived_batch_is_processing_order_independent():
    candidates = [
        {"pdb_resname": "ONE", "smiles": "N[C@@H](CC#N)C(=O)O"},
        {"pdb_resname": "TWO", "smiles": "N[C@@H](CCCl)C(=O)O"},
    ]
    forward = build_derived_batch(candidates)
    reverse = build_derived_batch(reversed(candidates))
    assert forward == reverse


def test_unified_alias_requires_matching_r3_anchor_and_cap():
    import pytest

    aspartate = "N[C@@H](CC(=O)O)C(=O)O"
    correct = "N[C@@H](C[C:9003](=O)O)C(=O)O"
    rows, manifests, aliases = build_derived_batch([{
        "pdb_resname": "ASP_OK",
        "smiles": aspartate,
        "r3_mapped_smiles": correct,
        "r3_port": {"cap": "OH"},
    }])
    assert rows == []
    assert manifests == []
    assert aliases[0]["target_symbol"] == "D"

    wrong = "N[C@@H]([CH2:9003]C(=O)O)C(=O)O"
    with pytest.raises(PortSemanticMismatchError, match="R3 anchor/cap differs"):
        build_derived_batch([{
            "pdb_resname": "ASP_WRONG",
            "smiles": aspartate,
            "r3_mapped_smiles": wrong,
            "r3_port": {"cap": "H"},
        }])

    rows, manifests, aliases = build_derived_batch([{
        "pdb_resname": "ASP_CCD",
        "smiles": aspartate,
        "r3_mapped_smiles": wrong,
        "r3_port": {"cap": "H"},
        "resolution_mode": "embedded_mmcif_chem_comp",
        "component_snapshot_sha256": "1" * 64,
        "component_source_input_sha256": "2" * 64,
    }])
    assert len(rows) == len(manifests) == 1
    assert aliases == []
    assert manifests[0]["resolution_mode"] == "embedded_mmcif_chem_comp"
    assert rows[0]["symbol"].startswith("LCL_ASP_CCD_")
