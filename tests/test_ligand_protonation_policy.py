"""Focused tests for the strict molecule-level pH-7.4 microstate policy.

These tests cover the docking/protonation.py strict helper
``protonate_molecule_ph74`` (rule correctness, the tertiary-amine H1 fix,
conservative neutral sites, preservation guarantees, strict failure
surfacing) and the legacy ``protonate_ph74`` facade contract.

No benchmark truth, deposited reference, or frozen artifact is read at
runtime; the optional local ligand-parent probe only checks arithmetic
self-consistency (charge conservation and heavy-atom preservation), never
reference charge targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cycpep_master.docking.protonation import (
    ProtonationPolicyError,
    WARNING_CHARGED_HISTIDINE_PRESERVED,
    WARNING_HISTIDINE_NEUTRAL,
    WARNING_ISOTOPE_SITE_SKIPPED,
    WARNING_THIOL_PHENOL_NEUTRAL,
    protonate_molecule_ph74,
    protonate_ph74,
)

_STAGE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

_EVIDENCE_PARENTS = {
    case: (
        _STAGE_PACKAGE_ROOT
        / ".zcode_cocrystal_execution_002"
        / shard
        / f"case_{case}"
        / "arm_A"
        / "ligand_A_parent_normalized.mol"
    )
    for case, shard in (
        ("6q1u", "run_001_shard1"),
        ("6u8g", "run_001_shard2"),
        ("5vb9", "run_001_shard2"),
    )
}


def _apply(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return protonate_molecule_ph74(molecule)


def _output_smiles(smiles: str) -> str:
    output, _report = _apply(smiles)
    return Chem.MolToSmiles(output)


def _charge(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.GetFormalCharge(molecule)


class TestLegacyFacadeContract:
    @pytest.mark.parametrize(
        ("source", "expected"),
        (
            ("CC(=O)O", "CC(=O)[O-]"),
            ("NCC(=O)O", "[NH3+]CC(=O)[O-]"),
            ("NCCCCN", "[NH3+]CCCC[NH3+]"),
            ("NC(=N)N", "NC(N)=[NH2+]"),
            ("CS(=O)(=O)O", "CS(=O)(=O)[O-]"),
            ("OP(=O)(O)O", "O=P([O-])([O-])O"),
            ("c1ncc[nH]1", "c1c[nH]cn1"),
            ("not-smiles", "not-smiles"),
        ),
    )
    def test_legacy_fixtures_unchanged(self, source, expected):
        assert protonate_ph74(source) == expected

    def test_legacy_facade_delegates_to_strict_engine(self):
        output, report = _apply("NCC(=O)O")
        assert Chem.MolToSmiles(output) == "[NH3+]CC(=O)[O-]"
        assert report["policy"] == "physiological"
        assert report["ph"] == 7.4


class TestAmineRules:
    def test_primary_amine_protonated(self):
        output, report = _apply("CCN")
        assert Chem.MolToSmiles(output) == "CC[NH3+]"
        entry = report["changed_atoms"][0]
        assert entry["rule"] == "primary_aliphatic_amine_protonated"
        assert entry["total_h"] == [2, 3]

    def test_secondary_amine_protonated(self):
        assert _output_smiles("CCNC(C)C") == "CC[NH2+]C(C)C"

    def test_tertiary_amine_protonated_with_exactly_one_hydrogen(self):
        output, report = _apply("CCN(CC)CC")
        assert Chem.MolToSmiles(output) == "CC[NH+](CC)CC"
        entry = report["changed_atoms"][0]
        assert entry["rule"] == "tertiary_aliphatic_amine_protonated"
        assert entry["total_h"] == [0, 1]
        assert Chem.GetFormalCharge(output) == 1

    def test_tertiary_amine_h1_fix_on_noimplicit_geometry_style_mol(self):
        """Regression: the legacy rule wrote charge +1 with H0, which on
        NoImplicit atoms (the geometry-path parent style) produces a
        trivalent nitrenium cation instead of a protonated amine."""
        editable = Chem.RWMol()
        for element in ("N", "C", "C", "C"):
            editable.AddAtom(Chem.Atom(element))
        nitrogen = editable.GetAtomWithIdx(0)
        nitrogen.SetNoImplicit(True)
        nitrogen.SetNumExplicitHs(0)
        for other in (1, 2, 3):
            editable.AddBond(0, other, Chem.BondType.SINGLE)
        molecule = editable.GetMol()
        molecule.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(molecule)
        assert molecule.GetAtomWithIdx(0).GetTotalNumHs() == 0

        output, report = protonate_molecule_ph74(molecule)
        nitrogen_out = output.GetAtomWithIdx(0)
        assert nitrogen_out.GetFormalCharge() == 1
        assert nitrogen_out.GetTotalNumHs() == 1
        assert Chem.MolToSmiles(output) == "C[NH+](C)C"
        assert report["changed_atoms"][0]["total_h"] == [0, 1]

    def test_quaternary_ammonium_unchanged_zero_hydrogens(self):
        output, report = _apply("C[N+](C)(C)C")
        assert Chem.MolToSmiles(output) == "C[N+](C)(C)C"
        assert report["changed_atoms"] == []
        atom = output.GetAtomWithIdx(1)
        assert atom.GetFormalCharge() == 1
        assert atom.GetTotalNumHs() == 0

    def test_amide_nitrogen_excluded(self):
        output, report = _apply("CCNC(C)=O")
        assert report["changed_atoms"] == []
        assert _charge("CCNC(C)=O") == 0

    def test_sulfonamide_nitrogen_excluded(self):
        output, report = _apply("CS(=O)(=O)NC")
        assert report["changed_atoms"] == []
        assert "N" not in [entry["element"] for entry in report["changed_atoms"]]


class TestAcidRules:
    def test_carboxylic_acid_deprotonated(self):
        output, report = _apply("CCC(=O)O")
        assert Chem.MolToSmiles(output) == "CCC(=O)[O-]"
        assert report["changed_atoms"][0]["rule"] == "carboxylic_acid_deprotonated"

    def test_dicarboxylic_acid_both_deprotonated(self):
        output, _report = _apply("OC(=O)C(=O)O")
        assert Chem.GetFormalCharge(output) == -2

    def test_phosphoric_acid_at_most_two(self):
        output, report = _apply("OP(=O)(O)O")
        assert Chem.GetFormalCharge(output) == -2
        rules = [entry["rule"] for entry in report["changed_atoms"]]
        assert rules.count("phosphoric_acid_deprotonated") == 2

    def test_sulfonic_acid_deprotonated(self):
        assert _output_smiles("CS(=O)(=O)O") == "CS(=O)(=O)[O-]"

    def test_already_deprotonated_carboxylate_preserved(self):
        output, report = _apply("CCC(=O)[O-]")
        assert report["changed_atoms"] == []
        assert Chem.GetFormalCharge(output) == -1

    def test_salt_components_and_non_target_ions_preserved(self):
        smiles = "CC(=O)[O-].[Na+]"
        output, report = _apply(smiles)
        assert report["changed_atoms"] == []
        assert len(Chem.GetMolFrags(output)) == 2
        assert Chem.GetFormalCharge(output) == 0
        symbols = sorted(atom.GetSymbol() for atom in output.GetAtoms())
        assert symbols == ["C", "C", "Na", "O", "O"]


class TestGuanidineAmidineRules:
    def test_guanidine_protonated_on_imino_nitrogen(self):
        output, report = _apply("CCCNC(=N)N")
        assert Chem.MolToSmiles(output) == "CCCNC(N)=[NH2+]"
        entry = report["changed_atoms"][0]
        assert entry["rule"] == "guanidine_protonated"
        assert entry["total_h"] == [1, 2]

    def test_already_protonated_guanidinium_preserved(self):
        output, report = _apply("CCCNC(=[NH2+])N")
        assert report["changed_atoms"] == []
        assert Chem.GetFormalCharge(output) == 1

    def test_acetylated_lysine_sidechain_amide_untouched(self):
        """6u8g carries ALY (N-epsilon-acetyl-lysine): the side-chain amide
        nitrogen must stay neutral while the free lysine amine protonates."""
        smiles = "NCCCCNC(C)=O"
        output, report = _apply(smiles)
        rules = {entry["rule"] for entry in report["changed_atoms"]}
        assert rules == {"primary_aliphatic_amine_protonated"}
        assert Chem.GetFormalCharge(output) == 1


class TestConservativeNeutralSites:
    def test_neutral_histidine_reported_not_modified(self):
        output, report = _apply("Cc1c[nH]cn1")
        assert report["changed_atoms"] == []
        assert WARNING_HISTIDINE_NEUTRAL in report["warnings"]
        assert report["conservative_neutral_sites"]["histidine_imidazole_like"]

    def test_charged_histidine_preserved_and_reported(self):
        smiles = "Cc1c[nH]c[nH+]1"
        output, report = _apply(smiles)
        assert WARNING_CHARGED_HISTIDINE_PRESERVED in report["warnings"]
        assert report["preserved_charged_histidine_like"]
        # The azolium nitrogen is neither deprotonated nor further changed.
        nitrogen_indices = report["preserved_charged_histidine_like"]
        for index in nitrogen_indices:
            before = Chem.MolFromSmiles(smiles).GetAtomWithIdx(index)
            after = output.GetAtomWithIdx(index)
            assert before.GetFormalCharge() == after.GetFormalCharge() == 1
        assert Chem.GetFormalCharge(output) == 1

    def test_thiol_and_phenol_neutral_conservative(self):
        for smiles in ("CCS", "Cc1ccccc1O"):
            output, report = _apply(smiles)
            assert report["changed_atoms"] == []
            assert WARNING_THIOL_PHENOL_NEUTRAL in report["warnings"]


class TestStrictFailures:
    def test_non_mol_input_raises(self):
        with pytest.raises(ProtonationPolicyError):
            protonate_molecule_ph74("CCO")

    def test_empty_mol_raises(self):
        with pytest.raises(ProtonationPolicyError):
            protonate_molecule_ph74(Chem.Mol())

    def test_unsanitizable_mol_raises(self):
        editable = Chem.RWMol()
        carbon = Chem.Atom("C")
        carbon.SetNoImplicit(True)
        carbon.SetNumExplicitHs(0)
        editable.AddAtom(carbon)
        for _ in range(5):
            editable.AddAtom(Chem.Atom("Cl"))
        for other in range(1, 6):
            editable.AddBond(0, other, Chem.BondType.SINGLE)
        molecule = editable.GetMol()
        molecule.UpdatePropertyCache(strict=False)
        with pytest.raises(ValueError):
            protonate_molecule_ph74(molecule)


class TestReportContract:
    def test_report_is_json_ready_with_policy_fields(self):
        _output, report = _apply("NCCCCNC(=N)C(=O)O")
        payload = json.dumps(report)
        assert payload
        assert report["policy"] == "physiological"
        assert report["ph"] == 7.4
        assert report["input_formal_charge"] == 0
        assert report["output_formal_charge"] == 1
        assert report["claim_boundary"]["forbidden"]
        assert "not experimentally validated" in report["rule_basis"]
        for entry in report["changed_atoms"]:
            assert set(entry) == {
                "atom_index",
                "input_atom_index",
                "element",
                "rule",
                "formal_charge",
                "total_h",
            }
            assert len(entry["formal_charge"]) == 2
            assert len(entry["total_h"]) == 2
        assert "atom_index" in report["indexing"]
        assert "input_atom_index" in report["indexing"]
        # Implicit inputs: folded index equals the input index.
        assert report["changed_atoms"][0]["atom_index"] == report["changed_atoms"][0]["input_atom_index"]

    def test_charge_arithmetic_is_consistent(self):
        smiles = "NCCCCNC(=N)C(=O)O"
        _output, report = _apply(smiles)
        delta = sum(
            entry["formal_charge"][1] - entry["formal_charge"][0]
            for entry in report["changed_atoms"]
        )
        assert (
            report["output_formal_charge"]
            == report["input_formal_charge"] + delta
        )


class TestPreservationGuarantees:
    @staticmethod
    def _embedded_l_alanine():
        molecule = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
        assert AllChem.EmbedMolecule(molecule, randomSeed=17) == 0
        info = Chem.AtomPDBResidueInfo()
        info.SetName(" CA ")
        info.SetResidueName("ALA")
        info.SetResidueNumber(1)
        info.SetChainId("L")
        molecule.GetAtomWithIdx(1).SetPDBResidueInfo(info)
        return molecule

    def test_input_molecule_not_modified(self):
        molecule = self._embedded_l_alanine()
        before_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)))
        before_charge = Chem.GetFormalCharge(molecule)
        protonate_molecule_ph74(molecule)
        assert Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule))) == before_smiles
        assert Chem.GetFormalCharge(molecule) == before_charge

    def test_explicit_hydrogen_input_preservation_and_stereo(self):
        molecule = self._embedded_l_alanine()
        heavy_before = [
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() != 1
        ]
        positions_before = [
            tuple(molecule.GetConformer().GetAtomPosition(index))
            for index in heavy_before
        ]
        tag_before = molecule.GetAtomWithIdx(1).GetChiralTag()

        output, report = protonate_molecule_ph74(molecule)

        # Heavy order, coordinates, fragments, PDB metadata preserved.
        heavy_after = [
            atom.GetIdx()
            for atom in output.GetAtoms()
            if atom.GetAtomicNum() != 1
        ]
        assert heavy_after == heavy_before
        assert [
            tuple(output.GetConformer().GetAtomPosition(index))
            for index in heavy_after
        ] == positions_before
        assert len(Chem.GetMolFrags(output)) == 1
        info = output.GetAtomWithIdx(1).GetPDBResidueInfo()
        assert info is not None and info.GetName().strip() == "CA"

        # Chiral tag survives; the CIP code must still be assigned.
        assert output.GetAtomWithIdx(1).GetChiralTag() == tag_before
        probe = Chem.Mol(output)
        Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
        assert probe.GetAtomWithIdx(1).GetProp("_CIPCode")

        # Net hydrogen count is conserved: amine +1 H, acid -1 H.
        assert output.GetNumAtoms() == molecule.GetNumAtoms()
        assert report["explicit_hydrogens_rebuilt"] is True
        assert report["preservation"]["heavy_coordinates_unchanged"] is True

        # Chemistry: zwitterionic alanine with the original stereocenter.
        output_no_h = Chem.RemoveHs(Chem.Mol(output))
        charges = {
            atom.GetIdx(): atom.GetFormalCharge()
            for atom in output_no_h.GetAtoms()
            if atom.GetFormalCharge()
        }
        assert sum(charges.values()) == 0
        nitrogen = next(
            atom for atom in output_no_h.GetAtoms() if atom.GetAtomicNum() == 7
        )
        assert nitrogen.GetFormalCharge() == 1
        assert nitrogen.GetTotalNumHs() == 3

    def test_multi_fragment_molecule_all_components_retained(self):
        smiles = "CC(=O)O.CCN"
        output, report = _apply(smiles)
        assert len(Chem.GetMolFrags(output)) == 2
        assert Chem.GetFormalCharge(output) == 0
        assert len(report["changed_atoms"]) == 2


class TestReviewGapContracts:
    """Provenance/indexing, bond-topology, and isotope-site guarantees."""

    @staticmethod
    def _interleaved_ethylamine():
        """C H H H C H H N H H: protium interleaved with heavy atoms."""
        editable = Chem.RWMol()
        for element in ("C", "H", "H", "H", "C", "H", "H", "N", "H", "H"):
            atom = Chem.Atom(element)
            atom.SetNoImplicit(element == "H")
            if element != "H":
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(0)
            editable.AddAtom(atom)
        editable.AddBond(0, 4, Chem.BondType.SINGLE)   # C-C
        editable.AddBond(4, 7, Chem.BondType.SINGLE)   # C-N
        for h, heavy in ((1, 0), (2, 0), (3, 0), (5, 4), (6, 4), (8, 7), (9, 7)):
            editable.AddBond(h, heavy, Chem.BondType.SINGLE)
        molecule = editable.GetMol()
        molecule.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(molecule)
        return molecule

    def test_input_atom_index_mapping_with_interleaved_hydrogens(self):
        molecule = self._interleaved_ethylamine()
        assert molecule.GetAtomWithIdx(7).GetAtomicNum() == 7  # N at input 7
        output, report = protonate_molecule_ph74(molecule)
        entry = report["changed_atoms"][0]
        assert entry["element"] == "N"
        assert entry["rule"] == "primary_aliphatic_amine_protonated"
        assert entry["input_atom_index"] == 7
        assert entry["atom_index"] == 2  # heavy relative order: C, C, N
        nitrogen = output.GetAtomWithIdx(entry["atom_index"])
        attached = sum(
            1
            for neighbour in nitrogen.GetNeighbors()
            if neighbour.GetAtomicNum() == 1 and neighbour.GetIsotope() < 2
        )
        assert attached + nitrogen.GetTotalNumHs() == 3

    def test_survivor_bond_topology_verified_aromatic(self):
        output, report = _apply("Cc1c[nH]cn1")
        assert report["preservation"]["heavy_bond_topology"] is True
        input_mol = Chem.MolFromSmiles("Cc1c[nH]cn1")

        def bond_type_counts(mol):
            return sorted(
                bond.GetBondType().name for bond in mol.GetBonds()
            )

        assert bond_type_counts(output) == bond_type_counts(input_mol)

    def test_isotope_hydrogen_site_refused_and_reported(self):
        editable = Chem.RWMol()
        for element in ("C", "N", "N", "N"):
            atom = Chem.Atom(element)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            editable.AddAtom(atom)
        deuterium = Chem.Atom("H")
        deuterium.SetIsotope(2)
        deuterium.SetNoImplicit(True)
        editable.AddAtom(deuterium)
        editable.AddBond(0, 1, Chem.BondType.DOUBLE)   # C=N(D)
        editable.AddBond(0, 2, Chem.BondType.SINGLE)   # C-NH2
        editable.AddBond(0, 3, Chem.BondType.SINGLE)   # C-NH2
        editable.AddBond(1, 4, Chem.BondType.SINGLE)   # N-D
        editable.GetAtomWithIdx(2).SetNumExplicitHs(2)
        editable.GetAtomWithIdx(3).SetNumExplicitHs(2)
        molecule = editable.GetMol()
        molecule.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(molecule)

        output, report = protonate_molecule_ph74(molecule)
        assert report["changed_atoms"] == []
        assert WARNING_ISOTOPE_SITE_SKIPPED in report["warnings"]
        assert report["skipped_isotope_hydrogen_sites"]
        # The deuterium atom itself is preserved.
        assert sum(
            1
            for atom in output.GetAtoms()
            if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 2
        ) == 1
        assert Chem.GetFormalCharge(output) == 0


    @pytest.mark.parametrize("case", sorted(_EVIDENCE_PARENTS))
    def test_parent_mol_policy_is_self_consistent(self, case):
        """Development-evidence probe (read-only): the policy applies to the
        real co-crystal parent molecules and preserves heavy geometry; the
        assertion is arithmetic self-consistency only, never a benchmark
        reference charge target."""
        path = _EVIDENCE_PARENTS[case]
        if not path.is_file():
            pytest.skip(f"local evidence parent missing: {path}")
        molecule = Chem.MolFromMolFile(str(path), removeHs=False)
        if molecule is None:
            pytest.skip("parent MOL not parseable in this environment")

        heavy_before = [
            (atom.GetAtomicNum(), atom.GetIsotope())
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() != 1
        ]
        positions_before = [
            tuple(molecule.GetConformer().GetAtomPosition(atom.GetIdx()))
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() != 1
        ]
        output, report = protonate_molecule_ph74(molecule)

        heavy_after = [
            (atom.GetAtomicNum(), atom.GetIsotope())
            for atom in output.GetAtoms()
            if atom.GetAtomicNum() != 1
        ]
        assert heavy_after == heavy_before
        assert [
            tuple(output.GetConformer().GetAtomPosition(atom.GetIdx()))
            for atom in output.GetAtoms()
            if atom.GetAtomicNum() != 1
        ] == positions_before
        assert len(Chem.GetMolFrags(output)) == len(Chem.GetMolFrags(molecule))

        delta = sum(
            entry["formal_charge"][1] - entry["formal_charge"][0]
            for entry in report["changed_atoms"]
        )
        assert (
            report["output_formal_charge"]
            == report["input_formal_charge"] + delta
        )
        # Every changed atom's hydrogen target was reached exactly.
        for entry in report["changed_atoms"]:
            atom = output.GetAtomWithIdx(entry["atom_index"])
            assert atom.GetFormalCharge() == entry["formal_charge"][1]
