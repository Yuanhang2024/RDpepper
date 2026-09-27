"""Linear-topology opt-in contract tests.

``allow_linear_topology`` is a keyword-only switch threading through the
application -> reconstruction -> remediation_v6 strict entry -> remediation_v5
input audit chain. When off (the default) a selected chain with neither
explicit nor geometrically-inferred cyclization evidence is rejected with
``V5_NO_CYCLIZATION_EVIDENCE``. When on, such a chain is accepted as
``topology_class="linear"`` and, if the v6 evidence dimensions otherwise
adjudicate, emitted as a qualified-strict success whose provenance records the
linear classification at the top level (never inside the frozen path-a/e
generation-provenance dicts that remediation_v6 replays by exact equality).
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from cycpep_master import application
from cycpep_master import reconstruction
from cycpep_master import remediation_v5
from cycpep_master.remediation_v6 import reconstruct_structure_fail_closed_v6

from .conftest import canonical

# A 3-residue linear Ala-Gly-Ala tripeptide with a free C-terminal OXT,
# selected-chain A, written with valid PDB column geometry so remediation_v5's
# residue-bond geometry audit and the geometric cyclization scan both accept a
# normal open chain and find no closure.  It carries no CONECT/SSBOND/LINK
# records and its N/C termini are far enough apart that no geometric closure
# candidate appears.
_LINEAR_AGA = """\
ATOM      1    N ALA L   1       0.883  -1.372  -1.785                       N
ATOM      2   CA ALA L   1       2.167  -1.467  -1.047                       C
ATOM      3    C ALA L   1       1.991  -0.810   0.338                       C
ATOM      4    O ALA L   1       2.759   0.021   0.807                       O
ATOM      5   CB ALA L   1       3.296  -0.827  -1.843                       C
TER
ATOM      6    N GLY L   2       0.868  -1.275   0.995                       N
ATOM      7   CA GLY L   2       0.410  -0.702   2.250                       C
ATOM      8    C GLY L   2      -0.699   0.344   2.072                       C
ATOM      9    O GLY L   2      -1.346   0.775   3.024                       O
TER
ATOM     10    N ALA L   3      -0.883   0.789   0.779                       N
ATOM     11   CA ALA L   3      -2.047   1.596   0.423                       C
ATOM     12    C ALA L   3      -3.243   0.719   0.035                       C
ATOM     13    O ALA L   3      -4.361   1.117  -0.257                       O
ATOM     14   CB ALA L   3      -1.701   2.513  -0.741                       C
ATOM     15  OXT ALA L   3      -2.967  -0.604  -0.029                       O
TER
END
"""


def _linear_pdb(tmp_path):
    path = tmp_path / "linear_tripeptide.pdb"
    path.write_text(_LINEAR_AGA, encoding="ascii")
    return path


def _heavy_atoms_in_pdb(path):
    """Count observed heavy atoms in an ATOM/HETATM PDB chain-agnostic file."""
    count = 0
    for line in path.read_text(encoding="ascii").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        element = line[76:78].strip()
        if element and element.upper() != "H":
            count += 1
    return count


def _heavy_atoms_in_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"linear output unparseable: {smiles!r}"
    return mol.GetNumHeavyAtoms()


def test_default_off_rejects_closure_less_chain(tmp_path):
    """Flag off: a linear tripeptide PDB still fails the fail-closed gate."""
    pdb = _linear_pdb(tmp_path)

    validation = remediation_v5.validate_pdb_reconstruction_input_v5(pdb, "L")
    assert validation.accepted is False
    assert "V5_NO_CYCLIZATION_EVIDENCE" in validation.warning_codes

    strict = reconstruct_structure_fail_closed_v6(pdb, "L")
    assert strict.status == "rejected"
    assert "V5_NO_CYCLIZATION_EVIDENCE" in strict.warning_codes

    envelope = application.reconstruct_structure(pdb, mode="strict")
    assert envelope["status"] == "failed"
    assert "V5_NO_CYCLIZATION_EVIDENCE" in envelope["data"]["warning_codes"]


def test_allow_linear_accepts_closure_less_chain(
    tmp_path,
):
    """Flag on: same input is a qualified linear strict success."""
    pdb = _linear_pdb(tmp_path)

    validation = remediation_v5.validate_pdb_reconstruction_input_v5(
        pdb, "L", allow_linear_topology=True
    )
    assert validation.accepted is True
    context = validation.context
    assert context["topology_class"] == "linear"
    assert context["cyclization_bonds"] == []
    assert "CONNECTIVITY_INFERRED_FROM_COORDINATES" not in context["repair_codes"]

    strict = reconstruct_structure_fail_closed_v6(
        pdb, "L", allow_linear_topology=True
    )
    assert strict.status == "success"
    assert strict.support_status == "qualified"
    assert strict.qualified_success is True
    assert strict.input_evidence["topology_class"] == "linear"
    assert strict.output_smiles
    assert strict.warning_codes == []
    family = strict.output_evidence["evidence_dimensions"][
        "independent_family_consensus"
    ]
    assert family["passed"] is True
    assert family["required"] is False
    diagnostic = strict.output_evidence["evidence_dimensions"][
        "diagnostic_identity_consistency"
    ]
    assert diagnostic["passed"] is True
    assert diagnostic["required"] is False
    assert diagnostic["observed_identity_consistency"] is False

    result = reconstruction.reconstruct_structure(
        pdb, chain_id="L", mode="strict", allow_linear_topology=True
    )
    assert result.status == "success"
    assert result.quality == "exact"
    assert result.smiles
    assert result.provenance["topology_class"] == "linear"

    envelope = application.reconstruct_structure(
        pdb, mode="strict", allow_linear_topology=True
    )
    assert envelope["status"] == "success"
    data = envelope["data"]
    assert data["provenance"]["topology_class"] == "linear"

    # The reconstruct_unified compatibility alias resolves identically.
    alias = application.reconstruct_unified(
        pdb, mode="strict", allow_linear_topology=True
    )
    assert alias["status"] == "success"
    assert alias["data"]["provenance"]["topology_class"] == "linear"
    # The assembly of the tripeptide (N-term open, C-term free acid + OXT)
    # must conserve the exact observed heavy-atom inventory.
    assert _heavy_atoms_in_pdb(pdb) == 15
    assert _heavy_atoms_in_smiles(data["smiles"]) == 15
    # A genuine linear chain must not report any ring.
    assert Chem.MolFromSmiles(data["smiles"]).GetRingInfo().NumRings() == 0


_CYCLIC_TOPOLOGY_CLASSES = {"monocyclic", "bicyclic", "tricyclic", "polycyclic"}


def _first_gold_cyclic(pdb_files):
    """Pick the first gold PDB that reconstructs as a cyclic strict success."""
    for pdb in pdb_files:
        strict = reconstruct_structure_fail_closed_v6(pdb, "L")
        if (
            strict.status == "success"
            and strict.output_smiles
            and strict.input_evidence.get("topology_class") in _CYCLIC_TOPOLOGY_CLASSES
        ):
            return pdb, strict
    pytest.skip("no gold-standard cyclic PDB reconstructs through strict V6")


def test_cyclic_gold_unaffected_by_flag(pdb_files):
    """Flag on/off must be byte-identical on a gold cyclic peptide."""
    pdb, strict = _first_gold_cyclic(pdb_files)
    base_canonical = canonical(strict.output_smiles)
    base_topology = strict.input_evidence["topology_class"]

    with_flag = reconstruct_structure_fail_closed_v6(
        pdb, "L", allow_linear_topology=True
    )
    assert with_flag.status == "success"
    assert canonical(with_flag.output_smiles) == base_canonical
    assert with_flag.input_evidence["topology_class"] == base_topology
    assert with_flag.input_evidence["topology_class"] != "linear"


def _strip_closure_records(src, dst):
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith(("CONECT", "SSBOND", "LINK")):
                continue
            fout.write(line)


def test_stripped_cyclic_never_qualified_linear(pdb_files, tmp_path):
    """A gold cyclic PDB with closure records stripped must not become linear.

    With the flag on, such a chain must still be classified cyclically (via
    the geometric cyclization path) or rejected -- never a qualified linear
    success whose SMILES would silently flatten a macrocycle.
    """
    pdb, strict = _first_gold_cyclic(pdb_files)
    base_canonical = canonical(strict.output_smiles)

    stripped = tmp_path / "stripped_cyclic.pdb"
    _strip_closure_records(pdb, stripped)

    # The input audit still classifies this as cyclic via the geometric scan,
    # so the linear branch must not engage.
    result = reconstruct_structure_fail_closed_v6(
        str(stripped), "L", allow_linear_topology=True
    )
    assert result.input_evidence.get("topology_class") != "linear"
    if result.status == "success":
        # If v6 still adjudicates, it must be the same cyclic macrocycle, not
        # a flattened linear chain.
        assert canonical(result.output_smiles) == base_canonical
        assert result.input_evidence.get("topology_class") in _CYCLIC_TOPOLOGY_CLASSES


def test_mixed_batch_classifies_each_input_correctly(pdb_files, tmp_path):
    """Alternating linear + cyclic inputs, flag on, classify per input."""
    linear_pdb = _linear_pdb(tmp_path)
    cyclic_pdb, cyclic_strict = _first_gold_cyclic(pdb_files)
    cyclic_canonical = canonical(cyclic_strict.output_smiles)

    inputs = [linear_pdb, cyclic_pdb, linear_pdb, cyclic_pdb]
    expected_linear = [True, False, True, False]
    for pdb, is_linear in zip(inputs, expected_linear):
        env = application.reconstruct_structure(
            str(pdb), mode="strict", chain_id="L",
            allow_linear_topology=True,
        )
        assert env["status"] == "success"
        topology = env["data"]["provenance"].get("topology_class")
        if is_linear:
            assert topology == "linear"
            assert _heavy_atoms_in_smiles(env["data"]["smiles"]) == 15
        else:
            assert topology != "linear"
            assert canonical(env["data"]["smiles"]) == cyclic_canonical