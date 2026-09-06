"""Tests for branched / multi-chain peptide assembly (Part 3).

Multi-polymer HELM (``PEPTIDE1{...}|PEPTIDE2{...}$conns$$$``) is converted to a
single MAP string with ``{br}`` chain-break markers and global cyclization
positions, then assembled with the inter-chain backbone bond suppressed at each
break so the chains are joined only by the explicit connection records
(disulfides / inter-chain peptide bonds).

The headline case is insulin (two chains A+B, one intra-chain and two
inter-chain disulfides) assembling into a single connected molecule with three
S-S bonds. Single-chain behaviour must be unchanged.
"""
from rdkit import Chem

from cycpep_master.paths._map_utils import (
    helm_to_map, get_smi_from_map, cyclize_linpep_from_map,
)
from cycpep_master.paths.path_b import (
    build_helm_multichain, generate_multichain,
)
import os
import pytest

_4INS = os.environ.get("RDPEPPER_TEST_4INS_PDB", "")
_have_4ins = os.path.exists(_4INS)

# Insulin A and B chains (4INS), HELM with three disulfides:
#   A6-A11 (intra-A), A7-B7 and A20-B19 (inter-chain).
_INS_A = "G.I.V.E.Q.C.C.T.S.I.C.S.L.Y.Q.L.E.N.Y.C.N"   # 21 residues
_INS_B = "F.V.N.Q.H.L.C.G.S.H.L.V.E.A.L.Y.L.V.C.G.E.R.G.F.F.Y.T.P.K.A"  # 30
_INS_CONNS = ("PEPTIDE1,PEPTIDE1,6:R3-11:R3|"
              "PEPTIDE1,PEPTIDE2,7:R3-7:R3|"
              "PEPTIDE1,PEPTIDE2,20:R3-19:R3")
_INSULIN_HELM = f"PEPTIDE1{{{_INS_A}}}|PEPTIDE2{{{_INS_B}}}${_INS_CONNS}$$$"


def _ss_count(smi):
    m = Chem.MolFromSmiles(smi)
    return len(m.GetSubstructMatches(Chem.MolFromSmarts("[#16X2]-[#16X2]")))


def _frags(smi):
    m = Chem.MolFromSmiles(smi)
    return len(Chem.GetMolFrags(m))


def _pdb_atom(serial, chain, residue, *, name="SG", element="S"):
    return (
        f"ATOM  {serial:5d} {name:>4s} CYS {chain}{residue:4d}    "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
        f"          {element:>2s}"
    )


def _cross_chain_link():
    record = list(" " * 80)
    record[0:6] = "LINK  "
    record[12:16] = f"{'SG':>4s}"
    record[17:20] = "CYS"
    record[21] = "A"
    record[22:26] = f"{2:4d}"
    record[42:46] = f"{'SG':>4s}"
    record[47:50] = "CYS"
    record[51] = "B"
    record[52:56] = f"{2:4d}"
    return "".join(record)


def _two_cysteine_chains(tmp_path, *connection_records):
    path = tmp_path / "cross-chain.pdb"
    atoms = [
        _pdb_atom(index, chain, residue)
        for index, (chain, residue) in enumerate(
            [("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2), ("B", 3)],
            start=1,
        )
    ]
    path.write_text(
        "\n".join([*connection_records, *atoms, "END", ""]),
        encoding="ascii",
    )
    return path


# ── helm_to_map multi-chain translation ─────────────────────────────────────

def test_helm_to_map_multichain_emits_break_and_global_positions():
    mp = helm_to_map(_INSULIN_HELM)
    assert "{br}" in mp                       # chain boundary marked
    # inter-chain disulfides translated to global positions (B offset = 21)
    assert "{cyc:7:R3-28:R3}" in mp           # A7 - B7  -> 7, 21+7=28
    assert "{cyc:20:R3-40:R3}" in mp          # A20 - B19 -> 20, 21+19=40
    assert "{cyc:6:R3-11:R3}" in mp           # intra-A unchanged


def test_helm_to_map_single_chain_unchanged():
    """The single-polymer path must produce exactly the original MAP."""
    assert helm_to_map("PEPTIDE1{A.C.G.C}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$") \
        == "ACGC{cyc:2:R3-4:R3}"
    assert helm_to_map("PEPTIDE1{A.C.D.E.F}$$$$") == "ACDEF"
    assert helm_to_map("PEPTIDE1{C.Y.I.Q.N.C.P.L.G}$PEPTIDE1,PEPTIDE1,1:R1-9:R2$$$") \
        == "CYIQNCPLG{cyc:N-C}"


# ── Multi-chain assembly ────────────────────────────────────────────────────

def test_insulin_assembles_single_connected_molecule():
    smi = get_smi_from_map(helm_to_map(_INSULIN_HELM))
    assert smi is not None
    assert _frags(smi) == 1          # one connected molecule
    assert _ss_count(smi) == 3       # three disulfide bonds


def test_chain_break_suppresses_backbone_bond():
    """Two chains C-A and A-C joined only by a disulfide between the cysteines:
    a chain break at index 1 must prevent the A(1)->A(2) backbone bond, so the
    result is one molecule bridged solely by the S-S bond (not a 4-mer)."""
    smi = cyclize_linpep_from_map(["C", "A", "A", "C"], ["1:R3-4:R3"],
                                  chain_breaks={1})
    assert smi is not None
    assert _frags(smi) == 1
    assert _ss_count(smi) == 1
    # Without the break, the same call would peptide-bond all four into a ring.
    bonded = cyclize_linpep_from_map(["C", "A", "A", "C"], ["1:R3-4:R3"])
    assert bonded is not None  # different topology, but must still build


def test_multichain_no_connections_stays_disconnected():
    """Two independent chains with no connection records assemble as two
    separate fragments (chain break, no bridging bond)."""
    mp = helm_to_map("PEPTIDE1{A.G}|PEPTIDE2{G.A}$$$$")
    assert "{br}" in mp
    smi = get_smi_from_map(mp)
    assert smi is not None
    assert _frags(smi) == 2


@pytest.mark.parametrize(
    "records",
    [
        (_cross_chain_link(),),
        ("CONECT    2    5",),
        (_cross_chain_link(), "CONECT    2    5"),
    ],
)
def test_multichain_link_and_conect_are_preserved_and_deduplicated(
    tmp_path, records
):
    path = _two_cysteine_chains(tmp_path, *records)

    helm = build_helm_multichain(str(path), ["A", "B"])
    smiles, error = generate_multichain(str(path), ["A", "B"])

    assert helm.split("$")[1].split("|") == ["PEPTIDE1,PEPTIDE2,2:R3-2:R3"]
    assert error is None
    assert smiles is not None
    assert _frags(smiles) == 1
    assert _ss_count(smiles) == 1


def test_path_b_maps_nh2_cap_to_registered_symbol(tmp_path):
    path = tmp_path / "amide-cap.pdb"
    path.write_text(
        "\n".join([
            _pdb_atom(1, "A", 1),
            _pdb_atom(2, "A", 2),
            _pdb_atom(3, "A", 3),
            _pdb_atom(4, "A", 4, name="N", element="N").replace("CYS", "NH2"),
            "END",
            "",
        ]),
        encoding="ascii",
    )

    helm = build_helm_multichain(str(path), ["A"])

    assert "[nh2]" in helm


# ── PDB -> multi-chain HELM end to end ───────────────────────────────────────

@pytest.mark.skipif(not _have_4ins, reason="4INS insulin PDB absent")
def test_build_helm_multichain_from_insulin_pdb():
    """SSBOND records become global HELM connections (one insulin = A+B)."""
    helm = build_helm_multichain(_4INS, ["A", "B"])
    assert helm is not None
    assert helm.count("PEPTIDE") >= 4  # 2 block ids + >=2 connection refs
    conns = helm.split("$")[1]
    assert "6:R3-11:R3" in conns       # intra-A disulfide
    assert "7:R3-7:R3" in conns        # A7-B7 inter-chain


@pytest.mark.skipif(not _have_4ins, reason="4INS insulin PDB absent")
def test_generate_multichain_insulin_single_molecule():
    smi, err = generate_multichain(_4INS, ["A", "B"])
    assert smi is not None, f"failed: {err}"
    assert _frags(smi) == 1            # one connected insulin
    assert _ss_count(smi) == 3         # three disulfides


@pytest.mark.skipif(not _have_4ins, reason="4INS insulin PDB absent")
def test_generate_multichain_all_chains_two_insulins():
    """4INS has two insulin molecules (A+B and C+D); the all-chains default
    yields two disconnected molecules with six disulfides total."""
    smi, err = generate_multichain(_4INS)
    assert smi is not None, f"failed: {err}"
    assert _frags(smi) == 2
    assert _ss_count(smi) == 6
