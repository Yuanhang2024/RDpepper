"""Focused component tests for the receptor charge and valence policy.

Covers the ``cycpep_master.docking.receptor_pdbqt`` behaviors added for
coordinated monoatomic ions and distorted-input valence repair:

* genuine monoatomic Na/K/Mg/Ca/Zn/Ni ions get finite formal ionic charges
  (input explicit declarations take precedence over model assumptions) and
  the coordination detachment is recorded in REMARKs as a docking
  representation choice;
* ``Na`` (sodium) and ``NA`` (nitrogen acceptor) atom types stay distinct
  and the Na/K whitelist extension is receptor-scoped only;
* variable-valent metals and PEOE-parameterless phosphate contexts fail
  closed as ``not_supported`` naming the element and residue;
* undeclared spurious proximity edges at a valence-invalid atom are
  repaired, while CONECT/LINK/SSBOND-declared bonds, peptide links, and
  disulfides are preserved and unrepairable valence fails with a named
  atom.

Fixture coordinates are real excerpts from the frozen development inputs
(``.zcode_paper_supplement_001/chain_inputs``); they are unit-test data,
not benchmark truth.
"""

import math
import os

from rdkit import Chem

from cycpep_master.docking import receptor_pdbqt
from cycpep_master.docking.pdbqt_validation import _AUTODOCK4_TYPES
from cycpep_master.docking.receptor_pdbqt import pdb_to_receptor_pdbqt

# 6wi3 excerpt: ILE180-ASP181-ILE182-HIS183-HIS184 and ALA268-ASP269-SER270
# with ZN A401 (coordinated by OD2 ASP181, ND1 HIS183, OD1 ASP269), NA A402
# (coordinated by O ASP181, O HIS183), and NA A403, plus the declaring
# CONECT records.
_ZN_LINE = (
    "HETATM 9103 ZN    ZN A 401      65.238  30.908  -3.034  1.00 28.38"
    "          ZN  \n"
)
_ZN_NA_FIXTURE = """ATOM   1402  N   ILE A 180      58.613  26.944  -7.737  1.00 18.32           N
ATOM   1403  CA  ILE A 180      58.646  28.257  -8.374  1.00 22.32           C
ATOM   1404  C   ILE A 180      59.528  29.241  -7.611  1.00 20.69           C
ATOM   1405  O   ILE A 180      59.523  30.437  -7.918  1.00 21.04           O
ATOM   1406  CB  ILE A 180      57.230  28.825  -8.566  1.00 20.67           C
ATOM   1407  CG1 ILE A 180      56.526  29.002  -7.220  1.00 19.46           C
ATOM   1408  CG2 ILE A 180      56.417  27.930  -9.494  1.00 18.66           C
ATOM   1409  CD1 ILE A 180      55.119  29.544  -7.341  1.00 25.86           C
ATOM   1410  N   ASP A 181      60.275  28.767  -6.615  1.00 27.00           N
ATOM   1411  CA  ASP A 181      61.315  29.582  -6.002  1.00 25.10           C
ATOM   1412  C   ASP A 181      62.319  30.014  -7.065  1.00 30.18           C
ATOM   1413  O   ASP A 181      62.574  29.294  -8.033  1.00 30.17           O
ATOM   1414  CB  ASP A 181      62.020  28.788  -4.898  1.00 24.17           C
ATOM   1415  CG  ASP A 181      62.904  29.649  -4.020  1.00 28.66           C
ATOM   1416  OD1 ASP A 181      62.491  29.950  -2.880  1.00 27.17           O
ATOM   1417  OD2 ASP A 181      64.011  30.017  -4.464  1.00 31.94           O
ATOM   1418  N   ILE A 182      62.889  31.209  -6.890  1.00 27.03           N
ATOM   1419  CA  ILE A 182      63.841  31.693  -7.884  1.00 25.97           C
ATOM   1420  C   ILE A 182      65.117  30.861  -7.881  1.00 28.24           C
ATOM   1421  O   ILE A 182      65.839  30.834  -8.885  1.00 27.05           O
ATOM   1422  CB  ILE A 182      64.157  33.187  -7.671  1.00 33.15           C
ATOM   1423  CG1 ILE A 182      64.852  33.761  -8.909  1.00 25.46           C
ATOM   1424  CG2 ILE A 182      65.018  33.392  -6.434  1.00 21.71           C
ATOM   1425  CD1 ILE A 182      65.194  35.225  -8.802  1.00 24.08           C
ATOM   1426  N   HIS A 183      65.406  30.163  -6.787  1.00 24.33           N
ATOM   1427  CA  HIS A 183      66.555  29.274  -6.726  1.00 26.58           C
ATOM   1428  C   HIS A 183      66.145  27.855  -7.100  1.00 27.81           C
ATOM   1429  O   HIS A 183      64.990  27.454  -6.934  1.00 28.33           O
ATOM   1430  CB  HIS A 183      67.182  29.290  -5.332  1.00 28.16           C
ATOM   1431  CG  HIS A 183      67.726  30.626  -4.930  1.00 30.47           C
ATOM   1432  ND1 HIS A 183      66.935  31.622  -4.400  1.00 29.80           N
ATOM   1433  CD2 HIS A 183      68.981  31.131  -4.985  1.00 35.89           C
ATOM   1434  CE1 HIS A 183      67.679  32.682  -4.141  1.00 29.06           C
ATOM   1435  NE2 HIS A 183      68.924  32.411  -4.488  1.00 36.63           N
ATOM   1436  N   HIS A 184      67.111  27.098  -7.613  1.00 32.24           N
ATOM   1437  CA  HIS A 184      66.849  25.732  -8.045  1.00 28.83           C
ATOM   1438  C   HIS A 184      66.609  24.827  -6.843  1.00 29.84           C
ATOM   1439  O   HIS A 184      67.366  24.856  -5.868  1.00 30.83           O
ATOM   1440  CB  HIS A 184      68.021  25.213  -8.878  1.00 26.51           C
ATOM   1441  CG  HIS A 184      67.958  23.745  -9.165  1.00 29.00           C
ATOM   1442  ND1 HIS A 184      68.958  22.871  -8.795  1.00 26.39           N
ATOM   1443  CD2 HIS A 184      67.016  22.997  -9.785  1.00 26.85           C
ATOM   1444  CE1 HIS A 184      68.635  21.648  -9.175  1.00 30.88           C
ATOM   1445  NE2 HIS A 184      67.461  21.697  -9.779  1.00 25.63           N
ATOM   2096  N   ALA A 268      58.726  32.986  -2.487  1.00 29.58           N
ATOM   2097  CA  ALA A 268      58.364  34.339  -2.885  1.00 24.52           C
ATOM   2098  C   ALA A 268      59.554  35.172  -3.339  1.00 30.24           C
ATOM   2099  O   ALA A 268      59.366  36.327  -3.736  1.00 26.75           O
ATOM   2100  CB  ALA A 268      57.649  35.053  -1.733  1.00 26.71           C
ATOM   2101  N   ASP A 269      60.769  34.625  -3.297  1.00 24.75           N
ATOM   2102  CA  ASP A 269      61.938  35.384  -3.726  1.00 29.44           C
ATOM   2103  C   ASP A 269      62.033  35.518  -5.239  1.00 31.19           C
ATOM   2104  O   ASP A 269      63.007  36.099  -5.731  1.00 31.12           O
ATOM   2105  CB  ASP A 269      63.209  34.745  -3.164  1.00 25.56           C
ATOM   2106  CG  ASP A 269      63.278  33.258  -3.428  1.00 27.83           C
ATOM   2107  OD1 ASP A 269      64.378  32.682  -3.296  1.00 26.07           O
ATOM   2108  OD2 ASP A 269      62.236  32.668  -3.781  1.00 33.81           O
ATOM   2109  N   SER A 270      61.058  34.999  -5.982  1.00 33.24           N
ATOM   2110  CA  SER A 270      60.944  35.240  -7.412  1.00 26.81           C
ATOM   2111  C   SER A 270      60.075  36.451  -7.726  1.00 30.89           C
ATOM   2112  O   SER A 270      59.757  36.687  -8.896  1.00 28.67           O
ATOM   2113  CB  SER A 270      60.387  33.999  -8.113  1.00 29.88           C
ATOM   2114  OG  SER A 270      59.169  33.581  -7.521  1.00 26.27           O
HETATM 9104 NA    NA A 402      62.543  26.886  -8.424  1.00 18.99          NA
HETATM 9105 NA    NA A 403      62.607  11.702  -9.994  1.00 32.60          NA
CONECT 1413 9104
CONECT 1417 9103
CONECT 1429 9104
CONECT 1432 9103
CONECT 2107 9103
CONECT 9103 1417 1432 2107
""" + _ZN_LINE

# 1h0i excerpt A315-Y318: the C(ASP316)--CD(PRO317) 1.71 A proximity artifact
# (undeclared: no CONECT/LINK/SSBOND record mentions serials 2448/2460).
_ASP_PRO_FIXTURE = """ATOM   2437  N   GLU A 315      20.070  28.776  92.838  1.00 75.38           N
ATOM   2438  CA  GLU A 315      18.625  28.773  92.691  1.00 78.49           C
ATOM   2439  C   GLU A 315      17.923  29.176  93.987  1.00 79.89           C
ATOM   2440  O   GLU A 315      17.985  28.455  94.980  1.00 80.03           O
ATOM   2441  CB  GLU A 315      18.168  27.377  92.283  1.00 79.27           C
ATOM   2442  CG  GLU A 315      18.713  26.870  90.937  1.00 80.28           C
ATOM   2443  CD  GLU A 315      20.200  26.552  90.955  1.00 80.91           C
ATOM   2444  OE1 GLU A 315      21.016  27.493  90.878  1.00 81.17           O
ATOM   2445  OE2 GLU A 315      20.551  25.355  91.049  1.00 81.12           O
ATOM   2446  N   ASP A 316      17.260  30.334  93.964  1.00 81.99           N
ATOM   2447  CA  ASP A 316      16.523  30.819  95.123  1.00 83.96           C
ATOM   2448  C   ASP A 316      15.148  30.176  95.049  1.00 84.88           C
ATOM   2449  O   ASP A 316      14.457  30.271  94.037  1.00 85.28           O
ATOM   2450  CB  ASP A 316      16.441  32.341  95.095  1.00 84.68           C
ATOM   2451  CG  ASP A 316      17.815  32.974  94.955  1.00 85.29           C
ATOM   2452  OD1 ASP A 316      18.680  32.660  95.805  1.00 85.74           O
ATOM   2453  OD2 ASP A 316      18.040  33.746  94.002  1.00 85.89           O
ATOM   2454  N   PRO A 317      14.781  29.435  96.106  1.00 85.32           N
ATOM   2455  CA  PRO A 317      14.282  28.610  97.198  1.00 85.31           C
ATOM   2456  C   PRO A 317      15.127  27.352  97.249  1.00 85.23           C
ATOM   2457  O   PRO A 317      14.736  26.293  96.750  1.00 85.87           O
ATOM   2458  CB  PRO A 317      12.839  28.369  96.789  1.00 85.59           C
ATOM   2459  CG  PRO A 317      12.462  29.718  96.168  1.00 86.02           C
ATOM   2460  CD  PRO A 317      13.768  30.495  95.999  1.00 85.67           C
ATOM   2461  N   TYR A 318      16.314  27.499  97.843  1.00 84.62           N
ATOM   2462  CA  TYR A 318      17.236  26.408  97.992  1.00 83.85           C
ATOM   2463  C   TYR A 318      16.599  25.058  97.701  1.00 83.73           C
ATOM   2464  O   TYR A 318      15.634  24.665  98.356  1.00 83.99           O
ATOM   2465  CB  TYR A 318      17.794  26.392  99.406  1.00 82.95           C
ATOM   2466  CG  TYR A 318      18.775  25.269  99.631  1.00 82.75           C
ATOM   2467  CD1 TYR A 318      18.572  24.318 100.636  1.00 82.90           C
ATOM   2468  CD2 TYR A 318      19.914  25.156  98.840  1.00 82.34           C
ATOM   2469  CE1 TYR A 318      19.493  23.287 100.848  1.00 82.35           C
ATOM   2470  CE2 TYR A 318      20.835  24.137  99.045  1.00 82.04           C
ATOM   2471  CZ  TYR A 318      20.622  23.208 100.046  1.00 82.19           C
ATOM   2472  OH  TYR A 318      21.574  22.241 100.263  1.00 82.16           O
"""

# 1h0i excerpt CYS328/CYS331 with its declaring SSBOND + CONECT records.
_CYS_BLOCK = """ATOM   2546  N   CYS A 328      25.281  24.715 111.501  1.00 48.71           N
ATOM   2547  CA  CYS A 328      24.303  25.785 111.404  1.00 48.19           C
ATOM   2548  C   CYS A 328      22.897  25.257 111.615  1.00 48.17           C
ATOM   2549  O   CYS A 328      22.246  24.810 110.672  1.00 48.87           O
ATOM   2550  CB  CYS A 328      24.404  26.488 110.048  1.00 47.69           C
ATOM   2551  SG  CYS A 328      23.195  27.834 109.870  1.00 45.69           S
ATOM   2570  N   CYS A 331      20.189  27.424 109.902  1.00 46.87           N
ATOM   2571  CA  CYS A 331      19.981  27.134 108.485  1.00 48.19           C
ATOM   2572  C   CYS A 331      18.881  26.095 108.328  1.00 50.26           C
ATOM   2573  O   CYS A 331      18.112  26.134 107.366  1.00 50.94           O
ATOM   2574  CB  CYS A 331      21.267  26.612 107.856  1.00 46.41           C
ATOM   2575  SG  CYS A 331      22.652  27.794 107.910  1.00 47.43           S
SSBOND   1 CYS A  328    CYS A  331                          1555   1555  2.03
CONECT 2551 2575
CONECT 2575 2551
"""

# Synthetic tetrahedral phosphate: all P--O bonds are single and neutral,
# which PEOE has no parameters for (the 7rov GCP situation).
_PO4_FIXTURE = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      5  P   PO4 A  10       0.000   0.000   0.000  1.00  0.00           P
ATOM      6  O1  PO4 A  10       1.480   0.000   0.000  1.00  0.00           O
ATOM      7  O2  PO4 A  10      -0.490   1.400   0.000  1.00  0.00           O
ATOM      8  O3  PO4 A  10      -0.490  -0.700   1.210  1.00  0.00           O
ATOM      9  O4  PO4 A  10      -0.490  -0.700  -1.210  1.00  0.00           O
"""

_ORGANIC_FIXTURE = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.100   1.200   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.600   2.300   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.000  -1.100   0.500  1.00  0.00           C
"""

_REAL_LINK_LINE = (
    "LINK         NE2 HIS A 188                NI    NI A 501     "
    "1555   1555  2.14  \n"
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text + "END\n", encoding="ascii", newline="\n")
    return str(path)


def _output_lines(path):
    return [
        line
        for line in open(path, encoding="utf-8").read().splitlines()
        if line.startswith("ATOM  ")
    ]


def _heavy_lines(path):
    return [line for line in _output_lines(path) if line.split()[-1] != "HD"]


def _site(line):
    return (
        line[12:16].strip(),
        line[17:20].strip(),
        line[21],
        int(line[22:26]),
        line[26],
    )


def test_monoatomic_ions_get_formal_charges_and_heavy_atoms_preserved(tmp_path):
    source = _write(tmp_path, "zn_na.pdb", _ZN_NA_FIXTURE)
    output = str(tmp_path / "zn_na.pdbqt")

    error = pdb_to_receptor_pdbqt(source, output)

    assert error is None
    heavy = _heavy_lines(output)
    assert len(heavy) == 66
    zn_line = next(line for line in heavy if line[17:20].strip() == "ZN")
    assert zn_line.split()[-1] == "Zn"
    assert float(zn_line.split()[-2]) == 2.0
    assert _site(zn_line) == ("ZN", "ZN", "A", 401, " ")
    assert tuple(
        float(zn_line[start:end])
        for start, end in ((30, 38), (38, 46), (46, 54))
    ) == (65.238, 30.908, -3.034)
    na_lines = [line for line in heavy if line[17:20].strip() == "NA"]
    assert len(na_lines) == 2
    for line in na_lines:
        assert line.split()[-1] == "Na"
        assert float(line.split()[-2]) == 1.0
    assert all(math.isfinite(float(line.split()[-2])) for line in heavy)
    text = open(output, encoding="utf-8").read()
    assert "ion charges:" in text
    assert "explicit model assumptions" in text
    assert "NOT experimentally determined" in text
    assert "rigid docking representation" in text
    assert "can protonate coordinating groups" in text
    assert "ZN A401" in text and "NA A402" in text and "NA A403" in text


def test_sodium_type_case_and_receptor_only_whitelist_scope(tmp_path):
    source = _write(tmp_path, "zn_na.pdb", _ZN_NA_FIXTURE)
    output = str(tmp_path / "zn_na.pdbqt")

    error = pdb_to_receptor_pdbqt(source, output)

    assert error is None
    heavy = _heavy_lines(output)
    sodium = [line for line in heavy if line.split()[-1] == "Na"]
    assert len(sodium) == 2
    assert all(line[17:20].strip() == "NA" for line in sodium)
    # "Na" (sodium) is distinct from "NA" (nitrogen acceptor); the Na/K
    # extension must not leak into the shared ligand whitelist.
    assert receptor_pdbqt._RECEPTOR_ION_ATOM_TYPES == frozenset({"Na", "K"})
    assert "Na" not in _AUTODOCK4_TYPES
    assert "NA" in _AUTODOCK4_TYPES
    assert receptor_pdbqt._MONOATOMIC_ION_FORMAL_CHARGES["K"] == 1


def test_input_explicit_formal_charge_precedence(tmp_path):
    charged_zn = _ZN_LINE[:-3] + "1+\n"
    assert charged_zn != _ZN_LINE
    source = _write(
        tmp_path, "zn_explicit.pdb", _ZN_NA_FIXTURE.replace(_ZN_LINE, charged_zn)
    )
    output = str(tmp_path / "zn_explicit.pdbqt")

    error = pdb_to_receptor_pdbqt(source, output)

    assert error is None
    zn_line = next(
        line for line in _heavy_lines(output) if line[17:20].strip() == "ZN"
    )
    assert float(zn_line.split()[-2]) == 1.0
    text = open(output, encoding="utf-8").read()
    assert "input explicit formal charge" in text


def _pdb_info_atom(symbol, name, resname, chain, resnum, serial=1):
    atom = Chem.Atom(symbol)
    info = Chem.AtomPDBResidueInfo()
    info.SetName(f"{name:<4s}")
    info.SetResidueName(f"{resname:<3s}")
    info.SetChainId(chain)
    info.SetResidueNumber(resnum)
    info.SetInsertionCode(" ")
    info.SetSerialNumber(serial)
    atom.SetPDBResidueInfo(info)
    return atom


def test_nonfinite_error_names_variable_valent_metal_and_residue():
    # A bonded variable-valent metal receives no PEOE parameters; the typed
    # error must name the element/residue and demand a chemistry reference
    # instead of guessing an oxidation state.  (Isolated neutral Fe parses
    # with a finite 0.0 charge on this RDKit build, so the classification is
    # exercised directly.)
    holder = Chem.RWMol()
    for resnum in (501, 502):
        holder.AddAtom(_pdb_info_atom("Fe", "FE", "FE", "A", resnum))

    error = receptor_pdbqt._nonfinite_charge_error(list(holder.GetAtoms()))

    assert error.startswith("not_supported:")
    assert "Fe" in error
    assert "FE" in error
    assert "chemistry reference" in error
    assert "refusing to assign a guessed charge" in error


def test_phosphate_charge_uncertainty_stays_not_supported_named(tmp_path):
    source = _write(tmp_path, "po4.pdb", _PO4_FIXTURE)
    output = str(tmp_path / "po4.pdbqt")

    error = pdb_to_receptor_pdbqt(source, output)

    assert error is not None
    assert error.startswith("not_supported:")
    assert "P" in error and "PO4" in error
    assert "chemistry reference" in error
    assert not os.path.exists(output)


def test_distorted_proline_artifact_repaired(tmp_path):
    source = _write(tmp_path, "proline.pdb", _ASP_PRO_FIXTURE)
    output = str(tmp_path / "proline.pdbqt")

    error = pdb_to_receptor_pdbqt(source, output)

    assert error is None
    heavy = _heavy_lines(output)
    assert len(heavy) == 36
    cd_line = next(
        line
        for line in heavy
        if line[12:16].strip() == "CD" and line[17:20].strip() == "PRO"
    )
    assert _site(cd_line) == ("CD", "PRO", "A", 317, " ")
    assert tuple(
        float(cd_line[start:end])
        for start, end in ((30, 38), (38, 46), (46, 54))
    ) == (13.768, 30.495, 95.999)
    assert all(math.isfinite(float(line.split()[-2])) for line in heavy)
    text = open(output, encoding="utf-8").read()
    assert "proximity repair" in text
    assert "C ASP A316" in text and "CD PRO A317" in text


def test_declared_noncanonical_bond_is_not_removed(tmp_path):
    source = _write(
        tmp_path,
        "proline_declared.pdb",
        _ASP_PRO_FIXTURE + "CONECT 2448 2460\n",
    )
    output = str(tmp_path / "proline_declared.pdbqt")

    error = pdb_to_receptor_pdbqt(source, output)

    assert error is not None
    assert "cannot be sanitized" in error
    assert "AtomValenceException" in error
    assert "C ASP" in error or "CD PRO" in error
    assert "preserved" in error
    assert not os.path.exists(output)


def test_peptide_and_disulfide_survive_repair(tmp_path):
    for name, extra in (
        ("frag_ss_declared.pdb", _CYS_BLOCK),
        ("frag_ss_undeclared.pdb", _CYS_BLOCK.split("SSBOND")[0]),
    ):
        source = _write(tmp_path, name, _ASP_PRO_FIXTURE + extra)
        molecule = Chem.MolFromPDBFile(source, removeHs=False, sanitize=False)
        declared = receptor_pdbqt._read_declared_pdb_edges(source)
        assert declared["ok"]

        repaired, error, edges = receptor_pdbqt._sanitize_with_spurious_edge_repair(
            molecule, declared
        )

        assert error is None, name
        assert len(edges) == 1
        sites = edges[0][0] + " " + edges[0][1]
        assert "C ASP" in sites and "CD PRO" in sites

        def find(resname, resnum, atom):
            return next(
                atom_
                for atom_ in repaired.GetAtoms()
                if atom_.GetPDBResidueInfo() is not None
                and atom_.GetPDBResidueInfo().GetResidueName().strip() == resname
                and atom_.GetPDBResidueInfo().GetResidueNumber() == resnum
                and atom_.GetPDBResidueInfo().GetName().strip() == atom
            )

        c_asp = find("ASP", 316, "C")
        n_pro = find("PRO", 317, "N")
        cd_pro = find("PRO", 317, "CD")
        sg_a = find("CYS", 328, "SG")
        sg_b = find("CYS", 331, "SG")
        assert repaired.GetBondBetweenAtoms(c_asp.GetIdx(), n_pro.GetIdx()) is not None
        assert repaired.GetBondBetweenAtoms(c_asp.GetIdx(), cd_pro.GetIdx()) is None
        assert repaired.GetBondBetweenAtoms(sg_a.GetIdx(), sg_b.GetIdx()) is not None


def test_declared_edge_parser_reads_conect_link_ssbond(tmp_path):
    source = _write(
        tmp_path,
        "records.pdb",
        "CONECT 2551 2575\n" + _REAL_LINK_LINE + _CYS_BLOCK,
    )

    declared = receptor_pdbqt._read_declared_pdb_edges(source)

    assert declared["ok"]
    assert declared["conect"] == {frozenset((2551, 2575))}
    assert declared["ssbond"] == {frozenset((("A", 328, ""), ("A", 331, "")))}
    assert declared["link"] == {
        frozenset(
            (("A", 188, "", "HIS", "NE2"), ("A", 501, "", "NI", "NI"))
        )
    }

    # A LINK-declared pair must be recognized as declared (ok=True parse,
    # not protection-by-failed-parser), and a different site must not be.
    holder = Chem.RWMol()
    holder.AddAtom(_pdb_info_atom("N", "NE2", "HIS", "A", 188, serial=100))
    holder.AddAtom(_pdb_info_atom("Ni", "NI", "NI", "A", 501, serial=101))
    holder.AddBond(0, 1, Chem.BondType.SINGLE)
    molecule = holder.GetMol()
    assert receptor_pdbqt._bond_declared_in_input(
        molecule.GetAtomWithIdx(0), molecule.GetAtomWithIdx(1), declared
    )

    other = Chem.RWMol()
    other.AddAtom(_pdb_info_atom("N", "NE2", "HIS", "A", 189, serial=102))
    other.AddAtom(_pdb_info_atom("Ni", "NI", "NI", "A", 501, serial=103))
    other.AddBond(0, 1, Chem.BondType.SINGLE)
    mismatch = other.GetMol()
    assert not receptor_pdbqt._bond_declared_in_input(
        mismatch.GetAtomWithIdx(0), mismatch.GetAtomWithIdx(1), declared
    )


def test_trimmed_ssbond_record_keeps_declaration(tmp_path):
    line = next(line for line in _CYS_BLOCK.splitlines() if line.startswith("SSBOND"))[:35]
    source = _write(tmp_path, "trimmed_ssbond.pdb", line + "\n")
    declared = receptor_pdbqt._read_declared_pdb_edges(source)
    assert declared["ok"]
    assert declared["ssbond"] == {frozenset((("A", 328, ""), ("A", 331, "")))}


def test_organic_receptor_keeps_default_behavior(tmp_path):
    source = _write(tmp_path, "organic.pdb", _ORGANIC_FIXTURE)
    output = str(tmp_path / "organic.pdbqt")

    error = pdb_to_receptor_pdbqt(source, output)

    assert error is None
    text = open(output, encoding="utf-8").read()
    assert "pH not assigned" in text
    assert "hydrogens:" in text
    assert "ion charges" not in text
    assert "proximity repair" not in text
    assert len(_heavy_lines(output)) == 5


def test_monoatomic_chloride_is_not_neutralized_or_hydrogenated(tmp_path):
    source = _write(tmp_path, "chloride.pdb",
                    "HETATM    1 CL    CL A   1      10.000  10.000  10.000  1.00 20.00          CL  \nEND\n")
    output = str(tmp_path / "chloride.pdbqt")
    assert pdb_to_receptor_pdbqt(source, output) is None
    lines = _heavy_lines(output)
    assert len(lines) == 1
    assert lines[0].split()[-1] == "Cl"
    assert float(lines[0][70:76]) == -1.0
    molecule = Chem.MolFromSmiles("CCl")
    for index, atom in enumerate(molecule.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetName("CL" if atom.GetSymbol() == "Cl" else "C")
        info.SetResidueName("LIG")
        info.SetResidueNumber(2)
        atom.SetPDBResidueInfo(info)
    assigned, error, remarks = receptor_pdbqt._apply_monoatomic_ion_policy(molecule)
    assert error is None and not remarks
    assert Chem.MolToSmiles(assigned) == "CCl"
