"""
Build unified monomer library from NNAA and CycPeptMPDB sources.

Input:
  NNAA_10000.txt            — 9,998 compiled entries: SMILES <tab> 3-letter-code
                               Source: Amarasinghe et al., JCIM 2022,
                               DOI 10.1021/acs.jcim.2c00193 (diverse
                               10,000-amino-acid Associated Content subset).
  CycPeptMPDB_Monomer_All.csv — 385 entries with ADMET descriptors + HELM fields

Output:
  unified_monomer_library.csv — ~10,383 rows with:
    • identity & classification
    • IUPAC names (PubChem > RDKit > InChI descriptor fallback)
    • 200+ RDKit ADMET descriptors
    • HELM fields (replaced_SMILES, CXSMILES, Monomer_Type, Polymer_Type, R1, R2, R3)

Usage:
  python build_monomer_library.py [--skip-iupac] [--skip-helm]
"""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    Descriptors, AllChem, Fragments,
    Crippen, Lipinski, GraphDescriptors, EState,
    rdMolDescriptors,
)
from rdkit.Chem.QED import qed as calc_qed

RDLogger.logger().setLevel(RDLogger.CRITICAL)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_source_path(filename: str, base_dir: str = BASE_DIR) -> str:
    """Resolve a source file without assuming an extra nested package dir.

    Historical checkouts placed source tables beside this script, in a data
    directory, or one level above the package. Prefer an existing candidate,
    then return the package-local path for a deterministic missing-file error.
    """
    package_dir = os.path.abspath(base_dir)
    parent_dir = os.path.dirname(package_dir)
    candidates = (
        os.path.join(package_dir, filename),
        os.path.join(package_dir, "data", filename),
        os.path.join(parent_dir, filename),
        os.path.join(parent_dir, "data", filename),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def resolve_build_paths(base_dir: str = BASE_DIR) -> Dict[str, str]:
    """Return input/output paths for the builder's current repository layout."""
    package_dir = os.path.abspath(base_dir)
    return {
        "nnaa": _resolve_source_path("NNAA_10000.txt", package_dir),
        "cycpep": _resolve_source_path("CycPeptMPDB_Monomer_All.csv", package_dir),
        "output": os.path.join(package_dir, "unified_monomer_library.csv"),
        "cache": os.path.join(package_dir, ".iupac_cache.json"),
    }


_BUILD_PATHS = resolve_build_paths()
NNAA_PATH = _BUILD_PATHS["nnaa"]
CYCPEP_PATH = _BUILD_PATHS["cycpep"]
OUTPUT_PATH = _BUILD_PATHS["output"]
CACHE_PATH = _BUILD_PATHS["cache"]

# Rate limit for PubChem (seconds between requests)
PUBCHEM_DELAY = 0.3
PUBCHEM_TIMEOUT = 10


# ============================================================================
# SECTION 1: SMILES utilities
# ============================================================================

def _neutralized_mol(smi: str):
    """Return a conservatively neutralized RDKit molecule.

    Only the two unambiguous peptide-zwitterion transformations are applied:
    a protonated, non-aromatic terminal amine and a carboxylate oxygen whose
    carbon also has a double-bonded oxygen. Other charge patterns are kept
    intact. In particular, nitro, quaternary ammonium and aromatic cations
    must not be rewritten by textual substitution.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None

    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        # [NH3+] at the peptide N terminus. Quaternary ammonium has no H and
        # aromatic cations are intentionally excluded.
        if (
            atom.GetAtomicNum() == 7
            and atom.GetFormalCharge() == 1
            and not atom.GetIsAromatic()
            and len(atom.GetNeighbors()) == 1
            and atom.GetNumExplicitHs() >= 2
        ):
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
            atom.SetNoImplicit(False)

        # [O-] in a carboxylate. A negatively charged oxygen attached to a
        # positively charged N (e.g. nitro) is not a carboxylate and remains
        # charged.
        if atom.GetAtomicNum() != 8 or atom.GetFormalCharge() != -1:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            continue
        carbon = neighbors[0]
        if carbon.GetAtomicNum() != 6:
            continue
        is_carboxyl = any(
            bond.GetOtherAtom(carbon).GetAtomicNum() == 8
            and bond.GetBondType() == Chem.BondType.DOUBLE
            for bond in carbon.GetBonds()
        )
        if is_carboxyl:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(1)
            atom.SetNoImplicit(False)

    result = rw.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception:
        # Let callers retain the original molecule/SMILES if a partially
        # specified input cannot be sanitized after the safe edits.
        return None
    return result


def _neutralize(smi: str) -> str:
    mol = _neutralized_mol(smi)
    if mol is None:
        return smi
    try:
        return Chem.MolToSmiles(mol, canonical=False)
    except Exception:
        return smi


def neutralize_backbone(smi: str) -> str:
    """Neutralize only unambiguous peptide backbone zwitterion atoms."""
    return _neutralize(smi)


def neutralize_full(smi: str) -> str:
    """Return a safe neutralized representation for descriptor calculation.

    The historical name is retained for API compatibility. Unlike the old
    implementation, non-backbone ionic groups are not rewritten blindly.
    """
    return _neutralize(smi)


def canonicalize(smi: str, strip_stereo: bool = False) -> Tuple[str, object]:
    """Return (canonical_smiles, mol) or (smi, None) on failure."""
    neut = neutralize_full(smi)
    mol = Chem.MolFromSmiles(neut)
    if mol is None:
        return smi, None
    if strip_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True), mol


def morgan_fp(smi: str) -> object:
    mol = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol else None


# ============================================================================
# SECTION 2: ADMET descriptors (200+ fields)
# ============================================================================

DESCRIPTOR_FIELDS = [
    "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
    "qed", "MolWt", "HeavyAtomMolWt", "ExactMolWt",
    "NumValenceElectrons", "NumRadicalElectrons",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
    "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
    "BalabanJ", "BertzCT",
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3",
    "LabuteASA",
    "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14",
    "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7",
    "PEOE_VSA8", "PEOE_VSA9",
    "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5",
    "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9",
    "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
    "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5",
    "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9",
    "TPSA",
    "EState_VSA1", "EState_VSA10", "EState_VSA11",
    "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
    "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9",
    "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4",
    "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9",
    "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "RingCount", "MolLogP", "MolMR",
    "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO",
    "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2",
    "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine",
    "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1",
    "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde",
    "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide",
    "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo",
    "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic",
    "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether",
    "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone",
    "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone",
    "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy",
    "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom",
    "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime",
    "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond",
    "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine",
    "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN",
    "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene",
    "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene",
    "fr_unbrch_alkane", "fr_urea",
]

_std_desc_map = {
    "MolWt": Descriptors.MolWt, "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "ExactMolWt": Descriptors.ExactMolWt,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
    "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
    "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
    "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
    "FpDensityMorgan3": Descriptors.FpDensityMorgan3,
    "BalabanJ": GraphDescriptors.BalabanJ,
    "BertzCT": GraphDescriptors.BertzCT,
    "Chi0": Descriptors.Chi0, "Chi0n": Descriptors.Chi0n, "Chi0v": Descriptors.Chi0v,
    "Chi1": Descriptors.Chi1, "Chi1n": Descriptors.Chi1n, "Chi1v": Descriptors.Chi1v,
    "Chi2n": Descriptors.Chi2n, "Chi2v": Descriptors.Chi2v,
    "Chi3n": Descriptors.Chi3n, "Chi3v": Descriptors.Chi3v,
    "Chi4n": Descriptors.Chi4n, "Chi4v": Descriptors.Chi4v,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Ipc": Descriptors.Ipc,
    "Kappa1": Descriptors.Kappa1, "Kappa2": Descriptors.Kappa2, "Kappa3": Descriptors.Kappa3,
    "LabuteASA": Descriptors.LabuteASA, "TPSA": Descriptors.TPSA,
    "FractionCSP3": Lipinski.FractionCSP3,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "NHOHCount": Lipinski.NHOHCount, "NOCount": Lipinski.NOCount,
    "NumAliphaticCarbocycles": Lipinski.NumAliphaticCarbocycles,
    "NumAliphaticHeterocycles": Lipinski.NumAliphaticHeterocycles,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "NumAromaticCarbocycles": Lipinski.NumAromaticCarbocycles,
    "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumSaturatedCarbocycles": Lipinski.NumSaturatedCarbocycles,
    "NumSaturatedHeterocycles": Lipinski.NumSaturatedHeterocycles,
    "NumSaturatedRings": Lipinski.NumSaturatedRings,
    "RingCount": Lipinski.RingCount,
    "MolLogP": Crippen.MolLogP, "MolMR": Crippen.MolMR,
    "qed": calc_qed,
}

_frag_funcs = {
    "fr_Al_COO": Fragments.fr_Al_COO, "fr_Al_OH": Fragments.fr_Al_OH,
    "fr_Al_OH_noTert": Fragments.fr_Al_OH_noTert, "fr_ArN": Fragments.fr_ArN,
    "fr_Ar_COO": Fragments.fr_Ar_COO, "fr_Ar_N": Fragments.fr_Ar_N,
    "fr_Ar_NH": Fragments.fr_Ar_NH, "fr_Ar_OH": Fragments.fr_Ar_OH,
    "fr_COO": Fragments.fr_COO, "fr_COO2": Fragments.fr_COO2,
    "fr_C_O": Fragments.fr_C_O, "fr_C_O_noCOO": Fragments.fr_C_O_noCOO,
    "fr_C_S": Fragments.fr_C_S, "fr_HOCCN": Fragments.fr_HOCCN,
    "fr_Imine": Fragments.fr_Imine, "fr_NH0": Fragments.fr_NH0,
    "fr_NH1": Fragments.fr_NH1, "fr_NH2": Fragments.fr_NH2,
    "fr_N_O": Fragments.fr_N_O, "fr_Ndealkylation1": Fragments.fr_Ndealkylation1,
    "fr_Ndealkylation2": Fragments.fr_Ndealkylation2,
    "fr_Nhpyrrole": Fragments.fr_Nhpyrrole, "fr_SH": Fragments.fr_SH,
    "fr_aldehyde": Fragments.fr_aldehyde,
    "fr_alkyl_carbamate": Fragments.fr_alkyl_carbamate,
    "fr_alkyl_halide": Fragments.fr_alkyl_halide,
    "fr_allylic_oxid": Fragments.fr_allylic_oxid, "fr_amide": Fragments.fr_amide,
    "fr_amidine": Fragments.fr_amidine, "fr_aniline": Fragments.fr_aniline,
    "fr_aryl_methyl": Fragments.fr_aryl_methyl, "fr_azide": Fragments.fr_azide,
    "fr_azo": Fragments.fr_azo, "fr_barbitur": Fragments.fr_barbitur,
    "fr_benzene": Fragments.fr_benzene,
    "fr_benzodiazepine": Fragments.fr_benzodiazepine,
    "fr_bicyclic": Fragments.fr_bicyclic, "fr_diazo": Fragments.fr_diazo,
    "fr_dihydropyridine": Fragments.fr_dihydropyridine,
    "fr_epoxide": Fragments.fr_epoxide, "fr_ester": Fragments.fr_ester,
    "fr_ether": Fragments.fr_ether, "fr_furan": Fragments.fr_furan,
    "fr_guanido": Fragments.fr_guanido, "fr_halogen": Fragments.fr_halogen,
    "fr_hdrzine": Fragments.fr_hdrzine, "fr_hdrzone": Fragments.fr_hdrzone,
    "fr_imidazole": Fragments.fr_imidazole, "fr_imide": Fragments.fr_imide,
    "fr_isocyan": Fragments.fr_isocyan, "fr_isothiocyan": Fragments.fr_isothiocyan,
    "fr_ketone": Fragments.fr_ketone,
    "fr_ketone_Topliss": Fragments.fr_ketone_Topliss,
    "fr_lactam": Fragments.fr_lactam, "fr_lactone": Fragments.fr_lactone,
    "fr_methoxy": Fragments.fr_methoxy, "fr_morpholine": Fragments.fr_morpholine,
    "fr_nitrile": Fragments.fr_nitrile, "fr_nitro": Fragments.fr_nitro,
    "fr_nitro_arom": Fragments.fr_nitro_arom,
    "fr_nitro_arom_nonortho": Fragments.fr_nitro_arom_nonortho,
    "fr_nitroso": Fragments.fr_nitroso, "fr_oxazole": Fragments.fr_oxazole,
    "fr_oxime": Fragments.fr_oxime,
    "fr_para_hydroxylation": Fragments.fr_para_hydroxylation,
    "fr_phenol": Fragments.fr_phenol,
    "fr_phenol_noOrthoHbond": Fragments.fr_phenol_noOrthoHbond,
    "fr_phos_acid": Fragments.fr_phos_acid,
    "fr_phos_ester": Fragments.fr_phos_ester,
    "fr_piperdine": Fragments.fr_piperdine, "fr_piperzine": Fragments.fr_piperzine,
    "fr_priamide": Fragments.fr_priamide,
    "fr_prisulfonamd": Fragments.fr_prisulfonamd,
    "fr_pyridine": Fragments.fr_pyridine, "fr_quatN": Fragments.fr_quatN,
    "fr_sulfide": Fragments.fr_sulfide, "fr_sulfonamd": Fragments.fr_sulfonamd,
    "fr_sulfone": Fragments.fr_sulfone,
    "fr_term_acetylene": Fragments.fr_term_acetylene,
    "fr_tetrazole": Fragments.fr_tetrazole, "fr_thiazole": Fragments.fr_thiazole,
    "fr_thiocyan": Fragments.fr_thiocyan, "fr_thiophene": Fragments.fr_thiophene,
    "fr_unbrch_alkane": Fragments.fr_unbrch_alkane, "fr_urea": Fragments.fr_urea,
}


_BCUT_FIELDS = (
    "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
)


def _smiles_input_summary(smiles: str) -> str:
    """Return a bounded, readable identifier for warning messages."""
    text = str(smiles)
    preview = text[:80]
    suffix = "..." if len(text) > 80 else ""
    return f"{preview!r}{suffix} (length={len(text)})"


def compute_descriptors(
    smiles: str, *, return_status: bool = False
) -> Dict[str, float]:
    """Compute all 200+ RDKit ADMET descriptors.

    The default return value remains the historical descriptor dictionary.
    ``return_status=True`` additionally returns a diagnostic dictionary so a
    descriptor calculation failure cannot be mistaken for a genuine zero:
    ``(values, {"parse_ok": bool, "failed_descriptors": [...]})``.
    """
    status = {"parse_ok": False, "failed_descriptors": []}

    def finish(values):
        return (values, status) if return_status else values

    neut = neutralize_full(smiles)
    try:
        mol = Chem.MolFromSmiles(neut)
    except Exception:
        mol = None
    if mol is None:
        warnings.warn(
            "RDKit SMILES parse failed for input "
            f"{_smiles_input_summary(smiles)}; descriptor values defaulted to 0.0",
            RuntimeWarning,
            stacklevel=2,
        )
        status["failed_descriptors"] = list(DESCRIPTOR_FIELDS)
        return finish({f: 0.0 for f in DESCRIPTOR_FIELDS})
    status["parse_ok"] = True

    result = {}

    # Standard descriptors
    for name, func in _std_desc_map.items():
        try:
            val = func(mol)
            result[name] = float(val) if val is not None else 0.0
        except Exception:
            result[name] = 0.0
            status["failed_descriptors"].append(name)

    # EState indices
    try:
        estates = EState.EStateIndices(mol)
        has_estates = len(estates) > 0
        result["MaxEStateIndex"] = max(estates) if has_estates else 0.0
        result["MinEStateIndex"] = min(estates) if has_estates else 0.0
        result["MaxAbsEStateIndex"] = max(abs(v) for v in estates) if has_estates else 0.0
        result["MinAbsEStateIndex"] = min(abs(v) for v in estates) if has_estates else 0.0
    except Exception:
        for k in ["MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex"]:
            result[k] = 0.0
        status["failed_descriptors"].extend([
            "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex",
            "MinAbsEStateIndex",
        ])

    # BCUT2D
    try:
        bcuts = rdMolDescriptors.BCUT2D(mol)
        if len(bcuts) < len(_BCUT_FIELDS):
            raise ValueError(
                f"BCUT2D returned {len(bcuts)} values; expected {len(_BCUT_FIELDS)}"
            )
        for i, label in enumerate(_BCUT_FIELDS):
            result[label] = float(bcuts[i])
    except Exception:
        for label in _BCUT_FIELDS:
            result[label] = 0.0
        status["failed_descriptors"].extend(_BCUT_FIELDS)
        warnings.warn(
            "BCUT2D descriptor calculation failed for input "
            f"{_smiles_input_summary(smiles)}; BCUT2D fields defaulted to 0.0",
            RuntimeWarning,
            stacklevel=2,
        )

    # VSA descriptor groups — use named descriptors (vector funcs removed in RDKit 2026)
    _vsa_groups = [
        ("PEOE_VSA", 14), ("SMR_VSA", 10), ("SlogP_VSA", 12),
        ("EState_VSA", 11), ("VSA_EState", 10),
    ]
    for prefix, n in _vsa_groups:
        for i in range(1, n + 1):
            name = f"{prefix}{i}"
            try:
                result[name] = float(getattr(Descriptors, name)(mol))
            except Exception:
                result[name] = 0.0
                status["failed_descriptors"].append(name)

    # Fragment counts
    for name, func in _frag_funcs.items():
        try:
            result[name] = int(func(mol))
        except Exception:
            result[name] = 0
            status["failed_descriptors"].append(name)

    return finish(result)


def compute_descriptors_with_status(smiles: str):
    """Explicit diagnostic form of :func:`compute_descriptors`."""
    return compute_descriptors(smiles, return_status=True)


# ============================================================================
# SECTION 3: IUPAC naming (PubChem → InChI fallback)
# ============================================================================

def _pubchem_iupac(smiles: str) -> str:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{urllib.parse.quote(smiles)}/property/IUPACName/JSON"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=PUBCHEM_TIMEOUT) as resp:
            data = json.loads(resp.read())
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props and props[0].get("CID", 0) > 0:
                return props[0].get("IUPACName", "").strip()
    except Exception:
        pass
    return ""


def _inchi_descriptor(smiles: str) -> str:
    """Generate a chemically-descriptive name from InChI formula layer."""
    neut = neutralize_backbone(smiles)
    mol = Chem.MolFromSmiles(neut)
    if mol is None:
        return ""
    inchi = Chem.MolToInchi(mol)
    if not inchi or "/" not in inchi:
        return ""

    parts = inchi.split("/")
    if len(parts) < 2:
        return ""
    formula = parts[1]

    atom_counts = {}
    i = 0
    s = formula
    while i < len(s):
        elem = s[i]
        i += 1
        while i < len(s) and s[i].islower():
            elem += s[i]
            i += 1
        num_str = ""
        while i < len(s) and s[i].isdigit():
            num_str += s[i]
            i += 1
        atom_counts[elem] = int(num_str) if num_str else 1

    has_n = atom_counts.get("N", 0)
    has_o = atom_counts.get("O", 0)
    has_s = atom_counts.get("S", 0)
    has_p = atom_counts.get("P", 0)

    name_parts = []
    if has_n >= 1 and has_o >= 2:
        name_parts.append("amino acid derivative")
    elif has_n >= 1:
        name_parts.append("amino compound")
    if has_s:
        name_parts.append("S-containing")
    if has_p:
        name_parts.append("P-containing")
    halogens = [h for h in ["F", "Cl", "Br", "I"] if h in atom_counts]
    if halogens:
        name_parts.append("halogenated-" + "/".join(halogens))

    formula_str = "".join(
        f"{k}{v}" if v > 1 else k
        for k, v in sorted(atom_counts.items())
        if k not in ["H"]
    )
    name_parts.append(f"[{formula_str}]")
    return " ".join(name_parts)


def generate_iupac_name(smiles: str) -> str:
    """Generate IUPAC name: PubChem → InChI descriptor fallback."""
    can, _ = canonicalize(smiles)
    if not can:
        return ""

    name = _pubchem_iupac(can)
    if name:
        return name
    return _inchi_descriptor(smiles)


def generate_iupac_names(records: List[dict], cache: dict) -> Tuple[int, int, int]:
    """Fill iupac_name for records that need it. Returns (filled, pubchem, fallback)."""
    filled = 0
    pubchem = 0
    fallback = 0
    t0 = time.time()

    for idx, r in enumerate(records):
        if r.get("iupac_name", "").strip():
            continue
        smi = r["smiles_original"]

        if smi in cache:
            if cache[smi]:
                r["iupac_name"] = cache[smi]
                filled += 1
            continue

        can, _ = canonicalize(smi)
        if not can:
            continue

        name = _pubchem_iupac(can)
        if name:
            pubchem += 1
        else:
            name = _inchi_descriptor(smi)
            if name:
                fallback += 1

        if name:
            r["iupac_name"] = name
            cache[smi] = name
            filled += 1
        else:
            cache[smi] = ""

        time.sleep(PUBCHEM_DELAY)

        if (idx + 1) % 2000 == 0:
            elapsed = max(time.time() - t0, 1)
            rate = (idx + 1) / elapsed
            n_done = sum(1 for r2 in records if r2.get("iupac_name", "").strip())
            print(f"    {idx + 1}/{len(records)} rate={rate:.1f}/s "
                  f"filled={n_done} PC={pubchem} FB={fallback}", flush=True)

    return filled, pubchem, fallback


# ============================================================================
# SECTION 4: HELM fields (CXSMILES generation)
# ============================================================================

HELM_FIELDS = ["replaced_SMILES", "CXSMILES", "Monomer_Type",
               "Polymer_Type", "R1", "R2", "R3"]

# Backbone SMARTS in priority order.
# Each matches: N, [bridge C atoms...], C(=O)OH (carboxyl C, =O, OH).
# H counts relaxed to handle both side-chain-substituted and unsubstituted cases.
BACKBONE_PATTERNS = [
    ("alpha_aa", Chem.MolFromSmarts("[N;!R;H2]-[C;!R]-[C;!R](=[O;!R])-[O;!R;H1]")),
    ("beta_aa", Chem.MolFromSmarts("[N;!R;H2]-[C;!R]-[C;!R]-[C;!R](=[O;!R])-[O;!R;H1]")),
    ("gamma_aa", Chem.MolFromSmarts("[N;!R;H2]-[C;!R]-[C;!R]-[C;!R]-[C;!R](=[O;!R])-[O;!R;H1]")),
    ("n_alkyl_aa", Chem.MolFromSmarts("[N;!R;H1]-[C;!R]-[C;!R](=[O;!R])-[O;!R;H1]")),
]

# Side-chain functional groups for R3 inference.
# Semantics:
#   R3 = "OH" → side chain has an extra -COOH (leaves as H2O on connection)
#   R3 = "H"  → side chain has -NH2 / -OH / -SH (leaves as H2 on connection)
#   R3 = "-"  → no modifiable side-chain group
#
# Key: must distinguish "side chain" from "backbone variant":
#   - CONH2 at C-term is R2's variant (C(=O)NH2 vs C(=O)OH), NOT an R3 site
#   - Guanidino NH2 is not a standard attachment point
#   - Aromatic NH2 (aniline) is not a standard attachment point
#   - Only aliphatic NH2/OH/SH count for R3

# COOH anywhere not in backbone → R3=OH
SIDECHAIN_COOH = Chem.MolFromSmarts("[C;!R](=[O;!R])-[O;!R;H1]")

# Aliphatic primary amine: NH2–CH2– (connected to sp3 carbon, not amide/guanidine/aniline)
SIDECHAIN_ALIPH_NH2 = Chem.MolFromSmarts("[N;!R;H2]-[C;!R;H2]")

# Aliphatic hydroxyl: OH–CH– or OH–CH2– (not phenol, not carboxyl OH)
SIDECHAIN_ALIPH_OH = Chem.MolFromSmarts("[O;!R;H1]-[C;!R;H1,H2]")

# Aliphatic thiol: SH–CH– or SH–CH2– (not thiophenol)
SIDECHAIN_ALIPH_SH = Chem.MolFromSmarts("[S;!R;H1]-[C;!R;H1,H2]")


def _infer_r3(mol: object, backbone_atom_indices: tuple) -> str:
    """Infer R3 (side-chain attachment point leaving group) by scanning for
    functional groups NOT belonging to the backbone match.
    Returns one of: \"OH\" (extra COOH), \"H\" (extra NH2/OH/SH), \"-\" (none)."""
    bb_set = set(backbone_atom_indices)

    # Extra COOH not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_COOH):
        c_atom = match[0]
        if c_atom not in bb_set:
            return "OH"

    # Extra aliphatic NH2 not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_ALIPH_NH2):
        if match[0] not in bb_set:
            return "H"

    # Extra aliphatic OH not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_ALIPH_OH):
        if match[0] not in bb_set:
            return "H"

    # Extra aliphatic SH not in backbone
    for match in mol.GetSubstructMatches(SIDECHAIN_ALIPH_SH):
        if match[0] not in bb_set:
            return "H"

    return "-"


def _gen_cxsmiles(neutral_smi: str) -> dict:
    """Core CXSMILES generator for one neutral SMILES."""
    mol = Chem.MolFromSmiles(neutral_smi)
    if mol is None:
        return _helm_fallback(neutral_smi)

    for patt_name, patt in BACKBONE_PATTERNS:
        matches = mol.GetSubstructMatches(patt)
        if matches:
            return _build_cx(mol, matches[0], neutral_smi)

    return _fallback_cx(mol, neutral_smi)


def _build_cx(mol, match: tuple, neutral_smi: str) -> dict:
    """Build CXSMILES from a backbone SMARTS match.
    Backbone match layout (applies to all patterns):
      match[0] = N, match[-3] = carboxyl-C, match[-2] = =O, match[-1] = -OH
    """
    n_idx = match[0]
    o_sgl_idx = match[-1]

    mol_h = Chem.AddHs(mol)
    h_on_n = None
    for nb in mol_h.GetAtomWithIdx(n_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_n = nb.GetIdx()
            break
    h_on_oh = None
    for nb in mol_h.GetAtomWithIdx(o_sgl_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_oh = nb.GetIdx()
            break

    if h_on_n is None or h_on_oh is None:
        return _fallback_cx(mol, neutral_smi)

    rw = Chem.RWMol(mol_h)
    rw.GetAtomWithIdx(h_on_n).SetAtomicNum(0)
    rw.RemoveAtom(h_on_oh)
    rw.GetAtomWithIdx(o_sgl_idx).SetAtomicNum(0)

    final_mol = rw.GetMol()
    final_mol = Chem.RemoveHs(final_mol, updateExplicitCount=True)
    try:
        Chem.SanitizeMol(final_mol)
    except Exception:
        pass

    cx_raw = Chem.MolToSmiles(final_mol)
    cx_mol = Chem.MolFromSmiles(cx_raw)
    if cx_mol is None:
        return _helm_fallback(neutral_smi)

    star_atoms = [a.GetIdx() for a in cx_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(star_atoms) < 2:
        return _helm_fallback(neutral_smi)

    r1_idx = r2_idx = None
    for sa in star_atoms:
        for nb in cx_mol.GetAtomWithIdx(sa).GetNeighbors():
            if nb.GetAtomicNum() == 7:
                r1_idx = sa
            elif nb.GetAtomicNum() == 6:
                r2_idx = sa
    if r1_idx is None:
        r1_idx = star_atoms[0]
    if r2_idx is None:
        r2_idx = star_atoms[1]
    if r1_idx == r2_idx:
        others = [s for s in star_atoms if s != r1_idx]
        r2_idx = others[0] if others else r1_idx

    n_atoms = cx_mol.GetNumAtoms()
    labels = [""] * n_atoms
    labels[r1_idx] = "_R1"
    labels[r2_idx] = "_R2"

    r3 = _infer_r3(mol, match)

    return {
        "replaced_SMILES": neutral_smi,
        "CXSMILES": f"{cx_raw} |${';'.join(labels)}$|",
        "Monomer_Type": "Backbone",
        "Polymer_Type": "PEPTIDE",
        "R1": "H",
        "R2": "OH",
        "R3": r3,
    }


def _fallback_cx(mol: object, neutral_smi: str) -> dict:
    """Find any amine + any carboxyl, mark as connection points."""
    amine_m = mol.GetSubstructMatches(SIDECHAIN_ALIPH_NH2)
    carboxyl_m = mol.GetSubstructMatches(SIDECHAIN_COOH)
    if not amine_m or not carboxyl_m:
        return _helm_fallback(neutral_smi)

    n_idx = amine_m[0][0]
    o_sgl_idx = carboxyl_m[0][2]

    mol_h = Chem.AddHs(mol)
    h_on_n = None
    for nb in mol_h.GetAtomWithIdx(n_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_n = nb.GetIdx()
            break
    h_on_oh = None
    for nb in mol_h.GetAtomWithIdx(o_sgl_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_on_oh = nb.GetIdx()
            break

    if h_on_n is None or h_on_oh is None:
        return _helm_fallback(neutral_smi)

    rw = Chem.RWMol(mol_h)
    rw.GetAtomWithIdx(h_on_n).SetAtomicNum(0)
    rw.RemoveAtom(h_on_oh)
    rw.GetAtomWithIdx(o_sgl_idx).SetAtomicNum(0)

    final_mol = rw.GetMol()
    final_mol = Chem.RemoveHs(final_mol, updateExplicitCount=True)
    try:
        Chem.SanitizeMol(final_mol)
    except Exception:
        pass

    cx_raw = Chem.MolToSmiles(final_mol)
    cx_mol = Chem.MolFromSmiles(cx_raw)
    if cx_mol is None:
        return _helm_fallback(neutral_smi)

    star_atoms = [a.GetIdx() for a in cx_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not star_atoms:
        return _helm_fallback(neutral_smi)

    n_atoms = cx_mol.GetNumAtoms()
    labels = [""] * n_atoms
    if len(star_atoms) >= 1:
        labels[star_atoms[0]] = "_R1"
    if len(star_atoms) >= 2:
        labels[star_atoms[1]] = "_R2"

    # Build backbone atom set for R3 inference
    bb_set = {amine_m[0][0], carboxyl_m[0][0], carboxyl_m[0][1], o_sgl_idx}
    r3 = _infer_r3(mol, tuple(bb_set))

    return {
        "replaced_SMILES": neutral_smi,
        "CXSMILES": f"{cx_raw} |${';'.join(labels)}$|",
        "Monomer_Type": "Backbone",
        "Polymer_Type": "PEPTIDE",
        "R1": "H",
        "R2": "OH",
        "R3": r3,
    }


def _helm_fallback(neutral_smi: str) -> dict:
    return {
        "replaced_SMILES": neutral_smi,
        "CXSMILES": "",
        "Monomer_Type": "Backbone",
        "Polymer_Type": "PEPTIDE",
        "R1": "H",
        "R2": "OH",
        "R3": "-",
    }


def generate_helm_fields(records: List[dict], cyc_data: dict) -> int:
    """Fill HELM fields. CycPeptMPDB entries imported directly; NNAA generated.

    After generation, validates each CXSMILES by parsing it back and
    checking that both [_R1] and [_R2] labels are present. Invalid
    entries are regenerated via fallback or marked with empty CXSMILES.
    """
    filled = 0
    validated = 0
    invalid = 0

    for r in records:
        sym = r.get("symbol", "").strip()
        if sym in cyc_data:
            for k, v in cyc_data[sym].items():
                r[k] = v
            if r.get("CXSMILES"):
                filled += 1
            continue

        if r.get("CXSMILES", "").strip():
            filled += 1
            continue

        smi = r.get("smiles_original", "")
        if not smi:
            continue

        neutral = neutralize_backbone(smi)
        _, mol = canonicalize(neutral)
        if mol is None:
            mol = Chem.MolFromSmiles(neutral)
        if mol is None:
            result = _helm_fallback(neutral)
        else:
            result = _gen_cxsmiles(neutral)

        for hf in HELM_FIELDS:
            r[hf] = result.get(hf, "")
        if r.get("CXSMILES"):
            filled += 1

    # ── Post-generation validation ──────────────────────────────────────
    print("  Validating CXSMILES labels...")
    for r in records:
        cx = r.get("CXSMILES", "").strip()
        if not cx:
            continue
        # Inline validation: parse CXSMILES and check for R1/R2 labels
        try:
            smi_list = cx.split('|')
            smi_part = smi_list[0].strip()
            pos_part = smi_list[1] if len(smi_list) > 1 else ""
            # Check position block for _R1 and _R2
            has_r1 = '_R1' in pos_part
            has_r2 = '_R2' in pos_part
            if has_r1 and has_r2:
                validated += 1
                continue
        except Exception:
            pass
        # Validation failed — try regenerating
        invalid += 1
        smi = r.get("smiles_original", "")
        if smi:
            neutral = neutralize_backbone(smi)
            result = _gen_cxsmiles(neutral)
            new_cx = result.get("CXSMILES", "")
            if new_cx:
                try:
                    smi_list2 = new_cx.split('|')
                    pos_part2 = smi_list2[1] if len(smi_list2) > 1 else ""
                    if '_R1' in pos_part2 and '_R2' in pos_part2:
                        for hf in HELM_FIELDS:
                            r[hf] = result.get(hf, "")
                        validated += 1
                        invalid -= 1
                        continue
                except Exception:
                    pass
        # Still invalid — mark as fallback
        for hf in HELM_FIELDS:
            r[hf] = _helm_fallback(smi if smi else "")[hf]

    print(f"  CXSMILES validated: {validated}, invalid (fallback): {invalid}")
    return filled


# ============================================================================
# SECTION 5: Data loading & matching
# ============================================================================

def load_nnaa(filepath: str) -> List[dict]:
    """Load NNAA_10000.txt: SMILES <tab> 3-letter-code."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            records.append({"source": "NNAA", "symbol": parts[-1],
                           "smiles_original": parts[0]})
    return records


def load_cycpep(filepath: str) -> List[dict]:
    """Load CycPeptMPDB_Monomer_All.csv."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


# ============================================================================
# SECTION 6: Output schema
# ============================================================================

UNIFIED_HEADER = [
    "monomer_id", "symbol", "source",
    "smiles_canonical", "smiles_canonical_nostereo", "smiles_original",
    "iupac_name", "compound_name", "iupac_condensed",
    "monomer_type", "polymer_type", "natural_analog",
    "pubchem_cid", "version",
    "capped_smiles", "contain_pepnum", "contain_perme",
    "best_cycpep_match_id", "best_cycpep_match_symbol", "tanimoto_similarity",
] + DESCRIPTOR_FIELDS + HELM_FIELDS


def build_record(uid: int, rec: dict) -> dict:
    """Fill unified schema from record dict."""
    row = {}
    row["monomer_id"] = uid
    row["symbol"] = rec.get("symbol", "")
    row["source"] = rec.get("source", "")
    row["smiles_canonical"] = rec.get("smiles_canonical", "")
    row["smiles_canonical_nostereo"] = rec.get("smiles_canonical_nostereo", "")
    row["smiles_original"] = rec.get("smiles_original", "")
    row["iupac_name"] = rec.get("iupac_name", "")
    row["compound_name"] = rec.get("compound_name", rec.get("Compound_Name", ""))
    row["iupac_condensed"] = rec.get("iupac_condensed", rec.get("IUPAC_Condensed", ""))
    row["monomer_type"] = rec.get("monomer_type", rec.get("Monomer_Type", "NNAA"))
    row["polymer_type"] = rec.get("polymer_type", rec.get("Polymer_Type", ""))
    row["natural_analog"] = rec.get("natural_analog", rec.get("Natural_Analog", ""))
    row["pubchem_cid"] = rec.get("pubchem_cid", rec.get("PubChem_CID", ""))
    row["version"] = rec.get("version", "1.0")
    row["capped_smiles"] = rec.get("capped_smiles", rec.get("capped_SMILES", ""))
    row["contain_pepnum"] = rec.get("contain_pepnum", "")
    row["contain_perme"] = rec.get("contain_perme", "")
    row["best_cycpep_match_id"] = rec.get("best_cycpep_match_id", "")
    row["best_cycpep_match_symbol"] = rec.get("best_cycpep_match_symbol", "")
    row["tanimoto_similarity"] = rec.get("tanimoto_similarity", 0.0)
    for f in DESCRIPTOR_FIELDS:
        row[f] = rec.get(f, 0.0)
    for f in HELM_FIELDS:
        row[f] = rec.get(f, "")
    return row


# ============================================================================
# SECTION 7: Main pipeline
# ============================================================================

def _option_value(argv: List[str], option: str, default: str) -> str:
    """Read a simple ``--option value`` or ``--option=value`` argument."""
    for index, arg in enumerate(argv):
        if arg == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            return argv[index + 1]
        prefix = option + "="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def main():
    argv = list(sys.argv[1:])
    args = set(argv)
    skip_iupac = "--skip-iupac" in args
    skip_helm = "--skip-helm" in args
    input_nnaa = os.path.abspath(_option_value(argv, "--nnaa", NNAA_PATH))
    input_cycpep = os.path.abspath(_option_value(argv, "--cycpep", CYCPEP_PATH))
    output_path = os.path.abspath(_option_value(argv, "--output", OUTPUT_PATH))
    cache_path = os.path.abspath(_option_value(argv, "--cache", CACHE_PATH))

    print("=" * 65)
    print("  Monomer Library Builder")
    print("  NNAA + CycPeptMPDB -> unified_monomer_library.csv")
    print("=" * 65)

    # ── 1. Load datasets ──────────────────────────────────────────────
    print("\n[1/6] Loading source datasets...")
    nnaa = load_nnaa(input_nnaa)
    print(f"  NNAA: {len(nnaa):,} entries")

    cycpep_rows = []
    cyc_data = {}
    if os.path.exists(input_cycpep):
        cycpep_raw = load_cycpep(input_cycpep)
        print(f"  CycPeptMPDB: {len(cycpep_raw):,} entries")
        for cr in cycpep_raw:
            sym = cr.get("Symbol", "").strip()
            cr["source"] = "CycPeptMPDB"
            cr["symbol"] = sym
            if sym:
                cyc_data[sym] = {f: cr.get(f, "") for f in HELM_FIELDS}
        cycpep_rows = cycpep_raw
    else:
        print("  CycPeptMPDB: not found, will build NNAA-only library")

    # ── 2. Canonicalize & match ───────────────────────────────────────
    print("\n[2/6] Canonicalizing SMILES & matching...")

    for r in nnaa:
        can, _ = canonicalize(r["smiles_original"])
        can_ns, _ = canonicalize(r["smiles_original"], strip_stereo=True)
        r["smiles_canonical"] = can
        r["smiles_canonical_nostereo"] = can_ns

    for r in cycpep_rows:
        smi = r.get("replaced_SMILES", r.get("SMILES", ""))
        can, _ = canonicalize(smi)
        can_ns, _ = canonicalize(smi, strip_stereo=True)
        r["smiles_canonical"] = can
        r["smiles_canonical_nostereo"] = can_ns
        r["smiles_original"] = smi

    # Build CycPeptMPDB lookup
    cyc_by_can = {}
    for cr in cycpep_rows:
        cyc_by_can[cr["smiles_canonical"]] = cr
    cyc_by_can_ns = {}
    for cr in cycpep_rows:
        if cr["smiles_canonical_nostereo"] not in cyc_by_can_ns:
            cyc_by_can_ns[cr["smiles_canonical_nostereo"]] = cr

    matched_cyc = set()
    nnaa_cyc_match = {}
    for i, r in enumerate(nnaa):
        key = r["smiles_canonical"]
        if key in cyc_by_can:
            nnaa_cyc_match[i] = cyc_by_can[key]
            matched_cyc.add(id(cyc_by_can[key]))
        elif r["smiles_canonical_nostereo"] in cyc_by_can_ns:
            nnaa_cyc_match[i] = cyc_by_can_ns[r["smiles_canonical_nostereo"]]
            matched_cyc.add(id(cyc_by_can_ns[r["smiles_canonical_nostereo"]]))

    # Compute Tanimoto for unmatched NNAA
    cyc_fps = [(cr, morgan_fp(cr["smiles_canonical"])) for cr in cycpep_rows]

    nnaa_sim = {}
    for i, r in enumerate(nnaa):
        if i in nnaa_cyc_match:
            continue
        fp = morgan_fp(r["smiles_canonical"])
        if fp is None:
            nnaa_sim[i] = ("", "", 0.0)
            continue
        best = 0.0
        best_cr = None
        for cr, cfp in cyc_fps:
            if cfp is None:
                continue
            s = DataStructs.TanimotoSimilarity(fp, cfp)
            if s > best:
                best = s
                best_cr = cr
        nnaa_sim[i] = (best_cr["ID"] if best_cr else "",
                       best_cr["Symbol"] if best_cr else "",
                       round(best, 4))

    print(f"  NNAA matched to CycPeptMPDB: {len(nnaa_cyc_match)}")
    print(f"  Unmatched NNAA: {len(nnaa) - len(nnaa_cyc_match)}")

    # ── 3. Compute descriptors ────────────────────────────────────────
    print("\n[3/6] Computing RDKit ADMET descriptors...")
    nnaa_desc = {}
    for i, r in enumerate(nnaa):
        nnaa_desc[i] = compute_descriptors(r["smiles_original"])
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(nnaa)}...")
    print(f"  Done: {len(nnaa_desc)} NNAA descriptor sets")

    # ── 4. IUPAC naming ───────────────────────────────────────────────
    print("\n[4/6] Generating IUPAC names...")
    cache = load_cache(cache_path) if not skip_iupac else {}

    if not skip_iupac:
        # For CycPeptMPDB entries: use their existing IUPAC_Name
        for cr in cycpep_rows:
            if not cr.get("iupac_name"):
                cr["iupac_name"] = cr.get("IUPAC_Name", "")

        # For NNAA: generate
        filled, pc, fb = generate_iupac_names(nnaa, cache)
        save_cache(cache_path, cache)
        print(f"  NNAA IUPAC: {filled} filled (PubChem={pc}, InChI={fb})")
    else:
        print("  Skipped (--skip-iupac)")
        for cr in cycpep_rows:
            cr.setdefault("iupac_name", cr.get("IUPAC_Name", ""))

    # ── 5. Build unified records ──────────────────────────────────────
    print("\n[5/6] Building unified records...")
    unified = []
    uid = 0

    # CycPeptMPDB-only entries
    for cr in cycpep_rows:
        if id(cr) not in matched_cyc:
            uid += 1
            cr["best_cycpep_match_id"] = cr.get("ID", "")
            cr["best_cycpep_match_symbol"] = cr.get("Symbol", "")
            cr["tanimoto_similarity"] = 1.0
            unified.append(build_record(uid, cr))

    # NNAA entries (all — matched + novel)
    for i, nr in enumerate(nnaa):
        uid += 1
        if i in nnaa_cyc_match:
            cr = nnaa_cyc_match[i]
            merged = dict(nr)
            merged["iupac_name"] = nr.get("iupac_name", "") or cr.get("IUPAC_Name", "")
            merged["compound_name"] = cr.get("Compound_Name", "")
            merged["iupac_condensed"] = cr.get("IUPAC_Condensed", "")
            merged["best_cycpep_match_id"] = cr.get("ID", "")
            merged["best_cycpep_match_symbol"] = cr.get("Symbol", "")
            merged["tanimoto_similarity"] = 1.0
            for f in DESCRIPTOR_FIELDS:
                merged[f] = cr.get(f, 0.0)
            unified.append(build_record(uid, merged))
        else:
            nr["monomer_type"] = "NNAA"
            best_id, best_sym, best_sim = nnaa_sim.get(i, ("", "", 0.0))
            nr["best_cycpep_match_id"] = best_id
            nr["best_cycpep_match_symbol"] = best_sym
            nr["tanimoto_similarity"] = best_sim
            desc = nnaa_desc.get(i, {})
            for f in DESCRIPTOR_FIELDS:
                nr[f] = desc.get(f, 0.0)
            unified.append(build_record(uid, nr))

    print(f"  Total unified records: {len(unified):,}")

    # ── 6. HELM fields ────────────────────────────────────────────────
    print("\n[6/6] Generating HELM fields...")

    if not skip_helm:
        n_helm = generate_helm_fields(unified, cyc_data)
        print(f"  Records with CXSMILES: {n_helm:,}/{len(unified):,} "
              f"({100 * n_helm / len(unified):.1f}%)")
    else:
        print("  Skipped (--skip-helm)")

    # ── Write output ──────────────────────────────────────────────────
    print(f"\nWriting {output_path}...")
    tmp = output_path + ".tmp"

    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNIFIED_HEADER, extrasaction="ignore")
        w.writeheader()
        w.writerows(unified)

    # Replace original (retry on lock)
    for attempt in range(10):
        try:
            os.replace(tmp, output_path)
            break
        except PermissionError:
            if attempt < 9:
                time.sleep(0.5)
            else:
                alt = output_path + ".new"
                os.replace(tmp, alt)
                print(f"WARNING: Output locked. Written to {alt}")
                print("Close any programs holding the file and rename manually.")

    size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
    print(f"  {len(unified):,} rows x {len(UNIFIED_HEADER)} cols, {size_mb:.1f} MB")

    # ── Summary ───────────────────────────────────────────────────────
    sources = Counter(r["source"] for r in unified)
    types = Counter(r["monomer_type"] for r in unified)
    iupac_ok = sum(1 for r in unified if r.get("iupac_name", "").strip())
    cx_ok = sum(1 for r in unified if r.get("CXSMILES", "").strip())

    print(f"\n{'=' * 65}")
    print(f"  BUILD SUMMARY")
    print(f"  Total: {len(unified):,} records")
    print(f"  Sources: {dict(sources)}")
    print(f"  Monomer types: {dict(types)}")
    print(f"  IUPAC names: {iupac_ok}/{len(unified)}")
    print(f"  CXSMILES:    {cx_ok}/{len(unified)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
