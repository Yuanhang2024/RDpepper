# Third-party data and redistribution status

Each bundled third-party resource keeps the license of its source as
identified below. The software code is MIT; the data are not MIT and are
redistributed under the terms stated per resource.

Paths below are relative to the source root; installed paths have the
`cycpep_master/` prefix. The original runtime data are retained unchanged so
this preparation does not silently create a different benchmarked product.

## Diverse NNAA collection: CC BY-NC 4.0

- Amarasinghe, K. N.; De Maria, L.; Tyrchan, C.; Eriksson, L. A.;
  Sadowski, J.; Petrovic, D. *Virtual Screening Expands the Non-Natural
  Amino Acid Palette for Peptide Optimization.* Journal of Chemical
  Information and Modeling 62, 2999-3007 (2022).
  https://doi.org/10.1021/acs.jcim.2c00193
- Official TXT dataset: https://doi.org/10.1021/acs.jcim.2c00193.s001
  (ACS Figshare record 20066931; `ci2c00193_si_001.txt`).
  Official metadata: https://api.figshare.com/v2/articles/20066931
- License: Creative Commons Attribution-NonCommercial 4.0 International,
  https://creativecommons.org/licenses/by-nc/4.0/.
  Attribution, a license link, identification of modifications and the
  non-commercial condition apply.
- The official file contains 9,998 nonempty SMILES-plus-identifier
  records; its published MD5 is `c069401cdd68f7278128049795a1f6e6`
  (552,588 bytes).
- Distributed resources: `libraries/nnaa_diverse.csv` and `source=NNAA`
  rows in `unified_monomer_library.csv` (9,998 rows).
- Changes: source molecule descriptions were converted to RDpepper
  monomer representations with attachment-port and library metadata,
  then compiled into CSV tables. Format conversion does not remove the
  source license. The article PDF itself is not redistributed.

## CycPeptMPDB monomers: CC BY 4.0

- Li, X.; Yanagisawa, K.; Sugita, M.; Fujie, T.; Ohue, M.; Akiyama, Y.
  *CycPeptMPDB: A Comprehensive Database of Membrane Permeability of
  Cyclic Peptides.* Journal of Chemical Information and Modeling 63,
  2240-2250 (2023). https://doi.org/10.1021/acs.jcim.2c01573
  Open-access article page:
  https://pubs.acs.org/jcisd8/article/63/7/2240/850420/
  CycPeptMPDB-A-Comprehensive-Database-of-Membrane
- Official database: http://cycpeptmpdb.com/
  Download page: http://cycpeptmpdb.com/download/
  Monomer file: `CycPeptMPDB_Monomer_All.csv` (385 upstream rows).
- License: Creative Commons Attribution 4.0 International,
  https://creativecommons.org/licenses/by/4.0/, per the open-access
  license of the article, whose data availability statement directs all
  recorded information to the free official download above.
- Distributed resources: 384 `source=CycPeptMPDB` rows in
  `unified_monomer_library.csv` and `libraries/curated_cycpep.csv`.
  Compilation merges one entry that also appears in the NNAA collection
  and adds RDpepper metadata columns.
- Changes: structural descriptors and attachment-port representations
  were added to the official rows. Attribution and a link to this
  license are retained here as the required notice.

A legacy source-tree table `data/MAP_momomers_library_new.csv`, obtained
from the MAP_HELM_SMILES repository (which states no repository-wide
license), is not included in this public distribution. Its monomer
content duplicates the official CycPeptMPDB table above; rebuild it from
the official download if that historical mapping column is needed.

## HELM-GPT consolidated monomers: MIT (project license)

- Xu, X. et al. *HELM-GPT: de novo macrocyclic peptide design using
  generative pre-trained transformer.* Bioinformatics 40, btae364
  (2024). https://doi.org/10.1093/bioinformatics/btae364
- Repository and license: https://github.com/charlesxu90/helm-gpt,
  MIT License, Copyright (c) 2021-2024 Charles Xu and others
  (copy retained at `licenses/HELM-GPT-MIT.txt`).
- The public distribution treats the repository MIT license as covering
  the project's monomer library, as determined by the repository
  maintainer.
- Distributed resources: 2,770 `source=HELM-GPT` rows in
  `unified_monomer_library.csv` and `libraries/helm_gpt.csv`.
- Upstream context, retained for attribution: the HELM-GPT paper
  describes its 3,104-monomer library as consolidating monomers from
  ChEMBL, CycPeptMPDB and KRAS HELMs. ChEMBL data are distributed under
  CC BY-SA 3.0 (official notice and required attribution retained at
  `licenses/ChEMBL-32-LICENSE.txt` and
  `licenses/ChEMBL-32-ATTRIBUTION.txt`; Mendez, D. et al., Nucleic Acids
  Research 47, D930-D940 (2019), DOI 10.1093/nar/gky1075). Rows in this
  subset follow the HELM-GPT project symbols; row-level upstream
  provenance within the consolidated library is not separately recorded
  by the source project.

## Structural templates and torsion summaries: CC BY 4.0

The package contains **coordinate templates as well as aggregate priors**:
794 files under `data/templates/`, comprising 568 third-party-derived
representatives and 226 RDpepper-generated synthetic templates.
`templates_index.json` records per-template source provenance.

### CPSea, CPBind and CPSea_PDB

Yang, Z.; Xie, H.; Jia, Y.; Kong, X.; Zheng, J.; Zhang, Z.; Liu, Y.; Liu, L.;
Lan, Y. *CPSea* (dataset), Zenodo.
https://doi.org/10.5281/zenodo.17324994

The record includes `CPBind.zip` and `CPSea_PDB.zip`, licensed CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/.
The supplied template subset contains 331 CPBind-derived and 228
CPSea_PDB-derived representatives. The same archives supply offline torsion
observations. CPBind is derived from AlphaFold Database structures and
CPSea_PDB from PDB structures, as described by the source record.

### AfCycDesign scaffold set

Rettie, S.; Bhardwaj, G.; Campbell, K. *Scaffolds from “Cyclic peptide
structure prediction and design using AlphaFold2”* (dataset), Zenodo.
https://doi.org/10.5281/zenodo.15164650

The record's `scaffold_set_7-13.tar.gz` and `scaffold_set_14-16.tar.gz`
are licensed CC BY 4.0. Nine derived representative templates are included.
Related article (citation only): Rettie et al., Nature Communications 16,
4730 (2025), https://doi.org/10.1038/s41467-025-59940-7.
The dataset license is distinct from the article license.

### Modifications and scope

RDpepper extracted peptide structures, assigned molecular representations,
clustered candidates and re-exported selected representatives as PDB files.
The complete source archives and their original headers are not included;
`source` and `source_filename` fields preserve source attribution.
`data/torsion_priors/` contains derived circular statistics, calibration
metadata and source inventory rather than coordinates. That statement about
aggregate priors does not apply to the separate coordinate template files.
These derived resources do not by themselves validate docking poses or affinity.
No source author endorsement of RDpepper is implied.

## Standard and special-residue tables

The 20-amino-acid core table and three cap rows are locally authored.
`special_residue_library.csv` and `libraries/special.csv` contain seven
locally authored descriptions referencing wwPDB Chemical Component Dictionary
codes (0EH, MK8, KYN, LME, DBU, FGA, DHA). wwPDB provides public-domain
structural data: https://www.wwpdb.org/. These references are retained.
The packaged derived-monomer table is empty in this freeze.

## Code and optional tools

Retain the inline MIT notices and original attribution for conversion
utilities in `paths/_map_utils.py`; `LICENSE` records the historical
MAP_HELM_SMILES attribution. It does not grant a new license to any
unlicensed third-party table. HELM-GPT's code notice is in `licenses/`.
RDKit, Gemmi and other installed dependencies retain their own licenses.
Open Babel, Meeko, PyQt5, ADMET backends and Vina are optional components,
not relicensed by RDpepper. No Vina executable is included in this candidate.

Full notices available with the candidate include CC BY 4.0, CC BY-NC 4.0,
HELM-GPT MIT, and the verified ChEMBL32 license and attribution text. Their
inclusion records source terms; it does not resolve the pending rights above.
