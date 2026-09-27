# Third-Party Data Attribution

RDpepper can include a compiled torsion-prior runtime index. The
index contains aggregate circular statistics and calibration metadata; it does
not contain or redistribute the source PDB coordinate files.

## Diverse 10,000 Non-Natural Amino Acid Library

- Publication: Amarasinghe, K. N.; De Maria, L.; Tyrchan, C.; Eriksson,
  L. A.; Sadowski, J.; Petrović, D. “Virtual Screening Expands the
  Non-Natural Amino Acid Palette for Peptide Optimization.” *J. Chem. Inf.
  Model.* **2022**, *62*, 2999–3007.
- DOI: https://doi.org/10.1021/acs.jcim.2c00193
- Source scope: the publication reports enumeration of nearly 380,000
  synthesizable non-natural amino acids and selection of a diverse 10,000
  member subset. Its Associated Content lists “Molecular formula strings for
  the diverse 10,000 amino acid library” as a TXT data file.
- Use in RDpepper: `build_monomer_library.py` reads `NNAA_10000.txt` and the
  current compiled unified library contains 9,998 rows with `source=NNAA`.
  The compiled-row count is reported separately from the publication’s
  10,000-member source subset.
- License/redistribution boundary: the local article PDF is © 2022 American
  Chemical Society and does not state an open-data redistribution license for
  the Associated Content TXT. Citation alone is not treated as redistribution
  permission. Public GitHub/PyPI release of the derived NNAA rows remains
  contingent on confirming the publisher/data terms or obtaining permission.

## CPBind and CPSea_PDB

- Record: CPSea
- DOI: https://doi.org/10.5281/zenodo.17324994
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Use in RDpepper: offline torsion observations from the CPBind and
  CPSea_PDB structure archives, followed by chemical-entity weighting and
  leave-one-entity-out / leave-one-source-out calibration.

## AfCycDesign Scaffold

- Record: Scaffolds from Cyclic peptide structure prediction and design using
  AlphaFold2
- DOI: https://doi.org/10.5281/zenodo.15164650
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Related publication: Rettie et al., "Cyclic peptide structure prediction and
  design using AlphaFold2," Nature Communications 16, 4730 (2025),
  https://doi.org/10.1038/s41467-025-59940-7.
- Local dataset alias used during the V4 build: `Scaffold`.
- Use in RDpepper: offline torsion observations from predicted/design
  cyclic-peptide structures.

The runtime manifest binds the exact source inventory, builder, calibration
protocol, and compiled resource by SHA-256. These data-derived priors are
flexibility evidence only and do not establish docking, pose, affinity, or
binding benefit.
