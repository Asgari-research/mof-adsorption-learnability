# Data sources and upstream provenance

## ARC-MOF source used by the benchmark

The four adsorption targets and the principal descriptor/grouping inputs in this repository originate from **ARC-MOF (ab initio REPEAT Charge MOF Database)**. Raw ARC-MOF files are not copied into this repository.

For strict historical provenance, use the **ARC-MOF v6** Zenodo release updated on **2024-10-04**:

- **Exact v6 record DOI:** https://doi.org/10.5281/zenodo.13891643
- **Persistent ARC-MOF concept DOI (all versions):** https://doi.org/10.5281/zenodo.6908727

The preserved historical pipeline includes the archive alias `ARCMOF_20241004.tar.gz`, which aligns with the v6 record. A later ARC-MOF version should not be substituted silently when attempting strict historical reproduction; record the version and checksums used.

## Peer-reviewed ARC-MOF paper

Please cite the database paper when using ARC-MOF:

> J. Burner, J. Luo, A. White, A. Mirmiran, O. Kwon, P. G. Boyd, S. Maley, M. Gibaldi, S. Simrod, V. Ogden, and T. K. Woo, "ARC-MOF: A Diverse Database of Metal-Organic Frameworks with DFT-Derived Partial Atomic Charges and Descriptors for Machine Learning," *Chemistry of Materials* **35** (3), 900-916 (2023). DOI: https://doi.org/10.1021/acs.chemmater.2c02485

## ARC-MOF files used by the historical pipeline

The core four-target benchmark is assembled from the following ARC-MOF v6 files. The MD5 values below are the checksums published by the v6 Zenodo record; they can be used to verify downloaded upstream files before a rerun.

| ARC-MOF v6 file | Historical role in this project | Upstream MD5 |
|---|---|---|
| `geometric_properties.csv` | 23 core geometry descriptors and MOF identifiers | `345ecb861674a3a25e7be580cd1ab716` |
| `post_comb_vsa-CO2.csv` | CO2 uptake targets at 0.015 and 0.150 bar, 298 K | `9d516ec31dd1c97f0e2b929df50ca249` |
| `methane.csv` | CH4 uptake targets at 5.8 and 65 bar, 298 K | `48ee74a8bc5e02fd90c043a09b63eb0b` |
| `RACs.csv` | RAC chemistry descriptors used by `geometry_plus_racs` | `0bc7315e8bab41cc115c2baf31ba2ca4` |
| `geo-clusters.csv` | geometry-grouped split labels | `82df1c16d0b7bff9eca621f20da272a7` |
| `mc-clusters.csv` | metal-chemistry-grouped split labels | `01cfac0b7ef5baf8532e70a599a55b9c` |
| `func-clusters.csv` | functional-group-grouped split labels | `b122413fe5015b9735a3e3d4f9856344` |
| `flig-clusters.csv` | ligand/linker-grouped split labels | `b58a0b8b608a76c9ba87457d18c9f01f` |
| `all_topology_lists.csv` | topology information used for the topology-grouped split when available | `14da9901247d08e450fb31d84c6ac912` |
| `overall_process.csv` | optional process-level contextual metrics | `04b17cf2fc3431b05a7882bdff2abc3a` |

`RDFs.csv` and other process/adsorption files are recognized by the historical source but are not required to reproduce the supplied principal `geometry_only` and `geometry_plus_racs` benchmark representation. The preserved result tables audited for the manuscript contain no evaluated RDF feature contribution.

The structure archive `ARCMOF_20241004.tar.gz` belongs to ARC-MOF v6, but the main tabular model-fitting benchmark does not require the CIF archive merely to reconstruct the four target/descriptor tables above.

## Target mapping implemented in the preserved source

The historical pipeline filters long-format ARC-MOF adsorption files at these conditions:

| Project target | Upstream file | Temperature | Pressure | Value |
|---|---|---:|---:|---|
| `CO2_0p015bar_298K_mmolg` | `post_comb_vsa-CO2.csv` | 298 K | 0.015 bar | `mmol/g` |
| `CO2_0p150bar_298K_mmolg` | `post_comb_vsa-CO2.csv` | 298 K | 0.150 bar | `mmol/g` |
| `CH4_5p8bar_298K_mmolg` | `methane.csv` | 298 K | 5.8 bar | `mmol/g` |
| `CH4_65bar_298K_mmolg` | `methane.csv` | 298 K | 65 bar | `mmol/g` |

These are **gravimetric uptake** targets.

## Local layout expected by the historical source

The source resolver accepts several historical aliases. The preferred organized layout is documented in [`modeling/INPUT_PATH_ALIASES.csv`](modeling/INPUT_PATH_ALIASES.csv). A minimal practical layout is:

```text
<DATA_ROOT>/
  data_raw/
    arc_mof/
      adsorption/
        post_comb_vsa_co2.csv
        methane.csv
      descriptors/
        geometric_properties.csv
        racs.csv
      clusters/
        geo_clusters.csv
        mc_clusters.csv
        func_clusters.csv
        flig_clusters.csv
      topology/
        all_topology_lists.csv
      process/
        overall_process.csv
```

The resolver also accepts the original ARC-MOF filenames directly. Use an explicit `--data-root` for intentional reruns rather than relying on machine-specific historical fallbacks.

## Third-party data and redistribution

ARC-MOF is an upstream third-party research dataset. This repository does not grant new rights over ARC-MOF and does not redistribute the raw ARC-MOF source files. Obtain the dataset from Zenodo, cite the ARC-MOF paper and dataset, and follow the terms attached to the upstream record.

The MIT license in this repository applies to repository-authored software/documentation as described by that license; it should not be interpreted as relicensing third-party ARC-MOF inputs.

## Optional historical external overlays

The preserved pipeline also contains optional CoRE-MOF/MOSAEC overlay stages. Their exact historical input versions and external-column mapping are not fully certified by the currently released provenance. They are therefore not substituted or reconstructed in this final release step. See [`modeling/SCIENTIFIC_AND_PROVENANCE_NOTES.md`](modeling/SCIENTIFIC_AND_PROVENANCE_NOTES.md) and [`figures/S3_PROVENANCE_STATUS.md`](figures/S3_PROVENANCE_STATUS.md).
