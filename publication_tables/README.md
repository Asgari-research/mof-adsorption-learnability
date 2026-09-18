# Publication tables

This directory contains the complete machine-readable tables associated with the manuscript **“Adsorption conditions shape the information needed for metal–organic framework screening.”**

The tables are publication-level derivative data. They are not a redistribution of the raw ARC-MOF database. Raw ARC-MOF inputs must be obtained from the upstream ARC-MOF v6 Zenodo record and cited with the database paper.

## Directory layout

```text
publication_tables/
  README.md
  MANIFEST_SHA256.csv
  main/
    Table_1_dataset_and_targets.csv
    Table_1_performance_summary_V2.csv
    Table_2_main_claims_summary.csv
    Table_3_top5_consensus_candidates_per_target.csv
  si/
    Table_S1_all_aggregated_metrics_with_CI.csv
    Table_S2_feature_set_integrity_RDF_audit.csv
    Table_S3_model_stability_quality_gate.csv
    Table_S4_prediction_range_sanity_by_model.csv
    Table_S5_precision_yield_frontier.csv
    Table_S6_consensus_true_enrichment.csv
    Table_S7_top25_consensus_candidates_per_target.csv
    Table_S8_external_domain_overlap_summary.csv
```

## Main-manuscript tables

| File | Rows | Role |
|---|---:|---|
| `main/Table_1_dataset_and_targets.csv` | 4 | Target-level dataset statistics for the four adsorption conditions. |
| `main/Table_1_performance_summary_V2.csv` | 4 | Values used in the manuscript ranking-performance summary at `B = 1000`. |
| `main/Table_2_main_claims_summary.csv` | 4 | Target-level summary of ranking, representation-gain, topology-penalty and candidate-set quantities. |
| `main/Table_3_top5_consensus_candidates_per_target.csv` | 20 | Five highest-scoring displayed consensus candidates per target, with prediction/support information. |

## Supplementary tables

The SI typesets compact summaries where a full row-level table would be unreadable. The files below are the complete machine-readable counterparts.

| SI table | File | Rows | Complete content |
|---|---|---:|---|
| S1 | `si/Table_S1_all_aggregated_metrics_with_CI.csv` | 7,776 | Aggregated regression, ranking, calibration, interval-width, tier and runtime metrics with seed-bootstrap limits. |
| S2 | `si/Table_S2_feature_set_integrity_RDF_audit.csv` | 3 | Feature-set inventory and RDF-branch integrity audit. |
| S3 | `si/Table_S3_model_stability_quality_gate.csv` | 12,960 | Experiment-level model/split/budget/seed records and candidate-quality gate fields. |
| S4 | `si/Table_S4_prediction_range_sanity_by_model.csv` | 612 | Compact candidate-related prediction-range and sanity summaries. |
| S5 | `si/Table_S5_precision_yield_frontier.csv` | 24 | Retention-fraction / reference-tail precision frontier. |
| S6 | `si/Table_S6_consensus_true_enrichment.csv` | 4 | Target-level reference-uptake enrichment statistics for the 250-member consensus summaries. |
| S7 | `si/Table_S7_top25_consensus_candidates_per_target.csv` | 100 | Candidate-level records for the 25 displayed consensus structures per target, including support and pore descriptors. |
| S8 | `si/Table_S8_external_domain_overlap_summary.csv` | 4 | Archived external-domain annotation summary. Its geometry-mapping provenance limitation is documented in the SI and `docs/figures/S3_PROVENANCE_STATUS.md`. |

## Integrity

`MANIFEST_SHA256.csv` records the byte size, data-row count, column count and SHA-256 hash of every CSV in this directory. The publication-table files deposited here should remain byte-identical to the corresponding `tables/` files in the finalized Overleaf/manuscript package.

## ARC-MOF source and citation

The benchmark uses **ARC-MOF v6**, updated 4 October 2024:

- exact Zenodo record: https://doi.org/10.5281/zenodo.13891643
- persistent ARC-MOF concept DOI: https://doi.org/10.5281/zenodo.6908727
- database paper: J. Burner *et al.*, *Chemistry of Materials* **35** (2023) 900–916, https://doi.org/10.1021/acs.chemmater.2c02485

See `docs/DATA_SOURCES.md` for the exact upstream ARC-MOF files, target mapping and published checksums.
