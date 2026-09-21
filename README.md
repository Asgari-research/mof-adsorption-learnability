# MOF adsorption learnability

Code and publication assets for a benchmark of adsorption-tail prediction in metal-organic frameworks under restricted model-fitting budgets, multiple descriptor families, grouped transfer tests, and empirical uncertainty calibration.

## Repository map

The repository separates historical modelling provenance from the current publication-figure workflow:

```text
modeling_pipeline/
  README.md
  src/                          Byte-identical historical modelling source
  legacy_publication_renderer/ Historical renderer retained for provenance
  requirements-historical.txt

figure_generation/
  code/                         Current plotting implementation
  data/                         Saved source tables read by the plotting code
  run_figure.py
  run_all.py
  verify_outputs.py

publication_tables/
  main/                         Complete manuscript-level CSV tables
  si/                           Complete machine-readable SI Tables S1-S8
  README.md                     Table map and interpretation notes
  MANIFEST_SHA256.csv           File hashes, sizes and row/column counts

figures/
  main/                         Final main-text PDFs (Figures 1-5)
  si/                           Final SI PDFs (Figures S1-S3)

docs/
  WORKFLOW_AND_REPRODUCIBILITY.md
  modeling/                     Static code/provenance maps and execution notes
  figures/                      Figure manifests and scientific notes

tools/
  verify_repository.py          Read-only repository integrity/static check

AUTHORS.md
LICENSE
```

## Start here

For the relationship between inputs, historical modelling outputs, saved publication tables, current plotting code, and finalized figures, read:

`docs/WORKFLOW_AND_REPRODUCIBILITY.md`

For the historical modelling implementation, read:

`modeling_pipeline/README.md`

For current figure regeneration, read:

`figure_generation/README.md`

## Publication tables

The complete machine-readable publication tables are deposited under `publication_tables/`. The `main/` subdirectory contains manuscript-level summary/candidate tables, while `si/` contains the complete CSV counterparts of Supplementary Tables S1-S8. Large tables such as S1 and S3 are intentionally summarized in the typeset SI rather than printed row-by-row. `publication_tables/MANIFEST_SHA256.csv` provides row counts, dimensions and SHA-256 hashes for integrity checks.

These publication-level tables are derived outputs and do not replace the upstream ARC-MOF source data.

## Primary data source and citation

The benchmark is derived from the **ARC-MOF** database. For strict historical provenance, use the ARC-MOF **v6** Zenodo snapshot updated on 2024-10-04:

- exact v6 record: [10.5281/zenodo.13891643](https://doi.org/10.5281/zenodo.13891643)
- persistent ARC-MOF concept DOI (all versions): [10.5281/zenodo.6908727](https://doi.org/10.5281/zenodo.6908727)
- peer-reviewed database paper: Burner *et al.*, *Chemistry of Materials* **35** (2023) 900-916, [10.1021/acs.chemmater.2c02485](https://doi.org/10.1021/acs.chemmater.2c02485)

The historical source explicitly recognizes the `ARCMOF_20241004.tar.gz` archive name and the tabular files available in that v6 record. Raw ARC-MOF source files are **not** redistributed by this repository. Download third-party inputs from the upstream record and retain their original provenance and terms.

See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for the exact benchmark input files, target mapping, upstream checksums, and reproduction notes.

## Modelling pipeline

`modeling_pipeline/src/fewshot_mof_risk_controlled_external_pipeline_v4_8_supplied.py` is preserved byte-for-byte from the supplied historical analysis package. Repository cleanup does not refit models or modify its scientific logic. Its source SHA-256 and static architecture maps are recorded under `docs/modeling/`.

The tested budgets are **model-fitting budgets**, not total label-acquisition costs. Calibration/test reservoirs and retrospective target-stratified fitting-set selection are separate parts of the workflow.

The historical source contains repeated publication-layer overrides, import-time filesystem side effects, existence-based restart caches, and some budget/resource-profile settings that can change scientific outputs. These execution semantics are documented in:

`docs/modeling/EXECUTION_SEMANTICS.md`

## Figure integrity

The PDFs in `figures/` are the publication-authority files. Their SHA-256 hashes are recorded in `docs/figures/FINAL_FIGURE_MANIFEST.csv`.

Figure regeneration writes to `figure_generation/outputs/` and must not silently replace the publication-authority PDFs. The current plotting package uses saved panel/table values only; it does not refit models, recompute shortlist membership, reconstruct raw adsorption observations, or recalculate external-domain neighbors.

## Regenerating current figures

The plotting workflow uses saved publication-level source tables only and writes review outputs under `figure_generation/outputs/`. It does not replace the locked PDFs under `figures/`.

Windows / Anaconda:

```text
conda activate mofenv
cd figure_generation
python check_environment.py
python run_all.py
python verify_outputs.py
```

WSL2 users can run `bash setup_wsl.sh` first to expose Windows Arial, then use the same Python commands. Figure 1 is a locked static asset; the current renderer covers Figures 2-5 and S1-S3.

## Repository verification

A read-only repository check is included:

```bash
python tools/verify_repository.py
```

It verifies the locked historical source hashes, finalized figure hashes, Python syntax, required release files, and basic repository hygiene. It does **not** run models or certify exact end-to-end numerical reproduction.

## Reproduction scope

The repository includes the historical modelling source and saved publication-level figure inputs, but the supplied project package did not establish a complete historical environment lock, a release-ready raw-database snapshot, complete split-index archives, or every historical cache manifest.

`modeling_pipeline/requirements-historical.txt` is therefore an unpinned dependency list, not a claim of exact environment reconstruction.

Known scientific/provenance limits are documented in:

- `docs/modeling/SCIENTIFIC_AND_PROVENANCE_NOTES.md`
- `docs/modeling/EXECUTION_SEMANTICS.md`
- `docs/modeling/REPRODUCTION_STATUS.md`
- `docs/figures/SCIENTIFIC_NOTES.md`


## License

See `LICENSE`.
