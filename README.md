# MOF adsorption learnability

Code and publication assets for a benchmark of adsorption-tail prediction in metal-organic frameworks under restricted model-fitting budgets, multiple descriptor families, grouped transfer tests, and empirical uncertainty calibration.

## Repository structure

This repository separates the historical modelling implementation from the current publication-figure workflow:

```text
modeling_pipeline/
  README.md
  src/                         Byte-identical historical modelling pipeline
  legacy_publication_renderer/ Historical renderer retained for provenance
  requirements-historical.txt
figure_generation/
  code/                        Current plotting implementation
  data/                        Saved source tables read by the plotting code
  run_figure.py
  run_all.py
  verify_outputs.py
figures/
  main/                        Final main-text PDFs (Figures 1-5)
  si/                          Final SI PDFs (Figures S1-S3)
docs/
  modeling/                    Static code/provenance audit maps
  figures/                     Figure manifests and scientific notes
AUTHORS.md
LICENSE
```

## Modelling pipeline

`modeling_pipeline/src/fewshot_mof_risk_controlled_external_pipeline_v4_8_supplied.py` is preserved byte-for-byte from the supplied historical analysis package. The Phase 2 repository cleanup does not refit models or modify its scientific logic. Its source SHA-256 and static architecture maps are recorded under `docs/modeling/`.

The historical file is a monolithic restart-safe pipeline with repeated publication-layer overrides. It also has import-time filesystem side effects, so static inspection should use the supplied maps rather than importing the module. See `modeling_pipeline/README.md` before any intentional rerun.

The tested fitting budgets should not be interpreted as total label acquisition cost: calibration/test reservoirs and retrospective target-stratified fitting-set selection are separate parts of the workflow.

## Figure integrity

The PDFs in `figures/` are the publication-authority files. Their SHA-256 hashes are recorded in `docs/figures/FINAL_FIGURE_MANIFEST.csv`. Figure regeneration writes to `figure_generation/outputs/` and must not silently replace those files.

The current plotting package uses saved panel/table values only. It does not refit machine-learning models, recompute shortlist membership, reconstruct raw adsorption observations, or recalculate external-domain neighbors.

## Regenerating current figures in Ubuntu

The plotting workflow uses the Python environment already active in the shell. It does not create or activate a new Conda environment or venv.

```bash
cd figure_generation
bash setup_wsl.sh
bash RUN_ALL_WSL.sh
```

Final exports require Arial; font files are not distributed with this repository. Regenerated outputs are review artifacts. Replacing any file under `figures/` should be an explicit, separately reviewed change accompanied by an updated manifest.

## Reproduction scope

The repository includes the historical source and saved publication-level figure inputs, but the supplied project package did not establish a complete historical environment lock, a release-ready raw-database snapshot, or every historical split/cache manifest. `modeling_pipeline/requirements-historical.txt` is therefore an unpinned dependency list rather than a claim of exact environment reconstruction.

Known scientific/provenance limitations are documented in `docs/modeling/SCIENTIFIC_AND_PROVENANCE_NOTES.md` and `docs/figures/SCIENTIFIC_NOTES.md`.

Figure S3 remains in the current SI asset set, but its historical external-geometry mapping is not independently certified from the available provenance. See `docs/figures/S3_PROVENANCE_STATUS.md`.

## License

See `LICENSE`.
