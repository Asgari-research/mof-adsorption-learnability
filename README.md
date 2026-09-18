# MOF adsorption learnability

Code and publication assets for a benchmark of adsorption-tail prediction in metal-organic frameworks under restricted model-fitting budgets, multiple descriptor families, grouped transfer tests, and empirical uncertainty calibration.

## Repository status

This release is being assembled in reviewable stages. The current repository layer contains the finalized manuscript figure assets, the current figure-generation source, and the saved tables required by that plotting workflow. The historical full modelling pipeline is intentionally not altered or added by this figure-release step; it is handled separately because its release cleanup must preserve numerical behavior and several historical provenance questions remain documented.

## Contents

```text
figures/
  main/            Final main-text PDFs (Figures 1-5)
  si/              Final SI PDFs (Figures S1-S3)
figure_generation/
  code/            Current plotting implementation
  data/            Saved source tables read by the plotting code
  run_figure.py    Regenerate one figure into a scratch output directory
  run_all.py       Regenerate the current figure set into scratch outputs
  verify_outputs.py
  setup_wsl.sh
  RUN_ALL_WSL.sh
docs/figures/
  README.md
  FINAL_FIGURE_MANIFEST.csv
  FINAL_FIGURE_SOURCE_MANIFEST.csv
  CURRENT_CODE_MANIFEST.csv
  FIGURE_PANEL_MAP.csv
  SCIENTIFIC_NOTES.md
  S3_PROVENANCE_STATUS.md
AUTHORS.md
LICENSE
```

## Figure integrity

The PDFs in `figures/` are the publication-authority files. Their SHA-256 hashes are recorded in `docs/figures/FINAL_FIGURE_MANIFEST.csv`. Figure regeneration writes to `figure_generation/outputs/` and must not silently replace the publication assets.

## Regenerating figures in WSL2

The plotting workflow requires Python 3.11+ and the packages in `figure_generation/requirements.txt`. Final exports require Arial; font files are not distributed with this repository.

```bash
cd figure_generation
bash setup_wsl.sh
bash RUN_ALL_WSL.sh
```

The setup script uses the Python environment already active in the shell. It does not create or activate a new Conda environment or venv; missing dependencies are reported explicitly.

Regenerated outputs are review artifacts. Replacing any file under `figures/` should be an explicit, separately reviewed change accompanied by an updated manifest.

## Scope of the saved plotting data

The plotting package uses saved panel/table values only. It does not refit machine-learning models, recompute shortlist membership, reconstruct raw adsorption observations, or recalculate external-domain neighbors. Scientific interpretation limits are summarized in `docs/figures/SCIENTIFIC_NOTES.md`.

Figure S3 is retained in the current SI asset set, but its historical external-geometry mapping is not independently certified from the currently available provenance. See `docs/figures/S3_PROVENANCE_STATUS.md`.

## License

See `LICENSE`.
