# Workflow and reproducibility map

## Purpose

This repository contains two different computational layers that should not be confused:

1. a **historical modelling pipeline** that produced the saved benchmark outputs; and
2. a **current publication-figure workflow** that reads saved publication-level tables and redraws the finalized figures without refitting models.

The repository intentionally preserves that distinction.

## Data-to-figure chain

```text
External/raw MOF data
  ARC-MOF v6 adsorption + geometry
  exact record: 10.5281/zenodo.13891643
  RAC / grouping / optional process data
  optional external CoRE/MOSAEC resources
          |
          |  raw snapshots are not fully redistributed in this repository
          v
Historical modelling pipeline
  modeling_pipeline/src/...
          |
          |  fitting, grouped/random splits, metrics,
          |  uncertainty, candidate collection, saved tables
          v
Historical/saved numerical outputs
  publication-facing derivative tables are deposited in publication_tables/
          |
          +--> Complete manuscript/SI publication tables
          |    publication_tables/main/
          |    publication_tables/si/
          |
          v
Saved figure source tables
  figure_generation/data/
          |
          |  no model fitting
          v
Current plotting implementation
  figure_generation/code/
          |
          v
Scratch regenerated figures
  figure_generation/outputs/
          |
          |  review + explicit replacement only
          v
Publication-authority PDFs
  figures/main/
  figures/si/
```

## Upstream ARC-MOF source

The historical benchmark is tied to ARC-MOF **v6** (2024-10-04), exact record DOI `10.5281/zenodo.13891643`. The persistent ARC-MOF concept DOI is `10.5281/zenodo.6908727`, and the peer-reviewed database paper DOI is `10.1021/acs.chemmater.2c02485`. Raw ARC-MOF files are not redistributed here. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for exact input filenames, target mapping, and upstream checksums.

## What can be checked from a fresh clone

A fresh clone can:

- verify the exact historical source hashes;
- inspect configuration, CLI, path and override maps without importing the historical module;
- inspect the complete machine-readable publication tables under `publication_tables/`;
- inspect the saved panel-level source tables used by the plotting workflow;
- regenerate the current manuscript figures if compatible plotting dependencies and Arial are available;
- verify the finalized figure hashes;
- run the read-only repository integrity check.

Use:

```bash
python tools/verify_repository.py
```

## What a fresh clone cannot establish by itself

The repository does not currently establish exact end-to-end reproduction of the historical model-fitting run because the supplied release materials do not contain all of the following:

- an exact historical environment lock;
- a complete release-ready raw database snapshot;
- complete saved split-index archives;
- complete cache lineage/configuration fingerprints for every historical stage;
- verified historical external-column mapping records.

These are provenance limits, not instructions to regenerate or guess missing artifacts.

## Historical reruns

An intentional historical rerun is a separate scientific task from figure regeneration.

Before rerunning:

1. read `modeling_pipeline/README.md`;
2. read `docs/modeling/EXECUTION_SEMANTICS.md`;
3. provide an explicit `--data-root`;
4. record the complete configuration and package environment;
5. use a clean output location or deliberately clear/force stale cache state;
6. do not interpret a successful run as proof that it reproduces the exact historical execution unless inputs/configuration/environment have also been matched.

## Current figure regeneration

The current figure workflow is intentionally narrower. It consumes saved tables and does not:

- refit models;
- reconstruct the raw adsorption database;
- recompute candidate consensus;
- recalculate external-domain neighbors.

Its outputs are written to a scratch directory. Files under `figures/` are replaced only through an explicit reviewed change with updated hashes.

## Provenance hierarchy

When two files appear to represent the same concept, use this hierarchy:

- `figures/` — authority for finalized manuscript figure PDFs;
- `docs/figures/FINAL_FIGURE_MANIFEST.csv` — authority for their hashes;
- `publication_tables/` — complete machine-readable manuscript and SI publication tables;
- `figure_generation/` — current plotting implementation/panel-level source tables;
- `modeling_pipeline/src/` — preserved historical modelling implementation;
- `modeling_pipeline/legacy_publication_renderer/` — historical renderer only;
- `docs/modeling/` — static maps and limitations, not executable scientific logic.
