# Reproduction status

## Available in this repository

- byte-identical historical modelling source;
- byte-identical historical publication renderer;
- the supplied unpinned dependency list;
- static configuration, CLI, import, path, and override maps;
- current publication figure-generation source and saved figure tables from Phase 1;
- locked finalized figure PDFs and their hashes.

## Not established by the supplied package

- the exact Python/package environment used for every historical run;
- a complete raw ARC-MOF / CoRE-MOF / MOSAEC input snapshot with release-ready redistribution rights;
- the exact realized topology-column choice for historical grouped splits;
- complete saved split indices for every run;
- complete candidate-tier collection manifests and cache lineage for every historical stage;
- the exact historical external-column mapping and its numerical impact.

`requirements-historical.txt` must therefore be treated as an import/dependency list, not an exact reproducibility lock file. Do not invent package pins after the fact.

The manuscript audit verified the supplied saved numerical tables directly. Phase 2 is a source-release/static-provenance task and performs no model fitting.
