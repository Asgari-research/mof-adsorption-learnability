# Static architecture and override audit

This document describes the supplied historical source **without executing or importing it**. The release preparation did not fit models or regenerate numerical results.

## Historical pipeline

- Source: `modeling_pipeline/src/fewshot_mof_risk_controlled_external_pipeline_v4_8_supplied.py`
- SHA-256: `1553d2b27118b400b53b17d436e76b33f10629445b62c311d70e2f9e907b3c3d`
- Size: 11,532 lines
- Top-level definitions: 319
- Unique top-level definition names: 262
- Repeated top-level names: 13
- `main()` is defined at line 7138; the `__main__` guard occurs at line 11523.

Python resolves the global function names used inside `main()` when `main()` executes. Because several names are redefined later in the file, the final definitions listed below are the active globals when the end-of-file `main()` call runs. This is why visually deleting earlier or later override blocks would not be a cosmetic edit.

| Name | Definitions | Lines | Active line |
|---|---:|---|---:|
| `collect_candidate_tiers` | 2 | 2798, 4676 | 4676 |
| `build_consensus_candidate_table` | 2 | 2908, 4798 | 4798 |
| `build_external_realism_table` | 2 | 4032, 4909 | 4909 |
| `chemical_failure_anatomy` | 2 | 3009, 5006 | 5006 |
| `save_manuscript_si_tables` | 3 | 3439, 5582, 6449 | 6449 |
| `plot_external_si_figures` | 2 | 4303, 6978 | 6978 |
| `plot_figure_1_framework` | 7 | 3114, 6457, 8178, 8956, 9883, 11052, 11494 | 11494 |
| `plot_figure_2_performance` | 8 | 3168, 5640, 6496, 8183, 8961, 9888, 11057, 11499 | 11499 |
| `plot_figure_3_calibration` | 8 | 3205, 5712, 6608, 8187, 8965, 9892, 11061, 11503 | 11503 |
| `plot_figure_4_failure_anatomy` | 9 | 3279, 5075, 5771, 6727, 8191, 8969, 9896, 11065, 11507 | 11507 |
| `plot_figure_5_external_realism` | 9 | 4194, 5167, 5844, 6861, 8196, 8974, 9901, 11070, 11511 | 11511 |
| `plot_si_figures` | 8 | 3352, 5917, 7029, 8201, 8979, 9906, 11075, 11515 | 11515 |
| `write_run_report` | 8 | 3501, 5973, 7080, 8206, 8984, 9911, 11080, 11519 | 11519 |

## Import-time behavior

The historical module is **not side-effect-free on import**. Before the `__main__` guard it configures warnings/loggers and Matplotlib, evaluates environment-dependent configuration, defines data-root fallbacks, and creates multiple output directories with `mkdir(...)`. Static inspection should therefore parse the file rather than importing it merely to inspect configuration.

## Legacy publication renderer

- Source: `modeling_pipeline/legacy_publication_renderer/regenerate_publication_figures_and_tables_v4_9.py`
- SHA-256: `c5f1fdfe8f7d758834124996bffeead5e46cbf4eccb33650cdb2abfc079a59ae`
- Size: 448 lines
- This script assumes the older `01_main_text/` and `02_supporting_information/` directory layout and writes a claims CSV in addition to figures.
- It is retained for historical provenance only. The current publication figure workflow is `figure_generation/`, added in Phase 1.

## What this audit does not establish

- No numerical experiment was rerun.
- Static parsing does not certify historical package versions, raw-input versions, or cache provenance.
- The override map describes Python name resolution in this supplied source; it does not prove that every historical result was generated from every final override layer shown here.
