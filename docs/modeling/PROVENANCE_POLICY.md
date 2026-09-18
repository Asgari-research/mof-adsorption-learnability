# Source preservation and authorship/provenance policy

The historical Python sources in `modeling_pipeline/` are preserved byte-for-byte from the supplied project package. Their hashes are recorded in `SOURCE_MANIFEST.csv` and checked by the repository verifier.

Release preparation deliberately does **not** rewrite source history to make the code appear to have a different origin. Historical comments, titles, internal version layers, and machine-specific fallback paths remain in the preserved source even when public-facing documentation has since been clarified.

Source-text scans and stylistic inspection cannot establish authorship. This repository therefore makes no authorship claim based on such scans. Any disclosure required by a journal, institution, funder, collaborator agreement, or research-integrity policy remains a separate obligation.

No Git history is rewritten by the release helpers. The original historical source SHA-256 values remain reviewable in the repository.
