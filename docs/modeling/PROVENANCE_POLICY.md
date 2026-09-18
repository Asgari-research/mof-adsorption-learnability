# Source preservation and authorship/provenance policy

The historical Python sources in `modeling_pipeline/` are preserved byte-for-byte from the supplied project package. Their hashes are recorded in `SOURCE_MANIFEST.csv` and checked by the Phase 2 static verifier.

The release preparation deliberately does **not** rewrite source history to make the code appear to have a different origin. Historical comments, titles, and internal version layers remain in the preserved source even when the manuscript wording has since changed. Public-facing documentation is kept separate from that historical implementation.

A literal text scan of the two supplied Python files found no named ChatGPT/OpenAI/Claude/Gemini/Copilot/LLM attribution markers. That observation is only a text-scan result; it is not evidence that can establish authorship, nor is this repository described as “AI-free.” Any disclosure required by a journal, institution, funder, or collaborator remains a separate authorship/research-integrity matter.

No Git history was rewritten in Phase 2. The original source SHA-256 values remain reviewable in the repository.
