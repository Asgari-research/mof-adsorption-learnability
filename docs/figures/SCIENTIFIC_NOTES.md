# Scientific notes for figure interpretation

These notes are release guardrails for the saved figure sources. They are not a substitute for the manuscript Methods.

1. `B=10-1000` denotes model-fitting budget, not total labelled-information cost.
2. Figure 2 shading is the 10th-90th percentile spread over 30 configuration means (five designated model families x six split strategies); each configuration mean summarizes five seeds.
3. For equal-size top-k sets, precision equals recall in the supplied aggregate table. At a 5% selection fraction, enrichment is 20 x recall.
4. The split-conformal implementation uses one `q` per run, so the theoretical interval width is constant within a run and lower-bound ranking preserves point-prediction order apart from ties.
5. The historical retention analysis used 30,000 compacted prediction rows per target, can contain repeated MOF identities, and is not a retained fraction of the full library.
6. Figure 4 panels do not all summarize the same population; do not combine them as though they did.
7. The displayed top-25 candidate identities and the top-250 aggregate shortlist summaries are different cohorts.
8. Support counts are repeated appearances/opportunities in stored outputs, not independent probabilities.
9. `D_i` is described using the conventional largest-cavity-diameter meaning. Do not relabel the stored external-distance calculation as corrected evidence.
10. Figure S3 external-domain overlap remains provenance-limited until the historical external column mapping is verified.
11. Figure S1 is reconstructed from saved histogram bins, not individual uptake observations.
12. The plotting workflow must not be used to refit models, select new candidates, rebuild uncertainty estimates, recompute external neighbors, or run simulations.
