# evidence-spectral

EvidenceSpectral applies random matrix theory to the 5x5 correlation matrix of
trust-component scores across Cochrane meta-analyses.

`spectral_engine.py` computes the Pearson correlation matrix of the five
component scores (audit, consistency, robustness, stability, power), then runs:

- eigendecomposition (descending eigenvalues / eigenvectors),
- Marchenko-Pastur bulk-edge bounds and a count of signal eigenvalues,
- a Tracy-Widom (TW1) statistic for the largest eigenvalue with a normal-tail
  p-value approximation,
- participation ratios per eigenvector,
- per-domain correlation matrices (domains with >= 30 observations).

`build_dashboard.py` renders an offline HTML dashboard (`dashboard.html` /
`index.html`) from the pipeline results, and `build_e156.py` builds the E156
submission page.

## Tests

```
python -m pytest -q
```

The pipeline-integration tests skip automatically when the source score/group
CSVs are not present on disk.
