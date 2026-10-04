# Evidence ageing in LLM-based software engineering agents: analysis scripts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23097803.svg)](https://doi.org/10.5281/zenodo.23097803)

## Description

Analysis scripts and summary results for the article *Evidence Ageing in LLM-Based Software Engineering
Agents: An Empirical Characterisation and the Limits of Uniform Probing* (R. Saosing and C. Tritham).

The study asks which evaluation results of a software engineering agent still hold after one dimension of
the agent (its model, its harness or a configuration setting) changes. It compares pairs of public
SWE-bench Verified submissions that differ in one declared dimension with reruns of unchanged
configurations, evaluates history-based selection of tasks to re-run without look-ahead, and confirms an
analytic bound on uniformly sampled probes. The scripts recompute every empirical table and figure of the
article.

## Dataset information

- **Source.** Public per-task outcomes of SWE-bench submissions in
  [SWE-bench/experiments](https://github.com/SWE-bench/experiments), splits `evaluation/verified`
  (500 tasks) and `evaluation/lite` (300 tasks; 93 shared with Verified).
- **Version used.** Archive commit `40f164d5b8f1d249bf95a6df8b74b577fd8e519d` (3 September 2026).
- **Unit.** One submission × one task, with a binary *resolved* outcome.
- **Not redistributed.** The archive declares no licence, so its files are not included here. The scripts
  download them on first use into `results/external/` (git-ignored).
- **Included here.** `results/*.json`: per-pair aggregates (counts, rates, test statistics) and per-strategy
  recall summaries. They contain no per-task archive data.

## Code information

| Path | Purpose | Article items |
|---|---|---|
| `experiments/swebench_pairs.py` | Builds model-, harness- and configuration-change pairs and the noise floor; regression rates, Wilson intervals, exact McNemar tests with Holm correction, repository homogeneity | Tables 5–6, Figure 2 |
| `experiments/v6_selection.py` | Six selection strategies evaluated without look-ahead (Algorithm 2), rerun control, paired Wilcoxon tests | Table 7, Figure 3 |
| `experiments/prop1_real.py` | Proposition 1 on real pairs: uniform versus ranked probes outside a targeted set | Table 8 |
| `experiments/robustness.py` | Effect sizes (McNemar odds ratios, Cliff's delta) and a cluster bootstrap over pairs sharing a submission; reads only `results/*.json` | RQ1 and RQ3 robustness |
| `experiments/rq1_sensitivity.py` | Noise floor on the 93 shared tasks, classification audit, data-source and harness-family splits; needs the cached archive | RQ1 and RQ3 sensitivity |
| `figures/fig_pairs.py`, `figures/fig_v6.py` | Figures from the result files; `figures/style.py` holds the shared style | Figures 2–3 |
| `results/*.json` | Outputs of the three experiments, as reported in the article | — |

## Requirements

- Python 3.11 or later. The experiments use the Python standard library only.
- `matplotlib` (tested with 3.11.1) for the figures only.
- The GitHub CLI `gh`, authenticated, which `swebench_pairs.py` uses to list the archive.
- Internet access on the first run, to download the archive files.

## Usage

```bash
python experiments/swebench_pairs.py   # downloads the archive files once, then writes results/swebench_pairs.json
python experiments/v6_selection.py     # fixed seeds; byte-identical output on repeated runs
python experiments/prop1_real.py
python experiments/robustness.py      # effect sizes and cluster bootstrap, from results/*.json only
python experiments/rq1_sensitivity.py # noise-floor sensitivity, audit, source and family splits
python figures/fig_pairs.py            # -> figures/figure2_pairs.pdf
python figures/fig_v6.py               # -> figures/figure3_selection.pdf
```

Every archive listing and download is pinned to the commit above (`REF` in `experiments/swebench_pairs.py`), so a
run reproduces the committed `results/*.json` byte for byte.

## Methodology

1. **Outcomes.** For each submission, per-task outcomes are read from `per_instance_details.json` where
   present, otherwise from the `resolved` list of `results/results.json`, complemented by the split's task
   list (tasks not listed as resolved count as unresolved).
2. **Dates.** A submission's date is the `YYYYMMDD` prefix of its name; it orders submissions for the
   no-look-ahead rule.
3. **Pairs.** Model change: every pair of models under the same mini-SWE-agent version. Harness change: the
   same model under two harnesses. Configuration change: the same model and harness with a different
   reasoning-effort setting. Noise floor: one submission name present in both Lite and Verified, compared on
   the 93 shared tasks; pairs with identical outcomes are treated as re-grades, the others as reruns.
4. **Measures.** Regressions (solved before, not after), improvements, regression rate relative to the
   previously solved tasks, and an exact McNemar test per pair, Holm-corrected within each group.
5. **Selection.** Each strategy scores the previously solved tasks using only submissions dated before the
   pair, selects a share of them, and is scored by its recall of the pair's regressions; 20 seeds for
   tie-breaking and sampling.

No other preprocessing, filtering or imputation is applied.

## Computing infrastructure

The reported results were produced on Windows 11 (build 26200), Intel Core i7-7700HQ (4 cores, 8 threads),
16 GB RAM, Python 3.11.15. No GPU and no model inference are required: the scripts analyse archived outcomes
only.

## Citation

If you use these scripts, please cite the article and this software (see `CITATION.cff`;
doi: [10.5281/zenodo.23097803](https://doi.org/10.5281/zenodo.23097803)). Please also cite SWE-bench
(Jimenez et al., ICLR 2024) and the SWE-bench/experiments archive.

## Licence and contributions

MIT (see `LICENSE`). The licence covers the scripts and summary results in this repository only, not the
SWE-bench archive. Issues and pull requests are welcome; please describe how a change affects the
reported numbers.
