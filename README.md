# Evidence ageing in LLM-based software engineering agents: analysis scripts

Analysis scripts and summary results for the article *Evidence Ageing in LLM-Based Software Engineering
Agents: An Empirical Characterisation and the Limits of Uniform Probing* (R. Saosing and C. Tritham).

All results are computed from the public per-task outcomes of SWE-bench submissions in
[SWE-bench/experiments](https://github.com/SWE-bench/experiments). The analysis used the archive at commit
`40f164d5b8f1d249bf95a6df8b74b577fd8e519d` (3 September 2026). The archive declares no licence, so its files
are **not redistributed here**: the scripts download them on first use into `results/external/` (git-ignored).

## Contents

| Path | Recomputes |
|---|---|
| `experiments/swebench_pairs.py` | Change pairs, noise floor, McNemar/Holm tests, repository locality (Tables 4–5, Figure 4) → `results/swebench_pairs.json` |
| `experiments/v6_selection.py` | History-based selection without look-ahead, rerun control (Table 6, Figure 5) → `results/v6_selection.json` |
| `experiments/prop1_real.py` | Proposition 1 on real pairs (Table 7) → `results/prop1_real.json` |
| `figures/fig_pairs.py`, `figures/fig_v6.py` | Figures 4 and 5 from the result files (`figures/style.py` holds the shared style) |
| `results/*.json` | Summary results reported in the article (aggregates per pair; no per-task archive data) |

## Running

Python 3.11 or later. The experiments use the standard library only; the figures need `matplotlib`.
`swebench_pairs.py` lists the archive through the GitHub CLI (`gh api`), so `gh` must be installed and authenticated.

```bash
python experiments/swebench_pairs.py   # downloads the archive files once, then writes results/swebench_pairs.json
python experiments/v6_selection.py     # fixed seeds; byte-identical output on repeated runs
python experiments/prop1_real.py
python figures/fig_pairs.py
python figures/fig_v6.py
```

The archive is fetched from its default branch. To reproduce the article exactly, check out the commit above
locally and point the cache at it, or compare your `results/*.json` with the committed ones.

## Licence

MIT (see `LICENSE`). The licence covers the scripts and summary results in this repository only, not the
SWE-bench archive.
