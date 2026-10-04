"""Robustness of the reported effects: effect sizes and a cluster bootstrap over shared submissions.

    python experiments/robustness.py   -> results/robustness.json

Reads only the committed result files (results/swebench_pairs.json, results/v6_per_pair.json); needs no archive data.

1. Effect sizes for RQ1:
   - per pair, the McNemar odds ratio OR = (regressions + 0.5) / (improvements + 0.5), reported as the median
     log2(OR) per group with an interquartile range;
   - Cliff's delta between the regression rates of change pairs and of reruns (the effect size that goes with the
     one-sided Mann-Whitney test reported in the article).
2. Dependence between pairs for RQ3: pairs that share a submission are not independent. The mean recall gain
   over random selection is re-estimated with a cluster bootstrap that resamples whole clusters of pairs, a
   cluster being all pairs that share their earlier submission. 10,000 resamples, fixed seed.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
B_BOOT = 10_000
SEED = 20261004


def quartiles(xs):
    q = statistics.quantiles(xs, n=4)
    return {"median": statistics.median(xs), "q1": q[0], "q3": q[2]}


def cliffs_delta(xs, ys):
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    return (gt - lt) / (len(xs) * len(ys))


def magnitude(d):  # Romano et al. (2006) thresholds
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def main():
    pairs = json.loads((RES / "swebench_pairs.json").read_text(encoding="utf-8"))["pairs"]
    out = {"effect_sizes": {}, "cluster_bootstrap": {}}

    groups = {"model": "M", "harness": "H", "rerun": "N-rerun"}
    for name, cat in groups.items():
        g = [p for p in pairs if p["category"] == cat]
        lor = [math.log2((p["regressions"] + 0.5) / (p["improvements"] + 0.5)) for p in g]
        alor = [abs(x) for x in lor]
        out["effect_sizes"][name] = {"pairs": len(g), "log2_odds_ratio": quartiles(lor),
                                     "abs_log2_odds_ratio": quartiles(alor)}
    rr = [p["regression_rate"] for p in pairs if p["category"] == "N-rerun"]
    for name, cat in (("model", "M"), ("harness", "H")):
        xs = [p["regression_rate"] for p in pairs if p["category"] == cat]
        d = cliffs_delta(xs, rr)
        out["effect_sizes"][name]["cliffs_delta_rate_vs_rerun"] = {"delta": d, "magnitude": magnitude(d)}
        g = [p for p in pairs if p["category"] == cat]
        xa = [abs(math.log2((p["regressions"] + 0.5) / (p["improvements"] + 0.5))) for p in g]
        ra = [abs(math.log2((p["regressions"] + 0.5) / (p["improvements"] + 0.5))) for p in pairs if p["category"] == "N-rerun"]
        d2 = cliffs_delta(xa, ra)
        out["effect_sizes"][name]["cliffs_delta_direction_vs_rerun"] = {"delta": d2, "magnitude": magnitude(d2)}

    per = json.loads((RES / "v6_per_pair.json").read_text(encoding="utf-8"))
    clusters = {}
    for p in per:
        clusters.setdefault(p["before"], []).append(p)
    keys = sorted(clusters)
    rnd = random.Random(SEED)
    out["cluster_bootstrap"] = {"pairs": len(per), "clusters": len(keys), "resamples": B_BOOT, "results": {}}
    budgets = list(per[0]["recall"])
    strategies = [s for s in per[0]["recall"][budgets[0]] if s != "random"]
    draws = [[rnd.choice(keys) for _ in keys] for _ in range(B_BOOT)]
    for b in budgets:
        res = {}
        for st in strategies:
            gain = {k: [p["recall"][b][st] - p["recall"][b]["random"] for p in clusters[k]] for k in keys}
            point = statistics.fmean(x for k in keys for x in gain[k])
            boots = []
            for d in draws:
                xs = [x for k in d for x in gain[k]]
                boots.append(statistics.fmean(xs))
            boots.sort()
            lo, hi = boots[int(0.025 * B_BOOT)], boots[int(0.975 * B_BOOT) - 1]
            res[st] = {"mean_gain_over_random": point, "cluster_ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0}
        out["cluster_bootstrap"]["results"][b] = res

    (RES / "robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    es = out["effect_sizes"]
    for n in groups:
        print(n, es[n]["pairs"], "median log2 OR", round(es[n]["log2_odds_ratio"]["median"], 3),
              "median |log2 OR|", round(es[n]["abs_log2_odds_ratio"]["median"], 3))
    for n in ("model", "harness"):
        print(n, "Cliff rate vs rerun", es[n]["cliffs_delta_rate_vs_rerun"], "direction", es[n]["cliffs_delta_direction_vs_rerun"])
    print("clusters", len(keys))
    for b in budgets:
        print(b, {st: (round(v["mean_gain_over_random"], 3), [round(x, 3) for x in v["cluster_ci95"]]) for st, v in out["cluster_bootstrap"]["results"][b].items()})


if __name__ == "__main__":
    main()
