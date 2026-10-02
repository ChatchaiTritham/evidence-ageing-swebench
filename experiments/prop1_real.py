"""Proposition 1 on real data: uniform probes outside a targeted set cannot beat random selection; ranked probes can.

For each real SWE-bench Verified change pair (no look-ahead, as in v6_selection.py) and total budget B:
  T        targeted set = the |T| candidates with the highest regression history (predictive score)
  uniform  B - |T| probes drawn uniformly from the candidates outside T
  ranked   B - |T| probes = the candidates outside T with the highest historical difficulty
  random   B candidates drawn uniformly from all candidates (budget-matched baseline)
Outcome: recovery of the regressions that lie OUTSIDE T (the failures the targeted set misses).
Proposition 1 predicts p_S = (B-|T|)/(|C|-|T|) for uniform probes and p_R = B/|C| for random, with p_S <= p_R.
Run: python experiments/prop1_real.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swebench_pairs as sp  # noqa: E402
from v6_selection import ALPHA, ci, date, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
B = 0.25
T_SHARES = (0.05, 0.10, 0.20)
SEEDS = range(20)


def main():
    pairs = [p for p in json.loads((ROOT / "results" / "swebench_pairs.json").read_text(encoding="utf-8"))["pairs"]
             if p["category"] in ("M", "H")]
    verified = [s["name"] for s in sp.gh(f"{sp.API}/verified")]
    data = {s: v for s in verified if (v := sp.load("verified", s))}
    out = {"budget": B, "seeds": len(SEEDS), "results": {}}
    for ts in T_SHARES:
        rows = {k: [] for k in ("pred_uniform", "obs_uniform", "pred_random", "obs_random", "obs_ranked")}
        for p in sorted(pairs, key=lambda q: max(date(q["before"]), date(q["after"]))):
            a, b = data[p["before"]], data[p["after"]]
            t0 = min(date(p["before"]), date(p["after"]))
            hist = [s for s in data if date(s) < t0]
            prior = [q for q in pairs if max(date(q["before"]), date(q["after"])) < t0]
            if len(hist) < 5 or len(prior) < 3:
                continue
            cand = sorted(i for i in a if a[i] and i in b)
            reg, sol = {i: 0 for i in cand}, {i: 0 for i in cand}
            for q in prior:
                A, Bq = data[q["before"]], data[q["after"]]
                for i in cand:
                    if A.get(i) and i in Bq:
                        sol[i] += 1; reg[i] += not Bq[i]
            pred = {i: (reg[i] + ALPHA) / (sol[i] + 2 * ALPHA) for i in cand}
            frag = {i: 1 - sum(data[s].get(i, False) for s in hist) / len(hist) for i in cand}
            n = len(cand); nb = max(2, round(B * n)); nt = min(nb - 1, max(1, round(ts * n))); k = nb - nt
            rnd0 = random.Random(zlib.crc32((p["before"] + p["after"]).encode()))
            tie = {i: rnd0.random() for i in cand}
            T = set(sorted(cand, key=lambda i: (-pred[i], tie[i]))[:nt])
            rest = [i for i in cand if i not in T]
            hidden = {i for i in rest if not b[i]}  # regressions the targeted set misses
            if not hidden:
                continue
            ranked = set(sorted(rest, key=lambda i: (-frag[i], tie[i]))[:k])
            u, r = [], []
            for seed in SEEDS:
                rnd = random.Random(seed * 7919 + zlib.crc32(p["after"].encode()) % 10007)
                u.append(len(hidden & set(rnd.sample(rest, k))) / len(hidden))
                r.append(len(hidden & set(rnd.sample(cand, nb))) / len(hidden))
            rows["pred_uniform"].append(k / (n - nt)); rows["obs_uniform"].append(statistics.fmean(u))
            rows["pred_random"].append(nb / n); rows["obs_random"].append(statistics.fmean(r))
            rows["obs_ranked"].append(len(hidden & ranked) / len(hidden))
        res = {key: {"mean": ci(v)[0], "ci95": list(ci(v)[1:])} for key, v in rows.items()}
        res["pairs"] = len(rows["obs_uniform"])
        res["bound_holds_in_all_pairs"] = all(x <= y + 1e-12 for x, y in zip(rows["pred_uniform"], rows["pred_random"]))
        res["p_ranked_vs_random"] = wilcoxon([x - y for x, y in zip(rows["obs_ranked"], rows["obs_random"])])
        res["p_uniform_vs_random"] = wilcoxon([x - y for x, y in zip(rows["obs_uniform"], rows["obs_random"])])
        out["results"][str(ts)] = res
        print(f"|T|={ts:.2f} pairs={res['pairs']} uniform pred={res['pred_uniform']['mean']:.3f} obs={res['obs_uniform']['mean']:.3f} | "
              f"random pred={res['pred_random']['mean']:.3f} obs={res['obs_random']['mean']:.3f} | ranked obs={res['obs_ranked']['mean']:.3f} "
              f"| bound holds: {res['bound_holds_in_all_pairs']} | p(ranked vs random)={res['p_ranked_vs_random']:.2g}")
    (ROOT / "results" / "prop1_real.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
