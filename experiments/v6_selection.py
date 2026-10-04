"""V6: which selection strategy finds the tasks a change will break, on real SWE-bench Verified data?

For every model-change and harness-change pair (results/swebench_pairs.json), the candidates are the tasks the
"before" submission solved; the targets are those the "after" submission no longer solves. Each strategy ranks the
candidates using ONLY submissions dated strictly before the pair (no look-ahead) and selects a fraction B of them.
Recall = share of the pair's regressions inside the selection. Random selection has expected recall B.

Strategies (each adapted from published work):
  random       budget-matched uniform selection (baseline)
  coarse       whole repositories, ranked by their historical regression rate (module-level RTS analogue)
  predictive   per-task historical regression rate in earlier change pairs (after Machalica et al., predictive selection)
  flaky        per-task outcome instability across earlier submissions, p(1-p) (after flaky-test studies)
  fragile      per-task low historical solve rate (difficulty)
  informed     Eq. (5) of the proposal: smoothed historical regression rate, mixed with a uniform share eps

Statistics: per-pair recall; mean with 95% CI; paired Wilcoxon signed-rank vs random (normal approximation),
Holm-corrected over strategies. Reported for all pairs and for pairs with a significant directional shift.
Run: python experiments/v6_selection.py   (uses the cache filled by experiments/swebench_pairs.py)
"""
from __future__ import annotations

import json
import math
import zlib
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swebench_pairs as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (0.10, 0.25, 0.50)
SEEDS = range(20)
ALPHA, EPS = 1.0, 0.1


def date(sub):
    return sub[:8]


def wilcoxon(diffs):
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    ranked = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(d[ranked[j]]) == abs(d[ranked[i]]):
            j += 1
        for k in range(i, j):
            ranks[ranked[k]] = (i + j + 1) / 2
        i = j
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sd
    return math.erfc(abs(z) / math.sqrt(2))  # two-sided


def ci(xs):
    m = statistics.fmean(xs)
    h = 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, m - h, m + h


def main():
    pairs = [p for p in json.loads((ROOT / "results" / "swebench_pairs.json").read_text(encoding="utf-8"))["pairs"]
             if p["category"] in ("M", "H")]
    subs = sorted({s for p in pairs for s in (p["before"], p["after"])} |
                  {s for s in sp.ids("verified") and []})
    verified = [s["name"] for s in sp.gh(f"{sp.API}/verified")]
    data = {s: sp.load("verified", s) for s in verified}
    data = {s: v for s, v in data.items() if v}
    rows = {st: {b: [] for b in BUDGETS} for st in ("random", "coarse", "predictive", "flaky", "fragile", "informed")}
    used = []
    for p in sorted(pairs, key=lambda q: max(date(q["before"]), date(q["after"]))):
        a, b = data[p["before"]], data[p["after"]]
        t0 = min(date(p["before"]), date(p["after"]))
        hist = [s for s in data if date(s) < t0]
        prior = [q for q in pairs if max(date(q["before"]), date(q["after"])) < t0]
        if len(hist) < 5 or len(prior) < 3:
            continue  # not enough history to learn from
        cand = sorted(i for i in a if a[i] and i in b)
        target = {i for i in cand if not b[i]}
        if not target:
            continue
        # per-task history
        solve = {i: [data[s].get(i, False) for s in hist] for i in cand}
        reg, sol = {i: 0 for i in cand}, {i: 0 for i in cand}
        repo_reg, repo_sol = {}, {}
        for q in prior:
            A, B = data[q["before"]], data[q["after"]]
            for i in cand:
                if A.get(i) and i in B:
                    sol[i] += 1
                    reg[i] += not B[i]
            for i, v in A.items():
                if v and i in B:
                    r = i.split("__")[0]
                    repo_sol[r] = repo_sol.get(r, 0) + 1
                    repo_reg[r] = repo_reg.get(r, 0) + (not B[i])
        p_hat = {i: sum(solve[i]) / len(solve[i]) for i in cand}
        scores = {
            "predictive": {i: (reg[i] + ALPHA) / (sol[i] + 2 * ALPHA) for i in cand},
            "flaky": {i: p_hat[i] * (1 - p_hat[i]) for i in cand},
            "fragile": {i: 1 - p_hat[i] for i in cand},
            "coarse": {i: (repo_reg.get(i.split("__")[0], 0) + ALPHA) / (repo_sol.get(i.split("__")[0], 0) + 2 * ALPHA) for i in cand},
        }
        used.append(p)
        for bud in BUDGETS:
            k = max(1, round(bud * len(cand)))
            rows["random"][bud].append(k / len(cand) * 0 + len(target & set(cand[:0])) or None)
            rec = {st: [] for st in rows}
            for seed in SEEDS:
                rnd = random.Random(seed * 7919 + zlib.crc32((p["before"] + p["after"]).encode()) % 10007)  # stable across runs
                tie = {i: rnd.random() for i in cand}
                rec["random"].append(len(target & set(rnd.sample(cand, k))) / len(target))
                for st, sc in scores.items():
                    sel = sorted(cand, key=lambda i: (-sc[i], tie[i]))[:k]
                    rec[st].append(len(target & set(sel)) / len(target))
                # informed: sample without replacement from pi = (1-eps)*norm(score) + eps/|C|
                w = scores["predictive"]; tot = sum(w.values())
                pool, picked = list(cand), []
                pi = {i: (1 - EPS) * w[i] / tot + EPS / len(cand) for i in cand}
                for _ in range(k):
                    r, acc = rnd.random() * sum(pi[i] for i in pool), 0.0
                    for i in pool:
                        acc += pi[i]
                        if acc >= r:
                            picked.append(i); pool.remove(i); break
                rec["informed"].append(len(target & set(picked)) / len(target))
            rows["random"][bud][-1] = statistics.fmean(rec["random"])
            for st in rows:
                if st != "random":
                    rows[st][bud].append(statistics.fmean(rec[st]))
    # Control: do the same scores predict flips between two runs of an UNCHANGED configuration?
    allp = json.loads((ROOT / "results" / "swebench_pairs.json").read_text(encoding="utf-8"))["pairs"]
    ctrl = {st: {b: [] for b in BUDGETS} for st in ("random", "flaky", "fragile")}
    n_ctrl = 0
    for p in [q for q in allp if q["category"] == "N-rerun"]:
        a, b = sp.load("lite", p["before"]), data.get(p["after"])
        if not (a and b):
            continue
        hist = [s for s in data if date(s) < date(p["before"])]
        cand = sorted(i for i in a if a[i] and i in b)
        target = {i for i in cand if not b[i]}
        if len(hist) < 5 or not target:
            continue
        n_ctrl += 1
        ph = {i: sum(data[s].get(i, False) for s in hist) / len(hist) for i in cand}
        sc = {"flaky": {i: ph[i] * (1 - ph[i]) for i in cand}, "fragile": {i: 1 - ph[i] for i in cand}}
        for bud in BUDGETS:
            k = max(1, round(bud * len(cand)))
            rec = {st: [] for st in ctrl}
            for seed in SEEDS:
                rnd = random.Random(seed * 7919 + zlib.crc32(p["before"].encode()) % 10007)
                tie = {i: rnd.random() for i in cand}
                rec["random"].append(len(target & set(rnd.sample(cand, k))) / len(target))
                for st in sc:
                    sel = sorted(cand, key=lambda i: (-sc[st][i], tie[i]))[:k]
                    rec[st].append(len(target & set(sel)) / len(target))
            for st in ctrl:
                ctrl[st][bud].append(statistics.fmean(rec[st]))
    sig = [i for i, p in enumerate(used) if p.get("p_holm", 1) < 0.05]
    out = {"pairs_evaluated": len(used), "pairs_significant_shift": len(sig), "budgets": list(BUDGETS),
           "alpha": ALPHA, "eps": EPS, "seeds": len(SEEDS), "results": {}}
    for subset, idx in (("all", list(range(len(used)))), ("significant_shift", sig)):
        res = {}
        for bud in BUDGETS:
            base = [rows["random"][bud][i] for i in idx]
            ps = {}
            for st in rows:
                xs = [rows[st][bud][i] for i in idx]
                m, lo, hi = ci(xs)
                res.setdefault(str(bud), {})[st] = {"mean_recall": m, "ci95": [lo, hi]}
                if st != "random":
                    ps[st] = wilcoxon([x - y for x, y in zip(xs, base)])
            order = sorted(ps, key=ps.get)
            run = 0.0
            for r, st in enumerate(order):
                run = max(run, min(1.0, (len(order) - r) * ps[st]))
                res[str(bud)][st]["p_vs_random_holm"] = run
        out["results"][subset] = res
    out["rerun_control"] = {"pairs": n_ctrl, "results": {
        str(b): {st: {"mean_recall": ci(ctrl[st][b])[0], "ci95": list(ci(ctrl[st][b])[1:])} for st in ctrl} for b in BUDGETS}} if n_ctrl > 1 else {"pairs": n_ctrl}
    (ROOT / "results" / "v6_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Per-pair mean recalls, for the cluster bootstrap in robustness.py (no per-task archive data).
    per_pair = [{"before": p["before"], "after": p["after"], "p_holm": p.get("p_holm"),
                 "recall": {str(b): {st: rows[st][b][i] for st in rows} for b in BUDGETS}} for i, p in enumerate(used)]
    (ROOT / "results" / "v6_per_pair.json").write_text(json.dumps(per_pair, indent=2), encoding="utf-8")
    print(f"pairs evaluated: {len(used)} (significant shift: {len(sig)})")
    for subset in ("all", "significant_shift"):
        print(f"== {subset}")
        for bud in BUDGETS:
            r = out["results"][subset][str(bud)]
            print(f"  B={bud:.2f} " + "  ".join(f"{st}={r[st]['mean_recall']:.3f}" + (f"(p={r[st]['p_vs_random_holm']:.3g})" if st != 'random' else "") for st in rows))
    print("== rerun control, pairs:", n_ctrl)
    for b in BUDGETS:
        if n_ctrl > 1:
            print(f"  B={b:.2f} " + "  ".join(f"{st}={out['rerun_control']['results'][str(b)][st]['mean_recall']:.3f}" for st in ctrl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
