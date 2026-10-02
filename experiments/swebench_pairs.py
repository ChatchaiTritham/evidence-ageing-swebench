"""Systematic single-dimension change pairs on SWE-bench Verified, with a noise floor and exact tests.

Extends swebench_flips.py (four hand-picked pairs) to every pair the public archive supports:

  M  model change    same mini-SWE-agent version, different model (all combinations within a version)
  H  harness change  same model, different harness (mini-SWE-agent versions, or different scaffolds)
  C  config change   same model and harness, different reasoning-effort setting
  N  noise floor     same configuration evaluated twice:
                       N-lite : one submission name present in both the Lite and Verified archives,
                                compared on the tasks the two benchmarks share
                       N-resub: one system name resubmitted to Verified on a later date

For every pair: regressions b (solved -> unsolved), improvements c, the regression rate b/solved-before
with a Wilson 95% interval, and an exact McNemar test (two-sided binomial on b vs c), Holm-corrected
within each category. Raw archive files are cached in results/external/ (git-ignored) and never
redistributed. Run: python experiments/swebench_pairs.py
"""
from __future__ import annotations

import itertools
import json
import math
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "external" / "swebench"
API = "repos/SWE-bench/experiments/contents/evaluation"


def gh(path):
    return json.loads(subprocess.run(["gh", "api", "--paginate", path], capture_output=True, text=True, check=True).stdout)


def ids(split):
    """Task universe of a split (instance ids from the princeton-nlp/SWE-bench_{Lite,Verified} datasets)."""
    f = CACHE / f"ids_{split}.json"
    if not f.exists():
        name = {"lite": "SWE-bench_Lite", "verified": "SWE-bench_Verified"}[split]
        out, off = [], 0
        while True:
            u = (f"https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2F{name}"
                 f"&config=default&split=test&offset={off}&length=100")
            d = json.load(urllib.request.urlopen(u, timeout=60))
            out += [r["row"]["instance_id"] for r in d["rows"]]; off += 100
            if off >= d["num_rows_total"]:
                break
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(sorted(out)), encoding="utf-8")
    return json.loads(f.read_text(encoding="utf-8"))


def fetch(split, sub, rel, dest):
    if not dest.exists():
        try:
            url = gh(f"{API}/{split}/{sub}/{rel}")["download_url"]
        except (subprocess.CalledProcessError, KeyError, TypeError):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return True


def load(split, sub):
    """Per-task resolved flags: per_instance_details.json when present (mini-SWE-agent), otherwise
    results/results.json, whose 'resolved' list is complemented by the split's task universe."""
    f = CACHE / split / f"{sub}.json"
    if fetch(split, sub, "per_instance_details.json", f):
        d = json.loads(f.read_text(encoding="utf-8"))
        return {k: bool(v.get("resolved")) for k, v in d.items()}
    g = CACHE / split / f"{sub}.results.json"
    if fetch(split, sub, "results/results.json", g):
        solved = set(json.loads(g.read_text(encoding="utf-8")).get("resolved", []))
        return {i: i in solved for i in ids(split)}
    return None


def binom_two_sided(b, c):
    """Exact McNemar p-value: two-sided binomial test of b successes in b+c trials at p = 0.5."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def compare(a, b):
    common = sorted(set(a) & set(b))
    reg = sum(a[i] and not b[i] for i in common)
    imp = sum(b[i] and not a[i] for i in common)
    solved = sum(a[i] for i in common)
    lo, hi = wilson(reg, solved)
    return {"n": len(common), "solved_before": solved, "solved_after": sum(b[i] for i in common),
            "regressions": reg, "improvements": imp, "regression_rate": reg / solved if solved else 0.0,
            "rate_ci95": [lo, hi], "flip_rate": (reg + imp) / len(common) if common else 0.0,
            "mcnemar_p": binom_two_sided(reg, imp)}


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["mcnemar_p"])
    m, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["mcnemar_p"]))
        rows[i]["p_holm"] = running


# Same model under different harnesses (model identity checked by name; see docs/DATASETS.md).
HARNESS = [
    ("gpt-5-mini", "20250807_mini-v1.7.0_gpt-5-mini", "20260217_mini-v2.0.0_gpt-5-mini"),
    ("gpt-5.2 (high)", "20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high", "20260217_mini-v2.0.0_gpt-5-2-high"),
    ("gpt-5", "20250929_Prometheus_v1.2_gpt5", "20251015_Prometheus_v1.2.1_gpt5"),
    ("gpt-5", "20250807_mini-v1.7.0_gpt-5", "20250807_openhands_gpt5"),
    ("Claude Sonnet 4", "20250726_mini-v1.0.0_claude-sonnet-4-20250514", "20250522_sweagent_claude-4-sonnet-20250514"),
    ("Claude Sonnet 4", "20250726_mini-v1.0.0_claude-sonnet-4-20250514", "20250524_openhands_claude_4_sonnet"),
    ("Claude Sonnet 4", "20250726_mini-v1.0.0_claude-sonnet-4-20250514", "20250611_moatless_claude-4-sonnet-20250514"),
    ("Claude Sonnet 4", "20250522_sweagent_claude-4-sonnet-20250514", "20250524_openhands_claude_4_sonnet"),
    ("Claude Sonnet 4", "20250522_sweagent_claude-4-sonnet-20250514", "20250611_moatless_claude-4-sonnet-20250514"),
    ("Claude Sonnet 4", "20250524_openhands_claude_4_sonnet", "20250611_moatless_claude-4-sonnet-20250514"),
    ("Claude Opus 4.5", "20251124_mini-v1.16.0_claude-opus-4-5-20251101", "20251127_openhands_claude-opus-4-5"),
    ("Claude Opus 4.5", "20251124_mini-v1.16.0_claude-opus-4-5-20251101", "20251215_livesweagent_claude-opus-4-5"),
    ("Claude Opus 4.5", "20251127_openhands_claude-opus-4-5", "20251215_livesweagent_claude-opus-4-5"),
    ("Kimi K2", "20250807_mini-v1.7.0_kimi-k2-instruct", "20250716_openhands_kimi_k2"),
    ("Kimi K2", "20250807_mini-v1.7.0_kimi-k2-instruct", "20250804_codesweep_sweagent_kimi_k2_instruct"),
    ("Qwen3-Coder-480B", "20250802_mini-v1.0.0_qwen3-coder-480b-a35b-instruct", "20250805_openhands-Qwen3-Coder-480B-A35B-Instruct"),
    ("Gemini 3 Pro (preview)", "20251118_mini-v1.15.0_gemini-3-pro-preview-20251118", "20251120_livesweagent_gemini-3-pro-preview"),
]
CONFIG = [("gpt-5.2 reasoning default -> high", "20251211_mini-v1.17.2_gpt-5.2-2025-12-11", "20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high")]


def main():
    verified = [s["name"] for s in gh(f"{API}/verified")]
    lite = [s["name"] for s in gh(f"{API}/lite")]
    rows = []

    def add(cat, label, a, b, split_a="verified", split_b="verified", restrict=None):
        A, B = load(split_a, a), load(split_b, b)
        if not (A and B):
            print("skip (no per-instance data):", a, "|", b); return
        if restrict is not None:
            A = {k: v for k, v in A.items() if k in restrict}; B = {k: v for k, v in B.items() if k in restrict}
        rows.append({"category": cat, "label": label, "before": a, "after": b, **compare(A, B)})

    # M: every model pair within one mini-SWE-agent version.
    by_ver = {}
    for s in verified:
        m = re.match(r"\d{8}_mini-(v[\d.]+)[-_](.+)", s)
        if m:
            by_ver.setdefault(m[1], []).append(s)
    for ver, subs in sorted(by_ver.items()):
        for a, b in itertools.combinations(sorted(subs), 2):
            add("M", f"model change within mini-SWE-agent {ver}", a, b)
    for label, a, b in HARNESS:
        add("H", label, a, b)
    for label, a, b in CONFIG:
        add("C", label, a, b)
    # N-lite: same submission name on both benchmarks, restricted to the shared tasks.
    for s in sorted(set(verified) & set(lite)):
        A, B = load("lite", s), load("verified", s)
        if A and B:
            add("N-lite", "same submission on Lite and Verified", s, s, "lite", "verified", restrict=set(A) & set(B))
    # N-resub: one system name resubmitted later.
    base = {}
    for s in verified:
        base.setdefault(re.sub(r"^\d{8}_", "", s), []).append(s)
    for name, subs in sorted(base.items()):
        if len(subs) > 1 and "mini-v" not in name:
            for a, b in zip(sorted(subs), sorted(subs)[1:]):
                add("N-resub", f"resubmission of {name}", a, b)

    # A Lite/Verified pair with zero flips on the shared tasks re-grades the same predictions; one with flips
    # is a separate run of the same declared configuration and so estimates run-to-run noise.
    for r in rows:
        if r["category"] == "N-lite":
            r["category"] = "N-regrade" if r["regressions"] + r["improvements"] == 0 else "N-rerun"
    for cat in {r["category"] for r in rows}:
        holm([r for r in rows if r["category"] == cat])
    summary = {}
    for cat in ("M", "H", "C", "N-rerun", "N-regrade", "N-resub"):
        rs = [r for r in rows if r["category"] == cat]
        if not rs:
            continue
        rates = sorted(r["regression_rate"] for r in rs)
        summary[cat] = {"pairs": len(rs), "median_regression_rate": rates[len(rates) // 2],
                        "min": rates[0], "max": rates[-1],
                        "median_flip_rate": sorted(r["flip_rate"] for r in rs)[len(rs) // 2],
                        "significant_after_holm": sum(r["p_holm"] < 0.05 for r in rs),
                        "pooled_regression_rate": sum(r["regressions"] for r in rs) / max(1, sum(r["solved_before"] for r in rs)),
                        "median_asymmetry": sorted(abs(r["improvements"] - r["regressions"]) / max(1, r["improvements"] + r["regressions"]) for r in rs)[len(rs) // 2]}
    # Do change pairs lose more solved tasks than same-configuration reruns? One-sided Mann-Whitney U
    # (normal approximation) on per-pair regression rates.
    def mwu(x, y):
        allv = sorted((v, g) for g, vs in ((0, x), (1, y)) for v in vs)
        ranks, i = {}, 0
        while i < len(allv):
            j = i
            while j < len(allv) and allv[j][0] == allv[i][0]:
                j += 1
            for k in range(i, j):
                ranks.setdefault(allv[k][1], []).append((i + j + 1) / 2)
            i = j
        n1, n2 = len(x), len(y)
        u = sum(ranks.get(0, [])) - n1 * (n1 + 1) / 2
        z = (u - n1 * n2 / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        return {"U": u, "z": z, "p_one_sided": 0.5 * math.erfc(z / math.sqrt(2)), "auc": u / (n1 * n2)}
    noise = [r["regression_rate"] for r in rows if r["category"] == "N-rerun"]
    tests = {}
    for cat in ("H", "M"):
        xs = [r["regression_rate"] for r in rows if r["category"] == cat]
        if xs and noise:
            tests[f"{cat}_vs_N-rerun_regression_rate"] = mwu(xs, noise)
        xa = [abs(r["improvements"] - r["regressions"]) / max(1, r["improvements"] + r["regressions"]) for r in rows if r["category"] == cat]
        na = [abs(r["improvements"] - r["regressions"]) / max(1, r["improvements"] + r["regressions"]) for r in rows if r["category"] == "N-rerun"]
        if xa and na:
            tests[f"{cat}_vs_N-rerun_asymmetry"] = mwu(xa, na)
    summary["tests"] = tests
    out = {"source": "SWE-bench/experiments, evaluation/{verified,lite}/*/per_instance_details.json",
           "summary": summary, "pairs": rows}
    (ROOT / "results" / "swebench_pairs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cat, s in [(k, v) for k, v in summary.items() if k != "tests"]:
        print(f"{cat:8s} pairs={s['pairs']:3d} regression rate median={s['median_regression_rate']:.3f} "
              f"[{s['min']:.3f}, {s['max']:.3f}] flip median={s['median_flip_rate']:.3f} sig(Holm)={s['significant_after_holm']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
