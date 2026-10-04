"""Sensitivity analyses requested in review: is the noise floor comparable, are the pairs classified correctly,
and do the results depend on the data source or the harness family?

    python experiments/rq1_sensitivity.py   -> results/rq1_sensitivity.json

1. Comparable task set. The reruns are measured on the 93 tasks that SWE-bench Lite and Verified share. Every
   model- and harness-change pair is re-measured on the same 93 tasks, and its regression rate is compared with
   the reruns (Cliff's delta, and a bootstrap 90% interval for the difference in medians, read as equivalence
   when it lies inside +/-0.05).
2. Classification audit. Every pair is checked against the naming rule that defines its category: model-change
   pairs must share the mini-SWE-agent version and differ in model; harness-change pairs are listed by hand with
   the model held fixed, so their model tokens must agree after normalisation.
3. Data source. Per-task outcomes come from per_instance_details.json (complete) or from the resolved list of
   results/results.json complemented by the task list. RQ1 rates are reported separately for the two sources.
4. Harness family. RQ3 gains (results/v6_per_pair.json) are reported separately for pairs inside the
   mini-SWE-agent family and pairs that involve another harness.
Reads the cached archive files written by swebench_pairs.py (results/external/, not redistributed).
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swebench_pairs as sp  # noqa: E402
from robustness import cliffs_delta, magnitude  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED, B_BOOT, MARGIN = 20261004, 10_000, 0.05


def source(sub):
    return "per_instance" if (sp.CACHE / "verified" / f"{sub}.json").exists() else "resolved_list"


def model_token(sub):
    s = re.sub(r"^\d{8}_", "", sub).lower()
    s = re.sub(r"^(mini-v[\d.]+[-_]|prometheus_v[\d.]+_|openhands[-_]|sweagent_|moatless_|livesweagent_|codesweep_sweagent_)", "", s)
    s = re.sub(r"[-_.]", "", s)
    s = re.sub(r"(instruct|preview|\d{8}|\d{4}\d{2}\d{2}|high)$", "", s)
    return s


def main():
    pairs = json.loads((RES / "swebench_pairs.json").read_text(encoding="utf-8"))["pairs"]
    shared = set(sp.ids("lite")) & set(sp.ids("verified"))
    out = {"shared_tasks": len(shared)}

    # 1. Change pairs on the 93 shared tasks.
    rates = {"M": [], "H": []}
    for p in pairs:
        if p["category"] not in rates:
            continue
        a, b = sp.load("verified", p["before"]), sp.load("verified", p["after"])
        common = [i for i in shared if i in a and i in b]
        solved = [i for i in common if a[i]]
        if len(solved) < 5:
            continue
        rates[p["category"]].append(sum(not b[i] for i in solved) / len(solved))
    rerun = [p["regression_rate"] for p in pairs if p["category"] == "N-rerun"]
    rnd = random.Random(SEED)
    res = {}
    for cat, name in (("M", "model"), ("H", "harness")):
        xs = rates[cat]
        diffs = []
        for _ in range(B_BOOT):
            bx = [rnd.choice(xs) for _ in xs]; by = [rnd.choice(rerun) for _ in rerun]
            diffs.append(statistics.median(bx) - statistics.median(by))
        diffs.sort()
        lo, hi = diffs[int(0.05 * B_BOOT)], diffs[int(0.95 * B_BOOT) - 1]
        d = cliffs_delta(xs, rerun)
        res[name] = {"pairs": len(xs), "median_rate_shared93": statistics.median(xs),
                     "rerun_median": statistics.median(rerun), "median_diff_ci90": [lo, hi],
                     "equivalent_within_margin": -MARGIN < lo and hi < MARGIN,
                     "cliffs_delta": d, "magnitude": magnitude(d)}
    out["shared_task_comparison"] = res

    # 2. Classification audit.
    bad = []
    for p in pairs:
        if p["category"] == "M":
            va = re.match(r"\d{8}_mini-(v[\d.]+)", p["before"]); vb = re.match(r"\d{8}_mini-(v[\d.]+)", p["after"])
            if not (va and vb and va[1] == vb[1] and model_token(p["before"]) != model_token(p["after"])):
                bad.append([p["category"], p["before"], p["after"]])
        elif p["category"] == "H":
            if p["before"] == p["after"]:
                bad.append([p["category"], p["before"], p["after"]])
    hm = sorted({(model_token(p["before"]), model_token(p["after"])) for p in pairs if p["category"] == "H"})
    out["classification_audit"] = {"model_pairs_checked": sum(p["category"] == "M" for p in pairs),
                                   "harness_pairs_checked": sum(p["category"] == "H" for p in pairs),
                                   "violations": bad, "harness_model_tokens": hm}

    # 3. Data source.
    src = {}
    for p in pairs:
        if p["category"] in ("M", "H"):
            k = "per_instance" if source(p["before"]) == source(p["after"]) == "per_instance" else "mixed_or_resolved_list"
            src.setdefault(k, []).append(p["regression_rate"])
    out["data_source"] = {k: {"pairs": len(v), "median_regression_rate": statistics.median(v)} for k, v in src.items()}

    # 4. Harness family (RQ3).
    per = json.loads((RES / "v6_per_pair.json").read_text(encoding="utf-8"))
    fam = {}
    for p in per:
        k = "mini_swe_agent_only" if "_mini-" in p["before"] and "_mini-" in p["after"] else "other_harness_involved"
        fam.setdefault(k, []).append(p)
    out["harness_family_rq3"] = {
        k: {"pairs": len(v), **{f"gain_{st}_B{b}": statistics.fmean(q["recall"][b][st] - q["recall"][b]["random"] for q in v)
                                for st in ("fragile", "predictive") for b in ("0.1", "0.25")}}
        for k, v in fam.items()}

    (RES / "rq1_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "classification_audit"}, indent=1))
    ca = out["classification_audit"]
    print("audit: M", ca["model_pairs_checked"], "H", ca["harness_pairs_checked"], "violations", len(ca["violations"]))
    print("harness model tokens:", ca["harness_model_tokens"])


if __name__ == "__main__":
    main()
