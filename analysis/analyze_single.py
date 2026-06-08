"""
Generic analyzer for a single combined records file (records tagged by `source`).
Used for the Phi-3.5 generality run. Reports, per dataset and pooled:
AUROC of signals, conformal operating point + coverage dial, adaptive skip rate,
and the fixed-threshold cross-distribution drift.

Run: ./.venv/bin/python analysis/analyze_single.py <records.json>
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import conformal_core as core

FEATURES = ["semantic_entropy", "ignorance_1msc", "neg_mean_logprob", "answer_len"]


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pack(recs):
    return {"X": np.array([[r[f] for f in FEATURES] for r in recs], float),
            "wrong": np.array([bool(r["wrong"]) for r in recs]),
            "lp": np.array([r["neg_mean_logprob"] for r in recs], float),
            "recs": recs}


def conf_op(d, alpha):
    return core.repeated_split_eval(d["X"], d["wrong"], alpha=alpha,
                                    n_splits=300, seed=1)["methods"]["conformal"]


def fixed_drift(src, tgt, alpha=0.10, n=200, seed=2):
    rng = np.random.default_rng(seed); harm = []
    for _ in range(n):
        si = rng.choice(len(src["wrong"]), len(src["wrong"]), replace=True)
        ti = rng.choice(len(tgt["wrong"]), len(tgt["wrong"]), replace=True)
        tau = core.tuned_threshold(src["lp"][si], src["wrong"][si], alpha)
        harm.append(float(((tgt["lp"][ti] <= tau) & tgt["wrong"][ti]).mean()))
    return float(np.mean(harm))


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        c = glob.glob(str(ROOT / "results" / "**" / "phase3_phi_records.json"), recursive=True)
        path = sorted(c)[-1]
    with open(path) as f:
        data = json.load(f)
    recs = data["records"] if isinstance(data, dict) else data
    print(f"  {path}  n={len(recs)}")

    by = {}
    for r in recs:
        by.setdefault(r["source"], []).append(r)
    D = {k: pack(v) for k, v in by.items()}
    D["pooled"] = pack(recs)

    print("\n  domain      acc   AUROC(SE/1msc/logprob)     conf_harm  retr   skip")
    for name, d in D.items():
        acc = 1 - d["wrong"].mean()
        a_se = auroc([r["semantic_entropy"] for r in d["recs"]], d["wrong"])
        a_sc = auroc([r["ignorance_1msc"] for r in d["recs"]], d["wrong"])
        a_lp = auroc(d["lp"], d["wrong"])
        c = conf_op(d, 0.10)
        print(f"  {name:9s} {acc:5.3f}   {a_se:.3f}/{a_sc:.3f}/{a_lp:.3f}        "
              f"{c['harmful_mean']:.3f}    {c['retrieval_mean']:.3f}  {1-c['retrieval_mean']:.0%}")
        core.format_result(f"phi_{name}_acc", round(float(acc), 3))
        core.format_result(f"phi_{name}_auroc_logprob", round(a_lp, 3))
        core.format_result(f"phi_{name}_conf_harm", round(c["harmful_mean"], 3))
        core.format_result(f"phi_{name}_retr", round(c["retrieval_mean"], 3))

    print("\n  coverage dial (pooled):")
    for a in (0.02, 0.05, 0.10, 0.15, 0.20):
        c = conf_op(D["pooled"], a)
        print(f"    alpha={a:.2f}: harm={c['harmful_mean']:.3f} retrieval={c['retrieval_mean']:.3f}")
        core.format_result(f"phi_cov_a{a}_harm", round(c["harmful_mean"], 3))

    if "triviaqa" in D and "popqa" in D:
        print("\n  fixed-threshold drift (logprob cutoff reused across distributions):")
        for s, t in (("popqa", "triviaqa"), ("triviaqa", "popqa")):
            h = fixed_drift(D[s], D[t])
            print(f"    {s} -> {t}: harm={h:.3f} (target 0.10, drift {h-0.10:+.3f})")
            core.format_result(f"phi_fixed_{s}_to_{t}", round(h, 3))
    print("\nDone.")


if __name__ == "__main__":
    main()
