"""
Baseline frontier: what does the conformal gate buy over the alternatives?

For TriviaQA, PopQA, and pooled, at alpha=0.10 unless swept:
  - never-retrieve  : harm = wrong-rate, retrieval = 0   (the do-nothing risk)
  - always-retrieve : harm = 0,         retrieval = 1     (the safe, costly one)
  - fixed-confidence threshold (NOT calibrated per-distribution): a single raw
    neg_logprob cutoff reused across distributions -> harm is UNCONTROLLED and
    drifts by distribution (the FLARE/Self-RAG-style failure mode).
  - conformal gate (recalibrated per distribution): harm == target alpha with a
    finite-sample guarantee, on EACH distribution.
  - cross-distribution transfer: calibrate on source, deploy on target -> how
    far does realized harm drift from alpha?

The contribution this isolates: only the conformal gate lets you DIAL the
hallucination-causing error to a chosen level, per distribution, with a
guarantee. Run: ./.venv/bin/python analysis/baselines.py
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


def _load(pat):
    cands = glob.glob(str(ROOT / "results" / "**" / pat), recursive=True)
    with open(sorted(cands)[-1]) as f:
        d = json.load(f)
    return d["records"] if isinstance(d, dict) else d


def domains():
    pop = _load("phase2_records.json"); tri = _load("phase2_trivia_records.json")
    def pack(recs):
        X = np.array([[r[f] for f in FEATURES] for r in recs], float)
        wrong = np.array([bool(r["wrong"]) for r in recs])
        lp = np.array([r["neg_mean_logprob"] for r in recs], float)
        return {"X": X, "wrong": wrong, "lp": lp, "recs": recs}
    return {"triviaqa": pack(tri), "popqa": pack(pop),
            "pooled": pack(tri + pop)}


def conformal_op(X, wrong, alpha, n_splits=300, seed=1):
    res = core.repeated_split_eval(X, wrong, alpha=alpha, n_splits=n_splits, seed=seed)
    return res["methods"]["conformal"]


def fixed_threshold_harm(d_src, d_tgt, alpha=0.10, n_draws=200, seed=2):
    """Pick a raw neg_logprob cutoff on the SOURCE to hit alpha, reuse the SAME
    raw cutoff on the TARGET (no recalibration). Returns target harm/retrieval."""
    rng = np.random.default_rng(seed)
    harm, retr = [], []
    for _ in range(n_draws):
        si = rng.choice(len(d_src["wrong"]), len(d_src["wrong"]), replace=True)
        ti = rng.choice(len(d_tgt["wrong"]), len(d_tgt["wrong"]), replace=True)
        tau = core.tuned_threshold(d_src["lp"][si], d_src["wrong"][si], alpha)
        s_te, w_te = d_tgt["lp"][ti], d_tgt["wrong"][ti]
        harm.append(float(((s_te <= tau) & w_te).mean()))
        retr.append(float((s_te > tau).mean()))
    return float(np.mean(harm)), float(np.mean(retr))


def main():
    D = domains()
    alpha = 0.10
    print(f"=== Reference points & conformal (alpha={alpha}) ===")
    print(f"  {'domain':9s} {'wrong':>6s} | {'conformal harm':>14s} {'conf retr':>10s}")
    for name, d in D.items():
        wr = float(d["wrong"].mean())
        c = conformal_op(d["X"], d["wrong"], alpha)
        print(f"  {name:9s} {wr:6.3f} | {c['harmful_mean']:14.3f} {c['retrieval_mean']:10.3f}")
        core.format_result(f"bl_{name}_wrongrate", round(wr, 3))
        core.format_result(f"bl_{name}_conf_harm", round(c["harmful_mean"], 3))
        core.format_result(f"bl_{name}_conf_retr", round(c["retrieval_mean"], 3))

    print("\n=== Fixed (un-recalibrated) confidence threshold drifts by distribution ===")
    print("  A single neg_logprob cutoff set to hit alpha=0.10 on SOURCE, reused on TARGET:")
    for src in ("triviaqa", "popqa"):
        for tgt in ("triviaqa", "popqa"):
            if src == tgt:
                continue
            h, r = fixed_threshold_harm(D[src], D[tgt], alpha)
            drift = h - alpha
            print(f"  {src:9s} -> {tgt:9s}: target harm={h:.3f} (drift {drift:+.3f} from "
                  f"alpha), retrieval={r:.3f}")
            core.format_result(f"bl_fixed_{src}_to_{tgt}_harm", round(h, 3))

    print("\n=== Conformal harm/retrieval frontier (pooled, dial-able) ===")
    for a in (0.02, 0.05, 0.10, 0.15, 0.20):
        c = conformal_op(D["pooled"]["X"], D["pooled"]["wrong"], a)
        print(f"  alpha={a:.2f}: harm={c['harmful_mean']:.3f}  retrieval={c['retrieval_mean']:.3f}")
    print("\n  Reference: never-retrieve harm = pooled wrong-rate = "
          f"{D['pooled']['wrong'].mean():.3f} (retrieval 0); "
          f"always-retrieve harm = 0 (retrieval 1).")
    print("\nDone.")


if __name__ == "__main__":
    main()
