"""
Combined cross-distribution analysis: TriviaQA (known-heavy) + PopQA (long-tail).

This produces the paper's main results on REAL data:
  1. Per-dataset operating points (efficiency where the model KNOWS vs long-tail).
  2. Coverage-vs-alpha on the pooled data (multivariate conformal gate).
  3. HEADLINE cross-distribution shift: calibrate on a known-heavy mix, deploy on
     a long-tail-heavy mix. Marginal conformal blows the safety budget; group-
     conditional (Mondrian) over a known->long-tail novelty bucket restores it.

Novelty buckets (known -> unknown):
  0 = TriviaQA            (model usually knows)
  1 = PopQA popular third
  2 = PopQA middle third
  3 = PopQA long-tail third

Run: ./.venv/bin/python analysis/combined_analysis.py
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


def _load(glob_pat):
    cands = glob.glob(str(ROOT / "results" / "**" / glob_pat), recursive=True)
    if not cands:
        return None
    with open(sorted(cands)[-1]) as f:
        d = json.load(f)
    return d["records"] if isinstance(d, dict) else d


def auroc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    return float((ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))


def build():
    pop = _load("phase2_records.json")          # PopQA
    tri = _load("phase2_trivia_records.json")    # TriviaQA
    if pop is None or tri is None:
        raise FileNotFoundError(f"need both; popqa={pop is not None} trivia={tri is not None}")
    # PopQA popularity terciles -> buckets 1,2,3 (popular..long-tail)
    lp = np.array([r["log_pop"] for r in pop], float)
    q = np.quantile(lp, [1/3, 2/3])
    pop_b = 1 + (2 - np.digitize(lp, q))   # high pop -> 1, low pop -> 3
    recs, buckets, src = [], [], []
    for r in tri:
        recs.append(r); buckets.append(0); src.append("triviaqa")
    for r, b in zip(pop, pop_b):
        recs.append(r); buckets.append(int(b)); src.append("popqa")
    X = np.array([[r[f] for f in FEATURES] for r in recs], float)
    wrong = np.array([bool(r["wrong"]) for r in recs])
    buckets = np.array(buckets); src = np.array(src)
    return recs, X, wrong, buckets, src


def per_dataset(recs, X, wrong, src, alpha=0.10):
    print("\n=== 1. Per-dataset operating points (alpha=0.10) ===")
    out = {}
    for name in ("triviaqa", "popqa"):
        m = src == name
        acc = 1 - wrong[m].mean()
        res = core.repeated_split_eval(X[m], wrong[m], alpha=alpha, n_splits=300, seed=1)
        c = res["methods"]["conformal"]
        print(f"  {name:9s}: acc={acc:.3f}  conformal E[harm]={c['harmful_mean']:.3f}  "
              f"retrieval={c['retrieval_mean']:.3f}  (skip {1-c['retrieval_mean']:.0%})")
        core.format_result(f"{name}_accuracy", round(float(acc), 3))
        core.format_result(f"{name}_retrieval_at_a10", round(c["retrieval_mean"], 3))
        core.format_result(f"{name}_harm_at_a10", round(c["harmful_mean"], 3))
        out[name] = {"acc": float(acc), **c}
    return out


def coverage(X, wrong, alphas=(0.02, 0.05, 0.10, 0.15, 0.20)):
    print("\n=== 2. Coverage vs alpha (pooled, multivariate conformal) ===")
    rows = {}
    for a in alphas:
        res = core.repeated_split_eval(X, wrong, alpha=a, n_splits=300, seed=3)
        c = res["methods"]["conformal"]
        rows[str(a)] = c
        print(f"  alpha={a:.2f}: E[harm]={c['harmful_mean']:.3f}  retrieval={c['retrieval_mean']:.3f}")
        core.format_result(f"cov_a{a}_harm", round(c["harmful_mean"], 3))
        core.format_result(f"cov_a{a}_retr", round(c["retrieval_mean"], 3))
    return rows


def headline_shift(X, wrong, buckets, alpha=0.10, n_draws=200, seed=0):
    """Calibrate on a KNOWN-heavy mix, deploy on a LONG-TAIL-heavy mix."""
    rng = np.random.default_rng(seed)
    pools = {b: np.where(buckets == b)[0] for b in np.unique(buckets)}
    known = [0, 1]          # triviaqa + popqa-popular
    unkn = [2, 3]           # popqa mid + long-tail

    def draw(n, frac_unknown):
        n_u = int(n * frac_unknown); n_k = n - n_u
        idx_k = np.concatenate([rng.choice(pools[b], n_k // len(known), replace=True) for b in known])
        idx_u = np.concatenate([rng.choice(pools[b], n_u // len(unkn), replace=True) for b in unkn])
        idx = np.concatenate([idx_k, idx_u])
        return idx[rng.permutation(len(idx))]

    acc = {k: [] for k in ("marg_overall", "marg_unknown", "grp_overall", "grp_unknown",
                           "marg_retr", "grp_retr")}
    for _ in range(n_draws):
        ci = draw(1500, frac_unknown=0.20)   # calibration: known-heavy
        ti = draw(1500, frac_unknown=0.80)   # deploy: long-tail-heavy
        h = len(ci) // 2
        model = core.IgnoranceModel().fit(X[ci[:h]], wrong[ci[:h]])
        s_cal, w_cal, b_cal = model.score(X[ci[h:]]), wrong[ci[h:]], buckets[ci[h:]]
        s_te, w_te, b_te = model.score(X[ti]), wrong[ti], buckets[ti]
        tau_m = core.conformal_risk_threshold(s_cal, w_cal, alpha)
        taus = core.group_conformal_thresholds(s_cal, w_cal, b_cal, alpha)
        gm = core.group_gate_metrics(s_te, w_te, b_te, taus)
        um = np.isin(b_te, unkn)
        acc["marg_overall"].append(float(((s_te <= tau_m) & w_te).mean()))
        acc["marg_unknown"].append(float(((s_te[um] <= tau_m) & w_te[um]).mean()))
        acc["marg_retr"].append(float((s_te > tau_m).mean()))
        acc["grp_overall"].append(gm["overall"]["harmful_skip_rate"])
        gu = [gm["per_group"][str(b)]["harmful_skip_rate"] for b in unkn if str(b) in gm["per_group"]]
        acc["grp_unknown"].append(float(np.mean(gu)) if gu else float("nan"))
        acc["grp_retr"].append(gm["overall"]["retrieval_freq"])
    M = {k: float(np.nanmean(v)) for k, v in acc.items()}
    print("\n=== 3. HEADLINE: cross-distribution shift + Mondrian (real) ===")
    print(f"  alpha={alpha}; calibrate known-heavy (20% unknown) -> deploy 80% unknown")
    print(f"  marginal conformal : E[harm] overall={M['marg_overall']:.3f}  "
          f"unknown={M['marg_unknown']:.3f}  retrieval={M['marg_retr']:.3f}")
    print(f"  group conformal    : E[harm] overall={M['grp_overall']:.3f}  "
          f"unknown={M['grp_unknown']:.3f}  retrieval={M['grp_retr']:.3f}")
    for k, v in M.items():
        core.format_result(f"head_{k}", round(v, 3))
    return M


def main():
    recs, X, wrong, buckets, src = build()
    print(f"  pooled n={len(recs)}  overall acc={1-wrong.mean():.3f}  "
          f"buckets={np.bincount(buckets).tolist()}")
    for f in FEATURES:
        core.format_result(f"auroc_{f}", round(auroc([r[f] for r in recs], wrong), 3))
    results = {
        "n": len(recs), "accuracy": float(1 - wrong.mean()),
        "per_dataset": per_dataset(recs, X, wrong, src),
        "coverage": coverage(X, wrong),
        "headline": headline_shift(X, wrong, buckets),
    }
    core.save_results(results, ROOT / "results" / "combined_analysis.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
