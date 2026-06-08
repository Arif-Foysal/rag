"""
Stress probe: WHERE does marginal conformal break under real shift, and does
group-conditional (Mondrian) fix it? Realistic source->target protocol:
fit ignorance model + calibrate threshold on SOURCE, deploy on TARGET.

We sweep:
  - shift severity (source pool -> target pool)
  - score choice (multivariate model vs single raw signals)
and report deploy E[harmful-skip] for marginal vs dataset/group-conditional.

The point: find the regime (if any) where a well-separated but conditionally
MIScalibrated score makes marginal violate alpha while Mondrian holds — that is
the honest real-data case for group-conditioning. If none exists, the paper's
honest message is that a conditionally-calibrated multivariate score makes
marginal control shift-robust on its own.

Run: ./.venv/bin/python analysis/shift_probe.py
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


def build():
    pop = _load("phase2_records.json"); tri = _load("phase2_trivia_records.json")
    lp = np.array([r["log_pop"] for r in pop], float)
    q = np.quantile(lp, [1/3, 2/3])
    pop_b = 1 + (2 - np.digitize(lp, q))      # 1 popular .. 3 long-tail
    recs, bucket = [], []
    for r in tri:
        recs.append(r); bucket.append(0)
    for r, b in zip(pop, pop_b):
        recs.append(r); bucket.append(int(b))
    X = np.array([[r[f] for f in FEATURES] for r in recs], float)
    wrong = np.array([bool(r["wrong"]) for r in recs])
    return recs, X, wrong, np.array(bucket)


def single_score(recs, name):
    return np.array([r[name] for r in recs], float)


def run_shift(X, wrong, bucket, src_buckets, tgt_buckets, score_mode="multi",
              alpha=0.10, n_draws=200, seed=0, recs=None):
    """Fit+calibrate on source buckets, deploy on target buckets."""
    rng = np.random.default_rng(seed)
    src_idx = np.where(np.isin(bucket, src_buckets))[0]
    tgt_idx = np.where(np.isin(bucket, tgt_buckets))[0]
    raw = None if score_mode == "multi" else single_score(recs, score_mode)

    marg, cond = [], []
    for _ in range(n_draws):
        s_tr = rng.choice(src_idx, len(src_idx), replace=True)
        # split source into fit/cal
        h = len(s_tr) // 2
        fit_i, cal_i = s_tr[:h], s_tr[h:]
        t_i = rng.choice(tgt_idx, len(tgt_idx), replace=True)
        if score_mode == "multi":
            model = core.IgnoranceModel().fit(X[fit_i], wrong[fit_i])
            s_cal, s_te = model.score(X[cal_i]), model.score(X[t_i])
        else:
            s_cal, s_te = raw[cal_i], raw[t_i]
        w_cal, w_te = wrong[cal_i], wrong[t_i]
        b_cal, b_te = bucket[cal_i], bucket[t_i]

        tau_m = core.conformal_risk_threshold(s_cal, w_cal, alpha)
        marg.append(float(((s_te <= tau_m) & w_te).mean()))
        # group-conditional: per-bucket tau. Target buckets unseen in source get
        # the most conservative (skip-none) tau -> falls back safely.
        taus = core.group_conformal_thresholds(s_cal, w_cal, b_cal, alpha)
        skip = np.zeros(len(t_i), bool)
        for b in np.unique(b_te):
            tau_b = taus.get(b, -np.inf)
            skip[b_te == b] = s_te[b_te == b] <= tau_b
        cond.append(float((skip & w_te).mean()))
    return float(np.mean(marg)), float(np.mean(cond))


def main():
    recs, X, wrong, bucket = build()
    print(f"  n={len(recs)} buckets(0=trivia,1=pop,2=mid,3=longtail)={np.bincount(bucket).tolist()}")
    alpha = 0.10
    configs = [
        ("trivia -> popqa-longtail", [0], [3]),
        ("trivia -> popqa-all",      [0], [1, 2, 3]),
        ("trivia+pop -> mid+long",   [0, 1], [2, 3]),
        ("popqa-pop -> popqa-long",  [1], [3]),
    ]
    scores = ["multi", "neg_mean_logprob", "ignorance_1msc", "semantic_entropy"]
    print(f"\n  alpha={alpha}. Deploy E[harmful-skip]: marginal / group-conditional")
    print(f"  {'shift':28s} {'score':18s} {'marg':>7s} {'group':>7s}  flag")
    worst = None
    for label, sb, tb in configs:
        for sm in scores:
            m, c = run_shift(X, wrong, bucket, sb, tb, score_mode=sm,
                             alpha=alpha, recs=recs)
            flag = ""
            if m > alpha + 0.02:
                flag = "MARGINAL VIOLATES"
                if c <= alpha + 0.01:
                    flag += " -> Mondrian fixes"
                    if worst is None or m - c > worst[1]:
                        worst = ((label, sm, m, c), m - c)
            print(f"  {label:28s} {sm:18s} {m:7.3f} {c:7.3f}  {flag}")
    if worst:
        (lbl, sm, m, c), gap = worst
        print(f"\n  >>> Best real Mondrian case: {lbl} | {sm}: marginal {m:.3f} vs group {c:.3f}")
        core.format_result("probe_best_shift", lbl)
        core.format_result("probe_best_score", sm)
        core.format_result("probe_marg", round(m, 3))
        core.format_result("probe_group", round(c, 3))
    else:
        print("\n  >>> No real config makes marginal violate by >0.02. "
              "Honest message: conditionally-calibrated score => marginal is shift-robust.")
        core.format_result("probe_marginal_robust", True)


if __name__ == "__main__":
    main()
