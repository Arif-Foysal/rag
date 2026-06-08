"""
Local synthetic validation of the conformal retrieve/skip gate.

No GPU / no Kaggle. Validates the statistical claims before spending compute:
  A. Marginal CRC controls test harmful-skip <= alpha; an uncorrected tuned
     threshold violates the budget far more often (finite-sample value).
  B. A better ignorance score (well-separated "semantic entropy") yields lower
     retrieval frequency at matched safety than a weak "logprob" score.
  C. HEADLINE: under a shift in group proportions (long-tail queries grow at
     deployment), marginal methods blow the budget; group-conditional (Mondrian)
     conformal maintains per-group and overall control.

Run: ./.venv/bin/python analysis/synth_validation.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import conformal_core as core


def gen_pool(n, p_know, sep_se, sep_lp, noise, rng):
    """Synthetic QA pool.

    known ~ Bernoulli(p_know); wrong = ~known with small label noise.
    Two ignorance features (higher => more likely wrong):
      se : strong separator (stands in for semantic entropy)
      lp : weak separator   (stands in for sequence log-prob confidence)
    """
    known = rng.random(n) < p_know
    wrong = ~known
    flip = rng.random(n) < noise
    wrong = np.where(flip, ~wrong, wrong)
    se = rng.normal(np.where(known, 0.0, sep_se), 1.0)
    lp = rng.normal(np.where(known, 0.0, sep_lp), 1.0)
    X = np.column_stack([se, lp])
    return X, wrong


def expt_A_finite_sample(rng, alpha=0.10):
    """CRC controls the EXPECTED harmful-skip rate (not a per-split PAC bound).
    The controlled quantity is harmful_mean; conformal keeps it <= alpha while
    the uncorrected tuned threshold exceeds alpha in expectation, and the gap
    widens as the calibration set shrinks (finite-sample value of CRC)."""
    print("\n=== Experiment A: finite-sample risk control (iid), n-sweep ===")
    print(f"  alpha = {alpha}.  Reporting E[harmful-skip] over splits (what CRC controls).")
    rows = {}
    for n in (90, 150, 300, 900):
        X, wrong = gen_pool(n, p_know=0.6, sep_se=2.2, sep_lp=0.8, noise=0.02, rng=rng)
        res = core.repeated_split_eval(X, wrong, alpha=alpha, n_splits=600, seed=1)
        c, t = res["methods"]["conformal"], res["methods"]["tuned"]
        n_cal = n // 3
        print(f"  n_cal~{n_cal:4d} | conformal E[harm]={c['harmful_mean']:.3f} "
              f"(retr {c['retrieval_mean']:.3f})  |  tuned E[harm]={t['harmful_mean']:.3f} "
              f"(retr {t['retrieval_mean']:.3f})")
        rows[n_cal] = {"conformal": c, "tuned": t}
    small = rows[min(rows)]
    core.format_result("A_smalln_conformal_Eharm", round(small["conformal"]["harmful_mean"], 3))
    core.format_result("A_smalln_tuned_Eharm", round(small["tuned"]["harmful_mean"], 3))
    return rows


def expt_B_score_quality(rng, alpha=0.10):
    X, wrong = gen_pool(900, p_know=0.6, sep_se=2.5, sep_lp=0.7, noise=0.02, rng=rng)
    # Conformal on the STRONG score only vs the WEAK score only.
    out = {}
    for name, col in [("semantic_entropy", 0), ("logprob", 1)]:
        r = core.repeated_split_eval(X[:, [col]], wrong, alpha=alpha, n_splits=300, seed=2)
        out[name] = r["methods"]["conformal"]
    print("\n=== Experiment B: score quality -> retrieval savings ===")
    for name, m in out.items():
        print(f"  {name:16s}: harmful_mean={m['harmful_mean']:.3f}  "
              f"violation_rate={m['violation_rate']:.3f}  retrieval={m['retrieval_mean']:.3f}")
    core.format_result("B_retrieval_semantic_entropy", round(out["semantic_entropy"]["retrieval_mean"], 3))
    core.format_result("B_retrieval_logprob", round(out["logprob"]["retrieval_mean"], 3))
    return out


def expt_C_shift_mondrian(rng, alpha=0.10, n_draws=200):
    """Calibration mix is mostly 'popular' (easy); deployment is mostly
    'long-tail' (hard). Same per-group conditionals, only the proportions shift.
    Averaged over many draws (CRC controls the EXPECTED rate)."""
    sep_se, sep_lp, noise = 2.2, 0.8, 0.02

    def mixed(n, frac_longtail):
        n1 = int(n * frac_longtail); n0 = n - n1
        X0, w0 = gen_pool(n0, 0.85, sep_se, sep_lp, noise, rng)  # group 0 popular
        X1, w1 = gen_pool(n1, 0.25, sep_se, sep_lp, noise, rng)  # group 1 long-tail
        X = np.vstack([X0, X1]); w = np.concatenate([w0, w1])
        g = np.concatenate([np.zeros(n0, int), np.ones(n1, int)])
        perm = rng.permutation(len(w))
        return X[perm], w[perm], g[perm]

    acc = {k: [] for k in ("marg_overall", "marg_lt", "tuned_overall", "tuned_lt",
                            "grp_overall", "grp_lt", "marg_retr", "grp_retr")}
    for _ in range(n_draws):
        Xc, wc, gc = mixed(1800, 0.20)        # calibration distribution
        Xt, wt, gt = mixed(1800, 0.70)        # shifted deployment
        n_tr = len(wc) // 2
        model = core.IgnoranceModel().fit(Xc[:n_tr], wc[:n_tr])
        s_cal, w_cal, g_cal = model.score(Xc[n_tr:]), wc[n_tr:], gc[n_tr:]
        s_te = model.score(Xt)

        tau_marg = core.conformal_risk_threshold(s_cal, w_cal, alpha)
        tau_tuned = core.tuned_threshold(s_cal, w_cal, alpha)
        taus = core.group_conformal_thresholds(s_cal, w_cal, g_cal, alpha)
        gm = core.group_gate_metrics(s_te, wt, gt, taus)

        lt = gt == 1
        acc["marg_overall"].append(float(((s_te <= tau_marg) & wt).mean()))
        acc["marg_lt"].append(float(((s_te[lt] <= tau_marg) & wt[lt]).mean()))
        acc["tuned_overall"].append(float(((s_te <= tau_tuned) & wt).mean()))
        acc["tuned_lt"].append(float(((s_te[lt] <= tau_tuned) & wt[lt]).mean()))
        acc["grp_overall"].append(gm["overall"]["harmful_skip_rate"])
        acc["grp_lt"].append(gm["per_group"]["1"]["harmful_skip_rate"])
        acc["marg_retr"].append(float((s_te > tau_marg).mean()))
        acc["grp_retr"].append(gm["overall"]["retrieval_freq"])

    M = {k: float(np.mean(v)) for k, v in acc.items()}
    print("\n=== Experiment C: distribution shift + Mondrian (HEADLINE) ===")
    print(f"  alpha={alpha}; cal long-tail 0.20 -> deploy 0.70; mean over {n_draws} draws")
    print(f"  marginal conformal : E[harm] overall={M['marg_overall']:.3f} "
          f"long-tail={M['marg_lt']:.3f}  retrieval={M['marg_retr']:.3f}")
    print(f"  tuned threshold    : E[harm] overall={M['tuned_overall']:.3f} "
          f"long-tail={M['tuned_lt']:.3f}")
    print(f"  group conformal    : E[harm] overall={M['grp_overall']:.3f} "
          f"long-tail={M['grp_lt']:.3f}  retrieval={M['grp_retr']:.3f}")
    core.format_result("C_marginal_overall_harmful", round(M["marg_overall"], 3))
    core.format_result("C_group_overall_harmful", round(M["grp_overall"], 3))
    core.format_result("C_marginal_longtail_harmful", round(M["marg_lt"], 3))
    core.format_result("C_group_longtail_harmful", round(M["grp_lt"], 3))
    core.format_result("C_group_retrieval", round(M["grp_retr"], 3))
    core.format_result("C_marginal_retrieval", round(M["marg_retr"], 3))
    return M


def main():
    rng = np.random.default_rng(0)
    results = {}
    results["A_finite_sample"] = expt_A_finite_sample(rng)
    results["B_score_quality"] = expt_B_score_quality(rng)
    results["C_shift_mondrian"] = expt_C_shift_mondrian(rng)
    out = Path(__file__).resolve().parents[1] / "results" / "synth_validation.json"
    core.save_results(results, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
