"""
Conformal gate analysis on REAL PopQA LLM signals (phase2 output).

Consumes results/<...>/phase2_records.json and runs the same A/B/C battery as
the synthetic study, plus a coverage-vs-alpha curve, on real model behavior:
  A. finite-sample control (conformal vs tuned), iid splits
  B. score quality: which single ignorance signal gives lowest retrieval at
     matched safety (semantic_entropy vs 1-self_consistency vs neg_logprob)
  C. HEADLINE: popularity-shift (calibrate popular-heavy, deploy long-tail-heavy)
     -> marginal vs group-conditional (Mondrian) conformal
  + coverage-vs-alpha curve for the multivariate conformal gate.

Run: ./.venv/bin/python analysis/real_analysis.py [records.json]
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


def load_records(path=None):
    if path is None:
        cands = glob.glob(str(ROOT / "results" / "*phase2*" / "phase2_records.json"))
        if not cands:
            cands = glob.glob(str(ROOT / "results" / "**" / "phase2_records.json"),
                              recursive=True)
        if not cands:
            raise FileNotFoundError("phase2_records.json not found under results/")
        path = sorted(cands)[-1]
    print(f"  loading {path}")
    with open(path) as f:
        data = json.load(f)
    return data["records"] if isinstance(data, dict) else data


def to_arrays(records):
    X = np.array([[r[f] for f in FEATURES] for r in records], float)
    wrong = np.array([bool(r["wrong"]) for r in records])
    logpop = np.array([r["log_pop"] for r in records], float)
    # 3 popularity buckets: 0=long-tail, 1=mid, 2=popular
    q = np.quantile(logpop, [1/3, 2/3])
    bucket = np.digitize(logpop, q)
    return X, wrong, logpop, bucket


def auroc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    return float((ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))


def expt_A(X, wrong, alpha=0.10):
    res = core.repeated_split_eval(X, wrong, alpha=alpha, n_splits=400, seed=1)
    c, t = res["methods"]["conformal"], res["methods"]["tuned"]
    print("\n=== A: finite-sample control (real, iid) ===")
    print(f"  conformal E[harm]={c['harmful_mean']:.3f} retr={c['retrieval_mean']:.3f} | "
          f"tuned E[harm]={t['harmful_mean']:.3f} retr={t['retrieval_mean']:.3f}")
    core.format_result("realA_conformal_Eharm", round(c["harmful_mean"], 3))
    core.format_result("realA_tuned_Eharm", round(t["harmful_mean"], 3))
    core.format_result("realA_conformal_retrieval", round(c["retrieval_mean"], 3))
    return {"conformal": c, "tuned": t}


def expt_B(records, wrong, alpha=0.10):
    print("\n=== B: score quality -> retrieval at matched safety (real) ===")
    out = {}
    for f in ["semantic_entropy", "ignorance_1msc", "neg_mean_logprob"]:
        col = np.array([r[f] for r in records], float)[:, None]
        r = core.repeated_split_eval(col, wrong, alpha=alpha, n_splits=300, seed=2)
        m = r["methods"]["conformal"]
        out[f] = m
        print(f"  {f:18s}: E[harm]={m['harmful_mean']:.3f}  retrieval={m['retrieval_mean']:.3f}  "
              f"AUROC={auroc(col[:,0], wrong):.3f}")
        core.format_result(f"realB_retrieval_{f}", round(m["retrieval_mean"], 3))
    return out


def expt_C(X, wrong, bucket, alpha=0.10, n_draws=200, seed=0):
    """Popularity shift: calibrate popular-heavy, deploy long-tail-heavy.
    Resample with replacement from bucket pools to set the mix."""
    rng = np.random.default_rng(seed)
    pools = {b: np.where(bucket == b)[0] for b in np.unique(bucket)}
    bvals = sorted(pools)            # 0=long-tail ... last=popular
    lt, pop = bvals[0], bvals[-1]

    def draw(n, frac_lt):
        n_lt = int(n * frac_lt); n_pop = n - n_lt
        i_lt = rng.choice(pools[lt], n_lt, replace=True)
        i_pop = rng.choice(pools[pop], n_pop, replace=True)
        idx = np.concatenate([i_lt, i_pop])
        g = np.concatenate([np.zeros(n_lt, int), np.ones(n_pop, int)])  # 0=lt,1=pop
        perm = rng.permutation(len(idx))
        return idx[perm], g[perm]

    acc = {k: [] for k in ("marg_overall","marg_lt","grp_overall","grp_lt",
                            "marg_retr","grp_retr")}
    for _ in range(n_draws):
        ci, cg = draw(1200, 0.20)    # calibration: popular-heavy
        ti, tg = draw(1200, 0.70)    # deploy: long-tail-heavy
        # fit ignorance model on half of calibration, calibrate threshold on rest
        h = len(ci) // 2
        model = core.IgnoranceModel().fit(X[ci[:h]], wrong[ci[:h]])
        s_cal, w_cal, g_cal = model.score(X[ci[h:]]), wrong[ci[h:]], cg[h:]
        s_te, w_te, g_te = model.score(X[ti]), wrong[ti], tg

        tau_m = core.conformal_risk_threshold(s_cal, w_cal, alpha)
        taus = core.group_conformal_thresholds(s_cal, w_cal, g_cal, alpha)
        gm = core.group_gate_metrics(s_te, w_te, g_te, taus)
        ltm = g_te == 0
        acc["marg_overall"].append(float(((s_te <= tau_m) & w_te).mean()))
        acc["marg_lt"].append(float(((s_te[ltm] <= tau_m) & w_te[ltm]).mean()))
        acc["marg_retr"].append(float((s_te > tau_m).mean()))
        acc["grp_overall"].append(gm["overall"]["harmful_skip_rate"])
        acc["grp_lt"].append(gm["per_group"]["0"]["harmful_skip_rate"])
        acc["grp_retr"].append(gm["overall"]["retrieval_freq"])
    M = {k: float(np.mean(v)) for k, v in acc.items()}
    print("\n=== C: popularity shift + Mondrian (real, HEADLINE) ===")
    print(f"  marginal : E[harm] overall={M['marg_overall']:.3f} long-tail={M['marg_lt']:.3f} "
          f"retr={M['marg_retr']:.3f}")
    print(f"  group    : E[harm] overall={M['grp_overall']:.3f} long-tail={M['grp_lt']:.3f} "
          f"retr={M['grp_retr']:.3f}")
    for k, v in M.items():
        core.format_result(f"realC_{k}", round(v, 3))
    return M


def coverage_curve(X, wrong, alphas=(0.02,0.05,0.10,0.15,0.20)):
    print("\n=== coverage vs alpha (multivariate conformal) ===")
    rows = {}
    for a in alphas:
        res = core.repeated_split_eval(X, wrong, alpha=a, n_splits=300, seed=3)
        c = res["methods"]["conformal"]
        rows[a] = c
        print(f"  alpha={a:.2f}: E[harm]={c['harmful_mean']:.3f}  retrieval={c['retrieval_mean']:.3f}")
        core.format_result(f"cov_alpha_{a}_Eharm", round(c["harmful_mean"], 3))
        core.format_result(f"cov_alpha_{a}_retrieval", round(c["retrieval_mean"], 3))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    records = load_records(path)
    X, wrong, logpop, bucket = to_arrays(records)
    print(f"  n={len(records)}  accuracy={1-wrong.mean():.3f}  "
          f"buckets={np.bincount(bucket).tolist()}")
    core.format_result("real_n", len(records))
    core.format_result("real_accuracy", round(float(1 - wrong.mean()), 4))
    for f in FEATURES:
        col = np.array([r[f] for r in records], float)
        core.format_result(f"auroc_{f}", round(auroc(col, wrong), 4))

    results = {
        "n": len(records), "accuracy": float(1 - wrong.mean()),
        "A": expt_A(X, wrong),
        "B": expt_B(records, wrong),
        "C": expt_C(X, wrong, bucket),
        "coverage": coverage_curve(X, wrong),
    }
    core.save_results(results, ROOT / "results" / "real_analysis.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
