"""
End-to-end gate evaluation on PopQA: does the conformal gate deliver near
full-RAG accuracy at reduced retrieval cost?

Fuses phase2 (closed-book signals + correctness) with phase4 (RAG correctness)
on the SAME questions. For each query the gate chooses:
    skip  -> outcome = closed-book correctness
    retrieve -> outcome = RAG correctness
We report the accuracy-vs-retrieval frontier and key operating points.

Mechanism stats:
    retrieval FIX rate     = P(RAG correct | closed-book wrong)
    context POISON rate    = P(RAG wrong  | closed-book correct)

Baselines: never-retrieve, always-retrieve, oracle (retrieve iff cb wrong),
random-skip frontier, and the score-based conformal gate.

Run: ./.venv/bin/python analysis/endtoend.py
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
    c = glob.glob(str(ROOT / "results" / "**" / pat), recursive=True)
    if not c:
        raise FileNotFoundError(pat)
    with open(sorted(c)[-1]) as f:
        d = json.load(f)
    return d["records"] if isinstance(d, dict) else d


def fuse():
    cb = _load("phase2_records.json")            # closed-book (PopQA, Qwen)
    rag = _load("phase4_rag_records.json")       # RAG outcomes (same questions)
    rag_by_q = {r["question"]: r for r in rag}
    X, cb_correct, rag_correct, feats = [], [], [], []
    for r in cb:
        m = rag_by_q.get(r["question"])
        if m is None:
            continue
        X.append([r[f] for f in FEATURES])
        cb_correct.append(not r["wrong"])
        rag_correct.append(bool(m["rag_correct"]))
        feats.append(r)
    return (np.array(X, float), np.array(cb_correct, bool),
            np.array(rag_correct, bool), feats)


def gate_accuracy(skip, cb_correct, rag_correct):
    """outcome = cb on skipped, rag on retrieved."""
    out = np.where(skip, cb_correct, rag_correct)
    return float(out.mean()), float((~skip).mean())  # accuracy, retrieval fraction


def conformal_points(X, cb_correct, rag_correct, alphas, n_splits=300, seed=1):
    """Score-based gate; tau from CRC bounding harmful-skip at alpha (cb wrong)."""
    rng = np.random.default_rng(seed)
    cb_wrong = ~cb_correct
    n = len(cb_correct)
    rows = {a: {"acc": [], "retr": []} for a in alphas}
    for _ in range(n_splits):
        idx = rng.permutation(n); h = n // 2
        cal, te = idx[:h], idx[h:]
        model = core.IgnoranceModel().fit(X[cal], cb_wrong[cal])
        s_cal, s_te = model.score(X[cal]), model.score(X[te])
        for a in alphas:
            tau = core.conformal_risk_threshold(s_cal, cb_wrong[cal], a)
            skip = s_te <= tau
            acc, retr = gate_accuracy(skip, cb_correct[te], rag_correct[te])
            rows[a]["acc"].append(acc); rows[a]["retr"].append(retr)
    return {a: {"acc": float(np.mean(v["acc"])), "retr": float(np.mean(v["retr"]))}
            for a, v in rows.items()}


def score_frontier(X, cb_correct, rag_correct, n_splits=200, seed=2):
    """Sweep skip fraction by score percentile -> achievable acc/retrieval curve,
    vs random-skip at the same retrieval fraction."""
    rng = np.random.default_rng(seed)
    cb_wrong = ~cb_correct
    fracs = np.linspace(0, 0.6, 13)   # skip fraction
    sc = {f: [] for f in fracs}; rnd = {f: [] for f in fracs}
    n = len(cb_correct)
    for _ in range(n_splits):
        idx = rng.permutation(n); h = n // 2
        cal, te = idx[:h], idx[h:]
        model = core.IgnoranceModel().fit(X[cal], cb_wrong[cal])
        s_te = model.score(X[te])
        order = np.argsort(s_te)                # ascending ignorance -> skip lowest
        for f in fracs:
            k = int(f * len(te))
            skip = np.zeros(len(te), bool)
            if k > 0:
                skip[order[:k]] = True
            acc, _ = gate_accuracy(skip, cb_correct[te], rag_correct[te])
            sc[f].append(acc)
            # random skip baseline at same fraction
            rsk = np.zeros(len(te), bool)
            rsk[rng.choice(len(te), k, replace=False)] = True
            racc, _ = gate_accuracy(rsk, cb_correct[te], rag_correct[te])
            rnd[f].append(racc)
    return {float(f): {"score": float(np.mean(sc[f])), "random": float(np.mean(rnd[f]))}
            for f in fracs}


def main():
    X, cb_correct, rag_correct, feats = fuse()
    n = len(cb_correct)
    cb_acc = float(cb_correct.mean()); rag_acc = float(rag_correct.mean())
    fix = float(rag_correct[~cb_correct].mean()) if (~cb_correct).any() else float("nan")
    poison = float((~rag_correct[cb_correct]).mean()) if cb_correct.any() else float("nan")
    oracle_acc = float((cb_correct | (~cb_correct & rag_correct)).mean())
    oracle_retr = float((~cb_correct).mean())

    print(f"  fused n={n}")
    print(f"  never-retrieve  acc={cb_acc:.3f}  (retrieval 0)")
    print(f"  always-retrieve acc={rag_acc:.3f}  (retrieval 1)")
    print(f"  oracle gate     acc={oracle_acc:.3f}  (retrieval {oracle_retr:.3f})")
    print(f"  retrieval FIX rate  P(rag ok | cb wrong)   = {fix:.3f}")
    print(f"  context POISON rate P(rag wrong | cb ok)   = {poison:.3f}")
    for k, v in [("ete_n", n), ("ete_neverretrieve_acc", round(cb_acc, 3)),
                 ("ete_alwaysretrieve_acc", round(rag_acc, 3)),
                 ("ete_oracle_acc", round(oracle_acc, 3)),
                 ("ete_fix_rate", round(fix, 3)), ("ete_poison_rate", round(poison, 3))]:
        core.format_result(k, v)

    print("\n  conformal gate operating points:")
    pts = conformal_points(X, cb_correct, rag_correct, alphas=(0.05, 0.10, 0.15, 0.20))
    for a, m in pts.items():
        saved = 1 - m["retr"]
        print(f"    alpha={a:.2f}: end-to-end acc={m['acc']:.3f}  retrieval={m['retr']:.3f}  "
              f"(saved {saved:.0%} vs always)  d_vs_always={m['acc']-rag_acc:+.3f}")
        core.format_result(f"ete_a{a}_acc", round(m["acc"], 3))
        core.format_result(f"ete_a{a}_retr", round(m["retr"], 3))

    print("\n  score-gate vs random-skip frontier (accuracy at skip fraction):")
    fr = score_frontier(X, cb_correct, rag_correct)
    for f in sorted(fr):
        print(f"    skip={f:.2f} (retrieval={1-f:.2f}): score-gate acc={fr[f]['score']:.3f} | "
              f"random acc={fr[f]['random']:.3f}")

    core.save_results({"n": n, "never": cb_acc, "always": rag_acc, "oracle": oracle_acc,
                       "fix_rate": fix, "poison_rate": poison,
                       "conformal_points": pts, "frontier": fr},
                      ROOT / "results" / "endtoend.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
