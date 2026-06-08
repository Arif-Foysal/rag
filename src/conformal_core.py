"""
conformal-rag core library.

Uploaded as a Kaggle dataset by `pipeline.py upload-src`. All notebooks
import this module after locating it under /kaggle/input/.

Contents
--------
- Result IO helpers (format_result / save_results)
- QA correctness (normalize_answer, em_match, token_f1)
- Uncertainty from sampled answers (semantic_entropy, self_consistency)
- A multivariate "ignorance" model: P(model wrong | features)
- Conformal Risk Control threshold for the retrieve/skip gate
- Baselines: tuned-threshold gate, plug-in Bayesian gate
- Group-conditional (Mondrian) conformal control for distribution shift
- Repeated-split evaluation harness

Conventions
-----------
- `scores` : ignorance score s, HIGHER = more likely the model is WRONG.
- `wrong`  : boolean array, True = model answers incorrectly WITHOUT retrieval
             (i.e. skipping retrieval here is the harmful, hallucination-causing
             action).
- Decision : SKIP retrieval iff s <= tau ; RETRIEVE iff s > tau.
- Harmful skip : (s <= tau) AND wrong. We bound its marginal rate at alpha.
- Retrieval frequency : mean(s > tau). Lower is better at matched safety.

Torch is intentionally NOT imported here — keep this module importable for
pure-CPU analysis. LLM generation lives in the Kaggle notebook templates.
"""

import json
import re
import string
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result IO
# ---------------------------------------------------------------------------

def format_result(key, value, comment=None):
    """Print a greppable [RESULT] line and return the formatted string."""
    line = f"[RESULT] {key} = {value}"
    if comment:
        line += f"  # {comment}"
    print(line)
    return line


def save_results(data: dict, path):
    """Write a results dict as JSON. Use /kaggle/working/<phase>_results.json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Saved: {p}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---------------------------------------------------------------------------
# QA correctness (short-form)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/space."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def em_match(pred: str, golds) -> bool:
    """Exact match of normalized prediction against any gold alias."""
    if isinstance(golds, str):
        golds = [golds]
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def token_f1(pred: str, golds) -> float:
    """Max token-level F1 of prediction against any gold alias."""
    if isinstance(golds, str):
        golds = [golds]
    best = 0.0
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks and not g_toks:
            best = max(best, 1.0)
            continue
        if not p_toks or not g_toks:
            continue
        common = {}
        for t in p_toks:
            common[t] = min(p_toks.count(t), g_toks.count(t))
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(p_toks)
        rec = n_same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def is_correct(pred: str, golds, mode: str = "em", f1_thresh: float = 0.5) -> bool:
    """Boolean correctness used to derive `wrong = not is_correct`."""
    if mode == "em":
        return em_match(pred, golds)
    return token_f1(pred, golds) >= f1_thresh


# ---------------------------------------------------------------------------
# Uncertainty from a set of sampled answers
# ---------------------------------------------------------------------------

def _cluster_sizes(samples):
    """Cluster sampled answer strings by normalized-string equality (a cheap,
    discrete proxy for semantic clustering; documented limitation vs NLI-based
    semantic entropy). Returns a list of cluster sizes."""
    clusters = {}
    for s in samples:
        key = normalize_answer(s)
        clusters[key] = clusters.get(key, 0) + 1
    return list(clusters.values())


def semantic_entropy(samples) -> float:
    """Discrete semantic entropy (nats) over normalized answer clusters.
    High => the model disagrees with itself about the *answer* => likely doesn't
    know. 0 when all samples agree."""
    sizes = _cluster_sizes(samples)
    n = sum(sizes)
    if n == 0:
        return 0.0
    p = np.array(sizes, dtype=float) / n
    return float(-(p * np.log(p)).sum())


def self_consistency(samples) -> float:
    """Fraction of samples in the majority cluster. High => consistent => more
    likely correct. (Ignorance = 1 - self_consistency.)"""
    sizes = _cluster_sizes(samples)
    n = sum(sizes)
    return float(max(sizes) / n) if n else 0.0


# ---------------------------------------------------------------------------
# Multivariate ignorance model: P(model wrong | features)
# ---------------------------------------------------------------------------

class IgnoranceModel:
    """Logistic regression predicting P(wrong) from uncertainty features.

    Used as the nonconformity / ignorance *score* fed to the conformal gate.
    Standardizes features. Falls back gracefully on degenerate inputs.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.clf = None

    def fit(self, X, wrong):
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, dtype=float)
        y = np.asarray(wrong, dtype=int)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        # If only one class present, store the prior and skip the fit.
        if len(np.unique(y)) < 2:
            self._prior = float(y.mean())
            self.clf = None
        else:
            self.clf = LogisticRegression(max_iter=1000, C=1.0)
            self.clf.fit(Xs, y)
        return self

    def score(self, X):
        """Return P(wrong) in [0,1] — the ignorance score (higher = worse)."""
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        if self.clf is None:
            return np.full(X.shape[0], getattr(self, "_prior", 0.5))
        return self.clf.predict_proba(Xs)[:, 1]


# ---------------------------------------------------------------------------
# Conformal Risk Control for the retrieve/skip gate
# ---------------------------------------------------------------------------

def conformal_risk_threshold(scores, wrong, alpha, B=1.0):
    """Conformal Risk Control threshold (Angelopoulos et al. 2024) for the
    monotone 0/1 loss L(tau) = 1[skip & wrong], skip := s <= tau.

    Returns the LARGEST tau (=> most skipping => lowest retrieval) such that the
    finite-sample-corrected empirical risk satisfies
        (sum_i L_i(tau) + B) / (n + 1) <= alpha.
    Guarantee: E[L_test(tau_hat)] <= alpha under exchangeability.

    If even skipping nothing cannot be certified (alpha < B/(n+1)), returns
    -inf (retrieve everything; safe but no savings).
    """
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    n = len(scores)
    if n == 0:
        return -np.inf
    # Candidate thresholds: -inf (skip none) then each distinct score ascending.
    cands = np.concatenate([[-np.inf], np.unique(scores)])
    best = -np.inf
    for tau in cands:
        skip = scores <= tau
        risk_count = np.sum(skip & wrong)
        if (risk_count + B) / (n + 1) <= alpha:
            best = tau  # cands ascending => keep the largest feasible
        else:
            break  # risk is monotone non-decreasing in tau
    return float(best)


def gate_metrics(scores, wrong, tau):
    """Operating-point metrics for skip := s <= tau."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    n = len(scores)
    skip = scores <= tau
    harmful = skip & wrong
    n_skip = int(skip.sum())
    return {
        "n": n,
        "tau": float(tau),
        "harmful_skip_rate": float(harmful.mean()) if n else 0.0,
        "retrieval_freq": float((~skip).mean()) if n else 1.0,
        "skip_rate": float(skip.mean()) if n else 0.0,
        "cond_wrong_given_skip": float(wrong[skip].mean()) if n_skip else 0.0,
        "n_skip": n_skip,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def tuned_threshold(scores, wrong, alpha):
    """Candidate A: largest tau with *uncorrected* empirical risk <= alpha.
    No finite-sample guarantee — expected to violate alpha on test more often,
    especially at small n. Same score, same skip rule as conformal."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    n = len(scores)
    if n == 0:
        return -np.inf
    cands = np.concatenate([[-np.inf], np.unique(scores)])
    best = -np.inf
    for tau in cands:
        skip = scores <= tau
        if skip.sum() == 0:
            risk = 0.0
        else:
            risk = float((skip & wrong).mean())
        if risk <= alpha:
            best = tau
        else:
            break
    return float(best)


def plugin_bayes_threshold(scores, wrong, alpha):
    """Candidate B (plug-in): treat `scores` as a recalibrated P(wrong) and pick
    the threshold that hits the alpha budget on the calibration set with no
    conformal correction. With a single fitted score this is monotone-equivalent
    to `tuned_threshold`; the meaningful difference from conformal is the absence
    of a finite-sample / distribution-free guarantee (shows up under shift and at
    small n)."""
    return tuned_threshold(scores, wrong, alpha)


# ---------------------------------------------------------------------------
# Group-conditional (Mondrian) conformal control
# ---------------------------------------------------------------------------

def group_conformal_thresholds(scores, wrong, groups, alpha, B=1.0):
    """One CRC threshold per group. Yields group-conditional risk control
    E[L | group=g] <= alpha for each g — robust to shifts in group proportions
    (the dominant shift when long-tail queries grow at deployment)."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    groups = np.asarray(groups)
    taus = {}
    for g in np.unique(groups):
        m = groups == g
        taus[g] = conformal_risk_threshold(scores[m], wrong[m], alpha, B=B)
    return taus


def group_gate_metrics(scores, wrong, groups, taus):
    """Per-group + overall metrics when each group uses its own tau."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    groups = np.asarray(groups)
    skip = np.zeros(len(scores), dtype=bool)
    per_group = {}
    for g in np.unique(groups):
        m = groups == g
        tau = taus.get(g, -np.inf)
        skip[m] = scores[m] <= tau
        per_group[str(g)] = gate_metrics(scores[m], wrong[m], tau)
    harmful = skip & wrong
    overall = {
        "n": len(scores),
        "harmful_skip_rate": float(harmful.mean()) if len(scores) else 0.0,
        "retrieval_freq": float((~skip).mean()) if len(scores) else 1.0,
        "skip_rate": float(skip.mean()) if len(scores) else 0.0,
    }
    return {"overall": overall, "per_group": per_group}


# ---------------------------------------------------------------------------
# Repeated-split evaluation harness
# ---------------------------------------------------------------------------

def repeated_split_eval(features, wrong, alpha, n_splits=200, cal_frac=0.5,
                        groups=None, seed=0, feature_cols=None):
    """Fit ignorance model on a train split, calibrate threshold on a cal split,
    evaluate on a held-out test split. Repeat over random splits.

    Compares, on identical splits and identical fitted ignorance score:
      - conformal (marginal CRC)
      - tuned (uncorrected threshold)
      - group_conformal (Mondrian, requires `groups`)

    Returns per-method arrays of test harmful_skip_rate and retrieval_freq,
    plus (if groups given) per-group test harmful_skip_rate for the marginal
    methods (to expose the marginal-coverage gap).
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    wrong = np.asarray(wrong, dtype=bool)
    groups = None if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    n = len(wrong)

    out = {m: {"harmful": [], "retrieval": []}
           for m in ("conformal", "tuned", "group_conformal")}
    pergroup_harm = {}  # method -> group -> list

    for _ in range(n_splits):
        idx = rng.permutation(n)
        n_tr = n // 3
        n_cal = n // 3
        tr, cal, te = idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]

        model = IgnoranceModel().fit(X[tr], wrong[tr])
        s_cal, s_te = model.score(X[cal]), model.score(X[te])

        # marginal conformal
        tau_c = conformal_risk_threshold(s_cal, wrong[cal], alpha)
        mc = gate_metrics(s_te, wrong[te], tau_c)
        out["conformal"]["harmful"].append(mc["harmful_skip_rate"])
        out["conformal"]["retrieval"].append(mc["retrieval_freq"])

        # tuned (uncorrected)
        tau_t = tuned_threshold(s_cal, wrong[cal], alpha)
        mt = gate_metrics(s_te, wrong[te], tau_t)
        out["tuned"]["harmful"].append(mt["harmful_skip_rate"])
        out["tuned"]["retrieval"].append(mt["retrieval_freq"])

        if groups is not None:
            # group-conditional conformal
            taus = group_conformal_thresholds(s_cal, wrong[cal], groups[cal], alpha)
            gm = group_gate_metrics(s_te, wrong[te], groups[te], taus)
            out["group_conformal"]["harmful"].append(gm["overall"]["harmful_skip_rate"])
            out["group_conformal"]["retrieval"].append(gm["overall"]["retrieval_freq"])
            # marginal-conformal per-group harmful (exposes the gap)
            for g in np.unique(groups[te]):
                mm = groups[te] == g
                hk = float((( s_te[mm] <= tau_c) & wrong[te][mm]).mean())
                pergroup_harm.setdefault("conformal_marginal", {}).setdefault(str(g), []).append(hk)
                gg = gm["per_group"].get(str(g))
                if gg:
                    pergroup_harm.setdefault("group_conformal", {}).setdefault(str(g), []).append(gg["harmful_skip_rate"])

    summary = {}
    for m, d in out.items():
        if not d["harmful"]:
            continue
        h = np.array(d["harmful"]); r = np.array(d["retrieval"])
        summary[m] = {
            "harmful_mean": float(h.mean()),
            "harmful_p95": float(np.quantile(h, 0.95)),
            "violation_rate": float((h > alpha).mean()),  # fraction of splits over budget
            "retrieval_mean": float(r.mean()),
        }
    pergroup_summary = {}
    for m, gd in pergroup_harm.items():
        pergroup_summary[m] = {g: {"harmful_mean": float(np.mean(v)),
                                    "violation_rate": float((np.array(v) > alpha).mean())}
                               for g, v in gd.items()}
    return {"alpha": alpha, "n": n, "methods": summary, "per_group": pergroup_summary}


# ---------------------------------------------------------------------------
# Project-specific code goes below.
# ---------------------------------------------------------------------------
