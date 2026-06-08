# Calibrating "Retrieve or Not" as a Risk-Controlled Decision Under Asymmetric Cost

*A problem-first proposal for adaptive Retrieval-Augmented Generation*

---

## Contributions

This proposal makes four claims, in increasing order of risk:

1. **Reframing.** The retrieve/skip gate in adaptive RAG is not symmetric binary classification; it is a *selective-prediction decision under asymmetric, query-dependent cost*, and should be optimized as one.
2. **A bound where the field has only had measurements.** We place a **finite-sample, distribution-free upper bound** on the safety-critical error — skipping retrieval on a question the model gets wrong — by conformalizing the *gate decision itself*, before any retrieval happens. To our knowledge no adaptive-RAG gate carries such a guarantee.
3. **A non-decorative nonconformity score.** We commit to a specific ignorance signal (semantic entropy + self-consistency dispersion) and argue it captures epistemic uncertainty that raw token-confidence does not — the hinge on which the whole contribution turns.
4. **Robustness to the regime that breaks naïve conformal prediction.** Because the motivating queries are precisely those under distribution shift, we make **group-conditional (Mondrian) risk control by query novelty** the core method, not an afterthought — recovering a guarantee exactly where the harm concentrates.

The honest framing: contribution (2) is textbook once (3) and (4) hold, and is worthless if they don't. The proposal lives or dies on the score and the shift.

---

## The Problem

Adaptive RAG must decide whether to retrieve at all before generating an answer. This decision is consequential in both directions:

- **Retrieving when unnecessary** — on a question the model already answers correctly from parametric knowledge — wastes latency and, worse, can inject distracting or wrong context that *degrades* an answer the model would otherwise have gotten right (context poisoning).
- **Not retrieving when necessary** — on a question the model does not know — causes hallucination.

Current retrieve/skip gates — a learned classifier (Self-RAG; Asai et al. 2023), a heuristic on generation confidence (FLARE; Jiang et al. 2023), or a complexity router (Adaptive-RAG; Jeong et al. 2024) — treat this as symmetric binary classification. But the two errors are not equally costly, and their relative cost is *query-dependent*. A single global threshold tuned to average behavior cannot adapt to where the costs actually sit for a given query, and it offers **no guarantee** on the failure mode we care about most: silent hallucination on questions the model didn't know it didn't know.

This is a concrete, reproducible failure: existing gates fire or skip on uncalibrated signals, and their error rate on the safety-critical direction is only ever measured empirically, never bounded.

---

## Structural Diagnosis

*(stated before naming any mathematical field)*

This is a **decision under uncertainty** where:

1. The value of an action (retrieve / skip) depends on an **unobserved variable** — does the model actually know this fact? We never observe this at decision time; we see only proxy signals.
2. The two failure types carry **asymmetric cost**: an unnecessary retrieval costs latency and some risk of context-poisoning; a missed retrieval costs a confident falsehood. The asymmetry shifts per query.
3. The right behavior is therefore **"act only when expected benefit exceeds expected cost,"** with a decision boundary that should move per query rather than sit at a fixed global threshold.

**Best guess at the underlying cause of current failure:** the gate is trained to predict a *binary label* (was retrieval helpful in the training data?) rather than to *control a cost-weighted risk*. A classifier minimizing label error implicitly assumes symmetric costs and a stationary threshold — exactly the two things this problem violates. The decision boundary is placed by the wrong objective.

The shape, in one line: **bound the cost of the worst error type while minimizing the frequency of the cheap-but-annoying one.**

---

## Candidate Approaches & The Choice

The structure invites at least three mathematically distinct treatments. I steelman each, because the choice only matters if the alternatives are presented at their strongest.

### Candidate A — Elementary: tuned-threshold confidence gate
A logistic-regression (or single-threshold) gate over confidence features, threshold tuned on validation. This is the current standard and the honest baseline. Cheap, simple, sometimes hard to beat. Its weakness is the diagnosis above: one global threshold, symmetric objective, no guarantee on the asymmetric failure.

### Candidate B — Bayesian decision theory done *competently*
Estimate a posterior $P(\text{correct}\mid x)$, **recalibrate it** (temperature/Platt scaling on held-out data), specify an asymmetric loss matrix $L$, and retrieve iff expected cost of skipping exceeds expected cost of retrieving:

$$\text{retrieve} \iff \mathbb{E}[L_{\text{skip}}\mid x] > \mathbb{E}[L_{\text{retrieve}}\mid x].$$

This is the textbook-correct formulation and genuinely good *if the recalibrated posterior is trustworthy point-wise*. That is the fatal "if." Recalibration fixes *average* calibration (ECE) but provides **no finite-sample guarantee on any chosen error direction**, and post-RLHF LLM confidence is known to be unreliable on out-of-distribution and long-tail inputs (Kadavath et al. 2022; OpenAI 2023) — exactly the inputs that matter here. The machinery is sound; its guarantee is asymptotic and average-case, not finite-sample and direction-specific.

> **B is not a strawman — it is the baseline to beat.** A competent recalibrated Bayesian gate is included in the experiments. If we cannot beat it on bounded safety at matched efficiency, the contribution fails, and we will report that.

### Candidate C — Conformal risk control
Produce a calibrated decision rule whose risk on the chosen direction is **bounded with finite-sample, distribution-free guarantees**, *without assuming the underlying scores are well-calibrated*. We define a nonconformity ("ignorance") score, choose a threshold by conformal risk control on held-out data, and obtain a guarantee that the gate's harmful-skip rate is $\le \alpha$ for a user-chosen $\alpha$.

### The Choice: **Conformal Risk Control (Angelopoulos et al. 2024), group-conditional by query novelty.**

Candidate B's guarantee is only as good as the posterior, and we have strong evidence that posterior is unreliable precisely on long-tail inputs. Building the decision on a recalibrated-but-still-asymptotic estimate reintroduces the original failure under a more sophisticated name.

Conformal risk control's decisive advantage: it converts an *uncalibrated* score into a decision rule with a real, finite-sample risk bound, requiring only **exchangeability** of calibration and test data — not that the score means what it claims. It lets us encode the **asymmetry directly**: bound the harmful-skip rate at $\alpha$, then minimize unnecessary retrievals subject to that bound. (We use CRC rather than vanilla split-conformal because our target is a *tunable risk* on a monotone loss, not set coverage — CRC is the exact tool; LTT, Angelopoulos et al. 2021, is the fallback if we control multiple risks jointly.)

---

## Method

### The decision and its loss

Let $s(x)\in\mathbb{R}$ be an **ignorance score**: high $s$ ⇒ the model is likely wrong *without* retrieval. Decision rule, parameterized by threshold $\tau$:

$$\textbf{skip retrieval} \iff s(x) < \tau.$$

Define the harmful event and its loss:

$$L_i(\tau) = \mathbb{1}\big[\,s(x_i) < \tau \;\wedge\; \text{model is wrong on } x_i \text{ without retrieval}\,\big].$$

$L_i(\tau)$ is **monotone non-decreasing in $\tau$** (raising $\tau$ only skips more), which is exactly the structure CRC requires. We bound the **marginal harmful-skip rate** $\mathbb{E}[L(\tau)]\le\alpha$. (We will also report the conditional $P(\text{wrong}\mid\text{skip})$; a reviewer will ask, and the two answer different operational questions — joint = "what fraction of all traffic is a harmful skip," conditional = "if I skipped, how often did I err.")

### The risk-control procedure and guarantee

Given calibration set $\{(x_i, \text{wrong}_i)\}_{i=1}^n$ (exchangeable with test), choose

$$\hat\tau = \sup\Big\{\tau : \widehat R_n(\tau) + \tfrac{B}{n} \le \alpha\Big\}, \qquad \widehat R_n(\tau)=\tfrac1n\sum_i L_i(\tau),$$

where $B$ is the loss bound ($=1$ here). This is the largest $\tau$ — i.e., the *most aggressive skipping*, hence the *lowest retrieval frequency* — that still satisfies the risk budget.

> **Proposition (CRC; Angelopoulos, Bates, Fisch, Lei, Schuster 2024).** If $(x_1,\dots,x_n,x_{\text{test}})$ are exchangeable and $L(\tau)$ is monotone and bounded by $B$, then $\mathbb{E}\big[L_{\text{test}}(\hat\tau)\big]\le\alpha$, where the expectation is over the calibration draw and the test point.

The guarantee is finite-sample, distribution-free, and makes **no assumption that $s$ is calibrated** — $s$ can be arbitrarily bad and the bound still holds; a bad $s$ merely forces $\hat\tau$ low and retrieval frequency high (you pay in efficiency, never in safety). That property is what makes the safety claim robust.

### The nonconformity score — the hinge

The entire contribution collapses if $s$ is repackaged token-confidence. We therefore commit to a score designed to capture *epistemic* uncertainty (does the model know?) rather than *aleatoric/lexical* uncertainty (surface phrasing):

- **Semantic entropy** (Kuhn, Gal, Farquhar 2023; Farquhar et al. 2024, *Nature*): sample $k$ generations, cluster by bidirectional entailment, compute entropy over *meaning* clusters. High when the model is internally inconsistent about the *answer*, not merely the wording.
- **Self-consistency dispersion**: agreement rate of the majority semantic cluster across the $k$ samples.
- (Reported but expected weak) **P(True) self-evaluation** (Kadavath et al. 2022) and raw sequence log-prob, as ablations to demonstrate the gap.

**Hypothesis under test:** semantic entropy separates "model knows" from "model is bluffing fluently" in a way raw confidence does not, and this separation is what conformal control monetizes into lower retrieval at fixed safety. If the ablation shows raw log-prob does just as well, contribution (3) is falsified and we say so.

### Calibration labels

`wrong_i` requires a ground-truth correctness judgment per calibration query, with **no retrieval**. Protocol:
- Short-form QA (TriviaQA, PopQA, Natural Questions): normalized exact-match / token-F1 against gold answers — high-reliability labels.
- **Judge-sensitivity check:** re-derive `wrong_i` with an LLM judge and report how much $\hat\tau$ and the realized risk move. If the guarantee is fragile to label noise, that is a finding.

### Handling the regime that breaks naïve conformal prediction

The motivating queries are *fresh and long-tail* — i.e., the deployment distribution differs from calibration, which **violates exchangeability** and voids the marginal guarantee. This is the proposal's central technical risk, so we confront it in the method, not the appendix:

- **Mondrian / group-conditional risk control.** Partition queries into novelty buckets via a cheap, retrieval-free estimator (e.g., entity-frequency / time-stamp / a novelty probe), and run CRC *within each bucket*. This yields **group-conditional** risk control: $\mathbb{E}[L\mid \text{bucket } g]\le\alpha$ for each $g$, which is *robust to shifts in the proportions of buckets* — the dominant form of shift here (more long-tail queries at test time). This simultaneously dissolves the "marginal-coverage hides long-tail failure" problem.
- **Weighted conformal** (covariate-shift correction) as a secondary route when a likelihood ratio is estimable.

This is the difference between "conformal prediction, which happens to break on the cases we care about" and "conformal prediction *engineered for* the cases we care about."

---

## Why It's Novel

**Closest existing work, and the precise gap:**

- *Adaptive RAG gates* (FLARE, Self-RAG, Adaptive-RAG): learned/heuristic retrieve-skip with **no guarantee** on either error direction.
- *Conformal RAG*: applies conformal prediction to the **output** — which retrieved passages or candidate answers to trust, with coverage over answer/evidence sets.
- *Conformal abstention / selective generation* (Quach et al. 2024; conformal-abstention work): structurally the nearest relative — abstain-vs-answer is the same shape as skip-vs-retrieve. **The difference:** abstention's recourse is silence; our recourse is a *corrective action* (retrieve) with its own asymmetric cost, so we control a *two-action* decision and explicitly trade the bounded safety direction against the unbounded efficiency direction.

**The structural difference:** prior conformal-RAG conformalizes the *answer*; we conformalize the *control decision* — a finite-sample bound on "skip-when-needed" failures *before* retrieval happens. The guarantee attaches to the gate, not the content.

**The assumption it breaks:** that the retrieve/skip gate can only be *evaluated* empirically, never *bounded*. We assert it can be bounded — distribution-free, finite-sample, group-conditional — in the safety-critical direction.

---

## Experimental Plan

**Models.** At least two open-weight LLMs of different scale (e.g., an 8B and a 70B-class model) to show the method is not tuned to one model's calibration profile.

**Data — a mixed pool with controlled novelty.**
- *Parametric-knowledge* (model usually knows): TriviaQA, PopQA (high-popularity slice), Natural Questions.
- *Long-tail / fresh* (model usually does not know): PopQA rare-entity slice, recent-events QA, RealTimeQA-style time-stamped questions.
- Novelty buckets defined by entity frequency and timestamp for the Mondrian split.

**Baselines.**
1. Self-RAG native gate.
2. FLARE confidence-trigger.
3. Tuned-threshold confidence classifier (Candidate A).
4. **Recalibrated Bayesian gate with asymmetric loss (Candidate B done competently)** — the key comparison.
5. Always-retrieve and never-retrieve (the trivial endpoints).

**Metrics.**
- *Realized harmful-skip rate* — must stay $\le\alpha$, reported **marginally and per novelty bucket** (the latter is the real test).
- *Retrieval frequency* — lower is better at matched safety; the operational payoff.
- *End-to-end answer accuracy / F1* and *latency* — because the goal is correct, fast answers, not just a satisfied bound.
- *Coverage-vs-$\alpha$ curve* over $\alpha\in\{1,5,10,20\}\%$ — does the guarantee track the budget across the range?
- *Variance over $\ge 100$ random calibration/test splits* — the guarantee is in expectation over the split; show the distribution, not one lucky draw.

**Success criterion (pre-registered).** At matched realized harmful-skip rate, conformal control achieves a retrieval-frequency reduction over the *best* baseline (expected to be Candidate B) of a margin that is statistically significant across splits — *and* maintains $\le\alpha$ harmful-skip rate **within every novelty bucket**, where the best baseline does not.

---

## The Experiment That Could Kill It

**Setup.** The mixed pool above, $\alpha=5\%$.

**The outcome that proves me wrong.** If conformal control **cannot reduce retrieval frequency below the recalibrated-Bayesian gate (B) at matched harmful-skip rate**, then the guarantee bought nothing operationally useful — we paid for distribution-free safety and got no efficiency, and a competently-calibrated classifier dominates in practice.

**Why the result is informative either way.** Success shows calibration-free guarantees translate into real efficiency gains and should inform adaptive-RAG design. Failure teaches the field something sharp: that distribution-free guarantees on the gate, while theoretically clean, do not beat good recalibration on this problem — redirecting effort toward fixing Candidate B's input rather than replacing its framework.

---

## Threats to Validity / Honest Failure Modes

1. **Decorative collapse.** If the ablation shows raw log-prob matches semantic entropy as $s$, the novelty is gone and conformal becomes a fancy threshold-picker. *Test:* the score ablation directly measures this; we report it whether or not it favors us.
2. **Exchangeability violation under shift** *(the big one).* If the deployment distribution differs from calibration in ways the novelty buckets don't capture, even group-conditional CRC degrades. *Mitigation:* Mondrian by novelty + weighted conformal; *residual risk:* unmodeled shift dimensions, which we will probe with held-out time-split tests.
3. **Label noise in `wrong_i`.** Correctness labels are imperfect; a biased judge biases $\hat\tau$. *Test:* judge-sensitivity analysis on $\hat\tau$ and realized risk.
4. **Joint vs. conditional bound mismatch.** Bounding the marginal $P(\text{skip}\wedge\text{wrong})$ at 5% can still permit a high $P(\text{wrong}\mid\text{skip})$ if skipping is rare. We report both and let the operational target dictate which to control.
5. **Cost of the corrective action is itself stochastic.** Retrieval can poison context; the "cost of retrieve" is not constant. We *measure* the context-poisoning rate rather than assuming it, and feed it into Candidate B's loss matrix for a fair fight.

---

## One-Line Summary

Turn the retrieve/skip gate from an empirically-tuned classifier into a decision with a **distribution-free, finite-sample, group-conditional bound** on its safety-critical error — by conformalizing the *gate*, not the answer, over a semantic-uncertainty score — and justify it precisely because the Bayesian alternative's guarantee is asymptotic and average-case on a posterior that fails on the long-tail queries that matter.

---

## References *(indicative)*

- Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.* ICLR 2024.
- Angelopoulos, Bates, et al. *Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021.
- Angelopoulos & Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2023.
- Asai et al. *Self-RAG.* ICLR 2024.
- Jiang et al. *Active Retrieval-Augmented Generation (FLARE).* EMNLP 2023.
- Jeong et al. *Adaptive-RAG.* NAACL 2024.
- Kuhn, Gal, Farquhar. *Semantic Uncertainty.* ICLR 2023. / Farquhar et al. *Detecting hallucinations using semantic entropy.* Nature 2024.
- Kadavath et al. *Language Models (Mostly) Know What They Know.* 2022.
- Quach et al. *Conformal Language Modeling.* ICLR 2024.
- Vovk et al. *Algorithmic Learning in a Random World* (Mondrian conformal prediction).
