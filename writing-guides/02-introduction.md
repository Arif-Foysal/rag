# Introduction (with Merged Literature Review)

## Purpose

The introduction must do three things in exactly this order:
1. Establish that the problem is real and important (Paragraph 1)
2. Show that prior work is genuinely insufficient — not just incomplete (Paragraph 2)
3. Make the reader believe your contribution resolves the gaps (Paragraph 3)

When there is no separate Related Work section, the literature review lives inside Paragraph 2 of the Introduction. This is the structure described here.

---

## Paragraph 1 — The Problem

**Four things to cover (in this order):**
1. What is currently happening in this field? (cite 3–4 recent papers by name)
2. What are the research gaps or unsolved challenges? (be specific — a technical gap, not "the problem is complex")
3. What do you want to do, and what is your high-level approach?
4. What is the scope of your dataset and proposed methods?

**Pattern:**
> [What currently exists, with 3–4 refs] → [Specific gap or limitation] → [Your goal and approach] → [Scope: domain / dataset / method type]

**Example:**
> "Local explanation methods such as KernelSHAP [1], LIME [2], Integrated Gradients [3], and SmoothGrad [4] sample inputs to construct a feature importance vector for any network prediction. These methods have no certified region of validity — they return an explanation without specifying how large an input neighbourhood it covers. We propose a certification framework that exposes the exact region on which a piecewise-linear network is affine, enabling auditing of any local explanation method. The framework applies to feedforward ReLU networks on tabular and image inputs."

**Rules:**
- Name each cited paper for what it does, not just as "[3]"
- The gap must be a specific technical gap: "no closed-form region", "no statistical significance test", "no guarantee under class imbalance" — not "the problem is challenging"
- Do not claim "we are the first" in Paragraph 1 unless you prove it in the experiments
- Scope sentence must be concrete: name the dataset, architecture, or domain

---

## Paragraph 2 — Literature Review (5–6 Papers)

**What to cover:**
For each paper: exactly 1–2 sentences.
- Sentence 1: What the paper proposes and what it achieves (method + result)
- Sentence 2: What it lacks — the gap that is directly relevant to YOUR contribution

**Per-Paper Formula:**
> "[Authors (Year)] [strong verb] [method], which [achieves / shows / proves X]. However, [specific limitation relevant to your work]."

**Example:**
> "Ribeiro, Singh and Guestrin (2016) introduce LIME, a sparse linear surrogate fitted to network outputs at perturbed inputs weighted by proximity. LIME provides no validity certificate — its explanation can change arbitrarily for inputs just outside the sampled region."

**Grouping rule:** Do not list papers chronologically. Group them by theme (approach type, problem axis, method family). Use a transition sentence at the start of each theme group:
> "A separate line of work addresses stability rather than validity..."

**Bridging sentence:** The last sentence of Paragraph 2 must connect the literature to your solution:
> "None of these works expose the maximal input-space region on which the network is exactly affine; we address this directly."

**Differentiation verbs for the bridging sentence:**
- "Unlike prior work, we..."
- "No existing method [does X]; we address this by..."
- "The present paper is the first to derive [X] in a single [Y] pass."
- "This paper departs from [approach type] by..."

**Rules:**
- Every cited paper's gap should motivate a different aspect of your contribution
- Avoid "also", "additionally", "furthermore" as the only connective — use thematic transitions
- Do not describe your own method here — save it for Paragraph 3 and Section 3+
- 5–6 papers is the target; fewer looks thin, more dilutes the focus

---

## Paragraph 3 — Your Contribution

**Four things to cover (in this order):**
1. What remains unsolved or unmitigated in the literature — and why?
2. How do you resolve those unsolved issues specifically?
3. Why are your proposed methods and dataset acceptable/applicable/suitable for these issues?
4. What are the concrete outcomes after the experimental journey?

**Pattern:**
> [What the literature leaves open, and why it was hard] → [Your specific mechanism] → [Why it is valid / what properties make it suitable] → [Concrete experimental outcomes with numbers]

**Example:**
> "Prior methods cannot certify local explanations because they treat the network as a black box and have no access to the activation geometry. We expose this geometry algebraically by defining the attribution invariance polytope — the maximal convex region on which all activations are constant, computed in a single layer-wise Jacobian pass. Because the network is exactly affine inside this region, any faithful explanation method must return the same attribution vector, making the region a natural correctness certificate. Applying a falsifiability test derived from this structure to four widely used methods reveals KernelSHAP failing on all three tabular benchmarks, while LIME's instability is shown to be a sampling artefact that disappears when background samples are restricted to the certified region."

**Contribution Bullet Format:**
After the paragraph, list 3–5 contributions as bullets. Each bullet must:
- Start with a **bold noun phrase** naming the contribution type
- Follow with a clause that uses a strong, specific verb
- Be falsifiable — a reader should be able to verify or disprove it from your paper

**Strong verbs for contribution bullets:**
`propose`, `introduce`, `define`, `prove`, `establish`, `derive`, `construct`, `reveal`, `demonstrate`, `certify`

**Good bullet:**
> **A one-pass escape-radius algorithm** that computes the Euclidean distance from any input to the nearest face of its polytope in a single layer-wise Jacobian pass, at a cost comparable to one gradient evaluation.

**Bad bullet:**
> We improve upon existing approaches by proposing a better method.

**Section roadmap (last sentence of the Introduction):**
> "Section 2 reviews prior work on post-hoc explanations, tropical geometry, and counterfactual methods. Section 3 defines the polytope, the escape radius, and the rank-one bound. Section 4 reports experiments E1–E7. Section 5 discusses limitations and scope. Section 6 concludes."

---

## When to Use a Separate Related Work Section Instead

Use this merged (3-paragraph) format when:
- The venue or template does not require a separate Related Work section
- The paper is short (≤8 pages, workshop papers, letters)
- The literature is narrow enough that 5–6 papers cover the relevant work

Use a **separate Related Work section** (see `03-related-work.md`) when:
- The venue explicitly requires it (most full conference papers: NeurIPS, ICML, ICLR)
- Your paper engages with multiple distinct lines of prior work that need thematic subsections
- Word/page count allows it (full journal papers, 9+ page conference papers)

---

## Introduction Checklist

**Paragraph 1:**
- [ ] Cites 3–4 recent papers, each named for what it does
- [ ] States a specific technical gap (not just "the problem is hard")
- [ ] States your high-level approach in one sentence
- [ ] Names the scope (dataset, domain, method type)

**Paragraph 2:**
- [ ] Covers 5–6 papers
- [ ] Each paper gets 1–2 sentences: what it does + what it lacks
- [ ] Papers are grouped by theme, not listed chronologically
- [ ] Ends with a bridging sentence that points toward your solution
- [ ] Every cited gap maps to one of your contributions

**Paragraph 3:**
- [ ] Names what specifically remains unsolved and why
- [ ] Explains your mechanism and why it is valid for this problem
- [ ] Includes at least one quantified experimental outcome
- [ ] Followed by 3–5 contribution bullets (bold noun phrase + specific verb)
- [ ] Ends with a section roadmap

**Overall:**
- [ ] No contribution in the bullets is absent from the experiments or proofs
- [ ] No method is described in detail here (save for Section 3+)
- [ ] No vague claims ("novel", "significant", "various") without specifics
