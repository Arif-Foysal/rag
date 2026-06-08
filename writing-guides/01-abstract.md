# Abstract: The 5-Sentence Formula

## Purpose

The abstract is read by everyone. It is the only section most people read. It must standalone — no citations, no undefined notation, no assumed context.

## The 5-Sentence Structure

**Sentence 1 — The Problem (context + gap)**
State what exists and what is missing. One sentence. Be specific about the gap, not just the field.
> "Local explanations of neural network predictions lack certified regions of validity, causing post-hoc methods to drift even on inputs where the network is provably linear."

**Sentence 2 — Your Approach (what you propose)**
Name your method or framework. One concrete noun phrase. Avoid vague words like "novel" or "new".
> "We propose a certification framework that exposes the exact local affine geometry of piecewise-linear neural networks."

**Sentence 3 — How it Works (mechanism, briefly)**
The key technical idea in plain terms. One sentence. No equations.
> "We define the attribution invariance polytope (AIP) to expose the maximal closed-form region of exact affine equivalence, computing the exact distance to its boundary in a single layer-wise Jacobian pass."

**Sentence 4 — What You Found (key results)**
Quantified outcomes. Use numbers. Avoid "significantly" or "substantially" without numbers.
> "Applying it to four widely used methods reveals that KernelSHAP fails on all three tabular benchmarks under the in-polytope cohort, while LIME stabilises when sampling is restricted to this region."

**Sentence 5 — Why It Matters (impact)**
The takeaway for the field. What can practitioners or researchers now do that they could not before?
> "By making the network's local geometry explicit and computable, the framework enables a new class of auditing tools that distinguish estimator failure from actual network behaviour."

## Abstract Checklist

- [ ] No undefined acronyms on first use (define them: "attribution invariance polytope (AIP)")
- [ ] Contains at least one quantified result (%, ×, dB, F1, etc.)
- [ ] No citations
- [ ] No "In this paper, we..." — just state the contribution directly
- [ ] Present tense for contributions ("We propose", "We prove"), past tense for findings ("We found", "Results showed")
- [ ] Under 250 words
- [ ] No equations or mathematical notation that cannot be spelled out in words
- [ ] The method name appears (if it has one)
- [ ] The application domain is clear by sentence 2
- [ ] The limitation or scope is implicit (if the method applies to ReLU networks, say so)

## Common Abstract Mistakes

| Mistake | Fix |
|---------|-----|
| "We propose a novel method..." | Remove "novel" — it adds nothing |
| "...which outperforms baselines." | Add numbers: "17× closer than gradient-descent baseline" |
| "This is important because..." | Embed the importance in the impact sentence |
| "We study the problem of X" | "We solve / address / certify X" |
| Describing only methods, not results | Add Sentence 4 with numbers |
| Ending with "Future work includes..." | End with impact, not limitations |
