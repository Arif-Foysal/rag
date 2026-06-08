# Conclusion: The 3-Part Close

## Purpose

The Conclusion must be readable without the rest of the paper. A reviewer who skims to the conclusion should leave knowing exactly what was done, what was proved, and why it matters.

---

## The 3-Part Structure (one paragraph each)

**Part 1 — Restate the problem and gap (2–3 sentences)**
Not a copy of the introduction. A compressed version that reminds the reader why this was worth solving.

> "Local explanations of neural network predictions have lacked certified regions of validity, leaving practitioners unable to distinguish genuine network properties from sampling artefacts of estimation methods."

**Part 2 — Summarize contributions (3–5 sentences)**
Map directly to the contribution bullets in the Introduction. Use past tense.

> "We introduced the attribution invariance polytope (AIP), the maximal closed-form region on which a piecewise-linear network is exactly affine. We proved that the escape radius to its boundary is computable in a single layer-wise Jacobian pass. Applying a falsifiability test derived from Theorem 1 revealed that KernelSHAP fails on all three tabular benchmarks under the in-polytope cohort, while LIME's instability was shown to be entirely a sampling artefact."

**Part 3 — Broader impact and what comes next (2–3 sentences)**
What can the field now do that it could not before? One concrete direction for future work.

> "By making local geometry explicit and computable, the framework enables auditing tools that distinguish estimator error from network behaviour — a capability previously unavailable. The chamber-walk algorithm provides a foundation for geometry-aware recourse methods that respect network linearity by construction."

---

## Conclusion Rules

- **Past tense** for what you did: "We proved", "We introduced", "Results showed"
- **Present tense** for general truths that remain true: "The AIP is the maximal region...", "This bound holds for..."
- No new claims — every statement must be supported by something in the paper
- No apologizing — do not end with "despite limitations, this is a first step"
- Do not start with "In this paper, we..." — start with the problem or the contribution directly

---

## Conclusion Checklist

- [ ] All major contributions are mentioned (map to the Introduction bullets)
- [ ] Past tense used for contributions and findings
- [ ] No new results or claims introduced
- [ ] Ends with impact, not limitations
- [ ] Readable as a standalone paragraph — no undefined terms
- [ ] Under 300 words (conclusions are not a second discussion section)
