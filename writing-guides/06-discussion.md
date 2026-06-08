# Discussion: Meaning, Limits, and Scope

## Purpose

Discussion interprets results beyond what the numbers show. It answers: "So what?" — connecting findings to the broader field and honestly addressing limits.

Not all venues require a separate Discussion section. Some papers fold it into Results or place it before Conclusion. Either is acceptable as long as the content exists.

---

## Structure

```
Discussion
  ├── Interpretation of key findings
  ├── Connection to related work / prior understanding
  ├── Limitations
  └── Future work
```

---

## Interpretation of Key Findings

Do not repeat numbers from the Results section. Instead, explain what the pattern of results means.

**Weak (just repeating):**
> "Our method achieves 99.38% accuracy on the thyroid dataset."

**Strong (interpreting):**
> "The near-ceiling accuracy on the thyroid dataset suggests that the seven feature selection methods form a consensus that eliminates the noise sources responsible for prior classifiers' lower performance. The 3% gap between the stacking ensemble and the best single classifier (XGBoost) is consistent with ensemble theory: stacking exploits disagreements between classifiers rather than correlating their errors."

---

## Connection to Prior Work

Explain how your results change or confirm the field's understanding:
- "This confirms the finding of [X] that..."
- "In contrast to [Y], our results suggest that..."
- "The failure of [method] under our certified cohort suggests the instability documented by [Z] is a sampling artefact, not a model property."

---

## Limitations

Be direct. Reviewers who find limitations you did not disclose will penalize you more than the limitation itself.

**Framing pattern:**
> "[Specific limitation]. This means [what it rules out]. Future work could address this by [direction], though [constraint]."

**Common limitations to address honestly:**
- Scope: which datasets / architectures / domains does the method apply to?
- Assumptions: what must be true for the method to work?
- Computational cost: what scale does the method break down at?
- Generalization: is the evaluation broad enough to claim general applicability?

---

## Future Work

Keep this brief (2–4 sentences). Only name directions that are genuinely open — do not list things you could have done but did not.

**Weak:**
> "In future work, we plan to extend our method to other domains."

**Strong:**
> "The chamber-walk algorithm currently targets prediction-flip counterfactuals; extending it to handle actionability constraints (immutability, plausibility under data manifold) would require combining the geometric traversal with a feasibility oracle."

---

## Discussion Checklist

- [ ] No result is simply repeated from the Experiments section
- [ ] At least one finding is connected to a specific prior work
- [ ] Limitations section names at least 2 specific constraints (not just "our method has limitations")
- [ ] Future work is concrete and non-obvious
- [ ] Discussion does not introduce new claims that were not backed by experiments
