# Results and Experiments: Presenting Evidence

## Purpose

The experiments section must prove every claim made in the Introduction. Map each contribution bullet to at least one experiment.

---

## Structure

```
4.1 Experimental Setup
    - Datasets (name, size, splits, source)
    - Baselines (named, cited, why chosen)
    - Evaluation metrics (defined, justified)
    - Hardware / compute (for reproducibility)
4.2 [Experiment 1 — Testing Contribution 1]
4.3 [Experiment 2 — Testing Contribution 2]
...
4.N Ablation Study (what happens when you remove key components?)
```

---

## Reporting Numbers

### Always report:
- Mean ± standard deviation (or standard error) across runs
- Number of runs/seeds
- Statistical significance if comparing to baselines (p-value or confidence interval)
- Whether higher or lower is better (e.g., "↑ means higher is better")

### Formatting tables:
- Bold the best result per row/column
- Underline the second best
- Use the same number of decimal places throughout
- Add a footnote for any dagger (†) or asterisk (*) entries
- Caption must describe what the table shows AND what the key takeaway is

**Strong caption example:**
> "Table 2: PESQ scores under 10% packet loss. Our method (Burg+Residual) outperforms all baselines on clean speech, with the largest gain over standard Burg (0.23 PESQ points, p < 0.01)."

**Weak caption example:**
> "Table 2: Results."

---

## Describing Results in Prose

**Always interpret, not just report:**

| Weak (just reporting) | Strong (reporting + interpreting) |
|----------------------|----------------------------------|
| "Our method achieves 97.3% accuracy." | "Our method achieves 97.3% accuracy, a 4.1 pp gain over the strongest baseline (XGBoost, 93.2%), confirming that stacking captures cross-model complementarity that single classifiers miss." |
| "The escape radius is smaller for deeper networks." | "The escape radius shrinks with depth (Fig. 3), consistent with the exponential growth in activation hyperplanes predicted by Theorem 2." |

**Pattern:** [Quantified result] + [comparison to baseline] + [interpretation / why it matters]

---

## Ablation Studies

An ablation removes one component at a time to show each component contributes.

Format:
- One row per removed component
- Use "-X" naming: "Full model", "- feature selection", "- SMOTE", "- stacking"
- Report the metric drop for each removal

---

## Figures

- Every figure must earn its space — if the takeaway can be stated in one sentence, the figure may be unnecessary
- Use the caption to state the takeaway explicitly: "Figure 4 shows that LIME stabilises inside the certified region (blue), confirming that instability is a sampling artefact."
- Axes must be labelled with units
- Legends must be readable at print size (≥8pt font)
- Color should not be the only differentiator (use shape/pattern for accessibility)

---

## Results Checklist

- [ ] Every contribution from the Introduction has a corresponding experiment
- [ ] All numbers are mean ± std over multiple seeds
- [ ] Best results are bolded in tables
- [ ] Figure captions state the takeaway, not just the content
- [ ] Ablation covers all key design choices
- [ ] Baselines are named and cited (not just "prior methods")
- [ ] Statistical tests are reported where claims of superiority are made
