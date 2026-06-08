# Presentation: Figures, Tables, Algorithms, and Formatting

## The Core Rule

Every visual element (figure, table, algorithm) must be self-contained. A reader who looks only at the figure/table and its caption must understand what it shows and what the key takeaway is — without reading the main text.

---

## Figures

### Caption Formula
**[What the figure shows] + [Key takeaway] + [How to read it (if non-obvious)]**

> **Weak:** "Figure 3: Results on three datasets."
> **Strong:** "Figure 3: Escape radius across three tabular benchmarks. Our streaming algorithm (blue) matches the brute-force oracle (black dashed) on all inputs, confirming Theorem 2. Smaller radius indicates tighter certification."

### Figure Checklist
- [ ] Both axes labelled with units (e.g., "PESQ score", "Accuracy (%)", "Epoch")
- [ ] Font size in figure ≥ 8pt at final print size
- [ ] Legend present if multiple series; entries match line style AND color
- [ ] Color is not the only differentiator — use shape/dash/pattern too (accessibility)
- [ ] Caption states the takeaway, not just the content
- [ ] Figure referenced in the main text before it appears: "Figure 3 shows..."
- [ ] No unnecessary whitespace or borders around the figure

### Common Figure Types and Their Purpose

| Figure type | When to use |
|-------------|-------------|
| Line plot | Trends over epochs, sizes, thresholds |
| Bar chart | Comparing discrete categories (datasets, methods) |
| Heatmap / confusion matrix | Per-class performance, feature correlation |
| Architecture diagram | System overview, pipeline flow |
| Scatter plot | Distribution, correlation between two variables |
| Waveform / spectral plot | Signal processing results |

---

## Tables

### Table Caption Placement
Table captions go **above** the table (figure captions go below). This is the IEEE/Elsevier standard.

### Table Design Rules
- Bold the best result in each column (or row, consistently)
- Underline the second best
- Align numbers by decimal point
- Use the same number of decimal places in each column
- Add ↑ or ↓ in the header to indicate whether higher or lower is better
- Use horizontal rules only: top rule, rule below header, bottom rule (no vertical lines — IEEE style)
- Include a "method" column and a "reference" column where appropriate

### Table Caption Formula
**[What the table measures] + [Main finding] + [Any special notation explained]**

> **Weak:** "Table 1: Performance comparison."
> **Strong:** "Table 1: Classification performance on the UCI thyroid dataset (↑ higher is better). Our stacking ensemble (bottom row) achieves the highest accuracy (99.38%) and F1 (0.9941) across all metrics. † indicates results reproduced from [12]."

---

## Algorithms

### Algorithm Block Structure
```
Algorithm N  Name of the Algorithm
Require: [all inputs, with types or ranges]
Ensure:  [output guarantee or what is returned]
1: [line]
2: [line]
...
Output: [return value]
```

### Algorithm Writing Rules
- Every line is unambiguous — no natural language that could be interpreted differently
- Index variables must be declared: "for $i = 1, \ldots, N$ do"
- Comment lines (▷) explain the WHY, not the WHAT
- After the algorithm, prose must explain the key lines: "Line 4 computes the dual norm, which is the $\ell_\infty$ norm since $\ell_2$ is self-dual."
- State time and space complexity after the algorithm

---

## Cross-References and Consistency

### Internal references
- Always refer to figures, tables, algorithms, equations, theorems by number: "Figure 3", "Table 2", "Algorithm 1", "Equation (5)", "Theorem 2"
- Never: "the figure above", "the following table" — page layout may change
- Equation references use parentheses: "from (5)" or "Equation (5)"

### Consistency rules
- If you use "ReLU network" in Section 1, do not switch to "rectified linear network" in Section 3
- If you define an acronym (AIP), use it consistently throughout — no switching back to full form
- Mathematical notation must be identical everywhere: if you write $\mathcal{P}(x_0)$ in the definition, write $\mathcal{P}(x_0)$ everywhere, not $P(x_0)$ or $\mathcal{P}$

---

## Formatting Checklist

- [ ] All figures have captions below; all tables have captions above
- [ ] Best results in tables are bolded
- [ ] All figures have labelled axes with units
- [ ] Every visual element is referenced by number in the main text
- [ ] Consistent notation throughout (same symbol = same thing)
- [ ] No vertical lines in tables
- [ ] Algorithms have Require/Ensure and stated complexity
- [ ] Font sizes in figures are readable at print scale
