# Methodology: Writing the Technical Core

## Purpose

The methodology section must be reproducible without the code. A reader with domain knowledge should be able to implement your approach from this section alone.

---

## Structure for a Theory/Algorithm Paper

```
3.1 Notation and Setup
3.2 Core Definition(s)
3.3 Main Algorithm(s)
3.4 Theoretical Guarantees (theorems, proofs or proof sketches)
3.5 Practical Considerations (complexity, extensions)
```

## Structure for an Applied ML Paper

```
3.1 Problem Formulation
3.2 Dataset and Preprocessing
3.3 Feature Selection / Data Pipeline
3.4 Model Architecture / Training Setup
3.5 Evaluation Metrics
```

---

## Writing Rules for Mathematical Content

### Notation
- Define every symbol before first use: "Let $F : \mathbb{R}^n \to \mathbb{R}^k$ be a feedforward network..."
- Use a notation table if you introduce more than 8 symbols
- Be consistent — never use the same symbol for two different things
- Subscript/superscript conventions must be stated: "superscript $(\ell)$ denotes layer index"

### Definitions
- Use the format: **Definition N (Name).** *Definition body.*
- Immediately follow with intuition: "Intuitively, this captures..."
- Give a simple example or a figure reference if the concept is geometric

### Theorems
- Use the format: **Theorem N (Name).** *Theorem statement.*
- **Proof.** ... $\square$
- If the proof is long, give a proof sketch in the main text and defer the full proof to the appendix
- After the theorem, write one paragraph of interpretation: "This theorem says that... The practical implication is..."

### Algorithms
- Use a numbered pseudocode block (Algorithm N)
- Every line must be executable — no ambiguous natural language inside the algorithm
- State: **Require:** (inputs) and **Ensure:** (output guarantee)
- After the algorithm, explain the key steps in prose: "Line 4 computes... Line 7 updates..."
- State the time and space complexity

---

## Writing Rules for Applied/ML Methods

- Every preprocessing step must be justified: "We apply SMOTE balancing because the dataset has 3:1 class imbalance (Table 1)"
- Every hyperparameter must be stated: optimizer, learning rate, batch size, number of epochs, random seed
- Data splits must be explicit: "We use an 80/10/10 train/validation/test split with stratified sampling"
- If you use an existing method, cite it and explain what you changed (if anything)

---

## Methodology Checklist

- [ ] Every symbol is defined before use
- [ ] All algorithms have Require/Ensure and stated complexity
- [ ] Every theorem is followed by an interpretation paragraph
- [ ] No result is stated without a proof or citation
- [ ] All hyperparameters are reported
- [ ] The methodology is sufficient to reproduce the work without the code
- [ ] Figures referenced in this section have self-contained captions

## Geometric Intuition Blocks

For geometric or abstract concepts, add a clearly labelled paragraph:

> *Geometric Intuition.* A ReLU network partitions input space into convex polyhedral chambers, and on each chamber it acts as one fixed linear map. The AIP $\mathcal{P}(x_0)$ is the largest such chamber containing $x_0$, exposed in closed form. Inside the AIP, the network behaves exactly like a linear model.

These blocks help reviewers and readers build mental models without losing technical precision.
