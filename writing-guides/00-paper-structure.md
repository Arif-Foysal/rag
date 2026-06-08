# Paper Structure: Architecture of a Strong CS/ML Paper

## The Core Principle

Every section must answer one question for the reader. If a section cannot be summarized in one sentence, it is doing too many things.

## Standard Section Order and Purpose

| Section | Reader's Question | Typical Length |
|---------|-------------------|----------------|
| Abstract | What did you do and why should I care? | 150–250 words |
| Introduction | What is the problem, why is it hard, and what exactly did you contribute? | 1–2 pages |
| Related Work | How does this differ from everything that came before? | 1–2 pages |
| Methodology | How does the approach actually work? | 2–4 pages |
| Experiments | Does it work, and how well? | 2–3 pages |
| Discussion | What does this mean, and what are the limits? | 0.5–1 page |
| Conclusion | What did you prove, and what comes next? | 0.5 page |

## Section Dependencies (write in this order)

1. **Contributions list** (bullet points) — define these first; every other section must be consistent with them
2. **Methodology** — the technical core
3. **Experiments** — validates the methodology
4. **Introduction** — written last; the contributions are now proven
5. **Abstract** — written last; distills everything
6. **Related Work** — can be written early but refined after intro is stable

## What Makes a Paper Feel "Complete"

- Every claim in the Introduction is validated in Experiments or proved in Methodology
- Every technique in Methodology is evaluated in Experiments
- Every result in Experiments is interpreted in Discussion or Conclusion
- No section introduces a concept not defined or cited elsewhere

## Red Flags for Reviewers

- "We leave X for future work" when X is central to the claimed contribution
- Results reported without statistical significance or error bars
- Figures without axis labels or captions that explain the takeaway
- Related work that lists papers without explaining how this paper differs
- Introduction contributions that do not map 1-to-1 to sections
- Abstract that describes methods but not results (the outcome must appear)

## Contribution Types and How to Frame Them

| Type | Framing verb | Example |
|------|-------------|---------|
| New algorithm | "We propose / introduce" | "We propose a one-pass algorithm..." |
| New theory | "We prove / establish / derive" | "We prove a tight Lipschitz bound..." |
| New dataset | "We present / release" | "We present a benchmark of..." |
| New evaluation | "We show / demonstrate / reveal" | "We show that KernelSHAP fails on..." |
| New framework | "We develop / construct" | "We develop a certification framework..." |

Never use vague verbs: "we study", "we look at", "we consider" — these do not describe a contribution.
