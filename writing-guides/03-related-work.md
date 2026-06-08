# Related Work: Standalone Section (Positioning, Not Listing)

> **Note:** If your venue or paper format embeds the literature review inside the Introduction (common for short papers, workshops, or letters), use `02-introduction.md` instead — it covers the merged format. This guide is for papers that have a **dedicated Related Work section** (Section 2 or later).

## Purpose

Related Work must answer one question: **How is this paper different from everything before it?**

A Related Work that only summarizes prior papers without explaining differences will be rejected. Every paragraph must end by clarifying how your work relates.

---

## Structure

### Option A: Thematic subsections (recommended for papers with broad scope)

Group related papers by theme (not by chronology). Each subsection covers one line of work and ends with a sentence explaining how your paper relates or departs.

**Example subsection structure:**
```
2.1 Post-hoc explanation methods
2.2 Tropical geometry and linear regions of neural networks
2.3 Counterfactual explanations
2.4 Lipschitz analysis of networks
```

Each subsection: 3–5 papers, 1–2 sentences each, ending with a bridging sentence.

### Option B: Single section with thematic paragraphs

For shorter related work (1 page), use paragraph-level themes without subsection headers.

---

## The Per-Paper Formula

For each cited paper, use this structure:
1. **What they do** — method + key result (1 sentence)
2. **What they lack** — the gap relevant to your work (1 sentence, optional if the gap is addressed in the bridge)

**Strong example:**
> "Moosavi-Dezfooli, Fawzi and Frossard (2016) linearise the classifier at each iterate and project to the nearest decision hyperplane. Croce and Hein (2020a) adapt the boundary projection with a fast adaptive step to escape local minima. The chamber walk is closer in spirit to these white-box methods than to Wachter or Growing Spheres."

**Weak example (do not write this way):**
> "Many methods have been proposed for counterfactual explanations [3, 7, 12, 15]. These methods have various advantages and disadvantages."

---

## Positioning Rules

**Every subsection or theme must end with a differentiating sentence:**
> "The present paper is the first to derive all three quantities — the maximal affine region, its escape radius, and a closed-form attribution bound — in a single layer-wise pass."

**Differentiation verbs to use:**
- "The present paper differs in that..."
- "Unlike [prior work], we..."
- "This paper extends / departs from / builds on [X] by..."
- "No prior work has [X]; we address this by..."

---

## Related Work Checklist

- [ ] Papers are grouped by theme, not listed chronologically
- [ ] Each paper is cited with author name(s) + year, not just a number
- [ ] Each cited paper's gap is named, not just its contribution
- [ ] Every subsection ends with a sentence explaining how YOUR work differs
- [ ] No paper is cited that is not referenced in the main contribution
- [ ] The most closely related work is addressed explicitly (not minimized)
- [ ] You do not describe your own method in Related Work — save it for Section 3+

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| "X et al. proposed Y [5]." with no follow-up | Add the limitation or contrast |
| Listing 15 papers in one paragraph | Group into themes; each gets 1–2 sentences |
| Describing your method to compare | Just name the contrast: "unlike us, they do not certify..." |
| Ignoring the most similar prior work | Address it directly — reviewers will notice |
| "Many works have studied X" | Name the specific works |
