# Language Editing: 35 Rules for Clear Academic Writing

This guide addresses the most common reasons reviewers write "Language editing is necessary." Apply these rules to every sentence before submission.

---

## Part 1: Sentence-Level Rules

### Rule 1: One idea per sentence
Split any sentence that contains more than one independent claim.

> **Before:** "We propose a method that uses ensemble learning and also includes SHAP and LIME for interpretability and achieves high accuracy."
> **After:** "We propose a stacking ensemble that achieves 99.38% accuracy. We add SHAP and LIME explanations to make predictions interpretable to clinicians."

### Rule 2: 35-word limit
If a sentence exceeds 35 words, split it. Count and cut.

### Rule 3: Subject near the verb
The subject and its verb should not be separated by more than one clause.

> **Before:** "The method, which we developed to handle the class imbalance problem that is common in medical datasets, uses SMOTE."
> **After:** "We address the class imbalance common in medical datasets using SMOTE."

### Rule 4: Avoid "there is / there are" openers
These delay the real subject.

> **Before:** "There are many methods that have been proposed for this task."
> **After:** "Many methods address this task."

### Rule 5: Avoid "it is" openers
> **Before:** "It is important to note that the results may vary."
> **After:** "Results vary across datasets (Table 3)."

---

## Part 2: Verb Choice

### Rule 6: Use strong verbs, not weak + noun constructions

| Weak | Strong |
|------|--------|
| "make an analysis of" | "analyse" |
| "perform a comparison" | "compare" |
| "provide an explanation" | "explain" |
| "give an indication" | "indicate" |
| "carry out an investigation" | "investigate" |

### Rule 7: Prefer active voice for your contributions

> **Passive (weak):** "The attribution invariance polytope is defined as..."
> **Active (strong):** "We define the attribution invariance polytope as..."

Passive voice is acceptable for describing established facts: "ReLU networks are piecewise-linear."

### Rule 8: Tense consistency

| Context | Tense |
|---------|-------|
| Established facts / prior work findings | Present: "ReLU networks partition input space into..." |
| Your methodology | Present: "We define / propose / prove..." |
| Your experimental results | Past: "We observed / found / measured..." |
| What the paper does | Present: "Section 3 introduces..." |

Never mix tenses within a single paragraph without a reason.

### Rule 9: Avoid "study", "investigate", "look at" as contribution verbs
These describe process, not achievement.

> **Weak:** "We study the stability of post-hoc explanations."
> **Strong:** "We certify the stability of post-hoc explanations by constructing a closed-form validity region."

---

## Part 3: Word Choice

### Rule 10: Remove "novel"
Every submitted paper claims novelty. The word adds nothing and signals weak writing.

> **Before:** "We propose a novel framework..."
> **After:** "We propose a framework..."

### Rule 11: Remove "very", "quite", "rather", "somewhat"
These are hedges that weaken claims.

> **Before:** "The results are quite good."
> **After:** "The results exceed all baselines by at least 4 pp."

### Rule 12: Replace "significant" with numbers
"Significantly better" means nothing to a reviewer. Use the actual number.

> **Before:** "Our method significantly outperforms the baseline."
> **After:** "Our method outperforms the baseline by 3.2% (p < 0.01, paired t-test)."

### Rule 13: Avoid vague scope words

| Vague | Specific |
|-------|---------|
| "many methods" | "five widely used methods [1–5]" |
| "some papers" | "prior work [3, 7, 12]" |
| "various approaches" | name them |
| "several experiments" | "seven experiments (E1–E7)" |

### Rule 14: "which" vs "that"
- "that" introduces a defining clause (no comma): "The method that uses ReLU..."
- "which" introduces a non-defining clause (with comma): "The method, which we call AIP, uses..."

### Rule 15: Article rules (a / an / the)

| Rule | Example |
|------|---------|
| First mention of a concept: use "a/an" | "We propose **a** certification framework..." |
| Second mention of the same concept: use "the" | "**The** framework exposes the local geometry..." |
| Unique or previously defined: use "the" | "**The** attribution invariance polytope (AIP)..." |
| General class: no article | "**Neural networks** are piecewise-linear..." |

### Rule 16: Hyphenation
Compound adjectives before a noun are hyphenated:
- "piecewise-linear network" ✓
- "layer-wise Jacobian pass" ✓
- "high-accuracy classifier" ✓
- "well-known method" ✓

### Rule 17: Avoid "etc." in academic writing
Name what you mean or say "among others."

> **Before:** "We use XGBoost, Random Forest, etc."
> **After:** "We use XGBoost, Random Forest, and Decision Tree."

---

## Part 4: Paragraph Structure

### Rule 18: Every paragraph has a topic sentence
The first sentence must state the paragraph's main claim. The rest supports it.

### Rule 19: Every paragraph has a forward link
The last sentence must connect to the next paragraph or section.

> "These results confirm the escape-radius bound. Section 4.3 now tests whether restricting LIME's sampling to this region stabilises its attribution."

### Rule 20: Paragraph length
3–6 sentences per paragraph. A one-sentence paragraph is a red flag (either expand or merge). A 10-sentence paragraph needs to be split.

---

## Part 5: Common Grammar Errors

### Rule 21: Subject-verb agreement with collective nouns

> **Wrong:** "The set of features **are** selected..."
> **Right:** "The set of features **is** selected..."
> **Right:** "The features **are** selected..."

### Rule 22: "Data" is plural

> **Wrong:** "The data **was** collected..."
> **Right:** "The data **were** collected..." or "The dataset **was** collected..."

### Rule 23: Comma before "which"

> **Wrong:** "The method which we call AIP is exact."
> **Right:** "The method, which we call AIP, is exact."

### Rule 24: No comma before "that"

> **Wrong:** "The region, that the network certifies, is convex."
> **Right:** "The region that the network certifies is convex."

### Rule 25: "Based on" vs "on the basis of"
Use "based on" for adjective use, not as a sentence opener.

> **Wrong:** "Based on these results, we conclude..."
> **Right:** "These results indicate..." or "From these results, we conclude..."

### Rule 26: "Respectively" needs parallel lists

> **Wrong:** "Table 1 and Table 2 show accuracy and F1, respectively." (if tables are not in this order in the sentence, reorder)
> **Right:** "Accuracy and F1 are reported in Tables 1 and 2, respectively."

### Rule 27: "Former" and "latter" — avoid them
They force the reader to look back. Repeat the noun instead.

### Rule 28: Possessive apostrophe for singular nouns

> **Wrong:** "The network's output" → ✓ this is correct
> **Wrong:** "The networks output" → missing apostrophe

### Rule 29: "i.e." vs "e.g."
- i.e. (id est) = "that is" — gives the only possible explanation
- e.g. (exempli gratia) = "for example" — gives one of several options
- Both must be followed by a comma: "e.g., ReLU networks"

### Rule 30: Serial comma (Oxford comma)
Use it in technical writing: "XGBoost, Random Forest, and Decision Tree" (not "XGBoost, Random Forest and Decision Tree").

---

## Part 6: Academic Register

### Rule 31: Avoid contractions
"Don't" → "do not", "It's" → "It is", "We've" → "We have"

### Rule 32: Avoid colloquial phrases

| Colloquial | Academic |
|-----------|---------|
| "a lot of" | "many / numerous / substantial" |
| "big" | "large / substantial / significant" |
| "get" | "obtain / achieve / compute" |
| "show up" | "appear / emerge / manifest" |
| "deal with" | "address / handle / resolve" |
| "figure out" | "determine / derive / identify" |

### Rule 33: Avoid rhetorical questions
"But how can we certify this?" → "Certifying this requires..."

### Rule 34: Avoid "Note that" and "It should be noted that"
These are filler. If something is worth saying, say it directly.

### Rule 35: Parallel structure in lists
All items in a bulleted or numbered list must have the same grammatical form.

> **Wrong:** "Our contributions are: (1) we define the AIP, (2) an algorithm, (3) proving a bound."
> **Right:** "Our contributions are: (1) a definition of the AIP, (2) an algorithm that computes the escape radius, (3) a proof of the rank-one attribution bound."

---

## Quick Pre-Submission Checklist

- [ ] Every sentence under 35 words
- [ ] No "novel", "very", "quite", "significant" without numbers
- [ ] Active voice for contributions, passive acceptable for background
- [ ] Consistent tense within each section
- [ ] Every paragraph starts with a topic sentence
- [ ] Oxford commas throughout
- [ ] "which" always preceded by a comma; "that" never preceded by a comma
- [ ] No contractions
- [ ] All lists are grammatically parallel
- [ ] Every abbreviation defined on first use: "attribution invariance polytope (AIP)"
