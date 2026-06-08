# Reviewer Response: Turning Rejection Into Acceptance

## The Core Principle

Reviewers are not your enemies. They are busy experts who read your paper in 90 minutes and wrote what they noticed. Your job is to address every point completely and make it easy for them to see that you did.

---

## Step 1: Classify Every Comment

Create a table with every reviewer comment, classified as:

| Type | Description | How to Respond |
|------|-------------|----------------|
| **Language** | Grammar, clarity, word choice | Fix in paper; list changes made |
| **Structure** | Section order, missing elements | Reorganize; explain why if structure is retained |
| **Content gap** | Missing experiment, analysis, citation | Add content; or argue out of scope with justification |
| **Correctness** | Wrong claim, broken proof, flawed experiment | Fix or acknowledge the limitation explicitly |
| **Scope** | Reviewer wanted broader claims | Clarify scope in paper; do not overclaim |

---

## Step 2: Write the Response Letter

### Format for each response

**Reviewer N, Comment M:**
> [Quote the reviewer's comment verbatim in a blockquote]

**Response:**
[Your answer — be direct, not defensive]

**Changes made:**
> [Quote the new text from the paper, or describe the location of the change]

---

### Response Letter Rules

**Do respond to every comment** — even if only to say "We thank the reviewer for this observation and have added a sentence in Section X."

**Do not argue without evidence.** If a reviewer misunderstood, the misunderstanding is your paper's fault. Fix the paper AND clarify in the response.

**Do not say "we agree"** without also saying what you changed. An "I agree" with no change is meaningless.

**Do not over-promise** — "Future work will..." is acceptable but should not substitute for addressing the actual concern.

**Quantify your changes** — "We added 2 paragraphs in Section 3.2", "We added Table 4 with ablation results", "We revised 6 sentences in the Introduction."

---

## Step 3: The "Language Editing" Comment

When a reviewer says "Language editing is necessary," it means they found errors throughout, not in one place. Treat this as a full-paper pass.

**Workflow:**
1. Run the full `08-language-editing.md` checklist on every section
2. Use Claude Code with: "Apply the rules in `writing-guides/08-language-editing.md` to Section [X]. Quote problematic sentences, state which rule is violated, and provide the corrected version."
3. Do a second pass specifically for tense consistency and article usage (Rules 8 and 15)
4. In your response letter: "We have revised the paper throughout for language clarity, addressing grammar, tense consistency, active/passive voice, and sentence length. We list the major changes by section below."

---

## Step 4: Track Changes

When resubmitting, mark all changes in the manuscript using:
- LaTeX: `\textcolor{blue}{new text}` or use the `changes` package
- Word: Track Changes mode

This lets reviewers verify your changes without re-reading the whole paper.

---

## Step 5: Revision Checklist

- [ ] Every reviewer comment has a numbered response
- [ ] No comment is dismissed without explanation
- [ ] All paper changes are quoted or located in the response letter
- [ ] Language changes are listed by section, not just acknowledged in aggregate
- [ ] New experiments are described with their results in the response letter
- [ ] The response letter has a summary table: Comment → Action taken → Location in paper
- [ ] Changes are highlighted in the revised manuscript

---

## Response Letter Template

```
Dear Editor and Reviewers,

We thank the reviewers for their careful reading and constructive comments.
We have revised the manuscript to address all concerns. Below we respond to
each comment in turn. Changes in the revised manuscript are highlighted in blue.

---

REVIEWER 1

Comment 1.1: [reviewer text]
Response: ...
Changes: [Section X, lines Y–Z]

Comment 1.2: ...

---

REVIEWER 2

...

---

We believe the revised manuscript fully addresses the reviewers' concerns
and is now suitable for publication.

Sincerely,
[Authors]
```
