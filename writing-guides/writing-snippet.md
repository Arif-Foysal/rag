# Writing Snippet — Paste This Into Any Existing CLAUDE.md

Copy the block below and append it to the `CLAUDE.md` in any paper repo.
Then copy the `writing-guides/` folder into that repo alongside it.

---

```markdown
## Academic Writing Assistant

Writing guides are in `writing-guides/`. For any writing or revision task,
read the relevant guide before suggesting changes.

| Task | Guide |
|------|-------|
| Language / grammar issues | `writing-guides/08-language-editing.md` |
| Paper structure overview | `writing-guides/00-paper-structure.md` |
| Abstract | `writing-guides/01-abstract.md` |
| Introduction (with merged literature review) | `writing-guides/02-introduction.md` |
| Methodology | `writing-guides/04-methodology.md` |
| Results / Experiments | `writing-guides/05-results.md` |
| Discussion | `writing-guides/06-discussion.md` |
| Conclusion | `writing-guides/07-conclusion.md` |
| Figures, tables, formatting | `writing-guides/09-presentation.md` |
| Responding to reviewers | `writing-guides/10-reviewer-response.md` |
| Related Work (standalone section only) | `writing-guides/03-related-work.md` |

When revising text: quote the original sentence, name the rule violated
(referencing the guide), and provide the corrected version.
When auditing a section: list all violations found before proposing fixes.
```

---

## How to Deploy to a New Paper Repo

```bash
# From this master workspace:
cp -r writing-guides/ /path/to/your-paper-repo/writing-guides/

# Then open the existing CLAUDE.md in your paper repo and paste the block above.
```
