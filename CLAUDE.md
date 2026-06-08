# conformal-rag Pipeline — Operating Guide

## Project
Adaptive Retrieval-Augmented Generation research: reframing the retrieve/skip gate as a risk-controlled decision under asymmetric cost. We conformalize the gate decision itself (not the answer) using Conformal Risk Control over a semantic-entropy "ignorance" score, placing a finite-sample, distribution-free, group-conditional bound on the safety-critical error — skipping retrieval when the model would otherwise hallucinate. Baselines: Self-RAG, FLARE, a tuned-threshold confidence gate, and a competently recalibrated Bayesian gate with asymmetric loss. See `proposal.md` for the full framing, formal guarantee, and experimental plan.

## Architecture
- `src/conformal_core.py` — core library, uploaded to Kaggle as dataset `mdariffaysalnayem/conformal-rag-core`
- `pipeline.py` — CLI for push/status/fetch/upload-src/upload-results
- `config.yaml` — username, datasets, phases, success criteria
- `templates/` — pristine notebook templates (checked in)
- `notebooks/` — working copies you edit (gitignored)
- `results/` — downloaded Kaggle outputs (gitignored)

## Environment
The Kaggle API token lives in `.env` (gitignored) and is loaded automatically by `pipeline.py`. Just activate the venv:
```bash
source .venv/bin/activate
python pipeline.py push notebooks/<phase>.py    # token loads from .env
```
If `.env` is missing, create it with `KAGGLE_API_TOKEN=KGAT_...` from kaggle.com/settings.

## Running an experiment
1. `cp templates/<phase>.py notebooks/<phase>.py` (or `python pipeline.py generate <phase>`)
2. Edit `notebooks/<phase>.py` if needed (parameters, etc.)
3. `python pipeline.py push notebooks/<phase>.py`
4. `python pipeline.py wait <phase>` — single blocking call; prints one line per state change. Returns 0 on complete, 1 on kernel error, 2 on timeout. If Kaggle's status endpoint is down (see gotcha), `wait` automatically falls back to probing the output endpoint and still returns 0 on completion — so `wait && fetch` keeps working untouched.
5. `python pipeline.py fetch <phase>`
6. Read `results/<slug>/*.json` and grep `[RESULT]` lines to judge success. Use `python pipeline.py tail <phase>` for the last 40 log lines if you need to debug — never `cat` the full `.log`.

## Iteration protocol
When a notebook fails or success criteria aren't met:
- **Notebook bug** (syntax, shapes, logic) → edit `notebooks/<phase>.py`, re-push
- **Core math/lib bug** → edit `src/conformal_core.py`, run `python pipeline.py upload-src`, **wait ~5 min**, re-push the notebook
- **Parameter issue** → tweak the notebook's config section, re-push
- **After 5 retries on the same notebook** → stop and ask the human

## Between phases
When phase N's outputs feed phase N+1:
1. `python pipeline.py upload-results phaseN`
2. Add the new dataset slug to `config.yaml` under `kaggle.datasets` (the CLI prints the line to add)
3. In `config.yaml` under the next notebook, add the key to its `datasets:` list

## Output conventions (do not break)
- `core.format_result(key, value)` — emits a greppable `[RESULT]` line
- `core.save_results(dict, path)` — writes JSON to `/kaggle/working/`
- Plots saved as PNG to `/kaggle/working/`

## Token-cost discipline (read this — it's why the harness exists)
- **Only three signals matter for "did this work?"**: `results/<slug>/*.json`, `[RESULT]` lines (use `pipeline.py results <phase>` or `pipeline.py fetch`), and the kernel state from `pipeline.py wait`. Don't Read raw `.log` files into context; use `pipeline.py tail <phase>` if you need recent log lines.
- **Use `wait`, not a polling loop.** One `python pipeline.py wait <phase>` is dramatically cheaper in context tokens than running `status` 6× in a row.
- **Keep `pipeline.py`, `src/conformal_core.py`, `templates/` stable across iterations.** They are the cacheable prompt prefix; edit them only for durable changes. Iterate on `notebooks/<phase>.py`.
- **Don't print large arrays.** Truncate to a few items + length. `_pretty_print` already enforces this for fetched JSON.

## When to stop and ask
- Phase boundary transitions (gate decisions)
- 5 retries exhausted
- Borderline / ambiguous results
- Anything that needs scientific judgement
- Before destructive actions (deleting kernels, force-pushing, etc.)

## Gotchas
- **Kaggle dataset propagation lag**: new dataset versions take ~5 minutes to mount in kernels. After `upload-src` or `upload-results`, wait before re-pushing.
- **`datasets create` silently no-ops** on existing datasets. The CLI handles this by trying `datasets version` first; don't bypass that pattern.
- **Mount paths drift**: always `os.walk` to find `conformal_core.py` rather than hardcoding `/kaggle/input/<slug>/`.
- **GPU quota** is ~30h/week on free Kaggle. Set `gpu: false` in `config.yaml` for any notebook that doesn't need it.
- **Auth**: `pipeline.py` loads `KAGGLE_API_TOKEN` from `.env` automatically. If commands fail with 401/403, verify `.env` exists at repo root and contains a valid `KGAT_*` token.
- **Kaggle status endpoint (`GetKernelSessionStatus`) intermittently 500s.** When it does, `status`/`list` report `unknown`. This is *not* a kernel failure. `wait` handles it automatically: after ~90s of `unknown` it probes the independent `kernels output` endpoint to detect completion and returns 0 normally. If you ever call `status` directly and see `unknown`, just run `python pipeline.py fetch <phase>` — a successful fetch with the expected `[RESULT]` lines / `*_results.json` means the run succeeded. (Observed and hardened during first-run validation 2026-06-08.)
