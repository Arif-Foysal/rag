# Pipeline Replication Guide — Read This and Build It

> **Instructions to Claude Code:** This file is a self-contained spec for a Claude-Code + Kaggle ML-Ops pipeline. When a human drops this file into a fresh repository and asks you to "set up the pipeline," your job is to:
>
> 1. Ask the human the **5 inputs** in §1.
> 2. Create all files listed in §3 with the exact content given (substituting placeholders with the human's answers).
> 3. Walk the human through the **first-run validation** in §4.
> 4. Hand off — from then on, your operating instructions live in the `CLAUDE.md` you just created.
>
> Do **not** start writing experiment code. Your job here is the harness, not the science. The human (or a follow-up session) will fill in `src/` and `templates/`.

---

## 1. Inputs to gather from the human

Before creating any files, ask the human (one question at a time, or batched — your call):

| # | Input | Example | Used for |
|---|---|---|---|
| 1 | **Project slug** (kebab-case, no spaces) | `stph`, `protein-fold`, `mri-seg` | Kernel/dataset naming: `<user>/<slug>-<phase>` |
| 2 | **Kaggle username** | `mdariffaysalnayem` | Owner of all kernels and datasets |
| 3 | **Core library module name** (snake_case) | `stph_core`, `fold_core` | The `.py` file in `src/` that all notebooks import |
| 4 | **Public dataset slug on Kaggle** (if any) | `khyeh0719/ptb-xl-dataset` | Auto-mounted into notebooks that opt in. Skip if synthetic-only. |
| 5 | **One-paragraph project description** | "Multi-lead ECG analysis with sheaf-theoretic persistent homology…" | Goes into `CLAUDE.md` so future sessions have context |

If the human can't answer any of these, stop and ask. Don't guess slugs or usernames.

The human must also have:
- A Kaggle account with an API token (`KGAT_*` from kaggle.com/settings) exported as `KAGGLE_API_TOKEN`, **or** `~/.kaggle/kaggle.json` in place.
- Python 3.10+ with `venv` available.

---

## 2. Architecture (so you understand what you're building)

```
        ┌──────────────┐
        │  Human (you) │  goals · gates · final calls
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │  Claude Code │  edits notebooks · interprets · iterates
        └──┬───────┬───┘
       push│       │fetch
        ┌──▼───────▼───┐
        │    Kaggle    │  runs notebooks · stores datasets
        └──────────────┘
```

**The loop:** `edit notebook → push → poll status → fetch → grep [RESULT] → fix or advance`. Everything is text (JSON, plain `.py`, greppable log lines) so an LLM can drive it without screen scraping.

**Key design choices (do not deviate without reason):**

1. **`src/` is uploaded as a single Kaggle dataset, not vendored.** Notebooks `import <core>` after locating the mounted dataset path. One source of truth; one `upload-src` to update everywhere.
2. **`templates/` is checked-in and pristine; `notebooks/` is mutable and gitignored.** Templates are the "blueprint"; notebooks are the working copy you edit. This keeps diffs tractable.
3. **Phase dependencies live in `config.yaml`, not in `pipeline.py`.** Each notebook lists which datasets it needs (raw + intermediate-results-from-prior-phases). The CLI reads this list when building `kernel-metadata.json`.
4. **Three result conventions:** `[RESULT] key = value` lines in stdout (greppable), `<phase>_results.json` saved to `/kaggle/working/`, and PNG plots. That's the contract that lets the agent judge "did this work."

---

## 3. Files to create

Create these in the project root unless noted. **Substitute** every `{{PROJECT_SLUG}}`, `{{KAGGLE_USERNAME}}`, `{{CORE_MODULE}}`, `{{PUBLIC_DATASET}}`, and `{{PROJECT_DESCRIPTION}}` with the human's answers from §1. If `{{PUBLIC_DATASET}}` is empty, leave the `raw_data` line out of `config.yaml` and the corresponding line out of the example template.

### 3.1 `requirements.txt`

```txt
kaggle>=1.6
pyyaml>=6.0
python-dotenv>=1.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
scikit-learn>=1.3
pandas>=2.0
```

Trim or extend based on what the project actually needs — most heavy deps (torch, wfdb, etc.) are pip-installed inside Kaggle notebooks rather than locally.

### 3.2 `.gitignore`

```
.venv/
__pycache__/
*.pyc
.env
.pipeline_runs.json
_kaggle_push/
_kaggle_dataset_*/
results/
notebooks/
```

The `notebooks/` ignore is intentional — those are mutable working copies. `templates/` is checked in. **Never commit `.env`** — it holds the Kaggle API token.

### 3.3 `config.yaml`

```yaml
# {{PROJECT_SLUG}} pipeline configuration.
# Edit before running. The pipeline.py CLI reads this on every command.

project:
  slug: "{{PROJECT_SLUG}}"          # kebab-case; used as kernel/dataset prefix

kaggle:
  username: "{{KAGGLE_USERNAME}}"

  # Datasets referenced by notebooks. The 'core' entry is created/updated by
  # `pipeline.py upload-src`. Phase-result entries are created by
  # `pipeline.py upload-results <phase>`. Add public dataset slugs as needed.
  datasets:
    core: "{{KAGGLE_USERNAME}}/{{PROJECT_SLUG}}-core"
    raw_data: "{{PUBLIC_DATASET}}"   # remove if no public dataset
    # Intermediate results, populated as phases complete:
    # phase1_results: "{{KAGGLE_USERNAME}}/{{PROJECT_SLUG}}-phase1-results"
    # phase2_results: "{{KAGGLE_USERNAME}}/{{PROJECT_SLUG}}-phase2-results"

  # Optional extra files bundled into the core dataset alongside src/*.py.
  # Useful for fixed inputs like train/val splits that all notebooks should see.
  core_extras: []
  # core_extras: ["results/splits_v1.json"]

# Pipeline control
pipeline:
  phases:
    phase1:
      description: "Synthetic / smoke validation"
      notebooks:
        - id: "phase1a"
          title: "First experiment"
          gpu: false
          timeout_hours: 3
          datasets: []                # core is always included; list extras here
          success_criteria:
            - "experiment_complete == true"

    # phase2:
    #   description: "Real-data preliminary"
    #   notebooks:
    #     - id: "phase2a"
    #       title: "..."
    #       gpu: false
    #       datasets: ["raw_data"]
    #     - id: "phase2b"
    #       gpu: true
    #       datasets: ["raw_data", "phase2a_results"]

  max_retries_per_notebook: 5
  poll_interval_seconds: 120
  auto_advance: false
```

**`datasets:` per notebook is the dependency graph.** Each string is a key into `kaggle.datasets`. The `core` dataset is always added automatically; you only list extras.

### 3.4 `.env` (not committed)

Hold the Kaggle API token here so it loads automatically on every run — no more `export KAGGLE_API_TOKEN=...` per session.

```
KAGGLE_API_TOKEN=KGAT_your_token_here
```

Get the token from https://www.kaggle.com/settings → "Create New Token". The `KGAT_*` format is the modern token; you do **not** need a `kaggle.json` file alongside it.

`pipeline.py` loads this file automatically via `python-dotenv` at startup, and the `kaggle` CLI subprocess inherits the env var. Make sure `.env` is in `.gitignore` (it is, per §3.2).

### 3.5 `pipeline.py`

This is the entire CLI — generic, config-driven, no project-specific logic. Copy verbatim.

```python
#!/usr/bin/env python3
"""
Kaggle ML-Ops Pipeline CLI
==========================
Drives the Kaggle-based experiment lifecycle. Reads config.yaml.

Commands:
    upload-src                  Upload src/*.py to Kaggle as the project's core dataset
    generate <phase_id>         Copy templates/<phase>.py to notebooks/<phase>.py
    push <notebook>             Push notebook to Kaggle
    status [kernel|phase]       One-line kernel state. Add --quiet to print
                                only the bare state token (e.g. "complete").
    wait <kernel|phase>         Block until the kernel is complete or errored,
                                printing one line only on terminal state.
                                Replaces an N-call polling loop with a single
                                command — keeps Claude's context lean.
    fetch <kernel|phase>        Download outputs to results/
    tail <kernel|phase>         Print the last N lines of the kernel log
                                (default 40). Use this instead of `cat` to
                                avoid blowing context on multi-MB Kaggle logs.
    results <phase_id>          Pretty-print result JSON + grep [RESULT] lines
    upload-results <phase_id>   Promote phase outputs to a Kaggle dataset
    list                        Same as status with no arg
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).parent
CONFIG_PATH = PIPELINE_DIR / "config.yaml"
NOTEBOOKS_DIR = PIPELINE_DIR / "notebooks"
RESULTS_DIR = PIPELINE_DIR / "results"
SRC_DIR = PIPELINE_DIR / "src"
TEMPLATES_DIR = PIPELINE_DIR / "templates"

# Load .env from the project root so KAGGLE_API_TOKEN doesn't need to be
# exported manually each session. python-dotenv is a no-op if .env is missing.
# override=True is REQUIRED — without it, a stale KAGGLE_API_TOKEN inherited
# from the shell (e.g. from another project) silently authenticates as the
# wrong account and `upload-src` lands in someone else's namespace.
try:
    from dotenv import load_dotenv
    load_dotenv(PIPELINE_DIR / ".env", override=True)
except ImportError:
    pass


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _kaggle_bin():
    # Prefer the kaggle binary that sits next to the active Python (i.e., the
    # venv's bin/), so `python pipeline.py ...` works whether or not the venv
    # is activated. Falls back to PATH lookup.
    candidate = Path(sys.executable).parent / "kaggle"
    if candidate.exists():
        return str(candidate)
    return "kaggle"


def kaggle_cmd(args, check=True, silent=False):
    cmd = [_kaggle_bin()] + args
    if not silent:
        print(f"  $ {' '.join(cmd)}")
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" in os.environ:
        env["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_API_TOKEN"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and result.returncode != 0:
        # Kaggle CLI 2.1+ writes API errors to stdout, not stderr — so check both.
        blob = (result.stdout or "") + "\n" + (result.stderr or "")
        print(f"  STDOUT: {result.stdout.strip()[:400]}")
        print(f"  STDERR: {result.stderr.strip()[:400]}")
        if "403" in blob or "401" in blob:
            print("\n  ERROR: Kaggle authentication failed.")
            print("  Set KAGGLE_API_TOKEN env var (KGAT_* token from kaggle.com/settings)")
            print("  OR put credentials in ~/.kaggle/kaggle.json")
        sys.exit(1)
    return result


def _looks_like_missing_dataset(result) -> bool:
    """Decide whether a failed `datasets version` call meant the dataset
    doesn't exist yet (so we should fall back to `datasets create`).

    Kaggle CLI 2.1+ writes errors to stdout, not stderr — and signals
    "no such dataset" with a 403 on the CreateDatasetVersion endpoint.
    """
    blob = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
    return any(s in blob for s in (
        "not found",
        "does not exist",
        "createdatasetversion",   # 403 on this endpoint => dataset missing
        "404",
    ))


def _project(config):
    return config["project"]["slug"]


def _expand_slug(config, kernel):
    """Accept either 'phase1a' or 'user/project-phase1a'; return the full slug."""
    if "/" in kernel:
        return kernel
    return f"{config['kaggle']['username']}/{_project(config)}-{kernel}"


def _find_notebook_config(config, stem):
    """Look up a notebook's config block by id (with _ or - normalization)."""
    norm = stem.replace("_", "-")
    for phase_name, phase in config["pipeline"]["phases"].items():
        for nb in phase["notebooks"]:
            if nb["id"] == stem or nb["id"].replace("_", "-") == norm:
                return phase_name, nb
    return None, None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_upload_src(config):
    project = _project(config)
    dataset_slug = config["kaggle"]["datasets"]["core"]
    dataset_dir = PIPELINE_DIR / "_kaggle_dataset_core"
    dataset_dir.mkdir(exist_ok=True)

    py_files = sorted(SRC_DIR.glob("*.py"))
    if not py_files:
        print(f"  Error: no .py files in {SRC_DIR}")
        sys.exit(1)
    for f in py_files:
        shutil.copy2(f, dataset_dir / f.name)
        print(f"  + {f.name}")

    for extra in config["kaggle"].get("core_extras", []) or []:
        p = PIPELINE_DIR / extra
        if p.exists():
            shutil.copy2(p, dataset_dir / p.name)
            print(f"  + {p.name} ({p.stat().st_size:,} bytes)")
        else:
            print(f"  Warning: core_extras path not found: {extra}")

    metadata = {
        "title": f"{project} core",
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nUploading core dataset: {dataset_slug}")
    # Try `version` first; only fall back to `create` on "not found".
    # Kaggle's `datasets create` silently no-ops on existing datasets, so
    # starting with version avoids that trap.
    result = kaggle_cmd(
        ["datasets", "version", "-p", str(dataset_dir),
         "-m", f"Updated {datetime.now().isoformat()}"],
        check=False,
    )
    if result.returncode != 0:
        if _looks_like_missing_dataset(result):
            print("  Dataset does not exist yet, creating...")
            kaggle_cmd(["datasets", "create", "-p", str(dataset_dir)])
        else:
            print(f"  Error stdout: {result.stdout}")
            print(f"  Error stderr: {result.stderr}")
            sys.exit(1)

    print(f"  Done: https://www.kaggle.com/datasets/{dataset_slug}")
    print("  NOTE: New versions take ~5 min to become available to kernels.")
    shutil.rmtree(dataset_dir)


def cmd_push(config, notebook_path):
    project = _project(config)
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        print(f"Error: {nb_path} not found")
        sys.exit(1)

    stem = nb_path.stem.replace("_", "-").lower()
    kernel_slug = f"{config['kaggle']['username']}/{project}-{stem}"

    if nb_path.suffix == ".py":
        kernel_type, language = "script", "python"
    elif nb_path.suffix == ".ipynb":
        kernel_type, language = "notebook", "python"
    else:
        print(f"Error: unsupported file type {nb_path.suffix}")
        sys.exit(1)

    _, nb_cfg = _find_notebook_config(config, stem)
    if nb_cfg is None:
        print(f"  Warning: '{stem}' not found in config.pipeline.phases; using defaults")
        nb_cfg = {}
    enable_gpu = "true" if nb_cfg.get("gpu", False) else "false"

    # Build dataset list. Core is always included.
    datasets_map = config["kaggle"]["datasets"]
    datasets = [datasets_map["core"]]
    for ds_key in nb_cfg.get("datasets", []) or []:
        if ds_key not in datasets_map:
            print(f"  Warning: dataset key '{ds_key}' not in config.kaggle.datasets; skipping")
            continue
        if datasets_map[ds_key]:
            datasets.append(datasets_map[ds_key])

    push_dir = PIPELINE_DIR / "_kaggle_push"
    push_dir.mkdir(exist_ok=True)
    shutil.copy2(nb_path, push_dir / nb_path.name)

    metadata = {
        "id": kernel_slug,
        "title": f"{project} {stem}",
        "code_file": nb_path.name,
        "language": language,
        "kernel_type": kernel_type,
        "is_private": "true",
        "enable_gpu": enable_gpu,
        "enable_internet": "true",
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(push_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPushing notebook: {nb_path.name}")
    print(f"  Kernel: {kernel_slug}")
    print(f"  GPU: {enable_gpu}")
    print(f"  Datasets: {datasets}")
    kaggle_cmd(["kernels", "push", "-p", str(push_dir)])
    print(f"\n  URL: https://www.kaggle.com/code/{kernel_slug}")
    print(f"  Status: python pipeline.py status {stem}")
    shutil.rmtree(push_dir)
    _record_run(kernel_slug, str(nb_path))


_TERMINAL_STATES = ("complete", "error", "cancelacknowledged", "cancelrequested")


def _kernel_state(config, kernel, silent=False):
    """Return a short state token like 'running' / 'complete' / 'error'.

    Parses the verbose kaggle CLI output once so callers don't have to.
    silent=True suppresses the per-call "$ kaggle ..." print line —
    important inside polling loops to keep agent-context tokens low.
    """
    slug = _expand_slug(config, kernel)
    result = kaggle_cmd(["kernels", "status", slug], check=False, silent=silent)
    raw = (result.stdout or "").strip()
    # Format: '<slug> has status "KernelWorkerStatus.RUNNING"'
    tail = raw.split("has status")[-1].strip().strip('"')
    state = tail.split(".")[-1].lower() if tail else "unknown"
    return slug, state


def cmd_status(config, kernel=None, quiet=False):
    if kernel:
        slug, state = _kernel_state(config, kernel)
        if quiet:
            print(state)
            return
        print(f"\n  {slug}: {state}")
        if state == "complete":
            print(f"  Fetch: python pipeline.py fetch {kernel}")
        elif state == "error":
            print(f"  Failed. Logs: https://www.kaggle.com/code/{slug}")
        return

    runs = _load_runs()
    if not runs:
        print("\nNo kernels tracked yet. Push one first.")
        return
    print(f"\n{'Kernel':<50} {'Status':<15} {'Pushed':<20}")
    print("-" * 85)
    for slug, info in runs.items():
        _, state = _kernel_state(config, slug)
        print(f"  {slug:<48} {state:<15} {info.get('pushed_at', '?'):<20}")


def _output_ready(config, kernel):
    """Liveness probe used when the status endpoint is unavailable.

    `kernels output` hits a *different* Kaggle endpoint than status, and only
    returns files once the kernel has finished. We download into a throwaway
    temp dir; if any file lands, the kernel is complete. Never raises — a
    failed/empty probe just means "not ready (or can't tell)" -> keep waiting.
    """
    import tempfile
    slug = _expand_slug(config, kernel)
    with tempfile.TemporaryDirectory() as td:
        kaggle_cmd(["kernels", "output", slug, "-p", td],
                   check=False, silent=True)
        try:
            return any(p.is_file() for p in Path(td).iterdir())
        except OSError:
            return False


def cmd_wait(config, kernel, interval=30, max_minutes=180, probe_after=3):
    """Poll kernel status from inside the CLI — one process, one final line.

    The agent-facing point: replace `N × pipeline.py status` (each pulling
    verbose CLI output into Claude's context) with one blocking call whose
    only output is the terminal state. Cuts polling-loop token cost ~Nx.

    Robustness: Kaggle's GetKernelSessionStatus endpoint intermittently
    returns 500, which surfaces here as state="unknown". A naive loop then
    spins for the whole timeout and a chained `wait && fetch` never fetches.
    Instead, once status has been "unknown" for `probe_after` polls (~90s at
    interval=30), we switch to probing the independent kernels-output endpoint
    to detect completion ourselves — so `wait` still returns 0 on success even
    with status fully down, and the chained fetch proceeds normally.
    """
    import time
    slug = _expand_slug(config, kernel)
    deadline = time.time() + max_minutes * 60
    last = None
    unknown_streak = 0
    while time.time() < deadline:
        _, state = _kernel_state(config, kernel, silent=True)
        if state != last:
            print(f"  {slug}: {state}", flush=True)
            last = state
        if state in _TERMINAL_STATES:
            return 0 if state == "complete" else 1
        if state == "unknown":
            unknown_streak += 1
            # Status endpoint is flaky; fall back to the output endpoint.
            if unknown_streak >= probe_after and _output_ready(config, kernel):
                print(f"  {slug}: complete (via output probe; "
                      f"status endpoint unavailable)", flush=True)
                return 0
        else:
            unknown_streak = 0
        time.sleep(interval)
    print(f"  {slug}: timeout after {max_minutes} min", flush=True)
    return 2


def cmd_fetch(config, kernel):
    slug = _expand_slug(config, kernel)
    slug_name = slug.split("/")[-1]
    output_dir = RESULTS_DIR / slug_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFetching: {slug}")
    print(f"  Output dir: {output_dir}")
    kaggle_cmd(["kernels", "output", slug, "-p", str(output_dir)])

    files = list(output_dir.iterdir())
    if files:
        print(f"\n  Downloaded {len(files)} file(s):")
        for f in files:
            print(f"    {f.name} ({f.stat().st_size:,} bytes)")
    else:
        print("  No output files found.")
    _parse_results(output_dir)


def cmd_results(config, phase_id):
    found = False
    for d in sorted(RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []:
        if not (d.is_dir() and phase_id in d.name):
            continue
        for f in d.iterdir():
            if f.suffix == ".json":
                found = True
                with open(f) as fh:
                    data = json.load(fh)
                print(f"\n--- {d.name}/{f.name} ---")
                _pretty_print(data)
        for f in d.iterdir():
            if f.suffix in (".log", ".txt", ""):
                _parse_file(f)
    if not found:
        print(f"\nNo results for phase: {phase_id}")
        print(f"  Look in {RESULTS_DIR} for downloaded outputs.")


def cmd_generate(config, phase_id):
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    template = TEMPLATES_DIR / f"{phase_id}.py"
    if template.exists():
        dest = NOTEBOOKS_DIR / f"{phase_id}.py"
        shutil.copy2(template, dest)
        print(f"\nCopied template: {dest}")
        print(f"To push: python pipeline.py push {dest}")
        return
    print(f"\nNo template at {template}")
    if TEMPLATES_DIR.exists():
        print("Available templates:")
        for f in sorted(TEMPLATES_DIR.glob("*.py")):
            print(f"  {f.stem}")


def cmd_tail(config, kernel, n=40):
    """Print the last N lines of the kernel log without `cat`-ing the whole
    multi-MB file. Always prefer this over reading log files directly into
    an LLM session — Kaggle logs are JSON-Lines and can be huge.
    """
    slug = _expand_slug(config, kernel)
    slug_name = slug.split("/")[-1]
    output_dir = RESULTS_DIR / slug_name
    if not output_dir.exists():
        print(f"  No fetched output for {kernel}. Run `pipeline.py fetch {kernel}` first.")
        return
    logs = sorted(output_dir.glob("*.log"))
    if not logs:
        print(f"  No .log file in {output_dir}")
        return
    log = logs[-1]
    lines = log.read_text(errors="replace").splitlines()
    print(f"--- tail -n {n} {log.name} ---")
    for line in lines[-n:]:
        print(line)


def cmd_upload_results(config, phase_id):
    project = _project(config)
    dataset_slug = f"{config['kaggle']['username']}/{project}-{phase_id}-results"
    result_dirs = [d for d in RESULTS_DIR.iterdir()
                   if d.is_dir() and phase_id in d.name] if RESULTS_DIR.exists() else []
    if not result_dirs:
        print(f"No results for {phase_id}")
        sys.exit(1)

    upload_dir = PIPELINE_DIR / f"_kaggle_dataset_{phase_id}_results"
    upload_dir.mkdir(exist_ok=True)
    for d in result_dirs:
        for f in d.iterdir():
            shutil.copy2(f, upload_dir / f"{d.name}_{f.name}")

    metadata = {
        "title": f"{project} {phase_id} results",
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(upload_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nUploading {phase_id} results: {dataset_slug}")
    result = kaggle_cmd(
        ["datasets", "version", "-p", str(upload_dir),
         "-m", f"Updated {datetime.now().isoformat()}"],
        check=False,
    )
    if result.returncode != 0:
        if _looks_like_missing_dataset(result):
            print("  Dataset does not exist yet, creating...")
            kaggle_cmd(["datasets", "create", "-p", str(upload_dir)])
        else:
            print(f"  Error stdout: {result.stdout}")
            print(f"  Error stderr: {result.stderr}")
            shutil.rmtree(upload_dir)
            sys.exit(1)

    shutil.rmtree(upload_dir)
    print(f"  Done: https://www.kaggle.com/datasets/{dataset_slug}")
    print("  NOTE: new versions take ~5 min to become available.")
    print(f"  Add to config.yaml: datasets.{phase_id}_results: \"{dataset_slug}\"")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_run(slug, notebook_path):
    runs = _load_runs()
    runs[slug] = {"notebook": notebook_path, "pushed_at": datetime.now().isoformat()}
    with open(PIPELINE_DIR / ".pipeline_runs.json", "w") as f:
        json.dump(runs, f, indent=2)


def _load_runs():
    p = PIPELINE_DIR / ".pipeline_runs.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _parse_results(d):
    for f in d.iterdir():
        _parse_file(f)


_MAX_RESULT_LINES = 200   # bound — Claude's context, not the human's screen


def _parse_file(f):
    try:
        text = f.read_text(errors="replace")
    except Exception:
        return
    lines = [l for l in text.splitlines() if "[RESULT]" in l]
    if not lines:
        return
    print(f"\n  [RESULT] lines from {f.name}:")
    for l in lines[:_MAX_RESULT_LINES]:
        print(f"    {l.strip()}")
    if len(lines) > _MAX_RESULT_LINES:
        print(f"    ... ({len(lines) - _MAX_RESULT_LINES} more — see {f})")


def _pretty_print(data, indent=4):
    p = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{p}{k}:")
            _pretty_print(v, indent + 4)
        elif isinstance(v, list) and len(v) > 5:
            print(f"{p}{k}: [{v[0]}, {v[1]}, ..., {v[-1]}] (len={len(v)})")
        else:
            print(f"{p}{k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kaggle ML-Ops Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=[
        "upload-src", "push", "status", "wait", "fetch", "tail", "list",
        "results", "generate", "upload-results",
    ])
    parser.add_argument("args", nargs="*", default=[])
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output (intended for `status` in agent loops).")
    args = parser.parse_args()

    config = load_config()
    if config["kaggle"]["username"] in ("YOUR_USERNAME", "", None):
        print("ERROR: edit config.yaml; set kaggle.username")
        sys.exit(1)

    def need(arg_name):
        if not args.args:
            print(f"Usage: python pipeline.py {args.command} <{arg_name}>")
            sys.exit(1)
        return args.args[0]

    if args.command == "upload-src":
        cmd_upload_src(config)
    elif args.command == "push":
        cmd_push(config, need("notebook_path"))
    elif args.command == "status":
        cmd_status(config, args.args[0] if args.args else None, quiet=args.quiet)
    elif args.command == "wait":
        sys.exit(cmd_wait(config, need("kernel_or_phase")))
    elif args.command == "fetch":
        cmd_fetch(config, need("kernel_or_phase"))
    elif args.command == "tail":
        n = int(args.args[1]) if len(args.args) > 1 else 40
        cmd_tail(config, need("kernel_or_phase"), n=n)
    elif args.command == "list":
        cmd_status(config, None)
    elif args.command == "results":
        cmd_results(config, need("phase_id"))
    elif args.command == "generate":
        cmd_generate(config, need("phase_id"))
    elif args.command == "upload-results":
        cmd_upload_results(config, need("phase_id"))


if __name__ == "__main__":
    main()
```

### 3.6 `src/{{CORE_MODULE}}.py`

A tiny stub that establishes the result-saving conventions. The human (or a follow-up session) fills in the project's actual math/feature code. Do not delete the helpers — every notebook calls them.

```python
"""
{{PROJECT_SLUG}} core library.

Uploaded as a Kaggle dataset by `pipeline.py upload-src`. All notebooks
import this module after locating it under /kaggle/input/.

Add project-specific functions below the helpers.
"""

import json
from pathlib import Path


def format_result(key, value, comment=None):
    """Print a greppable [RESULT] line and return the formatted string.

    Notebooks should call this for every metric so `pipeline.py fetch`
    can surface it without parsing free-form stdout.
    """
    line = f"[RESULT] {key} = {value}"
    if comment:
        line += f"  # {comment}"
    print(line)
    return line


def save_results(data: dict, path):
    """Write a results dict as JSON. Use /kaggle/working/<phase>_results.json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# Project-specific code goes below.
# ---------------------------------------------------------------------------
```

### 3.7 `templates/phase1a.py`

A runnable smoke-test template. The human can use this verbatim for a first push to validate the pipeline end-to-end before any real experiments.

```python
"""
{{PROJECT_SLUG}} phase1a — smoke test.

Validates the full pipeline: core dataset mounts, [RESULT] lines are
emitted, JSON results are saved. Replace the body with a real experiment
once the pipeline is verified.

Outputs to /kaggle/working/:
  - phase1a_results.json
"""

import os
import sys

# Locate the core library inside Kaggle's input mount. Mount paths vary,
# so walk to find it rather than hardcoding.
def _find_core(name):
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        if f"{name}.py" in filenames:
            return dirpath
    return None

_core_path = _find_core("{{CORE_MODULE}}")
if _core_path is None:
    print("ERROR: {{CORE_MODULE}}.py not found under /kaggle/input/")
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        for f in filenames:
            print(" ", os.path.join(dirpath, f))
    raise ImportError("{{CORE_MODULE}}.py not found")
sys.path.insert(0, _core_path)

import {{CORE_MODULE}} as core

# ---------------------------------------------------------------------------
# Smoke-test experiment: replace with the real thing.
# ---------------------------------------------------------------------------

print("=" * 60)
print("PHASE 1A: smoke test")
print("=" * 60)

import numpy as np
rng = np.random.default_rng(42)
x = rng.standard_normal(1000)

results = {
    "experiment_complete": True,
    "n_samples": int(x.size),
    "mean": float(x.mean()),
    "std": float(x.std()),
}

core.format_result("experiment_complete", results["experiment_complete"])
core.format_result("n_samples", results["n_samples"])
core.format_result("mean", f"{results['mean']:.4f}")
core.format_result("std", f"{results['std']:.4f}")

core.save_results(results, "/kaggle/working/phase1a_results.json")

print("\nDone.")
```

### 3.8 `CLAUDE.md`

Operating instructions for every future Claude Code session in this repo.

```markdown
# {{PROJECT_SLUG}} Pipeline — Operating Guide

## Project
{{PROJECT_DESCRIPTION}}

## Architecture
- `src/{{CORE_MODULE}}.py` — core library, uploaded to Kaggle as dataset `{{KAGGLE_USERNAME}}/{{PROJECT_SLUG}}-core`
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
4. `python pipeline.py wait <phase>` — single blocking call; prints one line per state change. Returns 0 on complete, non-zero on error/timeout.
5. `python pipeline.py fetch <phase>`
6. Read `results/<slug>/*.json` and grep `[RESULT]` lines to judge success. Use `python pipeline.py tail <phase>` for the last 40 log lines if you need to debug — never `cat` the full `.log`.

## Iteration protocol
When a notebook fails or success criteria aren't met:
- **Notebook bug** (syntax, shapes, logic) → edit `notebooks/<phase>.py`, re-push
- **Core math/lib bug** → edit `src/{{CORE_MODULE}}.py`, run `python pipeline.py upload-src`, **wait ~5 min**, re-push the notebook
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
- **Keep `pipeline.py`, `src/<core>.py`, `templates/` stable across iterations.** They are the cacheable prompt prefix; edit them only for durable changes. Iterate on `notebooks/<phase>.py`.
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
- **Mount paths drift**: always `os.walk` to find `{{CORE_MODULE}}.py` rather than hardcoding `/kaggle/input/<slug>/`.
- **GPU quota** is ~30h/week on free Kaggle. Set `gpu: false` in `config.yaml` for any notebook that doesn't need it.
- **Auth**: `pipeline.py` loads `KAGGLE_API_TOKEN` from `.env` automatically. If commands fail with 401/403, verify `.env` exists at repo root and contains a valid `KGAT_*` token.
```

### 3.9 `PIPELINE.md` (optional, user-facing)

If you want a human-facing quickstart separate from `CLAUDE.md`, create one summarizing §2 + §4. Skip if the human says they don't need it.

---

## 4. First-run validation

Once all files are in place, walk the human through this end-to-end. **Do not skip.** This catches auth, slug, and propagation issues before any real experiments.

```bash
# 1. Set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Create .env with your Kaggle token (one-time)
echo 'KAGGLE_API_TOKEN=KGAT_your_token_here' > .env
# Get the token from https://www.kaggle.com/settings if needed

# 3. Verify Kaggle auth works (pipeline.py loads .env automatically)
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('OK' if os.environ.get('KAGGLE_API_TOKEN') else 'MISSING')"
# Should print "OK"

# 4. Upload the core library as a Kaggle dataset
python pipeline.py upload-src
#    -> "Done: https://www.kaggle.com/datasets/<user>/<slug>-core"

# 5. WAIT ~5 minutes. New dataset versions don't propagate instantly.
sleep 300

# 6. Push the smoke-test notebook
cp templates/phase1a.py notebooks/phase1a.py
python pipeline.py push notebooks/phase1a.py

# 7. Wait for completion (single blocking call — emits one line per state change)
python pipeline.py wait phase1a
#    -> "<slug>: running" then "<slug>: complete"

# 8. Fetch results
python pipeline.py fetch phase1a
#    -> should print [RESULT] lines parsed from the kernel log

# 9. Verify results landed
ls results/<slug>-phase1a/
cat results/<slug>-phase1a/phase1a_results.json
```

If any of those steps fail, **fix the harness before writing real experiment code**. Common failures:

| Symptom | Cause | Fix |
|---|---|---|
| 401/403 on any kaggle command | `.env` missing, malformed, or token expired | Verify `.env` at repo root has `KAGGLE_API_TOKEN=KGAT_...` (no quotes, no spaces); regenerate token at kaggle.com/settings if needed |
| `upload-src` succeeds but `push` fails with "dataset not found" | Propagation lag | Wait 5 min and retry |
| Notebook fails with `ImportError: {{CORE_MODULE}} not found` | `os.walk` didn't find it | Check the dataset uploaded; re-run `upload-src` |
| `status` says "error" | Kernel runtime error | Open the kernel URL in browser; read the log; fix the notebook; re-push |
| `fetch` succeeds but no `[RESULT]` lines printed | Notebook didn't call `format_result` | Make sure every metric goes through `core.format_result(...)` |

---

## 5. After the harness works

The pipeline is now verified. Hand back to the human with this hand-off:

> The harness is in place and the smoke test passed. From here, the project-specific work is:
>
> 1. **Fill in `src/{{CORE_MODULE}}.py`** with the actual math / feature code shared across notebooks.
> 2. **Write real templates** in `templates/`. Use `templates/phase1a.py` as the boilerplate (the `os.walk` import block and `format_result`/`save_results` calls are required).
> 3. **Define your phases** in `config.yaml` — give each notebook a `gpu` flag, a `datasets:` list, and `success_criteria:` strings.
> 4. **Run the loop**: `cp template → push → status → fetch → interpret → fix or advance`.
>
> `CLAUDE.md` has the operating instructions for future sessions. Just point Claude Code at it.

---

## 6. Things that will bite you (memorize these)

- **Dataset propagation lag is real.** New versions take ~5 min. The CLI prints reminders; respect them.
- **`kaggle datasets create` silently succeeds when the dataset already exists** without uploading. Always try `datasets version` first and fall back to `create` only on "not found." The CLI does this — don't replace it with a "simpler" version.
- **Mount-path drift.** Kaggle sometimes mounts datasets at `/kaggle/input/<slug>/` and sometimes one level deeper. Always `os.walk` to find your file.
- **Free GPU quota** is ~30h/week. Set `gpu: false` aggressively. CPU-only smoke tests stay free forever.
- **Treat pushed kernels as immutable.** Re-pushing is fine; "amending" by editing a running kernel is not. Each push gets a new timestamp in `.pipeline_runs.json`.
- **Don't skip the smoke test** in §4 just because everything "looks right." The five-minute propagation lag has wasted more time than any other single thing.
- **Two ambient tokens silently send work to the wrong account.** If your shell already has a `KAGGLE_API_TOKEN` from another project, plain `load_dotenv()` won't override it — so `upload-src` will publish under the *wrong* user with no error. The provided `pipeline.py` uses `load_dotenv(..., override=True)`; do not "simplify" that flag away.
- **Kaggle CLI 2.1+ writes API errors to stdout, not stderr.** A 403 on `CreateDatasetVersion` (the "dataset doesn't exist yet" signal) appears in stdout. Any fallback logic that only inspects stderr will fail to fall through to `datasets create`. Use the `_looks_like_missing_dataset` helper that checks both streams plus the `CreateDatasetVersion` / `404` markers.
- **The `kaggle` binary is in the venv, not on system PATH.** `subprocess.run(["kaggle", ...])` only works if the venv is activated. The provided `pipeline.py` resolves it relative to `sys.executable` so it works either way.
- **Kaggle CLI cannot delete datasets.** No `kaggle datasets delete` exists; cleanup must be done in the web UI. So: never create probe / one-off datasets you'll later regret — always commit to a real slug.
- **The status endpoint (`GetKernelSessionStatus`) intermittently 500s** — sometimes for an entire run, even though the kernel itself completes fine. When it does, `kernels status` returns nothing, `_kernel_state` reports `"unknown"`, and a naive `wait` loop spins until timeout — so a chained `wait && fetch` never fetches. The provided `cmd_wait` handles this by falling back to `_output_ready` (a probe against the *independent* `kernels output` endpoint) after a short streak of `unknown`, and returns 0 on detected completion. Do not "simplify" `cmd_wait` back to a status-only loop, and do not chain `fetch` in a way that's skipped when `wait` is non-zero — the `kernels output` endpoint is the reliable completion signal, not status.

---

## 6.5. Token-cost discipline for the agent loop

The whole point of the harness is that an LLM (Claude Code) drives it. Every byte the harness emits ends up in Claude's context window — and prompt-caching only helps the *static* part of context. Treat agent-facing output as a budget.

**Rules the CLI already enforces (keep them):**
- **`pipeline.py wait <kernel>`** — single blocking call that prints one line per state change instead of N noisy `kaggle kernels status` invocations. Use this in agent loops.
- **`pipeline.py status --quiet`** — emits just the bare state token (`running`/`complete`/`error`). Bash-friendly, agent-friendly.
- **`pipeline.py tail <kernel> [N]`** — last N lines of the kernel log (default 40). Never `cat` a Kaggle log into Claude's context; logs are JSON-Lines and routinely exceed 1 MB.
- **`_MAX_RESULT_LINES = 200` cap in `_parse_file`** — bounds how many `[RESULT]` lines any single command surfaces.

**Rules the agent (you, Claude) must follow:**
- The **only** signals you should pull into context for "did this work?" are: (a) the JSON file under `results/<slug>/*.json`, (b) the grepped `[RESULT]` lines, (c) the kernel state token from `wait`. Do not Read raw `.log` files; use `tail` if you need recent log lines.
- **Don't poll.** If you find yourself running `pipeline.py status` more than twice in a row, switch to `pipeline.py wait`. Each poll cycle re-pays the prompt-cache miss for everything after the most recent stable prefix.
- **Keep CLAUDE.md, src/, templates/ stable across iterations.** They form the cacheable prefix of every session in this repo. Edit them only when the change is durable; don't rewrite them per experiment.
- **Edit notebooks, not the CLI.** `pipeline.py` is fixed; project iteration happens in `notebooks/<phase>.py`. Keeping the CLI pinned keeps the cached prefix valid.
- **Treat success criteria as boolean greps.** `success_criteria` strings in `config.yaml` should be parseable from `[RESULT]` lines (e.g. `experiment_complete == true`). The agent decides advance/retry by grepping, not by reading the whole notebook output.
- **Don't print large arrays.** `_pretty_print` already truncates lists >5 elements; respect that pattern in any custom result helpers you add to `src/<core>.py`.

The cost model: a single tight loop of *push → wait → fetch → grep [RESULT] → decide* should be ~1k tokens per iteration of *new* context (most of the prefix prompt-caches). A loop that polls every 30 s, reads full logs, and re-prints the kaggle CLI invocation 12 times is ~30k tokens for the same information.

---

## 7. What this guide is NOT

- A research methodology guide. The phases (`phase1a`, `phase2a`, …) are placeholders — your project picks its own stage names.
- A model definition. `src/{{CORE_MODULE}}.py` ships empty; that's where the actual work goes.
- A replacement for Kaggle docs. If you need to do something exotic with the `kaggle` CLI, read its docs; this guide only covers the eight commands the harness uses.

The point is the **harness** — a small, config-driven CLI plus three conventions (`[RESULT]` lines, `_results.json`, PNG plots) that let an LLM drive a remote compute backend with no human in the loop except at gates.
