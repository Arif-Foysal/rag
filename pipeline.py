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
