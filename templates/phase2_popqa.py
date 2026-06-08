"""
conformal-rag phase2 — real LLM uncertainty signals on PopQA.

Generates, per question (closed-book, NO retrieval):
  - greedy answer + correctness (EM / F1 vs gold aliases)  -> `wrong` label
  - ignorance features: semantic_entropy, 1-self_consistency, neg_mean_logprob,
    answer_len
  - subject popularity s_pop  -> novelty bucket for group-conditional conformal

Saves per-example records to /kaggle/working/phase2_records.json for local
conformal analysis. PopQA's popularity column is the real long-tail axis the
group-conditional (Mondrian) headline needs.

P100-SAFE: fp16 (no bf16), attn_implementation="sdpa" (no flash-attn). See
the kaggle-p100-gpu-constraints note.
"""

import json
import os
import subprocess
import sys

# CRITICAL (P100/sm_60): Kaggle's API runners ship torch 2.10+cu128, which has
# DROPPED Pascal (sm_60) support — so the P100 Kaggle assigns is unusable with
# the preinstalled torch ("no kernel image is available for execution on the
# device"). Install a P100-compatible torch BEFORE importing torch: cu118 wheels
# still bundle sm_60 kernels, and Kaggle's driver is backward-compatible.
print("Installing P100-compatible torch (2.4.1+cu118, retains sm_60)...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.4.1", "torchvision==0.19.1",
                "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)

import torch
import numpy as np

# Fail-fast GPU sanity check — prove sm_60 kernels actually run before we spend
# time downloading a model and looping over 1000 questions.
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      flush=True)
assert torch.cuda.is_available(), "No CUDA device"
_probe = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
print(f"  GPU matmul probe OK ({_probe:.1f}) — sm_60 kernels work", flush=True)

# ---- locate core lib (mount path drifts) ----
def _find_core(name):
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        if f"{name}.py" in filenames:
            return dirpath
    return None

_cp = _find_core("conformal_core")
if _cp is None:
    raise ImportError("conformal_core.py not found under /kaggle/input/")
sys.path.insert(0, _cp)
import conformal_core as core

# ---------------------------------------------------------------------------
# Config (tune for quota; P100 has 16 GB, ~9 h kernel limit)
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_QUESTIONS = 1000          # stratified across popularity
K_SAMPLES = 8               # samples for semantic entropy / self-consistency
MAX_NEW = 24
TEMP = 0.7
TOP_P = 0.9
SEED = 0

print("=" * 60)
print(f"PHASE 2: {MODEL} on PopQA  (N={N_QUESTIONS}, K={K_SAMPLES})")
print("=" * 60)
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data: PopQA (single dataset, real popularity -> novelty buckets)
# ---------------------------------------------------------------------------
from datasets import load_dataset

ds = load_dataset("akariasai/PopQA", split="test")
print(f"  PopQA rows: {len(ds)}")

# Stratify by popularity so we get both popular and long-tail subjects.
pops = np.array([float(r) for r in ds["s_pop"]])
order = np.argsort(pops)                       # ascending popularity
idx = order[np.linspace(0, len(order) - 1, N_QUESTIONS).astype(int)]
rows = [ds[int(i)] for i in idx]

def gold_answers(row):
    g = []
    for key in ("possible_answers", "s_aliases", "o_aliases"):
        v = row.get(key)
        if not v:
            continue
        try:
            parsed = json.loads(v) if isinstance(v, str) else v
            if isinstance(parsed, list):
                g.extend([str(x) for x in parsed])
            elif parsed:
                g.append(str(parsed))
        except Exception:
            if isinstance(v, str):
                g.append(v)
    # de-dup, drop empties
    return list({x for x in g if x and x.strip()})

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="sdpa",
).to("cuda").eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

SYS = ("You are a helpful assistant. Answer the question with ONLY the short "
       "factual answer (a few words), no explanation.")

def build_prompt(q):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Question: {q}\nAnswer:"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def greedy_with_logprob(prompt):
    inp = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                         return_dict_in_generate=True, output_scores=True,
                         pad_token_id=tok.pad_token_id)
    gen = out.sequences[0, inp.input_ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    # mean token log-prob (compute in fp32 for stability)
    lps = []
    for tkn, score in zip(gen, out.scores):
        logp = torch.log_softmax(score[0].float(), dim=-1)
        lps.append(logp[tkn].item())
    mean_lp = float(np.mean(lps)) if lps else -20.0
    n_tok = len(gen)
    return text, mean_lp, n_tok

@torch.no_grad()
def sample_answers(prompt, k):
    inp = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=True,
                         temperature=TEMP, top_p=TOP_P, num_return_sequences=k,
                         pad_token_id=tok.pad_token_id)
    texts = []
    for seq in out:
        g = seq[inp.input_ids.shape[1]:]
        texts.append(tok.decode(g, skip_special_tokens=True).strip())
    return texts

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
records = []
t_correct = 0
for i, row in enumerate(rows):
    q = row["question"]
    golds = gold_answers(row)
    if not golds:
        continue
    try:
        ans, mean_lp, n_tok = greedy_with_logprob(build_prompt(q))
        samples = sample_answers(build_prompt(q), K_SAMPLES)
    except RuntimeError as e:
        print(f"  [skip {i}] {e}")
        torch.cuda.empty_cache()
        continue

    correct = core.is_correct(ans, golds, mode="em")
    f1 = core.token_f1(ans, golds)
    se = core.semantic_entropy(samples)
    sc = core.self_consistency(samples)
    t_correct += int(correct)

    records.append({
        "question": q,
        "gold": golds,
        "answer": ans,
        "correct": bool(correct),
        "wrong": bool(not correct),
        "f1": f1,
        "semantic_entropy": se,
        "self_consistency": sc,
        "ignorance_1msc": 1.0 - sc,
        "neg_mean_logprob": -mean_lp,
        "answer_len": n_tok,
        "s_pop": float(row["s_pop"]),
        "log_pop": float(np.log10(max(float(row["s_pop"]), 1.0))),
    })
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(rows)}  running acc={t_correct/len(records):.3f}")

# ---------------------------------------------------------------------------
# Summary diagnostics: is the ignorance signal real?
# ---------------------------------------------------------------------------
n = len(records)
wrong = np.array([r["wrong"] for r in records])
acc = 1.0 - wrong.mean()

def auroc(score, label):
    # AUROC of `score` predicting label==1 (wrong). Rank-based.
    s = np.asarray(score, float); y = np.asarray(label, int)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

se_all = np.array([r["semantic_entropy"] for r in records])
lp_all = np.array([r["neg_mean_logprob"] for r in records])
sc_all = np.array([r["ignorance_1msc"] for r in records])

core.format_result("experiment_complete", True)
core.format_result("n_records", n)
core.format_result("accuracy", round(acc, 4))
core.format_result("auroc_semantic_entropy", round(auroc(se_all, wrong), 4))
core.format_result("auroc_1msc", round(auroc(sc_all, wrong), 4))
core.format_result("auroc_neg_logprob", round(auroc(lp_all, wrong), 4))

core.save_results({"records": records,
                   "summary": {"n": n, "accuracy": acc, "model": MODEL,
                               "n_questions": N_QUESTIONS, "k": K_SAMPLES}},
                  "/kaggle/working/phase2_records.json")
print("\nDone.")
