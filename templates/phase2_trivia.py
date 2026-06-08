"""
conformal-rag phase2 — real LLM uncertainty signals on TriviaQA.

Same protocol as phase2_popqa.py but on TriviaQA (general-knowledge questions
the model often KNOWS). Provides the "popular / known" population for:
  - a real efficiency story (the gate can skip a lot when the model knows),
  - a dramatic cross-distribution shift vs PopQA long-tail for the
    group-conditional (Mondrian) headline.

Saves /kaggle/working/phase2_trivia_records.json.

P100-SAFE: installs torch 2.4.1+cu118 (retains sm_60) before importing torch;
Kaggle's preinstalled torch 2.10+cu128 dropped Pascal support.
"""

import json
import os
import subprocess
import sys

print("Installing P100-compatible torch (2.4.1+cu118, retains sm_60)...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.4.1", "torchvision==0.19.1",
                "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)

import torch
import numpy as np

print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      flush=True)
assert torch.cuda.is_available(), "No CUDA device"
_probe = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
print(f"  GPU matmul probe OK ({_probe:.1f}) — sm_60 kernels work", flush=True)

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
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_QUESTIONS = 1000
K_SAMPLES = 8
MAX_NEW = 24
TEMP = 0.7
TOP_P = 0.9
SEED = 0

print("=" * 60)
print(f"PHASE 2: {MODEL} on TriviaQA  (N={N_QUESTIONS}, K={K_SAMPLES})")
print("=" * 60)
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data: TriviaQA (streaming to avoid the full multi-GB download)
# ---------------------------------------------------------------------------
from datasets import load_dataset

stream = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                      split="validation", streaming=True)
rows = []
for ex in stream:
    ans = ex.get("answer", {}) or {}
    golds = []
    for k in ("aliases", "normalized_aliases"):
        v = ans.get(k)
        if v:
            golds.extend([str(x) for x in v])
    if ans.get("value"):
        golds.append(str(ans["value"]))
    golds = list({g for g in golds if g and g.strip()})
    if not golds:
        continue
    rows.append({"question": ex["question"], "golds": golds})
    if len(rows) >= N_QUESTIONS:
        break
print(f"  TriviaQA questions: {len(rows)}")

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
    lps = []
    for tkn, score in zip(gen, out.scores):
        logp = torch.log_softmax(score[0].float(), dim=-1)
        lps.append(logp[tkn].item())
    return text, (float(np.mean(lps)) if lps else -20.0), len(gen)

@torch.no_grad()
def sample_answers(prompt, k):
    inp = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=True,
                         temperature=TEMP, top_p=TOP_P, num_return_sequences=k,
                         pad_token_id=tok.pad_token_id)
    return [tok.decode(seq[inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
            for seq in out]

# ---------------------------------------------------------------------------
records = []
t_correct = 0
for i, row in enumerate(rows):
    q, golds = row["question"], row["golds"]
    try:
        ans, mean_lp, n_tok = greedy_with_logprob(build_prompt(q))
        samples = sample_answers(build_prompt(q), K_SAMPLES)
    except RuntimeError as e:
        print(f"  [skip {i}] {e}"); torch.cuda.empty_cache(); continue
    correct = core.is_correct(ans, golds, mode="em")
    se = core.semantic_entropy(samples); sc = core.self_consistency(samples)
    t_correct += int(correct)
    records.append({
        "question": q, "gold": golds, "answer": ans,
        "correct": bool(correct), "wrong": bool(not correct),
        "f1": core.token_f1(ans, golds),
        "semantic_entropy": se, "self_consistency": sc, "ignorance_1msc": 1.0 - sc,
        "neg_mean_logprob": -mean_lp, "answer_len": n_tok,
        "source": "triviaqa", "log_pop": float("nan"),
    })
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(rows)}  running acc={t_correct/len(records):.3f}")

# ---------------------------------------------------------------------------
n = len(records)
wrong = np.array([r["wrong"] for r in records])

def auroc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    return float((ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))

core.format_result("experiment_complete", True)
core.format_result("n_records", n)
core.format_result("accuracy", round(float(1 - wrong.mean()), 4))
core.format_result("auroc_semantic_entropy",
                   round(auroc([r["semantic_entropy"] for r in records], wrong), 4))
core.format_result("auroc_neg_logprob",
                   round(auroc([r["neg_mean_logprob"] for r in records], wrong), 4))
core.save_results({"records": records,
                   "summary": {"n": n, "accuracy": float(1 - wrong.mean()),
                               "model": MODEL, "dataset": "triviaqa"}},
                  "/kaggle/working/phase2_trivia_records.json")
print("\nDone.")
