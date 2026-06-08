"""
conformal-rag phase3 — generality check with a DIFFERENT model family.

Runs Phi-3.5-mini-instruct (3.8B, Microsoft) on BOTH TriviaQA and PopQA in one
kernel, producing one combined records file so we can confirm the conformal-gate
results (coverage dial, AUROC, adaptive efficiency) are not Qwen-specific.

Saves /kaggle/working/phase3_phi_records.json (records tagged by `source`,
PopQA rows carry log_pop).

P100-SAFE: torch 2.4.1+cu118 before importing torch; fp16; SDPA.
"""

import json
import os
import subprocess
import sys

print("Installing P100-compatible torch (2.4.1+cu118)...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.4.1", "torchvision==0.19.1",
                "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)

import torch
import numpy as np

print("torch", torch.__version__, "| device",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
assert torch.cuda.is_available(), "No CUDA device"
_ = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
print("  GPU probe OK", flush=True)

def _find_core(name):
    for dp, _, fs in os.walk("/kaggle/input"):
        if f"{name}.py" in fs:
            return dp
    return None
sys.path.insert(0, _find_core("conformal_core"))
import conformal_core as core

MODEL = "microsoft/Phi-3.5-mini-instruct"
N = 1000
K = 8
MAX_NEW = 24
TEMP, TOP_P = 0.7, 0.9
torch.manual_seed(0); np.random.seed(0)
print(f"PHASE 3: {MODEL} on TriviaQA + PopQA (N={N} each, K={K})")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- data loaders ----
def load_trivia(n):
    st = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                      split="validation", streaming=True)
    out = []
    for ex in st:
        a = ex.get("answer", {}) or {}
        g = []
        for k in ("aliases", "normalized_aliases"):
            if a.get(k):
                g += [str(x) for x in a[k]]
        if a.get("value"):
            g.append(str(a["value"]))
        g = list({x for x in g if x and x.strip()})
        if g:
            out.append({"question": ex["question"], "golds": g,
                        "source": "triviaqa", "log_pop": float("nan")})
        if len(out) >= n:
            break
    return out

def load_popqa(n):
    ds = load_dataset("akariasai/PopQA", split="test")
    pops = np.array([float(r) for r in ds["s_pop"]])
    idx = np.argsort(pops)[np.linspace(0, len(pops) - 1, n).astype(int)]
    out = []
    for i in idx:
        row = ds[int(i)]
        g = []
        for key in ("possible_answers", "s_aliases", "o_aliases"):
            v = row.get(key)
            if not v:
                continue
            try:
                p = json.loads(v) if isinstance(v, str) else v
                g += [str(x) for x in p] if isinstance(p, list) else [str(p)]
            except Exception:
                if isinstance(v, str):
                    g.append(v)
        g = list({x for x in g if x and x.strip()})
        if g:
            out.append({"question": row["question"], "golds": g, "source": "popqa",
                        "log_pop": float(np.log10(max(float(row["s_pop"]), 1.0)))})
    return out

rows = load_trivia(N) + load_popqa(N)
print(f"  loaded {len(rows)} questions", flush=True)

# ---- model ----
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
# Use transformers' NATIVE Phi3 impl (trust_remote_code=False): the model repo's
# custom modeling_phi3.py is stale and calls DynamicCache.from_legacy_cache,
# removed in current transformers. Native impl is compatible; eager attn for P100.
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager",
    trust_remote_code=False).to("cuda").eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
SYS = ("You are a helpful assistant. Answer the question with ONLY the short "
       "factual answer (a few words), no explanation.")

def prompt(q):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS},
         {"role": "user", "content": f"Question: {q}\nAnswer:"}],
        tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def greedy(p):
    inp = tok(p, return_tensors="pt").to("cuda")
    o = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                       return_dict_in_generate=True, output_scores=True,
                       pad_token_id=tok.pad_token_id)
    gen = o.sequences[0, inp.input_ids.shape[1]:]
    txt = tok.decode(gen, skip_special_tokens=True).strip()
    lps = [torch.log_softmax(s[0].float(), -1)[t].item() for t, s in zip(gen, o.scores)]
    return txt, (float(np.mean(lps)) if lps else -20.0), len(gen)

@torch.no_grad()
def sample(p, k):
    inp = tok(p, return_tensors="pt").to("cuda")
    o = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP,
                       top_p=TOP_P, num_return_sequences=k, pad_token_id=tok.pad_token_id)
    return [tok.decode(s[inp.input_ids.shape[1]:], skip_special_tokens=True).strip() for s in o]

records = []
for i, r in enumerate(rows):
    try:
        ans, lp, nt = greedy(prompt(r["question"]))
        smp = sample(prompt(r["question"]), K)
    except RuntimeError as e:
        print(f"  [skip {i}] {e}"); torch.cuda.empty_cache(); continue
    corr = core.is_correct(ans, r["golds"], mode="em")
    sc = core.self_consistency(smp)
    records.append({**{k: r[k] for k in ("question", "source", "log_pop")},
                    "gold": r["golds"], "answer": ans, "correct": bool(corr),
                    "wrong": bool(not corr), "f1": core.token_f1(ans, r["golds"]),
                    "semantic_entropy": core.semantic_entropy(smp),
                    "self_consistency": sc, "ignorance_1msc": 1.0 - sc,
                    "neg_mean_logprob": -lp, "answer_len": nt})
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(rows)}", flush=True)

wrong = np.array([r["wrong"] for r in records])
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1-y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s)+1)
    return float((rk[y==1].sum() - n1*(n1+1)/2)/(n1*n0))
core.format_result("experiment_complete", True)
core.format_result("n_records", len(records))
core.format_result("accuracy", round(float(1-wrong.mean()), 4))
core.format_result("auroc_semantic_entropy", round(auroc([r["semantic_entropy"] for r in records], wrong), 4))
core.format_result("auroc_neg_logprob", round(auroc([r["neg_mean_logprob"] for r in records], wrong), 4))
core.save_results({"records": records, "summary": {"model": MODEL, "n": len(records)}},
                  "/kaggle/working/phase3_phi_records.json")
print("\nDone.")
