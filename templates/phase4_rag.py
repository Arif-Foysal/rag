"""
conformal-rag phase4 — end-to-end RAG outcomes on PopQA.

For the SAME 1000 PopQA questions as phase2 (deterministic reload), retrieve a
Wikipedia context and produce the open-book (RAG) answer + correctness. Fused
locally with phase2 closed-book signals (analysis/endtoend.py) to measure, for
the conformal gate vs always/never/fixed-threshold:
    final answer accuracy  vs  retrieval fraction.

Retriever: MediaWiki API. Query = subject Wikipedia title (s_wiki_title) with a
question-text search fallback -> a strong, realistic single-entity retriever
(PopQA is single-entity QA). The model must still EXTRACT the answer from the
page, so this is genuine RAG, not gold injection.

Saves /kaggle/working/phase4_rag_records.json.
P100-SAFE: torch 2.4.1+cu118 before import; fp16; SDPA.
"""

import json
import os
import subprocess
import sys
import time

print("Installing P100-compatible torch (2.4.1+cu118)...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.4.1", "torchvision==0.19.1",
                "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)

import numpy as np
import requests
import torch

print("torch", torch.__version__, "| device",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
assert torch.cuda.is_available(), "No CUDA device"
_ = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
print("  GPU probe OK", flush=True)

def _find_core(name):
    for dp, _, fs in os.walk("/kaggle/input"):
        if f"{name}.py" in fs:
            return dp
sys.path.insert(0, _find_core("conformal_core"))
import conformal_core as core

MODEL = "Qwen/Qwen2.5-3B-Instruct"   # MUST match phase2 (closed-book signals)
N_QUESTIONS = 1000
MAX_NEW = 24
MAX_CTX_CHARS = 1500
torch.manual_seed(0); np.random.seed(0)
print(f"PHASE 4 RAG: {MODEL} on PopQA (N={N_QUESTIONS})")

# ---- reproduce the EXACT phase2 question set ----
from datasets import load_dataset
ds = load_dataset("akariasai/PopQA", split="test")
pops = np.array([float(r) for r in ds["s_pop"]])
order = np.argsort(pops)
idx = order[np.linspace(0, len(order) - 1, N_QUESTIONS).astype(int)]
rows = [ds[int(i)] for i in idx]
print(f"  reproduced {len(rows)} PopQA questions")

def gold_answers(row):
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
    return list({x for x in g if x and x.strip()})

# ---- retriever (MediaWiki API) ----
# Lesson from a failed run: exact-title extracts fail on PopQA's disambiguated
# titles (e.g. "Valea Seacă River (Mara)"), and a 0.03s burst gets rate-limited
# (silently empty). Fix: SEARCH by title/subject (resolves disambiguation) and
# pace requests at ~0.4s. Validated locally at 25/25 context hit, 21/25 answer
# present in context.
S = requests.Session()
S.headers.update({"User-Agent": "conformal-rag-research/1.0 (academic; mailto wayne11nayem@gmail.com)"})
API = "https://en.wikipedia.org/w/api.php"

def _search_title(query):
    try:
        r = S.get(API, params={"action": "query", "list": "search", "srsearch": query,
                               "format": "json", "srlimit": 1}, timeout=20)
        h = r.json().get("query", {}).get("search", [])
        return h[0]["title"] if h else None
    except Exception:
        return None

def _extract(title):
    try:
        r = S.get(API, params={"action": "query", "prop": "extracts", "explaintext": 1,
                               "titles": title, "format": "json", "redirects": 1}, timeout=20)
        for _, p in r.json()["query"]["pages"].items():
            return p.get("extract", "")
    except Exception:
        pass
    return ""

def retrieve(s_wiki_title, subj, question):
    for q in (s_wiki_title, subj, question):
        if not q:
            continue
        t = _search_title(q)
        if t:
            ext = _extract(t)
            if ext:
                return ext[:MAX_CTX_CHARS]
    return ""

# ---- model ----
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

SYS = ("You are a helpful assistant. Use the provided context if relevant. Answer "
       "with ONLY the short factual answer (a few words), no explanation.")

def rag_prompt(q, ctx):
    user = (f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:" if ctx
            else f"Question: {q}\nAnswer:")
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def greedy(p):
    inp = tok(p, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    o = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

records, n_ctx, t_correct = [], 0, 0
for i, row in enumerate(rows):
    q = row["question"]; golds = gold_answers(row)
    if not golds:
        continue
    ctx = retrieve(row.get("s_wiki_title"), row.get("subj"), q)
    if not ctx:                       # one retry after a longer pause (anti-throttle)
        time.sleep(1.0)
        ctx = retrieve(row.get("s_wiki_title"), row.get("subj"), q)
    time.sleep(0.4)
    n_ctx += int(bool(ctx))
    try:
        ans = greedy(rag_prompt(q, ctx))
    except RuntimeError as e:
        print(f"  [skip {i}] {e}"); torch.cuda.empty_cache(); continue
    corr = core.is_correct(ans, golds, mode="em")
    t_correct += int(corr)
    records.append({"question": q, "gold": golds, "rag_answer": ans,
                    "rag_correct": bool(corr), "rag_f1": core.token_f1(ans, golds),
                    "has_context": bool(ctx), "ctx_len": len(ctx),
                    "s_pop": float(row["s_pop"])})
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(rows)}  rag_acc={t_correct/len(records):.3f}  "
              f"ctx_hit={n_ctx/(i+1):.2f}", flush=True)

n = len(records)
rag_acc = float(np.mean([r["rag_correct"] for r in records]))
core.format_result("experiment_complete", True)
core.format_result("n_records", n)
core.format_result("rag_accuracy", round(rag_acc, 4))
core.format_result("context_hit_rate", round(n_ctx / max(len(rows), 1), 4))
core.save_results({"records": records,
                   "summary": {"model": MODEL, "n": n, "rag_accuracy": rag_acc,
                               "context_hit_rate": n_ctx / max(len(rows), 1)}},
                  "/kaggle/working/phase4_rag_records.json")
print("\nDone.")
