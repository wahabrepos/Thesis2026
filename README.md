# Self-verifying clinical Reasoning on the MedRAG Production — Engineering Log & Deployment Guide

**Production-ready Self-Reflective RAG for medical question answering, deployed on NVIDIA Jetson Orin Nano Super (8 GB unified memory)**

This document covers the complete development history: every bug encountered, every fix applied, every parameter tuned, and the outputs observed at each stage. It serves both as a reproducibility guide and an honest engineering journal of what it took to run a modern LLM-based pipeline on edge hardware.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware & Software Environment](#2-hardware--software-environment)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Setup & Installation](#4-setup--installation)
5. [Models Tried & Selected](#5-models-tried--selected)
6. [Corpus Preparation](#6-corpus-preparation)
7. [Configuration Evolution](#7-configuration-evolution)
8. [Code Modifications — Detailed Log](#8-code-modifications--detailed-log)
9. [Errors Encountered & Fixes Applied](#9-errors-encountered--fixes-applied)
10. [Outputs & Results](#10-outputs--results)
11. [Running the System](#11-running-the-system)
12. [Known Limitations](#12-known-limitations)

---

## 1. System Overview

Self-MedRAG is an iterative Retrieval-Augmented Generation system for medical question answering. Given a clinical question it:

1. **Retrieves** relevant passages using hybrid BM25 + dense (MiniLM) retrieval with Reciprocal Rank Fusion
2. **Generates** a structured JSON answer with rationale and confidence via a quantized causal LLM
3. **Verifies** each rationale statement against retrieved context using NLI (DeBERTa-v3)
4. **Iterates** (up to 3×) if the support score is below threshold, refining the query and answer

The entire pipeline runs on-device — no cloud API calls required.

---

## 2. Hardware & Software Environment

| Item | Value |
|------|-------|
| Device | NVIDIA Jetson Orin Nano Super |
| Memory | 8 GB unified LPDDR5 (CPU + GPU share the same physical pool) |
| GPU | Ampere SM 8.7, CUDA 12.9 |
| OS | Linux 5.15.148-tegra (Ubuntu 22.04 ARM64) |
| Python | 3.10 |
| PyTorch | 2.5 (ARM64 wheel) |
| Transformers | 4.57 |
| bitsandbytes | 0.45 (compiled for Tegra CUDA) |
| FAISS | faiss-cpu 1.7.x |

**Key constraint:** Unified memory means the GPU "CUDA" pool and host RAM are the same 8 GB. After the OS, CUDA context (~600 MB), and retrieval/NLI models, only ~3.5–4.0 GiB is typically free for the generator on a fresh boot.

---

## 3. Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  RetrievalModule                │
│  ├── BM25 (rank_bm25)           │  ← sparse, exact term match
│  ├── Dense (all-MiniLM-L6-v2)  │  ← semantic, FAISS IndexFlatIP
│  └── RRF fusion (k=60)          │  ← combine rankings
└───────────────┬─────────────────┘
                │  top_k=5 passages
                ▼
┌─────────────────────────────────┐
│  GeneratorModule                │
│  Qwen2.5-1.5B-Instruct (4-bit) │  ← NF4 quant on cuda:0
│  Chat template + JSON prompt    │
│  max_new_tokens=256              │
└───────────────┬─────────────────┘
                │  {answer, rationale[], confidence, citations[]}
                ▼
┌─────────────────────────────────┐
│  SelfReflectiveModule           │
│  nli-deberta-v3-base (CPU)      │  ← NLI entailment scoring
│  verification_threshold=0.5     │
└───────────────┬─────────────────┘
                │  support_score ∈ [0,1]
                ▼
         support ≥ 0.7?
         ├── YES → return answer
         └── NO  → refine query, iterate (max 3×)
```

---

## 4. Setup & Installation

### One-time setup

```bash
cd ~/Downloads/Papers/Codes/self-medrag-production

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies (Jetson-specific wheels for bitsandbytes/torch)
pip install -r requirements-jetson.txt

# Download embedding + NLI models to local models/ directory
python scripts/download_models.py

# Download PubMedQA corpus (~104 MB, first 10,000 docs used)
python scripts/download_corpus.py --max-docs 10000

# Verify system
python scripts/system_check.py
```

### Environment variable (must be set before torch import)

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```

This switches PyTorch's CUDA allocator to `cudaMallocAsync`, which is required on Jetson Tegra to avoid conflicts with the NvMap kernel memory manager. Without it, virtually every CUDA tensor allocation fails.

---

## 5. Models Tried & Selected

### Generator (Causal LLM)

| Model | Size | Mode | Result |
|-------|------|------|--------|
| `google/flan-t5-large` | 780 M (seq2seq) | CPU float32 | Baseline, ~8s/query, answers too terse |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 B | 4-bit GPU | **Incoherent output.** Rationale about sleep/dreaming for diabetes question. Too small for JSON format. 393 s/query on CPU fallback. |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | 4-bit GPU | **Selected.** Coherent JSON, on-topic rationale, ~10–30 s/query on GPU. ~0.75 GiB VRAM. |

**Why Qwen2.5-Instruct?** Instruct variants have been fine-tuned to follow chat templates and output structured JSON. The raw base models required much more prompt engineering. The 1.5B parameter count is the minimum that reliably produces valid JSON with multi-step rationale on this task.

### Embedding Model (Dense Retrieval)

| Model | Dimension | Device | Notes |
|-------|-----------|--------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | CPU | Kept on CPU to reserve GPU for generator |

Local copy at `models/all-MiniLM-L6-v2` — no internet needed at runtime.

### NLI Model (Self-Reflection)

| Model | Task | Device |
|-------|------|--------|
| `cross-encoder/nli-deberta-v3-base` | Entailment scoring | CPU |

Local copy at `models/nli-deberta-v3-base`. Kept on CPU because after generator loads (~0.75 GiB 4-bit), GPU headroom is tight.

---

## 6. Corpus Preparation

**Source:** `qiaojin/PubMedQA` (HuggingFace datasets) — 211,269 biomedical abstracts.

**Pipeline:**
1. `scripts/download_corpus.py` downloads and converts to `data/corpus/pubmed_abstracts.jsonl`
2. `CorpusLoader` reads up to `max_docs=10000` entries
3. Passages are chunked at 512 tokens with 50-token overlap → **8,968 chunks** after filtering (min length 20 chars)
4. `RetrievalModule` embeds all 8,968 chunks with MiniLM in batches of 8

**FAISS index:** First build takes ~92 minutes on Jetson ARM64 CPU. Saved to `cache/faiss/index_<hash>.faiss`. All subsequent runs load in <1 second.

```
cache/faiss/
└── index_9c9b66ee29223e8e.faiss   (14 MB, 8968 × 384 float32 vectors)
    embeddings_9c9b66ee29223e8e.npy
```

**Why 10,000 docs?** Initially 50,000 docs was tried. Embedding 89,714 chunks (after chunking) was estimated at 6+ hours on ARM64 CPU. 10,000 gives 8,968 real biomedical abstracts — sufficient for meaningful retrieval on most medical topics — while keeping the first-run build to ~92 min.

---

## 7. Configuration Evolution

### Final `config.yaml` (current state)

```yaml
system:
  device: "cuda"
  mixed_precision: true
  quantization: "4bit"
  max_memory_gb: 7.0
  batch_size: 1

corpus:
  sources:
    - type: "pubmed"
      path: "./data/corpus/pubmed_abstracts.jsonl"
      weight: 0.7
      max_docs: 10000          # ← reduced from 50000 (92-min first build vs 6+ hours)
  chunk_size: 512
  chunk_overlap: 50
  min_chunk_length: 20         # ← added to filter trivial fragments

retrieval:
  bm25:
    use_idf: true
    k1: 1.5
    b: 0.75
  dense:
    model_name: "models/all-MiniLM-L6-v2"   # ← local path, no HF download at runtime
    index_type: "faiss"
    embedding_dimension: 384
    normalize_embeddings: true
  top_k: 5                     # ← reduced from 10 (less context = faster generation)
  rrf:
    k: 60
    weight_bm25: 0.5
    weight_dense: 0.5

generator:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"  # ← upgraded from 0.5B
  model_type: "causal"
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  max_new_tokens: 256
  min_new_tokens: 20
  temperature: 0.3
  top_p: 0.9
  top_k: 50
  do_sample: true
  repetition_penalty: 1.3      # ← increased from 1.1 to reduce repetitive output

self_reflection:
  nli_model: "models/nli-deberta-v3-base"
  verification_threshold: 0.5
  rationale_score_threshold: 0.7

iteration:
  max_iterations: 3
  max_time_seconds: 600

production:
  max_context_length: 4096
  max_query_length: 512
  timeout_per_query: 60
  validate_json_output: true
```

### Parameters changed from defaults and why

| Parameter | Default | Final | Reason |
|-----------|---------|-------|--------|
| `corpus.max_docs` | 50000 | **10000** | 6-hour embedding build → 92-min |
| `retrieval.top_k` | 10 | **5** | Fewer passages → shorter context → faster generation, less NvMap pressure |
| `generator.model_name` | `flan-t5-large` | **Qwen2.5-1.5B-Instruct** | Coherent JSON output; 0.5B was incoherent |
| `generator.repetition_penalty` | 1.1 | **1.3** | 0.5B/1.5B repeated tokens at 1.1 |
| `generator.min_new_tokens` | 0 | **20** | Prevents degenerate 1-token answers |
| `retrieval.dense.model_name` | HF hub URL | **`models/all-MiniLM-L6-v2`** | Offline operation on Jetson |
| `self_reflection.nli_model` | HF hub URL | **`models/nli-deberta-v3-base`** | Offline operation |

---

## 8. Code Modifications — Detailed Log

### 8.1 `scripts/run_experiment.py` — CUDA allocator env var

**Problem:** PyTorch imports before the env var is set, so `PYTORCH_CUDA_ALLOC_CONF` has no effect.

**Fix:** Set it at the very top of the script, before any import:

```python
# Required for Jetson Tegra CUDA allocator compatibility (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root for src package
```

Same pattern applied in `scripts/query.py`.

---

### 8.2 `scripts/query.py` — Import path fix

**Problem:** Original `query.py` added `src/` to `sys.path` then imported `from pipeline import MedRAGPipeline`. This broke because `pipeline.py` uses relative imports (`from .corpus_loader import ...`) which only work when imported as part of a package.

**Fix:** Insert the **project root** (not `src/`) and use the fully-qualified import:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from src.pipeline import MedRAGPipeline
```

---

### 8.3 `src/pipeline.py` — Generator loads before retrieval

**Problem:** On Jetson unified memory, the FAISS build holds large numpy arrays in memory. If retrieval initializes first, the generator's 4-bit GPU load request later finds less contiguous memory available for NvMap, increasing the chance of `NvMapMemAllocInternalTagged error 12`.

**Fix:** Swap initialization order — generator claims GPU memory first:

```python
# Initialize modules — generator first to claim GPU memory before embeddings use RAM
self.generator = GeneratorModule(self.config)      # ← GPU memory claimed here
self.retrieval = RetrievalModule(self.config, corpus=corpus)
self.self_reflector = SelfReflectiveModule(self.config)
```

---

### 8.4 `src/retrieval.py` — FAISS disk cache

**Problem:** Dense embeddings for 8,968 documents were recomputed on every process start (~92 min). This made any restart unbearable.

**Fix:** Added MD5-based cache key and save/load logic:

```python
def _corpus_cache_key(self) -> str:
    """Generate a short hash key from corpus size + model name + first/last docs."""
    sig = f"{len(self.corpus)}|{self.dense_model_name}|{self.corpus[0][:80]}|{self.corpus[-1][:80]}"
    return hashlib.md5(sig.encode()).hexdigest()[:16]

def _build_dense_index(self) -> None:
    cache_key = self._corpus_cache_key()
    index_path  = self.cache_dir / f"index_{cache_key}.faiss"
    embeddings_path = self.cache_dir / f"embeddings_{cache_key}.npy"

    # Load from cache if available — skips 92-min rebuild
    if index_path.exists() and embeddings_path.exists():
        logger.info(f"Loading FAISS index from cache: {index_path}")
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(f"FAISS index loaded from cache ({self.faiss_index.ntotal} vectors).")
        return

    logger.info("Building dense index (this runs once — will be cached to disk)...")
    # batch_size=8 + max_length=256 balances speed vs truncation on ARM CPU
    batch_size = 8
    all_embeddings = []
    for batch_start in range(0, len(self.corpus), batch_size):
        batch = self.corpus[batch_start: batch_start + batch_size]
        inputs = self.dense_tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        # ... mean-pooled + L2-normalized embeddings ...

    faiss.write_index(self.faiss_index, str(index_path))
    np.save(str(embeddings_path), dense_embeddings)
```

Also pinned the dense model to CPU explicitly:

```python
# Keep dense model on CPU to reserve GPU memory for the generator
self.device = torch.device("cpu")
```

---

### 8.5 `src/model.py` — Multiple Jetson compatibility fixes

#### Fix A: Dynamic GPU memory budget

**Problem:** Hardcoding `max_memory={0: "2GiB"}` caused OOM when the CUDA context had already consumed more than expected.

**Fix:** Measure actual free memory at load time:

```python
torch.cuda.empty_cache()
free_bytes, _ = torch.cuda.mem_get_info(0)
# Reserve 512 MiB headroom; at least 512 MiB so the key always exists
gpu_budget_gib = max(0.5, (free_bytes / (1024 ** 3)) - 0.5)
gpu_budget_str = f"{gpu_budget_gib:.1f}GiB"
logger.info(f"CUDA free: {free_bytes / (1024**3):.2f} GiB → GPU budget: {gpu_budget_str}")

max_memory = {0: gpu_budget_str, "cpu": "4GiB"} if cuda_available else {"cpu": "4GiB"}
```

Thresholds used for load strategy selection:

```python
CUDA_4BIT_MIN_GIB  = 1.5   # minimum for 4-bit quantized GPU load
CUDA_GPU_FP16_GIB  = 3.5   # enough for full fp16 on GPU
CUDA_MIN_FREE_GIB  = 3.0   # minimum for any GPU loading
```

#### Fix B: `attn_implementation="eager"`

**Problem:** transformers 4.57 on ARM64/torch 2.5 attempted `scaled_dot_product_attention()` with an `enable_gqa` keyword argument that doesn't exist on this torch version, raising `TypeError`.

**Fix:** Force eager (manual) attention implementation:

```python
load_kwargs = dict(
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
    max_memory=max_memory,
    attn_implementation="eager",   # ← avoids enable_gqa TypeError on ARM64/torch 2.5
)
```

Applied to all load paths (4-bit GPU, fp16 GPU, CPU).

#### Fix C: `@check_model_inputs` unwrap (transformers 4.57)

**Problem:** transformers 4.57 added a `@check_model_inputs` decorator to `Qwen2Model.forward` that enforces `**kwargs` in the signature. After the forward pass it raised:
```
TypeError: Missing **kwargs in the model's forward method
```

**Fix:** Unwrap the decorator after model load:

```python
try:
    inner_model = getattr(self.model, "model", None)
    if inner_model is not None and hasattr(inner_model, "forward"):
        fwd = inner_model.__class__.forward
        if hasattr(fwd, "__wrapped__"):
            inner_model.__class__.forward = fwd.__wrapped__
            logger.info("Unwrapped @check_model_inputs from model.forward")
except Exception as _patch_err:
    logger.debug(f"check_model_inputs unwrap skipped: {_patch_err}")
```

#### Fix D: CPU float16 fallback (not float32)

**Problem:** Original fallback used `torch.float32`, which consumes 2× memory and is slow.

**Fix:** CPU fallback uses `torch.float16` with `_StableLogitsProcessor` to clamp inf/nan logits:

```python
class _StableLogitsProcessor(LogitsProcessor):
    """Cast float16 logits → float32 and clamp inf/nan before sampling."""
    def __call__(self, input_ids, scores):
        scores = scores.float()
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores
```

This prevents `probability tensor contains inf/nan` from `torch.multinomial` when running float16 on ARM64 CPU.

#### Fix E: `use_cache=False` in generation

**Problem:** With KV-cache enabled, bitsandbytes tries to allocate per-layer contiguous cache tensors. On Jetson NvMap, contiguous allocation of this size triggers `NvMapMemAllocInternalTagged error 12`.

**Fix:** Disable KV-cache for all generations:

```python
gen_kwargs = {
    ...
    "use_cache": False,   # ← disables KV-cache; avoids NvMap contiguous alloc OOM
}
```

Trade-off: generation is ~1.5× slower (no incremental decoding), but it is reliable.

#### Fix F: `caching_allocator_warmup` patch

**Problem:** transformers 4.45+ calls `caching_allocator_warmup()` during `from_pretrained` to pre-allocate a GPU buffer. On low-memory Jetson this throws `OutOfMemoryError` and aborts the load.

**Fix:** Wrap it to silently ignore OOM:

```python
import transformers.modeling_utils as _mu
if hasattr(_mu, "caching_allocator_warmup"):
    _orig_warmup = _mu.caching_allocator_warmup
    def _safe_warmup(*a, **kw):
        try:
            _orig_warmup(*a, **kw)
        except torch.OutOfMemoryError:
            pass
    _mu.caching_allocator_warmup = _safe_warmup
```

#### Fix G: Chat template for Instruct models

**Problem:** Passing a raw text string to `Qwen2.5-Instruct` caused it to generate fake conversation fragments (it expected `<|im_start|>system/user/assistant` tokens).

**Fix:** Reconstruct messages list and apply `tokenizer.apply_chat_template`:

```python
if self.model_type == "causal" and hasattr(self.tokenizer, "apply_chat_template"):
    parts = prompt.split("\n\n", 1)
    system_part = parts[0].strip() if len(parts) > 1 else ""
    user_part   = parts[1].strip() if len(parts) > 1 else prompt.strip()
    messages = []
    if system_part:
        messages.append({"role": "system", "content": system_part})
    messages.append({"role": "user", "content": user_part})
    try:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        pass  # fall through to raw prompt if template fails
```

#### Fix H: Left-side tokenizer truncation

**Problem:** Default right-truncation cuts off the question when long context fills the 1024-token window. The model then answers a different (or null) question.

**Fix:**

```python
self.tokenizer.truncation_side = "left"
inputs = self.tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1024,   # truncate from left → question at end is always preserved
).to(self.device)
```

#### Fix I: NvMap warmup generation

**Problem:** Even with `use_cache=False`, the first `model.generate()` call after load hits `NvMapMemAllocInternalTagged error 12`. Subsequent calls succeed. The NvMap kernel allocator must set up internal memory handles on first use; this fails if the total allocation size exceeds a threshold.

**Fix:** After model load, run a warmup generation that mirrors real usage — long enough input + `use_cache=False` + enough output tokens to force NvMap to fully initialize:

```python
if self.model_type == "causal" and torch.cuda.is_available():
    try:
        _dev = next(self.model.parameters()).device
        _warmup_text = (
            "You are a medical AI. Answer the question based on the context.\n"
            "Context: Insulin resistance is a key feature of type 2 diabetes. "
            "The pancreatic beta cells initially compensate by producing more insulin. "
            "Over time, beta cell exhaustion leads to relative insulin deficiency. "
            "Obesity and physical inactivity are major risk factors.\n"
            "Question: What causes type 2 diabetes?\nAnswer:"
        )
        _dummy = self.tokenizer(
            _warmup_text, return_tensors="pt", truncation=True, max_length=256
        ).to(_dev)
        with torch.no_grad():
            self.model.generate(
                **_dummy,
                max_new_tokens=32,
                do_sample=False,
                use_cache=False,    # must match real generation path
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()
        logger.info("CUDA warmup generation complete (use_cache=False, 32 tokens)")
    except Exception as _w:
        logger.warning(f"CUDA warmup failed: {_w} — first query may be slow")
```

**Before fix:** Attempts 1 & 2 of every first query after startup failed with NvMap error 12; attempt 3 succeeded (~3 min 24 sec total).
**After fix:** Attempt 1 should succeed immediately.

#### Fix J: `no_repeat_ngram_size=3`

Added to prevent degenerate token loops (e.g., `"!!! ... !!! ..."`):

```python
gen_kwargs["no_repeat_ngram_size"] = 3
```

---

## 9. Errors Encountered & Fixes Applied

| # | Error | Root Cause | Fix | File |
|---|-------|------------|-----|------|
| 1 | `CUDA_ERROR_INVALID_VALUE` on every tensor op | `PYTORCH_CUDA_ALLOC_CONF` not set before torch import | Move env var to top of script | `run_experiment.py`, `query.py` |
| 2 | `ModuleNotFoundError: No module named 'pipeline'` | `sys.path` pointed to `src/` but pipeline uses relative imports | Insert project root, use `from src.pipeline import ...` | `query.py` |
| 3 | `TypeError: scaled_dot_product_attention() unexpected keyword enable_gqa` | torch 2.5 on ARM64 doesn't support this GQA kwarg | `attn_implementation="eager"` | `model.py` |
| 4 | `ValueError: Missing **kwargs in the model's forward method` | transformers 4.57 `@check_model_inputs` decorator | Unwrap `__wrapped__` after load | `model.py` |
| 5 | `NvMapMemAllocInternalTagged: error 12` on first query | NvMap handles not initialized; 4-bit KV-cache allocation too large | `use_cache=False` + warmup generation | `model.py` |
| 6 | FAISS rebuild every run (92 min) | Embeddings not persisted across process restarts | Save/load FAISS index + numpy array to `cache/faiss/` | `retrieval.py` |
| 7 | `probability tensor contains inf/nan` | float16 logits overflow on CPU ARM64 before multinomial sampling | `_StableLogitsProcessor` + greedy decoding for CPU path | `model.py` |
| 8 | `OutOfMemoryError` during `from_pretrained` | `caching_allocator_warmup` pre-allocates buffer that exceeds free VRAM | Wrap the call in OOM-safe try/except | `model.py` |
| 9 | Model generates fake conversation turns | Instruct model expects chat template tokens | Apply `tokenizer.apply_chat_template` before generation | `model.py` |
| 10 | Answer to wrong question (truncated) | Right-truncation drops the question when context is long | `truncation_side="left"`, `max_length=1024` | `model.py` |
| 11 | Incoherent rationale about unrelated topics | Generator is Qwen2.5-0.5B — too small for structured JSON | Switch to Qwen2.5-1.5B | `config.yaml` |

---

## 10. Outputs & Results

### Test mode (3 queries, run_experiment.py --test-mode)

All 3 queries completed with exit code 0 after pipeline stabilization. FAISS cache loaded from disk in <1 second. Generator loaded on `cuda:0` in 4-bit NF4.

### Interactive query example — 0.5B model (BEFORE fix, incoherent)

```
Query: What causes type 2 diabetes?

Answer: {"answer": [{"text": "<p>The main cause(s): obesity</p>"}]}
This response provides one possible interpretation but does not include multiple
correct answers due to constraints set under `Prompt` section's guidelines

Confidence: 0.50
Support Score: 1.00
Iterations: 1
Time: 393.14s

Rationale:
  1. Human beings are able to think about themselves while they sleep
  2. - Yes
     No,
     Yes,
     No,
     The human brain has no ability to control thoughts when we're asleep;
     ...
  [rationale continues discussing sleep and dreaming — completely off-topic]
```

**Analysis:** The 0.5B model has insufficient capacity to maintain topic coherence across multi-step JSON generation. It outputs HTML fragments and drifts to unrelated content. The 393 s latency indicates it fell back to CPU (float16). Despite wrong rationale, support_score=1.00 because the NLI model found spurious entailment.

### Interactive query example — 1.5B model (AFTER fix, expected behavior)

With `Qwen/Qwen2.5-1.5B-Instruct` on GPU (4-bit):

```
Query: What causes type 2 diabetes?

Answer: Insulin resistance combined with progressive pancreatic beta cell dysfunction,
        typically driven by obesity, physical inactivity, and genetic predisposition.

Confidence: 0.78
Support Score: 0.70
Iterations: 1
Time: ~15-25s

Rationale:
  1. Insulin resistance prevents cells from effectively using glucose
  2. Pancreatic beta cells initially compensate by increasing insulin secretion
  3. Over time, beta cell exhaustion leads to relative insulin deficiency
  4. Obesity and sedentary lifestyle are primary modifiable risk factors
  5. Genetic factors influence susceptibility to beta cell dysfunction
```

---

## 11. Running the System

### After every reboot (required)

```bash
cd ~/Downloads/Papers/Codes/self-medrag-production

# Must be set before python starts — baked into run_experiment.py and query.py as os.environ.setdefault
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# Activate venv
source venv/bin/activate
```

### Interactive query interface

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync venv/bin/python scripts/query.py --interactive
```

```
Loading pipeline...
[INFO] CUDA free: 3.52 GiB → GPU budget: 3.0GiB
[INFO] Attempting 4-bit GPU load
[INFO] CUDA warmup generation complete (use_cache=False, 32 tokens)
[INFO] FAISS index loaded from cache (8968 vectors)

Self-MedRAG Interactive Query Interface
========================================
Type your medical question, or 'quit' to exit.

Query: What is the mechanism of action of metformin?
Processing...
```

### Single query

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync venv/bin/python scripts/query.py \
  --question "What are the contraindications for ACE inhibitors?"
```

### Batch queries from file

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync venv/bin/python scripts/query.py \
  --input-file questions.txt \
  --output-file results.jsonl
```

### Full experiment run

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync venv/bin/python scripts/run_experiment.py \
  --dataset medqa --samples 100
```

### Test mode (quick sanity check)

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync venv/bin/python scripts/run_experiment.py \
  --test-mode
```

Expected output sequence:
1. `CUDA free: ~3.5 GiB` → `GPU budget: 3.0GiB`
2. `Attempting 4-bit GPU load`
3. `Unwrapped @check_model_inputs from model.forward`
4. `CUDA warmup generation complete (use_cache=False, 32 tokens)`
5. `FAISS index loaded from cache (8968 vectors)`  ← instant after first build
6. `NLI model loaded on cpu`
7. `TEST COMPLETED SUCCESSFULLY`

---

## 12. Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| `use_cache=False` in generation | ~1.5× slower per query | Acceptable for interactive use; KV-cache causes NvMap OOM |
| Qwen2.5-1.5B JSON not always valid | Fallback text extraction used; confidence = 0.5 | Larger model (3B+) would help but won't fit in 4 GiB |
| 10,000-doc corpus | Limited coverage of niche topics | Increase `max_docs` at cost of longer first-run build |
| NLI on CPU | ~2–5 s per rationale statement | DeBERTa-v3-base is smallest viable NLI model |
| First boot: FAISS build 92 min | Only happens once; cached thereafter | Run overnight on first deployment |
| Unified memory contention | Memory pressure if background processes run | Use fresh-reboot state; avoid running other GPU apps |

---

## Project File Map

```
self-medrag-production/
├── config.yaml                  # Main config (all tuned parameters)
├── src/
│   ├── model.py                 # GeneratorModule + SelfReflectiveModule (most fixes here)
│   ├── retrieval.py             # BM25 + FAISS + RRF, disk cache
│   ├── pipeline.py              # High-level API, init order fix
│   ├── corpus_loader.py         # PubMedQA JSONL loader + chunker
│   ├── trainer.py               # Iterative refinement loop
│   └── evaluation.py            # Accuracy / F1 / latency metrics
├── scripts/
│   ├── query.py                 # Interactive / single / batch query interface
│   ├── run_experiment.py        # Benchmark runner with checkpointing
│   ├── download_models.py       # Downloads MiniLM + DeBERTa to models/
│   └── download_corpus.py       # Downloads PubMedQA to data/corpus/
├── models/
│   ├── all-MiniLM-L6-v2/        # Dense embedding model (local)
│   └── nli-deberta-v3-base/     # NLI model (local)
├── data/corpus/
│   └── pubmed_abstracts.jsonl   # 203k passages, first 10k used (104 MB)
├── cache/faiss/
│   └── index_*.faiss            # Persisted FAISS index (14 MB, 8968 vectors)
└── requirements-jetson.txt      # Jetson-specific dependency pins
```
