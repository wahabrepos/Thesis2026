---
name: Self-MedRAG production pipeline state
description: Current state of the pipeline on Jetson Orin Nano Super, what works, what's next
type: project
---

## Pipeline status (as of 2026-03-22, pre-reboot)

Pipeline runs end-to-end. All 3 test queries pass (exit code 0).

**Why:** Getting real-data answers from Qwen2.5-1.5B on GPU with cached FAISS index.

**How to apply:** After reboot, run test mode and check if 4-bit GPU load succeeds.

---

## Config (config.yaml) — ready for next run

- `generator.model_name`: `Qwen/Qwen2.5-1.5B-Instruct`
- `generator.load_in_4bit`: `true`
- `corpus.max_docs`: `10000` (real PubMedQA abstracts)
- `corpus.path`: `./data/corpus/pubmed_abstracts.jsonl` (203k passages, first 10k used)

---

## After reboot — run this

```bash
cd ~/Downloads/Papers/Codes/self-medrag-production
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync venv/bin/python scripts/run_experiment.py --test-mode
```

Expected:
1. ~3.7 GiB CUDA free on fresh boot
2. Model tries 4-bit GPU (~0.75 GiB) → if NvMap fails, auto-falls back to float32 CPU
3. FAISS builds for 10k docs (~20-40 min first time) → cached to `cache/faiss/`
4. Subsequent runs: FAISS loads instantly from cache

---

## Key fixes applied (all in git-tracked source files)

| File | Fix |
|------|-----|
| `src/model.py` | 4-bit GPU path with OOM fallback to CPU; `@check_model_inputs` unwrap; `attn_implementation=eager` |
| `src/retrieval.py` | FAISS index saved/loaded from `cache/faiss/`; batch embedding (batch=8, max_len=256) |
| `src/pipeline.py` | Generator loads before retrieval (to claim GPU memory first) |
| `scripts/run_experiment.py` | `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` set before torch import |
| `config.yaml` | All paths, model names, thresholds tuned for Jetson |

---

## Known issues

- Qwen2.5-0.5B answers are incoherent (too small for JSON format) — need 1.5B on GPU
- 4-bit GPU load has historically failed with NvMapMemAllocInternalTagged (errno 12) — fresh reboot may fix this
- JSON parsing fails every generation → fallback extraction used (support scores still valid)
