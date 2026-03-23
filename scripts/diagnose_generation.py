"""
Minimal diagnostic: determines exactly WHY the model generates '!!!!!!'.
Tests three hypotheses in order:
  H1 — logits are NaN/Inf (float16 overflow or broken compute dtype)
  H2 — bitsandbytes 4-bit kernels produce garbage on Jetson SM8.7
  H3 — chat-template prompt format is wrong (model sees unrecognised input)
"""
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEP   = "=" * 60

def banner(msg):
    print(f"\n{SEP}\n  {msg}\n{SEP}")

# ── 0. Environment ─────────────────────────────────────────────
banner("0. Environment")
import bitsandbytes as bnb
print(f"bitsandbytes : {bnb.__version__}")
print(f"torch        : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print(f"CUDA free    : {free/1e9:.2f} GB / {total/1e9:.2f} GB")
    cap = torch.cuda.get_device_capability()
    print(f"SM capability: {cap[0]}.{cap[1]}")

# ── 1. Load tokeniser ──────────────────────────────────────────
banner("1. Tokeniser")
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"pad_token_id : {tok.pad_token_id}")
print(f"eos_token_id : {tok.eos_token_id}")

# Show what token '!' decodes to / encodes as
bang_ids = tok.encode("!", add_special_tokens=False)
print(f"token ids for '!' : {bang_ids}")
print(f"token 0 decodes to: {repr(tok.decode([0]))}")

# ── 2. Test H2: load WITHOUT 4-bit, use bfloat16 on CPU ────────
banner("2. H2 — Load WITHOUT 4-bit (CPU bfloat16)")
model_cpu = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model_cpu.eval()
print(f"Model device : {next(model_cpu.parameters()).device}")
print(f"Model dtype  : {next(model_cpu.parameters()).dtype}")

messages = [{"role": "user", "content": "What causes type 2 diabetes? Answer in one sentence."}]
prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt_str, return_tensors="pt", truncation=True, max_length=128)

# Check raw logits first
with torch.no_grad():
    logits = model_cpu(**inputs).logits[0, -1].float()
nan_count  = torch.isnan(logits).sum().item()
inf_count  = torch.isinf(logits).sum().item()
top5_vals, top5_ids = torch.topk(logits, 5)
print(f"NaN logits   : {nan_count}")
print(f"Inf logits   : {inf_count}")
print(f"Top-5 tokens : {[(tok.decode([i]), round(v, 2)) for i, v in zip(top5_ids.tolist(), top5_vals.tolist())]}")

with torch.no_grad():
    out = model_cpu.generate(**inputs, max_new_tokens=80, do_sample=False,
                              use_cache=False, pad_token_id=tok.pad_token_id,
                              eos_token_id=tok.eos_token_id)
generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n[CPU bfloat16 output]\n{generated}")
del model_cpu
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── 3. Test H2b: 4-bit GPU, check logits ──────────────────────
if torch.cuda.is_available():
    banner("3. H2b — 4-bit GPU (bfloat16 compute)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        attn_implementation="eager",
    )
    model_4bit.eval()
    dev = next(model_4bit.parameters()).device
    print(f"Model device : {dev}")

    inputs_gpu = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        logits4 = model_4bit(**inputs_gpu).logits[0, -1].float()
    nan4  = torch.isnan(logits4).sum().item()
    inf4  = torch.isinf(logits4).sum().item()
    top5_vals4, top5_ids4 = torch.topk(logits4, 5)
    print(f"NaN logits   : {nan4}")
    print(f"Inf logits   : {inf4}")
    print(f"Top-5 tokens : {[(tok.decode([i]), round(v, 2)) for i, v in zip(top5_ids4.tolist(), top5_vals4.tolist())]}")

    with torch.no_grad():
        out4 = model_4bit.generate(**inputs_gpu, max_new_tokens=80, do_sample=False,
                                    use_cache=False, pad_token_id=tok.pad_token_id,
                                    eos_token_id=tok.eos_token_id)
    gen4 = tok.decode(out4[0][inputs_gpu["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[4-bit GPU output]\n{gen4}")
    del model_4bit
    torch.cuda.empty_cache()

print(f"\n{SEP}\nDiagnosis complete.\n{SEP}")