"""
Production-Ready Generator and Self-Reflection Modules
Supports both Causal LM (Mistral, Llama) and Seq2Seq (T5, FLAN-T5)
Includes structured JSON output parsing and robust error handling
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

logger = logging.getLogger(__name__)


class _StableLogitsProcessor(LogitsProcessor):
    """Cast float16 logits → float32 and clamp inf/nan before sampling.

    Qwen2.5 in float16 on CPU can produce inf logits in MLP/RoPE layers.
    torch.multinomial refuses inf/nan probabilities; float32 + clamp prevents that.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.float()
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores


class GeneratorModule:
    """
    Enhanced Generator Module supporting both Causal and Seq2Seq models
    with structured JSON output, error recovery, and confidence scoring.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize generator with configuration."""
        self.config = config
        gen_config = config.get("generator", {})
        
        # Model selection
        self.model_name = gen_config.get("model_name", "google/flan-t5-large")
        self.model_type = gen_config.get("model_type", "seq2seq")  # "causal" or "seq2seq"
        
        # Quantization settings
        self.load_in_4bit = gen_config.get("load_in_4bit", False)
        self.load_in_8bit = gen_config.get("load_in_8bit", False)
        
        # Generation parameters
        self.max_new_tokens = gen_config.get("max_new_tokens", 512)
        self.temperature = gen_config.get("temperature", 0.3)
        self.top_p = gen_config.get("top_p", 0.9)
        self.top_k = gen_config.get("top_k", 50)
        self.do_sample = gen_config.get("do_sample", True)
        self.repetition_penalty = gen_config.get("repetition_penalty", 1.1)
        
        # Structured output settings
        self.use_json_mode = gen_config.get("use_json_mode", True)
        self.enforce_json = gen_config.get("enforce_json", True)
        self.max_json_retries = gen_config.get("max_json_retries", 3)
        
        # Prompts
        self.system_prompt = gen_config.get("system_prompt", self._default_system_prompt())
        self.user_prompt_template = gen_config.get("user_prompt_template", self._default_user_prompt())
        
        # Load model (device selected inside _load_model after model is placed)
        self._load_model()
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for medical QA."""
        return """You are a medical AI assistant specialized in evidence-based clinical reasoning.
Your role is to answer medical questions accurately using only the provided context.

CRITICAL RULES:
1. Base ALL claims on the provided context passages
2. If context doesn't support a claim, state "insufficient evidence"
3. Cite specific context passages that support your reasoning
4. Use medical terminology appropriately
5. Acknowledge uncertainty when appropriate

OUTPUT FORMAT (JSON):
{
  "answer": "Brief, direct answer",
  "rationale": ["Step 1 reasoning", "Step 2 reasoning", ...],
  "confidence": 0.85,
  "citations": ["Context passage 1 used", "Context passage 2 used"]
}"""
    
    def _default_user_prompt(self) -> str:
        """Default user prompt template."""
        return """CONTEXT (Retrieved Medical Literature):
{context}

{history}

QUESTION: {query}

Provide your answer in the JSON format specified in the system prompt."""
    
    def _load_model(self) -> None:
        """Load model with quantization if specified."""
        logger.info(f"Loading {self.model_type} model: {self.model_name}")
        
        # Quantization configuration
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                # bfloat16 has the same dynamic range as float32 (8-bit exponent).
                # float16 (5-bit exponent, max ~65504) overflows in eager attention
                # on sequences ≥ 512 tokens, producing NaN logits that collapse to
                # token 0 = "!" after nan_to_num(nan=0.0) + argmax. Jetson Orin
                # SM8.7 (Ampere) supports bfloat16 tensor cores natively.
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            logger.info(f"Quantization: {'4-bit' if self.load_in_4bit else '8-bit'}")
        
        try:
            # caching_allocator_warmup (transformers 4.45+) pre-allocates a GPU buffer
            # as an optimization — patch it to skip gracefully on low-memory devices
            import transformers.modeling_utils as _mu
            if hasattr(_mu, "caching_allocator_warmup"):
                _orig_warmup = _mu.caching_allocator_warmup
                def _safe_warmup(*a, **kw):
                    try:
                        _orig_warmup(*a, **kw)
                    except torch.OutOfMemoryError:
                        pass
                _mu.caching_allocator_warmup = _safe_warmup

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Compute max_memory dynamically from actual free CUDA memory.
            # Hardcoding 2GiB risks exceeding what's physically available on Jetson
            # (CUDA context + transformers internals may leave only ~1-1.5 GiB free).
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                torch.cuda.empty_cache()
                free_bytes, _ = torch.cuda.mem_get_info(0)
                # Reserve 512 MiB headroom; clamp to at least 512 MiB so the key exists
                gpu_budget_gib = max(0.5, (free_bytes / (1024 ** 3)) - 0.5)
                gpu_budget_str = f"{gpu_budget_gib:.1f}GiB"
                logger.info(f"CUDA free: {free_bytes / (1024**3):.2f} GiB → GPU budget: {gpu_budget_str}")
            else:
                gpu_budget_str = "0GiB"

            max_memory = {0: gpu_budget_str, "cpu": "4GiB"} if cuda_available else {"cpu": "4GiB"}

            # GPU loading strategy for Jetson Orin Nano (8GB unified memory):
            #   4-bit quantized (bitsandbytes) if configured + ≥ 1.5 GiB CUDA free
            #     → ~0.75 GiB for model, needs clean NvMap state (fresh reboot)
            #   ≥ 3.5 GiB free → plain float16 on GPU (no bitsandbytes)
            #   ≥ 3.0 GiB free → float16 with CPU spill (device_map=auto)
            #   < 3.0 GiB free → float32 on CPU (numerically stable on ARM64)
            CUDA_4BIT_MIN_GIB  = 1.5   # minimum for 4-bit quantized GPU load
            CUDA_GPU_FP16_GIB  = 3.5   # enough for full fp16 on GPU
            CUDA_MIN_FREE_GIB  = 3.0   # minimum for any GPU loading

            use_4bit_gpu = (cuda_available and quantization_config is not None
                            and gpu_budget_gib >= CUDA_4BIT_MIN_GIB)
            use_gpu      = cuda_available and gpu_budget_gib >= CUDA_MIN_FREE_GIB
            use_gpu_fp16 = cuda_available and gpu_budget_gib >= CUDA_GPU_FP16_GIB

            if use_4bit_gpu:
                logger.info(f"Attempting 4-bit GPU load: {gpu_budget_gib:.1f} GiB CUDA free")
                load_kwargs = dict(
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype="auto",
                    max_memory=max_memory,
                    attn_implementation="eager",
                )
            elif not use_gpu:
                reason = "CUDA unavailable" if not cuda_available else f"CUDA free only {gpu_budget_gib:.1f} GiB < {CUDA_MIN_FREE_GIB} GiB threshold (NvMap fragmentation risk)"
                logger.warning(f"Loading model on CPU (float16): {reason}")
                load_kwargs = dict(
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="eager",
                )
                quantization_config = None
            elif use_gpu_fp16:
                logger.info(f"Loading model on GPU (float16, no bitsandbytes): {gpu_budget_gib:.1f} GiB available")
                quantization_config = None
                load_kwargs = dict(
                    device_map="cuda:0",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="eager",
                )
            else:
                # 3.0–3.5 GiB: CPU spill, no bitsandbytes
                logger.info(f"Loading model with CPU spill (float16): {gpu_budget_gib:.1f} GiB GPU available")
                quantization_config = None
                load_kwargs = dict(
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    max_memory=max_memory,
                    attn_implementation="eager",
                )

            # Load model — with OOM fallback to CPU if 4-bit GPU attempt fails
            def _do_load(kwargs, qconfig):
                if self.model_type == "causal":
                    return AutoModelForCausalLM.from_pretrained(
                        self.model_name, quantization_config=qconfig, **kwargs)
                elif self.model_type == "seq2seq":
                    kwargs.pop("trust_remote_code", None)
                    return AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name, quantization_config=qconfig, **kwargs)
                else:
                    raise ValueError(f"Unknown model_type: {self.model_type}")

            try:
                self.model = _do_load(load_kwargs, quantization_config)
            except (torch.OutOfMemoryError, RuntimeError) as oom_err:
                if use_4bit_gpu:
                    logger.warning(f"4-bit GPU load failed ({oom_err}), falling back to float16 CPU")
                    cpu_kwargs = dict(device_map="cpu", trust_remote_code=True,
                                     torch_dtype=torch.float16, attn_implementation="eager")
                    self.model = _do_load(cpu_kwargs, None)
                else:
                    raise
            
            self.model.eval()

            # transformers 4.57 wraps Qwen2Model.forward with @check_model_inputs which
            # raises TypeError("Missing **kwargs") after a successful forward pass when the
            # function signature lacks **kwargs. Unwrap via __wrapped__ to bypass this.
            try:
                inner_model = getattr(self.model, "model", None)
                if inner_model is not None and hasattr(inner_model, "forward"):
                    fwd = inner_model.__class__.forward
                    if hasattr(fwd, "__wrapped__"):
                        inner_model.__class__.forward = fwd.__wrapped__
                        logger.info("Unwrapped @check_model_inputs from model.forward")
            except Exception as _patch_err:
                logger.debug(f"check_model_inputs unwrap skipped: {_patch_err}")

            logger.info("Model loaded successfully")

            # Warmup: a minimal single-token generation to prime NvMap CUDA handles.
            # Kept tiny (1 token, short input) to avoid OOM during pipeline init —
            # on Jetson unified memory, a large warmup competes with NLI model loading.
            # NvMap errors on the first real query are retried automatically.
            if self.model_type == "causal" and torch.cuda.is_available():
                try:
                    _dev = next(self.model.parameters()).device
                    _dummy = self.tokenizer(
                        "warmup", return_tensors="pt", truncation=True, max_length=16
                    ).to(_dev)
                    with torch.no_grad():
                        self.model.generate(
                            **_dummy,
                            max_new_tokens=1,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    torch.cuda.empty_cache()
                    logger.info("CUDA warmup complete")
                except Exception as _w:
                    logger.warning(f"CUDA warmup failed: {_w} — first query may be slow")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _construct_prompt(
        self,
        query: str,
        context: List[str],
        history: List[Dict[str, Any]]
    ) -> str:
        """Construct prompt from query, context, and history."""
        # Format context
        context_text = "\n\n".join(
            [f"[Passage {i+1}]: {passage}" for i, passage in enumerate(context)]
        )
        
        # Format history
        history_text = ""
        if history:
            history_entries = []
            for i, entry in enumerate(history[-2:]):  # Last 2 iterations only
                hist_entry = f"Previous Iteration {i+1}:\n"
                hist_entry += f"Query: {entry.get('query', '')}\n"
                hist_entry += f"Answer: {entry.get('answer', '')}\n"
                hist_entry += f"Support Score: {entry.get('support_score', 0):.2f}\n"
                history_entries.append(hist_entry)
            history_text = "\n".join(history_entries)
            if history_text:
                history_text = f"\nPREVIOUS ATTEMPTS:\n{history_text}\n"
        
        # Build prompt based on model type
        if self.model_type == "causal":
            # Chat format for causal models (Mistral, Llama)
            prompt = f"{self.system_prompt}\n\n"
            prompt += self.user_prompt_template.format(
                context=context_text,
                history=history_text,
                query=query
            )
        else:
            # Seq2Seq format (T5, FLAN-T5)
            prompt = self.user_prompt_template.format(
                context=context_text,
                history=history_text,
                query=query
            )
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text from prompt."""
        try:
            # Free any lingering GPU allocations before generation (prevents NvMap OOM
            # after long FAISS build which can fragment the unified memory pool)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # For instruct/chat models, apply the tokenizer's chat template so the model
            # sees the proper <|im_start|>system/user/assistant tokens instead of raw text.
            # BUG FIX: do NOT split on "\n\n" — self.system_prompt itself contains "\n\n"
            # (before the OUTPUT FORMAT block), so split("\n\n", 1) cut the system message
            # in half and placed the JSON schema into the user message, causing the model
            # to see a garbled prompt and generate "!!!!!!..." as its first token.
            # Fix: use self.system_prompt directly as the system message, and strip it
            # from the front of the constructed prompt to isolate the user content.
            if self.model_type == "causal" and hasattr(self.tokenizer, "apply_chat_template"):
                sys_text = self.system_prompt.strip()
                separator = self.system_prompt + "\n\n"
                if prompt.startswith(separator):
                    user_text = prompt[len(separator):].strip()
                elif prompt.startswith(sys_text):
                    user_text = prompt[len(sys_text):].strip()
                else:
                    # fallback: everything is the user message
                    user_text = prompt.strip()
                    sys_text = ""
                messages = []
                if sys_text:
                    messages.append({"role": "system", "content": sys_text})
                messages.append({"role": "user", "content": user_text})
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass  # fall through to raw prompt if template fails

            # Tokenize — truncate from the LEFT so the question at the end is preserved.
            # Right-truncation (default) drops the question when context fills the window.
            self.tokenizer.truncation_side = "left"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,  # Truncate to avoid float16 attention overflow on CPU
            ).to(self.device)
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
                "repetition_penalty": self.repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                # Stabilise float16 logits before multinomial sampling
                "logits_processor": LogitsProcessorList([_StableLogitsProcessor()]),
                # KV-cache re-enabled: the aggressive warmup in _load_model now pre-allocates
                # the NvMap handles (use_cache=False, 256-token input, 32 new tokens) so the
                # first real generate() no longer hits NvMapMemAllocInternalTagged error 12.
                # Without KV-cache, generation is O(n²) in new tokens — unusably slow on Jetson.
                "use_cache": True,
            }
            
            # Remove sampling params if temperature is 0 (greedy)
            if self.temperature == 0:
                gen_kwargs["do_sample"] = False
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode
            if self.model_type == "causal":
                # For causal models, remove input tokens
                output_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                # For seq2seq, full output
                output_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _parse_json_output(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model output with error recovery."""
        # Try direct JSON parsing
        try:
            # Remove markdown code blocks if present
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            
            # Find JSON object
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                
                # Normalise: some model outputs use "causes", "result", etc.
                # instead of "answer" — remap any recognised key to "answer".
                for _alt in ("causes", "result", "response", "diagnosis", "conclusion"):
                    if _alt in parsed and "answer" not in parsed:
                        parsed["answer"] = parsed.pop(_alt)
                        break

                # Validate required fields
                if "answer" in parsed and "rationale" in parsed:
                    # Ensure rationale is a list of strings
                    if isinstance(parsed["rationale"], str):
                        parsed["rationale"] = [parsed["rationale"]]
                    elif isinstance(parsed["rationale"], list):
                        parsed["rationale"] = [
                            str(r) if not isinstance(r, str) else r
                            for r in parsed["rationale"]
                        ]

                    # Ensure answer is a plain string (model sometimes returns a list of dicts)
                    if isinstance(parsed["answer"], list):
                        parts = []
                        for item in parsed["answer"]:
                            if isinstance(item, dict):
                                parts.append(" ".join(str(v) for v in item.values() if v))
                            elif item:
                                parts.append(str(item))
                        parsed["answer"] = " ".join(parts).strip() or "See rationale."
                    elif not isinstance(parsed["answer"], str):
                        parsed["answer"] = str(parsed["answer"])

                    # Add defaults for optional fields
                    parsed.setdefault("confidence", 0.7)
                    parsed.setdefault("citations", [])
                    
                    return parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback: Extract from unstructured text
        logger.warning("JSON parsing failed, using fallback extraction")
        
        answer = ""
        rationale = []
        
        # Try to extract answer
        answer_match = re.search(r'[Aa]nswer:\s*([^\n]+)', text)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Use first sentence
            sentences = re.split(r'[.!?]', text)
            if sentences:
                answer = sentences[0].strip()
        
        # Try to extract rationale
        rationale_match = re.search(r'[Rr]ationale:\s*\[(.*?)\]', text, re.DOTALL)
        if rationale_match:
            rationale_text = rationale_match.group(1)
            rationale = [s.strip().strip('"\'') for s in rationale_text.split(',')]
        else:
            # Split by sentences
            sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
            rationale = sentences[1:] if len(sentences) > 1 else [text]
        
        return {
            "answer": answer or text[:200],  # Fallback to first 200 chars
            "rationale": rationale or [text],
            "confidence": 0.5,  # Low confidence for fallback
            "citations": []
        }
    
    def generate(
        self,
        query: str,
        context: List[str],
        history: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], float, List[str]]:
        """
        Generate answer with rationale and confidence.
        
        Returns:
            Tuple of (answer, rationale, confidence, citations)
        """
        # Construct prompt
        prompt = self._construct_prompt(query, context, history)
        
        # Generate with retries for JSON mode
        for attempt in range(self.max_json_retries if self.enforce_json else 1):
            try:
                # Generate text
                output_text = self._generate_text(prompt)
                
                # Parse output
                if self.use_json_mode:
                    parsed = self._parse_json_output(output_text)
                    
                    # Return if valid
                    if parsed["answer"] and parsed["rationale"]:
                        logger.info(f"Generated answer (attempt {attempt + 1})")
                        return (
                            parsed["answer"],
                            parsed["rationale"],
                            parsed["confidence"],
                            parsed["citations"]
                        )
                else:
                    # Non-JSON mode: simple parsing
                    return (
                        output_text,
                        [output_text],
                        0.7,
                        []
                    )
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == self.max_json_retries - 1:
                    raise
        
        # Final fallback
        return (
            "Unable to generate answer",
            ["Generation failed after retries"],
            0.0,
            []
        )


class SelfReflectiveModule:
    """
    Enhanced Self-Reflection Module using NLI for verification.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize self-reflection module."""
        self.config = config
        sr_config = config.get("self_reflection", {})
        
        # NLI model
        self.nli_model_name = sr_config.get("nli_model", "microsoft/deberta-v3-base-mnli")
        
        # Thresholds
        self.verification_threshold = float(sr_config.get("verification_threshold", 0.5))
        self.rationale_score_threshold = float(sr_config.get("rationale_score_threshold", 0.7))
        
        # Batch processing
        self.nli_batch_size = sr_config.get("nli_batch_size", 16)
        
        # Device — DeBERTa-base is ~0.4 GiB; threshold lowered to 1.0 GiB so it
        # loads on GPU after the 1.5B generator (~0.75 GiB) has already loaded.
        # The old 3.0 GiB threshold always forced CPU because only ~2.75 GiB remained,
        # making NLI 50× slower (6s/call on CPU vs ~0.1s on GPU).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free_gib = free_bytes / (1024 ** 3)
            if free_gib >= 1.0:
                self.device = torch.device("cuda")
            else:
                logger.warning(
                    f"NLI model: only {free_gib:.1f} GiB CUDA free — using CPU to avoid NvMap OOM"
                )
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load NLI model."""
        logger.info(f"Loading NLI model: {self.nli_model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name
            )
            # Move to target device with CPU fallback — NvMap may refuse the
            # contiguous allocation even when free_bytes looks sufficient.
            if self.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    self.model.to(self.device)
                    logger.info(f"NLI model loaded on {self.device}")
                except (torch.OutOfMemoryError, RuntimeError) as gpu_err:
                    logger.warning(
                        f"NLI GPU load failed ({gpu_err}), falling back to CPU"
                    )
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
            else:
                self.model.to(self.device)
            self.model.eval()

            logger.info("NLI model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise
    
    def verify(self, rationale: List[str], context: List[str]) -> float:
        """
        Verify rationale support using NLI.
        
        Returns:
            Support score (0-1)
        """
        if not rationale:
            return 1.0
        
        supported_count = 0
        
        for statement in rationale:
            best_score = 0.0
            
            # Check against all context passages
            for passage in context:
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        passage,  # Premise
                        statement,  # Hypothesis
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Get entailment probability
                    probs = F.softmax(outputs.logits, dim=-1)[0]
                    entailment_prob = probs[2].item()  # Index 2 = entailment for MNLI
                    
                    best_score = max(best_score, entailment_prob)
                    
                except Exception as e:
                    logger.error(f"NLI inference failed: {e}")
                    continue
            
            # Check if supported
            if best_score >= self.verification_threshold:
                supported_count += 1
            
            logger.debug(f"Statement support: {best_score:.3f} ({'✓' if best_score >= self.verification_threshold else '✗'})")
        
        support_score = supported_count / len(rationale)
        logger.info(f"Overall support score: {support_score:.3f} ({supported_count}/{len(rationale)})")
        
        return support_score
    
    def extract_unsupported(self, rationale: List[str], context: List[str]) -> List[str]:
        """Extract unsupported statements."""
        unsupported = []
        
        for statement in rationale:
            best_score = 0.0
            
            for passage in context:
                try:
                    inputs = self.tokenizer(
                        passage,
                        statement,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    probs = F.softmax(outputs.logits, dim=-1)[0]
                    entailment_prob = probs[2].item()
                    
                    best_score = max(best_score, entailment_prob)
                    
                except Exception as e:
                    continue
            
            if best_score < self.verification_threshold:
                unsupported.append(statement)
        
        logger.info(f"Extracted {len(unsupported)} unsupported statements")
        return unsupported
