#!/usr/bin/env python3
"""
Quantize models to 4-bit or 8-bit for Jetson deployment
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def quantize_model(
    model_name: str,
    quantization: str = "4bit",
    output_dir: str = None
):
    """Quantize a model."""
    print(f"Quantizing {model_name} to {quantization}...")
    
    # Quantization config
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:  # 8bit
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load quantized model — use explicit max_memory for Jetson unified memory
        print(f"Loading model with {quantization} quantization...")
        gpu_free = torch.cuda.mem_get_info()[0]
        max_memory = {
            0: f"{int(gpu_free * 0.75 / 1024**3)}GiB",
            "cpu": "2GiB",
        }
        print(f"  Memory budget: GPU {max_memory[0]}, CPU {max_memory['cpu']}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
        )
        
        # Save if output_dir specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"✓ Quantized model saved to {output_dir}")
        else:
            print("✓ Model loaded and cached (not saved)")
        
        # Show size
        param_size = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  Parameters: {param_size:.2f}B")
        
    except Exception as e:
        print(f"✗ Quantization failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()
    
    quantize_model(args.model, args.quantization, args.output)


if __name__ == "__main__":
    main()
