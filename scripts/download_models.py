#!/usr/bin/env python3
"""
Download and cache HuggingFace models
"""

import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def download_model(model_name: str, model_type: str = "auto"):
    """Download a model from HuggingFace."""
    print(f"Downloading {model_name}...")
    
    try:
        # Download tokenizer
        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model
        print("  Downloading model...")
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        print(f"  ✓ Downloaded to cache")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--type", choices=["auto", "causal"], default="auto")
    args = parser.parse_args()
    
    download_model(args.model, args.type)


if __name__ == "__main__":
    main()
