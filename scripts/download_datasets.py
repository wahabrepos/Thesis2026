#!/usr/bin/env python3
"""
Download MedQA and PubMedQA datasets from HuggingFace
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def download_medqa(output_dir: Path, split: str = "train"):
    """Download MedQA USMLE dataset from HuggingFace.

    The HuggingFace schema for openlifescienceai/medqa is:
      question: str
      options:  dict  {"A": "...", "B": "...", "C": "...", "D": "..."}
      answer:   str   "A" | "B" | "C" | "D"

    We store:
      question:       original question text (no options appended — that happens in dataset_loader)
      options:        list of option texts in A-B-C-D order
      answer_letter:  correct option letter ("A" … "E")
      answer:         correct option text (for human readability)
    """
    print(f"Downloading MedQA ({split} split)...")

    try:
        dataset = load_dataset("openlifescienceai/medqa", split=split)

        data = []
        for item in dataset:
            # openlifescienceai/medqa nests all fields inside item["data"]
            inner = item.get("data", item)   # fall back to item itself if flat schema

            question_text   = str(inner.get("Question",        inner.get("question",       ""))).strip()
            options_raw     =     inner.get("Options",         inner.get("options",         {}))
            correct_letter  = str(inner.get("Correct Option",  inner.get("answer_letter",   ""))).strip().upper()
            correct_text    = str(inner.get("Correct Answer",  inner.get("answer",          ""))).strip()

            # options_raw may be a dict {"A": "...", "B": "...", ...} or a list
            if isinstance(options_raw, dict):
                sorted_keys  = sorted(options_raw.keys())
                options_list = [str(options_raw[k]).strip() for k in sorted_keys]
                # Derive correct_letter from correct_text if missing
                if not correct_letter and correct_text:
                    for k, v in options_raw.items():
                        if str(v).strip().lower() == correct_text.lower():
                            correct_letter = k.upper()
                            break
            elif isinstance(options_raw, list):
                options_list = [str(o).strip() for o in options_raw]
            else:
                options_list = []

            data.append({
                "question":      question_text,
                "options":       options_list,    # ["text_A", "text_B", "text_C", "text_D"]
                "answer_letter": correct_letter,  # "A" | "B" | "C" | "D"
                "answer":        correct_text,    # full text of the correct option
            })

        output_file = output_dir / f"medqa_{split}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {len(data)} samples to {output_file}")

    except Exception as e:
        print(f"✗ Failed to download MedQA: {e}")


def download_pubmedqa(output_dir: Path, split: str = "train"):
    """Download PubMedQA dataset."""
    print(f"Downloading PubMedQA ({split} split)...")
    
    try:
        # pqa_artificial only has 'train'; pqa_labeled contains the 1000-sample test set
        config = "pqa_labeled" if split == "test" else "pqa_artificial"
        actual_split = "train" if split == "test" else split
        dataset = load_dataset("qiaojin/PubMedQA", config, split=actual_split)
        
        # Convert to JSON format
        data = []
        for item in dataset:
            data.append({
                "question": item.get("question", ""),
                "abstract": item.get("context", {}).get("contexts", [""])[0],
                "final_decision": item.get("final_decision", ""),
                "answer": item.get("final_decision", ""),  # Alias
            })
        
        # Save
        output_file = output_dir / f"pubmedqa_{split}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(data)} samples to {output_file}")
        
    except Exception as e:
        print(f"✗ Failed to download PubMedQA: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["medqa", "pubmedqa", "all"], default="all")
    parser.add_argument("--split", default="train,test")
    parser.add_argument("--output-dir", default="./data/datasets")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = args.split.split(",")
    
    print("="*60)
    print("Downloading Datasets from HuggingFace")
    print("="*60 + "\n")
    
    if args.dataset in ["medqa", "all"]:
        for split in splits:
            download_medqa(output_dir, split.strip())
    
    if args.dataset in ["pubmedqa", "all"]:
        for split in splits:
            download_pubmedqa(output_dir, split.strip())
    
    print("\n" + "="*60)
    print("Download completed!")
    print("="*60)


if __name__ == "__main__":
    main()
