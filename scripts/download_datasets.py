#!/usr/bin/env python3
"""
Download MedQA and PubMedQA datasets from HuggingFace
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def download_medqa(output_dir: Path, split: str = "train"):
    """Download MedQA dataset."""
    print(f"Downloading MedQA ({split} split)...")
    
    try:
        dataset = load_dataset("openlifescienceai/medqa", split=split)
        
        # Convert to JSON format
        data = []
        for item in dataset:
            data.append({
                "question": item.get("question", ""),
                "options": item.get("options", {}).get("A", []),  # Simplified
                "answer": item.get("answer", ""),
            })
        
        # Save
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
