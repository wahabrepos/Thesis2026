#!/usr/bin/env python3
"""
Download Medical Corpus (PubMed abstracts)
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def download_pubmed(output_file: Path, max_docs: int = 50000):
    """
    Download PubMed abstracts.
    
    Note: This is a placeholder. In practice, you would:
    1. Use PubMed API or bulk download
    2. Or use pre-processed datasets from HuggingFace
    """
    print(f"Downloading PubMed abstracts (max {max_docs})...")
    print("Using sample data for demonstration\n")
    
    # Sample medical documents (in production, replace with actual PubMed API)
    sample_docs = [
        {
            "pmid": f"PMID{i}",
            "title": f"Medical study title {i}",
            "abstract": f"This is a sample medical abstract discussing clinical findings and treatment approaches for various medical conditions. Study {i} focuses on evidence-based medicine.",
            "text": f"Medical study title {i}. This is a sample medical abstract discussing clinical findings and treatment approaches for various medical conditions. Study {i} focuses on evidence-based medicine."
        }
        for i in range(min(max_docs, 1000))  # Limited sample
    ]
    
    # Save as JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for doc in tqdm(sample_docs, desc="Saving documents"):
            f.write(json.dumps(doc) + "\n")
    
    print(f"\n✓ Saved {len(sample_docs)} documents to {output_file}")
    print(f"\nNOTE: For production use, download actual PubMed corpus using:")
    print("  - PubMed E-utilities API")
    print("  - PubMed Baseline (bulk download)")
    print("  - Pre-processed datasets (HuggingFace, Kaggle)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="pubmed", choices=["pubmed"])
    parser.add_argument("--max-docs", type=int, default=50000)
    parser.add_argument("--output", default="./data/corpus/pubmed_abstracts.jsonl")
    args = parser.parse_args()
    
    output_file = Path(args.output)
    
    print("="*60)
    print("Medical Corpus Download")
    print("="*60 + "\n")
    
    if args.source == "pubmed":
        download_pubmed(output_file, args.max_docs)
    
    print("\n" + "="*60)
    print("Download completed!")
    print("="*60)


if __name__ == "__main__":
    main()
