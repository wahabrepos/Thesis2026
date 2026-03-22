#!/usr/bin/env python3
"""
Interactive Query Interface for Self-MedRAG
"""

import os
import sys
import argparse
from pathlib import Path

# Required for Jetson Tegra CUDA allocator compatibility (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root for src package

from src.pipeline import MedRAGPipeline


def interactive_mode(pipeline: MedRAGPipeline):
    """Run interactive query mode."""
    print("\n" + "="*60)
    print("Self-MedRAG Interactive Query Interface")
    print("="*60)
    print("\nType your medical question, or 'quit' to exit.\n")
    
    while True:
        try:
            question = input("Query: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\nProcessing...\n")
            
            result = pipeline.query(question, return_details=False)
            
            print("-" * 60)
            print(f"Answer: {result['answer']}")
            print(f"\nConfidence: {result.get('confidence', 0.0):.2f}")
            print(f"Support Score: {result['support_score']:.2f}")
            print(f"Iterations: {result['iterations']}")
            print(f"Time: {result['latency']:.2f}s")
            
            print(f"\nRationale:")
            for i, statement in enumerate(result['rationale'], 1):
                print(f"  {i}. {statement}")
            
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def single_query(pipeline: MedRAGPipeline, question: str):
    """Process a single query."""
    result = pipeline.query(question, return_details=True)
    
    print("\n" + "="*60)
    print(f"Question: {question}")
    print("="*60)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nConfidence: {result.get('confidence', 0.0):.2f}")
    print(f"Support Score: {result['support_score']:.2f}")
    print(f"Iterations: {result['iterations']}")
    print(f"Time: {result['latency']:.2f}s")
    
    print(f"\nRationale:")
    for i, statement in enumerate(result['rationale'], 1):
        print(f"  {i}. {statement}")
    
    if result.get('citations'):
        print(f"\nCitations: {len(result['citations'])} sources")
    
    print("="*60 + "\n")


def batch_mode(pipeline: MedRAGPipeline, input_file: str, output_file: str):
    """Process batch queries from file."""
    import json
    
    # Read questions
    with open(input_file, "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(questions)} queries from {input_file}...")
    
    # Process
    results = pipeline.batch_query(questions, show_progress=True)
    
    # Save results
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Query Self-MedRAG")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--question", "-q", help="Single question")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--input-file", help="Batch input file")
    parser.add_argument("--output-file", help="Batch output file")
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Loading pipeline...")
    pipeline = MedRAGPipeline(config_path=args.config)
    
    # Run mode
    if args.interactive:
        interactive_mode(pipeline)
    elif args.question:
        single_query(pipeline, args.question)
    elif args.input_file:
        output = args.output_file or "results.jsonl"
        batch_mode(pipeline, args.input_file, output)
    else:
        # Default to interactive
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
