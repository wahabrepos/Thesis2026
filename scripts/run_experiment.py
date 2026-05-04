#!/usr/bin/env python3
"""
Main Experiment Runner
Production-ready script with checkpointing, monitoring, and error recovery
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time

# Required for Jetson Tegra CUDA allocator compatibility (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# Add project root to path so src/ is importable as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MedRAGPipeline
from src.utils import save_json, load_json, format_time

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Self-MedRAG experiments")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["medqa", "pubmedqa", "all"],
        default="medqa",
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run single test query for verification"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (auto-generated if omitted)"
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint after this many queries (default: 50)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results"
    )
    
    return parser.parse_args()


def test_query(pipeline: MedRAGPipeline):
    """Run a test query to verify system."""
    test_questions = [
        "What is the first-line treatment for hypertension?",
        "What causes type 2 diabetes?",
        "How is pneumonia diagnosed?",
    ]
    
    print("\n" + "="*60)
    print("TEST MODE - Running sample queries")
    print("="*60 + "\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuery {i}: {question}")
        print("-" * 60)
        
        result = pipeline.query(question, return_details=True)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Support Score: {result['support_score']:.2f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Latency: {result['latency']:.2f}s")
        print(f"\nRationale:")
        for j, statement in enumerate(result['rationale'], 1):
            print(f"  {j}. {statement}")
        
        if result.get('citations'):
            print(f"\nCitations: {len(result['citations'])} sources")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY ✓")
    print("="*60 + "\n")


def main():
    """Main experiment runner."""
    args = parse_args()
    
    print("="*60)
    print("Self-MedRAG Production System")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize pipeline
        print(f"\nLoading configuration from: {args.config}")
        pipeline = MedRAGPipeline(config_path=args.config)
        
        # Test mode
        if args.test_mode:
            test_query(pipeline)
            return
        
        # Run evaluation
        if args.dataset == "all":
            datasets = ["medqa", "pubmedqa"]
        else:
            datasets = [args.dataset]
        
        all_results = {}
        
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Evaluating on {dataset_name.upper()}")
            print(f"{'='*60}\n")

            # Auto-generate checkpoint path per dataset if not provided via --resume
            if args.resume:
                ckpt_path = args.resume
            else:
                Path("./checkpoints").mkdir(parents=True, exist_ok=True)
                ckpt_path = f"./checkpoints/{dataset_name}_checkpoint.json"

            print(f"Checkpoint file: {ckpt_path}  (every {args.checkpoint_every} queries)")

            results = pipeline.evaluate_dataset(
                dataset_name=dataset_name,
                max_samples=args.samples,
                checkpoint_path=ckpt_path,
                checkpoint_every=args.checkpoint_every,
            )
            
            all_results[dataset_name] = results
            
            # Print metrics
            print(f"\n{dataset_name.upper()} Results:")
            print("-" * 40)
            metrics = results["metrics"]
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric_name}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"{metric_name}: {value}")
                else:
                    print(f"{metric_name}: {value}")
        
        # Save results
        output_path = args.output or f"./results/experiment_{int(time.time())}.json"
        save_json(all_results, output_path)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETED in {format_time(elapsed)}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
