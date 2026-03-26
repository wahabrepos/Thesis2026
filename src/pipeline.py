"""
MedRAG Pipeline - High-Level API
Provides easy-to-use interface for medical question answering
"""

import logging
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from .corpus_loader import CorpusLoader
from .dataset_loader import DatasetLoader
from .retrieval import RetrievalModule
from .model import GeneratorModule, SelfReflectiveModule
from .trainer import Trainer
from .evaluation import Evaluation
from .utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class MedRAGPipeline:
    """High-level pipeline for medical question answering."""
    
    @staticmethod
    def _check_thermals(warn_c: float = 75.0, abort_c: float = 85.0) -> float:
        """
        Read SoC thermal zones and warn/abort if too hot.
        Prevents the Jetson emergency-shutdown that occurs when the SoC
        hits its power/thermal budget during heavy model-load spikes.

        Returns the highest temperature found (°C).
        """
        import glob
        temps = []
        for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                raw = open(path, "rb").read()
                if raw:
                    temps.append(int(raw.strip()) / 1000.0)
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        if not temps:
            return 0.0
        peak = max(temps)
        if peak >= abort_c:
            raise RuntimeError(
                f"SoC temperature {peak:.0f}°C ≥ abort threshold {abort_c:.0f}°C. "
                "Let the board cool down before loading the model."
            )
        if peak >= warn_c:
            logger.warning(
                f"SoC temperature {peak:.0f}°C is high — consider running "
                "`sudo jetson_clocks --fan` to boost cooling before loading."
            )
        else:
            logger.info(f"Thermals OK: peak SoC temp {peak:.0f}°C")
        return peak

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        setup_logging(self.config)

        logger.info("Initializing MedRAG Pipeline")

        # Thermal gate — abort early rather than let the Jetson emergency-shutdown
        # mid-load when CPU+GPU power spikes during model weight transfer.
        self._check_thermals()
        
        # Load corpus
        corpus_loader = CorpusLoader(self.config)
        corpus = corpus_loader.load_corpus()
        logger.info(f"Loaded corpus with {len(corpus)} documents")
        
        # Initialize modules — generator first to claim GPU memory before embeddings use RAM
        self.generator = GeneratorModule(self.config)
        self.retrieval = RetrievalModule(self.config, corpus=corpus)
        self.self_reflector = SelfReflectiveModule(self.config)
        self.trainer = Trainer(
            self.retrieval,
            self.generator,
            self.self_reflector,
            self.config
        )
        
        logger.info("Pipeline initialized successfully")
    
    def query(
        self,
        question: str,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a single medical question.
        
        Args:
            question: Medical question to answer
            return_details: Whether to return detailed iteration history
        
        Returns:
            Dictionary with answer, rationale, confidence, etc.
        """
        start_time = time.time()
        
        logger.info(f"Processing query: '{question[:100]}...'")
        
        # Run iterative loop
        answer, rationale, iterations, support_score, history = self.trainer.run(question)
        
        elapsed_time = time.time() - start_time
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "rationale": rationale,
            "iterations": iterations,
            "support_score": support_score,
            "latency": elapsed_time,
        }
        
        # Add confidence if available
        if history and "confidence" in history[-1]:
            result["confidence"] = history[-1]["confidence"]
        
        # Add citations if available
        if history and "citations" in history[-1]:
            result["citations"] = history[-1]["citations"]
        
        # Add detailed history if requested
        if return_details:
            result["history"] = history
        
        logger.info(f"Query completed in {elapsed_time:.2f}s ({iterations} iterations)")
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions."""
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                questions = tqdm(questions, desc="Processing queries")
            except ImportError:
                logger.warning("tqdm not available, progress bar disabled")
        
        for question in questions:
            try:
                result = self.query(question, return_details=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{question[:50]}...': {e}")
                results.append({
                    "question": question,
                    "answer": "Error",
                    "rationale": [str(e)],
                    "iterations": 0,
                    "support_score": 0.0,
                    "latency": 0.0,
                })
        
        return results
    
    def evaluate_dataset(
        self,
        dataset_name: str = "medqa",
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate on a benchmark dataset."""
        logger.info(f"Evaluating on {dataset_name}")
        
        # Load dataset
        dataset_loader = DatasetLoader(self.config)
        datasets = dataset_loader.load_data()
        
        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset = datasets[dataset_name]
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset[:max_samples]
        
        logger.info(f"Processing {len(dataset)} samples")
        
        # Process queries
        predictions = []
        ground_truth = []
        
        for entry in dataset:
            question = entry.get("question", "")
            answer_gt = entry.get("answer", "")
            
            # Get prediction
            result = self.query(question, return_details=False)
            
            predictions.append({
                "final_answer": result["answer"],
                "rationale": result["rationale"],
                "iterations": result["iterations"],
                "support_score": result["support_score"],
                "confidence": result.get("confidence", 0.0),
                "latency": result["latency"],
            })
            
            ground_truth.append({
                "answer": answer_gt
            })
        
        # Evaluate
        evaluator = Evaluation()
        metrics = evaluator.evaluate(predictions, ground_truth)
        
        # Error analysis
        error_analysis = evaluator.error_analysis(predictions, ground_truth)
        
        return {
            "dataset": dataset_name,
            "num_samples": len(dataset),
            "metrics": metrics,
            "error_analysis": error_analysis,
            "predictions": predictions,
        }
