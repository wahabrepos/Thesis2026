"""
Self-verifying clinical Reasoning on the LOOP MedRAG Pipeline - High-Level API
Interface for medical question answering
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
        return_details: bool = False,
        dataset_type: str = None,
    ) -> Dict[str, Any]:
        """
        Answer a single medical question.

        Args:
            question: Medical question to answer
            return_details: Whether to return detailed iteration history
            dataset_type: Dataset type hint ("pubmedqa" enforces binary yes/no answer)

        Returns:
            Dictionary with answer, rationale, confidence, etc.
        """
        start_time = time.time()

        logger.info(f"Processing query: '{question[:100]}...'")

        # Run iterative loop
        answer, rationale, iterations, support_score, history = self.trainer.run(
            question, dataset_type=dataset_type
        )
        
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
        max_samples: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
    ) -> Dict[str, Any]:
        """Evaluate on a benchmark dataset with crash-safe checkpointing.

        Args:
            checkpoint_path: File to read/write incremental progress. If the
                file exists on entry, completed predictions are loaded and the
                run resumes from the next unanswered sample.
            checkpoint_every: Save the checkpoint after this many new queries.
        """
        import json as _json
        from datetime import datetime

        logger.info(f"Evaluating on {dataset_name}")

        # Load dataset
        dataset_loader = DatasetLoader(self.config)
        datasets = dataset_loader.load_data()

        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        dataset = datasets[dataset_name]

        if max_samples:
            dataset = dataset[:max_samples]

        total_samples = len(dataset)
        logger.info(f"Processing {total_samples} samples")

        # ── Resume from checkpoint ────────────────────────────────────────────
        predictions: list = []
        ground_truth: list = []
        start_index = 0

        if checkpoint_path:
            ckpt_file = Path(checkpoint_path)
            if ckpt_file.exists():
                try:
                    ckpt = _json.loads(ckpt_file.read_text())
                    predictions  = ckpt.get("predictions", [])
                    ground_truth = ckpt.get("ground_truth", [])
                    start_index  = len(predictions)
                    logger.info(
                        f"Resumed from checkpoint '{checkpoint_path}': "
                        f"{start_index}/{total_samples} already done"
                    )
                    print(f"[checkpoint] Resuming at sample {start_index + 1}/{total_samples}")
                except Exception as ckpt_err:
                    logger.warning(f"Could not load checkpoint ({ckpt_err}), starting fresh")

        def _save_checkpoint():
            if not checkpoint_path:
                return
            ckpt_file = Path(checkpoint_path)
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_file.with_suffix(".tmp")
            tmp.write_text(_json.dumps({
                "dataset":      dataset_name,
                "total":        total_samples,
                "completed":    len(predictions),
                "predictions":  predictions,
                "ground_truth": ground_truth,
                "saved_at":     datetime.utcnow().isoformat(),
            }, indent=2))
            tmp.replace(ckpt_file)   # atomic rename — safe against mid-write crashes
            logger.info(f"Checkpoint saved: {len(predictions)}/{total_samples}")

        # ── Main evaluation loop ──────────────────────────────────────────────
        queries_since_save = 0

        for i, entry in enumerate(dataset):
            if i < start_index:
                continue   # already completed in a previous run

            question  = entry.get("question", "")
            dtype     = entry.get("dataset_type", dataset_name)
            answer_gt = (
                entry["answer_letter"]
                if dtype == "medqa" and entry.get("answer_letter")
                else entry.get("answer", "")
            )

            result = self.query(question, return_details=False, dataset_type=dtype)

            predictions.append({
                "final_answer":  result["answer"],
                "rationale":     result["rationale"],
                "iterations":    result["iterations"],
                "support_score": result["support_score"],
                "confidence":    result.get("confidence", 0.0),
                "latency":       result["latency"],
                "dataset_type":  dtype,
            })
            ground_truth.append({"answer": answer_gt})

            queries_since_save += 1
            if queries_since_save >= checkpoint_every:
                _save_checkpoint()
                queries_since_save = 0

            completed = len(predictions)
            if completed % 10 == 0 or completed == total_samples:
                logger.info(f"Progress: {completed}/{total_samples}")

        # Final checkpoint flush (catches the tail that didn't hit the interval)
        _save_checkpoint()

        # ── Metrics ───────────────────────────────────────────────────────────
        evaluator    = Evaluation()
        metrics      = evaluator.evaluate(predictions, ground_truth)
        error_analysis = evaluator.error_analysis(predictions, ground_truth)

        # Remove checkpoint now that the run finished cleanly
        if checkpoint_path:
            ckpt_file = Path(checkpoint_path)
            if ckpt_file.exists():
                try:
                    ckpt_file.unlink()
                    logger.info(f"Checkpoint deleted after successful run")
                except OSError:
                    pass

        return {
            "dataset": dataset_name,
            "num_samples": total_samples,
            "metrics": metrics,
            "error_analysis": error_analysis,
            "predictions": predictions,
        }
