"""
Evaluation Module - Enhanced Metrics
Includes accuracy, F1, iteration stats, and groundedness metrics
"""

import logging
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class Evaluation:
    """Enhanced evaluation with groundedness and iteration metrics."""
    
    def __init__(self, metrics: List[str] = None):
        """Initialize evaluator with metric list."""
        if metrics is None:
            self.metrics = [
                "accuracy",
                "f1_score",
                "avg_iterations",
                "more_than_one_iter_pct",
                "avg_support_score",
                "avg_confidence",
                "latency_p50",
                "latency_p95",
            ]
        else:
            self.metrics = metrics
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.strip().lower()
    
    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.
        
        Each prediction should contain:
            - final_answer: str
            - rationale: List[str]
            - iterations: int
            - support_score: float
            - confidence: float (optional)
            - latency: float (optional)
        
        Each ground_truth should contain:
            - answer: str
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Collect data
        pred_answers = []
        gt_answers = []
        iteration_counts = []
        support_scores = []
        confidences = []
        latencies = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Answers
            pred_ans = self.normalize_text(pred.get("final_answer", ""))
            gt_ans = self.normalize_text(gt.get("answer", ""))
            pred_answers.append(pred_ans)
            gt_answers.append(gt_ans)
            
            # Iterations
            iteration_counts.append(pred.get("iterations", 1))
            
            # Support scores
            support_scores.append(pred.get("support_score", 0.0))
            
            # Confidence
            confidences.append(pred.get("confidence", 0.0))
            
            # Latency
            if "latency" in pred:
                latencies.append(pred["latency"])
        
        # Compute metrics
        results = {}
        
        # Accuracy
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(gt_answers, pred_answers)
            logger.info(f"Accuracy: {results['accuracy']:.4f}")
        
        # F1 Score
        if "f1_score" in self.metrics:
            try:
                results["f1_score"] = f1_score(
                    gt_answers,
                    pred_answers,
                    average="macro",
                    zero_division=0
                )
                logger.info(f"F1 Score: {results['f1_score']:.4f}")
            except Exception as e:
                logger.warning(f"F1 computation failed: {e}")
                results["f1_score"] = 0.0
        
        # Iteration statistics
        if "avg_iterations" in self.metrics:
            results["avg_iterations"] = np.mean(iteration_counts)
            logger.info(f"Avg Iterations: {results['avg_iterations']:.2f}")
        
        if "more_than_one_iter_pct" in self.metrics:
            more_than_one = sum(1 for cnt in iteration_counts if cnt > 1)
            results["more_than_one_iter_pct"] = (more_than_one / len(iteration_counts)) * 100
            logger.info(f"More than 1 iteration: {results['more_than_one_iter_pct']:.1f}%")
        
        # Iteration distribution
        if "iteration_distribution" in self.metrics:
            distribution = {}
            for cnt in iteration_counts:
                distribution[cnt] = distribution.get(cnt, 0) + 1
            results["iteration_distribution"] = distribution
            logger.info(f"Iteration distribution: {distribution}")
        
        # Support scores
        if "avg_support_score" in self.metrics and support_scores:
            results["avg_support_score"] = np.mean(support_scores)
            logger.info(f"Avg Support Score: {results['avg_support_score']:.4f}")
        
        # Confidence scores
        if "avg_confidence" in self.metrics and confidences:
            results["avg_confidence"] = np.mean(confidences)
            logger.info(f"Avg Confidence: {results['avg_confidence']:.4f}")
        
        # Latency statistics
        if "latency_p50" in self.metrics and latencies:
            results["latency_p50"] = np.percentile(latencies, 50)
            logger.info(f"Latency P50: {results['latency_p50']:.2f}s")
        
        if "latency_p95" in self.metrics and latencies:
            results["latency_p95"] = np.percentile(latencies, 95)
            logger.info(f"Latency P95: {results['latency_p95']:.2f}s")
        
        if "latency_mean" in self.metrics and latencies:
            results["latency_mean"] = np.mean(latencies)
            logger.info(f"Latency Mean: {results['latency_mean']:.2f}s")
        
        # Citation accuracy (if available)
        if "citation_accuracy" in self.metrics:
            cited_count = sum(
                1 for pred in predictions
                if pred.get("citations") and len(pred["citations"]) > 0
            )
            results["citation_accuracy"] = (cited_count / len(predictions)) * 100
            logger.info(f"Citation Rate: {results['citation_accuracy']:.1f}%")
        
        return results
    
    def error_analysis(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform error analysis on incorrect predictions."""
        errors = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            pred_ans = self.normalize_text(pred.get("final_answer", ""))
            gt_ans = self.normalize_text(gt.get("answer", ""))
            
            if pred_ans != gt_ans:
                errors.append({
                    "index": i,
                    "query": pred.get("query", ""),
                    "predicted": pred_ans,
                    "ground_truth": gt_ans,
                    "iterations": pred.get("iterations", 1),
                    "support_score": pred.get("support_score", 0.0),
                    "confidence": pred.get("confidence", 0.0),
                })
        
        # Analyze patterns
        analysis = {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(predictions) if predictions else 0,
            "errors": errors[:10],  # Top 10 errors
        }
        
        # Low support score errors
        low_support_errors = [e for e in errors if e["support_score"] < 0.5]
        analysis["low_support_errors"] = len(low_support_errors)
        
        # Low confidence errors
        low_conf_errors = [e for e in errors if e["confidence"] < 0.5]
        analysis["low_confidence_errors"] = len(low_conf_errors)
        
        # Single iteration errors
        single_iter_errors = [e for e in errors if e["iterations"] == 1]
        analysis["single_iteration_errors"] = len(single_iter_errors)
        
        logger.info(f"Error Analysis: {len(errors)} errors out of {len(predictions)} predictions")
        
        return analysis
