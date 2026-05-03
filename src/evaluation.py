"""
Evaluation Module - Enhanced Metrics
Includes accuracy, F1, iteration stats, and groundedness metrics
"""

import logging
import re
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

    @staticmethod
    def extract_option_letter(text: str) -> str:
        """Extract first A-E option letter from model output.

        Handles: "A", "A.", "A)", "(A)", "The answer is A", "A. Penicillin", etc.
        Returns the uppercase letter, or "" if none found.
        """
        # Highest priority: standalone letter at the very start
        m = re.match(r"^\s*([A-Ea-e])[.\):\s]", text)
        if m:
            return m.group(1).upper()
        # "answer is A", "correct answer: B", etc.
        m = re.search(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-Ea-e])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Parenthesised: (A), [B]
        m = re.search(r"[(\[]([A-Ea-e])[)\]]", text)
        if m:
            return m.group(1).upper()
        # Any isolated A-E letter as a word
        m = re.search(r"\b([A-Ea-e])\b", text)
        if m:
            return m.group(1).upper()
        return ""

    @staticmethod
    def extract_binary_answer(text: str) -> str:
        """Extract yes/no from model output for PubMedQA."""
        t = text.strip().lower()
        if t in ("yes", "no"):
            return t
        # "yes, ..." or "no, ..."
        m = re.match(r"^(yes|no)[^a-z]", t)
        if m:
            return m.group(1)
        # Anywhere in text
        for word in ("yes", "no"):
            if re.search(rf"\b{word}\b", t):
                return word
        return t  # fall through to exact match
    
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
        
        dataset_type = predictions[0].get("dataset_type", "pubmedqa") if predictions else "pubmedqa"

        for pred, gt in zip(predictions, ground_truth):
            raw_pred = pred.get("final_answer", "")
            raw_gt   = gt.get("answer", "")

            if dataset_type == "medqa":
                # Ground truth is an option letter (A-E); extract letter from prediction.
                pred_ans = self.extract_option_letter(raw_pred) or self.normalize_text(raw_pred)
                gt_ans   = self.normalize_text(raw_gt)
            else:
                # PubMedQA: extract bare yes/no from potentially verbose output.
                pred_ans = self.extract_binary_answer(raw_pred)
                gt_ans   = self.normalize_text(raw_gt)

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
