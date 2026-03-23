"""
Trainer Module - Iterative Refinement Loop
Implements Algorithm 1 from Self-MedRAG paper with proper iteration tracking
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from .retrieval import RetrievalModule
from .model import GeneratorModule, SelfReflectiveModule

logger = logging.getLogger(__name__)


class Trainer:
    """Implements the iterative retrieval-generation-reflection loop."""
    
    def __init__(
        self,
        retrieval: RetrievalModule,
        generator: GeneratorModule,
        self_reflector: SelfReflectiveModule,
        config: Dict[str, Any]
    ):
        """Initialize trainer with modules and configuration."""
        self.retrieval = retrieval
        self.generator = generator
        self.self_reflector = self_reflector
        self.config = config
        
        # Iteration settings
        iteration_config = config.get("iteration", {})
        self.max_iterations = iteration_config.get("max_iterations", 3)
        self.early_stopping = iteration_config.get("early_stopping", True)
        self.min_improvement = iteration_config.get("min_improvement", 0.05)
        self.max_time_seconds = iteration_config.get("max_time_seconds", 300)

        # top_k: how many passages to pass to generator and NLI after RRF fusion
        self.top_k = config.get("retrieval", {}).get("top_k", 5)
        
        # Self-reflection thresholds
        sr_config = config.get("self_reflection", {})
        self.rationale_threshold = sr_config.get("rationale_score_threshold", 0.7)
        
        logger.info(f"Trainer initialized: max_iterations={self.max_iterations}")
    
    def run(self, query: str) -> Tuple[str, List[str], int, float, List[Dict[str, Any]]]:
        """
        Execute iterative loop for a query.
        
        Returns:
            Tuple of (answer, rationale, iteration_count, support_score, history)
        """
        start_time = time.time()
        iteration = 0
        history: List[Dict[str, Any]] = []
        current_query = query.strip()
        
        final_answer = ""
        final_rationale = []
        final_support_score = 0.0
        final_confidence = 0.0
        final_citations = []
        
        logger.info(f"Starting iterative loop for query: '{current_query[:100]}...'")
        
        while iteration < self.max_iterations:
            logger.info(f"--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # Check timeout
            if time.time() - start_time > self.max_time_seconds:
                logger.warning(f"Timeout after {iteration + 1} iterations")
                break
            
            try:
                # 1. Retrieval Phase
                retrieval_results = self.retrieval.retrieve(current_query)
                bm25_results = retrieval_results.get("bm25", [])
                dense_results = retrieval_results.get("dense", [])
                fused_context = self.retrieval.fuse_results(bm25_results, dense_results)
                # Trim to top_k — RRF returns all unique passages from both lists,
                # but feeding all 19 to the generator and NLI explodes latency linearly.
                fused_context = fused_context[: self.top_k]

                logger.info(f"Retrieved {len(fused_context)} passages (top_k={self.top_k})")
                
                # 2. Generation Phase
                answer, rationale, confidence, citations = self.generator.generate(
                    current_query,
                    fused_context,
                    history
                )
                
                logger.info(f"Generated answer with {len(rationale)} rationale statements")
                logger.info(f"Confidence: {confidence:.3f}")
                
                # 3. Self-Reflection Phase — single NLI pass returns both score and unsupported
                support_score, unsupported_segments = self.self_reflector.verify_and_extract(
                    rationale, fused_context
                )
                logger.info(f"Support score: {support_score:.3f} (threshold: {self.rationale_threshold})")

                # Store iteration results
                iteration_entry = {
                    "iteration": iteration + 1,
                    "query": current_query,
                    "context": fused_context,
                    "answer": answer,
                    "rationale": rationale,
                    "confidence": confidence,
                    "support_score": support_score,
                    "citations": citations,
                }
                history.append(iteration_entry)

                # Check if we can stop
                if support_score >= self.rationale_threshold:
                    logger.info(f"✓ Support threshold met ({support_score:.3f} >= {self.rationale_threshold})")
                    final_answer = answer
                    final_rationale = rationale
                    final_support_score = support_score
                    final_confidence = confidence
                    final_citations = citations
                    iteration += 1
                    break

                # Check for minimal improvement (early stopping)
                if self.early_stopping and iteration > 0:
                    prev_score = history[-2]["support_score"]
                    improvement = support_score - prev_score

                    if improvement < self.min_improvement:
                        logger.info(f"Early stopping: improvement {improvement:.3f} < {self.min_improvement}")
                        final_answer = answer
                        final_rationale = rationale
                        final_support_score = support_score
                        final_confidence = confidence
                        final_citations = citations
                        iteration += 1
                        break

                # 4. Query Refinement — unsupported_segments already computed above
                
                if unsupported_segments:
                    logger.info(f"Found {len(unsupported_segments)} unsupported statements")
                    
                    # Refine query
                    refined_segments = " ".join(unsupported_segments[:3])  # Limit to top 3
                    current_query = self._refine_query(query, refined_segments, iteration + 1)
                    
                    logger.info(f"Refined query: '{current_query[:100]}...'")
                else:
                    logger.info("No unsupported segments found, but score below threshold")
                    # Keep original query for next iteration
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}")
                # Use last valid results or defaults
                if iteration > 0:
                    last_entry = history[-1]
                    final_answer = last_entry["answer"]
                    final_rationale = last_entry["rationale"]
                    final_support_score = last_entry["support_score"]
                    final_confidence = last_entry["confidence"]
                    final_citations = last_entry["citations"]
                else:
                    final_answer = "Error during generation"
                    final_rationale = [str(e)]
                    final_support_score = 0.0
                    final_confidence = 0.0
                    final_citations = []
                iteration += 1
                break
        
        # If we exhausted iterations without meeting threshold
        if iteration == self.max_iterations and not final_answer:
            logger.info(f"Max iterations reached without meeting threshold")
            if history:
                # Use best result
                best_entry = max(history, key=lambda x: x["support_score"])
                final_answer = best_entry["answer"]
                final_rationale = best_entry["rationale"]
                final_support_score = best_entry["support_score"]
                final_confidence = best_entry["confidence"]
                final_citations = best_entry["citations"]
            else:
                final_answer = "Unable to generate answer"
                final_rationale = ["No valid iterations completed"]
                final_support_score = 0.0
                final_confidence = 0.0
                final_citations = []
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed in {iteration} iterations ({elapsed_time:.2f}s)")
        logger.info(f"Final support score: {final_support_score:.3f}")
        
        return (
            final_answer,
            final_rationale,
            iteration,  # FIXED: Now returns actual iteration count
            final_support_score,
            history
        )
    
    def _refine_query(self, original_query: str, unsupported_text: str, iteration: int) -> str:
        """Refine query based on unsupported statements."""
        strategy = self.config.get("iteration", {}).get("refinement_strategy", "structured")
        
        if strategy == "structured":
            return (
                f"{original_query} "
                f"Find specific evidence for: {unsupported_text}"
            )
        elif strategy == "decomposition":
            return (
                f"Sub-question {iteration}: "
                f"What evidence supports that {unsupported_text}?"
            )
        else:  # concatenation
            return f"{original_query} {unsupported_text}"
