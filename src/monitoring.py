"""
Production Monitoring Module
Track metrics, performance, and system health
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class Monitor:
    """Production monitoring for Self-MedRAG system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitor with configuration."""
        self.config = config
        monitoring_config = config.get("monitoring", {})
        
        self.track_latency = monitoring_config.get("track_latency", True)
        self.track_memory = monitoring_config.get("track_memory", True)
        self.track_gpu = monitoring_config.get("track_gpu", True)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # Alerts
        self.enable_alerts = monitoring_config.get("enable_alerts", False)
        
        logger.info("Monitoring initialized")
    
    def log_query(self, query_data: Dict[str, Any]) -> None:
        """Log metrics for a single query."""
        if self.track_latency and "latency" in query_data:
            self.metrics["latency"].append(query_data["latency"])
        
        if "iterations" in query_data:
            self.metrics["iterations"].append(query_data["iterations"])
        
        if "support_score" in query_data:
            self.metrics["support_score"].append(query_data["support_score"])
        
        if "confidence" in query_data:
            self.metrics["confidence"].append(query_data["confidence"])
        
        # Track errors
        if query_data.get("error", False):
            self.metrics["errors"].append(1)
        else:
            self.metrics["errors"].append(0)
    
    def log_memory(self) -> Dict[str, float]:
        """Log current memory usage."""
        memory_info = {}
        
        if self.track_memory:
            # CPU memory
            process = psutil.Process()
            memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
            
            # GPU memory
            if self.track_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_info["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                        memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
                except:
                    pass
        
        return memory_info
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            "total_queries": len(self.metrics.get("latency", [])),
            "uptime_seconds": time.time() - self.start_time,
        }
        
        # Latency stats
        if "latency" in self.metrics and self.metrics["latency"]:
            latencies = self.metrics["latency"]
            summary["latency_mean"] = np.mean(latencies)
            summary["latency_p50"] = np.percentile(latencies, 50)
            summary["latency_p95"] = np.percentile(latencies, 95)
            summary["latency_max"] = np.max(latencies)
        
        # Iteration stats
        if "iterations" in self.metrics and self.metrics["iterations"]:
            summary["avg_iterations"] = np.mean(self.metrics["iterations"])
        
        # Support score stats
        if "support_score" in self.metrics and self.metrics["support_score"]:
            summary["avg_support_score"] = np.mean(self.metrics["support_score"])
        
        # Error rate
        if "errors" in self.metrics and self.metrics["errors"]:
            summary["error_rate"] = np.mean(self.metrics["errors"])
        
        # Memory
        summary["current_memory"] = self.log_memory()
        
        return summary
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file."""
        import json
        from datetime import datetime
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "raw_metrics": {k: list(v) for k, v in self.metrics.items()},
        }
        
        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
