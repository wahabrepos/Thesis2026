"""
Utility Functions
Configuration loading, logging setup, and helper functions
"""

import os
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution."""
    # Load environment variables
    load_dotenv()
    
    # Read YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    config = _substitute_env_vars(config)
    
    return config


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${ENV_VAR} in configuration."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Replace ${VAR} or ${VAR:default}
        if "${" in obj:
            import re
            pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.getenv(var_name, default)
            
            return re.sub(pattern, replacer, obj)
    return obj


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get("logging", {})
    
    # Create logs directory
    log_file = log_config.get("log_file", "./logs/medrag.log")
    log_file = log_file.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Console level
    console_level = getattr(logging, log_config.get("console_level", "INFO"))
    file_level = getattr(logging, log_config.get("file_level", "DEBUG"))
    
    # Format
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized: console={console_level}, file={file_level}")
    logging.info(f"Log file: {log_file}")


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename if {timestamp} present
    filepath = filepath.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_gpu_memory() -> Dict[str, float]:
    """Get GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
            }
    except:
        pass
    
    return {}


def clear_cuda_cache() -> None:
    """Clear CUDA cache to free memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.debug("CUDA cache cleared")
    except:
        pass
