"""
Self-verifying clinical Reasoning on the LOOP MedRAG Production Package
Medical Question Answering with Self-Reflective RAG
"""

__version__ = "1.0.0"
__author__ = "Self-MedRAG Team"

from .corpus_loader import CorpusLoader
from .dataset_loader import DatasetLoader
from .retrieval import RetrievalModule
from .model import GeneratorModule, SelfReflectiveModule
from .trainer import Trainer
from .evaluation import Evaluation
from .pipeline import MedRAGPipeline

__all__ = [
    "CorpusLoader",
    "DatasetLoader",
    "RetrievalModule",
    "GeneratorModule",
    "SelfReflectiveModule",
    "Trainer",
    "Evaluation",
    "MedRAGPipeline",
]
