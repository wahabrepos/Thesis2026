"""
Unit tests for retrieval module
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from retrieval import RetrievalModule


def test_retrieval_init():
    """Test retrieval module initialization."""
    config = {
        "retrieval": {
            "bm25": {"use_idf": True},
            "dense": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            "rrf": {"k": 60}
        }
    }
    
    corpus = ["Medical document 1", "Medical document 2"]
    retrieval = RetrievalModule(config, corpus=corpus)
    
    assert retrieval is not None
    assert len(retrieval.corpus) == 2


def test_retrieval_search():
    """Test retrieval search."""
    config = {
        "retrieval": {
            "bm25": {"use_idf": True},
            "dense": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            "rrf": {"k": 60},
            "top_k": 2
        }
    }
    
    corpus = [
        "Hypertension treatment includes ACE inhibitors",
        "Diabetes is managed with insulin",
    ]
    
    retrieval = RetrievalModule(config, corpus=corpus)
    results = retrieval.retrieve("hypertension treatment")
    
    assert "bm25" in results
    assert "dense" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
