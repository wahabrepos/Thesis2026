"""retrieval.py

This module implements the RetrievalModule class which performs both sparse and dense retrieval
over a biomedical corpus. Sparse retrieval is done using BM25 (via the rank_bm25 library)
and dense retrieval is performed with a pretrained Contriever model (using Hugging Face Transformers)
and FAISS for efficient similarity search. The individual ranked lists are then merged using
Reciprocal Rank Fusion (RRF) with the constant K read from configuration.
"""

import hashlib
import logging
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Stopword list — hardcoded to avoid an NLTK download dependency.
# Combines standard English stopwords with common medical filler words that
# carry no discriminative value in BM25 (e.g. "patient", "study", "also").
# Medical entity terms (drug names, anatomical terms, conditions) are kept.
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    # English function words
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "as", "not", "no",
    "nor", "so", "yet", "both", "either", "neither", "each", "than",
    "such", "up", "out", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "while", "where", "when", "who",
    "which", "their", "they", "them", "he", "she", "we", "our", "us",
    "also", "however", "therefore", "thus", "hence", "whereas",
    "moreover", "furthermore", "although", "because", "since", "whether",
    # High-frequency medical filler words
    "patient", "patients", "study", "studies", "result", "results",
    "method", "methods", "conclusion", "conclusions", "background",
    "objective", "purpose", "aim", "aims", "found", "showed", "shown",
    "using", "used", "based", "associated", "compared", "significantly",
    "among", "including", "included", "reported", "data", "analysis",
    "clinical", "case", "cases", "group", "groups", "effect", "effects",
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalModule:
    """
    RetrievalModule implements a dual retrieval approach using BM25 and dense retrieval (Contriever)
    and fuses the results using Reciprocal Rank Fusion (RRF).

    Attributes:
        use_idf (bool): Flag to indicate whether to use IDF weighting in BM25.
        dense_model_name (str): Name of the pretrained dense retrieval model (e.g., "Contriever-msmarco").
        rrf_k (int): Constant K for Reciprocal Rank Fusion.
        corpus (List[str]): List of documents that form the retrieval corpus.
        bm25 (BM25Okapi): BM25 index instance built on the tokenized corpus.
        dense_tokenizer (AutoTokenizer): Tokenizer for the dense retrieval model.
        dense_model (AutoModel): Dense retrieval model.
        faiss_index (faiss.IndexFlatIP): FAISS index built on the dense embeddings.
        device (torch.device): Device to run the dense model on.
    """

    def __init__(self, config: Dict[str, Any], corpus: List[str] = None) -> None:
        """
        Initialize the RetrievalModule with the given configuration and corpus.
        If no corpus is provided, a default dummy corpus is used.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            corpus (List[str], optional): List of document strings for the retrieval corpus.
                Defaults to a dummy corpus.
        """
        # Retrieve BM25 configuration
        bm25_config: Dict[str, Any] = config.get("retrieval", {}).get("bm25", {})
        self.use_idf: bool = bm25_config.get("use_idf", True)

        # Retrieve dense retrieval configuration
        dense_config: Dict[str, Any] = config.get("retrieval", {}).get("dense", {})
        self.dense_model_name: str = dense_config.get("model_name", "Contriever-msmarco")

        # Retrieve RRF constant
        rrf_config: Dict[str, Any] = config.get("retrieval", {}).get("rrf", {})
        self.rrf_k: int = rrf_config.get("k", 60)

        # Cache directory for FAISS index
        self.cache_dir = Path(config.get("system", {}).get("cache_dir", "./cache")) / "faiss"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"BM25 use_idf: {self.use_idf}")
        logger.info(f"Dense retrieval model: {self.dense_model_name}")
        logger.info(f"RRF constant K: {self.rrf_k}")

        # Prepare corpus: use provided corpus or a default dummy corpus
        if corpus is None:
            self.corpus: List[str] = [
                "This is an article on biomedical image processing.",
                "Clinical trial results in oncology show promising outcomes.",
                "A review on cardiovascular disease studies and treatments.",
                "Study of the effects of medication in pediatric patients.",
                "Meta analysis of randomized controlled trials in neurology."
            ]
            logger.info("No corpus provided. Using default dummy corpus.")
        else:
            self.corpus = corpus

        # Build BM25 index on tokenized corpus
        tokenized_corpus: List[List[str]] = [self._tokenize(doc) for doc in self.corpus]
        self.bm25: BM25Okapi = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built on the corpus.")

        # Set up dense retrieval: keep on CPU to reserve GPU memory for the generator
        self.device: torch.device = torch.device("cpu")
        self.dense_tokenizer = AutoTokenizer.from_pretrained(self.dense_model_name)
        self.dense_model = AutoModel.from_pretrained(self.dense_model_name)
        self.dense_model.to(self.device)
        self.dense_model.eval()
        logger.info(f"Dense retrieval model '{self.dense_model_name}' loaded on device: {self.device}.")

        # Compute dense embeddings for the corpus documents and build FAISS index
        self._build_dense_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing and querying.

        Pipeline:
          1. Lowercase
          2. Strip punctuation (preserves hyphens inside words like "HbA1c",
             "IL-6", "beta-blocker" — only leading/trailing punctuation removed)
          3. Split on whitespace
          4. Remove stopwords and single-character tokens

        Medical hyphenated terms and alphanumeric codes (e.g. "COVID-19",
        "TNF-alpha", "HbA1c") are preserved intact.
        """
        text = text.lower()
        # Remove punctuation except hyphens surrounded by word characters
        text = re.sub(r"[^\w\s-]", " ", text)          # drop non-word, non-hyphen chars
        text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)  # drop lone hyphens
        tokens = text.split()
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    def _corpus_cache_key(self) -> str:
        """Generate a short hash key from corpus size + model name + first/last docs."""
        sig = f"{len(self.corpus)}|{self.dense_model_name}|{self.corpus[0][:80]}|{self.corpus[-1][:80]}"
        return hashlib.md5(sig.encode()).hexdigest()[:16]

    def _build_dense_index(self) -> None:
        """
        Computes dense embeddings for the corpus documents using the pretrained dense model
        and builds a FAISS index for efficient similarity search.
        Saves/loads the index from disk cache to avoid recomputing on every run.
        """
        cache_key = self._corpus_cache_key()
        index_path = self.cache_dir / f"index_{cache_key}.faiss"
        embeddings_path = self.cache_dir / f"embeddings_{cache_key}.npy"

        # Load from cache if available
        if index_path.exists() and embeddings_path.exists():
            logger.info(f"Loading FAISS index from cache: {index_path}")
            self.faiss_index = faiss.read_index(str(index_path))
            logger.info(f"FAISS index loaded from cache ({self.faiss_index.ntotal} vectors).")
            return

        logger.info("Building dense index (this runs once — will be cached to disk)...")

        # Batch embedding — batch_size=8 + max_length=256 balances speed vs truncation on ARM CPU
        batch_size = 8
        all_embeddings: List[np.ndarray] = []
        for batch_start in range(0, len(self.corpus), batch_size):
            batch = self.corpus[batch_start: batch_start + batch_size]
            inputs = self.dense_tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=256, padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.dense_model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            if (batch_start // batch_size) % 500 == 0 and batch_start > 0:
                done = min(batch_start + batch_size, len(self.corpus))
                logger.info(f"Embedded {done}/{len(self.corpus)} documents...")

        dense_embeddings: np.ndarray = np.vstack(all_embeddings).astype("float32")
        num_docs, dim = dense_embeddings.shape
        logger.info(f"Computed dense embeddings for {num_docs} documents; embedding dimension: {dim}.")

        # Build and save FAISS index
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(dense_embeddings)
        faiss.write_index(self.faiss_index, str(index_path))
        np.save(str(embeddings_path), dense_embeddings)
        logger.info(f"FAISS index built and saved to cache: {index_path}")
        logger.info("FAISS index for dense retrieval built and embeddings added.")

    def retrieve(self, query: str) -> Dict[str, List[str]]:
        """
        Retrieve relevant documents for the given query using both BM25 sparse retrieval
        and dense retrieval with Contriever.

        Args:
            query (str): The input query string.

        Returns:
            Dict[str, List[str]]: A dictionary with keys "bm25" and "dense" containing lists of
            retrieved document texts.
        """
        # Tokenize query the same way the corpus was indexed
        query_tokens: List[str] = self._tokenize(query)
        top_k: int = min(10, len(self.corpus))  # Number of top documents to retrieve
        bm25_results: List[str] = self.bm25.get_top_n(query_tokens, self.corpus, n=top_k)
        logger.info(f"BM25 retrieval returned {len(bm25_results)} documents.")

        # Dense retrieval: encode query using the dense model
        inputs = self.dense_tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
        token_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # shape: (1, seq_len, 1)
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        query_embedding = sum_embeddings / sum_mask
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        query_vec: np.ndarray = query_embedding.cpu().numpy().astype("float32")

        # Search FAISS index for top_k similar documents
        if len(self.corpus) == 0:
            dense_results: List[str] = []
            logger.warning("Corpus is empty, dense retrieval yields no results.")
        else:
            distances, indices = self.faiss_index.search(query_vec, top_k)
            dense_results = [self.corpus[idx] for idx in indices[0] if idx < len(self.corpus)]
            logger.info(f"Dense retrieval returned {len(dense_results)} documents.")

        return {"bm25": bm25_results, "dense": dense_results}

    def fuse_results(self, bm25_results: List[str], dense_results: List[str]) -> List[str]:
        """
        Fuse the BM25 and dense retrieval results using Reciprocal Rank Fusion (RRF).

        Each document's contribution is computed as 1/(K + rank) over its rank in each list.
        If a document appears in both lists, the scores are additive.

        Args:
            bm25_results (List[str]): List of documents retrieved by BM25.
            dense_results (List[str]): List of documents retrieved by dense retrieval.

        Returns:
            List[str]: The final fused and ranked list of documents.
        """
        fused_scores: Dict[str, float] = {}

        # Process BM25 results with ranks starting at 1
        for rank, doc in enumerate(bm25_results, start=1):
            score: float = 1.0 / (self.rrf_k + rank)
            if doc in fused_scores:
                fused_scores[doc] += score
            else:
                fused_scores[doc] = score
            logger.debug(f"BM25 doc '{doc[:30]}...' at rank {rank} gets score {score:.4f}.")

        # Process Dense results with ranks starting at 1
        for rank, doc in enumerate(dense_results, start=1):
            score = 1.0 / (self.rrf_k + rank)
            if doc in fused_scores:
                fused_scores[doc] += score
            else:
                fused_scores[doc] = score
            logger.debug(f"Dense doc '{doc[:30]}...' at rank {rank} gets score {score:.4f}.")

        # Sort documents by cumulative score in descending order
        sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        fused_list: List[str] = [doc for doc, _ in sorted_docs]
        logger.info(f"Fused results produced {len(fused_list)} documents after RRF.")
        return fused_list
