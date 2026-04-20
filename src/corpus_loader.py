"""
Corpus Loader for Medical Literature
Loads PubMed abstracts, textbooks, and clinical guidelines
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class CorpusLoader:
    """Load and process medical corpus from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize corpus loader with configuration."""
        self.config = config
        corpus_config = config.get("corpus", {})
        
        self.sources = corpus_config.get("sources", [])
        self.chunk_size = corpus_config.get("chunk_size", 512)
        self.chunk_overlap = corpus_config.get("chunk_overlap", 50)
        self.min_chunk_length = corpus_config.get("min_chunk_length", 100)
        
        self.corpus: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
    
    def load_corpus(self) -> List[str]:
        """Load corpus from all configured sources."""
        logger.info("Loading medical corpus from configured sources")
        
        all_documents = []
        
        for source in self.sources:
            source_type = source.get("type", "unknown")
            source_path = source.get("path", "")
            weight = source.get("weight", 1.0)
            max_docs = source.get("max_docs", None)
            
            logger.info(f"Loading {source_type} from {source_path}")
            
            try:
                docs = self._load_source(source_path, max_docs)
                logger.info(f"Loaded {len(docs)} documents from {source_type}")
                all_documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Failed to load {source_type}: {e}")
                continue
        
        # Chunk documents
        self.corpus = self._chunk_documents(all_documents)
        logger.info(f"Total corpus size: {len(self.corpus)} chunks")
        
        return self.corpus
    
    def _load_source(self, path: str, max_docs: Optional[int] = None) -> List[str]:
        """Load documents from a single source file."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            logger.warning(f"Corpus file not found: {path}")
            return []
        
        documents = []
        
        try:
            if path_obj.suffix == ".jsonl":
                # JSONL format (one document per line)
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if max_docs and i >= max_docs:
                            break
                        
                        doc = json.loads(line)
                        text = self._extract_text(doc)
                        if text:
                            documents.append(text)
            
            elif path_obj.suffix == ".json":
                # JSON array format
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for i, doc in enumerate(data):
                            if max_docs and i >= max_docs:
                                break
                            text = self._extract_text(doc)
                            if text:
                                documents.append(text)
            
            elif path_obj.suffix == ".txt":
                # Plain text (one document per line or entire file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split by double newline for documents
                    docs = content.split("\n\n")
                    documents.extend([d.strip() for d in docs if d.strip()])
            
            else:
                logger.warning(f"Unsupported file format: {path_obj.suffix}")
        
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
        
        return documents[:max_docs] if max_docs else documents
    
    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Extract text from document dictionary."""
        # Try common fields
        for field in ["text", "abstract", "content", "body", "passage"]:
            if field in doc and doc[field]:
                return str(doc[field]).strip()
        
        # If document has title and abstract, combine them
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        if title and abstract:
            return f"{title}. {abstract}".strip()
        
        return ""
    
    def _chunk_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks with overlap."""
        chunks = []
        
        for doc in documents:
            # Split by sentences (simple approach)
            sentences = doc.split(". ")
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunk_text = ". ".join(current_chunk) + "."
                        if len(chunk_text.split()) >= self.min_chunk_length:
                            chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_chunk[-(self.chunk_overlap // 50):]
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add final chunk
            if current_chunk:
                chunk_text = ". ".join(current_chunk) + "."
                if len(chunk_text.split()) >= self.min_chunk_length:
                    chunks.append(chunk_text)
        
        return chunks
    
    def get_corpus(self) -> List[str]:
        """Get loaded corpus."""
        if not self.corpus:
            self.load_corpus()
        return self.corpus
