
import os
import logging
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SemanticRetrieval:
    """Simple semantic search using LangChain FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embeddings.

        Args:
            model_name: Sentence transformer model to use
        """
        logger.info(f"Loading embedding model: {model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vectorstores: Dict[str, FAISS] = {}
        logger.info("Retrieval system initialized")

    def add_document(self, index_name: str, filename: str, text: str):
        """
        Add a document to the specified index.

        Args:
            index_name: Name of the index
            filename: Document filename
            text: Document text content
        """
        doc = Document(
            page_content=text,
            metadata={"filename": filename}
        )

        if index_name not in self.vectorstores:
            # Create new vector store
            self.vectorstores[index_name] = FAISS.from_documents([doc], self.embeddings)
            logger.info(f"Created new index: {index_name}")
        else:
            # Add to existing vector store
            self.vectorstores[index_name].add_documents([doc])
            logger.info(f"Added document to index: {index_name}")

        # Save index
        self._save_index(index_name)

    def search(self, index_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search documents in the specified index.

        Args:
            index_name: Name of the index to search
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results
        """
        if index_name not in self.vectorstores:
            logger.warning(f"Index not found: {index_name}")
            return []

        try:
            # Perform similarity search
            results = self.vectorstores[index_name].similarity_search_with_score(
                query,
                k=top_k
            )

            # Format results
            formatted_results = []
            for i, (doc, score) in enumerate(results):
                formatted_results.append({
                    'rank': i + 1,
                    'filename': doc.metadata.get('filename', 'unknown'),
                    'score': round(float(score), 4),
                    'text_snippet': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                })

            logger.info(f"Found {len(formatted_results)} results in index '{index_name}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _save_index(self, index_name: str):
        """Save index to disk."""
        if index_name not in self.vectorstores:
            return

        try:
            index_path = f"data/models/{index_name}"
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.vectorstores[index_name].save_local(index_path)
            logger.info(f"Saved index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load_index(self, index_name: str) -> bool:
        """
        Load index from disk.

        Args:
            index_name: Name of the index to load

        Returns:
            True if loaded successfully
        """
        try:
            index_path = f"data/models/{index_name}"
            if os.path.exists(index_path):
                self.vectorstores[index_name] = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded index: {index_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False