from typing import List, Dict, Any, Optional

from app.utils.utils import get_logger
from app.database_manager.chroma_client import ChromaManager
from app.retrieval_manager.rag_config import rag_config

logger = get_logger(__name__)

class RAGService:
    """
    Handles RAG (Retrieval-Augmented Generation) operations:
    - Vector document retrieval (Semantic Search)
    """

    def __init__(self, db_client: ChromaManager) -> None:
        """
        Initializes the service using Dependency Injection.
        """
        if not db_client.collection:
            logger.error("RAGService initialized with an inactive ChromaDB collection.")
            raise ValueError("Database collection is not ready.")
            
        self._collection = db_client.collection

    def search_similar(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search in the Vector DB to retrieve the most relevant chunks.
        """
        top_k = top_k or rag_config.DEFAULT_QUERY_TOP_K

        logger.info(f"Executing vector retrieval for query: '{query}'")

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            if not results.get("documents") or not results["documents"][0]:
                logger.info("No documents found in the vector database.")
                return []

            retrieved_docs: List[Dict[str, Any]] = []
            
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # Optional distance filtering (Depends on metric, e.g., L2 or Cosine)
                if threshold is not None and float(distance) > threshold:
                    continue

                retrieved_docs.append({
                    "content": doc,
                    "metadata": metadata,
                    "vector_distance": float(distance)
                })

            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents.")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process retrieval query: {str(e)}")