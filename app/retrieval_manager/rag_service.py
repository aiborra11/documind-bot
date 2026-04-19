from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

# --------- Project Imports ---------
from app.utils.utils import get_logger
from app.database_manager.chroma_client import ChromaManager
from app.retrieval_manager.rag_config import rag_config

logger = get_logger(__name__)

class CrossEncoderReRanker:
    """
    Implements advanced re-ranking using a Cross-Encoder model.
    """
    def __init__(self) -> None:
        model_name = getattr(rag_config, "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info(f"Initializing CrossEncoder with model: {model_name}")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """Scores and reorders documents based on cross-encoder similarity."""
        if not documents:
            return []

        logger.info(f"Re-ranking {len(documents)} documents for query: '{query}'")
        
        # Prepare pairs for the cross-encoder, forcing string types
        sentence_pairs = [[str(query), str(doc["content"])] for doc in documents]
        scores = self._model.predict(sentence_pairs)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        best_docs = reranked_docs[:top_n]
        
        logger.debug(f"Selected top {len(best_docs)} documents after re-ranking.")
        return best_docs

class RAGService:
    """
    Handles RAG (Retrieval-Augmented Generation) operations:
    - Vector document retrieval (Semantic Search)
    """

    def __init__(self, db_client: ChromaManager, reranker: CrossEncoderReRanker) -> None:
        """
        Initializes the service using Dependency Injection.
        """
        if not db_client.collection:
            logger.error("RAGService initialized with an inactive ChromaDB collection.")
            raise ValueError("Database collection is not ready.")
            
        self._collection = db_client.collection
        self._reranker = reranker

    def search_similar(
        self,
        query: str,
        initial_top_k: Optional[int] = None,
        final_top_n: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a two-stage retrieval process.
        """
        initial_top_k = initial_top_k or getattr(rag_config, "DEFAULT_INITIAL_TOP_K", 10)
        final_top_n = final_top_n or getattr(rag_config, "DEFAULT_FINAL_TOP_N", 3)

        logger.info(f"Executing Stage 1: Vector retrieval for query: '{query}'")

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=initial_top_k,
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
                if not doc:
                    logger.warning(f"Skipping empty document from database. Metadata: {metadata}")
                    continue

                if threshold is not None and float(distance) > threshold:
                    continue

                retrieved_docs.append({
                    "content": str(doc),
                    "metadata": metadata,
                    "vector_distance": float(distance)
                })

            logger.info(f"Retrieved {len(retrieved_docs)} candidate documents. Moving to Stage 2.")
            
            best_docs = self._reranker.rerank(
                query=query,
                documents=retrieved_docs,
                top_n=final_top_n
            )

            return best_docs

        except Exception as e:
            logger.error(f"Error during similarity search and re-ranking: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process retrieval query: {str(e)}")
        