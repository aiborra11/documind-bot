import hashlib
from typing import List, Dict, Any, Set
from chromadb.api.models.Collection import Collection

from app.utils.utils import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    """
    Handles optimized storage of text chunks in ChromaDB.
    Implements idempotency checks to avoid redundant embedding generation.
    """

    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    def store_embeddings(self, chunks: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
        """
        Stores chunks in ChromaDB using an upsert strategy.
        Filters out already existing chunks to save compute resources.
        """
        if not chunks:
            raise ValueError("No chunks provided for processing.")

        # Generate IDs to prevent duplicates and enable traceability.
        all_ids = [
            self._generate_id(filename, i, chunk["text"]) 
            for i, chunk in enumerate(chunks)
        ]

        existing_ids = self._get_existing_ids(all_ids)
        
        # Filter chunks that actually need embedding
        new_indices = [i for i, _id in enumerate(all_ids) if _id not in existing_ids]
        
        if not new_indices:
            logger.info(f"All chunks from {filename} already exist. Skipping embedding generation.")
            return {
                "status": "skipped",
                "total_chunks": len(chunks),
                "new_chunks_added": 0
            }

        # Prepare data for new chunks only
        filtered_documents = [chunks[i]["text"] for i in new_indices]
        filtered_metadatas = [chunks[i]["metadata"] for i in new_indices]
        filtered_ids = [all_ids[i] for i in new_indices]

        try:
            self._collection.upsert(
                documents=filtered_documents,
                metadatas=filtered_metadatas,
                ids=filtered_ids
            )
            logger.info(f"Stored {len(filtered_ids)} new embeddings for {filename}.")
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors into database: {e}")
            raise RuntimeError("Database persistence failed.")

        return {
            "status": "processed",
            "total_chunks": len(chunks),
            "new_chunks_added": len(new_indices),
            "collection_total": self._collection.count()
        }

    def _get_existing_ids(self, ids: List[str]) -> Set[str]:
        """Checks the collection for existing IDs to avoid re-processing."""
        try:
            results = self._collection.get(ids=ids, include=[])
            return set(results.get("ids", []))
        except Exception:
            return set()

    def _generate_id(self, filename: str, index: int, text: str) -> str:
        """
        Generates a unique ID. 
        Note: If the filename changes, the ID changes. 
        To avoid duplicates even with different names, use only the text hash.
        """
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{filename}_{index}_{text_hash[:12]}"