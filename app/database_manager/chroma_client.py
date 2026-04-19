import chromadb

from typing import Optional, Any, Dict
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection

# -------------- Project Imports --------------
from app.utils.utils import get_logger
from app.database_manager.database_config import db_config


logger = get_logger(__name__)

class ChromaManager:
    """
    Manages the lifecycle of the ChromaDB connection and vector collection.
    Designed to be instantiated safely and passed via Dependency Injection.
    """
    def __init__(self) -> None:
        self._client: Optional[chromadb.PersistentClient] = None
        self._embedding_fn: Optional[embedding_functions.EmbeddingFunction] = None
        self._collection: Optional[Collection] = None

    @property
    def collection(self) -> Collection:
        """
        Read-only property to safely access the collection.
        Prevents external mutation and ensures connection state.
        """
        if not self._collection:
            raise RuntimeError("ChromaDB connection is not established. Call 'connect()' first.")
        return self._collection

    def connect(self) -> None:
        """Initializes client, embedding function, and vector collection."""
        if self._client is not None:
            logger.warning("ChromaDB client is already connected. Skipping initialization.")
            return

        try:
            logger.info(f"Connecting to ChromaDB at {db_config.VECTOR_DB_PATH}")
            self._client = chromadb.PersistentClient(path=db_config.VECTOR_DB_PATH)
            
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=db_config.EMBEDDING_MODEL
            )
            
            self._collection = self._initialize_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB connection: {str(e)}")
            self.close() 
            raise

    def close(self) -> None:
        """Cleans up memory references after closing connection."""
        logger.info("Closing ChromaDB resources.")
        self._client = None
        self._embedding_fn = None
        self._collection = None

    def _initialize_collection(self) -> Collection:
        """Retrieves or creates the vector collection with specified metrics."""
        if not self._client or not self._embedding_fn:
            raise RuntimeError("Client and embedding function must be initialized before collection creation.")
            
        metadata: Dict[str, Any] = {
            db_config.METADATA_DISTANCE_SPACE: db_config.DISTANCE_METRIC
        }
        
        collection = self._client.get_or_create_collection(
            name=db_config.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata=metadata 
        )
        logger.info(f"Collection '{db_config.COLLECTION_NAME}' is ready.")

        return collection