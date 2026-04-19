from typing import Optional
from fastapi import Request, HTTPException, status, Depends

# ----- Project Imports -----
from app.utils.utils import get_logger
from app.database_manager.chroma_client import ChromaManager 
from app.documents_manager.document_service import DocumentProcessor
from app.database_manager.embedding_service import EmbeddingService
from app.rag_manager.rag_service import CrossEncoderReRanker, RAGService
from app.llm_manager.llm_service import LLMService, PromptBuilder

logger = get_logger(__name__)


# DATABASE DEPENDENCIES
def get_db_client(request: Request) -> ChromaManager:
    """
    Dependency to safely extract the database client from the application state.
    """
    # Retrieve the DB instance yielded during the lifespan startup
    db_client: Optional[ChromaManager] | None = getattr(request.app.state, "db", None)
    
    if db_client is None:
        logger.error("Database client not found in app.state")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not initialized."
        )
    return db_client


# EMBEDDINGS DEPENDENCIES
def get_document_processor() -> DocumentProcessor:
    """
    Dependency to inject the DocumentProcessor.
    Currently stateless, so we instantiate it directly.
    """
    return DocumentProcessor()

def get_embedding_service(
    db_client: ChromaManager = Depends(get_db_client)
) -> EmbeddingService:
    """
    Dependency to inject the EmbeddingService.
    Extracts the collection from the active DB client safely.
    """
    return EmbeddingService(collection=db_client.collection)



# RAG DEPENDENCIES
_reranker_instance: Optional[CrossEncoderReRanker] = None

def get_reranker() -> CrossEncoderReRanker:
    """Dependency to safely inject the CrossEncoderReRanker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReRanker()
    return _reranker_instance

def get_rag_service(
    db_client: ChromaManager = Depends(get_db_client),
    reranker: CrossEncoderReRanker = Depends(get_reranker)
) -> RAGService:
    """Dependency to inject the full RAG service."""
    return RAGService(db_client=db_client, reranker=reranker)


# LLM DEPENDENCIES
_llm_service_instance = None

def get_llm_service() -> LLMService:
    """
    Dependency to safely inject the LLM Service.
    Maintains a single instance to reuse the LCEL chain.
    """
    global _llm_service_instance
    if _llm_service_instance is None:
        prompt_builder = PromptBuilder()
        _llm_service_instance = LLMService(prompt_builder=prompt_builder) 
    return _llm_service_instance
