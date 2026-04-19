from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

# --------- Project Imports ---------
from app.utils.utils import get_logger
from app.routers_manager.dependencies import get_rag_service
from app.retrieval_manager.rag_service import RAGService

logger = get_logger(__name__)

rag_router = APIRouter(prefix="/qa", tags=["Question & RAG Retrieval"])

class QueryRequest(BaseModel):
    """Payload schema for two-stage semantic search requests."""
    query: str = Field(
        ..., 
        min_length=3, 
        description="The natural language question to search for."
    )
    initial_top_k: Optional[int] = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Number of results to retrieve from the vector database (Stage 1)."
    )
    final_top_n: Optional[int] = Field(
        default=3, 
        ge=1, 
        le=20, 
        description="Number of results to return after cross-encoder re-ranking (Stage 2)."
    )
    threshold: Optional[float] = Field(
        default=None, 
        description="Optional distance threshold for Stage 1 filtering. Leave null to avoid strict cutoffs."
    )


@rag_router.post("/search", status_code=status.HTTP_200_OK)
async def semantic_search_endpoint(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Endpoint to test two-stage semantic retrieval.
    Takes a natural language query, retrieves candidates from ChromaDB, 
    and re-ranks them using a Cross-Encoder for maximum relevance.
    """
    try:
        results = rag_service.search_similar(
            query=request.query,
            initial_top_k=request.initial_top_k,
            final_top_n=request.final_top_n,
            threshold=request.threshold
        )

        return {
            "status": "success",
            "query": request.query,
            "total_returned": len(results),
            "data": results
        }

    except Exception as e:
        logger.error(f"Failed to execute search endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred during semantic search and re-ranking."
        )