from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.utils.utils import get_logger
from app.routers_manager.dependencies import get_rag_service
from app.retrieval_manager.rag_service import RAGService

logger = get_logger(__name__)

rag_router = APIRouter(prefix="/qa", tags=["Question & Answering"])

class QueryRequest(BaseModel):
    """Payload schema for semantic search requests."""
    query: str = Field(..., min_length=3, description="The natural language question to search for.")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to retrieve.")
    threshold: Optional[float] = Field(default=None, description="Optional distance threshold for filtering.")

@rag_router.post("/search", status_code=status.HTTP_200_OK)
async def semantic_search_endpoint(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Endpoint to test semantic retrieval.
    Takes a natural language query and returns the most relevant document chunks.
    """
    try:
        results = rag_service.search_similar(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )

        return {
            "status": "success",
            "query": request.query,
            "total_results": len(results),
            "data": results
        }

    except Exception as e:
        logger.error(f"Failed to execute search endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred during semantic search."
        )