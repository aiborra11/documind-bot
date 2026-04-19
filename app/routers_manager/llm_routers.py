from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status

from app.utils.utils import get_logger
from app.llm_manager.llm_service import LLMService
from app.retrieval_manager.rag_service import RAGService
from app.routers_manager.dependencies import get_rag_service, get_llm_service

logger = get_logger(__name__)

llm_router = APIRouter(prefix="/qa", tags=["Question & Answering"])


class QueryRequest(BaseModel):
    """Payload schema for end-to-end RAG requests."""
    query: str = Field(..., min_length=3, description="The natural language question to ask the financial bot.")
    initial_top_k: Optional[int] = Field(default=10, ge=1, le=50, description="Documents to retrieve from Vector DB.")
    final_top_n: Optional[int] = Field(default=3, ge=1, le=20, description="Documents to send to the LLM after re-ranking.")
    threshold: Optional[float] = Field(default=1.0, description="Optional distance threshold. Note is cosine distance, so lower is more similar.")

@llm_router.post("/ask", status_code=status.HTTP_200_OK)
async def ask_question_endpoint(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    End-to-End RAG Pipeline Endpoint.
    1. Retrieves candidate chunks from ChromaDB.
    2. Re-ranks them using the Cross-Encoder.
    3. Generates a cited answer using the local LLM via LangChain.
    """
    try:
        # Stage 1 & 2: Broad Retrieval & Deep Re-ranking
        retrieved_chunks = rag_service.search_similar(
            query=request.query,
            initial_top_k=request.initial_top_k,
            final_top_n=request.final_top_n,
            threshold=request.threshold
        )

        # Stage 3: LLM Generation (Asynchronous)
        final_answer = await llm_service.generate_response(
            query=request.query,
            retrieved_chunks=retrieved_chunks
        )

        return {
            "status": "success",
            "query": request.query,
            "answer": final_answer,
            "sources_used": len(retrieved_chunks),
            "context_metadata": [chunk["metadata"] for chunk in retrieved_chunks]
        }

    except Exception as e:
        logger.error(f"Failed to execute full RAG pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred while generating the answer."
        )