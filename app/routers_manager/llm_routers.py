from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status

from app.utils.utils import get_logger
from app.llm_manager.llm_service import LLMService
from app.rag_manager.rag_service import RAGService
from app.routers_manager.dependencies_service import get_rag_service, get_llm_service
from app.routers_manager.routers_config import QueryRequest

logger = get_logger(__name__)

llm_router = APIRouter(prefix="/qa", tags=["Question & Answering"])


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