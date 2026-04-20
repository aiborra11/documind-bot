import json
import time
from pathlib import Path

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.utils.utils import get_logger, extract_keywords
from app.llm_manager.llm_service import LLMService
from app.rag_manager.rag_service import RAGService
from app.routers_manager.dependencies_service import get_rag_service, get_llm_service
from app.routers_manager.routers_config import QueryRequest, EvalRequest

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
    


@llm_router.post("/evaluate", status_code=status.HTTP_200_OK)
async def evaluate_single_query_endpoint(
    request: EvalRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    Single-query evaluation endpoint.
    Executes the RAG pipeline and compares the output against expected ground truth.
    Returns the generated answer along with computed metrics (Hit Rate, Overlap).
    """
    try:
        # 1. Execute Retrieval
        retrieved_chunks = rag_service.search_similar(
            query=request.query,
            initial_top_k=request.initial_top_k,
            final_top_n=request.final_top_n,
            threshold=request.threshold
        )

        # 2. Execute LLM Generation
        final_answer = await llm_service.generate_response(
            query=request.query,
            retrieved_chunks=retrieved_chunks
        )

        # 3. Compute Metrics
        contexts = [chunk["metadata"] for chunk in retrieved_chunks]
        
        # Hit Calculations
        source_hit = any(request.expected_source in ctx.get("source", "") for ctx in contexts)
        page_hit = any(str(request.expected_page) == str(ctx.get("page", "")) for ctx in contexts)
        is_full_hit = source_hit and page_hit
        
        # Citation Calculation
        has_valid_citation = request.expected_source in final_answer

        # Overlap Calculation
        expected_kw = extract_keywords(request.expected_answer)
        generated_kw = extract_keywords(final_answer)
        
        overlap_score = 1.0
        if expected_kw:
            intersection = expected_kw.intersection(generated_kw)
            overlap_score = len(intersection) / len(expected_kw)

        return {
            "status": "success",
            "evaluation_metrics": {
                "retrieval_hit": is_full_hit,
                "valid_citation": has_valid_citation,
                "overlap_score": round(overlap_score, 2),
                "matching_kw": intersection,
            },
            "generated_answer": final_answer,
            "expected_answer": request.expected_answer,
            "retrieved_context": contexts
        }

    except Exception as e:
        logger.error(f"Failed to execute evaluation pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred while evaluating the query."
        )


@llm_router.post("/evaluate/batch", status_code=status.HTTP_200_OK)
async def evaluate_batch_endpoint(
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    Batch evaluation endpoint.
    Reads the eval.jsonl file, runs the full RAG pipeline for each question,
    and returns a quantitative summary of the system's performance.
    Warning: This is a synchronous long-running task.
    """
    # Dynamically resolve the path to the eval.jsonl file
    base_dir = Path(__file__).resolve().parent.parent.parent
    eval_file_path = base_dir / "data" / "input_data" / "eval.jsonl"

    if not eval_file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation file not found at {eval_file_path}"
        )

    total_questions = 0
    retrieval_hits = 0
    valid_citations = 0
    overlap_scores = []
    latencies = []
    detailed_results = []

    try:
        with open(eval_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                query = data["question"]
                expected_answer = data["expected_answer"]
                expected_source = data["expected_source"]
                expected_page = str(data["expected_page"])

                start_time = time.time()

                # 1. Pipeline Execution
                retrieved_chunks = rag_service.search_similar(
                    query=query, initial_top_k=10, final_top_n=3
                )
                final_answer = await llm_service.generate_response(
                    query=query, retrieved_chunks=retrieved_chunks
                )

                latency = time.time() - start_time
                latencies.append(latency)

                # 2. Compute Metrics for this specific query
                contexts = [chunk["metadata"] for chunk in retrieved_chunks]
                
                source_hit = any(expected_source in ctx.get("source", "") for ctx in contexts)
                page_hit = any(expected_page == str(ctx.get("page", "")) for ctx in contexts)
                is_full_hit = source_hit and page_hit
                
                if is_full_hit:
                    retrieval_hits += 1

                has_valid_citation = expected_source in final_answer
                if has_valid_citation:
                    valid_citations += 1

                # TODO: THIS SHOULD BE A LLM AS A JUDGE FUNCTION INSTEAD OF KEYWORD OVERLAP
                expected_kw = extract_keywords(expected_answer)
                generated_kw = extract_keywords(final_answer)
                
                overlap_score = 1.0
                if expected_kw:
                    intersection = expected_kw.intersection(generated_kw)
                    overlap_score = len(intersection) / len(expected_kw)
                
                overlap_scores.append(overlap_score)

                # Append to detailed report
                detailed_results.append({
                    "query": query,
                    "metrics": {
                        "retrieval_hit": is_full_hit,
                        "final_answer": final_answer,
                        "expected_answer": expected_answer,
                        "matching_kw": intersection,
                        "valid_citation": has_valid_citation,
                        "overlap_score": round(overlap_score, 2),
                        "latency_seconds": round(latency, 2)
                    }
                })

                total_questions += 1

        # 3. Aggregate Final Metrics
        if total_questions == 0:
            return {"status": "error", "message": "Evaluation file is empty."}

        hit_rate = (retrieval_hits / total_questions) * 100
        citation_rate = (valid_citations / total_questions) * 100
        avg_accuracy = (sum(overlap_scores) / len(overlap_scores)) * 100
        avg_latency = sum(latencies) / len(latencies)

        return {
            "status": "success",
            "summary": {
                "total_questions_analyzed": total_questions,
                "retrieval_hit_rate_percentage": round(hit_rate, 2),
                "valid_citation_rate_percentage": round(citation_rate, 2),
                "average_response_accuracy_percentage": round(avg_accuracy, 2),
                "average_latency_seconds": round(avg_latency, 2)
            },
            "detailed_results": detailed_results
        }

    except Exception as e:
        logger.error(f"Failed during batch evaluation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    