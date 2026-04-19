import os
import tempfile
from typing import Dict, Any

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.utils.utils import get_logger
from app.database_manager.chroma_client import ChromaManager    
from app.routers_manager.dependencies import get_db_client, get_document_processor, get_embedding_service
from app.ingestion_manager.document_processor import DocumentProcessor
from app.database_manager.embedding_service import EmbeddingService

logger = get_logger(__name__)

embeddings_router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

@embeddings_router.post("/embed-pdf", status_code=status.HTTP_200_OK)
async def embed_pdf_endpoint(
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> Dict[str, Any]:
    """
    End-to-end endpoint for PDF ingestion.
    Extracts text, applies recursive chunking, generates embeddings, 
    and stores them in the vector database.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid file format. Only PDF files are supported."
        )

    tmp_path = ""
    try:
        # Create a temporary file to allow PyMuPDF to read from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Processing uploaded file: {file.filename}")

        # 1. Extraction & Chunking
        raw_documents = processor.extract_text(tmp_path)
        chunks = processor.chunk_documents(raw_documents, file.filename)
        
        # 2. Save raw file for auditing/backup
        saved_path = processor.save_document(tmp_path, file.filename)

        # 3. Generate Embeddings & Store in Vector DB
        db_stats = embedding_service.store_embeddings(chunks, file.filename)

        return {
            "status": "success",
            "filename": file.filename,
            "saved_path": saved_path,
            "total_pages": len(raw_documents),
            "ingestion_results": db_stats
        }

    except ValueError as ve:
        logger.warning(f"Validation error during processing: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Database error during ingestion: {re}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store vectors in database.")
    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error processing document.")
    finally:
        # Ensure temporary file is always deleted
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@embeddings_router.delete("/reset-db", status_code=status.HTTP_200_OK)
async def reset_database(
    db: ChromaManager = Depends(get_db_client)
) -> dict[str, str]:
    """
    WARNING: Completely wipes the vector database collection.
    Exposed purely for testing and evaluation of the RAG pipeline.
    """
    try:
        db.reset_collection()
        return {
            "status": "success",
            "message": "Vector database has been completely reset."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset the database."
        )