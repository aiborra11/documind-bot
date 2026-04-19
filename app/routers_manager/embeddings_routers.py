import os
import tempfile
from typing import Dict, Any

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.utils.utils import get_logger
from app.routers_manager.dependencies import get_document_processor
from app.ingestion_manager.document_processor import DocumentProcessor

logger = get_logger(__name__)

processpdf_router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

@processpdf_router.post("/process-pdf", status_code=status.HTTP_200_OK)
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_document_processor)
) -> Dict[str, Any]:
    """
    Endpoint to test PDF extraction and text chunking.
    Uploads a file, processes it, and returns the chunk metadata.
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

        # Execute processing pipeline
        raw_documents = processor.extract_text(tmp_path)
        chunks = processor.chunk_documents(raw_documents, file.filename)
        saved_path = processor.save_document(tmp_path, file.filename)

        return {
            "status": "success",
            "filename": file.filename,
            "saved_path": saved_path,
            "total_pages": len(raw_documents),
            "total_chunks": len(chunks),
            "sample_chunk": chunks[0] if chunks else None
        }

    except ValueError as ve:
        logger.warning(f"Validation error during processing: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error processing document.")
    finally:
        # Ensure temporary file is always deleted
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)