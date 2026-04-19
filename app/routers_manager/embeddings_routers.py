import os
import uuid
import tempfile
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status

# --------- Project Imports ---------
from app.utils.utils import get_logger
from app.database_manager.chroma_client import ChromaManager    
from app.routers_manager.dependencies import get_db_client, get_document_processor, get_embedding_service
from app.documents_manager.document_processor import DocumentProcessor
from app.database_manager.embedding_service import EmbeddingService
from app.documents_manager.ingestion_config import ingestion_config

logger = get_logger(__name__)

embeddings_router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


TASK_STORE: Dict[str, Dict[str, Any]] = {}

# We need to use background tasks since annual reports might be pretty long what will lead to timeouts. 
def process_pdf_background(
    task_id: str,
    file_path: str, 
    filename: str, 
    processor: DocumentProcessor, 
    embedding_service: EmbeddingService
) -> None:
    """
    Background worker function that handles the heavy lifting.
    """
    logger.info(f"--- BACKGROUND TASK STARTED: Processing {filename} ---")
    TASK_STORE[task_id]["status"] = "processing"
    try:
        # 1. Extraction & Chunking (Using the permanent file_path)
        raw_documents = processor.extract_text(file_path)
        chunks = processor.chunk_documents(raw_documents, filename)
        
        # 2. Generate Embeddings & Store in Vector DB
        embedding_service.store_embeddings(chunks, filename)
        
        logger.info(f"--- BACKGROUND TASK COMPLETED: {filename} successfully ingested ---")
        TASK_STORE[task_id]["status"] = "completed"

    except Exception as e:
        logger.error(f"--- BACKGROUND TASK FAILED for {filename}: {e} ---", exc_info=True)
        TASK_STORE[task_id]["status"] = "failed"
        TASK_STORE[task_id]["error"] = str(e)

        # TODO: in a future we might consider redis to track failed ingestion and retry. We now use a global dict: TASK_STORE

@embeddings_router.post("/embed-pdf", status_code=status.HTTP_200_OK)
async def embed_pdf_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> Dict[str, Any]:
    """
    End-to-end endpoint for PDF ingestion.
    Extracts text, applies recursive chunking, generates embeddings, 
    and stores them in the vector database.
    """
    allowed_exts = tuple(ingestion_config.ALLOWED_EXTENSIONS)
    if not file.filename or not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid file format. Allowed: {allowed_exts}"
        )

    tmp_path = ""
    saved_path = ""
    task_id = str(uuid.uuid4())
    
    # Initialize task state
    TASK_STORE[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "status": "pending"
    }

    try:
        # Create a temporary file to allow PyMuPDF to read from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=ingestion_config.TEMP_FILE_SUFFIX) as tmp:
            tmp_path = tmp.name

            file_size_bytes = 0
            max_bytes = ingestion_config.MAX_FILE_SIZE_MB * 1024 * 1024
            while chunk := await file.read(ingestion_config.UPLOAD_CHUNK_SIZE_BYTES):
                file_size_bytes += len(chunk)

                if file_size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {ingestion_config.MAX_FILE_SIZE_MB} MB."
                    )
                tmp.write(chunk)

        # Save it to its permanent raw data folder
        saved_path = processor.save_document(tmp_path, file.filename)
        logger.info(f"File {file.filename} saved to persistent storage: {saved_path}")

        # Add the heavy process to BackgroundTasks so we prevent blocking the main thread and can return a quick response to the user.
        background_tasks.add_task(
            process_pdf_background,
            task_id=task_id,
            file_path=saved_path,
            filename=file.filename,
            processor=processor,
            embedding_service=embedding_service
        )

        return {
            "status": "accepted",
            "message": f"Document '{file.filename}' is processing in the background.",
            "task_id": task_id
        }

    except HTTPException:
        TASK_STORE[task_id]["status"] = "failed"
        TASK_STORE[task_id]["error"] = "File too large."
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    except Exception as e:
        logger.error(f"Unexpected error saving PDF: {e}")
        TASK_STORE[task_id]["status"] = "failed"
        TASK_STORE[task_id]["error"] = "Failed during file upload."
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal server error saving document."
        )
    finally:
        logger.info(f"Cleaning up temporary file for {file.filename}")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@embeddings_router.get("/list-all-tasks", status_code=status.HTTP_200_OK)
async def list_all_tasks(task_status: Optional[str] = None) -> Dict[str, Any]:
    """
    Lists all ingestion tasks in the system.
    Optionally filter by status using a query parameter (e.g., ?task_status=processing).
    """
    all_tasks = list(TASK_STORE.values())
    
    if task_status:
        filtered_tasks = [
            task for task in all_tasks 
            if task.get("status") == task_status.lower()
        ]
    else:
        filtered_tasks = all_tasks
        
    return {
        "status": "success",
        "total_tasks": len(filtered_tasks),
        "filter_applied": task_status or "none",
        "data": filtered_tasks
    }

@embeddings_router.get("/status/{task_id}", status_code=status.HTTP_200_OK)
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Checks the current status of an ingestion task.
    """
    task_info = TASK_STORE.get(task_id)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found."
        )
    return task_info


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