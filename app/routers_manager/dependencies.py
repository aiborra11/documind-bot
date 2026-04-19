from typing import Optional
from fastapi import Request, HTTPException, status

# ----- Project Imports -----
from app.database_manager.chroma_client import ChromaManager 
from app.utils.utils import get_logger

logger = get_logger(__name__)


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