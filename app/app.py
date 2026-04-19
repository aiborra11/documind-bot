import uvicorn
from fastapi import FastAPI, Depends
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

# -------------- Project Imports --------------
from app.config import app_settings
from app.utils.utils import get_logger
from app.database_manager.chroma_client import ChromaManager
from app.routers_manager.dependencies import get_db_client
from app.database_manager.database_config import db_config
from app.routers_manager.embeddings_routers import embeddings_router
from app.routers_manager.rag_routers import rag_router
from app.routers_manager.llm_routers import llm_router

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    """
    chroma_client = ChromaManager()

    try:
        logger.info("[app] Starting application: Initializing resources")

        # We do not add with() since we want to keep the connection open to reduce latency among requests +  prevent async issues with multiple connections. Instead, we will manually close it on shutdown.
        chroma_client.connect()
        app.state.db = chroma_client

        logger.info("Vector DB connected and registered in dependencies")
        yield 

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise e
    finally:
        chroma_client.close()
        logger.info("Application shutdown: Connection closed")


app = FastAPI(
    title=app_settings.APP_NAME,
    lifespan=lifespan
)

app.include_router(embeddings_router)
app.include_router(rag_router)
app.include_router(llm_router)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """
    Redirects the root URL to the interactive API documentation.
    """
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
async def health_check(db: ChromaManager = Depends(get_db_client)) -> dict:
    """
    Check the service and database health status.
    """
    return {
        "status": "online",
        "database": db_config.COLLECTION_NAME,
        "vector_count": db.collection.count() if db.collection else 0
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=app_settings.HOST,
        port=app_settings.PORT,
        reload=app_settings.RELOAD,
    )