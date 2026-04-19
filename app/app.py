import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from app.config import app_settings
from app.utils.utils import get_logger


logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    """
    try:
        logger.info("Application started!")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
    yield


app = FastAPI(
    title=app_settings.APP_NAME,
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
async def root_redirect():
    """
    Redirects the root URL to the interactive API documentation.
    """
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run("main:app", host=app_settings.HOST, port=app_settings.PORT, reload=app_settings.RELOAD)