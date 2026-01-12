"""
FastAPI Main Application - Web Chatbot.
Connects to remote AI service for embedding and chat.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.api import chat_router
from app.core import get_qdrant_service, get_ai_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    settings = get_settings()
    logger.info("=" * 50)
    logger.info("Starting RAG Chatbot (Web)")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"AI Service: {settings.ai_service_url}")
    logger.info("=" * 50)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG Chatbot - Indonesian Stock Market",
    description="Chatbot using RAG with Qdrant and remote AI service",
    version="1.0.0",
    lifespan=lifespan
)

# Static files and templates
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(STATIC_DIR))

# Include routers
app.include_router(chat_router, tags=["Chat"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    
    # Check Qdrant
    qdrant = get_qdrant_service(
        settings.qdrant_host,
        settings.qdrant_port,
        settings.qdrant_collection
    )
    qdrant_ok = qdrant.health_check()
    
    # Check AI Service
    ai_client = get_ai_client(settings.ai_service_url)
    ai_ok = await ai_client.health_check()
    
    return {
        "status": "healthy" if (qdrant_ok and ai_ok) else "degraded",
        "qdrant": {"connected": qdrant_ok},
        "ai_service": {"connected": ai_ok, "url": settings.ai_service_url}
    }
