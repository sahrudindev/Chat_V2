"""
Chat API endpoints.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from app.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from app.config import get_settings, Settings
from app.core.qdrant import get_qdrant_service
from app.core.ai_client import get_ai_client
from app.services.rag import get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag(settings: Settings = Depends(get_settings)):
    """Dependency to get RAG service."""
    ai_client = get_ai_client(settings.ai_service_url)
    qdrant_service = get_qdrant_service(
        settings.qdrant_host,
        settings.qdrant_port,
        settings.qdrant_collection
    )
    return get_rag_service(
        ai_client,
        qdrant_service,
        settings.rag_top_k,
        settings.rag_score_threshold
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, rag=Depends(get_rag)):
    """
    Chat endpoint - send message, get AI response.
    
    The response is generated using RAG:
    1. Query is embedded via AI service
    2. Relevant documents are retrieved from Qdrant
    3. Context + query sent to AI service for chat
    4. Response returned with source documents
    """
    try:
        history = [{"role": m.role, "content": m.content} for m in request.history]
        answer, sources = await rag.chat(request.message, history)
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-debug", response_model=SearchResponse)
async def search_debug(request: SearchRequest, rag=Depends(get_rag)):
    """Debug endpoint - search Qdrant without LLM."""
    try:
        sources, _ = await rag.retrieve(request.query)
        return SearchResponse(
            query=request.query,
            results=sources[:request.top_k]
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
