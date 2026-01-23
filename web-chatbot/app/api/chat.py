"""
Chat API endpoints.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from app.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from app.config import get_settings, Settings
from app.core.qdrant import get_qdrant_service
from app.core.ai_client import get_ai_client
from app.core.session_manager import get_session_store
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
    
    Session-based short-term memory:
    - Automatically creates session if none provided
    - Stores conversation history server-side
    - Filters out low-value messages (greetings, acknowledgements)
    - Limits context window to last 10 meaningful turns
    - Session expires after 30 minutes of inactivity
    
    The response is generated using RAG:
    1. Query is embedded via AI service
    2. Relevant documents are retrieved from Qdrant
    3. Context + query + session history sent to AI service for chat
    4. Response returned with source documents and session_id
    """
    try:
        # Get or create session
        session_store = get_session_store()
        session_id = session_store.get_or_create_session(request.session_id)
        
        logger.info(f"[Session] ID: {session_id[:8]}... | Received session_id: {request.session_id[:8] if request.session_id else 'None'}")
        
        # Add user message to session (may be filtered if low-value)
        session_store.add_message(session_id, "user", request.message)
        
        # Get filtered history from session (server-side, not from client)
        # This returns up to 10 turns of meaningful conversation
        history = session_store.get_history(session_id, max_turns=10)
        
        logger.info(f"[Session] {session_id[:8]}... | History: {len(history)} messages | Query: {request.message[:30]}...")
        
        # Generate response using RAG with session history
        answer, sources = await rag.chat(request.message, history)
        
        # Add assistant response to session
        session_store.add_message(session_id, "assistant", answer)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
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


@router.get("/session-debug/{session_id}")
async def session_debug(session_id: str):
    """Debug endpoint - check session status."""
    session_store = get_session_store()
    
    with session_store._lock:
        session = session_store._sessions.get(session_id)
        if not session:
            return {
                "found": False,
                "session_id": session_id,
                "message": "Session not found",
                "total_sessions": len(session_store._sessions)
            }
        
        return {
            "found": True,
            "session_id": session_id,
            "message_count": len(session.messages),
            "messages": [
                {"role": m.role, "content": m.content[:50] + "..." if len(m.content) > 50 else m.content}
                for m in session.messages
            ],
            "total_sessions": len(session_store._sessions)
        }


@router.get("/sessions-debug")
async def sessions_debug():
    """Debug endpoint - list all sessions."""
    session_store = get_session_store()
    
    with session_store._lock:
        sessions = []
        for sid, session in session_store._sessions.items():
            sessions.append({
                "session_id": sid[:8] + "...",
                "message_count": len(session.messages),
                "last_accessed": session.last_accessed
            })
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions
        }
