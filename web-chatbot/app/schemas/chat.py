"""
Pydantic schemas for chat API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., description="User message", min_length=1)
    history: List[ChatMessage] = Field(default=[], description="Chat history")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Apa itu perusahaan kelapa sawit terbesar?",
                "history": []
            }
        }


class SourceDocument(BaseModel):
    """Retrieved source document."""
    exchange: Optional[str] = None
    name: Optional[str] = None
    text: str
    score: float


class ChatResponse(BaseModel):
    """Chat response payload."""
    answer: str = Field(..., description="Assistant response")
    sources: List[SourceDocument] = Field(default=[], description="Source documents")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Berdasarkan data yang tersedia, PT Lonsum adalah salah satu perusahaan kelapa sawit terbesar...",
                "sources": [
                    {"exchange": "LSIP", "name": "PT Lonsum", "text": "...", "score": 0.85}
                ],
                "timestamp": "2026-01-09T15:00:00"
            }
        }


class SearchRequest(BaseModel):
    """Debug search request."""
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Debug search response."""
    query: str
    results: List[SourceDocument]
