"""
AI Service - Local embedding with ROCm GPU + Gemini for chat.

Endpoints:
- POST /embed - Generate embeddings using BGE-M3 (Local GPU)
- POST /chat  - Generate chat response using Gemini (Google API)
- GET /health - Health check
"""
import os
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# SCHEMAS
# =============================================================================

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    dense: List[float]
    sparse: Dict[int, float]

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 1024

class ChatResponse(BaseModel):
    content: str

class HealthResponse(BaseModel):
    status: str
    gpu: Optional[str] = None
    embedding_model: str
    llm_model: str

# =============================================================================
# EMBEDDING SERVICE (Local GPU)
# =============================================================================

class EmbeddingService:
    """BGE-M3 embedding service with GPU support."""
    
    def __init__(self):
        self.model = None
        self.device = None
    
    def load(self):
        """Load embedding model."""
        import torch
        from FlagEmbedding import BGEM3FlagModel
        
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
        else:
            self.device = "cpu"
            logger.warning("No GPU detected, using CPU")
        
        logger.info("Loading BGE-M3 model...")
        self.model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=(self.device == "cuda"),
            device=self.device
        )
        logger.info("BGE-M3 model loaded!")
    
    def embed(self, text: str) -> tuple:
        """Generate embeddings."""
        output = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        dense = output['dense_vecs'][0].tolist()
        sparse = {int(k): float(v) for k, v in output['lexical_weights'][0].items()}
        
        return dense, sparse

# =============================================================================
# GEMINI SERVICE (New SDK: google-genai)
# =============================================================================

class GeminiService:
    """Gemini service using new google-genai SDK."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model
        self._client = None
    
    def initialize(self):
        """Initialize Gemini client."""
        from google import genai
        
        self._client = genai.Client(api_key=self.api_key)
        logger.info(f"Gemini initialized with model: {self.model_name}")
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate chat response using Gemini with retry logic."""
        import time
        
        # Build the full prompt from messages
        system_prompt = ""
        conversation_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                conversation_parts.append(f"User: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
        
        # Combine into single prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += "\n".join(conversation_parts)
        
        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                # Check if it's a 503 overloaded error
                if "503" in error_str or "overloaded" in error_str.lower() or "UNAVAILABLE" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                        logger.warning(f"Gemini overloaded, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"Gemini still overloaded after {max_retries} retries")
                        raise
                else:
                    # For other errors, don't retry
                    raise
    
    def health_check(self) -> bool:
        """Check Gemini connection."""
        return self._client is not None

# =============================================================================
# APP
# =============================================================================

embedding_service = EmbeddingService()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

gemini_service = GeminiService(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("=" * 50)
    logger.info("Starting AI Service")
    logger.info(f"LLM: {GEMINI_MODEL}")
    logger.info("=" * 50)
    
    embedding_service.load()
    
    if GEMINI_API_KEY:
        gemini_service.initialize()
    else:
        logger.warning("GEMINI_API_KEY not set!")
    
    yield
    logger.info("Shutting down AI Service")

app = FastAPI(
    title="AI Service",
    description="Local embedding (BGE-M3 GPU) + Gemini chat",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings using BGE-M3."""
    try:
        dense, sparse = embedding_service.embed(request.text)
        return EmbedResponse(dense=dense, sparse=sparse)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate chat response using Gemini."""
    try:
        content = gemini_service.chat(
            request.messages,
            request.temperature,
            request.max_tokens
        )
        return ChatResponse(content=content)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    import torch
    
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    
    gemini_ok = gemini_service.health_check()
    
    return HealthResponse(
        status="healthy" if gemini_ok else "degraded",
        gpu=gpu,
        embedding_model="BAAI/bge-m3",
        llm_model=GEMINI_MODEL
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
