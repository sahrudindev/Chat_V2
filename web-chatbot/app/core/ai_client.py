"""
Remote AI service client for embedding and chat.
Connects to local AI service running on host machine.
"""
import logging
from typing import List, Dict, Tuple
import httpx

logger = logging.getLogger(__name__)


class AIServiceClient:
    """Client for remote AI service."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        self.timeout = 120.0
    
    async def embed(self, text: str) -> Tuple[List[float], Dict[int, float]]:
        """
        Get embeddings from AI service.
        
        Args:
            text: Text to embed
            
        Returns:
            Tuple of (dense_vector, sparse_vector)
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embed",
                json={"text": text}
            )
            response.raise_for_status()
            
            data = response.json()
            dense = data["dense"]
            sparse = {int(k): float(v) for k, v in data["sparse"].items()}
            
            return dense, sparse
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Get chat response from AI service.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Generated response text
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat",
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return data["content"]
    
    async def health_check(self) -> bool:
        """Check AI service health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False


_ai_client = None


def get_ai_client(base_url: str = "http://localhost:8001") -> AIServiceClient:
    """Get singleton AI client."""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIServiceClient(base_url)
    return _ai_client
