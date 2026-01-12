"""
RAG (Retrieval-Augmented Generation) Service.
Uses remote AI service for embedding and chat.
"""
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from app.core.qdrant import QdrantService
from app.core.ai_client import AIServiceClient
from app.core.query_parser import query_parser
from app.core.sector_mapping import get_sector_name
from app.schemas import SourceDocument

logger = logging.getLogger(__name__)

# Load system prompt
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "system.txt").read_text()


class RAGService:
    """RAG service using remote AI service for embedding and chat."""
    
    def __init__(
        self,
        ai_client: AIServiceClient,
        qdrant_service: QdrantService,
        top_k: int = 10,  # Increased for better coverage on broad queries
        score_threshold: float = 0.3
    ):
        self.ai_client = ai_client
        self.qdrant_service = qdrant_service
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    async def retrieve(self, query: str) -> Tuple[List[SourceDocument], str]:
        """
        Retrieve relevant documents for a query with hybrid search.
        
        Uses semantic search (embeddings) combined with Qdrant filtering
        for numerical/date queries.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (source_documents, formatted_context)
        """
        # Parse query for filters
        parsed = query_parser.parse(query)
        search_text = parsed.search_text
        qdrant_filter = query_parser.filters_to_qdrant(parsed.filters)
        
        # Use parsed values for sorting and count
        sort_field = parsed.sort_field
        sort_descending = parsed.sort_descending
        requested_count = parsed.requested_count
        
        # Fallback sort field from filter if not determined
        if not sort_field and parsed.filters:
            for f in parsed.filters:
                sort_field = f.get("key")
                break
        
        logger.info(f"Query: count={requested_count}, sort={sort_field}, desc={sort_descending}, filters={parsed.filters}")
        
        if parsed.filters:
            # Use PURE FILTER search (scroll) for filter queries
            results = self.qdrant_service.scroll_with_filter(
                query_filter=qdrant_filter,
                limit=50  # Always get up to 50 for filter queries
            )
            
            # Sort results by sort field
            if sort_field and results:
                results = sorted(
                    results,
                    key=lambda x: x.get("payload", {}).get(sort_field) or 0,
                    reverse=sort_descending
                )
        else:
            # Use semantic vector search for non-filter queries (including sort-only)
            dense_vector, sparse_vector = await self.ai_client.embed(search_text)
            results = self.qdrant_service.search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=50,  # Get more results for sorting
                score_threshold=self.score_threshold,
                query_filter=None
            )
            
            # Sort results by sort field if specified
            if sort_field and results:
                results = sorted(
                    results,
                    key=lambda x: x.get("payload", {}).get(sort_field) or 0,
                    reverse=sort_descending
                )
        
        # Format results - 5 detailed + rest as codes only
        sources = []
        detailed_parts = []
        additional_codes = []
        
        # Safe number formatting
        def fmt_num(val, prefix="", suffix=""):
            if val is None:
                return "N/A"
            try:
                return f"{prefix}{val:,.0f}{suffix}"
            except:
                return str(val)
        
        def fmt_pct(val):
            if val is None:
                return "N/A"
            try:
                return f"{val:.2f}%"
            except:
                return str(val)
        
        for i, result in enumerate(results):
            payload = result.get("payload", {})
            exchange = payload.get("exchange") or "N/A"
            name = payload.get("name") or "Unknown"
            
            source = SourceDocument(
                exchange=exchange,
                name=name,
                text=payload.get("text", "")[:500],
                score=result.get("score", 0.0)
            )
            sources.append(source)
            
            if i < 5:
                # First 5: Full detailed context
                sector_code = payload.get('si_code') or ''
                sector_display = get_sector_name(sector_code) if sector_code else 'N/A'
                
                detailed_parts.append(f"""
[Perusahaan {i+1}]
Nama: {name} ({exchange})
Sektor: {sector_display}
Established: {payload.get('established') or 'N/A'}
Listing Date: {payload.get('listing') or 'N/A'}
Harga Penutupan: {fmt_num(payload.get('close_price'), "Rp ")}
Harga Pembukaan: {fmt_num(payload.get('open_price'), "Rp ")}
Harga Tertinggi: {fmt_num(payload.get('day_high'), "Rp ")}
Harga Terendah: {fmt_num(payload.get('day_low'), "Rp ")}
Perubahan Harga: {fmt_num(payload.get('price_change'))} ({fmt_pct(payload.get('percentage_price'))})
Volume: {fmt_num(payload.get('tradable_volume'))}
Kapitalisasi Pasar: {fmt_num(payload.get('capitalization'), "Rp ")}
P/E Ratio: {payload.get('price_earning_ratio') or 'N/A'}
EPS: {payload.get('earning_per_share') or 'N/A'}
""")
            else:
                # Rest: Just code
                additional_codes.append(exchange)
        
        # Build context
        context = "\n".join(detailed_parts)
        
        # Add additional codes if any
        if additional_codes:
            codes_str = ", ".join(additional_codes[:10])  # Max 10 codes
            remaining = len(additional_codes) - 10 if len(additional_codes) > 10 else 0
            if remaining > 0:
                context += f"\n\n[Data Tambahan]\nSelain 5 perusahaan di atas, ada juga: {codes_str}, dan {remaining} perusahaan lainnya."
            else:
                context += f"\n\n[Data Tambahan]\nSelain 5 perusahaan di atas, ada juga: {codes_str}."
            context += "\nSarankan user untuk cek idnfinancials.com untuk informasi lengkap."
        
        return sources, context
    
    async def generate_response(
        self,
        query: str,
        context: str,
        history: Optional[List[dict]] = None
    ) -> str:
        """
        Generate response using AI service chat.
        
        Args:
            query: User query
            context: Retrieved context
            history: Chat history (optional)
            
        Returns:
            Generated response text
        """
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add history if provided
        if history:
            for msg in history[-4:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add context and query
        if context:
            user_message = f"""<context>
{context}
</context>

User Question: {query}"""
        else:
            user_message = f"""User Question: {query}

Note: No relevant context was found in the knowledge base."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate via AI service
        response = await self.ai_client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return response
    
    async def chat(
        self,
        query: str,
        history: Optional[List[dict]] = None
    ) -> Tuple[str, List[SourceDocument]]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            query: User query
            history: Chat history
            
        Returns:
            Tuple of (response, source_documents)
        """
        logger.info(f"Processing query: {query[:50]}...")
        
        # Retrieve
        sources, context = await self.retrieve(query)
        logger.info(f"Retrieved {len(sources)} documents")
        
        # Generate
        response = await self.generate_response(query, context, history)
        logger.info("Generated response")
        
        return response, sources


_rag_service = None


def get_rag_service(
    ai_client: AIServiceClient,
    qdrant_service: QdrantService,
    top_k: int = 5,
    score_threshold: float = 0.3
) -> RAGService:
    """Get singleton RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(
            ai_client,
            qdrant_service,
            top_k,
            score_threshold
        )
    return _rag_service
