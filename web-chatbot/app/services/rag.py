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
SYSTEM_PROMPT = (PROMPTS_DIR / "system2.txt").read_text()


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
        
        if parsed.filters or sort_field:
            # Use PURE FILTER/SCROLL for filter queries OR ranking queries (with sort_field)
            # This ensures ranking queries get ALL data for accurate sorting
            logger.info(f"Using scroll_with_filter: filter={qdrant_filter}")
            results = self.qdrant_service.scroll_with_filter(
                query_filter=qdrant_filter,  # Will be None for pure ranking queries
                limit=1000  # Get ALL data for accurate ranking (BEI has ~800 stocks)
            )
            logger.info(f"Scroll returned {len(results)} results")
            
            # Sort results by sort field
            if sort_field and results:
                results = sorted(
                    results,
                    key=lambda x: x.get("payload", {}).get(sort_field) or 0,
                    reverse=sort_descending
                )
                logger.info(f"Sorted by {sort_field}, first 3: {[r.get('payload',{}).get('exchange') for r in results[:3]]}")
        else:
            # Use semantic vector search ONLY for descriptive queries (no filter, no sort)
            dense_vector, sparse_vector = await self.ai_client.embed(search_text)
            results = self.qdrant_service.search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=1000,  # Full dataset coverage (~800 stocks)
                score_threshold=self.score_threshold,
                query_filter=None
            )
        
        # SHAREHOLDER CROSS-REFERENCE: Detect if query is asking about a person at a specific company
        # Pattern: "Apakah [NAME] pemegang saham [COMPANY]?"
        query_lower = query.lower()
        shareholder_keywords = ['pemegang saham', 'shareholder', 'kepemilikan', 'dimiliki']
        is_shareholder_query = any(kw in query_lower for kw in shareholder_keywords)
        
        if is_shareholder_query and results:
            # Check if the top result actually contains info about the queried entity
            # If not, search semantically for the person name
            top_result = results[0] if results else {}
            top_text = top_result.get("payload", {}).get("text", "").lower()
            
            # Try to extract person/entity names from query (words before shareholder keywords)
            import re
            # Common patterns: "Apakah X pemegang saham Y", "X adalah pemegang saham Y"
            name_match = re.search(r'(?:apakah|adalah)\s+([^?]+?)\s+(?:pemegang|shareholder)', query_lower)
            if name_match:
                entity_name = name_match.group(1).strip()
                # Check if entity exists in top results
                entity_found = any(entity_name in r.get("payload", {}).get("text", "").lower() for r in results[:5])
                
                if not entity_found and len(entity_name) > 2:
                    # Do secondary search for the person/entity name
                    logger.info(f"Shareholder cross-ref: searching for entity '{entity_name}'")
                    entity_dense, entity_sparse = await self.ai_client.embed(entity_name)
                    entity_results = self.qdrant_service.search(
                        dense_vector=entity_dense,
                        sparse_vector=entity_sparse,
                        top_k=10,
                        score_threshold=self.score_threshold,
                        query_filter=None
                    )
                    # Merge entity results with original (avoiding duplicates)
                    seen_ids = {r.get("id") for r in results[:10]}
                    for er in entity_results:
                        if er.get("id") not in seen_ids:
                            results.insert(5, er)  # Insert after top 5
                            seen_ids.add(er.get("id"))
                    logger.info(f"Added {len(entity_results)} cross-reference results")
        
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
                
                # Format shareholders for display
                shareholders_list = payload.get('shareholders', [])
                shareholders_text = ""
                if shareholders_list:
                    shareholders_text = "\nPemegang Saham:\n"
                    for j, sh in enumerate(shareholders_list[:5]):  # Top 5 shareholders
                        sh_name = sh.get('name', 'Unknown')
                        sh_pct = sh.get('percentage', 0)
                        sh_shares = sh.get('shares', 0)
                        shareholders_text += f"  {j+1}. {sh_name}: {sh_pct:.2f}% ({sh_shares:,} saham)\n"
                    if len(shareholders_list) > 5:
                        shareholders_text += f"  ... dan {len(shareholders_list) - 5} pemegang saham lainnya\n"
                
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
EPS: {payload.get('earning_per_share') or 'N/A'}{shareholders_text}""")
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
