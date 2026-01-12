"""
Qdrant vector database client.
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class QdrantService:
    """Qdrant vector database service for semantic search."""
    
    def __init__(self, host: str, port: int, collection_name: str):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = None
    
    @property
    def client(self) -> QdrantClient:
        """Lazy load Qdrant client."""
        if self._client is None:
            logger.info(f"Connecting to Qdrant: {self.host}:{self.port}")
            self._client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant, collection: {self.collection_name}")
        return self._client
    
    def search(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        query_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search on Qdrant collection with optional filtering.
        
        Args:
            dense_vector: Dense embedding vector
            sparse_vector: Optional sparse vector (not used currently)
            top_k: Number of results to return
            score_threshold: Minimum score threshold
            query_filter: Optional Qdrant filter conditions
        """
        try:
            logger.info(f"Searching with vector length: {len(dense_vector)}, filter: {query_filter}")
            
            # Build Qdrant filter if provided
            qdrant_filter = None
            if query_filter and "must" in query_filter:
                filter_conditions = []
                for condition in query_filter["must"]:
                    key = condition.get("key")
                    range_cond = condition.get("range", {})
                    
                    # Build range filter
                    range_filter = models.Range(
                        lt=range_cond.get("lt"),
                        gt=range_cond.get("gt"),
                        lte=range_cond.get("lte"),
                        gte=range_cond.get("gte")
                    )
                    
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=range_filter
                        )
                    )
                
                if filter_conditions:
                    qdrant_filter = models.Filter(must=filter_conditions)
            
            # Use query_points with NamedVector
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            
            # Format results
            formatted_results = []
            for point in results.points:
                formatted_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def health_check(self) -> bool:
        """Check Qdrant connection health."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.name
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def scroll_with_filter(
        self,
        query_filter: Dict[str, Any],
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Scroll through collection with filter only (no vector search).
        
        Use for filter-based queries like "price < 1000" where we want
        ALL matching results, not semantically similar ones.
        """
        try:
            logger.info(f"Scrolling with filter: {query_filter}, limit: {limit}")
            
            # Build Qdrant filter
            filter_conditions = []
            if query_filter and "must" in query_filter:
                for condition in query_filter["must"]:
                    key = condition.get("key")
                    range_cond = condition.get("range", {})
                    
                    range_filter = models.Range(
                        lt=range_cond.get("lt"),
                        gt=range_cond.get("gt"),
                        lte=range_cond.get("lte"),
                        gte=range_cond.get("gte")
                    )
                    
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=range_filter
                        )
                    )
            
            qdrant_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            # Use scroll to get all matching results
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
            
            # Format results
            formatted_results = []
            for point in results:
                formatted_results.append({
                    "id": point.id,
                    "score": 1.0,  # No score for scroll
                    "payload": point.payload
                })
            
            logger.info(f"Scroll found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Qdrant scroll error: {e}")
            import traceback
            traceback.print_exc()
            return []


_qdrant_service = None


def get_qdrant_service(host: str, port: int, collection_name: str) -> QdrantService:
    """Get singleton Qdrant service."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService(host, port, collection_name)
    return _qdrant_service
