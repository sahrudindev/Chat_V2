"""
Query Parser for Hybrid Search.

Detects numerical/date filter conditions from natural language queries.
Extracts filters for Qdrant while keeping semantic search text.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ParsedQuery:
    """Result of parsing a user query."""
    search_text: str  # Text for semantic search (cleaned)
    filters: List[Dict[str, Any]]  # Qdrant filter conditions
    original_query: str  # Original user query
    requested_count: int = 10  # How many results user requested (default 10)
    sort_field: str = None  # Field to sort by (e.g., 'capitalization', 'listing_year')
    sort_descending: bool = True  # Sort direction


class QueryParser:
    """
    Parse user queries to extract filter conditions.
    
    Supports Indonesian and English patterns for:
    - Price filters (harga dibawah/diatas, price below/above)
    - Volume filters (volume diatas)
    - Date filters (listing sebelum/setelah)
    - Market cap filters (kapitalisasi diatas)
    """
    
    # Number patterns with Indonesian multipliers
    NUMBER_PATTERN = r'([\d.,]+)\s*(juta|miliar|triliun|ribu|rb|jt|m|b|t)?'
    
    # Filter patterns: (regex, field_name, operator, multiplier_group)
    FILTER_PATTERNS = [
        # Price filters
        (r'harga\s+(?:dibawah|kurang\s+dari|<)\s+' + NUMBER_PATTERN, 'close_price', 'lt'),
        (r'harga\s+(?:diatas|lebih\s+dari|>)\s+' + NUMBER_PATTERN, 'close_price', 'gt'),
        (r'price\s+(?:below|under|<)\s+' + NUMBER_PATTERN, 'close_price', 'lt'),
        (r'price\s+(?:above|over|>)\s+' + NUMBER_PATTERN, 'close_price', 'gt'),
        
        # Volume filters
        (r'volume\s+(?:diatas|lebih\s+dari|>)\s+' + NUMBER_PATTERN, 'tradable_volume', 'gt'),
        (r'volume\s+(?:dibawah|kurang\s+dari|<)\s+' + NUMBER_PATTERN, 'tradable_volume', 'lt'),
        
        # Market cap filters
        (r'(?:market\s*cap|kapitalisasi)\s+(?:diatas|lebih\s+dari|>)\s+' + NUMBER_PATTERN, 'capitalization', 'gt'),
        (r'(?:market\s*cap|kapitalisasi)\s+(?:dibawah|kurang\s+dari|<)\s+' + NUMBER_PATTERN, 'capitalization', 'lt'),
        
        # Listing date filters (year only for simplicity)
        (r'listing\s+(?:sebelum|sebelum\s+tahun|<)\s+(\d{4})', 'listing', 'lt_year'),
        (r'listing\s+(?:sesudah\s+tahun|sesudah|setelah\s+tahun|setelah|>)\s+(\d{4})', 'listing', 'gt_year'),
        (r'listed\s+(?:before|<)\s+(\d{4})', 'listing', 'lt_year'),
        (r'listed\s+(?:after|>)\s+(\d{4})', 'listing', 'gt_year'),
    ]
    
    # Multipliers for Indonesian number words
    MULTIPLIERS = {
        'ribu': 1_000,
        'rb': 1_000,
        'juta': 1_000_000,
        'jt': 1_000_000,
        'm': 1_000_000,
        'miliar': 1_000_000_000,
        'b': 1_000_000_000,
        'triliun': 1_000_000_000_000,
        't': 1_000_000_000_000,
    }
    
    # Sector keywords mapping to si_code (IDX-IC codes)
    SECTOR_KEYWORDS = {
        'bank': 'G111',
        'perbankan': 'G111',
        'banking': 'G111',
        'asuransi': 'G121',
        'insurance': 'G121',
        'properti': 'H111',
        'property': 'H111',
        'real estat': 'H111',
        'teknologi': 'I',
        'technology': 'I',
        'telekomunikasi': 'J121',
        'telco': 'J121',
        'telecom': 'J121',
        'farmasi': 'F121',
        'pharma': 'F121',
        'tambang': 'B111',
        'mining': 'B111',
        'batubara': 'A112',
        'coal': 'A112',
        'minyak': 'A111',
        'oil': 'A111',
        'gas': 'A111',
        'konstruksi': 'C121',
        'construction': 'C121',
        'consumer': 'D',
        'konsumen': 'D',
    }
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse query to extract filters and search text.
        
        Args:
            query: User's natural language query
            
        Returns:
            ParsedQuery with filters and cleaned search text
        """
        filters = []
        search_text = query
        
        for pattern, field, operator in self.FILTER_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    filter_condition = self._build_filter(match, field, operator)
                    if filter_condition:
                        filters.append(filter_condition)
                        # Remove matched pattern from search text
                        search_text = re.sub(pattern, '', search_text, flags=re.IGNORECASE)
                except Exception:
                    # Skip invalid matches
                    pass
        
        query_lower = query.lower()
        
        # Check if this is a descriptive query (should NOT apply sector filter)
        descriptive_keywords = [
            'ceritakan', 'jelaskan', 'apa itu', 'tentang', 'siapa', 'bagaimana', 
            'describe', 'about', 'what is',
            # Shareholder-related keywords
            'pemegang saham', 'shareholder', 'kepemilikan', 'ownership', 'dimiliki',
            'pemilik', 'investor', 'stakeholder', 'saham terbesar', 'mayoritas'
        ]
        is_descriptive = any(kw in query_lower for kw in descriptive_keywords)
        
        # Check if this is a ranking/list query (SHOULD apply sector filter)
        ranking_keywords = ['terbesar', 'terkecil', 'tertinggi', 'terendah', 'list', 'daftar', 'tertua', 'terbaru']
        is_ranking = any(kw in query_lower for kw in ranking_keywords)
        
        # Detect sector keywords and add sector filter ONLY for ranking queries
        detected_sector = None
        if is_ranking and not is_descriptive:
            for keyword, sector_code in self.SECTOR_KEYWORDS.items():
                if keyword in query_lower:
                    detected_sector = sector_code
                    break
            
            if detected_sector:
                # Add sector filter using si_code field
                filters.append({
                    "key": "si_code",
                    "match": detected_sector
                })
        
        # Extract requested count (e.g., "5 saham", "10 bank", "3 perusahaan")
        requested_count = 10  # default
        count_match = re.search(r'\b(\d+)\s*(?:saham|bank|perusahaan|emiten|stock)', query, re.IGNORECASE)
        if count_match:
            requested_count = int(count_match.group(1))
        
        # Determine sort field based on query context
        sort_field = None
        sort_descending = True
        
        if any(kw in query_lower for kw in ['earning', 'eps', 'laba', 'laba bersih']):
            # EPS fundamental - highest earnings per share
            sort_field = 'earning_per_share'
            sort_descending = True
        elif any(kw in query_lower for kw in ['dividend yield', 'yield dividen', 'yield tertinggi']):
            # Dividend yield - highest yield first
            sort_field = 'latest_dividend_yield'
            sort_descending = True
        elif any(kw in query_lower for kw in ['dividen terbesar', 'dividen tertinggi', 'dps terbesar', 'dividend per share terbesar']):
            # Highest dividend per share
            sort_field = 'latest_dividend_per_share'
            sort_descending = True
        elif any(kw in query_lower for kw in ['gainer', 'gainers', 'naik', 'untung', 'keuntungan', 'profit', 'cuan']):
            # Market gainers - highest % price increase
            sort_field = 'percentage_price'
            sort_descending = True
        elif any(kw in query_lower for kw in ['loser', 'losers', 'turun', 'rugi', 'loss']):
            # Market losers - lowest % price (most negative first)
            sort_field = 'percentage_price'
            sort_descending = False
        elif 'terbesar' in query_lower or 'tertinggi' in query_lower:
            sort_field = 'capitalization'
            sort_descending = True
        elif 'terkecil' in query_lower or 'terendah' in query_lower:
            sort_field = 'capitalization'
            sort_descending = False
        elif 'tertua' in query_lower or 'oldest' in query_lower:
            sort_field = 'listing_year'
            sort_descending = False
        elif 'terbaru' in query_lower or 'newest' in query_lower:
            sort_field = 'listing_year'
            sort_descending = True
        elif filters:
            # Use filter field as sort field with appropriate direction
            for f in filters:
                if 'range' in f:
                    filter_key = f.get('key')
                    range_cond = f.get('range', {})
                    
                    # For listing_year filter, use 'listing' (full date) for better sorting
                    if filter_key == 'listing_year':
                        sort_field = 'listing'  # Use full date string for proper chronological order
                    else:
                        sort_field = filter_key
                    
                    # For "sebelum/dibawah" (lt): sort descending (closest to boundary first)
                    # For "sesudah/diatas" (gt): sort ascending (closest to boundary first)
                    if 'lt' in range_cond:
                        sort_descending = True  # 2018-12-31, 2018-12-30, ...
                    elif 'gt' in range_cond:
                        sort_descending = False  # 2011-01-01, 2011-01-02, ...
                    break
        
        # Clean up search text
        search_text = re.sub(r'\s+', ' ', search_text).strip()
        
        return ParsedQuery(
            search_text=search_text if search_text else query,
            filters=filters,
            original_query=query,
            requested_count=requested_count,
            sort_field=sort_field,
            sort_descending=sort_descending
        )
    
    def _build_filter(self, match: re.Match, field: str, operator: str) -> Optional[Dict[str, Any]]:
        """Build Qdrant filter condition from regex match."""
        
        if operator in ('lt_year', 'gt_year'):
            # Date filter - use listing_year integer field
            year = int(match.group(1))
            
            if operator == 'lt_year':
                return {
                    "key": "listing_year",  # Use integer field
                    "range": {"lt": year}
                }
            else:
                return {
                    "key": "listing_year",
                    "range": {"gt": year}
                }
        else:
            # Numeric filter
            number_str = match.group(1).replace(',', '').replace('.', '')
            value = float(number_str)
            
            # Apply multiplier if present
            multiplier_str = match.group(2)
            if multiplier_str:
                multiplier = self.MULTIPLIERS.get(multiplier_str.lower(), 1)
                value *= multiplier
            
            if operator == 'lt':
                return {
                    "key": field,
                    "range": {"lt": value}
                }
            elif operator == 'gt':
                return {
                    "key": field,
                    "range": {"gt": value}
                }
        
        return None
    
    def filters_to_qdrant(self, filters: List[Dict[str, Any]]) -> Optional[Dict]:
        """
        Convert parsed filters to Qdrant filter format.
        
        Args:
            filters: List of filter conditions from parse()
            
        Returns:
            Qdrant-compatible filter dict, or None if no filters
        """
        if not filters:
            return None
        
        conditions = []
        for f in filters:
            key = f["key"]
            
            # Handle range filter
            if "range" in f:
                conditions.append({
                    "key": key,
                    "range": f["range"]
                })
            # Handle match filter (for sector)
            elif "match" in f:
                conditions.append({
                    "key": key,
                    "match": f["match"]
                })
        
        if not conditions:
            return None
        
        return {"must": conditions}


# Singleton instance
query_parser = QueryParser()
