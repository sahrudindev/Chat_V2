#!/usr/bin/env python3
"""Debug script to test query parsing for ICBP and GOTO."""
import sys
sys.path.insert(0, '/home/fiqri/Desktop/IDN/AI_v2/web-chatbot')

from app.core.query_parser import query_parser

# Test queries
queries = [
    "Operating margin ICBP?",
    "COGS GOTO?",
    "Net profit BBCA berapa?",
    "ROE BBCA berapa?",
]

for q in queries:
    parsed = query_parser.parse(q)
    qdrant_filter = query_parser.filters_to_qdrant(parsed.filters)
    
    print(f"\n=== Query: {q} ===")
    print(f"  Filters: {parsed.filters}")
    print(f"  Qdrant Filter: {qdrant_filter}")
    print(f"  Sort Field: {parsed.sort_field}")
    print(f"  Sort Desc: {parsed.sort_descending}")
