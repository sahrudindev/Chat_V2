#!/usr/bin/env python3
"""
Full debug script to trace the RAG flow for financial queries.
This will show exactly where the problem occurs.
"""
import sys
import os
sys.path.insert(0, '/home/fiqri/Desktop/IDN/AI_v2/web-chatbot')
os.chdir('/home/fiqri/Desktop/IDN/AI_v2/web-chatbot')

from dotenv import load_dotenv
load_dotenv()

from app.core.query_parser import query_parser
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Test query
test_queries = [
    "Operating margin ICBP?",
    "COGS GOTO?",
]

# Connect to Qdrant Cloud
client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)
    
    # Step 1: Parse query
    parsed = query_parser.parse(query)
    print(f"\n[1] QUERY PARSER:")
    print(f"    Filters: {parsed.filters}")
    print(f"    Sort Field: {parsed.sort_field}")
    print(f"    Sort Desc: {parsed.sort_descending}")
    
    # Step 2: Convert to Qdrant filter
    qdrant_filter = query_parser.filters_to_qdrant(parsed.filters)
    print(f"\n[2] QDRANT FILTER:")
    print(f"    {qdrant_filter}")
    
    # Step 3: Query Qdrant directly
    if parsed.filters:
        # Extract stock code from filter
        stock_code = None
        for f in parsed.filters:
            if f.get('key') == 'exchange':
                stock_code = f.get('match')
                break
        
        if stock_code:
            print(f"\n[3] QDRANT QUERY for {stock_code}:")
            results = client.scroll(
                collection_name='financial_data',
                scroll_filter=Filter(must=[FieldCondition(key='exchange', match=MatchValue(value=stock_code))]),
                limit=1,
                with_payload=True
            )
            
            if results[0]:
                p = results[0][0].payload
                print(f"    Found: {p.get('name')} ({p.get('exchange')})")
                print(f"    financial_period: {p.get('financial_period')}")
                print(f"    operating_margin: {p.get('operating_margin')}")
                print(f"    cogs: {p.get('cogs')}")
                print(f"    gross_margin: {p.get('gross_margin')}")
                print(f"    net_margin: {p.get('net_margin')}")
                
                # Check if sort_field value exists
                if parsed.sort_field:
                    val = p.get(parsed.sort_field)
                    print(f"\n    [SORT FIELD VALUE] {parsed.sort_field} = {val}")
                    if val is None:
                        print(f"    ⚠️  WARNING: {parsed.sort_field} is None!")
            else:
                print(f"    ❌ NOT FOUND in Qdrant!")
