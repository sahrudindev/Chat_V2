#!/usr/bin/env python3
"""Check ICBP and GOTO data in Qdrant."""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

def check_company(code):
    result = client.scroll(
        collection_name='financial_data',
        scroll_filter=Filter(must=[FieldCondition(key='exchange', match=MatchValue(value=code))]),
        limit=1,
        with_payload=True
    )
    
    if result[0]:
        p = result[0][0].payload
        print(f"=== {code} ===")
        print(f"name: {p.get('name')}")
        print(f"is_bank: {p.get('is_bank')}")
        print(f"financial_period: {p.get('financial_period')}")
        print(f"operating_margin: {p.get('operating_margin')}")
        print(f"cogs: {p.get('cogs')}")
        print(f"gross_margin: {p.get('gross_margin')}")
        print(f"net_margin: {p.get('net_margin')}")
        print(f"revenue: {p.get('revenue')}")
        print(f"net_profit: {p.get('net_profit')}")
    else:
        print(f"{code} not found")

check_company('ICBP')
print()
check_company('GOTO')
