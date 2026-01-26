#!/usr/bin/env python3
"""
Create payload indices on existing Qdrant collection.
This fixes the "Index required" error for filtering.
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# Connect to Qdrant Cloud
client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

collection_name = os.getenv('QDRANT_COLLECTION', 'financial_data')

print(f"Creating indices on collection: {collection_name}")

# Fields that need indexing for filtering
fields = [
    ("exchange", "keyword"),
    ("si_code", "keyword"),
    ("is_bank", "bool"),
]

for field_name, field_type in fields:
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type
        )
        print(f"✓ Created index: {field_name} ({field_type})")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"• Index already exists: {field_name}")
        else:
            print(f"✗ Error creating index for {field_name}: {e}")

print("\nDone! Now try the queries again.")
