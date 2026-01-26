#!/usr/bin/env python3
"""
Full end-to-end RAG debug - shows exactly what context is sent to LLM.
"""
import sys
import os
import asyncio
sys.path.insert(0, '/home/fiqri/Desktop/IDN/AI_v2/web-chatbot')
os.chdir('/home/fiqri/Desktop/IDN/AI_v2/web-chatbot')

from dotenv import load_dotenv
load_dotenv()

from app.core.qdrant import get_qdrant_service
from app.core.ai_client import AIServiceClient
from app.services.rag import RAGService

async def debug_rag():
    # Initialize components
    qdrant = get_qdrant_service(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333)),
        collection_name=os.getenv('QDRANT_COLLECTION', 'financial_data')
    )
    
    ai_client = AIServiceClient(
        base_url=os.getenv('AI_SERVICE_URL', 'http://localhost:8001')
    )
    
    rag = RAGService(
        ai_client=ai_client,
        qdrant_service=qdrant,
    )
    
    query = "Operating margin ICBP?"
    print(f"Query: {query}")
    print("="*60)
    
    # Call retrieve
    sources, context = await rag.retrieve(query)
    
    print(f"\n[SOURCES] Count: {len(sources)}")
    for i, s in enumerate(sources[:3]):
        # Use correct attribute names
        name = getattr(s, 'name', getattr(s, 'title', 'Unknown'))
        score = getattr(s, 'score', 0)
        print(f"  {i+1}. {name} (score: {score:.2f})")
    
    print(f"\n[CONTEXT LENGTH]: {len(context)} chars")
    print(f"\n[CONTEXT SENT TO LLM]:")
    print("-"*60)
    print(context[:4000])  # First 4000 chars
    print("-"*60)
    
    if "operating margin" in context.lower() or "operating_margin" in context.lower():
        print("\n✓ 'operating margin' FOUND in context")
    else:
        print("\n✗ 'operating margin' NOT FOUND in context!")
        
    if "21.5" in context or "21,5" in context:
        print("✓ Value '21.5' FOUND in context")
    else:
        print("✗ Value '21.5' NOT FOUND in context!")

asyncio.run(debug_rag())
