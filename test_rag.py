#!/usr/bin/env python3
"""
Test script untuk debug RAG pipeline.
"""
import requests
import json

AI_SERVICE = "http://localhost:8001"
WEB_CHATBOT = "http://localhost:8080"
QDRANT = "http://localhost:6333"

def test_services():
    print("=" * 50)
    print("Testing RAG Services")
    print("=" * 50)
    
    # 1. Test Qdrant
    print("\n[1] Testing Qdrant...")
    try:
        r = requests.get(f"{QDRANT}/collections/financial_data", timeout=5)
        data = r.json()
        points = data.get("result", {}).get("points_count", 0)
        print(f"    ✓ Qdrant OK - {points} points in collection")
    except Exception as e:
        print(f"    ✗ Qdrant Error: {e}")
        return
    
    # 2. Test AI Service Health
    print("\n[2] Testing AI Service...")
    try:
        r = requests.get(f"{AI_SERVICE}/health", timeout=5)
        data = r.json()
        print(f"    ✓ AI Service OK - GPU: {data.get('gpu')}, LLM: {data.get('llm_model')}")
    except Exception as e:
        print(f"    ✗ AI Service Error: {e}")
        return
    
    # 3. Test Embedding
    print("\n[3] Testing Embedding...")
    try:
        r = requests.post(
            f"{AI_SERVICE}/embed",
            json={"text": "kelapa sawit"},
            timeout=30
        )
        data = r.json()
        dense_len = len(data.get("dense", []))
        sparse_len = len(data.get("sparse", {}))
        print(f"    ✓ Embedding OK - Dense: {dense_len}, Sparse: {sparse_len}")
        
        # Save for next test
        dense_vector = data["dense"]
    except Exception as e:
        print(f"    ✗ Embedding Error: {e}")
        return
    
    # 4. Test Qdrant Search Directly
    print("\n[4] Testing Qdrant Search Directly...")
    try:
        search_payload = {
            "vector": {
                "name": "dense",
                "vector": dense_vector
            },
            "limit": 5,
            "with_payload": True
        }
        r = requests.post(
            f"{QDRANT}/collections/financial_data/points/search",
            json=search_payload,
            timeout=10
        )
        data = r.json()
        results = data.get("result", [])
        print(f"    ✓ Qdrant Search OK - Found {len(results)} results")
        
        if results:
            for i, hit in enumerate(results[:3]):
                payload = hit.get("payload", {})
                name = payload.get("name", "Unknown")
                score = hit.get("score", 0)
                print(f"       [{i+1}] {name} (score: {score:.4f})")
        else:
            print("    ⚠ No results found - checking vector format...")
    except Exception as e:
        print(f"    ✗ Qdrant Search Error: {e}")
    
    # 5. Test Web Chatbot
    print("\n[5] Testing Web Chatbot...")
    try:
        r = requests.get(f"{WEB_CHATBOT}/health", timeout=5)
        data = r.json()
        print(f"    ✓ Web Chatbot OK - Status: {data.get('status')}")
    except Exception as e:
        print(f"    ✗ Web Chatbot Error: {e}")
    
    # 6. Test RAG Chat
    print("\n[6] Testing RAG Chat...")
    try:
        r = requests.post(
            f"{WEB_CHATBOT}/chat",
            json={"message": "kelapa sawit", "history": []},
            timeout=60
        )
        data = r.json()
        sources = data.get("sources", [])
        answer = data.get("answer", "")[:100]
        print(f"    ✓ RAG Chat OK - Sources: {len(sources)}, Answer: {answer}...")
    except Exception as e:
        print(f"    ✗ RAG Chat Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test Complete")
    print("=" * 50)

if __name__ == "__main__":
    test_services()
