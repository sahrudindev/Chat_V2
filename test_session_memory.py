#!/usr/bin/env python3
"""
Test script untuk debug session memory.
Jalankan: python test_session_memory.py
"""
import requests
import json

WEB_CHATBOT = "http://localhost:8080"

def test_session_memory():
    print("=" * 60)
    print("Testing Session Memory")
    print("=" * 60)
    
    # 1. First request (no session_id)
    print("\n[1] First request: 'Apa itu TLKM?'")
    print("-" * 40)
    
    response1 = requests.post(
        f"{WEB_CHATBOT}/chat",
        json={"message": "Apa itu TLKM?", "session_id": None},
        timeout=60
    )
    data1 = response1.json()
    
    session_id = data1.get("session_id")
    answer1 = data1.get("answer", "")[:100]
    
    print(f"    Session ID: {session_id}")
    print(f"    Answer: {answer1}...")
    
    if not session_id:
        print("\n❌ ERROR: No session_id in response!")
        return
    
    print(f"\n    ✓ Got session_id: {session_id[:8]}...")
    
    # 2. Second request (with session_id)
    print("\n[2] Second request: 'Berapa harganya?' (with session_id)")
    print("-" * 40)
    
    response2 = requests.post(
        f"{WEB_CHATBOT}/chat",
        json={"message": "Berapa harganya?", "session_id": session_id},
        timeout=60
    )
    data2 = response2.json()
    
    session_id2 = data2.get("session_id")
    answer2 = data2.get("answer", "")
    
    print(f"    Session ID: {session_id2}")
    print(f"    Answer: {answer2[:200]}...")
    
    # Check if session_id is the same
    if session_id == session_id2:
        print(f"\n    ✓ Session ID preserved: {session_id[:8]}...")
    else:
        print(f"\n    ❌ Session ID changed! {session_id[:8]} -> {session_id2[:8] if session_id2 else 'None'}")
    
    # Check if answer mentions TLKM or Telkom
    if "tlkm" in answer2.lower() or "telkom" in answer2.lower() or "harga" in answer2.lower():
        print("    ✓ Answer contains context from previous question!")
    else:
        print("    ❌ Answer does NOT mention TLKM/Telkom - memory not working!")
        print(f"\n    Full answer: {answer2}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_session_memory()
