#!/usr/bin/env python3
"""
Comprehensive stress test untuk membuktikan semua fitur bekerja.
Test:
1. Basic query (tanpa session)
2. Session memory (follow-up questions)
3. Multiple sessions (isolation)
4. Various query types
"""
import requests
import time
import json

WEB_CHATBOT = "http://localhost:8080"

def test_query(message, session_id=None, expected_in_response=None, test_name="Test"):
    """Send a query and check response."""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"   Query: {message}")
    print(f"   Session: {session_id[:8] if session_id else 'NEW'}...")
    
    try:
        response = requests.post(
            f"{WEB_CHATBOT}/chat",
            json={"message": message, "session_id": session_id},
            timeout=90
        )
        data = response.json()
        
        answer = data.get("answer", "")[:200]
        new_session = data.get("session_id", "")
        sources = len(data.get("sources", []))
        
        print(f"   ✓ Response: {answer}...")
        print(f"   ✓ Session ID: {new_session[:8]}...")
        print(f"   ✓ Sources: {sources}")
        
        if expected_in_response:
            found = expected_in_response.lower() in answer.lower()
            if found:
                print(f"   ✓ Contains '{expected_in_response}': YES")
            else:
                print(f"   ✗ Contains '{expected_in_response}': NO")
            return new_session, found
        
        return new_session, True
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return None, False


def run_stress_test():
    print("="*60)
    print("🚀 STRESS TEST - Session Memory & RAG")
    print("="*60)
    
    results = []
    
    # Test 1: Basic single query (no session)
    sid1, ok1 = test_query(
        "Apa itu BBRI?",
        session_id=None,
        expected_in_response="Bank",
        test_name="1. Basic Query (New Session)"
    )
    results.append(("Basic Query", ok1))
    
    # Test 2: Follow-up question (same session)
    sid2, ok2 = test_query(
        "Berapa harganya?",
        session_id=sid1,
        expected_in_response=None,  # Just check it works
        test_name="2. Follow-up Query (Same Session)"
    )
    results.append(("Follow-up Query", ok2 and sid1 == sid2))
    
    # Test 3: Another follow-up
    sid3, ok3 = test_query(
        "Bagaimana performanya?",
        session_id=sid2,
        expected_in_response=None,
        test_name="3. Second Follow-up (Same Session)"
    )
    results.append(("Second Follow-up", ok3 and sid2 == sid3))
    
    # Test 4: New session (different topic)
    sid4, ok4 = test_query(
        "List perusahaan di sektor perbankan",
        session_id=None,
        expected_in_response="bank",
        test_name="4. New Session (Different Topic)"
    )
    results.append(("New Session", ok4 and sid4 != sid1))
    
    # Test 5: Filter query
    sid5, ok5 = test_query(
        "Saham dengan harga dibawah 1000",
        session_id=None,
        expected_in_response=None,
        test_name="5. Filter Query"
    )
    results.append(("Filter Query", ok5))
    
    # Test 6: Comparison query
    sid6, ok6 = test_query(
        "Bandingkan BBRI dan BMRI",
        session_id=None,
        expected_in_response=None,
        test_name="6. Comparison Query"
    )
    results.append(("Comparison Query", ok6))
    
    # Test 7: Session isolation (old session should still work)
    sid7, ok7 = test_query(
        "Apa market cap-nya?",
        session_id=sid1,  # Use original session
        expected_in_response=None,
        test_name="7. Session Isolation (Back to Session 1)"
    )
    results.append(("Session Isolation", ok7 and sid7 == sid1))
    
    # Summary
    print("\n" + "="*60)
    print("📊 HASIL STRESS TEST")
    print("="*60)
    
    passed = 0
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"   {status}: {name}")
        if ok:
            passed += 1
    
    print(f"\n   Total: {passed}/{len(results)} tests passed")
    print("="*60)
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_stress_test()
    exit(0 if success else 1)
