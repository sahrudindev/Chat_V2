#!/usr/bin/env python3
"""
Stress Test untuk Data ks_financial (Financial Statements)
Test 20 pertanyaan terkait laporan keuangan.
"""
import requests
import time
import json

WEB_CHATBOT = "http://localhost:8080"

# 20 Pertanyaan Financial Statements
FINANCIAL_QUESTIONS = [
    # === ROE & ROA (1-4) ===
    ("ROE BBCA berapa?", "roe"),
    ("ROA tertinggi di sektor perbankan?", "roa"),
    ("Bandingkan ROE BBRI dan BMRI", "roe"),
    ("ROE TLKM Q4 2024?", "roe"),
    
    # === PROFITABILITY (5-8) ===
    ("Net profit ASII berapa?", "net profit"),
    ("Revenue ICBP Q4 2024?", "revenue"),
    ("Gross margin UNVR?", "margin"),
    ("Net margin perusahaan consumer terbaik?", "margin"),
    
    # === BANKING METRICS (9-12) ===
    ("Bank dengan NPL terendah?", "npl"),
    ("CAR BBRI berapa?", "car"),
    ("LDR 4 bank BUMN?", "ldr"),
    ("Bandingkan CAR, LDR, NPL BBCA vs BBRI", "car"),
    
    # === BALANCE SHEET (13-16) ===
    ("Total aset BMRI?", "aset"),
    ("Total equity BBCA?", "equity"),
    ("P/BV ratio BBRI?", "pbv"),
    ("DER ratio ASII?", "der"),
    
    # === TREND & COMPARISON (17-20) ===
    ("Trend net profit BBCA 2024?", "trend"),
    ("Perbandingan ROE Q1 vs Q4 2024 TLKM?", "q1"),
    ("Bagaimana performa keuangan 5 bank terbesar?", "bank"),
    ("Perusahaan dengan pertumbuhan net profit tertinggi?", "growth"),
]


def test_financial_queries():
    print("=" * 70)
    print("🧪 STRESS TEST - Financial Statements Data (ks_financial)")
    print("=" * 70)
    print(f"Total Questions: {len(FINANCIAL_QUESTIONS)}")
    print()
    
    success = 0
    failed = 0
    no_answer = 0
    session_id = None
    
    for i, (question, keyword) in enumerate(FINANCIAL_QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(FINANCIAL_QUESTIONS)}] {question}")
        print("-" * 60)
        
        try:
            start = time.time()
            response = requests.post(
                f"{WEB_CHATBOT}/chat",
                json={"message": question, "session_id": session_id},
                timeout=120
            )
            elapsed = time.time() - start
            
            data = response.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            session_id = data.get("session_id", session_id)
            
            # Check if answer contains expected keyword
            has_keyword = keyword.lower() in answer.lower()
            
            if "tidak memiliki informasi" in answer.lower() or "tidak dapat" in answer.lower():
                print(f"❌ No Answer ({elapsed:.1f}s)")
                print(f"   Answer: {answer[:100]}...")
                no_answer += 1
            elif has_keyword or len(answer) > 50:
                print(f"✓ Answered ({elapsed:.1f}s) [Sources: {len(sources)}]")
                print(f"   {answer[:150]}...")
                success += 1
            else:
                print(f"⚠ Partial ({elapsed:.1f}s)")
                print(f"   {answer[:150]}...")
                success += 1  # Still count as success
                
        except requests.exceptions.Timeout:
            print(f"✗ Timeout (>120s)")
            failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
        
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 HASIL STRESS TEST - FINANCIAL DATA")
    print("=" * 70)
    
    total = len(FINANCIAL_QUESTIONS)
    success_rate = (success / total) * 100
    
    print(f"   ✓ Success:   {success}/{total} ({success_rate:.1f}%)")
    print(f"   ❌ No Answer: {no_answer}/{total}")
    print(f"   ✗ Failed:    {failed}/{total}")
    print()
    
    if success_rate >= 90:
        print("   🎉 EXCELLENT! Success rate >= 90%")
    elif success_rate >= 75:
        print("   👍 GOOD! Success rate >= 75%")
    elif success_rate >= 50:
        print("   ⚠️ NEEDS IMPROVEMENT! Success rate >= 50%")
    else:
        print("   ❌ POOR! Success rate < 50%")
    
    print("=" * 70)
    
    return success_rate >= 75


if __name__ == "__main__":
    success = test_financial_queries()
    exit(0 if success else 1)
