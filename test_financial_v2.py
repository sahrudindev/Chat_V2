#!/usr/bin/env python3
"""
Stress Test v2 untuk Data ks_financial (Financial Statements)
Disesuaikan dengan perbedaan metrics Bank vs Non-Bank
"""
import requests
import time
import json

WEB_CHATBOT = "http://localhost:8080"

# 20 Pertanyaan Financial Statements (Update v2)
FINANCIAL_QUESTIONS = [
    # === GENERAL METRICS (Bisa untuk Semua) ===
    ("Net profit BBCA berapa?", "net profit"),
    ("Total aset BBRI?", "aset"),
    ("Total equity BMRI?", "equity"),
    ("EPS TLKM?", "eps"),
    
    # === BANK SPECIFIC (Calculated ROE/ROA) ===
    ("ROE BBCA berapa?", "roe"),
    ("ROA BBRI?", "roa"),
    ("Net interest income BMRI?", "interest"),
    ("Bandingkan ROE BBCA vs BBRI", "roe"),
    
    # === NON-BANK SPECIFIC ===
    ("Revenue ASII?", "revenue"),
    ("Gross margin UNVR?", "margin"),
    ("Operating margin ICBP?", "margin"),
    ("COGS GOTO?", "cogs"),
    
    # === RATIOS ===
    ("P/BV ratio BBCA?", "pbv"),
    ("DER ratio ASII?", "der"),
    ("Bandingkan P/BV BBCA vs BBRI", "pbv"),
    
    # === GROWTH & TREND ===
    ("Trend net profit TLKM 2024?", "trend"),
    ("Pertumbuhan net profit BBCA YoY?", "growth"),
    ("Revenue growth ASII?", "growth"),
    
    # === COMPLEX/RANKING ===
    ("Perusahaan dengan net profit terbesar?", "profit"),
    ("Bandingkan kinerja keuangan BBCA dan ASII", "banding"),
]


def test_financial_queries():
    print("=" * 70)
    print("🧪 STRESS TEST v2 - Financial Statements Data")
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
            
            # Check if answer contains expected keyword or meaningful data
            has_keyword = keyword.lower() in answer.lower()
            
            if "tidak memiliki informasi" in answer.lower() or "tidak dapat" in answer.lower():
                print(f"❌ No Answer ({elapsed:.1f}s)")
                print(f"   Answer: {answer[:100]}...")
                no_answer += 1
            elif has_keyword or len(answer) > 50:
                print(f"✓ Answered ({elapsed:.1f}s) [Sources: {len(sources)}]")
                print(f"   {answer[:150].replace(chr(10), ' ')}...")
                success += 1
            else:
                print(f"⚠ Partial ({elapsed:.1f}s)")
                print(f"   {answer[:150].replace(chr(10), ' ')}...")
                success += 1  # Still count as success if not "no info"
                
        except requests.exceptions.Timeout:
            print(f"✗ Timeout (>120s)")
            failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
        
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 HASIL STRESS TEST v2")
    print("=" * 70)
    
    total = len(FINANCIAL_QUESTIONS)
    success_rate = (success / total) * 100
    
    print(f"   ✓ Success:   {success}/{total} ({success_rate:.1f}%)")
    print(f"   ❌ No Answer: {no_answer}/{total}")
    print(f"   ✗ Failed:    {failed}/{total}")
    print()
    
    return success_rate >= 75


if __name__ == "__main__":
    success = test_financial_queries()
    exit(0 if success else 1)
