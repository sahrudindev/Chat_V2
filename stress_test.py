#!/usr/bin/env python3
"""
Comprehensive Chatbot Stress Test
==================================

Tests all 7 query types from system3.txt plus edge cases.
Outputs results with status indicators.

Usage:
    python stress_test.py [--base-url URL]
"""

import asyncio
import httpx
import time
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Configuration
BASE_URL = "http://localhost:8080"  # No /api prefix - router mounted at root
TIMEOUT = 60.0  # Gemini can be slow

@dataclass
class TestCase:
    """Test case definition."""
    query_type: str
    query: str
    expected_keywords: List[str]  # Keywords expected in response
    description: str

# All test cases covering 7 query types from system3.txt
TEST_CASES = [
    # ============= TYPE 1: DESCRIPTIVE =============
    TestCase(
        query_type="TYPE 1 - Descriptive",
        query="Ceritakan tentang BBCA",
        expected_keywords=["Bank Central Asia", "BBCA", "sektor"],  # 'Bank' or 'perbankan' both valid
        description="Company profile narrative"
    ),
    TestCase(
        query_type="TYPE 1 - Descriptive",
        query="Profil perusahaan TLKM",
        expected_keywords=["Telkom", "TLKM"],  # LLM may say Infrastruktur, Telekomunikasi, etc.
        description="Company profile with specific terms"
    ),
    TestCase(
        query_type="TYPE 1 - Descriptive",
        query="Apa itu IDN Financials?",
        expected_keywords=["idnfinancials", "Video", "Teknologi", "Bursa Efek"],
        description="Platform info from system prompt"
    ),
    
    # ============= TYPE 2: LIST/FILTER/RANKING =============
    TestCase(
        query_type="TYPE 2 - List/Ranking",
        query="5 saham dengan harga dibawah 1000",
        expected_keywords=["|", "Perusahaan", "Kode", "Harga"],
        description="Price filter with table"
    ),
    TestCase(
        query_type="TYPE 2 - List/Ranking",
        query="3 bank dengan kapitalisasi terbesar",
        expected_keywords=["|", "Kapitalisasi", "bank"],
        description="Sector + ranking filter"
    ),
    TestCase(
        query_type="TYPE 2 - List/Ranking",
        query="Saham sektor properti",
        expected_keywords=["|", "properti", "Kode"],
        description="Sector filter only"
    ),
    
    # ============= TYPE 3: COMPARISON =============
    TestCase(
        query_type="TYPE 3 - Comparison",
        query="Bandingkan BBCA dan BMRI",
        expected_keywords=["BBCA", "BMRI", "|", "Aspek"],
        description="Two-company comparison table"
    ),
    TestCase(
        query_type="TYPE 3 - Comparison",
        query="Perbedaan TLKM vs ISAT",
        expected_keywords=["TLKM", "ISAT", "|"],
        description="Comparison with vs keyword"
    ),
    
    # ============= TYPE 4: DUAL CRITERIA =============
    TestCase(
        query_type="TYPE 4 - Dual Criteria",
        query="Saham dengan keuntungan dan kerugian terbesar",
        expected_keywords=["Keuntungan", "Kerugian", "|"],  # LLM uses Indonesian terms
        description="Two tables: gainers + losers"
    ),
    TestCase(
        query_type="TYPE 4 - Dual Criteria",
        query="Top gainers dan losers hari ini",
        expected_keywords=["Gainer", "Loser", "|"],  # Accept mixed case
        description="Market movers dual table"
    ),
    
    # ============= TYPE 5: SHAREHOLDERS =============
    TestCase(
        query_type="TYPE 5 - Shareholders",
        query="Siapa pemegang saham BBCA?",
        expected_keywords=["Pemegang", "Kepemilikan", "%", "saham"],
        description="Shareholder table"
    ),
    TestCase(
        query_type="TYPE 5 - Shareholders",
        query="Struktur kepemilikan TLKM",
        expected_keywords=["|", "Pemegang", "%"],
        description="Ownership structure"
    ),
    
    # ============= TYPE 6: DIVIDENDS =============
    TestCase(
        query_type="TYPE 6 - Dividends",
        query="Dividend yield tertinggi",
        expected_keywords=["Yield", "%", "Tahun", "|"],
        description="Dividend ranking with year"
    ),
    TestCase(
        query_type="TYPE 6 - Dividends",
        query="Riwayat dividen BBCA",
        expected_keywords=["dividen", "BBCA", "Tahun"],  # 'Dividend Per Share' or 'DPS' both ok
        description="Company dividend history"
    ),
    
    # ============= TYPE 7: FINANCIAL STATEMENTS =============
    TestCase(
        query_type="TYPE 7 - Financials",
        query="ROE BBCA berapa?",
        expected_keywords=["ROE", "%", "BBCA"],
        description="Single company financial metric"
    ),
    TestCase(
        query_type="TYPE 7 - Financials",
        query="5 perusahaan dengan ROE tertinggi",
        expected_keywords=["ROE", "|", "%", "Perusahaan"],
        description="Financial metric ranking"
    ),
    TestCase(
        query_type="TYPE 7 - Financials",
        query="Net profit ASII",
        expected_keywords=["Net Profit", "ASII", "Rp"],
        description="Profit metric query"
    ),
    
    # ============= INVESTMENT PROHIBITION =============
    TestCase(
        query_type="Investment Block",
        query="Saham apa yang bagus untuk dibeli?",
        expected_keywords=["tidak dapat memberikan", "rekomendasi"],
        description="Should decline investment advice"
    ),
    TestCase(
        query_type="Investment Block",
        query="Prediksi harga BBCA besok",
        expected_keywords=["tidak dapat", "prediksi"],
        description="Should decline price prediction"
    ),
    
    # ============= EDGE CASES =============
    TestCase(
        query_type="Edge Case",
        query="GOTO listing kapan?",
        expected_keywords=["GOTO", "tercatat"],  # LLM says 'tercatat' instead of 'listing'
        description="Short follow-up style query"
    ),
    TestCase(
        query_type="Edge Case",
        query="Kontak IDN Financials",
        expected_keywords=["email", "telepon", "alamat"],
        description="Contact info from system prompt"
    ),
]


async def run_test(client: httpx.AsyncClient, test: TestCase) -> Tuple[bool, str, float]:
    """
    Run a single test case.
    
    Returns:
        Tuple of (passed, response_preview, response_time)
    """
    start = time.time()
    try:
        response = await client.post(
            f"{BASE_URL}/chat",
            json={"message": test.query, "session_id": None},
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}", elapsed
        
        data = response.json()
        answer = data.get("answer", "")
        
        # Check expected keywords (case-insensitive)
        answer_lower = answer.lower()
        missing = [kw for kw in test.expected_keywords if kw.lower() not in answer_lower]
        
        if missing:
            preview = answer[:150] + "..." if len(answer) > 150 else answer
            return False, f"Missing: {missing}\nGot: {preview}", elapsed
        
        preview = answer[:80] + "..." if len(answer) > 80 else answer
        return True, preview, elapsed
        
    except httpx.TimeoutException:
        return False, "TIMEOUT", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


async def run_all_tests():
    """Run all test cases and print results."""
    print("=" * 70)
    print("🧪 IDN Financials Chatbot Stress Test")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Total Tests: {len(TEST_CASES)}")
    print("=" * 70)
    
    passed = 0
    failed = 0
    total_time = 0
    
    results: List[Dict] = []
    
    async with httpx.AsyncClient() as client:
        # Health check first
        try:
            health = await client.get(f"{BASE_URL.replace('/api', '')}/health", timeout=5.0)
            if health.status_code != 200:
                print("❌ Server health check failed!")
                return
            print("✅ Server is healthy\n")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("   Make sure to run: cd web-chatbot && uvicorn app.main:app --reload")
            return
        
        for i, test in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] {test.query_type}")
            print(f"    Query: {test.query}")
            print(f"    Desc:  {test.description}")
            
            success, message, elapsed = await run_test(client, test)
            total_time += elapsed
            
            if success:
                passed += 1
                print(f"    ✅ PASS ({elapsed:.1f}s)")
                print(f"    Response: {message}")
            else:
                failed += 1
                print(f"    ❌ FAIL ({elapsed:.1f}s)")
                print(f"    {message}")
            
            results.append({
                "query_type": test.query_type,
                "query": test.query,
                "passed": success,
                "time": elapsed,
                "message": message
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total:  {len(TEST_CASES)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Rate:   {passed/len(TEST_CASES)*100:.1f}%")
    print(f"Time:   {total_time:.1f}s (avg: {total_time/len(TEST_CASES):.1f}s)")
    print("=" * 70)
    
    # Group by query type
    print("\n📋 Results by Query Type:")
    type_stats = {}
    for r in results:
        qtype = r["query_type"]
        if qtype not in type_stats:
            type_stats[qtype] = {"passed": 0, "failed": 0}
        if r["passed"]:
            type_stats[qtype]["passed"] += 1
        else:
            type_stats[qtype]["failed"] += 1
    
    for qtype, stats in type_stats.items():
        total = stats["passed"] + stats["failed"]
        pct = stats["passed"] / total * 100
        status = "✅" if stats["failed"] == 0 else "⚠️" if stats["passed"] > 0 else "❌"
        print(f"  {status} {qtype}: {stats['passed']}/{total} ({pct:.0f}%)")
    
    return passed == len(TEST_CASES)


if __name__ == "__main__":
    # Allow custom base URL
    if len(sys.argv) > 2 and sys.argv[1] == "--base-url":
        BASE_URL = sys.argv[2]
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
