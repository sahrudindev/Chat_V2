#!/usr/bin/env python3
"""
Comprehensive 100-Question Stress Test for IDN Financials Chatbot
==================================================================

Tests all 7 query types from system3.txt with extensive coverage.

Usage:
    python stress_test_100.py
    
Output: Results saved to stress_test_results.txt
"""

import asyncio
import httpx
import time
import sys
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8080"
TIMEOUT = 120.0  # Longer timeout for complex queries

@dataclass
class TestCase:
    """Test case definition."""
    id: int
    query_type: str
    query: str
    description: str

# 100 Test Cases covering all query types
TEST_CASES = [
    # ==================== TYPE 1: DESCRIPTIVE (15 questions) ====================
    TestCase(1, "TYPE 1 - Descriptive", "Ceritakan tentang BBCA", "Profile bank terbesar"),
    TestCase(2, "TYPE 1 - Descriptive", "Profil perusahaan TLKM", "Profile telekomunikasi"),
    TestCase(3, "TYPE 1 - Descriptive", "Apa itu ASII?", "Profile otomotif"),
    TestCase(4, "TYPE 1 - Descriptive", "Jelaskan tentang BMRI", "Profile bank BUMN"),
    TestCase(5, "TYPE 1 - Descriptive", "Deskripsi UNVR", "Profile consumer goods"),
    TestCase(6, "TYPE 1 - Descriptive", "Siapa GOTO?", "Profile teknologi"),
    TestCase(7, "TYPE 1 - Descriptive", "Ceritakan ICBP", "Profile makanan"),
    TestCase(8, "TYPE 1 - Descriptive", "Apa itu BBRI?", "Profile bank BRI"),
    TestCase(9, "TYPE 1 - Descriptive", "Jelaskan ANTM", "Profile tambang"),
    TestCase(10, "TYPE 1 - Descriptive", "Profil PGAS", "Profile gas"),
    TestCase(11, "TYPE 1 - Descriptive", "Apa itu IDN Financials?", "Platform info"),
    TestCase(12, "TYPE 1 - Descriptive", "Kapan BBCA didirikan?", "Founding date"),
    TestCase(13, "TYPE 1 - Descriptive", "Dimana kantor pusat TLKM?", "HQ location"),
    TestCase(14, "TYPE 1 - Descriptive", "Bidang usaha ASII apa saja?", "Business segments"),
    TestCase(15, "TYPE 1 - Descriptive", "INDF bergerak di sektor apa?", "Sector info"),
    
    # ==================== TYPE 2: LIST/FILTER/RANKING (25 questions) ====================
    TestCase(16, "TYPE 2 - List/Ranking", "5 saham harga dibawah 1000", "Price filter <1000"),
    TestCase(17, "TYPE 2 - List/Ranking", "10 saham harga diatas 10000", "Price filter >10000"),
    TestCase(18, "TYPE 2 - List/Ranking", "3 bank kapitalisasi terbesar", "Top 3 bank by mcap"),
    TestCase(19, "TYPE 2 - List/Ranking", "5 bank kapitalisasi terbesar", "Top 5 bank by mcap"),
    TestCase(20, "TYPE 2 - List/Ranking", "Saham sektor properti", "Property sector"),
    TestCase(21, "TYPE 2 - List/Ranking", "Daftar saham perbankan", "Banking list"),
    TestCase(22, "TYPE 2 - List/Ranking", "Saham sektor teknologi", "Tech sector"),
    TestCase(23, "TYPE 2 - List/Ranking", "Saham sektor kesehatan", "Healthcare sector"),
    TestCase(24, "TYPE 2 - List/Ranking", "Saham sektor energi", "Energy sector"),
    TestCase(25, "TYPE 2 - List/Ranking", "5 saham tertua di BEI", "Oldest stocks"),
    TestCase(26, "TYPE 2 - List/Ranking", "10 perusahaan market cap terbesar", "Top 10 mcap"),
    TestCase(27, "TYPE 2 - List/Ranking", "5 saham termurah di BEI", "Cheapest stocks"),
    TestCase(28, "TYPE 2 - List/Ranking", "Saham listing sebelum tahun 2000", "Pre-2000 listing"),
    TestCase(29, "TYPE 2 - List/Ranking", "Saham listing sesudah 2020", "Post-2020 listing"),
    TestCase(30, "TYPE 2 - List/Ranking", "5 saham volume tertinggi", "High volume"),
    TestCase(31, "TYPE 2 - List/Ranking", "Saham sektor konstruksi", "Construction"),
    TestCase(32, "TYPE 2 - List/Ranking", "Saham sektor farmasi", "Pharma"),
    TestCase(33, "TYPE 2 - List/Ranking", "Daftar saham telekomunikasi", "Telecom list"),
    TestCase(34, "TYPE 2 - List/Ranking", "3 saham tambang terbesar", "Top 3 mining"),
    TestCase(35, "TYPE 2 - List/Ranking", "Saham sektor otomotif", "Automotive"),
    TestCase(36, "TYPE 2 - List/Ranking", "Saham sektor ritel", "Retail"),
    TestCase(37, "TYPE 2 - List/Ranking", "5 saham makanan terbesar", "Top 5 F&B"),
    TestCase(38, "TYPE 2 - List/Ranking", "Saham sektor transportasi", "Transport"),
    TestCase(39, "TYPE 2 - List/Ranking", "3 asuransi kapitalisasi terbesar", "Top 3 insurance"),
    TestCase(40, "TYPE 2 - List/Ranking", "Saham sektor media", "Media"),
    
    # ==================== TYPE 3: COMPARISON - GENERAL (10 questions) ====================
    TestCase(41, "TYPE 3 - Comparison", "Bandingkan BBCA dan BMRI", "Bank comparison"),
    TestCase(42, "TYPE 3 - Comparison", "Perbedaan TLKM vs ISAT", "Telecom comparison"),
    TestCase(43, "TYPE 3 - Comparison", "Bandingkan ASII dengan UNTR", "Automotive comparison"),
    TestCase(44, "TYPE 3 - Comparison", "BBRI vs BBNI mana lebih besar?", "BRI vs BNI"),
    TestCase(45, "TYPE 3 - Comparison", "Perbandingan ICBP dan INDF", "Food comparison"),
    TestCase(46, "TYPE 3 - Comparison", "GOTO vs BUKA mana lebih besar?", "Tech comparison"),
    TestCase(47, "TYPE 3 - Comparison", "Bandingkan ANTM dan INCO", "Mining comparison"),
    TestCase(48, "TYPE 3 - Comparison", "ADRO vs PTBA", "Coal comparison"),
    TestCase(49, "TYPE 3 - Comparison", "HMSP vs GGRM", "Tobacco comparison"),
    TestCase(50, "TYPE 3 - Comparison", "SMGR vs INTP", "Cement comparison"),
    
    # ==================== TYPE 3B: FINANCIAL COMPARISON (10 questions) ====================
    TestCase(51, "TYPE 3B - Financial Comparison", "Bandingkan ROE BBCA dan BMRI", "ROE bank comparison"),
    TestCase(52, "TYPE 3B - Financial Comparison", "ROA BBRI vs BBNI mana lebih tinggi?", "ROA comparison"),
    TestCase(53, "TYPE 3B - Financial Comparison", "Perbandingan laba bersih TLKM dan ISAT", "Net profit comparison"),
    TestCase(54, "TYPE 3B - Financial Comparison", "EPS ASII vs UNTR mana lebih besar?", "EPS comparison"),
    TestCase(55, "TYPE 3B - Financial Comparison", "Bandingkan dividen BBCA dan BMRI", "Dividend comparison"),
    TestCase(56, "TYPE 3B - Financial Comparison", "Revenue TLKM vs ISAT", "Revenue comparison"),
    TestCase(57, "TYPE 3B - Financial Comparison", "Perbandingan net profit ICBP dan INDF", "Food profit comparison"),
    TestCase(58, "TYPE 3B - Financial Comparison", "ROE tertinggi antara BBCA, BMRI, dan BBRI", "Multi-bank ROE"),
    TestCase(59, "TYPE 3B - Financial Comparison", "Margin laba UNVR vs MYOR", "Profit margin comparison"),
    TestCase(60, "TYPE 3B - Financial Comparison", "Kapitalisasi GOTO vs BUKA vs EMTK", "Multi-tech mcap"),
    
    # ==================== TYPE 4: DUAL CRITERIA (8 questions) ====================
    TestCase(61, "TYPE 4 - Dual Criteria", "Saham keuntungan dan kerugian terbesar", "Gainers & losers"),
    TestCase(62, "TYPE 4 - Dual Criteria", "Top gainers dan losers hari ini", "Daily movers"),
    TestCase(63, "TYPE 4 - Dual Criteria", "Saham naik dan turun paling banyak", "Price change"),
    TestCase(64, "TYPE 4 - Dual Criteria", "Top 5 gainers dan 5 losers", "Top 5 each"),
    TestCase(65, "TYPE 4 - Dual Criteria", "Saham paling untung dan paling rugi", "Extreme movers"),
    TestCase(66, "TYPE 4 - Dual Criteria", "Performa terbaik dan terburuk", "Best & worst"),
    TestCase(67, "TYPE 4 - Dual Criteria", "Saham kenaikan dan penurunan tertinggi", "Highest change"),
    TestCase(68, "TYPE 4 - Dual Criteria", "Top performers dan bottom performers", "Performance"),
    
    # ==================== TYPE 5: SHAREHOLDERS (10 questions) ====================
    TestCase(69, "TYPE 5 - Shareholders", "Siapa pemegang saham BBCA?", "BBCA shareholders"),
    TestCase(70, "TYPE 5 - Shareholders", "Struktur kepemilikan TLKM", "TLKM ownership"),
    TestCase(71, "TYPE 5 - Shareholders", "Pemegang saham utama ASII", "ASII shareholders"),
    TestCase(72, "TYPE 5 - Shareholders", "Kepemilikan saham BMRI", "BMRI ownership"),
    TestCase(73, "TYPE 5 - Shareholders", "Siapa mayoritas di BBRI?", "BBRI majority"),
    TestCase(74, "TYPE 5 - Shareholders", "Pemegang saham GOTO", "GOTO shareholders"),
    TestCase(75, "TYPE 5 - Shareholders", "Struktur kepemilikan UNVR", "UNVR ownership"),
    TestCase(76, "TYPE 5 - Shareholders", "Siapa pemilik ICBP?", "ICBP owners"),
    TestCase(77, "TYPE 5 - Shareholders", "Kepemilikan saham INDF", "INDF ownership"),
    TestCase(78, "TYPE 5 - Shareholders", "Pemegang saham terbesar ANTM", "ANTM majority"),
    
    # ==================== TYPE 6: DIVIDENDS (10 questions) ====================
    TestCase(79, "TYPE 6 - Dividends", "Dividend yield tertinggi", "Top dividend yield"),
    TestCase(80, "TYPE 6 - Dividends", "Riwayat dividen BBCA", "BBCA dividend history"),
    TestCase(81, "TYPE 6 - Dividends", "Dividen TLKM tahun ini", "TLKM dividend"),
    TestCase(82, "TYPE 6 - Dividends", "5 saham dividen terbesar", "Top 5 dividends"),
    TestCase(83, "TYPE 6 - Dividends", "Histori dividen ASII", "ASII dividend history"),
    TestCase(84, "TYPE 6 - Dividends", "Dividen yield BMRI", "BMRI yield"),
    TestCase(85, "TYPE 6 - Dividends", "Saham dengan DPS tertinggi", "Top DPS"),
    TestCase(86, "TYPE 6 - Dividends", "Riwayat dividen UNVR", "UNVR dividends"),
    TestCase(87, "TYPE 6 - Dividends", "Dividen bank terbesar", "Bank dividends"),
    TestCase(88, "TYPE 6 - Dividends", "Saham dividen konsisten", "Consistent dividends"),
    
    # ==================== TYPE 7: FINANCIAL STATEMENTS (10 questions) ====================
    TestCase(89, "TYPE 7 - Financials", "ROE BBCA berapa?", "BBCA ROE"),
    TestCase(90, "TYPE 7 - Financials", "5 perusahaan ROE tertinggi", "Top 5 ROE"),
    TestCase(91, "TYPE 7 - Financials", "Net profit ASII", "ASII net profit"),
    TestCase(92, "TYPE 7 - Financials", "EPS tertinggi di BEI", "Top EPS"),
    TestCase(93, "TYPE 7 - Financials", "ROA BMRI berapa?", "BMRI ROA"),
    TestCase(94, "TYPE 7 - Financials", "Revenue TLKM", "TLKM revenue"),
    TestCase(95, "TYPE 7 - Financials", "Laba bersih BBRI", "BBRI profit"),
    TestCase(96, "TYPE 7 - Financials", "5 bank dengan ROE tertinggi", "Top ROE banks"),
    TestCase(97, "TYPE 7 - Financials", "Kinerja keuangan UNVR", "UNVR financials"),
    TestCase(98, "TYPE 7 - Financials", "Profit margin tertinggi", "Top profit margin"),
    
    # ==================== INVESTMENT BLOCK & EDGE CASES (2 questions) ====================
    TestCase(99, "Investment Block", "Saham apa yang bagus dibeli?", "Should decline"),
    TestCase(100, "Edge Case", "Kontak IDN Financials", "Contact info"),
]


async def run_test(client: httpx.AsyncClient, test: TestCase) -> Tuple[bool, str, float]:
    """Run a single test case."""
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
        
        # Basic validation - response should not be empty
        if not answer or len(answer) < 10:
            return False, "Empty or too short response", elapsed
        
        # Check for error messages
        if "error" in answer.lower() or "maaf, terjadi kesalahan" in answer.lower():
            return False, f"Error in response: {answer[:100]}", elapsed
        
        preview = answer[:100] + "..." if len(answer) > 100 else answer
        return True, preview, elapsed
        
    except httpx.TimeoutException:
        return False, "TIMEOUT", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


async def run_all_tests():
    """Run all 100 test cases."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 70)
    print("🧪 IDN Financials Chatbot - 100 Question Stress Test")
    print("=" * 70)
    print(f"Started: {timestamp}")
    print(f"Base URL: {BASE_URL}")
    print(f"Total Tests: {len(TEST_CASES)}")
    print("=" * 70)
    
    results = []
    passed = 0
    failed = 0
    total_time = 0
    
    async with httpx.AsyncClient() as client:
        # Health check
        try:
            health = await client.get(f"{BASE_URL}/health", timeout=10.0)
            if health.status_code != 200:
                print("❌ Server health check failed!")
                return False
            print("✅ Server is healthy\n")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("   Make sure Docker is running: docker compose up -d")
            return False
        
        for test in TEST_CASES:
            print(f"[{test.id:3d}/100] {test.query_type}")
            print(f"         Query: {test.query}")
            
            success, message, elapsed = await run_test(client, test)
            total_time += elapsed
            
            result = {
                "id": test.id,
                "type": test.query_type,
                "query": test.query,
                "passed": success,
                "time": elapsed,
                "response": message
            }
            results.append(result)
            
            if success:
                passed += 1
                print(f"         ✅ PASS ({elapsed:.1f}s)")
            else:
                failed += 1
                print(f"         ❌ FAIL ({elapsed:.1f}s) - {message[:50]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total:  {len(TEST_CASES)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Rate:   {passed/len(TEST_CASES)*100:.1f}%")
    print(f"Time:   {total_time:.1f}s (avg: {total_time/len(TEST_CASES):.1f}s per query)")
    print("=" * 70)
    
    # Stats by type
    print("\n📋 Results by Query Type:")
    type_stats = {}
    for r in results:
        qtype = r["type"]
        if qtype not in type_stats:
            type_stats[qtype] = {"passed": 0, "failed": 0, "total_time": 0}
        if r["passed"]:
            type_stats[qtype]["passed"] += 1
        else:
            type_stats[qtype]["failed"] += 1
        type_stats[qtype]["total_time"] += r["time"]
    
    for qtype, stats in type_stats.items():
        total = stats["passed"] + stats["failed"]
        pct = stats["passed"] / total * 100
        avg_time = stats["total_time"] / total
        status = "✅" if stats["failed"] == 0 else "⚠️" if stats["passed"] > 0 else "❌"
        print(f"  {status} {qtype}: {stats['passed']}/{total} ({pct:.0f}%) avg: {avg_time:.1f}s")
    
    # Save results to file
    with open("stress_test_results.txt", "w") as f:
        f.write(f"IDN Financials Chatbot - 100 Question Stress Test\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"SUMMARY: {passed}/{len(TEST_CASES)} passed ({passed/len(TEST_CASES)*100:.1f}%)\n")
        f.write(f"Total Time: {total_time:.1f}s\n\n")
        
        f.write("FAILED TESTS:\n")
        f.write("-" * 70 + "\n")
        for r in results:
            if not r["passed"]:
                f.write(f"[{r['id']}] {r['type']}\n")
                f.write(f"    Query: {r['query']}\n")
                f.write(f"    Error: {r['response']}\n\n")
        
        f.write("\nALL RESULTS:\n")
        f.write("-" * 70 + "\n")
        for r in results:
            status = "✅" if r["passed"] else "❌"
            f.write(f"[{r['id']:3d}] {status} {r['query'][:50]}... ({r['time']:.1f}s)\n")
    
    print(f"\n📁 Results saved to: stress_test_results.txt")
    
    return passed == len(TEST_CASES)


if __name__ == "__main__":
    print("\n🚀 Starting 100-question stress test...\n")
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
