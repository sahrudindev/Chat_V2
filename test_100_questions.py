#!/usr/bin/env python3
"""
100 Test Questions untuk RAG Chatbot.
"""
import requests
import json
import time

WEB_CHATBOT = "http://localhost:8080"

# 100 Pertanyaan Test
QUESTIONS = [
    # === BASIC INFO (1-20) ===
    "Apa itu BBCA?",
    "Ceritakan tentang Bank BCA",
    "Perusahaan apa saja yang bergerak di bidang perbankan?",
    "Sebutkan 5 perusahaan bank terbesar",
    "Apa bidang usaha TLKM?",
    "Siapa yang mengelola BMRI?",
    "Alamat kantor BBNI dimana?",
    "Website resmi BBRI apa?",
    "Email kontak ASII berapa?",
    "Nomor telepon UNVR berapa?",
    
    # === LISTING & ESTABLISHMENT (11-20) ===
    "Kapan BBCA listing di bursa?",
    "Tahun berapa TLKM didirikan?",
    "Kapan ASII mulai tercatat di BEI?",
    "Perusahaan mana yang listing sebelum tahun 2000?",
    "Bank apa yang paling lama listing?",
    "Kapan GOTO listing?",
    "Tahun pendirian BMRI?",
    "Kapan UNVR didirikan?",
    "Perusahaan tertua di sektor perbankan?",
    "Listing date ICBP?",
    
    # === HARGA SAHAM (21-40) ===
    "Berapa harga saham BBCA?",
    "Harga close BBRI berapa?",
    "Harga open BMRI hari ini?",
    "Berapa harga tertinggi TLKM?",
    "Harga terendah ASII berapa?",
    "Bandingkan harga BBCA, BBRI, BMRI",
    "5 saham bank dengan harga tertinggi",
    "Saham perbankan dengan harga dibawah 1000",
    "Berapa perubahan harga BBCA?",
    "Persentase kenaikan GOTO?",
    "Harga previous UNVR?",
    "Selisih harga open dan close BBNI?",
    "Saham dengan kenaikan harga terbesar?",
    "Saham dengan penurunan harga terbesar?",
    "Daftar harga 10 saham bank",
    "Harga saham sektor teknologi",
    "Perbandingan harga 5 bank terbesar",
    "Berapa range harga ICBP hari ini?",
    "Volatilitas harga BBCA?",
    "Harga rata-rata saham perbankan?",
    
    # === VOLUME & TRADING (41-55) ===
    "Volume trading BBCA berapa?",
    "Nilai transaksi BBRI?",
    "Frekuensi trading ASII?",
    "Saham dengan volume tertinggi?",
    "5 saham paling aktif diperdagangkan",
    "Bandingkan volume 5 bank besar",
    "Total value trading TLKM?",
    "Saham dengan frekuensi trading terendah?",
    "Likuiditas saham GOTO?",
    "Volume harian BMRI?",
    "Nilai transaksi harian UNVR?",
    "Trading activity ICBP?",
    "Saham dengan turnover tertinggi?",
    "Perbandingan likuiditas bank BUMN",
    "Volume trading sektor consumer goods?",
    
    # === MARKET CAP & VALUASI (56-70) ===
    "Kapitalisasi pasar BBCA berapa?",
    "Market cap 5 bank terbesar?",
    "P/E ratio BBRI?",
    "EPS BMRI berapa?",
    "Valuasi TLKM?",
    "Bandingkan P/E ratio bank-bank besar",
    "Saham dengan market cap terbesar?",
    "P/E ratio tertinggi di sektor bank?",
    "EPS tertinggi di perbankan?",
    "Kapitalisasi GOTO?",
    "Valuasi perusahaan teknologi?",
    "P/E ratio ASII?",
    "Market cap sektor consumer?",
    "Rasio keuangan UNVR?",
    "Bandingkan valuasi BBCA vs BBRI",
    
    # === SEKTOR & INDUSTRI (71-85) ===
    "Perusahaan apa saja di sektor perbankan?",
    "List perusahaan teknologi di BEI",
    "Saham consumer goods apa saja?",
    "Perusahaan tambang terdaftar?",
    "Sektor apa ASII?",
    "Industri TLKM?",
    "Perusahaan properti di BEI?",
    "Saham infrastruktur?",
    "Perusahaan energi terdaftar?",
    "Sektor ICBP apa?",
    "Perusahaan farmasi di bursa?",
    "Saham retail apa saja?",
    "Perusahaan otomotif?",
    "List perusahaan manufacturing?",
    "Sektor perusahaan GOTO?",
    
    # === PERBANDINGAN (86-95) ===
    "Bandingkan BBCA dan BBRI",
    "Perbandingan 4 bank BUMN",
    "BBCA vs BMRI, mana lebih besar?",
    "Bandingkan harga dan volume 5 bank",
    "Perbandingan P/E ratio bank swasta vs BUMN",
    "TLKM vs ISAT, siapa lebih besar?",
    "Bandingkan market cap ASII dan UNVR",
    "Perbandingan EPS bank-bank besar",
    "GOTO vs BUKA, bandingkan valuasinya",
    "Perbandingan lengkap 3 bank terbesar",
    
    # === ANALISIS (96-100) ===
    "Saham bank mana yang paling murah?",
    "Rekomendasi saham bank dengan P/E rendah?",
    "Saham dengan likuiditas tinggi dan harga stabil?",
    "Bank dengan kapitalisasi terbesar?",
    "Rangkum profil 5 bank terbesar di Indonesia",
    
    # === FINANCIAL STATEMENTS - NEW (101-120) ===
    "ROE BBCA berapa?",
    "ROA tertinggi di sektor perbankan?",
    "Net profit TLKM Q4 2024?",
    "Gross margin UNVR?",
    "Net margin ASII berapa?",
    "Operating margin perusahaan consumer?",
    "Bank dengan NPL terendah?",
    "CAR BBRI berapa?",
    "LDR 4 bank BUMN?",
    "Revenue ICBP Q4 2024?",
    "Total aset BMRI?",
    "Total equity BBCA?",
    "P/BV ratio BBRI?",
    "DER ratio ASII?",
    "Trend net profit BBCA 2024?",
    "Perbandingan ROE Q1 vs Q4 2024 TLKM?",
    "Bank dengan CAR tertinggi?",
    "Perusahaan dengan net margin terbaik?",
    "Laba bersih GOTO Q4 2024?",
    "Bagaimana performa keuangan 5 bank terbesar?"
]

def test_chatbot():
    print("=" * 60)
    print("Testing RAG Chatbot - 100 Questions (With Session Support)")
    print("=" * 60)
    
    success = 0
    failed = 0
    no_answer = 0
    session_id = None  # Will be set after first request
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i:03d}/100] {question}")
        print("-" * 50)
        
        try:
            r = requests.post(
                f"{WEB_CHATBOT}/chat",
                json={"message": question, "session_id": session_id},  # Include session_id
                timeout=120
            )
            data = r.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            session_id = data.get("session_id", session_id)  # Update session_id
            
            if "tidak memiliki informasi" in answer.lower():
                print(f"❌ No Answer (Sources: {len(sources)})")
                no_answer += 1
            else:
                print(f"✓ Answered (Sources: {len(sources)})")
                print(f"   {answer[:150]}...")
                success += 1
                
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
        
        # Small delay to avoid overload
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: ✓ {success} answered | ❌ {no_answer} no answer | ✗ {failed} errors")
    print(f"Success Rate: {success}/{len(QUESTIONS)} ({100*success/len(QUESTIONS):.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    test_chatbot()
