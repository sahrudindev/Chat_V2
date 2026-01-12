# 🚀 Tutorial Menjalankan RAG Chatbot

## Arsitektur Sistem

```
┌─────────────────────────────────────────────┐
│              DOCKER CONTAINERS               │
│  ┌─────────────┐       ┌─────────────┐      │
│  │  Web Chat   │       │   Qdrant    │      │
│  │  :8080      │       │   :6333     │      │
│  └──────┬──────┘       └─────────────┘      │
└─────────┼───────────────────────────────────┘
          │ HTTP (host.docker.internal:8001)
          ▼
┌─────────────────────────────────────────────┐
│              LOCAL (ROCm GPU)               │
│  ┌─────────────────────────────────────┐   │
│  │  AI Service :8001                   │   │
│  │  - /embed → BGE-M3 (GPU)            │   │
│  │  - /chat  → Gemini 2.0 Flash        │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## Langkah 1: Persiapan Awal (Sekali Saja)

### 1.1 Install Dependencies AI Service
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/ai-service
pip install -r requirements.txt
```

### 1.2 Setup Gemini API Key
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/ai-service
nano .env
```
Isi dengan:
```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```
> Dapatkan API key di: https://aistudio.google.com/app/apikey

---

## Langkah 2: Setup Data (Sekali Saja / Jika Data Berubah)

### 2.1 Jalankan Qdrant
```bash
cd /home/fiqri/Desktop/IDN/AI_v2
docker compose up -d qdrant
```

### 2.2 Jalankan ETL (Populate Vector Database)
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/etl
python mysql_to_qdrant_etl.py --qdrant-host localhost
```
⏱️ Waktu: ~10-15 menit untuk 866 perusahaan

---

## Langkah 3: Jalankan Sistem

### Terminal 1: Start AI Service
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/ai-service
python main.py
```
Tunggu sampai muncul:
```
INFO - BGE-M3 model loaded!
INFO - Gemini service initialized
INFO - Uvicorn running on http://0.0.0.0:8001
```

### Terminal 2: Start Docker (Web + Qdrant)
```bash
cd /home/fiqri/Desktop/IDN/AI_v2
docker compose up -d
```

---

## Langkah 4: Akses Chatbot

Buka browser: **http://localhost:8080**

### Contoh Pertanyaan:
- "Ceritakan tentang Bank Central Asia"
- "Bandingkan BBRI dan BMRI"
- "Perusahaan dengan harga dibawah 1000"
- "Saham listing sebelum 2000"
- "Perusahaan kelapa sawit terdaftar"

---

## Langkah 5: Verifikasi

```bash
# Cek semua service
curl http://localhost:8001/health  # AI Service
curl http://localhost:8080/health  # Web Chatbot
curl http://localhost:6333         # Qdrant

# Cek Docker
docker compose ps
```

---

## Menghentikan Sistem

```bash
# Stop Docker
cd /home/fiqri/Desktop/IDN/AI_v2
docker compose down

# Stop AI Service (Ctrl+C di terminal)
```

---

## Troubleshooting

### Qdrant Kosong (Belum Ada Data)
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/etl
python mysql_to_qdrant_etl.py --qdrant-host localhost
```

### GPU Error (ROCm)
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/scripts
bash fix_all_rocm_libs.sh
```

### Gemini Quota Exceeded (429 Error)
- Tunggu 1-2 menit lalu coba lagi
- Atau upgrade ke Gemini paid tier

---

## Quick Reference

| Service | Port | URL |
|---------|------|-----|
| Web Chatbot | 8080 | http://localhost:8080 |
| AI Service | 8001 | http://localhost:8001 |
| Qdrant | 6333 | http://localhost:6333/dashboard |

---

## Fitur Hybrid Search

Chatbot mendukung **hybrid search**:

| Query Type | Contoh | Cara Kerja |
|------------|--------|------------|
| Semantic | "Bank terbesar" | Embedding similarity |
| Filter Numerik | "Harga dibawah 1000" | Qdrant filter |
| Filter Tanggal | "Listing sebelum 2000" | Semantic + context |
