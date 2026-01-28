
Dokumen ini berisi panduan lengkap untuk menjalankan, mengelola, dan memonitor AI Chatbot di DigitalOcean Droplet secara manual.

## 🔑 1. Persiapan Credentials (.env)

Pastikan file `.env` di komputer lokal Anda sudah lengkap dengan key berikut:

```properties
# MySQL Connection (Local - for ETL only)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=idnrag

# Local Qdrant (for ETL backup & development)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Production Qdrant (DigitalOcean Droplet)
QDRANT_URL=http://165.227.178.68:6333
QDRANT_API_KEY=rahasia_banget_123
QDRANT_COLLECTION=financial_data

# Gemini API (Direct Integration)
GEMINI_API_KEY=AIzaSyAWd_F5z8V22xbf9GG6-6vdUY3qalDb6T8
GEMINI_MODEL=gemini-2.5-flash

# ETL Configuration
BATCH_SIZE=50
```

> **Catatan:** Jangan bagikan file `.env` ini ke publik!

---

## 🌐 2. Deployment Web Chatbot (Manual)

Jika Anda ingin deploy ulang atau update code chatbot ke Droplet:

### Cara Otomatis (Recommended)
Jalankan script auto-deploy:
```bash
./deploy_to_droplet.sh
```

### Cara Manual Step-by-Step
Jika ingin menjalankan perintah satu per satu via SSH:

1. **Login ke Droplet:**
   ```bash
   ssh root@165.227.178.68
   ```

2. **Masuk ke folder aplikasi:**
   ```bash
   cd /opt/chatbot
   ```

3. **Pull update code (jika pakai git) atau copy manual** (biasanya dihandle script).

4. **Rebuild & Restart Container:**
   ```bash
   # Stop container lama
   docker compose down || docker-compose down

   # Build dan start container baru (background)
   docker compose up --build -d
   ```

5. **Cek Status Container:**
   ```bash
   docker compose ps
   ```
   Pastikan statusnya `Up (healthy)`.

6. **Lihat Logs (untuk debugging):**
   ```bash
   docker compose logs -f --tail=50
   ```
   Tekan `Ctrl+C` untuk keluar dari logs.

---

## 🔄 3. Update Data (ETL)

Untuk mengupdate data saham terbaru dari MySQL lokal ke Qdrant Cloud:

**Jalankan Script ETL:**
```bash
./run_etl_cloud.sh
```

Script ini akan:
1. Mengambil data dari MySQL lokal.
2. Membuat embedding menggunakan GPU lokal (cepat).
3. Mengupload hasil vector ke **Local Qdrant** (backup) DAN **Cloud Qdrant** (production) secara bersamaan.

---

## 🛠️ 4. Monitoring & Troubleshooting

### Cek Apakah Chatbot Online
Buka di browser: http://165.227.178.68:8000
Atau cek health endpoint:
```bash
curl http://165.227.178.68:8000/health
```

### Cek Qdrant Dashboard
Buka: http://165.227.178.68:6333/dashboard
Ini berguna untuk melihat apakah data vector sudah masuk.

### Jika Chatbot Error
1. **Cek Koneksi Gemini:** Pastikan API Key valid.
2. **Cek Koneksi Qdrant:** Pastikan Qdrant container running di Droplet.
3. **Restart Service:**
   ```bash
   ssh root@165.227.178.68 "cd /opt/chatbot && docker compose restart"
   ```

---

## 📋 Ringkasan Command Penting

| Command | Fungsi |
|---------|--------|
| `./deploy_to_droplet.sh` | Deploy update web chatbot ke cloud |
| `./run_etl_cloud.sh` | Sync data MySQL -> Qdrant Cloud |
| `ssh root@165.227.178.68` | Masuk ke server cloud |
| `docker compose logs -f` | Lihat log error chatbot di server |

Selamat menjalankan AI Chatbot! 🚀
