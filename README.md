# MySQL to Qdrant ETL Pipeline

RAG-ready ETL pipeline untuk migrasi data finansial dari MySQL ke Qdrant Vector Database menggunakan **BGE-M3 Hybrid Search** (Dense + Sparse vectors).

## 🎯 Features

- **Memory Efficient**: Server-Side Cursor (SSCursor) untuk streaming jutaan row
- **Hybrid Search**: Dense vectors (semantic) + Sparse vectors (keyword/ticker matching)
- **AMD GPU Ready**: Support ROCm untuk RX 6600 XT
- **Batch Processing**: Configurable batch size untuk optimal throughput

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- MySQL database dengan data finansial
- Qdrant running di localhost:6333

### Development Mode (Recommended untuk testing)

```bash
# 1. Install system dependencies (Fedora)
sudo dnf install -y libxml2-devel libxslt-devel

# 2. Install Python packages
pip3 install qdrant-client pymysql FlagEmbedding tqdm python-dotenv

# 3. Start Qdrant (jika belum running)
docker start qdrant
# atau: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 4. Copy dan edit environment variables
cp .env.example .env
# Edit .env dengan credentials MySQL Anda

# 5. Run ETL (dry run dulu untuk testing)
python mysql_to_qdrant_etl.py --dry-run --max-rows 100

# 6. Run ETL (full)
python mysql_to_qdrant_etl.py --batch-size 50
```

### Test Retrieval

```bash
python test_retrieval.py
```

---

## 🐳 Production Mode (Docker)

Untuk deployment production, gunakan Docker dengan AMD GPU passthrough:

```bash
# Build dan run semua services
docker compose up --build

# Atau run ETL saja (jika Qdrant sudah running terpisah)
docker compose up etl --build
```

> ⚠️ **Note**: Build Docker dengan ROCm memakan waktu ~30-60 menit karena base image ~15GB

### AMD GPU Requirements

Pastikan ROCm driver terinstall di host:
```bash
# Check ROCm
rocm-smi

# Pastikan user di group video dan render
sudo usermod -aG video,render $USER
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MYSQL_HOST` | localhost | MySQL server host |
| `MYSQL_PORT` | 3306 | MySQL server port |
| `MYSQL_USER` | root | MySQL username |
| `MYSQL_PASSWORD` | root | MySQL password |
| `MYSQL_DATABASE` | local_news | Database name |
| `MYSQL_TABLE` | stock_session1_data | Table to process |
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `QDRANT_COLLECTION` | financial_data | Collection name |
| `BATCH_SIZE` | 50 | Rows per batch |

### CLI Options

```bash
python mysql_to_qdrant_etl.py \
  --mysql-host localhost \
  --mysql-database idnrag \
  --mysql-table company \
  --batch-size 50 \
  --max-rows 1000 \    # Optional: limit rows
  --dry-run \          # Optional: skip actual inserts
  --no-shareholders    # Optional: skip shareholder data
```

---

## 📊 Data Schema

### Input (MySQL)

| Column | Type | Example |
|--------|------|---------|
| ticker | VARCHAR | BBCA |
| report_year | INT | 2023 |
| metric | VARCHAR | Net Profit |
| value | DECIMAL | 45000000 |

### Output (Qdrant)

```json
{
  "id": 1,
  "vector": {
    "dense": [0.123, ...],  // 1024-dim
    "sparse": {"indices": [...], "values": [...]}
  },
  "payload": {
    "ticker": "BBCA",
    "report_year": 2023,
    "metric": "Net Profit",
    "value": 45000000,
    "text": "In 2023, the stock BBCA reported a Net Profit of 45,000,000.",
    "source": "idnfinancials.com"
  }
}
```

---

## 🔍 Verify Data

1. Buka Qdrant Dashboard: http://localhost:6333/dashboard
2. Check collection `financial_data`
3. Verify sample points memiliki dense dan sparse vectors

---

## 📁 Project Structure

```
AI_v2/
├── mysql_to_qdrant_etl.py   # Main ETL script
├── test_retrieval.py        # Test hybrid search
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
├── .env                     # Your config (gitignored)
├── Dockerfile               # Production container (ROCm)
└── docker-compose.yml       # Orchestration
```

---

## 🛠️ Troubleshooting

### Error: No module named 'qdrant_client'
```bash
pip3 install qdrant-client
```

### Error: libxml2/libxslt not found
```bash
sudo dnf install -y libxml2-devel libxslt-devel  # Fedora
sudo apt install -y libxml2-dev libxslt-dev      # Ubuntu
```

### GPU not detected
- Pastikan ROCm terinstall: `rocm-smi`
- Untuk Docker, pastikan `--device /dev/kfd --device /dev/dri`

---

## 📝 License

Data source: [idnfinancials.com](https://idnfinancials.com)
