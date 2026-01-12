# RAG Chatbot - Indonesian Stock Market

Web chatbot menggunakan Retrieval-Augmented Generation dengan Qdrant dan AI Service.

## Arsitektur

```
┌─────────────────────────────────────┐
│           DOCKER                    │
│  ┌──────────┐  ┌──────────────┐    │
│  │  Web UI  │  │   Qdrant     │    │
│  │  :8000   │  │   :6333      │    │
│  └────┬─────┘  └──────────────┘    │
└───────┼─────────────────────────────┘
        │ HTTP
        ▼
┌─────────────────────────────────────┐
│        LOCAL (ROCm GPU)             │
│  ┌──────────────────────────────┐  │
│  │  AI Service :8001            │  │
│  │  - /embed (BGE-M3)           │  │
│  │  - /chat (Ollama/Qwen)       │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Quick Start

### 1. Start AI Service (Local)
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/ai-service
pip install -r requirements.txt
python main.py
# Running on http://localhost:8001
```

### 2. Start Ollama + Qwen
```bash
ollama serve &
ollama pull qwen3:4b
```

### 3. Start Web + Qdrant (Docker)
```bash
cd /home/fiqri/Desktop/IDN/AI_v2/web-chatbot
docker compose up -d --build
```

### 4. Open Chatbot
http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat Web UI |
| `/chat` | POST | Send message |
| `/health` | GET | Health check |
| `/search-debug` | POST | Debug search |
