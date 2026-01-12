# AI Service

Local AI service for embedding and chat with ROCm GPU support.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate BGE-M3 embeddings |
| `/chat` | POST | Chat with Qwen via Ollama |
| `/health` | GET | Health check |

## Run

```bash
cd /home/fiqri/Desktop/IDN/AI_v2/ai-service
pip install -r requirements.txt
python main.py
```

Server runs on http://localhost:8001
