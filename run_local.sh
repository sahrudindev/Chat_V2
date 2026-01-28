#!/bin/bash

# run_local.sh
# Script untuk menjalankan IDN Financials AI Chatbot secara lokal (Manual Mode)
# Menjalankan:
# 1. Qdrant (via Docker)
# 2. AI Service (Python :8001)
# 3. Web Chatbot (Python :8080)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== IDN Financials AI Chatbot Launcher ===${NC}"

# 1. Check/Start Qdrant
echo -e "\n${BLUE}[1/3] Checking Qdrant...${NC}"
if docker ps | grep -q rag-qdrant; then
    echo -e "${GREEN}✓ Qdrant is running.${NC}"
else
    echo "Starting Qdrant via Docker..."
    docker compose up -d qdrant
    echo "Waiting for Qdrant to be ready..."
    sleep 5
fi

# Function to kill background processes on exit
cleanup() {
    echo -e "\n\n${RED}Shutting down services...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

# 2. Start AI Service
echo -e "\n${BLUE}[2/3] Starting AI Service (Port 8001)...${NC}"
cd ai-service
# Check if venv exists (optional, assuming user has environment set up or uses system python)
# Using python3 directly as per previous manual steps
python3 main.py > ../ai-service.log 2>&1 &
AI_PID=$!
echo "AI Service PID: $AI_PID (Logs: ai-service.log)"
cd ..

# Wait for AI Service to be ready
echo "Waiting for AI Service..."
until curl -s http://localhost:8001/health > /dev/null; do
    sleep 1
    echo -n "."
done
echo -e "\n${GREEN}✓ AI Service is ready!${NC}"

# 3. Start Web Chatbot
echo -e "\n${BLUE}[3/3] Starting Web Chatbot (Port 8080)...${NC}"
cd web-chatbot
# Using uvicorn with reload for development
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload &
WEB_PID=$!
echo "Web Chatbot PID: $WEB_PID"
cd ..

echo -e "\n${GREEN}=== ALL SERVICES STARTED ===${NC}"
echo -e "Web Interface: ${BLUE}http://localhost:8080${NC}"
echo -e "AI Service:    http://localhost:8001"
echo -e "Qdrant:        http://localhost:6333/dashboard"
echo -e "\nPress ${RED}Ctrl+C${NC} to stop all services."

# Keep script running
wait
