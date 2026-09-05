#!/usr/bin/env bash
echo "===================================================="
echo "Starting VibeVoice Realtime Container via Docker..."
echo "===================================================="

docker compose up --build -d

echo ""
echo "VibeVoice service is starting up!"
echo "Web UI will be available at: http://localhost:3001"
echo ""
echo "To view logs:      docker compose logs -f"
echo "To stop service:   docker compose down"
