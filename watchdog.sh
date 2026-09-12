#!/bin/bash
# Music DNA Agent Watchdog: Ensures aiogram bot and pipeline worker stay running

# Ensure permissions
chmod 777 /opt/hermes-data/music-dna/data 2>/dev/null || true

# Check if supervisor is running inside sandbox
if ! docker exec hermes-music-dna-sandbox pgrep -f "src/supervisor.py" > /dev/null 2>&1; then
    echo "[Sat Sep 12 21:46:52 +03 2026] Supervisor not running. Starting Music DNA supervisor..."
    docker exec -d hermes-music-dna-sandbox bash -c "export PYTHONPATH=/workspace/workspace; cd /workspace/workspace/music_dna_agent; python3 src/supervisor.py"
fi
