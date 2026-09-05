FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (ffmpeg and libsndfile for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md /app/
COPY vibevoice /app/vibevoice
COPY demo /app/demo
COPY vllm_plugin /app/vllm_plugin

# Install VibeVoice and Python dependencies
RUN pip install --no-cache-dir -e .[streamingtts]

# Expose Web UI port
EXPOSE 3000

# Default command to run real-time TTS web service
CMD ["python", "demo/vibevoice_realtime_demo.py", "--port", "3000", "--model_path", "microsoft/VibeVoice-Realtime-0.5B"]
