#!/usr/bin/env bash
set -e

echo "===================================================================="
echo "      VibeVoice + SadTalker Talking-Head Avatar Pipeline"
echo "===================================================================="
echo ""

# 1. Verify Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker is not running or not accessible."
    echo "Please ensure Docker daemon is running with NVIDIA GPU support."
    exit 1
fi

# 2. Parameters
TEXT="${1:-}"
PORTRAIT_PATH="${2:-inputs/portrait.png}"
SPEAKER="${3:-en-Carter_man}"

mkdir -p inputs outputs checkpoints/sadtalker

if [ ! -f "$PORTRAIT_PATH" ]; then
    echo "[WARNING] Portrait image '$PORTRAIT_PATH' not found!"
    echo "Please place a front-facing portrait photo in: inputs/portrait.png"
    read -rp "Enter image path: " USER_IMAGE
    if [ -n "$USER_IMAGE" ]; then
        PORTRAIT_PATH="$USER_IMAGE"
    fi
fi

if [ ! -f "$PORTRAIT_PATH" ]; then
    echo "[ERROR] Image file not found: $PORTRAIT_PATH. Aborting."
    exit 1
fi

PORTRAIT_FILE=$(basename "$PORTRAIT_PATH")
if [ ! -f "inputs/$PORTRAIT_FILE" ]; then
    echo "Copying $PORTRAIT_PATH to inputs/$PORTRAIT_FILE..."
    cp "$PORTRAIT_PATH" "inputs/$PORTRAIT_FILE"
fi

if [ -z "$TEXT" ]; then
    read -rp "Enter text for avatar to say [Default: Hello! Welcome to the presentation.]: " USER_TEXT
    TEXT="${USER_TEXT:-Hello! Welcome to the presentation.}"
fi

echo ""
echo "--------------------------------------------------------------------"
echo "[Configuration]"
echo "Text:     \"$TEXT\""
echo "Portrait: \"inputs/$PORTRAIT_FILE\""
echo "Speaker:  \"$SPEAKER\""
echo "--------------------------------------------------------------------"
echo ""

# 3. Stage 1: Audio source (custom audio or VibeVoice generation)
if [[ -f "$TEXT" && ("$TEXT" == *.wav || "$TEXT" == *.mp3 || "$TEXT" == *.m4a || "$TEXT" == *.ogg || "$TEXT" == *.flac) ]]; then
    echo "[Stage 1/2] Using provided audio file: $TEXT"
    cp "$TEXT" "outputs/speech.wav"
    echo "[Stage 1/2 Complete] Custom audio copied to outputs/speech.wav"
else
    echo "[Stage 1/2] Generating speech audio with VibeVoice..."
    docker compose run --rm vibevoice python demo/generate_speech_cli.py \
        --text "$TEXT" \
        --speaker_name "$SPEAKER" \
        --output_path /app/outputs/speech.wav
fi

if [ ! -f "outputs/speech.wav" ]; then
    echo "[ERROR] outputs/speech.wav was not created."
    exit 1
fi
echo "[Stage 1/2 Complete] Audio ready at outputs/speech.wav"
echo ""

# 4. Stage 2: Generate Video with SadTalker
echo "[Stage 2/2] Generating talking avatar video with SadTalker..."
docker run --gpus all --rm \
    -v "$(pwd)/inputs:/host_dir/inputs" \
    -v "$(pwd)/outputs:/host_dir/outputs" \
    wawa9000/sadtalker \
    --driven_audio /host_dir/outputs/speech.wav \
    --source_image "/host_dir/inputs/$PORTRAIT_FILE" \
    --enhancer gfpgan \
    --result_dir /host_dir/outputs

echo ""
echo "===================================================================="
echo "[SUCCESS] Talking avatar generation completed!"
echo "Check output files in: outputs/"
echo "===================================================================="
