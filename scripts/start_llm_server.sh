#!/usr/bin/env bash
# Start llama.cpp server for MockVehicle2D NL→JSON inference.
#
# Usage:
#   bash scripts/start_llm_server.sh                  # defaults: GPU 0, Qwen3-8B
#   bash scripts/start_llm_server.sh 1                # GPU 1
#   bash scripts/start_llm_server.sh 0 Qwen3-14B-Q4_K_M  # GPU 0, 14B model
#
# Defaults (persisted server configuration):
#   GPU:          0 (single GPU, CUDA_VISIBLE_DEVICES)
#   Model:        Qwen3-8B-Q4_K_M
#   Port:         8000
#   GPU Layers:   36 (all layers offloaded)
#   Context:      2048

set -euo pipefail

GPU_ID="${1:-0}"
MODEL_NAME="${2:-Qwen3-8B-Q4_K_M}"
PORT="${3:-8000}"
N_GPU_LAYERS="${4:-36}"
N_CTX="${5:-2048}"

# Model base path (HuggingFace cache)
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
HF_CACHE="${HF_CACHE:-/vepfs-mlp2/c20250205/241905024/jmx/.cache/huggingface}"
MODEL_DIR="${HF_CACHE}/hub/models--Qwen--${MODEL_NAME%-*}-GGUF"
MODEL_FILE="${MODEL_DIR}/snapshots/"*"/${MODEL_NAME}.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model not found at $MODEL_FILE"
    echo "Download it first: hf download Qwen/${MODEL_NAME%-*}-GGUF ${MODEL_NAME}.gguf --local-dir /vepfs-mlp2/c20250205/241905024/jmx/models/"
    exit 1
fi

echo "=== MockVehicle2D LLM Server ==="
echo "GPU:          $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
echo "Model:        $MODEL_NAME"
echo "Port:         $PORT"
echo "GPU Layers:   $N_GPU_LAYERS"
echo "Context:      $N_CTX"
echo "Model path:   $MODEL_FILE"
echo "================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m llama_cpp.server \
    --model "$MODEL_FILE" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --n_ctx "$N_CTX"
