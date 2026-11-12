#!/usr/bin/env bash
# Convert the merged model (scripts/merge_adapter.py) to GGUF with the official llama.cpp image:
# language model in f16, Q8_0 and Q4_K_M, vision projector (mmproj) in f16.
set -euo pipefail

IMAGE=${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:full-b12282}
MODELS=$(realpath "${1:-models}")
NAME=qwen3-vl-2b-receipts
RUN=(docker run --rm -u "$(id -u):$(id -g)" -v "$MODELS:/models" --entrypoint)

mkdir -p "$MODELS/gguf"
"${RUN[@]}" python3 "$IMAGE" /app/convert_hf_to_gguf.py /models/merged --model-name "$NAME" \
    --outtype f16 --outfile "/models/gguf/$NAME-f16.gguf"
"${RUN[@]}" python3 "$IMAGE" /app/convert_hf_to_gguf.py /models/merged --model-name "$NAME" \
    --mmproj --outtype f16 --outfile "/models/gguf/mmproj-$NAME-f16.gguf"
for quant in Q8_0 Q4_K_M; do
    "${RUN[@]}" /app/llama-quantize "$IMAGE" \
        "/models/gguf/$NAME-f16.gguf" "/models/gguf/$NAME-$quant.gguf" "$quant"
done
ls -la "$MODELS/gguf"
