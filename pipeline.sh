#!/usr/bin/env bash
#
# Full-cycle indexing job: drain the bot's pack queue, then scrape -> caption ->
# embed, once. Makes every queued pack searchable and exits. Runs linearly with
# no looping or polling; any stage failing aborts the whole job (non-zero exit).
#
# Ollama model lifecycle: captioning loads a VLM (qwen3-vl:8b) and
# embedding loads a separate model (bge-m3). To stop them competing for VRAM,
# each model is unloaded from Ollama before the next stage's model is loaded.
#
# Config via env (defaults match the rest of the project):
#   TELEGRAM_BOT_TOKEN  (required) the same bot that scraped the packs
#   CAPTION_MODEL       default qwen3-vl:8b
#   EMBED_MODEL         default bge-m3
#   OLLAMA_HOST         default http://localhost:11434
#   QDRANT_URL          default http://localhost:6333
#
set -euo pipefail

# Run from the repo root so the default relative paths (stickers/meta.sqlite)
# resolve regardless of the caller's working directory.
cd "$(dirname "$0")"

CAPTION_MODEL="${CAPTION_MODEL:-qwen3-vl:8b}"
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

: "${TELEGRAM_BOT_TOKEN:?set TELEGRAM_BOT_TOKEN (the bot that scraped the packs)}"

log() { printf '\n=== %s ===\n' "$1"; }

# Unload a model from Ollama so the next stage's model has room. Prefer the CLI
# `ollama stop` (works for chat and embed models alike); fall back to the
# keep_alive:0 API. Never fatal — a failed unload only costs VRAM.
unload() {
  local model="$1"
  if command -v ollama >/dev/null 2>&1 && ollama stop "$model" >/dev/null 2>&1; then
    echo "unloaded $model"
  elif curl -fsS "$OLLAMA_URL/api/generate" \
        -d "{\"model\":\"$model\",\"keep_alive\":0}" >/dev/null 2>&1; then
    echo "unloaded $model (api)"
  else
    echo "warning: could not unload $model (continuing)"
  fi
}

# Build once up front so a compile error aborts before any partial work.
log "build"
cargo build --release -p scrapper -p captioner -p embedder

log "scrape — draining the bot queue"
./target/release/scrapper --from-queue

log "caption — $CAPTION_MODEL"
./target/release/captioner run --model "$CAPTION_MODEL" --ollama-url "$OLLAMA_URL"
unload "$CAPTION_MODEL"

log "embed — $EMBED_MODEL"
./target/release/embedder \
  --embed-model "$EMBED_MODEL" --ollama-url "$OLLAMA_URL" --qdrant-url "$QDRANT_URL"
unload "$EMBED_MODEL"

log "done — queued packs are now searchable"
