# Roadmap

Pipeline: **scrape → caption → embed → search**. Stages run offline as a batch
to build the index; only the query path is live. Each sticker is keyed by its
UUID across all stages.

## Done

- **Scraper** (`scrapper` + `core`/`infra`). Bot API → static thumbnail + metadata
  → `stickers/<pack>/<uuid>.<ext>` and `stickers/meta.sqlite`. Idempotent on
  `file_unique_id`. See `README.md`.

## Remaining

### Captioner
- **Target:** a retrieval-optimized text per sticker — literal scene, verbatim
  on-image text (OCR), emotional tone, and the situations it's sent in.
- **Solution:** offline batch over the thumbnails via a local VLM (Ollama HTTP;
  `qwen2.5-vl` for OCR). Structured JSON prompt; persist a `captions` row keyed by
  sticker UUID. New `core` port + `infra` adapter; reuse the SQLite store.

### Embedder
- **Target:** vectors that make query text and stickers comparable.
- **Solution:** text-embedding model over captions (ONNX via `ort`, or Ollama
  `nomic-embed`). Optionally also CLIP/SigLIP image vectors (`ort`) for hybrid
  scoring. Invariant: query text and stored captions go through the *same*
  embedder. Store vectors keyed by sticker UUID.

### Index + query
- **Target:** text query → ranked stickers.
- **Solution:** at sticker scale, brute-force cosine in Rust (load vectors, sort);
  swap to Qdrant only if it outgrows memory. Query path: embed query → search →
  return UUIDs → resolve images from `meta.sqlite`.

### Bot (live interface)
- **Target:** users query and receive stickers in Telegram.
- **Solution:** `teloxide` bot wrapping the query path. Composition root only —
  the search logic stays in `core`.

## Open questions

- Thumbnail (~320px) quality sufficient for embeddings, or render originals?
- Local VLM strong enough on memes/OCR, or fall back to a cloud VLM for captioning?
- Caption embeddings alone, or hybrid with CLIP image vectors?
