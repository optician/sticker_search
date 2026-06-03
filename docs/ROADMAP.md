# Roadmap

Pipeline: **scrape → caption → embed → search**. Stages run offline as a batch
to build the index; only the query path is live. Each sticker is keyed by its
UUID across all stages.

## Done

- **Scraper** (`scrapper` + `core`/`infra`). Bot API → static thumbnail + metadata
  → `stickers/<pack>/<uuid>.<ext>` and `stickers/meta.sqlite`. Idempotent on
  `file_unique_id`. See `README.md`.

- **Captioner** (`captioner` + `core`/`infra`). Offline batch over thumbnails via a
  local VLM on Ollama (`qwen3-vl:32b`, validated incl. verbatim Cyrillic OCR).
  Structured JSON → `scene`, `on_image_text` (OCR), `tone`, `situations`. New
  `CaptionGateway` port + `OllamaCaptionGateway` adapter; `captions`/`prompts`
  tables in the same SQLite store. Captions keyed by `(sticker_id, model,
  prompt_version)` so models/prompts coexist for comparison; idempotent skip
  checked *before* the model call; live per-sticker progress. Subcommands:
  `run`, `stats`, `list`, `search`, `show`, `gallery` (static HTML),
  `serve` (review server: filter by pack, sort by date), `prompts`. See `README.md`.

## Remaining

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

- Thumbnail quality sufficient for embeddings, or render originals? (Captioning
  works well at thumbnail res — OCR reads stylized Cyrillic on 128–320px images.)
- Caption embeddings alone, or hybrid with CLIP image vectors?

## Resolved / notes

- **Local VLM strong enough on memes/OCR?** Yes for `qwen3-vl:32b` — verbatim
  Cyrillic OCR incl. deliberately-misspelled text, useful scene/tone/situations.
  No cloud fallback needed for captioning.
- **Don't ask the VLM to identify people/brands.** It hallucinates confidently
  (named Oleg Tinkov as "Ilya Mikhaylov" at 80% confidence). Keep captions
  descriptive; if identity search is ever needed, add deterministic manual tags.
- **Model swaps** are a one-liner (`--model`) and coexist in the `captions` table.
  Watch **Gemma 4 12B** (encoder-free, multilingual OCR) once Ollama ships its
  vision support — currently Ollama's Gemma 4 build is text-only.
