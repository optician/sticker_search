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

- **Embedder** (`embedder` + `core`/`infra`). Offline batch over captions via a
  local text-embedding model on Ollama (`bge-m3`, multilingual, 1024-dim,
  cosine). Reads the captions of one `(caption_model, prompt_version)` set,
  composes `Caption::embed_text()` (scene + verbatim OCR + tone + situations),
  and stores vectors in Qdrant — **one collection per
  `(caption_model, prompt_version, embed_model)`** (`collection_name`), point id
  = sticker UUID, provenance in the payload. New `EmbeddingGateway` +
  `VectorStore` + `CaptionReader` ports; `OllamaEmbeddingGateway` +
  `QdrantVectorStore` (REST, no gRPC) adapters. Idempotent skip checked before
  the model call; `--force` re-embeds; dimension mismatch is a per-sticker
  failure. Invariant: query text and stored captions must go through the *same*
  embedder. Caption-only for now; hybrid CLIP image vectors deferred until
  caption-only search shows concrete misses.

## Remaining

### Query
- **Target:** text query → ranked stickers.
- **Solution:** embed the query with the *same* `EmbeddingGateway`, search the
  active collection via Qdrant (`VectorStore::search`, to be added), return point
  ids (sticker UUIDs) → resolve images from `meta.sqlite`. Storage already lives
  in Qdrant; this stage is the read path against it.

### Bot (live interface)
- **Target:** users query and receive stickers in Telegram.
- **Solution:** `teloxide` bot wrapping the query path. Composition root only —
  the search logic stays in `core`.

## Open questions

- Is `embed_text()`'s field composition (scene + OCR + tone + situations) the
  best document for retrieval? Revisit once the query stage lets us judge real
  searches.

## Resolved / notes

- **Caption embeddings alone, or hybrid with CLIP image vectors?** Caption-only
  for now: one model, simplest path, and the OCR-rich captions already carry the
  meme text CLIP would miss. Add CLIP image vectors only if caption-only search
  shows concrete "the image shows X but the caption never said X" misses. (This
  also moots image resolution for embeddings — the embedder reads text, not the
  thumbnail.)

- **Local VLM strong enough on memes/OCR?** Yes for `qwen3-vl:32b` — verbatim
  Cyrillic OCR incl. deliberately-misspelled text, useful scene/tone/situations.
  No cloud fallback needed for captioning.
- **Don't ask the VLM to identify people/brands.** It hallucinates confidently
  (named Oleg Tinkov as "Ilya Mikhaylov" at 80% confidence). Keep captions
  descriptive; if identity search is ever needed, add deterministic manual tags.
- **Model swaps** are a one-liner (`--model`) and coexist in the `captions` table.
  Watch **Gemma 4 12B** (encoder-free, multilingual OCR) once Ollama ships its
  vision support — currently Ollama's Gemma 4 build is text-only.
