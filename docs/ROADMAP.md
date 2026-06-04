# Roadmap

Pipeline: **scrape → caption → embed → search**. The offline stages build the
index as a batch; the **bot** is the live interface (inline search + queueing
new packs). Each sticker is keyed by its UUID across all stages.

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

- **Query** (`search` use-case in `core` + `vsearch` server in `captioner`). Text
  query → ranked stickers. `SearchStickers` embeds the query with the *same*
  `EmbeddingGateway` the captions used, searches the matching collection
  (`VectorStore::search`, top-k + optional cosine `score_threshold`), and resolves
  each ranked UUID back to its sticker + caption via two narrow read ports
  (`StickerRepository::find_sticker_by_id`, new `CaptionLookup`). A hit whose
  sticker/caption row is missing is skipped, not fatal. Driven by
  `captioner vsearch`: a `tiny_http` server (sync loop bridged to the async query
  path via `spawn_blocking` + `Handle::block_on`) serving a search box → ranked
  result grid with thumbnails, scores, captions, and per-query latency. Query text
  is percent-decoded so Cyrillic searches work. See `README.md`.

- **Bot** (`bot` crate, `teloxide` 0.17). The live interface, composition root
  only — search/status logic stays in `core`. Two surfaces:
  - **Inline search.** `@bot <query>` in any chat → ranked stickers as you type,
    via the existing `SearchStickers` query path. Hits become
    `InlineQueryResultCachedSticker`s keyed by each sticker's `file_id`. Requires
    inline mode enabled in @BotFather, and the bot **must run under the same token
    that scraped the packs** (`file_id` is per-bot).
  - **`/add <link|name>`** (or sending a sticker — its `set_name` is read from the
    update, no API call). Verifies the pack exists, then queues it in a new
    `pack_requests` table (new `PackRequests` port). Open to any user. Re-running
    `/add` (or `/add` on a known pack) reports the pack's **derived** stage —
    `queued → scraped → captioned → ready` plus counts — computed on demand by the
    `PackStatus` use-case from the pipeline's own data (stickers, captions,
    vectors), so the offline batch never writes progress back. The scraper drains
    the queue with `--from-queue`. `normalize_pack_name` moved into `core` and is
    shared by scraper and bot. `SqliteRepository` now guards its connection with a
    `Mutex` so it's `Sync` for the async multi-threaded dispatcher.

## Open questions

- Is `embed_text()`'s field composition (scene + OCR + tone + situations) the
  best document for retrieval? Now judgeable: `captioner vsearch` shows each hit's
  score next to the caption that produced it. Run real queries and look for misses
  the field composition causes before tweaking it.

## Resolved / notes

- **Can the bot import a user's installed packs automatically?** No. The Bot API
  has no method to enumerate a user's sticker sets — only an MTProto *userbot*
  (the user's own account) can, via `messages.getAllStickers`. So packs enter the
  index one of two ways: list links in `packs.txt`, or `/add` them (by link, name,
  or by sending a sticker the bot reads `set_name` from). A one-off userbot script
  could bulk-export, but that's out of scope for the bot token.
- **Pack-add is queue-only, status is derived.** The bot deliberately does *not*
  run the VLM/embedding pipeline live (a 50-sticker pack is minutes on the local
  models). It only records the request; the operator runs `scrapper --from-queue`
  then the usual caption/embed batch. Status counts come free from the same data,
  so `/add` shows "captioned 12/50" without any progress bookkeeping.

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
