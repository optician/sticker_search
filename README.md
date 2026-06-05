# sticker-search

Search Telegram stickers by what they *mean*, not their filename. A local vision
model captions each sticker, a text-embedding model turns the captions into
vectors, and a Telegram bot answers inline queries against them.

Pipeline: **scrape** packs → **caption** (VLM) → **embed** → **search**. Search
is exposed as an inline Telegram bot.

For per-stage flags, subcommands, and tuning, see [`docs/guide.md`](docs/guide.md).

## Install

- **Rust** (edition 2024 toolchain).
- **[Ollama](https://ollama.com)** with a vision model and an embedding model:
  ```bash
  ollama pull qwen3-vl:8b   # captioning
  ollama pull bge-m3        # embedding (multilingual, incl. Cyrillic)
  ```
- **[Qdrant](https://qdrant.tech)** (vector store):
  ```bash
  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
  ```
- A **Telegram bot token** from [@BotFather](https://t.me/BotFather).

## Run and use the bot

One-time setup in [@BotFather](https://t.me/BotFather):

- **Enable inline mode** (`/setinline`, with a placeholder like `search
  stickers…`). Without it Telegram never sends inline queries.
- **Use one bot for everything.** Inline results reference each sticker's
  per-bot `file_id`, so the bot that *serves* search must be the same one that
  *scraped* the packs.

Start it (Ollama and Qdrant must be running, with an index already built — see
below):

```bash
export TELEGRAM_BOT_TOKEN=<your bot token>
cargo run -p bot
```

- **Search:** in any chat, type `@your_bot some query`. Ranked stickers appear as
  you type. Hits below a cosine score of `0.44` are dropped as noise (tune with
  `--min-score`).
- **Add a pack:** `/add <link or name>`, or forward the bot a sticker from the
  pack. This only **queues** the pack; build the index with the pipeline below.

## Run the pipeline

`/add` queues packs but doesn't index them. Drain the queue and run
scrape → caption → embed in one shot (each Ollama model is unloaded before the
next stage loads, so the VLM and embedder don't fight for VRAM):

```bash
export TELEGRAM_BOT_TOKEN=<the same bot token>
./pipeline.sh
```

Models and endpoints are overridable via env: `CAPTION_MODEL`, `EMBED_MODEL`,
`OLLAMA_HOST`, `QDRANT_URL`.

To bootstrap a fresh index from specific packs (instead of the bot queue), or to
run stages by hand:

```bash
cargo run -p scrapper -- <pack_or_link> [<pack_or_link> ...]   # download
cargo run -p captioner                                          # caption (VLM)
cargo run -p embedder                                           # embed → Qdrant
```

Re-runs are resumable: already-scraped, already-captioned, and already-embedded
stickers are skipped.

## Known issues

- **No animation support.** Animated and video stickers are captioned from their
  static thumbnail only — motion, and any text that appears mid-animation, is
  lost.
- **Limited context.** The VLM sees a 512px thumbnail in isolation, with no
  surrounding conversation. Abstract stickers, in-jokes, and culture-specific
  references often caption poorly, and search quality is only as good as the
  caption. Captioning each sticker takes a few seconds of model time.

## Tests

```bash
cargo nextest run --workspace
```
