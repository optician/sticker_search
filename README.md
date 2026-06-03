# sticker-search

Search Telegram stickers by text. Pipeline stages: scrape packs → caption →
embed → vector search. The **scraper** and **captioner** exist so far.

## Running the scraper

Downloads each pack's static thumbnail + metadata into `stickers/` (images per
pack) and `stickers/meta.sqlite`.

### Prerequisites

- Rust (edition 2024 toolchain).
- A bot token from [@BotFather](https://t.me/BotFather).

### Usage

```bash
export TELEGRAM_BOT_TOKEN=<your bot token>

# accepts bare ids or share links, interchangeably:
#   crazy_klutzy
#   https://t.me/addstickers/crazy_klutzy
cargo run -p scrapper -- <pack_or_link> [<pack_or_link> ...]
```

Or list packs in a file (one name per line; `#` comments and blank lines are
ignored) — merged with any names passed as arguments:

```bash
cat > packs.txt <<'EOF'
# my packs
some_pack_name
another_pack
EOF

cargo run -p scrapper
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--packs-file <path>` | `packs.txt` | File of pack names, one per line. |
| `--out <dir>` | `stickers` | Output directory for images + `meta.sqlite`. |

Re-runs are safe and resumable: existing stickers (keyed by `file_unique_id`)
keep their UUID and already-downloaded images are skipped. On completion it
prints a summary:

```
done: packs 2 ok / 0 failed | stickers 48 downloaded, 0 skipped, 0 failed
```

## Running the captioner

Captions each scraped thumbnail with a local vision model (via
[Ollama](https://ollama.com)) and writes structured, retrieval-oriented text
into the existing `stickers/meta.sqlite`: a literal `scene`, verbatim
`on_image_text` (OCR), emotional `tone`, and the `situations` it's sent in.

### Prerequisites

- A running Ollama with a vision model pulled, e.g.:

  ```bash
  ollama pull qwen3-vl:32b
  ```

- A populated `stickers/meta.sqlite` (run the scraper first).

### Running a batch

The captioner is one binary with subcommands. The `run` subcommand does the
captioning; bare `cargo run -p captioner` (no subcommand) is shorthand for
`run` with defaults — caption every scraped sticker with `qwen3-vl:32b`:

```bash
# caption everything, default model
cargo run -p captioner

# scope to packs and cap the run (flags go after `run`)
cargo run -p captioner -- run --pack crazy_klutzy --limit 2
```

Re-runs are safe and resumable: a sticker already captioned by the same
`(model, prompt_version)` is skipped. Use `--force` to re-caption it.

While it runs it shows a live progress line — the in-flight sticker (each takes
~15s on the model), updated in place with the outcome and an `[n/total]` counter:

```
[ 12/48] mne_pochuj/8f3c….webp ✓
```

On completion it prints a summary:

```
done: captioned 48, skipped 0, failed 0 (model qwen3-vl:32b, prompt v1)
```

`run` options:

| Flag | Default | Description |
|------|---------|-------------|
| `--pack <name>` | _all packs_ | Restrict to a pack; repeatable. |
| `--model <tag>` | `qwen3-vl:32b` | Ollama model tag. |
| `--images-dir <dir>` | `stickers` | Root holding the per-pack thumbnails. |
| `--force` | off | Re-caption stickers already done for this model + prompt. |
| `--limit <n>` | _none_ | Stop after N stickers per pack. |
| `--ollama-url <url>` | `$OLLAMA_HOST` or `http://localhost:11434` | Ollama base URL. |

`--db <path>` (default `stickers/meta.sqlite`) is global — it works with every
subcommand, before or after the subcommand name.

### Changing the model

The model is a runtime flag — no rebuild:

```bash
cargo run -p captioner -- run --model qwen3-vl:8b
```

It must be pulled in Ollama first (`ollama pull <tag>`). Captions are keyed by
model, so a second model's output is stored **alongside** the first rather than
replacing it — caption with both, then compare a sticker's rows:

```bash
cargo run -p captioner -- show <sticker-uuid>
```

To change the default tag, edit the `--model` default in
`captioner/src/main.rs`.

### Changing the prompt

The prompt lives as two consts in `captioner/src/main.rs`:

```rust
const PROMPT_VERSION: &str = "v1";
const PROMPT_TEXT: &str = "You are captioning a Telegram sticker ...";
```

When you edit `PROMPT_TEXT`, **bump `PROMPT_VERSION`** (e.g. `"v2"`). Captions
are keyed by `(sticker_id, model, prompt_version)`, and the `prompts` table pins
each version to exactly one text — so re-running with an edited prompt under the
old version is rejected at startup:

```
error: prompt version "v1" already exists with different text; bump the version when editing the prompt
```

After bumping, just re-run: the new version's rows are written next to the old
ones (no `--force` needed — the new key doesn't exist yet), so you can diff
prompt revisions the same way you diff models.

### Inspecting captions

Captions are keyed by `(sticker_id, model, prompt_version)`. Inspect them
through subcommands — no SQL needed:

```bash
# counts per model + prompt version
cargo run -p captioner -- stats

# list captions (filterable: --pack, --model, --prompt-version, --limit)
cargo run -p captioner -- list --pack crazy_klutzy --limit 5

# search the scene and on-image text
cargo run -p captioner -- search chicken

# every caption for one sticker (all models / prompt versions)
cargo run -p captioner -- show <sticker-uuid>

# the registered prompt versions and their text
cargo run -p captioner -- prompts
```

`stats` prints a count table; `list`/`search`/`show` print each caption with its
pack, sticker UUID, model/version, scene, on-image text, tone, and situations.
The UUIDs shown by `list`/`search` are what you pass to `show`.

### Viewing captions next to the images

To judge caption quality you need to see the thumbnail too. `gallery` writes a
self-contained HTML page (thumbnails + caption fields) you open in a browser —
the best way to eyeball results, especially OCR on memes:

```bash
# all captions → captions.html, then open it
cargo run -p captioner -- gallery
open captions.html            # macOS; or just open the file in any browser

# same filters as `list`, plus --out and --images-dir
cargo run -p captioner -- gallery --pack mne_pochuj --out ru.html
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pack` / `--model` / `--prompt-version` / `--limit` | _none_ | Same filters as `list`. |
| `--images-dir <dir>` | `stickers` | Root used to build the `<img>` paths. |
| `--out <file>` | `captions.html` | Output HTML file. |

The `<img>` paths are written relative to where you run the command, so open the
generated file from the project root (next to `stickers/`).

### Review server

For interactive review — re-filtering without regenerating a file — `serve`
runs a local web UI with a **pack** dropdown and a **date sort** (freshest
first by default):

```bash
cargo run -p captioner -- serve            # http://localhost:8080
cargo run -p captioner -- serve --port 9000
```

Then open `http://localhost:8080/` and use the dropdowns; the page reloads with
the new filter/sort. The server reads the database live, so re-running a caption
batch and refreshing the page shows the new captions. It binds to localhost only
and serves thumbnails from `--images-dir` (default `stickers`).

| Flag | Default | Description |
|------|---------|-------------|
| `--port <n>` | `8080` | Port to listen on. |
| `--images-dir <dir>` | `stickers` | Root for the served thumbnails. |

## Tests

```bash
cargo nextest run --workspace
```
