# sticker-search

Search Telegram stickers by text. Pipeline stages: scrape packs → caption →
embed → vector search. Only the **scraper** exists so far.

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

### Tests

```bash
cargo nextest run --workspace
```
