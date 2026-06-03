//! Composition root for the captioner. Subcommands:
//! - `run` (default): caption scraped thumbnails via Ollama.
//! - `stats` / `list` / `search` / `show` / `prompts`: inspect what's stored.

use clap::{Args, Parser, Subcommand};
use sticker_core::error::{CaptionError, CaptionGatewayError, RepoError};
use sticker_core::{CaptionProgress, CaptionRun, CaptionStickers, CaptionSummary, Prompt, ProgressEvent};
use sticker_infra::{
    CaptionFilter, CaptionSort, CaptionView, FsImageStore, OllamaCaptionGateway, SqliteRepository,
};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;
use time::OffsetDateTime;
use tiny_http::{Header, Response, Server};
use uuid::Uuid;

/// The captioning prompt. **Bump `PROMPT_VERSION` whenever this text changes** —
/// captions are keyed by `(sticker, model, prompt_version)`, and the store
/// rejects reusing a version with different text.
const PROMPT_VERSION: &str = "v1";
const PROMPT_TEXT: &str = "You are captioning a Telegram sticker for text search. \
Return JSON with keys: scene (literal description of what is shown), \
on_image_text (text shown in the image, copied verbatim, empty string if none), \
tone (the emotional tone), situations (an array of situations someone would send \
this sticker in). Be concise.";

/// Per-request timeout for the model. A cold 32B load can take ~70s.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

const DEFAULT_MODEL: &str = "qwen3-vl:32b";

/// Caption scraped sticker thumbnails with a local VLM (Ollama) and inspect the
/// results. Captions land in the existing meta.sqlite, keyed by
/// (sticker, model, prompt version).
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Metadata SQLite database (written by the scraper).
    #[arg(long, default_value = "stickers/meta.sqlite", global = true)]
    db: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Caption stickers (the default when no subcommand is given).
    Run(RunArgs),
    /// Show caption counts per model + prompt version.
    Stats,
    /// List captions, optionally filtered.
    List(ListArgs),
    /// Find captions by text in the scene or on-image text.
    Search(SearchArgs),
    /// Show every caption for one sticker (by UUID).
    Show(ShowArgs),
    /// Write an HTML page of thumbnails + captions to view in a browser.
    Gallery(GalleryArgs),
    /// Serve a browser UI for reviewing captions (filter by pack, sort by date).
    Serve(ServeArgs),
    /// List the registered prompt versions and their text.
    Prompts,
}

#[derive(Args, Debug)]
struct RunArgs {
    /// Restrict to these pack names (repeatable). Default: every pack.
    #[arg(long = "pack")]
    packs: Vec<String>,
    /// Ollama model tag.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,
    /// Root directory holding the per-pack thumbnail images.
    #[arg(long, default_value = "stickers")]
    images_dir: PathBuf,
    /// Re-caption stickers already captioned by this model + prompt version.
    #[arg(long)]
    force: bool,
    /// Stop after N stickers (per pack) — handy for cheap test runs.
    #[arg(long)]
    limit: Option<usize>,
    /// Ollama base URL. Falls back to $OLLAMA_HOST, then localhost.
    #[arg(long)]
    ollama_url: Option<String>,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            packs: Vec::new(),
            model: DEFAULT_MODEL.to_string(),
            images_dir: PathBuf::from("stickers"),
            force: false,
            limit: None,
            ollama_url: None,
        }
    }
}

#[derive(Args, Debug)]
struct ListArgs {
    #[arg(long)]
    pack: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long = "prompt-version")]
    prompt_version: Option<String>,
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Args, Debug)]
struct SearchArgs {
    /// Text to look for in the scene or on-image text.
    query: String,
    #[arg(long)]
    pack: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Args, Debug)]
struct ShowArgs {
    /// Sticker UUID (as printed by `list`/`search`).
    sticker_id: String,
}

#[derive(Args, Debug)]
struct GalleryArgs {
    #[arg(long)]
    pack: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long = "prompt-version")]
    prompt_version: Option<String>,
    #[arg(long)]
    limit: Option<usize>,
    /// Root holding the per-pack thumbnails (used to build <img> paths).
    #[arg(long, default_value = "stickers")]
    images_dir: PathBuf,
    /// Output HTML file.
    #[arg(long, default_value = "captions.html")]
    out: PathBuf,
}

#[derive(Args, Debug)]
struct ServeArgs {
    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Root holding the per-pack thumbnails (served under /images/).
    #[arg(long, default_value = "stickers")]
    images_dir: PathBuf,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("database error: {0}")]
    Repo(#[from] RepoError),
    #[error("building the Ollama gateway: {0}")]
    Gateway(#[from] CaptionGatewayError),
    #[error(transparent)]
    Run(#[from] CaptionError),
    #[error("not a valid sticker UUID: {0}")]
    BadStickerId(String),
    #[error("writing {path}: {source}")]
    WriteFile {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("starting the review server: {0}")]
    Serve(String),
    #[error("initializing tracing: {0}")]
    Tracing(String),
}

fn open_db(db: &Path) -> Result<SqliteRepository, AppError> {
    Ok(SqliteRepository::open(db)?)
}

async fn run() -> Result<(), AppError> {
    let Cli { db, command } = Cli::parse();
    match command.unwrap_or_else(|| Command::Run(RunArgs::default())) {
        Command::Run(args) => cmd_run(&db, args).await,
        Command::Stats => cmd_stats(&open_db(&db)?),
        Command::List(args) => cmd_list(&open_db(&db)?, args),
        Command::Search(args) => cmd_search(&open_db(&db)?, args),
        Command::Show(args) => cmd_show(&open_db(&db)?, args),
        Command::Gallery(args) => cmd_gallery(&open_db(&db)?, args),
        Command::Serve(args) => cmd_serve(&open_db(&db)?, args),
        Command::Prompts => cmd_prompts(&open_db(&db)?),
    }
}

/// Run the captioning batch.
async fn cmd_run(db: &Path, args: RunArgs) -> Result<(), AppError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .try_init()
        .map_err(|e| AppError::Tracing(e.to_string()))?;

    let url = args
        .ollama_url
        .clone()
        .or_else(|| std::env::var("OLLAMA_HOST").ok())
        .unwrap_or_else(|| "http://localhost:11434".to_string());

    let gateway = OllamaCaptionGateway::new(&url, &args.model, PROMPT_TEXT, REQUEST_TIMEOUT)?;
    let repo = open_db(db)?;
    let images = FsImageStore::new(&args.images_dir);

    let app = CaptionStickers::new(gateway, repo, images).on_progress(print_progress);
    let prompt = Prompt {
        version: PROMPT_VERSION.to_string(),
        text: PROMPT_TEXT.to_string(),
        created_at: OffsetDateTime::now_utc(),
    };

    tracing::info!(model = %args.model, url = %url, db = %db.display(), "starting captioner");

    let scopes: Vec<Option<&str>> = if args.packs.is_empty() {
        vec![None]
    } else {
        args.packs.iter().map(|p| Some(p.as_str())).collect()
    };

    let mut total = CaptionSummary::default();
    for pack in scopes {
        let cfg = CaptionRun { pack, force: args.force, limit: args.limit };
        let s = app.run(&prompt, cfg).await?;
        total.captioned += s.captioned;
        total.skipped += s.skipped;
        total.failed += s.failed;
    }

    println!(
        "done: captioned {}, skipped {}, failed {} (model {}, prompt {})",
        total.captioned, total.skipped, total.failed, args.model, PROMPT_VERSION,
    );
    Ok(())
}

/// Live one-line progress for a captioning run: shows the in-flight sticker
/// while the model works, then overwrites it with the outcome and moves on.
fn print_progress(p: CaptionProgress) {
    use std::io::Write;
    match p.event {
        ProgressEvent::Start => {
            print!("\r[{:>3}/{}] {} …", p.index, p.total, p.image_path);
            let _ = std::io::stdout().flush();
        }
        ProgressEvent::Captioned => {
            println!("\r[{:>3}/{}] {} ✓        ", p.index, p.total, p.image_path)
        }
        ProgressEvent::Skipped => {
            println!("\r[{:>3}/{}] {} – skipped ", p.index, p.total, p.image_path)
        }
        ProgressEvent::Failed => {
            println!("\r[{:>3}/{}] {} ✗ failed ", p.index, p.total, p.image_path)
        }
    }
}

fn cmd_stats(repo: &SqliteRepository) -> Result<(), AppError> {
    let stats = repo.caption_stats()?;
    if stats.is_empty() {
        println!("no captions yet — run `captioner` first");
        return Ok(());
    }
    println!("{:<18} {:<8} COUNT", "MODEL", "PROMPT");
    for s in &stats {
        println!("{:<18} {:<8} {}", s.model, s.prompt_version, s.count);
    }
    println!("total: {}", stats.iter().map(|s| s.count).sum::<u64>());
    Ok(())
}

fn cmd_list(repo: &SqliteRepository, args: ListArgs) -> Result<(), AppError> {
    let filter = CaptionFilter {
        pack: args.pack,
        model: args.model,
        prompt_version: args.prompt_version,
        limit: args.limit,
        ..Default::default()
    };
    print_views(&repo.query_captions(&filter)?);
    Ok(())
}

fn cmd_search(repo: &SqliteRepository, args: SearchArgs) -> Result<(), AppError> {
    let filter = CaptionFilter {
        text: Some(args.query),
        pack: args.pack,
        model: args.model,
        limit: args.limit,
        ..Default::default()
    };
    print_views(&repo.query_captions(&filter)?);
    Ok(())
}

fn cmd_show(repo: &SqliteRepository, args: ShowArgs) -> Result<(), AppError> {
    let id = Uuid::parse_str(&args.sticker_id)
        .map_err(|_| AppError::BadStickerId(args.sticker_id.clone()))?;
    let filter = CaptionFilter { sticker_id: Some(id), ..Default::default() };
    print_views(&repo.query_captions(&filter)?);
    Ok(())
}

fn cmd_gallery(repo: &SqliteRepository, args: GalleryArgs) -> Result<(), AppError> {
    let filter = CaptionFilter {
        pack: args.pack,
        model: args.model,
        prompt_version: args.prompt_version,
        limit: args.limit,
        ..Default::default()
    };
    let views = repo.query_captions(&filter)?;
    let html = render_gallery(&views, &args.images_dir);
    std::fs::write(&args.out, html)
        .map_err(|source| AppError::WriteFile { path: args.out.display().to_string(), source })?;
    println!(
        "wrote {} caption(s) to {} — open it in a browser",
        views.len(),
        args.out.display(),
    );
    Ok(())
}

/// Minimal HTML escaping for text placed in element bodies / quoted attributes.
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Static gallery file: no controls, image paths relative to `images_dir`.
fn render_gallery(views: &[CaptionView], images_dir: &Path) -> String {
    render_page(views, &images_dir.to_string_lossy(), "")
}

/// Shared page renderer. `img_prefix` is prepended to each stored image path
/// (`stickers` for the file gallery, `/images` for the server). `controls` is
/// optional HTML inserted above the grid.
fn render_page(views: &[CaptionView], img_prefix: &str, controls: &str) -> String {
    let mut h = String::from(
        "<!doctype html><meta charset=utf-8><title>sticker captions</title>\
         <style>\
         body{font-family:system-ui,sans-serif;margin:1rem;background:#111;color:#eee}\
         h1{font-size:1rem;font-weight:600}\
         .controls{margin:.5rem 0 1rem}\
         .controls select{background:#222;color:#eee;border:1px solid #444;\
           border-radius:5px;padding:.25rem}\
         .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem}\
         .card{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:.75rem}\
         .card img{width:128px;height:128px;object-fit:contain;background:#000;\
           border-radius:6px;display:block;margin:0 auto .5rem}\
         .meta{font-size:.7rem;color:#888;word-break:break-all}\
         .scene{margin:.4rem 0}.text{color:#7cf;font-weight:600}\
         .tone{color:#fc7}.uses{color:#9d9;font-size:.85rem;margin-top:.3rem}\
         </style>",
    );
    h.push_str(controls);
    h.push_str(&format!("<h1>{} caption(s)</h1><div class=grid>", views.len()));
    for v in views {
        let src = format!("{}/{}", img_prefix, v.image_path);
        h.push_str("<div class=card>");
        h.push_str(&format!("<img src=\"{}\" loading=lazy alt=\"\">", escape(&src)));
        h.push_str(&format!(
            "<div class=meta>{} · {} / {}</div>",
            escape(&v.pack),
            escape(&v.model),
            escape(&v.prompt_version),
        ));
        h.push_str(&format!("<div class=scene>{}</div>", escape(&v.scene)));
        if !v.on_image_text.is_empty() {
            h.push_str(&format!("<div class=text>“{}”</div>", escape(&v.on_image_text)));
        }
        h.push_str(&format!("<div class=tone>{}</div>", escape(&v.tone)));
        if !v.situations.is_empty() {
            h.push_str(&format!("<div class=uses>{}</div>", escape(&v.situations.join(" · "))));
        }
        h.push_str(&format!("<div class=meta>{}</div>", escape(&v.sticker_id.to_string())));
        h.push_str("</div>");
    }
    h.push_str("</div>");
    h
}

// ---- review server ----

fn cmd_serve(repo: &SqliteRepository, args: ServeArgs) -> Result<(), AppError> {
    let addr = format!("127.0.0.1:{}", args.port);
    let server = Server::http(&addr).map_err(|e| AppError::Serve(e.to_string()))?;
    println!("captioner review server on http://localhost:{}/  (Ctrl-C to stop)", args.port);
    for request in server.incoming_requests() {
        let response = build_response(repo, &args.images_dir, request.url());
        if let Err(e) = request.respond(response) {
            tracing::warn!(error = %e, "failed to send response");
        }
    }
    Ok(())
}

fn build_response(
    repo: &SqliteRepository,
    images_dir: &Path,
    url: &str,
) -> Response<Cursor<Vec<u8>>> {
    if let Some(rel) = url.strip_prefix("/images/") {
        return serve_image(images_dir, rel);
    }
    match render_review(repo, url) {
        Ok(html) => with_type(Response::from_data(html.into_bytes()), "text/html; charset=utf-8"),
        Err(e) => text_response(500, &format!("error: {e}")),
    }
}

fn render_review(repo: &SqliteRepository, url: &str) -> Result<String, RepoError> {
    let pack = query_param(url, "pack").filter(|s| !s.is_empty());
    let sort = match query_param(url, "sort").as_deref() {
        Some("date_asc") => CaptionSort::DateAsc,
        Some("pack") => CaptionSort::PackPosition,
        _ => CaptionSort::DateDesc, // freshest first by default
    };
    let packs = repo.caption_packs()?;
    let filter = CaptionFilter { pack: pack.clone(), sort, ..Default::default() };
    let views = repo.query_captions(&filter)?;
    let controls = render_controls(&packs, pack.as_deref(), sort);
    Ok(render_page(&views, "/images", &controls))
}

fn render_controls(packs: &[String], current_pack: Option<&str>, sort: CaptionSort) -> String {
    let mut s = String::from("<form method=get class=controls>");
    s.push_str("<label>pack <select name=pack onchange=\"this.form.submit()\">");
    s.push_str(&option("", "all packs", current_pack.is_none()));
    for p in packs {
        s.push_str(&option(p, p, current_pack == Some(p.as_str())));
    }
    s.push_str("</select></label> ");
    s.push_str("<label>sort <select name=sort onchange=\"this.form.submit()\">");
    s.push_str(&option("date_desc", "freshest first", sort == CaptionSort::DateDesc));
    s.push_str(&option("date_asc", "oldest first", sort == CaptionSort::DateAsc));
    s.push_str(&option("pack", "by pack", sort == CaptionSort::PackPosition));
    s.push_str("</select></label></form>");
    s
}

fn option(value: &str, label: &str, selected: bool) -> String {
    format!(
        "<option value=\"{}\"{}>{}</option>",
        escape(value),
        if selected { " selected" } else { "" },
        escape(label),
    )
}

fn serve_image(images_dir: &Path, rel: &str) -> Response<Cursor<Vec<u8>>> {
    if is_unsafe_rel(rel) {
        return text_response(403, "forbidden");
    }
    match std::fs::read(images_dir.join(rel)) {
        Ok(bytes) => with_type(Response::from_data(bytes), mime_of(rel)),
        Err(_) => text_response(404, "not found"),
    }
}

/// True for paths that must not be served: empty, absolute (a `join` of an
/// absolute path escapes `images_dir`), or containing a `..` segment.
fn is_unsafe_rel(rel: &str) -> bool {
    rel.is_empty() || rel.starts_with('/') || rel.split('/').any(|seg| seg == "..")
}

fn mime_of(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("webp") => "image/webp",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        _ => "application/octet-stream",
    }
}

fn with_type(mut resp: Response<Cursor<Vec<u8>>>, content_type: &str) -> Response<Cursor<Vec<u8>>> {
    if let Ok(h) = Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes()) {
        resp.add_header(h);
    }
    resp
}

fn text_response(status: u16, body: &str) -> Response<Cursor<Vec<u8>>> {
    Response::from_data(body.as_bytes().to_vec()).with_status_code(status)
}

/// First value of `key` in the URL's query string (with `+` decoded to space).
/// Pack/sort values are ASCII, so no percent-decoding is needed.
fn query_param(url: &str, key: &str) -> Option<String> {
    let query = url.split_once('?')?.1;
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| v.replace('+', " "))
    })
}

fn cmd_prompts(repo: &SqliteRepository) -> Result<(), AppError> {
    let prompts = repo.list_prompts()?;
    if prompts.is_empty() {
        println!("no prompts registered yet");
        return Ok(());
    }
    for p in &prompts {
        println!("{}: {}", p.version, p.text);
    }
    Ok(())
}

fn print_views(views: &[CaptionView]) {
    if views.is_empty() {
        println!("no matching captions");
        return;
    }
    for v in views {
        println!("{} [{}]  {} / {}", v.pack, v.sticker_id, v.model, v.prompt_version);
        println!("  scene: {}", v.scene);
        if !v.on_image_text.is_empty() {
            println!("  text:  {}", v.on_image_text);
        }
        println!("  tone:  {}", v.tone);
        if !v.situations.is_empty() {
            println!("  uses:  {}", v.situations.join("; "));
        }
        println!();
    }
    println!("{} caption(s)", views.len());
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn escape_neutralizes_html_metacharacters() {
        assert_eq!(escape("a<b>&\"c"), "a&lt;b&gt;&amp;&quot;c");
    }

    fn sample_view() -> CaptionView {
        CaptionView {
            sticker_id: Uuid::nil(),
            pack: "packA".into(),
            image_path: "packA/x.webp".into(),
            model: "qwen3-vl:32b".into(),
            prompt_version: "v1".into(),
            scene: "a <chicken> & friends".into(),
            on_image_text: "ЗАПАХЛО".into(),
            tone: "humorous".into(),
            situations: vec!["cooking".into()],
            created_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn gallery_embeds_image_path_and_escaped_caption() {
        let html = render_gallery(std::slice::from_ref(&sample_view()), Path::new("stickers"));

        assert!(html.contains("src=\"stickers/packA/x.webp\""));
        assert!(html.contains("a &lt;chicken&gt; &amp; friends"), "scene escaped");
        assert!(html.contains("ЗАПАХЛО"), "Cyrillic OCR preserved");
        assert!(html.contains("1 caption(s)"));
    }

    #[test]
    fn server_page_uses_image_route_and_controls() {
        let controls = render_controls(&["packA".into(), "packB".into()], Some("packB"), CaptionSort::DateDesc);
        let html = render_page(std::slice::from_ref(&sample_view()), "/images", &controls);

        assert!(html.contains("src=\"/images/packA/x.webp\""), "served image route");
        assert!(html.contains("<form method=get"), "controls present");
        assert!(html.contains("<option value=\"packB\" selected>packB</option>"), "pack preselected");
        assert!(
            html.contains("<option value=\"date_desc\" selected>freshest first</option>"),
            "sort preselected",
        );
    }

    #[test]
    fn query_param_extracts_values() {
        assert_eq!(query_param("/?pack=mne_pochuj&sort=date_asc", "pack").as_deref(), Some("mne_pochuj"));
        assert_eq!(query_param("/?pack=mne_pochuj&sort=date_asc", "sort").as_deref(), Some("date_asc"));
        assert_eq!(query_param("/?pack=", "pack").as_deref(), Some(""));
        assert_eq!(query_param("/", "pack"), None);
    }

    #[test]
    fn mime_of_maps_known_extensions() {
        assert_eq!(mime_of("a/b.webp"), "image/webp");
        assert_eq!(mime_of("x.png"), "image/png");
        assert_eq!(mime_of("noext"), "application/octet-stream");
    }

    #[test]
    fn unsafe_rel_blocks_traversal_and_absolute_paths() {
        assert!(is_unsafe_rel(""));
        assert!(is_unsafe_rel("/etc/passwd"));
        assert!(is_unsafe_rel("../Cargo.toml"));
        assert!(is_unsafe_rel("packA/../../secret"));
        assert!(!is_unsafe_rel("crazy_klutzy/uuid.webp"));
    }
}
