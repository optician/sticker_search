//! Pure helpers for the bot: building inline answers, parsing `/add`, and
//! wording the status reply. No teloxide, no I/O — all unit-testable.

use sticker_core::{PackReport, PackStage, SearchHit, normalize_pack_name};

/// `(result_id, sticker_file_id)` pairs for an inline answer, in rank order.
/// The id is the sticker UUID — unique per hit and well under Telegram's 64-byte
/// inline-result-id limit; the file_id is what a cached-sticker result needs to
/// render. The caller maps these onto teloxide's `InlineQueryResultCachedSticker`.
pub fn inline_entries(hits: &[SearchHit]) -> Vec<(String, String)> {
    hits.iter()
        .map(|h| (h.sticker.id.to_string(), h.sticker.file_id.clone()))
        .collect()
}

/// The pack name from a `/add …` message, or `None` if no name was given.
/// Accepts `/add <name>`, `/add <link>`, and the group-addressed
/// `/add@botname <name>`; strips the command and normalizes the rest. Returns
/// `None` for a bare `/add` and for non-command text like `/added`.
pub fn add_target_from_text(text: &str) -> Option<String> {
    let rest = text.strip_prefix("/add")?;
    // The char right after "/add" must end the command token — otherwise this is
    // a different word (e.g. "/added"), not the add command.
    match rest.chars().next() {
        None => return None,                           // bare "/add"
        Some(c) if c.is_whitespace() || c == '@' => {} // "/add x" or "/add@bot x"
        Some(_) => return None,                        // "/addfoo"
    }
    // Drop a leading "@botname" token when present, then take the remainder.
    let rest = rest.trim_start();
    let rest = match rest.strip_prefix('@') {
        Some(after_at) => after_at
            .split_once(char::is_whitespace)
            .map(|(_, r)| r)
            .unwrap_or(""),
        None => rest,
    };
    let name = normalize_pack_name(rest);
    (!name.is_empty()).then_some(name)
}

/// The reply to an `/add`, worded for the pack's derived stage. Plain text (no
/// Markdown) so pack names containing `_` need no escaping.
pub fn format_status(r: &PackReport, bot_username: &str) -> String {
    match r.stage {
        PackStage::Queued => format!(
            "📥 Queued “{}”. It'll be scraped, captioned and embedded on the next \
             batch run, then it's searchable. Send /add again to check progress.",
            r.name,
        ),
        PackStage::Scraped => format!(
            "⏳ “{}”: scraped, captioning… ({}/{} captioned)",
            r.name, r.captioned_count, r.sticker_count,
        ),
        PackStage::Captioned => format!(
            "⏳ “{}”: captioned, embedding… ({}/{} embedded)",
            r.name, r.embedded_count, r.sticker_count,
        ),
        PackStage::Ready => format!(
            "✅ “{}” is ready ({} stickers). Search it: type @{} then your query in any chat.",
            r.name, r.sticker_count, bot_username,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sticker_core::{Caption, Sticker, StickerFormat};
    use time::OffsetDateTime;
    use uuid::Uuid;

    fn hit(file_id: &str) -> SearchHit {
        let id = Uuid::new_v4();
        SearchHit {
            score: 0.9,
            sticker: Sticker {
                id,
                pack_id: Uuid::nil(),
                file_unique_id: format!("u-{file_id}"),
                file_id: file_id.into(),
                emoji: None,
                format: StickerFormat::Static,
                width: 512,
                height: 512,
                position: 0,
                image_path: "p/x.webp".into(),
                created_at: OffsetDateTime::UNIX_EPOCH,
            },
            caption: Caption {
                sticker_id: id,
                model: "qwen".into(),
                prompt_version: "v1".into(),
                scene: "s".into(),
                on_image_text: String::new(),
                tone: "t".into(),
                situations: vec![],
                raw: String::new(),
                created_at: OffsetDateTime::UNIX_EPOCH,
            },
        }
    }

    #[test]
    fn inline_entries_pairs_uuid_with_file_id_in_order() {
        let hits = [hit("fileA"), hit("fileB")];
        let entries = inline_entries(&hits);
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].0,
            hits[0].sticker.id.to_string(),
            "id is the sticker UUID"
        );
        assert_eq!(entries[0].1, "fileA");
        assert_eq!(entries[1].1, "fileB", "rank order preserved");
        assert!(entries[0].0.len() <= 64, "id within Telegram's limit");
    }

    #[rstest::rstest]
    #[case("/add crazy_klutzy", Some("crazy_klutzy"))]
    #[case("/add   https://t.me/addstickers/crazy_klutzy", Some("crazy_klutzy"))]
    #[case("/add@my_bot crazy_klutzy", Some("crazy_klutzy"))]
    #[case("/add", None)]
    #[case("/add   ", None)]
    #[case("/add@my_bot", None)]
    #[case("/added something", None)]
    #[case("hello", None)]
    fn add_target_parsing(#[case] input: &str, #[case] expected: Option<&str>) {
        assert_eq!(add_target_from_text(input).as_deref(), expected);
    }

    fn report(stage: PackStage, captioned: usize, embedded: usize) -> PackReport {
        PackReport {
            name: "my_pack".into(),
            stage,
            sticker_count: 50,
            captioned_count: captioned,
            embedded_count: embedded,
        }
    }

    #[test]
    fn status_wording_per_stage() {
        assert!(format_status(&report(PackStage::Queued, 0, 0), "b").contains("Queued"));

        let scraped = format_status(&report(PackStage::Scraped, 12, 0), "b");
        assert!(scraped.contains("captioning"));
        assert!(scraped.contains("12/50"));

        let captioned = format_status(&report(PackStage::Captioned, 50, 30), "b");
        assert!(captioned.contains("embedding"));
        assert!(captioned.contains("30/50"));

        let ready = format_status(&report(PackStage::Ready, 50, 50), "my_bot");
        assert!(ready.contains("ready"));
        assert!(ready.contains("@my_bot"), "tells the user how to search");
    }
}
