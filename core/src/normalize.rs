//! Text normalization applied to embedding inputs — both caption documents at
//! index time and user queries at search time — so the two sides meet in the
//! same lexical space.

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

/// Normalize text for embedding: Unicode NFKC, lowercase, trim, and collapse
/// internal whitespace runs to a single space.
pub fn normalize_for_embedding(text: &str) -> String {
    let folded = text.nfkc().collect::<String>().to_lowercase();
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether embedding inputs are normalized. Off exists for embed models that
/// are tolerant to case/width/whitespace variance, where raw text is preferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// NFKC + lowercase + whitespace collapse (see [`normalize_for_embedding`]).
    #[default]
    Nfkc,
    /// Pass text through untouched.
    Off,
}

impl Normalization {
    pub fn apply<'a>(&self, text: &'a str) -> Cow<'a, str> {
        match self {
            Self::Nfkc => Cow::Owned(normalize_for_embedding(text)),
            Self::Off => Cow::Borrowed(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::lowercases_cyrillic("ЗАПАХЛО Жареным", "запахло жареным")]
    #[case::trims_and_collapses_whitespace("  a \t b\n\nc ", "a b c")]
    #[case::nfkc_folds_ligatures("ﬁsh", "fish")]
    #[case::nfkc_folds_fullwidth("Ｑｗｅｎ", "qwen")]
    #[case::nfkc_composes_combining_accent("cafe\u{0301}", "café")]
    #[case::keeps_emoji_and_punctuation("cat: 🥹, panic!", "cat: 🥹, panic!")]
    fn normalizes(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(normalize_for_embedding(input), expected);
    }

    #[test]
    fn nfkc_mode_applies_normalization() {
        assert_eq!(Normalization::Nfkc.apply("  ЗАПАХЛО  "), "запахло");
    }

    #[test]
    fn off_mode_passes_text_through() {
        assert_eq!(Normalization::Off.apply("  ЗАПАХЛО  "), "  ЗАПАХЛО  ");
    }

    #[test]
    fn default_is_nfkc() {
        assert_eq!(Normalization::default(), Normalization::Nfkc);
    }
}
