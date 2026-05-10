//! Tokenizer loading and ASR chat-template prompt construction for Qwen3-ASR.

use candle_core::Result as CandleResult;
use std::path::Path;

// Special token IDs from tokenizer_config.json
pub const AUDIO_START_TOKEN_ID: u32 = 151669;
pub const AUDIO_END_TOKEN_ID: u32 = 151670;
pub const AUDIO_PAD_TOKEN_ID: u32 = 151676;
pub const IM_START_TOKEN_ID: u32 = 151644;
pub const IM_END_TOKEN_ID: u32 = 151645;
pub const ENDOFTEXT_TOKEN_ID: u32 = 151643;

/// EOS token IDs — generation stops when any of these is produced.
pub const EOS_TOKEN_IDS: &[u32] = &[IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID];

pub struct Qwen3ASRTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen3ASRTokenizer {
    /// Load from a model directory containing `tokenizer.json`.
    pub fn load(model_dir: &Path) -> CandleResult<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self { tokenizer })
    }

    /// Build the full `input_ids` for an ASR request.
    ///
    /// Returns `(input_ids, audio_token_positions)` where `audio_token_positions`
    /// are the indices within `input_ids` that should be replaced with audio features.
    ///
    /// When `language` is `Some(code)`, the natural-language instruction is
    /// adjusted to "Transcribe the audio to text in {Language}." This matches
    /// how the official Qwen3-ASR-Toolkit injects language hints — without it,
    /// the model can mis-detect the source language (notably German often
    /// collapsed into garbled English/Latin text).
    pub fn encode_asr_prompt(
        &self,
        n_audio_tokens: usize,
        language: Option<&str>,
    ) -> CandleResult<(Vec<u32>, Vec<usize>)> {
        let mut ids: Vec<u32> = Vec::new();

        // System turn
        ids.push(IM_START_TOKEN_ID);
        ids.extend(self.encode_text("system\nYou are a helpful assistant.")?);
        ids.push(IM_END_TOKEN_ID);
        ids.extend(self.encode_text("\n")?);

        // User turn
        ids.push(IM_START_TOKEN_ID);
        ids.extend(self.encode_text("user\n")?);

        // Audio placeholder
        ids.push(AUDIO_START_TOKEN_ID);
        let audio_start_pos = ids.len();
        for _ in 0..n_audio_tokens {
            ids.push(AUDIO_PAD_TOKEN_ID);
        }
        let audio_positions: Vec<usize> = (audio_start_pos..audio_start_pos + n_audio_tokens).collect();
        ids.push(AUDIO_END_TOKEN_ID);

        // Text instruction (with optional language hint)
        let instruction = build_instruction(language);
        ids.extend(self.encode_text(&format!("\n{}", instruction))?);
        ids.push(IM_END_TOKEN_ID);
        ids.extend(self.encode_text("\n")?);

        // Assistant turn start
        ids.push(IM_START_TOKEN_ID);
        ids.extend(self.encode_text("assistant\n")?);

        Ok((ids, audio_positions))
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, token_ids: &[u32]) -> CandleResult<String> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer decode error: {}", e)))
    }

    /// Encode a text string into token IDs (without special tokens).
    fn encode_text(&self, text: &str) -> CandleResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer encode error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }
}

/// Build the natural-language transcription instruction, optionally
/// specialised to a target language.
///
/// Accepts ISO codes (e.g. "de", "fr"), `BCP-47`-style codes ("de-DE", "pt-BR"),
/// or full English names ("german", "Mandarin"). Unknown values fall back to
/// the language-agnostic instruction so we never inject a malformed prompt.
fn build_instruction(language: Option<&str>) -> String {
    match language.and_then(language_full_name) {
        Some(name) => format!("Transcribe the audio to text in {}.", name),
        None => "Transcribe the audio to text.".to_string(),
    }
}

/// Map a language code or English name to the canonical English label that
/// the Qwen3 LLM understands. Returning `None` skips the hint entirely.
///
/// Coverage targets the languages officially documented for Qwen3-ASR-1.7B
/// plus a handful of additional European/Asian languages the underlying LLM
/// reliably recognises.
pub fn language_full_name(code: &str) -> Option<&'static str> {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Take the leading subtag for "de-DE", "pt-BR", "zh_CN", etc.
    let primary = trimmed
        .split(|c: char| c == '-' || c == '_')
        .next()
        .unwrap_or(trimmed)
        .to_ascii_lowercase();
    match primary.as_str() {
        "zh" | "chinese" | "mandarin" | "cmn" => Some("Chinese"),
        "yue" | "cantonese" => Some("Cantonese"),
        "en" | "english" => Some("English"),
        "de" | "german" | "deutsch" => Some("German"),
        "fr" | "french" | "français" | "francais" => Some("French"),
        "es" | "spanish" | "español" | "espanol" => Some("Spanish"),
        "it" | "italian" | "italiano" => Some("Italian"),
        "pt" | "portuguese" | "português" | "portugues" => Some("Portuguese"),
        "ja" | "japanese" | "日本語" => Some("Japanese"),
        "ko" | "korean" | "한국어" => Some("Korean"),
        "ru" | "russian" | "русский" => Some("Russian"),
        "ar" | "arabic" | "العربية" => Some("Arabic"),
        "nl" | "dutch" | "nederlands" => Some("Dutch"),
        "pl" | "polish" | "polski" => Some("Polish"),
        "tr" | "turkish" | "türkçe" | "turkce" => Some("Turkish"),
        "sv" | "swedish" | "svenska" => Some("Swedish"),
        "vi" | "vietnamese" => Some("Vietnamese"),
        "th" | "thai" => Some("Thai"),
        "id" | "indonesian" | "bahasa" => Some("Indonesian"),
        "hi" | "hindi" => Some("Hindi"),
        _ => None,
    }
}

/// Compute the number of audio tokens produced by the feature extractor
/// for a given number of mel spectrogram frames.
///
/// This mirrors `_get_feat_extract_output_lengths` from the Python model.
pub fn get_feat_extract_output_lengths(input_lengths: usize) -> usize {
    let leave = input_lengths % 100;
    let feat = if leave == 0 { 0 } else { (leave - 1) / 2 + 1 };
    let feat = if feat == 0 { 0 } else { (feat - 1) / 2 + 1 };
    let feat = if feat == 0 { 0 } else { (feat - 1) / 2 + 1 };
    feat + (input_lengths / 100) * 13
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feat_extract_output_lengths() {
        // Verified against Python: _get_feat_extract_output_lengths
        assert_eq!(get_feat_extract_output_lengths(100), 13);
        assert_eq!(get_feat_extract_output_lengths(200), 26);
        assert_eq!(get_feat_extract_output_lengths(3000), 390);

        // Edge cases
        assert_eq!(get_feat_extract_output_lengths(0), 0);
        assert_eq!(get_feat_extract_output_lengths(1), 1);
        // input=50: leave=50, feat=(49/2+1)=25, (24/2+1)=13, (12/2+1)=7 => 7+0=7
        assert_eq!(get_feat_extract_output_lengths(50), 7);
    }

    #[test]
    fn test_eos_tokens() {
        assert!(EOS_TOKEN_IDS.contains(&IM_END_TOKEN_ID));
        assert!(EOS_TOKEN_IDS.contains(&ENDOFTEXT_TOKEN_ID));
    }

    #[test]
    fn build_instruction_default() {
        assert_eq!(build_instruction(None), "Transcribe the audio to text.");
        assert_eq!(build_instruction(Some("")), "Transcribe the audio to text.");
        assert_eq!(build_instruction(Some("xx")), "Transcribe the audio to text.");
    }

    #[test]
    fn build_instruction_iso_codes() {
        assert_eq!(build_instruction(Some("de")), "Transcribe the audio to text in German.");
        assert_eq!(build_instruction(Some("DE")), "Transcribe the audio to text in German.");
        assert_eq!(build_instruction(Some("de-DE")), "Transcribe the audio to text in German.");
        assert_eq!(build_instruction(Some("fr")), "Transcribe the audio to text in French.");
        assert_eq!(build_instruction(Some("zh")), "Transcribe the audio to text in Chinese.");
        assert_eq!(build_instruction(Some("pt-BR")), "Transcribe the audio to text in Portuguese.");
    }

    #[test]
    fn build_instruction_english_names() {
        assert_eq!(build_instruction(Some("German")), "Transcribe the audio to text in German.");
        assert_eq!(build_instruction(Some("italian")), "Transcribe the audio to text in Italian.");
    }
}
