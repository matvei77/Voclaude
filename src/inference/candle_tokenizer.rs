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

/// Which chat template to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptStyle {
    /// The template shipped in the model's `chat_template.json`:
    /// `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<audio><|im_end|>\n<|im_start|>assistant\n`
    /// The system slot is empty by default and is the model's context/biasing slot.
    Official,
    /// The template Voclaude used before 2026-09-04: a "You are a helpful
    /// assistant." system turn plus a "Transcribe the audio to text."
    /// instruction after the audio. Kept only for A/B benchmarking.
    Legacy,
}

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
    /// `context` is placed in the system slot (official template only). It is
    /// used to carry the tail of the previous segment's transcript so that
    /// language and vocabulary stay consistent across segment boundaries.
    pub fn encode_asr_prompt(
        &self,
        n_audio_tokens: usize,
        context: Option<&str>,
        style: PromptStyle,
    ) -> CandleResult<(Vec<u32>, Vec<usize>)> {
        let mut ids: Vec<u32> = Vec::new();

        // System turn
        ids.push(IM_START_TOKEN_ID);
        match style {
            PromptStyle::Official => {
                ids.extend(self.encode_text("system\n")?);
                if let Some(ctx) = context.map(str::trim).filter(|c| !c.is_empty()) {
                    ids.extend(self.encode_text(ctx)?);
                }
            }
            PromptStyle::Legacy => {
                ids.extend(self.encode_text("system\nYou are a helpful assistant.")?);
            }
        }
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

        if style == PromptStyle::Legacy {
            ids.extend(self.encode_text("\nTranscribe the audio to text.")?);
        }
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
}
