//! Tokenizer loading and ASR chat-template prompt construction for Qwen3-ASR.

use candle_core::Result as CandleResult;
use serde_json::{json, Value};
use std::{fs, path::Path};
use tracing::info;

const QWEN_SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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
    /// Load `tokenizer.json`, or reconstruct the equivalent fast tokenizer
    /// from the files published by the official Qwen3-ASR repositories.
    pub fn load(model_dir: &Path) -> CandleResult<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?
        } else {
            info!(
                "tokenizer.json is absent; building tokenizer from vocab.json, merges.txt, and tokenizer_config.json"
            );
            build_from_qwen_files(model_dir).map_err(candle_core::Error::Msg)?
        };
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
        let audio_positions: Vec<usize> =
            (audio_start_pos..audio_start_pos + n_audio_tokens).collect();
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

fn build_from_qwen_files(model_dir: &Path) -> Result<tokenizers::Tokenizer, String> {
    let vocab = read_json(&model_dir.join("vocab.json"))?;
    let vocab_len = vocab
        .as_object()
        .map(|entries| entries.len())
        .ok_or_else(|| "vocab.json must contain a JSON object".to_string())?;

    let merges_path = model_dir.join("merges.txt");
    let merges_text = fs::read_to_string(&merges_path)
        .map_err(|e| format!("Failed to read {}: {}", merges_path.display(), e))?;
    let mut merges = Vec::new();
    for (index, raw_line) in merges_text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid BPE merge in {} at line {}",
                merges_path.display(),
                index + 1
            ));
        }
        merges.push(json!([parts[0], parts[1]]));
    }

    let config_path = model_dir.join("tokenizer_config.json");
    let config = read_json(&config_path)?;
    let decoder = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!(
                "{} must contain added_tokens_decoder",
                config_path.display()
            )
        })?;

    let mut definitions = decoder
        .iter()
        .map(|(raw_id, definition)| {
            let id = raw_id.parse::<u32>().map_err(|_| {
                format!(
                    "Invalid added-token ID {:?} in {}",
                    raw_id,
                    config_path.display()
                )
            })?;
            let fields = definition.as_object().cloned().ok_or_else(|| {
                format!(
                    "Added-token definition {} in {} must be an object",
                    raw_id,
                    config_path.display()
                )
            })?;
            Ok((id, fields))
        })
        .collect::<Result<Vec<_>, String>>()?;
    definitions.sort_by_key(|(id, _)| *id);

    let mut added_tokens = Vec::with_capacity(definitions.len());
    for (index, (id, mut fields)) in definitions.into_iter().enumerate() {
        let expected = vocab_len as u32 + index as u32;
        if id != expected {
            return Err(format!(
                "Added-token IDs in {} must be contiguous after the {}-token base vocabulary (expected {}, found {})",
                config_path.display(),
                vocab_len,
                expected,
                id
            ));
        }
        fields.insert("id".to_string(), json!(id));
        added_tokens.push(Value::Object(fields));
    }

    // This is the fast-tokenizer pipeline used by Qwen's Qwen2Tokenizer. The
    // ASR repositories publish its vocabulary, merges, and added-token table,
    // but currently omit the combined tokenizer.json artifact.
    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": { "type": "NFC" },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": { "Regex": QWEN_SPLIT_PATTERN },
                    "behavior": "Isolated",
                    "invert": false
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                }
            ]
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges
        }
    });
    let bytes = serde_json::to_vec(&tokenizer_json)
        .map_err(|e| format!("Failed to serialize reconstructed tokenizer: {}", e))?;
    tokenizers::Tokenizer::from_bytes(bytes)
        .map_err(|e| format!("Failed to build tokenizer from Qwen files: {}", e))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
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
    use std::time::{SystemTime, UNIX_EPOCH};

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
    fn builds_tokenizer_when_combined_json_is_missing() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_dir = std::env::temp_dir().join(format!(
            "voclaude-tokenizer-test-{}-{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&model_dir).unwrap();

        fs::write(model_dir.join("vocab.json"), r#"{"a":0}"#).unwrap();
        fs::write(model_dir.join("merges.txt"), "#version: 0.2\n").unwrap();
        fs::write(
            model_dir.join("tokenizer_config.json"),
            r#"{
                "added_tokens_decoder": {
                    "1": {
                        "content": "<|special|>",
                        "lstrip": false,
                        "normalized": false,
                        "rstrip": false,
                        "single_word": false,
                        "special": true
                    },
                    "2": {
                        "content": "<regular>",
                        "lstrip": false,
                        "normalized": false,
                        "rstrip": false,
                        "single_word": false,
                        "special": false
                    }
                }
            }"#,
        )
        .unwrap();

        let tokenizer = build_from_qwen_files(&model_dir).unwrap();
        assert_eq!(tokenizer.token_to_id("a"), Some(0));
        assert_eq!(tokenizer.token_to_id("<|special|>"), Some(1));
        assert_eq!(tokenizer.token_to_id("<regular>"), Some(2));
        assert_eq!(tokenizer.encode("a", false).unwrap().get_ids(), &[0]);

        fs::remove_dir_all(model_dir).unwrap();
    }
}
