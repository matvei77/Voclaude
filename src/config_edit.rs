//! Change single settings in config.toml and keep the rest of the file, with
//! its comments, as written. Used by `voclaude phrases` and the console.
//!
//! A change is written only if the app can load the result (`Config` parses
//! and validates). The running app applies it within a few seconds (the
//! settings watcher in app.rs).

use crate::config::Config;
use std::error::Error;
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn Error>>;

/// One change to config.toml.
pub enum Edit {
    /// Set `key` to a value.
    Set(&'static str, toml_edit::Value),
    /// Remove `key`, so the app uses its default.
    Remove(&'static str),
}

/// The config path and its current text. Creates the commented example file
/// when there is none.
pub fn read() -> Res<(PathBuf, String)> {
    let path = Config::config_path().ok_or("could not determine the config path")?;
    if !path.exists() {
        Config::load()?; // writes the commented example
    }
    let text = std::fs::read_to_string(&path)?;
    Ok((path, text))
}

/// `text` with `edits` applied. Fails if the app could not load the result.
pub fn apply(text: &str, edits: &[Edit]) -> Res<String> {
    let mut doc: toml_edit::Document = text.parse()?;
    for edit in edits {
        match edit {
            Edit::Set(key, value) => match doc.get_mut(key) {
                Some(item) if item.is_value() => {
                    // Keep the comments and spacing around the old value.
                    let decor = item.as_value().map(|v| v.decor().clone());
                    *item = toml_edit::Item::Value(value.clone());
                    if let (Some(decor), Some(v)) = (decor, item.as_value_mut()) {
                        *v.decor_mut() = decor;
                    }
                }
                _ => doc[key] = toml_edit::Item::Value(value.clone()),
            },
            Edit::Remove(key) => {
                doc.remove(key);
            }
        }
    }
    let out = doc.to_string();
    let config: Config = toml::from_str(&out)?;
    config.validate()?;
    Ok(out)
}

/// Apply `edits` to config.toml on disk.
pub fn save(edits: &[Edit]) -> Res<()> {
    let (path, text) = read()?;
    let out = apply(&text, edits)?;
    if out != text {
        Config::write_atomic(&path, &out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_keeps_comments_and_validates() {
        let text = "# top\nhotkey = \"F4\" # record\nuse_gpu = true\n";
        let out = apply(text, &[Edit::Set("use_gpu", false.into()), Edit::Set("cpu_threads", 4i64.into())]).unwrap();
        assert!(out.starts_with("# top\nhotkey = \"F4\" # record\n"), "{}", out);
        assert!(out.contains("use_gpu = false"));
        assert!(out.contains("cpu_threads = 4"));
        assert!(apply(text, &[Edit::Set("hotkey", "".into())]).is_err());
        assert!(apply(text, &[Edit::Set("engine", "nope".into())]).is_err());
    }

    #[test]
    fn remove_falls_back_to_the_default() {
        let out = apply("segment_min_seconds = 3.0\nhotkey = \"F4\"\n", &[Edit::Remove("segment_min_seconds")]).unwrap();
        assert_eq!(out, "hotkey = \"F4\"\n");
    }
}
