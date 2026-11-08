//! `voclaude phrases`: manage the `parakeet_phrases` list in config.toml and
//! see what it does.
//!
//! The command edits only the `parakeet_phrases` array. The rest of the file,
//! comments included, stays as written, and a file the app could not load is
//! never written. A running Voclaude applies the change within a few seconds
//! (the settings watcher in app.rs).

use crate::config::{project_dirs, Config};
use crate::inference::{letter_similarity, parse_phrases, unsure_log_path, word_core, AsrEngine, Engine, Phrase};
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

type Res<T> = Result<T, Box<dyn Error>>;

const USAGE: &str = "usage:
  voclaude phrases                         list the entries
  voclaude phrases add TERM [FORM ...]     add TERM, or spoken forms to an existing TERM
  voclaude phrases remove TERM [FORM ...]  remove TERM, or only the given spoken forms
  voclaude phrases suggest                 spellings worth adding (history + unsure-word log)
  voclaude phrases check [N]               what the list changes on the N newest kept recordings (20)

A spoken form is how Parakeet writes the term when it mishears it, e.g.
  voclaude phrases add Hemingway Hemengwe Megway
Quote values with spaces: voclaude phrases add \"Claude Code\" \"Cloud code\"";

/// A word Parakeet was unsure of is suggested when it is at least this similar
/// to a list spelling (the threshold the phrase re-check uses for candidates).
const SUGGEST_SIM_UNSURE: f64 = 0.6;
/// A history word (no confidence known) needs more: this similarity and at
/// least `SUGGEST_MIN_LEN` letters. On the user's history, 0.6 also proposed
/// short ordinary words for short names ("GPUs" for Opus).
const SUGGEST_SIM_HISTORY: f64 = 0.75;
const SUGGEST_MIN_LEN: usize = 5;
/// History words used this often are ordinary words, not misheard names.
const SUGGEST_MAX_HISTORY_COUNT: usize = 20;

pub fn cli(args: &[String]) -> i32 {
    let result = match args.first().map(String::as_str) {
        None | Some("list") => list(),
        Some("add") => add(&args[1..]),
        Some("remove") => remove(&args[1..]),
        Some("suggest") => suggest(),
        Some("check") => match args.get(1).map(|n| n.parse::<usize>()) {
            None => check(20),
            Some(Ok(n)) if n > 0 => check(n),
            _ => Err("check takes a positive number of recordings".into()),
        },
        Some("help") | Some("-h") | Some("--help") => {
            println!("{}", USAGE);
            Ok(())
        }
        Some(other) => Err(format!("unknown command '{}'\n{}", other, USAGE).into()),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("voclaude phrases: {}", e);
            1
        }
    }
}

// ---------------------------------------------------------------------------
// The list in config.toml
// ---------------------------------------------------------------------------

pub(crate) fn render(p: &Phrase) -> String {
    if p.forms.is_empty() {
        p.term.clone()
    } else {
        format!("{} = {}", p.term, p.forms.join(", "))
    }
}

fn config_text() -> Res<(PathBuf, String)> {
    crate::config_edit::read()
}

pub(crate) fn entries_of(text: &str) -> Res<Vec<String>> {
    let doc: toml_edit::Document = text.parse()?;
    match doc.get("parakeet_phrases") {
        None => Ok(Vec::new()),
        Some(item) => item
            .as_array()
            .ok_or("parakeet_phrases is not an array")?
            .iter()
            .map(|v| v.as_str().map(String::from).ok_or_else(|| "parakeet_phrases must hold strings".into()))
            .collect(),
    }
}

/// `text` with `parakeet_phrases` set to `entries` (one per line). Everything
/// else in the file is kept as written. Fails if the app could not load the result.
pub(crate) fn with_entries(text: &str, entries: &[String]) -> Res<String> {
    let mut doc: toml_edit::Document = text.parse()?;
    let mut array = toml_edit::Array::new();
    for e in entries {
        array.push(e.as_str());
    }
    for v in array.iter_mut() {
        v.decor_mut().set_prefix("\n  ");
    }
    array.set_trailing(if entries.is_empty() { "" } else { "\n" });
    array.set_trailing_comma(!entries.is_empty());
    match doc.get_mut("parakeet_phrases") {
        Some(item) => {
            // Keep the key, its comments and the spacing around "=".
            let decor = item.as_value().map(|v| v.decor().clone());
            *item = toml_edit::value(array);
            if let (Some(decor), Some(v)) = (decor, item.as_value_mut()) {
                *v.decor_mut() = decor;
            }
        }
        None => doc["parakeet_phrases"] = toml_edit::value(array),
    }
    let out = doc.to_string();
    let config: Config = toml::from_str(&out)?;
    config.validate()?;
    Ok(out)
}

pub(crate) fn save(path: &Path, text: &str, phrases: &[Phrase]) -> Res<()> {
    let entries: Vec<String> = phrases.iter().map(render).collect();
    Config::write_atomic(path, &with_entries(text, &entries)?)
}

fn same(a: &str, b: &str) -> bool {
    a.to_lowercase() == b.to_lowercase()
}

/// Add `term`, or spoken forms to an existing `term`. Returns false if nothing changed.
pub(crate) fn add_to(phrases: &mut Vec<Phrase>, term: &str, forms: &[String]) -> bool {
    let (index, mut changed) = match phrases.iter().position(|p| p.term == term) {
        Some(i) => (i, false),
        None => {
            phrases.push(Phrase { term: term.to_string(), forms: Vec::new() });
            (phrases.len() - 1, true)
        }
    };
    let p = &mut phrases[index];
    for f in forms {
        if !same(f, &p.term) && !p.forms.iter().any(|x| same(x, f)) {
            p.forms.push(f.clone());
            changed = true;
        }
    }
    changed
}

/// Remove `term` (no forms given) or only the given forms. Returns false if not found.
pub(crate) fn remove_from(phrases: &mut Vec<Phrase>, term: &str, forms: &[String]) -> bool {
    let Some(index) = phrases.iter().position(|p| p.term == term).or_else(|| phrases.iter().position(|p| same(&p.term, term)))
    else {
        return false;
    };
    if forms.is_empty() {
        phrases.remove(index);
        return true;
    }
    let p = &mut phrases[index];
    let before = p.forms.len();
    p.forms.retain(|x| !forms.iter().any(|f| same(f, x)));
    p.forms.len() != before
}

/// TERM and FORMs from the command line; "=" and "," would break the entry syntax.
fn term_and_forms(args: &[String]) -> Res<(String, Vec<String>)> {
    let mut values = args.iter().map(|a| tidy(a)).filter(|a| !a.is_empty());
    let term = values.next().ok_or_else(|| format!("missing TERM\n{}", USAGE))?;
    let forms: Vec<String> = values.collect();
    if std::iter::once(&term).chain(&forms).any(|v| !allowed(v)) {
        return Err("terms and spoken forms cannot contain '=' or ','".into());
    }
    Ok((term, forms))
}

/// One space between words, no leading or trailing space.
pub(crate) fn tidy(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// "=" and "," would break the entry syntax ("Term = form, form").
pub(crate) fn allowed(v: &str) -> bool {
    !v.contains('=') && !v.contains(',')
}

fn list() -> Res<()> {
    let (path, text) = config_text()?;
    let phrases = parse_phrases(&entries_of(&text)?);
    println!("{}", path.display());
    if phrases.is_empty() {
        println!("  (no entries)  add one: voclaude phrases add Qwen Quan Quen");
    }
    for p in &phrases {
        println!("  {}", render(p));
    }
    Ok(())
}

fn add(args: &[String]) -> Res<()> {
    let (term, forms) = term_and_forms(args)?;
    let (path, text) = config_text()?;
    let mut phrases = parse_phrases(&entries_of(&text)?);
    if !add_to(&mut phrases, &term, &forms) {
        println!("Nothing to add: {}", phrases.iter().find(|p| p.term == term).map(render).unwrap_or_default());
        return Ok(());
    }
    save(&path, &text, &phrases)?;
    println!("Saved: {}", phrases.iter().find(|p| p.term == term).map(render).unwrap_or_default());
    println!("A running Voclaude applies it within a few seconds.");
    Ok(())
}

fn remove(args: &[String]) -> Res<()> {
    let (term, forms) = term_and_forms(args)?;
    let (path, text) = config_text()?;
    let mut phrases = parse_phrases(&entries_of(&text)?);
    if !remove_from(&mut phrases, &term, &forms) {
        return Err(format!("'{}' (or the given spoken forms) is not in the list", term).into());
    }
    save(&path, &text, &phrases)?;
    println!("Removed. The list now has {} entries.", phrases.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// suggest
// ---------------------------------------------------------------------------

/// One line of data/unsure-words.jsonl.
pub(crate) struct UnsureWord {
    pub word: String,
    pub p: f64,
    pub term: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Suggestion {
    pub term: String,
    pub word: String,
    pub unsure: usize,
    pub history: usize,
}

pub(crate) fn is_latin_word(w: &str) -> bool {
    !w.is_empty() && w.chars().all(|c| c.is_ascii_alphabetic() || c == '-' || c == '\'') && w.chars().any(|c| c.is_ascii_alphabetic())
}

/// Spellings that look like a list spelling but are not one: from the unsure
/// words of the phrase re-check (acoustic doubt), and from capitalised words
/// inside sentences in the history (names), skipping words used so often that
/// they are ordinary words.
pub(crate) fn suggestions(phrases: &[Phrase], history: &[String], unsure: &[UnsureWord]) -> Vec<Suggestion> {
    let spellings: Vec<(&str, &str)> = phrases
        .iter()
        .flat_map(|p| std::iter::once((p.term.as_str(), p.term.as_str())).chain(p.forms.iter().map(move |f| (f.as_str(), p.term.as_str()))))
        .collect();
    // "Jonathan's" is the term, not a spelling of it.
    let base = |w: &str| -> String {
        let w = w.strip_suffix("'s").or_else(|| w.strip_suffix("’s")).unwrap_or(w);
        w.to_string()
    };
    let known = |w: &str| spellings.iter().any(|(s, _)| same(s, &base(w)));
    let nearest = |w: &str, min_sim: f64| {
        spellings
            .iter()
            .map(|(s, t)| (letter_similarity(w, s), *t))
            .filter(|(sim, _)| *sim >= min_sim)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, t)| t.to_string())
    };
    let mut found: BTreeMap<(String, String), Suggestion> = BTreeMap::new();
    for u in unsure.iter().filter(|u| u.term.is_none() && is_latin_word(&u.word) && !known(&u.word)) {
        if let Some(term) = nearest(&u.word, SUGGEST_SIM_UNSURE) {
            let s = found.entry((term.clone(), u.word.to_lowercase())).or_default();
            (s.term, s.word) = (term, u.word.clone());
            s.unsure += 1;
        }
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for text in history {
        for w in text.split_whitespace() {
            *counts.entry(word_core(w).to_lowercase()).or_default() += 1;
        }
    }
    for text in history {
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, w) in words.iter().enumerate() {
            let core = word_core(w);
            let mid_sentence = i > 0 && !words[i - 1].ends_with(['.', '?', '!']);
            if !mid_sentence || !core.starts_with(|c: char| c.is_ascii_uppercase()) || !is_latin_word(core) || known(core) {
                continue;
            }
            if core.chars().filter(|c| c.is_ascii_alphabetic()).count() < SUGGEST_MIN_LEN
                || counts.get(&core.to_lowercase()).copied().unwrap_or(0) > SUGGEST_MAX_HISTORY_COUNT
            {
                continue;
            }
            if let Some(term) = nearest(core, SUGGEST_SIM_HISTORY) {
                let s = found.entry((term.clone(), core.to_lowercase())).or_default();
                if s.word.is_empty() {
                    (s.term, s.word) = (term, core.to_string());
                }
                s.history += 1;
            }
        }
    }
    let mut out: Vec<Suggestion> = found.into_values().collect();
    out.sort_by(|a, b| (b.unsure * 3 + b.history).cmp(&(a.unsure * 3 + a.history)).then(a.term.cmp(&b.term)));
    out
}

pub(crate) fn read_unsure_log() -> Vec<UnsureWord> {
    let Some(text) = unsure_log_path().and_then(|p| std::fs::read_to_string(p).ok()) else { return Vec::new() };
    text.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| {
            Some(UnsureWord {
                word: v.get("word")?.as_str()?.to_string(),
                p: v.get("p").and_then(|p| p.as_f64()).unwrap_or(f64::NAN),
                term: v.get("term").and_then(|t| t.as_str()).map(String::from),
            })
        })
        .collect()
}

/// History entries, newest first: (created_at_ms, text, kept audio file).
pub(crate) fn read_history() -> Res<Vec<(u64, String, Option<PathBuf>)>> {
    let path = project_dirs().ok_or("could not determine the data folder")?.data_dir().join("history.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let mut entries: Vec<(u64, String, Option<PathBuf>)> = value
        .as_array()
        .ok_or("history.json is not a list")?
        .iter()
        .map(|e| {
            (
                e.get("created_at_ms").and_then(|t| t.as_u64()).unwrap_or(0),
                e.get("text").and_then(|t| t.as_str()).unwrap_or_default().to_string(),
                e.get("audio_path").and_then(|a| a.as_str()).map(PathBuf::from).filter(|p| p.is_file()),
            )
        })
        .collect();
    entries.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(entries)
}

fn suggest() -> Res<()> {
    let config = Config::load()?;
    let phrases = parse_phrases(&config.parakeet_phrases);
    if phrases.is_empty() {
        return Err("the list is empty; add a term first: voclaude phrases add TERM".into());
    }
    let history: Vec<String> = read_history()?.into_iter().map(|(_, t, _)| t).collect();
    let unsure = read_unsure_log();
    let list = suggestions(&phrases, &history, &unsure);
    if list.is_empty() {
        println!("No new spellings found ({} history entries, {} unsure words).", history.len(), unsure.len());
    } else {
        println!("Spellings close to a list entry (check each one before you add it):");
        for s in list.iter().take(30) {
            let quote = |v: &str| if v.contains(' ') { format!("\"{}\"", v) } else { v.to_string() };
            println!(
                "  {:<14} {:<16} unsure {:>2}x, history {:>2}x   voclaude phrases add {} {}",
                s.term, s.word, s.unsure, s.history, quote(&s.term), quote(&s.word)
            );
        }
    }
    // Feedback: what the list wrote, and recurring doubts no entry covers.
    let written = written_counts(&unsure);
    if !written.is_empty() {
        let parts: Vec<String> = written.iter().map(|(t, n)| format!("{} {}x", t, n)).collect();
        println!("Written by the list so far: {}", parts.join(", "));
    }
    let recurring = recurring_doubts(&phrases, &unsure);
    if !recurring.is_empty() {
        println!("Words Parakeet was unsure of more than once (a name? add it with its right spelling):");
        for (w, n, p) in recurring.iter().take(15) {
            println!("  {:<18} {:>2}x, lowest confidence {:.2}", w, n, p);
        }
    }
    Ok(())
}

/// How often the list wrote each term, most first.
pub(crate) fn written_counts(unsure: &[UnsureWord]) -> Vec<(String, usize)> {
    let mut written: BTreeMap<&str, usize> = BTreeMap::new();
    for t in unsure.iter().filter_map(|u| u.term.as_deref()) {
        *written.entry(t).or_default() += 1;
    }
    let mut out: Vec<(String, usize)> = written.into_iter().map(|(t, n)| (t.to_string(), n)).collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Fillers Parakeet is often unsure of; never a name.
const FILLERS: &[&str] = &["uh", "um", "uhm", "er", "erm", "ah", "eh", "oh", "hmm", "mm", "mhm", "huh"];

/// Words Parakeet was unsure of at least twice that no entry covers:
/// (word, times, lowest confidence), most frequent first. Spellings already
/// in the list and filler words are left out.
pub(crate) fn recurring_doubts(phrases: &[Phrase], unsure: &[UnsureWord]) -> Vec<(String, usize, f64)> {
    let known = |w: &str| phrases.iter().any(|p| same(&p.term, w) || p.forms.iter().any(|f| same(f, w)));
    let mut doubts: BTreeMap<String, (String, usize, f64)> = BTreeMap::new();
    for u in unsure.iter().filter(|u| u.term.is_none()) {
        let d = doubts.entry(u.word.to_lowercase()).or_insert((u.word.clone(), 0, f64::INFINITY));
        d.1 += 1;
        d.2 = d.2.min(u.p);
    }
    let mut out: Vec<(String, usize, f64)> = doubts
        .into_values()
        .filter(|(w, n, _)| *n >= 2 && is_latin_word(w) && !known(w) && !FILLERS.contains(&w.to_lowercase().as_str()))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

/// Word spans that differ between `a` and `b` (longest common subsequence).
pub(crate) fn word_diff(a: &str, b: &str) -> Vec<(String, String)> {
    let x: Vec<&str> = a.split_whitespace().collect();
    let y: Vec<&str> = b.split_whitespace().collect();
    let (n, m) = (x.len(), y.len());
    let mut lcs = vec![0u32; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[at(i, j)] = if x[i] == y[j] { lcs[at(i + 1, j + 1)] + 1 } else { lcs[at(i + 1, j)].max(lcs[at(i, j + 1)]) };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    let (mut old, mut new): (Vec<&str>, Vec<&str>) = (Vec::new(), Vec::new());
    let flush = |old: &mut Vec<&str>, new: &mut Vec<&str>, out: &mut Vec<(String, String)>| {
        if !old.is_empty() || !new.is_empty() {
            out.push((old.join(" "), new.join(" ")));
            old.clear();
            new.clear();
        }
    };
    while i < n || j < m {
        if i < n && j < m && x[i] == y[j] {
            flush(&mut old, &mut new, &mut out);
            i += 1;
            j += 1;
        } else if j < m && (i == n || lcs[at(i, j + 1)] >= lcs[at(i + 1, j)]) {
            new.push(y[j]);
            j += 1;
        } else {
            old.push(x[i]);
            i += 1;
        }
    }
    flush(&mut old, &mut new, &mut out);
    out
}

fn age(created_at_ms: u64) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let minutes = now.saturating_sub(created_at_ms) / 60_000;
    match minutes {
        0..=59 => format!("{} min ago", minutes),
        60..=2879 => format!("{} h ago", minutes / 60),
        _ => format!("{} days ago", minutes / 1440),
    }
}

fn transcribe_all(config: &Config, files: &[&Path]) -> Res<Vec<String>> {
    let mut engine = Engine::from_config(config)?;
    engine.prepare(None)?;
    files.iter().map(|f| engine.transcribe_file_with_progress(f, None)).collect()
}

fn check(n: usize) -> Res<()> {
    let config = Config::load()?;
    if !config.uses_parakeet() {
        return Err("the phrase list applies to the Parakeet engine only".into());
    }
    if config.parakeet_phrases.is_empty() {
        return Err("the list is empty; add a term first: voclaude phrases add TERM".into());
    }
    let recordings: Vec<(u64, PathBuf)> =
        read_history()?.into_iter().filter_map(|(t, _, a)| a.map(|a| (t, a))).take(n).collect();
    if recordings.is_empty() {
        return Err("no kept recordings in the history (see audio_retention_count)".into());
    }
    println!("Transcribing {} recordings with and without the list...", recordings.len());
    let files: Vec<&Path> = recordings.iter().map(|(_, p)| p.as_path()).collect();
    let mut with_list = config.clone();
    with_list.parakeet_unsure_log = false;
    let mut without_list = with_list.clone();
    without_list.parakeet_phrases.clear();
    let with = transcribe_all(&with_list, &files)?;
    let without = transcribe_all(&without_list, &files)?;
    let mut changed = 0;
    for ((t, _), (a, b)) in recordings.iter().zip(without.iter().zip(&with)) {
        let diff = word_diff(a, b);
        if diff.is_empty() {
            continue;
        }
        changed += diff.len();
        let parts: Vec<String> = diff.iter().map(|(o, n)| format!("[{}] -> [{}]", o, n)).collect();
        println!("  {:<12} {}", age(*t), parts.join("   "));
    }
    println!("{} change(s) in {} recordings. Every change should be a word you meant.", changed, recordings.len());
    println!("A wrong one: remove the spoken form that caused it, or the term (voclaude phrases remove ...).");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(term: &str, forms: &[&str]) -> Phrase {
        Phrase { term: term.into(), forms: forms.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn editing_keeps_comments_and_other_settings() {
        let text = "# My settings\nhotkey = \"F4\"\n\n# names\nparakeet_phrases = [\"Qwen = Quan\"]\nuse_gpu = true\n";
        let out = with_entries(text, &["Qwen = Quan, Quen".to_string(), "Jonathan".to_string()]).unwrap();
        assert!(out.starts_with("# My settings\nhotkey = \"F4\"\n\n# names\nparakeet_phrases = ["), "{}", out);
        assert!(out.contains("use_gpu = true"));
        assert_eq!(entries_of(&out).unwrap(), vec!["Qwen = Quan, Quen", "Jonathan"]);
        let config: Config = toml::from_str(&out).unwrap();
        assert_eq!(config.parakeet_phrases.len(), 2);
    }

    #[test]
    fn editing_adds_a_missing_key() {
        let out = with_entries("hotkey = \"F4\"\n", &["Codex = codecs".to_string()]).unwrap();
        assert_eq!(entries_of(&out).unwrap(), vec!["Codex = codecs"]);
        assert!(out.starts_with("hotkey = \"F4\"\n"));
    }

    #[test]
    fn editing_refuses_a_file_the_app_cannot_load() {
        assert!(with_entries("hotkey = \"\"\n", &["Qwen".to_string()]).is_err(), "an empty hotkey fails validation");
    }

    #[test]
    fn add_merges_forms_and_remove_takes_them_out() {
        let mut list = vec![p("Qwen", &["Quan"])];
        assert!(add_to(&mut list, "Qwen", &["quan".into(), "Quen".into()]));
        assert_eq!(list, vec![p("Qwen", &["Quan", "Quen"])], "case-insensitive duplicates are skipped");
        assert!(add_to(&mut list, "Codex", &[]));
        assert_eq!(list.len(), 2);
        assert!(!add_to(&mut list, "Codex", &["codex".into()]), "a form equal to its term is not added");
        assert!(remove_from(&mut list, "Qwen", &["quen".into()]));
        assert_eq!(list[0], p("Qwen", &["Quan"]));
        assert!(remove_from(&mut list, "codex", &[]));
        assert_eq!(list.len(), 1);
        assert!(!remove_from(&mut list, "Missing", &[]));
    }

    #[test]
    fn command_line_values_are_checked() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            term_and_forms(&args(&["Claude  Code", "Cloud code"])).unwrap(),
            ("Claude Code".to_string(), vec!["Cloud code".to_string()])
        );
        assert!(term_and_forms(&args(&["Qwen = Quan"])).is_err());
        assert!(term_and_forms(&args(&[])).is_err());
    }

    #[test]
    fn suggestions_come_from_doubts_and_names_not_common_words() {
        let phrases =
            vec![p("Hemingway", &["Hemengwe", "Hebenquewe"]), p("Qwen", &[]), p("Opus", &[]), p("Lisa", &[]), p("Jonathan", &[])];
        let unsure = vec![
            UnsureWord { word: "Hemenque".into(), p: 0.8, term: None },
            UnsureWord { word: "Hemengwe".into(), p: 0.8, term: Some("Hemingway".into()) },
        ];
        let mut history = vec![
            "Send it to Hebenque please.".to_string(),
            "When can we start?".to_string(),
            "We need more GPUs and Tron is down, ask Jonathan's team.".to_string(),
        ];
        history.extend(std::iter::repeat("and when it runs".to_string()).take(30));
        let s = suggestions(&phrases, &history, &unsure);
        let words: Vec<&str> = s.iter().map(|s| s.word.as_str()).collect();
        assert!(words.contains(&"Hemenque"), "{:?}", words);
        assert!(words.contains(&"Hebenque"), "{:?}", words);
        assert!(!words.iter().any(|w| w.eq_ignore_ascii_case("when")), "sentence-initial and common words are skipped: {:?}", words);
        assert!(!words.contains(&"Hemengwe"), "known spellings are skipped");
        for noise in ["GPUs", "Tron", "Jonathan's"] {
            assert!(!words.contains(&noise), "{} is short, unlike enough or a possessive: {:?}", noise, words);
        }
    }

    #[test]
    fn recurring_doubts_skip_known_spellings_and_fillers() {
        let phrases = vec![p("Jonathan", &["Jonathon"]), p("Lisa", &[])];
        let u = |w: &str, p: f64| UnsureWord { word: w.into(), p, term: None };
        let unsure = vec![u("Uh", 0.3), u("uh", 0.2), u("lisa", 0.5), u("Lisa", 0.4), u("Jonathon", 0.4), u("Jonathon", 0.3), u("Lyza", 0.4), u("lyza", 0.2), u("Once", 0.5)];
        let out = recurring_doubts(&phrases, &unsure);
        assert_eq!(out.len(), 1, "{:?}", out);
        assert_eq!((out[0].0.as_str(), out[0].1), ("Lyza", 2));
        assert!((out[0].2 - 0.2).abs() < 1e-9);
    }

    #[test]
    fn word_diff_finds_replaced_spans() {
        let d = word_diff("send it to uh Hemengwe. and the Quan model", "send it to uh Hemingway. and the Qwen model");
        assert_eq!(d, vec![("Hemengwe.".into(), "Hemingway.".into()), ("Quan".into(), "Qwen".into())]);
        assert!(word_diff("same text", "same text").is_empty());
    }
}
