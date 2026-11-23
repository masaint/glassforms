//! Weaves: edits of a shader's text that find a line and put something else in its place. A weave whose
//! line has been reworded changes nothing and says nothing, and the picture quietly loses what it added.
//! So every line a weave looks for has a name here, and `census` counts how often each was found while a
//! tracer was built (app/tests/weave_anchors.rs holds the counts for the sample tracers and fails, by name,
//! when one changes).
//!
//! Outside a census nothing is counted: `replace` is `str::replace` and costs nothing more.

use std::cell::RefCell;
use std::collections::BTreeMap;

/// For each anchor's name, how many times it was found at each look, in order.
pub type Census = BTreeMap<&'static str, Vec<usize>>;

thread_local! {
    static CENSUS: RefCell<Option<Census>> = const { RefCell::new(None) };
}

/// `f` run with every look for a named anchor counted (on this thread).
pub fn census<T>(f: impl FnOnce() -> T) -> (T, Census) {
    let outer = CENSUS.with(|c| c.borrow_mut().replace(Census::new()));
    let out = f();
    let counted = CENSUS.with(|c| std::mem::replace(&mut *c.borrow_mut(), outer)).unwrap_or_default();
    (out, counted)
}

fn counting() -> bool {
    CENSUS.with(|c| c.borrow().is_some())
}

fn note(name: &'static str, found: usize) {
    CENSUS.with(|c| {
        if let Some(m) = c.borrow_mut().as_mut() {
            m.entry(name).or_default().push(found);
        }
    });
}

/// How often `anchor` stands in `src` (always counted, and noted in a census).
pub fn found(src: &str, name: &'static str, anchor: &str) -> usize {
    let n = src.matches(anchor).count();
    note(name, n);
    n
}

/// `src` with every `anchor` replaced by `by` (`str::replace`), the look noted in a census.
pub fn replace(src: &str, name: &'static str, anchor: &str, by: &str) -> String {
    if counting() { note(name, src.matches(anchor).count()); }
    src.replace(anchor, by)
}

/// Where `anchor` first stands in `src` (`str::find`), the look noted in a census.
pub fn find(src: &str, name: &'static str, anchor: &str) -> Option<usize> {
    if counting() { note(name, src.matches(anchor).count()); }
    src.find(anchor)
}

/// The same with at most `n` replaced (`str::replacen`).
pub fn replacen(src: &str, name: &'static str, anchor: &str, by: &str, n: usize) -> String {
    if counting() { note(name, src.matches(anchor).count()); }
    src.replacen(anchor, by, n)
}

/// `replace` as a method, for chains of weaves.
pub trait Woven {
    fn woven(&self, name: &'static str, anchor: &str, by: &str) -> String;
}

impl Woven for str {
    fn woven(&self, name: &'static str, anchor: &str, by: &str) -> String {
        replace(self, name, anchor, by)
    }
}

/// A weave that went in only partly, or matched where it should match once: its text has drifted from
/// the shader's. Said once per anchor in debug builds (a release build stays quiet; the census test is
/// what fails).
pub fn drifted(name: &'static str, what: &str) {
    #[cfg(debug_assertions)]
    {
        use std::sync::Mutex;
        static SAID: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if !said.contains(&name) {
            said.push(name);
            eprintln!("weave {name}: {what} (its anchor no longer matches the shader text; see app/tests/weave_anchors.rs)");
        }
    }
    #[cfg(not(debug_assertions))]
    let _ = (name, what);
}

/// A group of anchors woven together: all found once, or all absent (a tracer without that part). Some
/// found and some not, or any found more than once, is drift (`drifted`). True when all were found once.
pub fn all_once(name: &'static str, counts: &[usize]) -> bool {
    let once = counts.iter().filter(|&&c| c == 1).count();
    if counts.iter().any(|&c| c > 1) || (once > 0 && once < counts.len()) {
        drifted(name, &format!("found {counts:?} times, expected all once or none"));
    }
    once == counts.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_census_counts_each_look_and_leaves_the_text_as_str_replace_does() {
        let src = "a b a c";
        let ((one, two), counted) = census(|| (replace(src, "t.a", "a", "x"), replacen(src, "t.c", "q", "y", 1)));
        assert_eq!(one, src.replace("a", "x"));
        assert_eq!(two, src);
        assert_eq!(counted["t.a"], vec![2]);
        assert_eq!(counted["t.c"], vec![0]);
        // outside a census nothing is kept
        let _ = replace(src, "t.a", "a", "x");
        assert!(census(|| ()).1.is_empty());
    }

    #[test]
    fn a_group_is_whole_or_absent() {
        assert!(all_once("t.g1", &[1, 1]));
        assert!(!all_once("t.g2", &[0, 0]));
        assert!(!all_once("t.g3", &[1, 0]));
        assert!(!all_once("t.g4", &[2, 1]));
    }
}
