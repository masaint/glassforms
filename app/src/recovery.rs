//! Keeping work through a crash: the setup is written to a recovery folder now and then while it changes,
//! and a program that did not close cleanly last time offers it back at the next start.
//!
//! The folder is the user's own (`%LOCALAPPDATA%\GlassForms\recovery`, or beside the settings file where
//! there is no such place), never the workspace. In it: `last.json`, the setup as last written, and
//! `running`, a file held open (shared with no one) for as long as a window runs. A `running` file that
//! can be opened at start belongs to no running program: the last one ended without closing, and its
//! `last.json` is offered back. One that cannot be opened is another window's: this one then neither
//! offers nor writes, so two windows never write over each other.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// How often a changed setup is written at most.
pub const EVERY: Duration = Duration::from_secs(30);

pub struct Recovery {
    dir: PathBuf,
    /// the `running` file, held open while this window runs (none: another window holds it, or the
    /// folder could not be made; then nothing is written)
    held: Option<File>,
    /// what was last written (to write only when the setup has changed), and when it was last looked at
    last: Option<u64>,
    looked: Instant,
    /// the setup the last session left, when it ended without closing (taken once, by `offer`)
    left: Option<(PathBuf, std::time::SystemTime)>,
}

/// The folder the recovery files go in.
pub fn folder() -> PathBuf {
    match std::env::var_os("LOCALAPPDATA") {
        Some(d) if !d.is_empty() => PathBuf::from(d).join("GlassForms").join("recovery"),
        _ => crate::ui::store::program_folder().join("recovery"),
    }
}

/// Open `path` for writing so that no one else can open it while it is held.
fn hold(path: &Path) -> std::io::Result<File> {
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(false);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        o.share_mode(0);
    }
    o.open(path)
}

impl Recovery {
    /// Start keeping work in `dir`: note whether the last window there ended without closing.
    pub fn start(dir: PathBuf) -> Recovery {
        let running = dir.join("running");
        let last = dir.join("last.json");
        let had = running.exists();
        let held = std::fs::create_dir_all(&dir).ok().and_then(|()| hold(&running).ok());
        // (a `running` file left behind and now free: the window that held it is gone without closing)
        let left = if held.is_some() && had { std::fs::metadata(&last).and_then(|m| m.modified()).ok().map(|t| (last, t)) } else { None };
        Recovery { dir, held, last: None, looked: Instant::now(), left }
    }

    /// The setup the last session left and when it was written, if it ended without closing; asked once.
    pub fn offer(&mut self) -> Option<(PathBuf, std::time::SystemTime)> {
        self.left.take()
    }

    /// Write the setup if it has changed since it was last written and `EVERY` has passed since the last
    /// look (`setup` is only asked for then). Returns what went wrong, if anything did.
    pub fn tick(&mut self, setup: impl FnOnce() -> String) -> Result<bool, String> {
        if self.held.is_none() || self.looked.elapsed() < EVERY { return Ok(false); }
        self.looked = Instant::now();
        self.write(&setup())
    }

    /// Write this setup now, if it differs from the one last written.
    pub fn write(&mut self, text: &str) -> Result<bool, String> {
        if self.held.is_none() { return Ok(false); }
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut h);
        let hash = h.finish();
        if self.last == Some(hash) { return Ok(false); }
        // (written beside and then moved over, so a crash while writing never leaves half a file)
        let tmp = self.dir.join("last.json.part");
        std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, self.dir.join("last.json"))).map_err(|e| format!("the recovery copy was not written: {e}"))?;
        self.last = Some(hash);
        Ok(true)
    }

    /// A clean close: the `running` file goes, so the next start offers nothing.
    pub fn close(&mut self) {
        if self.held.take().is_some() { let _ = std::fs::remove_file(self.dir.join("running")); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gf_recovery_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A clean close offers nothing next time; a window gone without closing offers its last setup; a
    /// second window while the first runs neither offers nor writes.
    #[test]
    fn work_is_offered_back_only_after_an_unclean_end() {
        let d = dir("ends");
        let mut a = Recovery::start(d.clone());
        assert!(a.offer().is_none());
        assert_eq!(a.write("{\"one\":1}"), Ok(true));
        assert_eq!(a.write("{\"one\":1}"), Ok(false), "an unchanged setup is not written again");
        // a second window while the first runs
        let mut b = Recovery::start(d.clone());
        if cfg!(windows) {
            assert!(b.offer().is_none());
            assert_eq!(b.write("{\"two\":2}"), Ok(false));
        }
        drop(b);
        a.close();
        assert!(Recovery::start(d.clone()).offer().is_none(), "a clean close offers nothing");
        // (that start held the file and was dropped without closing: as a crash leaves it)
        let mut c = Recovery::start(d.clone());
        let (path, _) = c.offer().expect("an unclean end offers the last setup");
        assert_eq!(std::fs::read_to_string(path).unwrap(), "{\"one\":1}");
        c.close();
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Nothing is asked for before the interval has passed.
    #[test]
    fn the_setup_is_only_made_when_it_is_due() {
        let d = dir("due");
        let mut r = Recovery::start(d.clone());
        assert_eq!(r.tick(|| panic!("asked for too soon")), Ok(false));
        r.looked -= EVERY;
        assert_eq!(r.tick(|| "{}".into()), Ok(true));
        r.close();
        let _ = std::fs::remove_dir_all(&d);
    }
}
