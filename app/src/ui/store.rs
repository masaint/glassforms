//! What the window remembers from one run to the next: which groups are folded away, how big the window
//! was, whether the light room and the materials lab are open and where, how videos and stills are to be
//! made, and the folders videos, stills and setups last went to. It lives in one small file beside the
//! program, and nowhere else.

use std::path::{Path, PathBuf};

use gf_core::settings::RenderOpts;
use serde::{Deserialize, Serialize};

pub const FILE_NAME: &str = "glassforms-ui.json";

/// Which arrangement of the panels this program has. A file written for an earlier arrangement keeps
/// everything it says except which groups are folded: its groups were other groups, in other places,
/// and folding them the old way would hide what this arrangement means a newcomer to see.
pub const ARRANGEMENT: u32 = 4;

/// Every group that folds: those of the left panel from the top, those of the right, those of the light room.
/// (The chooser of named materials and the layers of the materials lab fold too. They are not listed
/// here: the material's own code names them, see `material::folds`.)
pub const GROUPS: [&str; 17] = [
    "material", "light", "camera", "camera-character", "camera-grade", "output", "quality", "video", "joins", "placement", "animate",
    "room-surround", "room-floor", "room-lights", "room-cast", "room-turn", "lab-casts",
];

/// The groups that start folded away: what is set once or seldom. Material and light on the left and the
/// animation on the right start open; inside the camera, the lens's character and the grade are folded;
/// inside the output, quality and video.
pub const FOLDED_AT_FIRST: [&str; 7] = ["camera", "camera-character", "camera-grade", "output", "quality", "video", "placement"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Store {
    /// the arrangement of the panels the file was written for (a file from before there were any says nothing: 0)
    #[serde(default)]
    pub arrangement: u32,
    /// the groups that are folded away
    pub closed: Vec<String>,
    /// the size of the window's inside, in points
    pub window: Option<[f32; 2]>,
    pub maximized: bool,
    /// the light room is open, and where its top left corner was last left (None = where it first appears)
    pub room_open: bool,
    pub room_at: Option<[f32; 2]>,
    /// the same for the materials lab
    pub lab_open: bool,
    pub lab_at: Option<[f32; 2]>,
    /// how videos are made
    pub video: RenderOpts,
    /// where videos are written; None = the "renders" folder beside the program
    pub folder: Option<String>,
    /// where a setup was last saved or loaded; None = the "setups" folder beside the program
    pub setup_folder: Option<String>,
    /// where stills are written; None = the "stills" folder beside the program
    pub still_folder: Option<String>,
    /// stills are written with 16 bits a channel
    pub still_16: bool,
}
impl Default for Store {
    fn default() -> Self {
        Self {
            arrangement: ARRANGEMENT, closed: FOLDED_AT_FIRST.iter().map(|g| g.to_string()).collect(), window: None, maximized: false,
            room_open: false, room_at: None, lab_open: false, lab_at: None, video: RenderOpts::default(), folder: None, setup_folder: None, still_folder: None, still_16: false,
        }
    }
}

/// The folder the program itself is in.
pub fn program_folder() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."))
}

impl Store {
    /// Where the file is kept unless another place is asked for.
    pub fn default_path() -> PathBuf {
        program_folder().join(FILE_NAME)
    }

    /// Read the file; anything missing or unreadable is simply left at its default.
    pub fn load(path: &Path) -> Store {
        std::fs::read_to_string(path).ok().map(|t| Self::read(&t)).unwrap_or_default()
    }

    /// What a file's text says. One written for another arrangement of the panels starts with this
    /// arrangement's groups folded as they first are, and keeps the rest.
    pub fn read(text: &str) -> Store {
        let mut s: Store = serde_json::from_str(text).unwrap_or_default();
        if s.arrangement != ARRANGEMENT {
            s.closed = Store::default().closed;
            s.arrangement = ARRANGEMENT;
        }
        s
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn is_closed(&self, group: &str) -> bool {
        self.closed.iter().any(|g| g == group)
    }

    /// Fold a group away, or bring it back.
    pub fn fold(&mut self, group: &str) {
        match self.closed.iter().position(|g| g == group) {
            Some(i) => { self.closed.remove(i); }
            None => self.closed.push(group.to_owned()),
        }
    }

    /// The folder videos go to.
    pub fn video_folder(&self) -> PathBuf {
        folder_or(&self.folder, "renders")
    }
    /// The folder the setup dialogs open in.
    pub fn setups_folder(&self) -> PathBuf {
        folder_or(&self.setup_folder, "setups")
    }
    /// The folder stills go to.
    pub fn stills_folder(&self) -> PathBuf {
        folder_or(&self.still_folder, "stills")
    }
}

/// A remembered folder, or the one of this name beside the program.
fn folder_or(remembered: &Option<String>, name: &str) -> PathBuf {
    match remembered {
        Some(f) if !f.trim().is_empty() => PathBuf::from(f),
        _ => program_folder().join(name),
    }
}

/// What to remember for a folder a file has just gone to (or come from): nothing, if it is the usual one.
pub fn remembered(file: &Path, usual: &str) -> Option<String> {
    let whole = |p: &Path| std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let dir = whole(file.parent().filter(|d| !d.as_os_str().is_empty())?);
    (dir != whole(&program_folder().join(usual))).then(|| dir.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_fold_and_unfold() {
        let mut s = Store::default();
        assert!(!s.is_closed("material") && !s.is_closed("light") && !s.is_closed("animate"), "{:?}", s.closed);
        assert!(s.is_closed("video") && s.is_closed("output") && s.is_closed("camera"));
        s.fold("camera");
        s.fold("material");
        assert!(!s.is_closed("camera") && s.is_closed("material"));
    }

    #[test]
    fn a_file_with_parts_missing_is_still_read() {
        let s = Store::read(r#"{ "arrangement": 4, "closed": ["animate"], "video": { "fps": 24 } }"#);
        assert_eq!(s.closed, vec!["animate".to_string()]);
        assert_eq!((s.video.fps, s.video.samples, s.window, s.room_open, s.room_at), (24, RenderOpts::default().samples, None, false, None));
        assert_eq!((s.lab_open, s.lab_at), (false, None));
        let again = Store::read(&serde_json::to_string(&s).unwrap());
        assert_eq!(again, s);
        assert!(s.video_folder().ends_with("renders") && s.setups_folder().ends_with("setups") && s.stills_folder().ends_with("stills"));
        let elsewhere = Store { folder: Some("D:/videos".into()), still_folder: Some("D:/pictures".into()), ..Store::default() };
        assert_eq!(elsewhere.video_folder(), PathBuf::from("D:/videos"));
        assert_eq!(elsewhere.stills_folder(), PathBuf::from("D:/pictures"));
        // a file in the usual folder leaves nothing to remember; one elsewhere leaves its folder
        assert_eq!(remembered(&program_folder().join("stills").join("a.png"), "stills"), None);
        let other = std::env::temp_dir().join("a.png");
        assert!(remembered(&other, "stills").is_some_and(|f| Path::new(&f).is_dir()));
        assert_eq!(remembered(Path::new("a.png"), "stills"), None);
        // nonsense is no file at all
        assert_eq!(Store::read("not json"), Store::default());
    }

    #[test]
    fn a_file_from_an_earlier_arrangement_keeps_all_but_its_folds() {
        // as the program wrote it when the families and the motion were on the left and glass was folded
        let old = r#"{ "closed": ["picture", "glass", "motion"], "window": [1219.0, 900.0], "maximized": true,
                       "video": { "size": "2560x1440", "fps": 30, "samples": 64, "lens": 8, "shutter": 0.5, "codec": "h265", "crf": 20 },
                       "folder": "D:/videos", "setup_folder": null, "still_folder": "D:/pictures", "still_16": true }"#;
        let s = Store::read(old);
        assert_eq!(s.closed, Store::default().closed, "the old folds were kept");
        assert!(!s.is_closed("glass") && !s.is_closed("material") && s.is_closed("video"));
        assert_eq!(s.arrangement, ARRANGEMENT);
        assert_eq!((s.window, s.maximized, s.still_16), (Some([1219.0, 900.0]), true, true));
        assert_eq!((s.video.size.as_str(), s.video.fps, s.video.samples, s.video.codec.as_str()), ("2560x1440", 30, 64, "h265"));
        assert_eq!((s.folder.as_deref(), s.still_folder.as_deref()), (Some("D:/videos"), Some("D:/pictures")));
        // written out and read again it is this arrangement's file, and its folds are now the user's
        let mut mine = s.clone();
        mine.fold("material");
        assert_eq!(Store::read(&serde_json::to_string(&mine).unwrap()), mine);
        // a file from an arrangement yet to come is treated the same way
        assert_eq!(Store::read(r#"{ "arrangement": 9, "closed": ["glass"] }"#).closed, Store::default().closed);
    }
}
