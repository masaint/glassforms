//! Which build this is (worked out by `build.rs` when it was compiled): shown in the window's title, the
//! help window and `--version`.

/// `<branch>@<commit>[+dirty]` (a detached checkout's branch is "detached"), GF_BUILD_ID where there was
/// no git checkout, otherwise "unknown".
pub const ID: &str = env!("GF_BUILD_ID");
/// When it was compiled: `yyyy-mm-dd hh:mm UTC`.
pub const TIME: &str = env!("GF_BUILD_TIME");
/// Built with DLSS.
pub const DLSS: bool = cfg!(feature = "dlss");

/// The line naming a build.
pub fn line(id: &str, time: &str, dlss: bool) -> String {
    format!("Glass Forms · {id} · {time}{}", if dlss { " · DLSS" } else { "" })
}

/// This build's line.
pub fn this() -> String { line(ID, TIME, DLSS) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_line_names_branch_commit_state_time_and_dlss() {
        assert_eq!(line("main@1a2b3c4", "2026-11-21 14:05 UTC", false), "Glass Forms · main@1a2b3c4 · 2026-11-21 14:05 UTC");
        assert_eq!(line("detached@1a2b3c4+dirty", "2026-11-21 14:05 UTC", true), "Glass Forms · detached@1a2b3c4+dirty · 2026-11-21 14:05 UTC · DLSS");
        assert_eq!(line("unknown", "2026-01-01 00:00 UTC", false), "Glass Forms · unknown · 2026-01-01 00:00 UTC");
    }

    #[test]
    fn this_build_has_an_id_and_a_time() {
        assert!(!ID.is_empty());
        assert!(TIME.len() == "2026-11-21 14:05 UTC".len() && TIME.ends_with(" UTC"), "{TIME}");
        assert!(this().starts_with("Glass Forms · "));
    }
}
