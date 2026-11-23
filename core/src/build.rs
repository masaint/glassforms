//! How carefully a shape is built, and who asked for the build.
//!
//! A build is either a **draft** or **final**:
//! - a draft is cheap and made while something moves (a slider or handle is held, an animation plays).
//!   It is shown only until the movement stops, and is never kept as if it were final;
//! - a final build is the settled shape at its full detail (a still, or a video frame).
//!
//! `Detail` says which of the four kinds of build is asked for. Every choice that depends on it (the
//! voxel count, whether boxes are measured, the stage's cap, whether the build is waited for...) is
//! made by one of its methods below, and nowhere else. Cache keys record `Quality`, so a draft volume
//! can never be taken for a final one.
//!
//! | Detail | Quality | Asked for by | Voxels a side |
//! |---|---|---|---|
//! | Quick | draft | a slider being dragged | 256 (`RES_DRAG`), at most the still detail |
//! | Anim | draft | a played frame, a handle being dragged | the shape's moving detail, capped by frame-rate keeping |
//! | Full | final | an edit, a release, a pause | the shape's still detail |
//! | Hq | final | a video frame | the shape's still detail |
//!
//! The request itself is a `BuildRequest`: the detail plus why it was asked for. There is one slot in
//! `State` (`State::request`), and the last request in a frame wins, as it always has.

use crate::state::{valid_res, ShapeState, RES_DRAG};

/// How carefully a shape is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// full detail
    Full,
    /// a slider is being dragged
    Quick,
    /// one frame of live animation (or a handle being dragged)
    Anim,
    /// one frame of a video: full detail, nothing approximated
    Hq,
}

/// What a build is good for: shown while something moves, or the settled shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quality {
    Draft,
    Final,
}

impl Quality {
    /// The word a cache key records.
    pub fn tag(self) -> &'static str {
        match self {
            Quality::Draft => "draft",
            Quality::Final => "final",
        }
    }
    pub fn is_draft(self) -> bool {
        self == Quality::Draft
    }
}

impl Detail {
    /// Draft (Quick, Anim) or final (Full, Hq).
    pub fn quality(self) -> Quality {
        match self {
            Detail::Quick | Detail::Anim => Quality::Draft,
            Detail::Full | Detail::Hq => Quality::Final,
        }
    }
    /// A cheap build, shown only while something moves.
    pub fn is_draft(self) -> bool {
        self.quality() == Quality::Draft
    }
    /// The settled shape: exact, at the still detail.
    pub fn is_final(self) -> bool {
        self.quality() == Quality::Final
    }
    /// One frame of an animation, live or a video's (Anim, Hq). A clock frame keeps the shape's face-on
    /// turn from the last still build, so it does not swing about, and does not measure the glass's
    /// thickness afresh.
    pub fn is_clock_frame(self) -> bool {
        matches!(self, Detail::Anim | Detail::Hq)
    }
    /// A played frame (Anim): rebuilt every frame, so the tube cell index is kept coarse (density 3.5
    /// against 4.5; see plans/playback_regression_fix.md).
    pub fn is_played(self) -> bool {
        self == Detail::Anim
    }
    /// Whether the parts' boxes are measured afresh (not for a played frame: the last ones stand).
    pub fn measures_boxes(self) -> bool {
        self != Detail::Anim
    }
    /// Whether the build is the one a blocking caller waits for and times to the end (Full only: a
    /// video frame is waited for by the video itself).
    pub fn is_settle(self) -> bool {
        self == Detail::Full
    }
    /// Whether an imported model keeps its base volume between builds (all but Full).
    pub fn keeps_model_base(self) -> bool {
        self != Detail::Full
    }
    /// The most voxels a side of a grown stage (Full: 1024; every other build 512, a video frame
    /// included, as it has been).
    pub fn box_cap(self) -> f64 {
        if self == Detail::Full { crate::layers::BOX_RES_FULL } else { crate::layers::BOX_RES_ANIM }
    }
    /// Voxels per side a shape is built at, this carefully: the one place this is decided.
    /// `anim_cap` is frame-rate keeping's cap for played frames.
    pub fn res(self, st: &ShapeState, anim_cap: Option<u32>) -> u32 {
        let asked = self.asked_res(st);
        let res = valid_res(if self == Detail::Anim { asked.min(anim_cap.unwrap_or(u32::MAX)) } else { asked } as f64);
        Self::studio_res(res, st)
    }
    /// The voxels a side the shape's own settings ask for, this carefully (before frame-rate keeping,
    /// the studio's largest and the memory budget).
    pub fn asked_res(self, st: &ShapeState) -> u32 {
        match self {
            Detail::Anim => st.anim_res.min(st.res),
            Detail::Quick => RES_DRAG.min(st.res),
            Detail::Full | Detail::Hq => st.res,
        }
    }
    /// (the surface studio works on volumes up to its own largest)
    pub fn studio_res(res: u32, st: &ShapeState) -> u32 {
        if !st.mods.any_live() { res } else { res.min(crate::sculpt::RES_MAX) }
    }
    /// The detail a command line names (`detail=anim|quick|hq`, anything else full).
    pub fn parse(s: Option<&str>) -> Detail {
        match s {
            Some("anim") => Detail::Anim,
            Some("quick") => Detail::Quick,
            Some("hq") => Detail::Hq,
            _ => Detail::Full,
        }
    }
}

/// Why a build was asked for (kept with the request, for the record and for checks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// the shape, or how it is built, changed
    Edit,
    /// a slider was dragged, or let go
    Slider,
    /// a handle on the view (or a part's row in the tree panel) was dragged, or let go
    Handle,
    /// a played frame moved something the shape is made from
    Played,
    /// the animation was paused
    Paused,
    /// the clock was set by hand (a still or a video frame at a time)
    Clock,
    /// a video ended and everything was put back
    VideoDone,
    /// the stage took a new size
    Stage,
    /// lighting needs the bodies' volume made
    Lighting,
    /// a model finished reading
    ModelReady,
    /// the studio's pre-filter is out of date for the view
    Refilter,
    /// a staged refinement stopped short, or finished: the plain build takes over
    Refinement,
    /// a build could not finish (what it was to reuse had gone): the same again
    Again,
    /// a command line, a script, a bench or a check
    Script,
}

/// A build asked for: how carefully, and why. Made only through the constructors, one per cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildRequest {
    pub detail: Detail,
    pub cause: Cause,
}

impl BuildRequest {
    const fn new(detail: Detail, cause: Cause) -> Self {
        Self { detail, cause }
    }
    /// The shape, or how it is built, changed: build it again, settled.
    pub const fn edit() -> Self {
        Self::new(Detail::Full, Cause::Edit)
    }
    /// A slider: a quick draft while it is held, final once let go.
    pub const fn slider(held: bool) -> Self {
        Self::new(if held { Detail::Quick } else { Detail::Full }, Cause::Slider)
    }
    /// A handle on the view, or a part's row in the tree panel: drafted as a played frame is while
    /// held (moving detail, boxes not measured), final once let go.
    pub const fn handle(held: bool) -> Self {
        Self::new(if held { Detail::Anim } else { Detail::Full }, Cause::Handle)
    }
    /// A played frame changed the shape.
    pub const fn played() -> Self {
        Self::new(Detail::Anim, Cause::Played)
    }
    /// The animation was paused: the shape as it stands, settled.
    pub const fn paused() -> Self {
        Self::new(Detail::Full, Cause::Paused)
    }
    /// The clock was set by hand: a video frame (`hq`) or a played frame.
    pub const fn clock(hq: bool) -> Self {
        Self::new(if hq { Detail::Hq } else { Detail::Anim }, Cause::Clock)
    }
    /// A video ended: everything is back as it was, settled.
    pub const fn video_done() -> Self {
        Self::new(Detail::Full, Cause::VideoDone)
    }
    /// The stage took a new size: drafted while playing, else settled.
    pub const fn stage(playing: bool) -> Self {
        Self::new(if playing { Detail::Anim } else { Detail::Full }, Cause::Stage)
    }
    /// Lighting needs the bodies' volume.
    pub const fn lighting() -> Self {
        Self::new(Detail::Full, Cause::Lighting)
    }
    /// A model finished reading.
    pub const fn model_ready() -> Self {
        Self::new(Detail::Full, Cause::ModelReady)
    }
    /// The studio's pre-filter is out of date for the view.
    pub const fn refilter() -> Self {
        Self::new(Detail::Full, Cause::Refilter)
    }
    /// A staged refinement stopped short or finished: the plain build takes over.
    pub const fn refinement() -> Self {
        Self::new(Detail::Full, Cause::Refinement)
    }
    /// A build could not finish: the same again next frame.
    pub const fn again(detail: Detail) -> Self {
        Self::new(detail, Cause::Again)
    }
    /// A command line, script, bench or check asks for this detail.
    pub const fn script(detail: Detail) -> Self {
        Self::new(detail, Cause::Script)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_are_the_moving_builds_and_finals_the_settled_ones() {
        assert_eq!([Detail::Quick, Detail::Anim].map(Detail::quality), [Quality::Draft; 2]);
        assert_eq!([Detail::Full, Detail::Hq].map(Detail::quality), [Quality::Final; 2]);
        // every held gesture drafts, every release settles
        for held in [true, false] {
            for r in [BuildRequest::slider(held), BuildRequest::handle(held)] {
                assert_eq!(r.detail.is_draft(), held, "{r:?}");
            }
        }
        assert!(BuildRequest::played().detail.is_draft() && BuildRequest::paused().detail.is_final());
        assert!(BuildRequest::clock(true).detail.is_final() && BuildRequest::clock(false).detail.is_draft());
    }

    #[test]
    fn the_command_line_names_every_detail() {
        for (s, d) in [("anim", Detail::Anim), ("quick", Detail::Quick), ("hq", Detail::Hq), ("full", Detail::Full)] {
            assert_eq!(Detail::parse(Some(s)), d);
        }
        assert_eq!(Detail::parse(None), Detail::Full);
    }
}
