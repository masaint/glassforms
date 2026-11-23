//! The quality tiers and which one draws each frame (docs: plans/app_integration_plan.md 1.4 and 2; the
//! rules were measured in the lab: plans/lab_tiered.md).
//!
//! This file is the policy only, on the CPU: no engine, no GPU. The window says what is happening
//! (`PhaseTracker`: moving, the release, at rest), what the machine and the scene can do (`Avail`), what the
//! owner chose (`TierChoice`, remembered in the window's own file); `decide` answers with the tier that draws
//! this frame. Nothing here changes a picture: until the engine asks `decide`, every frame is drawn as before.
//!
//! The tiers:
//!   matcap  the first surface shaded by a matcap (no glass)
//!   draft   soft glass from a quarter of the pixels, cleaned up by Ray Reconstruction (NVIDIA, optional build)
//!   live    glass drawn from the parts engine, the dragged part live
//!   full    the full render as the program has always drawn it (the default)
//!
//! The hand-over rules (1.4, each tested below):
//!   1. a tier switch happens only at a frame boundary: `decide` is called once a frame;
//!   2. motion starts on the frame the step arrives: that frame is already the motion tier;
//!   3. motion ends on the release frame, which the REST tier draws (it holds the rebake); after a turn
//!      the first idle frame is that hand-over frame;
//!   4. the rest tier's accumulation starts again on every motion step (the tracer's own rule; the engine
//!      does it, `Decision::restart` says so);
//!   5. glass-to-glass motion tiers are gated: a measured pop over twice the settle noise falls back to the
//!      rest tier, with the reason said;
//!   6. matcap to glass is a change of look, reported (`Decision::look_change`), never gated;
//!   7. every fallback carries its reason, never silent;
//!   8. the settled rest picture is the rest tier's alone (a motion tier never draws at rest).

use serde::{Deserialize, Serialize};

/// What draws a frame, from the quickest look to the fullest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Matcap,
    Draft,
    Live,
    Full,
}
pub const TIERS: [Tier; 4] = [Tier::Matcap, Tier::Draft, Tier::Live, Tier::Full];

impl Tier {
    /// The name on the command line and in the window's file.
    pub fn id(self) -> &'static str {
        match self {
            Tier::Matcap => "matcap",
            Tier::Draft => "draft",
            Tier::Live => "live",
            Tier::Full => "full",
        }
    }
    /// The words the row and the foot line use.
    pub fn label(self) -> &'static str {
        match self {
            Tier::Matcap => "matcap",
            Tier::Draft => "draft",
            Tier::Live => "live glass",
            Tier::Full => "full render",
        }
    }
    pub fn parse(s: &str) -> Option<Tier> {
        TIERS.iter().copied().find(|t| t.id() == s)
    }
    /// Does it draw glass? (A matcap picture is another look.)
    pub fn glass(self) -> bool {
        self != Tier::Matcap
    }
    fn at(self) -> usize {
        TIERS.iter().position(|t| *t == self).unwrap_or(0)
    }
}

/// What draws the moving frames under Auto.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Motion {
    /// the chosen tier's own moving frames
    #[default]
    Same,
    /// the draft (only where it is available)
    Draft,
    /// matcap: a change of look when you let go
    Matcap,
}
impl Motion {
    pub fn id(self) -> &'static str {
        match self {
            Motion::Same => "same",
            Motion::Draft => "draft",
            Motion::Matcap => "matcap",
        }
    }
    pub fn parse(s: &str) -> Option<Motion> {
        [Motion::Same, Motion::Draft, Motion::Matcap].into_iter().find(|m| m.id() == s)
    }
}

/// What the owner chose: the tier at rest (the lit button), whether Auto is on, and what draws while moving.
/// A choice of this machine and of viewing, not of the scene: kept in the window's own file, never in a
/// setup, never in undo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TierChoice {
    pub rest: Tier,
    pub auto: bool,
    pub motion: Motion,
}
impl Default for TierChoice {
    /// The full render, no Auto: the program draws as it always has.
    fn default() -> Self {
        TierChoice { rest: Tier::Full, auto: false, motion: Motion::Same }
    }
}

/// `tier=matcap|draft|live|full`, `tierAuto=0|1`, `tierMotion=same|draft|matcap` from the command line, laid
/// over `base` (None when none was given). `get` looks a name up.
pub fn from_args<'a>(base: TierChoice, get: impl Fn(&str) -> Option<&'a str>) -> Result<Option<TierChoice>, String> {
    let (tier, auto, motion) = (get("tier"), get("tierAuto"), get("tierMotion"));
    if tier.is_none() && auto.is_none() && motion.is_none() {
        return Ok(None);
    }
    let mut c = base;
    if let Some(t) = tier {
        c.rest = Tier::parse(t).ok_or_else(|| format!("Unknown tier={t}. Choose matcap, draft, live or full."))?;
    }
    if let Some(a) = auto {
        c.auto = match a {
            "1" | "on" | "true" => true,
            "0" | "off" | "false" => false,
            _ => return Err(format!("Unknown tierAuto={a}. Choose 0 or 1.")),
        };
    }
    if let Some(m) = motion {
        c.motion = Motion::parse(m).ok_or_else(|| format!("Unknown tierMotion={m}. Choose same, draft or matcap."))?;
    }
    Ok(Some(c))
}

/// What the Ray Reconstruction draft needs of this build and machine (plan 1.6); every condition must hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftFacts {
    /// the build has the cargo feature `dlss` (the draft is a runtime switch on it, not a feature of its own)
    pub built: bool,
    /// the window's device was opened through dlss_wgpu, on an NVIDIA adapter on Vulkan
    pub device: bool,
    /// the SDK reports `ray_reconstruction_supported`
    pub supported: bool,
    /// nvngx_dlssd.dll and nvngx_dlss.dll sit beside the program
    pub dlls: bool,
}
impl DraftFacts {
    /// Is the `dlss` cargo feature in this build? (A default build: no.)
    pub const fn built_in() -> bool {
        cfg!(feature = "dlss")
    }
    /// Ok, or the first thing that is missing, in the words the tooltip uses.
    pub fn check(&self) -> Result<(), String> {
        if !self.built {
            Err("this build has no Ray Reconstruction".into())
        } else if !self.device {
            Err("needs an NVIDIA graphics card on Vulkan, opened for Ray Reconstruction at start".into())
        } else if !self.supported {
            Err("this graphics card does not support Ray Reconstruction".into())
        } else if !self.dlls {
            Err("the NVIDIA runtime files are not beside the program".into())
        } else {
            Ok(())
        }
    }
}

/// Does the parts engine accept the scene? (`gf_core::parts::accepts` once it exists: WP1.) Err: the reason.
pub trait Accepts {
    fn accepts(&self) -> Result<(), String>;
}
/// Until the parts engine is in: it accepts nothing.
pub struct NoParts;
impl Accepts for NoParts {
    fn accepts(&self) -> Result<(), String> {
        Err("the parts engine is not in this build".into())
    }
}
impl<F: Fn() -> Result<(), String>> Accepts for F {
    fn accepts(&self) -> Result<(), String> {
        self()
    }
}

/// How a button of the row is shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shown {
    Normal,
    /// drawn dimmed, with the reason in its tooltip (it may still be chosen: the full render's path draws it)
    Dimmed(String),
    /// not in the row (the row closes up)
    Hidden,
}

/// What this machine and this scene can do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Avail {
    pub draft: Result<(), String>,
    pub live: Result<(), String>,
}
impl Avail {
    pub fn probe(draft: &DraftFacts, live: &dyn Accepts) -> Avail {
        Avail { draft: draft.check(), live: live.accepts() }
    }
    /// A machine that has everything (tests, and a build that wants every tier).
    pub fn all() -> Avail {
        Avail { draft: Ok(()), live: Ok(()) }
    }
    pub fn shown(&self, t: Tier) -> Shown {
        match t {
            Tier::Matcap | Tier::Full => Shown::Normal,
            Tier::Draft => if self.draft.is_ok() { Shown::Normal } else { Shown::Hidden },
            Tier::Live => match &self.live {
                Ok(()) => Shown::Normal,
                Err(why) => Shown::Dimmed(why.clone()),
            },
        }
    }
    /// The tiers in the row, in order.
    pub fn row(&self) -> Vec<(Tier, Shown)> {
        TIERS.iter().map(|t| (*t, self.shown(*t))).filter(|(_, s)| *s != Shown::Hidden).collect()
    }
    /// The tier after `from` in the row (shift+Z), skipping hidden ones.
    pub fn next(&self, from: Tier) -> Tier {
        let row = self.row();
        let i = row.iter().position(|(t, _)| *t == from);
        match i {
            Some(i) => row[(i + 1) % row.len()].0,
            None => row.iter().map(|(t, _)| *t).find(|t| t.at() > from.at()).unwrap_or(Tier::Matcap),
        }
    }
}

/// What is happening, as the window sees it each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// the view or the shape is being moved (a turn, orbit, fly, a handle or slider drag, playing)
    Moving,
    /// the frame after the last moving one: a drag's release, or the first idle frame after a turn
    Release,
    /// nothing is moving
    Rest,
}

/// Turns "is something moving this frame" into phases and counts the rest frames since.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTracker {
    was_moving: bool,
    pub rest_frames: u32,
}
impl PhaseTracker {
    pub fn next(&mut self, moving: bool) -> Phase {
        let p = if moving {
            self.rest_frames = 0;
            Phase::Moving
        } else if self.was_moving {
            self.rest_frames = 0;
            Phase::Release
        } else {
            self.rest_frames = self.rest_frames.saturating_add(1);
            Phase::Rest
        };
        self.was_moving = moving;
        p
    }
}

/// The measured hand-over of a glass motion tier into glass: pop over the settle noise (the lab's gate: 2).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PopGate {
    pub ratio: Option<f64>,
}
pub const POP_LINE: f64 = 2.0;
impl PopGate {
    /// Unknown passes (nothing measured against it yet); a measured ratio over the line does not.
    pub fn ok(&self) -> bool {
        self.ratio.is_none_or(|r| r <= POP_LINE)
    }
}

/// What the window knows this frame.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub phase: Phase,
    /// the tier that drew the previous frame (None: the first frame)
    pub prev: Option<Tier>,
    /// at rest: the picture has settled, nothing more to draw
    pub settled: bool,
    /// the draft's context is built (it costs 0.9 s the first time: build it when the tier is chosen)
    pub draft_ready: bool,
    pub pop: PopGate,
}
impl Frame {
    pub fn new(phase: Phase) -> Frame {
        Frame { phase, prev: None, settled: false, draft_ready: true, pop: PopGate::default() }
    }
}

/// Why a frame was not drawn by what was asked for, in words for the dropdown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fallback(pub String);

/// The answer: what draws this frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub tier: Tier,
    /// false: settled at rest, nothing is drawn (the picture on screen stands)
    pub draws: bool,
    /// this frame's tier differs from the last frame's
    pub switched: bool,
    /// the switch is between matcap and glass: another look, reported not gated
    pub look_change: bool,
    /// the tier's accumulation starts again (a moving frame)
    pub restart: bool,
    pub fallback: Option<Fallback>,
}

/// The rest tier as chosen, made real: a draft that cannot run is the live glass.
pub fn rest_tier(choice: &TierChoice, avail: &Avail) -> (Tier, Option<Fallback>) {
    if choice.rest == Tier::Draft {
        if let Err(why) = &avail.draft {
            return (Tier::Live, Some(Fallback(format!("draft unavailable: {why}"))));
        }
    }
    (choice.rest, None)
}

/// Which tier draws this frame. Pure: the same inputs, the same answer.
pub fn decide(choice: &TierChoice, avail: &Avail, f: &Frame) -> Decision {
    let (rest, mut fallback) = rest_tier(choice, avail);
    let tier = match f.phase {
        // (3: the release frame, and after a turn the first idle frame, belong to the rest tier)
        Phase::Release | Phase::Rest => rest,
        Phase::Moving => {
            if !choice.auto {
                rest
            } else {
                match choice.motion {
                    Motion::Same => rest,
                    // (a matcap rest tier is already as quick as a tier gets: nothing to draw instead)
                    Motion::Draft if rest == Tier::Matcap => rest,
                    Motion::Draft => match (&avail.draft, f.draft_ready, f.pop.ok()) {
                        (Err(why), _, _) => {
                            fallback = Some(Fallback(format!("draft unavailable: {why}")));
                            rest
                        }
                        (Ok(()), false, _) => {
                            fallback = Some(Fallback("draft is still starting".into()));
                            rest
                        }
                        (Ok(()), true, false) => {
                            fallback = Some(Fallback(format!("draft to glass is over {POP_LINE} times the settle noise here: drawn by {}", rest.label())));
                            rest
                        }
                        (Ok(()), true, true) => Tier::Draft,
                    },
                    Motion::Matcap => Tier::Matcap,
                }
            }
        }
    };
    let moving = f.phase == Phase::Moving;
    Decision {
        tier,
        draws: moving || f.phase == Phase::Release || !f.settled,
        switched: f.prev.is_some_and(|p| p != tier),
        look_change: f.prev.is_some_and(|p| p != tier && p.glass() != tier.glass()),
        restart: moving,
        fallback,
    }
}

/// How many frames of each tier's GPU time are kept.
pub const RING: usize = 120;

/// The last `RING` GPU times of one kind of frame.
#[derive(Clone, Debug, Default)]
pub struct Ring {
    v: Vec<f32>,
    at: usize,
}
impl Ring {
    pub fn push(&mut self, ms: f64) {
        if self.v.len() < RING {
            self.v.push(ms as f32);
        } else {
            self.v[self.at] = ms as f32;
            self.at = (self.at + 1) % RING;
        }
    }
    pub fn len(&self) -> usize {
        self.v.len()
    }
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    /// The p-th share (0..1) by the nearest rank, as the lab and the app's tools take it.
    pub fn percentile(&self, q: f64) -> Option<f64> {
        if self.v.is_empty() {
            return None;
        }
        let mut s = self.v.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        Some(s[((q * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)] as f64)
    }
    pub fn p80(&self) -> Option<f64> {
        self.percentile(0.8)
    }
    pub fn max(&self) -> Option<f64> {
        self.percentile(1.0)
    }
}

/// The GPU times by tier, moving and at rest, for the dropdown and the foot line.
#[derive(Clone, Debug, Default)]
pub struct Times {
    moving: [Ring; 4],
    rest: [Ring; 4],
}
impl Times {
    pub fn push(&mut self, tier: Tier, moving: bool, ms: f64) {
        let r = if moving { &mut self.moving } else { &mut self.rest };
        r[tier.at()].push(ms);
    }
    pub fn ring(&self, tier: Tier, moving: bool) -> &Ring {
        if moving { &self.moving[tier.at()] } else { &self.rest[tier.at()] }
    }
    /// "live glass 9.6 ms" for the foot line (the p80 of the frames of the kind just drawn), None before any.
    pub fn foot(&self, tier: Tier, moving: bool) -> Option<String> {
        self.ring(tier, moving).p80().map(|ms| format!("{} {:.1} ms", tier.label(), ms))
    }
    /// One line of the dropdown: "live glass 9.6 ms (moving) · 9.0 ms (rest)", None for a tier not drawn yet.
    pub fn line(&self, tier: Tier) -> Option<String> {
        let (m, r) = (self.moving[tier.at()].p80(), self.rest[tier.at()].p80());
        if m.is_none() && r.is_none() {
            return None;
        }
        let part = |v: Option<f64>, what: &str| v.map_or_else(|| format!("\u{2014} ({what})"), |ms| format!("{ms:.1} ms ({what})"));
        Some(format!("{} {} \u{b7} {}", tier.label(), part(m, "moving"), part(r, "rest")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glass_auto(motion: Motion) -> TierChoice {
        TierChoice { rest: Tier::Live, auto: true, motion }
    }

    /// Drive a session: `moving` per frame, with the settling of the picture counted as the engine would
    /// (settled after `depth` rest frames); returns the decisions.
    fn session(choice: TierChoice, avail: &Avail, moving: &[bool], depth: u32, draft_ready: bool, pop: PopGate) -> Vec<Decision> {
        let mut tr = PhaseTracker::default();
        let mut prev = None;
        let mut out = Vec::new();
        for &m in moving {
            let phase = tr.next(m);
            let f = Frame { phase, prev, settled: phase == Phase::Rest && tr.rest_frames > depth, draft_ready, pop };
            let d = decide(&choice, avail, &f);
            prev = Some(d.tier);
            out.push(d);
        }
        out
    }
    fn tiers(d: &[Decision]) -> Vec<Tier> {
        d.iter().map(|d| d.tier).collect()
    }
    const L: Tier = Tier::Live;
    const D: Tier = Tier::Draft;

    #[test]
    fn a_default_window_draws_the_full_render_always() {
        let c = TierChoice::default();
        assert_eq!(c, TierChoice { rest: Tier::Full, auto: false, motion: Motion::Same });
        let d = session(c, &Avail::all(), &[false, true, true, false, false, true], 30, true, PopGate::default());
        assert!(d.iter().all(|d| d.tier == Tier::Full && !d.switched && d.fallback.is_none()));
    }

    #[test]
    fn a_session_of_a_drag_a_release_a_rest_and_a_turn() {
        // rest 2, drag 3, release (drag over), rest 2, turn 2, idle (hand-over), rest 1
        let m = [false, false, true, true, true, false, false, false, true, true, false, false];
        let d = session(glass_auto(Motion::Draft), &Avail::all(), &m, 30, true, PopGate::default());
        assert_eq!(tiers(&d), vec![L, L, D, D, D, L, L, L, D, D, L, L]);
        // motion starts on the frame the step arrives (2): already the draft, switched, and it restarts the accumulation
        assert!(d[2].switched && d[2].restart && !d[2].look_change);
        assert!(d[3].restart && !d[3].switched);
        // the release frame (5) is the rest tier's, and so is the first idle frame after the turn (10)
        assert!(d[5].switched && !d[5].restart && d[5].draws);
        assert!(d[10].switched && !d[10].restart);
        // a glass-to-glass hand-over is no change of look
        assert!(d.iter().all(|d| !d.look_change));
    }

    #[test]
    fn the_rest_tier_stops_drawing_once_the_picture_has_settled() {
        let m = [true, false, false, false, false, false, false, true, false];
        let d = session(glass_auto(Motion::Same), &Avail::all(), &m, 3, true, PopGate::default());
        // moving, release, then rest frames 1..3 draw, 4.. do not; moving draws again, and so does its release
        let draws: Vec<bool> = d.iter().map(|d| d.draws).collect();
        assert_eq!(draws, vec![true, true, true, true, true, false, false, true, true]);
        // the tier stays the rest tier all the while: nothing else ever draws at rest
        assert!(d.iter().all(|d| d.tier == L));
    }

    #[test]
    fn matcap_while_moving_is_a_change_of_look_not_a_gated_pop() {
        let over = PopGate { ratio: Some(26.0) };
        let m = [false, true, true, false, false];
        let d = session(glass_auto(Motion::Matcap), &Avail::all(), &m, 30, true, over);
        assert_eq!(tiers(&d), vec![L, Tier::Matcap, Tier::Matcap, L, L]);
        assert!(d[1].look_change && d[3].look_change && !d[2].look_change && !d[4].look_change);
        assert!(d.iter().all(|d| d.fallback.is_none()), "the pop gate is for glass hand-overs only");
    }

    #[test]
    fn a_draft_over_the_pop_gate_falls_back_and_says_why() {
        let over = PopGate { ratio: Some(3.2) };
        let d = session(glass_auto(Motion::Draft), &Avail::all(), &[true, true, false], 30, true, over);
        assert_eq!(tiers(&d), vec![L, L, L]);
        let why = d[0].fallback.clone().unwrap().0;
        assert!(why.contains("settle noise") && why.contains("live glass"), "{why}");
        // at the line itself it passes; unmeasured passes
        assert_eq!(session(glass_auto(Motion::Draft), &Avail::all(), &[true], 30, true, PopGate { ratio: Some(2.0) })[0].tier, D);
        assert_eq!(session(glass_auto(Motion::Draft), &Avail::all(), &[true], 30, true, PopGate::default())[0].tier, D);
    }

    #[test]
    fn a_draft_that_is_unavailable_or_not_started_is_never_silent() {
        let none = Avail { draft: DraftFacts::default().check(), live: Ok(()) };
        let d = session(glass_auto(Motion::Draft), &none, &[true, false], 30, true, PopGate::default());
        assert_eq!(tiers(&d), vec![L, L]);
        assert!(d[0].fallback.as_ref().unwrap().0.starts_with("draft unavailable: this build has no Ray Reconstruction"));
        // available, but the context is not built yet: the rest tier draws, and says so
        let d = session(glass_auto(Motion::Draft), &Avail::all(), &[true], 30, false, PopGate::default());
        assert_eq!((d[0].tier, d[0].fallback.clone().map(|f| f.0)), (L, Some("draft is still starting".to_string())));
        // chosen as the rest tier without being there: the live glass
        let c = TierChoice { rest: D, auto: false, motion: Motion::Same };
        let d = session(c, &none, &[false], 30, true, PopGate::default());
        assert_eq!(d[0].tier, L);
        assert!(d[0].fallback.is_some());
    }

    #[test]
    fn without_auto_moving_frames_are_the_chosen_tiers_own() {
        let c = TierChoice { rest: Tier::Full, auto: false, motion: Motion::Draft };
        let d = session(c, &Avail::all(), &[false, true, true, false], 30, true, PopGate::default());
        assert_eq!(tiers(&d), vec![Tier::Full; 4]);
        // a matcap rest tier stays matcap whatever the motion tier says
        let c = TierChoice { rest: Tier::Matcap, auto: true, motion: Motion::Draft };
        let d = session(c, &Avail::all(), &[true, false], 30, true, PopGate::default());
        assert_eq!(tiers(&d), vec![Tier::Matcap; 2]);
    }

    #[test]
    fn a_turn_then_idle_hands_over_on_the_first_idle_frame() {
        let c = TierChoice { rest: Tier::Full, auto: true, motion: Motion::Draft };
        let d = session(c, &Avail::all(), &[true, true, true, false, false], 30, true, PopGate::default());
        assert_eq!(tiers(&d), vec![D, D, D, Tier::Full, Tier::Full]);
        assert_eq!(d.iter().map(|d| d.switched).collect::<Vec<_>>(), vec![false, false, false, true, false]);
        // (the first frame has no earlier one: no switch)
    }

    #[test]
    fn the_phase_tracker_counts_rest_frames_from_the_release() {
        let mut t = PhaseTracker::default();
        let p: Vec<Phase> = [false, true, false, false, false, true].iter().map(|m| t.next(*m)).collect();
        assert_eq!(p, vec![Phase::Rest, Phase::Moving, Phase::Release, Phase::Rest, Phase::Rest, Phase::Moving]);
        let mut t = PhaseTracker::default();
        t.next(true);
        t.next(false);
        assert_eq!(t.rest_frames, 0);
        t.next(false);
        t.next(false);
        assert_eq!(t.rest_frames, 2);
    }

    #[test]
    fn the_row_hides_the_draft_dims_the_live_glass_and_steps_with_shift_z() {
        let none = Avail { draft: DraftFacts::default().check(), live: Err("a model is in the scene".into()) };
        let row = none.row();
        assert_eq!(row.iter().map(|(t, _)| *t).collect::<Vec<_>>(), vec![Tier::Matcap, L, Tier::Full]);
        assert_eq!(none.shown(L), Shown::Dimmed("a model is in the scene".into()));
        assert_eq!(none.shown(D), Shown::Hidden);
        assert_eq!(none.next(Tier::Matcap), L);
        assert_eq!(none.next(L), Tier::Full);
        assert_eq!(none.next(Tier::Full), Tier::Matcap);
        // from a hidden tier: the next one that is shown
        assert_eq!(none.next(D), L);
        let all = Avail::all();
        assert_eq!(all.row().len(), 4);
        assert_eq!(all.next(Tier::Matcap), D);
        // the default build has no draft
        assert!(!DraftFacts::built_in());
        assert_eq!(Avail::probe(&DraftFacts { built: false, ..Default::default() }, &NoParts).shown(D), Shown::Hidden);
        let ok = DraftFacts { built: true, device: true, supported: true, dlls: true };
        assert_eq!(Avail::probe(&ok, &|| Ok(())), Avail::all());
        assert!(DraftFacts { dlls: false, ..ok }.check().unwrap_err().contains("beside the program"));
        assert!(DraftFacts { device: false, ..ok }.check().unwrap_err().contains("NVIDIA"));
    }

    #[test]
    fn the_rings_keep_the_last_frames_and_give_the_p80() {
        let mut t = Times::default();
        assert_eq!(t.line(L), None);
        for i in 1..=10 {
            t.push(L, true, i as f64);
        }
        t.push(L, false, 9.0);
        assert_eq!(t.ring(L, true).p80(), Some(8.0));
        assert_eq!(t.foot(L, true).as_deref(), Some("live glass 8.0 ms"));
        assert_eq!(t.line(L).as_deref(), Some("live glass 8.0 ms (moving) \u{b7} 9.0 ms (rest)"));
        assert_eq!(t.line(D), None);
        t.push(D, true, 5.8);
        assert!(t.line(D).unwrap().contains("\u{2014} (rest)"));
        // only the last RING frames count
        let mut r = Ring::default();
        for i in 0..(RING + 50) {
            r.push(i as f64);
        }
        assert_eq!((r.len(), r.max()), (RING, Some((RING + 49) as f64)));
        assert_eq!(r.percentile(0.0), Some(50.0));
    }

    #[test]
    fn the_command_line_names_a_tier_auto_and_a_motion_tier() {
        let args = |p: &'static [(&'static str, &'static str)]| move |k: &str| p.iter().find(|x| x.0 == k).map(|x| x.1);
        let base = TierChoice::default();
        assert_eq!(from_args(base, args(&[])).unwrap(), None);
        let c = from_args(base, args(&[("tier", "live"), ("tierAuto", "1"), ("tierMotion", "draft")])).unwrap().unwrap();
        assert_eq!(c, TierChoice { rest: L, auto: true, motion: Motion::Draft });
        // each alone is laid over what was remembered
        let c = from_args(c, args(&[("tierAuto", "0")])).unwrap().unwrap();
        assert_eq!(c, TierChoice { rest: L, auto: false, motion: Motion::Draft });
        assert!(from_args(base, args(&[("tier", "clay")])).unwrap_err().contains("Choose matcap, draft, live or full"));
        assert!(from_args(base, args(&[("tierAuto", "2")])).is_err());
        assert!(from_args(base, args(&[("tierMotion", "fast")])).is_err());
    }

    #[test]
    fn the_choice_round_trips_and_an_old_file_says_nothing_of_it() {
        let c = TierChoice { rest: Tier::Draft, auto: true, motion: Motion::Matcap };
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(s, r#"{"rest":"draft","auto":true,"motion":"matcap"}"#);
        assert_eq!(serde_json::from_str::<TierChoice>(&s).unwrap(), c);
        // a partial or empty object takes the defaults
        assert_eq!(serde_json::from_str::<TierChoice>(r#"{"auto":true}"#).unwrap(), TierChoice { auto: true, ..Default::default() });
        assert_eq!(serde_json::from_str::<TierChoice>("{}").unwrap(), TierChoice::default());
        // inside the window's file (ui/store.rs)
        let st = crate::ui::store::Store { tiers: c, ..Default::default() };
        let again = crate::ui::store::Store::read(&serde_json::to_string(&st).unwrap());
        assert_eq!(again.tiers, c);
        assert_eq!(crate::ui::store::Store::read(r#"{ "closed": [] }"#).tiers, TierChoice::default());
    }
}
