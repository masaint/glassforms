//! Undo and redo, and the ten camera slots.
//!
//! **What an entry is.** The design manual this follows (its "transactions") makes every change an object
//! carrying old and new values, with the gesture as the unit: a whole drag is one entry. Here an entry
//! is a snapshot of the setup instead: the same `Setup` a file keeps (every slider, the shapes, the layers
//! and their joins, the surface studio, the bodies' materials, the lights, the lamps, the camera, what is
//! animated), taken at the quiet moment after a gesture.
//!
//! Why snapshots, not commands. The program changes its state in hundreds of places, most of them
//! direct writes from the panels (`st.p.dof = ...`, a lamp's kind, a studio's lights). Commands would only
//! be as complete as the places that remembered to make one, and a missed place is an undo that silently
//! does nothing, or worse, undoes the wrong thing. A snapshot is complete by construction: whatever a
//! setup keeps, an undo puts back, so it can never fall out of step with the program. What it costs: a
//! setup is a few kilobytes to a few tens (most of it the numbers of the shapes and their tracks), so the
//! 200 entries kept cost a few megabytes. It keeps the manual's point that history is data: an entry is
//! exactly what a setup file holds, and could be written out as one.
//!
//! **Gestures.** The window takes a snapshot at the quiet moment after input (no button down, nothing
//! typed or turned for a moment) and at every press. A change between two of them is one entry, however
//! many frames a drag lasted. Esc while a button is down puts back the snapshot taken at the press.
//!
//! **What an entry leaves out.** It leaves out what moves by itself: the animation's clock and what
//! the animation writes into the animated sliders, the spin's phase, and the lights' own circling.
//! The snapshot is taken with those set aside, so playing never makes an entry, and an undo leaves the
//! animation running where it is. The camera slots are left out as well (storing one is not undone).

use serde::{Deserialize, Serialize};

use crate::setup::Setup;
use crate::state::{Rig, State};

/// How many entries are kept.
pub const CAP: usize = 200;

/// The settings that make up a view: where the camera stands and which way it looks (orbiting or flying),
/// the lens and where it is focused.
pub const VIEW_NAMES: &[&str] = &[
    "dist", "eyeX", "eyeY", "pivotX", "pivotY", "pivotZ", "pivotR", "azim", "elev", "roll", "shiftX", "shiftY",
    "proj", "fov", "focal", "sensor", "aperture", "fstop", "fnear", "ffar", "tiltX", "tiltY",
];
/// How many camera slots there are.
pub const SLOTS: usize = 10;

/// One stored view: those settings by name, and how the shape is turned in front of the camera.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewSlot {
    pub values: std::collections::BTreeMap<String, f64>,
    pub orientation: [f64; 9],
}

impl State {
    /// The setup as an undo entry keeps it (see the head of this file): what moves by itself is set aside.
    pub fn undo_snapshot(&mut self) -> Setup {
        // the animated sliders at a fixed place for the moment (their lowest), the clock and the spin out
        let active = self.active();
        let keep: Vec<f64> = active.iter().map(|k| self.get(&k.r)).collect();
        let rev = self.lamps_rev;
        for k in &active { self.set(&k.r, k.lo); }
        let (r, clock, lphi) = (self.r, self.clock, self.lphi);
        (self.r, self.clock, self.lphi) = (self.r0, 0.0, 0.0);
        let mut s = self.setup();
        s.views = Vec::new();
        (self.r, self.clock, self.lphi) = (r, clock, lphi);
        for (k, v) in active.iter().zip(keep) { self.set(&k.r, v); }
        self.lamps_rev = rev;
        s
    }

    /// Put an undo entry back. Where only settings, the lights, the lamps and what is animated differ,
    /// they are put back by themselves (nothing is built again that needs not be); otherwise the setup is
    /// put up whole, as loading it does, with everything that follows. The animation's clock, the spin's
    /// phase, whether it plays and the camera slots stay as they are.
    pub fn restore_snapshot(&mut self, s: &Setup) {
        let (playing, clock, theta, lphi) = (self.playing, self.clock, self.theta, self.lphi);
        // (what the animation has the animated sliders at now, to the last place: worked out again below, it
        // could come out a rounding away)
        let had: std::collections::BTreeMap<String, f64> = self.active().iter().map(|k| (k.key.clone(), self.get(&k.r))).collect();
        let views = std::mem::take(&mut self.views);
        let cur = self.undo_snapshot();
        let same_scene = cur.family == s.family && cur.preset == s.preset && cur.values == s.values && cur.shape == s.shape
            && cur.layers == s.layers && cur.bodies == s.bodies && cur.hdri == s.hdri;
        if same_scene {
            self.p = s.settings.clone();
            crate::camera::sync(&mut self.p);
            self.tracks = s.tracks.clone();
            self.render = s.render.clone();
            if self.lamps != s.lamps {
                self.lamps = s.lamps.clone();
                self.lamps_rev += 1;
            }
            match &s.rig {
                Some(r) => { let r = Rig::from(r); if *self.rig() != r { *self.rig_mut() = r; } }
                None => { if !self.rig_is_default() { self.reset_rig(); } }
            }
            self.cut = true;
        } else {
            // (it was a setup a moment ago: it puts up)
            let _ = self.apply_setup(s);
        }
        self.views = views;
        (self.clock, self.theta, self.lphi) = (clock, theta, lphi);
        self.r0 = crate::math::orthonormalise(&s.orientation);
        self.update_r();
        if playing != self.playing { self.set_playing(playing); }
        // the animated sliders where the animation has them now (as they were, where that is the same)
        self.apply_time();
        for k in self.active() {
            if let Some(v) = had.get(&k.key) {
                if (self.get(&k.r) - v).abs() < 1e-6 { self.set(&k.r, *v); }
            }
        }
    }

    /// Store the view as it stands in slot `i` (0 to 9).
    pub fn store_view(&mut self, i: usize) {
        if i >= SLOTS { return; }
        let values = VIEW_NAMES.iter().filter_map(|n| crate::camera::get(&self.p, n).or_else(|| self.p.get(n)).map(|v| (n.to_string(), v))).collect();
        if self.views.len() < SLOTS { self.views.resize(SLOTS, None); }
        self.views[i] = Some(ViewSlot { values, orientation: self.r });
    }
    /// Go to the view in slot `i`; false if it is empty.
    pub fn recall_view(&mut self, i: usize) -> bool {
        let Some(slot) = self.views.get(i).cloned().flatten() else { return false };
        // (the focal length after the field of view: a lens said both ways keeps the later)
        for n in VIEW_NAMES {
            if let Some(v) = slot.values.get(*n) {
                if !crate::camera::set(&mut self.p, n, *v) { self.p.set(n, *v); }
            }
        }
        crate::camera::sync(&mut self.p);
        self.set_r(&slot.orientation);
        self.cut = true;
        true
    }
    pub fn view_stored(&self, i: usize) -> bool {
        self.views.get(i).is_some_and(|v| v.is_some())
    }
}

/// The undo and redo stacks, and the snapshot of the state as it last settled.
#[derive(Default)]
pub struct History {
    undo: Vec<Setup>,
    redo: Vec<Setup>,
    last: Option<Setup>,
}

impl History {
    /// A quiet moment, or a gesture about to begin: if the state has changed since the last snapshot, that
    /// change becomes one entry. True if it did.
    pub fn settle(&mut self, st: &mut State) -> bool {
        let now = st.undo_snapshot();
        match &self.last {
            Some(l) if *l == now => false,
            None => {
                self.last = Some(now);
                false
            }
            Some(_) => {
                let before = self.last.replace(now).expect("just matched");
                self.undo.push(before);
                if self.undo.len() > CAP { self.undo.remove(0); }
                self.redo.clear();
                true
            }
        }
    }
    /// Esc during a gesture: the state goes back to the snapshot taken when it began.
    pub fn cancel(&mut self, st: &mut State) {
        if let Some(l) = &self.last {
            let l = l.clone();
            if st.undo_snapshot() != l { st.restore_snapshot(&l); }
        }
    }
    pub fn undo(&mut self, st: &mut State) -> bool {
        self.settle(st);
        let Some(before) = self.undo.pop() else { return false };
        if let Some(now) = self.last.replace(before.clone()) { self.redo.push(now); }
        st.restore_snapshot(&before);
        true
    }
    pub fn redo(&mut self, st: &mut State) -> bool {
        self.settle(st);
        let Some(after) = self.redo.pop() else { return false };
        if let Some(now) = self.last.replace(after.clone()) { self.undo.push(now); }
        st.restore_snapshot(&after);
        true
    }
    pub fn can_undo(&self) -> usize { self.undo.len() }
    pub fn can_redo(&self) -> usize { self.redo.len() }
}
