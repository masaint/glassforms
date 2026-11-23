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
//! **Selection.** An entry also keeps what was selected (`selection.rs`), so a click or a drag-select is an
//! entry of its own (taken at the same quiet moments, so a whole drag-select is one), and undo puts back
//! the selection with the setup. Moving between levels (entering an object, a group) is navigation:
//! it makes no entry, and a selection is put back only at the level it was made at. Entries read from a
//! history file have no selection: undoing them leaves the selection as it is.
//!
//! **What an entry leaves out.** It leaves out what moves by itself: the animation's clock and what
//! the animation writes into the animated sliders, the spin's phase, and the lights' own circling.
//! The snapshot is taken with those set aside, so playing never makes an entry, and an undo leaves the
//! animation running where it is. The camera slots are left out as well (storing one is not undone).

use serde::{Deserialize, Serialize};

use crate::selection::Selection;
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
        // (how the viewport draws is not part of what is undone)
        s.view = Default::default();
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
        let scope = self.edit_scope.clone();
        let outline = self.outline();
        let enclosing = scope.world.and_then(|id| outline.ancestors(crate::groups::Item::Group(id))).unwrap_or_default();
        let object_count = self.object_count();
        let (playing, clock, theta, lphi) = (self.playing, self.clock, self.theta, self.lphi);
        // (what the animation has the animated sliders at now, to the last place: worked out again below, it
        // could come out a rounding away)
        let had: std::collections::BTreeMap<String, f64> = self.active().iter().map(|k| (k.key.clone(), self.get(&k.r))).collect();
        let views = std::mem::take(&mut self.views);
        let view = self.view.clone();
        let cur = self.undo_snapshot();
        let same_scene = cur.family == s.family && cur.preset == s.preset && cur.values == s.values && cur.shape == s.shape
            && cur.layers == s.layers && cur.bodies == s.bodies && cur.hdri == s.hdri && cur.empty == s.empty;
        if same_scene {
            self.p = s.settings.clone();
            self.canonical = s.canonical.as_ref().map(|saved| crate::setup::CanonicalMaterials {
                scene: saved.scene.clone(),
                bodies: {
                    let mut bodies: [Option<Box<crate::pbr::MaterialSpec>>; crate::material::MAX_BODIES] = Default::default();
                    for (key, spec) in &saved.bodies {
                        if let Ok(layer) = key.parse::<usize>() {
                            if (2..=crate::material::MAX_BODIES).contains(&layer) {
                                bodies[layer - 1] = Some(Box::new(spec.clone()));
                            }
                        }
                    }
                    bodies
                },
            });
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
            // the tree of formula parts (csg.rs): put back by itself, and built again (which part the panel
            // has selected stays, if there still is one)
            // (a setup's own tree is the first object's: the others' are with their layers, above)
            if cur.construction != s.construction || cur.csg != s.csg || cur.csg_place != s.csg_place || cur.csg_coarse != s.csg_coarse || cur.csg_baked != s.csg_baked {
                if let Some(c) = self.layer_csg_mut(0) {
                    c.construction = s.construction.clone();
                    c.parts = s.csg.iter().take(crate::csg::MAX_PARTS).cloned().collect();
                    c.place = s.csg_place;
                    c.coarse = s.csg_coarse;
                    c.baked = s.csg_baked;
                    if c.sel.is_some_and(|i| i >= c.parts.len()) { c.sel = None; }
                    if c.parts.is_empty() { c.whole = false; }
                }
                self.request(crate::build::BuildRequest::edit());
            }
        } else {
            // (it was a setup a moment ago: it puts up)
            let _ = self.apply_setup(s);
        }
        if scope.inside.is_some() || scope.world.is_some() || !scope.selection.is_empty() {
            use crate::groups::Item;
            let tree = self.outline();
            self.edit_scope = scope;
            self.edit_scope.world = self.edit_scope.world.filter(|&id| tree.group(id).is_some())
                .or_else(|| enclosing.into_iter().rev().find(|&id| tree.group(id).is_some()));
            if let Some(o) = &mut self.edit_scope.inside { o.return_world = o.return_world.filter(|&id| tree.group(id).is_some()); }
            self.edit_scope.selection.retain(|&item| tree.ancestors(item).is_some());
            if let Some(i) = self.edit_scope.object().filter(|&i| object_count == self.object_count() && i < self.object_count()) {
                self.select_layer(i);
                self.edit_scope.world = tree.ancestors(Item::Object(i)).and_then(|p| p.last().copied());
                let construction = self.csg.construction_tree();
                let o = self.edit_scope.inside.as_mut().expect("object scope");
                while construction.children(&o.construction_level).is_none() {
                    o.construction_level.pop();
                }
                let path = &mut o.construction;
                while !path.is_empty() && construction.node(path).is_none() { path.pop(); }
                o.marked.retain(|p| construction.node(p).is_some());
                o.isolated = o.isolated.take().filter(|p| construction.node(p).is_some());
            } else {
                self.edit_scope.inside = None;
                match self.edit_scope.selection.last().copied() {
                    Some(Item::Object(i)) => self.select_layer(i),
                    Some(Item::Group(id)) => self.selected_group = Some(id),
                    None => self.selected_group = None,
                }
            }
        }
        self.views = views;
        self.view = view;
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

/// A history as a file keeps it.
#[derive(Serialize, Deserialize)]
struct SavedHistory {
    version: u32,
    undo: Vec<Setup>,
    #[serde(default)]
    redo: Vec<Setup>,
}

/// Where a setup's history is kept: beside it, `name.history.json` for `name.json`.
pub fn history_path(setup: &std::path::Path) -> std::path::PathBuf {
    setup.with_extension("history.json")
}

/// One entry: the setup, and what was selected (none for an entry read from a file).
#[derive(Clone, Debug, PartialEq)]
struct Entry {
    setup: Setup,
    sel: Option<Selection>,
}

impl Entry {
    fn of(st: &mut State) -> Self {
        Self { setup: st.undo_snapshot(), sel: Some(st.selection_snapshot()) }
    }
    fn restore(&self, st: &mut State) {
        st.restore_snapshot(&self.setup);
        if let Some(s) = &self.sel { st.restore_selection(s); }
    }
    /// Only where the editor is changed between the two: navigation, not an entry.
    fn moved_only(&self, now: &Entry) -> bool {
        self.setup == now.setup && match (&self.sel, &now.sel) {
            (Some(a), Some(b)) => a.level != b.level,
            _ => true,
        }
    }
}

/// The undo and redo stacks, and the snapshot of the state as it last settled.
#[derive(Default)]
pub struct History {
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    last: Option<Entry>,
}

impl History {
    /// A quiet moment, or a gesture about to begin: if the state has changed since the last snapshot, that
    /// change becomes one entry. True if it did.
    pub fn settle(&mut self, st: &mut State) -> bool {
        let now = Entry::of(st);
        match &self.last {
            Some(l) if *l == now => false,
            Some(l) if l.moved_only(&now) => {
                self.last = Some(now);
                false
            }
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
    /// The newest entry and the state it led to: what the last settle recorded.
    pub fn latest(&self) -> Option<(&Setup, &Setup)> {
        Some((&self.undo.last()?.setup, &self.last.as_ref()?.setup))
    }
    /// Esc during a gesture: the state goes back to the snapshot taken when it began.
    pub fn cancel(&mut self, st: &mut State) {
        if let Some(l) = &self.last {
            let l = l.clone();
            if Entry::of(st) != l { l.restore(st); }
        }
    }
    pub fn undo(&mut self, st: &mut State) -> bool {
        self.settle(st);
        let Some(before) = self.undo.pop() else { return false };
        if let Some(now) = self.last.take() { self.redo.push(now); }
        before.restore(st);
        // (taken again: whatever putting it back tidied is not a change, and redo is kept)
        self.last = Some(Entry::of(st));
        true
    }
    pub fn redo(&mut self, st: &mut State) -> bool {
        self.settle(st);
        let Some(after) = self.redo.pop() else { return false };
        if let Some(now) = self.last.take() { self.undo.push(now); }
        after.restore(st);
        self.last = Some(Entry::of(st));
        true
    }
    /// The history as a file keeps it beside a setup: every entry is itself a setup.
    pub fn to_json(&self) -> String {
        serde_json::to_string(&SavedHistory {
            version: crate::setup::VERSION,
            undo: self.undo.iter().map(|e| e.setup.clone()).collect(),
            redo: self.redo.iter().map(|e| e.setup.clone()).collect(),
        }).expect("setups always serialise")
    }
    /// Take up a history read from beside a setup that has just been put up (`st` is that setup): undo
    /// then walks back through what was done before it was saved. Fails, changing nothing, if the text
    /// is not a history.
    pub fn adopt_json(&mut self, text: &str, st: &mut State) -> Result<(), String> {
        let h: SavedHistory = serde_json::from_str(text).map_err(|e| format!("not a history: {e}"))?;
        let read = |setups: Vec<Setup>| setups.into_iter().map(|setup| Entry { setup, sel: None }).collect::<Vec<_>>();
        self.undo = read(h.undo);
        self.redo = read(h.redo);
        let cut = self.undo.len().saturating_sub(CAP);
        self.undo.drain(..cut);
        self.last = Some(Entry::of(st));
        Ok(())
    }
    pub fn can_undo(&self) -> usize { self.undo.len() }
    pub fn can_redo(&self) -> usize { self.redo.len() }
}
