//! Everything that floats over the picture, and the controls in it.
//!
//! The right panel is the shape and how it moves; the left panel is how it looks. What is set once or
//! seldom is folded away, and what is not a control at all (how fast things are going) is a quiet line at
//! the foot of the window.
//!
//!   theme     colours, fonts, sizes
//!   widgets   buttons, menus, sliders, the range of an animated slider, floating windows
//!   knob      one slider's row, with its wave button and what appears under it when lit
//!   right     the right panel: the families, the family's shapes, the mathematics of the one chosen (for a
//!             model: its file, and what was found on examining it), its own sliders
//!   layers    several shapes in one space: the layers at the head of the right panel, how they are joined,
//!             where each is placed
//!   animate   the foot of the right panel: play, time, speed, easing, spin, the loop
//!   left      the left panel: material, light, camera (with the view), and folded away the output (still,
//!             setup, quality with the shape's detail, video)
//!   camera    the camera group of the left panel: lens, focus, where it stands, exposure, the lens's character
//!   studio    the light room's studio (backdrop, floor) and shadows and caustics; a body's own casting switches
//!   material  what the shape is made of: the material section at the head of the left panel (which
//!             material, its recipe, clay), and the materials lab, a floating window with every control of it
//!   sculpt    the surface studio: a floating window with a stack of procedural layers that reshape the
//!             selected layer's surface, opened from the right panel
//!   room      the light room: a floating window with everything about the light (surroundings, floor,
//!             lights, shadows and caustics, turning)
//!   gizmo     the ball that shows where the lights are
//!   handles   arrows and rings over the picture that move, turn and size the selected layer
//!   tools     the bars over the picture: the camera's (orbit or fly, frame, the camera slots) with the cube
//!             of the room's axes beside the left panel; the object tools (select, all, move, turn, size)
//!             beside the right
//!   picker    the shape picker: a window of pictures of every shape, family by family, that makes an object
//!             (from the "+") or changes the selected object's shape (from its card in the right panel)
//!   overlay   the line of figures at the foot of the window, and the help with its "?"
//!   store     what is remembered between runs
//!   script    driving the window without a mouse, for tests

pub mod animate;
pub mod camera;
pub mod tiers;
pub mod viewport;
pub mod csg;
pub mod gizmo;
pub mod handles;
pub mod draw_cut;
pub mod knob;
pub mod lamps;
pub mod layers;
pub mod left;
pub mod material;
pub mod overlay;
pub mod paths;
pub mod picker;
pub mod pick;
pub mod outline;
pub mod right;
pub mod room;
pub mod script;
pub mod select;
pub mod settings;
pub mod mods;
mod modifier_pivot;
mod world;
mod construction;
pub mod world_modifiers;
pub mod sculpt;
pub mod store;
pub mod studio;
pub mod theme;
pub mod tools;
pub mod widgets;
#[cfg(test)]
mod gesture_tests;

use std::collections::{BTreeMap, BTreeSet};

use eframe::egui::{self, Id, Rect, TextureId, Ui};

/// Something a panel asks the window to do (the panels themselves only change the state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    RenderVideo,
    CancelVideo,
    Benchmark,
    OpenFolder,
    ChooseFolder,
    SaveSetup,
    LoadSetup,
    /// the last session's work, offered back after it ended without closing: taken (true) or let go
    Restore(bool),
    SaveStill,
    SaveStillAs,
    /// choose an environment picture in a dialog
    LoadPicture,
    /// choose a model's file in a dialog
    LoadModel,
    /// set the layer at this place in the list down on the floor
    ToFloor(usize),
    /// choose the picture of the surface studio's layer at this place
    LoadSculptPicture(usize),
    /// write the shape as a mesh (OBJ or STL), to a file chosen in a dialog
    ExportMesh,
    /// choose a picture to use as the matcap (view_mode.rs)
    LoadMatcap,
}

/// What the window knows of the lighting that the state does not say: the engine's doing.
#[derive(Default)]
pub struct Lighting {
    /// the small picture of the lighting as it stands, once the engine has drawn it
    pub preview: Option<TextureId>,
    /// the environment picture in use: its file's name, its width and height
    pub picture: Option<(String, u32, u32)>,
    /// why the last picture asked for could not be used
    pub error: Option<String>,
    /// a picture is being read
    pub loading: bool,
}

/// What the window knows of the selected layer's model that the state does not say: the engine's doing.
#[derive(Clone, Default)]
pub struct Model {
    /// it has been read and built, and what follows is known
    pub known: bool,
    /// its file with what it is made of ("teapot.obj · 89,600 triangles · 4 pieces"), what examining it
    /// found, and the rest of that for whoever hovers over the two
    pub file: String,
    pub line: String,
    pub tip: String,
    /// it is drawn as a shell; examining it found a shell
    pub shell: bool,
    pub found_shell: bool,
    /// how far its own facets call for its detail to be melted, and how wide those facets are
    pub own_smooth: f64,
    pub facet: f64,
    /// a file is being read: its name
    pub reading: Option<String>,
    /// why the last file given could not be used
    pub error: Option<String>,
    /// what else there is to say of the last file given (it took a layer's place, there being no room for another)
    pub said: String,
}

/// What the window knows of the shadows and caustics that the state does not say: the engine's doing.
#[derive(Clone, Default)]
pub struct Casting {
    /// something casts and something can show it
    pub on: bool,
    /// something in the picture can show a shadow: clay, a milky body, a floor
    pub receivers: bool,
    /// caustics are being made
    pub focusing: bool,
    /// how many lights' shadows are followed
    pub lights: usize,
    /// what each body casts at present, in the order of the state's bodies: (a shadow, caustics)
    pub casters: Vec<(bool, bool)>,
    /// the volume of arriving light: voxels a side and bytes of video memory (while caustics are made)
    pub caustics: Option<(u32, u64)>,
    /// where the floor stands under the middle of the shape (while there is one)
    pub floor: Option<f64>,
}

/// The benchmark button: untouched, about to run (its label says so for one frame first), or showing a result.
#[derive(Clone, Debug, PartialEq)]
pub enum Bench {
    Idle,
    Asked,
    Done(String),
}

/// What the panels share with the window that shows them.
pub struct Panels {
    pub store: store::Store,
    /// the quality-tier row (ui/tiers.rs): what the scene allows (looked at again every few frames), the chevron's window and where it stands,
    /// the choices inside the tiers, the frame times by tier, the tier drawing now when it is not the lit one, and a check aid that shows every tier
    pub tier_cache: Option<(u64, crate::tier::Avail)>,
    pub tier_menu: bool,
    pub tier_menu_at: Option<[f32; 2]>,
    pub tier_cfg: tiers::Cfg,
    pub tier_times: crate::tier::Times,
    pub tier_drawing: Option<crate::tier::Tier>,
    pub tier_all: bool,
    /// the phase of the interaction as the window sees it, a count of frames, and why a frame was not drawn by what was asked for
    pub tier_phase: crate::tier::PhaseTracker,
    pub tier_frame: u64,
    pub tier_note: Option<String>,
    pub hide_left: bool,
    pub hide_right: bool,
    /// the help (what the mouse and the keys do) is open
    pub help_open: bool,
    /// the picture is bright from edge to edge (an environment picture is shown behind the shape): what
    /// lies over it must stand out more
    pub bright: bool,
    /// which light the light room is showing the sliders of
    pub light: usize,
    /// the handles over the picture (shown with G)
    pub handles: handles::Handles,
    pub draw_cut: draw_cut::Drawing,
    /// the path builder, and the path drawn over the picture (while the shape is a path)
    pub paths: paths::PathEdit,
    /// how the camera is moved: orbiting or flying
    pub nav: tools::Nav,
    /// which lamp the light room is showing the sliders of
    pub lamp: usize,
    /// The last selected preset; a selection from a setup or another body scrolls into view.
    pub lit_mat: Option<usize>,
    /// The specific Object Mode construction node being edited in the Materials Lab.
    pub lab_node: Option<(usize, Vec<usize>)>,
    /// Canonical attributes added for editing in this session, and the last rejected body edit.
    /// These are UI affordances only; full canonical owners in `State` remain authoritative.
    pub canonical_added: BTreeSet<(usize, String)>,
    pub canonical_error: Option<(usize, String)>,
    /// which combination of layers has its row open in the joins (one at a time), as a mask of the layers;
    /// and whether the rows for three layers, and for all four, are shown under their headings
    pub combo_open: Option<gf_core::layers::Mask>,
    pub meets_open: [bool; 2],
    /// the layers were last combined into a volume this many voxels a side, in this many milliseconds
    /// (none: one layer, nothing to combine)
    pub joined: Option<(u32, f64)>,
    /// the surface studio last ran on a volume this many voxels a side, in this many milliseconds; what
    /// went wrong with it (a picture that could not be read); which of its layers is open
    pub sculpted: Option<(u32, f64)>,
    /// the selected layer's shape's own size, which the studio's sizes as shares are taken of (0: not
    /// measured yet)
    pub sculpt_size: f64,
    pub sculpt_error: Option<String>,
    /// why the matcap picture asked for could not be used
    pub matcap_error: Option<String>,
    /// the starter matcaps as small balls for the picker (made the first time it is shown)
    pub matcap_thumbs: Vec<eframe::egui::TextureHandle>,
    pub sculpt_layer: Option<usize>,
    /// the modifier whose controls are open, "+ add modifier" open, its frame's rows open; how the
    /// selected object's stack was applied when that is worth a note (ui/mods.rs)
    pub mods_sel: Option<usize>,
    pub mods_adding: bool,
    pub mods_frame: bool,
    pub mods_note: Option<String>,
    /// the modifier being carried to another place (its row, and where in it it was picked up), each row's
    /// slide from its place (easing to where it is going), the height the open controls took
    pub mods_drag: Option<(usize, f32)>,
    pub mods_slide: Vec<f32>,
    pub mods_ctrl_h: f32,
    /// the csg object's own strokes are open in the panel
    pub strokes_open: bool,
    pub lighting: Lighting,
    pub model: Model,
    pub casting: Casting,
    /// the picture's width over its height (what the camera's field of view is worked out for)
    pub aspect: f64,
    /// the f-stop the lens was last open at, for when depth of field is switched on again
    pub last_fstop: f64,
    /// the line at the foot of the window: picture size, speed or how far the picture has refined, the shape's build
    pub perf: String,
    /// how far the still picture has settled (0 to 1), drawn as a hairline under that line; none while it moves or has settled
    pub settle: Option<f32>,
    /// the short note above the foot line: what was just saved or loaded, or what went wrong (overlay::message)
    pub message: Option<overlay::Message>,
    /// the last session ended without closing and its work can be brought back: when it was last kept
    pub restore: Option<String>,
    /// what keeping the frame rate is doing now, for the quality section (empty while it is off)
    pub keep_says: String,
    pub bench: Bench,
    /// a video is being rendered: everything but its cancel button is out of reach
    pub rendering: bool,
    /// ffmpeg was found, so videos can be made
    pub ffmpeg: bool,
    /// what the video group says under its button: progress, then the result
    pub video_text: String,
    /// what the file group says under its buttons: the file saved or loaded, or what went wrong
    pub file_text: String,
    /// the words in the loop-length field while it is being typed in
    pub loop_text: String,
    /// what the panels ask of the window this frame
    pub requests: Vec<Request>,
    /// which panel or window is being drawn: a slider shown in two of them at once is two controls
    pub place: &'static str,
    /// note where the controls are drawn (for tests that point at them)
    pub record: bool,
    pub rects: BTreeMap<String, Rect>,
    /// where the two panels are, when they are shown (the line at the foot keeps clear of them)
    pub left_at: Option<Rect>,
    pub right_at: Option<Rect>,
    /// the family and shape the picker last showed lit (when it changes, the list scrolls to it)
    pub lit_shape: Option<(usize, usize)>,
    /// the right panel's tab, volume (false) or csg (true), when it is not the selected object's kind: picking a
    /// shape then makes a new object of the tab's kind (right.rs). Put back to the object's kind whenever the
    /// selection changes (	ab_seen: the selected object, how many there are, and its kind, as last seen)
    pub tab: Option<bool>,
    pub tab_seen: (usize, usize, bool),
    /// the family shown while making a new object (the selected object's family is left as it is)
    pub browse_fam: usize,
    /// a new object of the selected object's own kind is being made (its tab pressed again, or the objects' +)
    pub adding: bool,
    /// how tall the animation at the foot of the right panel was drawn (the controls above are given the rest)
    pub anim_h: f32,
    /// where the selected object's controls are to scroll to (a script's scroll:right)
    pub right_scroll: Option<f32>,
    /// while controls are drawn in an area that scrolls: what of it is in sight (`note` leaves the rest out)
    pub clip: Option<Rect>,
    /// how far the selected object's controls can scroll (as last drawn)
    pub controls_max: f32,
    /// scroll a panel to here on the next frame: [left, right]
    pub scroll_to: [Option<f32>; 2],
    /// the most height a panel's contents may take before it scrolls, and the height each one's take: [left, right]
    pub room_h: f32,
    pub needs: [f32; 2],
    /// how tall the right panel's contents were last frame apart from its list of shapes, which gets the
    /// height that is left
    pub right_rest: f32,
    /// the inside of the program's window, this frame
    pub full: Rect,
    /// a control marked costly is in view in the materials lab (its legend is shown only then)
    pub costly_shown: bool,
    /// the shape picker: whether it is open and for what, its search, its pictures
    pub picker: picker::Picker,
    pub palette: outline::Palette,
}

impl Panels {
    pub fn new(store: store::Store) -> Self {
        Self {
            store, tier_cache: None, tier_menu: false, tier_menu_at: None, tier_cfg: Default::default(), tier_times: Default::default(), tier_drawing: None, tier_all: false, tier_phase: Default::default(), tier_frame: 0, tier_note: None, hide_left: false, hide_right: false, help_open: false, bright: false, light: 0, handles: handles::Handles::starting(), draw_cut: Default::default(), paths: paths::PathEdit::default(), nav: tools::Nav::default(), lamp: 0, lit_mat: None, lab_node: None, canonical_added: BTreeSet::new(), canonical_error: None, combo_open: None, meets_open: [false; 2], joined: None, sculpted: None, sculpt_size: 0.0, sculpt_error: None, matcap_error: None, matcap_thumbs: Vec::new(), sculpt_layer: Some(0), mods_sel: None, mods_adding: false, mods_frame: false, mods_note: None, mods_drag: None, mods_slide: Vec::new(), mods_ctrl_h: 0.0, strokes_open: false, lighting: Lighting::default(), model: Model::default(), casting: Casting::default(), aspect: 16.0 / 9.0, last_fstop: 4.0, perf: String::new(), settle: None, message: None, restore: None, keep_says: String::new(), bench: Bench::Idle, rendering: false, ffmpeg: false,
            video_text: String::new(), file_text: String::new(), loop_text: String::new(), requests: Vec::new(), place: "", record: false, rects: BTreeMap::new(),
            left_at: None, right_at: None, lit_shape: None, tab: None, tab_seen: (0, 1, false), browse_fam: 0, adding: false, anim_h: 120.0, right_scroll: None, clip: None, controls_max: 0.0, scroll_to: [None, None], room_h: 600.0, needs: [0.0; 2], right_rest: 0.0, full: Rect::ZERO, costly_shown: false, picker: picker::Picker::default(), palette: outline::Palette::default(),
        }
    }

    /// Note where something was drawn, if a test has asked for that.
    pub fn note(&mut self, name: impl Into<String>, rect: Rect) {
        // (a control scrolled out of sight is not there to be pressed)
        if self.clip.is_some_and(|c| !c.intersects(rect)) { return; }
        if self.record { self.rects.insert(name.into(), rect); }
    }

    /// What a test knows a control by is prefixed with this when it is drawn in the light room (the same
    /// control may also be in a panel).
    pub fn whose(&self) -> &'static str {
        match self.place {
            "room" => "room/",
            // (the materials lab holds the only copy of the material's controls: they keep their plain names)
            "cam" => "cam/",
            _ => "",
        }
    }

    /// A group's title; true if the group is open and its contents should follow. `says`: a few words
    /// shown beside the title while the group is folded, about what it is set to.
    pub fn group(&mut self, ui: &mut Ui, id: &str, title: &str, says: &str) -> bool {
        let closed = self.store.is_closed(id);
        self.group_saying(ui, id, title, if closed { says } else { "" })
    }

    /// The same, with words beside the title whether the group is folded or not.
    pub fn group_saying(&mut self, ui: &mut Ui, id: &str, title: &str, says: &str) -> bool {
        let closed = self.store.is_closed(id);
        let (clicked, at) = widgets::group_title(ui, Id::unique(("group", id)), title, closed, says);
        self.note(format!("group:{id}"), at);
        if clicked { self.store.fold(id); }
        !self.store.is_closed(id)
    }
}

/// The levels below a section (see docs/ui_layout.md, "The hierarchy"): a group in a panel has a heading
/// and never folds; a group in a floating window folds; a sub-group folds and stands in from the edge
/// with a bar down its left.
impl Panels {
    /// Let go of every gesture in progress. They hold places in the scene (layer, modifier and item
    /// indices) that an undo or redo can give to something else.
    pub fn drop_gestures(&mut self) {
        self.handles.grab = None;
        self.paths.grab = None;
        self.draw_cut = Default::default();
        self.palette.drag = None;
        self.mods_drag = None;
    }
    /// A group's heading in a panel. (Noted as `group:<id>` for tests, as the folding ones are.)
    pub fn heading(&mut self, ui: &mut Ui, id: &str, title: &str) {
        let at = widgets::group_heading(ui, title);
        self.note(format!("group:{id}"), at);
    }

    /// A group in a floating window: its heading, which folds it; true if it is open. `says` shows beside
    /// the title while it is folded.
    pub fn window_group(&mut self, ui: &mut Ui, id: &str, title: &str, says: &str) -> bool {
        let closed = self.store.is_closed(id);
        let (clicked, at) = widgets::fold_heading(ui, Id::unique(("group", id)), title, closed, says);
        self.note(format!("group:{id}"), at);
        if clicked { self.store.fold(id); }
        !self.store.is_closed(id)
    }

    /// A sub-group: its heading, and if it is open its rows, set in from the edge with a bar down their
    /// left. `says` shows beside the title whether it is open or not.
    pub fn sub(&mut self, ui: &mut Ui, id: &str, title: &str, says: &str, body: impl FnOnce(&mut Ui, &mut Panels)) {
        if !self.window_group(ui, id, title, says) { return; }
        let (top, left) = (ui.cursor().top(), ui.cursor().left());
        let inner = Rect::from_min_max(egui::pos2(left + SUB_INSET, top), egui::pos2(ui.max_rect().right(), top + 8000.0));
        let names = widgets::label_width(ui);
        ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
            ui.set_width(inner.width());
            widgets::set_label_width(ui, names - SUB_INSET);
            body(ui, self);
        });
        widgets::set_label_width(ui, names);
        widgets::side_bar(ui, left + 3.0, top + 1.0, ui.cursor().top() - 2.0, 0.3);
        widgets::gap(ui, 2.0);
    }
}

/// How far a sub-group's rows stand in from its heading.
pub const SUB_INSET: f32 = 10.0;

/// Digits shown after the point for a slider with this step.
pub fn digits_of(step: f64) -> usize {
    if step < 0.001 { 4 } else if step < 0.01 { 3 } else if step < 1.0 { 2 } else { 0 }
}

/// What an undo or a redo changed, in a few words: the setting it moved and from what to what ("index
/// 1.80 → 1.50"), and how many more it moved with it; otherwise what part of the scene it changed.
pub fn undo_words(was: &gf_core::setup::Setup, now: &gf_core::setup::Setup) -> String {
    // (an entry that changed nothing a setup keeps changed what was selected)
    if was == now { return "the selection".to_owned(); }
    let (a, b) = (serde_json::to_value(&was.settings).unwrap_or_default(), serde_json::to_value(&now.settings).unwrap_or_default());
    let mut moved: Vec<String> = Vec::new();
    if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
        let knobs = gf_core::state::State::engine_knobs();
        for (name, va) in a {
            let (Some(x), Some(y)) = (va.as_f64(), b.get(name).and_then(|v| v.as_f64())) else { continue };
            if x == y { continue; }
            let (label, digits) = match knobs.iter().find(|k| k.id == *name) {
                Some(k) => (k.label.clone(), digits_of(k.step)),
                None => (name.clone(), 2),
            };
            moved.push(format!("{label} {x:.digits$} → {y:.digits$}"));
        }
    }
    if let Some(first) = moved.first() {
        return if moved.len() > 1 { format!("{first} and {} more", moved.len() - 1) } else { first.clone() };
    }
    // nothing among the settings: say which part of the scene it was
    let (a, b) = (serde_json::to_value(was).unwrap_or_default(), serde_json::to_value(now).unwrap_or_default());
    let names = [
        ("family", "the shape"), ("preset", "the shape"), ("values", "the shape's sliders"), ("shape", "the shape's build"), ("orientation", "the view's turn"),
        ("tracks", "what is animated"), ("rig", "the lights"), ("hdri", "the environment picture"), ("layers", "the layers"), ("bodies", "a body's material"),
        ("lamps", "the lamps"), ("csg", "the csg parts"), ("render", "the video's choices"),
    ];
    let parts: Vec<&str> = names.iter().filter(|(k, _)| a.get(k) != b.get(k)).map(|(_, w)| *w).fold(Vec::new(), |mut v, w| { if !v.contains(&w) { v.push(w); } v });
    if parts.is_empty() { "a change".to_owned() } else { parts.join(", ") }
}
