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
//!   material  what the shape is made of: the material group at the head of the left panel, and the
//!             materials lab, a floating window with the rest of a material's controls
//!   room      the light room: a floating window with everything about the light (surroundings, floor,
//!             lights, shadows and caustics, turning)
//!   gizmo     the ball that shows where the lights are
//!   handles   arrows and rings over the picture that move, turn and size the selected layer
//!   overlay   the line of figures at the foot of the window, and the help with its "?"
//!   store     what is remembered between runs
//!   script    driving the window without a mouse, for tests

pub mod animate;
pub mod camera;
pub mod gizmo;
pub mod handles;
pub mod knob;
pub mod lamps;
pub mod layers;
pub mod left;
pub mod material;
pub mod overlay;
pub mod right;
pub mod room;
pub mod script;
pub mod store;
pub mod studio;
pub mod theme;
pub mod widgets;

use std::collections::BTreeMap;

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
    SaveStill,
    SaveStillAs,
    /// choose an environment picture in a dialog
    LoadPicture,
    /// choose a model's file in a dialog
    LoadModel,
    /// set the layer at this place in the list down on the floor
    ToFloor(usize),
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
    /// which lamp the light room is showing the sliders of
    pub lamp: usize,
    /// which group of the named materials the list is showing (none: the group of the one chosen), and
    /// the material the list last showed lit (when another is chosen from elsewhere, the list goes to it)
    pub mat_group: Option<&'static str>,
    pub lit_mat: Option<usize>,
    /// which combination of layers has its row open in the joins (one at a time), as a mask of the layers;
    /// and whether the rows for three layers, and for all four, are shown under their headings
    pub combo_open: Option<u8>,
    pub meets_open: [bool; 2],
    /// the layers were last combined into a volume this many voxels a side, in this many milliseconds
    /// (none: one layer, nothing to combine)
    pub joined: Option<(u32, f64)>,
    pub lighting: Lighting,
    pub model: Model,
    pub casting: Casting,
    /// the picture's width over its height (what the camera's field of view is worked out for)
    pub aspect: f64,
    /// the f-stop the lens was last open at, for when depth of field is switched on again
    pub last_fstop: f64,
    /// the line at the foot of the window: picture size, speed or how far the picture has refined, the shape's build
    pub perf: String,
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
    /// scroll a panel to here on the next frame: [left, right]
    pub scroll_to: [Option<f32>; 2],
    /// the most height a panel's contents may take before it scrolls, and the height each one's take: [left, right]
    pub room_h: f32,
    pub needs: [f32; 2],
    /// how tall the right panel's contents were last frame apart from its list of shapes, which gets the
    /// height that is left
    pub right_rest: f32,
}

impl Panels {
    pub fn new(store: store::Store) -> Self {
        Self {
            store, hide_left: false, hide_right: false, help_open: false, bright: false, light: 0, handles: handles::Handles::default(), lamp: 0, mat_group: None, lit_mat: None, combo_open: None, meets_open: [false; 2], joined: None, lighting: Lighting::default(), model: Model::default(), casting: Casting::default(), aspect: 16.0 / 9.0, last_fstop: 4.0, perf: String::new(), bench: Bench::Idle, rendering: false, ffmpeg: false,
            video_text: String::new(), file_text: String::new(), loop_text: String::new(), requests: Vec::new(), place: "", record: false, rects: BTreeMap::new(),
            left_at: None, right_at: None, lit_shape: None, scroll_to: [None, None], room_h: 600.0, needs: [0.0; 2], right_rest: 0.0,
        }
    }

    /// Note where something was drawn, if a test has asked for that.
    pub fn note(&mut self, name: impl Into<String>, rect: Rect) {
        if self.record { self.rects.insert(name.into(), rect); }
    }

    /// What a test knows a control by is prefixed with this when it is drawn in the light room or in the
    /// materials lab (the same control may also be in a panel).
    pub fn whose(&self) -> &'static str {
        match self.place {
            "room" => "room/",
            "lab" => "lab/",
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
