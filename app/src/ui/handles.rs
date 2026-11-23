//! Handles on the picture: arrows, small squares and a dot to move the selected layer, rings to turn it,
//! and a ring round it to size it, dragged with the pointer. They are only paint over the picture (no
//! pass of their own), and they are drawn and followed through the camera's own arithmetic
//! (`Camera::project` and `Camera::ray`), so they sit right under every kind of lens: a fisheye bends
//! them as it bends the shape.
//!
//! What a drag changes is the layer's placement, the same numbers as its sliders (`L<n>.x` and so on):
//! they are kept in a setup, can be animated, and come back with the placement's reset. Nothing is
//! shown until asked for (G, or the placement's "handles" button); T steps through move, turn and size;
//! Esc puts them away, or during a drag puts the layer back where the drag began.
//!   on a handle   Ctrl: in steps (0.05 of a unit, 15 degrees, sizes by 0.05); Shift: finely (a tenth)
//!   a click       a click on a shape selects its layer, on a lamp the lamp, on nothing lets go (Alt+click
//!                 focuses); what is chosen is marked with corners round it (paint only)
//!
//! The tool bar over the picture (tools.rs) chooses among them: select (no handles), move, turn, size, or
//! the camera (no handles; a click focuses while the lens has an aperture). The handles' axes are the
//! scene's (world) or the chosen thing's own (local); snapping may be left on.

use eframe::egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, Vec2};
use gf_core::camera::Camera;
use gf_core::math::{apply, axis_angle, euler, euler_of, mul, transpose, Mat3, DEG};
use gf_core::state::State;

/// What the handles do.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    /// everything at once: arrows, rings, the middle, the rings round it, the ball inside
    #[default]
    All,
    Move,
    Turn,
    Size,
}
impl Mode {
    pub fn next(self) -> Mode {
        match self {
            Mode::All => Mode::Move,
            Mode::Move => Mode::Turn,
            Mode::Turn => Mode::Size,
            Mode::Size => Mode::All,
        }
    }
    pub fn word(self) -> &'static str {
        match self {
            Mode::All => "all",
            Mode::Move => "move",
            Mode::Turn => "turn",
            Mode::Size => "size",
        }
    }
}

/// A part of the handles that can be taken hold of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Part {
    /// an arrow along one of the scene's axes (0 = x, 1 = y, 2 = z)
    Axis(usize),
    /// a small square: the plane across that axis
    Plane(usize),
    /// the dot in the middle: across the picture
    Centre,
    /// a ring about that axis
    Ring(usize),
    /// the ring round everything: the size
    Scale,
    /// the pale ring: a turn about the line of sight
    Spin,
    /// the faint ball inside the rings: a turn any way, as a ball rolled under the hand
    Ball,
    /// the small square near the end of the thing's own axis: its size along that axis (lamps)
    Stretch(usize),
}

/// What the handles act on: a layer (the selected one) or a lamp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Pivot(gf_core::modifier_pivot::PivotOwner, usize),
    World(gf_core::groups::Item),
    Lamp(usize),
    /// a part of the tree of formulas (csg.rs), selected in its list: moved within the object. Held by
    /// its stable identity (`Csg::part_id`), not its place in the list
    Part(u64),
    /// The base layer or a nested construction group selected in Object Mode.
    ConstructionNode,
    /// the tree's object, the whole shape: moved in the scene, its parts with it
    Whole,
}

impl Target {
    pub(super) fn construction(self, st: &State) -> Option<gf_core::csg::ConstructionHandle> {
        match self {
            Self::Part(id) => st.construction_handle(Some(st.csg.construction_part(id)?)),
            Self::Whole => st.construction_handle(None),
            Self::ConstructionNode => st.selected_node_handle(),
            _ => None,
        }
    }
}

/// A drag in progress.
#[derive(Clone, Debug)]
pub struct Grab {
    pub part: Part,
    target: Target,
    /// the numbers when the drag began: x, y, z, three turns, size (and a lamp's second size; a part's
    /// stretch along its own three axes)
    start: [f64; 11],
    /// where the pointer was pressed
    at0: Pos2,
    /// the point of the constraint the pointer's ray met at the press (camera frame), or the length
    /// along the axis
    hit0: [f64; 3],
    s0: f64,
    /// a ring: the way the ring runs on the screen where it was taken hold of (a turn the right way about
    /// its axis moves that point this way), and how far that point is from the middle, in points
    tan: Vec2,
    rad: f32,
    /// what the target's place pointed into when the drag began (see `Seen`)
    seen: Seen,
}

/// The lists a drag's target is found in, as they were at the press. A part is held by its identity and
/// a group by its id, but an object, a lamp, a modifier and a construction node are held by their place;
/// if any of these lists changes under the drag (something added or removed, another layer or node
/// selected), the place may name something else, and the drag lets go rather than move it.
#[derive(Clone, Debug, PartialEq)]
struct Seen {
    layer: usize,
    objects: usize,
    lamps: usize,
    parts: usize,
    node: Vec<usize>,
    /// a pivot's modifier stack
    mods: Option<usize>,
}

impl Seen {
    fn of(st: &State, t: Target) -> Self {
        let mods = match t { Target::Pivot(owner, _) => owner.stack_len(st), _ => None };
        Self { layer: st.stack.sel, objects: st.object_count(), lamps: st.lamps.len(), parts: st.csg.parts.len(), node: st.edit_scope.construction().to_vec(), mods }
    }
}

impl Grab {
    /// The target is still there, and still the one taken hold of.
    fn holds(&self, st: &State) -> bool {
        exists(st, self.target) && self.seen == Seen::of(st, self.target)
    }
}

/// What the tool bar has chosen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Select,
    All,
    Move,
    Turn,
    Size,
    Camera,
}

/// The handles' state, kept by the window.
#[derive(Default)]
pub struct Handles {
    /// the handles show (the tool is move, turn or size: `mode` says which)
    pub on: bool,
    pub mode: Mode,
    /// the tool is select (when the handles do not show)
    pub select: bool,
    /// something is chosen: the selected layer, or `lamp`. Nothing is marked or handled while not
    pub picked: bool,
    /// the handles' axes are the chosen thing's own, not the scene's
    pub local: bool,
    /// the handles move in steps without Ctrl being held
    pub snap: bool,
    pub move_origin: bool,
    pub construction_origin: bool,
    pub snap_reference: Option<gf_core::world_transform::Reference>,
    pub align_reference: gf_core::world_transform::Reference,
    pub grab: Option<Grab>,
    pub hover: Option<Part>,
    /// they act on this lamp (chosen in the light room, or picked on the picture), otherwise on the
    /// selected layer
    pub lamp: Option<usize>,
    pub pivot: Option<(gf_core::modifier_pivot::PivotOwner, usize)>,
    /// where each part of them was last painted (for tests that take hold of them)
    pub spots: Vec<(&'static str, Pos2)>,
}

const AXIS_INK: [Color32; 3] = [Color32::from_rgb(235, 90, 85), Color32::from_rgb(110, 205, 95), Color32::from_rgb(95, 145, 240)];
/// A handle under the pointer is brightened in its own hue (towards white), so that it keeps its axis's
/// colour and is not mistaken for the orange size ring.
fn lit(c: Color32) -> Color32 {
    let up = |v: u8| (v as f32 + (255.0 - v as f32) * 0.5).round() as u8;
    Color32::from_rgba_unmultiplied(up(c.r()), up(c.g()), up(c.b()), c.a())
}

/// Words over the picture on a small dark pill, as the hint under the tool bar has, so that they read over
/// bright glass and over the handles' own lines.
fn pill_text(p: &Painter, at: Pos2, anchor: eframe::egui::Align2, text: &str, font: eframe::egui::FontId, ink: Color32) {
    let g = p.layout_no_wrap(text.to_owned(), font, ink);
    let r = anchor.anchor_size(at, g.size());
    p.rect_filled(r.expand2(Vec2::new(5.0, 2.0)), 5, Color32::from_black_alpha(150));
    p.galley(r.min, g, ink);
}
/// how long the arrows look, in points, and how near the pointer must come to a handle to take it: its
/// drawn half width (the dark edge under the lines is 4 points wide at most) and a little slack
const ARROW_PT: f64 = 90.0;
const REACH_PT: f32 = 2.0 + super::pick::SLACK_PT;
/// an arrow's whole shaft is a handle, taken as near as its other handles are
const SHAFT_PT: f32 = REACH_PT;
/// the dot at the handles' middle
const CENTRE_PT: f32 = 4.0;
/// the combined handles: where their arrows end (a share of the rings' radius), the pale ring and the size
/// ring (multiples of the arrows' length on the screen), and the size ring's colour
const ALL_ARROW: f64 = 0.74;
const SPIN_R: f64 = 1.14;
const SIZE_R: f64 = 1.30;
const SIZE_INK: Color32 = Color32::from_rgb(240, 160, 60);

/// The camera and where the selected layer stands, as the picture shows them.
struct View {
    cam: Camera,
    /// the object's frame -> the camera's
    m: Mat3,
    /// the layer's middle, in the camera's frame
    c: [f64; 3],
    /// the handles' three axes, in the camera's frame, and in the frame the thing is placed in (the scene's
    /// own axes, or the thing's own turned ones)
    axes: [[f64; 3]; 3],
    dirs: [[f64; 3]; 3],
    /// the thing's own axes, in the camera's frame (for sizes along them)
    own: [[f64; 3]; 3],
    /// a length in the scene that looks ARROW_PT points long where the layer stands
    len: f64,
    rect: Rect,
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn scale(a: [f64; 3], k: f64) -> [f64; 3] { [a[0] * k, a[1] * k, a[2] * k] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }

impl View {
    fn of(st: &State, rect: Rect, target: Target, local: bool) -> Option<View> {
        let cam = st.camera();
        // a layer stands in the object's frame, a lamp in the lights' frame
        let (m, pos, rot) = match target {
            Target::Pivot(layer, i) => (st.view_r(), layer.position(st, i)?, layer.frame(st).placement.rot),
            Target::World(item) => { let p = st.world_placement(item)?; (st.view_r(), p.pos, p.rot) }
            Target::Lamp(i) => { let l = st.lamps.get(i)?; (st.light_m(), l.pos, l.rot) }
            // (a part is moved in a frame turned with the object: see csg.rs `part_handle`)
            Target::Part(_) | Target::Whole | Target::ConstructionNode => {
                let h = target.construction(st)?;
                (gf_core::math::mul(&st.view_r(), &h.axes), h.pose.pos, h.pose.rot)
            }
        };
        let c = apply(&m, pos);
        let own = if local { euler(rot[0], rot[1], rot[2]) } else { [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
        let dirs = [apply(&own, [1.0, 0.0, 0.0]), apply(&own, [0.0, 1.0, 0.0]), apply(&own, [0.0, 0.0, 1.0])];
        let axes = dirs.map(|d| apply(&m, d));
        let turned = euler(rot[0], rot[1], rot[2]);
        let own = [0, 1, 2].map(|k| { let mut e = [0.0; 3]; e[k] = 1.0; apply(&m, apply(&turned, e)) });
        let mut v = View { cam, m, c, axes, dirs, own, len: 0.4, rect };
        // how long a step across the picture looks where the layer stands
        let step = 1e-3 * (v.cam.along(c).max(0.1));
        if let (Some(a), Some(b)) = (v.screen(c), v.screen(add(c, [step, 0.0, 0.0]))) {
            let d = (b - a).length() as f64;
            if d > 1e-9 { v.len = ARROW_PT * step / d; }
        }
        Some(v)
    }
    /// Where a point of the camera's frame shows on the screen.
    fn screen(&self, p: [f64; 3]) -> Option<Pos2> {
        let uv = self.cam.project(p)?;
        let h = self.rect.height() as f64 / 2.0;
        Some(Pos2::new((self.rect.center().x as f64 + uv[0] * h) as f32, (self.rect.center().y as f64 - uv[1] * h) as f32))
    }
    /// The ray through a point of the screen, in the camera's frame.
    fn ray(&self, at: Pos2) -> Option<([f64; 3], [f64; 3])> {
        let h = self.rect.height() as f64 / 2.0;
        self.cam.ray([(at.x - self.rect.center().x) as f64 / h, -(at.y - self.rect.center().y) as f64 / h])
    }
    /// A line of the scene as points on the screen (bent as the lens bends it); runs broken where the
    /// lens cannot see.
    fn polyline(&self, pts: impl Iterator<Item = [f64; 3]>) -> Vec<Vec<Pos2>> {
        let mut runs = vec![Vec::new()];
        for p in pts {
            match self.screen(p) {
                Some(s) => runs.last_mut().expect("one").push(s),
                None => if !runs.last().expect("one").is_empty() { runs.push(Vec::new()); },
            }
        }
        runs.retain(|r| r.len() > 1);
        runs
    }
    /// The two directions across axis k (for its ring and its plane's square).
    fn across(&self, k: usize) -> ([f64; 3], [f64; 3]) {
        (self.axes[(k + 1) % 3], self.axes[(k + 2) % 3])
    }
    fn arrow(&self, k: usize) -> Vec<Vec<Pos2>> {
        self.arrow_to(k, 1.0)
    }
    /// The arrow along axis k, ending at this share of its usual length.
    fn arrow_to(&self, k: usize, f: f64) -> Vec<Vec<Pos2>> {
        self.polyline((0..=12).map(|i| add(self.c, scale(self.axes[k], self.len * (0.18 + (f - 0.18) * i as f64 / 12.0)))))
    }
    /// Where the size square of the thing's own axis k stands.
    fn stretch_at(&self, k: usize) -> Option<Pos2> {
        self.screen(add(self.c, scale(self.own[k], self.len * 0.9)))
    }
    fn square(&self, k: usize) -> Option<Vec<Pos2>> {
        let (a, b) = self.across(k);
        let (s0, s1) = (0.22 * self.len, 0.42 * self.len);
        [(s0, s0), (s1, s0), (s1, s1), (s0, s1)].iter().map(|(x, y)| self.screen(add(self.c, add(scale(a, *x), scale(b, *y))))).collect()
    }
    /// A ring about axis k: its points, and whether each is on the camera's side of the middle.
    fn ring(&self, k: usize) -> Vec<(Pos2, bool)> {
        let (a, b) = self.across(k);
        let to_cam = self.toward_camera(self.c);
        (0..=72)
            .filter_map(|i| {
                let t = i as f64 / 72.0 * std::f64::consts::TAU;
                let off = add(scale(a, t.cos() * self.len), scale(b, t.sin() * self.len));
                self.screen(add(self.c, off)).map(|s| (s, dot(off, to_cam) >= 0.0))
            })
            .collect()
    }
    fn toward_camera(&self, p: [f64; 3]) -> [f64; 3] {
        let (o, _) = self.cam.ray([0.0, 0.0]).unwrap_or(([0.0, 0.0, self.cam.dist], [0.0, 0.0, -1.0]));
        let d = if self.cam.proj == gf_core::camera::Projection::Orthographic { [0.0, 0.0, 1.0] } else { sub(o, p) };
        let l = dot(d, d).sqrt().max(1e-12);
        scale(d, 1.0 / l)
    }
    fn scale_radius(&self) -> f32 {
        (ARROW_PT * 1.25) as f32
    }
    /// How far the rings reach from the middle on the screen (their radius, near enough).
    fn ring_radius(&self) -> f32 {
        let Some(c) = self.screen(self.c) else { return ARROW_PT as f32 };
        let to = self.toward_camera(self.c);
        // a direction across the line of sight
        let side = { let a = cross(to, [0.0, 1.0, 0.0]); let l = dot(a, a).sqrt(); if l > 1e-6 { scale(a, 1.0 / l) } else { [1.0, 0.0, 0.0] } };
        self.screen(add(self.c, scale(side, self.len))).map_or(ARROW_PT as f32, |q| (q - c).length())
    }
}

fn dist_to_runs(p: Pos2, runs: &[Vec<Pos2>]) -> f32 {
    let mut best = f32::MAX;
    for r in runs {
        for w in r.windows(2) {
            let (a, b) = (w[0], w[1]);
            let ab = b - a;
            let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
            best = best.min((a + ab * t - p).length());
        }
    }
    best
}

fn inside(p: Pos2, poly: &[Pos2]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let c = (b - a).x * (p - a).y - (b - a).y * (p - a).x;
        if c != 0.0 {
            if sign != 0.0 && c.signum() != sign { return false; }
            sign = c.signum();
        }
    }
    true
}

/// The ray (o, d) and the plane through c across n: where they meet.
fn meet_plane(o: [f64; 3], d: [f64; 3], c: [f64; 3], n: [f64; 3]) -> Option<[f64; 3]> {
    let dn = dot(d, n);
    if dn.abs() < 1e-9 { return None; }
    let t = dot(sub(c, o), n) / dn;
    Some(add(o, scale(d, t)))
}

/// How far along the line through c along unit a lies the point of it nearest to the ray (o, d).
fn along_line(o: [f64; 3], d: [f64; 3], c: [f64; 3], a: [f64; 3]) -> Option<f64> {
    // the two lines' nearest points: solve for s on the axis
    let w = sub(c, o);
    let (aa, ad, dd) = (dot(a, a), dot(a, d), dot(d, d));
    let den = aa * dd - ad * ad;
    if den.abs() < 1e-9 { return None; }
    Some((ad * dot(w, d) - dd * dot(w, a)) / den)
}

fn values(st: &State, t: Target) -> [f64; 11] {
    match t {
        Target::Pivot(layer, i) => { let p = layer.position(st, i).unwrap_or_default(); [p[0], p[1], p[2], 0., 0., 0., 1., 0., 1., 1., 1.] }
        Target::World(item) => {
            let p = st.world_placement(item).unwrap_or_default();
            [p.pos[0], p.pos[1], p.pos[2], p.rot[0], p.rot[1], p.rot[2], p.size, 0.0, 1., 1., 1.]
        }
        Target::Lamp(i) => {
            let l = st.lamps[i];
            [l.pos[0], l.pos[1], l.pos[2], l.rot[0], l.rot[1], l.rot[2], l.size[0], l.size[1], 1., 1., 1.]
        }
        Target::Part(_) | Target::Whole | Target::ConstructionNode => {
            let p = t.construction(st).map(|h| h.pose).unwrap_or_default();
            let s = match t { Target::Part(id) => st.csg.construction_part(id).and_then(|i| st.csg.parts.get(i)).and_then(|p| p.stretch).unwrap_or([1.0; 3]), _ => [1.0; 3] };
            [p.pos[0], p.pos[1], p.pos[2], p.rot[0], p.rot[1], p.rot[2], p.size, 0.0, s[0], s[1], s[2]]
        }
    }
}

fn exists(st: &State, t: Target) -> bool {
    match t {
        Target::Pivot(layer, i) => layer.position(st, i).is_some(),
        Target::World(item) => st.world_placement(item).is_some(),
        Target::Lamp(i) => i < st.lamps.len(),
        Target::Part(id) => st.csg.construction_part(id).is_some(),
        Target::ConstructionNode => st.selected_node_handle().is_some(),
        Target::Whole => st.edit_scope.object().is_some(),
    }
}

/// Which of the numbers a size square along the thing's own axis k changes: a tube's length along its
/// x and its radius across; a panel's half width and half height; a part's stretch along that axis
/// (gf_core::stretch). None: a layer (it has one size), a ball.
fn stretch_slot(st: &State, t: Target, k: usize) -> Option<usize> {
    if let Target::Part(id) = t { return st.csg.construction_part(id).map(|_| 8 + k); }
    let Target::Lamp(i) = t else { return None };
    match (st.lamps.get(i)?.kind, k) {
        (gf_core::lamps::LampKind::Tube, 0) => Some(7),
        (gf_core::lamps::LampKind::Tube, _) => Some(6),
        (gf_core::lamps::LampKind::Panel, 0) => Some(6),
        (gf_core::lamps::LampKind::Panel, 1) => Some(7),
        _ => None,
    }
}
fn part_axis(p: Part) -> usize {
    match p { Part::Axis(k) | Part::Plane(k) | Part::Ring(k) | Part::Stretch(k) => k, _ => 0 }
}

impl Handles {
    /// What the handles act on, if they show.
    pub fn target(&self, st: &State) -> Option<Target> {
        if !self.on { return None; }
        self.chosen(st)
    }
    /// What is chosen, whether or not the handles show.
    pub fn chosen(&self, st: &State) -> Option<Target> {
        if !self.picked { return None; }
        if let Some((layer, i)) = self.pivot.filter(|(l, i)| l.selected(st) && l.position(st, *i).is_some()) { return Some(Target::Pivot(layer, i)); }
        if let Some(i) = self.lamp.filter(|i| *i < st.lamps.len()) { return Some(Target::Lamp(i)); }
        if st.edit_scope.object().is_none() {
            return st.world_selection().last().copied().map(Target::World);
        }
        // (a part selected in the tree's list: it is what the handles move)
        if let Some(i) = st.csg.sel.filter(|i| *i < st.csg.parts.len()) { return st.csg.part_id(i).map(Target::Part); }
        if !st.edit_scope.construction().is_empty() { return Some(Target::ConstructionNode); }
        st.csg.whole.then_some(Target::Whole)
    }

    /// What the corners mark: what is chosen, or, before anything has been clicked, the objects the panel
    /// shows as selected (the window opens with one selected; its corners say so, while the handles wait
    /// for a click on it, see `starting`).
    pub fn marked(&self, st: &State) -> Option<Target> {
        self.chosen(st).or_else(|| if st.edit_scope.object().is_none() { st.world_selection().last().copied().map(Target::World) } else { None })
    }

    /// As the window opens: the "all" tool taken up (docs/view_tools.md), with nothing chosen yet, so that
    /// nothing is drawn over the first picture until a shape is clicked.
    pub fn starting() -> Self {
        Self { on: true, mode: Mode::All, ..Self::default() }
    }

    /// Show or hide them (G). Shown, they are on whatever is selected.
    pub fn toggle(&mut self) {
        self.on = !self.on;
        self.grab = None;
        if self.on {
            self.picked = true;
            self.select = false;
        }
    }

    /// The tool in use.
    pub fn tool(&self) -> Tool {
        match (self.on, self.mode) {
            (true, Mode::All) => Tool::All,
            (true, Mode::Move) => Tool::Move,
            (true, Mode::Turn) => Tool::Turn,
            (true, Mode::Size) => Tool::Size,
            _ if self.select => Tool::Select,
            _ => Tool::Camera,
        }
    }
    /// Take up a tool. One with handles puts them on what is selected.
    pub fn set_tool(&mut self, t: Tool) {
        self.construction_origin = false;
        self.pivot = None;
        self.grab = None;
        match t {
            Tool::All | Tool::Move | Tool::Turn | Tool::Size => {
                self.mode = match t { Tool::All => Mode::All, Tool::Move => Mode::Move, Tool::Turn => Mode::Turn, _ => Mode::Size };
                self.on = true;
                self.picked = true;
                self.select = false;
            }
            Tool::Select => (self.on, self.select) = (false, true),
            Tool::Camera => (self.on, self.select) = (false, false),
        }
    }
    /// Choose a layer (None) or a lamp, as a click on it does.
    pub fn pick(&mut self, lamp: Option<usize>) {
        self.pivot = None;
        self.construction_origin = false;
        self.lamp = lamp;
        self.picked = true;
    }
    /// Show the handles on this lamp (None: on the selected layer), as its "handles" button does.
    pub fn show_on(&mut self, lamp: Option<usize>) {
        self.pick(lamp);
        if !self.on { self.toggle(); }
    }
    /// The handles are drawn on this lamp (None: on what is chosen in the scene). Its button reads
    /// "hide handles" exactly while this holds.
    pub fn shown_on(&self, st: &State, lamp: Option<usize>) -> bool {
        match (self.target(st), lamp) {
            (Some(Target::Lamp(i)), Some(j)) => i == j,
            (Some(Target::Lamp(_)), None) | (None, _) | (Some(_), Some(_)) => false,
            (Some(_), None) => true,
        }
    }
    /// Let go of what is chosen, as a click on nothing does.
    pub fn let_go(&mut self) {
        self.picked = false;
        self.grab = None;
    }

    /// The part of the handles under the pointer, if any.
    pub fn hit(&self, st: &State, rect: Rect, at: Pos2) -> Option<Part> {
        let target = self.target(st)?;
        let v = View::of(st, rect, target, self.local)?;
        let c = v.screen(v.c)?;
        let mut best: Option<(f32, Part)> = None;
        let mut take = |d: f32, part: Part| {
            if d <= REACH_PT && best.is_none_or(|(b, _)| d < b) { best = Some((d, part)); }
        };
        match self.mode {
            Mode::Move => {
                take((at - c).length() - CENTRE_PT, Part::Centre);
                for k in 0..3 {
                    take(dist_to_runs(at, &v.arrow(k)), Part::Axis(k));
                    if let Some(sq) = v.square(k) { if inside(at, &sq) { take(REACH_PT - 0.5, Part::Plane(k)); } }
                }
            }
            Mode::Turn => {
                for k in 0..3 {
                    let pts: Vec<Pos2> = v.ring(k).into_iter().map(|(p, _)| p).collect();
                    take(dist_to_runs(at, &[pts]), Part::Ring(k));
                }
            }
            Mode::Size => take(((at - c).length() - v.scale_radius()).abs(), Part::Scale),
            Mode::All => {
                let rr = v.ring_radius();
                let r = (at - c).length();
                take(r - CENTRE_PT, Part::Centre);
                for k in 0..3 {
                    // (the whole shaft takes the drag, and a little more than the other handles' reach)
                    let d = dist_to_runs(at, &v.arrow_to(k, ALL_ARROW));
                    if d <= SHAFT_PT { take(d.min(REACH_PT - 0.25), Part::Axis(k)); }
                    let pts: Vec<Pos2> = v.ring(k).into_iter().map(|(p, _)| p).collect();
                    take(dist_to_runs(at, &[pts]), Part::Ring(k));
                    if let (Some(q), Some(_)) = (v.stretch_at(k), stretch_slot(st, target, k)) { take((at - q).length() - 3.0, Part::Stretch(k)); }
                }
                take((r - rr * SPIN_R as f32).abs(), Part::Spin);
                take((r - rr * SIZE_R as f32).abs(), Part::Scale);
                // inside the rings and on none of them: the ball
                if best.is_none() && r < rr - 4.0 { best = Some((REACH_PT, Part::Ball)); }
            }
        }
        best.map(|(_, p)| p)
    }

    /// Take hold of a part at the pointer.
    pub fn begin(&mut self, st: &State, rect: Rect, at: Pos2, part: Part) {
        let Some(target) = self.target(st) else { return };
        let Some(v) = View::of(st, rect, target, self.local) else { return };
        let mut g = Grab { part, target, start: values(st, target), at0: at, hit0: v.c, s0: 0.0, tan: Vec2::X, rad: v.ring_radius().max(20.0), seen: Seen::of(st, target) };
        if let (Part::Stretch(k), Some(c), Some(q)) = (part, v.screen(v.c), v.stretch_at(part_axis(part))) {
            // the way the thing's own axis runs on the screen
            let _ = k;
            if (q - c).length() > 1e-3 { g.tan = (q - c).normalized(); }
        }
        if let Part::Ring(k) = part {
            // the point of the ring nearest the pointer, and the way it runs there on the screen
            let (a, b) = v.across(k);
            let mut best = (f32::MAX, [0.0; 3]);
            for i in 0..144 {
                let t = i as f64 / 144.0 * std::f64::consts::TAU;
                let off = add(scale(a, t.cos() * v.len), scale(b, t.sin() * v.len));
                if let Some(s) = v.screen(add(v.c, off)) {
                    let d = (s - at).length();
                    if d < best.0 { best = (d, off); }
                }
            }
            let off = best.1;
            let step = scale(cross(v.axes[k], off), 0.02);
            if let (Some(c), Some(p0), Some(p1)) = (v.screen(v.c), v.screen(add(v.c, off)), v.screen(add(add(v.c, off), step))) {
                let d = p1 - p0;
                if d.length() > 1e-6 { g.tan = d.normalized(); }
                g.rad = (p0 - c).length().max(20.0);
            }
        }
        if let Some((o, d)) = v.ray(at) {
            match part {
                Part::Axis(k) => g.s0 = along_line(o, d, v.c, v.axes[k]).unwrap_or(0.0),
                Part::Plane(k) | Part::Ring(k) => g.hit0 = meet_plane(o, d, v.c, v.axes[k]).unwrap_or(v.c),
                Part::Centre => g.hit0 = meet_plane(o, d, v.c, v.toward_camera(v.c)).unwrap_or(v.c),
                Part::Scale | Part::Spin | Part::Ball | Part::Stretch(_) => {}
            }
        }
        self.grab = Some(g);
    }

    /// The pointer has moved to `at` during a drag: change the placement. `snap`: in steps; `fine`: a
    /// tenth as far.
    pub fn drag(&mut self, st: &mut State, rect: Rect, at: Pos2, snap: bool, fine: bool) {
        // (with snapping left on, Ctrl lets the handle go freely)
        let snap = snap != self.snap;
        let Some(g) = self.grab.clone() else { return };
        if !g.holds(st) { self.grab = None; return; }
        // the frame as it was at the press: the middle moves under the drag, the constraint does not
        let Some(mut v) = View::of(st, rect, g.target, self.local) else { return };
        v.c = apply(&v.m, [g.start[0], g.start[1], g.start[2]]);
        let Some((o, d)) = v.ray(at) else { return };
        let k_fine = if fine { 0.1 } else { 1.0 };
        let mut out = g.start;
        match g.part {
            Part::Axis(k) => {
                let Some(s) = along_line(o, d, v.c, v.axes[k]) else { return };
                for i in 0..3 { out[i] = g.start[i] + v.dirs[k][i] * (s - g.s0) * k_fine; }
            }
            Part::Plane(_) | Part::Centre => {
                let n = match g.part { Part::Plane(k) => v.axes[k], _ => v.toward_camera(v.c) };
                // a plane seen edge on would throw the layer far off: it moves no further than the drag would across the picture
                let Some(hit) = meet_plane(o, d, v.c, n) else { return };
                let delta = apply(&transpose(&v.m), scale(sub(hit, g.hit0), k_fine));
                let far = 6.0 * v.len;
                let l = dot(delta, delta).sqrt();
                let delta = if l > far { scale(delta, far / l) } else { delta };
                for i in 0..3 { out[i] = g.start[i] + delta[i]; }
            }
            Part::Ring(k) => {
                // the pointer carried along the ring as it runs on the screen: as far as its distance from
                // the middle, a radian (the same however the ring is seen, edge on included)
                let angle = ((at - g.at0).dot(g.tan) / g.rad) as f64;
                let mut deg = angle / DEG * k_fine;
                if snap { deg = (deg / 15.0).round() * 15.0; }
                // a turn about the handle's axis (the scene's, or the layer's own as it was at the press), laid
                // onto the turn the layer had
                let ax = if self.local { apply(&euler(g.start[3], g.start[4], g.start[5]), { let mut a = [0.0; 3]; a[k] = 1.0; a }) } else { v.dirs[k] };
                let r = mul(&axis_angle(ax[0], ax[1], ax[2], deg * DEG), &euler(g.start[3], g.start[4], g.start[5]));
                let t = euler_of(&r);
                out[3] = t[0];
                out[4] = t[1];
                out[5] = t[2];
            }
            Part::Spin | Part::Ball => {
                let Some(c) = v.screen(v.c) else { return };
                // the turn in the camera's frame: about the line of sight by how far the pointer has gone
                // round the middle; or rolled like a ball by the drag, a radian for the rings' radius
                let (axis, mut angle) = if g.part == Part::Spin {
                    let (a0, a1) = ((g.at0 - c).angle(), (at - c).angle());
                    let mut da = (a1 - a0) as f64;
                    while da > std::f64::consts::PI { da -= std::f64::consts::TAU; }
                    while da < -std::f64::consts::PI { da += std::f64::consts::TAU; }
                    // (the screen counts its angles clockwise; a turn towards the camera, anticlockwise)
                    (v.toward_camera(v.c), -da)
                } else {
                    let d = at - g.at0;
                    let l = d.length() as f64;
                    if l < 1e-3 { return; }
                    ([d.y as f64 / l, d.x as f64 / l, 0.0], l / g.rad as f64)
                };
                angle *= k_fine;
                if snap { angle = (angle / DEG / 15.0).round() * 15.0 * DEG; }
                let turn = mul(&transpose(&v.m), &mul(&axis_angle(axis[0], axis[1], axis[2], angle), &v.m));
                let t = euler_of(&mul(&turn, &euler(g.start[3], g.start[4], g.start[5])));
                out[3] = t[0];
                out[4] = t[1];
                out[5] = t[2];
            }
            Part::Stretch(k) => {
                let Some(c) = v.screen(v.c) else { return };
                let (s0, s1) = ((g.at0 - c).dot(g.tan).max(4.0) as f64, (at - c).dot(g.tan) as f64);
                let f = (1.0 + (s1 / s0 - 1.0) * k_fine).max(0.02);
                if let Some(i) = stretch_slot(st, g.target, k) {
                    out[i] = g.start[i] * f;
                    if snap { out[i] = (out[i] / 0.05).round().max(1.0) * 0.05; }
                }
            }
            Part::Scale => {
                let Some(c) = v.screen(v.c) else { return };
                let (r0, r1) = ((g.at0 - c).length().max(4.0) as f64, (at - c).length() as f64);
                let f = 1.0 + (r1 / r0 - 1.0) * k_fine;
                out[6] = g.start[6] * f;
                out[7] = g.start[7] * f;
                if snap { out[6] = (out[6] / 0.05).round() * 0.05; }
                out[6] = out[6].max(0.01);
            }
        }
        if snap && matches!(g.part, Part::Axis(_) | Part::Plane(_) | Part::Centre) {
            if let Target::World(item) = g.target {
                use gf_core::world_transform::DragConstraint;
                if out[..3] != g.start[..3] {
                    let constraint = match g.part {
                        Part::Axis(k) => DragConstraint::Axis(v.dirs[k]),
                        Part::Plane(k) => DragConstraint::Plane { normal: v.dirs[k] },
                        _ => DragConstraint::Plane { normal: apply(&transpose(&v.m), v.toward_camera(v.c)) },
                    };
                    let source = if self.move_origin { Some(gf_core::world_transform::Reference::Origin) } else { self.snap_reference };
                    let dest = st.world_snap(item, [out[0], out[1], out[2]], constraint, source, v.len * 0.07);
                    out[..3].copy_from_slice(&dest);
                }
            } else {
                for x in out.iter_mut().take(3) { *x = (*x / 0.05).round() * 0.05; }
            }
        }
        self.write(st, g.target, &out, true);
    }

    /// Let go: the placement is built at full detail.
    pub fn end(&mut self, st: &mut State) {
        if let Some(g) = self.grab.take() {
            if g.holds(st) {
                let now = values(st, g.target);
                self.write(st, g.target, &now, false);
            }
        }
    }

    /// Esc during a drag: the layer goes back to where the drag began. True if there was a drag.
    pub fn cancel(&mut self, st: &mut State) -> bool {
        let Some(g) = self.grab.take() else { return false };
        if g.holds(st) { self.write(st, g.target, &g.start, false); }
        true
    }

    /// Write the numbers through their sliders (so that they are what the sliders say, within their range).
    fn write(&self, st: &mut State, t: Target, v: &[f64; 11], quick: bool) {
        let turn = |c: usize| (v[3 + c] + 180.0).rem_euclid(360.0) - 180.0;
        if let Target::World(item) = t {
            let pos = [v[0], v[1], v[2]];
            if self.move_origin { st.set_world_origin(item, pos); } else {
                let frame = gf_core::layers::Placement { pos, rot: [turn(0), turn(1), turn(2)], size: v[6].max(0.01) };
                st.transform_world_items(&st.world_selection(), item, frame);
            }
            st.request(gf_core::build::BuildRequest::handle(quick));
            return;
        }
        if let Target::Pivot(layer, i) = t {
            layer.set_position(st, i, [v[0], v[1], v[2]]);
            st.sculpt_changed(quick);
            return;
        }
        // (a layer's sliders: x, y, z, three turns, size; a lamp's begin the same, then its second size)
        // (the tree's placements have no knobs: they are written as they are, within their sliders' ranges)

        if let Some(h) = t.construction(st) {
            if self.construction_origin && matches!(t, Target::Part(_) | Target::ConstructionNode) {
                st.set_construction_origin(apply(&h.axes, [v[0], v[1], v[2]]));
                st.request(gf_core::build::BuildRequest::handle(quick));
                return;
            }
            let place = h.local(gf_core::layers::Placement { pos: [v[0], v[1], v[2]], rot: [turn(0), turn(1), turn(2)], size: v[6].max(0.01) });
            match t {
                Target::Part(id) => if let Some(i) = st.csg.construction_part(id) {
                    st.csg.parts[i].place = place;
                    // (its stretch, once it has one or a square has given it one)
                    let s = [v[8], v[9], v[10]];
                    if s != [1.0; 3] || st.csg.parts[i].stretch.is_some() { st.csg.parts[i].stretch = Some(s.map(|x| x.max(1e-3))); }
                },
                Target::ConstructionNode => {
                    if let Some(node) = st.csg.edit_construction().node_mut(&st.edit_scope.construction()) { node.place = place; }
                }
                _ => {
                    if !st.has_tree(st.stack.sel) { st.csg.edit_construction(); }
                    st.csg.place = place;
                }
            }
            st.request(gf_core::build::BuildRequest::handle(quick));
            return;
        }
        let Target::Lamp(i) = t else { return };
        for (c, k) in st.lamp_knobs_of(i).into_iter().take(8).enumerate() {
            // a number that is being animated is the animation's: the handle leaves it alone
            if st.tracks.contains_key(&k.key) { continue; }
            let x = if c >= 3 && c <= 5 { ((v[c] + 540.0) % 360.0) - 180.0 } else { v[c] };
            let x = x.clamp(k.lo, k.hi);
            if (st.get(&k.r) - x).abs() > 1e-12 || !quick { st.user_set(&k, x, quick); }
        }
    }

    /// Corners round what is chosen, where its box (or the ball it fits in) shows on the screen, and its name over
    /// them: paint only, quiet enough not to be mistaken for part of the picture.
    fn mark(&self, p: &Painter, st: &State, rect: Rect) {
        let Some(t) = self.marked(st) else { return };
        let Some((c, r, name)) = (match t {
            Target::World(item) => st.world_framing(item).map(|(p, r)| (apply(&st.view_r(), p), r, st.world_name(item))),
            Target::Pivot(_, _) => None,
            Target::Part(_) | Target::Whole | Target::ConstructionNode => t.construction(st).map(|h| {
                let (c, size) = h.in_scene();
                let name = match t {
                    Target::Part(id) => format!("part {}", st.csg.construction_part(id).unwrap_or(0) + super::csg::first_number(st)),
                    Target::ConstructionNode => st.csg.construction_tree().node(&st.edit_scope.construction()).map(|n| n.name.clone()).unwrap_or_default(),
                    _ => "csg shape".into(),
                };
                (apply(&st.view_r(), c), gf_core::FIT * size.max(1e-3), name)
            }),
            Target::Lamp(i) => st.lamp_in_view(i).map(|(c, _)| {
                let l = &st.lamps[i];
                (c, l.size[0].max(l.size[1]).max(0.05) * 1.15, format!("lamp {}", i + 1))
            }),
        }) else { return };
        let v = View { cam: st.camera(), m: st.view_r(), c, axes: [[0.0; 3]; 3], dirs: [[0.0; 3]; 3], own: [[0.0; 3]; 3], len: 0.4, rect };
        // an object or group in the world: the eight corners of its box, which hug it far closer than the ball
        // round the box does (that ball is wider than the window for a shape that fills the view);
        // otherwise the ball's outline, near enough: three great circles round it
        let (mut lo, mut hi, mut seen, mut all) = (Pos2::new(f32::MAX, f32::MAX), Pos2::new(f32::MIN, f32::MIN), 0, 0);
        let world_box = if let Target::World(item) = t { world_screen_box(st, item, rect) } else { None };
        if let Some(b) = world_box { (lo, hi, seen, all) = (b.min, b.max, 1, 1); }
        for k in 0..if world_box.is_some() { 0 } else { 3 } {
            for i in 0..32 {
                let a = i as f64 / 32.0 * std::f64::consts::TAU;
                let mut d = [0.0; 3];
                d[(k + 1) % 3] = a.cos() * r;
                d[(k + 2) % 3] = a.sin() * r;
                all += 1;
                if let Some(q) = v.screen(add(c, d)) {
                    seen += 1;
                    lo = lo.min(q);
                    hi = hi.max(q);
                }
            }
        }
        if seen * 2 < all { return; }
        let b = Rect::from_min_max(lo, hi).expand(4.0);
        if !b.intersects(rect) || b.width() > 4.0 * rect.width() { return; }
        // (kept inside the window, so that the corners of a shape that overflows it still show)
        let b = b.intersect(rect.shrink(6.0));
        let arm = (b.width().min(b.height()) * 0.18).clamp(5.0, 16.0);
        let ink = Color32::from_white_alpha(150);
        let shade = Stroke::new(3.0, Color32::from_black_alpha(70));
        for (corner, dx, dy) in [(b.left_top(), 1.0, 1.0), (b.right_top(), -1.0, 1.0), (b.left_bottom(), 1.0, -1.0), (b.right_bottom(), -1.0, -1.0)] {
            let pts = vec![corner + eframe::egui::vec2(dx * arm, 0.0), corner, corner + eframe::egui::vec2(0.0, dy * arm)];
            p.add(Shape::line(pts.clone(), shade));
            p.add(Shape::line(pts, Stroke::new(1.2, ink)));
        }
        // (on a small dark backing, as the line at the foot of the window has: readable over bright glass)
        let font = eframe::egui::FontId::monospace(10.0);
        let at = Pos2::new(b.left().max(rect.left() + 4.0), (b.top() - 3.0).max(rect.top() + 12.0));
        pill_text(p, at + eframe::egui::vec2(5.0, -2.0), eframe::egui::Align2::LEFT_BOTTOM, &name, font, Color32::from_white_alpha(190));
    }

    /// Paint the handles over the picture, and the mark round what is chosen.
    pub fn paint(&mut self, p: &Painter, st: &State, rect: Rect, pointer: Option<Pos2>) {
        self.mark(p, st, rect);
        self.spots.clear();
        let Some(target) = self.target(st) else { self.hover = None; return };
        let Some(v) = View::of(st, rect, target, self.local) else { return };
        let Some(c) = v.screen(v.c) else { return };
        // (a point well inside each part, for tests)
        self.spots.push(("handle:centre", c));
        self.spots.push(("handle:size", c + Vec2::new(v.scale_radius(), 0.0)));
        for k in 0..3 {
            if let Some(q) = v.arrow(k).first().and_then(|r| r.get(r.len() * 2 / 3).copied()) { self.spots.push((["handle:x", "handle:y", "handle:z"][k], q)); }
            // (half way between two other axes, on the near side: where no other ring crosses)
            let (a, b) = v.across(k);
            let to_cam = v.toward_camera(v.c);
            let spot = [45.0f64, 135.0, 225.0, 315.0].iter().filter_map(|deg| {
                let t = deg.to_radians();
                let off = add(scale(a, t.cos() * v.len), scale(b, t.sin() * v.len));
                if dot(off, to_cam) < 0.0 { return None; }
                v.screen(add(v.c, off))
            }).max_by(|p, q| (*p - c).length().total_cmp(&(*q - c).length()));
            if let Some(q) = spot { self.spots.push((["handle:ring-x", "handle:ring-y", "handle:ring-z"][k], q)); }
        }
        {
            let rr = v.ring_radius();
            self.spots.push(("handle:spin", c + Vec2::new(0.0, -rr * SPIN_R as f32)));
            self.spots.push(("handle:sizering", c + Vec2::new(0.0, rr * SIZE_R as f32)));
            // (inside the rings, clear of the arrows: between two of them, a third of the way out)
            let ball = (0..12).map(|i| { let a = i as f32 / 12.0 * std::f32::consts::TAU; c + Vec2::new(a.cos(), a.sin()) * rr * 0.45 })
                .max_by(|p, q| {
                    let clear = |at: &Pos2| (0..3).map(|k| dist_to_runs(*at, &v.arrow_to(k, ALL_ARROW))).fold(f32::MAX, f32::min);
                    clear(p).total_cmp(&clear(q))
                });
            if let Some(b) = ball { self.spots.push(("handle:ball", b)); }
            for k in 0..3 {
                if let Some(q) = v.arrow_to(k, ALL_ARROW).first().and_then(|r| r.get(r.len() / 2).copied()) { self.spots.push((["handle:all-x", "handle:all-y", "handle:all-z"][k], q)); }
                if let (Some(q), Some(_)) = (v.stretch_at(k), stretch_slot(st, target, k)) { self.spots.push((["handle:stretch-x", "handle:stretch-y", "handle:stretch-z"][k], q)); }
            }
        }
        let lit = self.grab.as_ref().map(|g| g.part).or_else(|| pointer.and_then(|at| self.hit(st, rect, at)));
        self.hover = lit;
        let ink = |part: Part, base: Color32| if lit == Some(part) { self::lit(base) } else { base };
        let shadow = Stroke::new(4.0, Color32::from_black_alpha(90));
        let thin = Stroke::new(3.0, Color32::from_black_alpha(70));
        // a ring about axis k, its far half faint
        let ring = |k: usize, width: f32| {
            let col = ink(Part::Ring(k), AXIS_INK[k]);
            let pts = v.ring(k);
            for front in [false, true] {
                let mut run: Vec<Pos2> = Vec::new();
                let flush = |run: &mut Vec<Pos2>| {
                    if run.len() > 1 {
                        let s = if front { Stroke::new(width, col) } else { Stroke::new(1.0, col.gamma_multiply(0.3)) };
                        if front { p.add(Shape::line(run.clone(), thin)); }
                        p.add(Shape::line(std::mem::take(run), s));
                    }
                    run.clear();
                };
                for (q, near) in &pts {
                    if *near != front { flush(&mut run); continue; }
                    run.push(*q);
                }
                flush(&mut run);
            }
        };
        // an arrow along axis k, to this share of its length
        let arrow = |k: usize, f: f64, width: f32, head: f32| {
            let col = ink(Part::Axis(k), AXIS_INK[k]);
            for run in v.arrow_to(k, f) {
                p.add(Shape::line(run.clone(), thin));
                p.add(Shape::line(run.clone(), Stroke::new(if lit == Some(Part::Axis(k)) { width + 1.0 } else { width }, col)));
                let (a, b) = (run[run.len() - 2], run[run.len() - 1]);
                let dir = (b - a).normalized();
                let side = eframe::egui::vec2(-dir.y, dir.x);
                p.add(Shape::convex_polygon(vec![b + dir * head, b + side * head * 0.5, b - side * head * 0.5], col, Stroke::NONE));
            }
        };
        match self.mode {
            Mode::All => {
                let rr = v.ring_radius();
                // the ball: barely there, a little more when the pointer is on it
                p.circle_filled(c, rr, Color32::from_white_alpha(if lit == Some(Part::Ball) { 22 } else { 9 }));
                p.circle_stroke(c, rr, Stroke::new(1.0, Color32::from_white_alpha(if lit == Some(Part::Ball) { 90 } else { 30 })));
                for k in 0..3 { ring(k, 1.6); }
                let spin = ink(Part::Spin, Color32::from_white_alpha(120));
                p.circle_stroke(c, rr * SPIN_R as f32, thin);
                p.circle_stroke(c, rr * SPIN_R as f32, Stroke::new(if lit == Some(Part::Spin) { 2.2 } else { 1.4 }, spin));
                let size = ink(Part::Scale, SIZE_INK);
                p.circle_stroke(c, rr * SIZE_R as f32, thin);
                p.circle_stroke(c, rr * SIZE_R as f32, Stroke::new(if lit == Some(Part::Scale) { 2.4 } else { 1.6 }, size));
                for k in 0..3 { arrow(k, ALL_ARROW, 1.8, 8.0); }
                for k in 0..3 {
                    if let (Some(q), Some(_)) = (v.stretch_at(k), stretch_slot(st, target, k)) {
                        let col = ink(Part::Stretch(k), AXIS_INK[k]);
                        p.rect(Rect::from_center_size(q, Vec2::splat(7.0)), 1.0, col, Stroke::new(1.0, Color32::from_black_alpha(120)), eframe::egui::StrokeKind::Outside);
                    }
                }
                p.circle_filled(c, 6.0, Color32::from_black_alpha(110));
                p.circle_filled(c, 4.5, ink(Part::Centre, Color32::from_rgb(110, 225, 230)));
            }
            Mode::Move => {
                for k in 0..3 {
                    if let Some(sq) = v.square(k) {
                        let col = ink(Part::Plane(k), AXIS_INK[k]);
                        p.add(Shape::convex_polygon(sq, col.gamma_multiply(0.35), Stroke::new(1.0, col)));
                    }
                }
                for k in 0..3 { arrow(k, 1.0, 2.0, 9.0); }
                p.circle_filled(c, 5.5, Color32::from_black_alpha(110));
                p.circle_filled(c, 4.0, ink(Part::Centre, Color32::from_gray(235)));
            }
            Mode::Turn => {
                for k in 0..3 { ring(k, 2.0); }
                p.circle_filled(c, 3.0, Color32::from_gray(235));
            }
            Mode::Size => {
                let col = ink(Part::Scale, Color32::from_gray(225));
                p.circle_stroke(c, v.scale_radius(), shadow);
                p.circle_stroke(c, v.scale_radius(), Stroke::new(2.0, col));
                p.circle_filled(c, 3.0, col);
            }
        }
        // a word to say what they do, under the middle
        let font = eframe::egui::FontId::monospace(10.0);
        // (the combined handles say what the one under the pointer does instead)
        let word = match (self.mode, lit) {
            (Mode::All, Some(Part::Axis(k))) => format!("move along {}", ["x", "y", "z"][k]),
            (Mode::All, Some(Part::Ring(k))) => format!("turn about {}", ["x", "y", "z"][k]),
            (Mode::All, Some(Part::Centre)) => "move across the picture".into(),
            (Mode::All, Some(Part::Spin)) => "turn about the line of sight".into(),
            (Mode::All, Some(Part::Ball)) => "turn any way".into(),
            (Mode::All, Some(Part::Scale)) => "size".into(),
            (Mode::All, Some(Part::Stretch(k))) => format!("size along its own {}", ["x", "y", "z"][k]),
            (Mode::All, _) => String::new(),
            (m, _) => format!("{} · T", m.word()),
        };
        if !word.is_empty() { pill_text(p, c + eframe::egui::vec2(0.0, 16.0), eframe::egui::Align2::CENTER_TOP, &word, font, Color32::from_white_alpha(200)); }
    }
}

/// The shown lamp under a point of the screen, if any, and how far along the pixel's ray: lamps are known
/// exactly, so this is worked out here with the shapes the tracer draws them with (ui/pick.rs).
/// Where an object or group of the world shows on the screen: the box round the eight corners of its
/// bounds as they project (none if fewer than half are in front of the camera).
pub fn world_screen_box(st: &State, item: gf_core::groups::Item, rect: Rect) -> Option<Rect> {
    let bx = st.world_bounds(item).filter(|b| b[0] <= b[3])?;
    let (cam, m, h) = (st.camera(), st.view_r(), rect.height() as f64 / 2.0);
    let (mut lo, mut hi, mut seen) = (Pos2::new(f32::MAX, f32::MAX), Pos2::new(f32::MIN, f32::MIN), 0);
    for mask in 0..8 {
        let p: [f64; 3] = std::array::from_fn(|k| bx[if mask & (1 << k) == 0 { k } else { k + 3 }]);
        if let Some(uv) = cam.project(apply(&m, p)) {
            let q = Pos2::new((rect.center().x as f64 + uv[0] * h) as f32, (rect.center().y as f64 - uv[1] * h) as f32);
            seen += 1;
            lo = lo.min(q);
            hi = hi.max(q);
        }
    }
    (seen >= 4).then(|| Rect::from_min_max(lo, hi))
}

pub fn pick_lamp(st: &State, rect: Rect, at: Pos2) -> Option<(f64, usize)> {
    let h = rect.height() as f64 / 2.0;
    let (o, d) = st.camera().ray([(at.x - rect.center().x) as f64 / h, -(at.y - rect.center().y) as f64 / h])?;
    super::pick::lamp_on_ray(st, o, d)
}

/// The layer whose shape is under a point of the picture (fx, fy: fractions of its width and height
/// from the top left), from the distance stored with the picture's pixels and each layer's own volume.
/// None if the point shows no shape.
pub fn pick_layer(e: &crate::engine::Engine, fx: f64, fy: f64) -> Option<usize> {
    pick_occurrence(e, fx, fy).map(|hit| hit.source)
}

/// The transient occurrence ordinal is useful to the viewport. Editing always
/// addresses the source layer, whose stored placement is independent of a hit.
#[derive(Clone, Copy, Debug)]
pub struct OccurrenceHit { pub source: usize, pub occurrence: usize, /** how far along the pixel's ray the shape is */ pub depth: f64 }

pub fn pick_occurrence(e: &crate::engine::Engine, fx: f64, fy: f64) -> Option<OccurrenceHit> {
    let (_, (tw, th)) = e.renderer.size();
    if tw < 2 || th < 2 { return None; }
    let (x, y) = ((fx * tw as f64).clamp(0.0, tw as f64 - 1.0) as u32, (fy * th as f64).clamp(0.0, th as f64 - 1.0) as u32);
    let px = e.renderer.read_hdr(x, y, 1, 1).ok()?;
    let t = px[3] as f64;
    if t <= 0.0 { return None; }
    let st = &e.state;
    let n = st.layer_count();
    if let Some(source) = st.edit_scope.object().filter(|i| *i < n) { return Some(OccurrenceHit { source, occurrence: 0, depth: t }); }
    if n < 2 && !st.stack.has_world_modifiers() { return Some(OccurrenceHit { source: 0, occurrence: 0, depth: t }); }
    // the point in the camera's frame, then in the object's
    let uv = [((x as f64 + 0.5) * 2.0 - tw as f64) / th as f64, (th as f64 - (y as f64 + 0.5) * 2.0) / th as f64];
    let (o, d) = st.camera().ray(uv)?;
    let p = apply(&transpose(&st.view_r()), add(o, scale(d, t)));
    let plan = st.stack.world_plan();
    let mut best: Option<(f64, OccurrenceHit)> = None;
    let mut pending = Vec::new();
    let mut field_probes = Vec::new();
    let mut consider = |dist: f64, hit| {
        if best.is_none_or(|(v, _)| dist.abs() < v) { best = Some((dist.abs(), hit)); }
    };
    for (source, instances) in plan.placements.iter().enumerate() {
        if !st.stack.layers[source].show || st.edit_scope.object().is_some_and(|i| i != source) { continue; }
        if let Some((volume,scene_scale))=e.layers.world_fields.volume(source) {
            let hb=gf_core::HB*scene_scale;
            let at=p.map(|x| (((x/hb+1.)*0.5*f64::from(volume.res)-0.5).round()).clamp(0.,f64::from(volume.res-1)) as u32);
            field_probes.push((e.gpu.read_regions_later(&volume.tex,&[at],1,1,8),source,scene_scale));
            continue;
        }
        let volume = e.csg.layer_vols.get(source).and_then(|v| v.as_ref()).map(|(_, v)| v)
            .or_else(|| e.baker.layer_volume(source));
        let mut origins = Vec::new();
        let mut hits = Vec::new();
        for (occurrence, pl) in instances.iter().enumerate() {
            let size = pl.size.abs().max(1e-9);
            let q = scale(apply(&transpose(&euler(pl.rot[0], pl.rot[1], pl.rot[2])), sub(p, pl.pos)), pl.size.signum() / size);
            let hb = gf_core::HB;
            if q.iter().any(|v| v.abs() >= hb) { continue; }
            let hit = OccurrenceHit { source, occurrence, depth: t };
            if let Some(vol) = volume {
                origins.push(q.map(|v| (((v / hb + 1.0) * 0.5 * vol.res as f64 - 0.5).round()).clamp(0.0, vol.res as f64 - 1.0) as u32));
                hits.push((hit, size));
            } else {
                consider((dot(q, q).sqrt() - gf_core::FIT) * size, hit);
            }
        }
        if let Some(vol) = volume.filter(|_| !origins.is_empty()) {
            pending.push((e.gpu.read_regions_later(&vol.tex, &origins, 1, 1, 2), hits));
        }
    }
    if !pending.is_empty() || !field_probes.is_empty() { e.gpu.wait(); }
    for (read,source,scale) in field_probes {
        if let Some(pixels)=read.take(&e.gpu) {
            let pixel=&pixels[0];
            let channel=|c:usize| half::f16::from_le_bytes([pixel[c*2],pixel[c*2+1]]).to_f64();
            let occurrence=channel(1) as usize+1024*channel(2) as usize;
            consider(channel(0)*scale,OccurrenceHit {source,occurrence,depth:t});
        }
    }
    for (read, hits) in pending {
        if let Some(texels) = read.take(&e.gpu) {
            for (value, (hit, size)) in texels.into_iter().zip(hits) {
                consider(half::f16::from_le_bytes([value[0], value[1]]).to_f64() * size, hit);
            }
        }
    }
    best.map(|(_, hit)| hit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gf_core::{groups::Item, layers::Placement, world_transform::Reference};

    #[test]
    fn snapped_local_arrow_follows_the_same_line_under_the_pointer() {
        let mut st = State::new();
        let moving = Item::Object(0);
        let peer = Item::Group(st.new_world_group(None).unwrap());
        st.set_world_placement(moving, Placement { rot: [0., 0., 45.], ..Default::default() });
        st.select_world_item(moving, false);
        let mut handles = Handles::starting();
        handles.set_tool(Tool::Move);
        handles.pick(None);
        handles.snap = true;
        handles.local = true;
        handles.snap_reference = Some(Reference::Origin);
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(1200., 800.));
        let view = View::of(&st, rect, Target::World(moving), true).unwrap();
        let on_arrow = |distance| view.screen(add(view.c, scale(view.axes[0], distance))).unwrap();
        let drag = view.len * 0.4;
        let guide_x = view.dirs[0][0] * drag + view.len * 0.02;
        st.set_world_origin(peer, [guide_x, 3., 3.]);
        handles.begin(&st, rect, on_arrow(view.len * 0.8), Part::Axis(0));
        handles.drag(&mut st, rect, on_arrow(view.len * 0.8 + drag), false, false);
        handles.end(&mut st);
        let [x, y, z] = st.world_placement(moving).unwrap().pos;
        assert!((x-guide_x).abs() < 1e-8 && (y-x).abs() < 1e-8 && z.abs() < 1e-8, "{x}, {y}, {z}");
    }

    #[test]
    fn an_arrow_is_taken_within_its_drawn_width_and_a_little_slack_only() {
        let mut st = State::new();
        st.select_world_item(Item::Object(0), false);
        let mut handles = Handles::starting();
        handles.set_tool(Tool::Move);
        handles.pick(None);
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(3840., 2160.));
        let view = View::of(&st, rect, Target::World(Item::Object(0)), false).unwrap();
        // halfway along the x arrow, and across it on the screen
        let a = view.screen(add(view.c, scale(view.axes[0], view.len * 0.5))).unwrap();
        let b = view.screen(add(view.c, scale(view.axes[0], view.len * 0.6))).unwrap();
        let across = Vec2::new(-(b - a).y, (b - a).x).normalized();
        assert_eq!(handles.hit(&st, rect, a + across * (REACH_PT - 0.5)), Some(Part::Axis(0)));
        // beside it, where the old reach (7 points) still took it, is not the arrow
        assert_ne!(handles.hit(&st, rect, a + across * 6.0), Some(Part::Axis(0)));
    }

    #[test]
    fn scale_snap_never_collapses_csg_or_construction_size_to_zero() {
        let mut st = State::new();
        let studio = st.families.iter().position(|f| f.id == "studio").unwrap();
        st.set_scene(studio);
        st.shape_mut().trace = gf_core::sdf::Trace::Csg;
        st.csg.parts = gf_core::csg::parse_arg("studio/sphere@0,0,0@0,0,0@1").unwrap();
        st.enter_world_item(Item::Object(0));
        st.csg.sel = Some(0);
        st.csg.whole = false;
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(1200., 800.));
        for snap in [false, true] {
            st.csg.parts[0].place.size = 1.0;
            let mut handles = Handles::starting();
            handles.set_tool(Tool::Size);
            handles.pick(None);
            let view = View::of(&st, rect, Target::Part(st.csg.part_id(0).unwrap()), false).unwrap();
            let centre = view.screen(view.c).unwrap();
            let rim = Pos2::new(centre.x + 80.0, centre.y);
            handles.begin(&st, rect, rim, Part::Scale);
            handles.drag(&mut st, rect, centre, snap, false);
            handles.end(&mut st);
            assert!(st.csg.parts[0].place.size >= 0.01, "snap={snap}: part size collapsed to {}", st.csg.parts[0].place.size);
        }
    }

    /// A csg object of three spheres along x, inside it, the last part chosen, the move tool on.
    fn three_parts() -> (State, Handles, Rect) {
        let mut st = State::new();
        let studio = st.families.iter().position(|f| f.id == "studio").unwrap();
        st.set_scene(studio);
        st.shape_mut().trace = gf_core::sdf::Trace::Csg;
        st.csg.parts = gf_core::csg::parse_arg("studio/sphere@-0.5,0,0@0,0,0@0.3;studio/sphere@0,0,0@0,0,0@0.3;studio/sphere@0.5,0,0@0,0,0@0.3").unwrap();
        st.csg.edit_construction();
        st.enter_world_item(Item::Object(0));
        st.csg.sel = Some(2);
        st.csg.whole = false;
        let mut handles = Handles::starting();
        handles.set_tool(Tool::Move);
        handles.pick(None);
        (st, handles, Rect::from_min_size(Pos2::ZERO, Vec2::new(1200., 800.)))
    }

    #[test]
    fn a_part_drag_holds_its_part_by_identity_when_the_list_is_reordered() {
        let (mut st, mut handles, rect) = three_parts();
        let id = st.csg.part_id(2).unwrap();
        let t = handles.target(&st).unwrap();
        assert_eq!(t, Target::Part(id));
        let c = View::of(&st, rect, t, false).unwrap();
        let at = c.screen(c.c).unwrap();
        handles.begin(&st, rect, at, Part::Centre);
        // the held part moves to the front of the list under the drag
        let held = st.csg.parts.remove(2);
        st.csg.parts.insert(0, held);
        let others: Vec<_> = st.csg.parts[1..].iter().map(|p| p.place).collect();
        handles.drag(&mut st, rect, at + Vec2::new(60.0, 0.0), false, false);
        handles.end(&mut st);
        let i = st.csg.construction_part(id).unwrap();
        assert_eq!(i, 0);
        assert!(st.csg.parts[0].place.pos[0] != 0.5, "the held part did not move");
        assert_eq!(st.csg.parts[1..].iter().map(|p| p.place).collect::<Vec<_>>(), others, "another part moved");
    }

    #[test]
    fn a_drag_lets_go_when_a_list_it_points_into_changes() {
        let (mut st, mut handles, rect) = three_parts();
        let t = handles.target(&st).unwrap();
        let c = View::of(&st, rect, t, false).unwrap();
        let at = c.screen(c.c).unwrap();
        handles.begin(&st, rect, at, Part::Centre);
        handles.drag(&mut st, rect, at + Vec2::new(30.0, 0.0), false, false);
        // a part taken away under the drag: it lets go, and Esc writes nothing anywhere
        st.csg.parts.remove(0);
        let now: Vec<_> = st.csg.parts.iter().map(|p| p.place).collect();
        handles.drag(&mut st, rect, at + Vec2::new(90.0, 0.0), false, false);
        assert!(handles.grab.is_none());
        assert!(!handles.cancel(&mut st));
        assert_eq!(st.csg.parts.iter().map(|p| p.place).collect::<Vec<_>>(), now);
    }
}
