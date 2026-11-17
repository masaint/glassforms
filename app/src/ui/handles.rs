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
    Layer(usize),
    Lamp(usize),
    /// a part of the tree of formulas (csg.rs), selected in its list: moved within the object
    Part(usize),
    /// the tree's object, the whole shape: moved in the scene, its parts with it
    Whole,
}

/// A drag in progress.
#[derive(Clone, Debug)]
pub struct Grab {
    pub part: Part,
    target: Target,
    /// the numbers when the drag began: x, y, z, three turns, size (and a lamp's second size)
    start: [f64; 8],
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
    pub grab: Option<Grab>,
    pub hover: Option<Part>,
    /// they act on this lamp (chosen in the light room, or picked on the picture), otherwise on the
    /// selected layer
    pub lamp: Option<usize>,
    /// where each part of them was last painted (for tests that take hold of them)
    pub spots: Vec<(&'static str, Pos2)>,
}

const AXIS_INK: [Color32; 3] = [Color32::from_rgb(235, 90, 85), Color32::from_rgb(110, 205, 95), Color32::from_rgb(95, 145, 240)];
const HOT: Color32 = Color32::from_rgb(255, 214, 90);
/// how long the arrows look, in points, and how near the pointer must come to a handle to take it
const ARROW_PT: f64 = 90.0;
const REACH_PT: f32 = 7.0;
/// an arrow's whole shaft is a handle: this near to it takes it
const SHAFT_PT: f32 = 8.0;
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
            Target::Layer(i) => { let p = st.stack.layers.get(i)?.place; (st.view_r(), p.pos, p.rot) }
            Target::Lamp(i) => { let l = st.lamps.get(i)?; (st.light_m(), l.pos, l.rot) }
            // (a part is moved in a frame turned with the object: see csg.rs `part_handle`)
            Target::Part(i) => { let (ro, pos, rot, _) = st.csg.part_handle(i)?; (gf_core::math::mul(&st.view_r(), &ro), pos, rot) }
            Target::Whole => { let p = st.csg.place; (st.view_r(), p.pos, p.rot) }
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

fn values(st: &State, t: Target) -> [f64; 8] {
    match t {
        Target::Layer(i) => {
            let p = st.stack.layers[i].place;
            [p.pos[0], p.pos[1], p.pos[2], p.rot[0], p.rot[1], p.rot[2], p.size, 0.0]
        }
        Target::Lamp(i) => {
            let l = st.lamps[i];
            [l.pos[0], l.pos[1], l.pos[2], l.rot[0], l.rot[1], l.rot[2], l.size[0], l.size[1]]
        }
        Target::Part(i) => {
            let (_, pos, rot, size) = st.csg.part_handle(i).unwrap_or_default();
            [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], size, 0.0]
        }
        Target::Whole => {
            let p = st.csg.place;
            [p.pos[0], p.pos[1], p.pos[2], p.rot[0], p.rot[1], p.rot[2], p.size, 0.0]
        }
    }
}

fn exists(st: &State, t: Target) -> bool {
    match t {
        Target::Layer(i) => i < st.object_count(),
        Target::Lamp(i) => i < st.lamps.len(),
        Target::Part(i) => i < st.csg.parts.len(),
        Target::Whole => !st.csg.parts.is_empty(),
    }
}

/// Which of the numbers a size square along the thing's own axis k changes: a tube's length along its
/// x and its radius across; a panel's half width and half height. None: a layer (it has one size), a ball.
fn stretch_slot(st: &State, t: Target, k: usize) -> Option<usize> {
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
        if let Some(i) = self.lamp.filter(|i| *i < st.lamps.len()) { return Some(Target::Lamp(i)); }
        // (a part selected in the tree's list: it is what the handles move)
        if let Some(i) = st.csg.sel.filter(|i| *i < st.csg.parts.len() && st.plain()) { return Some(Target::Part(i)); }
        if st.csg.whole && !st.csg.parts.is_empty() && st.plain() { return Some(Target::Whole); }
        // (with no object there is nothing to move)
        (st.stack.sel < st.object_count()).then_some(Target::Layer(st.stack.sel))
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
        self.lamp = lamp;
        self.picked = true;
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
                take((at - c).length() - 3.0, Part::Centre);
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
                take(r - 4.0, Part::Centre);
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
        let mut g = Grab { part, target, start: values(st, target), at0: at, hit0: v.c, s0: 0.0, tan: Vec2::X, rad: v.ring_radius().max(20.0) };
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
        if !exists(st, g.target) { self.grab = None; return; }
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
            }
        }
        if snap && matches!(g.part, Part::Axis(_) | Part::Plane(_) | Part::Centre) {
            for x in out.iter_mut().take(3) { *x = (*x / 0.05).round() * 0.05; }
        }
        self.write(st, g.target, &out, true);
    }

    /// Let go: the placement is built at full detail.
    pub fn end(&mut self, st: &mut State) {
        if let Some(g) = self.grab.take() {
            if exists(st, g.target) {
                let now = values(st, g.target);
                self.write(st, g.target, &now, false);
            }
        }
    }

    /// Esc during a drag: the layer goes back to where the drag began. True if there was a drag.
    pub fn cancel(&mut self, st: &mut State) -> bool {
        let Some(g) = self.grab.take() else { return false };
        if exists(st, g.target) { self.write(st, g.target, &g.start, false); }
        true
    }

    /// Write the numbers through their sliders (so that they are what the sliders say, within their range).
    fn write(&self, st: &mut State, t: Target, v: &[f64; 8], quick: bool) {
        // (a layer's sliders: x, y, z, three turns, size; a lamp's begin the same, then its second size)
        // (the tree's placements have no knobs: they are written as they are, within their sliders' ranges)
        let turn = |c: usize| ((v[3 + c] + 540.0) % 360.0) - 180.0;
        if let Target::Part(i) = t {
            st.csg.set_part_handle(i, [v[0], v[1], v[2]], [turn(0), turn(1), turn(2)], v[6]);
            st.want = Some(if quick { gf_core::state::Detail::Anim } else { gf_core::state::Detail::Full });
            return;
        }
        if let Target::Whole = t {
            let pl = &mut st.csg.place;
            for c in 0..3 { pl.pos[c] = v[c].clamp(-1.4, 1.4); }
            pl.rot = [turn(0), turn(1), turn(2)];
            pl.size = v[6].clamp(0.05, 2.0);
            st.want = Some(if quick { gf_core::state::Detail::Anim } else { gf_core::state::Detail::Full });
            return;
        }
        let knobs = match t {
            Target::Part(_) | Target::Whole => unreachable!("written above"),
            Target::Layer(i) => st.place_knobs(i),
            Target::Lamp(i) => st.lamp_knobs_of(i).into_iter().take(8).collect(),
        };
        for (c, k) in knobs.into_iter().enumerate() {
            // a number that is being animated is the animation's: the handle leaves it alone
            if st.tracks.contains_key(&k.key) { continue; }
            let x = if c >= 3 && c <= 5 { ((v[c] + 540.0) % 360.0) - 180.0 } else { v[c] };
            let x = x.clamp(k.lo, k.hi);
            if (st.get(&k.r) - x).abs() > 1e-12 || !quick { st.user_set(&k, x, quick); }
        }
    }

    /// Corners round what is chosen, where the ball it fits in shows on the screen, and its name over
    /// them: paint only, quiet enough not to be mistaken for part of the picture.
    fn mark(&self, p: &Painter, st: &State, rect: Rect) {
        let Some(t) = self.chosen(st) else { return };
        let Some((c, r, name)) = (match t {
            Target::Layer(i) => {
                let pl = st.stack.layers[i].place;
                Some((apply(&st.view_r(), pl.pos), gf_core::FIT * pl.size.abs().max(1e-3), if st.layer_count() > 1 { format!("{} · {}", i + 1, st.layer_title(i)) } else { st.layer_title(i) }))
            }
            Target::Part(i) => st.csg.part_in_scene(i).map(|(c, size)| (apply(&st.view_r(), c), gf_core::FIT * size.max(1e-3), format!("part {}", i + 1))),
            Target::Whole => {
                let pl = st.csg.place;
                Some((apply(&st.view_r(), pl.pos), gf_core::FIT * pl.size.abs().max(1e-3), "csg shape".to_string()))
            }
            Target::Lamp(i) => st.lamp_in_view(i).map(|(c, _)| {
                let l = &st.lamps[i];
                (c, l.size[0].max(l.size[1]).max(0.05) * 1.15, format!("lamp {}", i + 1))
            }),
        }) else { return };
        let v = View { cam: st.camera(), m: st.view_r(), c, axes: [[0.0; 3]; 3], dirs: [[0.0; 3]; 3], own: [[0.0; 3]; 3], len: 0.4, rect };
        // the ball's outline, near enough: three great circles round it
        let (mut lo, mut hi, mut seen, mut all) = (Pos2::new(f32::MAX, f32::MAX), Pos2::new(f32::MIN, f32::MIN), 0, 0);
        for k in 0..3 {
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
        let arm = (b.width().min(b.height()) * 0.18).clamp(5.0, 16.0);
        let ink = Color32::from_white_alpha(150);
        let shade = Stroke::new(3.0, Color32::from_black_alpha(70));
        for (corner, dx, dy) in [(b.left_top(), 1.0, 1.0), (b.right_top(), -1.0, 1.0), (b.left_bottom(), 1.0, -1.0), (b.right_bottom(), -1.0, -1.0)] {
            let pts = vec![corner + eframe::egui::vec2(dx * arm, 0.0), corner, corner + eframe::egui::vec2(0.0, dy * arm)];
            p.add(Shape::line(pts.clone(), shade));
            p.add(Shape::line(pts, Stroke::new(1.2, ink)));
        }
        let font = eframe::egui::FontId::monospace(10.0);
        let at = Pos2::new(b.left().max(rect.left() + 4.0), (b.top() - 3.0).max(rect.top() + 12.0));
        p.text(at + eframe::egui::vec2(1.0, 1.0), eframe::egui::Align2::LEFT_BOTTOM, &name, font.clone(), Color32::from_black_alpha(120));
        p.text(at, eframe::egui::Align2::LEFT_BOTTOM, &name, font, Color32::from_white_alpha(150));
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
        let ink = |part: Part, base: Color32| if lit == Some(part) { HOT } else { base };
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
        if !word.is_empty() { p.text(c + eframe::egui::vec2(0.0, 14.0), eframe::egui::Align2::CENTER_TOP, word, font, Color32::from_white_alpha(170)); }
    }
}

/// The lamp under a point of the screen, if any (the nearest of those whose outline the pointer's ray
/// passes within): lamps are known exactly, so this is worked out here, not read from the picture.
pub fn pick_lamp(st: &State, rect: Rect, at: Pos2) -> Option<usize> {
    let cam = st.camera();
    let h = rect.height() as f64 / 2.0;
    let (o, d) = cam.ray([(at.x - rect.center().x) as f64 / h, -(at.y - rect.center().y) as f64 / h])?;
    let mut best: Option<(f64, usize)> = None;
    for i in 0..st.lamps.len() {
        let (c, _) = st.lamp_in_view(i)?;
        let l = &st.lamps[i];
        let r = match l.kind { gf_core::lamps::LampKind::Ball => l.size[0], _ => l.size[0].max(l.size[1]) } + 0.03;
        let oc = sub(c, o);
        let t = dot(oc, d);
        if t <= 0.0 { continue; }
        if dot(oc, oc) - t * t < r * r && best.is_none_or(|(b, _)| t < b) { best = Some((t, i)); }
    }
    best.map(|(_, i)| i)
}

/// The layer whose shape is under a point of the picture (fx, fy: fractions of its width and height
/// from the top left), from the distance stored with the picture's pixels and each layer's own volume.
/// None if the point shows no shape.
pub fn pick_layer(e: &crate::engine::Engine, fx: f64, fy: f64) -> Option<usize> {
    let (_, (tw, th)) = e.renderer.size();
    if tw < 2 || th < 2 { return None; }
    let (x, y) = ((fx * tw as f64).clamp(0.0, tw as f64 - 1.0) as u32, (fy * th as f64).clamp(0.0, th as f64 - 1.0) as u32);
    let px = e.renderer.read_hdr(x, y, 1, 1).ok()?;
    let t = px[3] as f64;
    if t <= 0.0 { return None; }
    let st = &e.state;
    let n = st.layer_count();
    if n < 2 { return Some(0); }
    // the point in the camera's frame, then in the object's
    let uv = [((x as f64 + 0.5) * 2.0 - tw as f64) / th as f64, (th as f64 - (y as f64 + 0.5) * 2.0) / th as f64];
    let (o, d) = st.camera().ray(uv)?;
    let p = apply(&transpose(&st.view_r()), add(o, scale(d, t)));
    let mut best: Option<(f64, usize)> = None;
    for i in 0..n {
        if !st.stack.layers[i].show { continue; }
        let pl = st.stack.layers[i].place;
        let s = if pl.size != 0.0 { pl.size } else { 1.0 };
        let q = scale(apply(&transpose(&euler(pl.rot[0], pl.rot[1], pl.rot[2])), sub(p, pl.pos)), 1.0 / s);
        let hb = gf_core::HB;
        // read the layer's own volume at the nearest voxel; outside its box, how far outside
        let dist = match e.baker.layer_volume(i) {
            Some(vol) if q.iter().all(|c| c.abs() < hb) => {
                let ix = q.map(|c| (((c / hb + 1.0) * 0.5 * vol.res as f64 - 0.5).round()).clamp(0.0, vol.res as f64 - 1.0) as u32);
                match e.gpu.read(&vol.tex, ix[0], ix[1], ix[2], 1, 1, 2) {
                    Ok(b) if b.len() >= 2 => half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() as f64 * s,
                    _ => continue,
                }
            }
            _ => (dot(q, q).sqrt() - gf_core::FIT) * s,
        };
        if best.is_none_or(|(b, _)| dist < b) { best = Some((dist, i)); }
    }
    best.map(|(_, i)| i)
}
