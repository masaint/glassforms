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
//!   a click       while the handles show, a click on a shape selects its layer (Alt+click focuses)

use eframe::egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, Vec2};
use gf_core::camera::Camera;
use gf_core::math::{apply, axis_angle, euler, euler_of, mul, transpose, Mat3, DEG};
use gf_core::state::State;

/// What the handles do.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    #[default]
    Move,
    Turn,
    Size,
}
impl Mode {
    pub fn next(self) -> Mode {
        match self {
            Mode::Move => Mode::Turn,
            Mode::Turn => Mode::Size,
            Mode::Size => Mode::Move,
        }
    }
    pub fn word(self) -> &'static str {
        match self {
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
}

/// What the handles act on: a layer (the selected one) or a lamp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Layer(usize),
    Lamp(usize),
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

/// The handles' state, kept by the window.
#[derive(Default)]
pub struct Handles {
    pub on: bool,
    pub mode: Mode,
    pub grab: Option<Grab>,
    pub hover: Option<Part>,
    /// they act on this lamp (chosen in the light room, or picked on the picture), otherwise on the
    /// selected layer
    pub lamp: Option<usize>,
}

const AXIS_INK: [Color32; 3] = [Color32::from_rgb(235, 90, 85), Color32::from_rgb(110, 205, 95), Color32::from_rgb(95, 145, 240)];
const HOT: Color32 = Color32::from_rgb(255, 214, 90);
/// how long the arrows look, in points, and how near the pointer must come to a handle to take it
const ARROW_PT: f64 = 90.0;
const REACH_PT: f32 = 7.0;

/// The camera and where the selected layer stands, as the picture shows them.
struct View {
    cam: Camera,
    /// the object's frame -> the camera's
    m: Mat3,
    /// the layer's middle, in the camera's frame
    c: [f64; 3],
    /// the scene's three axes, in the camera's frame
    axes: [[f64; 3]; 3],
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
    fn of(st: &State, rect: Rect, target: Target) -> Option<View> {
        let cam = st.camera();
        // a layer stands in the object's frame, a lamp in the lights' frame
        let (m, pos) = match target {
            Target::Layer(i) => (st.view_r(), st.stack.layers.get(i)?.place.pos),
            Target::Lamp(i) => (st.light_m(), st.lamps.get(i)?.pos),
        };
        let c = apply(&m, pos);
        let axes = [apply(&m, [1.0, 0.0, 0.0]), apply(&m, [0.0, 1.0, 0.0]), apply(&m, [0.0, 0.0, 1.0])];
        let mut v = View { cam, m, c, axes, len: 0.4, rect };
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
        self.polyline((0..=12).map(|i| add(self.c, scale(self.axes[k], self.len * (0.18 + 0.82 * i as f64 / 12.0)))))
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
    }
}

fn exists(st: &State, t: Target) -> bool {
    match t {
        Target::Layer(i) => i < st.layer_count(),
        Target::Lamp(i) => i < st.lamps.len(),
    }
}

impl Handles {
    /// What the handles act on, if they show.
    pub fn target(&self, st: &State) -> Option<Target> {
        if !self.on { return None; }
        if let Some(i) = self.lamp.filter(|i| *i < st.lamps.len()) { return Some(Target::Lamp(i)); }
        (st.stack.sel < st.layer_count()).then_some(Target::Layer(st.stack.sel))
    }

    /// Show or hide them (G).
    pub fn toggle(&mut self) {
        self.on = !self.on;
        self.grab = None;
    }

    /// The part of the handles under the pointer, if any.
    pub fn hit(&self, st: &State, rect: Rect, at: Pos2) -> Option<Part> {
        let target = self.target(st)?;
        let v = View::of(st, rect, target)?;
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
        }
        best.map(|(_, p)| p)
    }

    /// Take hold of a part at the pointer.
    pub fn begin(&mut self, st: &State, rect: Rect, at: Pos2, part: Part) {
        let Some(target) = self.target(st) else { return };
        let Some(v) = View::of(st, rect, target) else { return };
        let mut g = Grab { part, target, start: values(st, target), at0: at, hit0: v.c, s0: 0.0, tan: Vec2::X, rad: 40.0 };
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
                Part::Scale => {}
            }
        }
        self.grab = Some(g);
    }

    /// The pointer has moved to `at` during a drag: change the placement. `snap`: in steps; `fine`: a
    /// tenth as far.
    pub fn drag(&mut self, st: &mut State, rect: Rect, at: Pos2, snap: bool, fine: bool) {
        let Some(g) = self.grab.clone() else { return };
        if !exists(st, g.target) { self.grab = None; return; }
        // the frame as it was at the press: the middle moves under the drag, the constraint does not
        let Some(mut v) = View::of(st, rect, g.target) else { return };
        v.c = apply(&v.m, [g.start[0], g.start[1], g.start[2]]);
        let Some((o, d)) = v.ray(at) else { return };
        let k_fine = if fine { 0.1 } else { 1.0 };
        let mut out = g.start;
        match g.part {
            Part::Axis(k) => {
                let Some(s) = along_line(o, d, v.c, v.axes[k]) else { return };
                out[k] = g.start[k] + (s - g.s0) * k_fine;
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
                // a turn about the scene's axis, laid onto the turn the layer had
                let mut ax = [0.0; 3];
                ax[k] = 1.0;
                let r = mul(&axis_angle(ax[0], ax[1], ax[2], deg * DEG), &euler(g.start[3], g.start[4], g.start[5]));
                let t = euler_of(&r);
                out[3] = t[0];
                out[4] = t[1];
                out[5] = t[2];
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
        let knobs = match t {
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

    /// Paint the handles over the picture.
    pub fn paint(&mut self, p: &Painter, st: &State, rect: Rect, pointer: Option<Pos2>) {
        let Some(target) = self.target(st) else { return };
        let Some(v) = View::of(st, rect, target) else { return };
        let Some(c) = v.screen(v.c) else { return };
        let lit = self.grab.as_ref().map(|g| g.part).or_else(|| pointer.and_then(|at| self.hit(st, rect, at)));
        self.hover = lit;
        let ink = |part: Part, base: Color32| if lit == Some(part) { HOT } else { base };
        let shadow = Stroke::new(4.0, Color32::from_black_alpha(90));
        match self.mode {
            Mode::Move => {
                for k in 0..3 {
                    if let Some(sq) = v.square(k) {
                        let col = ink(Part::Plane(k), AXIS_INK[k]);
                        p.add(Shape::convex_polygon(sq, col.gamma_multiply(0.35), Stroke::new(1.0, col)));
                    }
                }
                for k in 0..3 {
                    let col = ink(Part::Axis(k), AXIS_INK[k]);
                    for run in v.arrow(k) {
                        p.add(Shape::line(run.clone(), shadow));
                        p.add(Shape::line(run.clone(), Stroke::new(2.0, col)));
                        // the head, pointing the way the end of the line runs
                        let (a, b) = (run[run.len() - 2], run[run.len() - 1]);
                        let dir = (b - a).normalized();
                        let side = eframe::egui::vec2(-dir.y, dir.x);
                        p.add(Shape::convex_polygon(vec![b + dir * 9.0, b + side * 4.5, b - side * 4.5], col, Stroke::NONE));
                    }
                }
                p.circle_filled(c, 5.5, Color32::from_black_alpha(110));
                p.circle_filled(c, 4.0, ink(Part::Centre, Color32::from_gray(235)));
            }
            Mode::Turn => {
                for k in 0..3 {
                    let col = ink(Part::Ring(k), AXIS_INK[k]);
                    let pts = v.ring(k);
                    // the half of the ring on the far side of the layer faintly, the near half strongly
                    for front in [false, true] {
                        let mut run: Vec<Pos2> = Vec::new();
                        let flush = |run: &mut Vec<Pos2>| {
                            if run.len() > 1 {
                                let s = if front { Stroke::new(2.0, col) } else { Stroke::new(1.0, col.gamma_multiply(0.35)) };
                                if front { p.add(Shape::line(run.clone(), shadow)); }
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
                }
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
        p.text(c + eframe::egui::vec2(0.0, 14.0), eframe::egui::Align2::CENTER_TOP, format!("{} · T", self.mode.word()), font, Color32::from_white_alpha(170));
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
