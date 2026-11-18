//! The path builder: laying a path down piece by piece (`gf_core::path`), and editing it on the picture.
//!
//! In the right panel, under the shape's sliders (the profile), while a shape of the paths family is the
//! selected layer:
//!   the pieces    a chip for every piece of the path, in order; a click selects one (another click, or a
//!                 click on the end chip, lets go). New pieces go at the end, or after the selected piece.
//!   add           a button for every kind of piece, each with a small drawing of it. Hovering one shows the
//!                 piece it would add on the picture, as a ghost at the end of the path.
//!   sizes         the angles a turn, a slope and a bank are made with (the usual fixed steps), and the radius
//!                 and length; while a piece is selected they are that piece's own, and change it.
//!   finishing     undo the last piece, close the loop (a curve back to the start), centre, start again.
//!
//! On the picture the path's centre line is drawn over the glass, with a dot at every joint. Dragged:
//!   the start ring        moves the whole path across the floor (with shift: up and down)
//!   a joint               a straight's or a bank's length, a turn's or a slope's radius, a curve's end
//!   a curve's handles     how it leaves (along the heading) and how it arrives
//! In steps (0.025 of a unit; a curve's arrival in steps of 15 degrees) while snapping is on; Ctrl drags
//! freely. A click on the line selects its piece.

use eframe::egui::{self, Align2, Color32, Id, Painter, Pos2, Rect, Sense, Shape, Stroke, Ui};
use gf_core::camera::Camera;
use gf_core::math::{apply, euler, transpose, Mat3};
use gf_core::path::{self, add, dot, len3, norm, scale, sub, Frame, Path, Piece};
use gf_core::presets::paths as pp;
use gf_core::state::{Detail, State};

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};

/// The grid lengths snap to, and the step a curve's arrival snaps to (degrees).
pub const GRID: f64 = 0.025;
pub const ANGLE_STEP: f64 = 15.0;
/// The angles offered for a turn, a slope and a bank.
pub const TURNS: [f64; 4] = [22.5, 45.0, 90.0, 180.0];
pub const SLOPES: [f64; 4] = [15.0, 30.0, 45.0, 90.0];
pub const BANKS: [f64; 4] = [15.0, 30.0, 45.0, 90.0];
/// How near the pointer must come to take hold of a joint or a handle, and to pick a piece by its line.
const REACH_PT: f32 = 8.0;
const LINE_PT: f32 = 6.0;

/// The kinds of piece the builder offers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Straight,
    Left,
    Right,
    Up,
    Down,
    BankLeft,
    BankRight,
    Loop,
    Curve,
}
pub const KINDS: [Kind; 9] = [Kind::Straight, Kind::Left, Kind::Right, Kind::Up, Kind::Down, Kind::BankLeft, Kind::BankRight, Kind::Loop, Kind::Curve];

impl Kind {
    pub fn of(p: &Piece) -> Kind {
        match *p {
            Piece::Straight { .. } => Kind::Straight,
            Piece::Turn { deg, .. } => if deg >= 0.0 { Kind::Left } else { Kind::Right },
            Piece::Slope { deg, .. } => if deg >= 0.0 { Kind::Up } else { Kind::Down },
            Piece::Bank { deg, .. } => if deg >= 0.0 { Kind::BankLeft } else { Kind::BankRight },
            Piece::Loop { .. } => Kind::Loop,
            Piece::Curve { .. } => Kind::Curve,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Straight => "straight",
            Kind::Left => "turn left",
            Kind::Right => "turn right",
            Kind::Up => "slope up",
            Kind::Down => "slope down",
            Kind::BankLeft => "bank left",
            Kind::BankRight => "bank right",
            Kind::Loop => "loop",
            Kind::Curve => "free curve",
        }
    }
    fn key(self) -> &'static str {
        match self {
            Kind::Straight => "straight",
            Kind::Left => "left",
            Kind::Right => "right",
            Kind::Up => "up",
            Kind::Down => "down",
            Kind::BankLeft => "bankleft",
            Kind::BankRight => "bankright",
            Kind::Loop => "loop",
            Kind::Curve => "curve",
        }
    }
}

/// Something of the path on the picture that can be taken hold of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hit {
    /// the ring at the start
    Start,
    /// the joint at the end of this piece
    Joint(usize),
    /// a curve's first handle (how it leaves), and its second (how it arrives)
    Out(usize),
    Arrive(usize),
    /// the line of this piece
    Line(usize),
}

/// A drag in progress.
#[derive(Clone, Debug)]
pub struct Grab {
    hit: Hit,
    /// the path's text when the drag began (Esc puts it back)
    text0: String,
    path0: Path,
    at0: Pos2,
    moved: f32,
}

/// The builder's state, kept by the window.
pub struct PathEdit {
    /// the selected piece (its values are what the sizes show and change)
    pub sel: Option<usize>,
    /// the angles new turns, slopes and banks are made with; the radius and length of new pieces
    pub turn: f64,
    pub slope: f64,
    pub bank: f64,
    pub radius: f64,
    pub length: f64,
    /// joints and handles move in steps
    pub snap: bool,
    /// the path is drawn over the picture
    pub show: bool,
    /// the piece a button under the pointer would add (shown as a ghost on the picture), as of this frame
    pub ghost: Option<Piece>,
    pub grab: Option<Grab>,
    pub hover: Option<Hit>,
    /// where each part was last painted (for tests that take hold of them)
    pub spots: Vec<(String, Pos2)>,
}

impl Default for PathEdit {
    fn default() -> Self {
        Self { sel: None, turn: 90.0, slope: 30.0, bank: 45.0, radius: 0.15, length: 0.3, snap: true, show: true, ghost: None, grab: None, hover: None, spots: Vec::new() }
    }
}

/// The selected layer's shape is a path (a volume object of the paths family).
pub fn editing(st: &State) -> bool {
    st.family().id == pp::FAMILY && !st.is_csg_object(st.stack.sel)
}
/// The selected layer's path as its text stands.
pub fn path_now(st: &State) -> Path {
    pp::path_of(st.preset().id, &st.shape().model)
}
/// Put a path in as the selected layer's (built roughly while `quick`, otherwise finely).
fn set_path(st: &mut State, p: &Path, quick: bool) {
    let text = p.text();
    if st.shape().model == text { return; }
    st.shape_mut().model = text;
    st.want = Some(if quick { Detail::Quick } else { Detail::Full });
}
/// How wide the profile is (a loop steps aside by a little more than this, so that it clears itself).
fn profile_width(st: &State) -> f64 {
    let pr = pp::profile(st.preset().id, &st.shape().vals);
    pr.w.max(pr.h) * pr.start.max(pr.end)
}

fn snap_to(v: f64, step: f64, on: bool) -> f64 {
    if on { (v / step).round() * step } else { v }
}

impl PathEdit {
    /// A new piece of this kind, with the builder's sizes.
    pub fn make(&self, k: Kind, width: f64) -> Piece {
        let (r, l) = (self.radius, self.length);
        match k {
            Kind::Straight => Piece::Straight { len: l },
            Kind::Left => Piece::Turn { deg: self.turn, radius: r },
            Kind::Right => Piece::Turn { deg: -self.turn, radius: r },
            Kind::Up => Piece::Slope { deg: self.slope, radius: r },
            Kind::Down => Piece::Slope { deg: -self.slope, radius: r },
            Kind::BankLeft => Piece::Bank { deg: self.bank, len: l },
            Kind::BankRight => Piece::Bank { deg: -self.bank, len: l },
            Kind::Loop => Piece::Loop { radius: r.max(1.2 * width), shift: snap_to((1.4 * width).max(GRID), GRID, true) },
            // an S that steps half its length to the left and heads on as before
            Kind::Curve => Piece::Curve { out: l / 3.0, h2: [2.0 * l / 3.0, 0.0, 0.5 * l], end: [l, 0.0, 0.5 * l] },
        }
    }
    /// Where new pieces go: after the selected one, or at the end.
    fn insert_at(&self, p: &Path) -> usize {
        self.sel.map_or(p.pieces.len(), |i| (i + 1).min(p.pieces.len()))
    }
    /// Keep the selection on a piece that is there.
    fn tidy(&mut self, p: &Path) {
        if self.sel.is_some_and(|i| i >= p.pieces.len()) { self.sel = None; }
    }
}

// ---- small drawings of the pieces ---------------------------------------------------------------------

/// A drawing of a kind of piece in `r`: the track as a stroke, the way it goes on at its end.
pub fn glyph(p: &Painter, r: Rect, k: Kind, ink: Color32) {
    let c = r.center();
    let s = 0.36 * r.width().min(r.height());
    let at = |x: f32, y: f32| Pos2::new(c.x + x * s, c.y - y * s);
    let line = |pts: Vec<Pos2>| Shape::line(pts, Stroke::new(1.5, ink));
    let arc = |cx: f32, cy: f32, rad: f32, a0: f32, a1: f32| -> Vec<Pos2> {
        (0..=16).map(|i| { let a = a0 + (a1 - a0) * i as f32 / 16.0; at(cx + rad * a.cos(), cy + rad * a.sin()) }).collect()
    };
    let head = |tip: Pos2, dir: egui::Vec2| {
        let d = dir.normalized() * (0.42 * s);
        let n = egui::vec2(-d.y, d.x) * 0.75;
        Shape::convex_polygon(vec![tip + d * 0.6, tip - d * 0.6 + n, tip - d * 0.6 - n], ink, Stroke::NONE)
    };
    use std::f32::consts::{FRAC_PI_2, PI};
    match k {
        Kind::Straight => {
            p.add(line(vec![at(0.0, -1.0), at(0.0, 0.75)]));
            p.add(head(at(0.0, 0.85), egui::vec2(0.0, -1.0)));
        }
        Kind::Left | Kind::Right => {
            let m = if k == Kind::Left { -1.0 } else { 1.0 };
            let mut pts = vec![at(0.35 * m, -1.0)];
            pts.extend(arc(-0.65 * m, -0.1, 1.0, if m > 0.0 { 0.0 } else { PI }, if m > 0.0 { FRAC_PI_2 * 0.85 } else { PI - FRAC_PI_2 * 0.85 }));
            let end = *pts.last().expect("points");
            p.add(line(pts));
            p.add(head(end + egui::vec2(-0.25 * s * m, -0.03 * s), egui::vec2(-m, 0.12)));
        }
        Kind::Up | Kind::Down => {
            // seen from the side: level, then rising (or falling) away to the right
            let m = if k == Kind::Up { 1.0 } else { -1.0 };
            let mut pts = vec![at(-1.0, -0.55 * m), at(-0.35, -0.55 * m)];
            for i in 0..=12 {
                let a = -FRAC_PI_2 + (i as f32 / 12.0) * 1.05;
                pts.push(at(-0.35 + 1.1 * a.cos(), m * (0.55 + 1.1 * a.sin())));
            }
            let (end, prev) = (pts[pts.len() - 1], pts[pts.len() - 2]);
            p.add(line(pts));
            p.add(head(end, end - prev));
        }
        Kind::BankLeft | Kind::BankRight => {
            // the track's section, tilted, with the turn it is given
            let m = if k == Kind::BankLeft { 1.0 } else { -1.0 };
            let a = 0.45f32 * m;
            let (ca, sa) = (a.cos(), a.sin());
            let rot = |x: f32, y: f32| at(x * ca - y * sa, x * sa + y * ca - 0.15);
            p.add(Shape::convex_polygon(vec![rot(-0.85, -0.12), rot(0.85, -0.12), rot(0.85, 0.12), rot(-0.85, 0.12)], fade(ink, 0.25), Stroke::new(1.3, ink)));
            let pts = arc(0.0, -0.15, 0.95, FRAC_PI_2 - 0.2 * m, FRAC_PI_2 + 0.9 * m);
            let end = *pts.last().expect("points");
            let prev = pts[pts.len() - 2];
            p.add(line(pts));
            p.add(head(end, end - prev));
        }
        Kind::Loop => {
            let mut pts = vec![at(-1.0, -0.85), at(0.15, -0.85)];
            pts.extend(arc(0.0, -0.05, 0.8, -FRAC_PI_2 + 0.15, 1.5 * PI - 0.35));
            p.add(line(pts));
            p.add(line(vec![at(-0.2, -0.85), at(0.95, -0.85)]));
            p.add(head(at(1.0, -0.85), egui::vec2(1.0, 0.0)));
        }
        Kind::Curve => {
            let (p0, p1, p2, p3) = ([-0.9f32, -0.9f32], [0.1f32, -0.9f32], [-0.1f32, 0.9f32], [0.9f32, 0.9f32]);
            let pts: Vec<Pos2> = (0..=20).map(|i| {
                let u = i as f32 / 20.0;
                let v = 1.0 - u;
                let b = |k: usize| v * v * v * p0[k] + 3.0 * v * v * u * p1[k] + 3.0 * v * u * u * p2[k] + u * u * u * p3[k];
                at(b(0), b(1))
            }).collect();
            p.add(line(pts));
            p.add(Shape::line_segment([at(p0[0], p0[1]), at(p1[0], p1[1])], Stroke::new(1.0, fade(ink, 0.6))));
            p.add(Shape::line_segment([at(p3[0], p3[1]), at(p2[0], p2[1])], Stroke::new(1.0, fade(ink, 0.6))));
            p.circle_filled(at(p1[0], p1[1]), 1.9, ink);
            p.circle_filled(at(p2[0], p2[1]), 1.9, ink);
        }
    }
}

/// A square button with a drawing on it. Returns what the pointer did to it.
fn glyph_button(ui: &mut Ui, rect: Rect, id: Id, k: Kind, on: bool) -> egui::Response {
    let resp = ui.interact(rect, id, Sense::click());
    let live = ui.is_enabled();
    let hot = live && resp.hovered();
    if hot { ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand); }
    let down = live && resp.is_pointer_button_down_on();
    let (fill, line, ink) = if on {
        (if hot { Color32::WHITE } else { pal().bright }, pal().bright, pal().ink)
    } else if hot {
        (white(if down { 0.12 } else { 0.05 }), white(0.3), Color32::WHITE)
    } else {
        (white(0.05), white(0.1), pal().button)
    };
    ui.painter().rect(rect, BUTTON_RADIUS, fill, Stroke::new(1.0, line), egui::StrokeKind::Inside);
    glyph(ui.painter(), rect.shrink(3.0), k, ink);
    resp
}

// ---- the panel -------------------------------------------------------------------------------------

/// The builder, in the right panel, while the selected layer's shape is a path.
pub fn panel(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    px.paths.ghost = None;
    if !editing(st) { return; }
    let mut path = path_now(st);
    px.paths.tidy(&path);
    let walk = path.walk();
    gap(ui, 4.0);
    px.heading(ui, "path", "path");
    // what the path comes to
    let n = path.pieces.len();
    let state = if walk.closed {
        "a closed loop".to_owned()
    } else if n > 0 {
        let (gap_to, angle) = path.end_gap();
        format!("open · its end {} from the start, heading {:.0}° off", path::fmt(gap_to), angle)
    } else {
        "nothing laid yet".to_owned()
    };
    let words = format!("{} {} · {} long · {}", n, if n == 1 { "piece" } else { "pieces" }, path::fmt(walk.length), state);
    let r = paragraph(ui, &words, mono(SMALL), pal().dim, LINE, 0.0);
    px.note("path:summary", r);
    gap(ui, 4.0);

    chips(ui, st, px, &path);
    gap(ui, 6.0);

    // ---- adding pieces
    let width = profile_width(st);
    let at = px.paths.insert_at(&path);
    let r = row(ui, 13.0);
    let says = if px.paths.sel.is_some() { format!("add after piece {}", at) } else { "add at the end".to_owned() };
    text_in(ui, r, Align2::LEFT_CENTER, &says, mono(10.0), pal().faint);
    let size = ((ui.available_width() - 3.0 * (KINDS.len() as f32 - 1.0)) / KINDS.len() as f32).floor().min(30.0);
    let r = row(ui, size);
    let mut added = None;
    for (i, k) in KINDS.iter().enumerate() {
        let place = Rect::from_min_size(Pos2::new(r.left() + i as f32 * (size + 3.0), r.top()), egui::vec2(size, size));
        px.note(format!("path:add:{}", k.key()), place);
        let resp = glyph_button(ui, place, Id::unique(("path add", i)), *k, false);
        let tip = format!("add a {}{}", k.name(), match k {
            Kind::Left | Kind::Right => format!(" of {}° on radius {}", path::fmt(px.paths.turn), path::fmt(px.paths.radius)),
            Kind::Up | Kind::Down => format!(" of {}° on radius {}", path::fmt(px.paths.slope), path::fmt(px.paths.radius)),
            Kind::BankLeft | Kind::BankRight => format!(" of {}° over {}", path::fmt(px.paths.bank), path::fmt(px.paths.length)),
            Kind::Straight => format!(" {} long", path::fmt(px.paths.length)),
            Kind::Loop => " that steps aside to clear itself".to_owned(),
            Kind::Curve => ": drag its end and handles on the picture".to_owned(),
        });
        let resp = resp.on_hover_text(tip);
        if resp.hovered() { px.paths.ghost = Some(px.paths.make(*k, width)); }
        if resp.clicked() { added = Some(*k); }
    }
    if let Some(k) = added {
        path.pieces.insert(at, px.paths.make(k, width));
        if px.paths.sel.is_some() { px.paths.sel = Some(at); }
        set_path(st, &path, false);
    }
    gap(ui, 4.0);

    // ---- the sizes: of the selected piece, or of new pieces
    match px.paths.sel {
        Some(i) => selected_rows(ui, st, px, &mut path, i),
        None => new_rows(ui, px),
    }

    // ---- finishing
    gap(ui, 4.0);
    finishing(ui, st, px, &mut path);
    start_rows(ui, st, px, &mut path);
    gap(ui, 4.0);
}

/// The pieces of the path, a chip each, and the end. A click selects a piece, or lets go of it.
fn chips(ui: &mut Ui, st: &mut State, px: &mut Panels, path: &Path) {
    let _ = st;
    let n = path.pieces.len();
    let size = 20.0;
    let per = ((ui.available_width() + 2.0) / (size + 2.0)).floor().max(1.0) as usize;
    let lines = (n + 1).div_ceil(per).max(1);
    let r = row(ui, lines as f32 * (size + 2.0) - 2.0);
    for i in 0..=n {
        let (line, col) = (i / per, i % per);
        let place = Rect::from_min_size(Pos2::new(r.left() + col as f32 * (size + 2.0), r.top() + line as f32 * (size + 2.0)), egui::vec2(size, size));
        if i == n {
            // the end: where new pieces go while nothing is selected
            let on = px.paths.sel.is_none();
            let resp = ui.interact(place, Id::unique("path end chip"), Sense::click());
            let ink = if on { pal().bright } else if resp.hovered() { Color32::WHITE } else { pal().dim };
            ui.painter().rect(place, BUTTON_RADIUS, Color32::TRANSPARENT, Stroke::new(1.0, if on { fade(pal().bright, 0.7) } else { white(0.15) }), egui::StrokeKind::Inside);
            text_in(ui, place, Align2::CENTER_CENTER, "+", mono(13.0), ink);
            px.note("path:chip:end", place);
            if resp.on_hover_text("the end of the path: new pieces go here").clicked() { px.paths.sel = None; }
            continue;
        }
        let k = Kind::of(&path.pieces[i]);
        let on = px.paths.sel == Some(i);
        px.note(format!("path:chip:{}", i + 1), place);
        let resp = glyph_button(ui, place, Id::unique(("path chip", i)), k, on).on_hover_text(format!("piece {}: {}", i + 1, path.pieces[i].words()));
        if resp.clicked() { px.paths.sel = if on { None } else { Some(i) }; }
    }
}

/// Buttons for a few fixed values, one lit if the value is one of them: the one clicked.
fn steps_row(ui: &mut Ui, px: &mut Panels, key: &str, name: &str, steps: &[f64], now: f64) -> Option<f64> {
    let words: Vec<String> = steps.iter().map(|v| path::fmt(*v)).collect();
    let items: Vec<Item> = words.iter().zip(steps).map(|(w, v)| Item::new(w, (v - now).abs() < 1e-9)).collect();
    let run = named_run(ui, Id::unique(("path steps", key)), name, &items);
    for (i, place) in run.places.iter().enumerate() { px.note(format!("path:{key}:{i}"), *place); }
    run.clicked.map(|i| steps[i])
}

/// The sizes new pieces are made with.
fn new_rows(ui: &mut Ui, px: &mut Panels) {
    if let Some(v) = steps_row(ui, px, "turn", "turn °", &TURNS, px.paths.turn) { px.paths.turn = v; }
    if let Some(v) = steps_row(ui, px, "slope", "slope °", &SLOPES, px.paths.slope) { px.paths.slope = v; }
    if let Some(v) = steps_row(ui, px, "bank", "bank °", &BANKS, px.paths.bank) { px.paths.bank = v; }
    if let Some(v) = knob::value_row(ui, px, "path radius", "radius", px.paths.radius, 0.02, 1.0, GRID).value { px.paths.radius = v; }
    if let Some(v) = knob::value_row(ui, px, "path length", "length", px.paths.length, GRID, 2.0, GRID).value { px.paths.length = v; }
}

/// The selected piece: what it is, its kind (another kind keeps its sizes where it can), its own numbers,
/// and taking it out.
fn selected_rows(ui: &mut Ui, st: &mut State, px: &mut Panels, path: &mut Path, i: usize) {
    let Some(piece) = path.pieces.get(i).cloned() else { return };
    let (top, left) = (ui.cursor().top(), ui.cursor().left());
    let r = row(ui, 22.0);
    text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 6.0, r.top()), r.max), Align2::LEFT_CENTER, &format!("piece {} · {}", i + 1, piece.words()), mono(SIZE), pal().bright);
    let font = mono(SIZE);
    let dsize = button_size(ui, "delete", &font, egui::vec2(7.0, 2.0));
    let dplace = Rect::from_min_size(Pos2::new(r.right() - dsize.x, r.center().y - dsize.y / 2.0), dsize);
    px.note("path:delete", dplace);
    if button(ui, dplace, Id::unique("path delete"), "delete", false, font).on_hover_text("take this piece out: the pieces after it move up to close the gap").clicked() {
        path.pieces.remove(i);
        px.paths.sel = if path.pieces.is_empty() { None } else { Some(i.saturating_sub(1)).filter(|_| i > 0) };
        set_path(st, path, false);
        return;
    }
    // its kind
    let k = Kind::of(&piece);
    let size = ((ui.available_width() - 6.0 - 3.0 * (KINDS.len() as f32 - 1.0)) / KINDS.len() as f32).floor().min(26.0);
    let kr = row(ui, size);
    let mut swap = None;
    for (j, kk) in KINDS.iter().enumerate() {
        let place = Rect::from_min_size(Pos2::new(kr.left() + 6.0 + j as f32 * (size + 3.0), kr.top()), egui::vec2(size, size));
        px.note(format!("path:kind:{}", kk.key()), place);
        if glyph_button(ui, place, Id::unique(("path kind", j)), *kk, *kk == k).on_hover_text(format!("make it a {}", kk.name())).clicked() && *kk != k { swap = Some(*kk); }
    }
    if let Some(kk) = swap {
        path.pieces[i] = reshaped(&piece, kk, profile_width(st), &px.paths);
        set_path(st, path, false);
    }
    gap(ui, 3.0);
    // its numbers
    let mut p = path.pieces[i].clone();
    let before = p.clone();
    let mut quick = false;
    let mut slide = |ui: &mut Ui, px: &mut Panels, key: &str, name: &str, v: &mut f64, lo: f64, hi: f64, step: f64| {
        let m = knob::value_row(ui, px, key, name, *v, lo, hi, step);
        if let Some(x) = m.value { *v = x; quick = true; }
        if m.released { quick = false; }
    };
    match &mut p {
        Piece::Straight { len } => slide(ui, px, "piece length", "length", len, GRID, 2.0, GRID),
        Piece::Turn { deg, radius } | Piece::Slope { deg, radius } => {
            let sign = deg.signum();
            let steps: &[f64] = if matches!(before, Piece::Turn { .. }) { &TURNS } else { &SLOPES };
            if let Some(v) = steps_row(ui, px, "piece angle", "angle °", steps, deg.abs()) { *deg = sign * v; }
            let mut a = deg.abs();
            slide(ui, px, "piece degrees", "", &mut a, 0.0, 360.0, 7.5);
            *deg = sign * a;
            slide(ui, px, "piece radius", "radius", radius, 0.02, 1.0, GRID / 5.0);
        }
        Piece::Bank { deg, len } => {
            let sign = deg.signum();
            if let Some(v) = steps_row(ui, px, "piece angle", "angle °", &BANKS, deg.abs()) { *deg = sign * v; }
            let mut a = deg.abs();
            slide(ui, px, "piece degrees", "", &mut a, 0.0, 360.0, 7.5);
            *deg = sign * a;
            slide(ui, px, "piece length", "over", len, GRID, 2.0, GRID);
        }
        Piece::Loop { radius, shift } => {
            slide(ui, px, "piece radius", "radius", radius, 0.05, 1.0, GRID / 5.0);
            slide(ui, px, "piece shift", "steps aside", shift, -1.0, 1.0, GRID / 5.0);
        }
        Piece::Curve { out, .. } => {
            slide(ui, px, "piece out", "leaves over", out, 0.0, 1.5, GRID / 5.0);
            gap(ui, 2.0);
            paragraph(ui, "drag its end and its handles on the picture: the arrival snaps to 15° steps", sans(SMALL), pal().faint, 1.4, 6.0);
        }
    }
    if p != before {
        path.pieces[i] = p;
        set_path(st, path, quick);
    }
    gap(ui, 2.0);
    side_bar(ui, left + 1.0, top + 2.0, ui.cursor().top() - 2.0, 0.55);
}

/// A piece made another kind, keeping what sizes it can.
fn reshaped(p: &Piece, k: Kind, width: f64, ed: &PathEdit) -> Piece {
    let len = p.length().max(GRID);
    let (angle, radius) = match *p {
        Piece::Turn { deg, radius } | Piece::Slope { deg, radius } => (Some(deg.abs()), radius),
        Piece::Bank { deg, .. } => (Some(deg.abs()), ed.radius),
        Piece::Loop { radius, .. } => (None, radius),
        _ => (None, ed.radius),
    };
    let fresh = PathEdit { radius, length: snap_to(len, GRID, true).max(GRID), ..PathEdit { turn: ed.turn, slope: ed.slope, bank: ed.bank, ..PathEdit::default() } };
    let mut out = fresh.make(k, width);
    if let Some(a) = angle {
        match &mut out {
            Piece::Turn { deg, .. } | Piece::Slope { deg, .. } | Piece::Bank { deg, .. } => *deg = deg.signum() * a,
            _ => {}
        }
    }
    out
}

/// Undo the last piece, close the loop, centre, start again, and whether joints snap and the path shows.
fn finishing(ui: &mut Ui, st: &mut State, px: &mut Panels, path: &mut Path) {
    let font = mono(SIZE);
    let pad = egui::vec2(6.0, 2.0);
    let closed = path.walk().closed;
    let words = ["undo last", "close loop", "centre", "clear"];
    let tips = [
        "take the last piece off the end",
        "lay a curve from the end back to the start, arriving the way the path starts: the loop closes smoothly",
        "move the whole path so that it is centred in the volume (its height kept)",
        "start again with no pieces",
    ];
    let can = [!path.pieces.is_empty(), !path.pieces.is_empty() && !closed, true, !path.pieces.is_empty()];
    let items: Vec<Item> = words.iter().zip(tips).map(|(w, t)| Item::new(w, false).tip(t)).collect();
    let _ = pad;
    let mut clicked = None;
    ui.scope(|ui| {
        let run = filled_run(ui, Id::unique("path finishing"), &items, egui::vec2(5.0, 2.0), &font);
        for (w, place) in words.iter().zip(&run.places) { px.note(format!("path:{w}"), *place); }
        clicked = run.clicked.filter(|i| can[*i]);
    });
    match clicked {
        Some(0) => {
            path.pieces.pop();
            px.paths.tidy(path);
            set_path(st, path, false);
        }
        Some(1) => {
            if let Some(c) = path.closing_curve() {
                path.pieces.push(c);
                px.paths.sel = None;
                set_path(st, path, false);
            }
        }
        Some(2) => {
            path.centre(true);
            set_path(st, path, false);
        }
        Some(3) => {
            path.pieces.clear();
            px.paths.sel = None;
            set_path(st, path, false);
        }
        _ => {}
    }
    gap(ui, 3.0);
    let items = [
        Item::new("snap", px.paths.snap).tip("joints and handles move in steps of 0.025, a curve's arrival in steps of 15° (ctrl while dragging: freely)"),
        Item::new("show on the picture", px.paths.show).tip("draw the path's line, joints and handles over the picture"),
    ];
    let run = filled_run(ui, Id::unique("path toggles"), &items, egui::vec2(6.0, 2.0), &font);
    px.note("path:snap", run.places[0]);
    px.note("path:show", run.places[1]);
    match run.clicked {
        Some(0) => px.paths.snap = !px.paths.snap,
        Some(1) => px.paths.show = !px.paths.show,
        _ => {}
    }
}

/// Where the path starts, and which way it heads (folded).
fn start_rows(ui: &mut Ui, st: &mut State, px: &mut Panels, path: &mut Path) {
    let says = format!("{} {} {} · heads {}°", path::fmt(path.start.p[0]), path::fmt(path.start.p[1]), path::fmt(path.start.p[2]), path::fmt(path.start.yaw));
    let mut changed = (false, false);
    let start = &mut path.start;
    px.sub(ui, "path-start", "start", &says, |ui, px| {
        for (k, (key, name)) in [("path start x", "x"), ("path start y", "height"), ("path start z", "z")].iter().enumerate() {
            let m = knob::value_row(ui, px, key, name, start.p[k], -1.4, 1.4, GRID / 5.0);
            if let Some(v) = m.value { start.p[k] = v; changed = (true, !m.released); }
        }
        for (key, name, v, lo, hi) in [("path start yaw", "heading °", &mut start.yaw, -180.0, 180.0), ("path start pitch", "climb °", &mut start.pitch, -90.0, 90.0), ("path start roll", "roll °", &mut start.roll, -180.0, 180.0)] {
            let m = knob::value_row(ui, px, key, name, *v, lo, hi, 7.5);
            if let Some(x) = m.value { *v = x; changed = (true, !m.released); }
        }
    });
    if changed.0 { set_path(st, path, changed.1); }
}

// ---- on the picture -------------------------------------------------------------------------------

/// The camera, and how the selected layer's own space lies in the camera's frame.
struct Look {
    cam: Camera,
    /// the layer's space -> the camera's: m q + c
    m: Mat3,
    c: [f64; 3],
    rect: Rect,
}

impl Look {
    fn of(st: &State, rect: Rect) -> Look {
        let cam = st.camera();
        let v = st.view_r();
        let place = st.stack.layers.get(st.stack.sel).map(|l| l.place).unwrap_or_default();
        let r = euler(place.rot[0], place.rot[1], place.rot[2]);
        let s = if place.size != 0.0 { place.size } else { 1.0 };
        let m = gf_core::math::mul(&v, &r).map(|x| x * s);
        Look { cam, m, c: apply(&v, place.pos), rect }
    }
    fn to_cam(&self, q: [f64; 3]) -> [f64; 3] {
        add(apply(&self.m, q), self.c)
    }
    /// A point of the camera's frame back in the layer's space.
    fn to_space(&self, p: [f64; 3]) -> [f64; 3] {
        let inv = invert(&self.m);
        apply(&inv, sub(p, self.c))
    }
    fn dir_to_space(&self, d: [f64; 3]) -> [f64; 3] {
        norm(apply(&invert(&self.m), d))
    }
    fn screen(&self, q: [f64; 3]) -> Option<Pos2> {
        let uv = self.cam.project(self.to_cam(q))?;
        let h = self.rect.height() as f64 / 2.0;
        Some(Pos2::new((self.rect.center().x as f64 + uv[0] * h) as f32, (self.rect.center().y as f64 - uv[1] * h) as f32))
    }
    /// The pointer's ray in the layer's space.
    fn ray(&self, at: Pos2) -> Option<([f64; 3], [f64; 3])> {
        let h = self.rect.height() as f64 / 2.0;
        let (o, d) = self.cam.ray([(at.x - self.rect.center().x) as f64 / h, -(at.y - self.rect.center().y) as f64 / h])?;
        Some((self.to_space(o), self.dir_to_space(d)))
    }
}

/// The inverse of a turn scaled evenly: its transpose over the square of its scale.
fn invert(m: &Mat3) -> Mat3 {
    let s2 = m[0] * m[0] + m[3] * m[3] + m[6] * m[6];
    transpose(m).map(|x| x / s2.max(1e-300))
}

/// The ray (o, d) and the plane through c across n: where they meet.
fn meet_plane(o: [f64; 3], d: [f64; 3], c: [f64; 3], n: [f64; 3]) -> Option<[f64; 3]> {
    let dn = dot(d, n);
    if dn.abs() < 1e-9 { return None; }
    let t = dot(sub(c, o), n) / dn;
    (t > 0.0).then(|| add(o, scale(d, t)))
}
/// How far along the line through c along unit a lies the point of it nearest to the ray (o, d).
fn along_line(o: [f64; 3], d: [f64; 3], c: [f64; 3], a: [f64; 3]) -> Option<f64> {
    let w = sub(c, o);
    let (aa, ad, dd) = (dot(a, a), dot(a, d), dot(d, d));
    let den = aa * dd - ad * ad;
    if den.abs() < 1e-9 { return None; }
    Some((ad * dot(w, d) - dd * dot(w, a)) / den)
}

/// A curve's four control points, in the layer's space.
fn curve_points(f: &Frame, out: f64, h2: [f64; 3], end: [f64; 3]) -> [[f64; 3]; 4] {
    [f.p, add(f.p, scale(f.axes().t, out)), f.to_world(h2), f.to_world(end)]
}

fn dist_to_line(p: Pos2, pts: &[Pos2]) -> f32 {
    let mut best = f32::MAX;
    for w in pts.windows(2) {
        let ab = w[1] - w[0];
        let t = if ab.length_sq() > 0.0 { ((p - w[0]).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
        best = best.min((w[0] + ab * t - p).length());
    }
    best
}

impl PathEdit {
    /// What of the path is under the pointer: handles first, then joints, then the line.
    pub fn hit(&self, st: &State, rect: Rect, at: Pos2) -> Option<Hit> {
        if !self.show || !editing(st) { return None; }
        let path = path_now(st);
        let walk = path.walk();
        let look = Look::of(st, rect);
        let near = |q: [f64; 3]| look.screen(q).is_some_and(|s| (s - at).length() <= REACH_PT);
        // the selected curve's handles
        if let Some(i) = self.sel {
            if let (Some(Piece::Curve { out, h2, end }), Some(f)) = (path.pieces.get(i), walk.joints.get(i)) {
                let c = curve_points(f, *out, *h2, *end);
                if near(c[1]) { return Some(Hit::Out(i)); }
                if near(c[2]) { return Some(Hit::Arrive(i)); }
            }
        }
        for i in (0..path.pieces.len()).rev() {
            if near(walk.joints[i + 1].p) { return Some(Hit::Joint(i)); }
        }
        if near(path.start.p) { return Some(Hit::Start); }
        // the line: the nearest piece's
        let mut best = (LINE_PT, None);
        for i in 0..path.pieces.len() {
            let pts: Vec<Pos2> = std::iter::once(walk.joints[i].p).chain(walk.samples.iter().filter(|s| s.piece == i && s.s > 0.0).map(|s| s.p)).filter_map(|q| look.screen(q)).collect();
            let d = dist_to_line(at, &pts);
            if d < best.0 { best = (d, Some(i)); }
        }
        best.1.map(Hit::Line)
    }

    pub fn begin(&mut self, st: &State, at: Pos2, hit: Hit) {
        let path = path_now(st);
        self.grab = Some(Grab { hit, text0: st.shape().model.clone(), path0: path, at0: at, moved: 0.0 });
        if let Hit::Joint(i) | Hit::Line(i) | Hit::Out(i) | Hit::Arrive(i) = hit { self.sel = Some(i); }
    }

    /// The pointer has moved to `at` while holding on. `free`: no steps (ctrl); `vertical`: up and down (shift).
    pub fn drag(&mut self, st: &mut State, rect: Rect, at: Pos2, free: bool, vertical: bool) {
        let Some(g) = &mut self.grab else { return };
        g.moved = g.moved.max((at - g.at0).length());
        let snap = self.snap && !free;
        let look = Look::of(st, rect);
        let Some((o, d)) = look.ray(at) else { return };
        let mut path = g.path0.clone();
        let walk = path.walk();
        let up = [0.0, 1.0, 0.0];
        // a point moved across the floor (the level plane through it), or up and down (the upright plane
        // through it that faces the camera best)
        let moved_point = |q: [f64; 3]| -> Option<[f64; 3]> {
            if vertical {
                let s = along_line(o, d, q, up)?;
                Some([q[0], snap_to(q[1] + s, GRID, snap), q[2]])
            } else {
                let p = meet_plane(o, d, q, up)?;
                Some([snap_to(p[0], GRID, snap), q[1], snap_to(p[2], GRID, snap)])
            }
        };
        match g.hit {
            Hit::Line(_) => return,
            Hit::Start => {
                let Some(p) = moved_point(path.start.p) else { return };
                path.start.p = p;
            }
            Hit::Joint(i) => {
                let f = walk.joints[i];
                let piece = path.pieces[i].clone();
                match piece {
                    Piece::Straight { .. } | Piece::Bank { .. } => {
                        let Some(s) = along_line(o, d, f.p, f.axes().t) else { return };
                        let len = snap_to(s, GRID, snap).clamp(0.0, path::LIM);
                        path.pieces[i] = match piece { Piece::Bank { deg, .. } => Piece::Bank { deg, len }, _ => Piece::Straight { len } };
                    }
                    Piece::Turn { deg, .. } | Piece::Slope { deg, .. } => {
                        // the end of an arc moves along a line through its start as its radius grows
                        let one = Path { start: f, pieces: vec![match piece { Piece::Turn { .. } => Piece::Turn { deg, radius: 1.0 }, _ => Piece::Slope { deg, radius: 1.0 } }] };
                        let e = one.walk().joints[1].p;
                        let v = sub(e, f.p);
                        if len3(v) < 1e-9 { return; }
                        let Some(s) = along_line(o, d, f.p, norm(v)) else { return };
                        let radius = snap_to(s / len3(v), GRID / 5.0, snap).clamp(0.02, path::LIM);
                        path.pieces[i] = match piece { Piece::Turn { .. } => Piece::Turn { deg, radius }, _ => Piece::Slope { deg, radius } };
                    }
                    Piece::Loop { .. } => return,
                    Piece::Curve { out, h2, end } => {
                        let p3 = f.to_world(end);
                        let Some(q) = moved_point(p3) else { return };
                        let e2 = f.to_local(q);
                        let shift = sub(e2, end);
                        path.pieces[i] = Piece::Curve { out, h2: add(h2, shift), end: e2 };
                    }
                }
            }
            Hit::Out(i) => {
                let f = walk.joints[i];
                let Piece::Curve { h2, end, .. } = path.pieces[i] else { return };
                let Some(s) = along_line(o, d, f.p, f.axes().t) else { return };
                path.pieces[i] = Piece::Curve { out: snap_to(s, GRID, snap).clamp(0.0, path::LIM), h2, end };
            }
            Hit::Arrive(i) => {
                let f = walk.joints[i];
                let Piece::Curve { out, h2, end } = path.pieces[i] else { return };
                let (p2, p3) = (f.to_world(h2), f.to_world(end));
                let q = if vertical { moved_point(p2) } else { meet_plane(o, d, p2, up) };
                let Some(mut q) = q else { return };
                if snap && !vertical {
                    // the arrival's heading in steps of 15 degrees, its handle's length on the grid
                    let v = sub(q, p3);
                    let flat = v[0].hypot(v[2]);
                    let a = (v[2].atan2(v[0]).to_degrees() / ANGLE_STEP).round() * ANGLE_STEP;
                    let l = snap_to(flat, GRID, true).max(GRID);
                    q = [p3[0] + l * a.to_radians().cos(), q[1], p3[2] + l * a.to_radians().sin()];
                }
                path.pieces[i] = Piece::Curve { out, h2: f.to_local(q), end };
            }
        }
        set_path(st, &path, true);
    }

    /// Let go: the shape is built finely; a press that hardly moved is a click (it selects).
    pub fn end(&mut self, st: &mut State) {
        let Some(g) = self.grab.take() else { return };
        if g.moved < 4.0 {
            match g.hit {
                Hit::Line(i) | Hit::Joint(i) => self.sel = if self.sel == Some(i) && matches!(g.hit, Hit::Line(_)) { None } else { Some(i) },
                Hit::Start => self.sel = None,
                _ => {}
            }
        }
        if st.shape().model != g.text0 { st.want = Some(Detail::Full); }
    }

    /// Esc during a drag: the path as it was when the drag began. True if there was a drag.
    pub fn cancel(&mut self, st: &mut State) -> bool {
        let Some(g) = self.grab.take() else { return false };
        st.shape_mut().model = g.text0;
        st.want = Some(Detail::Full);
        true
    }

    /// Draw the path over the picture: its line, the selected piece lit, the joints, the start and the end,
    /// a curve's handles, and the ghost of a piece about to be added.
    pub fn paint(&mut self, p: &Painter, st: &State, rect: Rect, pointer: Option<Pos2>) {
        self.spots.clear();
        if !self.show || !editing(st) { self.hover = None; return; }
        let path = path_now(st);
        let walk = path.walk();
        let look = Look::of(st, rect);
        self.hover = if self.grab.is_some() { self.grab.as_ref().map(|g| g.hit) } else { pointer.and_then(|at| self.hit(st, rect, at)) };
        let shadow = Stroke::new(3.2, Color32::from_black_alpha(110));
        let line_of = |i: usize| -> Vec<Pos2> {
            std::iter::once(walk.joints[i].p).chain(walk.samples.iter().filter(|s| s.piece == i && s.s > 0.0).map(|s| s.p)).filter_map(|q| look.screen(q)).collect()
        };
        // the line, piece by piece
        for i in 0..path.pieces.len() {
            let pts = line_of(i);
            if pts.len() < 2 { continue; }
            let lit = self.sel == Some(i);
            let hot = matches!(self.hover, Some(Hit::Line(j)) if j == i);
            let ink = if lit { pal().accent } else if hot { white(0.85) } else { white(0.5) };
            p.add(Shape::line(pts.clone(), shadow));
            p.add(Shape::line(pts, Stroke::new(if lit { 2.6 } else { 1.2 }, ink)));
        }
        // the ghost of the piece a button would add, from where it would go
        if let Some(g) = &self.ghost {
            let at = self.insert_at(&path);
            let mut q = path.clone();
            q.pieces.insert(at, g.clone());
            let w = q.walk();
            let pts: Vec<Pos2> = std::iter::once(w.joints[at].p).chain(w.samples.iter().filter(|s| s.piece == at && s.s > 0.0).map(|s| s.p)).filter_map(|x| look.screen(x)).collect();
            if pts.len() > 1 {
                p.add(Shape::line(pts.clone(), Stroke::new(4.0, fade(pal().past, 0.25))));
                p.add(Shape::line(pts.clone(), Stroke::new(1.6, pal().past)));
                if let Some(e) = pts.last() { p.circle_stroke(*e, 4.0, Stroke::new(1.4, pal().past)); }
            }
        }
        // the joints
        for i in 0..path.pieces.len() {
            let Some(s) = look.screen(walk.joints[i + 1].p) else { continue };
            self.spots.push((format!("path-joint:{}", i + 1), s));
            let hot = matches!(self.hover, Some(Hit::Joint(j)) if j == i);
            let lit = self.sel == Some(i);
            let r = if hot { 5.0 } else { 3.6 };
            p.circle(s, r, if lit { pal().accent } else if hot { Color32::WHITE } else { white(0.75) }, Stroke::new(1.2, Color32::from_black_alpha(160)));
        }
        // the start: a ring with the way it heads
        if let Some(s) = look.screen(path.start.p) {
            self.spots.push(("path-start".to_owned(), s));
            let hot = matches!(self.hover, Some(Hit::Start));
            let ink = if hot { Color32::WHITE } else { pal().bright };
            p.circle_stroke(s, 6.5, Stroke::new(3.0, Color32::from_black_alpha(110)));
            p.circle_stroke(s, 6.5, Stroke::new(1.5, ink));
            if let Some(t) = look.screen(add(path.start.p, scale(path.start.axes().t, 0.08))) {
                arrow(p, s, t, ink);
            }
        }
        // the end, where the next piece goes (unless the loop is closed)
        if !walk.closed {
            if let Some(f) = walk.joints.last() {
                if let (Some(a), Some(b)) = (look.screen(f.p), look.screen(add(f.p, scale(f.axes().t, 0.1)))) {
                    self.spots.push(("path-end".to_owned(), a));
                    arrow(p, a, b, if self.sel.is_none() { pal().past } else { white(0.6) });
                }
            }
        }
        // the selected curve's handles
        if let Some(i) = self.sel {
            if let (Some(Piece::Curve { out, h2, end }), Some(f)) = (path.pieces.get(i), walk.joints.get(i)) {
                let c = curve_points(f, *out, *h2, *end);
                let s: Vec<Option<Pos2>> = c.iter().map(|q| look.screen(*q)).collect();
                for (a, b, which) in [(0, 1, Hit::Out(i)), (3, 2, Hit::Arrive(i))] {
                    if let (Some(x), Some(y)) = (s[a], s[b]) {
                        p.add(Shape::line_segment([x, y], Stroke::new(1.0, fade(pal().accent, 0.8))));
                        let hot = self.hover == Some(which);
                        let r = egui::Rect::from_center_size(y, egui::vec2(if hot { 11.0 } else { 9.0 }, if hot { 11.0 } else { 9.0 }));
                        p.rect(r, 1.5, if hot { Color32::WHITE } else { pal().accent }, Stroke::new(1.0, Color32::from_black_alpha(160)), egui::StrokeKind::Outside);
                        self.spots.push((format!("path-{}:{}", if a == 0 { "out" } else { "arrive" }, i + 1), y));
                    }
                }
            }
        }
        // while dragging: what it is being set to
        if let (Some(g), Some(at)) = (&self.grab, pointer) {
            let words = match g.hit {
                Hit::Start => Some(format!("start {} {} {}", path::fmt(path.start.p[0]), path::fmt(path.start.p[1]), path::fmt(path.start.p[2]))),
                Hit::Joint(i) | Hit::Out(i) | Hit::Arrive(i) => path.pieces.get(i).map(|x| x.words()),
                Hit::Line(_) => None,
            };
            if let Some(w) = words.filter(|_| g.moved >= 4.0) {
                let gal = p.layout_no_wrap(w, mono(SIZE), pal().bright);
                let r = Align2::LEFT_BOTTOM.anchor_size(at + egui::vec2(14.0, -10.0), gal.size());
                p.rect_filled(r.expand2(egui::vec2(5.0, 2.0)), 5, Color32::from_black_alpha(150));
                p.galley(r.min, gal, pal().bright);
            }
        }
    }
}

/// A short arrow from a to b on the screen.
fn arrow(p: &Painter, a: Pos2, b: Pos2, ink: Color32) {
    let d = b - a;
    if d.length() < 1.0 { return; }
    let u = d.normalized();
    let tip = a + u * d.length().clamp(14.0, 40.0);
    p.add(Shape::line_segment([a, tip], Stroke::new(3.0, Color32::from_black_alpha(110))));
    p.add(Shape::line_segment([a, tip], Stroke::new(1.5, ink)));
    let n = egui::vec2(-u.y, u.x);
    p.add(Shape::convex_polygon(vec![tip + u * 5.0, tip - u * 2.0 + n * 4.0, tip - u * 2.0 - n * 4.0], ink, Stroke::new(1.0, Color32::from_black_alpha(120))));
}

/// What the line over the picture is, in a few words (for the hint under the tool bar).
pub fn hint(h: Hit) -> &'static str {
    match h {
        Hit::Start => "drag: move the whole path · shift: up and down",
        Hit::Joint(_) => "drag: this piece's length or radius (a curve: its end) · ctrl: freely",
        Hit::Out(_) => "drag: how far the curve leaves along the heading",
        Hit::Arrive(_) => "drag: how the curve arrives · in 15° steps while snapping",
        Hit::Line(_) => "click: select this piece",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_makes_the_piece_it_names() {
        let ed = PathEdit::default();
        for k in KINDS {
            let p = ed.make(k, 0.1);
            assert_eq!(Kind::of(&p), k, "{k:?}");
            // and made another kind keeps being a piece of that kind
            for k2 in KINDS { assert_eq!(Kind::of(&reshaped(&p, k2, 0.1, &ed)), k2); }
        }
    }

    #[test]
    fn a_scaled_turn_comes_back() {
        let m = gf_core::math::mul(&euler(10.0, 20.0, 30.0), &euler(0.0, 0.0, 0.0)).map(|x| x * 0.7);
        let i = invert(&m);
        let v = [0.3, -0.2, 0.5];
        let back = apply(&i, apply(&m, v));
        for k in 0..3 { assert!((back[k] - v[k]).abs() < 1e-12); }
    }
}
