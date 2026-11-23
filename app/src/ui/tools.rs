//! The tool bars over the picture, and the little cube of the room's axes.
//!
//! Split as the panels are (the owner's ruling: the left is everything that is not about objects, the
//! right everything that is): beside the left panel the camera's bar (orbiting or flying, "frame", which
//! brings what is chosen to the middle of the picture, and the ten camera slots) with the cube; beside the
//! right panel the object bar (select, all, move, turn, size; the handles' axes, the scene's or the chosen
//! thing's own; snapping). Every one of them says its key when hovered.
//! None of them changes anything but numbers the sliders also change (see handles.rs and gf_core::view),
//! so whatever they do is kept in a setup and animates as the sliders do.
//!
//! The cube shows which way the camera looks in the room: the room's x (red), y (green) and z (blue)
//! as the camera sees them, the ends that point at the camera drawn in front. A click on an end looks
//! along that axis at the pivot: from the right, from the top, from the front, or from the other side.

use std::time::{Duration, Instant};

use eframe::egui::{self, Align2, Color32, CursorIcon, Id, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui};
use gf_core::math::{apply, transpose};
use gf_core::state::State;

use super::handles::{Target, Tool};
use super::theme::*;
use super::widgets::*;
use super::Panels;

/// How the camera is moved about: orbiting what it looks at, or flying free; and how fast it flies.
pub struct Nav {
    pub fly: bool,
    /// scene units a second
    pub speed: f64,
    /// a few words shown under the bar for a moment (the speed, as the wheel changes it)
    pub said: Option<(String, Instant)>,
    /// they are about what the object tools did (shown under the object bar, not the camera's)
    pub said_by_objects: bool,
}
impl Default for Nav {
    fn default() -> Self {
        Self { fly: false, speed: 1.5, said: None, said_by_objects: false }
    }
}
impl Nav {
    /// What the line says now, if anything.
    pub fn said_text(&self) -> Option<&str> { self.said.as_ref().map(|(w, _)| w.as_str()) }
    pub fn say(&mut self, words: String) {
        self.said = Some((words, Instant::now()));
        self.said_by_objects = false;
    }
    /// The same, under the object bar.
    pub fn say_for_objects(&mut self, words: String) {
        self.said = Some((words, Instant::now()));
        self.said_by_objects = true;
    }
    /// Fly, or orbit again. Nothing in the picture moves; orbiting turns about what is in the middle of it.
    pub fn set_fly(&mut self, st: &mut State, fly: bool) {
        if self.fly && !fly { st.pivot_ahead(); }
        self.fly = fly;
        self.say(if fly { "flying: w a s d, q e, drag to look".into() } else { "orbiting".into() });
    }
}

/// Fly speeds the wheel may set, scene units a second.
pub const SPEED: (f64, f64) = (0.05, 40.0);

const B: f32 = 26.0;
const GAP: f32 = 3.0;
const SPLIT: f32 = 9.0;
/// The cube's square, and about how wide the bar is.
pub const CUBE: f32 = 84.0;
const AXIS_INK: [Color32; 3] = [Color32::from_rgb(235, 90, 85), Color32::from_rgb(110, 205, 95), Color32::from_rgb(95, 145, 240)];

#[derive(Clone, Copy)]
enum Icon {
    Select,
    All,
    Move,
    Turn,
    Size,
    Orbit,
    Fly,
}

fn p2(c: Pos2, x: f32, y: f32) -> Pos2 {
    Pos2::new(c.x + x, c.y + y)
}

/// The small drawings on the tool buttons, 16 points across, in `ink`.
fn icon(p: &Painter, r: Rect, which: Icon, ink: Color32) {
    let c = r.center();
    let s = Stroke::new(1.4, ink);
    let head = |tip: Pos2, dir: egui::Vec2, size: f32| {
        let side = egui::vec2(-dir.y, dir.x);
        Shape::convex_polygon(vec![tip, tip - dir * size + side * size * 0.6, tip - dir * size - side * size * 0.6], ink, Stroke::NONE)
    };
    match which {
        Icon::Select => {
            // a pointer
            let pts = vec![p2(c, -4.0, -7.0), p2(c, -4.0, 5.0), p2(c, -1.0, 2.2), p2(c, 1.4, 7.0), p2(c, 3.2, 6.0), p2(c, 0.9, 1.5), p2(c, 4.6, 1.2)];
            p.add(Shape::closed_line(pts, s));
        }
        Icon::All => {
            // a ring round three short arrows
            p.circle_stroke(c, 7.0, Stroke::new(1.1, ink));
            for d in [egui::vec2(1.0, 0.0), egui::vec2(0.0, -1.0), egui::vec2(-0.7, 0.7)] {
                p.line_segment([c, c + d * 3.6], s);
                p.add(head(c + d * 6.2, d, 2.6));
            }
            p.circle_filled(c, 1.6, ink);
        }
        Icon::Move => {
            // four arrows from the middle
            for d in [egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0), egui::vec2(0.0, 1.0), egui::vec2(0.0, -1.0)] {
                p.line_segment([c + d * 1.5, c + d * 5.5], s);
                p.add(head(c + d * 8.0, d, 3.2));
            }
        }
        Icon::Turn => {
            // most of a ring, with an arrowhead
            let pts: Vec<Pos2> = (0..=20).map(|i| {
                let a = (0.35 + 1.45 * i as f32 / 20.0) * std::f32::consts::PI;
                p2(c, 6.0 * a.cos(), 6.0 * a.sin())
            }).collect();
            let (a, b) = (pts[19], pts[20]);
            p.add(Shape::line(pts, s));
            p.add(head(b + (b - a).normalized() * 2.5, (b - a).normalized(), 3.4));
        }
        Icon::Size => {
            // a small square in a large one's corner, and the way it grows
            p.rect_stroke(Rect::from_min_size(p2(c, -6.5, -0.5), egui::vec2(7.0, 7.0)), 0, s, StrokeKind::Middle);
            p.add(Shape::dashed_line(&[p2(c, -6.5, -6.5), p2(c, 6.5, -6.5), p2(c, 6.5, 6.5)], Stroke::new(1.1, ink), 2.0, 1.6));
            p.line_segment([p2(c, 0.5, -0.5), p2(c, 4.0, -4.0)], s);
            p.add(head(p2(c, 6.0, -6.0), egui::vec2(0.7071, -0.7071), 3.0));
        }
        Icon::Orbit => {
            // a ball with a moon going round it on a tilted path
            p.circle_stroke(c, 3.2, s);
            let at = |a: f32| { let (x, y) = (8.0 * a.cos(), 3.0 * a.sin()); p2(c, x * 0.92 + y * 0.39, -x * 0.39 + y * 0.92) };
            // (the far side of the path passes behind the ball)
            let front: Vec<Pos2> = (0..=20).map(|i| at(i as f32 / 20.0 * std::f32::consts::PI)).collect();
            let back: Vec<Pos2> = (0..=20).map(|i| at(std::f32::consts::PI * (1.0 + i as f32 / 20.0))).filter(|q| (*q - c).length() > 4.6).collect();
            p.add(Shape::line(front, s));
            p.add(Shape::line(back, Stroke::new(1.0, ink.gamma_multiply(0.6))));
            p.circle_filled(at(0.55 * std::f32::consts::PI), 2.0, ink);
        }
        Icon::Fly => {
            // a paper dart, going up and to the right
            let pts = vec![p2(c, 7.0, -6.0), p2(c, -7.0, -0.5), p2(c, -1.5, 1.5), p2(c, 0.5, 7.0)];
            p.add(Shape::closed_line(pts, s));
            p.line_segment([p2(c, -1.5, 1.5), p2(c, 7.0, -6.0)], s);
        }
    }
}

/// A square button with a drawing on it. `on`: lit. `mark`: a dot under the drawing (the way the camera
/// moves, while another tool is in use).
fn icon_button(ui: &mut Ui, rect: Rect, id: Id, which: Icon, on: bool, mark: bool) -> Response {
    let resp = button(ui, rect, id, "", on, mono(SIZE));
    let ink = if on { pal().ink } else if resp.hovered() { pal().hot } else { pal().button };
    icon(ui.painter(), rect, which, ink);
    if mark && !on { ui.painter().circle_filled(Pos2::new(rect.center().x, rect.bottom() - 3.0), 1.6, fade(pal().bright, 0.8)); }
    resp
}

/// How tall a bar is, and how far below the bars (and the cube) a floating window starts by default,
/// so that it covers neither.
pub const BAR_H: f32 = B + 10.0;
const BELOW: f32 = 10.0;

/// The two bars, by the owner's split of the window: the camera's tools (orbit, fly, frame, the camera
/// slots) beside the left panel with the cube of the room's axes, which says which way the camera looks;
/// the object tools (select, all, move, turn, size, world or local, snap) beside the right panel, the
/// panel of objects. About how wide each is, in points.
pub const CAMERA_W: f32 = 2.0 * 5.0 + 2.0 * B + 2.0 * GAP + FRAME_W + SPLIT + 10.0 * DOT + 4.0;
/// The width of "frame".
const FRAME_W: f32 = 54.0;
pub const OBJECT_W: f32 = 2.0 * 5.0 + 5.0 * B + 4.0 * GAP + SPLIT + 2.0 * 50.0 + GAP;
/// One camera slot's width in the camera bar.
const DOT: f32 = 11.0;

/// Where the camera's bar stands: beside the left panel (at the window's corner when it is hidden).
fn camera_at(px: &Panels, full: Rect) -> Pos2 {
    Pos2::new(px.left_at.map_or(full.left() + EDGE, |r| r.right() + 10.0), full.top() + EDGE)
}
/// Where the object bar's right end stands: beside the right panel (at the window's corner when it is hidden).
fn object_end(px: &Panels, full: Rect) -> f32 {
    px.right_at.map_or(full.right() - EDGE, |r| r.left() - 10.0)
}

/// The lowest a floating window on the left starts without covering the camera's bar or the cube; and on
/// the right, without covering the object bar.
pub fn below_left(full: Rect) -> f32 {
    full.top() + EDGE + CUBE.max(2.0 * BAR_H + GAP) + BELOW
}
pub fn below_right(full: Rect) -> f32 {
    full.top() + EDGE + BAR_H + BELOW
}

/// The bars and the cube, between the panels; where the room between them is short, the cube gives way
/// first, then the object bar.
pub fn bars(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    let lo = px.left_at.map_or(full.left(), |r| r.right());
    let hi = px.right_at.map_or(full.right(), |r| r.left());
    let room = hi - lo;
    if room >= CAMERA_W + 24.0 {
        camera_bar(ctx, st, px, full);
        super::tiers::bar(ctx, st, px, camera_at(px, full));
    }
    if room >= CAMERA_W + OBJECT_W + 44.0 { object_bar(ctx, st, px, full); }
    if room >= CAMERA_W + OBJECT_W + CUBE + 60.0 { cube(ctx, st, px, full); }
    words(ctx, px, full);
}

/// The camera's bar: orbit or fly, frame, and the ten camera slots (a dot for each, filled where a view is
/// stored: a click goes to it, ctrl+click stores the view there).
fn camera_bar(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    let at = camera_at(px, full);
    let font = mono(SIZE);
    let shown = egui::Area::new(Id::unique("camera bar")).order(egui::Order::Middle).fixed_pos(at).constrain(false).fade_in(false).show(ctx, |ui| {
        let pad = 5.0;
        let w = CAMERA_W;
        let frame_w = FRAME_W;
        let _ = w;
        let (_, whole) = ui.allocate_space(egui::vec2(w, B + 2.0 * pad));
        ui.painter().add(glass(whole, PANEL_RADIUS, Lift::Bar, px.bright));
        let mut x = whole.left() + pad;
        let y = whole.top() + pad;
        let mut place = |wd: f32, gap: f32| {
            let r = Rect::from_min_size(Pos2::new(x, y), egui::vec2(wd, B));
            x += wd + gap;
            r
        };
        let camera = px.handles.tool() == Tool::Camera;
        let r = place(B, GAP);
        px.note("tool:orbit", r);
        if icon_button(ui, r, Id::unique(("tool", "orbit")), Icon::Orbit, camera && !px.nav.fly, !px.nav.fly)
            .on_hover_text("orbit: dragging turns the view round the shape (or what was framed), the wheel comes closer, a click focuses while the lens has an aperture · V: orbit or fly")
            .clicked()
        {
            px.handles.set_tool(Tool::Camera);
            px.nav.set_fly(st, false);
        }
        let r = place(B, GAP);
        px.note("tool:fly", r);
        if icon_button(ui, r, Id::unique(("tool", "fly")), Icon::Fly, camera && px.nav.fly, px.nav.fly)
            .on_hover_text("fly: W A S D move, Q and E down and up, a drag (either button) looks round, the wheel sets the speed; shift: fast, ctrl: slow · V: orbit or fly")
            .clicked()
        {
            px.handles.set_tool(Tool::Camera);
            px.nav.set_fly(st, true);
        }
        let r = place(frame_w, SPLIT);
        px.note("tool:frame", r);
        if button(ui, r, Id::unique(("tool", "frame")), "frame", false, font.clone())
            .on_hover_text("the camera slides to have what is chosen in the middle of the picture, and turns about it from then on (nothing chosen: the whole scene) · F")
            .clicked()
        {
            frame(st, px);
        }
        // the camera slots: 1 to 9, then 0, as the keys
        let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
        for i in 0..gf_core::history::SLOTS {
            let r = Rect::from_min_size(Pos2::new(x + i as f32 * DOT, y), egui::vec2(DOT, B));
            let n = (i + 1) % 10;
            px.note(format!("view-slot:{n}"), r);
            let resp = ui.interact(r, Id::unique(("view slot", i)), Sense::click());
            let stored = st.view_stored(i);
            let hot = resp.hovered();
            if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
            let c = Pos2::new(r.center().x, r.center().y - 4.0);
            let ink = if hot { pal().hot } else if stored { pal().bright } else { pal().hint };
            if stored { ui.painter().circle_filled(c, 3.2, ink); } else { ui.painter().circle_stroke(c, 2.8, Stroke::new(1.0, ink)); }
            ui.painter().text(Pos2::new(r.center().x, r.bottom() - 5.0), Align2::CENTER_CENTER, format!("{n}"), mono(8.5), if hot { pal().bright } else if stored { pal().dim } else { pal().hint });
            let tip = if stored { format!("view {n}: click to go to it (shift+{n}) · ctrl+click stores this view in its place (ctrl+{n})") } else { format!("no view in {n} yet: ctrl+click (or ctrl+{n}) stores this view there") };
            if resp.on_hover_text(tip).clicked() {
                if ctrl {
                    st.store_view(i);
                    px.nav.say(format!("view {n} stored · click its dot or shift+{n} to go back to it"));
                } else if st.recall_view(i) {
                    px.nav.say(format!("view {n}"));
                } else {
                    px.nav.say(format!("no view in {n} yet · ctrl+click its dot stores one"));
                }
            }
        }
        whole
    });
    px.note("camera bar", shown.response.rect);
}

/// The object bar: the tools that choose, move, turn and size what is in the scene, the handles' axes and
/// snapping.
fn object_bar(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    let font = mono(SIZE);
    let words = [if px.handles.local { "local" } else { "world" }, "snap"];
    let pad = 5.0;
    // (laid out from the right end leftwards: the words' buttons are as wide as the wider word could want)
    let wide = [50.0f32, 46.0];
    let w = 2.0 * pad + 5.0 * B + 4.0 * GAP + SPLIT + wide.iter().sum::<f32>() + GAP;
    let at = Pos2::new(object_end(px, full) - w, full.top() + EDGE);
    let shown = egui::Area::new(Id::unique("tool bar")).order(egui::Order::Middle).fixed_pos(at).constrain(false).fade_in(false).show(ctx, |ui| {
        let tool = px.handles.tool();
        let (_, whole) = ui.allocate_space(egui::vec2(w, B + 2.0 * pad));
        ui.painter().add(glass(whole, PANEL_RADIUS, Lift::Bar, px.bright));
        let mut x = whole.left() + pad;
        let y = whole.top() + pad;
        let mut place = |wd: f32, gap: f32| {
            let r = Rect::from_min_size(Pos2::new(x, y), egui::vec2(wd, B));
            x += wd + gap;
            r
        };
        let tools = [
            (Tool::Select, Icon::Select, "select", "select: a click on a shape chooses its layer, on a lamp the lamp, on nothing lets go · esc"),
            (Tool::All, Icon::All, "all", "all at once: arrows (the whole line) move along an axis, coloured rings turn about it, the middle moves across the picture, the pale ring turns about the line of sight, the faint ball turns any way, the orange ring sizes; a lamp's small squares size it along its own axes · G shows or hides the handles, T steps through all, move, turn, size; ctrl: in steps, shift: finely, esc during a drag puts it back"),
            (Tool::Move, Icon::Move, "move", "move only: arrows along the axes, squares across two of them, the dot across the picture · T"),
            (Tool::Turn, Icon::Turn, "turn", "turn: a ring about each axis · T"),
            (Tool::Size, Icon::Size, "size", "size: the ring round it · T"),
        ];
        for (k, (t, ic, name, tip)) in tools.iter().enumerate() {
            let r = place(B, if k == 4 { SPLIT } else { GAP });
            px.note(format!("tool:{name}"), r);
            if icon_button(ui, r, Id::unique(("tool", *name)), *ic, tool == *t, false).on_hover_text(*tip).clicked() {
                px.handles.set_tool(*t);
                if *t != Tool::Select && st.stack.sel >= st.layer_count() { px.handles.let_go(); }
            }
        }
        let r = place(wide[0], GAP);
        px.note("tool:axes", r);
        if button(ui, r, Id::unique(("tool", "axes")), words[0], false, font.clone())
            .on_hover_text(if px.handles.local { "the handles go along the chosen thing's own axes, turned with it: click for the scene's" } else { "the handles go along the scene's axes: click for the chosen thing's own, turned with it" })
            .clicked()
        {
            px.handles.local = !px.handles.local;
        }
        let r = place(wide[1], 0.0);
        px.note("tool:snap", r);
        if button(ui, r, Id::unique(("tool", "snap")), words[1], px.handles.snap, font.clone())
            .on_hover_text("the handles go in steps: 0.05 of a unit, 15 degrees, sizes by 0.05 (ctrl does the same while dragging, or the opposite while this is on)")
            .clicked()
        {
            px.handles.snap = !px.handles.snap;
        }
        whole
    });
    px.note("tool bar", shown.response.rect);
}

/// A few words under a bar for a moment (the fly speed, an undo, a camera slot under the camera's bar;
/// what the object tools did under the object bar), drawn whether or not the bar has room to show.
pub fn words(ctx: &egui::Context, px: &mut Panels, full: Rect) {
    if let Some((words, when)) = &px.nav.said {
        let age = when.elapsed();
        if age < Duration::from_millis(1600) {
            let a = if age < Duration::from_millis(1100) { 1.0 } else { 1.0 - (age.as_secs_f32() - 1.1) / 0.5 };
            let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, Id::unique("tool bar words")));
            let g = p.layout_no_wrap(words.clone(), mono(SMALL), fade(pal().bright, 0.85 * a));
            let at = if px.nav.said_by_objects {
                Pos2::new(object_end(px, full) - 6.0 - g.size().x, full.top() + EDGE + B + 16.0)
            } else {
                camera_at(px, full) + egui::vec2(6.0, B + 16.0)
            };
            p.rect_filled(Rect::from_min_size(at - egui::vec2(5.0, 2.0), g.size() + egui::vec2(10.0, 4.0)), BUTTON_RADIUS, fade(bed(px.bright), a));
            p.galley(at, g, pal().bright);
            ctx.request_repaint_after(Duration::from_millis(40));
        } else {
            px.nav.said = None;
        }
    }
}

/// Bring what is chosen (nothing chosen: the whole scene) to the middle of the picture: F, and the
/// bar's "frame".
pub fn frame(st: &mut State, px: &mut Panels) {
    let (centre, r) = match px.handles.chosen(st) {
        Some(Target::World(item)) => { let Some(bounds) = st.world_framing(item) else { return }; bounds }
        Some(Target::Pivot(layer, i)) => (layer.position(st, i).unwrap_or_default(), 0.2),
        Some(t @ (Target::Part(_) | Target::Whole | Target::ConstructionNode)) => {
            let Some(h) = t.construction(st) else { return };
            let (c, size) = h.in_scene();
            (c, gf_core::FIT * size.max(1e-3))
        }
        Some(Target::Lamp(i)) => {
            let Some((c, _)) = st.lamp_in_view(i) else { return };
            let l = &st.lamps[i];
            // (a lamp stands in the room: its place now, in the object's frame)
            (apply(&transpose(&st.view_r()), c), l.size[0].max(l.size[1]).max(0.05) * 2.0)
        }
        None => {
            let r = (0..st.layer_count()).filter(|i| st.stack.layers[*i].show).map(|i| {
                let pl = st.stack.layers[i].place;
                (pl.pos[0] * pl.pos[0] + pl.pos[1] * pl.pos[1] + pl.pos[2] * pl.pos[2]).sqrt() + gf_core::FIT * pl.size.abs()
            }).fold(0.0f64, f64::max);
            ([0.0; 3], r.max(0.2))
        }
    };
    st.frame_on(centre, r);
    px.nav.say(match px.handles.chosen(st) {
        Some(Target::World(item)) => format!("framed {}", st.world_name(item)),
        Some(Target::Lamp(i)) => format!("framed lamp {}", i + 1),
        Some(Target::Part(id)) => format!("framed part {}", st.csg.construction_part(id).unwrap_or(0) + 1),
        Some(Target::ConstructionNode) => "framed the construction layer".into(),
        Some(Target::Whole) => "framed the csg shape".into(),
        Some(Target::Pivot(_, _)) => "framed the pivot".into(),
        None => "framed the scene".into(),
    });
}

/// The cube of the room's axes, beside the camera's bar: it says which way the camera looks, and a click on
/// one of its ends looks from that side (the one home for "look from the left").
pub fn cube(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    let at = camera_at(px, full) + egui::vec2(CAMERA_W + 10.0, 0.0);
    let shown = egui::Area::new(Id::unique("view cube")).order(egui::Order::Middle).fixed_pos(at).constrain(false).fade_in(false).show(ctx, |ui| {
        let (_, sq) = ui.allocate_space(egui::vec2(CUBE, CUBE));
        let c = sq.center();
        let rad = CUBE * 0.5 - 2.0;
        let resp = ui.interact(sq, Id::unique("view cube"), Sense::click());
        let p = ui.painter();
        p.add(glass(Rect::from_center_size(c, egui::vec2(2.0 * rad, 2.0 * rad)), rad as u8, Lift::Bar, px.bright));
        // the room's axes as the camera sees them: (across, up on the screen, towards the camera)
        let m = st.room_m();
        let reach = rad - 11.0;
        let mut ends: Vec<(usize, bool, Pos2, f64)> = Vec::new();
        for k in 0..3 {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            let v = apply(&m, e);
            for pos in [true, false] {
                let s = if pos { 1.0 } else { -1.0 };
                ends.push((k, pos, Pos2::new(c.x + (s * v[0]) as f32 * reach, c.y - (s * v[1]) as f32 * reach), s * v[2]));
            }
        }
        // the one under the pointer
        let hover = resp.hover_pos().and_then(|q| {
            ends.iter().enumerate().filter(|(_, e)| (e.2 - q).length() < 10.0).min_by(|a, b| (a.1 .2 - q).length().total_cmp(&(b.1 .2 - q).length())).map(|(i, _)| i)
        });
        // back to front
        let mut order: Vec<usize> = (0..ends.len()).collect();
        order.sort_by(|a, b| ends[*a].3.total_cmp(&ends[*b].3));
        for i in order {
            let (k, pos, q, depth) = ends[i];
            let hot = hover == Some(i);
            let dim = if depth < -0.05 { 0.55 } else { 1.0 };
            let col = fade(AXIS_INK[k], dim);
            if pos { p.line_segment([c, q], Stroke::new(1.6, col)); }
            if pos {
                p.circle(q, 7.5, if hot { pal().hot } else { col }, Stroke::new(1.0, Color32::from_black_alpha(90)));
                p.text(q, Align2::CENTER_CENTER, ["x", "y", "z"][k], mono(10.0), pal().ink);
            } else {
                p.circle(q, 5.5, fade(AXIS_INK[k], 0.22 * dim), Stroke::new(1.2, if hot { pal().hot } else { col }));
            }
        }
        let names = [("right", "left"), ("top", "under"), ("front", "back")];
        let tip = match hover {
            Some(i) => { let (k, pos, ..) = ends[i]; format!("look from the {}", if pos { names[k].0 } else { names[k].1 }) }
            None => "which way the camera looks in the room: x red, y green (up), z blue (towards the front). A click on an end looks from that side".into(),
        };
        if hover.is_some() { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
        let clicked = resp.clicked();
        resp.on_hover_text(tip);
        if clicked {
            if let Some(i) = hover {
                let (k, pos, ..) = ends[i];
                st.snap_view(k, pos);
                px.nav.say(format!("from the {}", if pos { names[k].0 } else { names[k].1 }));
            }
        }
        for (i, (k, pos, q, _)) in ends.iter().enumerate() {
            let _ = i;
            px.note(format!("cube:{}{}", if *pos { "+" } else { "-" }, ["x", "y", "z"][*k]), Rect::from_center_size(*q, egui::vec2(10.0, 10.0)));
        }
        sq
    });
    px.note("view cube", shown.response.rect);
}
