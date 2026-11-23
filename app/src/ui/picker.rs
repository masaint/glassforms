//! The shape picker. It opens as a small popup against the button that asked for it: the "+" of the objects
//! (or the "add" button when there are none, or the other tab is up), where a shape picked makes a new object
//! of the kind the tab says; or the selected object's shape card in the right panel, where the shape picked
//! takes the place of that object's shape. The popup holds a search, the families as chips, and the chosen
//! family's shapes as small pictures. Its "browse all…" opens the explorer: a window over the picture that
//! shows every shape, family by family, each family headed by a portrait of three of its shapes.
//!
//! The pictures are made by `tools/thumbs.py` and kept in `thumbs/<family>/<shape>.png` (and `_hero.png`,
//! the family's portrait) beside the program, as `models/` and `hdri/` are. They are read on a thread of
//! their own as their cards come into view; a shape with no picture has a plain card.
//!
//! Keys while either is open (the window's own keys are off): the arrows move between the cards (on past a
//! family's ends into the next), enter picks the one lit, esc closes it. Whatever is typed goes into the
//! search, which keeps the shapes whose names (or family's) hold every word.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

use eframe::egui::{self, Align, Align2, Color32, CursorIcon, Id, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, TextureId, Ui};
use gf_core::presets::Kind;
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{Panels, Request};

/// The widest and tallest the window is made, and its parts: the head, the foot, the families down its left.
const MAX_W: f32 = 1180.0;
const MAX_H: f32 = 1000.0;
const HEAD: f32 = 54.0;
const FOOT: f32 = 46.0;
const RAIL: f32 = 200.0;
/// The room round the grid, between cards, and under a card's picture for its name.
const PAD: f32 = 20.0;
const GAP: f32 = 12.0;
const NAME: f32 = 22.0;
/// The size a card's picture aims at; the cards of a line are then widened to fill it exactly.
const CARD: f32 = 128.0;
/// How long a picture takes to come up once it has been read.
const FADE: f32 = 0.18;

/// What the picker is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// a new object, of this kind (true: csg)
    Add { csg: bool },
    /// a new shape for the selected object
    Replace,
}

/// Which way an arrow key moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Left,
    Right,
    Up,
    Down,
}

/// The picker's own state, kept with the panels.
#[derive(Default)]
pub struct Picker {
    /// open, and for what
    pub open: Option<Mode>,
    /// open as the small popup (otherwise as the explorer)
    pub small: bool,
    /// the button the popup was opened from: it hangs below it
    pub anchor: Option<Rect>,
    /// the words in the search
    pub search: String,
    /// the card the keys have lit: (family, shape)
    pub at: Option<(usize, usize)>,
    /// the card under the pointer
    hover: Option<(usize, usize)>,
    /// bring the lit card into view on the next frame; centre it (on opening) rather than scroll the least
    reveal: bool,
    centre: bool,
    /// scroll to this family's heading on the next frame
    jump: Option<usize>,
    /// the family whose section is at the top of the grid (lit in the list at the left)
    in_view: Option<usize>,
    /// cards a line, as last drawn (what up and down move by)
    cols: usize,
    /// opened this frame
    fresh: bool,
    /// the last shape picked to make an object (a new add starts there)
    last: Option<(usize, usize)>,
    pub thumbs: Thumbs,
}

impl Picker {
    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }
}

/// The kind of object a shape picked now would make: the tab's (the selected object's kind, unless the
/// other tab is up).
pub fn tab_csg(st: &State, px: &Panels) -> bool {
    px.tab.unwrap_or_else(|| st.object_count() > 0 && st.is_csg_object(st.stack.sel))
}

/// Open the picker to make a new object, of the tab's kind.
pub fn open_add(st: &State, px: &mut Panels) {
    px.palette.add_to = None;
    let csg = tab_csg(st, px);
    let here = (st.object_count() > 0).then(|| (st.fam, st.shape().preset));
    let p = &mut px.picker;
    p.open = Some(Mode::Add { csg });
    p.small = true;
    p.at = p.last.or(here);
    p.reveal = p.at.is_some();
    p.centre = true;
    p.fresh = true;
    p.search.clear();
}

/// Open the picker to change the selected object's shape: it opens at that shape.
pub fn open_replace(st: &State, px: &mut Panels) {
    let p = &mut px.picker;
    p.open = Some(Mode::Replace);
    p.small = true;
    p.at = Some((st.fam, st.shape().preset));
    p.reveal = true;
    p.centre = true;
    p.fresh = true;
    p.search.clear();
}

/// From the popup to the explorer, at the same shape and with the same search.
pub fn browse(px: &mut Panels) {
    let p = &mut px.picker;
    if p.open.is_none() { return; }
    p.small = false;
    p.reveal = p.at.is_some();
    p.centre = true;
    p.fresh = true;
}

/// Where the popup hangs from when no button has said (a script): the button that would have opened it.
pub fn anchor_from(px: &mut Panels, names: &[&str]) {
    px.picker.anchor = names.iter().find_map(|n| px.rects.get(*n).copied()).or(px.picker.anchor);
}

pub fn close(px: &mut Panels) {
    px.palette.add_to = None;
    px.picker.open = None;
    px.picker.hover = None;
}

/// Pick a shape, as a click on its card does: a new object of it (of the kind the picker was opened for,
/// or the tab's), or the selected object's shape changed to it. The picker closes.
pub fn pick(st: &mut State, px: &mut Panels, fam: usize, preset: usize) {
    if fam >= st.families.len() || preset >= st.families[fam].presets.len() { return; }
    let mode = px.picker.open.unwrap_or(Mode::Add { csg: tab_csg(st, px) });
    match mode {
        Mode::Add { csg } => {
            let parent = px.palette.add_to.or(st.edit_scope.world);
            let added = st.create_world_object(fam, preset, csg, parent);
            if let Err(error) = &added { px.palette.report(error.clone()); }
            if added.is_ok() {
                px.combo_open = None;
                px.picker.last = Some((fam, preset));
            }
        }
        Mode::Replace => {
            if fam != st.fam { st.set_scene(fam); }
            st.choose(preset);
        }
    }
    // (the shape that stands for a file of the user's asks for one)
    if st.needs_file(st.stack.sel) { px.requests.push(Request::LoadModel); }
    close(px);
}

/// The shapes to show: every family that has any whose name, id or family holds every word typed, with
/// those shapes, in order.
pub fn matches(st: &State, query: &str) -> Vec<(usize, Vec<usize>)> {
    let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
    st.families
        .iter()
        .enumerate()
        .filter_map(|(f, fam)| {
            let hits: Vec<usize> = fam
                .presets
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    let hay = format!("{} {} {} {}", p.name, p.id, fam.name, fam.id).to_lowercase();
                    words.iter().all(|w| hay.contains(w.as_str()))
                })
                .map(|(i, _)| i)
                .collect();
            (!hits.is_empty()).then_some((f, hits))
        })
        .collect()
}

/// Where an arrow key takes the lit card, among the shapes shown in lines of `cols`. Left and right go
/// through every card in order; up and down keep to the column, going on into the section above or below.
/// A card that is not shown goes to the first that is.
pub fn step(shown: &[(usize, Vec<usize>)], at: Option<(usize, usize)>, cols: usize, dir: Dir) -> Option<(usize, usize)> {
    let first = shown.first().map(|(f, ps)| (*f, ps[0]));
    let Some((s, i)) = at.and_then(|(f, p)| shown.iter().enumerate().find_map(|(s, (g, ps))| (*g == f).then(|| ps.iter().position(|q| *q == p).map(|i| (s, i))).flatten())) else {
        return first;
    };
    let cols = cols.max(1);
    let len = |s: usize| shown[s].1.len();
    let (s, i) = match dir {
        Dir::Right if i + 1 < len(s) => (s, i + 1),
        Dir::Right if s + 1 < shown.len() => (s + 1, 0),
        Dir::Left if i > 0 => (s, i - 1),
        Dir::Left if s > 0 => (s - 1, len(s - 1) - 1),
        Dir::Down if i + cols < len(s) => (s, i + cols),
        // (a shorter last line below: its last card)
        Dir::Down if i / cols < (len(s) - 1) / cols => (s, len(s) - 1),
        Dir::Down if s + 1 < shown.len() => (s + 1, (i % cols).min(len(s + 1) - 1)),
        Dir::Up if i >= cols => (s, i - cols),
        Dir::Up if s > 0 => {
            let l = len(s - 1);
            (s - 1, ((l - 1) / cols * cols + i % cols).min(l - 1))
        }
        _ => (s, i),
    };
    Some((shown[s].0, shown[s].1[i]))
}

/// A few words about each family, under its name.
fn about(id: &str) -> &'static str {
    match id {
        "attractors" => "orbits of chaotic flows: a point carried along by three equations, its path drawn as a glass tube",
        "knots" => "closed curves that tie themselves, and links of several loops",
        "surfaces" => "triply periodic minimal surfaces: walls that part space into two tangled halves",
        "algebraic" => "surfaces where a polynomial is zero: cubes with tunnels, hearts, many-pointed stars",
        "gems" => "faceted stones, cut to the classic plans",
        "solids" => "the regular solids, their stars and their kin",
        "forms" => "shapes from formulas: superformulas, harmonics, metaballs, shells and a fractal",
        "studio" => "plain shapes for product shots: slabs, rings, discs, lenses",
        "trees" => "a tree grown from rules: trunk, branches and roots",
        "models" => "polygon models from files: the ones that come with the program, or one of your own",
        "fractals" => "shapes that repeat themselves at every scale: bulbs, sponges, packings of spheres",
        "topology" => "surfaces with a twist: one-sided bottles, minimal surfaces, soap films",
        "orbitals" => "where a hydrogen atom's electron is likely to be, drawn as clouds",
        "polyhedra" => "solids with flat faces: Johnson solids, compounds, zonohedra, geodesic domes",
        "links" => "rings that pass through each other and cannot be pulled apart",
        "braids" => "strands crossing over and under: plaits, braids and knotwork",
        "fieldlines" => "the paths a field draws: a magnet's lines, a vortex ring, a chaotic flow",
        "crystals" => "minerals as they grow: stepped bismuth, quartz points, a geode",
        "foam" => "soap bubbles meeting at their films",
        "branching" => "growth that splits and splits again: coral and lightning",
        "snowflakes" => "ice crystals grown in changing weather",
        "text" => "your own words in 3D, from your installed fonts",
        "paths" => "a tube, strip or outline swept along a path you build piece by piece",
        _ => "",
    }
}

/// What a family's shapes are made from, in a word.
fn made_of(kind: Kind) -> &'static str {
    match kind {
        Kind::Curve => "curves",
        Kind::Field => "equations",
        Kind::Mesh => "files",
    }
}

// ---- the pictures --------------------------------------------------------------------------------------

enum Slot {
    /// being read
    Asked,
    Ready(TextureHandle, Instant),
    /// no picture (or one that could not be read)
    Missing,
}

/// The pictures of the cards, read on a thread of their own as they are first asked for, and kept.
#[derive(Default)]
pub struct Thumbs {
    slots: HashMap<String, Slot>,
    ask: Option<Sender<(String, PathBuf)>>,
    got: Option<Receiver<(String, Option<egui::ColorImage>)>>,
}

/// Where the pictures are kept: beside the program.
pub fn folder() -> PathBuf {
    crate::hdri::home().join("thumbs")
}

/// Read a PNG of 8 bits a channel (grey, RGB or RGBA) as a picture egui can show.
fn read_png(path: &Path) -> Option<egui::ColorImage> {
    let file = std::fs::File::open(path).ok()?;
    let mut dec = png::Decoder::new(std::io::BufReader::new(file));
    dec.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let px = &buf[..info.buffer_size()];
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => px.to_vec(),
        png::ColorType::Rgb => px.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => px.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        png::ColorType::Grayscale => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return None,
    };
    (rgba.len() == w * h * 4).then(|| egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

impl Thumbs {
    /// Take in whatever the thread has read since last time.
    fn drain(&mut self, ctx: &egui::Context) {
        let Some(got) = &self.got else { return };
        while let Ok((key, image)) = got.try_recv() {
            let slot = match image {
                Some(img) => {
                    let opts = egui::TextureOptions { mipmap_mode: Some(egui::TextureFilter::Linear), ..egui::TextureOptions::LINEAR };
                    Slot::Ready(ctx.load_texture(format!("thumb {key}"), img, opts), Instant::now())
                }
                None => Slot::Missing,
            };
            self.slots.insert(key, slot);
        }
    }

    /// The picture of a shape ("knots/trefoil") or a family's portrait ("knots/_hero"), and how far it has
    /// come up (0 to 1); none while it is being read or if there is none. The first time it is asked for,
    /// it is sent to be read.
    fn get(&mut self, ctx: &egui::Context, key: &str) -> Option<(TextureId, f32)> {
        match self.slots.get(key) {
            Some(Slot::Ready(tex, at)) => {
                let up = (at.elapsed().as_secs_f32() / FADE).min(1.0);
                if up < 1.0 { ctx.request_repaint(); }
                Some((tex.id(), up))
            }
            Some(_) => None,
            None => {
                if self.ask.is_none() { self.start(ctx); }
                let path = folder().join(format!("{key}.png"));
                let sent = self.ask.as_ref().is_some_and(|a| a.send((key.to_owned(), path)).is_ok());
                self.slots.insert(key.to_owned(), if sent { Slot::Asked } else { Slot::Missing });
                None
            }
        }
    }

    /// The thread that reads the pictures.
    fn start(&mut self, ctx: &egui::Context) {
        let (ask, asked) = channel::<(String, PathBuf)>();
        let (give, got) = channel();
        let ctx = ctx.clone();
        let spawned = std::thread::Builder::new().name("shape pictures".into()).spawn(move || {
            for (key, path) in asked {
                if give.send((key, read_png(&path))).is_err() { break; }
                ctx.request_repaint();
            }
        });
        if spawned.is_ok() {
            self.ask = Some(ask);
            self.got = Some(got);
        }
    }
}

/// A shape's picture in a rectangle, coming up as it arrives; a plain card until it does (and for good if
/// there is none). `ink`: how bright it is drawn.
fn picture(ui: &Ui, thumbs: &mut Thumbs, key: &str, rect: Rect, radius: u8, ink: f32) {
    ui.painter().rect_filled(rect, radius, white(0.035));
    if !ui.is_rect_visible(rect) { return; }
    match thumbs.get(ui.ctx(), key) {
        Some((tex, up)) => {
            let a = (up * ink).clamp(0.0, 1.0);
            egui::Image::new(egui::load::SizedTexture::new(tex, rect.size()))
                .corner_radius(radius)
                .tint(Color32::from_white_alpha((a * 255.0) as u8))
                .paint_at(ui, rect);
        }
        None => {
            // (a plain card: a faint ring where the shape would be)
            ui.painter().circle_stroke(rect.center(), rect.width().min(rect.height()) * 0.18, Stroke::new(1.0, white(0.07)));
        }
    }
}

// ---- in the right panel --------------------------------------------------------------------------------

/// The selected object's shape, where the families and the list of shapes were: its picture, its name and
/// its family. A press opens the picker to change it; ← and → still step through its family.
pub fn shape_card(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let r = row(ui, 62.0);
    let resp = ui.interact(r, Id::unique("shape card"), Sense::click());
    let hot = ui.is_enabled() && resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let open = px.picker.open == Some(Mode::Replace);
    ui.painter().rect(r, 6, white(if hot { 0.08 } else { 0.04 }), Stroke::new(1.0, if open { fade(pal().accent, 0.7) } else { white(if hot { 0.22 } else { 0.1 }) }), StrokeKind::Inside);
    let img = Rect::from_min_size(r.min + egui::vec2(6.0, 6.0), egui::vec2(50.0, 50.0));
    let (fam, preset) = (st.fam, st.shape().preset);
    let (f, p) = (&st.families[fam], &st.families[fam].presets[preset]);
    let key = format!("{}/{}", f.id, p.id);
    // (the card takes in what the reading thread has made itself: with the picker shut nothing else does,
    // and the card stayed a plain ring)
    px.picker.thumbs.drain(ui.ctx());
    picture(ui, &mut px.picker.thumbs, &key, img, 4, 1.0);
    let text_l = img.right() + 11.0;
    let icon = Rect::from_center_size(Pos2::new(r.right() - 17.0, r.center().y), egui::vec2(14.0, 14.0));
    text_in(ui, Rect::from_min_max(Pos2::new(text_l, r.top() + 11.0), Pos2::new(icon.left() - 6.0, r.top() + 29.0)), Align2::LEFT_CENTER, p.name, mono(12.5), pal().bright);
    let (n, i) = (f.presets.len(), preset + 1);
    let line = Rect::from_min_max(Pos2::new(text_l, r.top() + 32.0), Pos2::new(icon.left() - 6.0, r.top() + 48.0));
    text_in(ui, line, Align2::LEFT_CENTER, &format!("{} · {i} of {n}", f.name.to_lowercase()), mono(10.0), pal().dim);
    // four small squares: a choice of pictures
    let ink = if hot || open { pal().bright } else { pal().faint };
    for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let c = icon.center() + egui::vec2(dx * 3.6, dy * 3.6);
        ui.painter().rect_stroke(Rect::from_center_size(c, egui::vec2(5.0, 5.0)), 1, Stroke::new(1.1, ink), StrokeKind::Inside);
    }
    px.note("shape card", r);
    if resp.on_hover_text("choose another shape for this object\n← → step through its family").clicked() {
        if open { close(px) } else { px.picker.anchor = Some(r); open_replace(st, px) }
    }
    gap(ui, 2.0);
    let r = row(ui, 13.0);
    text_in(ui, r, Align2::RIGHT_CENTER, "← → to step", mono(10.0), pal().hint);
    // a model is a file: any file can be loaded in place of the ones that come with the program
    if st.family().kind == Kind::Mesh {
        let (resp, at) = wide_button(ui, Id::unique("load a model"), "load a file…", false);
        px.note("model load", at);
        if resp.on_hover_text("any OBJ, STL, PLY, glTF or GLB file. Or drop one on the window: it takes this layer's place, or with shift held becomes one more layer").clicked() {
            px.requests.push(Request::LoadModel);
        }
    }
}

/// While a new object is to be made (the other tab is up, or there is none): the button that opens the
/// picker to make one.
pub fn add_button(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let csg = tab_csg(st, px);
    let full = st.object_count() >= gf_core::layers::MAX_LAYERS;
    let open = matches!(px.picker.open, Some(Mode::Add { .. }));
    ui.scope(|ui| {
        if full { ui.disable(); }
        let (resp, at) = wide_button(ui, Id::unique("picker add"), if csg { "+  add a csg object…" } else { "+  add a volume object…" }, open);
        px.note("picker:add", at);
        if resp.on_hover_text("choose its shape from pictures of every shape").clicked() {
            if open { close(px) } else { px.picker.anchor = Some(at); open_add(st, px) }
        }
    });
    if full {
        gap(ui, 3.0);
        paragraph(ui, &format!("{} objects at most: take one away to make another", gf_core::layers::MAX_LAYERS), mono(SMALL), pal().hint, LINE, 9.0);
    }
}

// ---- the window ----------------------------------------------------------------------------------------

/// Where the picker goes: over the picture between the panels, under the bars, as wide as that room
/// allows (over the panels too when the room between them is narrow).
fn place(px: &Panels, full: Rect) -> Rect {
    let lo = px.left_at.map_or(full.left() + EDGE, |r| r.right() + 12.0);
    let hi = px.right_at.map_or(full.right() - EDGE, |r| r.left() - 12.0);
    let (lo, hi) = if hi - lo < 660.0 { (full.left() + EDGE, full.right() - EDGE) } else { (lo, hi) };
    let w = (hi - lo).min(MAX_W);
    let top = super::tools::below_right(full);
    // (clear of the line of figures at the foot of the window)
    let h = (full.bottom() - EDGE - 20.0 - top).min(MAX_H).max(240.0);
    Rect::from_min_size(Pos2::new(((lo + hi - w) / 2.0).round(), top), egui::vec2(w, h))
}

/// The picker, if it is open.
pub fn show(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    let Some(mode) = px.picker.open else { return };
    px.picker.thumbs.drain(ctx);
    if px.picker.small { return popup(ctx, st, px, full, mode); }
    let fresh = std::mem::take(&mut px.picker.fresh);
    let shown = matches(st, &px.picker.search);
    // the lit card stays among those shown
    let lit_shown = px.picker.at.is_some_and(|a| shown.iter().any(|(f, ps)| *f == a.0 && ps.contains(&a.1)));
    if !lit_shown { px.picker.at = shown.first().map(|(f, ps)| (*f, ps[0])); }

    // the keys: taken before anything else sees them (the search keeps only what is typed)
    let (mut chosen, mut shut) = (None, false);
    let keys = ctx.input_mut(|i| {
        let mut moves = Vec::new();
        for (key, dir) in [(egui::Key::ArrowLeft, Dir::Left), (egui::Key::ArrowRight, Dir::Right), (egui::Key::ArrowUp, Dir::Up), (egui::Key::ArrowDown, Dir::Down)] {
            while i.consume_key(egui::Modifiers::NONE, key) { moves.push(dir); }
        }
        (moves, i.consume_key(egui::Modifiers::NONE, egui::Key::Enter), i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
    });
    for dir in keys.0 {
        px.picker.at = step(&shown, px.picker.at, px.picker.cols, dir);
        px.picker.reveal = true;
    }
    if keys.1 { chosen = px.picker.at; }
    if keys.2 { shut = true; }

    let rect = place(px, full);
    // a press anywhere outside the window closes it
    if !fresh && ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| !rect.contains(p))) { shut = true; }

    egui::Area::new(Id::unique("shape picker")).order(egui::Order::Foreground).fixed_pos(rect.min).constrain(false).fade_in(false).show(ctx, |ui| {
        glass_frame(ui, egui::Margin::ZERO, PANEL_RADIUS, Lift::Float, false, |ui| {
            ui.set_min_size(rect.size());
            ui.set_max_size(rect.size());
            if px.rendering { ui.disable(); }
            let r = Rect::from_min_size(ui.cursor().min, rect.size());
            let head = Rect::from_min_max(r.min, Pos2::new(r.right(), r.top() + HEAD));
            let foot = Rect::from_min_max(Pos2::new(r.left(), r.bottom() - FOOT), r.max);
            let rail = Rect::from_min_max(Pos2::new(r.left(), head.bottom()), Pos2::new(r.left() + RAIL, foot.top()));
            let grid = Rect::from_min_max(Pos2::new(rail.right(), head.bottom()), Pos2::new(r.right(), foot.top()));
            let rule = Stroke::new(1.0, white(0.08));
            ui.painter().hline(r.x_range(), head.bottom(), rule);
            ui.painter().hline(r.x_range(), foot.top(), rule);
            ui.painter().vline(rail.right(), rail.y_range(), rule);

            if heading(ui, st, px, head, mode, fresh) { shut = true; }
            families(ui, st, px, rail, &shown);
            ui.scope_builder(egui::UiBuilder::new().max_rect(grid), |ui| {
                if let Some(p) = cards(ui, st, px, grid, &shown, mode, fresh) { chosen = Some(p); }
            });
            footing(ui, st, px, foot, mode);
        });
    });
    px.note("picker", rect);
    if let Some((f, p)) = chosen {
        pick(st, px, f, p);
    } else if shut {
        close(px);
    }
}

/// The head of the window: what it is for, the search, the cross. True if the cross was pressed.
fn heading(ui: &mut Ui, st: &State, px: &mut Panels, head: Rect, mode: Mode, fresh: bool) -> bool {
    let x = head.left() + PAD;
    let title = match mode { Mode::Add { .. } => "add a shape", Mode::Replace => "change the shape" };
    let g = spaced(ui, title, 12.0, pal().bright);
    ui.painter().galley(Pos2::new(x, head.top() + 12.0), g, pal().bright);
    let says = match mode {
        Mode::Add { csg: true } => "it is made a csg object: its formula traced exactly".to_owned(),
        Mode::Add { csg: false } if st.object_count() == 0 => "it is made a volume object: the shape baked into a grid of distances".to_owned(),
        Mode::Add { csg: false } => "it is made a volume object, joined to the others".to_owned(),
        Mode::Replace => format!("of object {}: {}, in its place", st.stack.sel + 1, st.preset().name),
    };
    text_in(ui, Rect::from_min_max(Pos2::new(x, head.top() + 31.0), Pos2::new(head.center().x, head.top() + 45.0)), Align2::LEFT_CENTER, &says, mono(10.0), pal().faint);

    // the cross, and the search before it
    let cross = Rect::from_center_size(Pos2::new(head.right() - 22.0, head.center().y), egui::vec2(22.0, 22.0));
    let closed = close_button(ui, cross, Id::unique("picker close")).on_hover_text("close · esc").clicked();
    px.note("picker:close", cross);
    let field = Rect::from_min_max(Pos2::new((cross.left() - 14.0 - 300.0).max(head.center().x), head.center().y - 13.0), Pos2::new(cross.left() - 14.0, head.center().y + 13.0));
    let id = Id::unique("picker search");
    let edit = egui::TextEdit::singleline(&mut px.picker.search).id(id).hint_text("search shapes").font(mono(11.5)).text_color(pal().bright).margin(egui::Margin::symmetric(9, 5)).desired_width(field.width());
    let resp = ui.put(field, edit);
    px.note("picker:search", field);
    // (whatever is typed goes to the search, unless something else has the keyboard)
    if fresh || ui.memory(|m| m.focused().is_none()) { resp.request_focus(); }
    if resp.changed() {
        px.picker.reveal = false;
        px.picker.jump = None;
    }
    closed
}

/// The families down the left: each with how many of its shapes are shown; the one whose section holds the
/// upper part of the grid lit. A press scrolls the grid to it.
fn families(ui: &mut Ui, st: &State, px: &mut Panels, rail: Rect, shown: &[(usize, Vec<usize>)]) {
    let mut y = rail.top() + 14.0;
    for (f, ps) in shown {
        let fam = &st.families[*f];
        let r = Rect::from_min_size(Pos2::new(rail.left() + 10.0, y), egui::vec2(rail.width() - 20.0, 27.0));
        y += 29.0;
        if r.bottom() > rail.bottom() { break; }
        let resp = ui.interact(r, Id::unique(("picker family", *f)), Sense::click());
        let hot = resp.hovered();
        if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
        let lit = px.picker.in_view == Some(*f);
        if lit || hot { ui.painter().rect_filled(r, 5, white(if lit { 0.08 } else { 0.05 })); }
        if lit { ui.painter().rect_filled(Rect::from_min_max(Pos2::new(r.left(), r.top() + 7.0), Pos2::new(r.left() + 2.0, r.bottom() - 7.0)), 1, pal().accent); }
        let name = Rect::from_min_max(Pos2::new(r.left() + 10.0, r.top()), Pos2::new(r.right() - 28.0, r.bottom()));
        text_in(ui, name, Align2::LEFT_CENTER, &fam.name.to_lowercase(), mono(11.0), if lit || hot { pal().bright } else { pal().button });
        text_in(ui, r.shrink2(egui::vec2(8.0, 0.0)), Align2::RIGHT_CENTER, &ps.len().to_string(), mono(10.0), pal().hint);
        px.note(format!("picker:family:{}", fam.id), r);
        if resp.clicked() {
            px.picker.jump = Some(*f);
            px.picker.at = Some((*f, ps[0]));
            px.picker.reveal = false;
        }
    }
}

/// The grid: a section a family, its portrait and words at its head (only its name while searching), then
/// its cards. Returns the shape clicked.
#[allow(clippy::too_many_arguments)]
fn cards(ui: &mut Ui, st: &State, px: &mut Panels, grid: Rect, shown: &[(usize, Vec<usize>)], mode: Mode, fresh: bool) -> Option<(usize, usize)> {
    let mut clicked = None;
    let searching = !px.picker.search.trim().is_empty();
    let width = grid.width() - 2.0 * PAD + 6.0;
    let cols = (((width + GAP) / (CARD + GAP)).floor() as usize).max(2);
    let card = ((width - GAP * (cols - 1) as f32) / cols as f32).floor();
    px.picker.cols = cols;
    let now = (st.fam, st.shape().preset);
    let mut area = egui::ScrollArea::vertical().id_salt("picker grid").auto_shrink([false, false]).max_height(grid.height());
    if fresh && !px.picker.reveal { area = area.vertical_scroll_offset(0.0); }
    let mut hover = None;
    // the card under the pointer is drawn again over its neighbours, larger
    let mut lifted: Option<(Rect, String, bool)> = None;
    let mut tops: Vec<(usize, f32, f32)> = Vec::new();
    area.show(ui, |ui| {
        ui.set_width(grid.width());
        let clip = ui.clip_rect();
        let left = ui.cursor().left() + PAD;
        gap(ui, PAD - 4.0);
        for (f, ps) in shown {
            let fam = &st.families[*f];
            let top = ui.cursor().top();
            // the head of the section
            let head = if searching {
                let r = row(ui, 22.0);
                let g = spaced(ui, fam.name, 10.5, pal().faint);
                ui.painter().galley(Pos2::new(left, r.center().y - g.size().y / 2.0), g, pal().faint);
                r
            } else {
                let hero_w = (width * 0.5).clamp(300.0, 520.0).floor();
                let r = row(ui, (hero_w / 2.0).floor());
                let hero = Rect::from_min_size(Pos2::new(left, r.top()), egui::vec2(hero_w, r.height()));
                picture(ui, &mut px.picker.thumbs, &format!("{}/_hero", fam.id), hero, 8, 1.0);
                ui.painter().rect_stroke(hero, 8, Stroke::new(1.0, white(0.07)), StrokeKind::Inside);
                let tx = hero.right() + 26.0;
                let tw = (left + width - tx).max(60.0);
                let g = spaced(ui, fam.name, 13.0, pal().bright);
                let mut y = hero.top() + 8.0;
                ui.painter().galley(Pos2::new(tx, y), g.clone(), pal().bright);
                y += g.size().y + 10.0;
                let words = wrapped(ui, about(fam.id), sans(12.5), pal().text, tw, 1.45);
                ui.painter().galley(Pos2::new(tx, y), words.clone(), pal().text);
                y += words.size().y + 10.0;
                text_in(ui, Rect::from_min_size(Pos2::new(tx, y), egui::vec2(tw, 14.0)), Align2::LEFT_CENTER, &format!("{} shapes · from {}", fam.presets.len(), made_of(fam.kind)), mono(10.0), pal().faint);
                r
            };
            if px.picker.jump == Some(*f) { ui.scroll_to_rect(head.expand2(egui::vec2(0.0, PAD - 4.0)), Some(Align::TOP)); }
            gap(ui, 14.0);
            for line in ps.chunks(cols) {
                let r = row(ui, card + NAME);
                for (k, &p) in line.iter().enumerate() {
                    let slot = Rect::from_min_size(Pos2::new(left + k as f32 * (card + GAP), r.top()), egui::vec2(card, card + NAME));
                    let img = Rect::from_min_size(slot.min, egui::vec2(card, card));
                    let preset = &fam.presets[p];
                    let id = Id::unique(("pick", *f, p));
                    let resp = ui.interact(slot, id, Sense::click());
                    let hot = resp.hovered();
                    let lit = px.picker.at == Some((*f, p));
                    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); hover = Some((*f, p)); }
                    if lit && px.picker.reveal { ui.scroll_to_rect(slot.expand(GAP), px.picker.centre.then_some(Align::Center)); }
                    let key = format!("{}/{}", fam.id, preset.id);
                    let mine = mode == Mode::Replace && now == (*f, p);
                    let lift = ui.ctx().animate_bool_with_time(id.with("lift"), hot, 0.12);
                    if lift > 0.0 {
                        lifted = Some((img, key.clone(), lit || mine));
                    } else {
                        picture(ui, &mut px.picker.thumbs, &key, img, 6, 0.92);
                        let line = if lit { Stroke::new(1.5, pal().accent) } else { Stroke::new(1.0, white(0.07)) };
                        ui.painter().rect_stroke(img, 6, line, StrokeKind::Inside);
                    }
                    if mine { ui.painter().circle_filled(img.right_top() + egui::vec2(-10.0, 10.0), 3.5, pal().accent); }
                    let name = Rect::from_min_max(Pos2::new(slot.left() + 2.0, img.bottom() + 3.0), Pos2::new(slot.right() - 2.0, slot.bottom()));
                    text_in(ui, name, Align2::CENTER_CENTER, preset.name, mono(10.5), if hot || lit { pal().bright } else if mine { pal().accent } else { pal().button });
                    if clip.contains_rect(slot) { px.note(format!("pick:{}/{}", fam.id, preset.id), slot); }
                    if resp.clicked() { clicked = Some((*f, p)); }
                }
                gap(ui, GAP);
            }
            tops.push((*f, top, ui.cursor().top()));
            gap(ui, 26.0);
        }
        if shown.is_empty() {
            gap(ui, 30.0);
            paragraph(ui, &format!("no shape is called anything like \"{}\"", px.picker.search.trim()), mono(SIZE), pal().dim, LINE, PAD);
        }
        gap(ui, PAD);
        // the card under the pointer, larger, over the others
        if let Some((img, key, lit)) = &lifted {
            let t = hover.map_or(0.0, |(f, p)| ui.ctx().animate_bool_with_time(Id::unique(("pick", f, p)).with("lift"), true, 0.12));
            let big = Rect::from_center_size(img.center(), img.size() * (1.0 + 0.07 * t));
            ui.painter().rect_filled(big.expand(1.0).translate(egui::vec2(0.0, 3.0 * t)), 8, Color32::from_black_alpha((120.0 * t) as u8));
            picture(ui, &mut px.picker.thumbs, key, big, 7, 0.92 + 0.08 * t);
            ui.painter().rect_stroke(big, 7, Stroke::new(if *lit { 1.5 } else { 1.0 }, if *lit { pal().accent } else { white(0.32) }), StrokeKind::Inside);
        }
        // the family whose section holds the upper third of the grid
        let mark = clip.top() + clip.height() / 3.0;
        px.picker.in_view = tops.iter().rev().find(|(_, top, _)| *top <= mark).or(tops.first()).map(|(f, _, _)| *f);
    });
    px.picker.hover = hover;
    px.picker.jump = None;
    px.picker.reveal = false;
    px.picker.centre = false;
    clicked
}

/// The foot of the window: the shape under the pointer (or the one lit), its family and its mathematics;
/// at the right, the keys.
fn footing(ui: &mut Ui, st: &State, px: &mut Panels, foot: Rect, mode: Mode) {
    let keys = match mode { Mode::Add { .. } => "arrows move · enter adds · esc closes", Mode::Replace => "arrows move · enter puts it in · esc closes" };
    let kw = text_width(ui, keys, &mono(10.0));
    text_in(ui, Rect::from_min_max(Pos2::new(foot.right() - PAD - kw - 4.0, foot.top()), Pos2::new(foot.right() - PAD, foot.bottom())), Align2::RIGHT_CENTER, keys, mono(10.0), pal().hint);
    let Some((f, p)) = px.picker.hover.or(px.picker.at) else { return };
    let (fam, preset) = (&st.families[f], &st.families[f].presets[p]);
    let x = foot.left() + PAD;
    let room = Rect::from_min_max(Pos2::new(x, foot.top()), Pos2::new(foot.right() - PAD - kw - 30.0, foot.bottom()));
    let name_w = text_width(ui, preset.name, &mono(12.0));
    text_in(ui, Rect::from_min_max(room.min, Pos2::new(room.right(), room.center().y + 1.0)).translate(egui::vec2(0.0, 3.0)), Align2::LEFT_CENTER, preset.name, mono(12.0), pal().bright);
    text_in(ui, Rect::from_min_max(Pos2::new(x + name_w + 12.0, room.top() + 3.0), Pos2::new(room.right(), room.center().y + 4.0)), Align2::LEFT_CENTER, &fam.name.to_lowercase(), mono(10.0), pal().dim);
    // its mathematics, as far as a line holds; otherwise its note
    let say = if preset.eq.is_empty() { preset.note.to_owned() } else { preset.eq.join("   ") };
    text_in(ui, Rect::from_min_max(Pos2::new(x, room.center().y + 2.0), Pos2::new(room.right(), room.bottom() - 4.0)), Align2::LEFT_CENTER, &say, mono(10.0), pal().faint);
}

// ---- the popup -----------------------------------------------------------------------------------------

/// The popup's pictures, the gap between them, how many a line, and how many lines show before it scrolls.
const POP_THUMB: f32 = 64.0;
const POP_GAP: f32 = 6.0;
const POP_COLS: usize = 5;
const POP_LINES: f32 = 3.5;
/// The width of its pictures' lines, and of everything in it (the lines and, at their right, the room for
/// the scroll bar); the room round its contents.
const POP_W: f32 = POP_COLS as f32 * POP_THUMB + (POP_COLS - 1) as f32 * POP_GAP;
const POP_IN: f32 = POP_W + 10.0;
const POP_PAD: i8 = 12;

/// Where the popup goes: hanging below the button that opened it, its right edge with the button's (or its
/// left edge, for a button on the left of the window), kept inside the window.
fn pop_place(px: &Panels, full: Rect, size: egui::Vec2) -> Pos2 {
    let a = px.picker.anchor.or(px.right_at.map(|r| Rect::from_min_size(r.min + egui::vec2(10.0, 10.0), egui::vec2(r.width() - 20.0, 24.0)))).unwrap_or(Rect::from_min_size(full.center(), egui::Vec2::ZERO));
    let x = if a.center().x > full.center().x { a.right() - size.x } else { a.left() };
    let mut y = a.bottom() + 6.0;
    // (no room below: above it, or as low as it goes)
    if y + size.y > full.bottom() - EDGE { y = if a.top() - 6.0 - size.y > full.top() + EDGE { a.top() - 6.0 - size.y } else { full.bottom() - EDGE - size.y }; }
    Pos2::new(x.clamp(full.left() + EDGE, (full.right() - EDGE - size.x).max(full.left() + EDGE)), y.max(full.top() + EDGE)).round()
}

/// The small popup: the search, the families as chips, the chosen family's shapes (every match while
/// searching), the name of the one under the pointer, and "browse all…".
fn popup(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect, mode: Mode) {
    let fresh = std::mem::take(&mut px.picker.fresh);
    let searching = !px.picker.search.trim().is_empty();
    // the keys move through every family's shapes (or the matches), in order
    let all = matches(st, &px.picker.search);
    let lit_shown = px.picker.at.is_some_and(|a| all.iter().any(|(f, ps)| *f == a.0 && ps.contains(&a.1)));
    if !lit_shown { px.picker.at = all.first().map(|(f, ps)| (*f, ps[0])); }
    let (mut chosen, mut shut, mut more) = (None, false, false);
    let keys = ctx.input_mut(|i| {
        let mut moves = Vec::new();
        for (key, dir) in [(egui::Key::ArrowLeft, Dir::Left), (egui::Key::ArrowRight, Dir::Right), (egui::Key::ArrowUp, Dir::Up), (egui::Key::ArrowDown, Dir::Down)] {
            while i.consume_key(egui::Modifiers::NONE, key) { moves.push(dir); }
        }
        (moves, i.consume_key(egui::Modifiers::NONE, egui::Key::Enter), i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
    });
    for dir in keys.0 {
        px.picker.at = step(&all, px.picker.at, POP_COLS, dir);
        px.picker.reveal = true;
    }
    if keys.1 { chosen = px.picker.at; }
    if keys.2 { shut = true; }
    // what the grid shows: the lit card's family, or every match
    let fam = px.picker.at.map_or(0, |a| a.0);
    let shown: Vec<(usize, Vec<usize>)> = if searching { all.clone() } else { vec![(fam, (0..st.families[fam].presets.len()).collect())] };

    let id = Id::unique("shape popup");
    // (its size as last drawn, for placing it; the first frame guesses)
    let size: egui::Vec2 = ctx.data(|d| d.get_temp(id.with("size"))).unwrap_or(egui::vec2(POP_IN + 2.0 * POP_PAD as f32 - 4.0, 420.0));
    let at = pop_place(px, full, size);
    let rect = Rect::from_min_size(at, size);
    if !fresh && ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| !rect.contains(p) && !px.picker.anchor.is_some_and(|a| a.contains(p)))) { shut = true; }

    let mut hover = None;
    let shown_area = egui::Area::new(id).order(egui::Order::Foreground).fixed_pos(at).constrain(false).fade_in(false).show(ctx, |ui| {
        glass_frame(ui, egui::Margin { left: POP_PAD, right: POP_PAD - 4, top: POP_PAD, bottom: POP_PAD }, PANEL_RADIUS, Lift::Float, false, |ui| {
            ui.set_width(POP_IN);
            if px.rendering { ui.disable(); }
            // the search, with what the shape will do beside its hint
            let field = row(ui, 26.0);
            let sid = Id::unique("popup search");
            let hint = match mode { Mode::Add { csg: true } => "search · makes a csg object", Mode::Add { csg: false } => "search · makes a volume object", Mode::Replace => "search · replaces this shape" };
            let edit = egui::TextEdit::singleline(&mut px.picker.search).id(sid).hint_text(hint).font(mono(11.0)).text_color(pal().bright).margin(egui::Margin::symmetric(8, 5)).desired_width(field.width());
            let resp = ui.put(field, edit);
            px.note("picker:search", field);
            if fresh || ui.memory(|m| m.focused().is_none()) { resp.request_focus(); }
            gap(ui, 8.0);

            // the families, as chips (hidden while searching: the matches come from all of them)
            if !searching {
                let font = mono(10.0);
                let (pad, h) = (egui::vec2(6.0, 0.0), 19.0);
                let (left, right) = (ui.cursor().left(), ui.cursor().left() + POP_IN);
                let mut places = Vec::with_capacity(st.families.len());
                let (mut x, mut y) = (left, 0.0f32);
                for f in &st.families {
                    let w = (text_width(ui, &f.name.to_lowercase(), &font) + 2.0 * pad.x).ceil();
                    if x + w > right + 0.5 && x > left { x = left; y += h + 3.0; }
                    places.push(Rect::from_min_size(Pos2::new(x, y), egui::vec2(w, h)));
                    x += w + 3.0;
                }
                let block = row(ui, y + h);
                for (i, (f, p)) in st.families.iter().zip(&places).enumerate() {
                    let r = p.translate(egui::vec2(0.0, block.top()));
                    let resp = ui.interact(r, Id::unique(("popup family", i)), Sense::click());
                    let hot = resp.hovered();
                    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
                    let lit = i == fam;
                    if lit { ui.painter().rect_filled(r, 4, pal().lit); } else if hot { ui.painter().rect_filled(r, 4, white(0.08)); }
                    text_in(ui, r, Align2::CENTER_CENTER, &f.name.to_lowercase(), font.clone(), if lit { pal().ink } else if hot { pal().hot } else { pal().button });
                    px.note(format!("picker:family:{}", f.id), r);
                    if resp.clicked() && !lit {
                        px.picker.at = Some((i, 0));
                        px.picker.reveal = false;
                        ui.ctx().data_mut(|d| d.insert_temp(Id::unique("popup top"), true));
                    }
                }
                gap(ui, 8.0);
                ui.painter().hline(egui::Rangef::new(left, right), ui.cursor().top() - 4.0, Stroke::new(1.0, white(0.08)));
            }

            // the shapes
            let line = POP_THUMB + POP_GAP;
            let mut area = egui::ScrollArea::vertical().id_salt("popup grid").max_height(POP_LINES * line).auto_shrink([false, true]);
            if ui.ctx().data_mut(|d| d.remove_temp::<bool>(Id::unique("popup top"))).is_some() || (fresh && !px.picker.reveal) { area = area.vertical_scroll_offset(0.0); }
            let now = (st.fam, st.shape().preset);
            area.show(ui, |ui| {
                ui.set_width(POP_IN);
                let clip = ui.clip_rect();
                for (f, ps) in &shown {
                    let family = &st.families[*f];
                    if searching {
                        let r = row(ui, 16.0);
                        text_in(ui, r, Align2::LEFT_BOTTOM, &family.name.to_lowercase(), mono(9.5), pal().faint);
                        gap(ui, 3.0);
                    }
                    for chunk in ps.chunks(POP_COLS) {
                        let r = row(ui, POP_THUMB);
                        for (k, &p) in chunk.iter().enumerate() {
                            let img = Rect::from_min_size(Pos2::new(r.left() + k as f32 * line, r.top()), egui::vec2(POP_THUMB, POP_THUMB));
                            let preset = &family.presets[p];
                            let resp = ui.interact(img, Id::unique(("popup pick", *f, p)), Sense::click());
                            let hot = resp.hovered();
                            let lit = px.picker.at == Some((*f, p));
                            let mine = mode == Mode::Replace && now == (*f, p);
                            if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); hover = Some((*f, p)); }
                            if lit && px.picker.reveal { ui.scroll_to_rect(img.expand(POP_GAP), px.picker.centre.then_some(Align::Center)); }
                            picture(ui, &mut px.picker.thumbs, &format!("{}/{}", family.id, preset.id), img, 5, if hot || lit { 1.0 } else { 0.9 });
                            let edge = if lit { Stroke::new(1.5, pal().accent) } else if hot { Stroke::new(1.0, white(0.4)) } else { Stroke::new(1.0, white(0.07)) };
                            ui.painter().rect_stroke(img, 5, edge, StrokeKind::Inside);
                            if mine { ui.painter().circle_filled(img.right_top() + egui::vec2(-7.0, 7.0), 3.0, pal().accent); }
                            if clip.contains_rect(img) { px.note(format!("pick:{}/{}", family.id, preset.id), img); }
                            if resp.on_hover_text(preset.name).clicked() { chosen = Some((*f, p)); }
                        }
                        gap(ui, POP_GAP);
                    }
                }
                if shown.is_empty() {
                    gap(ui, 10.0);
                    paragraph(ui, &format!("no shape is called anything like \"{}\"", px.picker.search.trim()), mono(SMALL), pal().dim, LINE, 0.0);
                    gap(ui, 10.0);
                }
            });
            px.picker.reveal = false;
            px.picker.centre = false;

            // the shape under the pointer (or the one lit), then "browse all…"
            gap(ui, 4.0);
            let r = row(ui, 18.0);
            if let Some((f, p)) = hover.or(px.picker.at) {
                let (family, preset) = (&st.families[f], &st.families[f].presets[p]);
                let w = text_width(ui, preset.name, &mono(11.5));
                text_in(ui, r, Align2::LEFT_CENTER, preset.name, mono(11.5), pal().bright);
                let rest = Rect::from_min_max(Pos2::new(r.left() + w + 10.0, r.top()), r.max);
                let (i, n) = (p + 1, family.presets.len());
                text_in(ui, rest, Align2::LEFT_CENTER, &format!("{} · {i} of {n}", family.name.to_lowercase()), mono(10.0), pal().dim);
            }
            gap(ui, 6.0);
            let foot = row(ui, 24.0);
            let label = "browse all…";
            let bw = text_width(ui, label, &mono(10.5)) + 22.0;
            let b = Rect::from_min_size(Pos2::new(foot.right() - bw, foot.top()), egui::vec2(bw, foot.height()));
            px.note("picker:browse", b);
            if button(ui, b, Id::unique("popup browse"), label, false, mono(10.5)).on_hover_text("every shape, family by family, with pictures of each family and the mathematics").clicked() { more = true; }
            let keys = match mode { Mode::Add { .. } => "enter adds · esc closes", Mode::Replace => "enter puts it in · esc closes" };
            text_in(ui, Rect::from_min_max(foot.min, Pos2::new(b.left() - 8.0, foot.bottom())), Align2::LEFT_CENTER, keys, mono(9.5), pal().hint);
        });
    });
    let drawn = shown_area.response.rect;
    if (drawn.size() - size).length() > 0.5 {
        ctx.data_mut(|d| d.insert_temp(id.with("size"), drawn.size()));
        ctx.request_repaint();
    }
    px.note("picker", drawn);
    px.picker.hover = hover;
    if let Some((f, p)) = chosen {
        pick(st, px, f, p);
    } else if more {
        browse(px);
    } else if shut {
        close(px);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_search_keeps_shapes_whose_names_hold_every_word() {
        let st = State::new();
        let all = matches(&st, "");
        assert_eq!(all.len(), st.families.len());
        assert_eq!(all.iter().map(|(_, ps)| ps.len()).sum::<usize>(), st.families.iter().map(|f| f.presets.len()).sum::<usize>());
        let knots = st.families.iter().position(|f| f.id == "knots").unwrap();
        let tre = st.families[knots].presets.iter().position(|p| p.id == "trefoil").unwrap();
        let found = matches(&st, "  TREFOIL ");
        assert!(found.iter().any(|(f, ps)| *f == knots && ps.contains(&tre)), "{found:?}");
        // a family's name finds all of its shapes
        let gems = st.families.iter().position(|f| f.id == "gems").unwrap();
        let g = matches(&st, "gem");
        assert!(g.iter().any(|(f, ps)| *f == gems && ps.len() == st.families[gems].presets.len()));
        // every word must be found
        assert!(matches(&st, "trefoil zzzz").is_empty());
    }

    #[test]
    fn the_arrows_move_along_lines_and_between_sections() {
        // two sections: one of 5 shapes (lines of 3: 0 1 2 / 3 4), one of 2
        let shown = vec![(0, vec![0, 1, 2, 3, 4]), (1, vec![0, 1])];
        let go = |at: (usize, usize), d: Dir| step(&shown, Some(at), 3, d).unwrap();
        assert_eq!(go((0, 0), Dir::Right), (0, 1));
        assert_eq!(go((0, 4), Dir::Right), (1, 0));
        assert_eq!(go((1, 0), Dir::Left), (0, 4));
        assert_eq!(go((0, 0), Dir::Left), (0, 0));
        assert_eq!(go((1, 1), Dir::Right), (1, 1));
        assert_eq!(go((0, 1), Dir::Down), (0, 4));
        // from the end of a full line over a shorter one: to the shorter one's last
        assert_eq!(go((0, 2), Dir::Down), (0, 4));
        assert_eq!(go((0, 4), Dir::Down), (1, 1));
        assert_eq!(go((0, 3), Dir::Down), (1, 0));
        assert_eq!(go((1, 1), Dir::Up), (0, 4));
        assert_eq!(go((1, 0), Dir::Up), (0, 3));
        assert_eq!(go((0, 4), Dir::Up), (0, 1));
        assert_eq!(go((0, 1), Dir::Up), (0, 1));
        // a card not shown: the first that is
        assert_eq!(step(&shown, Some((7, 7)), 3, Dir::Down), Some((0, 0)));
        assert_eq!(step(&shown, None, 3, Dir::Right), Some((0, 0)));
        assert_eq!(step(&[], None, 3, Dir::Right), None);
    }

    #[test]
    fn the_popup_hangs_below_its_button_inside_the_window() {
        let mut px = Panels::new(super::super::store::Store::default());
        let full = Rect::from_min_size(Pos2::ZERO, egui::vec2(1600.0, 900.0));
        let size = egui::vec2(368.0, 420.0);
        // a button at the right: the popup's right edge with the button's, just below it
        px.picker.anchor = Some(Rect::from_min_max(Pos2::new(1500.0, 50.0), Pos2::new(1560.0, 74.0)));
        let at = pop_place(&px, full, size);
        assert_eq!((at.x + size.x, at.y), (1560.0, 80.0));
        // a button near the foot: above it
        px.picker.anchor = Some(Rect::from_min_max(Pos2::new(1500.0, 800.0), Pos2::new(1560.0, 824.0)));
        assert_eq!(pop_place(&px, full, size).y, 800.0 - 6.0 - 420.0);
        // a button at the left: its left edge with the button's, kept inside
        px.picker.anchor = Some(Rect::from_min_max(Pos2::new(2.0, 50.0), Pos2::new(40.0, 74.0)));
        assert_eq!(pop_place(&px, full, size).x, EDGE);
    }

    #[test]
    fn every_family_has_a_few_words() {
        let st = State::new();
        for f in &st.families { assert!(!about(f.id).is_empty(), "{} has no words", f.id); }
    }

    #[test]
    fn a_shape_picked_is_made_of_the_tab_s_kind_or_takes_the_selected_one_s_place() {
        let mut st = State::new();
        let mut px = Panels::new(super::super::store::Store::default());
        let knots = st.families.iter().position(|f| f.id == "knots").unwrap();
        let before = st.object_count();
        // the csg tab up: a new csg object
        px.tab = Some(true);
        open_add(&st, &mut px);
        assert_eq!(px.picker.open, Some(Mode::Add { csg: true }));
        assert!(px.picker.small, "the \"+\" opens the small popup");
        browse(&mut px);
        assert!(!px.picker.small && px.picker.open == Some(Mode::Add { csg: true }), "browse all opens the explorer for the same thing");
        pick(&mut st, &mut px, knots, 1);
        assert_eq!(st.object_count(), before + 1);
        assert!(st.is_csg_object(st.stack.sel) && st.fam == knots && st.shape().preset == 1);
        assert!(px.picker.open.is_none());
        // the selected object's shape changed in its place
        let gems = st.families.iter().position(|f| f.id == "gems").unwrap();
        open_replace(&st, &mut px);
        assert_eq!(px.picker.at, Some((knots, 1)));
        pick(&mut st, &mut px, gems, 2);
        assert_eq!(st.object_count(), before + 1);
        assert!(st.fam == gems && st.shape().preset == 2 && st.is_csg_object(st.stack.sel));
    }
}
