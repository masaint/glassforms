//! The quality-tier row: a small glass bar under the camera's tools with a round button for each way the
//! picture can be drawn (matcap, draft, live glass, full render), the lit one the tier at rest, and a small
//! chevron that opens that tier's settings (plans/app_integration_plan.md, section 2).
//!
//! This is the layout and the look. What the buttons choose is the policy of `crate::tier` (the choice is
//! kept in the window's own file, `store.tiers`); nothing here is read by the engine yet, so the only
//! picture a click changes is the matcap one, which was always the view mode's to change. The glyphs are
//! drawn with the painter, like the tool buttons' drawings; no image is used.
//!
//! Every element notes its place under the name a check knows it by: `tier:matcap`, `tier:draft`,
//! `tier:live`, `tier:full`, `tier:menu`, and in the dropdown `tier/auto:off|on`, `tier/motion:<id>`.

use eframe::egui::{self, Align2, Color32, Id, Painter, Pos2, Rect, Sense, Shape, Stroke, Ui};
use gf_core::state::State;
use gf_core::view_mode::ViewMode;

use super::theme::*;
use super::widgets::*;
use super::Panels;
use crate::tier::{Avail, DraftFacts, Motion, Shown, Tier, Times, TIERS};

const B: f32 = 26.0;
const GAP: f32 = 3.0;
const SPLIT: f32 = 9.0;
const PAD: f32 = 5.0;
/// The chevron's hit area is narrower than a button.
const CHEV_W: f32 = 16.0;
/// The round buttons' diameter (the lit disc is a little smaller, so that its ring has a gap round it).
const DISC: f32 = 24.0;

/// The tier choices that live inside a tier's dropdown (stubs until the tiers are in the engine: only
/// kept here and shown).
#[derive(Clone, Debug, PartialEq)]
pub struct Cfg {
    /// draft: bounces 1 to 3, and the share of the pixels traced (0: a quarter, 1: a ninth)
    pub draft_bounces: u8,
    pub draft_pixels: usize,
    /// live glass: thin detail (0 off, 1 refine, 2 fitted box), the refine budget in MB, the volume 512 | 768 | 1024
    pub thin: usize,
    pub refine_mb: u32,
    pub volume: usize,
}
impl Default for Cfg {
    fn default() -> Self {
        Cfg { draft_bounces: 2, draft_pixels: 0, thin: 1, refine_mb: 250, volume: 0 }
    }
}

/// Where the row stands: directly under the camera bar, at its left edge.
pub fn at(camera: Pos2) -> Pos2 {
    Pos2::new(camera.x, camera.y + super::tools::BAR_H + GAP)
}

/// The tier that is lit: the matcap while the view is a matcap, else the glass tier chosen for rest.
pub fn lit(st: &State, px: &Panels) -> Tier {
    if st.view.mode == ViewMode::Matcap { return Tier::Matcap; }
    // (a stored matcap, from `tier=matcap`, is lit as the matcap: `tier_step` draws it so)
    px.store.tiers.rest
}

/// What this machine and this scene allow. The scene's part (does the parts engine accept it) is looked at
/// again every few frames, not every frame.
pub fn avail(st: &State, px: &mut Panels, frame: u64) -> Avail {
    let mut a = avail_machine(st, px, frame);
    a.draft_skip = px.draft_skip.clone();
    a
}

/// The machine's and the scene's part, cached.
fn avail_machine(st: &State, px: &mut Panels, frame: u64) -> Avail {
    if let Some((at, a)) = &px.tier_cache {
        if frame.wrapping_sub(*at) < 10 { return a.clone(); }
    }
    let mut facts = DraftFacts { built: DraftFacts::built_in(), ..Default::default() };
    #[cfg(feature = "dlss")]
    { facts = crate::dlss_rr::facts(None); }
    if px.tier_all { facts = DraftFacts { built: true, device: true, supported: true, dlls: true }; }
    let a = Avail::probe(&facts, &|| gf_core::parts::accepts(st));
    px.tier_cache = Some((frame, a.clone()));
    a
}

/// Choose a tier from the row (a click, or shift+Z). The matcap is the view mode; the others leave it.
pub fn select(st: &mut State, px: &mut Panels, t: Tier) {
    if t == Tier::Matcap {
        st.view.mode = ViewMode::Matcap;
    } else {
        st.view.mode = ViewMode::Render;
        px.store.tiers.rest = t;
    }
}

/// Shift+Z: the next tier in the row, skipping a hidden one.
pub fn step(st: &mut State, px: &mut Panels) {
    let a = px.tier_cache.as_ref().map(|(_, a)| a.clone()).unwrap_or_else(|| avail(st, px, 0));
    let next = a.next(lit(st, px));
    select(st, px, next);
}

/// How wide the row is with this many buttons.
pub fn width(n: usize) -> f32 {
    2.0 * PAD + n as f32 * B + (n.saturating_sub(1)) as f32 * GAP + SPLIT + CHEV_W
}

fn luma(c: Color32) -> f32 {
    (0.2126 * c.r() as f32 + 0.7152 * c.g() as f32 + 0.0722 * c.b() as f32) / 255.0
}
fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8;
    Color32::from_rgba_unmultiplied(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()), m(a.a(), b.a()))
}

fn arc(c: Pos2, r: f32, from: f32, to: f32, n: usize) -> Vec<Pos2> {
    (0..=n).map(|i| {
        let a = from + (to - from) * i as f32 / n as f32;
        Pos2::new(c.x + r * a.cos(), c.y + r * a.sin())
    }).collect()
}

fn star(p: &Painter, c: Pos2, r: f32, ink: Color32) {
    // a four-point star: long points up, down, left and right, short between
    let k = r * 0.28;
    let pts = vec![
        Pos2::new(c.x, c.y - r), Pos2::new(c.x + k, c.y - k), Pos2::new(c.x + r, c.y), Pos2::new(c.x + k, c.y + k),
        Pos2::new(c.x, c.y + r), Pos2::new(c.x - k, c.y + k), Pos2::new(c.x - r, c.y), Pos2::new(c.x - k, c.y - k),
    ];
    p.add(Shape::convex_polygon(pts, ink, Stroke::NONE));
}

/// The four drawings, 16 points across, centred by eye on `c`. `ink`: the glyph; `bg`: what it stands on
/// (the lit disc, or the glass), for the soft highlight of the clay ball.
fn glyph(p: &Painter, c: Pos2, t: Tier, ink: Color32, bg: Color32) {
    let s = Stroke::new(1.35, ink);
    match t {
        Tier::Matcap => {
            // a clay ball: a shaded sphere (its dark side at the lower right) with one soft highlight at the upper left; no dark centre
            let dark = luma(ink) < 0.5;
            let (body, shade, hi) = if dark { (mix(ink, Color32::WHITE, 0.7), mix(ink, Color32::WHITE, 0.3), Color32::WHITE) } else { (ink, mix(ink, Color32::BLACK, 0.45), mix(ink, Color32::WHITE, 0.85)) };
            let _ = bg;
            p.circle_filled(c, 6.7, shade);
            p.circle_filled(Pos2::new(c.x - 0.8, c.y - 0.8), 6.0, body);
            p.circle_filled(Pos2::new(c.x - 2.4, c.y - 2.6), 1.7, hi);
        }
        Tier::Draft => {
            // a ball not yet drawn: a dashed circle with one short solid arc at the lower right
            let r = 6.6;
            let dashes = 9;
            for i in 0..dashes {
                let a0 = std::f32::consts::TAU * (i as f32 + 0.15) / dashes as f32;
                let a1 = std::f32::consts::TAU * (i as f32 + 0.62) / dashes as f32;
                // (the lower right quarter is left to the solid arc)
                let mid = (a0 + a1) / 2.0;
                if mid > 0.1 && mid < 1.55 { continue; }
                p.add(Shape::line(arc(c, r, a0, a1, 4), Stroke::new(1.25, ink)));
            }
            p.add(Shape::line(arc(c, r, 0.12, 1.5, 10), Stroke::new(2.2, ink)));
        }
        Tier::Live | Tier::Full => {
            // light through glass: a circle, and a ray that comes in level from a dot at the left, bends down where it enters,
            // runs on inside and bends again where it leaves (a different drawing from the camera's orbit: no ring round a ball)
            p.circle_stroke(c, 6.6, s);
            let q = |x: f32, y: f32| Pos2::new(c.x + x, c.y + y);
            p.circle_filled(q(-9.0, -1.9), 1.15, ink);
            p.add(Shape::line(vec![q(-9.0, -1.9), q(-6.3, -1.9), q(6.1, 2.3), q(8.7, 4.6)], s));
            if t == Tier::Full { star(p, q(5.4, -5.3), 2.6, ink); }
        }
    }
}

fn chevron(p: &Painter, c: Pos2, ink: Color32) {
    let s = Stroke::new(1.5, ink);
    p.add(Shape::line(vec![Pos2::new(c.x - 3.2, c.y - 1.6), Pos2::new(c.x, c.y + 1.6), Pos2::new(c.x + 3.2, c.y - 1.6)], s));
}

fn tip(t: Tier, shown: &Shown) -> String {
    match (t, shown) {
        (Tier::Matcap, _) => "matcap: the first surface shaded by a matcap, no glass; immediate, for modelling · Z".into(),
        (Tier::Draft, Shown::Dimmed(why)) => format!("draft: not drawn for this scene just now, the full render draws it: {why}"),
        (Tier::Draft, _) => "draft: soft glass from a quarter of the pixels, rebuilt by NVIDIA Ray Reconstruction; fastest glass while things move · shift+Z steps through the tiers".into(),
        (Tier::Live, Shown::Dimmed(why)) => format!("live glass: this scene is drawn by the full render: {why}"),
        (Tier::Live, _) => "live glass: the whole scene in one volume, the dragged part drawn live; edits stay fast at any number of parts · shift+Z".into(),
        (Tier::Full, _) => "full render: each object exact, every material and light; edits rebuild the object they touch · shift+Z".into(),
    }
}

/// The row, at `camera`'s corner (the camera bar's place), and the dropdown if it is open.
pub fn bar(ctx: &egui::Context, st: &mut State, px: &mut Panels, camera: Pos2) {
    let frame = ctx.cumulative_frame_nr();
    let a = avail(st, px, frame);
    let row = a.row();
    let w = width(row.len());
    let now = lit(st, px);
    let at = at(camera);
    let auto = px.store.tiers.auto;
    let mut chosen: Option<Tier> = None;
    let mut menu_clicked = false;
    let shown = egui::Area::new(Id::unique("tier bar")).order(egui::Order::Middle).fixed_pos(at).constrain(false).fade_in(false).show(ctx, |ui| {
        let (_, whole) = ui.allocate_space(egui::vec2(w, B + 2.0 * PAD));
        ui.painter().add(glass(whole, PANEL_RADIUS, Lift::Bar, px.bright));
        let mut x = whole.left() + PAD;
        let y = whole.top() + PAD;
        for (t, sh) in &row {
            let r = Rect::from_min_size(Pos2::new(x, y), egui::vec2(B, B));
            x += B + GAP;
            px.note(format!("tier:{}", t.id()), r);
            let resp = ui.interact(r, Id::unique(("tier", t.id())), Sense::click());
            let on = *t == now;
            let hot = resp.hovered();
            if hot { ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand); }
            let c = r.center();
            let p = ui.painter();
            let dim = if matches!(sh, Shown::Dimmed(_)) { 0.45 } else { 1.0 };
            let (ink, bg) = if on {
                // lit: the disc in the lit colour with a ring round it, a gap between
                p.circle_filled(c, DISC * 0.5 - 1.5, pal().lit);
                p.circle_stroke(c, DISC * 0.5 + 0.2, Stroke::new(1.4, fade(pal().bright, 0.9)));
                (pal().ink, pal().lit)
            } else if hot {
                p.circle_filled(c, DISC * 0.5, white(0.10));
                (pal().hot, white(0.1))
            } else {
                (pal().button, pal().ink)
            };
            glyph(p, c, *t, fade(ink, dim), bg);
            // the dot under a tier that is drawing now while another one is lit (Auto, while moving)
            if px.tier_drawing == Some(*t) && !on {
                p.circle_filled(Pos2::new(c.x, r.bottom() - 1.5), 1.6, fade(pal().bright, 0.8));
            }
            if resp.on_hover_text(tip(*t, sh)).clicked() { chosen = Some(*t); }
        }
        // the chevron, set apart from the row
        x += SPLIT - GAP;
        let r = Rect::from_min_size(Pos2::new(x, y), egui::vec2(CHEV_W, B));
        px.note("tier:menu", r);
        let resp = ui.interact(r, Id::unique(("tier", "menu")), Sense::click());
        let hot = resp.hovered();
        if hot { ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand); }
        let open = px.tier_menu;
        let p = ui.painter();
        if open { p.rect_filled(r.shrink2(egui::vec2(1.0, 4.0)), 6.0, white(0.12)); }
        chevron(p, Pos2::new(r.center().x, r.center().y + 0.5), if hot || open { pal().hot } else { pal().dim });
        if auto {
            p.text(Pos2::new(r.right() - 0.5, r.top() + 5.5), Align2::CENTER_CENTER, "A", mono(7.5), pal().bright);
        }
        if resp.on_hover_text(format!("settings of {}; auto chooses the tier by what is happening", now.label())).clicked() { menu_clicked = true; }
        whole
    });
    px.note("tier bar", shown.response.rect);
    if let Some(t) = chosen { select(st, px, t); }
    if menu_clicked { px.tier_menu = !px.tier_menu; }
    if px.tier_menu { dropdown(ctx, st, px, &a, Pos2::new(at.x, at.y + super::tools::BAR_H + 6.0)); }
}

/// The chevron's window: the choices every tier shares, then those of the lit tier.
fn dropdown(ctx: &egui::Context, st: &mut State, px: &mut Panels, a: &Avail, first: Pos2) {
    let now = lit(st, px);
    let title = format!("{} settings", now.label());
    let first = px.tier_menu_at.map_or(first, |p| Pos2::new(p[0], p[1]));
    let max_h = (px.full.height() - 2.0 * EDGE - 60.0).max(120.0);
    let out = floating(ctx, "tiers", &title, 300.0, first, max_h, |ui| {
        px.place = "tier";
        set_label_width(ui, 74.0);
        gap(ui, 4.0);
        // auto: the lit tier at rest, and what draws while something moves
        let auto = px.store.tiers.auto;
        let run = named_run(ui, Id::unique(("tier auto", px.place)), "auto", &[
            Item::new("off", !auto).tip("the lit tier draws everything, moving or not"),
            Item::new("on", auto).tip("the lit tier at rest; while something moves, the choice below"),
        ]);
        for (r, n) in run.places.iter().zip(["tier/auto:off", "tier/auto:on"]) { px.note(n, *r); }
        match run.clicked { Some(0) => px.store.tiers.auto = false, Some(1) => px.store.tiers.auto = true, _ => {} }
        if px.store.tiers.auto {
            let m = px.store.tiers.motion;
            let mut names = vec![("same", Motion::Same)];
            if a.draft.is_ok() { names.push(("draft", Motion::Draft)); }
            names.push(("matcap", Motion::Matcap));
            let items: Vec<Item> = names.iter().map(|(n, mm)| Item::new(n, *mm == m)).collect();
            let run = named_run(ui, Id::unique(("tier motion", px.place)), "while moving", &items);
            for (r, (n, _)) in run.places.iter().zip(&names) { px.note(format!("tier/motion:{n}"), *r); }
            if let Some(i) = run.clicked { px.store.tiers.motion = names[i].1; }
            if px.store.tiers.motion == Motion::Matcap { note(ui, "matcap while moving: the picture changes look when you let go"); }
        }
        gap(ui, 2.0);
        times(ui, &px.tier_times, a);
        gap(ui, 4.0);
        match now {
            Tier::Matcap => super::viewport::matcap_settings(ui, st, px),
            Tier::Draft => {
                let c = px.tier_cfg.clone();
                let run = named_run(ui, Id::unique(("tier draft bounces", px.place)), "bounces", &[
                    Item::new("1", c.draft_bounces == 1), Item::new("2", c.draft_bounces == 2).tip("the lab's pass point: 5 ms at 4K"), Item::new("3", c.draft_bounces == 3).tip("over 5 ms at 4K on the busiest scene"),
                ]);
                if let Some(i) = run.clicked { px.tier_cfg.draft_bounces = i as u8 + 1; }
                let run = named_run(ui, Id::unique(("tier draft pixels", px.place)), "pixels traced", &[
                    Item::new("a quarter", c.draft_pixels == 0).tip("0.5 of each side"), Item::new("a ninth", c.draft_pixels == 1).tip("0.33 of each side: softer"),
                ]);
                if let Some(i) = run.clicked { px.tier_cfg.draft_pixels = i; }
                note(ui, "+0.5 GB on the graphics card; started when the program opens");
            }
            Tier::Live => {
                let c = px.tier_cfg.clone();
                // (none of the three is read by the engine yet: dimmed, and said so on hover)
                ui.add_enabled_ui(false, |ui| {
                    let run = named_run(ui, Id::unique(("tier live thin", px.place)), "thin detail", &[
                        Item::new("off", c.thin == 0), Item::new("refine", c.thin == 1).tip("finer bricks only where a thin piece sits"), Item::new("fitted box", c.thin == 2).tip("one choice: the two together look noisier"),
                    ]);
                    if let Some(i) = run.clicked { px.tier_cfg.thin = i; }
                    let run = named_run(ui, Id::unique(("tier live volume", px.place)), "volume", &[
                        Item::new("512", c.volume == 0), Item::new("768", c.volume == 1).tip("about 3 GB, rest over 11 ms at 4K"), Item::new("1024", c.volume == 2).tip("about 6.5 GB, drag 21 ms at 4K"),
                    ]);
                    if let Some(i) = run.clicked { px.tier_cfg.volume = i; }
                    let r = row(ui, 22.0);
                    label(ui, r, "refine MB");
                    text_in(ui, beside_label(ui, r, 22.0), Align2::LEFT_CENTER, &format!("{} MB", c.refine_mb), mono(SIZE), pal().bright);
                }).response.on_disabled_hover_text("not wired yet");
                match &a.live {
                    Ok(()) => note(ui, "parts, cell entries, device MB and last release ms will show here once live glass is in the engine"),
                    Err(why) => note(ui, &format!("not used for this scene: {why}; drawn by the full render")),
                }
            }
            Tier::Full => {
                note(ui, "full render: each object exact, every material and light, as the program has always drawn it. The keep-rate choice is in the camera section of the left panel.");
            }
        }
    });
    if out.closed { px.tier_menu = false; }
    px.tier_menu_at = Some([out.rect.min.x.round(), out.rect.min.y.round()]);
    px.note("tier/menu", out.rect);
    px.note("tier/menu:close", out.cross);
}

/// One line for each tier in the row: its frame time moving and at rest (from the last 120 frames), or a dash.
fn times(ui: &mut Ui, t: &Times, a: &Avail) {
    let r = row(ui, 20.0);
    label(ui, r, "frame times");
    gap(ui, 0.0);
    for (tier, _) in a.row() {
        let r = row(ui, 17.0);
        match t.line(tier) {
            // (the line names its tier already: it runs the whole width)
            Some(words) => { text_in(ui, r, Align2::LEFT_CENTER, &words, mono(SMALL), pal().bright); }
            None => {
                text_in(ui, Rect::from_min_size(r.min, egui::vec2(label_width(ui) + COL_GAP, r.height())), Align2::LEFT_CENTER, tier.label(), mono(SMALL), pal().dim);
                text_in(ui, Rect::from_min_max(Pos2::new(r.left() + label_width(ui) + COL_GAP, r.top()), r.max), Align2::LEFT_CENTER, "—", mono(SMALL), pal().faint);
            }
        }
    }
}

#[allow(dead_code)]
const _ALL: [Tier; 4] = TIERS;
