//! What lies over the picture without being a panel: the quiet line of figures at the foot of the window
//! (how large the picture is traced, how fast, how long the shape took to build), the small "?" after it,
//! and the help the "?" opens: what the mouse, the tools over the picture and the keys do.

use eframe::egui::{self, Align2, Color32, CursorIcon, Id, Pos2, Rect, Sense, Stroke, Ui};

use super::theme::*;
use super::widgets::*;
use super::Panels;

/// How far the line's middle is from the foot of the window, and the size of its letters.
const FOOT: f32 = 15.0;
const LETTERS: f32 = 10.5;
/// Width and height of the "?", the least distance between it and the figures, and how far to the right
/// of the middle of the figures it usually stands.
const MARK: f32 = 15.0;
const MARK_GAP: f32 = 9.0;
/// The gap between the "?" and the gear after it.
const MARK_SPACE: f32 = 5.0;
const MARK_REACH: f32 = 152.0;

/// The line of figures and its "?", painted straight onto the picture, in the middle at the foot of the
/// window. The figures are only paint: a press on them is a press on the picture. Where a panel reaches
/// down beside them in a narrow window they move over to stay clear, and if there is no room they are
/// left out. `ui`: the window's own, the one the picture is painted on.
pub fn foot(ui: &mut Ui, px: &mut Panels, full: Rect) {
    let font = mono(LETTERS);
    let text_w = if px.perf.is_empty() { 0.0 } else { text_width(ui, &px.perf, &font) };
    let y = full.bottom() - FOOT;
    // the room between the panels at this height
    let reaches = |r: &Option<Rect>| r.filter(|r| r.bottom() > y - 12.0);
    let lo = reaches(&px.left_at).map_or(full.left() + 6.0, |r| r.right() + 10.0);
    let hi = reaches(&px.right_at).map_or(full.right() - 6.0, |r| r.left() - 10.0);
    // The figures are centred, and the "?" stands a fixed way to their right, far enough for the longest
    // figures there usually are: it must not jump about as they change between frames a second and how far
    // the picture has refined. Only when the figures are longer than that (a video's progress), or the
    // room is short, does it stand right after them.
    let snug = text_w / 2.0 + MARK_GAP;
    let half = if hi - lo >= 2.0 * (MARK_REACH + MARK + 5.0) { snug.max(MARK_REACH) } else { snug };
    // (room on both sides for the "?" and the gear after it, so that the figures stay in the middle)
    let whole = 2.0 * (half + 2.0 * MARK + MARK_SPACE + 5.0);
    if hi - lo < whole { return; }
    let centre = full.center().x.clamp(lo + whole / 2.0, hi - whole / 2.0);
    let words = Rect::from_min_max(Pos2::new(centre - text_w / 2.0, y - 8.0), Pos2::new(centre + text_w / 2.0, y + 8.0));
    if text_w > 0.0 {
        // a soft dark bed keeps them readable over bright glass (on black it cannot be seen); over a picture
        // that is bright everywhere the bed is darker and the figures a little brighter
        let bed = Rect::from_min_max(Pos2::new(centre - whole / 2.0, y - 10.0), Pos2::new(centre + whole / 2.0, y + 10.0));
        ui.painter().add(on_glass(bed, px.bright));
        px.note("foot", bed);
        for d in [egui::vec2(-1.0, 0.0), egui::vec2(1.0, 0.0), egui::vec2(0.0, -1.0), egui::vec2(0.0, 1.0)] {
            text_in(ui, words.translate(d), Align2::LEFT_CENTER, &px.perf, font.clone(), halo());
        }
        text_in(ui, words, Align2::LEFT_CENTER, &px.perf, font.clone(), if px.bright || frost() != Frost::Off { fade(pal().bright, 0.8) } else { quiet() });
        px.note("perf", words);
    }

    let mark = Rect::from_center_size(Pos2::new(centre + half + MARK / 2.0, y), egui::vec2(MARK, MARK));
    // it takes a drag as well as a click, so that a press on it is never also a press on the picture under it
    let resp = ui.interact(mark.expand(2.0), Id::unique("help mark"), Sense::click_and_drag());
    let hot = resp.hovered() || px.help_open;
    if resp.hovered() { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let ink = if hot { pal().bright } else if px.bright || frost() != Frost::Off { fade(pal().bright, 0.8) } else { quiet() };
    ui.painter().add(on_glass(mark, hot || px.bright));
    ui.painter().circle_stroke(mark.center(), MARK / 2.0, Stroke::new(1.0, fade(ink, if hot { 0.9 } else { 0.6 })));
    text_in(ui, mark, Align2::CENTER_CENTER, "?", mono(10.0), ink);
    px.note("help mark", mark);
    if resp.on_hover_text("what the mouse and the keys do · key F1").clicked() { px.help_open = !px.help_open; }

    // the gear beside it: the settings (the look, the frost, the size)
    let gear = mark.translate(egui::vec2(MARK + MARK_SPACE, 0.0));
    let resp = ui.interact(gear.expand(2.0), Id::unique("settings mark"), Sense::click_and_drag());
    let hot = resp.hovered() || px.store.settings_open;
    if resp.hovered() { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let ink = if hot { pal().bright } else if px.bright || frost() != Frost::Off { fade(pal().bright, 0.8) } else { quiet() };
    ui.painter().add(on_glass(gear, hot || px.bright));
    ui.painter().circle_stroke(gear.center(), MARK / 2.0, Stroke::new(1.0, fade(ink, if hot { 0.9 } else { 0.6 })));
    gear_mark(ui, gear.center(), ink);
    px.note("settings mark", gear);
    if resp.on_hover_text("settings: dark or light, frosted glass, size").clicked() { px.store.settings_open = !px.store.settings_open; }
}

/// What the figures and the marks sit on: a pill of the panels' frosted glass, or while the frost is off a
/// soft bed, darker (or in the light look, paler) over a picture bright from edge to edge and under a mark
/// that is lit.
fn on_glass(r: Rect, strong: bool) -> egui::Shape {
    if frost() == Frost::Off { return egui::Shape::rect_filled(r, (r.height() / 2.0) as u8, bed(strong)); }
    glass(r, (r.height() / 2.0) as u8, Lift::Bar, strong)
}
/// The faint outline round the figures that keeps them clear of the glass behind.
fn halo() -> Color32 {
    if pal().dark { Color32::from_black_alpha(96) } else { Color32::from_white_alpha(150) }
}

/// A small gear: a ring with eight teeth.
fn gear_mark(ui: &Ui, c: Pos2, ink: Color32) {
    let p = ui.painter();
    p.circle_stroke(c, 2.6, Stroke::new(1.3, ink));
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let d = egui::vec2(a.cos(), a.sin());
        p.line_segment([c + d * 3.6, c + d * 5.3], Stroke::new(1.6, ink));
    }
}

/// What the mouse does on the picture, and what the keys do. `families`: how many there are.
fn gestures(families: usize) -> [(&'static str, Vec<(String, &'static str)>); 4] {
    let s = |t: &str| t.to_owned();
    [
        ("on the picture", vec![
            (s("drag"), "turn the view (flying: look round)"),
            (s("right-drag, shift-drag"), "slide the view (flying: right-drag looks round)"),
            (s("ctrl-drag, middle-drag"), "roll the view"),
            (s("wheel"), "closer, further away (flying: how fast)"),
            (s("shift-wheel"), "change the lens"),
            (s("click"), "choose the shape or lamp there (a lamp opens the light room at it); on nothing, let go"),
            (s("alt-click"), "focus there, while the lens is open (so does a click with the orbit tool)"),
            (s("double click"), "put the view back"),
            (s("drop a file"), "a setup opens, a panorama lights the shape, a model becomes the shape (with shift: one more layer)"),
        ]),
        ("on a slider", vec![
            (s("press, drag"), "set it"),
            (s("shift-drag"), "set it finely (ten times finer)"),
            (s("ctrl-shift-drag"), "finer still (a hundred times), between its steps"),
            (s("alt-drag"), "take it past its ends, as far as is safe (with shift or ctrl-shift: finely)"),
            (s("click its number"), "type a value: enter keeps it, esc leaves it; past its ends shows in green"),
            (s("∿"), "let it travel while animating"),
        ]),
        ("over the picture", vec![
            (s("orbit, fly, frame"), "the camera's tools, beside the left panel"),
            (s("dots 1 – 0"), "the camera slots: a click goes to a stored view, ctrl+click stores one"),
            (s("cube"), "which way the camera looks; a click on an end looks from that side"),
            (s("select … size"), "the object tools, beside the right panel: choose, move, turn and size"),
        ]),
        ("keys", vec![
            (s("space"), "spin, or stop spinning"),
            (s("P"), "animate, or pause"),
            (s("← →"), "the shape before, the next"),
            (format!("1 – {}", families.clamp(2, 9)), "the families"),
            (s("R"), "put the view back"),
            (s("L"), "the light room"),
            (s("M"), "the materials lab"),
            (s("K"), "fold the camera section, or open it"),
            (s("C"), "clay, or the material again"),
            (s("H"), "hide the panels, or show them"),
            (s("G"), "the handles on what is chosen, or not"),
            (s("T"), "the handles: all at once, move, turn, size"),
            (s("F"), "frame what is chosen"),
            (s("V"), "orbit, or fly"),
            (s("W A S D  Q E"), "flying: forward, left, back, right; down, up (shift fast, ctrl slow)"),
            (s("F11"), "fullscreen (Esc leaves it)"),
            (s("ctrl+Z, ctrl+shift+Z"), "undo, redo (also ctrl+Y): the line under the camera's tools says what changed"),
            (s("ctrl+1 – 0"), "store the view in a slot"),
            (s("shift+1 – 0"), "go to a stored view"),
            (s("Esc while dragging"), "call the drag off: back as it was"),
            (s("F1"), "this help (Esc closes it)"),
        ]),
    ]
}

/// How wide the help is, and its column of gestures and keys.
const HELP_W: f32 = 420.0;
const HELP_KEYS: f32 = 150.0;

/// The help, if it is open: a floating window. `full`: the inside of the program's window. Its words wrap,
/// so nothing is cut off at any size.
pub fn help(ctx: &egui::Context, px: &mut Panels, families: usize, full: Rect) {
    if !px.help_open { return; }
    let width = HELP_W;
    // it first appears in the middle, just above the line it was opened from
    let first = Pos2::new(full.center().x - (width + 24.0) / 2.0, (full.bottom() - 640.0).max(EDGE));
    let out = floating(ctx, "help", "mouse and keys", width, first, (full.height() - 2.0 * EDGE - 44.0).max(80.0), |ui| {
        set_label_width(ui, COL_LABEL);
        for (title, lines) in gestures(families) {
            section(ui, title, "");
            for (what, does) in lines {
                let top = ui.cursor().top();
                let left = ui.cursor().left();
                let g = wrapped(ui, does, sans(SMALL), pal().dim, ui.available_width() - HELP_KEYS - 4.0, 1.35);
                let h = g.size().y.max(17.0) + 2.0;
                let r = row(ui, h);
                text_in(ui, Rect::from_min_size(Pos2::new(left, top), egui::vec2(HELP_KEYS, 17.0)), Align2::LEFT_CENTER, &what, mono(SMALL), pal().bright);
                ui.painter().galley(Pos2::new(r.left() + HELP_KEYS + 4.0, top + (17.0 - SMALL * 1.35) / 2.0), g, pal().dim);
            }
        }
    });
    if out.closed { px.help_open = false; }
    px.note("help", out.rect);
    px.note("help:close", out.cross);
}
