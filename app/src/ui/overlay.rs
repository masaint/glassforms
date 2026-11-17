//! What lies over the picture without being a panel: the quiet line of figures at the foot of the window
//! (how large the picture is traced, how fast, how long the shape took to build), the small "?" after it,
//! and the help the "?" opens: what the mouse and the keys do, and the benchmark.

use eframe::egui::{self, Align2, Color32, CursorIcon, Id, Pos2, Rect, Sense, Stroke, Ui};

use super::theme::*;
use super::widgets::*;
use super::{Bench, Panels, Request};

/// How far the line's middle is from the foot of the window, and the size of its letters.
const FOOT: f32 = 15.0;
const LETTERS: f32 = 10.5;
/// Width and height of the "?", the least distance between it and the figures, and how far to the right
/// of the middle of the figures it usually stands.
const MARK: f32 = 15.0;
const MARK_GAP: f32 = 9.0;
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
    let whole = 2.0 * (half + MARK + 5.0);
    if hi - lo < whole { return; }
    let centre = full.center().x.clamp(lo + whole / 2.0, hi - whole / 2.0);
    let words = Rect::from_min_max(Pos2::new(centre - text_w / 2.0, y - 8.0), Pos2::new(centre + text_w / 2.0, y + 8.0));
    if text_w > 0.0 {
        // a soft dark bed keeps them readable over bright glass (on black it cannot be seen); over a picture
        // that is bright everywhere the bed is darker and the figures a little brighter
        let bed = Rect::from_min_max(Pos2::new(centre - whole / 2.0, y - 10.0), Pos2::new(centre + whole / 2.0, y + 10.0));
        ui.painter().rect_filled(bed, 10, Color32::from_black_alpha(if px.bright { 150 } else { 70 }));
        px.note("foot", bed);
        for d in [egui::vec2(-1.0, 0.0), egui::vec2(1.0, 0.0), egui::vec2(0.0, -1.0), egui::vec2(0.0, 1.0)] {
            text_in(ui, words.translate(d), Align2::LEFT_CENTER, &px.perf, font.clone(), Color32::from_black_alpha(96));
        }
        text_in(ui, words, Align2::LEFT_CENTER, &px.perf, font.clone(), if px.bright { fade(BRIGHT, 0.78) } else { quiet() });
        px.note("perf", words);
    }

    let mark = Rect::from_center_size(Pos2::new(centre + half + MARK / 2.0, y), egui::vec2(MARK, MARK));
    // it takes a drag as well as a click, so that a press on it is never also a press on the picture under it
    let resp = ui.interact(mark.expand(2.0), Id::unique("help mark"), Sense::click_and_drag());
    let hot = resp.hovered() || px.help_open;
    if resp.hovered() { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let ink = if hot { BRIGHT } else if px.bright { fade(BRIGHT, 0.78) } else { quiet() };
    ui.painter().circle(mark.center(), MARK / 2.0, Color32::from_black_alpha(if hot { 150 } else { 70 }), Stroke::new(1.0, fade(ink, if hot { 0.9 } else { 0.6 })));
    text_in(ui, mark, Align2::CENTER_CENTER, "?", mono(10.0), ink);
    px.note("help mark", mark);
    if resp.on_hover_text("what the mouse and the keys do · key F1").clicked() { px.help_open = !px.help_open; }
}

/// What the mouse does on the picture, and what the keys do. `families`: how many there are.
fn gestures(families: usize) -> [(&'static str, Vec<(String, &'static str)>); 3] {
    let s = |t: &str| t.to_owned();
    [
        ("on the picture", vec![
            (s("drag"), "turn the view (flying: look round)"),
            (s("right-drag, shift-drag"), "slide the view (flying, right-drag looks round)"),
            (s("ctrl-drag, middle-drag"), "roll the view"),
            (s("wheel"), "closer, further away (flying: how fast)"),
            (s("shift-wheel"), "change the lens"),
            (s("click"), "choose the shape or lamp there; on nothing, let go"),
            (s("alt-click"), "focus there (with an aperture; so does a click with the camera tool)"),
            (s("double click"), "put the view back"),
        ]),
        ("on a slider", vec![
            (s("press, drag"), "set it"),
            (s("shift-drag"), "set it finely (ten times finer)"),
            (s("∿"), "let it travel while animating"),
        ]),
        ("keys", vec![
            (s("space"), "spin, or stop spinning"),
            (s("P"), "animate, or pause"),
            (s("← →"), "the shape before, the next"),
            (format!("1 – {}", families.clamp(2, 9)), "the families"),
            (s("R"), "put the view back"),
            (s("L"), "the light room"),
            (s("M"), "the materials lab"),
            (s("K"), "the camera"),
            (s("C"), "clay, or the material again"),
            (s("H"), "hide the panels, or show them"),
            (s("G"), "the handles on what is chosen, or not"),
            (s("T"), "the handles: all at once, move, turn, size"),
            (s("F"), "frame what is chosen"),
            (s("V"), "orbit, or fly"),
            (s("W A S D  Q E"), "flying: forward, left, back, right; down, up (shift fast, ctrl slow)"),
            (s("F11"), "fullscreen (Esc leaves it)"),
            (s("F1"), "this help (Esc closes it)"),
        ]),
    ]
}

/// The help, if it is open: a floating window. `full`: the inside of the program's window.
pub fn help(ctx: &egui::Context, px: &mut Panels, families: usize, full: Rect) {
    if !px.help_open { return; }
    let width = FLOAT_W;
    // it first appears in the middle, just above the line it was opened from
    let first = Pos2::new(full.center().x - (width + 24.0) / 2.0, (full.bottom() - 550.0).max(EDGE));
    let out = floating(ctx, "help", "mouse and keys", width, first, (full.height() - 2.0 * EDGE - 44.0).max(80.0), |ui| {
        set_label_width(ui, COL_LABEL);
        for (title, lines) in gestures(families) {
            section(ui, title, "");
            for (what, does) in lines {
                let r = row(ui, 17.0);
                text_in(ui, Rect::from_min_size(r.min, egui::vec2(142.0, r.height())), Align2::LEFT_CENTER, &what, mono(SMALL), BRIGHT);
                text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 146.0, r.top()), r.max), Align2::LEFT_CENTER, does, sans(SMALL), DIM);
            }
        }
        // how fast this view can be drawn when nothing waits for the display
        section(ui, "speed", "");
        let caption = match &px.bench {
            Bench::Idle => "benchmark (no vsync)",
            Bench::Asked => "running...",
            Bench::Done(result) => result.as_str(),
        };
        ui.scope(|ui| {
            if px.rendering { ui.disable(); }
            let (resp, at) = wide_button(ui, Id::unique("benchmark"), caption, false);
            if resp.on_hover_text("times 60 frames of this view turning, drawn as fast as they will go").clicked() && px.bench != Bench::Asked {
                px.requests.push(Request::Benchmark);
            }
            if px.record { px.rects.insert("benchmark".into(), at); }
        });
    });
    if out.closed { px.help_open = false; }
    px.note("help", out.rect);
    px.note("help:close", out.cross);
}
