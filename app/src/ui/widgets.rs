//! The controls the panels are made of, each drawn by hand so that they look and measure alike: buttons,
//! runs of buttons, menus, sliders, the two-handled range of an animated slider, fold-away titles, and the
//! windows that float over the picture. None of them knows anything about the renderer; they take a value
//! and say what the user did to it.

use std::sync::Arc;

use eframe::egui::{
    self, text::LayoutJob, text::TextFormat, text::TextWrapping, Align2, Color32, CursorIcon, FontId, Galley, Id, Pos2, Rect, Response, Sense, Shape, Stroke,
    StrokeKind, Ui,
};

use super::theme::*;

/// Take the next `h` points of height across the whole width of the panel.
pub fn row(ui: &mut Ui, h: f32) -> Rect {
    let w = ui.available_width();
    ui.allocate_space(egui::vec2(w, h)).1
}

/// An empty gap.
pub fn gap(ui: &mut Ui, h: f32) {
    if h > 0.0 { row(ui, h); }
}

fn layout(ui: &Ui, text: &str, font: FontId, color: Color32, wrap: f32, line: Option<f32>, spacing: f32, one_row: bool, weight: f32) -> Arc<Galley> {
    let mut format = TextFormat { font_id: font, color, line_height: line, extra_letter_spacing: spacing, ..Default::default() };
    // a font that comes in every weight can be asked for a heavier one
    if weight > 0.0 { format.coords = egui::epaint::text::VariationCoords::new([(b"wght", weight)]); }
    let mut job = LayoutJob::single_section(text.to_owned(), format);
    job.wrap = TextWrapping { max_width: wrap, max_rows: if one_row { 1 } else { usize::MAX }, break_anywhere: one_row, overflow_character: Some('…') };
    ui.painter().layout_job(job)
}

/// Text wrapped at `width`, each line `line` times as tall as the font, ready to be painted.
pub fn wrapped(ui: &Ui, text: &str, font: FontId, color: Color32, width: f32, line: f32) -> Arc<Galley> {
    let h = font.size * line;
    layout(ui, text, font, color, width.max(1.0), Some(h), 0.0, false, 0.0)
}

/// One line of text in capitals set a little apart, as the titles are.
pub fn spaced(ui: &Ui, text: &str, size: f32, color: Color32) -> Arc<Galley> {
    layout(ui, &text.to_uppercase(), mono(size), color, f32::INFINITY, None, 1.0, false, 0.0)
}

/// How wide a piece of text is on one line.
pub fn text_width(ui: &Ui, text: &str, font: &FontId) -> f32 {
    layout(ui, text, font.clone(), Color32::WHITE, f32::INFINITY, None, 0.0, false, 0.0).size().x
}

/// One line of text inside a rectangle, centred vertically, at its left, middle or right. Text too long
/// for the rectangle is cut short with an ellipsis; returns true if that happened.
pub fn text_in(ui: &Ui, rect: Rect, anchor: Align2, text: &str, font: FontId, color: Color32) -> bool {
    let g = layout(ui, text, font, color, rect.width().max(1.0), None, 0.0, true, 0.0);
    let r = anchor.anchor_size(anchor.pos_in_rect(&rect), g.size());
    let cut = g.elided;
    ui.painter().galley(r.min, g, color);
    cut
}

/// A block of text that wraps at the panel's width (or `width`, counted from `indent`), each line `line`
/// times as tall as the font. Returns the space it took.
pub fn paragraph(ui: &mut Ui, text: &str, font: FontId, color: Color32, line: f32, indent: f32) -> Rect {
    let g = wrapped(ui, text, font, color, ui.available_width() - indent, line);
    let r = row(ui, g.size().y.ceil());
    ui.painter().galley(Pos2::new(r.left() + indent, r.top()), g, color);
    r
}

/// A few words of explanation under some controls.
pub fn note(ui: &mut Ui, text: &str) {
    gap(ui, 3.0);
    paragraph(ui, text, sans(SMALL), FAINT, 1.4, 0.0);
    gap(ui, 2.0);
}

/// Say how wide the column for names is in whatever is drawn from here on (a panel says so before it
/// draws its rows; it stays so until the next one says otherwise).
pub fn set_label_width(ui: &Ui, width: f32) {
    ui.data_mut(|d| d.insert_temp(Id::unique("width of the column for names"), width));
}

/// How wide the column for names is in the panel being drawn.
pub fn label_width(ui: &Ui) -> f32 {
    ui.data(|d| d.get_temp(Id::unique("width of the column for names"))).unwrap_or(COL_LABEL)
}

/// A name written in its column: one too long for it is set a little smaller, and only if it is still too
/// long cut short. Returns true if it was cut.
pub fn name_in(ui: &Ui, rect: Rect, text: &str, size: f32) -> bool {
    let fits = [size, size - 0.5, size - 1.0].into_iter().find(|s| text_width(ui, text, &mono(*s)) <= rect.width());
    text_in(ui, rect, Align2::LEFT_CENTER, text, mono(fits.unwrap_or(size - 1.0)), DIM)
}

/// The name of a control, in the column for names at the left of its row.
pub fn label(ui: &Ui, row: Rect, text: &str) {
    name_in(ui, Rect::from_min_size(row.min, egui::vec2(label_width(ui), row.height())), text, SIZE);
}

/// The part of a row to the right of the column for names: where its menu, field or buttons go.
pub fn beside_label(ui: &Ui, row: Rect, h: f32) -> Rect {
    Rect::from_min_max(Pos2::new(row.left() + label_width(ui) + COL_GAP, row.center().y - h / 2.0), Pos2::new(row.right(), row.center().y + h / 2.0))
}

/// How much room a button with these words needs. `pad` is the space around the words inside its border.
pub fn button_size(ui: &Ui, text: &str, font: &FontId, pad: egui::Vec2) -> egui::Vec2 {
    egui::vec2((text_width(ui, text, font) + 2.0 * pad.x + 2.0).ceil(), (font.size * LINE).round() + 2.0 * pad.y + 2.0)
}

/// The box of a button in a given place: what the pointer did to it, and the colour for whatever goes on it.
/// `on`: it is lit (the current choice, or something switched on).
fn button_box(ui: &mut Ui, rect: Rect, id: Id, on: bool) -> (Response, Color32) {
    let resp = ui.interact(rect, id, Sense::click());
    let live = ui.is_enabled();
    let hot = live && resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let down = live && resp.is_pointer_button_down_on();
    let (fill, line, ink) = if on {
        (if hot { Color32::WHITE } else { BRIGHT }, BRIGHT, INK)
    } else if hot {
        (white(if down { 0.12 } else { 0.05 }), white(0.3), Color32::WHITE)
    } else {
        (white(0.05), white(0.1), BUTTON)
    };
    ui.painter().rect(rect, BUTTON_RADIUS, fill, Stroke::new(1.0, line), StrokeKind::Inside);
    (resp, ink)
}

/// A button with words on it.
pub fn button(ui: &mut Ui, rect: Rect, id: Id, text: &str, on: bool, font: FontId) -> Response {
    let (resp, ink) = button_box(ui, rect, id, on);
    text_in(ui, rect.shrink2(egui::vec2(2.0, 0.0)), Align2::CENTER_CENTER, text, font, ink);
    resp
}

/// The button beside a slider that sets it travelling: one turn of a wave.
pub fn wave_button(ui: &mut Ui, rect: Rect, id: Id, on: bool) -> Response {
    let (resp, ink) = button_box(ui, rect, id, on);
    let c = rect.center();
    let curve: Vec<Pos2> = (0..=16).map(|i| {
        let u = i as f32 / 16.0;
        Pos2::new(c.x - 4.5 + 9.0 * u, c.y - 2.6 * (u * std::f32::consts::TAU).sin())
    }).collect();
    ui.painter().add(Shape::line(curve, Stroke::new(1.25, ink)));
    resp
}

/// A button as wide as the panel. Returns what the pointer did to it, and where it is.
pub fn wide_button(ui: &mut Ui, id: Id, text: &str, on: bool) -> (Response, Rect) {
    gap(ui, 5.0);
    let r = row(ui, 26.0);
    let resp = button(ui, r, id, text, on, mono(SIZE));
    gap(ui, 2.0);
    (resp, r)
}

/// One button of a run of buttons.
pub struct Item<'a> {
    pub text: &'a str,
    pub on: bool,
    pub tip: Option<String>,
    /// scroll whatever holds the run so that this button is in view
    pub reveal: bool,
    /// a dot of this colour before the words (a material's)
    pub colour: Option<Color32>,
    /// the mark of something costly after the words
    pub mark: bool,
}
impl<'a> Item<'a> {
    pub fn new(text: &'a str, on: bool) -> Self {
        Self { text, on, tip: None, reveal: false, colour: None, mark: false }
    }
    pub fn reveal(mut self, reveal: bool) -> Self {
        self.reveal = reveal;
        self
    }
    pub fn tip(mut self, tip: impl Into<String>) -> Self {
        self.tip = Some(tip.into());
        self
    }
    pub fn colour(mut self, colour: Color32) -> Self {
        self.colour = Some(colour);
        self
    }
    pub fn mark(mut self, mark: bool) -> Self {
        self.mark = mark;
        self
    }
}

/// The room a dot of colour takes before a button's words, and a mark after them.
const DOT_W: f32 = 14.0;
const MARK_W: f32 = 10.0;

/// How much room one button of a run needs.
fn item_size(ui: &Ui, it: &Item, font: &FontId, pad: egui::Vec2) -> egui::Vec2 {
    let words = button_size(ui, it.text, font, pad);
    egui::vec2(words.x + if it.colour.is_some() { DOT_W } else { 0.0 } + if it.mark { MARK_W } else { 0.0 }, words.y)
}

/// A dot of a material's colour. On a lit (bright) button its rim is dark, so that a pale colour still shows.
pub fn dot(ui: &Ui, centre: Pos2, colour: Color32, on_bright: bool) {
    ui.painter().circle(centre, 4.5, colour, Stroke::new(1.0, if on_bright { Color32::from_black_alpha(120) } else { white(0.28) }));
}

/// The mark of something that costs noticeably more to draw: a small diamond.
pub fn costly_mark(ui: &Ui, centre: Pos2) {
    let d = 3.2;
    let points = vec![centre + egui::vec2(0.0, -d), centre + egui::vec2(d, 0.0), centre + egui::vec2(0.0, d), centre + egui::vec2(-d, 0.0)];
    ui.painter().add(Shape::convex_polygon(points, COSTLY, Stroke::NONE));
}

/// The name of a costly control keeps this much of its column free for the mark.
pub const MARK_ROOM: f32 = 11.0;

/// Put the mark of something costly after a name written at the left of `rect` (its column, less the
/// room kept for the mark): close behind the name, or at the column's end if the name fills it.
pub fn mark_after(ui: &Ui, rect: Rect, text: &str) {
    let end = rect.left() + text_width(ui, text, &mono(SIZE));
    costly_mark(ui, Pos2::new((end + 7.0).min(rect.right() + MARK_ROOM - 5.0), rect.center().y));
}

/// One button of a run: its words, with the dot and the mark it may have.
fn item_button(ui: &mut Ui, rect: Rect, id: Id, it: &Item, font: FontId) -> Response {
    if it.colour.is_none() && !it.mark { return button(ui, rect, id, it.text, it.on, font); }
    let (resp, ink) = button_box(ui, rect, id, it.on);
    let mut words = rect.shrink2(egui::vec2(2.0, 0.0));
    if let Some(colour) = it.colour {
        dot(ui, Pos2::new(rect.left() + 11.0, rect.center().y), colour, it.on);
        words.min.x += DOT_W;
    }
    if it.mark {
        costly_mark(ui, Pos2::new(rect.right() - 9.0, rect.center().y));
        words.max.x -= MARK_W;
    }
    text_in(ui, words, Align2::CENTER_CENTER, it.text, font, ink);
    resp
}

/// What came of a run of buttons: which one was clicked, the space the whole run took (the caller may
/// write a name in its indent), and where each button went.
pub struct Run {
    pub clicked: Option<usize>,
    pub rect: Rect,
    pub places: Vec<Rect>,
}

/// The gap between neighbouring buttons of a run, and between its lines.
pub const RUN_SPACE: f32 = 4.0;

/// Lay buttons of these widths out in lines no wider than `limit`, `space` between neighbours: which
/// buttons are on which line.
fn lines_of(widths: &[f32], limit: f32, space: f32) -> Vec<Vec<usize>> {
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut x = 0.0f32;
    for (i, w) in widths.iter().enumerate() {
        match lines.last_mut() {
            Some(line) if x + space + w <= limit + 0.5 => {
                line.push(i);
                x += space + w;
            }
            _ => {
                lines.push(vec![i]);
                x = *w;
            }
        }
    }
    lines
}

/// The same in as few lines as `limit` allows, but with the lines as even in length as can be: laid out
/// to the narrowest limit that still gives that few lines.
fn even_lines(widths: &[f32], limit: f32, space: f32) -> Vec<Vec<usize>> {
    let fewest = lines_of(widths, limit, space).len();
    let (mut lo, mut hi) = (widths.iter().copied().fold(0.0, f32::max).min(limit), limit);
    while hi - lo > 0.5 {
        let mid = (lo + hi) / 2.0;
        if lines_of(widths, mid, space).len() <= fewest { hi = mid } else { lo = mid }
    }
    lines_of(widths, hi, space)
}

/// Draw the buttons of a run in the places worked out for them, which are relative to `origin`.
fn run_buttons(ui: &mut Ui, id: Id, items: &[Item], places: &mut [Rect], origin: egui::Vec2, font: &FontId) -> Option<usize> {
    let mut clicked = None;
    for (i, (it, place)) in items.iter().zip(places.iter_mut()).enumerate() {
        *place = place.translate(origin);
        let mut resp = item_button(ui, *place, id.with(i), it, font.clone());
        if it.reveal { ui.scroll_to_rect(place.expand(3.0), None); }
        if let Some(tip) = &it.tip { resp = resp.on_hover_text(tip.as_str()); }
        if resp.clicked() { clicked = Some(i); }
    }
    clicked
}

/// A run of buttons that wraps onto further lines, starting `indent` in from the left of the panel and at
/// least `min_h` tall.
pub fn button_run(ui: &mut Ui, id: Id, indent: f32, items: &[Item], pad: egui::Vec2, font: &FontId, min_h: f32) -> Run {
    let width = ui.available_width() - indent;
    let sizes: Vec<egui::Vec2> = items.iter().map(|it| item_size(ui, it, font, pad)).collect();
    let widths: Vec<f32> = sizes.iter().map(|s| s.x).collect();
    let line_h = sizes.first().map_or(0.0, |s| s.y);
    let mut places = vec![Rect::ZERO; items.len()];
    let lines = lines_of(&widths, width, RUN_SPACE);
    for (n, line) in lines.iter().enumerate() {
        let mut x = 0.0;
        for &i in line {
            places[i] = Rect::from_min_size(Pos2::new(x, n as f32 * (line_h + RUN_SPACE)), sizes[i]);
            x += widths[i] + RUN_SPACE;
        }
    }
    let used = (lines.len() as f32 * (line_h + RUN_SPACE) - RUN_SPACE).max(0.0);
    let rect = row(ui, used.max(min_h));
    let origin = egui::vec2(rect.left() + indent, rect.top() + ((min_h - used) / 2.0).max(0.0));
    let clicked = run_buttons(ui, id, items, &mut places, origin, font);
    Run { clicked, rect, places }
}

/// A run of buttons that fills the panel's width however many there are: as few lines as will hold them,
/// the lines as even as can be, and the buttons of each line widened until the line is full.
pub fn filled_run(ui: &mut Ui, id: Id, items: &[Item], pad: egui::Vec2, font: &FontId) -> Run {
    let width = ui.available_width();
    let sizes: Vec<egui::Vec2> = items.iter().map(|it| item_size(ui, it, font, pad)).collect();
    let widths: Vec<f32> = sizes.iter().map(|s| s.x).collect();
    let line_h = sizes.first().map_or(0.0, |s| s.y);
    let mut places = vec![Rect::ZERO; items.len()];
    let lines = even_lines(&widths, width, RUN_SPACE);
    for (n, line) in lines.iter().enumerate() {
        let natural: f32 = line.iter().map(|&i| widths[i]).sum::<f32>() + RUN_SPACE * (line.len() as f32 - 1.0);
        let more = ((width - natural) / line.len() as f32).max(0.0);
        let mut x = 0.0f32;
        for &i in line {
            // whole points, so that every edge is crisp; the last button of the line takes up the slack
            let w = if Some(&i) == line.last() { width - x } else { (widths[i] + more).round() };
            places[i] = Rect::from_min_size(Pos2::new(x, n as f32 * (line_h + RUN_SPACE)), egui::vec2(w.max(1.0), line_h));
            x += w + RUN_SPACE;
        }
    }
    let used = (lines.len() as f32 * (line_h + RUN_SPACE) - RUN_SPACE).max(0.0);
    let rect = row(ui, used);
    let clicked = run_buttons(ui, id, items, &mut places, rect.min.to_vec2(), font);
    Run { clicked, rect, places }
}

/// A name on the left and a run of buttons on the right.
pub fn named_run(ui: &mut Ui, id: Id, name: &str, items: &[Item]) -> Run {
    let run = button_run(ui, id, label_width(ui) + COL_GAP, items, egui::vec2(7.0, 2.0), &mono(SIZE), 24.0);
    label(ui, run.rect, name);
    run
}

/// A name on the left and a run of buttons on the right, one of them lit. Returns the one clicked.
pub fn choice(ui: &mut Ui, id: Id, name: &str, options: &[&str], current: usize) -> Option<usize> {
    let items: Vec<Item> = options.iter().enumerate().map(|(i, o)| Item::new(o, i == current)).collect();
    named_run(ui, id, name, &items).clicked
}

/// A menu in a given place: a box showing the current choice, which opens a list. Returns the new choice.
pub fn select(ui: &mut Ui, rect: Rect, id: Id, options: &[String], current: usize, font: FontId) -> Option<usize> {
    let resp = ui.interact(rect, id, Sense::click());
    let hot = ui.is_enabled() && resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let open = egui::Popup::is_id_open(ui.ctx(), egui::Popup::default_response_id(&resp));
    ui.painter().rect(rect, BUTTON_RADIUS, white(0.05), Stroke::new(1.0, white(if hot || open { 0.3 } else { 0.1 })), StrokeKind::Inside);
    let words = Rect::from_min_max(Pos2::new(rect.left() + 6.0, rect.top()), Pos2::new(rect.right() - 15.0, rect.bottom()));
    // a choice whose name does not fit is shown without its remark in brackets; the list has it in full
    let name = options.get(current).map_or("", |s| s.as_str());
    let fits = text_width(ui, name, &font) <= words.width();
    let brief = if fits { name } else { name.split(" (").next().unwrap_or(name) };
    text_in(ui, words, Align2::LEFT_CENTER, brief, font.clone(), BRIGHT);
    // the little arrow that says "this opens"
    let c = Pos2::new(rect.right() - 9.0, rect.center().y + 0.5);
    ui.painter().add(Shape::convex_polygon(vec![c + egui::vec2(-3.5, -2.0), c + egui::vec2(3.5, -2.0), c + egui::vec2(0.0, 2.5)], if hot || open { BRIGHT } else { DIM }, Stroke::NONE));

    let mut chosen = None;
    let mut places = Vec::new();
    let frame = egui::Frame::new().fill(MENU_FILL).stroke(Stroke::new(1.0, white(0.16))).corner_radius(BUTTON_RADIUS).inner_margin(3);
    let shown = egui::Popup::menu(&resp).frame(frame).gap(2.0).width(rect.width().max(92.0) - 8.0).show(|ui| {
        let w = (rect.width().max(92.0) - 8.0).max(options.iter().map(|o| text_width(ui, o, &font) + 14.0).fold(0.0, f32::max));
        ui.set_width(w);
        for (i, o) in options.iter().enumerate() {
            let (r, item) = ui.allocate_exact_size(egui::vec2(w, 19.0), Sense::click());
            places.push(r);
            if item.hovered() {
                ui.painter().rect_filled(r, 3, fade(ACCENT, 0.22));
                ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
            }
            let ink = if item.hovered() { Color32::WHITE } else if i == current { BRIGHT } else { BUTTON };
            text_in(ui, r.shrink2(egui::vec2(6.0, 0.0)), Align2::LEFT_CENTER, o, font.clone(), ink);
            if i == current {
                ui.painter().circle_filled(Pos2::new(r.right() - 7.0, r.center().y), 1.6, ACCENT);
            }
            if item.clicked() {
                chosen = Some(i);
                ui.close();
            }
        }
    });
    if shown.is_some() { ui.data_mut(|d| d.insert_temp(open_menu(), places)); }
    chosen.filter(|&i| i != current)
}

fn open_menu() -> Id {
    Id::unique("the items of the menu that is open")
}

/// Where the items of the menu that is open are, in order (for tests that point at them); empty if none is.
pub fn menu_places(ctx: &egui::Context) -> Vec<Rect> {
    ctx.data(|d| d.get_temp::<Vec<Rect>>(open_menu())).unwrap_or_default()
}

/// Forget where the last menu's items were: done before each frame is laid out, so that only a menu
/// shown in the frame is reported.
pub fn forget_menu(ctx: &egui::Context) {
    ctx.data_mut(|d| d.remove::<Vec<Rect>>(open_menu()));
}

/// A name on the left and a menu on the right.
pub fn pick(ui: &mut Ui, id: Id, name: &str, options: &[String], current: usize) -> Option<usize> {
    let r = row(ui, 24.0);
    label(ui, r, name);
    select(ui, beside_label(ui, r, 22.0), id, options, current, mono(SIZE))
}

/// A menu of values that are not simply 0, 1, 2...: returns the value chosen. A current value that is
/// not on the list (it came from the command line, say) is shown as one more entry.
pub fn pick_value<T: PartialEq + Clone + std::fmt::Display>(ui: &mut Ui, id: &str, name: &str, options: &[(T, String)], current: &T) -> Option<T> {
    let mut all: Vec<(T, String)> = options.to_vec();
    if !all.iter().any(|(v, _)| v == current) { all.push((current.clone(), current.to_string())); }
    let names: Vec<String> = all.iter().map(|(_, n)| n.clone()).collect();
    let at = all.iter().position(|(v, _)| v == current).unwrap_or(0);
    pick(ui, Id::unique(("pick", id)), name, &names, at).map(|i| all[i].0.clone())
}

/// What a slider says happened to it this frame.
#[derive(Default)]
pub struct Moved {
    /// it is being dragged, and this is the value under the pointer
    pub value: Option<f64>,
    /// the pointer has just let go of it
    pub released: bool,
    /// it is being dragged in the finest steps (ctrl+shift): show more of the value's digits
    pub finer: bool,
}

/// Width of a range's handles, and of the allowance a plain slider's round handle keeps at each end.
const HANDLE_W: f32 = 9.0;

/// The nearest value a slider with this step can take.
pub fn snap(v: f64, lo: f64, hi: f64, step: f64) -> f64 {
    snap_from(v, lo, (lo, hi), step)
}
/// The same with the steps counted from `origin` (the slider's own low end, so that a value past either
/// end still lands on its steps) and the value kept within `limits`.
pub fn snap_from(v: f64, origin: f64, (lo, hi): (f64, f64), step: f64) -> f64 {
    let s = if step > 0.0 { origin + ((v - origin) / step).round() * step } else { v };
    // rid of the dust that multiplying by a step like 0.01 leaves behind
    let s = (s * 1e9).round() / 1e9;
    if s.is_nan() { lo } else { s.clamp(lo, hi) }
}

struct Rail {
    left: f32,
    width: f32,
    lo: f64,
    hi: f64,
}
impl Rail {
    fn new(rect: Rect, lo: f64, hi: f64) -> Self {
        Self { left: rect.left() + HANDLE_W / 2.0, width: (rect.width() - HANDLE_W).max(1.0), lo, hi }
    }
    /// Where along the slot a value sits.
    fn at(&self, v: f64) -> f32 {
        let f = if self.hi > self.lo { ((v - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0) } else { 0.0 };
        self.left + f as f32 * self.width
    }
    /// The value at a place along the slot.
    fn value(&self, x: f32) -> f64 {
        self.lo + ((x - self.left) / self.width).clamp(0.0, 1.0) as f64 * (self.hi - self.lo)
    }
}

/// How much less a slider moves than the pointer while shift is held, and while ctrl and shift are.
pub const FINE: f64 = 0.1;
pub const FINER: f64 = 0.01;
/// With ctrl and shift held the steps are this much smaller too, so that a value can be set between two
/// of the slider's own steps.
const FINER_STEPS: f64 = 0.01;

/// How the keys held while dragging change a slider: how far it moves for the pointer's travel, how fine
/// its steps are, and whether it may go past its ends (as far as its hard limits).
///
///   (none)            the handle goes to the pointer, on the slider's steps, within its range
///   shift             a tenth as fast, from where it was
///   ctrl+shift        a hundredth as fast, in steps a hundredth as large
///   alt               past either end, as fast as the pointer
///   alt+shift         past either end, a tenth as fast
///   ctrl+alt+shift    past either end, a hundredth as fast, in steps a hundredth as large
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Keys {
    pub speed: f64,
    pub finer: bool,
    pub past: bool,
}
impl Keys {
    pub fn of(m: &egui::Modifiers) -> Self {
        let finer = m.shift && m.ctrl;
        Self { speed: if finer { FINER } else if m.shift { FINE } else { 1.0 }, finer, past: m.alt }
    }
    /// The handle follows the pointer from where it was rather than jumping to it.
    fn relative(&self) -> bool {
        self.speed < 1.0 || self.past
    }
    fn step(&self, step: f64) -> f64 {
        if self.finer { step * FINER_STEPS } else { step }
    }
}

/// What hovering over a slider's value says.
pub const VALUE_TIP: &str = "click to type a value (enter keeps it, esc leaves it)\ndrag the slider with shift: finer · ctrl+shift: finer still · alt: past its ends";

/// What a slider remembers while it is dragged from where it was: where the pointer was when the keys
/// last changed, the value then, and the keys.
#[derive(Clone, Copy)]
struct Hold {
    x: f32,
    value: f64,
    keys: Keys,
}

/// What the pointer does to a slider (nothing is drawn): press anywhere on it to bring the handle there,
/// and drag; the keys held change that (`Keys`). `hard`: how far it may go past its ends with alt.
/// Returns what happened, where the handle now is, and whether the pointer is on it.
#[allow(clippy::too_many_arguments)]
fn slide(ui: &mut Ui, rect: Rect, id: Id, value: f64, lo: f64, hi: f64, step: f64, hard: (f64, f64)) -> (Moved, f32, bool) {
    let rail = Rail::new(rect, lo, hi);
    let resp = ui.interact(rect.expand2(egui::vec2(0.0, 2.0)), id, Sense::drag());
    let live = ui.is_enabled();
    let mut out = Moved::default();
    let hold = id.with("held finely");
    if live && (resp.dragged() || resp.drag_started()) {
        if let Some(p) = resp.interact_pointer_pos() {
            let keys = ui.input(|i| Keys::of(&i.modifiers));
            let v = if keys.relative() {
                // (when the keys change, the slider carries on from where it is, without a jump)
                let from = ui.data(|d| d.get_temp::<Hold>(hold)).filter(|h| h.keys == keys).unwrap_or(Hold { x: p.x, value, keys });
                ui.data_mut(|d| d.insert_temp(hold, from));
                // (without alt: within the range, though a value already past an end may stay there)
                let limits = if keys.past { (hard.0.min(lo), hard.1.max(hi)) } else { (lo.min(from.value), hi.max(from.value)) };
                snap_from(from.value + (p.x - from.x) as f64 / rail.width as f64 * (hi - lo) * keys.speed, lo, limits, keys.step(step))
            } else {
                ui.data_mut(|d| d.remove::<Hold>(hold));
                snap(rail.value(p.x), lo, hi, step)
            };
            out.finer = keys.finer;
            if v != value { out.value = Some(v); }
        }
    } else if resp.drag_stopped() {
        ui.data_mut(|d| d.remove::<Hold>(hold));
    }
    out.released = live && resp.drag_stopped();
    let x = rail.at(out.value.unwrap_or(value));
    (out, x, live && (resp.hovered() || resp.dragged()))
}

/// A slider: a rail filled up to its handle. With alt it may be taken past its ends by twice its width
/// (`gf_core::limits::widen`); a knob's row gives it the knob's own hard limits instead (`slider_to`).
pub fn slider(ui: &mut Ui, rect: Rect, id: Id, value: f64, lo: f64, hi: f64, step: f64, accent: Color32) -> Moved {
    slider_to(ui, rect, id, value, lo, hi, step, accent, gf_core::limits::widen(lo, hi))
}

/// The same, with the hard limits it may be taken to with alt.
#[allow(clippy::too_many_arguments)]
pub fn slider_to(ui: &mut Ui, rect: Rect, id: Id, value: f64, lo: f64, hi: f64, step: f64, accent: Color32, hard: (f64, f64)) -> Moved {
    let (out, x, hot) = slide(ui, rect, id, value, lo, hi, step, hard);
    let y = rect.center().y;
    let p = ui.painter();
    p.rect_filled(Rect::from_min_max(Pos2::new(rect.left(), y - 1.5), Pos2::new(rect.right(), y + 1.5)), 2, white(0.16));
    p.rect_filled(Rect::from_min_max(Pos2::new(rect.left(), y - 1.5), Pos2::new(x, y + 1.5)), 2, fade(accent, 0.8));
    p.circle_filled(Pos2::new(x, y), if hot { 5.5 } else { 5.0 }, if hot { BRIGHT } else { accent });
    out
}

/// How many digits to show of a value: as many as its slider's steps have, more (up to four more) when it
/// lies between them (set in the finest steps, or typed).
pub fn digits_for(value: f64, digits: usize) -> usize {
    let mut d = digits;
    while d < digits + 4 {
        let s = value * 10f64.powi(d as i32);
        if !s.is_finite() || (s - s.round()).abs() < 1e-6 * s.abs().max(1.0) { break; }
        d += 1;
    }
    d
}

/// What typing into a value remembers: the text so far, and whether it has been given the keyboard yet.
#[derive(Clone)]
struct Typing {
    text: String,
    focused: bool,
}

/// A slider's value, written at the right of `rect`, with `digits` decimals if they fit (else fewer, but
/// never fewer than `least`: the slider's own). Past the slider's own ends (`lo`..`hi`) it is written
/// in `PAST`, so that a setting taken out of bounds shows that it is. Click it to type a value instead:
/// enter keeps it, esc (or a click elsewhere) leaves the value as it was. Returns a value typed and kept;
/// the caller holds it to the hard limits.
#[allow(clippy::too_many_arguments)]
pub fn value_text(ui: &mut Ui, rect: Rect, id: Id, value: f64, digits: usize, least: usize, lo: f64, hi: f64, size: f32) -> Option<f64> {
    let key = id.with("typing");
    let mut d = digits.max(least);
    let mut shown = format!("{value:.d$}");
    while d > least && text_width(ui, &shown, &mono(size)) > rect.width() {
        d -= 1;
        shown = format!("{value:.d$}");
    }
    if let Some(mut t) = ui.data(|d| d.get_temp::<Typing>(key)) {
        // (the field reaches over the end of the slider: a typed number may be longer than a shown one)
        let field = Rect::from_min_max(Pos2::new(rect.right() - 74.0, rect.center().y - 9.0), Pos2::new(rect.right() + 2.0, rect.center().y + 9.0));
        let edit_id = id.with("field");
        let edit = egui::TextEdit::singleline(&mut t.text).id(edit_id).font(mono(size)).text_color(BRIGHT).margin(egui::Margin::symmetric(4, 1)).align(Align2::RIGHT_CENTER).desired_width(field.width());
        let resp = ui.put(field, edit);
        if !t.focused {
            // (the whole of the old value chosen, so that what is typed takes its place)
            resp.request_focus();
            if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), edit_id) {
                state.cursor.set_char_range(Some(egui::text_selection::CCursorRange::select_all_str(&t.text)));
                egui::TextEdit::store_state(ui.ctx(), edit_id, state);
            }
            t.focused = true;
        }
        let (enter, esc) = ui.input(|i| (i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Escape)));
        if resp.lost_focus() || esc || (enter && !resp.has_focus()) {
            ui.data_mut(|d| d.remove::<Typing>(key));
            if enter && !esc {
                return t.text.trim().parse::<f64>().ok().filter(|v| v.is_finite());
            }
            return None;
        }
        ui.data_mut(|d| d.insert_temp(key, t));
        return None;
    }
    let past = value < lo - 1e-12 || value > hi + 1e-12;
    let resp = ui.interact(rect, id.with("value"), Sense::click());
    let hot = ui.is_enabled() && resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::Text); }
    let ink = if past { PAST } else if hot { Color32::WHITE } else { BRIGHT };
    text_in(ui, rect, Align2::RIGHT_CENTER, &shown, mono(size), ink);
    let tip = if past { format!("past this slider's ends ({lo} to {hi})\n{VALUE_TIP}") } else { VALUE_TIP.to_owned() };
    let resp = resp.on_hover_text(tip);
    if ui.is_enabled() && resp.clicked() {
        ui.data_mut(|d| d.insert_temp(key, Typing { text: shown, focused: false }));
    }
    None
}

/// A colour of full brightness from its hue (degrees) and how far it is from white (0 = white, 1 = the
/// pure hue): red, green, blue, the largest of them 1.
pub fn hue_rgb(hue: f64, strength: f64) -> [f64; 3] {
    let h = hue.rem_euclid(360.0) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let pure = match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    };
    let s = strength.clamp(0.0, 1.0);
    pure.map(|c| 1.0 - s * (1.0 - c))
}

/// The other way: the hue (degrees), the strength and the brightness (its largest part) of a colour.
/// White and grey have no hue, and black has no strength either: None for what the colour does not say.
pub fn rgb_hue(c: [f64; 3]) -> (Option<f64>, Option<f64>, f64) {
    let (hi, lo) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    if hi <= 0.0 { return (None, None, 0.0); }
    let d = hi - lo;
    if d <= hi * 1e-9 { return (None, Some(0.0), hi); }
    let h = if hi == c[0] { ((c[1] - c[2]) / d).rem_euclid(6.0) } else if hi == c[1] { (c[2] - c[0]) / d + 2.0 } else { (c[0] - c[1]) / d + 4.0 };
    (Some(h * 60.0), Some(d / hi), hi)
}

/// A colour given as light (not as bytes for a screen), as the screen shows it.
pub fn shown_colour(c: [f64; 3]) -> Color32 {
    Color32::from(egui::Rgba::from_rgb(c[0].clamp(0.0, 1.0) as f32, c[1].clamp(0.0, 1.0) as f32, c[2].clamp(0.0, 1.0) as f32))
}

/// A slider of hues: its rail is the hues themselves, at the strength given.
pub fn hue_slider(ui: &mut Ui, rect: Rect, id: Id, hue: f64, strength: f64) -> Moved {
    let (out, x, hot) = slide(ui, rect, id, hue, 0.0, 360.0, 1.0, (0.0, 360.0));
    let y = rect.center().y;
    let p = ui.painter();
    let (left, width, parts) = (rect.left() + HANDLE_W / 2.0, rect.width() - HANDLE_W, 36);
    for i in 0..parts {
        let (a, b) = (left + width * i as f32 / parts as f32, left + width * (i + 1) as f32 / parts as f32);
        let ink = shown_colour(hue_rgb((i as f64 + 0.5) / parts as f64 * 360.0, strength.max(0.35)));
        p.rect_filled(Rect::from_min_max(Pos2::new(a, y - 2.0), Pos2::new(b + 0.5, y + 2.0)), 0, fade(ink, if ui.is_enabled() { 0.9 } else { 0.4 }));
    }
    p.circle_filled(Pos2::new(x, y), if hot { 6.0 } else { 5.5 }, if hot { Color32::WHITE } else { BRIGHT });
    p.circle_filled(Pos2::new(x, y), 3.5, shown_colour(hue_rgb(out.value.unwrap_or(hue), strength)));
    out
}

/// A bar of a colour, which opens whatever sets that colour: brighter round its edge while open or under
/// the pointer, with the little arrow that says "this opens" at its end.
pub fn colour_bar(ui: &mut Ui, rect: Rect, id: Id, colour: Color32, open: bool) -> Response {
    let resp = ui.interact(rect, id, Sense::click());
    let hot = ui.is_enabled() && resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    ui.painter().rect(rect, 4, colour, Stroke::new(1.0, white(if hot || open { 0.55 } else { 0.22 })), StrokeKind::Outside);
    // the arrow is dark on a pale colour and pale on a dark one
    let light = 0.299 * colour.r() as f32 + 0.587 * colour.g() as f32 + 0.114 * colour.b() as f32;
    let ink = if light > 140.0 { Color32::from_black_alpha(190) } else { white(0.8) };
    let c = Pos2::new(rect.right() - 9.0, rect.center().y + 0.5);
    let arrow = if open {
        vec![c + egui::vec2(-3.5, 2.0), c + egui::vec2(0.0, -2.5), c + egui::vec2(3.5, 2.0)]
    } else {
        vec![c + egui::vec2(-3.5, -2.0), c + egui::vec2(3.5, -2.0), c + egui::vec2(0.0, 2.5)]
    };
    ui.painter().add(Shape::convex_polygon(arrow, ink, Stroke::NONE));
    resp
}

/// A small button that stands for one thing of several by its number and its colour (a light): lit when
/// it is the one chosen.
pub fn chip(ui: &mut Ui, rect: Rect, id: Id, text: &str, colour: Color32, on: bool) -> Response {
    let (resp, ink) = button_box(ui, rect, id, on);
    text_in(ui, Rect::from_min_max(rect.min, Pos2::new(rect.right(), rect.bottom() - 3.0)), Align2::CENTER_CENTER, text, mono(SIZE), ink);
    let bar = Rect::from_min_max(Pos2::new(rect.left() + 4.0, rect.bottom() - 5.0), Pos2::new(rect.right() - 4.0, rect.bottom() - 2.5));
    ui.painter().rect_filled(bar, 1, colour);
    resp
}

/// What a range says happened to it this frame.
#[derive(Default)]
pub struct Stops {
    /// new stops, if either was moved
    pub moved: Option<(f64, f64)>,
    pub released: bool,
}

/// What a range remembers while part of it is being dragged.
#[derive(Clone, Copy)]
struct Grip {
    /// where the pointer took hold
    x: f32,
    from: f64,
    to: f64,
    keys: Keys,
}

/// The slider of an animated knob: a handle for each stop it travels between, the stretch between them
/// (drag it to move both, kept inside the slider's range), and a mark at the present value.
/// The keys held change it as they do a slider (`Keys`): with alt the stops may go past its ends, to `hard`.
#[allow(clippy::too_many_arguments)]
pub fn range(ui: &mut Ui, rect: Rect, id: Id, from: f64, to: f64, now: f64, lo: f64, hi: f64, step: f64, digits: usize, hard: (f64, f64)) -> Stops {
    let rail = Rail::new(rect, lo, hi);
    let live = ui.is_enabled();
    let (xa, xb, y) = (rail.at(from), rail.at(to), rect.center().y);
    let handle = |x: f32| Rect::from_center_size(Pos2::new(x, y), egui::vec2(HANDLE_W, 12.0));
    let stretch = Rect::from_min_max(Pos2::new(xa.min(xb), y - 3.5), Pos2::new(xa.max(xb), y + 3.5));
    // later ones lie on top: the second handle wins where the two overlap, and both win over the stretch
    let parts = [
        (ui.interact(stretch.expand2(egui::vec2(0.0, 2.5)), id.with("both"), Sense::drag()), 2usize),
        (ui.interact(handle(xa), id.with("from"), Sense::drag()), 0),
        (ui.interact(handle(xb), id.with("to"), Sense::drag()), 1),
    ];
    let mut out = Stops::default();
    let fmt = |v: f64| format!("{v:.digits$}");
    let mut hot = [false; 3];
    for (resp, which) in parts {
        let resp = if !live {
            resp
        } else {
            match which {
                0 => resp.on_hover_text(format!("from {}", fmt(from))),
                1 => resp.on_hover_text(format!("to {}", fmt(to))),
                _ => resp.on_hover_text(format!("{} to {}: drag to move both", fmt(from), fmt(to))),
            }
        };
        if !live { continue; }
        hot[which] = resp.hovered() || resp.dragged();
        if hot[which] {
            ui.ctx().set_cursor_icon(if which < 2 { CursorIcon::ResizeHorizontal } else if resp.dragged() { CursorIcon::Grabbing } else { CursorIcon::Grab });
        }
        let Some(p) = resp.interact_pointer_pos() else {
            if resp.drag_stopped() { out.released = true; }
            continue;
        };
        let keys = ui.input(|i| Keys::of(&i.modifiers));
        // (taken hold of again whenever the keys change, so that the stops carry on without a jump)
        if resp.drag_started() || ui.data(|d| d.get_temp::<Grip>(id)).is_some_and(|g| g.keys != keys) {
            ui.data_mut(|d| d.insert_temp(id, Grip { x: p.x, from, to, keys }));
        }
        if resp.dragged() {
            if let Some(g) = ui.data(|d| d.get_temp::<Grip>(id)) {
                let moved = (p.x - g.x) as f64 / rail.width as f64 * (hi - lo) * keys.speed;
                // (within the slider's range, or past it with alt; a stop already past an end may stay there)
                let (a, b) = if keys.past { (hard.0.min(lo), hard.1.max(hi)) } else { (lo.min(g.from).min(g.to), hi.max(g.from).max(g.to)) };
                let st = keys.step(step);
                let new = match which {
                    0 => (snap_from(g.from + moved, lo, (a, b), st), to),
                    1 => (from, snap_from(g.to + moved, lo, (a, b), st)),
                    _ => {
                        let d = moved.clamp(a - g.from.min(g.to), b - g.from.max(g.to));
                        (g.from + d, g.to + d)
                    }
                };
                if new != (from, to) { out.moved = Some(new); }
            }
        }
        if resp.drag_stopped() { out.released = true; }
    }
    let (from, to) = out.moved.unwrap_or((from, to));
    let (xa, xb) = (rail.at(from), rail.at(to));
    let p = ui.painter();
    p.rect_filled(Rect::from_min_max(Pos2::new(rect.left(), y - 1.5), Pos2::new(rect.right(), y + 1.5)), 2, white(0.16));
    p.rect_filled(Rect::from_min_max(Pos2::new(xa.min(xb), y - 3.5), Pos2::new(xa.max(xb), y + 3.5)), 2, fade(ACCENT, if hot[2] { 0.75 } else { 0.55 }));
    p.rect_filled(handle(xa), 3, if hot[0] { Color32::WHITE } else { BRIGHT });
    p.rect_filled(handle(xb), 3, if hot[1] { ACCENT } else { HANDLE_TO });
    let xn = rail.at(now);
    p.rect_filled(Rect::from_min_max(Pos2::new(xn - 1.0, y - 6.0), Pos2::new(xn + 1.0, y + 6.0)), 1, Color32::WHITE);
    out
}

/// The title of a group of controls; clicking it folds the group away or brings it back. `says`: a few
/// words at the right of the title (what a folded group is set to, so that it need not be opened to see).
/// Returns true when clicked, and where the title is.
pub fn group_title(ui: &mut Ui, id: Id, title: &str, closed: bool, says: &str) -> (bool, Rect) {
    gap(ui, 9.0);
    let r = row(ui, 15.0);
    let resp = ui.interact(r, id, Sense::click());
    let hot = resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let ink = if hot { BRIGHT } else { FAINT };
    let c = Pos2::new(r.left() + 3.5, r.center().y);
    let arrow = if closed {
        vec![c + egui::vec2(-1.5, -3.5), c + egui::vec2(2.5, 0.0), c + egui::vec2(-1.5, 3.5)]
    } else {
        vec![c + egui::vec2(-3.5, -1.5), c + egui::vec2(3.5, -1.5), c + egui::vec2(0.0, 2.5)]
    };
    ui.painter().add(Shape::convex_polygon(arrow, ink, Stroke::NONE));
    let g = spaced(ui, title, 10.0, ink);
    let at = Pos2::new(r.left() + 13.0, r.center().y - g.size().y / 2.0);
    let end = at.x + g.size().x;
    ui.painter().galley(at, g, ink);
    if !says.is_empty() {
        let room = Rect::from_min_max(Pos2::new(end + 12.0, r.top()), r.max);
        text_in(ui, room, Align2::RIGHT_CENTER, says, mono(10.0), if hot { DIM } else { HINT });
    }
    gap(ui, 3.0);
    (resp.clicked(), r)
}

/// The heading of a group inside a panel's section: small lowercase words, the same as the camera's
/// headings always were. It does not fold.
pub fn group_heading(ui: &mut Ui, text: &str) -> Rect {
    gap(ui, 6.0);
    let r = row(ui, 14.0);
    text_in(ui, r, Align2::LEFT_CENTER, text, mono(10.0), FAINT);
    gap(ui, 1.0);
    r
}

/// The heading of something that folds below a section: a group in a floating window, or a sub-group.
/// The small arrow and the lowercase words of the joins' rows; `says` at the right. Returns true when
/// pressed, and where it is.
pub fn fold_heading(ui: &mut Ui, id: Id, title: &str, closed: bool, says: &str) -> (bool, Rect) {
    gap(ui, 4.0);
    let r = row(ui, 17.0);
    let resp = ui.interact(r, id, Sense::click());
    let hot = resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let ink = if hot { BRIGHT } else { FAINT };
    let c = Pos2::new(r.left() + 4.5, r.center().y);
    let tip = if closed {
        vec![c + egui::vec2(-1.3, -3.0), c + egui::vec2(2.3, 0.0), c + egui::vec2(-1.3, 3.0)]
    } else {
        vec![c + egui::vec2(-3.0, -1.5), c + egui::vec2(3.0, -1.5), c + egui::vec2(0.0, 2.2)]
    };
    ui.painter().add(Shape::convex_polygon(tip, ink, Stroke::NONE));
    let w = text_width(ui, title, &mono(10.5));
    text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 13.0, r.top()), r.max), Align2::LEFT_CENTER, title, mono(10.5), ink);
    if !says.is_empty() {
        text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 13.0 + w + 12.0, r.top()), r.max), Align2::RIGHT_CENTER, says, mono(10.0), if hot { DIM } else { HINT });
    }
    gap(ui, 1.0);
    (resp.clicked(), r)
}

/// A title inside a floating window, over the controls that belong together (it does not fold). `says`:
/// a few words at its right.
pub fn section(ui: &mut Ui, title: &str, says: &str) -> Rect {
    gap(ui, 9.0);
    let r = row(ui, 15.0);
    let g = spaced(ui, title, 10.0, FAINT);
    let end = r.left() + g.size().x;
    ui.painter().galley(Pos2::new(r.left(), r.center().y - g.size().y / 2.0), g, FAINT);
    if !says.is_empty() {
        text_in(ui, Rect::from_min_max(Pos2::new(end + 12.0, r.top()), r.max), Align2::RIGHT_CENTER, says, mono(10.0), HINT);
    }
    gap(ui, 3.0);
    r
}

/// A bar down the left of a block (the equations of a shape, the extra rows of an animated slider).
pub fn side_bar(ui: &Ui, x: f32, top: f32, bottom: f32, alpha: f32) {
    ui.painter().rect_filled(Rect::from_min_max(Pos2::new(x, top), Pos2::new(x + 2.0, bottom)), 0, fade(ACCENT, alpha));
}

/// A path with its beginning cut off if it does not fit in `width`: the end is the part worth reading.
pub fn tail(ui: &Ui, text: &str, font: &FontId, width: f32) -> String {
    if text_width(ui, text, font) <= width { return text.to_owned(); }
    let chars: Vec<char> = text.chars().collect();
    for start in 1..chars.len() {
        let s: String = std::iter::once('…').chain(chars[start..].iter().copied()).collect();
        if text_width(ui, &s, font) <= width { return s; }
    }
    String::new()
}

/// The small cross that closes a floating window.
fn close_button(ui: &mut Ui, rect: Rect, id: Id) -> Response {
    let resp = ui.interact(rect, id, Sense::click());
    let hot = resp.hovered();
    if hot {
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
        ui.painter().rect_filled(rect, BUTTON_RADIUS, white(0.1));
    }
    let (c, d, ink) = (rect.center(), 3.5, if hot { Color32::WHITE } else { DIM });
    for s in [1.0, -1.0] {
        ui.painter().line_segment([c + egui::vec2(-d, -d * s), c + egui::vec2(d, d * s)], Stroke::new(1.3, ink));
    }
    resp
}

/// What came of showing a floating window.
pub struct Floated {
    /// its cross was pressed
    pub closed: bool,
    /// where it is, and where its title (the part to drag it by) and its cross are
    pub rect: Rect,
    pub bar: Rect,
    pub cross: Rect,
}

/// A window that floats over the picture and the panels: a title, a cross to close it, and whatever
/// `body` puts in it. It is dragged by its title or by any part of it that is not a control, stays inside
/// the program's window, and scrolls if it is taller than `max_h`. `first`: where its top left corner is
/// when it first appears in a run (after that it stays where it is put).
pub fn floating(ctx: &egui::Context, name: &str, title: &str, width: f32, first: Pos2, max_h: f32, body: impl FnOnce(&mut Ui)) -> Floated {
    let frame = egui::Frame::new()
        .fill(float_fill())
        .stroke(Stroke::new(1.0, white(0.16)))
        .corner_radius(PANEL_RADIUS)
        .shadow(egui::Shadow { offset: [0, 6], blur: 18, spread: 0, color: Color32::from_black_alpha(110) })
        .inner_margin(egui::Margin { left: 12, right: 2, top: 9, bottom: 10 });
    let id = Id::unique(("floating", name));
    let (mut bar, mut cross, mut closed) = (Rect::ZERO, Rect::ZERO, false);
    let shown = egui::Area::new(id).order(egui::Order::Foreground).movable(true).constrain(true).fade_in(false).default_pos(first).show(ctx, |ui| {
        frame.show(ui, |ui| {
            ui.set_width(width + 10.0);
            let head = Rect::from_min_size(ui.cursor().min, egui::vec2(width, 18.0));
            ui.scope_builder(egui::UiBuilder::new().max_rect(head), |ui| {
                bar = row(ui, 18.0);
                let g = spaced(ui, title, 10.5, BRIGHT);
                ui.painter().galley(Pos2::new(bar.left(), bar.center().y - g.size().y / 2.0), g, BRIGHT);
                cross = Rect::from_center_size(Pos2::new(bar.right() - 8.0, bar.center().y), egui::vec2(18.0, 18.0));
                closed = close_button(ui, cross, id.with("close")).on_hover_text("close").clicked();
            });
            gap(ui, 2.0);
            // (the room is given outright: left to itself, a floating thing is offered only the height it had last frame)
            let space = Rect::from_min_size(ui.cursor().min, egui::vec2(width + 10.0, max_h));
            ui.scope_builder(egui::UiBuilder::new().max_rect(space), |ui| {
                egui::ScrollArea::vertical().id_salt((name, "scroll")).max_height(max_h).auto_shrink([false, true]).show(ui, |ui| {
                    ui.set_width(width);
                    body(ui);
                });
            });
        });
    });
    Floated { closed, rect: shown.response.rect, bar, cross }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slider_takes_only_whole_steps_inside_its_range() {
        assert_eq!(snap(1.234, 1.05, 2.4, 0.01), 1.23);
        assert_eq!(snap(0.3, 0.0, 0.4, 0.005), 0.3);
        assert_eq!(snap(0.0123, 0.0, 0.03, 0.0005), 0.0125);
        assert_eq!(snap(9.9, 1.0, 8.0, 1.0), 8.0);
        assert_eq!(snap(-3.0, 1.0, 8.0, 1.0), 1.0);
        // no dust from the arithmetic: the value prints as the step would have it
        assert_eq!(format!("{}", snap(0.1 + 0.2, 0.0, 1.0, 0.1)), "0.3");
    }

    #[test]
    fn a_value_sits_where_its_handle_is() {
        let rail = Rail::new(Rect::from_min_size(Pos2::new(100.0, 0.0), egui::vec2(109.0, 12.0)), 1.0, 3.0);
        assert_eq!(rail.at(1.0), 104.5);
        assert_eq!(rail.at(3.0), 204.5);
        assert_eq!(rail.at(9.0), 204.5);
        assert!((rail.value(154.5) - 2.0).abs() < 1e-9);
        assert_eq!(rail.value(0.0), 1.0);
    }

    #[test]
    fn a_colour_is_its_hue_its_strength_and_its_brightness() {
        assert_eq!(hue_rgb(0.0, 1.0), [1.0, 0.0, 0.0]);
        assert_eq!(hue_rgb(120.0, 1.0), [0.0, 1.0, 0.0]);
        assert_eq!(hue_rgb(240.0, 0.5), [0.5, 0.5, 1.0]);
        assert_eq!(hue_rgb(77.0, 0.0), [1.0, 1.0, 1.0]);
        // there and back, for the lights the families have and a few more
        for c in [[5.0, 4.9, 4.7], [0.5, 1.4, 4.0], [5.0, 1.6, 0.25], [0.8, 0.9, 1.1], [0.006, 0.011, 0.022], [0.0, 3.0, 0.0], [2.0, 0.0, 2.0]] {
            let (h, s, v) = rgb_hue(c);
            let back = hue_rgb(h.unwrap(), s.unwrap()).map(|x| x * v);
            assert!(c.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-12), "{c:?} came back as {back:?}");
        }
        // what a colour does not say is not made up
        assert_eq!(rgb_hue([3.0, 3.0, 3.0]), (None, Some(0.0), 3.0));
        assert_eq!(rgb_hue([0.0, 0.0, 0.0]), (None, None, 0.0));
        // a hue past the end of the circle is the same hue
        assert_eq!(hue_rgb(360.0, 1.0), hue_rgb(0.0, 1.0));
        assert_eq!(hue_rgb(-120.0, 1.0), hue_rgb(240.0, 1.0));
    }

    #[test]
    fn buttons_wrap_onto_as_few_lines_as_hold_them() {
        // four families' names as buttons, in a panel 262 wide
        let w = [81.0, 49.0, 120.0, 133.0];
        assert_eq!(lines_of(&w, 262.0, 4.0), vec![vec![0, 1, 2], vec![3]]);
        // evened out: two and two rather than three and one
        assert_eq!(even_lines(&w, 262.0, 4.0), vec![vec![0, 1], vec![2, 3]]);
        // seven: three lines either way, and no line left with a single short button
        let w = [81.0, 49.0, 120.0, 133.0, 43.0, 56.0, 49.0];
        assert_eq!(lines_of(&w, 262.0, 4.0).len(), 3);
        let even = even_lines(&w, 262.0, 4.0);
        assert_eq!(even.len(), 3);
        assert!(even.iter().all(|line| line.len() >= 2), "{even:?}");
        assert_eq!(even.concat(), (0..7).collect::<Vec<_>>());
        // one that is wider than the panel still gets a line of its own, and nothing is lost
        assert_eq!(even_lines(&[300.0, 40.0], 262.0, 4.0), vec![vec![0], vec![1]]);
        assert!(even_lines(&[], 262.0, 4.0).is_empty());
        assert_eq!(even_lines(&[50.0], 262.0, 4.0), vec![vec![0]]);
    }
}
