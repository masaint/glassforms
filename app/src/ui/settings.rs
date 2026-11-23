//! The settings: a small floating window, opened from the gear beside the "?" at the foot of the window.
//! It holds only what is a matter of taste rather than of the picture: the look (dark, light, or as
//! Windows has it), the panels' glass (how blurred the picture behind them is, and how much of it they
//! let through), and how large everything over the picture is drawn. All of
//! it is remembered in the store, none of it in a setup.

use eframe::egui::{self, Id, Pos2, Rect};

use super::knob;
use super::theme::*;
use super::widgets::*;
use super::Panels;

/// The looks, as the store names them and as the window shows them.
const LOOKS: [(&str, &str); 3] = [("dark", "dark"), ("light", "light"), ("system", "as windows")];
/// The sizes offered: none is as Windows sets it; the rest are so many times that.
pub const SCALES: [(Option<f32>, &str); 5] = [(None, "auto"), (Some(0.9), "90%"), (Some(1.1), "110%"), (Some(1.25), "125%"), (Some(1.5), "150%")];

const SETTINGS_W: f32 = 268.0;
const SETTINGS_LABEL: f32 = 62.0;

/// Show the settings if they are open. `full`: the inside of the program's window; `fixed_scale`: the
/// size was set on the command line, which wins over the one chosen here.
pub fn show(ctx: &egui::Context, px: &mut Panels, full: Rect, fixed_scale: bool) {
    if !px.store.settings_open { return; }
    // it first appears over the middle of the foot of the window, above the gear it was opened from
    let first = px.store.settings_at.map_or(Pos2::new(full.center().x - (SETTINGS_W + 24.0) / 2.0, (full.bottom() - 260.0).max(EDGE)), |p| Pos2::new(p[0], p[1]));
    let max_h = (full.height() - 2.0 * EDGE - 44.0).max(80.0);
    let out = floating(ctx, "settings", "settings", SETTINGS_W, first, max_h, |ui| {
        set_label_width(ui, SETTINGS_LABEL);
        px.place = "settings";
        gap(ui, 4.0);
        let now = LOOKS.iter().position(|(id, _)| *id == px.store.theme).unwrap_or(0);
        let names: Vec<&str> = LOOKS.iter().map(|(_, n)| *n).collect();
        let run = named_run(ui, Id::unique("settings look"), "look", &names.iter().enumerate().map(|(i, n)| Item::new(n, i == now)).collect::<Vec<_>>());
        for (i, r) in run.places.iter().enumerate() { px.note(format!("settings/look:{}", LOOKS[i].0), *r); }
        if let Some(i) = run.clicked { px.store.theme = LOOKS[i].0.into(); }

        // the panels' glass: two sliders in hundredths, each taking effect as it moves
        let (blur, opacity) = px.store.glass();
        let m = knob::value_row(ui, px, "blur", "blur", (blur * 100.0).round() as f64, 0.0, 100.0, 1.0);
        if let Some(v) = m.value { px.store.ui_blur = Some((v as f32 / 100.0).clamp(0.0, 1.0)); }
        let m = knob::value_row(ui, px, "opacity", "opacity", (opacity * 100.0).round() as f64, 0.0, 100.0, 1.0);
        if let Some(v) = m.value { px.store.ui_opacity = Some((v as f32 / 100.0).clamp(0.0, 1.0)); }
        let r = row(ui, 20.0);
        let says = if blur <= 0.0 { "plain panels: nothing behind them is blurred" } else { "the picture shows through the panels, blurred" };
        text_in(ui, Rect::from_min_max(r.min, egui::pos2(r.right() - 52.0, r.max.y)), egui::Align2::LEFT_CENTER, says, sans(SMALL), pal().faint);
        let at_default = (blur, opacity) == (BLUR_DEFAULT, OPACITY_DEFAULT);
        let reset = Rect::from_min_max(egui::pos2(r.right() - 48.0, r.top() + 2.0), egui::pos2(r.right(), r.bottom() - 2.0));
        let pressed = ui.scope(|ui| {
            if at_default { ui.disable(); }
            button(ui, reset, Id::unique("settings glass reset"), "reset", false, mono(SMALL)).on_hover_text("the glass as it first was: blur 75, opacity 50").clicked()
        }).inner;
        px.note("settings/glass:reset", reset);
        if pressed { (px.store.ui_blur, px.store.ui_opacity, px.store.frost) = (None, None, "full".into()); }
        gap(ui, 4.0);

        let now = SCALES.iter().position(|(s, _)| *s == px.store.ui_scale).unwrap_or(0);
        if fixed_scale { ui.disable(); }
        let items: Vec<Item> = SCALES.iter().enumerate().map(|(i, (_, n))| Item::new(n, i == now)).collect();
        let run = named_run(ui, Id::unique("settings scale"), "size", &items);
        for (i, r) in run.places.iter().enumerate() { px.note(format!("settings/size:{i}"), *r); }
        if let Some(i) = run.clicked { px.store.ui_scale = SCALES[i].0; }
        note(ui, if fixed_scale { "the size is set on the command line (--ui-scale) for this run" } else { "auto: as large as Windows draws everything else" });
    });
    if out.closed { px.store.settings_open = false; }
    px.store.settings_at = Some([out.rect.min.x.round(), out.rect.min.y.round()]);
    px.note("settings", out.rect);
    px.note("settings:close", out.cross);
}
