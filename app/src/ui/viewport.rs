//! How the matcap draws (gf_core::view_mode; the view mode itself is chosen in the tier row, ui/tiers.rs), with the
//! matcap's choice (a picker of the starter set, each shown as its own ball; the scene's material; a
//! picture), its light's turn, and what is drawn over it. Kept with the view, never with a material.

use eframe::egui::{self, Color32, Id, Rect, Sense, Stroke, Ui};
use gf_core::state::State;
use gf_core::view_mode::{shade, srgb, Builtin, MatcapSource, Tint};

use super::theme::*;
use super::widgets::*;
use super::Panels;

/// A built-in matcap on a ball, as a small picture for the picker (sRGB, the ball filling the square).
pub fn thumbnail(b: Builtin, side: usize) -> egui::ColorImage {
    let mut px = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            // four samples a pixel, for a smooth rim
            let mut acc = [0.0f64; 4];
            for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let nx = (x as f64 + ox) / side as f64 * 2.0 - 1.0;
                let ny = -((y as f64 + oy) / side as f64 * 2.0 - 1.0);
                let r2 = nx * nx + ny * ny;
                if r2 > 1.0 { continue; }
                let c = shade(b, [nx, ny, (1.0 - r2).sqrt()]);
                for i in 0..3 { acc[i] += srgb(c[i]); }
                acc[3] += 1.0;
            }
            let a = acc[3] / 4.0;
            let k = if acc[3] > 0.0 { 1.0 / acc[3] } else { 0.0 };
            let ch = |v: f64| (v * k * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            px.push(Color32::from_rgba_unmultiplied(ch(acc[0]), ch(acc[1]), ch(acc[2]), (a * 255.0 + 0.5) as u8));
        }
    }
    egui::ColorImage::new([side, side], px)
}

/// The matcap's choices, as the matcap tier's dropdown shows them (ui/tiers.rs): the picker of the starter set, the scene's
/// material or a picture, the colour, the light's turn and lift, what is drawn over it, and the smoothing of its edges.
pub fn matcap_settings(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    picker(ui, st, px);
    let m = &mut st.view.matcap;
    let pic_name = match &m.source {
        MatcapSource::Image(p) => format!("picture: {}", std::path::Path::new(p).file_name().map_or(p.clone(), |f| f.to_string_lossy().into_owned())),
        _ => "picture…".to_owned(),
    };
    let own = named_run(ui, Id::unique(("matcap own", px.place)), "or", &[
        Item::new("material", m.source == MatcapSource::Material).tip("approximate: the scene's material on a ball, lit by the studio's lamps and backdrop only (no floor, no lit sweep, no reflections of the scene: metals read darker than the render)"),
        Item::new(&pic_name, matches!(m.source, MatcapSource::Image(_))).tip("a picture of a lit ball filling a square (.png, .jpg, .hdr, .exr)"),
    ]);
    match own.clicked {
        Some(0) => m.source = MatcapSource::Material,
        Some(1) => px.requests.push(super::Request::LoadMatcap),
        _ => {}
    }
    if let Some(e) = &px.matcap_error { note(ui, e); }
    let tint = named_run(ui, Id::unique(("matcap tint", px.place)), "colour", &[
        Item::new("one", m.tint == Tint::None).tip("one colour for everything"),
        Item::new("by body", m.tint == Tint::Body).tip("each body its own colour"),
    ]);
    match tint.clicked { Some(0) => m.tint = Tint::None, Some(1) => m.tint = Tint::Body, _ => {} }
    for (name, tip, v, lo, hi, step) in [
        ("light turn", "the matcap's light turned round the view, degrees", &mut m.turn, -180.0, 180.0, 1.0),
        ("light lift", "the matcap's light raised or lowered over the top, degrees", &mut m.lift, -90.0, 90.0, 1.0),
        ("cavity", "creases and pits darkened", &mut m.cavity, 0.0, 1.0, 0.01),
        ("edges", "convex edges lifted, concave ones darkened", &mut m.edge, 0.0, 1.0, 0.01),
        ("outline", "lines where the depth jumps and around the silhouette", &mut m.outline, 0.0, 1.0, 0.01),
    ] {
        let r = row(ui, 22.0);
        label(ui, r, name);
        let s = beside_label(ui, r, 22.0);
        let moved = slider(ui, s, Id::unique(("matcap slider", name, px.place)), *v, lo, hi, step, pal().accent);
        if let Some(x) = moved.value { *v = x.clamp(lo, hi); }
        ui.interact(r, Id::unique(("matcap slider tip", name, px.place)), Sense::hover()).on_hover_text(tip);
    }
    let aa = named_run(ui, Id::unique(("matcap aa", px.place)), "smooth edges", &[
        Item::new("one frame", !m.aa).tip("one frame and done"),
        Item::new("8 frames", m.aa).tip("a still view is refined over eight jittered frames"),
    ]);
    match aa.clicked { Some(0) => m.aa = false, Some(1) => m.aa = true, _ => {} }
}

/// The starter set as balls, six to a row; the one in use ringed.
fn picker(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    const SIDE: f32 = 34.0;
    const GAP: f32 = 5.0;
    if px.matcap_thumbs.len() != Builtin::ALL.len() {
        px.matcap_thumbs = Builtin::ALL.iter().map(|b| {
            ui.ctx().load_texture(format!("matcap {b:?}"), thumbnail(*b, 64), egui::TextureOptions::LINEAR)
        }).collect();
    }
    let left = ui.cursor().left() + label_width(ui) + COL_GAP;
    let per_row = (((ui.available_width() - label_width(ui) - COL_GAP + GAP) / (SIDE + GAP)).floor() as usize).clamp(3, 6);
    let rows = Builtin::ALL.len().div_ceil(per_row);
    let r = row(ui, rows as f32 * (SIDE + GAP));
    label(ui, Rect::from_min_size(r.min, egui::vec2(label_width(ui), 22.0)), "matcap");
    let current = match st.view.matcap.source { MatcapSource::Builtin(b) => Some(b), _ => None };
    for (i, b) in Builtin::ALL.iter().enumerate() {
        let at = egui::pos2(left + (i % per_row) as f32 * (SIDE + GAP), r.top() + (i / per_row) as f32 * (SIDE + GAP));
        let cell = Rect::from_min_size(at, egui::vec2(SIDE, SIDE));
        let resp = ui.interact(cell, Id::unique(("matcap pick", i, px.place)), Sense::click()).on_hover_text(b.name());
        px.note(format!("{}matcap:{b:?}", px.whose()), cell);
        let p = ui.painter();
        p.image(px.matcap_thumbs[i].id(), cell, Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        if current == Some(*b) {
            p.circle_stroke(cell.center(), SIDE * 0.5 + 2.0, Stroke::new(1.5, pal().bright));
        } else if resp.hovered() {
            p.circle_stroke(cell.center(), SIDE * 0.5 + 2.0, Stroke::new(1.0, pal().dim));
        }
        if resp.clicked() { st.view.matcap.source = MatcapSource::Builtin(*b); }
    }
}
