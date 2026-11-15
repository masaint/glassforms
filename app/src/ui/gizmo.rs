//! The little ball in the light room: the family's lights as dots on a sphere seen from the camera,
//! with the sphere's horizon and one meridian. Dragging it turns the lighting.

use eframe::egui::{epaint::Mesh, Color32, CursorIcon, Id, Pos2, Rect, Sense, Shape, Stroke, Ui};
use gf_core::math::{apply, axis_angle, mul, DEG, TAU};
use gf_core::state::State;

use super::theme::fade;

/// Width and height of the ball's square, in points.
pub const SIZE: f32 = 92.0;

/// Turn the lights as dragging the ball by (dx, dy) points does.
pub fn drag(st: &mut State, dx: f64, dy: f64) {
    st.turn_lights(&mul(&axis_angle(0.0, 1.0, 0.0, dx * 0.02), &axis_angle(1.0, 0.0, 0.0, dy * 0.02)));
}

fn mix(a: [f32; 4], b: [f32; 4], t: f32) -> Color32 {
    let c = |i: usize| a[i] + (b[i] - a[i]) * t;
    Color32::from_rgba_unmultiplied(c(0) as u8, c(1) as u8, c(2) as u8, (c(3) * 255.0) as u8)
}

/// The ball itself: lit from the upper left, darker toward its rim.
fn ball(c: Pos2, r: f32) -> Mesh {
    // the shading runs from a small circle up and to the left of the middle out to the rim
    let (f, r0) = (Pos2::new(c.x - 0.35 * r, c.y - 0.4 * r), 0.1 * r);
    let (inner, outer) = ([120.0, 132.0, 165.0, 0.38], [20.0, 23.0, 34.0, 0.75]);
    let shade = |p: Pos2| -> Color32 {
        let (cd, pd, dr) = (c - f, p - f, r - r0);
        let (a, b, k) = (cd.dot(cd) - dr * dr, pd.dot(cd) + r0 * dr, pd.dot(pd) - r0 * r0);
        let t = if k <= 0.0 { 0.0 } else { ((b - (b * b - a * k).max(0.0).sqrt()) / a).clamp(0.0, 1.0) };
        mix(inner, outer, t)
    };
    let (rings, spokes) = (12u32, 56u32);
    let mut m = Mesh::default();
    for j in 0..=rings + 1 {
        // one last ring a point outside the rim, fully clear, softens the edge
        let rr = if j <= rings { r * j as f32 / rings as f32 } else { r + 1.0 };
        for i in 0..spokes {
            let a = i as f32 / spokes as f32 * std::f32::consts::TAU;
            let p = Pos2::new(c.x + rr * a.cos(), c.y + rr * a.sin());
            m.colored_vertex(p, if j <= rings { shade(p) } else { Color32::TRANSPARENT });
        }
    }
    for j in 0..=rings {
        for i in 0..spokes {
            let (a, b) = (j * spokes + i, j * spokes + (i + 1) % spokes);
            let (c2, d) = (a + spokes, b + spokes);
            m.add_triangle(a, b, c2);
            m.add_triangle(b, d, c2);
        }
    }
    m
}

/// Draw the ball in `rect` and let it be dragged. `chosen`: the light to pick out with a white ring (none:
/// the lights are not lit, and are drawn faintly).
pub fn light_ball(ui: &mut Ui, st: &mut State, rect: Rect, chosen: Option<usize>) {
    let resp = ui.interact(rect, Id::unique("light ball"), Sense::drag());
    if ui.is_enabled() {
        if resp.dragged() {
            let d = resp.drag_delta();
            drag(st, d.x as f64, d.y as f64);
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        } else if resp.hovered() {
            ui.ctx().set_cursor_icon(CursorIcon::Grab);
        }
    }
    resp.on_hover_text("Drag the ball to turn all the lighting. Dots are lights on your side of the shape, rings are behind it; the chosen light has a white ring.");

    let (c, r) = (rect.center(), SIZE / 2.0 - SIZE * 0.07);
    let m = st.light_m();
    let p = ui.painter();
    p.add(Shape::mesh(ball(c, r)));
    p.circle_stroke(c, r, Stroke::new(1.0, Color32::from_white_alpha(56)));
    // a circle on the sphere: the part on our side drawn strongly, the part behind faintly
    let ring = |f: &dyn Fn(f64) -> [f64; 3], col: Color32| {
        for front in [false, true] {
            let mut run: Vec<Pos2> = Vec::new();
            let flush = |run: &mut Vec<Pos2>| {
                if run.len() > 1 { p.add(Shape::line(std::mem::take(run), Stroke::new(1.0, fade(col, if front { 0.7 } else { 0.2 })))); }
                run.clear();
            };
            for i in 0..=72 {
                let v = apply(&m, f(i as f64 / 72.0 * TAU));
                if (v[2] > 0.0) != front {
                    flush(&mut run);
                    continue;
                }
                run.push(Pos2::new(c.x + v[0] as f32 * r, c.y - v[1] as f32 * r));
            }
            flush(&mut run);
        }
    };
    ring(&|a| [a.cos(), 0.0, a.sin()], Color32::from_rgb(0x9f, 0xb0, 0xd8));      // the lights' horizon
    ring(&|a| [0.0, a.cos(), a.sin()], Color32::from_rgb(0x6f, 0x7c, 0x9c));      // a meridian, to show roll
    let mut dots: Vec<([f64; 3], Color32, f32, bool)> = st
        .rig()
        .lights
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let ce = (l.el * DEG).cos();
            let v = apply(&m, [(l.az * DEG).sin() * ce, (l.el * DEG).sin(), -(l.az * DEG).cos() * ce]);
            let top = l.rgb[0].max(l.rgb[1]).max(l.rgb[2]).max(1e-6);
            let ch = |x: f64| (255.0 * (x / top).powf(0.6)).round() as u8;
            let ink = Color32::from_rgb(ch(l.rgb[0]), ch(l.rgb[1]), ch(l.rgb[2]));
            (v, if chosen.is_some() { ink } else { fade(ink, 0.3) }, (2.2 + 0.2 * (l.w * l.h).sqrt() as f32).min(8.0), chosen == Some(i))
        })
        .collect();
    dots.sort_by(|a, b| a.0[2].total_cmp(&b.0[2]));
    for (v, col, size, picked) in dots {
        let at = Pos2::new(c.x + v[0] as f32 * r, c.y - v[1] as f32 * r);
        if v[2] > 0.0 {
            p.circle_filled(at, size, col);
        } else {
            p.circle_stroke(at, size, Stroke::new(1.3, fade(col, 0.6)));
        }
        if picked { p.circle_stroke(at, size + 2.0, Stroke::new(1.0, Color32::WHITE)); }
    }
}
