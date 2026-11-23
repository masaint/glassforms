//! Freehand placement on a view-facing plane through the selected owner. Sampling
//! uses the current projection ray, including off-axis and fisheye cameras.
use super::{widgets::wide_button, Panels};
use eframe::egui::{self, Color32, Id, Pos2, Rect, Stroke, Ui};
use gf_core::{
    drawn_cut::{CutSpace, DrawnCut},
    math::{apply, transpose},
    modifier_pivot::{PivotFrame, PivotOwner, Point},
    mods::Kind,
    state::State,
};

#[derive(Default)]
pub struct Drawing {
    target: Option<(PivotOwner, usize)>,
    samples: Vec<Pos2>,
}

fn owner(st: &State) -> PivotOwner {
    st.construction_pivot_owner()
        .unwrap_or(PivotOwner::Object(st.stack.sel))
}

impl Drawing {
    pub fn arm(&mut self, st: &State, i: usize) {
        self.target = Some((owner(st), i));
        self.samples.clear();
    }
    /// The selected owner's modifier at `from` was moved to `to` (None: taken away): the armed
    /// cut follows its modifier, and is let go with it.
    pub fn list_moved(&mut self, st: &State, from: usize, to: Option<usize>) {
        let Some((target, i)) = self.target.filter(|(t, _)| *t == owner(st)) else { return };
        let left = if i > from { i - 1 } else { i };
        let mapped = if i == from { to } else { Some(to.map_or(left, |to| if left >= to { left + 1 } else { left })) };
        match mapped {
            Some(i) => self.target = Some((target, i)),
            None => *self = Self::default(),
        }
    }
    pub fn armed(&self) -> Option<usize> { self.target.map(|(_, i)| i) }
    pub fn gesture(
        &mut self,
        ctx: &egui::Context,
        picture: &egui::Response,
        st: &mut State,
    ) -> bool {
        let Some((target, i)) = self.target else {
            return false;
        };
        if !target.selected(st)
            || st
                .selected_modifiers()
                .list
                .get(i)
                .is_none_or(|m| m.kind != Kind::DrawCut)
        {
            *self = Self::default();
            return false;
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            *self = Self::default();
            return true;
        }
        let (pressed, down, released, at) = ctx.input(|i| {
            (
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.primary_released(),
                i.pointer.interact_pos(),
            )
        });
        if pressed && picture.is_pointer_button_down_on() {
            self.samples.clear();
            if let Some(p) = at {
                self.samples.push(p);
            }
        }
        if !self.samples.is_empty() {
            if down || released {
                if let Some(p) =
                    at.filter(|p| self.samples.last().is_none_or(|q| q.distance(*p) > 1.0))
                {
                    self.samples.push(p);
                }
            }
            ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                Id::unique("drawn-cut preview"),
            ))
            .add(egui::Shape::line(
                self.samples.clone(),
                Stroke::new(2.0, Color32::from_rgb(249, 168, 84)),
            ));
            if released {
                let frame = target.frame(st);
                let space = st.selected_modifiers().list[i]
                    .cut
                    .as_ref()
                    .map_or(CutSpace::Local, |c| c.space);
                if let Some(cut) = stroke(st, frame, space, picture.rect, &self.samples) {
                    st.selected_modifiers_mut().list[i].cut = Some(cut);
                    st.refresh_scope();
                }
                *self = Self::default();
            }
            return true;
        }
        if picture.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        picture.hovered() || picture.is_pointer_button_down_on()
    }
}

fn stroke(
    st: &State,
    frame: PivotFrame,
    space: CutSpace,
    rect: Rect,
    samples: &[Pos2],
) -> Option<DrawnCut> {
    if samples.len() < 2 {
        return None;
    }
    let cam = st.camera();
    let back = transpose(&st.view_r());
    let (_, dc) = cam.ray([0., 0.])?;
    let normal = apply(&back, dc);
    let plane = frame.to_world(frame.origin);
    let dot = |a: Point, b: Point| a.into_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let h = rect.height() as f64 * 0.5;
    let mut points = Vec::new();
    for p in samples {
        let uv = [
            (p.x - rect.center().x) as f64 / h,
            -(p.y - rect.center().y) as f64 / h,
        ];
        let (o, d) = cam.ray(uv)?;
        let (o, d) = (apply(&back, o), apply(&back, d));
        let denom = dot(normal, d);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = dot(normal, std::array::from_fn(|i| plane[i] - o[i])) / denom;
        if t < 0. {
            return None;
        }
        points.push(std::array::from_fn(|i| o[i] + t * d[i]));
    }
    let mut cut = DrawnCut {
        points,
        normal,
        space: CutSpace::World,
    };
    // A subpixel sampling tolerance preserves every visible change of direction.
    let pixel = cam.dist * cam.t / h;
    cut.simplify((0.65 * pixel.abs()).max(1e-5));
    cut.change_space(frame, space, &mut 1.0);
    cut.valid().then_some(cut)
}

pub(super) fn controls(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize) {
    let target = owner(st);
    let frame = target.frame(st);
    let active = px.draw_cut.target == Some((target, i));
    let m = &st.selected_modifiers().list[i];
    let Some(c) = m.cut.as_ref() else { return };
    let mut space = c.space;
    let label = if active {
        "Cancel drawing"
    } else if c.points.is_empty() {
        "Draw in viewport"
    } else {
        "Redraw in viewport"
    };
    ui.horizontal(|ui| {
        ui.label("Cut plane");
        ui.selectable_value(&mut space, CutSpace::Local, "Owner local");
        ui.selectable_value(&mut space, CutSpace::World, "Fixed world");
    });
    if space != c.space {
        let m = &mut st.selected_modifiers_mut().list[i];
        m.cut
            .as_mut()
            .unwrap()
            .change_space(frame, space, &mut m.size);
        st.refresh_scope();
    }
    if wide_button(ui, Id::unique(("draw cut", i)), label, active)
        .0
        .clicked()
    {
        if active {
            px.draw_cut = Default::default()
        } else {
            px.draw_cut.arm(st, i)
        }
    }
    if active {
        ui.label("Drag across the viewport to cut. Esc cancels.");
    }
}
