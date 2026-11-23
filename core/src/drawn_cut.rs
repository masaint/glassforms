//! A planar freehand slot retained by its owning modifier stack. The points
//! define its centreline; the cutter extends through the solid along the normal.
use crate::math::{dot, len, scale, sub};
use crate::modifier_pivot::{PivotFrame, Point};
use crate::mods::{Kind, Modifier, Mods};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CutSpace {
    #[default]
    Local,
    World,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DrawnCut {
    pub points: Vec<Point>,
    pub normal: Point,
    pub space: CutSpace,
}
impl Default for DrawnCut {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            normal: [0., 0., 1.],
            space: CutSpace::Local,
        }
    }
}

pub(crate) fn unit(v: Point) -> Point {
    let l = len(v);
    if l > 1e-12 {
        scale(v, 1.0 / l)
    } else {
        [0., 0., 1.]
    }
}

impl DrawnCut {
    pub fn valid(&self) -> bool {
        self.points.len() >= 2
            && self
                .points
                .iter()
                .flatten()
                .chain(self.normal.iter())
                .all(|v| v.is_finite())
            && dot(self.normal, self.normal) > 1e-12
    }

    /// Transfer an authored stroke between references without moving its cut.
    pub fn change_space(&mut self, frame: PivotFrame, to: CutSpace, width: &mut f64) {
        if self.space == to {
            return;
        }
        let transform = |p| {
            if to == CutSpace::Local {
                frame.to_local(p)
            } else {
                frame.to_world(p)
            }
        };
        self.normal = unit(sub(transform(self.normal), transform([0.; 3])));
        for p in &mut self.points {
            *p = transform(*p);
        }
        let scale = frame.placement.size.abs().max(1e-9);
        *width *= if to == CutSpace::Local {
            1. / scale
        } else {
            scale
        };
        self.space = to;
    }

    /// Remove intermediate samples which lie on the same visible stroke to within tolerance.
    pub fn simplify(&mut self, tolerance: f64) {
        if self.points.len() < 3 {
            return;
        }
        let mut keep = vec![false; self.points.len()];
        keep[0] = true;
        let last = self.points.len() - 1;
        keep[last] = true;
        let mut ranges = vec![(0, last)];
        while let Some((a, b)) = ranges.pop() {
            let p = self.points[a];
            let d = sub(self.points[b], p);
            let dd = dot(d, d).max(1e-30);
            let mut best = tolerance * tolerance;
            let mut split = None;
            for i in a + 1..b {
                let q = sub(self.points[i], p);
                let t = (dot(q, d) / dd).clamp(0., 1.);
                let error = sub(q, d.map(|x| t * x));
                let distance = dot(error, error);
                if distance > best {
                    best = distance;
                    split = Some(i);
                }
            }
            if let Some(i) = split {
                keep[i] = true;
                ranges.push((a, i));
                ranges.push((i, b));
            }
        }
        let mut i = 0;
        self.points.retain(|_| {
            i += 1;
            keep[i - 1]
        });
    }
}

impl Modifier {
    pub fn has_world_cut(&self) -> bool {
        self.kind == Kind::DrawCut
            && self
                .cut
                .as_ref()
                .is_some_and(|c| c.space == CutSpace::World)
    }
    pub fn localize_cut(&mut self, frame: PivotFrame) {
        if let Some(c) = self.cut.as_mut() {
            c.change_space(frame, CutSpace::Local, &mut self.size);
        }
    }
}
impl Mods {
    /// The saved stroke is one editable operation. Its exact signed distance
    /// is the minimum of its segments, so successive subtractions reproduce it.
    pub(crate) fn expanded_cuts(&self) -> Option<Self> {
        if !self
            .list
            .iter()
            .any(|m| m.kind == Kind::DrawCut && m.cut.as_ref().is_some_and(|c| c.points.len() > 2))
        {
            return None;
        }
        let mut list = Vec::new();
        for m in &self.list {
            if let Some(c) = m
                .cut
                .as_ref()
                .filter(|_| m.kind == Kind::DrawCut && m.live())
            {
                for pair in c.points.windows(2) {
                    let mut segment = m.clone();
                    segment.cut = Some(DrawnCut {
                        points: pair.to_vec(),
                        normal: c.normal,
                        space: c.space,
                    });
                    list.push(segment);
                }
            } else if m.kind != Kind::DrawCut || m.live() {
                list.push(m.clone())
            }
        }
        Some(Self { list })
    }
}
