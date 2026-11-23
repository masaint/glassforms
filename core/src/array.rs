//! Arrays: a field's copies in a grid, a ring or part of a ring, or along a path, read exactly.
//!
//! An array is a `mods::Kind::Repeat` entry. It is read one of two ways.
//!
//! **Fold** (shader code 8: a grid, or a whole ring with no steps and no blend). The point is taken into the
//! cell that holds it and one copy is read there, guarded by the cell's walls (`mods.rs`, `modBack` case 8):
//!
//! ```text
//!   R(p) = min(d_c(p), g(p)),   g = the way to a wall of the cell + the room the copy beyond it leaves
//! ```
//!
//! Every other copy lies beyond a wall of cell c, at least `room` from it, so the straight way from p to any
//! point of it crosses the wall: its distance is at least g. Hence `R <= min_k d_k`, the union's distance:
//! **a fold never overstates while the copies leave room** (`rooms` > 0). It is exact where `d_c <= g`, which
//! holds near the copy. Without the guard (`d_c` alone) the reading is the distance to one copy only and
//! overstates wherever another copy is nearer, which happens whenever the motif is not symmetric about its
//! cell (test `an_unguarded_fold_overstates`): a fold keeps a valid bound only if the neighbours are either
//! read or walled off. When copies touch, no wall exists, and the copies must be read: the overlap route
//! (`repeat_overlap.wgsl`) or a listed array.
//! The grid's rooms differ per side (`Modifier::clearance`): the reading is then still never above the
//! union, but it can step down where a wall is crossed, by at most the difference of the two rooms (test
//! `the_fold_steps_down_by_the_difference_of_its_rooms`). With equal rooms it is continuous and 1-Lipschitz.
//!
//! **Listed** (shader code 14, a node's stack only: an arc, a path, steps per copy, a blend, and grids of up to
//! `LIST_MAX` copies with those). Every copy is a rigid placement with a scale (`Place`): a point is taken
//! into the copy (`Place::back`) and read there, the reading multiplied by the copy's scale, which is exact
//! for a uniform scale. The reading is the least over all copies, but a copy is read only if the signed
//! distance to its box (the field's box carried by the placement) is below what is kept:
//!
//! ```text
//!   pass 1: the copy whose box is nearest;   pass 2: every other copy whose box is nearer than the best
//! ```
//!
//! The signed distance to a box never exceeds the distance to anything inside it, inside or out (a point
//! inside the solid is no farther from the solid's outside than from the box's), so a skipped copy could not
//! have been nearer: **the listed reading equals the union of every copy** (test
//! `the_listed_reading_is_the_union_of_its_copies`), overlapping or not. Blend joins the two nearest copies by
//! the quadratic smooth minimum (the fillet profile of the joins, `min - k h²/4`): the second-smallest of
//! 1-Lipschitz readings is 1-Lipschitz, and the join's slopes sum to 1, so the blend is 1-Lipschitz and never
//! overstates its own surface (Hart 1996, theorem 1); the cull keeps it equal to joining the two nearest of
//! all copies. Where three copies are within the blend of each other only the two nearest are joined.
//!
//! Nested arrays are stacks or nodes in turn: a listed array reads the field before it at each candidate, a
//! fold takes the point back once; each of the two readings above never overstates when the field it reads
//! does not, so their composition does not either (tests `a_ring_of_grids_...`, `a_grid_of_rings_...`).
//!
//! Counts: a grid folds any count at the same cost; `GRID_MAX` per axis. A listed array holds `LIST_MAX`
//! copies, the same rows reserved whatever the count, so a count is a number and changes no shader text.
//! Raising the part count to 256 (decision D8) waits for the layer work; nothing here depends on it.

use serde::{Deserialize, Serialize};

use crate::mods::{Box3, Kind, Modifier, Mods, MOD_FLOATS};

/// Copies along each axis of a grid (the fold costs the same for any count).
pub const GRID_MAX: f64 = 64.0;
/// Copies in a whole ring.
pub const RING_MAX: f64 = 64.0;
/// Copies of a listed array (arc, path, steps, blend). Its rows are reserved in full.
pub const LIST_MAX: usize = 64;
/// The shader's number for a listed array.
pub const LISTED: u32 = 14;

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}
fn zero3(v: &[f64; 3]) -> bool {
    *v == [0.0; 3]
}
fn not(v: &bool) -> bool {
    !*v
}

/// What an array has beyond a plain grid or ring. All plain numbers; the default is none of it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArrayOpts {
    /// a ring's span in degrees (0, or 360 and more: the whole ring) and where it starts
    #[serde(skip_serializing_if = "is_zero")]
    pub span: f64,
    #[serde(skip_serializing_if = "is_zero")]
    pub start: f64,
    /// the copies keep the shape's heading instead of turning with the ring or the path
    #[serde(skip_serializing_if = "not")]
    pub keep_heading: bool,
    /// per copy, one more of: a move (in the station's frame: a grid's axes; a ring's out, up and along; a
    /// path's along, up and across), a turn about the copy's own y (degrees), a growth (a share of its size)
    #[serde(skip_serializing_if = "zero3")]
    pub step_move: [f64; 3],
    #[serde(skip_serializing_if = "is_zero")]
    pub step_turn: f64,
    #[serde(skip_serializing_if = "is_zero")]
    pub step_grow: f64,
    /// neighbouring copies joined by a smooth seam this wide (0: a plain union)
    #[serde(skip_serializing_if = "is_zero")]
    pub blend: f64,
    /// along a path: the path's text (`path::Path::text`; empty: a half turn)
    #[serde(skip_serializing_if = "String::is_empty")]
    pub path: String,
}

impl ArrayOpts {
    pub fn is_plain(&self) -> bool {
        *self == Self::default()
    }
    /// Any step per copy.
    pub fn stepped(&self) -> bool {
        self.step_move != [0.0; 3] || self.step_turn != 0.0 || self.step_grow != 0.0
    }
    /// A part of a ring only.
    pub fn arc(&self) -> bool {
        self.span != 0.0 && self.span.abs() < 360.0
    }
}

/// The path an array follows when it has none of its own.
pub const DEFAULT_PATH: &str = "start -0.8 0 0 0 0 0; T 180 0.8";

/// 3 x 3 as columns: the images of x, y and z.
pub type Mat = [[f64; 3]; 3];
const ID: Mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn apply(m: &Mat, q: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[0][i] * q[0] + m[1][i] * q[1] + m[2][i] * q[2])
}
fn apply_t(m: &Mat, u: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|c| m[c][0] * u[0] + m[c][1] * u[1] + m[c][2] * u[2])
}
fn mul(a: &Mat, b: &Mat) -> Mat {
    [apply(a, b[0]), apply(a, b[1]), apply(a, b[2])]
}
/// Turned about y by t, from x toward z (as `modTurn(u, 0, 2, t)` turns a point).
fn turn_y(t: f64) -> Mat {
    let (s, c) = t.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn len(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// One copy: the field's point q stands at `u = at + scale * rot q` in the array's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub rot: Mat,
    pub at: [f64; 3],
    pub scale: f64,
}

impl Place {
    /// The point of the field read for u.
    pub fn back(&self, u: [f64; 3]) -> [f64; 3] {
        scale(apply_t(&self.rot, sub(u, self.at)), 1.0 / self.scale)
    }
    pub fn out(&self, q: [f64; 3]) -> [f64; 3] {
        add(self.at, scale(apply(&self.rot, q), self.scale))
    }
    /// Three rows of four: q = rows · (u, 1).
    pub fn rows(&self) -> [[f64; 4]; 3] {
        let k = 1.0 / self.scale;
        let c = apply_t(&self.rot, self.at);
        std::array::from_fn(|r| [self.rot[r][0] * k, self.rot[r][1] * k, self.rot[r][2] * k, -c[r] * k])
    }
}

/// Where each copy stands before its steps: a station (a position and a frame).
#[derive(Clone, Copy, Debug)]
pub struct Station {
    pub at: [f64; 3],
    pub frame: Mat,
    /// the copy's number along its steps
    pub k: f64,
}

/// A grid's stations: copy (i, j, l) at ((i, j, l) - (n - 1) / 2) · step, every other row (along y) moved
/// along x by `stagger` of a step; its number along the steps i + j + l.
pub fn grid_stations(n: [usize; 3], step: [f64; 3], stagger: f64) -> Vec<Station> {
    let mut out = Vec::with_capacity(n[0] * n[1] * n[2]);
    for l in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let id = [i, j, l];
                let mut at: [f64; 3] = std::array::from_fn(|a| (id[a] as f64 - 0.5 * (n[a] as f64 - 1.0)) * step[a]);
                if j % 2 == 1 {
                    at[0] += stagger * step[0];
                }
                out.push(Station { at, frame: ID, k: (i + j + l) as f64 });
            }
        }
    }
    out
}

/// The angles of a ring's copies: a whole ring (`span` 0 or a full turn and more) n evenly from `start`;
/// a part of one from `start` to `start + span`, both ends included (radians).
pub fn ring_angles(n: usize, start: f64, span: f64) -> Vec<f64> {
    let tau = std::f64::consts::TAU;
    let whole = span == 0.0 || span.abs() >= tau - 1e-12;
    let step = if whole { tau / n as f64 } else if n > 1 { span / (n as f64 - 1.0) } else { 0.0 };
    (0..n).map(|k| start + k as f64 * step).collect()
}

/// A ring's stations at `radius` from its axis (y), x pointing out from it.
pub fn ring_stations(n: usize, radius: f64, start: f64, span: f64) -> Vec<Station> {
    ring_angles(n, start, span)
        .into_iter()
        .enumerate()
        .map(|(k, a)| {
            let frame = turn_y(a);
            Station { at: apply(&frame, [radius, 0.0, 0.0]), frame, k: k as f64 }
        })
        .collect()
}

/// A path's stations, evenly by length along it (both ends of an open path; a closed one n evenly round),
/// each with the path's frame there: x along it, y its top, z across.
pub fn path_stations(path: &crate::path::Path, n: usize) -> Vec<Station> {
    let b = crate::path_formula::build(path, 0.0, false);
    if b.prims.is_empty() || n == 0 {
        return vec![Station { at: path.start.p, frame: ID, k: 0.0 }; n.min(1)];
    }
    let total = b.length;
    let gap = if b.closed { total / n as f64 } else if n > 1 { total / (n as f64 - 1.0) } else { 0.0 };
    (0..n)
        .map(|k| {
            let s = (k as f64 * gap).min(total);
            let prim = b.prims.iter().find(|p| s <= p.s1).unwrap_or(b.prims.last().expect("one at least"));
            let u = ((s - prim.s0) / (prim.s1 - prim.s0).max(1e-12)).clamp(0.0, 1.0);
            let (p, t, side) = prim.at(u);
            let top = cross(t, side);
            Station { at: p, frame: [t, top, cross(t, top)], k: k as f64 }
        })
        .collect()
}

/// Each station's copy: its steps, then the station, then the whole array scaled by `fit` about its middle.
/// Copy k is moved by k·`step_move` in the station's frame, turned k·`step_turn` about its own y and grown
/// by (1 + `step_grow`)^k; it turns with its station unless it keeps its heading.
pub fn places(stations: &[Station], o: &ArrayOpts, fit: f64) -> Vec<Place> {
    let grow = 1.0 + o.step_grow.max(-0.9);
    stations
        .iter()
        .map(|s| {
            let heading = if o.keep_heading { ID } else { s.frame };
            let rot = mul(&heading, &turn_y(s.k * o.step_turn.to_radians()));
            let at = add(s.at, apply(&s.frame, scale(o.step_move, s.k)));
            Place { rot, at: scale(at, fit), scale: fit * grow.powf(s.k) }
        })
        .collect()
}

/// The signed distance to a box.
pub fn box_dist(q: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let e: [f64; 3] = std::array::from_fn(|i| (lo[i] - q[i]).max(q[i] - hi[i]));
    len(e.map(|v| v.max(0.0))) + e[0].max(e[1]).max(e[2]).min(0.0)
}

/// The quadratic smooth minimum of two readings (the joins' fillet profile): `min - k h²/4`.
pub fn smin(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (1.0 - (a - b).abs() / k).max(0.0);
    a.min(b) - 0.25 * k * h * h
}

/// The listed reading at u (array frame): as the shader reads it (`ARRAY`), the field's box being
/// (`lo`, `hi`) in the field's own frame. Also returns how many copies were read.
pub fn listed<F: Fn([f64; 3]) -> f64>(places: &[Place], lo: [f64; 3], hi: [f64; 3], blend: f64, field: F, u: [f64; 3]) -> (f64, usize) {
    let bound = |p: &Place| p.scale * box_dist(p.back(u), lo, hi);
    let (mut first, mut near) = (0, f64::INFINITY);
    for (k, p) in places.iter().enumerate() {
        let b = bound(p);
        if b < near {
            near = b;
            first = k;
        }
    }
    let (mut best, mut second, mut read) = (f64::INFINITY, f64::INFINITY, 0);
    let order = std::iter::once(first).chain((0..places.len()).filter(|k| *k != first));
    for (j, k) in order.enumerate() {
        let p = &places[k];
        if j > 0 {
            let cut = if blend > 0.0 { second.min(best + blend) } else { best };
            if bound(p) >= cut {
                continue;
            }
        }
        read += 1;
        let d = p.scale * field(p.back(u));
        if d < best {
            second = best;
            best = d;
        } else {
            second = second.min(d);
        }
    }
    (if blend > 0.0 && second.is_finite() { smin(best, second, blend) } else { best }, read)
}

/// Every copy read, the nearest kept (and, with a blend, joined to the second nearest).
pub fn union<F: Fn([f64; 3]) -> f64>(places: &[Place], blend: f64, field: F, u: [f64; 3]) -> f64 {
    let mut d: Vec<f64> = places.iter().map(|p| p.scale * field(p.back(u))).collect();
    d.sort_by(f64::total_cmp);
    match d.len() {
        0 => f64::INFINITY,
        1 => d[0],
        _ if blend > 0.0 => smin(d[0], d[1], blend),
        _ => d[0],
    }
}

/// The fold of a grid as the shader reads it (`modBack` case 8 then `modOn`): `n` copies a side, `step`,
/// `stagger`, the rooms the copies leave beyond the − and + walls, the fit; u in the array's frame.
#[allow(clippy::too_many_arguments)]
pub fn fold_grid<F: Fn([f64; 3]) -> f64>(n: [f64; 3], step: [f64; 3], stagger: f64, minus: [f64; 3], plus: [f64; 3], fit: f64, field: F, u: [f64; 3]) -> f64 {
    let mut v = scale(u, 1.0 / fit);
    let mut g = 1e9f64;
    for o in 0..3 {
        let i = (o + 1) % 3;
        if n[i] > 1.5 {
            let s = step[i];
            let c = 0.5 * (n[i] - 1.0);
            let id = (v[i] / s + c).round_ties_even().clamp(0.0, n[i] - 1.0);
            v[i] -= s * (id - c);
            if id < n[i] - 1.5 {
                g = g.min(0.5 * s - v[i] + plus[i]);
            }
            if id > 0.5 {
                g = g.min(0.5 * s + v[i] + minus[i]);
            }
            if i == 1 {
                v[0] -= stagger * step[0] * (id - 2.0 * (0.5 * id).floor());
            }
        }
    }
    fit * field(v).min(g)
}

/// The fold of a whole ring as the shader reads it: `n` copies at `radius`, the room each leaves from its
/// sector's planes, slices mirrored or not, the fit.
pub fn fold_ring<F: Fn([f64; 3]) -> f64>(n: f64, radius: f64, room: f64, flip: bool, fit: f64, field: F, u: [f64; 3]) -> f64 {
    let mut v = scale(u, 1.0 / fit);
    let mut g = 1e9f64;
    if n > 1.5 {
        let al = std::f64::consts::TAU / n;
        let k = (v[2].atan2(v[0]) / al).round_ties_even();
        v = apply(&turn_y(-k * al), v);
        let rho = v[0].hypot(v[2]);
        g = rho * (0.5 * al - v[2].atan2(v[0]).abs()).sin() + room;
        if flip {
            v[2] = v[2].abs();
        }
    }
    v[0] -= radius;
    fit * field(v).min(g)
}

/// A grid's rooms beyond its − and + walls per axis, for a field within the box (lo, hi) (as
/// `Modifier::clearance` gives them).
pub fn grid_rooms(n: [f64; 3], step: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let mut minus = [1e9; 3];
    let mut plus = [1e9; 3];
    for i in 0..3 {
        if n[i] >= 2.0 {
            plus[i] = 0.5 * step[i] + lo[i];
            minus[i] = 0.5 * step[i] - hi[i];
        }
    }
    (minus, plus)
}

// ---- the modifier's side

impl Modifier {
    /// Its copies are listed one by one (shader code 14): a part of a ring, a path, steps or a blend. A grid
    /// with steps or a blend is listed only up to `LIST_MAX` copies; past that the extras are not drawn.
    pub fn listed(&self) -> bool {
        if self.kind != Kind::Repeat {
            return false;
        }
        let o = &self.array;
        match self.mode {
            2 => true,
            1 => o.arc() || o.stepped() || o.blend > 0.0 || o.keep_heading,
            _ => (o.stepped() || o.blend > 0.0) && self.grid_count() <= LIST_MAX,
        }
    }
    fn grid_count(&self) -> usize {
        self.a.get().iter().map(|v| v.round().clamp(1.0, GRID_MAX) as usize).product()
    }
    /// A listed array's copies in its own frame (fit included).
    pub fn places(&self) -> Vec<Place> {
        let a = self.a.get();
        let stations = match self.mode {
            1 => ring_stations(a[0].round().clamp(1.0, RING_MAX) as usize, self.detail, self.array.start.to_radians(), self.array.span.to_radians()),
            2 => path_stations(&self.array_path(), a[0].round().clamp(1.0, LIST_MAX as f64) as usize),
            _ => {
                let n = a.map(|v| v.round().clamp(1.0, GRID_MAX) as usize);
                grid_stations(n, self.b.get().map(|v| v.max(0.05)), self.stagger)
            }
        };
        let mut p = places(&stations, &self.array, self.fit_scale());
        p.truncate(LIST_MAX);
        p
    }
    /// The path a path array follows.
    pub fn array_path(&self) -> crate::path::Path {
        crate::path::Path::parse(&self.array.path).or_else(|| crate::path::Path::parse(DEFAULT_PATH)).expect("the default path reads")
    }
    /// A path array's fit: the path and the copies scaled together to stand where the shape stood.
    pub(crate) fn path_fit(&self) -> f64 {
        let (lo, hi) = self.array_path().bounds();
        let span = (0..3).map(|i| hi[i] - lo[i]).fold(0.0f64, f64::max);
        let w = 2.0 * crate::mods::H;
        w / (w + span)
    }
    /// The box of a listed array's field (object frame), from the box before it.
    pub(crate) fn listed_bound(&self, b: &Box3) -> Box3 {
        let f = self.frame();
        let lb = crate::mods::local_box(b, self.centre, &f);
        let mut o = [f64::MAX, f64::MAX, f64::MAX, f64::MIN, f64::MIN, f64::MIN];
        for p in self.places() {
            for k in 0..8 {
                let q = [lb[if k & 1 == 0 { 0 } else { 3 }], lb[if k & 2 == 0 { 1 } else { 4 }], lb[if k & 4 == 0 { 2 } else { 5 }]];
                let u = p.out(q);
                for i in 0..3 {
                    o[i] = o[i].min(u[i]);
                    o[i + 3] = o[i + 3].max(u[i]);
                }
            }
        }
        if o[0] > o[3] {
            return *b;
        }
        crate::mods::world_box(&o, self.centre, &f)
    }
    /// A listed array's entry (nine vec4): q0 (code, mode, -, copies), q1 (centre, 1), q2..q4 its frame
    /// (q2.w the blend, q3.w the blend's profile: 0 the fillet), q5 and q6 the field's box before it in the
    /// array's frame (w: -1, no guard).
    pub fn listed_entry(&self, before: &Box3) -> [f32; MOD_FLOATS] {
        let f = self.frame();
        let lb = crate::mods::local_box(before, self.centre, &f);
        let n = self.places().len();
        let c = self.centre;
        let rows: [[f64; 4]; 9] = [
            [LISTED as f64, self.mode as f64, 1.0, n as f64],
            [c[0], c[1], c[2], 1.0],
            [f[0][0], f[0][1], f[0][2], self.array.blend.max(0.0)],
            [f[1][0], f[1][1], f[1][2], 0.0],
            [f[2][0], f[2][1], f[2][2], 0.0],
            [lb[0], lb[1], lb[2], -1.0],
            [lb[3], lb[4], lb[5], -1.0],
            [0.0; 4],
            [0.0; 4],
        ];
        let mut q = [0f32; MOD_FLOATS];
        for (i, r) in rows.iter().enumerate() {
            for k in 0..4 {
                q[i * 4 + k] = r[k] as f32;
            }
        }
        q
    }
    /// A listed array's rows, one table record a copy (`LIST_MAX` reserved, so a count is only a number):
    /// the way into the copy (three rows of four) and its scale.
    pub fn listed_rows(&self) -> Vec<f32> {
        let stride = crate::csg::PART_FLOATS;
        let mut out = vec![0f32; LIST_MAX * stride];
        for (k, p) in self.places().iter().enumerate() {
            let r = p.rows();
            for (i, row) in r.iter().enumerate() {
                for c in 0..4 {
                    out[k * stride + i * 4 + c] = row[c] as f32;
                }
            }
            out[k * stride + 12] = p.scale as f32;
        }
        out
    }
}

impl Mods {
    /// The stack's shader numbers for a node: as `codes`, a listed array as `LISTED`.
    pub fn node_codes(&self) -> Vec<u32> {
        let codes = self.codes();
        if self.expanded_cuts().is_some() {
            return codes;
        }
        codes.into_iter().zip(self.live()).map(|(c, m)| if m.listed() { LISTED } else { c }).collect()
    }
    /// The stack's numbers for a node (`exact_numbers`, with each listed array's entry), and each live
    /// entry's rows (empty but for listed arrays).
    pub fn node_numbers(&self, shape: &Box3, object: &crate::layers::Placement) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut nums = self.exact_numbers(shape, object);
        if self.expanded_cuts().is_some() {
            return (nums, Vec::new());
        }
        let boxes = self.boxes(shape);
        let stride = crate::csg::PART_FLOATS;
        let mut rows = Vec::new();
        for (j, m) in self.live().take(crate::mods::MAX_MODS).enumerate() {
            if m.listed() {
                let q = m.listed_entry(&boxes[j]);
                nums[stride * (1 + j)..stride * (1 + j) + MOD_FLOATS].copy_from_slice(&q);
                rows.push(m.listed_rows());
            } else {
                rows.push(Vec::new());
            }
        }
        (nums, rows)
    }
}

/// The shader's listed array: one stage of a node's stack (`construction_pattern::wrap_stack`), `m` its
/// entry, `u` the point in its frame; ROWS the first row, INCOMING the field before it.
pub const ARRAY: &str = r#"
    // array:listed (gf_core::array: the copy with the nearest box, then every copy whose box is nearer)
    let n=i32(m.q0.w);
    var first=0;
    var near=1e9;
    for (var k=0;k<n;k++) {
        let r=csgP[ROWSu+u32(k)];
        let h=vec4f(u,1.0);
        let q=vec3f(dot(r.inv0,h),dot(r.inv1,h),dot(r.inv2,h));
        let e=max(m.q5.xyz-q,q-m.q6.xyz);
        let lb=r.b.x*(length(max(e,vec3f(0.0)))+min(max(e.x,max(e.y,e.z)),0.0));
        if (lb<near) { near=lb; first=k; }
    }
    var best=vec3f(1e9,0.0,-1.0);
    var second=1e9;
    for (var j=-1;j<n;j++) {
        if (j==first) { continue; }
        let k=select(j,first,j<0);
        let r=csgP[ROWSu+u32(k)];
        let h=vec4f(u,1.0);
        let q=vec3f(dot(r.inv0,h),dot(r.inv1,h),dot(r.inv2,h));
        if (j>=0) {
            let e=max(m.q5.xyz-q,q-m.q6.xyz);
            let lb=r.b.x*(length(max(e,vec3f(0.0)))+min(max(e.x,max(e.y,e.z)),0.0));
            let cut=select(best.x,min(second,best.x+m.q2.w),m.q2.w>0.0);
            if (lb>=cut) { continue; }
        }
        let hit=INCOMING(cn_modWorld(m,q));
        let d=r.b.x*hit.x;
        if (d<best.x) { second=best.x; best=vec3f(d,0.0,hit.z); } else { second=min(second,d); }
    }
    var d=best.x;
    // array:blend (the joins' fillet profile; q3.w is the profile's number for the join family)
    if (m.q2.w>0.0 && second<1e8) { let w=m.q2.w; let h=max(1.0-abs(second-best.x)/w,0.0); d=best.x-0.25*w*h*h; }
    return vec3f(d,0.0,best.z);
"#;

/// The named anchors of `ARRAY` (census style: each found once per listed stage, none in a tree without).
pub const ANCHORS: [(&str, &str); 2] = [
    ("array.listed", "    // array:listed (gf_core::array"),
    ("array.blend", "    // array:blend (the joins' fillet profile"),
];

/// The shader's listed reading done in f32 from the packed numbers: the entry (`listed_entry`) and the
/// rows (`listed_rows`), the field read in the array's input frame, x in the object frame. Line for line
/// `ARRAY` with `modLocal` and `modWorld`: the test that the numbers and the text agree with the formula.
pub fn read_packed<F: Fn([f32; 3]) -> f32>(m: &[f32], rows: &[f32], field: F, x: [f32; 3]) -> f32 {
    let v = |i: usize| [m[i * 4], m[i * 4 + 1], m[i * 4 + 2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let d0 = [x[0] - m[4], x[1] - m[5], x[2] - m[6]];
    let u = [dot(d0, v(2)), dot(d0, v(3)), dot(d0, v(4))];
    let world = |q: [f32; 3]| -> [f32; 3] { std::array::from_fn(|i| m[4 + i] + q[0] * m[8 + i] + q[1] * m[12 + i] + q[2] * m[16 + i]) };
    let stride = crate::csg::PART_FLOATS;
    let n = m[3] as usize;
    let blend = m[11];
    let (lo, hi) = (v(5), v(6));
    let into = |k: usize| -> ([f32; 3], f32) {
        let r = &rows[k * stride..];
        let q = std::array::from_fn(|i| r[i * 4] * u[0] + r[i * 4 + 1] * u[1] + r[i * 4 + 2] * u[2] + r[i * 4 + 3]);
        (q, r[12])
    };
    let bound = |q: [f32; 3], s: f32| {
        let e: [f32; 3] = std::array::from_fn(|i| (lo[i] - q[i]).max(q[i] - hi[i]));
        s * ((e.map(|v| v.max(0.0)).iter().map(|v| v * v).sum::<f32>()).sqrt() + e[0].max(e[1]).max(e[2]).min(0.0))
    };
    let (mut first, mut near) = (0, 1e9f32);
    for k in 0..n {
        let (q, s) = into(k);
        let b = bound(q, s);
        if b < near {
            near = b;
            first = k;
        }
    }
    let (mut best, mut second) = (1e9f32, 1e9f32);
    for j in -1..n as i64 {
        if j == first as i64 {
            continue;
        }
        let k = if j < 0 { first } else { j as usize };
        let (q, s) = into(k);
        if j >= 0 {
            let cut = if blend > 0.0 { second.min(best + blend) } else { best };
            if bound(q, s) >= cut {
                continue;
            }
        }
        let d = s * field(world(q));
        if d < best {
            second = best;
            best = d;
        } else {
            second = second.min(d);
        }
    }
    if blend > 0.0 && second < 1e8 {
        let h = (1.0 - (second - best).abs() / blend).max(0.0);
        best - 0.25 * blend * h * h
    } else {
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small random number source (no crate needed).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.next()
        }
        fn point(&mut self, r: f64) -> [f64; 3] {
            [self.range(-r, r), self.range(-r, r), self.range(-r, r)]
        }
    }

    /// Exact motifs: a box off its centre (not symmetric about its cell), and a torus.
    fn lump(q: [f64; 3]) -> f64 {
        box_dist(q, [-0.05, -0.12, -0.08], [0.17, 0.1, 0.06])
    }
    fn torus(q: [f64; 3]) -> f64 {
        (q[0].hypot(q[2]) - 0.12).hypot(q[1]) - 0.04
    }
    const LUMP: ([f64; 3], [f64; 3]) = ([-0.05, -0.12, -0.08], [0.17, 0.1, 0.06]);
    const TORUS: ([f64; 3], [f64; 3]) = ([-0.16, -0.04, -0.16], [0.16, 0.04, 0.16]);

    /// Copies written out one by one, independently of the fold.
    fn written_grid(n: [usize; 3], step: [f64; 3], stagger: f64, fit: f64) -> Vec<Place> {
        let mut out = Vec::new();
        for i in 0..n[0] {
            for j in 0..n[1] {
                for l in 0..n[2] {
                    let mut x = (i as f64 - (n[0] - 1) as f64 / 2.0) * step[0];
                    if j % 2 == 1 {
                        x += stagger * step[0];
                    }
                    let y = (j as f64 - (n[1] - 1) as f64 / 2.0) * step[1];
                    let z = (l as f64 - (n[2] - 1) as f64 / 2.0) * step[2];
                    out.push(Place { rot: ID, at: [fit * x, fit * y, fit * z], scale: fit });
                }
            }
        }
        out
    }

    fn slope_ok<F: Fn([f64; 3]) -> f64>(f: F, rng: &mut Rng, pairs: usize, r: f64) -> f64 {
        let mut worst = 0.0f64;
        for _ in 0..pairs {
            let p = rng.point(r);
            let dir = rng.point(1.0);
            let l = len(dir).max(1e-9);
            let h = rng.range(1e-4, 0.05);
            let q = add(p, scale(dir, h / l));
            worst = worst.max((f(p) - f(q)).abs() / h);
        }
        worst
    }

    #[test]
    fn a_row_of_sixty_four_folds_to_the_union_and_never_above_it() {
        // 64 copies, the count step 3 raises the grid to; the fold reads one cell whatever the count
        let n = [GRID_MAX, 1.0, 1.0];
        let step = [0.4, 0.4, 0.4];
        let (minus, plus) = grid_rooms(n, step, LUMP.0, LUMP.1);
        assert!(minus.iter().chain(&plus).all(|r| *r > 0.0));
        let copies = written_grid([64, 1, 1], step, 0.0, 0.8);
        let mut rng = Rng(7);
        let (mut exact, mut most_over) = (0, f64::MIN);
        for _ in 0..20000 {
            let u = [rng.range(-11.0, 11.0), rng.range(-0.6, 0.6), rng.range(-0.6, 0.6)];
            let f = fold_grid(n, step, 0.0, minus, plus, 0.8, lump, u);
            let t = union(&copies, 0.0, lump, u);
            most_over = most_over.max(f - t);
            if (f - t).abs() < 1e-12 {
                exact += 1;
            }
        }
        eprintln!("row of 64: most over the union {most_over:e}, exact at {exact} of 20000");
        assert!(most_over <= 1e-12, "the fold overstated by {most_over}");
        assert!(exact > 2000, "exact at {exact} points of 20000");
    }

    #[test]
    fn an_eight_cubed_grid_with_rows_offset_never_overstates() {
        let n = [8.0, 8.0, 8.0];
        let step = [0.45, 0.3, 0.35];
        let stagger = 0.3;
        let (minus, plus) = grid_rooms(n, step, LUMP.0, LUMP.1);
        let copies = written_grid([8, 8, 8], step, stagger, 1.0);
        let mut rng = Rng(11);
        for _ in 0..4000 {
            let u = rng.point(2.2);
            let f = fold_grid(n, step, stagger, minus, plus, 1.0, lump, u);
            let t = union(&copies, 0.0, lump, u);
            assert!(f <= t + 1e-12, "over by {} at {u:?}", f - t);
            if t < 0.03 {
                // near a copy the guard is not reached: the fold is the union exactly
                assert!((f - t).abs() < 1e-12, "{f} against {t} at {u:?}");
            }
        }
    }

    #[test]
    fn an_unguarded_fold_overstates() {
        // the same row read in one cell with no wall: the copy beyond the wall is nearer and is missed
        let n = [8.0, 1.0, 1.0];
        let step = [0.3, 1.0, 1.0];
        let copies = written_grid([8, 1, 1], step, 0.0, 1.0);
        let mut rng = Rng(3);
        let mut most = 0.0f64;
        for _ in 0..20000 {
            let u = [rng.range(-1.3, 1.3), rng.range(-0.3, 0.3), rng.range(-0.3, 0.3)];
            let f = fold_grid(n, step, 0.0, [1e9; 3], [1e9; 3], 1.0, lump, u);
            most = most.max(f - union(&copies, 0.0, lump, u));
        }
        eprintln!("unguarded fold: overstates by up to {most}");
        assert!(most > 0.02, "the lone fold should overstate; most {most}");
    }

    #[test]
    fn the_fold_steps_down_by_the_difference_of_its_rooms() {
        // per-side rooms (as `clearance` packs them): never above the union, continuous for a symmetric
        // motif, and for the lump stepping at a wall by no more than |plus - minus|
        let n = [GRID_MAX, 1.0, 1.0];
        let step = [0.4, 1.0, 1.0];
        let (minus, plus) = grid_rooms(n, step, LUMP.0, LUMP.1);
        let mut rng = Rng(5);
        let jump = |field: fn([f64; 3]) -> f64, mi: [f64; 3], pl: [f64; 3], rng: &mut Rng| {
            let mut most = 0.0f64;
            for _ in 0..4000 {
                // (64 cells about the middle: the walls are at whole steps)
                let wall = rng.range(-30.0, 30.0).round() * 0.4;
                let u = [wall, rng.range(-0.4, 0.4), rng.range(-0.4, 0.4)];
                let a = fold_grid(n, step, 0.0, mi, pl, 1.0, field, [u[0] - 1e-9, u[1], u[2]]);
                let b = fold_grid(n, step, 0.0, mi, pl, 1.0, field, [u[0] + 1e-9, u[1], u[2]]);
                most = most.max((a - b).abs());
            }
            most
        };
        let lump_jump = jump(lump, minus, plus, &mut rng);
        eprintln!("per-side rooms {:.3} and {:.3}: steps down by {lump_jump:.4} at most", minus[0], plus[0]);
        assert!(lump_jump <= (plus[0] - minus[0]).abs() + 1e-6, "{lump_jump}");
        let (tm, tp) = grid_rooms(n, step, TORUS.0, TORUS.1);
        assert!(jump(torus, tm, tp, &mut rng) < 1e-6);
        // and with one room, the least of the two, the lump's fold is continuous and 1-Lipschitz
        let r = plus[0].min(minus[0]);
        assert!(jump(lump, [r; 3], [r; 3], &mut rng) < 1e-6);
        let s = slope_ok(|u| fold_grid(n, step, 0.0, [r; 3], [r; 3], 1.0, lump, u), &mut rng, 100000, 3.0);
        assert!(s <= 1.0 + 1e-6, "slope {s}");
    }

    #[test]
    fn a_ring_of_sixty_four_folds_within_the_union() {
        // (64 tori of radius 0.16 leave room between them from a radius of 3.3)
        let radius = 4.0;
        let n = RING_MAX;
        let half = std::f64::consts::PI / n;
        // (the room as `clearance` takes it: the torus's box against its sector's planes)
        let mut room = f64::MAX;
        for k in 0..8 {
            let x = if k & 1 == 0 { TORUS.0[0] } else { TORUS.1[0] } + radius;
            let z = if k & 4 == 0 { TORUS.0[2] } else { TORUS.1[2] };
            room = room.min(x * half.sin() - z * half.cos()).min(x * half.sin() + z * half.cos());
        }
        assert!(room > 0.0, "room {room}");
        let copies = places(&ring_stations(64, radius, 0.0, 0.0), &ArrayOpts::default(), 1.0);
        let mut rng = Rng(13);
        for _ in 0..4000 {
            let u = [rng.range(-4.4, 4.4), rng.range(-0.3, 0.3), rng.range(-4.4, 4.4)];
            let f = fold_ring(n, radius, room, false, 1.0, torus, u);
            let t = union(&copies, 0.0, torus, u);
            assert!(f <= t + 1e-12, "over by {}", f - t);
            if t < room {
                assert!((f - t).abs() < 1e-9, "{f} {t}");
            }
        }
    }

    #[test]
    fn an_arc_puts_its_ends_where_asked() {
        let a = ring_angles(5, 0.3, 1.2);
        assert_eq!(a.len(), 5);
        assert!((a[0] - 0.3).abs() < 1e-15 && (a[4] - 1.5).abs() < 1e-15);
        // a whole ring does not put a copy on its first
        let w = ring_angles(4, 0.0, std::f64::consts::TAU);
        assert!((w[3] - 1.5 * std::f64::consts::PI).abs() < 1e-12);
        // each copy stands at its angle, turned to face out
        let s = ring_stations(5, 0.9, 0.3, 1.2);
        for (st, t) in s.iter().zip(&a) {
            assert!((st.at[0] - 0.9 * t.cos()).abs() < 1e-12 && (st.at[2] - 0.9 * t.sin()).abs() < 1e-12);
            let out = apply(&st.frame, [1.0, 0.0, 0.0]);
            assert!((out[0] - t.cos()).abs() < 1e-12 && (out[2] - t.sin()).abs() < 1e-12);
        }
    }

    fn check_listed(copies: &[Place], field: fn([f64; 3]) -> f64, bx: ([f64; 3], [f64; 3]), blend: f64, r: f64, seed: u64) -> f64 {
        let mut rng = Rng(seed);
        let mut reads = 0;
        for _ in 0..10000 {
            let u = rng.point(r);
            let (f, read) = listed(copies, bx.0, bx.1, blend, field, u);
            let t = union(copies, blend, field, u);
            assert!((f - t).abs() < 1e-12, "{f} against {t} at {u:?}");
            reads += read;
        }
        reads as f64 / 10000.0
    }

    #[test]
    fn the_listed_reading_is_the_union_of_its_copies() {
        // an arc of 64 lumps, overlapping (they touch and cross), turned and grown per copy
        let o = ArrayOpts { span: 200.0, start: 15.0, step_turn: 9.0, step_grow: 0.01, step_move: [0.0, 0.01, 0.0], ..Default::default() };
        let arc = places(&ring_stations(64, 0.9, o.start.to_radians(), o.span.to_radians()), &o, 1.0);
        let reads = check_listed(&arc, lump, LUMP, 0.0, 1.4, 21);
        // (the cull reads few of the 64)
        eprintln!("arc of 64 lumps: {reads:.2} copies read per point");
        assert!(reads < 8.0, "copies read per point: {reads}");
        // with a blend: equal to joining the two nearest of all
        check_listed(&arc, lump, LUMP, 0.06, 1.4, 22);
        // a grid of 4 x 4 x 4 with steps (so listed), and copies that keep their heading on a whole ring
        let g = places(&grid_stations([4, 4, 4], [0.25, 0.3, 0.2], 0.2), &ArrayOpts { step_turn: 20.0, ..Default::default() }, 0.9);
        check_listed(&g, torus, TORUS, 0.03, 1.0, 23);
        let k = places(&ring_stations(12, 0.5, 0.0, 0.0), &ArrayOpts { keep_heading: true, ..Default::default() }, 1.0);
        check_listed(&k, lump, LUMP, 0.0, 1.0, 24);
    }

    #[test]
    fn copies_along_a_path_stand_evenly_by_length() {
        let path = crate::path::Path::parse("start -0.8 0 0 0 0 0; S 0.4; T 90 0.5; P 30 0.4; C 0.3 0.6 0.2 0.1 0.9 0.3 -0.2").unwrap();
        let st = path_stations(&path, 17);
        let b = crate::path_formula::build(&path, 0.0, false);
        // the path walked finely: the length between stations is the same
        let mut walk = Vec::new();
        for p in &b.prims {
            for i in 0..=4000 {
                walk.push((p.s0 + (p.s1 - p.s0) * i as f64 / 4000.0, p.at(i as f64 / 4000.0).0));
            }
        }
        let gap = b.length / 16.0;
        for (k, s) in st.iter().enumerate() {
            let want = k as f64 * gap;
            let (_, at) = walk.iter().min_by(|a, b| (a.0 - want).abs().total_cmp(&(b.0 - want).abs())).unwrap();
            assert!(len(sub(*at, s.at)) < 1e-3, "station {k}");
            // a frame: unit, square, right-handed, x along the path
            let f = s.frame;
            assert!((len(f[0]) - 1.0).abs() < 1e-9 && (len(f[1]) - 1.0).abs() < 1e-9);
            assert!(len(sub(cross(f[0], f[1]), f[2])) < 1e-9);
        }
        assert!(len(sub(st[0].at, path.start.p)) < 1e-12);
        let copies = places(&st, &ArrayOpts::default(), 1.0);
        check_listed(&copies, torus, TORUS, 0.0, 1.5, 31);
        check_listed(&copies, torus, TORUS, 0.05, 1.5, 32);
    }

    #[test]
    fn the_listed_reading_is_one_lipschitz_with_a_blend() {
        let o = ArrayOpts { span: 120.0, blend: 0.08, ..Default::default() };
        let arc = places(&ring_stations(9, 0.5, 0.0, o.span.to_radians()), &o, 1.0);
        let mut rng = Rng(41);
        let f = |u| listed(&arc, LUMP.0, LUMP.1, 0.08, lump, u).0;
        let s = slope_ok(f, &mut rng, 100000, 1.0);
        eprintln!("listed arc with blend: steepest slope {s:.6}");
        assert!(s <= 1.0 + 1e-6, "slope {s}");
        // and continuous where the copy read first changes: close pairs far apart in the copies' order
        let s = slope_ok(|u| listed(&arc, LUMP.0, LUMP.1, 0.0, lump, u).0, &mut rng, 100000, 1.0);
        assert!(s <= 1.0 + 1e-6, "slope {s}");
    }

    /// A ring of grids: each ring copy a 3 x 2 grid of tori folded (code 8), the ring listed (code 14).
    #[test]
    fn a_ring_of_grids_never_overstates_and_is_exact_near_its_copies() {
        let n = [3.0, 2.0, 1.0];
        let step = [0.36, 0.36, 1.0];
        let (minus, plus) = grid_rooms(n, step, TORUS.0, TORUS.1);
        let inner = |q: [f64; 3]| fold_grid(n, step, 0.0, minus, plus, 1.0, torus, q);
        let inner_box = ([-0.36 - 0.16, -0.18 - 0.04, -0.16], [0.36 + 0.16, 0.18 + 0.04, 0.16]);
        let o = ArrayOpts { span: 270.0, ..Default::default() };
        let ring = places(&ring_stations(7, 1.0, 0.0, o.span.to_radians()), &o, 1.0);
        // all 42 tori written out
        let grid = written_grid([3, 2, 1], step, 0.0, 1.0);
        let all: Vec<Place> = ring.iter().flat_map(|r| grid.iter().map(|g| Place { rot: mul(&r.rot, &g.rot), at: r.out(g.at), scale: r.scale * g.scale })).collect();
        let room = minus.iter().chain(&plus).fold(f64::MAX, |a, b| a.min(*b));
        let mut rng = Rng(51);
        for _ in 0..6000 {
            let u = rng.point(1.5);
            let (f, _) = listed(&ring, inner_box.0, inner_box.1, 0.0, inner, u);
            let t = union(&all, 0.0, torus, u);
            assert!(f <= t + 1e-12, "over by {}", f - t);
            if t < room {
                assert!((f - t).abs() < 1e-9);
            }
        }
        let s = slope_ok(|u| listed(&ring, inner_box.0, inner_box.1, 0.0, inner, u).0, &mut rng, 50000, 1.5);
        assert!(s <= 1.0 + 1e-6, "slope {s}");
    }

    /// A grid of rings: each grid copy a listed ring of lumps, the grid folded.
    #[test]
    fn a_grid_of_rings_never_overstates() {
        let ring = places(&ring_stations(6, 0.2, 0.0, 0.0), &ArrayOpts { step_grow: 0.05, ..Default::default() }, 1.0);
        let rb = ([-0.45, -0.2, -0.45], [0.45, 0.2, 0.45]);
        let inner = |q: [f64; 3]| listed(&ring, LUMP.0, LUMP.1, 0.0, lump, q).0;
        let n = [4.0, 1.0, 3.0];
        let step = [1.0, 1.0, 1.0];
        let (minus, plus) = grid_rooms(n, step, rb.0, rb.1);
        let grid = written_grid([4, 1, 3], step, 0.0, 1.0);
        let all: Vec<Place> = grid.iter().flat_map(|g| ring.iter().map(|r| Place { rot: r.rot, at: g.out(r.at), scale: r.scale })).collect();
        let mut rng = Rng(61);
        for _ in 0..6000 {
            let u = rng.point(2.2);
            let f = fold_grid(n, step, 0.0, minus, plus, 1.0, inner, u);
            let t = union(&all, 0.0, lump, u);
            assert!(f <= t + 1e-12, "over by {}", f - t);
            if t < 0.05 {
                assert!((f - t).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn the_packed_numbers_read_as_the_formula() {
        // a modifier with an arc and steps, packed as the node's stack packs it, read in f32 line for line
        // as the shader text does, against the copies read in f64
        let mut m = Modifier::of(Kind::Repeat);
        m.mode = 1;
        m.a = crate::mods::Axes::new([24.0, 1.0, 1.0], false);
        m.detail = 0.8;
        m.array = ArrayOpts { span: 150.0, start: -40.0, step_turn: 7.0, step_grow: 0.02, blend: 0.04, ..Default::default() };
        m.centre = [0.1, -0.05, 0.02];
        m.turn = 30.0;
        m.tilt = 70.0;
        assert!(m.listed());
        let before = [LUMP.0[0], LUMP.0[1], LUMP.0[2], LUMP.1[0], LUMP.1[1], LUMP.1[2]];
        // (the field before, in the object frame; the motif placed in the array's frame through it)
        let f = m.frame();
        let c = m.centre;
        let lb = crate::mods::local_box(&before, c, &f);
        let entry = m.listed_entry(&before);
        let rows = m.listed_rows();
        let copies = m.places();
        assert_eq!(copies.len(), 24);
        let motif = |x: [f64; 3]| box_dist(x, [before[0], before[1], before[2]], [before[3], before[4], before[5]]);
        let to_local = |x: [f64; 3]| { let d = sub(x, c); [f[0][0] * d[0] + f[0][1] * d[1] + f[0][2] * d[2], f[1][0] * d[0] + f[1][1] * d[1] + f[1][2] * d[2], f[2][0] * d[0] + f[2][1] * d[1] + f[2][2] * d[2]] };
        let to_world = |q: [f64; 3]| add(c, add(add(scale(f[0], q[0]), scale(f[1], q[1])), scale(f[2], q[2])));
        let mut rng = Rng(71);
        let mut worst = 0.0f64;
        for _ in 0..10000 {
            let x = rng.point(1.4);
            let want = listed(&copies, [lb[0], lb[1], lb[2]], [lb[3], lb[4], lb[5]], 0.04, |q| motif(to_world(q)), to_local(x)).0;
            let truth = union(&copies, 0.04, |q| motif(to_world(q)), to_local(x));
            assert!((want - truth).abs() < 1e-12);
            let got = read_packed(&entry, &rows, |w| motif(w.map(f64::from)) as f32, x.map(|v| v as f32));
            worst = worst.max((got as f64 - want).abs());
        }
        eprintln!("packed f32 against the f64 formula: {worst:e} at most");
        assert!(worst < 2e-5, "f32 against f64: {worst}");
    }

    #[test]
    fn a_plain_repeat_is_not_listed_and_keeps_its_numbers() {
        let m = Modifier::of(Kind::Repeat);
        assert!(!m.listed() && m.array.is_plain());
        let mut ring = Modifier::of(Kind::Repeat);
        ring.mode = 1;
        ring.array.span = 360.0;
        assert!(!ring.listed());
        // a grid with steps is listed up to LIST_MAX copies only
        let mut g = Modifier::of(Kind::Repeat);
        g.array.step_turn = 10.0;
        g.a = crate::mods::Axes::new([8.0, 8.0, 1.0], false);
        assert!(g.listed());
        g.a = crate::mods::Axes::new([8.0, 9.0, 1.0], false);
        assert!(!g.listed());
        // saved: nothing new is written for a plain repeat
        let text = serde_json::to_string(&m).unwrap();
        assert!(!text.contains("array"), "{text}");
        let stack = Mods { list: vec![m.clone()] };
        assert_eq!(stack.node_codes(), vec![8]);
        assert_eq!(stack.node_numbers(&crate::mods::whole_box(), &Default::default()).0, stack.exact_numbers(&crate::mods::whole_box(), &Default::default()));
    }

    #[test]
    fn a_listed_arrays_box_holds_every_copy() {
        let mut m = Modifier::of(Kind::Repeat);
        m.mode = 2;
        m.a = crate::mods::Axes::new([9.0, 1.0, 1.0], false);
        m.array.step_grow = 0.1;
        let before = [-0.1, -0.2, -0.1, 0.15, 0.1, 0.1];
        let b = m.bound(&before);
        let f = m.frame();
        for p in m.places() {
            for k in 0..8 {
                let q = [before[if k & 1 == 0 { 0 } else { 3 }], before[if k & 2 == 0 { 1 } else { 4 }], before[if k & 4 == 0 { 2 } else { 5 }]];
                let local = crate::mods::local_box(&[q[0], q[1], q[2], q[0], q[1], q[2]], m.centre, &f);
                let u = p.out([local[0], local[1], local[2]]);
                let w = add(m.centre, add(add(scale(f[0], u[0]), scale(f[1], u[1])), scale(f[2], u[2])));
                for i in 0..3 {
                    assert!(w[i] >= b[i] - 1e-9 && w[i] <= b[i + 3] + 1e-9);
                }
            }
        }
    }
}
