//! The parts engine's CPU side: a scene of many parts as a table of pieces (one record each), their bounding
//! balls, and the lists of which pieces each cell of the scene's grid holds (`cells`). No GPU.
//!
//! A piece is one part of a tree (or one copy of a part, or the whole of an exact tube) in a form a compute
//! shader reads: the way from the scene into the part's own frame, the part's kind and sizes after the studio's
//! fit, its scale back to the scene, its join (union or cut) and seam. The GPU side (`app/src/parts_volume.rs`)
//! bakes the scene's volume from the pieces cell by cell; this module makes the pieces and keeps the lists.
//!
//! What is read: the studio's sphere, ball with a cut, block, cylinder, disc, capsule, ellipsoid, oval, washer,
//! bandring, slab, bar, frame and tube bodies (no bevel: a bevel is refused, `accepts`), the solids' round ring,
//! parts stretched unevenly (the app's own reading, `stretch.rs`: a lower bound), the app's exact tubes as an
//! object's own shape, an object's twist / bend / taper as a point warp (`Mods::exact_numbers`), an object's array
//! as separate pieces, one per copy, two objects joined, and several objects set apart or placed.
//!
//! Everything it cannot read is refused with a sentence (`accepts`); the caller then keeps the existing path.

pub mod cells;

use crate::csg::{GridFrom, Op, Resolved};
use crate::layers::Placement;
use crate::state::{Detail, ShapeJob, State};

/// The studio's fit (presets/studio.rs): a shape reaching further than this from its middle is drawn in to it.
pub const FIT: f64 = 1.27;

/// The piece kinds (`Piece::a[0]`).
pub mod kind {
    pub const SPHERE: f32 = 0.0;
    pub const BLOCK: f32 = 1.0;
    pub const CYLINDER: f32 = 2.0;
    pub const CAPSULE: f32 = 3.0;
    /// the solids' ring with a round section: b = (big radius, tube radius)
    pub const RING: f32 = 4.0;
    pub const ELLIPSOID: f32 = 5.0;
    pub const OVAL: f32 = 6.0;
    /// the app's exact tube table: b = (the table's offset in entries, half the side of its grid box)
    pub const TUBE: f32 = 7.0;
    /// a ring with a rectangular section lying along y (washer, bandring; the tube is the same with x and y swapped in its frame)
    pub const WASHER: f32 = 8.0;
    pub const SLAB: f32 = 9.0;
    /// a ball with a cut (a dome on a flat base)
    pub const CUT_BALL: f32 = 10.0;
    pub const FRAME: f32 = 12.0;
}

/// One piece: seven vec4 the shaders read as they stand (`as_f32`).
///
/// - `inv`: three rows of four, the way from the scene into the part's own frame (the part's placement, the object's,
///   an array copy's, an uneven stretch's all folded in);
/// - `a`: (kind, the scale from the part's frame back to the scene, 1 union / 2 cut / 3 a hole cut later in the list
///   (the live pass) / 0 left out, seam radius);
/// - `b`, `c`: the kind's sizes after the studio's fit, and more of them (see `kind`; `c[3]` is the point warp's
///   number, 1-based, 0 for none);
/// - `w`: a warped piece's object's guard sphere in the scene (centre, radius);
/// - `x`: x[0] is 1 for a piece of the second object of a joined pair (its own accumulator).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Piece {
    pub inv: [[f32; 4]; 3],
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub c: [f32; 4],
    pub w: [f32; 4],
    pub x: [f32; 4],
}

/// Floats in a piece.
pub const PIECE_FLOATS: usize = 28;

impl Piece {
    /// The record as the shaders read it (seven vec4).
    pub fn as_f32(&self) -> [f32; PIECE_FLOATS] {
        let mut o = [0.0; PIECE_FLOATS];
        for r in 0..3 { o[r * 4..r * 4 + 4].copy_from_slice(&self.inv[r]); }
        o[12..16].copy_from_slice(&self.a);
        o[16..20].copy_from_slice(&self.b);
        o[20..24].copy_from_slice(&self.c);
        o[24..28].copy_from_slice(&self.w);
        // (x is the eighth vec4: the table keeps it apart so that the records stay 32 floats)
        o
    }

    /// The record with all eight vec4 (x last): 32 floats, 128 bytes.
    pub fn as_f32_all(&self) -> [f32; 32] {
        let mut o = [0.0; 32];
        o[..PIECE_FLOATS].copy_from_slice(&self.as_f32());
        o[28..32].copy_from_slice(&self.x);
        o
    }

    pub fn is_warped(&self) -> bool { self.c[3] != 0.0 }
    pub fn is_cut(&self) -> bool { self.a[2] > 1.5 }
    pub fn seam(&self) -> f64 { self.a[3] as f64 }

    /// The point in the part's own frame.
    pub fn local(&self, p: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|r| self.inv[r][0] as f64 * p[0] + self.inv[r][1] as f64 * p[1] + self.inv[r][2] as f64 * p[2] + self.inv[r][3] as f64)
    }

    /// A lower bound of the piece's own distance at the scene point `p` (the exact distance for the kinds that have a closed
    /// form and an even stretch; the bounding block for the ellipsoid, oval, frame and tube; for a warped piece the way to its
    /// object's guard sphere, a bound inside it too: a ray out of the shape meets its surface before it leaves the sphere).
    pub fn lower(&self, p: [f64; 3]) -> f64 {
        if self.is_warped() { return self.warp_lower(p); }
        let q = self.local(p);
        let b = self.b.map(|x| x as f64);
        let len = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let boxed = |h: [f64; 3]| {
            let e = [0, 1, 2].map(|a| q[a].abs() - h[a]);
            len(&e.map(|x| x.max(0.0))) + e[0].max(e[1]).max(e[2]).min(0.0)
        };
        let d = match self.a[0] {
            k if k == kind::SPHERE => len(&q) - b[0],
            k if k == kind::BLOCK => {
                let e = [0, 1, 2].map(|a| q[a].abs() - (b[a] - b[3]));
                len(&e.map(|x| x.max(0.0))) + e[0].max(e[1]).max(e[2]).min(0.0) - b[3]
            }
            k if k == kind::CYLINDER => {
                let e = [(q[0] * q[0] + q[2] * q[2]).sqrt() - (b[0] - b[3]), q[1].abs() - (b[1] - b[3])];
                e[0].max(e[1]).min(0.0) + len(&e.map(|x| x.max(0.0))) - b[3]
            }
            k if k == kind::CAPSULE => len(&[(q[0].abs() - b[0]).max(0.0), q[1], q[2]]) - b[1],
            k if k == kind::ELLIPSOID || k == kind::OVAL || k == kind::FRAME => boxed([b[0], b[1], b[2]]),
            k if k == kind::TUBE => boxed([b[1]; 3]),
            k if k == kind::CUT_BALL => len(&[q[0], q[1] + self.c[0] as f64, q[2]]) - (b[0] + b[3]),
            k if k == kind::WASHER => boxed([b[0] + b[1], b[2], b[0] + b[1]]),
            k if k == kind::SLAB => boxed([b[0], b[1], b[2]]),
            _ => len(&[(q[0] * q[0] + q[2] * q[2]).sqrt() - b[0], q[1]]) - b[1],
        };
        d * self.a[1] as f64
    }

    fn warp_lower(&self, p: [f64; 3]) -> f64 {
        ((0..3).map(|a| (p[a] - self.w[a] as f64).powi(2)).sum::<f64>()).sqrt() - self.w[3] as f64
    }

    /// The piece's own distance at the scene point `p`: exact for the kinds with a closed form (the ellipsoid and oval by
    /// the app's own reading, `stretch::ell_f64`), a lower bound for the rest (`lower`); `exact` says which.
    pub fn dist(&self, p: [f64; 3]) -> f64 {
        if self.is_warped() { return self.warp_lower(p); }
        let q = self.local(p);
        let b = self.b.map(|x| x as f64);
        let len = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let sc = self.a[1] as f64;
        let k = self.a[0];
        if k == kind::ELLIPSOID { return crate::stretch::ell_f64::<3>(q, [b[0], b[1], b[2]]) * sc; }
        if k == kind::OVAL {
            let r = b[3];
            let a = crate::stretch::ell_f64::<2>([q[0], q[2]], [b[0], b[2]]) + r;
            let c = q[1].abs() - (b[1] - r);
            let (x, y) = (a.max(c), a.min(c));
            return ((if x <= 0.0 { x } else { len(&[x, y.max(0.0)]) }) - r) * sc;
        }
        if k == kind::WASHER {
            let e = [(q[0] * q[0] + q[2] * q[2]).sqrt() - b[0], q[1]].map(f64::abs);
            let e = [e[0] - b[1] + b[3], e[1] - b[2] + b[3]];
            let (x, y) = (e[0].max(e[1]), e[0].min(e[1]));
            return ((if x <= 0.0 { x } else { len(&[x, y.max(0.0)]) }) - b[3]) * sc;
        }
        if k == kind::SLAB {
            let w = [q[0].abs() - (b[0] - self.c[0] as f64), q[2].abs() - (b[2] - self.c[0] as f64)];
            let a = len(&w.map(|x| x.max(0.0))) + w[0].max(w[1]).min(0.0) - self.c[1] as f64;
            let c = q[1].abs() - (b[1] - b[3]);
            let (x, y) = (a.max(c), a.min(c));
            return ((if x <= 0.0 { x } else { len(&[x, y.max(0.0)]) }) - b[3]) * sc;
        }
        if k == kind::FRAME {
            let rr = b[3];
            let kk = self.c[0] as f64 - rr;
            let rim = self.c[1] as f64;
            let rrect = |hx: f64, hy: f64, k: f64| {
                let o = [q[0].abs() - (hx - k), q[1].abs() - (hy - k)];
                len(&o.map(|x| x.max(0.0))) + o[0].max(o[1]).min(0.0) - k
            };
            let outer = rrect(b[0] - rr, b[1] - rr, kk);
            let k2 = (self.c[0] as f64 - rim).max(0.0) + rr;
            let opening = rrect(b[0] - rim + rr, b[1] - rim + rr, k2);
            let a = outer.max(-opening);
            let c = q[2].abs() - (b[2] - rr);
            let (x, y) = (a.max(c), a.min(c));
            return ((if x <= 0.0 { x } else { len(&[x, y.max(0.0)]) }) - rr) * sc;
        }
        if k == kind::CUT_BALL {
            let u = [len(&[q[0], q[2]]), q[1] + self.c[0] as f64];
            let p1 = [b[2], b[1]];
            let (sn, cs) = (self.c[1] as f64, self.c[2] as f64);
            let on_ball = if u[0] * cs - u[1] * sn <= 0.0 { (len(&u) - b[0]).abs() } else { len(&[u[0] - p1[0], u[1] - p1[1]]) };
            let dd = on_ball.min(len(&[(u[0] - b[2]).max(0.0), u[1] - b[1]]));
            let outside = (len(&u) - b[0]).max(b[1] - u[1]);
            return ((if outside <= 0.0 { -dd } else { dd }) - b[3]) * sc;
        }
        self.lower(p)
    }

    /// Does `dist` give the exact distance (an even stretch, a closed form)?
    pub fn exact(&self) -> bool { !self.is_warped() && self.a[0] != kind::TUBE }

    /// The piece's bounding box in the scene (its box in its own frame carried out through the placement), grown by its seam.
    pub fn aabb(&self) -> ([f64; 3], [f64; 3]) {
        if self.is_warped() {
            let r = self.w[3] as f64;
            return ([0, 1, 2].map(|a| self.w[a] as f64 - r), [0, 1, 2].map(|a| self.w[a] as f64 + r));
        }
        let m = [0, 1, 2].map(|r| [0, 1, 2].map(|c| self.inv[r][c] as f64));
        let t = [0, 1, 2].map(|r| self.inv[r][3] as f64);
        let e = self.box_half();
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let id = 1.0 / det;
        let mi = [
            [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id],
            [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id],
            [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id],
        ];
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for corner in 0..8 {
            let q = [0, 1, 2].map(|a| (if corner >> a & 1 == 1 { e[a] } else { -e[a] }) - t[a]);
            for a in 0..3 {
                let v = mi[a][0] * q[0] + mi[a][1] * q[1] + mi[a][2] * q[2];
                lo[a] = lo[a].min(v);
                hi[a] = hi[a].max(v);
            }
        }
        let s = self.a[3] as f64;
        (lo.map(|x| x - s), hi.map(|x| x + s))
    }

    /// The half sizes of the block that holds the piece, in the part's own frame (before the scale back).
    pub fn box_half(&self) -> [f64; 3] {
        let b = self.b.map(|x| x as f64);
        let k = self.a[0];
        if k == kind::SPHERE { [b[0]; 3] }
        else if k == kind::BLOCK || k == kind::ELLIPSOID || k == kind::OVAL || k == kind::FRAME || k == kind::SLAB { [b[0], b[1], b[2]] }
        else if k == kind::TUBE { [b[1]; 3] }
        else if k == kind::WASHER { [b[0] + b[1], b[2], b[0] + b[1]] }
        else if k == kind::CUT_BALL { let r = b[0] + b[3]; [r, r + (self.c[0] as f64).abs(), r] }
        else if k == kind::CYLINDER { [b[0], b[1], b[0]] }
        else if k == kind::CAPSULE { [b[0] + b[1], b[1], b[1]] }
        else { [b[0] + b[1], b[1], b[0] + b[1]] }
    }

    /// How thick the piece is at its thinnest, in scene units (infinity for the kinds not read as thin: tubes, warped pieces, cuts).
    pub fn thickness(&self) -> f64 {
        if self.is_warped() || self.is_cut() { return f64::MAX; }
        let b = self.b.map(|x| x as f64);
        let k = self.a[0];
        let t = if k == kind::SPHERE { 2.0 * b[0] }
            else if k == kind::BLOCK || k == kind::ELLIPSOID || k == kind::OVAL { 2.0 * b[0].min(b[1]).min(b[2]) }
            else if k == kind::CYLINDER { 2.0 * b[0].min(b[1]) }
            else if k == kind::CAPSULE || k == kind::RING { 2.0 * b[1] }
            else if k == kind::WASHER { 2.0 * b[1].min(b[2]) }
            else if k == kind::SLAB { 2.0 * b[1] }
            else if k == kind::FRAME { (self.c[1] as f64).min(2.0 * b[2]) }
            else { return f64::MAX };
        t * self.a[1] as f64
    }
}

/// Which pieces are thin: their thickness is under `limit` (scene units).
pub fn thin_pieces(pieces: &[Piece], limit: f64) -> Vec<bool> {
    pieces.iter().map(|o| o.thickness() < limit).collect()
}

// ---------------------------------------------------------------- placements

/// A point in a placement's own units, out into the scene (the way back of `Placement::inverse`, whose rows are R^T / size).
pub fn to_scene(place: &Placement, p: [f64; 3]) -> [f64; 3] {
    let inv = place.inverse();
    let s2 = place.size * place.size;
    [0, 1, 2].map(|i| place.pos[i] + s2 * (0..3).map(|j| inv[j * 4 + i] * p[j]).sum::<f64>())
}

/// q = a . (b . p): two maps one after the other (three rows of four): b first, then a.
pub fn compose(a: &[f64; 12], b: &[f64; 12]) -> [f64; 12] {
    let mut o = [0.0; 12];
    for r in 0..3 {
        for c in 0..4 {
            let mut s = if c == 3 { a[r * 4 + 3] } else { 0.0 };
            for k in 0..3 { s += a[r * 4 + k] * b[k * 4 + c]; }
            o[r * 4 + c] = s;
        }
    }
    o
}

pub fn apply34(m: &[f64; 12], p: [f64; 3]) -> [f64; 3] { [0, 1, 2].map(|r| m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3]) }

/// The inverse of a 3 x 4 map (a turn or a scaled one).
pub fn inv34(m: &[f64; 12]) -> [f64; 12] {
    let a = |r: usize, c: usize| m[r * 4 + c];
    let det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
    let id = 1.0 / det;
    let r = [
        [(a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * id, (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * id, (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * id],
        [(a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * id, (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * id, (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * id],
        [(a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * id, (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * id, (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * id],
    ];
    let mut o = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 { o[i * 4 + j] = r[i][j]; }
        o[i * 4 + 3] = -(r[i][0] * m[3] + r[i][1] * m[7] + r[i][2] * m[11]);
    }
    o
}

// ---------------------------------------------------------------- objects and parts

/// One object of the scene (a layer that shows).
#[derive(Clone)]
pub struct Object {
    /// its place in the app's list of layers
    pub layer: usize,
    /// where it stands in the scene
    pub place: Placement,
    /// its parts in its own units, joined in order; the first is the object's own shape when the object is not a tree of its own
    pub parts: Vec<Resolved>,
}

/// A part with nothing but a shape (the object's own shape, as the first part of its tree).
pub fn bare(fam: usize, preset: usize, vals: Vec<f64>) -> Resolved {
    Resolved { fam, preset, vals, puff: 0.0, place: Placement::default(), op: Op::Union, smooth: 0.0, smooth_rel: false, level: 0, grid: None, source: GridFrom::Up, formula_at: None, formula_key: String::new(), model: String::new(), strokes: Default::default(), stretch: [1.0; 3] }
}

/// Every object that shows, with its parts (a tree's own shape first, as `State::csg_tree_layer` gives them).
pub fn objects(state: &State) -> Result<Vec<Object>, String> {
    let mut out = Vec::new();
    for i in 0..state.stack.layers.len() {
        let layer = &state.stack.layers[i];
        if !layer.show { continue; }
        let (mut parts, joined, _dropped) = state.csg_tree_layer(i);
        if joined || parts.is_empty() {
            let st = state.layer_shape(i).ok_or("a layer without a shape")?;
            parts.insert(0, bare(state.layer_fam(i), st.preset, st.vals.clone()));
        }
        out.push(Object { layer: i, place: layer.place, parts });
    }
    Ok(out)
}

/// An exact tube object: (its table's offset in entries, half the side of its grid box, the box's middle in the object's units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tube {
    pub base: f32,
    pub half: f32,
    pub mid: [f64; 3],
}

fn param(state: &State, part: &Resolved, id: &str) -> f64 {
    let p = &state.families[part.fam].presets[part.preset];
    p.params.iter().position(|d| d.id == id).map_or(0.0, |k| part.vals[k])
}

/// A part's kind and sizes: (kind, b, c, reach), in the part's own units after the studio's fit. Err: a sentence saying why it is not read.
pub fn kind_of(state: &State, part: &Resolved) -> Result<(f32, [f32; 4], [f32; 4], f64), String> {
    let f = &state.families[part.fam];
    let p = &f.presets[part.preset];
    let v = |id: &str| param(state, part, id);
    let no_bevel = |what: &str| if v("bevel") > 0.0 { Err(format!("the {what} with a bevel is not drawn by the parts engine (set the bevel to 0)")) } else { Ok(()) };
    let fit = |reach: f64| (FIT / reach).min(1.0);
    let f4 = |a: [f64; 4], s: f64| a.map(|x| (x * s) as f32);
    // the solids' ring with a round section (presets/solids.rs ONE_RING: fitted to 1.22 in its own way)
    if f.id == "solids" && p.id == "ring" && (v("section") - 2.0).abs() < 1e-6 && (v("flat") - 1.0).abs() < 1e-6 {
        // (presets/solids.rs ONE_RING: the farthest point is a corner of the section, for a round one the ring's own outside)
        let fat = v("fat");
        let big = crate::presets::solids::FIT / (1.0 + fat).hypot(fat * v("flat"));
        return Ok((kind::RING, [big as f32, (fat * big) as f32, 0.0, 0.0], [0.0; 4], big * (1.0 + fat)));
    }
    if f.id != "studio" {
        return Err(format!("the shape {}/{} is not one of the studio's sphere, ball with a cut, block, cylinder, disc, capsule, ellipsoid, oval, washer, band ring, slab, bar, frame or tube, or the solids' round ring", f.id, p.id));
    }
    match p.id {
        "sphere" => {
            if v("cut") <= 0.0 { return Ok((kind::SPHERE, [v("radius") as f32, 0.0, 0.0, 0.0], [0.0; 4], v("radius"))); }
            no_bevel("ball with a cut")?;
            let big = v("radius");
            let base = -big + 2.0 * big * v("cut");
            let r = v("round").min(0.45 * (big - base));
            let rad = big - r;
            let ground = base + r;
            let w = (rad * rad - ground * ground).max(0.0).sqrt();
            let s = fit(big);
            let (sn, cs) = (w / rad, (ground / rad).clamp(-1.0, 1.0));
            Ok((kind::CUT_BALL, f4([rad, ground, w, r], s), [(0.5 * (big + base) * s) as f32, sn as f32, cs as f32, 0.0], big * s))
        }
        "block" | "bar" => {
            no_bevel(p.id)?;
            let h = [0.5 * v(if p.id == "bar" { "length" } else { "width" }), 0.5 * v("height"), 0.5 * v("depth")];
            let reach = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
            let r = v("round").min(0.98 * h[0].min(h[1]).min(h[2]));
            Ok((kind::BLOCK, f4([h[0], h[1], h[2], r], fit(reach)), [0.0; 4], reach * fit(reach)))
        }
        "cylinder" | "disc" => {
            no_bevel(p.id)?;
            let h = [v("radius"), 0.5 * v("height")];
            let r = v("round").min(0.98 * h[0].min(h[1]));
            let reach = (h[0] * h[0] + h[1] * h[1]).sqrt();
            Ok((kind::CYLINDER, f4([h[0], h[1], 0.0, r], fit(reach)), [0.0; 4], reach * fit(reach)))
        }
        "capsule" => {
            let reach = (0.5 * v("length")).max(v("radius"));
            Ok((kind::CAPSULE, f4([(0.5 * v("length") - v("radius")).max(0.0), v("radius"), 0.0, 0.0], fit(reach)), [0.0; 4], reach * fit(reach)))
        }
        "ellipsoid" => {
            let reach = v("rx").max(v("ry")).max(v("rz"));
            let s = fit(reach);
            Ok((kind::ELLIPSOID, f4([v("rx"), v("ry"), v("rz"), 0.0], s), [0.0; 4], reach * s))
        }
        "oval" => {
            no_bevel("oval")?;
            let h = [v("rx"), 0.5 * v("height"), v("rz")];
            let reach = (h[0].max(h[2]).powi(2) + h[1] * h[1]).sqrt();
            let s = fit(reach);
            let round = v("round").min(0.98 * h[0].min(h[1]).min(h[2]));
            Ok((kind::OVAL, f4([h[0], h[1], h[2], round], s), [0.0; 4], reach * s))
        }
        "slab" => {
            no_bevel("slab")?;
            let h = [0.5 * v("width"), 0.5 * v("height"), 0.5 * v("depth")];
            let reach = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
            let s = fit(reach);
            let r = v("round").min(0.98 * h[0].min(h[1]).min(h[2]));
            let k = v("corner").clamp(r, h[0].min(h[2]));
            Ok((kind::SLAB, f4([h[0], h[1], h[2], r], s), f4([k, k - r, 0.0, 0.0], s), reach * s))
        }
        "rectframe" => {
            no_bevel("frame")?;
            let half = [0.5 * v("width"), 0.5 * v("height")];
            let rim = v("rim").min(half[0].min(half[1]) - 0.02);
            let deep = 0.5 * v("depth");
            let r = v("round").min(0.98 * (0.5 * rim).min(deep));
            let k = v("corner").clamp(r, half[0].min(half[1]));
            let reach = (half[0] * half[0] + half[1] * half[1] + deep * deep).sqrt();
            let s = fit(reach);
            Ok((kind::FRAME, f4([half[0], half[1], deep, r], s), f4([k, rim, 0.0, 0.0], s), reach * s))
        }
        "washer" | "bandring" | "tube" => {
            no_bevel(p.id)?;
            let hole = v("hole").min(v("outer") - 0.02);
            let (outer, height) = (v("outer"), if p.id == "tube" { v("length") } else { v("height") });
            let half = [0.5 * (outer - hole), 0.5 * height];
            let reach = (outer * outer + half[1] * half[1]).sqrt();
            let s = fit(reach);
            let r = v("round").min(0.98 * half[0].min(half[1]));
            // (the tube lies along x: the washer with its x and y swapped, done in the piece's frame: c[2] = 1)
            Ok((kind::WASHER, f4([0.5 * (outer + hole), half[0], half[1], r], s), [0.0, 0.0, if p.id == "tube" { 1.0 } else { 0.0 }, 0.0], reach * s))
        }
        other => Err(format!("the studio's {other} is not drawn by the parts engine (it draws the sphere, ball with a cut, block, cylinder, disc, capsule, ellipsoid, oval, washer, band ring, slab, bar, frame and tube)")),
    }
}

/// A part as a piece, and its bounding ball (centre in the scene, reach with its seam). `obj`: where the part's object stands
/// when it is placed or one of several (the part's frame is then composed with it); `tube`: the part is an exact tube object's own
/// shape; `copy`: one copy of an array (the map from the object's frame to the motif's, the scale back).
pub fn piece_of(state: &State, r: &Resolved, obj: Option<&Placement>, tube: Option<Tube>, copy: Option<(&[f64; 12], f64)>) -> Result<(Piece, [f64; 4]), String> {
    if r.puff != 0.0 { return Err("inflated (puff) parts are not drawn by the parts engine".into()); }
    if r.op == Op::Intersect { return Err("a part that intersects is not drawn by the parts engine (union and cut only)".into()); }
    let mut cc = [0.0f32; 4];
    let (kind, b, reach) = if let Some(t) = tube {
        (kind::TUBE, [t.base, t.half, 0.0, 0.0], t.half as f64 * 1.7320508)
    } else {
        let (k, b, c, rc) = kind_of(state, r)?;
        cc = c;
        (k, b, rc)
    };
    // (a part stretched unevenly, the app's reading: the stretched space measured, the distance times its least stretch, a lower bound; stretch.rs)
    let st = r.stretch.map(|x| x.max(1e-9));
    let (low, high) = (st[0].min(st[1]).min(st[2]), st[0].max(st[1]).max(st[2]));
    // (a tube: the frame is moved to its table's box middle, which the shader adds back)
    let own = tube.map_or(r.place.pos, |t| t.mid);
    // (a copy of an array: the object's frame taken to the motif's first, the copy's scale back on the value)
    let (pi, csc, own) = match copy {
        Some((m, s)) => (compose(&r.place.inverse(), m), s, apply34(&inv34(m), own)),
        None => (r.place.inverse(), 1.0, own),
    };
    let (mut inv, osize, centre) = match obj {
        Some(o) if !o.is_identity() => (compose(&pi, &o.inverse()), o.size.abs(), to_scene(o, own)),
        _ => (pi, 1.0, own),
    };
    if let Some(t) = tube { for row in 0..3 { inv[row * 4 + 3] -= t.mid[row]; } }
    let swap = kind == kind::WASHER && cc[2] == 1.0;
    if swap { cc[2] = 0.0; }
    let src = |row: usize| if swap { [1, 0, 2][row] } else { row };
    let op = if r.op == Op::Subtract { 2.0 } else { 1.0 };
    let p = Piece {
        inv: [0, 1, 2].map(|row| [0, 1, 2, 3].map(|c| (inv[src(row) * 4 + c] / st[src(row)]) as f32)),
        a: [kind, (r.place.size.abs() * osize * low * csc) as f32, op, r.smooth as f32],
        b,
        c: cc,
        w: [0.0; 4],
        x: [0.0; 4],
    };
    let rad = (reach * r.place.size.abs() * osize * high) * csc + r.smooth;
    Ok((p, [centre[0], centre[1], centre[2], rad]))
}

// ---------------------------------------------------------------- arrays

/// An object's array as separate pieces: each copy's map (3 x 4, object units to the motif's), the scale back, the parts of one copy.
#[derive(Clone, Debug)]
pub struct ArrayInfo {
    pub maps: Vec<[f64; 12]>,
    pub scale: f64,
    pub np: usize,
}

/// An array's copies (one modifier entry of `Mods::exact_numbers`, `modBack` case 8: a grid or a whole ring) as maps from the object's
/// frame to the motif's (3 x 4 each), and the entry's scale back. A copy is a piece of its own: the value is the least over the copies (the
/// union of the copies, which is what the app's overlap route and listed arrays read).
pub fn array_maps(q: &[f32]) -> Result<(Vec<[f64; 12]>, f64), String> {
    let g = |i: usize| q[i] as f64;
    let (mode, sc, flip) = (g(1), g(2), g(3));
    if flip > 0.5 { return Err("a ring of mirrored copies is not drawn by the parts engine".into()); }
    let centre = [g(4), g(5), g(6)];
    // the frame's columns (modLocal: u = F^T (x - centre))
    let f = [[g(8), g(9), g(10)], [g(12), g(13), g(14)], [g(16), g(17), g(18)]];
    let (a, st, rad, stagger) = ([g(20), g(21), g(22)], [g(24), g(25), g(26)], g(15), g(31));
    let mut copies: Vec<([[f64; 3]; 3], [f64; 3])> = Vec::new();
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if mode > 0.5 {
        if a[0] > 1.5 {
            let n = a[0].round() as usize;
            let al = std::f64::consts::TAU / a[0];
            for k in 0..n {
                let t = -(k as f64) * al;
                let (c, s) = (t.cos(), t.sin());
                copies.push(([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], [-rad, 0.0, 0.0]));
            }
        } else { copies.push((eye, [-rad, 0.0, 0.0])); }
    } else {
        let n = [0, 1, 2].map(|i| if a[i] > 1.5 { a[i].round() as usize } else { 1 });
        for ix in 0..n[0] { for iy in 0..n[1] { for iz in 0..n[2] {
            let id = [ix, iy, iz];
            let mut t = [0.0; 3];
            for i in 0..3 { if a[i] > 1.5 { t[i] = -st[i] * (id[i] as f64 - 0.5 * (a[i].round() - 1.0)); } }
            if a[1] > 1.5 { t[0] -= stagger * st[0] * (iy % 2) as f64; }
            copies.push((eye, t));
        } } }
    }
    if copies.len() > MAX_COPIES { return Err(format!("an array of {} copies is not drawn by the parts engine ({MAX_COPIES} at most)", copies.len())); }
    let mut maps = Vec::new();
    for (m, t) in copies {
        // x' = centre + F (M F^T (x - centre) / sc + t) = A x + b
        let mut aa = [[0.0; 3]; 3];
        for r in 0..3 { for c in 0..3 {
            let mut v = 0.0;
            for i in 0..3 { for j in 0..3 { v += f[i][r] * m[i][j] * f[j][c]; } }
            aa[r][c] = v / sc;
        } }
        let mut o = [0.0; 12];
        for r in 0..3 {
            let ft = f[0][r] * t[0] + f[1][r] * t[1] + f[2][r] * t[2];
            for c in 0..3 { o[r * 4 + c] = aa[r][c]; }
            o[r * 4 + 3] = centre[r] + ft - (aa[r][0] * centre[0] + aa[r][1] * centre[1] + aa[r][2] * centre[2]);
        }
        maps.push(o);
    }
    Ok((maps, sc))
}

/// The most copies of an array read as pieces.
pub const MAX_COPIES: usize = 256;

// ---------------------------------------------------------------- warps

/// An object's point warp as the shaders read it: the app's exact entries (`Mods::exact_numbers`: a header, then 36 numbers a
/// modifier), the guard sphere the warped shape stays within (scene centre and radius), and the stack's 1 / Lipschitz bound
/// (what a distance read through the warp is multiplied by, and a piece's seam with it).
#[derive(Clone, Debug)]
pub struct Warp {
    pub data: Vec<f32>,
    pub ball: [f32; 4],
    pub inv_lipschitz: f32,
}

// ---------------------------------------------------------------- the scene

/// Two objects joined (the app's `combine`: min(a, b) less what a seam of the pair's radius and profile takes off where they differ).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    pub radius: f64,
    /// 1 bevel, 2 round, 3 smooth
    pub power: u8,
    /// the two objects' sizes
    pub sizes: [f64; 2],
}

impl Pair {
    /// How far a seam reaches beyond the surfaces, in scene units. Measured against the app's own combined volume (two spheres of size 0.7,
    /// radius 0.4, bevel: the inside counts agree to 2% at the pair's radius over the objects' size, 13% short at the radius itself):
    /// the radius over the geometric mean of the two sizes. (The profiles' own factors the lab fitted are not used.)
    pub fn reach(&self) -> f64 { self.radius.max(0.0) / (self.sizes[0].abs() * self.sizes[1].abs()).sqrt().max(1e-6) }
}

/// The scene as pieces. Built by `PartsScene::build`.
pub struct PartsScene {
    /// per object: (its layer, its first piece, its pieces)
    pub objs_at: Vec<(usize, usize, usize)>,
    /// per piece: (its object, its place in the object's list as expanded: copy after copy)
    pub owner: Vec<(usize, usize)>,
    pub pieces: Vec<Piece>,
    pub balls: Vec<[f64; 4]>,
    /// per piece: its exact tube, if it is one
    pub tubes: Vec<Option<Tube>>,
    /// the tubes' tables, one after another (entries of four floats), and the first tube preset's formula text
    pub tube_table: Vec<f32>,
    pub tube_text: Option<String>,
    pub pair: Option<Pair>,
    /// per object: its array
    pub arrays: Vec<Option<ArrayInfo>>,
    /// per object: its warp (1-based into `warps`, 0 none)
    pub obj_warp: Vec<usize>,
    pub warps: Vec<Warp>,
    /// the objects stand placed or several (the parts' frames carry the object's placement)
    pub multi: bool,
    /// the widest seam any join in the scene reaches (the floor of the volume's value cap is twice this)
    pub smooth: f64,
    /// the distinct (family, preset) of the formula parts, in the order the generated kind text numbers them
    pub kinds: Vec<(usize, usize)>,
    /// per piece: its kind's number in `kinds` as the shader reads it (`TUBE_KIND` for an exact tube)
    pub part_kind: Vec<f32>,
    /// per piece: the record the shaders read (`gpu_record`)
    pub recs: Vec<[f32; GPU_FLOATS]>,
}

/// The kind number of an exact tube in a piece's record.
pub const TUBE_KIND: f32 = 1000.0;
/// Floats in a piece's record as the shaders read it: inv (3 vec4), a, v0, v1, w, x.
pub const GPU_FLOATS: usize = 32;

fn slider_block(r: &Resolved) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for (k, x) in r.vals.iter().take(8).enumerate() { v[k] = *x as f32; }
    v
}

/// A piece as the shaders read it: the frame, (kind, scale, op, seam), the part's own sliders (an exact tube: its table and box), the
/// block that holds it (the reading is never under its distance), (second object of a pair, warp number, 1 if it has the block).
pub fn gpu_record(p: &Piece, kind: f32, vals: &[f32; 8], boxed: bool) -> [f32; GPU_FLOATS] {
    let mut o = [0.0f32; GPU_FLOATS];
    for r in 0..3 { o[r * 4..r * 4 + 4].copy_from_slice(&p.inv[r]); }
    o[12..16].copy_from_slice(&[kind, p.a[1], p.a[2], p.a[3]]);
    if kind == TUBE_KIND { o[16..20].copy_from_slice(&p.b); } else { o[16..24].copy_from_slice(vals); }
    // (every kind is held to its block's distance where that is more, as the app's own tree reads a part far from its box (`csgBox`): an
    // estimate (a walk to the surface of limited length) runs out far from a small part, and the exact reading of a very flat ellipsoid
    // in single precision is too low far from it; a value too low there is a surface that is not there. The block is never nearer than the part.)
    if boxed { let e = p.box_half(); o[24..27].copy_from_slice(&[e[0] as f32, e[1] as f32, e[2] as f32]); }
    o[28..32].copy_from_slice(&[p.x[0], p.c[3], if boxed { 1.0 } else { 0.0 }, p.x[3]]);
    o
}

/// A piece made again from the state.
pub struct Derived {
    pub piece: Piece,
    pub ball: [f64; 4],
    pub rec: [f32; GPU_FLOATS],
}

/// The way each refusal is told: a sentence.
pub type Refusal = String;

/// Can the parts engine draw this scene? Every refusal is a sentence (the caller keeps the existing path and shows it).
pub fn accepts(state: &State) -> Result<(), Refusal> {
    PartsScene::plan(state).map(|_| ())
}

impl PartsScene {
    /// The scene as pieces. An exact tube is read as a placeholder box until `load_tubes` reads its table (building a job needs the state
    /// mutable); that is enough to accept or refuse a scene.
    pub fn plan(state: &State) -> Result<PartsScene, Refusal> {
        if state.body_materials().is_some() { return Err("per-body materials need the bodies' volume, which the parts engine does not keep".into()); }
        let objs = objects(state)?;
        if objs.is_empty() { return Err("no object shows".into()); }
        let mut sc = PartsScene { objs_at: vec![], owner: vec![], pieces: vec![], balls: vec![], tubes: vec![], tube_table: vec![], tube_text: None, pair: None, arrays: vec![], obj_warp: vec![], warps: vec![], multi: false, smooth: 0.0, kinds: vec![], part_kind: vec![], recs: vec![] };
        for o in &objs { sc.check_parts(state, o)?; }
        sc.arrays = sc.compute_arrays(state, &objs)?;
        let objs = sc.expand_arrays(objs);
        // (several objects, or one placed: their parts joined in one list, each with its object's placement composed in. Objects set apart or joined
        // in a pair only (no cuts or morphs between them), and no cut inside one of several that reaches another.)
        sc.multi = objs.len() > 1 || !objs[0].place.is_identity();
        if sc.multi && state.stack.bodies().iter().any(|m| m.count_ones() > 1) {
            let st = &state.stack;
            let po = (objs.len() == 2 && st.layers.len() == 2).then(|| st.pair(crate::layers::mask_of(&[objs[0].layer, objs[1].layer])));
            match po {
                Some(po) if po.op == crate::layers::Op::Join => sc.pair = Some(Pair { radius: po.radius, power: po.profile.power(), sizes: [objs[0].place.size, objs[1].place.size] }),
                _ => return Err("only two objects joined (a join with its seam profile) are drawn by the parts engine; three or more joined objects and cuts or morphs between objects are not".into()),
            }
        }
        let mut first = 0;
        sc.objs_at = objs.iter().map(|o| { let t = (o.layer, first, o.parts.len()); first += o.parts.len(); t }).collect();
        sc.owner = objs.iter().enumerate().flat_map(|(k, o)| (0..o.parts.len()).map(move |j| (k, j))).collect();
        sc.read_tubes(state, &objs)?;
        sc.compute_warps(state, &objs)?;
        for (i, (r, k0)) in objs.iter().enumerate().flat_map(|(k, o)| o.parts.iter().map(move |r| (r, k))).enumerate() {
            let (k, j) = sc.owner[i];
            let cp = sc.copy_args(k, j);
            let place = objs[k0].place;
            let (mut p, mut b) = piece_of(state, r, if sc.multi { Some(&place) } else { None }, sc.tubes[i], cp.as_ref().map(|c| (&c.0, c.1)))?;
            sc.warp_piece(k, &mut p, &mut b);
            sc.pair_piece(k, &mut p, &mut b);
            p.x[3] = k as f32;
            let kind = if sc.tubes[i].is_some() { TUBE_KIND } else {
                let at = sc.kinds.iter().position(|&x| x == (r.fam, r.preset)).unwrap_or_else(|| { sc.kinds.push((r.fam, r.preset)); sc.kinds.len() - 1 });
                at as f32
            };
            sc.recs.push(gpu_record(&p, kind, &slider_block(r), kind != TUBE_KIND));
            sc.part_kind.push(kind);
            sc.pieces.push(p);
            sc.balls.push(b);
        }
        sc.smooth = objs.iter().flat_map(|o| o.parts.iter()).map(|r| r.smooth).fold(0.0, f64::max).max(sc.pair.map_or(0.0, |p| 0.5 * p.reach()));
        // (a cut in one of several objects cuts, in the one list, every part before it: right only where it does not reach another object)
        Ok(sc)
    }

    /// The value cap's floor for a volume of `res` voxels over a cube of half-side `hb` (one cell, twice the widest seam, four voxels).
    pub fn band(&self, res: u32, hb: f64) -> f64 {
        let cs = 2.0 * hb / 64.0;
        cs.max(2.0 * self.smooth).max(4.0 * 2.0 * hb / res as f64)
    }

    /// A cut in one of several objects reaching another object (one list would cut both).
    pub fn check_cuts(&self, band: f64) -> Result<(), Refusal> {
        for i in 0..self.pieces.len() {
            if !self.pieces[i].is_cut() { continue; }
            for j in 0..self.pieces.len() {
                if self.pieces[j].is_cut() || self.owner[j].0 == self.owner[i].0 { continue; }
                let (a, b) = (self.balls[i], self.balls[j]);
                let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
                if d < a[3] + b[3] + band { return Err(format!("a cut in object {} reaches object {} (one list would cut both); only objects far enough apart are drawn by the parts engine", self.owner[i].0, self.owner[j].0)); }
            }
        }
        Ok(())
    }

    fn check_parts(&self, state: &State, o: &Object) -> Result<(), Refusal> {
        for (j, r) in o.parts.iter().enumerate() {
            let pre = &state.families[r.fam].presets[r.preset];
            if r.grid.is_some() || !r.model.is_empty() {
                return Err("parts read from a volume of their own (drawn curves, models, paths, sculpted volumes) are not drawn by the parts engine".into());
            }
            if pre.exact_tubes && (j != 0 || r.source != GridFrom::Up) { return Err("an exact tube (an attractor) is drawn by the parts engine only as an object's own shape, not inside a tree".into()); }
            if !pre.exact_tubes {
                // (the part's kind and sizes: Err is its sentence)
                kind_of(state, r)?;
            }
            if !r.strokes.is_none() { return Err("parts with strokes (shell, hole, round, cone, star, mirror, elongate) are not drawn by the parts engine".into()); }
            if r.puff != 0.0 { return Err("inflated (puff) parts are not drawn by the parts engine".into()); }
            if r.op == Op::Intersect { return Err("a part that intersects is not drawn by the parts engine (union and cut only)".into()); }
        }
        Ok(())
    }

    /// An object's shape box in its own units: its additive parts' boxes (a larger box only makes a bound safer).
    pub fn object_box(state: &State, o: &Object) -> Result<[f64; 6], String> {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for r in o.parts.iter().filter(|r| r.op != Op::Subtract) {
            let (p, _) = piece_of(state, r, None, None, None)?;
            let (l, h) = p.aabb();
            for a in 0..3 { lo[a] = lo[a].min(l[a]); hi[a] = hi[a].max(h[a]); }
        }
        Ok([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]])
    }

    /// Arrays: an object whose one modifier is a repeat (a grid or a whole ring) is held as separate pieces, one set of the parts for each
    /// copy (the value the least over the copies). Parts must be union only with no seam (a cut or seam in one copy would reach the others).
    fn compute_arrays(&self, state: &State, objs: &[Object]) -> Result<Vec<Option<ArrayInfo>>, Refusal> {
        let mut out = Vec::new();
        for o in objs {
            let Some(mods) = state.layer_mods(o.layer).filter(|m| m.any_live()) else { out.push(None); continue };
            if mods.codes() != [8] { out.push(None); continue; }
            if o.parts.iter().any(|r| r.op != Op::Union || r.smooth != 0.0) { return Err("an array of a tree with cuts or seams is not drawn by the parts engine (a cut in one copy would cut the others)".into()); }
            if o.parts.iter().any(|r| state.families[r.fam].presets[r.preset].exact_tubes) { return Err("an array of exact tubes is not drawn by the parts engine".into()); }
            let bx = Self::object_box(state, o)?;
            if !mods.exact(&bx) { return Err("this array cannot be traced exactly (its copies need a bake)".into()); }
            let nums = mods.exact_numbers(&bx, &o.place);
            let (maps, scale) = array_maps(&nums[crate::csg::PART_FLOATS..crate::csg::PART_FLOATS + 36])?;
            out.push(Some(ArrayInfo { maps, scale, np: o.parts.len() }));
        }
        Ok(out)
    }

    /// The objects with each array's parts repeated for every copy (copy after copy).
    fn expand_arrays(&self, objs: Vec<Object>) -> Vec<Object> {
        objs.into_iter().enumerate().map(|(k, mut o)| {
            if let Some(a) = self.arrays.get(k).and_then(|a| a.as_ref()) {
                let one = o.parts.clone();
                o.parts = (0..a.maps.len()).flat_map(|_| one.iter().cloned()).collect();
            }
            o
        }).collect()
    }

    /// Piece `j` of object `k` (in the expanded list): its copy's map and scale, if the object is an array.
    pub fn copy_args(&self, k: usize, j: usize) -> Option<([f64; 12], f64)> {
        let a = self.arrays.get(k)?.as_ref()?;
        Some((a.maps[j / a.np], a.scale))
    }

    /// Where piece `j` of object `k` (expanded) stands in the object's own list of parts.
    pub fn part_index(&self, k: usize, j: usize) -> usize {
        self.arrays.get(k).and_then(|a| a.as_ref()).map_or(j, |a| j % a.np)
    }

    /// Exact tubes (attractors): placeholders (a box) until `load_tubes`.
    fn read_tubes(&mut self, state: &State, objs: &[Object]) -> Result<(), Refusal> {
        for o in objs {
            for (j, r) in o.parts.iter().enumerate() {
                let pre = &state.families[r.fam].presets[r.preset];
                if !pre.exact_tubes { self.tubes.push(None); continue; }
                if j != 0 || r.source != GridFrom::Up { return Err("an exact tube (an attractor) is drawn by the parts engine only as an object's own shape, not inside a tree".into()); }
                self.tubes.push(Some(Tube { base: 0.0, half: 1.0, mid: [0.0; 3] }));
            }
        }
        Ok(())
    }

    /// The exact tubes' tables read from the app's own jobs (one table after another, an offset each), and their pieces made again.
    pub fn load_tubes(&mut self, state: &mut State) -> Result<(), Refusal> {
        self.tube_table.clear();
        self.tube_text = None;
        let tube_pieces: Vec<usize> = (0..self.tubes.len()).filter(|&i| self.tubes[i].is_some()).collect();
        for &i in &tube_pieces {
            let (k, _) = self.owner[i];
            let layer = self.objs_at[k].0;
            let r = objects(state)?.get(k).and_then(|o| o.parts.first().cloned()).ok_or("an exact tube without its part")?;
            let mut job = state.layer_job(layer, Detail::Full, None)?;
            let fams = &state.families;
            job.prepare_formula(fams)?;
            let ShapeJob::Field { table, .. } = job.shape else { return Err("an exact tube without its table".into()) };
            let (lo, w, n) = ([table[4] as f64, table[5] as f64, table[6] as f64], table[7] as f64, table[8] as f64);
            let half = 0.5 * n * w;
            let base = (self.tube_table.len() / 4) as f32;
            self.tube_table.extend_from_slice(&table);
            self.tubes[i] = Some(Tube { base, half: half as f32, mid: [lo[0] + half, lo[1] + half, lo[2] + half] });
            if self.tube_text.is_none() { self.tube_text = Some(fams[r.fam].presets[r.preset].formula().ok_or("an exact tube without its formula")?.0.to_string()); }
        }
        for &i in &tube_pieces {
            let d = self.derive(state, &[i])?.pop().ok_or("no tube piece")?;
            self.pieces[i] = d.piece;
            self.balls[i] = d.ball;
            self.recs[i] = d.rec;
        }
        Ok(())
    }

    /// A piece of a warped object: its warp's number, its sphere and its seam times the stack's bound (the shader joins the scaled distances).
    fn warp_piece(&self, obj: usize, p: &mut Piece, b: &mut [f64; 4]) {
        let w = self.obj_warp.get(obj).copied().unwrap_or(0);
        if w == 0 { return; }
        let wp = &self.warps[w - 1];
        let seam = p.a[3];
        p.c[3] = w as f32;
        p.w = wp.ball;
        p.a[3] = seam * wp.inv_lipschitz;
        *b = [wp.ball[0] as f64, wp.ball[1] as f64, wp.ball[2] as f64, wp.ball[3] as f64 + seam as f64];
    }

    /// Two objects joined: the second object's pieces are in the pair's second accumulator, and every piece of the pair reaches the seam further.
    fn pair_piece(&self, obj: usize, p: &mut Piece, b: &mut [f64; 4]) {
        let Some(pair) = self.pair else { return };
        if obj == 1 { p.x[0] = 1.0; }
        b[3] += pair.reach();
    }

    /// Point warps: an object's twist, bend or taper, as the app's tracer traces them (exact entries, the guard sphere, the bound).
    fn compute_warps(&mut self, state: &State, objs: &[Object]) -> Result<(), Refusal> {
        for o in objs {
            let Some(mods) = state.layer_mods(o.layer).filter(|m| m.any_live()) else { self.obj_warp.push(0); continue };
            let codes = mods.codes();
            if codes == [8] { self.obj_warp.push(0); continue; }
            if codes.iter().any(|c| !matches!(c, 1 | 3 | 4)) {
                return Err(format!("the parts engine draws the point warps twist, bend and taper, or one array (a grid or a whole ring) alone (modifier codes {codes:?}: the other kinds, and arrays with other modifiers, are not)"));
            }
            let bx = Self::object_box(state, o)?;
            if !mods.exact(&bx) { return Err("this stack of modifiers cannot be traced exactly (its copies touch or it needs a bake)".into()); }
            let nums = mods.exact_numbers(&bx, &o.place);
            let n = nums[0] as usize;
            let mut data = nums[..28].to_vec();
            for j in 0..n { data.extend_from_slice(&nums[crate::csg::PART_FLOATS * (1 + j)..crate::csg::PART_FLOATS * (1 + j) + crate::mods::MOD_FLOATS]); }
            // (the last modifier's guard: the sphere the warped shape stays within, about its centre; scene units through the object's rows)
            let last = &nums[crate::csg::PART_FLOATS * n..];
            let (c, gr, so) = ([last[4] as f64, last[5] as f64, last[6] as f64], last[23] as f64, nums[1] as f64);
            if gr <= 0.0 { return Err("a point warp without a guard sphere".into()); }
            let f = &nums[16..28];
            let centre = [0, 1, 2].map(|r| (f[r * 4] as f64 * c[0] + f[r * 4 + 1] as f64 * c[1] + f[r * 4 + 2] as f64 * c[2] + f[r * 4 + 3] as f64) as f32);
            let inv_l: f64 = (0..n).map(|j| nums[crate::csg::PART_FLOATS * (1 + j) + 7] as f64).product();
            self.warps.push(Warp { data, ball: [centre[0], centre[1], centre[2], (gr * so) as f32], inv_lipschitz: inv_l as f32 });
            self.obj_warp.push(self.warps.len());
        }
        Ok(())
    }

    /// Piece `i` and its ball as the state stands now (a slider or a drag changed a part): the same reading as `plan`'s for one piece.
    /// The scene's arrays and warps are as they were made: call `refresh` after a change to a modifier's own numbers or a shape's size.
    pub fn rederive(&self, state: &State, i: usize) -> Result<(Piece, [f64; 4]), Refusal> {
        let (k, j) = self.owner[i];
        let objs = objects(state)?;
        let o = objs.get(k).ok_or("no such object")?;
        let r = o.parts.get(self.part_index(k, j)).ok_or("no such part")?;
        let cp = self.copy_args(k, j);
        let (mut p, mut b) = piece_of(state, r, if self.multi { Some(&o.place) } else { None }, self.tubes[i], cp.as_ref().map(|c| (&c.0, c.1)))?;
        self.warp_piece(k, &mut p, &mut b);
        self.pair_piece(k, &mut p, &mut b);
        Ok((p, b))
    }

    /// Pieces `idx` as the state stands now, with their records (one reading of the state for all). A part of a kind the scene was not built
    /// with is refused (the kinds' text is compiled once).
    pub fn derive(&self, state: &State, idx: &[usize]) -> Result<Vec<Derived>, Refusal> {
        let objs = objects(state)?;
        idx.iter().map(|&i| {
            let (k, j) = self.owner[i];
            let o = objs.get(k).ok_or("no such object")?;
            let r = o.parts.get(self.part_index(k, j)).ok_or("no such part")?;
            let cp = self.copy_args(k, j);
            let (mut p, mut b) = piece_of(state, r, if self.multi { Some(&o.place) } else { None }, self.tubes[i], cp.as_ref().map(|c| (&c.0, c.1)))?;
            self.warp_piece(k, &mut p, &mut b);
            self.pair_piece(k, &mut p, &mut b);
            p.x[3] = k as f32;
            let kind = self.part_kind[i];
            if kind != TUBE_KIND && self.kinds.get(kind as usize) != Some(&(r.fam, r.preset)) { return Err("the part is a different kind of shape than it was built with".into()); }
            Ok(Derived { rec: gpu_record(&p, kind, &slider_block(r), kind != TUBE_KIND), piece: p, ball: b })
        }).collect()
    }

    /// The arrays' maps and the warps made again from the state (a slider moved a shape's size, which the shape box and the modifiers' bounds follow).
    /// The number of copies must not change.
    pub fn refresh(&mut self, state: &State) -> Result<(), Refusal> {
        let objs = objects(state)?;
        let before: Vec<usize> = self.arrays.iter().map(|a| a.as_ref().map_or(0, |x| x.maps.len())).collect();
        self.arrays = self.compute_arrays(state, &objs)?;
        let after: Vec<usize> = self.arrays.iter().map(|a| a.as_ref().map_or(0, |x| x.maps.len())).collect();
        if before != after { return Err("the change altered the number of an array's copies".into()); }
        let n = self.warps.iter().map(|w| w.data.len()).collect::<Vec<_>>();
        self.warps.clear();
        self.obj_warp.clear();
        self.compute_warps(state, &objs)?;
        if self.warps.iter().map(|w| w.data.len()).collect::<Vec<_>>() != n { return Err("the change altered the structure of a point warp".into()); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(args: &[&str]) -> State {
        let pairs: Vec<(String, String)> = args.iter().map(|a| { let (k, v) = a.split_once('=').unwrap(); (k.to_string(), v.to_string()) }).collect();
        State::with_args(&pairs, true)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) }
        fn range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.next() }
    }

    const ARRAY: &str = r#"mods=[{"kind":"repeat","mode":1,"a":{"v":[8,1,1],"link":false},"b":{"v":[3.0,2.2,3.8],"link":false},"detail":2.4}]"#;
    const GRID: &str = r#"mods=[{"kind":"repeat","mode":0,"a":{"v":[3,2,2],"link":false},"b":{"v":[2.2,2.4,2.2],"link":false},"stagger":0.5}]"#;
    const BEND: &str = r#"mods=[{"kind":"bend","a":{"v":[0,140,0],"link":false}}]"#;
    const TWIST: &str = r#"mods=[{"kind":"twist","a":{"v":[0,180,0],"link":false}}]"#;

    /// lib_array's eight balls: each copy's piece is a ball of the copy's radius on the ring, and the least of their distances is the
    /// least over the eight analytic balls.
    #[test]
    fn ring_copies_are_balls_on_the_ring() {
        let state = st(&["scene=studio", "preset=sphere", "material=glass", ARRAY, "dist=6.0"]);
        let objs = objects(&state).unwrap();
        let o = &objs[0];
        let mods = state.layer_mods(o.layer).expect("mods");
        let bx = PartsScene::object_box(&state, o).unwrap();
        let nums = mods.exact_numbers(&bx, &o.place);
        let (maps, sc) = array_maps(&nums[crate::csg::PART_FLOATS..crate::csg::PART_FLOATS + 36]).unwrap();
        assert_eq!(maps.len(), 8);
        let (plain, _) = piece_of(&state, &o.parts[0], None, None, None).unwrap();
        let r0 = plain.b[0] as f64 * plain.a[1] as f64;
        let (mut pieces, mut balls) = (Vec::new(), Vec::new());
        for m in &maps {
            let (p, b) = piece_of(&state, &o.parts[0], None, None, Some((m, sc))).unwrap();
            pieces.push(p);
            balls.push(b);
        }
        let ring = 2.4 * sc;
        for b in &balls {
            assert!(((b[0] * b[0] + b[2] * b[2]).sqrt() - ring).abs() < 1e-5, "centre off the ring: {b:?}");
            assert!(b[1].abs() < 1e-6);
            assert!((b[3] - r0 * sc).abs() < 1e-5, "radius {} against {}", b[3], r0 * sc);
        }
        for i in 0..8 { for j in i + 1..8 { let d = ((balls[i][0] - balls[j][0]).powi(2) + (balls[i][2] - balls[j][2]).powi(2)).sqrt(); assert!(d > 0.1); } }
        for p in [[0.3, 0.2, -0.4], [1.5, 0.1, 0.2], [-0.8, 0.5, 0.9], [0.0, 0.0, 0.0]] {
            let least = pieces.iter().map(|q| q.dist(p)).fold(f64::MAX, f64::min);
            let analytic = balls.iter().map(|b| ((p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2) + (p[2] - b[2]).powi(2)).sqrt() - b[3]).fold(f64::MAX, f64::min);
            assert!((least - analytic).abs() < 1e-4, "{least} against {analytic} at {p:?}");
        }
    }

    /// lib_chamfer: two objects, one join, radius 0.4, bevel (power 1), sizes 0.7.
    #[test]
    fn chamfer_pair_is_read() {
        let state = st(&["scene=studio", "preset=sphere", "material=glass", "layer=studio:sphere", "J12.op=join", "J12.profile=bevel", "J12.radius=0.4", "L1.x=-0.5", "L2.x=0.5", "L1.size=0.7", "L2.size=0.7", "dist=5.0"]);
        let sc = PartsScene::plan(&state).expect("accepted");
        let pair = sc.pair.expect("a pair");
        assert!((pair.radius - 0.4).abs() < 1e-9);
        assert_eq!(pair.power, 1);
        assert_eq!(sc.objs_at.len(), 2);
        assert!(sc.pieces.iter().enumerate().all(|(i, p)| (p.x[0] == 1.0) == (sc.owner[i].0 == 1)));
        assert!(accepts(&state).is_ok());
    }

    /// A rigging capsule is thin, a hull block is not, a cut is never thin.
    #[test]
    fn thin_pieces_are_the_stays() {
        let mut o = Piece::default();
        o.a = [kind::CAPSULE, 1.0, 1.0, 0.0];
        o.b = [0.5, 0.004, 0.0, 0.0];
        assert!(o.thickness() < 0.01);
        o.a = [kind::BLOCK, 1.0, 1.0, 0.0];
        o.b = [0.5, 0.3, 0.2, 0.0];
        assert!(o.thickness() > 0.3);
        o.a[2] = 2.0;
        assert_eq!(o.thickness(), f64::MAX);
        let v = thin_pieces(&[o, Piece { a: [kind::SPHERE, 1.0, 1.0, 0.0], b: [0.001, 0.0, 0.0, 0.0], ..Default::default() }], 0.01);
        assert_eq!(v, vec![false, true]);
    }

    // ---- the studio's bodies, read from presets/studio.rs (edge2, edge3, rrect, the bodies, the fit of the tail), in f64

    fn edge2(a: f64, b: f64, c: f64) -> f64 {
        let rt2 = 0.70710678f64;
        let (x, y) = (a.max(b), a.min(b));
        let s = (x + y + c) * rt2;
        if x <= 0.0 && s <= 0.0 { return x.max(s); }
        let on_face = x.hypot((y + c).max(0.0));
        let t = ((x - y) * rt2).clamp(0.0, c * rt2);
        let on_flat = (x - (-0.5 * c + t * rt2)).hypot(y - (-0.5 * c - t * rt2));
        on_face.min(on_flat)
    }
    fn edge3(q: [f64; 3], c: f64) -> f64 {
        let rt2 = 0.70710678f64;
        let x = q[0].max(q[1]).max(q[2]);
        let z = q[0].min(q[1]).min(q[2]);
        let y = q[0] + q[1] + q[2] - x - z;
        let s = (x + y + c) * rt2;
        if x <= 0.0 && s <= 0.0 { return x.max(s); }
        let on_face = (x * x + (y + c).max(0.0).powi(2) + (z + c).max(0.0).powi(2)).sqrt();
        let t = (x - y) * rt2;
        let w = c * rt2;
        let mut d = 0.0f64;
        if t > w || z + t * rt2 + 0.5 * c > 0.0 {
            let ab = [w, -0.5 * c];
            let pa = [t, z + 0.5 * c];
            let k = ((pa[0] * ab[0] + pa[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-12)).clamp(0.0, 1.0);
            let slant = (pa[0] - ab[0] * k).hypot(pa[1] - ab[1] * k);
            let straight = (t - w).hypot((z + c).max(0.0));
            d = slant.min(straight);
        }
        on_face.min(s.hypot(d))
    }
    fn rrect(p: [f64; 2], half: [f64; 2], k: f64) -> f64 {
        let o = [p[0].abs() - (half[0] - k), p[1].abs() - (half[1] - k)];
        o[0].max(0.0).hypot(o[1].max(0.0)) + o[0].max(o[1]).min(0.0) - k
    }
    fn seg(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let pa = [q[0] - a[0], q[1] - a[1]];
        let ba = [b[0] - a[0], b[1] - a[1]];
        let k = ((pa[0] * ba[0] + pa[1] * ba[1]) / (ba[0] * ba[0] + ba[1] * ba[1]).max(1e-12)).clamp(0.0, 1.0);
        (pa[0] - ba[0] * k).hypot(pa[1] - ba[1] * k)
    }
    fn off(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let ba = [b[0] - a[0], b[1] - a[1]];
        let l = ba[0].hypot(ba[1]);
        if l < 1e-6 { return -1e9; }
        (ba[1] * (q[0] - a[0]) - ba[0] * (q[1] - a[1])) / l
    }

    /// The studio's own distance for a preset at these sliders (bevel 0): (body at p, reach), before the tail's fit.
    fn studio_body(id: &str, v: &dyn Fn(&str) -> f64, p: [f64; 3]) -> (f64, f64) {
        let trims = |m: f64| (v("round").min(0.98 * m), 0.0);
        let box_reach = |a: &str| 0.5 * ((v(a)).powi(2) + v("height").powi(2) + v("depth").powi(2)).sqrt();
        match id {
            "sphere" if v("cut") <= 0.0 => ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - v("radius"), v("radius")),
            "sphere" => {
                let big = v("radius");
                let base = -big + 2.0 * big * v("cut");
                let pp = [p[0], p[1] + 0.5 * (big + base), p[2]];
                let r = v("round").min(0.45 * (big - base));
                let rad = big - r;
                let ground = base + r;
                let w = (rad * rad - ground * ground).max(0.0).sqrt();
                let rim = (ground / rad).clamp(-1.0, 1.0).acos();
                let c = 0.0f64.min(0.9 * w).min(0.5 * rad * rim);
                let up = rim - c / rad;
                let p1 = [rad * up.sin(), rad * up.cos()];
                let p2 = [w - c, ground];
                let q = [pp[0].hypot(pp[2]), pp[1]];
                let on_ball = if q[0] * up.cos() - q[1] * up.sin() <= 0.0 { (q[0].hypot(q[1]) - rad).abs() } else { (q[0] - p1[0]).hypot(q[1] - p1[1]) };
                let d = on_ball.min(seg(q, p2, p1).min((q[0] - p2[0]).max(0.0).hypot(q[1] - ground)));
                let outside = (q[0].hypot(q[1]) - rad).max(ground - q[1]).max(off(q, p2, p1));
                ((if outside <= 0.0 { -d } else { d }) - r, big)
            }
            "block" | "bar" => {
                let a = if id == "bar" { "length" } else { "width" };
                let half = [0.5 * v(a), 0.5 * v("height"), 0.5 * v("depth")];
                let t = trims(half[0].min(half[1]).min(half[2]));
                (edge3([p[0].abs() - (half[0] - t.0), p[1].abs() - (half[1] - t.0), p[2].abs() - (half[2] - t.0)], t.1) - t.0, box_reach(a))
            }
            "cylinder" | "disc" => {
                let half = [v("radius"), 0.5 * v("height")];
                let t = trims(half[0].min(half[1]));
                let a = [p[0].hypot(p[2]) - (half[0] - t.0), p[1].abs() - (half[1] - t.0)];
                (edge2(a[0], a[1], t.1) - t.0, half[0].hypot(half[1]))
            }
            "capsule" => {
                let line = (0.5 * v("length") - v("radius")).max(0.0);
                (((p[0].abs() - line).max(0.0)).hypot(p[1]).hypot(p[2]) - v("radius"), (0.5 * v("length")).max(v("radius")))
            }
            "ellipsoid" => (crate::stretch::ell_f64::<3>(p, [v("rx"), v("ry"), v("rz")]), v("rx").max(v("ry")).max(v("rz"))),
            "oval" => {
                let half = [v("rx"), 0.5 * v("height"), v("rz")];
                let t = trims(half[0].min(half[2]).min(half[1]));
                (edge2(crate::stretch::ell_f64::<2>([p[0], p[2]], [half[0], half[2]]) + t.0, p[1].abs() - (half[1] - t.0), t.1) - t.0, v("rx").max(v("rz")).hypot(0.5 * v("height")))
            }
            "slab" => {
                let half = [0.5 * v("width"), 0.5 * v("height"), 0.5 * v("depth")];
                let t = trims(half[0].min(half[1]).min(half[2]));
                let k = v("corner").clamp(t.0 + t.1, half[0].min(half[2]));
                (edge2(rrect([p[0], p[2]], [half[0] - t.0, half[2] - t.0], k - t.0), p[1].abs() - (half[1] - t.0), t.1) - t.0, box_reach("width"))
            }
            "rectframe" => {
                let half = [0.5 * v("width"), 0.5 * v("height")];
                let rim = v("rim").min(half[0].min(half[1]) - 0.02);
                let deep = 0.5 * v("depth");
                let t = trims((0.5 * rim).min(deep));
                let k = v("corner").clamp(t.0 + t.1, half[0].min(half[1]));
                let outer = rrect([p[0], p[1]], [half[0] - t.0, half[1] - t.0], k - t.0);
                let opening = rrect([p[0], p[1]], [half[0] - rim + t.0, half[1] - rim + t.0], (k - rim).max(0.0) + t.0);
                (edge2(outer.max(-opening), p[2].abs() - (deep - t.0), t.1) - t.0, box_reach("width"))
            }
            "washer" | "bandring" | "tube" => {
                let hole = v("hole").min(v("outer") - 0.02);
                let long = if id == "tube" { v("length") } else { v("height") };
                let half = [0.5 * (v("outer") - hole), 0.5 * long];
                let t = trims(half[0].min(half[1]));
                let (radial, along) = if id == "tube" { (p[1].hypot(p[2]), p[0]) } else { (p[0].hypot(p[2]), p[1]) };
                let a = [(radial - 0.5 * (v("outer") + hole)).abs() - (half[0] - t.0), along.abs() - (half[1] - t.0)];
                (edge2(a[0], a[1], t.1) - t.0, v("outer").hypot(0.5 * long))
            }
            other => panic!("no reference for {other}"),
        }
    }

    /// piece_of of every studio body the parts engine draws equals the studio's own distance (its body in the tail's fit) at 1000 random points.
    #[test]
    fn studio_bodies_equal_the_studios_own_distance() {
        let state = st(&["scene=studio", "preset=sphere"]);
        let fam = state.families.iter().position(|f| f.id == "studio").unwrap();
        let mut rng = Rng(7);
        let mut seen = 0;
        for (pi, pre) in state.families[fam].presets.iter().enumerate() {
            if !matches!(pre.id, "sphere" | "block" | "bar" | "cylinder" | "disc" | "capsule" | "ellipsoid" | "oval" | "slab" | "rectframe" | "washer" | "bandring" | "tube") { continue; }
            // (the sliders at their defaults, a bevel of none; a ball once without and once with a cut)
            let cases: Vec<Vec<(&str, f64)>> = if pre.id == "sphere" { vec![vec![], vec![("cut", 0.4), ("round", 0.05)]] } else { vec![vec![]] };
            for case in cases {
                let mut vals: Vec<f64> = pre.params.iter().map(|d| d.value).collect();
                for (k, d) in pre.params.iter().enumerate() {
                    if d.id == "bevel" { vals[k] = 0.0; }
                    if let Some(c) = case.iter().find(|c| c.0 == d.id) { vals[k] = c.1; }
                }
                let part = bare(fam, pi, vals.clone());
                let (piece, ball) = piece_of(&state, &part, None, None, None).unwrap_or_else(|e| panic!("{}: {e}", pre.id));
                let v = |id: &str| pre.params.iter().position(|d| d.id == id).map_or(0.0, |k| vals[k]);
                let reach = studio_body(pre.id, &v, [0.0; 3]).1;
                let s = (FIT / reach).min(1.0);
                for _ in 0..1000 {
                    let p = [rng.range(-1.6, 1.6), rng.range(-1.6, 1.6), rng.range(-1.6, 1.6)];
                    let want = studio_body(pre.id, &v, [p[0] / s, p[1] / s, p[2] / s]).0 * s;
                    let got = piece.dist(p);
                    assert!((got - want).abs() < 1e-5, "{} at {p:?}: piece {got} studio {want}", pre.id);
                }
                assert!(ball[3] > 0.0);
                seen += 1;
            }
        }
        assert_eq!(seen, 14, "every body the parts engine draws was compared");
    }

    /// The pieces of a tree with a placed object equal what `rederive` makes again.
    #[test]
    fn rederive_makes_the_same_pieces() {
        let state = st(&["scene=studio", "preset=block", "material=glass", "csg=studio/sphere@0.3,0.2,0@10,20,30@0.4@union@0.05@0;studio/cylinder@-0.5,0,0.1@0,0,45@0.3@subtract@0@0", "L1.x=0.4", "L1.size=0.8"]);
        let sc = PartsScene::plan(&state).unwrap();
        assert!(sc.pieces.len() >= 3);
        for i in 0..sc.pieces.len() {
            let (p, b) = sc.rederive(&state, i).unwrap();
            assert_eq!(p, sc.pieces[i]);
            assert_eq!(b, sc.balls[i]);
        }
        assert!(sc.pieces.iter().any(|p| p.is_cut()));
    }

    #[test]
    fn warps_arrays_and_many_parts_are_read() {
        let bend = st(&["scene=studio", "preset=capsule", "material=glass", BEND, "dist=5.0"]);
        let sc = PartsScene::plan(&bend).unwrap();
        assert_eq!(sc.warps.len(), 1);
        assert!(sc.pieces[0].is_warped());
        assert!(sc.warps[0].inv_lipschitz > 0.0 && sc.warps[0].inv_lipschitz <= 1.0);
        let twist = st(&["scene=studio", "preset=block", "material=glass", TWIST, "dist=5.0"]);
        assert!(accepts(&twist).is_ok());
        let ring = st(&["scene=studio", "preset=sphere", "material=glass", ARRAY]);
        assert_eq!(PartsScene::plan(&ring).unwrap().pieces.len(), 8);
        let grid = st(&["scene=studio", "preset=sphere", "material=glass", GRID]);
        assert_eq!(PartsScene::plan(&grid).unwrap().pieces.len(), 12);
        let mut sc = PartsScene::plan(&ring).unwrap();
        sc.refresh(&ring).unwrap();
        assert_eq!(sc.pieces.len(), 8);
    }

    /// The generated kind text (the app's own formulas) is valid WGSL for every kind the parts engine draws.
    #[test]
    fn generated_kinds_text_validates() {
        let state = st(&["scene=studio", "preset=sphere"]);
        let fam = state.families.iter().position(|f| f.id == "studio").unwrap();
        let kinds: Vec<(usize, usize)> = state.families[fam].presets.iter().enumerate()
            .filter(|(_, p)| matches!(p.id, "sphere" | "block" | "bar" | "cylinder" | "disc" | "capsule" | "ellipsoid" | "oval" | "slab" | "rectframe" | "washer" | "bandring" | "tube"))
            .map(|(i, _)| (fam, i)).collect();
        assert_eq!(kinds.len(), 13);
        let text = format!("struct Piece {{ inv0: vec4f, inv1: vec4f, inv2: vec4f, a: vec4f, v0: vec4f, v1: vec4f, w: vec4f, x: vec4f }}
@group(0) @binding(0) var<storage, read> pcs: array<Piece>;
{}{}
@compute @workgroup_size(1) fn cs() {{ _ = partKind(0, 0u, vec3f(0.0)); }}
", crate::stretch::ELL_WGSL, crate::csg::parts_kinds_wgsl(&state.families, &kinds));
        let module = naga::front::wgsl::parse_str(&text).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
    }

    /// Every refusal is a sentence.
    #[test]
    fn accepts_refuses_each_case_with_its_sentence() {
        let refused = |args: &[&str], want: &str| {
            let r = accepts(&st(args));
            match r { Err(e) => assert!(e.contains(want), "{args:?}: '{e}' does not say '{want}'"), Ok(()) => panic!("{args:?} was accepted") }
        };
        refused(&["scene=studio", "preset=lens", "material=glass"], "not drawn");
        refused(&["scene=gems", "preset=emerald", "material=glass"], "is not one of the studio's");
        refused(&["scene=studio", "preset=block", "p.bevel=0.03", "material=glass"], "bevel");
        refused(&["scene=studio", "preset=block", "csg=studio/sphere@0,0,0@0,0,0@0.4@intersect@0@0"], "intersects");
        refused(&["scene=studio", "preset=sphere", r#"mods=[{"kind":"ripple","a":{"v":[0.1,0.1,0.1],"link":false}}]"#], "point warps");
        refused(&["scene=studio", "preset=sphere", "csg=studio/block@0.3,0,0@0,0,0@0.4@subtract@0@0", ARRAY], "cuts or seams");
        refused(&["scene=studio", "preset=sphere", r#"mods=[{"kind":"repeat","mode":0,"a":{"v":[12,12,12],"link":false},"b":{"v":[3.0,3.0,3.0],"link":false}}]"#], "copies");
        refused(&["scene=studio", "preset=sphere", "layer=studio:sphere", "layer=studio:sphere", "J12.op=join", "J13.op=join", "J23.op=join"], "two objects joined");
        refused(&["scene=studio", "preset=sphere", "layer=studio:sphere", "J12.op=a_cuts_b", "L1.x=-0.3", "L2.x=0.3"], "two objects joined");
    }
}
