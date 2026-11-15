//! Polygon models: a mesh file read, examined, and made ready to be turned into a distance volume.
//!
//! Nothing here needs a GPU. `read` turns a file into triangles. `examine` (this file) merges corners
//! that coincide, drops triangles that are none, winds every piece one way, finds the gaps in the surface
//! and closes them, and says in one line what it found and what it did. `solid` then works out what a
//! volume needs that only the CPU gives exactly: which voxels are inside, which triangles are the outside
//! of the solid, and the distance far from the surface.
//!
//! A hole THROUGH a shape (a ring, a handle) is not a gap: the surface around it is whole. A gap is where
//! the surface itself stops: an edge with a triangle on one side only. Gaps are closed with flat fans of
//! triangles, which makes the whole thing a closed surface however broken the file was; whether what it
//! then encloses is worth calling a solid is judged from how much of it the fans are.

pub mod read;
pub mod solid;

use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A triangle mesh as it comes out of a file: positions, and triangles as three places in `pos`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub pos: Vec<[f32; 3]>,
    pub tri: Vec<[u32; 3]>,
}

/// Corners closer together than this share of the model's size are one corner.
pub const WELD: f64 = 1e-6;
/// A gap whose fan is no more than this share of the closed surface is a small one: little of the solid
/// is a guess. (A gap may be long and still small: a slit takes little to close.)
pub const SMALL_GAP: f64 = 0.03;
/// Fans that make up more than this share of the closed surface: there is no solid, only a sheet.
pub const OPEN_SHARE: f64 = 0.45;
/// A closed-up model thinner than this share of its own size (volume over the fans' area) is a sheet too.
pub const THIN: f64 = 0.04;

/// Two faces that meet at more than this many degrees meet in a crease that is meant; at less than
/// `FLAT` degrees they are one flat face cut in two; in between they are facets of a surface that was
/// meant to be smooth.
pub const CREASE: f64 = 30.0;
pub const FLAT: f64 = 0.25;
/// How far a model's detail is melted to begin with, as a share of the width of its facets (once it is
/// on the stage): enough that glass no longer shows them as steps in its reflections. Never more than
/// `SMOOTH_MOST`: facets wider than that allows are a model's look, not a fault to melt away.
pub const SMOOTH_PER_FACET: f64 = 0.25;
pub const SMOOTH_MOST: f64 = 0.008;
/// With this share of its edges (by length, flat ones aside) creased, a model is one of flat faces and
/// meant edges, and nothing is melted; with fewer, in proportion.
pub const CREASED_NONE: f64 = 0.3;
/// The step the `smooth` slider moves in.
pub const SMOOTH_STEP: f64 = 0.0005;

/// What is done with a model's surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Treatment {
    /// the surface is whole: a solid
    Closed,
    /// every gap is small (its fan a small share of the surface): closed, a solid
    SmallGaps,
    /// there are large openings: closed with flat fans, a solid, but a shell may suit it better
    LargeOpenings,
    /// the surface encloses nothing to speak of: a shell with a thickness
    Shell,
}

/// One gap in the surface: a closed run of edges that have a triangle on one side only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gap {
    pub edges: usize,
    /// its largest extent as a share of the model's
    pub size: f64,
    /// the area of the fan that closes it, in the file's units
    pub area: f64,
}

/// What examining a model found.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// as read
    pub vertices_in: usize,
    pub triangles_in: usize,
    /// after merging corners that coincide and dropping triangles that are none
    pub vertices: usize,
    pub triangles: usize,
    /// triangles dropped: with two corners the same, and exact copies of another
    pub degenerate: usize,
    pub duplicates: usize,
    /// separate pieces (triangles joined through shared edges)
    pub pieces: usize,
    /// every edge is shared by exactly two triangles that run along it in opposite directions, as read
    pub closed: bool,
    /// edges with one triangle only, and with three or more, as read
    pub open_edges: usize,
    pub crowded_edges: usize,
    /// pieces whose triangles were wound inconsistently and were put right; pieces that cannot be (one-sided)
    pub rewound: usize,
    pub one_sided: usize,
    /// closed pieces that were inside out and were turned; pieces kept inside out because they are hollows
    pub turned: usize,
    pub hollows: usize,
    /// the gaps, largest first
    pub gaps: Vec<Gap>,
    /// corners of the box around the model, in the file's units
    pub bounds: [[f32; 3]; 2],
    /// area of the surface, of the fans that close its gaps, and the volume enclosed once closed
    pub area: f64,
    pub fans: f64,
    pub volume: f64,
    /// Of the edges where two faces meet at an angle: the share (by length) that are creases; the width
    /// of the facets either side of the others, in the stage's units; and how far the model's detail
    /// is melted to begin with, which follows from the two (see `SMOOTH_PER_FACET`).
    pub creased: f64,
    pub facet: f64,
    pub smooth: f64,
    pub treatment: Treatment,
    /// the result in one line
    pub line: String,
}

/// What the GPU side is asked to build for a model: the file, and what the model's own controls say.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshJob {
    /// the file: a full path, or one below the program's folder
    pub file: String,
    /// the file's up axis is z (otherwise y)
    pub up_z: bool,
    /// 0 = solid or shell as the examiner found it, 1 = solid, 2 = shell
    pub mode: u8,
    /// a shell's wall thickness
    pub wall: f64,
    /// its size as a share of the size that fills the stage
    pub size: f64,
    /// how far fine detail is melted; below 0: as far as the model's own facets call for (`Report::smooth`)
    pub smooth: f64,
    /// how far the surface is moved outward
    pub inflate: f64,
    /// voxels a side of the shape's volume at full detail (the model is prepared once, at this size)
    pub full_res: u32,
}

/// A model examined: the report, the surface as it is (merged, wound one way), and the surface with
/// every gap closed. The closed one begins with the same triangles as the open one, in the same order.
#[derive(Clone, Debug)]
pub struct Examined {
    pub report: Report,
    pub surface: Mesh,
    pub closed: Mesh,
}

// ---- a quick hash for whole numbers (the standard one is slow for millions of small keys)
#[derive(Default, Clone, Copy)]
struct Fx(u64);
impl Fx {
    fn mix(&mut self, v: u64) { self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x517c_c1b7_2722_0a95); }
}
impl Hasher for Fx {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) { for b in bytes { self.mix(*b as u64); } }
    fn write_u32(&mut self, v: u32) { self.mix(v as u64); }
    fn write_u64(&mut self, v: u64) { self.mix(v); }
    fn write_i64(&mut self, v: i64) { self.mix(v as u64); }
    fn write_usize(&mut self, v: usize) { self.mix(v as u64); }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn len(a: [f64; 3]) -> f64 { dot(a, a).sqrt() }
fn f64s(p: [f32; 3]) -> [f64; 3] { [p[0] as f64, p[1] as f64, p[2] as f64] }

impl Mesh {
    /// The corners of the box around every position.
    pub fn bounds(&self) -> [[f32; 3]; 2] {
        let mut b = [[f32::MAX; 3], [f32::MIN; 3]];
        for p in &self.pos {
            for a in 0..3 {
                b[0][a] = b[0][a].min(p[a]);
                b[1][a] = b[1][a].max(p[a]);
            }
        }
        b
    }
    fn corners(&self, t: usize) -> [[f64; 3]; 3] {
        let v = self.tri[t];
        [f64s(self.pos[v[0] as usize]), f64s(self.pos[v[1] as usize]), f64s(self.pos[v[2] as usize])]
    }
    /// Area of the triangles from `from` on.
    fn area(&self, from: usize) -> f64 {
        (from..self.tri.len()).into_par_iter().map(|t| { let c = self.corners(t); 0.5 * len(cross(sub(c[1], c[0]), sub(c[2], c[0]))) }).sum()
    }
}

/// Every triangle's three edges, sorted so that the triangles along one edge sit together. Each entry:
/// the edge as (lower corner, higher corner) packed into one number, then the triangle and which of its
/// edges it is (triangle * 4 + edge).
fn edge_list(tri: &[[u32; 3]]) -> Vec<(u64, u32)> {
    let mut e: Vec<(u64, u32)> = tri
        .par_iter()
        .enumerate()
        .flat_map_iter(|(t, v)| {
            (0..3).map(move |k| {
                let (a, b) = (v[k], v[(k + 1) % 3]);
                (((a.min(b) as u64) << 32) | a.max(b) as u64, (t * 4 + k) as u32)
            })
        })
        .collect();
    e.par_sort_unstable();
    e
}
/// Does the triangle in this entry run along its edge from the lower corner to the higher?
fn forward(tri: &[[u32; 3]], entry: u32) -> bool {
    let (t, k) = ((entry / 4) as usize, (entry % 4) as usize);
    tri[t][k] < tri[t][(k + 1) % 3]
}
/// Call `f` with each run of entries that share an edge.
fn for_each_edge(e: &[(u64, u32)], mut f: impl FnMut(&[(u64, u32)])) {
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j].0 == e[i].0 { j += 1; }
        f(&e[i..j]);
        i = j;
    }
}

/// Sets of things that have been joined: `find` gives a set's name.
struct Sets(Vec<u32>);
impl Sets {
    fn new(n: usize) -> Self { Sets((0..n as u32).collect()) }
    fn find(&mut self, mut a: u32) -> u32 {
        while self.0[a as usize] != a {
            self.0[a as usize] = self.0[self.0[a as usize] as usize];
            a = self.0[a as usize];
        }
        a
    }
    fn join(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b { self.0[a.max(b) as usize] = a.min(b); }
    }
}

/// Merge corners that coincide, drop triangles with two corners the same and exact copies of another
/// triangle. Returns the mesh, and how many triangles of each kind were dropped.
fn weld(m: &Mesh) -> (Mesh, usize, usize) {
    let b = m.bounds();
    let size = len(sub(f64s(b[1]), f64s(b[0])));
    // corners are merged by the cell of a fine lattice they fall in (a millionth of the model's size)
    let cell = (size * WELD).max(1e-30);
    let mut seen: FxMap<[i64; 3], u32> = FxMap::default();
    let mut out = Mesh::default();
    let mut to = vec![u32::MAX; m.pos.len()];
    let mut place = |i: u32, out: &mut Mesh| -> u32 {
        if to[i as usize] != u32::MAX { return to[i as usize]; }
        let p = m.pos[i as usize];
        let key = [0, 1, 2].map(|a| ((p[a] as f64 - b[0][a] as f64) / cell).round() as i64);
        let n = *seen.entry(key).or_insert_with(|| { out.pos.push(p); out.pos.len() as u32 - 1 });
        to[i as usize] = n;
        n
    };
    let (mut none, mut copies) = (0, 0);
    // which way round each set of three corners has already been seen: 1 = as sorted, 2 = the other way
    let mut had: FxMap<[u32; 3], u8> = FxMap::default();
    for t in &m.tri {
        let v = [place(t[0], &mut out), place(t[1], &mut out), place(t[2], &mut out)];
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            none += 1;
            continue;
        }
        let mut s = v;
        s.sort_unstable();
        // turned so that its lowest corner comes first, a triangle reads (lowest, middle, highest) one way round
        let k = (0..3).find(|&k| v[k] == s[0]).unwrap_or(0);
        let way = if v[(k + 1) % 3] == s[1] { 1 } else { 2 };
        let h = had.entry(s).or_insert(0);
        if *h & way != 0 {
            copies += 1;
            continue;
        }
        *h |= way;
        out.tri.push(v);
    }
    (out, none, copies)
}

/// Wind every piece one way: across each edge that two triangles share, the two must run along it in
/// opposite directions. Returns how many pieces had triangles turned over, and how many cannot be put
/// right at all (a one-sided surface, like a Möbius band).
fn rewind(m: &mut Mesh, edges: &[(u64, u32)]) -> (usize, usize) {
    let n = m.tri.len();
    // each triangle's neighbour across each of its edges, where exactly two triangles share the edge:
    // neighbour * 2 + (1 if the two agree as they stand)
    let mut next = vec![[u32::MAX; 3]; n];
    for_each_edge(edges, |run| {
        if run.len() != 2 { return; }
        let (a, b) = (run[0].1, run[1].1);
        let agree = (forward(&m.tri, a) != forward(&m.tri, b)) as u32;
        next[(a / 4) as usize][(a % 4) as usize] = (b / 4) * 2 + agree;
        next[(b / 4) as usize][(b % 4) as usize] = (a / 4) * 2 + agree;
    });
    // 0 = not reached, 1 = stays, 2 = turned over
    let mut state = vec![0u8; n];
    let (mut rewound, mut one_sided) = (0, 0);
    let mut stack = Vec::new();
    let mut members = Vec::new();
    for start in 0..n {
        if state[start] != 0 { continue; }
        state[start] = 1;
        stack.push(start as u32);
        members.clear();
        let mut clash = false;
        while let Some(t) = stack.pop() {
            members.push(t);
            for k in 0..3 {
                let e = next[t as usize][k];
                if e == u32::MAX { continue; }
                let (o, agree) = ((e / 2) as usize, e % 2 == 1);
                let want = if agree { state[t as usize] } else { 3 - state[t as usize] };
                if state[o] == 0 {
                    state[o] = want;
                    stack.push(o as u32);
                } else if state[o] != want {
                    clash = true;
                }
            }
        }
        let turned = members.iter().filter(|t| state[**t as usize] == 2).count();
        if clash { one_sided += 1; }
        if turned == 0 { continue; }
        if !clash { rewound += 1; }
        // turn over whichever are fewer
        let flip = if turned * 2 > members.len() { 1 } else { 2 };
        for t in &members {
            if state[*t as usize] == flip { m.tri[*t as usize].swap(1, 2); }
        }
    }
    (rewound, one_sided)
}

/// The gaps of a surface wound one way, each as a closed run of corners in the direction the surface's
/// own edges run along it. Wherever more triangles run along an edge one way than the other, the edge
/// is part of a gap, once for each triangle too many.
fn gaps(m: &Mesh, edges: &[(u64, u32)]) -> Vec<Vec<u32>> {
    let mut open: Vec<(u32, u32)> = Vec::new();
    for_each_edge(edges, |run| {
        let fwd = run.iter().filter(|e| forward(&m.tri, e.1)).count() as i64;
        let net = 2 * fwd - run.len() as i64;
        let (lo, hi) = ((run[0].0 >> 32) as u32, run[0].0 as u32);
        for _ in 0..net.abs() {
            open.push(if net > 0 { (lo, hi) } else { (hi, lo) });
        }
    });
    if open.is_empty() { return Vec::new(); }
    open.sort_unstable();
    // where each corner's outgoing edges begin in the sorted list
    let mut first: FxMap<u32, usize> = FxMap::default();
    for (i, e) in open.iter().enumerate() { first.entry(e.0).or_insert(i); }
    let mut used = vec![false; open.len()];
    let mut cursor: FxMap<u32, usize> = first.clone();
    let mut take = |from: u32, used: &mut Vec<bool>| -> Option<u32> {
        let c = cursor.get_mut(&from)?;
        while *c < open.len() && open[*c].0 == from {
            if !used[*c] {
                used[*c] = true;
                return Some(open[*c].1);
            }
            *c += 1;
        }
        None
    };
    let mut loops = Vec::new();
    for s in 0..open.len() {
        if used[s] { continue; }
        // walk from corner to corner; each time the walk comes back to a corner it has been at, the
        // stretch since then is one closed run
        let mut path = vec![open[s].0];
        let mut at: FxMap<u32, usize> = FxMap::default();
        at.insert(open[s].0, 0);
        while let Some(&cur) = path.last() {
            let Some(nxt) = take(cur, &mut used) else { break };
            if let Some(&i) = at.get(&nxt) {
                loops.push(path[i..].to_vec());
                for v in &path[i + 1..] { at.remove(v); }
                path.truncate(i + 1);
            } else {
                at.insert(nxt, path.len());
                path.push(nxt);
            }
        }
    }
    loops
}

/// Is `p` inside the closed set of triangles `tris` of `m`? Counted along a line straight up from it.
fn inside_piece(m: &Mesh, tris: &[u32], p: [f64; 3]) -> bool {
    let mut n = 0;
    for &t in tris {
        let c = m.corners(t as usize);
        let e = |a: [f64; 3], b: [f64; 3]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let (w0, w1, w2) = (e(c[1], c[2]), e(c[2], c[0]), e(c[0], c[1]));
        let all = w0 + w1 + w2;
        if all == 0.0 || !((w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0)) { continue; }
        let z = (w0 * c[0][2] + w1 * c[1][2] + w2 * c[2][2]) / all;
        if z > p[2] { n += 1; }
    }
    n % 2 == 1
}

/// Turn every closed piece the right way out. A piece inside another keeps the sense it has relative to
/// the one around it: wound against it, it is a hollow and stays one. Returns (pieces, turned, hollows,
/// volume enclosed).
fn face_out(closed: &mut Mesh, surface: &mut Mesh) -> (usize, usize, usize, f64) {
    let n = closed.tri.len();
    let edges = edge_list(&closed.tri);
    let mut sets = Sets::new(n);
    for_each_edge(&edges, |run| {
        for e in &run[1..] { sets.join(run[0].1 / 4, e.1 / 4); }
    });
    let b = closed.bounds();
    let mid = [0, 1, 2].map(|a| 0.5 * (b[0][a] as f64 + b[1][a] as f64));
    struct Piece { tris: Vec<u32>, volume: f64, lo: [f64; 3], hi: [f64; 3] }
    let mut name: FxMap<u32, usize> = FxMap::default();
    let mut pieces: Vec<Piece> = Vec::new();
    for t in 0..n {
        let s = sets.find(t as u32);
        let i = *name.entry(s).or_insert_with(|| { pieces.push(Piece { tris: Vec::new(), volume: 0.0, lo: [f64::MAX; 3], hi: [f64::MIN; 3] }); pieces.len() - 1 });
        let c = closed.corners(t);
        let p = &mut pieces[i];
        p.tris.push(t as u32);
        p.volume += dot(sub(c[0], mid), cross(sub(c[1], mid), sub(c[2], mid))) / 6.0;
        for q in c {
            for a in 0..3 {
                p.lo[a] = p.lo[a].min(q[a]);
                p.hi[a] = p.hi[a].max(q[a]);
            }
        }
    }
    // The pieces from the largest down. A piece is inside another if its box is inside the other's and
    // some of its own corners are inside the other's surface. (Only so much work is spent on this: a
    // model of thousands of pieces is taken as pieces side by side.)
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_by(|a, b| pieces[*b].volume.abs().total_cmp(&pieces[*a].volume.abs()));
    let mut keep_sense = vec![1.0f64; pieces.len()];     // +1: wanted the right way out; -1: wanted inside out
    let mut budget: i64 = 200_000_000;
    let (mut turned, mut hollows) = (0, 0);
    let mut flip = vec![false; pieces.len()];
    for (rank, &i) in order.iter().enumerate() {
        let p = &pieces[i];
        // the smallest piece around this one, among the larger ones
        let mut parent = None;
        for &j in order[..rank].iter().rev() {
            let q = &pieces[j];
            if !(0..3).all(|a| q.lo[a] <= p.lo[a] && q.hi[a] >= p.hi[a]) { continue; }
            budget -= 3 * q.tris.len() as i64;
            if budget < 0 { break; }
            let probes = [p.tris[0], p.tris[p.tris.len() / 2], p.tris[p.tris.len() - 1]];
            if probes.iter().all(|t| inside_piece(closed, &q.tris, closed.corners(*t as usize)[0])) {
                parent = Some(j);
                break;
            }
        }
        let was = if p.volume < 0.0 { -1.0 } else { 1.0 };
        let want = match parent {
            // the same sense relative to the piece around it as the file has
            Some(j) => keep_sense[j] * was * if pieces[j].volume < 0.0 { -1.0 } else { 1.0 },
            None => 1.0,
        };
        keep_sense[i] = want;
        if want < 0.0 { hollows += 1; }
        if want != was && p.volume != 0.0 {
            flip[i] = true;
            if parent.is_none() { turned += 1; }
        }
    }
    let mut volume = 0.0;
    for (i, p) in pieces.iter().enumerate() {
        volume += if flip[i] { -p.volume } else { p.volume };
        if !flip[i] { continue; }
        for &t in &p.tris {
            closed.tri[t as usize].swap(1, 2);
            if (t as usize) < surface.tri.len() { surface.tri[t as usize].swap(1, 2); }
        }
    }
    (pieces.len(), turned, hollows, volume)
}

/// What a surface's edges say about how it was meant: the share of them (by length, leaving out those
/// between faces that lie flat together) that are creases, and the width of its facets, in the file's
/// units. A ball in facets has no creases and facets of some width; a box has nothing but creases; a
/// gear has flat faces, creases, and a bore in facets.
///
/// A facet's width is taken across each of its edges that is neither flat nor a crease (the mean of
/// the heights of the two triangles above the edge), and a triangle counts with the widest of its
/// three: that is the spacing of the steps it makes in a reflection. The widths are then averaged by
/// the triangles' areas: the broad facets are the ones that show, however many small ones there are.
fn facets(m: &Mesh, edges: &[(u64, u32)]) -> (f64, f64) {
    let (mut creased, mut faceted) = (0.0, 0.0);
    let mut widest = vec![0.0f64; m.tri.len()];
    for_each_edge(edges, |run| {
        if run.len() != 2 { return; }
        let (lo, hi) = ((run[0].0 >> 32) as usize, run[0].0 as u32 as usize);
        let long = len(sub(f64s(m.pos[hi]), f64s(m.pos[lo])));
        let normal = |entry: u32| { let c = m.corners((entry / 4) as usize); cross(sub(c[1], c[0]), sub(c[2], c[0])) };
        let (a, b) = (normal(run[0].1), normal(run[1].1));
        if long == 0.0 || len(a) == 0.0 || len(b) == 0.0 { return; }
        let angle = len(cross(a, b)).atan2(dot(a, b)).to_degrees();
        if angle > CREASE {
            creased += long;
        } else if angle >= FLAT {
            faceted += long;
            // (a triangle's height above an edge is twice its area over the edge's length)
            let across = 0.5 * (len(a) + len(b)) / long;
            for e in run {
                let t = (e.1 / 4) as usize;
                widest[t] = widest[t].max(across);
            }
        }
    });
    let (mut width, mut weight) = (0.0, 0.0);
    for (t, w) in widest.iter().enumerate() {
        if *w == 0.0 { continue; }
        let c = m.corners(t);
        let area = 0.5 * len(cross(sub(c[1], c[0]), sub(c[2], c[0])));
        width += area * w;
        weight += area;
    }
    let all = creased + faceted;
    if all == 0.0 { (1.0, 0.0) } else { (creased / all, if weight > 0.0 { width / weight } else { 0.0 }) }
}

/// How far a model's detail is melted to begin with: by a share of its facets' width (`facet`, in the
/// stage's units), less as more of its edges are creases, and nothing once enough are. In whole steps
/// of the slider.
pub fn smooth_for(creased: f64, facet: f64) -> f64 {
    let s = (SMOOTH_PER_FACET * facet).min(SMOOTH_MOST) * (1.0 - creased / CREASED_NONE).clamp(0.0, 1.0);
    (s / SMOOTH_STEP).round() * SMOOTH_STEP
}

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 { format!("1 {one}") } else { format!("{n} {many}") }
}

/// Examine a mesh as read from a file: clean it, wind it, close its gaps, and say what was found.
pub fn examine(read: &Mesh) -> Result<Examined, String> {
    let (mut surface, degenerate, duplicates) = weld(read);
    if surface.tri.is_empty() { return Err("the model has no triangles with any size".into()); }
    let bounds = surface.bounds();
    let size = len(sub(f64s(bounds[1]), f64s(bounds[0])));
    if !(size > 0.0 && size.is_finite()) { return Err("the model has no size".into()); }

    // as read: is every edge shared by exactly two triangles running opposite ways?
    let edges = edge_list(&surface.tri);
    let (mut open_edges, mut crowded_edges, mut closed_as_read) = (0, 0, true);
    for_each_edge(&edges, |run| {
        match run.len() {
            1 => open_edges += 1,
            2 => {}
            _ => crowded_edges += 1,
        }
        if run.len() != 2 || forward(&surface.tri, run[0].1) == forward(&surface.tri, run[1].1) { closed_as_read = false; }
    });
    let (rewound, one_sided) = rewind(&mut surface, &edges);
    // the gaps that are left once every piece is wound one way, each closed by a cap about its middle
    let edges = edge_list(&surface.tri);
    // what the edges say of how it was meant, with the facets' width as it will be on the stage (where
    // the model's farthest corner is `FIT` from its middle)
    let (creased, facet) = facets(&surface, &edges);
    let middle = [0, 1, 2].map(|a| 0.5 * (bounds[0][a] as f64 + bounds[1][a] as f64));
    let far = surface.pos.par_iter().map(|p| len(sub(f64s(*p), middle))).reduce(|| 0.0, f64::max);
    let facet = facet * crate::FIT / far.max(1e-30);
    let loops = gaps(&surface, &edges);
    let mut closed = surface.clone();
    let mut found: Vec<Gap> = Vec::new();
    for l in &loops {
        if l.len() < 3 { continue; }
        let mut c = [0.0f64; 3];
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for v in l {
            let p = f64s(closed.pos[*v as usize]);
            for a in 0..3 {
                c[a] += p[a] / l.len() as f64;
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let k = l.len();
        let first = closed.tri.len();
        // The gap's own plane, as nearly as it has one: the direction its edge turns about its middle.
        let mut n = [0.0f64; 3];
        for i in 0..k {
            let x = cross(sub(f64s(closed.pos[l[i] as usize]), c), sub(f64s(closed.pos[l[(i + 1) % k] as usize]), c));
            n = [n[0] + x[0], n[1] + x[1], n[2] + x[2]];
        }
        let nl = len(n);
        // A gap whose edge is not flat (a hole in a scan) is closed flat all the same, but for a rim:
        // between the edge and the middle go a few rings, each a smaller copy of the edge that has come
        // towards the gap's plane faster than towards its middle. (A flat edge gives a flat cap either way.)
        let rings = if k >= 6 && nl > 0.0 { 3 } else { 0 };
        let mut outer: Vec<u32> = l.clone();
        for r in 1..=rings {
            let t = 1.0 - r as f64 / (rings + 1) as f64;
            let inner: Vec<u32> = l
                .iter()
                .map(|v| {
                    let d = sub(f64s(closed.pos[*v as usize]), c);
                    let h = dot(d, n) / nl;
                    let q = [0, 1, 2].map(|a| c[a] + t * (d[a] - h * n[a] / nl) + t * t * h * n[a] / nl);
                    closed.pos.push([q[0] as f32, q[1] as f32, q[2] as f32]);
                    closed.pos.len() as u32 - 1
                })
                .collect();
            for i in 0..k {
                let j = (i + 1) % k;
                // against the direction the surface runs along the gap, so that the cap carries the surface on
                closed.tri.push([outer[j], outer[i], inner[i]]);
                closed.tri.push([outer[j], inner[i], inner[j]]);
            }
            outer = inner;
        }
        let at = closed.pos.len() as u32;
        closed.pos.push([c[0] as f32, c[1] as f32, c[2] as f32]);
        for i in 0..k {
            closed.tri.push([outer[(i + 1) % k], outer[i], at]);
        }
        found.push(Gap { edges: k, size: len(sub(hi, lo)) / size, area: closed.area(first) });
    }
    found.sort_by(|a, b| b.size.total_cmp(&a.size));
    let (pieces, turned, hollows, volume) = face_out(&mut closed, &mut surface);
    let area = surface.area(0);
    let fans: f64 = found.iter().map(|g| g.area).sum();

    let treatment = if found.is_empty() {
        Treatment::Closed
    } else if fans > OPEN_SHARE * (area + fans) || volume.abs() < THIN * size * fans {
        Treatment::Shell
    } else if found.iter().all(|g| g.area <= SMALL_GAP * (area + fans)) {
        Treatment::SmallGaps
    } else {
        Treatment::LargeOpenings
    };
    let mut line = match treatment {
        Treatment::Closed => "closed".to_string(),
        Treatment::SmallGaps => format!("{}, closed automatically", plural(found.len(), "small gap", "small gaps")),
        Treatment::LargeOpenings => {
            let big = found.iter().filter(|g| g.area > SMALL_GAP * (area + fans)).count();
            let rest = found.len() - big;
            let small = if rest > 0 { format!(" and {}", plural(rest, "small gap", "small gaps")) } else { String::new() };
            format!("{}{small} (the largest {:.0}% of the model's size), closed by a flat guess: shell may suit it better", plural(big, "large opening", "large openings"), 100.0 * found[0].size)
        }
        Treatment::Shell => "open surface: made a shell".to_string(),
    };
    if rewound > 0 { line += &format!("; {} wound inconsistently, put right", plural(rewound, "piece", "pieces")); }
    if turned > 0 { line += &format!("; {} inside out, turned", plural(turned, "piece", "pieces")); }
    if one_sided > 0 { line += &format!("; {} one-sided", plural(one_sided, "piece is", "pieces are")); }
    if crowded_edges > 0 { line += &format!("; {} shared by more than two faces", plural(crowded_edges, "edge", "edges")); }
    let report = Report {
        vertices_in: read.pos.len(), triangles_in: read.tri.len(), vertices: surface.pos.len(), triangles: surface.tri.len(),
        degenerate, duplicates, pieces, closed: closed_as_read, open_edges, crowded_edges, rewound, one_sided, turned, hollows,
        gaps: found, bounds, area, fans, volume, creased, facet, smooth: smooth_for(creased, facet), treatment, line,
    };
    Ok(Examined { report, surface, closed })
}

impl Report {
    /// The report as a piece of JSON (for scripts).
    pub fn json(&self) -> String {
        let gaps: Vec<String> = self.gaps.iter().take(40).map(|g| format!("{{\"edges\":{},\"size\":{:.4},\"area\":{:.6}}}", g.edges, g.size, g.area)).collect();
        let b = self.bounds;
        format!(
            "{{\"verticesIn\":{},\"trianglesIn\":{},\"vertices\":{},\"triangles\":{},\"degenerate\":{},\"duplicates\":{},\"pieces\":{},\"closed\":{},\"openEdges\":{},\"crowdedEdges\":{},\"rewound\":{},\"oneSided\":{},\"turned\":{},\"hollows\":{},\"gapCount\":{},\"gaps\":[{}],\"bounds\":[[{},{},{}],[{},{},{}]],\"area\":{:.6},\"fans\":{:.6},\"volume\":{:.6},\"creased\":{:.4},\"facet\":{:.5},\"smooth\":{:.4},\"treatment\":\"{}\",\"line\":\"{}\"}}",
            self.vertices_in, self.triangles_in, self.vertices, self.triangles, self.degenerate, self.duplicates, self.pieces, self.closed, self.open_edges, self.crowded_edges,
            self.rewound, self.one_sided, self.turned, self.hollows, self.gaps.len(), gaps.join(","), b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2], self.area, self.fans, self.volume, self.creased, self.facet, self.smooth,
            match self.treatment { Treatment::Closed => "closed", Treatment::SmallGaps => "small gaps", Treatment::LargeOpenings => "large openings", Treatment::Shell => "shell" },
            self.line.replace('"', "'"),
        )
    }
}

/// Read a model file and examine it; with `res`, also prepare it for a volume of that many voxels a
/// side (as a solid unless the examiner made it a shell, or `shell` says which). The findings and how
/// long each part took, as JSON: what `glassforms --model-info` prints.
pub fn describe(path: &std::path::Path, up_z: bool, res: Option<u32>, shell: Option<bool>) -> Result<String, String> {
    let t = std::time::Instant::now();
    let mesh = read::read_file(path)?;
    let read_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let file = path.file_name().map(|n| n.to_string_lossy().replace('"', "'")).unwrap_or_default();
    let ex = examine(&mesh).map_err(|e| format!("{file}: {e}"))?;
    let examine_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut out = format!("{{\"file\":\"{file}\",\"readMs\":{read_ms:.1},\"examineMs\":{examine_ms:.1},\"report\":{}", ex.report.json());
    if let Some(res) = res {
        let shell = shell.unwrap_or(ex.report.treatment == Treatment::Shell);
        let t = std::time::Instant::now();
        let p = solid::prepare(&ex, up_z, shell, res).map_err(|e| format!("{file}: {e}"))?;
        out += &format!(
            ",\"prepare\":{{\"res\":{res},\"shell\":{shell},\"ms\":{:.1},\"parts\":[{:.1},{:.1},{:.1},{:.1},{:.1}],\"triangles\":{},\"hidden\":{},\"unclosed\":{},\"flips\":{},\"midRes\":{}}}",
            t.elapsed().as_secs_f64() * 1e3, p.ms[0], p.ms[1], p.ms[2], p.ms[3], p.ms[4], p.tris.len(), p.hidden, p.unclosed, p.flips.len(), p.mid_res
        );
    }
    Ok(out + "}")
}
