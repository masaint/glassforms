//! Geometric solids, in glass: the regular and semi-regular polyhedra and their edge frames, star
//! polyhedra, prisms and antiprisms, superellipsoids, rings and pierced blocks.
//!
//! A handful of generators, each with sliders, rather than a formula per shape:
//!
//!   kaleidoscope   Wythoff's construction. Three mirrors through the centre fold space into one chamber;
//!                  a point placed in the chamber, reflected in the mirrors, gives every corner of a
//!                  polyhedron. Where the point stands decides which: the five Platonic solids, their
//!                  truncations and everything between. Drawn as the solid (rounded, hollowed) or as its
//!                  edges in tubes.
//!   stellation     the face planes of a regular solid, extended: the points lying beyond at most k of them.
//!   hull           the convex hull of two regular polygons, one turned: prisms, antiprisms and between.
//!   superquadric   |x|^p and |y|^q in place of squares: from a diamond through a ball to a block.
//!   ring           a torus whose cross-section is a superellipse; and several of them linked in a row.
//!   pierced        a block with a round bore along each axis and a hollow middle.
//!
//! Every shape is scaled inside the shader to reach about FIT from the centre whatever its sliders say,
//! so none is ever cut by the edge of the volume; that is why they ask for no clipping (`clip` = -1).
//!
//! No edge is left perfectly sharp: each is rounded to a radius of two voxels (`fine()`), exactly where the
//! generator can and smoothly where it cannot. The walk that turns an equation into distances takes its
//! direction from the slope of the function, and along the plane that halves the angle at a sharp edge
//! the slope belongs to neither face; voxels there come out with wrong distances (by more than a voxel
//! at an acute edge), which shows as saw-toothed edges. With the edge rounded, the function is smooth
//! for two voxels around the surface and the distances there are right. Two voxels is about what the
//! smooth lookup of the volume rounds an edge to anyway.
use super::*;

/// How far from the centre the farthest point of each shape lies.
pub(crate) const FIT: f64 = 1.22;
/// No clipping: these shapes are their own bound, and a clip would only round their corners.
const NO_CLIP: &str = "-1.0";
// The least rounding any edge gets: two voxels of the volume being built.
pub(crate) const FINE: &str = "fn fine() -> f32 { return 4.0 * u.hb / u.res; }\n";

// ---------------------------------------------------------------------------------------- kaleidoscope

// The three mirrors, the fold into the chamber between them, and the generating point.
pub(crate) const KALEIDOSCOPE: &str = "
// The kaleidoscope: three mirrors through the centre, the planes x = 0, y = 0 and the one with normal nc,
// meeting at angles pi/2, pi/3 and pi/n. For n = 3, 4, 5 their reflections are the symmetries of the
// tetrahedron, of the cube and octahedron, and of the dodecahedron and icosahedron.
// c = cos(pi/n). s: how far each of the chamber's corner vectors (see kal_vertex) stands from the mirror opposite it.
struct Kal { nc: vec3f, c: f32, s: f32 }
fn kal(n: f32) -> Kal {
  let c = cos(3.14159265 / clamp(round(n), 3.0, 5.0));
  let s = sqrt(0.75 - c * c);
  return Kal(vec3f(-0.5, -c, s), c, s);
}
// Reflect p in the mirrors until it lies in the chamber between them. Five rounds of three reflections
// are as many as the largest of the three groups ever needs.
fn kal_fold(p0: vec3f, nc: vec3f) -> vec3f {
  var p = p0;
  for (var i = 0; i < 5; i++) {
    p = vec3f(abs(p.xy), p.z);
    p -= 2.0 * min(0.0, dot(p, nc)) * nc;
  }
  return p;
}
// The generating point, on the sphere of radius r: weights on the chamber's three corners, which lie on
// the 2-fold, the n-fold and the 3-fold axes. The corner vectors are sized so that each stands equally
// far from the mirror opposite it; equal weights then put the point equally far from those mirrors,
// which is what makes all the edges of the polyhedron equally long.
fn kal_vertex(k: Kal, weights: vec3f, r: f32) -> vec3f {
  var w = max(weights, vec3f(0.0));
  if (w.x + w.y + w.z < 1e-4) { w = vec3f(1.0); }       // no weight at all would be no point at all
  return r * normalize(w.x * vec3f(0.0, 0.0, 1.0) + w.y * vec3f(k.s, 0.0, 0.5) + w.z * vec3f(0.0, k.s, k.c));
}
// How far a point (given from the generating point) is from each of the three edges that leave the
// generating point. Each runs straight at a mirror, where its own reflection carries it on.
fn kal_edges(w: vec3f, nc: vec3f) -> vec3f {
  return vec3f(length(w + max(-w.x, 0.0) * vec3f(1.0, 0.0, 0.0)),
               length(w + max(-w.y, 0.0) * vec3f(0.0, 1.0, 0.0)),
               length(w + max(-dot(w, nc), 0.0) * nc));
}
";

// The polyhedron as a solid. Within the chamber it is bounded by just three planes, the faces that meet
// at the generating point, each square to one of the chamber's corners; so its distance there is the
// distance to the corner those three planes make, which can be had exactly.
const SOLID: &str = "
// Signed distance to the corner made by the three faces that meet at the generating point, for a point
// given from it. nb, nq: unit normals of the faces across the n-fold and the 3-fold axis (the third is z).
// Inside, it is the largest of the three plane distances. Outside, the nearest point is on a face, on
// one of the three edges, or the corner itself.
fn kal_corner(w: vec3f, nc: vec3f, c: f32, nb: vec3f, nq: vec3f) -> f32 {
  let fa = w.z;
  let fb = dot(w, nb);
  let fc = dot(w, nq);
  let inside = max(fa, max(fb, fc));
  if (inside <= 0.0) { return inside; }
  let e = kal_edges(w, nc);
  var d = min(e.x, min(e.y, e.z));
  // a face counts where the foot of the perpendicular falls between the two edges that bound it
  let p1 = -w.x;
  let p2 = -w.y;
  let p3 = -dot(w, nc);
  if (fa > 0.0 && p1 >= 0.0 && p2 >= 0.0) { d = min(d, fa); }
  if (fb > 0.0 && p2 + c * p3 >= 0.0 && p3 + c * p2 >= 0.0) { d = min(d, fb); }
  if (fc > 0.0 && p1 + 0.5 * p3 >= 0.0 && p3 + 0.5 * p1 >= 0.0) { d = min(d, fc); }
  return d;
}
// A corner moved towards the centre may cross a mirror (when a face is smaller than the rounding asked
// for): hold it in the chamber.
fn kal_hold(v0: vec3f, nc: vec3f) -> vec3f {
  var v = v0;
  for (var i = 0; i < 3; i++) {
    v = vec3f(max(v.xy, vec2f(0.0)), v.z);
    v -= min(0.0, dot(v, nc)) * nc;
  }
  return vec3f(max(v.xy, vec2f(0.0)), v.z);
}
fn raw(p: vec3f) -> f32 {
  let k = kal(pr.n);
  let q = kal_fold(p, k.nc);
  let lb = sqrt(1.0 - k.c * k.c);
  let nb = vec3f(k.s, 0.0, 0.5) / lb;
  let nq = vec3f(0.0, k.s, k.c) / 0.8660254;
  let v = kal_vertex(k, vec3f(pr.a2, pr.an, pr.a3), CIRCUM);
  // moving all three faces in by one unit moves the corner they share by this much
  let inward = vec3f((lb - 0.5) / k.s, (0.8660254 - k.c) / k.s, 1.0);
  // Rounding: every face is moved in by the rounding and the result swollen again by as much, which
  // leaves the faces where they were and rounds edges and corners to that radius.
  let rd = max(pr.round, fine());
  var d = kal_corner(q - kal_hold(v - rd * inward, k.nc), k.nc, k.c, nb, nq) - rd;
  // Hollow: the same solid again, its faces further in by the wall, taken out of the middle; as long as
  // there is room for it (the nearest face of the hollow must still be short of the centre).
  // (a wall is never thinner than six voxels: thinner than that the volume cannot hold it)
  let wall = max(pr.wall, 3.0 * fine());
  let room = min(v.z, min(dot(v, nb), dot(v, nq))) - wall;
  if (pr.wall > 0.0005 && room > fine()) {
    let rc = min(max(rd - wall, fine()), 0.5 * room);
    let hollow = kal_corner(q - kal_hold(v - (wall + rc) * inward, k.nc), k.nc, k.c, nb, nq) - rc;
    d = max(d, -hollow);
  }
  return d;
}";

// The same polyhedron as its edges alone, in tubes, with a ball at every corner.
const FRAME: &str = "
fn raw(p: vec3f) -> f32 {
  let k = kal(pr.n);
  let q = kal_fold(p, k.nc);
  let v = kal_vertex(k, vec3f(pr.a2, pr.an, pr.a3), CIRCUM - max(pr.tube, pr.ball));
  let w = q - v;
  let e = kal_edges(w, k.nc);
  return smin(min(e.x, min(e.y, e.z)) - pr.tube, length(w) - pr.ball, 0.03);
}";

const KAL_EQ: [&str; 3] = [
    "three mirrors through the centre, at angles π/2, π/3, π/n",
    "v = a·P₂ + b·Pₙ + c·P₃  (the axes where the mirrors cross)",
    "faces: the planes through v square to P₂, Pₙ, P₃, reflected",
];

fn kal_weights(n: f64, w: [f64; 3]) -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("n", "symmetry n", n, 3.0, 5.0, 1.0),
        ParamDef::labelled("a2", "a (2-fold)", w[0], 0.0, 1.0, 0.01),
        ParamDef::labelled("an", "b (n-fold)", w[1], 0.0, 1.0, 0.01),
        ParamDef::labelled("a3", "c (3-fold)", w[2], 0.0, 1.0, 0.01),
    ]
}

/// A polyhedron of the kaleidoscope, as a solid: symmetry n, where the generating point stands, the
/// slider that travels when the animation plays.
fn wythoff(id: &'static str, name: &'static str, n: f64, w: [f64; 3], animate: &'static str, rot: [f64; 3], note: &'static str) -> Preset {
    let mut params = kal_weights(n, w);
    params.push(ParamDef::labelled("round", "rounding", 0.0, 0.0, 0.3, 0.005));
    params.push(ParamDef::labelled("wall", "hollow: wall", 0.0, 0.0, 0.3, 0.005));
    Preset {
        eq: KAL_EQ.to_vec(),
        note,
        params,
        rot: Some(rot),
        animate: Some(vec![animate]),
        ..Preset::new(id, name, Shape::Field {
            wgsl: format!("const CIRCUM: f32 = {FIT:.4};\n{FINE}{KALEIDOSCOPE}{SOLID}"),
            clip: NO_CLIP.to_string(),
            shell: "0.0".to_string(),
        })
    }
}

/// The same as its edges in tubes.
fn frame(id: &'static str, name: &'static str, n: f64, w: [f64; 3], tube: f64, ball: f64, animate: &'static str, rot: [f64; 3], note: &'static str) -> Preset {
    let mut params = kal_weights(n, w);
    params.push(ParamDef::labelled("tube", "tube", tube, 0.02, 0.16, 0.002));
    params.push(ParamDef::labelled("ball", "corner balls", ball, 0.0, 0.3, 0.005));
    Preset {
        eq: vec![KAL_EQ[0], KAL_EQ[1], "edges: from v straight to each mirror, reflected"],
        note,
        params,
        rot: Some(rot),
        animate: Some(vec![animate]),
        ..Preset::new(id, name, Shape::Field {
            wgsl: format!("const CIRCUM: f32 = {FIT:.4};\n{KALEIDOSCOPE}{FRAME}"),
            clip: NO_CLIP.to_string(),
            shell: "0.0".to_string(),
        })
    }
}

// ------------------------------------------------------------------------------------------ stellation

/// A regular solid whose face planes are extended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Core {
    Octahedron,
    Dodecahedron,
    Icosahedron,
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

impl Core {
    /// The faces come in opposite pairs: one unit normal of each pair.
    pub fn normals(self) -> Vec<[f64; 3]> {
        let f = (1.0 + 5f64.sqrt()) / 2.0;
        let g = 1.0 / f;
        let cube = [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]];
        let v: Vec<[f64; 3]> = match self {
            Core::Octahedron => cube.to_vec(),
            Core::Dodecahedron => vec![[0.0, 1.0, f], [0.0, -1.0, f], [1.0, f, 0.0], [-1.0, f, 0.0], [f, 0.0, 1.0], [f, 0.0, -1.0]],
            Core::Icosahedron => cube.into_iter().chain([[0.0, g, f], [0.0, -g, f], [g, f, 0.0], [-g, f, 0.0], [f, 0.0, g], [f, 0.0, -g]]).collect(),
        };
        v.into_iter().map(unit).collect()
    }

    /// The highest order that is still a bounded figure: beyond it the cells run off to infinity.
    pub fn orders(self) -> usize {
        match self {
            Core::Octahedron => 1,
            Core::Dodecahedron => 3,
            Core::Icosahedron => 7,
        }
    }

    /// How far the figure of order `order` (whole or not) reaches from the centre when the face planes
    /// are one unit away.
    ///
    /// Along a direction u the planes are met at distances 1 / |n.u|, so the figure of whole order k
    /// ends where the (k+1)-th nearest plane is met, and a blend of two orders where the blend of their
    /// two rules is met: at 1 / ((1 - t) a_k + t a_k+1), a_j being the (j+1)-th largest |n.u|. That is
    /// largest where the blend of a's is smallest, which can only be where great circles of the kind
    /// n.u = 0 or |n_i.u| = |n_j.u| cross; so those crossings are all that need trying.
    pub fn reach(self, order: f64) -> f64 {
        let normals = self.normals();
        let o = order.clamp(0.0, self.orders() as f64);
        let k = (o.floor() as usize).min(self.orders() - 1);
        let t = o - k as f64;
        // the planes through the centre on which two of the |n.u| tie or one vanishes
        let mut ties: Vec<[f64; 3]> = normals.clone();
        for i in 0..normals.len() {
            for j in i + 1..normals.len() {
                let (a, b) = (normals[i], normals[j]);
                ties.push([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
                ties.push([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
            }
        }
        let mut least = f64::MAX;
        for i in 0..ties.len() {
            for j in i + 1..ties.len() {
                let (a, b) = (ties[i], ties[j]);
                let u = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
                let l = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                if l < 1e-9 { continue; }
                let mut along: Vec<f64> = normals.iter().map(|n| ((n[0] * u[0] + n[1] * u[1] + n[2] * u[2]) / l).abs()).collect();
                along.sort_by(|x, y| y.partial_cmp(x).unwrap());
                least = least.min((1.0 - t) * along[k] + t * along[k + 1]);
            }
        }
        1.0 / least
    }
}

/// Steps to each whole order in the table of sizes the shader reads between.
pub const ORDER_STEPS: usize = 16;

// The figures are turned out of line with the volume's axes. Along the plane that halves the angle at an
// edge, the walk that measures distances cannot tell which of the two faces is nearer, and where such
// planes run exactly through the centres of voxels (as the planes x = y and the like do) every voxel on
// them is affected, which shows as saw-toothed edges. Turned, only a scattering of voxels is.
pub(crate) const TURN: [[f64; 3]; 3] = [
    [0.8044187403, -0.2849119348, 0.5212827253],
    [0.4437075685, 0.8716242528, -0.2083140795],
    [-0.3950114985, 0.3988688399, 0.8275684652],
];

// Where two planes meet in an edge: the exact distance, which is what lets the edge be rounded exactly.
const WEDGE: &str = "
// Signed distance to the wedge where two planes both read zero or less. a, b: distances from the two
// planes; c: the dot product of their unit normals.
fn wedge(a: f32, b: f32, c0: f32) -> f32 {
  if (a <= 0.0 && b <= 0.0) { return max(a, b); }
  let c = clamp(c0, -0.999, 0.999);
  if (a > 0.0 && b - a * c <= 0.0) { return a; }           // nearest is a point of the first face
  if (b > 0.0 && a - b * c <= 0.0) { return b; }           // or of the second
  return sqrt(max(a * a + b * b - 2.0 * a * b * c, 0.0) / (1.0 - c * c));      // or the edge itself
}
";

// The stellation's rule picks one plane out of the sorted list, and where that plane changes places with a
// neighbour in the list the figure has an edge or a crease.
const EASED: &str = "
// The rule 'the middle one of three neighbours in the sorted list' (hi >= mid >= lo), with its edges
// eased to radius r. Where mid ties with hi the glass has a crease running inwards, which is filled to
// that radius; where it ties with lo the glass has an outward edge, which is rounded to it.
// c_hi, c_lo: dot products of mid's normal with the normals of hi and lo.
fn eased(hi: f32, mid: f32, lo: f32, c_hi: f32, c_lo: f32, r: f32) -> f32 {
  let filled = r - wedge(r - hi, r - mid, c_hi);
  return wedge(filled + r, lo + r, c_lo) - r;
}
";

/// The points lying beyond at most `order` of the core's face planes.
///
/// The shader is written out plane by plane, with nothing kept in lists: it is run some thirty times for
/// every voxel, and lists indexed as it goes are several times slower on a GPU than named values.
fn stellation(id: &'static str, name: &'static str, core: Core, order: f64, rot: [f64; 3], eq: Vec<&'static str>, note: &'static str) -> Preset {
    use std::fmt::Write;
    let top = core.orders();
    let normals: Vec<[f64; 3]> = core.normals().iter().map(|n| {
        let r = |row: &[f64; 3]| row[0] * n[0] + row[1] * n[1] + row[2] * n[2];
        [r(&TURN[0]), r(&TURN[1]), r(&TURN[2])]
    }).collect();
    let m = normals.len();
    // how far the core's faces stand from the centre at each step of the order, so that the figure always reaches FIT
    let n_sizes = top * ORDER_STEPS + 1;
    let sizes: Vec<String> = (0..n_sizes).map(|i| format!("{:.7}", FIT / core.reach(i as f64 / ORDER_STEPS as f64))).collect();

    let mut w = format!(
        "{FINE}{WEDGE}{EASED}
// What depends on the sliders alone is worked out once for each voxel and kept: how far the core's faces
// stand from the centre, which whole order lies below the order asked for, and how far past it that is.
var<private> st_ready: bool = false;
var<private> st_h: f32;
var<private> st_k: i32;
var<private> st_t: f32;
fn st_setup() {{
  // the core's size at every {step}th of an order, chosen so that the figure always just fits
  var sizes = array<f32, {n_sizes}>({sizes});
  let o = clamp(pr.order, 0.0, {top}.0);
  let at = o * {step}.0;
  let cell = min(i32(floor(at)), {n_sizes} - 2);
  st_h = mix(sizes[cell], sizes[cell + 1], at - f32(cell));
  st_k = min(i32(floor(o)), {top} - 1);
  st_t = o - f32(st_k);
  st_ready = true;
}}
fn raw(p: vec3f) -> f32 {{
  if (!st_ready) {{ st_setup(); }}
  // The faces come in opposite pairs. For each pair: how far along its normal the point is, and its
  // signed distance from the face of the pair on its own side.
",
        step = ORDER_STEPS, sizes = sizes.join(", "),
    );
    for (i, n) in normals.iter().enumerate() {
        writeln!(w, "  let g{i} = dot(p, vec3f({:.9}, {:.9}, {:.9}));\n  let d{i} = abs(g{i}) - st_h;", n[0], n[1], n[2]).unwrap();
    }
    // the distances in order, the farthest beyond first: neighbours compared and exchanged until sorted
    w += "  // the same distances in order, the farthest beyond first\n";
    for i in 0..m {
        writeln!(w, "  var s{i} = d{i};").unwrap();
    }
    w += "  var x: f32;\n";
    for pass in 0..m - 1 {
        for j in 0..m - 1 - pass {
            writeln!(w, "  x = max(s{j}, s{k}); s{k} = min(s{j}, s{k}); s{j} = x;", k = j + 1).unwrap();
        }
    }
    // Beyond at most k planes: the (k+1)-th in order is not beyond. The rule for k, eased, needs its
    // neighbours either side; the blend with the rule for k + 1 needs one more.
    w += "  // Beyond at most k planes: the (k+1)-th of them is not beyond. Between whole orders the rules for k
  // and k + 1 are blended. Four neighbours in the order are needed (a >= b >= c >= e), and whose they are.
  var a = 1e3;
  var b = 0.0;
  var c = 0.0;
  var e = 0.0;
  switch (st_k) {\n";
    for k in 0..top {
        let above = if k > 0 { format!("a = s{}; ", k - 1) } else { String::new() };
        let case = if k + 1 < top { format!("case {k}") } else { "default".to_string() };
        writeln!(w, "    {case}: {{ {above}b = s{k}; c = s{}; e = s{}; }}", k + 1, k + 2).unwrap();
    }
    w += "  }
  var pa = vec3f(0.0);
  var pb = vec3f(0.0);
  var pc = vec3f(0.0);
  var pe = vec3f(0.0);
  var nn: vec3f;\n";
    for (i, n) in normals.iter().enumerate() {
        writeln!(
            w,
            "  nn = sign(g{i}) * vec3f({:.9}, {:.9}, {:.9});\n  pa = select(pa, nn, d{i} == a); pb = select(pb, nn, d{i} == b); pc = select(pc, nn, d{i} == c); pe = select(pe, nn, d{i} == e);",
            n[0], n[1], n[2]
        ).unwrap();
    }
    w += "  let r = fine();
  let cbc = dot(pb, pc);
  return mix(eased(a, b, c, dot(pa, pb), cbc, r), eased(b, c, e, cbc, dot(pc, pe), r), st_t);
}";
    Preset {
        eq,
        note,
        params: vec![ParamDef::labelled("order", "order k", order, 0.0, top as f64, 0.01)],
        rot: Some(rot),
        animate: Some(vec!["order"]),
        ..Preset::new(id, name, Shape::Field { wgsl: w, clip: NO_CLIP.to_string(), shell: "0.0".to_string() })
    }
}

// ------------------------------------------------------------------------------------------------ hull

// The convex hull of two regular n-gons, one above the other and turned against it. Each side of the
// lower polygon makes a triangle with one corner of the upper, each side of the upper with one corner
// of the lower; with no turn, the two triangles lie in one plane and make the rectangle of a prism.
const HULL: &str = "
// What depends on the sliders alone is worked out once for each voxel and kept: the half-height, the
// angle from one side to the next, and the planes of three neighbouring sides (normal and distance from
// the centre): the side between the first two corners of the lower polygon, the one before it and the
// one after. Each side is two triangles, so two planes.
var<private> hl_ready: bool = false;
var<private> hl_step: f32;
var<private> hl_h: f32;
var<private> hl_n: array<vec3f, 6>;
var<private> hl_d: vec2f;
// the first corner is set this far off the x axis, so that with eight sides no edge lies exactly in a
// plane x = z, where the walk that measures distances would leave it saw-toothed
const HL_OFF: f32 = 0.2;
fn hl_turned(v: vec3f, a: f32) -> vec3f { return vec3f(cos(a) * v.x - sin(a) * v.z, v.y, sin(a) * v.x + cos(a) * v.z); }
fn hl_setup() {
  let n = clamp(round(pr.sides), 3.0, 12.0);
  let step = 6.2831853 / n;
  let turn = clamp(pr.twist, 0.0, 1.0) * 0.5 * step;
  // with corners at distance 1 from the axis: the side of the base, and the half-height at which the
  // upright edges are as long as it
  let side = 2.0 * sin(0.5 * step);
  let even = 0.5 * sqrt(max(side * side - 4.0 * sin(0.5 * turn) * sin(0.5 * turn), 1e-6));
  let tall = pr.height * even;
  let r = FIT / sqrt(1.0 + tall * tall);
  let h = tall * r;
  // one side of the lower polygon with the corner of the upper that stands over it, and one side of the
  // upper with the corner of the lower beneath it: every other side face is one of these two, turned
  // about the axis by a whole number of steps
  let b0 = vec3f(r * cos(HL_OFF), -h, r * sin(HL_OFF));
  let b1 = vec3f(r * cos(HL_OFF + step), -h, r * sin(HL_OFF + step));
  let t0 = vec3f(r * cos(HL_OFF + turn), h, r * sin(HL_OFF + turn));
  let t1 = vec3f(r * cos(HL_OFF + step + turn), h, r * sin(HL_OFF + step + turn));
  var n0 = normalize(cross(b1 - b0, t0 - b0));
  n0 *= sign(dot(n0, b0 + b1 + t0));
  var n1 = normalize(cross(t1 - t0, b1 - t0));
  n1 *= sign(dot(n1, t0 + t1 + b1));
  hl_step = step;
  hl_h = h;
  hl_n[0] = hl_turned(n0, -step);
  hl_n[1] = hl_turned(n1, -step);
  hl_n[2] = n0;
  hl_n[3] = n1;
  hl_n[4] = hl_turned(n0, step);
  hl_n[5] = hl_turned(n1, step);
  hl_d = vec2f(dot(b0, n0), dot(t0, n1));
  hl_ready = true;
}
// one more plane: keep the two the point is farthest beyond
fn hl_keep(d: f32, n: vec3f, da: ptr<function, f32>, na: ptr<function, vec3f>, db: ptr<function, f32>, nb: ptr<function, vec3f>) {
  if (d > *da) { *db = *da; *nb = *na; *da = d; *na = n; } else if (d > *db) { *db = d; *nb = n; }
}
fn raw(p: vec3f) -> f32 {
  if (!hl_ready) { hl_setup(); }
  // the point turned back about the axis by a whole number of steps, to where the side it faces is the first
  let q = hl_turned(p, -floor((atan2(p.z, p.x) - HL_OFF) / hl_step) * hl_step);
  // The two planes the point is farthest beyond: of the two ends, and of that side and its two neighbours.
  var da = q.y - hl_h;
  var na = vec3f(0.0, 1.0, 0.0);
  var db = -q.y - hl_h;
  var nb = vec3f(0.0, -1.0, 0.0);
  if (db > da) { let x = da; da = db; db = x; let y = na; na = nb; nb = y; }
  hl_keep(dot(q, hl_n[0]) - hl_d.x, hl_n[0], &da, &na, &db, &nb);
  hl_keep(dot(q, hl_n[1]) - hl_d.y, hl_n[1], &da, &na, &db, &nb);
  hl_keep(dot(q, hl_n[2]) - hl_d.x, hl_n[2], &da, &na, &db, &nb);
  hl_keep(dot(q, hl_n[3]) - hl_d.y, hl_n[3], &da, &na, &db, &nb);
  hl_keep(dot(q, hl_n[4]) - hl_d.x, hl_n[4], &da, &na, &db, &nb);
  hl_keep(dot(q, hl_n[5]) - hl_d.y, hl_n[5], &da, &na, &db, &nb);
  // Inside, the distance is the larger of the two. The edge the two planes make is rounded to the least
  // radius, exactly.
  let ease = fine();
  return wedge(da + ease, db + ease, dot(na, nb)) - ease;
}";

fn hull(id: &'static str, name: &'static str, sides: f64, twist: f64, height: f64, rot: [f64; 3], note: &'static str) -> Preset {
    Preset {
        eq: vec!["the convex hull of two regular n-gons at y = ±h,", "the upper one turned by τ · 180°/n"],
        note,
        params: vec![
            ParamDef::labelled("sides", "sides n", sides, 3.0, 12.0, 1.0),
            ParamDef::labelled("twist", "turn τ", twist, 0.0, 1.0, 0.01),
            ParamDef::new("height", height, 0.2, 3.0, 0.02),
        ],
        rot: Some(rot),
        animate: Some(vec!["twist"]),
        ..Preset::new(id, name, Shape::Field { wgsl: format!("const FIT: f32 = {FIT:.4};\n{FINE}{WEDGE}{HULL}"), clip: NO_CLIP.to_string(), shell: "0.0".to_string() })
    }
}

// ---------------------------------------------------------------------------- superquadric, ring, pierced

// |u|^e + |v|^e = 1 is a diamond for e = 1, a circle for e = 2 and tends to a square as e grows. Its
// farthest point is a corner, 2^(1/2 - 1/e) from the centre, once e passes 2.
const SUPER: &str = "
fn sup2(u: vec2f, e: f32) -> f32 { let a = abs(u) + 1e-6; return pow(pow(a.x, e) + pow(a.y, e), 1.0 / e); }
fn reach2(e: f32) -> f32 { return max(1.0, pow(2.0, 0.5 - 1.0 / e)); }
";

const SUPERQUADRIC: &str = "
fn raw(p: vec3f) -> f32 {
  let ax = vec3f(1.0, pr.tall, pr.deep);
  let s = FIT / (reach2(pr.p) * reach2(pr.q) * max(1.0, max(pr.tall, pr.deep)));
  let u = p / (ax * s);
  // the outline round the axis, then the outline from pole to pole
  let f = sup2(vec2f(sup2(u.xz, pr.p), u.y), pr.q) - 1.0;
  return f * s * min(1.0, min(pr.tall, pr.deep));
}";

const RING: &str = "
// a ring of radius big about the y axis whose cross-section is a superellipse, `small` wide and `small * flat` high
fn ring(p: vec3f, big: f32, small: f32, flat: f32, e: f32) -> f32 {
  let u = vec2f(length(p.xz) - big, p.y) / vec2f(small, small * flat);
  return (sup2(u, e) - 1.0) * small * min(1.0, flat);
}
";

const ONE_RING: &str = "
fn raw(p: vec3f) -> f32 {
  // the farthest point is a corner of the cross-section on the outside of the ring
  let big = FIT / length(vec2f(1.0 + pr.fat, pr.fat * pr.flat));
  return ring(p, big, pr.fat * big, pr.flat, pr.section);
}";

const LINKS: &str = "
fn raw(p: vec3f) -> f32 {
  let n = clamp(round(pr.rings), 2.0, 5.0);
  let reach = pr.fat * reach2(pr.section);
  // a ring passes through its neighbour without touching it while their centres are more than one tube's
  // width apart and less than a ring's width less a tube's
  let gap = clamp(pr.spacing, 2.0 * reach + 0.06, 2.0 - 2.0 * reach - 0.06);
  let big = FIT / (0.5 * (n - 1.0) * gap + 1.0 + reach);
  var d = 1e9;
  for (var i = 0; i < 5; i++) {
    if (f32(i) >= n) { break; }
    var q = p - vec3f((f32(i) - 0.5 * (n - 1.0)) * gap * big, 0.0, 0.0);
    if (i % 2 == 1) { q = q.xzy; }                       // every other ring stands at right angles to its neighbours
    d = min(d, ring(q, big, pr.fat * big, 1.0, pr.section));
  }
  return d;
}";

const PIERCED: &str = "
fn sup3(u: vec3f, e: f32) -> f32 { let a = abs(u) + 1e-6; return pow(pow(a.x, e) + pow(a.y, e) + pow(a.z, e), 1.0 / e); }
fn raw(p: vec3f) -> f32 {
  // the block: |x|^e + |y|^e + |z|^e = s^e, sized so that its corners reach FIT
  let s = FIT / max(1.0, pow(3.0, 0.5 - 1.0 / pr.body));
  let block = sup3(p, pr.body) - s;
  // a bore along each axis, and a hollow in the middle shaped like the block
  let bore = min(sup2(p.xy, pr.bore), min(sup2(p.yz, pr.bore), sup2(p.zx, pr.bore))) - pr.hole * s;
  let hollow = sup3(p, pr.body) - pr.cavity * s;
  let k = max(pr.round, 1.5 * fine());
  return smax(smax(block, -bore, k), -hollow, k);
}";

fn light(az: f64, el: f64, w: f64, h: f64, rgb: [f64; 3], soft: f64) -> Light {
    Light { az, el, w, h, rgb, soft }
}

pub fn family() -> Family {
    let tail = |body: &str| Shape::Field { wgsl: format!("const FIT: f32 = {FIT:.4};\n{FINE}{SUPER}{body}"), clip: NO_CLIP.to_string(), shell: "0.0".to_string() };
    Family {
        // The star polyhedra's distances are only bounds deep inside (the rule is not a distance there), so
        // rays step a little short of them; frames, hollow solids and stars need rays that leave the glass
        // followed into the next piece of it.
        step_scale: 0.85,
        steps_out: 220,
        steps_in: 64,
        layers: 2,
        presets: vec![
            wythoff("tetrahedron", "Tetrahedron", 3.0, [0.0, 1.0, 0.0], "a2", [20.0, 30.0, 0.0],
                "Four triangles: the simplest solid there is. Raise a and its corners are cut off; at a = 1, b = 0 it has become an octahedron."),
            wythoff("cube", "Cube", 4.0, [0.0, 0.0, 1.0], "a2", [24.0, 32.0, 0.0],
                "Six squares. The point v stands on a 3-fold axis; move it to the 4-fold axis (b = 1, c = 0) and the same mirrors make the octahedron."),
            wythoff("octahedron", "Octahedron", 4.0, [0.0, 1.0, 0.0], "a2", [20.0, 28.0, 0.0],
                "Eight triangles, four at each corner: the cube’s dual."),
            wythoff("dodecahedron", "Dodecahedron", 5.0, [0.0, 0.0, 1.0], "a2", [18.0, 22.0, 0.0],
                "Twelve pentagons. Raise a and each corner becomes a small triangle."),
            wythoff("icosahedron", "Icosahedron", 5.0, [0.0, 1.0, 0.0], "a2", [18.0, 22.0, 0.0],
                "Twenty triangles, five at each corner. Raise a to 1 and every corner is cut to a pentagon: the football."),
            wythoff("cuboctahedron", "Cuboctahedron", 4.0, [1.0, 0.0, 0.0], "a3", [24.0, 32.0, 0.0],
                "Half way between cube and octahedron: v stands on a 2-fold axis. Raise b or c for the truncated octahedron or cube; all three at 1 gives the great rhombicuboctahedron."),
            wythoff("football", "Truncated icosahedron", 5.0, [1.0, 1.0, 0.0], "a2", [18.0, 22.0, 0.0],
                "Twelve pentagons and twenty hexagons, every edge the same length: a football, and the carbon cage C₆₀. Take a to 0 and c to 1 for the rhombicosidodecahedron, with its thirty squares."),
            frame("frame", "Edge frame", 5.0, [0.0, 0.0, 1.0], 0.06, 0.11, "a2", [18.0, 22.0, 0.0],
                "The edges alone. Starts as the dodecahedron’s thirty; the sliders reach every polyhedron the solids do."),
            stellation("stellaoctangula", "Stella octangula", Core::Octahedron, 1.0, [20.0, 30.0, 0.0],
                vec!["nᵢ·x = h : the eight face planes of an octahedron", "Sₖ = { x : at most k of the planes lie between x and the centre }"],
                "Kepler’s eight-pointed star: two tetrahedra through each other. Order 0 is the octahedron itself; half way, the pyramids on its faces flatten into the rhombic dodecahedron."),
            stellation("stardodecahedron", "Stellated dodecahedron", Core::Dodecahedron, 1.0, [18.0, 22.0, 0.0],
                vec!["nᵢ·x = h : the twelve face planes of a dodecahedron", "Sₖ = { x : at most k of the planes lie between x and the centre }"],
                "k = 1, 2, 3 are the star polyhedra of Kepler and Poinsot: the small stellated, the great, and the great stellated dodecahedron. Between whole numbers neighbouring figures are blended."),
            stellation("staricosahedron", "Stellated icosahedron", Core::Icosahedron, 2.0, [18.0, 22.0, 0.0],
                vec!["nᵢ·x = h : the twenty face planes of an icosahedron", "Sₖ = { x : at most k of the planes lie between x and the centre }"],
                "Of the icosahedron’s 59 stellations, the eight that take whole layers: k = 2 is the compound of five octahedra, k = 6 the great icosahedron, k = 7 the last: a ball of spikes."),
            hull("prism", "Prism", 6.0, 0.0, 1.6, [22.0, 24.0, 0.0],
                "τ = 0 is a prism, τ = 1 an antiprism. At height 1 the upright edges are as long as the base’s: squares on a prism, equilateral triangles on an antiprism."),
            hull("antiprism", "Antiprism", 5.0, 1.0, 1.0, [22.0, 24.0, 0.0],
                "A band of triangles between two pentagons: the middle of an icosahedron. With three sides it is the octahedron."),
            Preset {
                eq: vec!["( |x|ᵖ + |z/c|ᵖ )^(q/p) + |y/b|^q = 1"],
                note: "Exponent 1 gives a diamond, 2 a circle, large ones a square. p shapes the outline seen from above, q the outline seen from the side.",
                params: vec![
                    ParamDef::new("p", 2.5, 0.8, 8.0, 0.05),
                    ParamDef::new("q", 2.5, 0.8, 8.0, 0.05),
                    ParamDef::labelled("tall", "b", 1.0, 0.3, 1.8, 0.01),
                    ParamDef::labelled("deep", "c", 1.0, 0.3, 1.8, 0.01),
                ],
                rot: Some([24.0, 32.0, 0.0]),
                animate: Some(vec!["q", "p"]),
                ..Preset::new("superellipsoid", "Superellipsoid", tail(SUPERQUADRIC))
            },
            Preset {
                eq: vec!["( |ρ − R|ᵉ + |y/s|ᵉ )^(1/e) = r,   ρ = √(x² + z²)"],
                note: "A torus whose cross-section need not be round: e = 2 is the ordinary ring, large e a square band, e = 1 a diamond.",
                params: vec![
                    ParamDef::labelled("fat", "r / R", 0.42, 0.12, 0.9, 0.01),
                    ParamDef::labelled("section", "e", 2.0, 1.0, 10.0, 0.05),
                    ParamDef::labelled("flat", "s", 1.0, 0.3, 2.5, 0.01),
                ],
                rot: Some([55.0, 0.0, 20.0]),
                animate: Some(vec!["fat"]),
                ..Preset::new("ring", "Ring", tail(&format!("{RING}{ONE_RING}")))
            },
            Preset {
                eq: vec!["tori in a row, alternately lying and standing,", "each passing through the next"],
                note: "Two rings linked once cannot be parted without cutting one. With more rings it is a chain; e squares the section off.",
                params: vec![
                    ParamDef::new("rings", 2.0, 2.0, 5.0, 1.0),
                    ParamDef::labelled("fat", "r / R", 0.28, 0.1, 0.36, 0.005),
                    ParamDef::new("spacing", 1.0, 0.5, 1.5, 0.01),
                    ParamDef::labelled("section", "e", 2.0, 1.0, 10.0, 0.05),
                ],
                rot: Some([40.0, 20.0, 15.0]),
                animate: Some(vec!["spacing"]),
                ..Preset::new("links", "Linked rings", tail(&format!("{RING}{LINKS}")))
            },
            Preset {
                eq: vec!["|x|ᵉ + |y|ᵉ + |z|ᵉ ≤ 1,  less the bores", "x² + y² < a²,  y² + z² < a²,  z² + x² < a²"],
                note: "A block bored through along all three axes. Widen the bores and only the twelve edges are left.",
                params: vec![
                    ParamDef::labelled("body", "e", 8.0, 2.0, 12.0, 0.1),
                    ParamDef::labelled("hole", "a", 0.5, 0.1, 0.8, 0.01),
                    ParamDef::labelled("bore", "bore shape", 2.0, 2.0, 10.0, 0.1),
                    ParamDef::labelled("cavity", "hollow", 0.0, 0.0, 0.95, 0.01),
                    ParamDef::labelled("round", "rounding", 0.04, 0.0, 0.2, 0.005),
                ],
                rot: Some([24.0, 32.0, 0.0]),
                animate: Some(vec!["hole"]),
                ..Preset::new("pierced", "Pierced block", tail(PIERCED))
            },
        ],
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.20, 0.09, 0.05])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(4.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([22.0, 30.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        // A flat face mirrors one direction, so under a few small lights each face is either lit or black.
        // These are broad instead and fade from their centres all the way to their edges: a face then takes
        // its tone from where in a light it looks, and neighbouring faces differ. Two small bright ones
        // give the glints.
        lights: vec![
            light(40.0, 20.0, 30.0, 42.0, [3.0, 2.9, 2.7], 1.0),
            light(-60.0, -5.0, 26.0, 46.0, [0.4, 1.1, 3.3], 1.0),
            light(-140.0, 5.0, 22.0, 46.0, [3.9, 1.3, 0.22], 1.0),
            light(150.0, 30.0, 32.0, 24.0, [2.1, 2.05, 2.0], 1.0),
            light(0.0, 85.0, 40.0, 40.0, [0.8, 0.9, 1.1], 1.0),
            light(0.0, -80.0, 36.0, 36.0, [0.5, 0.22, 0.05], 1.0),
            light(95.0, 46.0, 5.0, 5.0, [7.0, 7.0, 7.0], 0.5),
            light(-100.0, -36.0, 4.0, 8.0, [5.0, 5.5, 6.5], 0.5),
        ],
        ..Family::new("solids", "Geometric solids", Kind::Field)
    }
}
