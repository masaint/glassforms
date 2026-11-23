//! Polyhedra beyond the regular and semi-regular ones of the geometric solids: Johnson solids, compounds,
//! zonohedra and geodesic spheres. The same machinery as solids.rs, carried further.
//!
//! Every convex solid here is a list of face planes, worked out on the CPU (the convex hull of its corners,
//! or the faces of a zonohedron straight from its star of vectors) and written into the shader one plane
//! a line. A point keeps the three planes it lies farthest beyond, and its distance is measured exactly to
//! the corner those three make (`corner3`): to a face, an edge or the corner itself, whichever is nearest.
//! Inside, that is the largest plane distance, which is exact for a convex solid; outside it is exact
//! wherever three faces meet at a corner, and never more than the truth elsewhere (the three planes bound
//! a larger region than the solid). So every edge can be rounded exactly, as the kaleidoscope's are: the
//! faces are moved in by the rounding and the result swollen again by as much. No edge is left sharper
//! than two voxels (`fine()`, see solids.rs).
//!
//!   johnson      convex solids with regular faces that are neither Platonic, Archimedean, prisms nor antiprisms
//!   compound     several solids through one another, each exact, joined with the creases between them filled
//!   zonohedron   the Minkowski sum of a star of segments: every face a parallelogram (or a centred polygon)
//!   geodesic     an icosahedron's faces divided into triangles, the corners pushed out to a sphere
//!
//! The polyhedra are turned out of line with the volume's axes (TURN, solids.rs): a face or edge plane on
//! the grid's diagonals leaves saw-toothed edges in the bake.
use super::*;
use super::solids::{FINE, FIT, KALEIDOSCOPE, TURN};

type V = [f64; 3];
use crate::math::{add, cross, dot, len, scale as mul, sub, unit};
/// a vector turned by TURN
fn turn(v: V) -> V { [dot(TURN[0], v), dot(TURN[1], v), dot(TURN[2], v)] }
const PHI: f64 = 1.618_033_988_749_895;

/// A 3x3 matrix as rows.
type M = [[f64; 3]; 3];
fn mat_t(m: M) -> M { [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]] }
fn mat_mul(a: M, b: M) -> M {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum(); } }
    r
}
/// the turn by angle `a` about the unit axis `u`
fn axis_turn(u: V, a: f64) -> M {
    let (s, c) = a.sin_cos();
    let t = 1.0 - c;
    [
        [c + u[0] * u[0] * t, u[0] * u[1] * t - u[2] * s, u[0] * u[2] * t + u[1] * s],
        [u[1] * u[0] * t + u[2] * s, c + u[1] * u[1] * t, u[1] * u[2] * t - u[0] * s],
        [u[2] * u[0] * t - u[1] * s, u[2] * u[1] * t + u[0] * s, c + u[2] * u[2] * t],
    ]
}
fn apply(m: M, v: V) -> V { [dot(m[0], v), dot(m[1], v), dot(m[2], v)] }
/// WGSL for a matrix given as rows (WGSL's constructor takes columns)
fn wgsl_mat(m: M) -> String {
    format!("mat3x3f({:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9})",
        m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2])
}
fn wgsl_v(v: V) -> String { format!("vec3f({:.9}, {:.9}, {:.9})", v[0], v[1], v[2]) }

// ------------------------------------------------------------------------------------------ the planes

/// The face planes of the convex hull of `pts` (unit outward normal, distance from the centre).
pub fn hull(pts: &[V]) -> Vec<(V, f64)> {
    let mut out: Vec<(V, f64)> = Vec::new();
    let scale = pts.iter().map(|p| len(*p)).fold(0.0, f64::max);
    let eps = 1e-7 * scale.max(1.0);
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            for k in j + 1..pts.len() {
                let n = cross(sub(pts[j], pts[i]), sub(pts[k], pts[i]));
                if len(n) < 1e-9 { continue; }
                let n = unit(n);
                let h = dot(n, pts[i]);
                let (mut above, mut below) = (false, false);
                for p in pts {
                    let s = dot(n, *p) - h;
                    if s > eps { above = true; }
                    if s < -eps { below = true; }
                }
                let (n, h) = match (above, below) {
                    (false, _) => (n, h),
                    (true, false) => (mul(n, -1.0), -h),
                    _ => continue,
                };
                if !out.iter().any(|(m, _)| len(sub(*m, n)) < 1e-6) { out.push((n, h)); }
            }
        }
    }
    out
}

/// Corners centred on the middle of their bounding box and sized so that the farthest is `r` out.
fn centred(pts: &[V], r: f64) -> Vec<V> {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in pts { for a in 0..3 { lo[a] = lo[a].min(p[a]); hi[a] = hi[a].max(p[a]); } }
    let c = mul(add(lo, hi), 0.5);
    let moved: Vec<V> = pts.iter().map(|p| sub(*p, c)).collect();
    let far = moved.iter().map(|p| len(*p)).fold(0.0, f64::max);
    moved.iter().map(|p| mul(*p, r / far)).collect()
}

/// `fn <name>(q: vec3f, s1: f32) -> Top`: every plane tried in a loop over a table of them (written out a
/// line a plane, the formula went into the shader dozens of times over and took seconds to compile). A
/// plane is (normal, distance, and how far the distance moves for each unit of `s1`).
fn planes_fn(name: &str, planes: &[(V, f64, f64)], prune_sectors: bool) -> String {
    let n = planes.len();
    let pl: Vec<String> = planes.iter().map(|(v, h, _)| format!("vec4f({:.9}, {:.9}, {:.9}, {h:.9})", v[0], v[1], v[2])).collect();
    let moves = planes.iter().any(|(_, _, k)| k.abs() > 1e-12);
    let (table_k, h) = if moves {
        let ks: Vec<String> = planes.iter().map(|(_, _, k)| format!("{k:.9}")).collect();
        (format!("var<private> {name}_k: array<f32, {n}> = array<f32, {n}>({});\n", ks.join(", ")), format!("pl.w + s1 * {name}_k[i]"))
    } else {
        (String::new(), "pl.w".to_string())
    };
    let (masks, scan) = if prune_sectors && (30..=96).contains(&n) {
        let masks = plane_masks(planes);
        let words: Vec<_> = masks.iter().map(|m| format!("vec3u({}u,{}u,{}u)", m[0], m[1], m[2])).collect();
        let masks = format!("const zono_masks = array<vec3u, 48>({});\n{ZONO_SECTOR}\n", words.join(","));
        let scan = format!("  let mask = zono_masks[zono_sector(q)];
  for (var word = 0u; word < {}u; word++) {{
    var bits = mask[word];
    while (bits != 0u) {{
      let i = 32u * word + firstTrailingBit(bits);
      bits &= bits - 1u;
      let pl = {name}_pl[i];
      let h = {h};
      top_add(&t, dot(q, pl.xyz) - h, pl.xyz, h);
    }}
  }}", n.div_ceil(32));
        (masks, scan)
    } else {
        (String::new(), format!("  for (var i = 0; i < {n} + opaque(); i++) {{
    let pl = {name}_pl[i];
    let h = {h};
    top_add(&t, dot(q, pl.xyz) - h, pl.xyz, h);
  }}"))
    };
    format!("var<private> {name}_pl: array<vec4f, {n}> = array<vec4f, {n}>({pls});
{table_k}{masks}fn {name}(q: vec3f, s1: f32) -> Top {{
  var t = top_new();
{scan}
  return t;
}}
", pls = pl.join(", "))
}

// Three ordered absolute coordinates and their signs select one of 48 cones. Within
// a cone every query point is a nonnegative combination of its three extreme rays.
// If three other faces score above a face on all three rays and have smaller offsets
// at both stretch endpoints, that face cannot enter the three-plane distance there.
fn plane_masks(planes: &[(V, f64, f64)]) -> Vec<[u32; 3]> {
    let orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut masks = Vec::with_capacity(48);
    for signs in 0..8 {
        for axes in orders {
            let mut rays = [[0.0; 3]; 3];
            for k in 0..3 {
                if k > 0 { rays[k] = rays[k - 1]; }
                rays[k][axes[k]] = if signs & (1 << axes[k]) == 0 { 1.0 } else { -1.0 };
            }
            let mut mask = [0u32; 3];
            for (i, &(n, h, slope)) in planes.iter().enumerate() {
                let above = planes.iter().filter(|&&(m, mh, ms)| {
                    let diff = sub(m, n);
                    rays.iter().all(|r| dot(diff, *r) > 1e-7)
                        && mh - 0.7 * ms <= h - 0.7 * slope + 1e-10
                        && mh + 1.5 * ms <= h + 1.5 * slope + 1e-10
                }).take(3).count();
                if above < 3 { mask[i / 32] |= 1 << (i % 32); }
            }
            masks.push(mask);
        }
    }
    masks
}

const ZONO_SECTOR: &str = "
fn zono_sector(q: vec3f) -> u32 {
  let a = abs(q);
  let signs = select(0u, 1u, q.x < 0.0) | select(0u, 2u, q.y < 0.0) | select(0u, 4u, q.z < 0.0);
  var order: u32;
  if (a.x >= a.y) {
    order = select(select(4u, 1u, a.x >= a.z), 0u, a.y >= a.z);
  } else {
    order = select(select(5u, 3u, a.y >= a.z), 2u, a.x >= a.z);
  }
  return signs * 6u + order;
}
";

#[cfg(test)]
mod plane_tests {
    use super::*;

    #[test]
    fn cone_pruning_preserves_three_nearest_planes_and_inradius() {
        let orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
        for id in ["triacontahedron", "enneacontahedron", "truncicosidodeca"] {
            let gens: Vec<_> = star(id).into_iter().map(unit).collect();
            let planes = zono_planes(&gens, 1.0);
            let masks = plane_masks(&planes);
            for signs in 0..8 {
                for (order, axes) in orders.iter().enumerate() {
                    let mask = masks[6 * signs + order];
                    for radii in [[0.0, 0.0, 0.0], [0.001, 0.001, 0.0], [1.9, 1.9, 1.9], [2.0, 0.85, 0.11], [70.0, 2.0, 0.0]] {
                        let mut q = [0.0; 3];
                        for (k, axis) in axes.iter().copied().enumerate() {
                            q[axis] = radii[k] * if signs & (1 << axis) == 0 { 1.0 } else { -1.0 };
                        }
                        for stretch in [-0.7, -0.25, 0.0, 0.31, 1.5] {
                            let bounds = |prune: bool| {
                                let mut values = Vec::new();
                                let mut least = f64::INFINITY;
                                for (i, &(n, h, slope)) in planes.iter().enumerate() {
                                    if !prune || mask[i / 32] & (1 << (i % 32)) != 0 {
                                        let offset = h + stretch * slope;
                                        values.push(dot(q, n) - offset);
                                        least = least.min(offset);
                                    }
                                }
                                values.sort_by(|a, b| b.total_cmp(a));
                                [values[0], values[1], values[2], least]
                            };
                            for (full, fast) in bounds(false).into_iter().zip(bounds(true)) {
                                assert!((full - fast).abs() < 1e-9, "{id}, q={q:?}, stretch={stretch}");
                            }
                        }
                    }
                }
            }
        }
    }
}

// The three planes a point lies farthest beyond, the exact distance to the corner they make, and the
// solid made from them rounded and hollowed.
const POLY: &str = "
// raw: exact inside, never more than the distance outside
// Zero, though the compiler cannot know it (the volume's size is never below 0): a loop counted to
// N + opaque() is left a loop, where one counted to a known N would be written out N times over and
// take seconds to compile.
fn opaque() -> i32 { return i32(min(u.res, 0.0)); }
// The three planes the point lies farthest beyond (values v.x >= v.y >= v.z, normals n1, n2, n3) and the
// least distance of any plane from the centre (the radius of the largest ball inside).
struct Top { v: vec3f, n1: vec3f, n2: vec3f, n3: vec3f, ih: f32 }
fn top_new() -> Top { return Top(vec3f(-1e9), vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), 1e9); }
fn top_add(t: ptr<function, Top>, v: f32, n: vec3f, h: f32) {
  (*t).ih = min((*t).ih, h);
  if (v > (*t).v.x) { (*t).n3 = (*t).n2; (*t).n2 = (*t).n1; (*t).n1 = n; (*t).v = vec3f(v, (*t).v.x, (*t).v.y); }
  else if (v > (*t).v.y) { (*t).n3 = (*t).n2; (*t).n2 = n; (*t).v = vec3f((*t).v.x, v, (*t).v.y); }
  else if (v > (*t).v.z) { (*t).n3 = n; (*t).v.z = v; }
}
// The distance to the edge where planes A and B meet, if its nearest point there lies inside plane C
// (a, b, c: the point's distances from the planes; cab, cac, cbc: dot products of their normals).
fn c3edge(a: f32, b: f32, c: f32, cab: f32, cac: f32, cbc: f32, na: vec3f, nb: vec3f) -> vec4f {
  let s = 1.0 - cab * cab;
  if (s < 1e-5) { return vec4f(0.0, 0.0, 0.0, 1e9); }
  let al = (a - b * cab) / s;
  let be = (b - a * cab) / s;
  if (al < 0.0 || be < 0.0 || c - al * cac - be * cbc > 0.0) { return vec4f(0.0, 0.0, 0.0, 1e9); }
  let d = sqrt(max(a * al + b * be, 0.0));
  return vec4f((al * na + be * nb) / max(d, 1e-12), d);
}
// Signed distance and outward normal. A feasible projection onto a face or edge, with
// nonnegative multipliers, is already the closest point of the convex intersection.
fn corner3(v: vec3f, n1: vec3f, n2: vec3f, n3: vec3f) -> vec4f {
  let a = v.x;
  let b = v.y;
  let c = v.z;
  if (a <= 0.0) { return vec4f(n1, a); }
  let c12 = dot(n1, n2);
  let c13 = dot(n1, n3);
  if (b <= a * c12 && c <= a * c13) { return vec4f(n1, a); }
  let c23 = dot(n2, n3);
  let ab = c3edge(a, b, c, c12, c13, c23, n1, n2);
  if (ab.w < 1e8) { return ab; }
  let ac = c3edge(a, c, b, c13, c12, c23, n1, n3);
  if (ac.w < 1e8) { return ac; }
  let bc = c3edge(b, c, a, c23, c12, c13, n2, n3);
  if (bc.w < 1e8) { return bc; }
  let det = 1.0 + 2.0 * c12 * c13 * c23 - c12 * c12 - c13 * c13 - c23 * c23;
  if (det <= 1e-5) { return vec4f(n1, a); }
  let l = vec3f(
    (1.0 - c23*c23)*a + (c13*c23 - c12)*b + (c12*c23 - c13)*c,
    (c13*c23 - c12)*a + (1.0 - c13*c13)*b + (c12*c13 - c23)*c,
    (c12*c23 - c13)*a + (c12*c13 - c23)*b + (1.0 - c12*c12)*c) / det;
  let d = sqrt(max(dot(v, l), 0.0));
  return vec4f((l.x*n1 + l.y*n2 + l.z*n3) / max(d, 1e-12), d);
}
fn top_gradient(t: Top, r: f32, wall: f32, fe: f32) -> vec4f {
  var outer = corner3(t.v + r, t.n1, t.n2, t.n3);
  outer.w -= r;
  let w = max(wall, 3.0 * fe);
  let room = t.ih - w;
  if (wall > 0.0005 && room > fe) {
    let rc = min(max(r - w, fe), 0.5 * room);
    let inner = corner3(t.v + w + rc, t.n1, t.n2, t.n3);
    if (rc - inner.w > outer.w) { return vec4f(-inner.xyz, rc - inner.w); }
  }
  return outer;
}
fn top_solid(t: Top, r: f32, wall: f32, fe: f32) -> f32 { return top_gradient(t, r, wall, fe).w; }

";

fn turn_wgsl() -> String {
    // the inverse of TURN (its transpose): a point of the volume taken into the polyhedron's own frame
    format!("const TT: mat3x3f = {};\n", wgsl_mat(mat_t(TURN)))
}

fn round_wall() -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("round", "rounding", 0.0, 0.0, 0.3, 0.005),
        ParamDef::labelled("wall", "hollow: wall", 0.0, 0.0, 0.3, 0.005),
    ]
}

// --------------------------------------------------------------------------------------------- johnson

/// n points round the y axis at radius r and height y, the first at angle `phase`
fn ring(n: usize, r: f64, y: f64, phase: f64) -> Vec<V> {
    (0..n).map(|k| { let a = phase + std::f64::consts::TAU * k as f64 / n as f64; [r * a.cos(), y, r * a.sin()] }).collect()
}
/// the radius of a regular n-gon of edge 1
fn rad(n: usize) -> f64 { 0.5 / (std::f64::consts::PI / n as f64).sin() }
/// a cupola of edge 1: an n-gon at height h over a 2n-gon at 0
fn cupola(n: usize) -> (Vec<V>, f64) {
    let (rt, rb) = (rad(n), rad(2 * n));
    let dh2 = rt * rt + rb * rb - 2.0 * rt * rb * (std::f64::consts::PI / (2 * n) as f64).cos();
    let h = (1.0 - dh2).sqrt();
    let mut v = ring(n, rt, h, 0.0);
    v.extend(ring(2 * n, rb, 0.0, std::f64::consts::PI / (2 * n) as f64));
    (v, h)
}
fn icosahedron() -> Vec<V> {
    let mut v = Vec::new();
    for a in [-1.0, 1.0] { for b in [-PHI, PHI] { v.push([0.0, a, b]); v.push([a, b, 0.0]); v.push([b, 0.0, a]); } }
    v
}
fn dodecahedron() -> Vec<V> {
    let g = 1.0 / PHI;
    let mut v = Vec::new();
    for a in [-1.0, 1.0] { for b in [-1.0, 1.0] {
        for c in [-1.0, 1.0] { v.push([a, b, c]); }
        v.push([0.0, a * g, b * PHI]); v.push([a * g, b * PHI, 0.0]); v.push([a * PHI, 0.0, b * g]);
    } }
    v
}
fn icosidodecahedron() -> Vec<V> {
    let mut v = vec![[0.0, 0.0, PHI], [0.0, 0.0, -PHI], [0.0, PHI, 0.0], [0.0, -PHI, 0.0], [PHI, 0.0, 0.0], [-PHI, 0.0, 0.0]];
    for a in [-0.5, 0.5] { for b in [-PHI / 2.0, PHI / 2.0] { for c in [-PHI * PHI / 2.0, PHI * PHI / 2.0] {
        v.push([a, b, c]); v.push([b, c, a]); v.push([c, a, b]);
    } } }
    v
}
/// points turned so that the unit axis `a` points up (y)
fn upright(pts: Vec<V>, a: V) -> Vec<V> {
    let ax = cross(a, [0.0, 1.0, 0.0]);
    if len(ax) < 1e-9 { return pts; }
    let m = axis_turn(unit(ax), dot(a, [0.0, 1.0, 0.0]).clamp(-1.0, 1.0).acos());
    pts.into_iter().map(|p| apply(m, p)).collect()
}

fn johnson(id: &'static str, name: &'static str, eq: Vec<&'static str>, note: &'static str, pts: Vec<V>, rot: [f64; 3]) -> Preset {
    let pts = centred(&pts, FIT);
    let planes: Vec<(V, f64, f64)> = hull(&pts).into_iter().map(|(n, h)| (turn(n), h, 0.0)).collect();
    let wgsl = format!("{FINE}{POLY}{tt}{f}
fn raw(p: vec3f) -> f32 {{
  return top_solid(solid(p, 0.0), max(pr.round, fine()), pr.wall, fine());
}}", tt = turn_wgsl(), f = planes_fn("solid", &planes, false));
    Preset {
        eq, note, params: round_wall(), rot: Some(view(rot)), animate: Some(vec!["round"]),
        ..Preset::new(id, name, Shape::Field { wgsl, clip: "-1.0".into(), shell: "0.0".into() })
    }
}

// -------------------------------------------------------------------------------------------- compound

/// One member of a compound: which of the base solids, and the matrix from the compound's frame to its own.
struct Member { base: usize, m: M }

/// The cube and the octahedron have closed-form distances in their own axes.
enum CompoundSolid { Cube, Octahedron, Hull(Vec<V>) }

impl CompoundSolid {
    fn wgsl(&self, i: usize) -> String {
        let head = format!("fn member{i}(q: vec3f, r: f32, limit: f32) -> vec4f");
        match self {
            Self::Cube => format!("{head} {{
  let d = abs(q) - vec3f(0.577350269) + r;
  let out = max(d, vec3f(0.0));
  let l = length(out);
  let hi = max(d.x, max(d.y, d.z));
  var n = select(vec3f(0, 0, 1), vec3f(0, 1, 0), d.y >= d.z);
  n = select(n, vec3f(1, 0, 0), d.x >= hi);
  n = select(n, out / max(l, 1e-12), l > 0.0);
  return vec4f(n * sign(q), l + min(hi, 0.0) - r);
}}\n"),
            // Reflect into the positive octant; the projection is on its dominant face, edge or vertex.
            Self::Octahedron => format!("{head} {{
  let p = abs(q);
  let s = 1.0 - 1.732050808 * r;
  let sum = p.x + p.y + p.z;
  let lo = min(p.x, min(p.y, p.z));
  let m = sum - s;
  let sg = sign(q);
  if (3.0 * lo >= m) {{ return vec4f(sg * 0.577350269, m * 0.577350269 - r); }}
  let hi = max(p.x, max(p.y, p.z));
  let mid = sum - lo - hi;
  let edge = 0.5 * (mid + hi - s);
  var delta = min(p, vec3f(edge));
  if (mid < edge) {{ delta = select(p, p - s, p == vec3f(hi)); }}
  let d = length(delta);
  return vec4f(sg * delta / max(d, 1e-12), d - r);
}}\n"),
            Self::Hull(points) => {
                let planes = hull(&centred(points, 1.0)).into_iter().map(|(n, h)| (n, h, 0.0)).collect::<Vec<_>>();
                let table = planes_fn(&format!("base{i}"), &planes, false);
                // The farthest plane is a lower bound. A member beyond limit cannot affect the union.
                format!("{table}{head} {{
  let t = base{i}(q, 0.0);
  if (t.v.x >= limit) {{ return vec4f(t.n1, t.v.x); }}
  return top_gradient(t, r, 0.0, 0.0);
}}\n")
            }
        }
    }
}

/// How a compound moves when it animates.
enum Motion {
    /// the members pushed apart along `dir` (in each member's own frame), by `explode`
    Explode(V),
    /// the second base solid's size against the first's, by `balance`
    Balance(f64),
}

/// A compound of solids, each of circumradius 1 in its own frame.
fn compound(id: &'static str, name: &'static str, eq: Vec<&'static str>, note: &'static str, bases: &[CompoundSolid], members: Vec<Member>, motion: Motion, rot: [f64; 3]) -> Preset {
    use std::fmt::Write;
    let mut w = format!("{FINE}{POLY}");
    for (i, b) in bases.iter().enumerate() {
        w += &b.wgsl(i);
    }
    let tt = mat_t(TURN);
    let (head, params, animate) = match motion {
        Motion::Explode(dir) => (
            format!("  let e = pr.explode;\n  let sc = (1.0 + e) / {FIT:.4};\n  let off = e * {};\n", wgsl_v(dir)),
            vec![
                ParamDef::labelled("explode", "pull apart", 0.0, 0.0, 0.8, 0.01),
                ParamDef::labelled("round", "rounding", 0.0, 0.0, 0.2, 0.005),
            ],
            "explode",
        ),
        Motion::Balance(v) => (
            format!("  let bal = pr.balance;\n  let sc = max(1.0, bal) / {FIT:.4};\n  let off = vec3f(0.0);\n"),
            vec![
                ParamDef::labelled("balance", "second solid's size", (v * 1000.0).round() / 1000.0, 0.5, 1.6, 0.001),
                ParamDef::labelled("round", "rounding", 0.0, 0.0, 0.2, 0.005),
            ],
            "balance",
        ),
    };
    let balanced = matches!(motion, Motion::Balance(_));
    let mut tables = String::new();
    w += "
fn join_member(a: vec4f, b: vec4f, frame: mat3x3f, scale: f32, k: f32) -> vec4f {
  let d = b.w * scale;
  if (d >= a.w + k) { return a; }
  let n = transpose(frame) * b.xyz;
  let t = clamp(0.5 + 0.5 * (a.w - d) / k, 0.0, 1.0);
  return vec4f(mix(a.xyz, n, t), smin(a.w, d, k));
}
fn rawGrad(p: vec3f) -> vec4f {
";
    w += &head;
    w += "  let r = max(pr.round, fine()) * sc;
  let k = 3.0 * fine() * sc;
  let q0 = p * sc;
  var d = vec4f(0.0, 0.0, 0.0, 1e9);
";
    for base in 0..bases.len() {
        let ms: Vec<String> = members.iter().filter(|m| m.base == base).map(|m| wgsl_mat(mat_mul(mat_t(m.m), tt))).collect();
        if balanced && base == 1 {
            writeln!(w, "  d = join_member(d, member1({m} * q0 / bal, r / bal, (d.w + k) / bal), {m}, bal, k);", m = ms[0]).unwrap();
        } else if ms.len() == 1 {
            writeln!(w, "  d = join_member(d, member{base}({m} * q0 - off, r, d.w + k), {m}, 1.0, k);", m = ms[0]).unwrap();
        } else {
            tables += &format!("var<private> members{base}: array<mat3x3f, {n}> = array<mat3x3f, {n}>({});\n", ms.join(", "), n = ms.len());
            writeln!(w, "  for (var i = 0; i < {n} + opaque(); i++) {{
    let frame = members{base}[i];
    d = join_member(d, member{base}(frame * q0 - off, r, d.w + k), frame, 1.0, k);
  }}", n = ms.len()).unwrap();
        }
    }
    w += "  return vec4f(d.xyz, d.w / sc);\n}\nfn raw(p: vec3f) -> f32 { return rawGrad(p).w; }";
    // (the ten tetrahedra: too many members to read at every surface inside, traced from the formula)
    let first = if id == "tentetra" { "// csg: the formula at the first surface only
" } else { "" };
    let w = format!("{first}{tables}{w}");
    let budget = match id {
        "tentetra" => Budget { thr_cut: Some(0.05), leave_cut: Some(true), ..Budget::default() },
        _ => Budget::default(),
    };
    Preset {
        eq, note, params, budget, rot: Some(view(rot)), animate: Some(vec![animate]),
        ..Preset::new(id, name, Shape::Field { wgsl: w, clip: "-1.0".into(), shell: "0.0".into() })
    }
}

/// The five turns about a five-fold axis of the dodecahedron (the compounds' members are one solid turned by each).
fn fivefold() -> (V, Vec<M>) {
    let a = unit([0.0, 1.0, PHI]);
    (a, (0..5).map(|k| axis_turn(a, std::f64::consts::TAU * k as f64 / 5.0)).collect())
}

// ------------------------------------------------------------------------------------------ zonohedron

fn zono_planes(gens: &[V], far: f64) -> Vec<(V, f64, f64)> {
    let support = |u: V| 0.5 * gens.iter().map(|g| dot(*g, u).abs()).sum::<f64>();
    let mut normals: Vec<V> = Vec::new();
    for i in 0..gens.len() {
        for j in i + 1..gens.len() {
            let c = cross(gens[i], gens[j]);
            if len(c) < 1e-9 { continue; }
            let c = unit(c);
            for n in [c, mul(c, -1.0)] {
                if !normals.iter().any(|m| len(sub(*m, n)) < 1e-6) { normals.push(n); }
            }
        }
    }
    // in units of the farthest corner: each plane's distance, and how it moves as the first vector stretches
    normals.iter().map(|n| (turn(*n), support(*n) / far, 0.5 * dot(gens[0], *n).abs() / far)).collect()
}

/// The zonohedron of a star of vectors: the sum of the segments from -g/2 to g/2. Its faces are square to
/// the cross products of pairs of the vectors; the first vector's length is a slider.
fn zonohedron(id: &'static str, name: &'static str, eq: Vec<&'static str>, note: &'static str, star: &[V], rot: [f64; 3]) -> Preset {
    let gens: Vec<V> = star.iter().map(|g| unit(*g)).collect();
    // the farthest corner: for a direction u the corner is the sum of ½ sign(g·u) g
    let mut far: f64 = 0.0;
    let n = 20000;
    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - y * y).sqrt();
        let a = i as f64 * 2.399_963_229_728_653;
        let u = [r * a.cos(), y, r * a.sin()];
        let c = gens.iter().fold([0.0; 3], |s, g| add(s, mul(*g, 0.5 * dot(*g, u).signum())));
        far = far.max(len(c));
    }
    let planes = zono_planes(&gens, far);
    let wgsl = format!("{FINE}{POLY}{f}
fn rawGrad(p: vec3f) -> vec4f {{
  let s1 = pr.stretch - 1.0;
  // stretching the first vector moves the farthest corner out by no more than half the stretch
  let sc = (1.0 + max(s1, 0.0) * {g0:.6}) / {FIT:.4};
  let q = p * sc;
  let hit = top_gradient(zono(q, s1), max(pr.round, fine()) * sc, pr.wall * sc, fine() * sc);
  return vec4f(hit.xyz, hit.w / sc);
}}
fn raw(p: vec3f) -> f32 {{ return rawGrad(p).w; }}", f = planes_fn("zono", &planes, true), g0 = 0.5 / far);
    let mut params = vec![ParamDef::labelled("stretch", "first vector's length", 1.0, 0.3, 2.5, 0.01)];
    params.extend(round_wall());
    Preset {
        eq, note, params, rot: Some(view(rot)), animate: Some(vec!["stretch"]),
        ..Preset::new(id, name, Shape::Field { wgsl, clip: "-1.0".into(), shell: "0.0".into() })
    }
}

/// The face count of a zonohedron's star (for the tests and notes).
pub fn zonohedron_faces(star: &[V]) -> usize {
    let gens: Vec<V> = star.iter().map(|g| unit(*g)).collect();
    let mut normals: Vec<V> = Vec::new();
    for i in 0..gens.len() { for j in i + 1..gens.len() {
        let c = unit(cross(gens[i], gens[j]));
        for n in [c, mul(c, -1.0)] { if !normals.iter().any(|m| len(sub(*m, n)) < 1e-6) { normals.push(n); } }
    } }
    normals.len()
}

// The polar zonohedron: n unit vectors evenly round a cone of slope θ, gₖ = (cos θ cos aₖ, sin θ,
// cos θ sin aₖ), aₖ = 2πk/n. Everything is had in closed form, since n is a slider. Its faces come in
// classes, one for each pair of vectors d apart (d up to n/2), each class n faces turned about the axis
// and their n opposites. For the pair (g₀, g_d), with φ = πd/n,
//   normal  ∝ (sin θ cos φ, −cos θ cos φ, sin θ sin φ)
//   its distance from the centre  ½ sin θ cos θ (2 sin((d+1)π/n) / sin(π/n) + (n − 2d − 2) cos φ) / |that|
// and the farthest corner is the sum of a run of m neighbouring vectors less the rest (m = 0..n).
// Within a class every face stands at the same slope, so the one a point lies farthest beyond is the one
// nearest it in azimuth: only the two bracketing the point's azimuth are tried, of each class and each
// side. (tools: the forms were checked against the faces and corners found by brute force.)
const POLAR: &str = "
fn raw(p: vec3f) -> f32 {
  let n = clamp(round(pr.sides), 3.0, 12.0);
  let al = radians(pr.rise);
  let S = sin(al);
  let C = cos(al);
  let step = 6.2831853 / n;
  let sn = sin(0.5 * step);
  var far = 0.0;
  for (var m = 0; m <= 12 + opaque(); m++) {
    if (f32(m) > n) { break; }
    let x = C * sin(f32(m) * 0.5 * step) / sn;
    let y = S * (f32(m) - 0.5 * n);
    far = max(far, x * x + y * y);
  }
  // the polyhedron's own units (vectors of length 1) for a point of the volume
  let sc = sqrt(far) / FIT;
  let q = TT * p * sc;
  let psi = atan2(q.z, q.x);
  var t = top_new();
  for (var d = 1; d <= 6 + opaque(); d++) {
    if (2.0 * f32(d) > n) { break; }
    let ph = 0.5 * step * f32(d);
    let cp = cos(ph);
    let nrm = inverseSqrt(S * S + C * C * cp * cp);
    let h = 0.5 * S * C * (2.0 * sin(0.5 * step * f32(d + 1)) / sn + (n - 2.0 * f32(d) - 2.0) * cp) * nrm;
    for (var s = 0; s < 2; s++) {
      let az0 = ph + select(0.0, 3.14159265, s == 1);
      let vy = select(-C * cp, C * cp, s == 1) * nrm;
      let j = floor((psi - az0) / step);
      for (var e = 0; e < 2; e++) {
        let az = az0 + (j + f32(e)) * step;
        let nn = vec3f(S * nrm * cos(az), vy, S * nrm * sin(az));
        top_add(&t, dot(q, nn) - h, nn, h);
      }
    }
  }
  return top_solid(t, max(pr.round, fine()) * sc, pr.wall * sc, fine() * sc) / sc;
}";

// -------------------------------------------------------------------------------------------- geodesic

/// The geodesic sphere's constants: the icosahedron face the kaleidoscope's chamber lies in (corners GA,
/// GB, GC on the unit sphere), its plane, the dual basis that gives a point's place in the face, and the
/// mirrors through its edges.
fn geodesic_consts() -> String {
    let a = unit([1.0, 0.0, PHI]);
    let b = unit([-1.0, 0.0, PHI]);
    let c = unit([0.0, PHI, 1.0]);
    let nf = unit(add(add(a, b), c));
    let hf = dot(a, nf);
    let (e1, e2) = (sub(b, a), sub(c, a));
    // dual basis in the face's plane: ds·e1 = 1, ds·e2 = 0, dt·e1 = 0, dt·e2 = 1
    let (g11, g12, g22) = (dot(e1, e1), dot(e1, e2), dot(e2, e2));
    let det = g11 * g22 - g12 * g12;
    let ds = mul(sub(mul(e1, g22), mul(e2, g12)), 1.0 / det);
    let dt = mul(sub(mul(e2, g11), mul(e1, g12)), 1.0 / det);
    let mab = unit(cross(a, b));
    let mac = unit(cross(a, c));
    let mbc = unit(cross(b, c));
    format!(
        "const GA: vec3f = {};\nconst GB: vec3f = {};\nconst GC: vec3f = {};\nconst GNF: vec3f = {};\nconst GHF: f32 = {hf:.9};\nconst GDS: vec3f = {};\nconst GDT: vec3f = {};\nconst GMAB: vec3f = {};\nconst GMAC: vec3f = {};\nconst GMBC: vec3f = {};\n",
        wgsl_v(a), wgsl_v(b), wgsl_v(c), wgsl_v(nf), wgsl_v(ds), wgsl_v(dt), wgsl_v(mab), wgsl_v(mac), wgsl_v(mbc)
    )
}

const GEODESIC: &str = "
var<private> geo_f: f32;
var<private> geo_s: f32;
fn geo_flat(i: f32, j: f32) -> vec3f { return GA + (GB - GA) * (i / geo_f) + (GC - GA) * (j / geo_f); }
fn geo_out(x: vec3f) -> vec3f { return mix(x, normalize(x), geo_s); }
fn geo_mirror(x: vec3f, m: vec3f) -> vec3f { return x - 2.0 * dot(x, m) * m; }
// corner (i, j) of the face's grid (i along GA-GB, j along GA-GC), or of the grid continued across one of
// the face's edges, which is the mirror image of a corner of this face
fn geo_v(i: f32, j: f32) -> vec3f {
  if (j < 0.0) { return geo_mirror(geo_out(geo_flat(i + j, -j)), GMAB); }
  if (i < 0.0) { return geo_mirror(geo_out(geo_flat(-i, j + i)), GMAC); }
  if (i + j > geo_f) { return geo_mirror(geo_out(geo_flat(geo_f - j, geo_f - i)), GMBC); }
  return geo_out(geo_flat(i, j));
}
// The small triangle a point of the chamber looks out through (its corners v0, v1, v2) and, across each of
// its edges (v1v2, v0v1, v0v2), the far corner of the neighbouring triangle (w0, w1, w2).
struct Cell { v0: vec3f, v1: vec3f, v2: vec3f, w0: vec3f, w1: vec3f, w2: vec3f }
fn geo_cell(q: vec3f) -> Cell {
  let g = q * (GHF / max(dot(q, GNF), 1e-4));
  let x = dot(g - GA, GDS) * geo_f;
  let y = dot(g - GA, GDT) * geo_f;
  let i = clamp(floor(x), 0.0, geo_f - 1.0);
  let j = clamp(floor(y), 0.0, geo_f - 1.0 - i);
  let up = (x - i) + (y - j) < 1.0 || i + j > geo_f - 2.0;
  if (up) {
    return Cell(geo_v(i, j), geo_v(i + 1.0, j), geo_v(i, j + 1.0), geo_v(i + 1.0, j + 1.0), geo_v(i + 1.0, j - 1.0), geo_v(i - 1.0, j + 1.0));
  }
  return Cell(geo_v(i + 1.0, j + 1.0), geo_v(i, j + 1.0), geo_v(i + 1.0, j), geo_v(i, j), geo_v(i, j + 2.0), geo_v(i + 2.0, j));
}
fn geo_setup() {
  geo_f = clamp(round(pr.freq), 1.0, 8.0);
  geo_s = clamp(pr.sphere, 0.0, 1.0);
}
";

const GEO_SOLID: &str = "
// the plane through three corners, facing out
fn geo_plane(t: ptr<function, Top>, q: vec3f, a: vec3f, b: vec3f, c: vec3f) {
  var n = normalize(cross(b - a, c - a));
  n *= sign(dot(n, a + b + c));
  let h = dot(n, a);
  top_add(t, dot(q, n) - h, n, h);
}
fn raw(p: vec3f) -> f32 {
  geo_setup();
  let sc = 1.0 / FIT;
  let k = kal(5.0);
  let q = kal_fold(TT * p, k.nc) * sc;
  let c = geo_cell(q);
  var t = top_new();
  geo_plane(&t, q, c.v0, c.v1, c.v2);
  geo_plane(&t, q, c.v1, c.v2, c.w0);
  geo_plane(&t, q, c.v0, c.v1, c.w1);
  geo_plane(&t, q, c.v0, c.v2, c.w2);
  return top_solid(t, max(pr.round, fine()) * sc, 0.0, fine() * sc) / sc;
}";

const GEO_FRAME: &str = "
fn geo_seg(q: vec3f, a: vec3f, b: vec3f) -> vec3f {
  let ab = b - a;
  let h = clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0);
  return q - a - ab * h;
}
// whether grid corner (i, j) is in this face or one mirror away from it (a corner of the icosahedron has
// five neighbours, not six: the sixth place round it would be two mirrors away, and is left out)
fn geo_near(i: f32, j: f32) -> bool {
  let f = geo_f;
  if (j < 0.0) { return i + j >= 0.0 && i <= f; }
  if (i < 0.0) { return j + i >= 0.0 && j <= f; }
  if (i + j > f) { return j <= f && i <= f; }
  return true;
}
// the nearest of the struts that leave grid corner (i, j), and of its hub
fn geo_star(q: vec3f, i: f32, j: f32, tube: f32, ball: f32) -> vec4f {
  let c = geo_v(i, j);
  var nearest = q - c;
  var best_sq = dot(nearest, nearest);
  let di = array<f32, 6>(1.0, -1.0, 0.0, 0.0, 1.0, -1.0);
  let dj = array<f32, 6>(0.0, 0.0, 1.0, -1.0, -1.0, 1.0);
  for (var k = 0; k < 6 + opaque(); k++) {
    let ni = i + di[k];
    let nj = j + dj[k];
    if (geo_near(ni, nj)) {
      let delta = geo_seg(q, c, geo_v(ni, nj));
      let sq = dot(delta, delta);
      if (sq < best_sq) { best_sq = sq; nearest = delta; }
    }
  }
  let strut = sqrt(best_sq);
  let hub = length(q - c);
  let mix_t = clamp(0.5 + 0.5 * ((hub - ball) - (strut - tube)) / 0.02, 0.0, 1.0);
  let normal = mix((q - c) / max(hub, 1e-10), nearest / max(strut, 1e-10), mix_t);
  return vec4f(normal, smin(strut - tube, hub - ball, 0.02));
}
// Fold the point with the same five mirror rounds as kal_fold. Record each reflection so the
// gradient of the winning strut or hub can be returned to the object's coordinate frame.
var<private> geo_mirrors: u32;
fn geo_frame(p: vec3f) -> vec4f {
  geo_setup();
  // the corners sit on a sphere this far out, so that struts and hubs just reach FIT
  let sc = 1.0 / (FIT - max(pr.tube, pr.ball));
  let k = kal(5.0);
  var chamber = TT * p;
  geo_mirrors = 0u;
  for (var j = 0; j < 5; j++) {
    var flips = select(0u, 1u, chamber.x < 0.0) | select(0u, 2u, chamber.y < 0.0);
    chamber = vec3f(abs(chamber.xy), chamber.z);
    let plane = dot(chamber, k.nc);
    flips |= select(0u, 4u, plane < 0.0);
    chamber -= 2.0 * min(plane, 0.0) * k.nc;
    geo_mirrors = (geo_mirrors << 3u) | flips;
  }
  let q = chamber * sc;
  // the small triangle the point looks out through: every strut near it leaves one of its corners
  let g = q * (GHF / max(dot(q, GNF), 1e-4));
  let x = dot(g - GA, GDS) * geo_f;
  let y = dot(g - GA, GDT) * geo_f;
  let i = clamp(floor(x), 0.0, geo_f - 1.0);
  let j = clamp(floor(y), 0.0, geo_f - 1.0 - i);
  let up = (x - i) + (y - j) < 1.0 || i + j > geo_f - 2.0;
  let t = pr.tube * sc;
  let bl = pr.ball * sc;
  let a = geo_star(q, i + 1.0, j, t, bl);
  let b = geo_star(q, i, j + 1.0, t, bl);
  let step = select(1.0, 0.0, up);
  let c = geo_star(q, i + step, j + step, t, bl);
  let ab = select(a, b, b.w < a.w);
  let nearest = select(ab, c, c.w < ab.w);
  return vec4f(nearest.xyz, nearest.w / sc);
}
fn raw(p: vec3f) -> f32 { return geo_frame(p).w; }
fn rawGrad(p: vec3f) -> vec4f {
  let nearest = geo_frame(p);
  var n = nearest.xyz;
  let nc = kal(5.0).nc;
  var bits = geo_mirrors;
  for (var j = 0; j < 5; j++) {
    if ((bits & 4u) != 0u) { n -= 2.0 * dot(n, nc) * nc; }
    n.x = select(n.x, -n.x, (bits & 1u) != 0u);
    n.y = select(n.y, -n.y, (bits & 2u) != 0u);
    bits >>= 3u;
  }
  return vec4f(transpose(TT) * n, nearest.w);
}";

fn geodesic(frame: bool) -> Preset {
    let head = format!("const FIT: f32 = {FIT:.4};\n{FINE}{KALEIDOSCOPE}{POLY}{tt}{c}{GEODESIC}", tt = turn_wgsl(), c = geodesic_consts());
    let eq = vec!["each face of an icosahedron cut into ν² triangles,", "every corner pushed out to the sphere by s"];
    if frame {
        Preset {
            eq,
            note: "Buckminster Fuller’s dome, closed into a ball: struts of nearly equal length, five meeting at twelve corners and six everywhere else. At s = 0 it is the flat icosahedron, cut up.",
            params: vec![
                ParamDef::labelled("freq", "frequency ν", 3.0, 1.0, 8.0, 1.0),
                ParamDef::labelled("sphere", "onto the sphere s", 1.0, 0.0, 1.0, 0.01),
                ParamDef::labelled("tube", "strut", 0.035, 0.012, 0.1, 0.001),
                ParamDef::labelled("ball", "hub", 0.05, 0.0, 0.14, 0.002),
            ],
            rot: Some(view([16.0, 24.0, 0.0])),
            animate: Some(vec!["sphere"]),
            ..Preset::new("geodesicframe", "Geodesic dome frame", Shape::Field { wgsl: format!("{head}{GEO_FRAME}"), clip: "-1.0".into(), shell: "0.0".into() })
        }
    } else {
        Preset {
            eq,
            note: "A geodesic sphere as a cut stone: 20ν² triangles, flat, nearly a ball. Slide s down and the facets flatten back onto the icosahedron’s twenty faces.",
            params: vec![
                ParamDef::labelled("freq", "frequency ν", 3.0, 1.0, 8.0, 1.0),
                ParamDef::labelled("sphere", "onto the sphere s", 1.0, 0.0, 1.0, 0.01),
                ParamDef::labelled("round", "rounding", 0.0, 0.0, 0.1, 0.002),
            ],
            rot: Some(view([16.0, 24.0, 0.0])),
            animate: Some(vec!["sphere"]),
            ..Preset::new("geodesic", "Geodesic sphere", Shape::Field { wgsl: format!("{head}{GEO_SOLID}"), clip: "-1.0".into(), shell: "0.0".into() })
        }
    }
}

// ----------------------------------------------------------------------------------------------- family

/// The stars of the zonohedra.
pub fn star(id: &str) -> Vec<V> {
    let g = 1.0 / PHI;
    match id {
        "rhombicdodeca" => vec![[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]],
        "triacontahedron" => vec![[0.0, 1.0, PHI], [0.0, 1.0, -PHI], [1.0, PHI, 0.0], [-1.0, PHI, 0.0], [PHI, 0.0, 1.0], [PHI, 0.0, -1.0]],
        "truncoct" => vec![[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0], [0.0, 1.0, -1.0]],
        "enneacontahedron" => vec![
            [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0],
            [0.0, g, PHI], [0.0, g, -PHI], [g, PHI, 0.0], [-g, PHI, 0.0], [PHI, 0.0, g], [PHI, 0.0, -g],
        ],
        "truncicosidodeca" => {
            // the fifteen two-fold axes: the edge midpoints of the icosahedron, one of each opposite pair
            let v = icosahedron();
            let mut out: Vec<V> = Vec::new();
            for i in 0..v.len() { for j in i + 1..v.len() {
                if (len(sub(v[i], v[j])) - 2.0).abs() > 1e-9 { continue; }
                let m = unit(add(v[i], v[j]));
                if !out.iter().any(|o| len(sub(*o, m)) < 1e-6 || len(add(*o, m)) < 1e-6) { out.push(m); }
            } }
            out
        }
        _ => panic!("no star called {id}"),
    }
}

fn presets() -> Vec<Preset> {
    let mut v = Vec::new();

    // Johnson solids
    v.push(johnson("squarepyramid", "Square pyramid", vec!["J₁: four equilateral triangles on a square"],
        "The first of the 92 Johnson solids: convex, every face a regular polygon, yet not one of the uniform solids. Half an octahedron.",
        { let mut p = ring(4, rad(4), 0.0, 0.0); p.push([0.0, (1.0 - rad(4) * rad(4)).sqrt(), 0.0]); p }, [24.0, 28.0, 0.0]));
    v.push(johnson("pentpyramid", "Pentagonal pyramid", vec!["J₂: five equilateral triangles on a pentagon"],
        "An icosahedron’s cap: cut one corner off an icosahedron and this is the piece that comes away.",
        { let mut p = ring(5, rad(5), 0.0, 0.0); p.push([0.0, (1.0 - rad(5) * rad(5)).sqrt(), 0.0]); p }, [24.0, 28.0, 0.0]));
    v.push(johnson("tricupola", "Triangular cupola", vec!["J₃: a triangle over a hexagon, joined by", "three squares and three triangles"],
        "Half a cuboctahedron, cut through one of its hexagonal waists.",
        cupola(3).0, [26.0, 24.0, 0.0]));
    v.push(johnson("pentcupola", "Pentagonal cupola", vec!["J₅: a pentagon over a decagon, joined by", "five squares and five triangles"],
        "A cap of the rhombicosidodecahedron.", cupola(5).0, [26.0, 24.0, 0.0]));
    v.push(johnson("rotunda", "Pentagonal rotunda", vec!["J₆: a pentagon over a decagon, joined by", "five pentagons and ten triangles"],
        "Half an icosidodecahedron: its domed top is a dome of pentagons and triangles.",
        upright(icosidodecahedron().into_iter().filter(|p| dot(*p, unit([0.0, 1.0, PHI])) > -1e-9).collect(), unit([0.0, 1.0, PHI])), [26.0, 24.0, 0.0]));
    v.push(johnson("gyrobicupola", "Square gyrobicupola", vec!["J₂₉: two square cupolas base to base,", "one turned by 45°"],
        "Turned the other way (the orthobicupola) its squares and triangles would line up; turned, it is the rhombicuboctahedron’s cousin with a twist at the waist.",
        { let (mut p, h) = cupola(4); p.extend(ring(4, rad(4), -h, std::f64::consts::FRAC_PI_4)); p }, [18.0, 22.5, 0.0]));
    v.push(johnson("gyrobipyramid", "Gyroelongated square bipyramid", vec!["J₁₇: a square antiprism capped", "by a pyramid at each end"],
        "Sixteen equilateral triangles: one of the eight convex deltahedra, the solids made of nothing else.",
        {
            let r = rad(4);
            let half = 0.5 * (1.0 - (2.0 * r * r - 2.0 * r * r * std::f64::consts::FRAC_PI_4.cos())).sqrt();
            let tip = half + (1.0 - r * r).sqrt();
            let mut p = ring(4, r, half, 0.0);
            p.extend(ring(4, r, -half, std::f64::consts::FRAC_PI_4));
            p.push([0.0, tip, 0.0]);
            p.push([0.0, -tip, 0.0]);
            p
        }, [10.0, 20.0, 0.0]));
    v.push(johnson("bidiminished", "Metabidiminished icosahedron", vec!["J₆₂: an icosahedron with two corners cut away,", "neither neighbours nor opposite"],
        "Take two caps off an icosahedron and two pentagons appear, meeting at a corner.",
        {
            let ico = icosahedron();
            let a = ico[0];
            let b = *ico.iter().find(|q| (len(sub(**q, a)) - 2.0 * PHI).abs() < 1e-9).unwrap();
            upright(ico.into_iter().filter(|q| len(sub(*q, a)) > 1e-9 && len(sub(*q, b)) > 1e-9).collect(), unit(add(a, b)))
        }, [30.0, 20.0, 0.0]));
    v.push(johnson("augdodeca", "Augmented dodecahedron", vec!["J₅₈: a dodecahedron with a pentagonal pyramid", "on one face"],
        "A dodecahedron that has grown a spire: five triangles rising from one of its pentagons.",
        {
            let mut d = dodecahedron();
            let n = unit([0.0, 1.0, PHI]);
            let edge = 2.0 / PHI;
            let inr = d.iter().map(|q| dot(*q, n)).fold(f64::MIN, f64::max);
            let tip = inr + (1.0 - rad(5) * rad(5)).sqrt() * edge;
            d.push(mul(n, tip));
            upright(d, n)
        }, [18.0, 20.0, 0.0]));

    // compounds
    let tet_a: Vec<V> = vec![[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
    let tet_b: Vec<V> = tet_a.iter().map(|p| mul(*p, -1.0)).collect();
    let (axis, five) = fivefold();
    let out = unit(cross(axis, [1.0, 0.0, 0.0]));
    let ring_of = |base: usize| -> Vec<Member> { five.iter().map(|m| Member { base, m: *m }).collect() };
    let id3: M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // each member's own frame: the direction it is pushed is `out` turned with it, the same in every frame
    v.push(compound("fivecubes", "Compound of five cubes", vec!["five cubes, each turned a fifth of a turn", "about an axis through two faces of a dodecahedron"],
        "Their 40 corners are the dodecahedron’s 20, each shared by two cubes; every cube is one of the five that fit inside it.",
        &[CompoundSolid::Cube], ring_of(0), Motion::Explode(out), [16.0, 14.0, 0.0]));
    v.push(compound("fivetetra", "Compound of five tetrahedra", vec!["five tetrahedra on the corners of a dodecahedron,", "each turned a fifth of a turn"],
        "It has a hand: its mirror image is a different compound, and the two together are the compound of ten.",
        &[CompoundSolid::Hull(tet_a.clone())], ring_of(0), Motion::Explode(out), [16.0, 14.0, 0.0]));
    v.push(compound("tentetra", "Compound of ten tetrahedra", vec!["the compound of five tetrahedra", "and its mirror image"],
            "Every corner of the dodecahedron is shared by two tetrahedra, one of each hand.",
            &[CompoundSolid::Hull(tet_a), CompoundSolid::Hull(tet_b)], { let mut m = ring_of(0); m.extend(ring_of(1)); m }, Motion::Explode(out), [16.0, 14.0, 0.0]));
    v.push(compound("fiveocta", "Compound of five octahedra", vec!["five octahedra whose corners are the 30", "corners of an icosidodecahedron"],
        "Pull it apart and five octahedra come away in a ring; together their faces make the shell of a great icosahedron-like star.",
        &[CompoundSolid::Octahedron], ring_of(0), Motion::Explode(out), [16.0, 14.0, 0.0]));
    v.push(compound("cubeocta", "Cube and octahedron", vec!["a cube and its dual, sized so that", "their edges cross at their midpoints"],
        "Each edge of one bisects an edge of the other at right angles. Shrink the octahedron and it hides inside the cube; grow it and the cube’s corners sink into its faces.",
        &[CompoundSolid::Cube, CompoundSolid::Octahedron], vec![Member { base: 0, m: id3 }, Member { base: 1, m: id3 }], Motion::Balance(2.0 / 3f64.sqrt()), [20.0, 28.0, 0.0]));
    v.push(compound("dodecaicosa", "Dodecahedron and icosahedron", vec!["a dodecahedron and its dual, sized so that", "their edges cross at their midpoints"],
        "Kepler drew it in Harmonices Mundi. The thirty crossings are the corners of an icosidodecahedron.",
        &[CompoundSolid::Hull(dodecahedron()), CompoundSolid::Hull(icosahedron())], vec![Member { base: 0, m: id3 }, Member { base: 1, m: id3 }],
        Motion::Balance({
            // the icosahedron's circumradius when its edge midpoints meet the dodecahedron's (circumradius 1)
            let d = dodecahedron(); let i = icosahedron();
            let mid = |pts: &[V], e: f64| -> f64 { let a = pts[0]; let b = pts.iter().find(|q| (len(sub(**q, a)) - e).abs() < 1e-9).unwrap(); len(mul(add(a, *b), 0.5)) / len(a) };
            mid(&d, 2.0 / PHI) / mid(&i, 2.0)
        }), [18.0, 22.0, 0.0]));

    // zonohedra
    v.push(zonohedron("rhombicdodeca", "Rhombic dodecahedron", vec!["the sum of four segments along", "the diagonals of a cube"],
        "Twelve rhombi. It fills space with copies of itself, as bees’ cells and garnet crystals do. Stretch one vector and four faces grow long.",
        &star("rhombicdodeca"), [18.0, 28.0, 0.0]));
    v.push(zonohedron("triacontahedron", "Rhombic triacontahedron", vec!["the sum of six segments along", "the five-fold axes of an icosahedron"],
        "Thirty golden rhombi, their diagonals in the golden ratio. The icosidodecahedron’s dual.",
        &star("triacontahedron"), [18.0, 22.0, 0.0]));
    v.push(zonohedron("truncoct", "Truncated octahedron", vec!["the sum of six segments along", "the face diagonals of a cube"],
        "Eight hexagons and six squares: Kelvin’s cell, which fills space with the least wall for its volume of any such shape he knew.",
        &star("truncoct"), [20.0, 30.0, 0.0]));
    v.push(zonohedron("enneacontahedron", "Rhombic enneacontahedron", vec!["the sum of ten segments along", "the three-fold axes of an icosahedron"],
        "Ninety rhombi of two kinds, sixty wide and thirty narrow.", &star("enneacontahedron"), [18.0, 22.0, 0.0]));
    v.push(zonohedron("truncicosidodeca", "Truncated icosidodecahedron", vec!["the sum of fifteen segments along", "the two-fold axes of an icosahedron"],
        "Thirty squares, twenty hexagons and twelve decagons: the largest of the Archimedean solids, and a zonohedron.",
        &star("truncicosidodeca"), [18.0, 22.0, 0.0]));
    v.push(Preset {
        eq: vec!["the sum of n segments evenly round a cone", "of slope θ: a polar zonohedron"],
        note: "Rhombi spiralling up from a point and down to another. Low θ flattens it to a star-like disc; high θ draws it into a spindle.",
        params: vec![
            ParamDef::labelled("sides", "vectors n", 9.0, 3.0, 12.0, 1.0),
            ParamDef::labelled("rise", "slope θ °", 30.0, 10.0, 75.0, 0.5),
            ParamDef::labelled("round", "rounding", 0.0, 0.0, 0.3, 0.005),
            ParamDef::labelled("wall", "hollow: wall", 0.0, 0.0, 0.3, 0.005),
        ],
        rot: Some(view([38.0, 10.0, 0.0])),
        animate: Some(vec!["rise"]),
        ..Preset::new("polarzono", "Polar zonohedron", Shape::Field {
            wgsl: format!("const FIT: f32 = {FIT:.4};\n{FINE}{POLY}{tt}{POLAR}", tt = turn_wgsl()),
            clip: "-1.0".into(), shell: "0.0".into(),
        })
    });

    // geodesic spheres
    v.push(geodesic(false));
    v.push(geodesic(true));
    v
}

pub fn family() -> Family {
    Family {
        step_scale: 0.85,
        steps_out: 220,
        steps_in: 64,
        layers: 2,
        presets: presets(),
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.20, 0.09, 0.05])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(4.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([22.0, 30.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        // the geometric solids' lights: broad ones that fade to their edges, so neighbouring faces differ, and
        // two small bright ones for glints
        lights: faceted_lights(),
        ..Family::new("polyhedra", "Polyhedra", Kind::Field)
    }
}
