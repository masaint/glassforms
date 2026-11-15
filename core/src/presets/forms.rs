//! Forms from other corners of mathematics than orbits, knots, minimal surfaces and polynomials: shapes
//! drawn by a formula for a radius (the superformula, spherical harmonics), by fields that add up or
//! multiply (metaballs, many-centred Cassini ovals), in the coordinates of a ring (bars and bands that
//! turn as they go round), by self-similarity (a seashell), by iteration (the equipotential of a Julia
//! set) and by inversion (a Dupin cyclide).
//!
//! Each is an equation in WGSL, `raw(p)`, negative inside the glass. None of them is a distance: the
//! bake walks every voxel to the surface. What that walk asks of `raw`, beyond crossing zero cleanly and
//! being continuous across any cut of the angles it uses, shaped several of these equations:
//!
//! - Away from the surface `raw` must keep a slope that leads to the surface. Where it lies level in a
//!   hollow that never reaches zero, a voxel steps to and fro across the hollow, ends where it began and
//!   records a surface that is not there. (Hence the stand-in far fields of the Cassini surface and the
//!   Julia set.)
//! - A thin wall cannot be written into `raw`: half way through it `raw` is at its lowest and the walk has
//!   nowhere to go. A hollow shape is the solid's equation with the bake's own wall laid about it. (The
//!   seashell.)
//!
//! Every shape is scaled inside its shader to fill the volume without leaving it, whatever its sliders
//! say; `core/tests/gens_forms.rs` checks the formulas that do that. These shapes have no counterpart in
//! the web version.
use super::*;
use crate::field::{CLIP_DEFAULT, SHELL_DEFAULT};

/// A shape that is its own bound: the smooth cut that other shapes get would only shave it.
const NO_CLIP: &str = "-1.0";

/// What every preset here gives.
struct Form {
    id: &'static str,
    name: &'static str,
    eq: &'static [&'static str],
    note: &'static str,
    params: Vec<ParamDef>,
    rot: [f64; 3],
    /// the slider that travels when the animation plays
    animate: &'static str,
    wgsl: String,
    clip: &'static str,
    /// half the thickness of a wall about the surface, as WGSL ("0.0" = fill the inside)
    shell: &'static str,
}

fn form(f: Form) -> Preset {
    Preset {
        eq: f.eq.to_vec(),
        note: f.note,
        params: f.params,
        rot: Some(f.rot),
        animate: Some(vec![f.animate]),
        ..Preset::new(f.id, f.name, Shape::Field { wgsl: f.wgsl, clip: f.clip.to_string(), shell: f.shell.to_string() })
    }
}

// ---- the superformula ---------------------------------------------------------------------------
//
// Gielis's formula gives a radius for every angle: r(a) = (|cos(m a/4)|^n2 + |sin(m a/4)|^n3)^(-1/n1).
// Two of them make a surface, one going round the vertical axis (r1 of the longitude) and one from pole
// to pole (r2 of the latitude):  x = r1 cos(lon) r2 cos(lat),  z = r1 sin(lon) r2 cos(lat),  y = r2 sin(lat).
// A point is inside when, having divided its distance from the axis by r1, it is nearer the centre than
// r2 of its own latitude.
//
// n2 and n3 are kept equal. That makes every whole m close up without a seam (with n2 != n3 an odd m
// leaves one), and the largest radius is then known in closed form, which is what scales the shape to fit.
const SUPER: &str = "
// the radius at angle a
fn gielis(a: f32, m: f32, n1: f32, n: f32) -> f32 {
  let t = 0.25 * m * a;
  return pow(pow(abs(cos(t)), n) + pow(abs(sin(t)), n), -1.0 / n1);
}
// the largest it gets: 1 where a lobe begins (n below 2), or its value half way along a lobe (n above 2)
fn gielisMax(n1: f32, n: f32) -> f32 { return max(1.0, pow(2.0, (0.5 * n - 1.0) / n1)); }
// How far the surface reaches from the centre. Sideways that is r1 at its largest times r2·cos(latitude),
// upward r2·sin(latitude); the latitude at which the two together are greatest is found by trying a
// spread of them (r2 is the same above and below the equator).
fn reach() -> f32 {
  let wide = select(gielisMax(pr.n1, pr.n2), 1.0, round(pr.m) == 0.0);     // m = 0 is a plain circle
  var most = 0.0;
  for (var i = 0; i <= 16; i++) {
    let lat = 1.5707963 * f32(i) / 16.0;
    let r2 = gielis(lat, round(pr.mv), pr.v1, pr.v2);
    most = max(most, r2 * length(vec2f(wide * cos(lat), sin(lat))));
  }
  return most;
}
// (worked out once for each voxel, not at every step of its walk to the surface)
var<private> fit: f32 = 0.0;
fn raw(p: vec3f) -> f32 {
  if (fit == 0.0) { fit = 1.25 / reach(); }
  let s = fit;
  let q = p / s;
  let r1 = gielis(atan2(q.z, q.x), round(pr.m), pr.n1, pr.n2);
  let rho = length(q.xz) / r1;
  let r2 = gielis(atan2(q.y, rho + 1e-9), round(pr.mv), pr.v1, pr.v2);
  return (length(vec2f(rho, q.y)) - r2) * s;
}";

/// around: m, n1, n2 of the longitude; upright: the same of the latitude
#[allow(clippy::too_many_arguments)]
fn supershape(id: &'static str, name: &'static str, note: &'static str, around: [f64; 3], upright: [f64; 3], rot: [f64; 3], animate: &'static str) -> Preset {
    form(Form {
        id, name, note, rot, animate,
        eq: &["r(a) = (|cos(m·a/4)|ⁿ² + |sin(m·a/4)|ⁿ²)^(−1/n₁)", "x = r₁(θ) cos θ · r₂(φ) cos φ", "z = r₁(θ) sin θ · r₂(φ) cos φ", "y = r₂(φ) sin φ"],
        params: vec![
            ParamDef::labelled("m", "around m", around[0], 0.0, 8.0, 1.0),
            ParamDef::labelled("n1", "around n₁", around[1], 1.5, 12.0, 0.05),
            ParamDef::labelled("n2", "around n₂", around[2], 1.0, 10.0, 0.05),
            ParamDef::labelled("mv", "upright m", upright[0], 0.0, 8.0, 1.0),
            ParamDef::labelled("v1", "upright n₁", upright[1], 1.5, 12.0, 0.05),
            ParamDef::labelled("v2", "upright n₂", upright[2], 1.0, 10.0, 0.05),
        ],
        wgsl: SUPER.to_string(),
        clip: NO_CLIP,
        shell: SHELL_DEFAULT,
    })
}

// ---- spherical harmonics --------------------------------------------------------------------------
//
// r = 1 + A·Y(l, m): a ball pushed out and in by one real spherical harmonic. The harmonic is worked out
// from the direction's own coordinates, with no angles: (x + iz)^m is sin^m(theta)·e^(i m phi), and what
// is left of the Legendre function is a polynomial in y, built by the usual recurrence in l. It is scaled
// so that the largest harmonic of each degree reaches 1 (the others stay below), which keeps A honest.
const HARMONICS: &str = "
fn harmonic(d: vec3f, l: i32, m: i32) -> f32 {
  var c = vec2f(1.0, 0.0);                 // (x + iz)^m
  var pl = 1.0;                            // (2m - 1)!!, the start of the recurrence
  for (var k = 1; k <= 8; k++) {
    if (k > m) { break; }
    c = vec2f(c.x * d.x - c.y * d.z, c.x * d.z + c.y * d.x);
    pl *= f32(2 * k - 1);
  }
  var below = 0.0;
  for (var k = 1; k <= 8; k++) {
    let ll = m + k;
    if (ll > l) { break; }
    let next = (f32(2 * ll - 1) * d.y * pl - f32(ll + m - 1) * below) / f32(ll - m);
    below = pl;
    pl = next;
  }
  var norm = 1.0;                          // (l - m)! / (l + m)!
  for (var k = 1; k <= 16; k++) {
    let j = l - m + k;
    if (j > l + m) { break; }
    norm /= f32(j);
  }
  return select(sqrt(2.0 * norm), 1.0, m == 0) * pl * c.x;
}
// (worked out once for each voxel, not at every step of its walk to the surface)
var<private> fit: f32 = 0.0;
fn raw(p: vec3f) -> f32 {
  let l = i32(clamp(round(pr.l), 0.0, 8.0));
  let m = min(i32(clamp(round(pr.m), 0.0, 8.0)), l);
  if (fit == 0.0) {
    // How far out the largest bulge reaches. The harmonic is a function of latitude times cos(m·longitude),
    // so its largest value is found by trying a spread of latitudes at longitude 0.
    var most = 0.0;
    for (var i = 0; i <= 32; i++) {
      let th = 3.14159265 * f32(i) / 32.0;
      most = max(most, abs(harmonic(vec3f(sin(th), cos(th), 0.0), l, m)));
    }
    fit = 1.25 / (1.0 + pr.amp * most);
  }
  let r = length(p) + 1e-6;
  return r - fit * (1.0 + pr.amp * harmonic(p / r, l, m));
}";

fn harmonics(id: &'static str, name: &'static str, note: &'static str, l: f64, m: f64, amp: f64, rot: [f64; 3]) -> Preset {
    form(Form {
        id, name, note, rot,
        animate: "amp",
        eq: &["r = 1 + A·Yₗᵐ(θ, φ)", "Yₗᵐ ∝ Pₗᵐ(cos θ) cos mφ"],
        params: vec![
            ParamDef::labelled("l", "degree l", l, 1.0, 8.0, 1.0),
            ParamDef::labelled("m", "order m", m, 0.0, 8.0, 1.0),
            ParamDef::labelled("amp", "A", amp, 0.0, 0.9, 0.01),
        ],
        wgsl: HARMONICS.to_string(),
        clip: NO_CLIP,
        shell: SHELL_DEFAULT,
    })
}

// ---- shapes in the coordinates of a ring ----------------------------------------------------------
//
// A cross-section carried round a circle, turning about the circle as it goes. `section(w)` is the
// cross-section (negative inside) and `turns` how many whole turns it makes in one circuit: that must
// be a multiple of the section's own symmetry, or the ends would not meet where the angle wraps.
// The slider `turn` then turns the section everywhere alike, which slides the whole twist round the ring.
fn ring_wgsl(section: &str, turns: &str) -> String {
    format!("{section}
fn raw(p: vec3f) -> f32 {{
  let th = atan2(p.z, p.x);
  let q = vec2f(length(p.xz) - pr.ring, p.y);
  let al = ({turns}) * th + radians(pr.turn);
  let c = cos(al);
  let s = sin(al);
  return section(vec2f(c * q.x + s * q.y, c * q.y - s * q.x));
}}")
}

// A regular polygon with rounded corners, as an exact distance: fold the angle into one side's sector,
// where the nearest edge is that side (x = apo, |y| <= h).
const POLYGON: &str = "
fn section(w: vec2f) -> f32 {
  let an = 6.2831853 / clamp(round(pr.sides), 3.0, 12.0);
  let a = atan2(w.y, w.x);
  let f = a - an * floor(a / an + 0.5);
  let l = length(w);
  let v = vec2f(l * cos(f), abs(l * sin(f)));
  let apo = pr.size * cos(0.5 * an) - pr.edge;
  let h = apo * tan(0.5 * an);
  return select(length(v - vec2f(apo, h)), v.x - apo, v.y <= h) - pr.edge;
}";

// A flat band with rounded edges.
const BAND: &str = "
fn section(w: vec2f) -> f32 {
  let r = min(pr.edge, pr.thick);
  let d = abs(w) - vec2f(pr.width, pr.thick) + r;
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0) - r;
}";

// A round section with ridges: its radius rises and falls n times round.
const RIDGED: &str = "
fn section(w: vec2f) -> f32 {
  return length(w) - pr.size * (1.0 + pr.depth * cos(round(pr.ridges) * atan2(w.y, w.x)));
}";

// ---- a seashell ---------------------------------------------------------------------------------
//
// A tube round a logarithmic spiral: its centre is rho = e^(b·angle) from the axis, and both its radius
// and its drop along the axis are in proportion to rho, so every turn is the turn before, enlarged.
// A point finds the turn it is nearest by how far out it is along the cone of tube centres; that turn
// and its two neighbours are tried. The spiral winds in for ever: turns too small for the voxels to
// show simply fuse into a solid tip. The last turn ends flat, at the mouth.
//
// Hollow, the equation stays that of the solid and the bake's own wall is laid about its surface: one
// skin, with one winding chamber inside. (Walls written into the equation itself would not do: half way
// through a wall the equation is at its lowest, and a voxel walking to the surface from there has no
// direction to go.) That wall closes the mouth too, so a window is then cut in it (`shellCut`), which
// leaves the mouth open with a narrow lip.
const SHELL: &str = "
// where a point is, in the shell's own terms
struct Spot {
  rho: f32,       // distance from the axis
  y: f32,         // distance down the axis from the tip
  th: f32,        // angle round the axis, -pi to pi; the mouth is at 0
  xz: vec2f,      // across the axis
  rm: f32,        // the last turn's centre is this far from the axis
  b: f32,         // rho = rm·e^(b·angle)
}
// the shell's size, and where its centre is in the shell's terms (x, down from the tip, z):
// worked out once for each voxel, not at every step of its walk to the surface
var<private> fitted: bool = false;
var<private> fitSize: f32;
var<private> fitCentre: vec3f;
fn spot(p: vec3f) -> Spot {
  let b = log(pr.growth) / 6.2831853;
  if (!fitted) {
    let a = pr.fat;
    let h = pr.spire;
    // Sizes are first worked out with the last turn's centre 1 from the axis. Seen along the axis the
    // outline is the spiral (1 + a)·e^(b·angle), which reaches furthest right at the mouth and furthest
    // left, down and up where its tangent turns back (the angles below): the shell is centred between
    // those. Along the axis it runs from its tip (or the top of the last turn, if that stands higher) to
    // the foot of the last turn. A ball about that centre, through the corner of that box, holds it all;
    // a hollow shell's wall stands out from the surface by half its thickness, which is allowed for.
    let be = atan(b);
    let out = (1.0 + a) * cos(be);
    let xl = out * exp(b * (be - 3.14159265));
    let zd = out * exp(b * (be - 1.5707963));
    let zu = out * exp(b * (be - 4.712389));
    let top = min(0.0, h - a);
    let plan = 0.5 * max(1.0 + a + xl, zd + zu);
    // (that ball is a little larger than it need be, never by less than a fiftieth: hence 1.3 and not 1.25)
    fitSize = 1.3 / (length(vec2f(plan, 0.5 * (h + a - top))) * (1.0 + 0.5 * pr.wall * a));
    fitCentre = fitSize * vec3f(0.5 * (1.0 + a - xl), 0.5 * (h + a + top), 0.5 * (zu - zd));
    fitted = true;
  }
  let xz = p.xz + fitCentre.xz;
  return Spot(max(length(xz), 1e-5), fitCentre.y - p.y, atan2(xz.y, xz.x), xz, fitSize, b);
}
// how far out along the cone of tube centres a point is: the size of the turn passing there
fn along(s: Spot) -> f32 { return max((s.rho + pr.spire * s.y) / (1.0 + pr.spire * pr.spire), 1e-6); }
// The tube of turn j, counted back from the last (0), which passes this angle with its centre
// rm·e^(b(th - 2 pi j)) from the axis. (Past the mouth, turn 0 is the last turn carried on.)
// Round the tube this is a distance. Along it the tube grows and drops, so its surface is tilted and
// the true distance is a little less: by a sixth or so near the surface of a fast-growing shell.
fn turn(s: Spot, j: f32) -> f32 {
  let rc = s.rm * exp(s.b * (s.th - 6.2831853 * j));
  return length(vec2f(s.rho - rc, s.y - pr.spire * rc)) - pr.fat * rc;
}
// how far past the mouth a point is: the mouth lies in the half-plane z = 0, x > 0 of the shell
fn beyond(s: Spot) -> f32 { return select(select(-s.rho, s.rho, s.xz.y >= 0.0), s.xz.y, s.xz.x > 0.0); }
// half the wall's thickness: in proportion to the turn, but never thinner than a few voxels (so the
// first turns are simply solid); 0 when the shell is solid
fn wallAt(s: Spot) -> f32 {
  return select(0.0, max(0.5 * pr.wall * pr.fat * along(s), 0.016), pr.wall > 0.0);
}
fn raw(p: vec3f) -> f32 {
  let s = spot(p);
  let j0 = max(round((s.th - log(along(s) / s.rm) / s.b) / 6.2831853), 0.0);
  // the nearest turn and its neighbours; the last turn stops at the mouth
  var d = 1e9;
  for (var i = -1; i <= 1; i++) {
    let j = j0 + f32(i);
    if (j > 0.0) { d = min(d, turn(s, j)); }
    if (j == 0.0) { d = min(d, max(turn(s, 0.0), beyond(s))); }
  }
  return d;
}
fn shellWall(p: vec3f) -> f32 { return wallAt(spot(p)); }
// What is cut away (positive there), on two counts.
// Anything the equation itself puts well clear of the solid's surface: it overstates distances, by half
// at the very most, so a part of it is a safe floor under them. That keeps stray flecks of wall out of
// the chamber, where a voxel that lost its way to the surface would otherwise leave one.
// And, hollow, the window in the wall across the mouth: everything in front of the back of that wall
// and within the last turn's tube, short of the tube's own wall and of the wall of the turn before
// (which passes through the mouth too) by a little, which is left as the lip.
fn shellCut(p: vec3f) -> f32 {
  let s = spot(p);
  let w = wallAt(s);
  let f = raw(p);
  var cut = select(f, abs(f), pr.wall > 0.0) / 1.5 - w - 0.03;
  if (pr.wall > 0.0) {
    let m = 1.25 * w + 0.03;
    cut = max(cut, min(min(-turn(s, 0.0), turn(s, 1.0)) - m, beyond(s) + m));
  }
  return cut;
}";

/// numbers: growth per turn, tube radius / distance from the axis, spire, wall
fn shell(id: &'static str, name: &'static str, note: &'static str, numbers: [f64; 4], rot: [f64; 3], animate: &'static str) -> Preset {
    form(Form {
        id, name, note, rot, animate,
        eq: &["a tube round the spiral ρ = e^(bθ),", "its radius and its drop along the axis both ∝ ρ"],
        params: vec![
            ParamDef::labelled("growth", "growth per turn", numbers[0], 1.3, 4.0, 0.01),
            ParamDef::labelled("fat", "tube ⁄ radius", numbers[1], 0.3, 0.9, 0.005),
            ParamDef::labelled("spire", "spire", numbers[2], 0.0, 2.5, 0.01),
            ParamDef::labelled("wall", "wall (0 = solid)", numbers[3], 0.0, 0.4, 0.01),
        ],
        wgsl: SHELL.to_string(),
        clip: "shellCut(p)",
        shell: "shellWall(p)",
    })
}

// ---- the family -----------------------------------------------------------------------------------
fn presets() -> Vec<Preset> {
    vec![
        supershape("superformula", "Superformula · starfish", "Johan Gielis’s formula (2003): one expression for starfish, petals, pods and rounded polygons. Here n₂ = n₃.",
            [5.0, 2.0, 7.0], [2.0, 2.0, 2.0], [62.0, 12.0, 0.0], "n2"),
        supershape("superpillow", "Superformula · pillow", "The same formula in both directions, with four lobes each way.",
            [4.0, 3.0, 7.0], [4.0, 3.0, 7.0], [58.0, 20.0, 0.0], "n2"),
        supershape("superpebble", "Superformula · pebble", "Three lobes round the axis and two from pole to pole, with high exponents: almost a polygon, rounded off.",
            [3.0, 4.5, 10.0], [2.0, 4.5, 10.0], [35.0, 30.0, 0.0], "n1"),

        harmonics("harmonics", "Spherical harmonic", "The shapes a drum-skin wrapped round a ball can ring in, and the angular part of an atom’s orbitals. Order m cannot exceed degree l.",
            3.0, 2.0, 0.6, [20.0, 30.0, 0.0]),

        form(Form {
            id: "metaballs", name: "Metaballs",
            eq: &["Σ rᵢ² ⁄ |p − cᵢ|² = 1", "a core, and n balls on a ring round it"],
            note: "Jim Blinn’s blobby molecules (1982): every ball adds to a field, and the surface is where the sum reaches 1. Balls that come near reach out and join.",
            params: vec![
                ParamDef::new("balls", 4.0, 2.0, 8.0, 1.0),
                ParamDef::new("spread", 0.72, 0.35, 0.8, 0.005),
                ParamDef::new("size", 0.32, 0.15, 0.36, 0.005),
                ParamDef::new("core", 0.36, 0.0, 0.5, 0.005),
                ParamDef::new("rise", 0.3, 0.0, 0.4, 0.005),
            ],
            rot: [58.0, 12.0, 0.0],
            animate: "spread",
            wgsl: "
fn raw(p: vec3f) -> f32 {
  let n = i32(clamp(round(pr.balls), 2.0, 8.0));
  var g = pr.core * pr.core / (dot(p, p) + 1e-6);
  for (var i = 0; i < 8; i++) {
    if (i >= n) { break; }
    let a = 6.2831853 * f32(i) / f32(n);
    // alternate balls sit above and below the ring
    let d = p - vec3f(pr.spread * cos(a), pr.rise * (1.0 - 2.0 * f32(i & 1)), pr.spread * sin(a));
    g += pr.size * pr.size / (dot(d, d) + 1e-6);
  }
  // for one ball alone this is its distance: |p - c| - r
  return (inverseSqrt(g) - 1.0) * pr.size;
}".to_string(),
            clip: CLIP_DEFAULT,
            shell: SHELL_DEFAULT,
        }),

        form(Form {
            id: "cassini", name: "Cassini surface",
            eq: &["|p − c₁| · |p − c₂| ··· |p − cₙ| = bⁿ", "the n centres on a circle of radius a"],
            note: "An ellipse keeps the sum of two distances fixed; Cassini’s ovals keep the product. At b = a the lobes meet in a single point; below it they come apart.",
            params: vec![
                ParamDef::labelled("foci", "centres n", 3.0, 2.0, 8.0, 1.0),
                ParamDef::labelled("level", "b / a", 1.06, 0.8, 1.4, 0.005),
            ],
            rot: [62.0, 15.0, 0.0],
            animate: "level",
            wgsl: "
fn raw(p: vec3f) -> f32 {
  let n = i32(clamp(round(pr.foci), 2.0, 8.0));
  let e = 1.0 / f32(n);
  // On the ray through a centre the product of the distances is |r^n - a^n|, so the surface crosses it
  // at (a^n + b^n)^(1/n), its farthest point, and, when the lobes are apart, at (a^n - b^n)^(1/n).
  // The circle of centres is sized to put the farthest point 1.22 from the middle.
  let bn = pow(pr.level, f32(n));
  let a = 1.22 / pow(1.0 + bn, e);
  let far = 1.22;
  let near = a * pow(max(1.0 - bn, 0.0), e);
  var s = 0.0;
  var ball = 1e9;
  for (var i = 0; i < 8; i++) {
    if (i >= n) { break; }
    let t = 6.2831853 * f32(i) / f32(n);
    let c = vec3f(cos(t), 0.0, sin(t));
    s += log(length(p - a * c) + 1e-7);
    ball = min(ball, length(p - 0.5 * (near + far) * c));
  }
  // the geometric mean of the distances, less b
  let f = exp(s * e) - pr.level * a;
  // Far from the surface that has too little slope to walk down (in the middle, between separate lobes,
  // it is level), so there the distance to balls that hold the lobes stands in: one about each lobe
  // while they are apart, one about them all once they have joined.
  let hold = select(length(p) - 1.02 * far - 0.01, ball - 0.525 * (far - near) - 0.01, pr.level < 1.0);
  return max(f, hold);
}".to_string(),
            clip: NO_CLIP,
            shell: SHELL_DEFAULT,
        }),

        form(Form {
            id: "twistedring", name: "Twisted ring",
            eq: &["a regular n-gon carried round a circle,", "turning k⁄n of a turn on the way"],
            note: "With one twist of a square bar, four faces become one: follow a face round and you return on its neighbour.",
            params: vec![
                ParamDef::new("sides", 4.0, 3.0, 8.0, 1.0),
                ParamDef::labelled("twist", "twists k", 1.0, 0.0, 8.0, 1.0),
                ParamDef::labelled("turn", "turn °", 0.0, 0.0, 360.0, 0.5),
                ParamDef::labelled("ring", "ring radius", 0.8, 0.6, 0.85, 0.005),
                ParamDef::labelled("size", "bar size", 0.36, 0.15, 0.42, 0.005),
                ParamDef::labelled("edge", "rounding", 0.02, 0.0, 0.06, 0.002),
            ],
            rot: [58.0, 20.0, 0.0],
            animate: "turn",
            wgsl: ring_wgsl(POLYGON, "round(pr.twist) / clamp(round(pr.sides), 3.0, 12.0)"),
            clip: NO_CLIP,
            shell: SHELL_DEFAULT,
        }),

        form(Form {
            id: "mobius", name: "Möbius band",
            eq: &["a flat bar carried round a circle,", "turning k half-turns on the way"],
            note: "One half-turn leaves a band with a single side and a single edge. Any odd number does; an even number gives two sides again.",
            params: vec![
                ParamDef::labelled("twist", "half-turns k", 1.0, 0.0, 5.0, 1.0),
                ParamDef::labelled("turn", "turn °", 0.0, 0.0, 360.0, 0.5),
                ParamDef::labelled("ring", "ring radius", 0.78, 0.6, 0.82, 0.005),
                ParamDef::labelled("width", "half-width", 0.4, 0.2, 0.42, 0.005),
                ParamDef::labelled("thick", "half-thickness", 0.11, 0.05, 0.2, 0.005),
                ParamDef::labelled("edge", "rounding", 0.06, 0.0, 0.2, 0.005),
            ],
            rot: [55.0, 25.0, 0.0],
            animate: "turn",
            wgsl: ring_wgsl(BAND, "0.5 * round(pr.twist)"),
            clip: NO_CLIP,
            shell: SHELL_DEFAULT,
        }),

        form(Form {
            id: "cruller", name: "Cruller",
            eq: &["a section r = s·(1 + d cos nφ) carried round a circle,", "turning k⁄n of a turn on the way"],
            note: "A ring whose ridges wind round it as they go, like the twisted doughnut it is named after. With n and k sharing no factor there is only one ridge.",
            params: vec![
                ParamDef::labelled("ridges", "ridges n", 5.0, 2.0, 9.0, 1.0),
                ParamDef::labelled("twist", "twists k", 3.0, 0.0, 12.0, 1.0),
                ParamDef::labelled("turn", "turn °", 0.0, 0.0, 360.0, 0.5),
                ParamDef::labelled("ring", "ring radius", 0.8, 0.6, 0.85, 0.005),
                ParamDef::labelled("size", "section", 0.3, 0.15, 0.34, 0.005),
                ParamDef::labelled("depth", "ridge depth", 0.2, 0.0, 0.3, 0.005),
            ],
            rot: [58.0, 20.0, 0.0],
            animate: "turn",
            wgsl: ring_wgsl(RIDGED, "round(pr.twist) / clamp(round(pr.ridges), 2.0, 12.0)"),
            clip: NO_CLIP,
            shell: SHELL_DEFAULT,
        }),

        shell("seashell", "Seashell", "Every turn is the turn before, enlarged: the growth rule of real shells, which D’Arcy Thompson and later David Raup described with a few numbers.",
            [1.5, 0.6, 1.6, 0.15], [12.0, -35.0, 0.0], "spire"),
        shell("nautilus", "Nautilus", "The same rule with no drop along the axis, and each turn three times the size of the last.",
            [3.0, 0.75, 0.0, 0.15], [62.0, -20.0, 0.0], "growth"),

        form(Form {
            id: "julia", name: "Julia set",
            eq: &["zₙ₊₁ = zₙ² + c  in the quaternions", "the surface where 2⁻ⁿ ln|zₙ| = g"],
            note: "Points that escape to infinity under the map carry a potential, as if the set they flee were charged. This is one of its level surfaces, cut through four dimensions; lower g and it closes in on the set.",
            params: vec![
                ParamDef::labelled("cr", "c real", -1.0, -1.3, 0.4, 0.005),
                ParamDef::labelled("ci", "c i", 0.2, -0.8, 0.8, 0.005),
                ParamDef::labelled("cj", "c j", 0.0, -0.8, 0.8, 0.005),
                ParamDef::labelled("ck", "c k", 0.3, -0.6, 0.6, 0.005),
                ParamDef::labelled("level", "g", 0.08, 0.03, 0.4, 0.005),
            ],
            rot: [20.0, 30.0, 0.0],
            animate: "ck",
            wgsl: "
// The potential of the filled Julia set of z -> z² + c at a point of the slice k = 0 of the quaternions:
// 2^-n ln|z_n|, taken once z_n is far enough out for the limit to have settled.
fn raw(p: vec3f) -> f32 {
  let c = vec4f(pr.cr, pr.ci, pr.cj, pr.ck);
  var z = vec4f(1.3 * p, 0.0);
  var pw = 1.0;
  var gone = false;
  for (var i = 0; i < 12; i++) {
    z = vec4f(z.x * z.x - dot(z.yzw, z.yzw), 2.0 * z.x * z.yzw) + c;
    pw *= 0.5;
    if (dot(z, z) > 1e8) { gone = true; break; }
  }
  // A point that has not left after twelve steps is deep inside, where the potential is flat (and what
  // twelve steps make of it is noise): it is given the potential 0 and the faintest slope outward, so
  // that a voxel there still has a way to walk to the surface.
  if (!gone) { return 0.0015 * length(p) - pr.level; }
  return 0.5 * log(dot(z, z)) * pw - pr.level;
}".to_string(),
            clip: CLIP_DEFAULT,
            shell: SHELL_DEFAULT,
        }),

        form(Form {
            id: "cyclide", name: "Dupin cyclide",
            eq: &["a torus turned inside out through a sphere:", "p ↦ c + k²·(p − c) ⁄ |p − c|²"],
            note: "Inversion sends spheres to spheres, so the torus’s two families of circles are still circles. With the centre on the axis the torus comes back unchanged.",
            params: vec![
                ParamDef::labelled("tube", "tube ⁄ ring", 0.42, 0.2, 0.6, 0.005),
                ParamDef::labelled("skew", "centre off axis", 0.55, 0.0, 0.85, 0.005),
            ],
            rot: [50.0, 20.0, 0.0],
            animate: "skew",
            wgsl: "
fn raw(p: vec3f) -> f32 {
  let r = pr.tube;                           // the torus: ring radius 1, tube radius r, a hole of radius g
  let g = 1.0 - r;
  let e = pr.skew * g;                       // the centre of inversion, in the hole, this far from the axis
  // The torus lies outside the ball that fills its hole, so its image lies inside that ball's image:
  // the sphere of inversion is sized, and the result moved, to make that image the ball of radius 1.22.
  let k2 = 1.22 * (g * g - e * e) / g;
  let c = vec3f(e, 0.0, 0.0);
  let d = p + vec3f(k2 * e / (g * g - e * e), 0.0, 0.0);      // p - c, the shape having been centred
  let dd = max(dot(d, d), 1e-9);
  let q = c + d * (k2 / dd);                 // where p comes from
  let axis = normalize(q.xz + vec2f(1e-9, 0.0));
  let mid = vec3f(axis.x, 0.0, axis.y);      // the middle of the tube nearest q
  let dt = length(q - mid);
  let foot = mid + (q - mid) * (r / max(dt, 1e-9));           // the torus's nearest point to q
  // inversion changes the distance between two points by k² over the product of their distances from c
  return (dt - r) * sqrt(dd) / length(foot - c);
}".to_string(),
            clip: NO_CLIP,
            shell: SHELL_DEFAULT,
        }),
    ]
}

pub fn family() -> Family {
    Family {
        step_scale: 0.85,
        layers: 2,
        presets: presets(),
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.30, 0.14, 0.06])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(3.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([24.0, 34.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        // broad soft lights, as the other smooth surfaces have: broad faces of glass want broad reflections
        lights: vec![
            Light { az:   38.0, el:  18.0, w: 14.0, h: 32.0, rgb: [5.0, 4.9, 4.7],   soft: 0.7 },
            Light { az:  -55.0, el:  -8.0, w: 10.0, h: 42.0, rgb: [0.5, 1.4, 4.0],   soft: 0.7 },
            Light { az: -140.0, el:   5.0, w:  8.0, h: 42.0, rgb: [5.0, 1.6, 0.25],  soft: 0.7 },
            Light { az:  150.0, el:  28.0, w: 24.0, h: 16.0, rgb: [3.2, 3.1, 3.0],   soft: 0.8 },
            Light { az:    0.0, el:  85.0, w: 34.0, h: 34.0, rgb: [0.8, 0.9, 1.1],   soft: 0.9 },
            Light { az:    0.0, el: -80.0, w: 30.0, h: 30.0, rgb: [0.5, 0.22, 0.05], soft: 0.9 },
        ],
        ..Family::new("forms", "Mathematical forms", Kind::Field)
    }
}
