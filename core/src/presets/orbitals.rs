//! Hydrogen orbitals: the surfaces on which the electron's probability density |ψ|² takes one value.
//!
//! ψ(n, l, m) = R(n, l) · Y(l, m), in units of the Bohr radius:
//!   R = ρ^l · L(n−l−1, 2l+1; ρ) · e^(−ρ/2),  ρ = 2r/n          (L: the associated Laguerre polynomial)
//!   Y = the real spherical harmonics, the chemists' p, d and f orbitals, written as polynomials in x, y, z.
//!
//! The glass is where |ψ|² is above the iso level, given as a fraction of the density's own peak (on a log
//! scale: 10^level). Lobes of either sign are one body; the nodes, where ψ changes sign (radial shells,
//! planes, cones), are gaps between them.
//!
//! `raw` is log|ψ|² measured down from the level: its slope is even far out (the density falls off as
//! e^(−2r/n), so its logarithm is a straight ramp) and it is divided by that slope, so far from the glass it
//! reads about as a distance. It is not a distance (the bake walks every voxel to the surface). At a node
//! log|ψ|² would fall to minus infinity; it is floored a hundredth of the level below the surface, so a node
//! is a smooth ridge between lobes, not a well. The whole orbital is scaled so that its farthest point, at
//! the level asked for, reaches FIT from the centre whatever n, l, m and the level are: the scale is worked
//! out once per voxel (`orb_setup`) from the radial density alone, read from a table made here on the CPU
//! (for each n and l, how far out in ρ the density falls to each level), between levels a quarter apart.
//!
//! The orbital is turned a few degrees inside the formula: a nodal plane through the centre of every voxel
//! (x = y for d x²−y², say) would leave the walk a level ridge with no slope at all, read as a surface.
use super::solids::FIT;
use super::*;

/// The real spherical harmonic (l, m) as a homogeneous polynomial in c, in the order l·l + l + m.
fn y_poly(c: [f64; 3], i: usize) -> f64 {
    let [x, y, z] = c;
    let r2 = x * x + y * y + z * z;
    match i {
        0 => 1.0,
        1 => y,
        2 => z,
        3 => x,
        4 => x * y,
        5 => y * z,
        6 => 3.0 * z * z - r2,
        7 => x * z,
        8 => x * x - y * y,
        9 => y * (3.0 * x * x - y * y),
        10 => x * y * z,
        11 => y * (5.0 * z * z - r2),
        12 => z * (5.0 * z * z - 3.0 * r2),
        13 => x * (5.0 * z * z - r2),
        14 => z * (x * x - y * y),
        _ => x * (x * x - 3.0 * y * y),
    }
}

/// The largest |Y| on the unit sphere, for each of the sixteen harmonics.
fn y_max() -> [f64; 16] {
    static C: std::sync::OnceLock<[f64; 16]> = std::sync::OnceLock::new();
    *C.get_or_init(y_max_worked)
}
fn y_max_worked() -> [f64; 16] {
    let mut most = [0.0f64; 16];
    let (nt, np) = (360, 720);
    for a in 0..=nt {
        let th = std::f64::consts::PI * a as f64 / nt as f64;
        for b in 0..np {
            let ph = 2.0 * std::f64::consts::PI * b as f64 / np as f64;
            let c = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
            for (i, m) in most.iter_mut().enumerate() {
                *m = m.max(y_poly(c, i).abs());
            }
        }
    }
    most
}

const LEV_LO: f64 = -6.0;
const LEV_HI: f64 = -0.3;
const LEV_STEP: f64 = 0.25;

/// The associated Laguerre polynomial L(k, a; x).
fn lag(k: usize, a: f64, x: f64) -> f64 {
    if k == 0 { return 1.0; }
    let (mut l0, mut l1) = (1.0, 1.0 + a - x);
    for i in 1..k {
        let l2 = ((2.0 * i as f64 + 1.0 + a - x) * l1 - (i as f64 + a) * l0) / (i + 1) as f64;
        l0 = l1;
        l1 = l2;
    }
    l1
}

/// The radial log density 2 ln|rho^l L| - rho.
fn g(n: usize, l: usize, rho: f64) -> f64 {
    2.0 * (rho.powi(l as i32) * lag(n - l - 1, 2.0 * l as f64 + 1.0, rho)).abs().max(1e-300).ln() - rho
}

/// For each (n, l), n up to 4, in the order n(n-1)/2 + l: the peak of the radial log density, and for each
/// level from LEV_LO to LEV_HI by LEV_STEP the largest rho at which the density is still that far below it.
pub fn radial_tables() -> (Vec<f64>, Vec<f64>, usize) {
    static C: std::sync::OnceLock<(Vec<f64>, Vec<f64>, usize)> = std::sync::OnceLock::new();
    C.get_or_init(radial_tables_worked).clone()
}
fn radial_tables_worked() -> (Vec<f64>, Vec<f64>, usize) {
    let levels = ((LEV_HI - LEV_LO) / LEV_STEP).round() as usize + 1;
    let (mut peaks, mut reach) = (Vec::new(), Vec::new());
    for n in 1..=4usize {
        for l in 0..n {
            let step = 0.002;
            let gm = (0..30000).map(|i| g(n, l, i as f64 * step)).fold(f64::MIN, f64::max);
            peaks.push(gm);
            for j in 0..levels {
                let thr = gm + (LEV_LO + j as f64 * LEV_STEP) * std::f64::consts::LN_10;
                let mut i = 30000;
                while i > 0 && g(n, l, i as f64 * step) < thr { i -= 1; }
                let (mut lo, mut hi) = (i as f64 * step, (i + 1) as f64 * step);
                for _ in 0..30 {
                    let mid = 0.5 * (lo + hi);
                    if g(n, l, mid) >= thr { lo = mid } else { hi = mid }
                }
                reach.push(lo);
            }
        }
    }
    (peaks, reach, levels)
}

/// For each (n, l): the lowest of the radial log density's local peaks (its faintest shell). A level above
/// it would leave that shell a hollow of the formula that never reaches the surface, which the walk to the
/// surface reads as a thin false sheet.
pub fn radial_lows() -> Vec<f64> {
    static C: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    C.get_or_init(radial_lows_worked).clone()
}
fn radial_lows_worked() -> Vec<f64> {
    let mut out = Vec::new();
    for n in 1..=4usize {
        for l in 0..n {
            let step = 0.002;
            let v: Vec<f64> = (0..30000).map(|i| g(n, l, i as f64 * step)).collect();
            let mut low = f64::MAX;
            for i in 0..v.len() - 1 {
                let left = if i == 0 { f64::MIN } else { v[i - 1] };
                if v[i] >= left && v[i] >= v[i + 1] && v[i] > -600.0 { low = low.min(v[i]); }
            }
            out.push(low);
        }
    }
    out
}

/// For each harmonic: the lowest of |Y|'s local peaks on the unit sphere (its faintest lobe, or ring).
pub fn y_lows() -> [f64; 16] {
    static C: std::sync::OnceLock<[f64; 16]> = std::sync::OnceLock::new();
    *C.get_or_init(y_lows_worked)
}
fn y_lows_worked() -> [f64; 16] {
    let (nt, np) = (180usize, 360usize);
    let mut out = [0.0f64; 16];
    for (i, o) in out.iter_mut().enumerate() {
        let at = |a: usize, b: usize| {
            let th = std::f64::consts::PI * (a as f64 + 0.5) / nt as f64;
            let ph = 2.0 * std::f64::consts::PI * b as f64 / np as f64;
            y_poly([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], i).abs()
        };
        let mut low = f64::MAX;
        for a in 0..nt {
            for b in 0..np {
                let v = at(a, b);
                if v < 1e-6 { continue; }
                let mut peak = true;
                for (da, db) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                    let aa = a as i64 + da;
                    if aa < 0 || aa >= nt as i64 { continue; }
                    let bb = (b as i64 + db).rem_euclid(np as i64) as usize;
                    if at(aa as usize, bb) > v { peak = false; break; }
                }
                if peak { low = low.min(v); }
            }
        }
        *o = low;
    }
    out
}

/// How high the level may go for (n, l, m) (log10 of the share of the peak): a little below the faintest
/// lobe's own peak, so that every lobe is still there; never above LEV_HI.
pub fn level_cap(n: usize, l: usize, m: i64) -> f64 {
    let row = n * (n - 1) / 2 + l;
    let i = (l * l + l) as i64 + m;
    let (peaks, _, _) = radial_tables();
    let rad = radial_lows()[row] - peaks[row];
    let ang = 2.0 * (y_lows()[i as usize] / y_max()[i as usize]).ln();
    (((rad + ang) / std::f64::consts::LN_10) - CAP_MARGIN).min(LEV_HI)
}
/// How low the level may go for (n, l): where the thinnest radial shell or gap between shells is still
/// FLOOR_VOXELS voxels of the full volume (768 across 3 units) along the lobes' axis. Lower, a node's gap
/// closes to less than a voxel: the volume cannot hold it and the surface there turns rough.
pub fn level_floor(n: usize, l: usize) -> f64 {
    static C: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let mut out = Vec::new();
        let step = 0.002;
        for n in 1..=4usize {
            for l in 0..n {
                let v: Vec<f64> = (0..30000).map(|i| g(n, l, i as f64 * step)).collect();
                let gm = v.iter().cloned().fold(f64::MIN, f64::max);
                let top = ((radial_lows()[n * (n - 1) / 2 + l] - gm) / std::f64::consts::LN_10 - CAP_MARGIN).min(LEV_HI);
                let mut floor = LEV_LO;
                let mut lv = LEV_LO;
                while lv <= top + 1e-9 {
                    let thr = gm + lv * std::f64::consts::LN_10;
                    // the edges of the stretches of rho inside the glass
                    let mut edges = Vec::new();
                    if v[0] >= thr { edges.push(0.0); }
                    for i in 0..v.len() - 1 {
                        if (v[i] >= thr) != (v[i + 1] >= thr) { edges.push((i as f64 + 0.5) * step); }
                    }
                    let reach = *edges.last().unwrap();
                    let voxel = reach / FIT * 3.0 / 768.0;
                    let thinnest = edges.windows(2).map(|w| w[1] - w[0]).fold(f64::MAX, f64::min) / voxel;
                    if thinnest < FLOOR_VOXELS { floor = lv + 0.05; }
                    lv += 0.05;
                }
                out.push((floor * 100.0).round() / 100.0);
            }
        }
        out
    })[n * (n - 1) / 2 + l]
}
const FLOOR_VOXELS: f64 = 2.5;

/// How far (in log10) below the faintest lobe's peak the level stops.
const CAP_MARGIN: f64 = 0.2;

/// A small turn, off every plane of the voxel grid: rows of the matrix (about x by 7°, then y by 11°).
fn turn() -> [[f64; 3]; 3] {
    let (a, b) = (7f64.to_radians(), 11f64.to_radians());
    let rx = [[1.0, 0.0, 0.0], [0.0, a.cos(), -a.sin()], [0.0, a.sin(), a.cos()]];
    let ry = [[b.cos(), 0.0, b.sin()], [0.0, 1.0, 0.0], [-b.sin(), 0.0, b.cos()]];
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| ry[i][k] * rx[k][j]).sum();
        }
    }
    m
}

const BODY: &str = "
// the associated Laguerre polynomial L(k, a; x), by its three-term recurrence
fn orb_lag(k: i32, a: f32, x: f32) -> f32 {
  if (k == 0) { return 1.0; }
  var l0 = 1.0;
  var l1 = 1.0 + a - x;
  for (var i = 1; i < k; i++) {
    let l2 = ((2.0 * f32(i) + 1.0 + a - x) * l1 - (f32(i) + a) * l0) / f32(i + 1);
    l0 = l1;
    l1 = l2;
  }
  return l1;
}
// the real spherical harmonic number i (= l·l + l + m), as a homogeneous polynomial in c
fn orb_y(c: vec3f, i: i32) -> f32 {
  let x = c.x;
  let y = c.y;
  let z = c.z;
  let r2 = dot(c, c);
  switch (i) {
    case 0: { return 1.0; }
    case 1: { return y; }
    case 2: { return z; }
    case 3: { return x; }
    case 4: { return x * y; }
    case 5: { return y * z; }
    case 6: { return 3.0 * z * z - r2; }
    case 7: { return x * z; }
    case 8: { return x * x - y * y; }
    case 9: { return y * (3.0 * x * x - y * y); }
    case 10: { return x * y * z; }
    case 11: { return y * (5.0 * z * z - r2); }
    case 12: { return z * (5.0 * z * z - 3.0 * r2); }
    case 13: { return x * (5.0 * z * z - r2); }
    case 14: { return z * (x * x - y * y); }
    default: { return x * (x * x - 3.0 * y * y); }
  }
}
// What depends on the sliders alone, worked out once for each voxel and kept.
var<private> orb_ready: bool = false;
var<private> orb_n: f32;
var<private> orb_l: f32;
var<private> orb_i: i32;
var<private> orb_k: i32;
var<private> orb_a: f32;
var<private> orb_s: f32;      // Bohr radii per unit of the volume
var<private> orb_t: f32;      // log|psi|^2 at the surface
var<private> orb_slope: f32;  // how fast log|psi|^2 falls far out, per unit of the volume
var<private> orb_f: f32;      // (2/n)^l
var<private> orb_r: f32;      // rho at the farthest point
fn orb_setup() {
  let q = orb_nlm();
  let n = clamp(round(q.x), 1.0, 4.0);
  let l = clamp(round(q.y), 0.0, n - 1.0);
  let m = clamp(round(q.z), -l, l);
  orb_n = n;
  orb_l = l;
  orb_i = i32(l * l + l + m);
  orb_k = i32(n - l - 1.0);
  orb_a = 2.0 * l + 1.0;
  var amax = array<f32, 16>(YMAX);
  // the radial density's peak, and how far out (in rho) it falls to the level: tables made on the CPU
  // (orbitals.rs), one row of levels for each (n, l)
  var peak = array<f32, 10>(PEAK);
  var reach = array<f32, REACHN>(REACH);
  let row = i32(n * (n - 1.0) * 0.5 + l);
  // the level, held below the faintest lobe's own peak (radial and angular tables made on the CPU)
  var radlow = array<f32, 10>(RADLOW);
  var anglow = array<f32, 16>(ANGLOW);
  let cap = min((radlow[row] - peak[row] + 2.0 * log(anglow[orb_i] / amax[orb_i])) / 2.3025851 - CAPM, LEVHI);
  var lows = array<f32, 10>(FLOORS);
  let lv = clamp(pr.level, min(lows[row], cap), cap);
  let at = (lv - LEVLO) / LEVSTEP;
  let j = min(i32(floor(at)), LEVN - 2);
  let base = REACHBASE;
  let rr = mix(reach[base + j], reach[base + j + 1], at - f32(j));
  orb_r = rr;
  orb_s = rr * 0.5 * n / FIT;
  orb_t = peak[row] + lv * 2.3025851 + 2.0 * log(amax[orb_i]);
  orb_slope = 2.0 * orb_s / n;
  orb_f = pow(2.0 / n, l);
  orb_ready = true;
}
fn raw(p: vec3f) -> f32 {
  if (!orb_ready) { orb_setup(); }
  let t = ORB_TURN * p;
  // the chemists' z is up
  let c = vec3f(t.x, t.z, t.y) * orb_s;
  let rho = 2.0 * length(c) / orb_n;
  let v = orb_f * orb_y(c, orb_i) * orb_lag(orb_k, orb_a, rho);
  let lp = 2.0 * log(max(abs(v), 1e-30)) - rho;
  // Floored, smoothly, a hundredth of the level below the surface out to the farthest point, and falling
  // as the density does beyond it: a node is a ridge, not a well, and the far field keeps its slope.
  let past = rho - orb_r;
  let lb = orb_t - 4.6051702 - (max(past, 0.0) + log(1.0 + exp(-abs(past))));
  let ls = max(lp, lb) + log(1.0 + exp(-abs(lp - lb)));
  return (orb_t - ls) / orb_slope;
}
";

/// The formula for the orbital `nlm` (WGSL giving n, l, m); `nl`: its n and l when they are fixed.
fn wgsl(nlm: &str, nl: Option<(usize, usize)>) -> String {
    let m = turn();
    let ymax: Vec<String> = y_max().iter().map(|v| format!("{v:.7}")).collect();
    let mat = format!(
        "mat3x3f({:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9})",
        // (WGSL takes columns)
        m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]
    );
    let (peaks, reach, levels) = radial_tables();
    let f = |v: &[f64]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(", ");
    let body = BODY.replace("YMAX", &ymax.join(", ")).replace("PEAK", &f(&peaks))
        .replace("RADLOW", &f(&radial_lows())).replace("ANGLOW", &f(&y_lows())).replace("CAPM", &format!("{CAP_MARGIN:.2}"))
        .replace("FLOORS", &(1..=4usize).flat_map(|n| (0..n).map(move |l| format!("{:.2}", level_floor(n, l)))).collect::<Vec<_>>().join(", "));
    let (reach, base) = match nl {
        Some((n, l)) => { let row = n * (n - 1) / 2 + l; (reach[row * levels..(row + 1) * levels].to_vec(), "0".to_string()) }
        None => (reach, format!("row * {levels}")),
    };
    let body = body.replace("REACHN", &reach.len().to_string()).replace("REACHBASE", &base)
        .replace("REACH", &f(&reach)).replace("LEVLO", &format!("{LEV_LO:.2}")).replace("LEVHI", &format!("{LEV_HI:.2}"))
        .replace("LEVSTEP", &format!("{LEV_STEP:.3}")).replace("LEVN", &levels.to_string());
    format!(
        "const FIT: f32 = {FIT:.4};\nconst ORB_TURN = {mat};\nfn orb_nlm() -> vec3f {{ return {nlm}; }}\n{}",
        body
    )
}

fn level(v: f64) -> ParamDef {
    ParamDef::labelled("level", "iso level (log₁₀)", v, LEV_LO, LEV_HI, 0.01)
}
/// The level slider of one orbital: from its floor up to its cap (in whole steps of the slider), so that no
/// gap closes below a voxel and no lobe vanishes.
fn level_to(v: f64, lo: f64, cap: f64) -> ParamDef {
    let hi = (cap * 100.0).floor() / 100.0;
    let lo = lo.min(hi - 0.1);
    ParamDef::labelled("level", "iso level (log₁₀)", v.clamp(lo, hi), lo, hi, 0.01)
}

/// One orbital: n, l, m, the name the chemists give it, its starting level and orientation.
fn orbital(id: &'static str, name: &'static str, nlm: [i32; 3], eq: &'static str, lv: f64, rot: [f64; 3], note: &'static str) -> Preset {
    Preset {
        eq: vec![eq, "|ψ|² = 10^level × its peak"],
        note,
        params: vec![level_to(lv, level_floor(nlm[0] as usize, nlm[1] as usize), level_cap(nlm[0] as usize, nlm[1] as usize, nlm[2] as i64))],
        rot: Some(rot),
        animate: Some(vec!["level"]),
        ..Preset::new(id, name, Shape::Field {
            wgsl: wgsl(&format!("vec3f({:.1}, {:.1}, {:.1})", nlm[0], nlm[1], nlm[2]), Some((nlm[0] as usize, nlm[1] as usize))),
            clip: "-1.0".to_string(),
            shell: "0.0".to_string(),
        })
    }
}

pub fn family() -> Family {
    let side = [18.0, 28.0, 0.0];
    let top = [52.0, 20.0, 0.0];
    let mut presets = vec![
        orbital("orb1s", "1s", [1, 0, 0], "ψ ∝ e^(−r)", -1.5, side,
            "The ground state of hydrogen: a ball, densest at the nucleus and fading outward without edge."),
        orbital("orb2s", "2s", [2, 0, 0], "ψ ∝ (2 − r) e^(−r/2)", -2.6, side,
            "One radial node: a ball inside a shell, with a sphere of nothing between them where ψ changes sign."),
        orbital("orb2p", "2p", [2, 1, 0], "ψ ∝ z e^(−r/2)", -1.6, side,
            "The dumbbell: two lobes of opposite sign either side of a nodal plane."),
        orbital("orb3s", "3s", [3, 0, 0], "ψ ∝ (27 − 18r + 2r²) e^(−r/3)", -2.65, side,
            "Two radial nodes: a ball in two nested shells."),
        orbital("orb3p", "3p", [3, 1, 0], "ψ ∝ z (6 − r) e^(−r/3)", -2.2, side,
            "A dumbbell with a radial node: each lobe is cupped by a larger one beyond it."),
        orbital("orb3dz2", "3d z²", [3, 2, 0], "ψ ∝ (3z² − r²) e^(−r/3)", -1.8, side,
            "Two lobes along the axis and a ring about the waist, parted by two nodal cones."),
        orbital("orb3dxy", "3d xy", [3, 2, -2], "ψ ∝ xy e^(−r/3)", -1.8, top,
            "A four-leaf clover between two nodal planes."),
        orbital("orb4s", "4s", [4, 0, 0], "ψ ∝ (192 − 144r + 24r² − r³) e^(−r/4)", -2.97, side,
            "Three radial nodes: a ball in three nested shells, each fainter than the last."),
        orbital("orb4p", "4p", [4, 1, 0], "ψ ∝ z (80 − 20r + r²) e^(−r/4)", -2.2, side,
            "Two radial nodes: a dumbbell inside two cupped dumbbells."),
        orbital("orb4dz2", "4d z²", [4, 2, 0], "ψ ∝ (3z² − r²)(12 − r) e^(−r/4)", -2.0, side,
            "The 3d z² with a radial node: every lobe and the ring echoed further out."),
        orbital("orb4dx2y2", "4d x²−y²", [4, 2, 2], "ψ ∝ (x² − y²)(12 − r) e^(−r/4)", -2.0, top,
            "A clover along the axes, with its echo beyond a radial node."),
        orbital("orb4fz3", "4f z³", [4, 3, 0], "ψ ∝ z(5z² − 3r²) e^(−r/4)", -1.6, side,
            "Two lobes along the axis and two rings, parted by a nodal plane and two cones."),
        orbital("orb4fxyz", "4f xyz", [4, 3, -2], "ψ ∝ xyz e^(−r/4)", -1.6, [30.0, 35.0, 10.0],
            "Eight lobes, one in each octant, between three nodal planes."),
        orbital("orb4fy3", "4f y(3x²−y²)", [4, 3, -3], "ψ ∝ y(3x² − y²) e^(−r/4)", -1.6, top,
            "Six lobes round the axis, like petals, between three nodal planes."),
        orbital("orb4fxz2", "4f xz²", [4, 3, 1], "ψ ∝ x(5z² − r²) e^(−r/4)", -1.6, [24.0, 30.0, 0.0],
            "Six lobes: two pairs leaning off the axis and a pair across the waist."),
    ];
    presets.push(Preset {
        eq: vec!["ψ = ρ^l L(n−l−1, 2l+1; ρ) e^(−ρ/2) · Y(l, m),  ρ = 2r/n", "|ψ|² = 10^level × its peak"],
        note: "Any orbital up to n = 4: l runs to n − 1 and m from −l to l (the real harmonics: m < 0 are the sine forms).",
        params: vec![
            level(-2.0),
            ParamDef::new("n", 4.0, 1.0, 4.0, 1.0),
            ParamDef::new("l", 2.0, 0.0, 3.0, 1.0),
            ParamDef::new("m", 1.0, -3.0, 3.0, 1.0),
        ],
        rot: Some([22.0, 30.0, 0.0]),
        animate: Some(vec!["level"]),
        ..Preset::new("orbital", "Any orbital", Shape::Field {
            wgsl: wgsl("vec3f(pr.n, pr.l, pr.m)", None),
            clip: "-1.0".to_string(),
            shell: "0.0".to_string(),
        })
    });
    Family {
        step_scale: 0.9,
        layers: 2,
        presets,
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.22, 0.12, 0.05])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(3.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([20.0, 30.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        lights: surface_lights(),
        ..Family::new("orbitals", "Hydrogen orbitals", Kind::Field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The faintest lobe of d z² is its ring, a quarter of the axis lobes' density: the level stops 0.2 below
    /// log10(1/4). The p and 3d xy orbitals have no fainter lobe and keep the whole range; s orbitals'
    /// outer shells are faint and stop it low. Every preset starts below its own cap.
    #[test]
    fn levels_stop_below_the_faintest_lobe() {
        assert!((level_cap(3, 2, 0) - (0.25f64.log10() - CAP_MARGIN)).abs() < 0.01, "{}", level_cap(3, 2, 0));
        assert_eq!(level_cap(2, 1, 0), LEV_HI);
        assert_eq!(level_cap(3, 2, -2), LEV_HI);
        assert!(level_cap(4, 0, 0) < -2.5);
        assert_eq!(level_floor(2, 1), LEV_LO);
        assert!(level_floor(4, 0) > -3.6 && level_floor(4, 0) < level_cap(4, 0, 0), "{}", level_floor(4, 0));
        for p in family().presets {
            let d = &p.params[0];
            assert!(d.lo < d.value && d.value < d.hi && d.hi <= LEV_HI && d.hi > -3.5, "{}: {} in {}..{}", p.id, d.value, d.lo, d.hi);
        }
    }
}
