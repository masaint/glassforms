//! A part's size along each of its own axes (csg_parity_design.md 2.1, steps 2a and 2b).
//!
//! A part stands at `pos + size · R · S · q`, S = diag(stretch). Where its shape has a size of its own
//! along an axis (a block's width, a disc's radius, across x and z alike), the stretch is taken into that
//! size (`route`): the shape stays exact and its edges keep their own radius. A sphere stretched unevenly
//! becomes the ellipsoid, a disc or cylinder stretched unevenly across its axis the oval: both measured
//! exactly (`ELL_WGSL`). What is left over is a true stretch, read by σ·f(q), σ the part's least size along
//! its axes: never more than the distance, its surface the stretched shape's own, and its normal (its own
//! slope carried back by the inverse transpose of its placement: gf_core::csg::with_stretch) the true one.
//! The volume the tree is measured into is then made a distance again (app/src/engine.rs), so the march
//! is not slowed. A seam is shaped by the readings it joins: on a stretched part it is wider along the
//! stretch, by as much. (A tangent-plane estimate near the surface, σ·f / |S'⁻¹ n|, kept seams to their
//! radius but cost 3 to 8 ms a frame at 4K: csg_scale_log.md.)

use crate::presets::{Family, Preset};

/// The most one axis may be stretched against another.
pub const MOST: f64 = 16.0;

/// How far a studio shape may reach from its middle before it is shrunk as a whole to fit (`FIT` of the
/// studio shapes' text; not crate::FIT).
const STUDIO_FIT: f64 = 1.27;

/// A stretch as it is used: finite, positive, and no axis less than 1 / MOST of the largest.
pub fn held(s: [f64; 3]) -> [f64; 3] {
    let s = s.map(|v| if v.is_finite() { v.abs().clamp(1e-3, 1e3) } else { 1.0 });
    let top = s[0].max(s[1]).max(s[2]);
    s.map(|v| v.max(top / MOST))
}

/// Is this more than a size alike along all three axes?
pub fn uneven(s: [f64; 3]) -> bool {
    let (lo, hi) = (s[0].min(s[1]).min(s[2]), s[0].max(s[1]).max(s[2]));
    hi > lo * (1.0 + 1e-9)
}

/// Axes as bits: 1 x, 2 y, 4 z.
const X: u8 = 1;
const Y: u8 = 2;
const Z: u8 = 4;

/// Which of a studio shape's sliders are its size along which of its axes: a group is taken into the shape
/// when the axes it names are stretched alike.
fn axes_of(id: &str) -> Option<&'static [(u8, &'static [&'static str])]> {
    Some(match id {
        "slab" | "block" | "rectframe" => &[(X, &["width"]), (Y, &["height"]), (Z, &["depth"])],
        "bar" | "wedge" => &[(X, &["length"]), (Y, &["height"]), (Z, &["depth"])],
        "washer" | "bandring" => &[(Y, &["height"]), (X | Z, &["outer", "hole"])],
        "tube" => &[(X, &["length"]), (Y | Z, &["outer", "hole"])],
        "disc" | "cylinder" => &[(Y, &["height"]), (X | Z, &["radius"])],
        "capsule" => &[(X, &["length"]), (Y | Z, &["radius"])],
        "cone" => &[(Y, &["height"]), (X | Z, &["radius", "top"])],
        "ellipsoid" => &[(X, &["rx"]), (Y, &["ry"]), (Z, &["rz"])],
        "oval" => &[(X, &["rx"]), (Y, &["height"]), (Z, &["rz"])],
        _ => return None,
    })
}

/// How far a studio shape reaches from its middle with these sliders (its text's `reach()`).
fn reach(p: &Preset, v: &[f64]) -> Option<f64> {
    let g = |id: &str| p.param_index(id).and_then(|i| v.get(i).copied());
    let half = |a: f64, b: f64, c: f64| 0.5 * (a * a + b * b + c * c).sqrt();
    Some(match p.id {
        "slab" | "block" | "rectframe" => half(g("width")?, g("height")?, g("depth")?),
        "bar" | "wedge" => half(g("length")?, g("height")?, g("depth")?),
        "washer" | "bandring" => g("outer")?.hypot(0.5 * g("height")?),
        "tube" => g("outer")?.hypot(0.5 * g("length")?),
        "disc" | "cylinder" => g("radius")?.hypot(0.5 * g("height")?),
        "capsule" => (0.5 * g("length")?).max(g("radius")?),
        "cone" => g("radius")?.max(g("top")?).hypot(0.5 * g("height")?),
        "sphere" => g("radius")?,
        "ellipsoid" => g("rx")?.max(g("ry")?).max(g("rz")?),
        "oval" => g("rx")?.max(g("rz")?).hypot(0.5 * g("height")?),
        _ => return None,
    })
}

/// A studio shape's sliders that are not lengths (a share, an angle): left as they are when the shape is
/// scaled.
fn not_length(id: &str) -> bool {
    matches!(id, "lean" | "cut" | "arc" | "turns")
}

/// What a stretch makes of a part: its shape (perhaps another of its family's), its sliders, and the
/// stretch left over, a gain alike on all three axes included (`gain`: the shape is held within the
/// volume by shrinking it, and a shape grown by its sizes is shrunk where the stretched one was not: it is
/// drawn smaller in its own frame and the frame made larger by this, so that it stands as large as before).
#[derive(Clone, Debug, PartialEq)]
pub struct Routed {
    pub preset: usize,
    pub vals: Vec<f64>,
    pub stretch: [f64; 3],
    pub gain: f64,
}

/// Take what can be of the stretch `s` into the sizes of the shape `preset` of `fam` (sliders `vals`).
pub fn route(fam: &Family, preset: usize, vals: &[f64], s: [f64; 3]) -> Routed {
    let s = held(s);
    let left = Routed { preset, vals: vals.to_vec(), stretch: s, gain: 1.0 };
    if fam.id != "studio" || !uneven(s) { return left; }
    let p = &fam.presets[preset];
    let get = |id: &str| p.param_index(id).and_then(|i| vals.get(i).copied());
    let find = |id: &str| fam.presets.iter().position(|q| q.id == id);
    // a whole ball stretched unevenly is an ellipsoid; a disc or cylinder stretched unevenly across its
    // axis, an oval
    let (to, mut v) = match p.id {
        "sphere" if get("cut") == Some(0.0) => match (find("ellipsoid"), get("radius")) {
            (Some(e), Some(r)) => (e, vec![r, r, r]),
            _ => return left,
        },
        "disc" | "cylinder" if s[0] != s[2] => match (find("oval"), get("radius"), get("height"), get("bevel"), get("round")) {
            (Some(o), Some(r), Some(h), Some(b), Some(k)) => (o, vec![r, r, h, b, k]),
            _ => return left,
        },
        _ => (preset, vals.to_vec()),
    };
    let q = &fam.presets[to];
    let Some(groups) = axes_of(q.id) else { return left };
    let mut rest = s;
    for (mask, ids) in groups {
        let axes: Vec<usize> = (0..3).filter(|k| mask & (1 << k) != 0).collect();
        let f = s[axes[0]];
        if axes.iter().any(|&k| s[k] != f) { continue; }
        for id in ids.iter() {
            if let Some(i) = q.param_index(id) { v[i] *= f; }
        }
        for &k in &axes { rest[k] = 1.0; }
    }
    if rest == s && to == preset { return left; }
    // held within the volume as the shape was: its own fit, as it stood, kept by the gain
    let (Some(r0), Some(r1)) = (reach(p, vals), reach(q, &v)) else { return left };
    let fit0 = (STUDIO_FIT / r0).min(1.0);
    let k = if r1 > STUDIO_FIT { STUDIO_FIT / r1 * (1.0 - 1e-6) } else { 1.0 };
    if k != 1.0 {
        for (i, d) in q.params.iter().enumerate() {
            if !not_length(d.id) { v[i] *= k; }
        }
    }
    let gain = fit0 / k;
    Routed { preset: to, vals: v, stretch: rest.map(|r| r * gain), gain }
}

/// The ellipse and ellipsoid readings the studio's oval and ellipsoid are made of (exact, both signs).
///
/// The nearest point of the ellipsoid of half-axes e to y = |p| is x_i = e_i² y_i / (t + e_i²), t the one
/// root right of −e_min² of  S(t) = Σ (e_i y_i / (t + e_i²))² = 1.  The steps solve H(t) = S^(-1/2) − 1 = 0
/// instead: H rises and bends downwards there (Cauchy-Schwarz), so Newton's steps from a point left of the
/// root climb to it and never pass it, and where one term rules (deep inside near the middle plane across
/// the least axis, where the root nears its pole) H is a straight line, met in one step. They are taken in
/// τ = t + e_min², which keeps its digits near that pole. Three starts are left of the root, and the steps
/// start from the furthest right of them: where the largest term alone is 1 (no term can be more at the
/// root); where every denominator were the largest; outside, t = 0; and where the terms off the pole
/// alone come to more than 1 at the pole, the first Newton step of those terms from there (they fall and
/// bend upwards: their tangent meets 1 left of where they do, so left of the root). The distance is
/// t·|y / (t + e²)|,
/// signed as t is, its terms along the least axis (and any as short) taken from the root's own equation:
/// together they are √(1 − the other terms of S) / e_min there, which f32 holds where t + e_min² is too
/// small to (a point near that middle plane whose nearest point is off it).
pub const ELL_WGSL: &str = r#"
fn ell3(p: vec3f, e: vec3f) -> f32 {
  let lo = min(e.x, min(e.y, e.z));
  let y = max(abs(p), vec3f(1e-6 * lo));
  let dl = e * e - lo * lo;
  let ey = e * y;
  let one = ey - dl;
  var u = max(max(one.x, max(one.y, one.z)), length(ey) - max(dl.x, max(dl.y, dl.z)));
  if (dot(y / e, y / e) > 1.0) { u = max(u, lo * lo); }
  let pole = dl <= vec3f(1e-4 * lo * lo);
  let k0 = select(1.0 / max(dl, vec3f(1e-30)), vec3f(0.0), pole);
  let w0 = ey * k0;
  let s0 = dot(w0, w0);
  if (s0 > 1.0) { u = max(u, (s0 - 1.0) / (2.0 * dot(w0 * w0, k0))); }
  for (var n = 0; n < 8; n++) {
    let k = 1.0 / (u + dl);
    let w = ey * k;
    let s = dot(w, w);
    u += 2.0 * s * (sqrt(s) - 1.0) / max(2.0 * dot(w * w, k), 1e-30);
  }
  let k = 1.0 / (u + dl);
  let w = select(ey * k, vec3f(0.0), pole);
  let r = select(y * k, vec3f(0.0), pole);
  return (u - lo * lo) * sqrt(dot(r, r) + max(1.0 - dot(w, w), 0.0) / (lo * lo));
}
fn ell2(p: vec2f, e: vec2f) -> f32 {
  let lo = min(e.x, e.y);
  let y = max(abs(p), vec2f(1e-6 * lo));
  let dl = e * e - lo * lo;
  let ey = e * y;
  let one = ey - dl;
  var u = max(max(one.x, one.y), length(ey) - max(dl.x, dl.y));
  if (dot(y / e, y / e) > 1.0) { u = max(u, lo * lo); }
  let pole = dl <= vec2f(1e-4 * lo * lo);
  let k0 = select(1.0 / max(dl, vec2f(1e-30)), vec2f(0.0), pole);
  let w0 = ey * k0;
  let s0 = dot(w0, w0);
  if (s0 > 1.0) { u = max(u, (s0 - 1.0) / (2.0 * dot(w0 * w0, k0))); }
  for (var n = 0; n < 8; n++) {
    let k = 1.0 / (u + dl);
    let w = ey * k;
    let s = dot(w, w);
    u += 2.0 * s * (sqrt(s) - 1.0) / max(2.0 * dot(w * w, k), 1e-30);
  }
  let k = 1.0 / (u + dl);
  let w = select(ey * k, vec2f(0.0), pole);
  let r = select(y * k, vec2f(0.0), pole);
  return (u - lo * lo) * sqrt(dot(r, r) + max(1.0 - dot(w, w), 0.0) / (lo * lo));
}
"#;

/// `ell3` / `ell2` of ELL_WGSL line for line in f32 (`n` axes), as the shader reads it.
pub fn ell_f32<const N: usize>(p: [f32; N], e: [f32; N]) -> f32 {
    ell_f32_steps(p, e, 8)
}

/// The same with this many steps (how few will do: the test below).
fn ell_f32_steps<const N: usize>(p: [f32; N], e: [f32; N], steps: usize) -> f32 {
    let lo = e.iter().copied().fold(f32::MAX, f32::min);
    let y: [f32; N] = std::array::from_fn(|i| p[i].abs().max(1e-6 * lo));
    let dl: [f32; N] = std::array::from_fn(|i| e[i] * e[i] - lo * lo);
    let ey: [f32; N] = std::array::from_fn(|i| e[i] * y[i]);
    let len = |v: &[f32; N]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let one = (0..N).map(|i| ey[i] - dl[i]).fold(f32::MIN, f32::max);
    let mut u = one.max(len(&ey) - dl.iter().copied().fold(f32::MIN, f32::max));
    if (0..N).map(|i| (y[i] / e[i]) * (y[i] / e[i])).sum::<f32>() > 1.0 { u = u.max(lo * lo); }
    let pole: [bool; N] = std::array::from_fn(|i| dl[i] <= 1e-4 * lo * lo);
    let k0: [f32; N] = std::array::from_fn(|i| if pole[i] { 0.0 } else { 1.0 / dl[i].max(1e-30) });
    let w0: [f32; N] = std::array::from_fn(|i| ey[i] * k0[i]);
    let s0 = w0.iter().map(|x| x * x).sum::<f32>();
    if s0 > 1.0 { u = u.max((s0 - 1.0) / (2.0 * (0..N).map(|i| w0[i] * w0[i] * k0[i]).sum::<f32>())); }
    for _ in 0..steps {
        let k: [f32; N] = std::array::from_fn(|i| 1.0 / (u + dl[i]));
        let w: [f32; N] = std::array::from_fn(|i| ey[i] * k[i]);
        let s = w.iter().map(|x| x * x).sum::<f32>();
        let g = 2.0 * (0..N).map(|i| w[i] * w[i] * k[i]).sum::<f32>();
        u += 2.0 * s * (s.sqrt() - 1.0) / g.max(1e-30);
    }
    let w: [f32; N] = std::array::from_fn(|i| if pole[i] { 0.0 } else { ey[i] / (u + dl[i]) });
    let r: [f32; N] = std::array::from_fn(|i| if pole[i] { 0.0 } else { y[i] / (u + dl[i]) });
    let ww = w.iter().map(|x| x * x).sum::<f32>();
    (u - lo * lo) * (r.iter().map(|x| x * x).sum::<f32>() + (1.0 - ww).max(0.0) / (lo * lo)).sqrt()
}

/// The same in f64, taken to convergence (Newton from the left, then bisection where it stalls): the
/// reference the shader's reading is held to.
pub fn ell_f64<const N: usize>(p: [f64; N], e: [f64; N]) -> f64 {
    let lo = e.iter().copied().fold(f64::MAX, f64::min);
    let y: [f64; N] = std::array::from_fn(|i| p[i].abs().max(1e-12 * lo));
    let dl: [f64; N] = std::array::from_fn(|i| e[i] * e[i] - lo * lo);
    let ey: [f64; N] = std::array::from_fn(|i| e[i] * y[i]);
    let f = |u: f64| (0..N).map(|i| (ey[i] / (u + dl[i])).powi(2)).sum::<f64>() - 1.0;
    // bracket: f > 0 at a, f < 0 at b
    let at = (0..N).find(|&i| (0..N).all(|j| e[i] <= e[j])).unwrap_or(0);
    let mut a = lo * y[at];
    let mut b = (0..N).map(|i| ey[i] * ey[i]).sum::<f64>().sqrt() + 1.0;
    for _ in 0..400 {
        let m = 0.5 * (a + b);
        if f(m) > 0.0 { a = m } else { b = m }
        if b - a <= 1e-15 * b.max(1e-300) { break; }
    }
    let u = 0.5 * (a + b);
    let r: [f64; N] = std::array::from_fn(|i| y[i] / (u + dl[i]));
    (u - lo * lo) * r.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 }
    }

    /// The distance to an ellipsoid by searching its surface (angles, then finer about the best), owing
    /// nothing to the root above.
    fn searched(p: [f64; 3], e: [f64; 3]) -> f64 {
        let at = |u: f64, v: f64| [e[0] * u.cos() * v.sin(), e[1] * v.cos(), e[2] * u.sin() * v.sin()];
        let d = |q: [f64; 3]| ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2)).sqrt();
        let (mut bu, mut bv, mut best) = (0.0, 0.0, f64::MAX);
        for i in 0..256 {
            for j in 0..=128 {
                let (u, v) = (i as f64 / 256.0 * std::f64::consts::TAU, j as f64 / 128.0 * std::f64::consts::PI);
                let x = d(at(u, v));
                if x < best { (bu, bv, best) = (u, v, x); }
            }
        }
        let mut h = 0.03;
        for _ in 0..40 {
            let (cu, cv) = (bu, bv);
            for i in -4..=4 {
                for j in -4..=4 {
                    let (u, v) = (cu + i as f64 * h, (cv + j as f64 * h).clamp(0.0, std::f64::consts::PI));
                    let x = d(at(u, v));
                    if x < best { (bu, bv, best) = (u, v, x); }
                }
            }
            h *= 0.5;
        }
        let inside = (p[0] / e[0]).powi(2) + (p[1] / e[1]).powi(2) + (p[2] / e[2]).powi(2) < 1.0;
        if inside { -best } else { best }
    }

    #[test]
    fn the_reference_ellipsoid_reading_is_the_searched_distance() {
        let mut r = rng(0x5eed_e111);
        let mut worst = 0.0f64;
        for k in 0..300 {
            let e = [0.2 + r(), 0.1 + 0.6 * r(), 0.3 + 0.9 * r()];
            // inside, near, far, and on the middle planes
            let s = [0.3, 1.0, 1.05, 2.5][k % 4];
            let mut p = [(2.0 * r() - 1.0) * e[0] * s, (2.0 * r() - 1.0) * e[1] * s, (2.0 * r() - 1.0) * e[2] * s];
            if k % 7 == 0 { p[k % 3] = 0.0; }
            worst = worst.max((ell_f64(p, e) - searched(p, e)).abs());
        }
        assert!(worst < 1e-6, "{worst}");
    }

    /// The shader's reading (f32, eight steps) against the reference, at points inside and out, near the
    /// surface, deep inside close to the middle plane across the least axis (where single precision is
    /// known to struggle), and far off.
    #[test]
    fn the_shaders_ellipse_and_ellipsoid_hold_to_the_reference() {
        let mut r = rng(0xe111_5e1d);
        let (mut worst3, mut worst2, mut near3) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..20000 {
            let e = [0.06 + 1.2 * r(), 0.06 + 1.2 * r(), 0.06 + 1.2 * r()];
            let lo = e[0].min(e[1]).min(e[2]);
            let s = [0.2, 0.9, 1.0, 1.1, 3.0, 0.6][k % 6];
            let mut p = [(2.0 * r() - 1.0) * e[0] * s, (2.0 * r() - 1.0) * e[1] * s, (2.0 * r() - 1.0) * e[2] * s];
            if k % 5 == 0 {
                // close to the middle plane across the least axis
                let at = (0..3).find(|&i| e[i] == lo).unwrap();
                p[at] *= 1e-4 * r();
            }
            let want = ell_f64(p, e);
            let got = ell_f32(p.map(|x| x as f32), e.map(|x| x as f32)) as f64;
            let err = (got - want).abs() / lo.max(0.1);
            if err > 1e-3 && std::env::var("ELL_SAY").is_ok() { eprintln!("p {p:?} e {e:?} want {want} got {got}"); }
            worst3 = worst3.max(err);
            if want.abs() < 0.05 { near3 = near3.max((got - want).abs()); }
            let want2 = ell_f64([p[0], p[2]], [e[0], e[2]]);
            let got2 = ell_f32([p[0] as f32, p[2] as f32], [e[0] as f32, e[2] as f32]) as f64;
            worst2 = worst2.max((got2 - want2).abs() / e[0].min(e[2]).max(0.1));
        }
        if std::env::var("ELL_SAY").is_ok() { eprintln!("near {near3:e}, ellipsoid {worst3:e}, ellipse {worst2:e}"); }
        // near the surface: well under a hundredth of a voxel of the finest volume (3 / 1536)
        assert!(near3 < 2e-5, "near the surface: {near3}");
        assert!(worst3 < 1e-3 && worst2 < 1e-3, "ellipsoid {worst3}, ellipse {worst2}");
    }

    /// How many steps the readings need (ELL_WGSL takes 8): ELL_SAY=1 prints the worst error by steps.
    #[test]
    fn how_many_steps_the_ellipsoid_needs() {
        let mut r = rng(0x57e9_5000);
        let pts: Vec<([f64; 3], [f64; 3])> = (0..20000).map(|k| {
            let e = [0.06 + 1.2 * r(), 0.06 + 1.2 * r(), 0.06 + 1.2 * r()];
            let s = [0.2, 0.9, 1.0, 1.1, 3.0, 0.6][k % 6];
            let mut p = [(2.0 * r() - 1.0) * e[0] * s, (2.0 * r() - 1.0) * e[1] * s, (2.0 * r() - 1.0) * e[2] * s];
            if k % 5 == 0 { let lo = e[0].min(e[1]).min(e[2]); let at = (0..3).find(|&i| e[i] == lo).unwrap(); p[at] *= 1e-4 * r(); }
            (p, e)
        }).collect();
        let mut say = Vec::new();
        for steps in [3, 4, 5, 6, 8] {
            let (mut near, mut all) = (0.0f64, 0.0f64);
            for (p, e) in &pts {
                let want = ell_f64(*p, *e);
                let got = ell_f32_steps(p.map(|x| x as f32), e.map(|x| x as f32), steps) as f64;
                all = all.max((got - want).abs() / e[0].min(e[1]).min(e[2]).max(0.1));
                if want.abs() < 0.05 { near = near.max((got - want).abs()); }
            }
            say.push(format!("{steps} steps: near {near:.2e}, anywhere {all:.2e}"));
            if steps == 8 { assert!(near < 2e-5 && all < 2e-3, "{}", say.last().unwrap()); }
        }
        if std::env::var("ELL_SAY").is_ok() { eprintln!("{}", say.join("\n")); }
        // (measured: 5 steps 1.8e-4 near the surface, 6 steps 1.7e-6, 8 steps 6.7e-7; anywhere 1e-3 from 5 on.
        // Six are not enough: on the points of the test above they leave 3.0e-4 near the surface, eight 4.5e-7.)
    }

    #[test]
    fn a_stretch_is_held_within_sixteen_to_one() {
        assert_eq!(held([1.0, 2.0, 64.0]), [4.0, 4.0, 64.0]);
        assert_eq!(held([f64::NAN, -2.0, 1.0]), [1.0, 2.0, 1.0]);
        assert!(!uneven([0.7; 3]) && uneven([1.0, 1.0, 1.0 + 1e-6]));
    }

    fn studio() -> Family { crate::presets::studio::family() }

    fn routed(id: &str, vals: &[(&str, f64)], s: [f64; 3]) -> (String, Vec<f64>, Routed) {
        let f = studio();
        let i = f.presets.iter().position(|p| p.id == id).unwrap();
        let mut v: Vec<f64> = f.presets[i].params.iter().map(|d| d.value).collect();
        for (k, x) in vals { v[f.presets[i].param_index(k).unwrap()] = *x; }
        let r = route(&f, i, &v, s);
        (f.presets[r.preset].id.to_string(), v, r)
    }

    #[test]
    fn a_stretched_box_is_a_box_of_other_sizes_with_its_own_edges() {
        let (id, v, r) = routed("block", &[("width", 0.6), ("height", 0.4), ("depth", 0.5)], [2.0, 0.5, 1.5]);
        assert_eq!(id, "block");
        assert!(!uneven(r.stretch));
        // within the volume: taken in whole, edges as they were
        assert_eq!(r.gain, 1.0);
        assert_eq!(&r.vals[..3], &[1.2, 0.2, 0.75]);
        assert_eq!(&r.vals[3..], &v[3..]);
    }

    #[test]
    fn a_shape_grown_past_the_volume_keeps_its_size_through_the_gain() {
        // a block whose own fit shrinks it, stretched: drawn as the shape whose sizes are stretched, shrunk
        // to fit, then grown back by the frame
        let (_, v, r) = routed("block", &[("width", 2.0), ("height", 1.0), ("depth", 1.0)], [2.0, 1.0, 1.0]);
        let fit0 = (STUDIO_FIT / (0.5 * (4.0f64 + 1.0 + 1.0).sqrt())).min(1.0);
        let r1 = 0.5 * (16.0f64 + 1.0 + 1.0).sqrt();
        let k = STUDIO_FIT / r1 * (1.0 - 1e-6);
        // the shape's sizes as drawn, times the gain: the stretched shape as the old fit drew it
        let drawn = [r.vals[0] * r.gain, r.vals[1] * r.gain, r.vals[2] * r.gain];
        for (d, want) in drawn.iter().zip([4.0 * fit0, 1.0 * fit0, 1.0 * fit0]) { assert!((d - want).abs() < 1e-9, "{drawn:?}"); }
        assert!((r.gain - fit0 / k).abs() < 1e-12);
        // and the edges, as the old fit drew them
        assert!((r.vals[4] * r.gain - v[4] * fit0).abs() < 1e-12);
        assert_eq!(r.stretch, [r.gain; 3]);
    }

    #[test]
    fn what_has_no_size_of_its_own_stays_a_stretch() {
        // a capsule stretched unevenly across its axis: its length taken, the rest left
        let (id, v, r) = routed("capsule", &[], [1.5, 1.0, 2.0]);
        assert_eq!(id, "capsule");
        assert!((r.vals[0] * r.gain - 1.5 * v[0]).abs() < 1e-9 && (r.vals[1] * r.gain - v[1]).abs() < 1e-9);
        assert_eq!(r.stretch, [r.gain, r.gain, 2.0 * r.gain]);
        // an arch has none: all of it is left
        let (id, v, r) = routed("arch", &[], [1.0, 1.0, 3.0]);
        assert_eq!((id.as_str(), &r.vals, r.stretch, r.gain), ("arch", &v, [1.0, 1.0, 3.0], 1.0));
    }

    #[test]
    fn a_ball_stretched_unevenly_is_an_ellipsoid_and_a_disc_an_oval() {
        let (id, _, r) = routed("sphere", &[("radius", 0.5)], [1.0, 2.0, 0.5]);
        assert_eq!((id.as_str(), r.vals.as_slice(), r.stretch), ("ellipsoid", &[0.5, 1.0, 0.25][..], [1.0; 3]));
        // a ball with a flat base is not: left stretched
        let (id, _, r) = routed("sphere", &[("radius", 0.5), ("cut", 0.3)], [1.0, 2.0, 0.5]);
        assert_eq!(id, "sphere");
        assert!(uneven(r.stretch));
        let (id, v, r) = routed("cylinder", &[], [1.0, 1.25, 0.5]);
        assert_eq!(id, "oval");
        assert_eq!(r.vals, [v[0], 0.5 * v[0], 1.25 * v[1], v[2], v[3]]);
        // across its axis alike: still a cylinder, of another radius
        let (id, v, r) = routed("cylinder", &[], [0.8, 1.25, 0.8]);
        assert_eq!((id.as_str(), r.vals[0], r.vals[1]), ("cylinder", 0.8 * v[0], 1.25 * v[1]));
    }
}
