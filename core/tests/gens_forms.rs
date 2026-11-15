//! The mathematical forms: the sums their shaders rest on, done again here in Rust and checked.
//!
//! Each of these shapes is scaled inside its shader so that it fills the volume without leaving it,
//! and several lean on a closed form (the largest radius the superformula reaches, how far a harmonic
//! bulges, which balls hold a Cassini surface's lobes, where an inverted torus ends up). A shader cannot
//! be run here, so the same formulas are written out again and tested against brute force: if one of
//! them were wrong, the shape would be cut off by the edge of the volume or shaved by its own bound.
use std::f64::consts::{PI, TAU};

use gf_core::presets::{self, Preset, Shape};

/// The largest radius a shape may reach: the volume's half-width is 1.5 and the tracer looks no further
/// than 1.47 from the centre.
const LIMIT: f64 = 1.3;

fn family() -> Vec<Preset> {
    presets::forms::family().presets
}
fn wgsl(p: &Preset) -> &str {
    match &p.shape {
        Shape::Field { wgsl, .. } => wgsl,
        _ => panic!("{}: not an equation", p.id),
    }
}
fn value(p: &Preset, id: &str) -> f64 {
    p.params.iter().find(|d| d.id == id).unwrap_or_else(|| panic!("{}: no slider {id}", p.id)).value
}
fn range(p: &Preset, id: &str) -> (f64, f64) {
    let d = p.params.iter().find(|d| d.id == id).unwrap_or_else(|| panic!("{}: no slider {id}", p.id));
    (d.lo, d.hi)
}
/// A few values across a slider's range, both ends included.
fn across(p: &Preset, id: &str, n: usize) -> Vec<f64> {
    let (lo, hi) = range(p, id);
    (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect()
}

#[test]
fn the_family_is_what_it_says() {
    let ids: Vec<&str> = family().iter().map(|p| p.id).collect();
    assert_eq!(ids, ["superformula", "superpillow", "superpebble", "harmonics", "metaballs", "cassini", "twistedring", "mobius", "cruller", "seashell", "nautilus", "julia", "cyclide"]);
    // every one is an equation with a `raw` of its own
    for p in family() {
        assert!(wgsl(&p).contains("fn raw(p: vec3f) -> f32"), "{}", p.id);
    }
}

// ---- the superformula ---------------------------------------------------------------------------

fn gielis(a: f64, m: f64, n1: f64, n: f64) -> f64 {
    let t = 0.25 * m * a;
    (t.cos().abs().powf(n) + t.sin().abs().powf(n)).powf(-1.0 / n1)
}
fn gielis_max(n1: f64, n: f64) -> f64 {
    1.0f64.max(2.0f64.powf((0.5 * n - 1.0) / n1))
}

#[test]
fn the_superformulas_largest_radius_is_the_closed_form() {
    let p = &family()[0];
    for &n1 in &across(p, "n1", 7) {
        for &n in &across(p, "n2", 9) {
            for m in 1..=8 {
                let most = (0..=4000).map(|i| gielis(TAU * i as f64 / 4000.0, m as f64, n1, n)).fold(0.0, f64::max);
                let want = gielis_max(n1, n);
                assert!((most - want).abs() < 1e-4 * want, "m {m} n1 {n1} n {n}: largest {most}, closed form {want}");
            }
        }
    }
}

#[test]
fn the_superformula_closes_up_for_every_whole_m() {
    // The radius is the same either side of the cut in the angle, and from n = 2 up so is its slope.
    // (Below 2 every lobe begins at a crease, or at a curve too tight to tell from one, and the cut
    // falls on such a place: there only the radius can agree.)
    for m in 0..=8 {
        for (n1, n) in [(1.5, 1.0), (1.5, 1.5), (2.0, 2.0), (2.0, 7.0), (3.0, 3.0), (12.0, 10.0)] {
            let r = |a: f64| gielis(a, m as f64, n1, n);
            assert!((r(PI) - r(-PI)).abs() < 1e-12);
            if n >= 2.0 {
                let e = 1e-5;
                let (before, after) = ((r(PI) - r(PI - e)) / e, (r(-PI + e) - r(-PI)) / e);
                // (one-sided slopes over a short step: they agree to within the curve's bend over that step)
                assert!((before - after).abs() < 5e-3 * (1.0 + before.abs()), "m {m} n1 {n1} n {n}: slopes {before} and {after}");
            }
        }
    }
}

/// How far the superformula surface reaches, as the shader works it out: 17 latitudes tried.
fn super_reach(v: &[f64; 6]) -> f64 {
    let wide = if v[0].round() == 0.0 { 1.0 } else { gielis_max(v[1], v[2]) };
    (0..=16)
        .map(|i| {
            let lat = 0.5 * PI * i as f64 / 16.0;
            gielis(lat, v[3].round(), v[4], v[5]) * (wide * lat.cos()).hypot(lat.sin())
        })
        .fold(0.0, f64::max)
}
/// The same by brute force over the whole surface.
fn super_reach_true(v: &[f64; 6]) -> f64 {
    let mut most = 0.0f64;
    for i in 0..=180 {
        let lat = 0.5 * PI * i as f64 / 180.0;
        let r2 = gielis(lat, v[3].round(), v[4], v[5]);
        for j in 0..360 {
            let r1 = gielis(TAU * j as f64 / 360.0, v[0].round(), v[1], v[2]);
            most = most.max(r2 * (r1 * lat.cos()).hypot(lat.sin()));
        }
    }
    most
}

#[test]
fn the_superformula_shapes_fit_the_volume() {
    for p in family().iter().filter(|p| p.id.starts_with("super")) {
        let d = p.defaults();
        let mut cases = vec![d.clone()];
        for i in 0..d.len() {
            for v in [p.params[i].lo, p.params[i].hi] {
                let mut c = d.clone();
                c[i] = v;
                cases.push(c);
            }
        }
        for c in cases {
            let v: [f64; 6] = c.clone().try_into().unwrap();
            // scaled by 1.25 over the reach the shader finds
            let fits = 1.25 * super_reach_true(&v) / super_reach(&v);
            assert!(fits <= LIMIT && fits > 1.1, "{} at {c:?}: reaches {fits}", p.id);
        }
    }
}

// ---- spherical harmonics --------------------------------------------------------------------------

/// The shader's harmonic: (x + iz)^m times the Legendre recurrence in y, scaled so that the zonal
/// harmonic is 1 at the pole.
fn harmonic(d: [f64; 3], l: i32, m: i32) -> f64 {
    let (mut c, mut pl) = ((1.0, 0.0), 1.0);
    for k in 1..=m {
        c = (c.0 * d[0] - c.1 * d[2], c.0 * d[2] + c.1 * d[0]);
        pl *= (2 * k - 1) as f64;
    }
    let mut below = 0.0;
    for ll in m + 1..=l {
        let next = ((2 * ll - 1) as f64 * d[1] * pl - (ll + m - 1) as f64 * below) / (ll - m) as f64;
        below = pl;
        pl = next;
    }
    let norm: f64 = (l - m + 1..=l + m).map(|j| 1.0 / j as f64).product();
    (if m == 0 { 1.0 } else { (2.0 * norm).sqrt() }) * pl * c.0
}
fn dir(theta: f64, phi: f64) -> [f64; 3] {
    [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()]
}

#[test]
fn the_harmonics_are_the_textbooks() {
    for i in 0..=20 {
        for j in 0..12 {
            let (th, ph) = (PI * i as f64 / 20.0, TAU * j as f64 / 12.0);
            let (d, x, s) = (dir(th, ph), th.cos(), th.sin());
            let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() < 1e-12, "{what} at {th} {ph}: {a} against {b}");
            close(harmonic(d, 1, 0), x, "Y10");
            close(harmonic(d, 2, 0), 0.5 * (3.0 * x * x - 1.0), "Y20");
            close(harmonic(d, 3, 0), 0.5 * (5.0 * x * x * x - 3.0 * x), "Y30");
            // with m above 0: sqrt(2 (l-m)!/(l+m)!) P_l^m(cos theta) cos(m phi), up to the sign convention
            close(harmonic(d, 1, 1), s * ph.cos(), "Y11");
            close(harmonic(d, 2, 1), (2.0f64 / 6.0).sqrt() * 3.0 * x * s * ph.cos(), "Y21");
            close(harmonic(d, 2, 2), (2.0f64 / 24.0).sqrt() * 3.0 * s * s * (2.0 * ph).cos(), "Y22");
            close(harmonic(d, 3, 2), (2.0f64 / 120.0).sqrt() * 15.0 * x * s * s * (2.0 * ph).cos(), "Y32");
        }
    }
}

#[test]
fn no_harmonic_exceeds_one_and_the_ball_they_shape_fits() {
    let p = family().into_iter().find(|p| p.id == "harmonics").unwrap();
    let (_, top) = range(&p, "amp");
    for l in 1..=8 {
        for m in 0..=l {
            // the largest value, by brute force and as the shader finds it (33 latitudes at longitude 0)
            let mut most = 0.0f64;
            for i in 0..=360 {
                for j in 0..=(if m == 0 { 0 } else { 180 }) {
                    most = most.max(harmonic(dir(PI * i as f64 / 360.0, PI * j as f64 / 180.0), l, m).abs());
                }
            }
            let found = (0..=32).map(|i| harmonic(dir(PI * i as f64 / 32.0, 0.0), l, m).abs()).fold(0.0, f64::max);
            assert!(most <= 1.0 + 1e-9, "l {l} m {m}: reaches {most}");
            if m == 0 { assert!((most - 1.0).abs() < 1e-9, "l {l}: the zonal harmonic is 1 at the pole"); }
            // the shape is scaled by 1.25 / (1 + A·found); what it truly reaches is that times 1 + A·most
            let fits = 1.25 * (1.0 + top * most) / (1.0 + top * found);
            assert!(fits <= LIMIT, "l {l} m {m}: reaches {fits}");
        }
    }
}

// ---- Cassini's surfaces ---------------------------------------------------------------------------

#[test]
fn the_balls_that_stand_in_far_from_a_cassini_surface_hold_all_of_it() {
    let p = family().into_iter().find(|p| p.id == "cassini").unwrap();
    for n in 2..=8usize {
        for &level in &across(&p, "level", 13) {
            let e = 1.0 / n as f64;
            let bn = level.powi(n as i32);
            let a = 1.22 / (1.0 + bn).powf(e);
            let (far, near) = (1.22, a * (1.0 - bn).max(0.0).powf(e));
            let centres: Vec<[f64; 2]> = (0..n).map(|i| { let t = TAU * i as f64 / n as f64; [t.cos(), t.sin()] }).collect();
            // the geometric mean of the distances to the centres, less b: negative inside the glass
            let f = |q: [f64; 3]| {
                let s: f64 = centres.iter().map(|c| ((q[0] - a * c[0]).powi(2) + q[1] * q[1] + (q[2] - a * c[1]).powi(2)).sqrt().ln()).sum();
                (s * e).exp() - level * a
            };
            let hold = |q: [f64; 3]| {
                if level < 1.0 {
                    let mid = 0.5 * (near + far);
                    centres.iter().map(|c| ((q[0] - mid * c[0]).powi(2) + q[1] * q[1] + (q[2] - mid * c[1]).powi(2)).sqrt()).fold(f64::MAX, f64::min) - 0.525 * (far - near) - 0.01
                } else {
                    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - 1.02 * far - 0.01
                }
            };
            // A grid through the upper half (the surface is the same below): over everything once the
            // lobes have joined, and about the lobe on the x axis while they are apart (they are all alike).
            let (x0, x1, half) = if level < 1.0 { (near - 0.03, far + 0.03, 0.6 * (far - near) + 0.03) } else { (-1.3, 1.3, 1.3) };
            let mut inside = 0;
            for i in 0..60 {
                for j in 0..30 {
                    for k in 0..60 {
                        let q = [x0 + (x1 - x0) * (i as f64 + 0.5) / 60.0, half * (j as f64 + 0.5) / 30.0, -half + 2.0 * half * (k as f64 + 0.5) / 60.0];
                        if f(q) < 0.0 {
                            inside += 1;
                            assert!(hold(q) < 0.0, "{n} centres, b/a {level}: glass at {q:?} lies outside the balls");
                            assert!(q.iter().map(|v| v * v).sum::<f64>().sqrt() <= 1.22 + 0.03, "{n} centres, b/a {level}: glass at {q:?}");
                        }
                    }
                }
            }
            assert!(inside > 200, "{n} centres, b/a {level}: only {inside} points of the grid are inside");
            // and the lobe does not reach the edge of the box it was looked for in
            if level < 1.0 {
                for k in 0..60 {
                    let z = -half + 2.0 * half * (k as f64 + 0.5) / 60.0;
                    assert!(f([0.5 * (near + far), half, z]) > 0.0 && f([0.5 * (near + far), 0.0, half]) > 0.0, "{n} centres, b/a {level}: the lobe is wider than supposed");
                }
            }
        }
    }
}

// ---- the seashell ---------------------------------------------------------------------------------

/// Where the shell's surface is, in the volume, for these numbers: points of the tube's surface over
/// the last five turns, placed as the shader places them, each with half the wall's thickness there.
fn shell_points(growth: f64, fat: f64, spire: f64, wall: f64) -> Vec<([f64; 3], f64)> {
    let b = growth.ln() / TAU;
    let (a, h) = (fat, spire);
    let be = b.atan();
    let out = (1.0 + a) * be.cos();
    let xl = out * (b * (be - PI)).exp();
    let zd = out * (b * (be - 0.5 * PI)).exp();
    let zu = out * (b * (be - 1.5 * PI)).exp();
    let top = (h - a).min(0.0);
    let plan = 0.5 * (1.0 + a + xl).max(zd + zu);
    let rm = 1.3 / (plan.hypot(0.5 * (h + a - top)) * (1.0 + 0.5 * wall * a));
    let (cx, cy, cz) = (0.5 * (1.0 + a - xl) * rm, 0.5 * (h + a + top) * rm, 0.5 * (zu - zd) * rm);
    let mut pts = Vec::new();
    for i in 0..=2000 {
        let t = -5.0 * TAU * i as f64 / 2000.0;                 // the angle, back from the mouth
        let rc = rm * (b * t).exp();
        for k in 0..48 {
            let ps = TAU * k as f64 / 48.0;
            let (r, y) = (rc * (1.0 + a * ps.cos()), rc * (h + a * ps.sin()));
            let half_wall = if wall > 0.0 { (0.5 * wall * fat * (r + h * y) / (1.0 + h * h)).max(0.016) } else { 0.0 };
            // in the shell: (r cos t, y down from the tip, r sin t); in the volume the shell is moved to its centre
            pts.push(([r * t.cos() - cx, cy - y, r * t.sin() - cz], half_wall));
        }
    }
    pts
}

#[test]
fn the_seashell_fits_the_volume_whatever_its_numbers() {
    let p = family().into_iter().find(|p| p.id == "seashell").unwrap();
    let (mut least, mut most) = (f64::MAX, 0.0f64);
    for &growth in &across(&p, "growth", 6) {
        for &fat in &across(&p, "fat", 5) {
            for &spire in &across(&p, "spire", 6) {
                for &wall in &across(&p, "wall", 3) {
                    // the surface, and the outside of the wall that stands on it
                    let pts = shell_points(growth, fat, spire, wall);
                    let reach = pts.iter().map(|(q, w)| (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() + w).fold(0.0, f64::max);
                    assert!(reach <= LIMIT, "growth {growth} fat {fat} spire {spire} wall {wall}: reaches {reach}");
                    if wall == 0.0 {
                        least = least.min(reach);
                        most = most.max(reach);
                    }
                }
            }
        }
    }
    // and a solid one is never left small in the middle of the volume
    assert!(least > 1.05 && most <= LIMIT, "solid shells reach from {least} to {most}");
    println!("a solid shell's farthest point lies between {least:.3} and {most:.3} from the centre");
}

#[test]
fn the_two_shells_start_as_a_spire_and_a_flat_coil() {
    let fam = family();
    let (shell, nautilus) = (fam.iter().find(|p| p.id == "seashell").unwrap(), fam.iter().find(|p| p.id == "nautilus").unwrap());
    assert!(value(shell, "spire") > 1.0 && value(nautilus, "spire") == 0.0);
    // neighbouring turns overlap, so each is one body: tube radii together exceed the gap between tube centres
    for p in [shell, nautilus] {
        let (w, a, h) = (value(p, "growth"), value(p, "fat"), value(p, "spire"));
        assert!(a * (1.0 + 1.0 / w) > (1.0 - 1.0 / w) * (1.0 + h * h).sqrt(), "{}: the turns do not touch", p.id);
    }
}

// ---- the cyclide ----------------------------------------------------------------------------------

#[test]
fn the_inverted_torus_fills_a_ball_of_radius_1_22_about_the_centre() {
    let p = family().into_iter().find(|p| p.id == "cyclide").unwrap();
    for &r in &across(&p, "tube", 5) {
        for &skew in &across(&p, "skew", 6) {
            let g = 1.0 - r;
            let e = skew * g;
            let k2 = 1.22 * (g * g - e * e) / g;
            let shift = k2 * e / (g * g - e * e);
            let (mut most, mut least_x, mut most_x) = (0.0f64, f64::MAX, f64::MIN);
            for i in 0..720 {
                for j in 0..360 {
                    let (u, v) = (TAU * i as f64 / 720.0, TAU * j as f64 / 360.0);
                    // a point of the torus (ring radius 1, tube radius r, axis y), then inverted about (e, 0, 0)
                    let q = [(1.0 + r * v.cos()) * u.cos(), r * v.sin(), (1.0 + r * v.cos()) * u.sin()];
                    let d = [q[0] - e, q[1], q[2]];
                    let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    // the shader's p is the image moved by `shift` along x
                    let img = [d[0] * k2 / dd - shift, d[1] * k2 / dd, d[2] * k2 / dd];
                    most = most.max((img[0] * img[0] + img[1] * img[1] + img[2] * img[2]).sqrt());
                    least_x = least_x.min(img[0]);
                    most_x = most_x.max(img[0]);
                }
            }
            assert!((most - 1.22).abs() < 1e-3, "tube {r} skew {skew}: reaches {most}");
            assert!((least_x + 1.22).abs() < 1e-3 && (most_x - 1.22).abs() < 1e-3, "tube {r} skew {skew}: spans {least_x} to {most_x}");
        }
    }
}

// ---- rings ----------------------------------------------------------------------------------------

#[test]
fn the_rings_stay_inside_the_volume_and_keep_their_holes() {
    let fam = family();
    // (preset, the slider that is the section's reach from the ring)
    for (id, size) in [("twistedring", "size"), ("mobius", "width"), ("cruller", "size")] {
        let p = fam.iter().find(|p| p.id == id).unwrap();
        let (ring_lo, ring_hi) = range(p, "ring");
        let mut reach = range(p, size).1;
        if id == "cruller" { reach *= 1.0 + range(p, "depth").1; }
        if id == "mobius" { reach = reach.hypot(range(p, "thick").1); }
        assert!(ring_hi + reach <= LIMIT, "{id}: reaches {}", ring_hi + reach);
        assert!(ring_lo - reach > 0.05, "{id}: the hole closes ({})", ring_lo - reach);
    }
}
