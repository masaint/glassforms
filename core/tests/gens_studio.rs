//! The studio shapes: what can be said of them without drawing anything.
//!
//! Their finish rests on two small pieces of arithmetic in the shader: the distance to an edge cut by a
//! flat (`edge2`), and to a corner where three such edges meet (`edge3`). Every flat, round and corner
//! of the family is one of those two. They are written out again here, the same arithmetic in Rust, and
//! held against the distance worked out a slower way that owes them nothing. (That the shader itself
//! draws these shapes is measured from its volumes by tools/gens_check.py.)
use gf_core::presets::{self, Family, Setting, Shape};
use gf_core::state::State;

const RT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Two faces meeting at a right angle, their edge cut by a flat that takes c off each: `edge2` of the shader.
fn edge2(a: f64, b: f64, c: f64) -> f64 {
    let (x, y) = (a.max(b), a.min(b));
    let s = (x + y + c) * RT2;
    if x <= 0.0 && s <= 0.0 { return x.max(s); }
    let on_face = x.hypot((y + c).max(0.0));
    let t = ((x - y) * RT2).clamp(0.0, c * RT2);
    let on_flat = (x - (-0.5 * c + t * RT2)).hypot(y - (-0.5 * c - t * RT2));
    on_face.min(on_flat)
}

/// Three faces meeting at right angles, each edge cut by a flat that takes c off both its faces: `edge3`.
fn edge3(q: [f64; 3], c: f64) -> f64 {
    let x = q[0].max(q[1]).max(q[2]);
    let z = q[0].min(q[1]).min(q[2]);
    let y = q[0] + q[1] + q[2] - x - z;
    let s = (x + y + c) * RT2;
    if x <= 0.0 && s <= 0.0 { return x.max(s); }
    let on_face = (x * x + (y + c).max(0.0).powi(2) + (z + c).max(0.0).powi(2)).sqrt();
    let t = (x - y) * RT2;
    let w = c * RT2;
    let mut d = 0.0;
    if t > w || z + t * RT2 + 0.5 * c > 0.0 {
        let (ab, pa) = ([w, -0.5 * c], [t, z + 0.5 * c]);
        let h = ((pa[0] * ab[0] + pa[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-12)).clamp(0.0, 1.0);
        let slant = (pa[0] - ab[0] * h).hypot(pa[1] - ab[1] * h);
        let straight = (t - w).hypot((z + c).max(0.0));
        d = slant.min(straight);
    }
    on_face.min(s.hypot(d))
}

/// Numbers between -1 and 1, the same every time.
fn scatter(n: usize) -> Vec<f64> {
    let mut seed = 0x9E3779B97F4A7C15u64;
    (0..n).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0 }).collect()
}

/// The distance from a point to the glass side of a set of planes n.x <= h (0 inside), found by
/// projecting onto each in turn with Dykstra's corrections, which ends at the nearest point of them all.
fn outside_distance(q: [f64; 3], planes: &[([f64; 3], f64)]) -> f64 {
    let mut x = q;
    let mut kept = vec![[0.0f64; 3]; planes.len()];
    for _ in 0..4000 {
        for (i, (n, h)) in planes.iter().enumerate() {
            let y = [x[0] + kept[i][0], x[1] + kept[i][1], x[2] + kept[i][2]];
            let over = (n[0] * y[0] + n[1] * y[1] + n[2] * y[2] - h).max(0.0);
            x = [y[0] - over * n[0], y[1] - over * n[1], y[2] - over * n[2]];
            kept[i] = [y[0] - x[0], y[1] - x[1], y[2] - x[2]];
        }
    }
    ((q[0] - x[0]).powi(2) + (q[1] - x[1]).powi(2) + (q[2] - x[2]).powi(2)).sqrt()
}

/// The true distance to the glass side of the planes: inside, as far as the nearest plane (negative).
fn true_distance(q: [f64; 3], planes: &[([f64; 3], f64)]) -> f64 {
    let deepest = planes.iter().map(|(n, h)| n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - h).fold(f64::MIN, f64::max);
    if deepest <= 0.0 { deepest } else { outside_distance(q, planes) }
}

#[test]
fn an_edge_cut_by_a_flat_is_measured_truly() {
    let r = scatter(4000);
    for c in [0.0, 0.07, 0.3] {
        // the two faces and the flat across their edge, in a plane (the third direction plays no part)
        let planes = [([1.0, 0.0, 0.0], 0.0), ([0.0, 1.0, 0.0], 0.0), ([RT2, RT2, 0.0], -c * RT2)];
        let mut worst = 0.0f64;
        for p in r.chunks(2) {
            let (a, b) = (p[0], p[1]);
            worst = worst.max((edge2(a, b, c) - true_distance([a, b, 0.0], &planes)).abs());
        }
        assert!(worst < 1e-9, "a flat of {c}: out by {worst}");
    }
}

#[test]
fn a_corner_of_three_cut_edges_is_measured_truly() {
    let r = scatter(6000);
    for c in [0.0, 0.07, 0.3] {
        // three faces, and a flat across each of their three edges
        let planes = [
            ([1.0, 0.0, 0.0], 0.0), ([0.0, 1.0, 0.0], 0.0), ([0.0, 0.0, 1.0], 0.0),
            ([RT2, RT2, 0.0], -c * RT2), ([0.0, RT2, RT2], -c * RT2), ([RT2, 0.0, RT2], -c * RT2),
        ];
        let (mut worst, mut outside) = (0.0f64, 0);
        for p in r.chunks(3) {
            let q = [p[0], p[1], p[2]];
            let want = true_distance(q, &planes);
            if want > 0.0 { outside += 1; }
            worst = worst.max((edge3(q, c) - want).abs());
        }
        assert!(outside > 1000, "{outside} points outside");
        assert!(worst < 1e-7, "a flat of {c}: out by {worst}");
        // where the three flats meet, straight out from the corner: the point c/2 in from each face
        let d = edge3([0.0; 3], c);
        assert!((d - 0.5 * c * 3f64.sqrt()).abs() < 1e-12, "{d}");
    }
    // with no flat it is the plain corner of a box
    for p in r.chunks(3) {
        let plain = (p[0].max(0.0).powi(2) + p[1].max(0.0).powi(2) + p[2].max(0.0).powi(2)).sqrt() + p[0].max(p[1]).max(p[2]).min(0.0);
        assert!((edge3([p[0], p[1], p[2]], 0.0) - plain).abs() < 1e-12);
    }
}

fn studio() -> Family {
    presets::studio::family()
}

#[test]
fn the_family_has_the_shapes_a_designer_reaches_for() {
    let f = studio();
    let ids: Vec<&str> = f.presets.iter().map(|p| p.id).collect();
    assert_eq!(ids, ["slab", "bar", "block", "washer", "bandring", "spiralband", "tube", "disc", "cylinder", "rectframe", "arch", "capsule", "lens", "sphere", "cone", "wedge"]);
    assert_eq!(f.name, "Studio shapes");
    // every one is an equation that is its own boundary: nothing shaves or rounds it after the fact
    for p in &f.presets {
        match &p.shape {
            Shape::Field { clip, shell, .. } => assert_eq!((clip.as_str(), shell.as_str()), ("-1.0", "0.0"), "{}", p.id),
            _ => panic!("{}: not an equation", p.id),
        }
    }
}

#[test]
fn every_edge_has_a_flat_and_a_round_and_they_come_last() {
    for p in &studio().presets {
        let ids: Vec<&str> = p.params.iter().map(|d| d.id).collect();
        if p.id == "capsule" {
            // the one shape with no edge
            assert_eq!(ids, ["length", "radius"]);
            continue;
        }
        let n = ids.len();
        assert_eq!(&ids[n - 2..], ["bevel", "round"], "{}", p.id);
        let (bevel, round) = (&p.params[n - 2], &p.params[n - 1]);
        // the same ranges on every shape, so that the two sliders mean the same everywhere; both can be nothing
        assert_eq!((bevel.lo, bevel.hi, bevel.step), (0.0, 0.15, 0.005), "{}", p.id);
        assert_eq!((round.lo, round.hi, round.step), (0.0, 0.2, 0.005), "{}", p.id);
        // and as it starts every shape has a soft edge of one kind or the other
        assert!(bevel.value > 0.0 || round.value > 0.0, "{}: starts with raw edges", p.id);
    }
}

#[test]
fn sizes_are_lengths_in_steps_of_a_hundredth() {
    for p in &studio().presets {
        for d in &p.params {
            if matches!(d.id, "bevel" | "round" | "arc") { continue; }
            assert_eq!(d.step, 0.01, "{}: {}", p.id, d.id);
        }
    }
    // the shapes that stand on a level face say how high they are, so that a floor can be put under them
    for id in ["slab", "bar", "block", "washer", "bandring", "spiralband", "disc", "cylinder", "rectframe", "cone", "wedge"] {
        let f = studio();
        assert!(f.preset(id).unwrap().param_index("height").is_some(), "{id}");
    }
}

#[test]
fn in_motion_the_family_is_built_finer_than_the_usual() {
    let f = studio();
    assert!(f.defaults.contains(&("animRes", Setting::Num(384.0))));
    assert_eq!(State::family_anim_res(&f), 384);
    // rings, frames and the coil are not convex: a ray leaving the glass must be followed into the next piece
    assert!(f.layers >= 1);
}
