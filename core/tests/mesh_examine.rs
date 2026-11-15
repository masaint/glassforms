//! Models examined and prepared, on shapes made here whose answers are known: what the examiner counts
//! and repairs, which treatment it chooses, and that inside and outside come out exactly.

use gf_core::mesh::solid::{self, Fit, Flips, Tree};
use gf_core::mesh::{examine, Mesh, Treatment};
use gf_core::{FIT, HB};
use std::f64::consts::PI;

// ---- shapes -------------------------------------------------------------------------------------------
fn sphere(r: f64, n: usize, c: [f64; 3], cut: Option<f64>) -> Mesh {
    let rows = n / 2;
    let mut m = Mesh::default();
    for j in 0..=rows {
        let th = PI * j as f64 / rows as f64;
        for i in 0..n {
            let ph = 2.0 * PI * i as f64 / n as f64;
            m.pos.push([(c[0] + r * th.sin() * ph.cos()) as f32, (c[1] + r * th.cos()) as f32, (c[2] + r * th.sin() * ph.sin()) as f32]);
        }
    }
    for j in 0..rows {
        for i in 0..n {
            let (a, b) = ((j * n + i) as u32, (j * n + (i + 1) % n) as u32);
            let (cc, d) = (a + n as u32, b + n as u32);
            // (with `cut`, the cap above that height, as a share of r, is left out)
            if cut.is_some_and(|h| (PI * (j + 1) as f64 / rows as f64).cos() >= h) { continue; }
            if j > 0 { m.tri.push([a, b, cc]); }
            if j < rows - 1 { m.tri.push([b, d, cc]); }
        }
    }
    m
}

fn torus(big: f64, small: f64, n: usize, k: usize) -> Mesh {
    let mut m = Mesh::default();
    for i in 0..n {
        let u = 2.0 * PI * i as f64 / n as f64;
        for j in 0..k {
            let v = 2.0 * PI * j as f64 / k as f64;
            m.pos.push([((big + small * v.cos()) * u.cos()) as f32, (small * v.sin()) as f32, ((big + small * v.cos()) * u.sin()) as f32]);
        }
    }
    for i in 0..n {
        for j in 0..k {
            let (a, b) = ((i * k + j) as u32, (i * k + (j + 1) % k) as u32);
            let (c, d) = ((((i + 1) % n) * k + j) as u32, (((i + 1) % n) * k + (j + 1) % k) as u32);
            m.tri.push([a, b, c]);
            m.tri.push([b, d, c]);
        }
    }
    m
}

/// A box; `skip`: a face to leave out (0..5: -x, +x, -y, +y, -z, +z).
fn cuboid(half: [f64; 3], c: [f64; 3], skip: Option<usize>) -> Mesh {
    let mut m = Mesh::default();
    for sz in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sx in [-1.0, 1.0] {
                m.pos.push([(c[0] + sx * half[0]) as f32, (c[1] + sy * half[1]) as f32, (c[2] + sz * half[2]) as f32]);
            }
        }
    }
    let faces = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];
    for (k, f) in faces.iter().enumerate() {
        if skip == Some(k) { continue; }
        m.tri.push([f[0], f[1], f[2]]);
        m.tri.push([f[0], f[2], f[3]]);
    }
    m
}

fn sheet(n: usize) -> Mesh {
    let mut m = Mesh::default();
    for j in 0..=n {
        for i in 0..=n {
            m.pos.push([2.0 * (i as f32 / n as f32 - 0.5), 0.05 * ((4 * i) as f32 / n as f32).sin(), 1.4 * (j as f32 / n as f32 - 0.5)]);
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = (j * (n + 1) + i) as u32;
            m.tri.push([a, a + n as u32 + 1, a + 1]);
            m.tri.push([a + 1, a + n as u32 + 1, a + n as u32 + 2]);
        }
    }
    m
}

/// A band with a half twist: one side, one edge.
fn mobius(n: usize) -> Mesh {
    let mut m = Mesh::default();
    for i in 0..n {
        let u = 2.0 * PI * i as f64 / n as f64;
        for s in [-0.3, 0.3] {
            let r = 1.0 + s * (u / 2.0).cos();
            m.pos.push([(r * u.cos()) as f32, (s * (u / 2.0).sin()) as f32, (r * u.sin()) as f32]);
        }
    }
    for i in 0..n {
        let (a, b) = ((2 * i) as u32, (2 * i + 1) as u32);
        // the last pair of the band meets the first the other way up
        let (c, d) = if i + 1 < n { ((2 * i + 2) as u32, (2 * i + 3) as u32) } else { (1, 0) };
        m.tri.push([a, b, c]);
        m.tri.push([b, d, c]);
    }
    m
}

fn join(a: &Mesh, b: &Mesh) -> Mesh {
    let mut m = a.clone();
    let off = m.pos.len() as u32;
    m.pos.extend_from_slice(&b.pos);
    m.tri.extend(b.tri.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    m
}
fn flipped(m: &Mesh, which: impl Fn(usize) -> bool) -> Mesh {
    let mut o = m.clone();
    for (k, t) in o.tri.iter_mut().enumerate() {
        if which(k) { t.swap(1, 2); }
    }
    o
}
/// Every triangle with corners of its own, as an STL file has them.
fn unwelded(m: &Mesh) -> Mesh {
    let mut o = Mesh::default();
    for t in &m.tri {
        let at = o.pos.len() as u32;
        for v in t { o.pos.push(m.pos[*v as usize]); }
        o.tri.push([at, at + 1, at + 2]);
    }
    o
}

fn close(a: f64, b: f64, share: f64) -> bool { (a - b).abs() <= share * b.abs() }

// ---- the examiner -------------------------------------------------------------------------------------
#[test]
fn whole_shapes_are_found_closed() {
    let s = examine(&sphere(1.0, 96, [0.0; 3], None)).unwrap();
    let r = &s.report;
    assert!(r.closed && r.gaps.is_empty() && r.treatment == Treatment::Closed, "{}", r.line);
    assert_eq!((r.line.as_str(), r.pieces, r.open_edges, r.crowded_edges, r.rewound, r.turned, r.hollows), ("closed", 1, 0, 0, 0, 0, 0));
    assert!(close(r.volume, 4.0 / 3.0 * PI, 0.01) && close(r.area, 4.0 * PI, 0.01), "{} {}", r.volume, r.area);
    // a hole through a shape is not a gap
    let t = examine(&torus(1.0, 0.4, 96, 48)).unwrap();
    assert!(t.report.closed && t.report.line == "closed", "{}", t.report.line);
    assert!(close(t.report.volume, 2.0 * PI * PI * 0.16, 0.01), "{}", t.report.volume);
    let c = examine(&cuboid([1.0, 0.5, 0.25], [0.0; 3], None)).unwrap();
    assert!(c.report.closed && close(c.report.volume, 1.0, 1e-6) && c.report.triangles == 12, "{}", c.report.line);
}

#[test]
fn corners_that_coincide_are_merged_and_triangles_that_are_none_are_dropped() {
    let whole = sphere(1.0, 48, [0.0; 3], None);
    let soup = unwelded(&whole);
    assert_eq!(soup.pos.len(), 3 * soup.tri.len());
    let r = examine(&soup).unwrap().report;
    let w = examine(&whole).unwrap().report;
    // the sphere's poles are single corners once merged, and it is closed again
    assert_eq!((r.vertices, r.triangles, r.closed, r.line.as_str()), (w.vertices, w.triangles, true, "closed"));
    assert!(r.vertices_in > r.vertices);
    // triangles with two corners the same, and a triangle given twice, are dropped
    let mut bad = whole.clone();
    bad.tri.push([5, 5, 9]);
    bad.tri.push([7, 9, 7]);
    bad.tri.push(bad.tri[10]);
    let b = examine(&bad).unwrap().report;
    assert_eq!((b.degenerate, b.duplicates, b.triangles, b.closed), (2, 1, w.triangles, true));
}

#[test]
fn faces_wound_the_wrong_way_are_put_right() {
    let whole = torus(1.0, 0.4, 64, 32);
    let w = examine(&whole).unwrap().report;
    // every third triangle turned over: not closed as read, one piece rewound, the same solid after
    let r = examine(&flipped(&whole, |k| k % 3 == 0)).unwrap().report;
    assert!(!r.closed && r.rewound == 1 && r.gaps.is_empty() && r.treatment == Treatment::Closed, "{}", r.line);
    assert!(close(r.volume, w.volume, 1e-9), "{} {}", r.volume, w.volume);
    assert!(r.line.starts_with("closed; 1 piece wound inconsistently"), "{}", r.line);
    // the whole thing inside out: turned
    let r = examine(&flipped(&whole, |_| true)).unwrap().report;
    assert!(r.turned == 1 && r.volume > 0.0 && close(r.volume, w.volume, 1e-9), "{}", r.line);
    // of two pieces side by side, one inside out
    let two = join(&sphere(1.0, 48, [0.0; 3], None), &flipped(&sphere(1.0, 48, [3.0, 0.0, 0.0], None), |_| true));
    let r = examine(&two).unwrap().report;
    assert_eq!((r.pieces, r.turned, r.hollows), (2, 1, 0), "{}", r.line);
    assert!(close(r.volume, 2.0 * 4.0 / 3.0 * PI, 0.02));
}

#[test]
fn gaps_are_found_measured_and_closed() {
    // a box with its top off: one gap of four edges, as large as the box, closed flat
    let open = cuboid([1.0; 3], [0.0; 3], Some(3));
    let e = examine(&open).unwrap();
    let r = &e.report;
    assert_eq!((r.gaps.len(), r.gaps[0].edges, r.open_edges, r.treatment), (1, 4, 4, Treatment::LargeOpenings), "{}", r.line);
    assert!(r.line.starts_with("1 large opening (the largest 82% of the model's size), closed by a flat guess"), "{}", r.line);
    assert!(close(r.gaps[0].area, 4.0, 1e-6) && close(r.volume, 8.0, 1e-6), "{} {}", r.gaps[0].area, r.volume);
    // the closed surface begins with the open one's triangles and has the fan after them
    assert_eq!(e.closed.tri[..e.surface.tri.len()], e.surface.tri[..]);
    assert_eq!(e.closed.tri.len(), e.surface.tri.len() + 4);
    // and it is closed: examined again it is found whole
    assert!(examine(&e.closed).unwrap().report.closed);

    // a ball with a small cap cut off: a small gap, closed, a solid
    let r = examine(&sphere(1.0, 96, [0.0; 3], Some(0.99))).unwrap().report;
    assert_eq!((r.gaps.len(), r.treatment), (1, Treatment::SmallGaps), "{}", r.line);
    assert_eq!(r.line, "1 small gap, closed automatically");
    assert!(r.gaps[0].area < 0.03 * r.area && close(r.volume, 4.0 / 3.0 * PI, 0.02), "{:?} {}", r.gaps[0], r.volume);

    // a sheet has no inside at all: a shell
    let r = examine(&sheet(24)).unwrap().report;
    assert_eq!((r.treatment, r.line.as_str()), (Treatment::Shell, "open surface: made a shell"));
    // a one-sided band too, and it is said to be one-sided
    let r = examine(&mobius(64)).unwrap().report;
    assert!(r.one_sided == 1 && r.treatment == Treatment::Shell, "{}", r.line);
}

#[test]
fn pieces_inside_pieces_keep_their_sense() {
    let outer = sphere(1.0, 64, [0.0; 3], None);
    let inner = sphere(0.6, 48, [0.0; 3], None);
    let shell_volume = 4.0 / 3.0 * PI * (1.0 - 0.216);
    // wound against the piece around it: a hollow
    let r = examine(&join(&outer, &flipped(&inner, |_| true))).unwrap().report;
    assert_eq!((r.pieces, r.hollows, r.turned, r.treatment), (2, 1, 0, Treatment::Closed), "{}", r.line);
    assert!(close(r.volume, shell_volume, 0.02), "{}", r.volume);
    // the whole file inside out: still a hollow ball
    let r = examine(&join(&flipped(&outer, |_| true), &inner)).unwrap().report;
    assert_eq!((r.hollows, r.turned), (1, 1), "{}", r.line);
    assert!(close(r.volume, shell_volume, 0.02), "{}", r.volume);
    // wound the same way as the piece around it: a solid inside a solid, no hollow
    let r = examine(&join(&outer, &inner)).unwrap().report;
    assert_eq!((r.hollows, r.turned), (0, 0), "{}", r.line);
}

#[test]
fn nothing_to_examine_is_an_error() {
    assert!(examine(&Mesh::default()).is_err());
    let flat = Mesh { pos: vec![[1.0, 1.0, 1.0]; 3], tri: vec![[0, 1, 2]] };
    assert!(examine(&flat).is_err());
}

// ---- inside and outside --------------------------------------------------------------------------------
/// The centre of voxel i of n.
fn at(i: u32, n: u32) -> f64 { ((i as f64 + 0.5) / n as f64 * 2.0 - 1.0) * HB }

/// Every voxel of a volume against a rule for which side it is on; voxels closer to the surface than
/// `near` voxels are passed over (the mesh is a polyhedron, not the smooth shape the rule knows).
fn sides(m: &Mesh, n: u32, near: f64, inside: impl Fn([f64; 3]) -> f64) -> (usize, usize) {
    let e = examine(m).unwrap();
    let fit = Fit::of(&e.surface, false);
    let pos = fit.place(&e.closed);
    let f = Flips::of(&pos, &e.closed.tri, n, 2);
    assert_eq!(f.unclosed, 0, "the surface is not closed along {} lines", f.unclosed);
    let h = 2.0 * HB / n as f64;
    let (mut wrong, mut tested) = (0, 0);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let d = inside([at(i, n), at(j, n), at(k, n)]);
                if d.abs() < near * h { continue; }
                tested += 1;
                if f.inside_voxel(i, j, k) != (d < 0.0) { wrong += 1; }
            }
        }
    }
    (wrong, tested)
}

#[test]
fn inside_and_outside_are_exact() {
    let n = 96;
    let len = |p: [f64; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    // a ball (fitted: radius FIT)
    let (wrong, tested) = sides(&sphere(1.0, 128, [0.0; 3], None), n, 0.2, |p| len(p) - FIT);
    assert!(wrong == 0 && tested > 800_000, "{wrong} of {tested}");
    // a ring
    let s = FIT / 1.4;
    let (wrong, _) = sides(&torus(1.0, 0.4, 192, 96), n, 0.2, |p| (((p[0] * p[0] + p[2] * p[2]).sqrt() - s).powi(2) + p[1] * p[1]).sqrt() - 0.4 * s);
    assert_eq!(wrong, 0);
    // a box whose faces lie exactly on voxel boundaries or centres somewhere: every line still counts right
    let b = FIT / 3f64.sqrt();
    let (wrong, _) = sides(&cuboid([1.0; 3], [0.0; 3], None), n, 0.01, |p| (p[0].abs() - b).max(p[1].abs() - b).max(p[2].abs() - b));
    assert_eq!(wrong, 0);
    // the same as loose triangles, with a third of them turned over
    let (wrong, _) = sides(&flipped(&unwelded(&sphere(1.0, 128, [0.0; 3], None)), |k| k % 3 == 1), n, 0.2, |p| len(p) - FIT);
    assert_eq!(wrong, 0);
    // two balls that pass through each other: inside either is inside
    let s = FIT / 1.5;
    let two = join(&sphere(1.0, 128, [-0.5, 0.0, 0.0], None), &sphere(1.0, 128, [0.5, 0.0, 0.0], None));
    let (wrong, _) = sides(&two, n, 0.2, |p| len([p[0] + 0.5 * s, p[1], p[2]]).min(len([p[0] - 0.5 * s, p[1], p[2]])) - s);
    assert_eq!(wrong, 0);
    // a ball with a hollow in it: the hollow is outside
    let hollow = join(&sphere(1.0, 128, [0.0; 3], None), &flipped(&sphere(0.6, 96, [0.0; 3], None), |_| true));
    let (wrong, _) = sides(&hollow, n, 0.2, |p| (len(p) - 0.8 * FIT).abs() - 0.2 * FIT);
    assert_eq!(wrong, 0);
    // a box with its top off, closed flat: exactly the box
    let (wrong, _) = sides(&cuboid([1.0; 3], [0.0; 3], Some(3)), n, 0.01, |p| (p[0].abs() - b).max(p[1].abs() - b).max(p[2].abs() - b));
    assert_eq!(wrong, 0);
}

#[test]
fn only_the_outside_of_the_solid_is_measured_to() {
    // two balls through each other: the parts of each inside the other are left out
    let s = FIT / 1.5;
    let two = join(&sphere(1.0, 128, [-0.5, 0.0, 0.0], None), &sphere(1.0, 128, [0.5, 0.0, 0.0], None));
    let e = examine(&two).unwrap();
    let p = solid::prepare(&e, false, false, 128).unwrap();
    assert!(p.hidden > 1000 && p.unclosed == 0, "{} hidden", p.hidden);
    let h = 2.0 * HB / 128.0;
    let len = |p: [f64; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let mut deepest = 0.0f64;
    for t in &p.tris {
        let c = [0, 1, 2].map(|a| (t[a] + t[3 + a] + t[6 + a]) as f64 / 3.0);
        // every triangle kept is on one ball's surface and not inside the other ball by more than a voxel or two
        let (d1, d2) = (len([c[0] + 0.5 * s, c[1], c[2]]) - s, len([c[0] - 0.5 * s, c[1], c[2]]) - s);
        assert!(d1.abs().min(d2.abs()) < 0.01, "a triangle off both balls");
        deepest = deepest.max(-d1.max(d2));
    }
    assert!(deepest < 2.0 * h, "a triangle {} voxels inside the solid was kept", deepest / h);
    // and nothing of the outside is missing: every point of the union's surface has a triangle within a voxel
    let tree = Tree::of(p.tris.clone());
    for k in 0..4000 {
        let (a, b) = (k as f64 * 0.61803399 % 1.0, (k as f64 * 0.7548777) % 1.0);
        let (th, ph) = ((1.0 - 2.0 * a).acos(), 2.0 * PI * b);
        let dir = [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()];
        for side in [-1.0, 1.0] {
            let q = [side * 0.5 * s + s * dir[0], s * dir[1], s * dir[2]];
            // (points of this ball that are inside the other are not on the union's surface)
            if len([q[0] + side * 0.5 * s, q[1], q[2]]) < s + h { continue; }
            let d = tree.dist2([q[0] as f32, q[1] as f32, q[2] as f32], 1.0).sqrt() as f64;
            assert!(d < 0.6 * h, "nothing within {} voxels of a point of the surface", d / h);
        }
    }
}

#[test]
fn a_model_with_too_much_surface_is_refused() {
    // sheets as large as the stage, thousands deep: every line of voxels would cross them thousands of
    // times, and every one would be drawn across the whole of every slice near it
    let mut m = Mesh::default();
    for k in 0..4000 {
        let y = k as f32 * 0.0004;
        let at = m.pos.len() as u32;
        m.pos.extend_from_slice(&[[-1.0, y, -1.0], [1.0, y, -1.0], [1.0, y, 1.0], [-1.0, y, 1.0]]);
        m.tri.push([at, at + 1, at + 2]);
        m.tri.push([at, at + 2, at + 3]);
    }
    let e = examine(&m).unwrap();
    for shell in [false, true] {
        let Err(why) = solid::prepare(&e, false, shell, 768) else { panic!("it was prepared") };
        assert!(why.contains("too much surface") && why.contains("less detail"), "{why}");
    }
    // at less detail there is room
    assert!(solid::prepare(&e, false, true, 64).is_ok());
}

#[test]
fn the_coarse_grid_holds_exact_distances() {
    let e = examine(&sphere(1.0, 192, [0.0; 3], None)).unwrap();
    let fit = Fit::of(&e.surface, false);
    let pos = fit.place(&e.closed);
    let tris: Vec<[f32; 9]> = e.closed.tri.iter().map(|t| { let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]); [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]] }).collect();
    let m = 48;
    let sign = Flips::of(&pos, &e.closed.tri, m, 2);
    let grid = solid::mid_field(&Tree::of(tris), m, Some(&sign));
    let mut worst = 0.0f64;
    for k in 0..m {
        for j in 0..m {
            for i in 0..m {
                let p = [at(i, m), at(j, m), at(k, m)];
                // (only the stage matters: the tracer looks no further out than 1.47)
                if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] > 1.47 * 1.47 { continue; }
                let want = ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - FIT).clamp(-1.0, 1.0);
                let got = grid[((k * m + j) * m + i) as usize] as f64;
                // near the surface it is measured to the triangles themselves; far off it is read from a coarser grid
                // (far off, where the distance has a peak or stops at its limit, the coarser grid rounds it down)
                let allow = if want.abs() < 0.3 { 6e-4 } else { 0.06 };
                assert!((got - want).abs() < allow && got.abs() - want.abs() < 0.012, "{got} against {want} at {p:?}");
                worst = worst.max((got - want).abs());
            }
        }
    }
    println!("coarse grid of a ball: worst {worst:.5}");
}

#[test]
fn a_model_is_fitted_to_the_stage() {
    // a long box off centre, with z as its long axis
    let m = cuboid([0.5, 1.0, 3.0], [10.0, -4.0, 7.0], None);
    for up_z in [false, true] {
        let fit = Fit::of(&m, up_z);
        let pos = fit.place(&m);
        let far = pos.iter().map(|p| (p[0] as f64).hypot(p[1] as f64).hypot(p[2] as f64)).fold(0.0, f64::max);
        assert!((far - FIT).abs() < 1e-5, "{far}");
        let (lo, hi) = pos.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), p| ([0, 1, 2].map(|a| lo[a].min(p[a])), [0, 1, 2].map(|a| hi[a].max(p[a]))));
        // centred, and its long axis is the stage's y when the file's up is z
        for a in 0..3 { assert!((lo[a] + hi[a]).abs() < 1e-5); }
        let long = (0..3).max_by(|a, b| (hi[*a] - lo[*a]).total_cmp(&(hi[*b] - lo[*b]))).unwrap();
        assert_eq!(long, if up_z { 1 } else { 2 });
    }
}

#[test]
fn the_slices_a_triangle_reaches_are_listed() {
    let n = 64;
    let h = 2.0 * HB as f32 / n as f32;
    // one triangle lying across slices 10 to 12 (their centres), reaching a band of two voxels either way
    let z = |k: f32| ((k + 0.5) / n as f32 * 2.0 - 1.0) * HB as f32;
    let tris = vec![[0.0, 0.0, z(10.0), 0.3, 0.0, z(12.0), 0.0, 0.3, z(11.0)]];
    let (pairs, start) = solid::reach(&tris, n, 2.0 * h as f64);
    let got: Vec<u32> = pairs.iter().map(|p| p[1]).collect();
    assert_eq!(got, (8..=14).collect::<Vec<u32>>());
    assert_eq!((start[8], start[9], start[15], start[n as usize]), (0, 1, 7, 7));
}
