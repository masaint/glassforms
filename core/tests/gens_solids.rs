//! The geometric solids: what their generators claim, worked out again here without the shaders.
//!
//!   - the kaleidoscope: the slider values each named polyhedron starts with really give that polyhedron
//!     (its corners, edges and faces are counted from the reflections themselves), with every edge the
//!     same length;
//!   - the stellations: how far each order reaches, against two other ways of finding it, and that the
//!     table the shader reads between keeps every order, whole or not, just inside the volume;
//!   - the prism and antiprism: "height 1" makes the upright edges as long as the base's.
use gf_core::presets::solids::{self, Core};
use gf_core::presets::Preset;

type V3 = [f64; 3];
fn dot(a: &V3, b: &V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn sub(a: &V3, b: &V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn len(a: &V3) -> f64 { dot(a, a).sqrt() }
fn unit(a: &V3) -> V3 { let l = len(a); [a[0] / l, a[1] / l, a[2] / l] }
fn cross(a: &V3, b: &V3) -> V3 { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }

/// How far every shape of the family is made to reach (the constant the generators share).
const FIT: f64 = 1.22;

fn preset(id: &str) -> Preset {
    solids::family().presets.into_iter().find(|p| p.id == id).unwrap_or_else(|| panic!("no shape called {id}"))
}
fn value(p: &Preset, id: &str) -> f64 {
    p.params.iter().find(|d| d.id == id).unwrap_or_else(|| panic!("{}: no slider called {id}", p.id)).value
}

// ---------------------------------------------------------------------------------------- kaleidoscope

/// The three mirrors of the kaleidoscope for symmetry n, as the shader has them.
fn mirrors(n: f64) -> [V3; 3] {
    let c = (std::f64::consts::PI / n).cos();
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -c, (0.75 - c * c).sqrt()]]
}

/// Every image of a point under the reflections, found by reflecting until nothing new turns up.
fn orbit(v: &V3, m: &[V3; 3]) -> Vec<V3> {
    let mut pts = vec![*v];
    let mut i = 0;
    while i < pts.len() {
        for n in m {
            let k = 2.0 * dot(&pts[i], n);
            let x = [pts[i][0] - k * n[0], pts[i][1] - k * n[1], pts[i][2] - k * n[2]];
            if pts.iter().all(|y| len(&sub(&x, y)) > 1e-7) { pts.push(x); }
        }
        i += 1;
        assert!(pts.len() <= 240, "the reflections do not close up");
    }
    pts
}

/// The generating point for these weights: the shader's rule (weights on the chamber's corners, each
/// corner vector as far from the mirror opposite it as the others are from theirs).
fn generating_point(n: f64, w: [f64; 3]) -> V3 {
    let c = (std::f64::consts::PI / n).cos();
    let s = (0.75 - c * c).sqrt();
    let corners = [[0.0, 0.0, 1.0], [s, 0.0, 0.5], [0.0, s, c]];
    let mut v = [0.0; 3];
    for (wk, corner) in w.iter().zip(&corners) {
        for i in 0..3 { v[i] += wk * corner[i]; }
    }
    let u = unit(&v);
    [FIT * u[0], FIT * u[1], FIT * u[2]]
}

/// Corners, edges, and the faces counted by their number of sides, of the polyhedron a preset starts as.
fn census(id: &str) -> (usize, usize, Vec<(usize, usize)>) {
    let p = preset(id);
    let n = value(&p, "n");
    let m = mirrors(n);
    // the corner vectors really are equally far from the mirrors opposite them
    let c = (std::f64::consts::PI / n).cos();
    let s = (0.75 - c * c).sqrt();
    let corners: [V3; 3] = [[0.0, 0.0, 1.0], [s, 0.0, 0.5], [0.0, s, c]];
    for (corner, (opposite, others)) in corners.iter().zip([(2, [0, 1]), (0, [1, 2]), (1, [0, 2])]) {
        assert!((dot(corner, &m[opposite]) - s).abs() < 1e-12, "{id}");
        for o in others { assert!(dot(corner, &m[o]).abs() < 1e-12, "{id}: a corner off its mirrors"); }
    }
    let v = generating_point(n, [value(&p, "a2"), value(&p, "an"), value(&p, "a3")]);
    let verts = orbit(&v, &m);
    for x in &verts { assert!((len(x) - FIT).abs() < 1e-9, "{id}: a corner off the sphere"); }

    // edges: every corner's nearest neighbours, all at one distance
    let mut edge = f64::MAX;
    for (i, a) in verts.iter().enumerate() {
        for b in &verts[i + 1..] { edge = edge.min(len(&sub(a, b))); }
    }
    let mut ends = 0;
    for a in &verts {
        ends += verts.iter().filter(|b| (len(&sub(a, b)) - edge).abs() < 1e-7).count();
    }
    let edges = ends / 2;

    // faces: across each kind of axis, the planes that carry at least three corners
    let mut faces: Vec<(usize, usize)> = Vec::new();
    let mut every_side = 0;
    for axis in &corners {
        for normal in orbit(&unit(axis), &m) {
            let far = verts.iter().map(|x| dot(x, &normal)).fold(f64::MIN, f64::max);
            let on: Vec<&V3> = verts.iter().filter(|x| dot(x, &normal) > far - 1e-7).collect();
            if on.len() < 3 { continue; }
            // a real face: each of its corners has exactly two neighbours on it, an edge away
            for a in &on {
                let near = on.iter().filter(|b| (len(&sub(a, b)) - edge).abs() < 1e-7).count();
                assert_eq!(near, 2, "{id}: a face whose sides are not all edges");
            }
            every_side += on.len();
            match faces.iter_mut().find(|f| f.0 == on.len()) {
                Some(f) => f.1 += 1,
                None => faces.push((on.len(), 1)),
            }
        }
    }
    faces.sort();
    // every edge is a side of exactly two faces, and Euler's formula holds
    assert_eq!(every_side, 2 * edges, "{id}");
    let f: usize = faces.iter().map(|f| f.1).sum();
    assert_eq!(verts.len() + f, edges + 2, "{id}: corners {} + faces {f} != edges {edges} + 2", verts.len());
    (verts.len(), edges, faces)
}

#[test]
fn each_named_polyhedron_is_what_its_name_says() {
    // corners, edges, faces as (sides, how many)
    let want: [(&str, usize, usize, &[(usize, usize)]); 7] = [
        ("tetrahedron", 4, 6, &[(3, 4)]),
        ("cube", 8, 12, &[(4, 6)]),
        ("octahedron", 6, 12, &[(3, 8)]),
        ("dodecahedron", 20, 30, &[(5, 12)]),
        ("icosahedron", 12, 30, &[(3, 20)]),
        ("cuboctahedron", 12, 24, &[(3, 8), (4, 6)]),
        ("football", 60, 90, &[(5, 12), (6, 20)]),
    ];
    for (id, corners, edges, faces) in want {
        let (v, e, f) = census(id);
        assert_eq!((v, e, f.as_slice()), (corners, edges, faces), "{id}");
    }
    // the frame starts as the dodecahedron's thirty edges
    let (v, e, _) = census("frame");
    assert_eq!((v, e), (20, 30));
}

#[test]
fn the_seven_uniform_polyhedra_of_each_symmetry_sit_at_whole_weights() {
    // weights of 0 and 1 only: every edge comes out the same length (census checks that each face's
    // sides are all edges), and the counts are those of the textbook's solids
    let cases: [(f64, [f64; 3], usize, usize); 10] = [
        (4.0, [1.0, 1.0, 0.0], 24, 36),      // truncated octahedron
        (4.0, [1.0, 0.0, 1.0], 24, 36),      // truncated cube
        (4.0, [0.0, 1.0, 1.0], 24, 48),      // rhombicuboctahedron
        (4.0, [1.0, 1.0, 1.0], 48, 72),      // great rhombicuboctahedron
        (5.0, [1.0, 0.0, 0.0], 30, 60),      // icosidodecahedron
        (5.0, [1.0, 0.0, 1.0], 60, 90),      // truncated dodecahedron
        (5.0, [0.0, 1.0, 1.0], 60, 120),     // rhombicosidodecahedron
        (5.0, [1.0, 1.0, 1.0], 120, 180),    // great rhombicosidodecahedron
        (3.0, [1.0, 0.0, 0.0], 6, 12),       // the octahedron again, from the tetrahedron's mirrors
        (3.0, [1.0, 1.0, 0.0], 12, 18),      // truncated tetrahedron
    ];
    for (n, w, corners, edges) in cases {
        let verts = orbit(&generating_point(n, w), &mirrors(n));
        let mut edge = f64::MAX;
        for (i, a) in verts.iter().enumerate() {
            for b in &verts[i + 1..] { edge = edge.min(len(&sub(a, b))); }
        }
        let ends: usize = verts.iter().map(|a| verts.iter().filter(|b| (len(&sub(a, b)) - edge).abs() < 1e-7).count()).sum();
        assert_eq!((verts.len(), ends / 2), (corners, edges), "symmetry {n}, weights {w:?}");
    }
}

// ------------------------------------------------------------------------------------------ stellation

/// Every face plane of the core (both of each opposite pair), one unit from the centre.
fn planes(core: Core) -> Vec<V3> {
    core.normals().into_iter().flat_map(|n| [n, [-n[0], -n[1], -n[2]]]).collect()
}

/// The stellation's rule at a point, as the shader has it without the easing of its edges: the planes'
/// signed distances in order, the (k+1)-th blended with the (k+2)-th.
fn rule(core: Core, order: f64, h: f64, x: &V3) -> f64 {
    let mut d: Vec<f64> = core.normals().iter().map(|n| dot(n, x).abs() - h).collect();
    d.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let k = (order.floor() as usize).min(core.orders() - 1);
    let t = order - k as f64;
    (1.0 - t) * d[k] + t * d[k + 1]
}

/// Well-spread directions.
fn directions(n: usize) -> Vec<V3> {
    (0..n).map(|i| {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - y * y).sqrt();
        let a = i as f64 * 2.399963229728653;
        [r * a.cos(), y, r * a.sin()]
    }).collect()
}

#[test]
fn whole_orders_reach_as_far_as_the_farthest_corner_of_the_planes() {
    for core in [Core::Octahedron, Core::Dodecahedron, Core::Icosahedron] {
        let all = planes(core);
        // another way to the same number: the farthest point where three planes meet that has at most k
        // planes between it and the centre
        let mut far = vec![0.0f64; core.orders() + 2];
        for i in 0..all.len() {
            for j in i + 1..all.len() {
                for k in j + 1..all.len() {
                    let (a, b, c) = (&all[i], &all[j], &all[k]);
                    let (bc, ca, ab) = (cross(b, c), cross(c, a), cross(a, b));
                    let det = dot(a, &bc);
                    if det.abs() < 1e-9 { continue; }
                    let x = [(bc[0] + ca[0] + ab[0]) / det, (bc[1] + ca[1] + ab[1]) / det, (bc[2] + ca[2] + ab[2]) / det];
                    let beyond = all.iter().filter(|n| dot(n, &x) - 1.0 > 1e-9).count();
                    for (order, f) in far.iter_mut().enumerate() {
                        if beyond <= order { *f = f.max(len(&x)); }
                    }
                }
            }
        }
        for order in 0..=core.orders() {
            let r = core.reach(order as f64);
            assert!((r - far[order]).abs() < 1e-9, "{core:?} order {order}: reach {r}, farthest corner {}", far[order]);
        }
        // one order further and the figure is no longer bounded: some direction never meets enough planes
        let mut least = f64::MAX;
        let mut dirs = directions(4000);
        // (the directions along which most planes run are those square to two of the normals)
        let normals = core.normals();
        for i in 0..normals.len() {
            for j in i + 1..normals.len() { dirs.push(unit(&cross(&normals[i], &normals[j]))); }
        }
        for u in &dirs {
            let mut along: Vec<f64> = normals.iter().map(|n| dot(n, u).abs()).collect();
            along.sort_by(|a, b| b.partial_cmp(a).unwrap());
            least = least.min(along[core.orders() + 1]);
        }
        assert!(least < 1e-9, "{core:?}: order {} would still be bounded", core.orders() + 1);
    }
    // the numbers the books give
    let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(near(Core::Octahedron.reach(0.0), 3f64.sqrt()) && near(Core::Octahedron.reach(1.0), 3.0));
    let five = 5f64.sqrt();
    assert!(near(Core::Dodecahedron.reach(0.0), (15.0 - 6.0 * five).sqrt()));
    assert!(near(Core::Dodecahedron.reach(1.0), five) && near(Core::Dodecahedron.reach(2.0), five));
    // great stellated dodecahedron: its points are 3 phi^3 / sqrt(15 + 6 sqrt 5)... checked by number
    assert!((Core::Dodecahedron.reach(3.0) - 5.3306).abs() < 1e-3, "{}", Core::Dodecahedron.reach(3.0));
}

#[test]
fn every_order_whole_or_not_just_fits() {
    let dirs = directions(6000);
    for core in [Core::Octahedron, Core::Dodecahedron, Core::Icosahedron] {
        let steps = solids::ORDER_STEPS;
        let table: Vec<f64> = (0..=core.orders() * steps).map(|i| FIT / core.reach(i as f64 / steps as f64)).collect();
        let mut smallest = f64::MAX;
        for i in 0..=core.orders() * 40 {
            let order = i as f64 / 40.0;
            // the size the shader takes: read between the table's entries
            let at = order * steps as f64;
            let cell = (at.floor() as usize).min(table.len() - 2);
            let h = table[cell] + (table[cell + 1] - table[cell]) * (at - cell as f64);
            // by the formula, the figure then reaches
            let reach = h * core.reach(order);
            assert!(reach <= FIT * (1.0 + 1e-9), "{core:?} order {order}: reaches {reach}");
            smallest = smallest.min(reach);
            // and marching outwards along many directions never finds glass beyond that
            for u in dirs.iter().step_by(7) {
                for r in [reach * 1.004, 1.3, 1.45] {
                    let x = [u[0] * r, u[1] * r, u[2] * r];
                    assert!(rule(core, order, h, &x) > 0.0, "{core:?} order {order}: glass at distance {r}");
                }
            }
        }
        // between the table's entries the figure may be a little smaller than it could be, never much
        println!("{core:?}: never smaller than {:.4} of the size it could be", smallest / FIT);
        assert!(smallest > 0.98 * FIT, "{core:?}: shrinks to {smallest}");
    }
}

// ------------------------------------------------------------------------------------------------ hull

/// The corners of the two polygons as the shader places them (before its common scale).
fn hull_corners(sides: f64, twist: f64, height: f64) -> (Vec<V3>, Vec<V3>) {
    let step = std::f64::consts::TAU / sides;
    let turn = twist * 0.5 * step;
    let side = 2.0 * (0.5 * step).sin();
    let even = 0.5 * (side * side - 4.0 * (0.5 * turn).sin().powi(2)).sqrt();
    let h = height * even;
    let n = sides as usize;
    let lower = (0..n).map(|i| [(i as f64 * step).cos(), -h, (i as f64 * step).sin()]).collect();
    let upper = (0..n).map(|i| [(i as f64 * step + turn).cos(), h, (i as f64 * step + turn).sin()]).collect();
    (lower, upper)
}

#[test]
fn at_height_one_the_upright_edges_are_as_long_as_the_bases() {
    for sides in 3..=12 {
        for twist in [0.0, 0.3, 1.0] {
            let (lower, upper) = hull_corners(sides as f64, twist, 1.0);
            let base = len(&sub(&lower[0], &lower[1]));
            assert!((len(&sub(&upper[0], &upper[1])) - base).abs() < 1e-12);
            // the edge from each lower corner to the upper corner that stands over it
            assert!((len(&sub(&lower[0], &upper[0])) - base).abs() < 1e-12, "{sides} sides, turn {twist}");
            if twist == 1.0 {
                // an antiprism: the other upright edge too, so its triangles are equilateral
                assert!((len(&sub(&lower[1], &upper[0])) - base).abs() < 1e-12, "{sides} sides");
            }
        }
    }
    // three sides, fully turned: the regular octahedron
    let (lower, upper) = hull_corners(3.0, 1.0, 1.0);
    let all: Vec<V3> = lower.into_iter().chain(upper).collect();
    let r = len(&all[0]);
    let mut edges = 0;
    for (i, a) in all.iter().enumerate() {
        assert!((len(a) - r).abs() < 1e-12);
        for b in &all[i + 1..] {
            let d = len(&sub(a, b));
            // every pair of corners is an edge apart, or opposite
            assert!((d - r * 2f64.sqrt()).abs() < 1e-12 || (d - 2.0 * r).abs() < 1e-12);
            if (d - r * 2f64.sqrt()).abs() < 1e-12 { edges += 1; }
        }
    }
    assert_eq!(edges, 12);
    // the presets start where their names say
    let (prism, antiprism) = (preset("prism"), preset("antiprism"));
    assert_eq!(value(&prism, "twist"), 0.0);
    assert_eq!(value(&antiprism, "twist"), 1.0);
}
