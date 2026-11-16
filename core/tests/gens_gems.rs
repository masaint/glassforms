//! The gem cuts, measured: that a slider called "crown angle" really is the crown angle, that the round
//! brilliant has the facets a round brilliant has and that they meet where a cutter makes them meet, that
//! what the shader measures (a folded point against a handful of planes) is the whole stone, and that the
//! expressions written into the shader come to the same planes as the arithmetic done here.
use std::f64::consts::PI;

use gf_core::presets::gems::{gems, Cut, Gem, Plane, Stone, Sym, V3};

const RADIUS: f64 = 1.1;
const DEG: f64 = PI / 180.0;

fn dot(a: V3<f64>, b: V3<f64>) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: V3<f64>, b: V3<f64>) -> V3<f64> { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn sub(a: V3<f64>, b: V3<f64>) -> V3<f64> { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn len(a: V3<f64>) -> f64 { dot(a, a).sqrt() }

/// Where three planes meet, if they meet in a point.
fn meet(a: &Plane<f64>, b: &Plane<f64>, c: &Plane<f64>) -> Option<V3<f64>> {
    let (bc, ca, ab) = (cross(b.n, c.n), cross(c.n, a.n), cross(a.n, b.n));
    let det = dot(a.n, bc);
    if det.abs() < 1e-9 { return None; }
    Some([0, 1, 2].map(|i| (a.h * bc[i] + b.h * ca[i] + c.h * ab[i]) / det))
}
fn on(p: &Plane<f64>, x: V3<f64>) -> f64 { dot(p.n, x) - p.h }

fn gem(id: &str) -> Gem { gems().into_iter().find(|g| g.id == id).unwrap_or_else(|| panic!("no gem called {id}")) }
fn defaults(g: &Gem) -> Vec<f64> { g.params.iter().map(|d| d.value).collect() }
fn planes<'a>(s: &'a Stone<f64>, tier: &str) -> Vec<&'a Plane<f64>> { s.facets.iter().filter(|f| f.tier == tier).map(|f| &f.plane).collect() }

/// The stone's facets, with its rim (where it has one) stood in for by 96 upright planes touching it.
fn all_planes(s: &Stone<f64>) -> Vec<Plane<f64>> {
    let mut out: Vec<Plane<f64>> = s.facets.iter().map(|f| f.plane).collect();
    if s.rim_dist(0.0, 0.0).is_finite() {
        for i in 0..96 {
            let (c, sn) = ((2.0 * PI * (i as f64 + 0.37) / 96.0).cos(), (2.0 * PI * (i as f64 + 0.37) / 96.0).sin());
            // out along this direction to the rim, then the rim's own direction there
            let (mut lo, mut hi) = (0.0, 3.0);
            for _ in 0..60 { let mid = 0.5 * (lo + hi); if s.rim_dist(c * mid, sn * mid) < 0.0 { lo = mid } else { hi = mid } }
            let (x, z, e) = (c * lo, sn * lo, 1e-6);
            let g = [s.rim_dist(x + e, z) - s.rim_dist(x - e, z), 0.0, s.rim_dist(x, z + e) - s.rim_dist(x, z - e)];
            let l = len(g);
            let n = [g[0] / l, 0.0, g[2] / l];
            out.push(Plane { n, h: n[0] * x + n[2] * z });
        }
    }
    out
}

/// Every corner of the solid these planes bound.
fn corners_of(planes: &[Plane<f64>]) -> Vec<V3<f64>> {
    let mut out: Vec<V3<f64>> = Vec::new();
    for i in 0..planes.len() {
        for j in i + 1..planes.len() {
            for k in j + 1..planes.len() {
                let Some(x) = meet(&planes[i], &planes[j], &planes[k]) else { continue };
                if planes.iter().any(|p| on(p, x) > 1e-7) { continue; }
                if !out.iter().any(|y| len(sub(*y, x)) < 1e-6) { out.push(x); }
            }
        }
    }
    out
}

/// How many corners of the solid lie on this plane.
fn corners_on(p: &Plane<f64>, corners: &[V3<f64>]) -> usize {
    corners.iter().filter(|c| on(p, **c).abs() < 1e-6).count()
}

/// A few thousand points in and around the stone, the same every time.
fn scatter(n: usize) -> Vec<V3<f64>> {
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 11) as f64 / (1u64 << 53) as f64 * 3.0 - 1.5 };
    (0..n).map(|_| [next(), next(), next()]).collect()
}

/// The slider settings a cut is tried at: as it starts, and each slider at each end.
fn settings(g: &Gem) -> Vec<Vec<f64>> {
    let mut out = vec![defaults(g)];
    for (i, d) in g.params.iter().enumerate() {
        for v in [d.lo, d.hi] {
            let mut s = defaults(g);
            s[i] = v;
            out.push(s);
        }
    }
    out
}

#[test]
fn the_round_brilliant_has_a_round_brilliants_facets() {
    let g = gem("brilliant");
    let stone = g.cut.stone::<f64>(&defaults(&g));
    assert_eq!(stone.count(), [("table", 1), ("bezel", 8), ("star", 8), ("upper girdle", 16), ("pavilion main", 8), ("lower girdle", 16), ("culet", 1)]);
    assert_eq!(stone.facets.len(), 58);          // the 57 of a brilliant, and the culet
    // each really is a face of the stone, with the corners its kind has
    let corners = corners_of(&all_planes(&stone));
    for (tier, want) in [("table", 8), ("bezel", 4), ("star", 3), ("pavilion main", 5), ("culet", 8)] {
        for p in planes(&stone, tier) {
            assert_eq!(corners_on(p, &corners), want, "a {tier} facet");
        }
    }
    // (the girdle facets end on the round rim, so their corners depend on how finely the rim is stood in for)
    for tier in ["upper girdle", "lower girdle"] {
        for p in planes(&stone, tier) { assert!(corners_on(p, &corners) >= 3, "a {tier} facet"); }
    }
    // with no culet the pavilion comes to a point and its mains are kites
    let mut v = defaults(&g);
    v[6] = 0.0;
    let pointed = g.cut.stone::<f64>(&v);
    let corners = corners_of(&all_planes(&pointed));
    for p in planes(&pointed, "pavilion main") { assert_eq!(corners_on(p, &corners), 4); }
}

#[test]
fn the_round_brilliants_sliders_are_the_proportions_they_are_named_for() {
    let g = gem("brilliant");
    // as it starts, the two ends of every slider, and a stone that is none of those
    let mut cases = settings(&g);
    cases.push(vec![61.0, 31.25, 42.3, 5.0, 62.0, 84.0, 3.0]);
    for v in cases {
        let (table, crown, pav, girdle, star, lower, culet) = (v[0] / 100.0, v[1], v[2], v[3] / 100.0, v[4] / 100.0, v[5] / 100.0, v[6] / 100.0);
        let stone = g.cut.stone::<f64>(&v);
        let tab = planes(&stone, "table")[0];
        let (bezels, stars, mains) = (planes(&stone, "bezel"), planes(&stone, "star"), planes(&stone, "pavilion main"));
        let (uppers, lowers) = (planes(&stone, "upper girdle"), planes(&stone, "lower girdle"));

        // crown angle and pavilion angle: between each main facet and the girdle's plane
        for b in &bezels { assert!((b.n[1].acos() / DEG - crown).abs() < 1e-9, "crown angle {} for a slider at {crown}", b.n[1].acos() / DEG); }
        for m in &mains { assert!(((-m.n[1]).acos() / DEG - pav).abs() < 1e-9, "pavilion angle {} for a slider at {pav}", (-m.n[1]).acos() / DEG); }
        // eight of each, evenly round the stone
        for (i, b) in bezels.iter().enumerate() {
            let az = b.n[2].atan2(b.n[0]).rem_euclid(2.0 * PI) / DEG;
            assert!((az - 45.0 * i as f64).abs() < 1e-9, "bezel {i} faces {az}");
        }

        for i in 0..8 {
            let j = (i + 1) % 8;
            // table width: its corners are where two stars meet it, and the bezel between them passes through too
            let corner = meet(tab, stars[(i + 7) % 8], stars[i]).unwrap();
            assert!((2.0 * corner[0].hypot(corner[2]) / (2.0 * RADIUS) - table).abs() < 1e-9, "table {} for a slider at {table}", corner[0].hypot(corner[2]) / RADIUS);
            assert!(on(bezels[i], corner).abs() < 1e-9, "bezel {i} misses its corner of the table");

            // star length: the star's lower point is where it meets two bezels; seen from above it lies
            // this fraction of the way from the table's edge to the girdle
            let m = meet(bezels[i], bezels[j], stars[i]).unwrap();
            let edge = table * RADIUS * (PI / 8.0).cos();
            assert!(((m[0].hypot(m[2]) - edge) / (RADIUS - edge) - star).abs() < 1e-9, "star length");
            // the two upper girdle facets under this star pass through that point as well: five facets meet there
            assert!(on(uppers[2 * i], m).abs() < 1e-9 && on(uppers[2 * i + 1], m).abs() < 1e-9, "upper girdle facets miss the star's point");

            // lower-half length: where two lower girdle facets meet the ridge between two mains
            let n = meet(mains[i], mains[j], lowers[2 * i]).unwrap();
            assert!(((RADIUS - n[0].hypot(n[2])) / RADIUS - lower).abs() < 1e-9, "lower half length");
            assert!(on(lowers[2 * i + 1], n).abs() < 1e-9);

            // girdle thickness, where the mains meet it: straight above and below each other on the rim
            let (c, s) = ((45.0 * i as f64 * DEG).cos(), (45.0 * i as f64 * DEG).sin());
            let height = |p: &Plane<f64>| (p.h - p.n[0] * RADIUS * c - p.n[2] * RADIUS * s) / p.n[1];
            assert!(((height(bezels[i]) - height(mains[i])) / (2.0 * RADIUS) - girdle).abs() < 1e-9, "girdle thickness");
            // and the girdle facets either side of a main meet it at that same point of the rim
            let top = [RADIUS * c, height(bezels[i]), RADIUS * s];
            assert!(on(uppers[2 * i], top).abs() < 1e-9 && on(uppers[(2 * i + 15) % 16], top).abs() < 1e-9, "upper girdle facets miss the bezel's foot");
        }

        // total depth, table to culet, against the sum of its parts
        let culet_plane = planes(&stone, "culet")[0];
        let depth = (tab.h + culet_plane.h) / (2.0 * RADIUS);
        let parts = 0.5 * (1.0 - table) * (crown * DEG).tan() + girdle + 0.5 * (pav * DEG).tan() - 0.5 * culet * (pav * DEG).tan();
        assert!((depth - parts).abs() < 1e-9, "depth {depth}, parts {parts}");
        // the culet's width, flat to flat: where a main crosses its plane
        let across = (mains[0].h + mains[0].n[1] * culet_plane.h) / mains[0].n[0];
        assert!((across / RADIUS - culet).abs() < 1e-9, "culet width {} for a slider at {culet}", across / RADIUS);
    }
    // Tolkowsky's stone comes to the depth the books give: 59.3% with a knife-edge girdle, here plus the girdle
    let stone = g.cut.stone::<f64>(&[53.0, 34.5, 40.75, 0.0, 50.0, 77.0, 0.0]);
    let depth = (planes(&stone, "table")[0].h + planes(&stone, "culet")[0].h) / (2.0 * RADIUS);
    assert!((depth - 0.593).abs() < 0.002, "{depth}");
}

#[test]
fn every_cut_is_a_stone_whose_facets_are_all_there() {
    for g in gems() {
        let stone = g.cut.stone::<f64>(&defaults(&g));
        assert!(stone.dist([0.0; 3]) < -0.15, "{}: the middle of the volume is not well inside the stone", g.id);
        let all = all_planes(&stone);
        let corners = corners_of(&all);
        // every facet the recipe cuts is a face of the stone (it has corners), bar those a recipe says may vanish
        for f in &stone.facets {
            let n = corners_on(&f.plane, &corners);
            let may_vanish = f.tier.ends_with("corner") || f.tier == "culet";
            assert!(n >= 3 || may_vanish, "{}: a {} facet touches the stone at {n} corners", g.id, f.tier);
        }
        // and as a whole it sits in the middle of the volume, within reach
        let far = corners.iter().map(|c| len([c[0], c[1] - stone.lift, c[2]])).fold(0.0, f64::max);
        assert!(far > 1.0 && far < 1.3, "{}: reaches {far}", g.id);
        println!("{:10} {:3} facets, {:3} corners, reaches {far:.3}: {:?}", g.id, stone.facets.len(), corners.len(), stone.count());
    }
}

#[test]
fn every_cut_stays_a_stone_inside_the_volume_at_each_sliders_two_ends() {
    for g in gems() {
        for v in settings(&g) {
            let stone = g.cut.stone::<f64>(&v);
            assert!(stone.facets.iter().all(|f| f.plane.h.is_finite() && f.plane.n.iter().all(|c| c.is_finite())), "{} at {v:?}", g.id);
            assert!(stone.dist([0.0; 3]) < -0.08, "{} at {v:?}: the middle is {} from the surface", g.id, stone.dist([0.0; 3]));
            // nothing of it beyond a radius of 1.3: every point out there is outside
            for i in 0..4000 {
                let (a, b) = (i as f64 * 2.399963, (1.0 - 2.0 * (i as f64 + 0.5) / 4000.0f64).acos());
                let p = [1.3 * b.sin() * a.cos(), 1.3 * b.cos(), 1.3 * b.sin() * a.sin()];
                assert!(stone.dist(p) > 0.0, "{} at {v:?}: reaches beyond 1.3", g.id);
            }
        }
    }
}

#[test]
fn folding_then_measuring_a_few_planes_is_the_whole_stone() {
    let points = scatter(3000);
    for g in gems() {
        // the planes the shader keeps are chosen once, at the starting values
        let kept = g.cut.kept(&defaults(&g));
        let total = g.cut.stone::<f64>(&defaults(&g)).facets.len();
        assert!(kept.len() < total, "{}: nothing saved by folding", g.id);
        for v in settings(&g) {
            let stone = g.cut.stone::<f64>(&v);
            for p in &points {
                let (whole, folded) = (stone.dist(*p), g.cut.folded_dist(&v, &kept, *p));
                assert!((whole - folded).abs() < 1e-9, "{} at {v:?}, point {p:?}: {whole} whole, {folded} folded", g.id);
            }
        }
        println!("{:10} {total:3} facets, {:2} measured per lookup", g.id, kept.len());
    }
}

#[test]
fn the_shaders_expressions_come_to_the_same_planes() {
    for g in gems() {
        let kept = g.cut.kept(&defaults(&g));
        for v in settings(&g) {
            let stone = g.cut.stone::<f64>(&v);
            let (_, roots) = g.cut.expressions(&kept, g.params.len());
            // in the order they are written: the lift, the rim's numbers, then four for each plane kept
            let got: Vec<f64> = roots.iter().map(|r| r.eval(&v)).collect();
            assert!((got[0] - stone.lift).abs() < 1e-12, "{}", g.id);
            let at = got.len() - 4 * kept.len();
            for (i, &f) in kept.iter().enumerate() {
                let p = &stone.facets[f].plane;
                for (a, b) in got[at + 4 * i..at + 4 * i + 4].iter().zip([p.n[0], p.n[1], p.n[2], p.h]) {
                    assert!((a - b).abs() < 1e-12, "{} at {v:?}: facet {f} ({}) is {a} written down, {b} worked out", g.id, stone.facets[f].tier);
                }
            }
        }
    }
    // a number known before any slider is read is worked out here, not in the shader
    Sym::reset();
    let c: Cut = gem("brilliant").cut;
    let (_, roots) = c.expressions(&c.kept(&defaults(&gem("brilliant"))), 7);
    let names: Vec<String> = (0..7).map(|i| format!("s{i}")).collect();
    let (text, named) = Sym::wgsl(&roots, &names);
    assert!(named.contains(&"1.1".to_string()), "the rim's radius is a plain number: {named:?}");
    assert!(text.lines().all(|l| l.contains("pr.") || l.contains("g")), "every step written depends on a slider");
}

/// What the tracer is given to find a ray's way out of a stone without marching (`gems::inside`): its
/// planes and the stone's field agree everywhere (the planes alone wherever the rim is not nearer), and
/// every point of the rim's wall lies within the band of heights given for it.
#[test]
fn inside_planes_and_rim_band() {
    use gf_core::presets::gems::{inside, MAX_PLANES};
    let mut seed = 12345u64;
    let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 11) as f64 / (1u64 << 53) as f64 };
    for g in gems() {
        let v = defaults(&g);
        let s = g.cut.stone::<f64>(&v);
        let ins = inside(g.id, &v, 0.0).unwrap_or_else(|| panic!("{}: more than {MAX_PLANES} planes", g.id));
        let (mut rim_pts, mut inside_band) = (0, 0);
        for _ in 0..200_000 {
            let p = [rnd() * 2.4 - 1.2, rnd() * 2.4 - 1.2, rnd() * 2.4 - 1.2];
            let planes = ins.planes.iter().map(|q| q[0] * p[0] + q[1] * p[1] + q[2] * p[2] - q[3]).fold(f64::NEG_INFINITY, f64::max);
            let rim = s.rim_dist(p[0], p[2]);
            let d = s.dist(p);
            assert!((planes.max(rim) - d).abs() < 1e-9, "{}: the planes give {} at {p:?}, the stone {d}", g.id, planes.max(rim));
            // on the wall: the rim is what bounds the stone here
            if d.abs() < 0.01 && rim > planes {
                rim_pts += 1;
                if p[1] >= ins.band.0 && p[1] <= ins.band.1 { inside_band += 1; }
            }
        }
        assert_eq!(rim_pts, inside_band, "{}: points of the rim's wall outside its band {:?}", g.id, ins.band);
        // it is worked out afresh for every frame of an animated stone: it must take next to no time
        // (the quickest of twenty: other tests run beside this one)
        let ms = (0..20).map(|_| { let t0 = std::time::Instant::now(); inside(g.id, &v, 0.0); t0.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        println!("{}: {} planes, band {:?}, {} wall points, {ms:.3} ms", g.id, ins.planes.len(), ins.band, rim_pts);
        assert!(ms < 0.5, "{}: working out what the tracer needs took {ms:.2} ms", g.id);
    }
}
