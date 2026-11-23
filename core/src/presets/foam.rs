//! Foam: soap bubbles stuck together, as thin glass films, traced from an exact formula.
//!
//! Bubbles of seeded sizes are packed by a few hundred rounds of relaxation: every pair of bubbles is
//! pushed apart until their centres are at least d apart, d² = ra² + rb² − ra·rb, and all are drawn gently
//! together. At that distance the two spheres and the wall between them meet at 120°: Plateau's rule for
//! three films meeting along a line.
//!
//! **The formula.** The bubbles (centre and radius) are the formula's table. At a point, the bubbles of least
//! *power* (|p − c|² − r²) are found: the point lies in the cell of the first in the power diagram of the
//! bubbles, whose cells are the bubbles' insides and whose faces are the walls between them. Of the four of
//! least power, it is measured against
//!   - each one's outer film: its sphere, |p − c| − r, thickened, on its own side of the radical plane of
//!     every neighbour among the four (the plane through the circle where the two spheres meet: where the
//!     sphere is not inside the neighbour);
//!   - each wall between two that touch: their radical plane, thickened, within their circle (inside the
//!     ball) and on its side of the planes it shares with the others: a facet of the power diagram.
//! (Written with no arrays and no loops over the four: a formula read at every hit of every ray is kept
//! straight; with arrays, seven bubbles cost ten times as much.)
//! Every term is a true distance (to a sphere, to a plane), and each film or wall is the largest of its
//! terms: exact inside, never more than the distance outside. Flat walls are exact for bubbles of one size;
//! between unequal bubbles the true wall bulges into the larger one (a sphere of radius ra·rb / (rb − ra)),
//! slightly, and is drawn flat. The corners where three films meet (the Plateau borders) are sharp.
//!
//! The films are as thin as `film` asks, and no thinner than two and a half voxels of whatever grid is
//! made from the formula (`u.res`): traced exactly ("csg") the formula is told of a grid four times finer.

use super::*;
use crate::field;
use geo::*;
use super::exact;

pub const FAMILY: &str = "foam";

/// A bubble: centre and radius.
#[derive(Clone, Copy, Debug)]
struct Bubble {
    c: V,
    r: f64,
}

/// The distance two bubbles' centres sit apart where their films meet at Plateau's angles.
fn contact(a: f64, b: f64) -> f64 {
    (a * a + b * b - a * b).sqrt()
}

/// Pack bubbles: push apart every pair closer than `contact`, draw all towards the middle (or, flat, onto
/// the plane y = 0 and towards its middle).
fn relax(b: &mut [Bubble], flat: bool) {
    let n = b.len();
    for round in 0..600 {
        let pull = if round < 450 { 0.04 } else { 0.0 };
        let mut mid = [0.0; 3];
        let mut w = 0.0;
        for q in b.iter() {
            mid = add(mid, mul(q.c, q.r * q.r * q.r));
            w += q.r * q.r * q.r;
        }
        let mid = mul(mid, 1.0 / w.max(1e-12));
        for q in b.iter_mut() {
            q.c = add(q.c, mul(sub(mid, q.c), pull));
            if flat { q.c[1] = 0.0; }
        }
        for _ in 0..4 {
            for i in 0..n {
                for j in i + 1..n {
                    let d0 = contact(b[i].r, b[j].r);
                    let v = sub(b[j].c, b[i].c);
                    let d = len(v);
                    if d >= d0 { continue; }
                    let u = if d > 1e-9 { mul(v, 1.0 / d) } else { unit([1.0, 0.37 * i as f64, 0.11 * j as f64]) };
                    // the smaller bubble gives way more
                    let (wi, wj) = (b[j].r / (b[i].r + b[j].r), b[i].r / (b[i].r + b[j].r));
                    b[i].c = sub(b[i].c, mul(u, (d0 - d) * wi));
                    b[j].c = add(b[j].c, mul(u, (d0 - d) * wj));
                    if flat {
                        b[i].c[1] = 0.0;
                        b[j].c[1] = 0.0;
                    }
                }
            }
        }
    }
}

/// Bubbles moved and scaled so that the cluster's farthest film is `size` from its middle (a raft: its
/// water level at y = 0 before it is moved).
fn placed(mut b: Vec<Bubble>, raft: bool, size: f64) -> (Vec<Bubble>, Option<f64>) {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in &b {
        for a in 0..3 {
            lo[a] = lo[a].min(q.c[a] - q.r);
            hi[a] = hi[a].max(q.c[a] + q.r);
        }
    }
    if raft { lo[1] = 0.0; }
    let mid = mul(add(lo, hi), 0.5);
    // the farthest point of the films (of a raft, of its domes above the water)
    let far = b.iter().map(|q| len(sub(q.c, mid)) + q.r).fold(0.0, f64::max).max(1e-9);
    let s = size / far;
    for q in &mut b {
        q.c = mul(sub(q.c, mid), s);
        q.r *= s;
    }
    (b, raft.then_some(-mid[1] * s))
}

/// The radical plane of bubbles a and b: n.p = d, n from a to b (a's side: n.p < d).
fn radical(a: &Bubble, b: &Bubble) -> (V, f64) {
    let v = sub(b.c, a.c);
    let l = len(v).max(1e-12);
    let n = mul(v, 1.0 / l);
    (n, (dot(b.c, b.c) - dot(a.c, a.c) - b.r * b.r + a.r * a.r) / (2.0 * l))
}

/// The films of packed bubbles (in the volume's units), as pieces: each bubble's outer film, its sphere
/// cut away where it lies inside a neighbour (past their radical plane); each wall between two that touch,
/// their radical plane within their circle, on its side of every other neighbour's plane (a facet of the
/// power diagram). `t`: the films' half-thickness; `water`: nothing below this height.
fn films(b: &[Bubble], t: f64, water: Option<f64>) -> Vec<Piece> {
    let n = b.len();
    let touch = |i: usize, j: usize| i != j && len(sub(b[i].c, b[j].c)) < 0.999 * (b[i].r + b[j].r);
    let nb: Vec<Vec<usize>> = (0..n).map(|i| (0..n).filter(|&j| touch(i, j)).collect()).collect();
    let mut out = Vec::new();
    for i in 0..n {
        let mut terms = vec![Term::Shell { c: b[i].c, r: b[i].r, t }];
        for &j in &nb[i] {
            let (nn, d) = radical(&b[i], &b[j]);
            terms.push(Term::Plane { n: nn, d });
        }
        let e = b[i].r + 3.0 * t;
        let mut lo = sub(b[i].c, [e; 3]);
        if let Some(w) = water { lo[1] = lo[1].max(w - 3.0 * t); }
        out.push(Piece { lo, hi: add(b[i].c, [e; 3]), terms });
        for &j in nb[i].iter().filter(|&&j| j > i) {
            let (nn, d) = radical(&b[i], &b[j]);
            // the circle the wall fills: its middle and radius
            let h = d - dot(nn, b[i].c);
            let rho2 = b[i].r * b[i].r - h * h;
            if rho2 <= 0.0 { continue; }
            let rho = rho2.sqrt();
            let cc = add(b[i].c, mul(nn, h));
            let mut terms = vec![Term::Slab { n: nn, d, t }, Term::Ball { c: b[i].c, r: b[i].r }];
            for &k in nb[i].iter().filter(|&&k| k != j) {
                let (m, e) = radical(&b[i], &b[k]);
                terms.push(Term::Plane { n: m, d: e });
            }
            for &k in nb[j].iter().filter(|&&k| k != i) {
                let (m, e) = radical(&b[j], &b[k]);
                terms.push(Term::Plane { n: m, d: e });
            }
            let ext: V = std::array::from_fn(|a| rho * (1.0 - nn[a] * nn[a]).max(0.0).sqrt() + 3.0 * t);
            out.push(Piece { lo: sub(cc, ext), hi: add(cc, ext), terms });
        }
    }
    out
}

/// The table: the films as pieces, through the shared engine's cell lists (exact.rs).
fn table(b: &[Bubble], water: Option<f64>, film: f64) -> Vec<[f32; 4]> {
    let pieces = films(b, film, water).iter().map(exact::written).collect();
    exact::write(pieces, water, 0.0)
}

/// How far the farthest film is from the middle, in the volume. The double bubble is smaller: two big
/// films that fill the picture, seen through each other, cost more than the 11.1 ms a frame at 4K allows.
const SIZE: f64 = crate::FIT;
const DOUBLE_SIZE: f64 = 1.05;

/// Seeded sizes: a log-normal spread of `variety`.
fn sizes(rng: &mut Rng, n: usize, variety: f64) -> Vec<f64> {
    (0..n).map(|_| (variety * rng.gauss()).clamp(-2.0, 2.0).exp()).collect()
}

fn cluster(v: &[f64]) -> Vec<[f32; 4]> {
    let (seed, n, variety) = (v[0], v[1].round().clamp(2.0, 30.0) as usize, v[2]);
    let mut rng = Rng::new(seed * 7919.0 + 11.0);
    let rs = sizes(&mut rng, n, variety);
    let spread = (n as f64).cbrt() * 0.8;
    let mut b: Vec<Bubble> = rs.iter().map(|&r| Bubble { c: mul(rng.dir(), spread * rng.next().cbrt()), r }).collect();
    relax(&mut b, false);
    let (b, w) = placed(b, false, SIZE);
    table(&b, w, v[3])
}

fn double(v: &[f64]) -> Vec<[f32; 4]> {
    let (ratio, third) = (v[0], v[1] >= 0.5);
    let mut b = vec![Bubble { c: [0.0; 3], r: 1.0 }, Bubble { c: [contact(1.0, ratio), 0.0, 0.0], r: ratio }];
    if third {
        // a third bubble in the groove: the standard triple bubble
        let r = 0.5 * (1.0 + ratio);
        b.push(Bubble { c: [0.5 * contact(1.0, ratio), 0.6, 0.6], r });
        relax(&mut b, false);
    }
    let (b, w) = placed(b, false, DOUBLE_SIZE);
    table(&b, w, v[2])
}

fn raft(v: &[f64]) -> Vec<[f32; 4]> {
    let (seed, n, variety) = (v[0], v[1].round().clamp(2.0, 40.0) as usize, v[2]);
    let mut rng = Rng::new(seed * 7919.0 + 23.0);
    let rs = sizes(&mut rng, n, variety);
    let spread = (n as f64).sqrt() * 0.8;
    let mut b: Vec<Bubble> = rs
        .iter()
        .map(|&r| {
            let a = TAU * rng.next();
            let d = spread * rng.next().sqrt();
            Bubble { c: [d * a.cos(), 0.0, d * a.sin()], r }
        })
        .collect();
    relax(&mut b, true);
    let (b, w) = placed(b, true, SIZE);
    table(&b, w, v[3])
}


fn film() -> ParamDef {
    // (the films' half-thickness, in the units of the volume)
    ParamDef::labelled("film", "film", 0.01, 0.002, 0.03, 0.0005)
}

fn presets() -> Vec<Preset> {
    let eq = vec!["neighbours: d² = ra² + rb² − ra·rb", "films meet three at a time, at 120°"];
    let common = |id, name, note, f: fn(&[f64]) -> Vec<[f32; 4]>, params: Vec<ParamDef>, rot: [f64; 3]| Preset {
        eq: eq.clone(),
        note,
        params,
        rot: Some(rot),
        table: Some(Box::new(move |v: &[f64], _: &str| f(v))),
        // playing: the bubbles swell and shrink against each other
        animate: Some(vec![if id == "double" { "ratio" } else { "variety" }]),
        ..Preset::new(id, name, Shape::Field { wgsl: exact::WGSL.to_string(), clip: field::CLIP_DEFAULT.to_string(), shell: field::SHELL_DEFAULT.to_string() })
    };
    vec![
        common(
            "cluster",
            "Bubble cluster",
            "Soap bubbles stuck together. Where two meet they share a wall; where three meet, the films are at 120° (Plateau's rules).",
            cluster,
            vec![
                ParamDef::new("seed", 4.0, 1.0, 999.0, 1.0),
                ParamDef::labelled("count", "bubbles", 7.0, 2.0, 24.0, 1.0),
                ParamDef::labelled("variety", "size variety", 0.3, 0.0, 0.6, 0.01),
                film(),
            ],
            [18.0, 30.0, 0.0],
        ),
        common(
            "double",
            "Double bubble",
            "Two bubbles share one wall; it is flat when they are the same size. The double bubble is the least film that holds two volumes.",
            double,
            vec![ParamDef::labelled("ratio", "size ratio", 0.8, 0.4, 1.0, 0.01), ParamDef::labelled("third", "third bubble", 0.0, 0.0, 1.0, 1.0), film()],
            [12.0, 20.0, 0.0],
        ),
        common(
            "raft",
            "Bubble raft",
            "Half-bubbles on a surface of water: the walls stand upright and meet three at a time, at 120°, like a honeycomb.",
            raft,
            vec![
                ParamDef::new("seed", 2.0, 1.0, 999.0, 1.0),
                ParamDef::labelled("count", "bubbles", 14.0, 2.0, 40.0, 1.0),
                ParamDef::labelled("variety", "size variety", 0.15, 0.0, 0.6, 0.01),
                film(),
            ],
            [32.0, 0.0, 0.0],
        ),
    ]
}

pub fn family() -> Family {
    Family {
        presets: presets(),
        defaults: vec![("ior", Setting::Num(1.33)), ("absorb", Setting::Vec3([0.02, 0.012, 0.006])), ("rot", Setting::Vec3([18.0, 30.0, 0.0])), ("dist", Setting::Num(6.0)), ("fov", Setting::Num(26.0))],
        lights: tube_lights(),
        ..Family::new(FAMILY, "Foam", Kind::Field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula on the CPU (the shared engine's; the film's thickness is in the table).
    fn dist(tb: &[[f32; 4]], p: V, _t: f64) -> f64 { exact::tests::dist(tb, p) }

    #[test]
    fn neighbours_meet_at_plateaus_distance() {
        let mut rng = Rng::new(3.0);
        let rs = sizes(&mut rng, 9, 0.3);
        let mut b: Vec<Bubble> = rs.iter().map(|&r| Bubble { c: mul(rng.dir(), rng.next()), r }).collect();
        relax(&mut b, false);
        let mut touching = 0;
        for i in 0..b.len() {
            for j in i + 1..b.len() {
                let d = len(sub(b[i].c, b[j].c));
                let d0 = contact(b[i].r, b[j].r);
                assert!(d > 0.98 * d0, "bubbles {i} and {j} overlap: {d} < {d0}");
                if d < 1.02 * d0 { touching += 1; }
            }
        }
        assert!(touching >= 9, "a loose cluster: {touching} contacts");
    }

    #[test]
    fn every_preset_gives_films_quickly() {
        for p in &family().presets {
            let t = std::time::Instant::now();
            let tb = field::table(p, &p.defaults(), "");
            assert!(t.elapsed().as_millis() < 200, "{} took {:?}", p.id, t.elapsed());
            assert!(tb.len() > 40 && tb[0] as usize * 4 == tb.len(), "{}", p.id);
        }
    }

    #[test]
    fn the_equal_double_bubble_has_a_flat_wall_through_the_middle_and_its_films_meet_at_120() {
        let b = vec![Bubble { c: [0.0; 3], r: 1.0 }, Bubble { c: [contact(1.0, 1.0), 0.0, 0.0], r: 1.0 }];
        let (b, _) = placed(b, false, DOUBLE_SIZE);
        let tb = table(&b, None, 0.0);
        let r = b[0].r;
        assert!((b[0].c[0] + b[1].c[0]).abs() < 1e-9);
        // the wall: the plane x = 0 between the two, within the circle where the spheres meet
        assert!(dist(&tb, [0.0, 0.3 * r, 0.0], 0.0).abs() < 1e-5);
        assert!((dist(&tb, [0.02, 0.3 * r, 0.0], 0.0) - 0.02).abs() < 1e-5);
        // the circle where the films meet: radius sqrt(r² - (d/2)²), d = r (equal bubbles)
        let rho = (r * r - (0.5 * r) * (0.5 * r)).sqrt();
        assert!(dist(&tb, [0.0, rho, 0.0], 0.0).abs() < 1e-5);
    }

    #[test]
    fn the_formula_is_never_more_than_the_distance() {
        let p = &family().presets[0];
        let tb = table_of(p);
        let mut r = Rng::new(9.0);
        for _ in 0..4000 {
            let q = [1.3 * r.sym(), 1.3 * r.sym(), 1.3 * r.sym()];
            let d = dist(&tb, q, 0.01);
            if d <= 0.002 { continue; }
            let dir = r.dir();
            let s = add(q, mul(dir, 0.999 * d));
            assert!(dist(&tb, s, 0.01) > -1e-9, "{q:?} {d}");
        }
    }
    fn table_of(p: &Preset) -> Vec<[f32; 4]> {
        (p.table.as_ref().unwrap())(&p.defaults(), "")
    }
}
