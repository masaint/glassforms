//! Foam: soap bubbles stuck together, as thin glass films.
//!
//! Bubbles of seeded sizes are packed by a few hundred rounds of relaxation: every pair of bubbles is
//! pushed apart until their centres are at least d apart, d² = ra² + rb² − ra·rb, and all are drawn gently
//! together. At that distance the two spheres and the wall between them meet at 120°: Plateau's rule for
//! three films meeting along a line.
//!
//! Each bubble's outer film is its sphere (`Term::Shell`) cut away where it lies inside a neighbour: past
//! the radical plane of the two (where both spheres have equal power, the plane through their circle of
//! intersection). Each pair that touches shares a flat wall on that plane (`Term::Slab`), inside the
//! bubble's ball and on the right side of every other neighbour's radical plane: a facet of the power
//! diagram of the bubbles. Flat walls are exact for bubbles of one size; between unequal bubbles the true
//! wall bulges into the larger one (a sphere of radius ra·rb / (rb − ra)), slightly, and is drawn flat.
//! The corners where three films meet (the Plateau borders) are sharp: see `family` for why no melt.

use super::*;
use geo::*;

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

/// The radical plane of bubbles a and b: n.p = d, n from a to b (a's side: n.p < d).
fn radical(a: &Bubble, b: &Bubble) -> (V, f64) {
    let v = sub(b.c, a.c);
    let l = len(v).max(1e-12);
    let n = mul(v, 1.0 / l);
    (n, (dot(b.c, b.c) - dot(a.c, a.c) - b.r * b.r + a.r * a.r) / (2.0 * l))
}

/// The films of packed bubbles, as pieces. `t`: the films' half-thickness.
fn films(b: &[Bubble], t: f64, open: Option<f64>) -> Vec<Piece> {
    let n = b.len();
    let touch = |i: usize, j: usize| i != j && len(sub(b[i].c, b[j].c)) < 0.999 * (b[i].r + b[j].r);
    let nb: Vec<Vec<usize>> = (0..n).map(|i| (0..n).filter(|&j| touch(i, j)).collect()).collect();
    let cut = |terms: &mut Vec<Term>| {
        // a raft: nothing below the water
        if let Some(y) = open { terms.push(Term::Plane { n: [0.0, -1.0, 0.0], d: -y }); }
    };
    let mut out = Vec::new();
    for i in 0..n {
        let mut terms = vec![Term::Shell { c: b[i].c, r: b[i].r, t }];
        for &j in &nb[i] {
            let (nn, d) = radical(&b[i], &b[j]);
            terms.push(Term::Plane { n: nn, d });
        }
        cut(&mut terms);
        let e = b[i].r + 3.0 * t;
        out.push(Piece { lo: sub(b[i].c, [e; 3]), hi: add(b[i].c, [e; 3]), terms });
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
            cut(&mut terms);
            let ext: V = std::array::from_fn(|a| rho * (1.0 - nn[a] * nn[a]).max(0.0).sqrt() + 3.0 * t);
            out.push(Piece { lo: sub(cc, ext), hi: add(cc, ext), terms });
        }
    }
    out
}

/// Bubbles scaled so that the cluster's farthest film is `size` (the preset's `fit`) from its middle (so the films' thickness is
/// in the units of the volume), then their films.
fn finish(mut b: Vec<Bubble>, t: f64, raft: bool, size: f64) -> RawCurve {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in &b {
        for a in 0..3 {
            lo[a] = lo[a].min(q.c[a] - q.r);
            hi[a] = hi[a].max(q.c[a] + q.r);
        }
    }
    if raft { lo[1] = 0.0; }
    let mid = mul(add(lo, hi), 0.5);
    let far = b.iter().map(|q| len(sub(q.c, mid)) + q.r).fold(0.0, f64::max).max(1e-9);
    let s = size / far;
    for q in &mut b {
        q.c = mul(sub(q.c, if raft { [mid[0], 0.0, mid[2]] } else { mid }), s);
        q.r *= s;
    }
    let mut raw = RawCurve { pieces: films(&b, t, raft.then_some(0.0)), ..Default::default() };
    // placed by points on the bubbles (the corners of their boxes would place them too small); a raft by
    // its domes alone, cut flat at the water
    for q in &b {
        for i in -2..=2 {
            for j in 0..8 {
                let (el, az) = (i as f64 * 0.25 * std::f64::consts::PI, j as f64 * 0.25 * std::f64::consts::PI);
                let p = add(q.c, mul([el.cos() * az.cos(), el.sin(), el.cos() * az.sin()], q.r));
                if !raft || p[1] >= 0.0 { raw.frame.extend_from_slice(&p); }
            }
        }
    }
    if raft { raw.ground = Some(0.0); }
    raw
}

/// The films' half-thickness, in the units of the volume (the thickness slider scales it). Thinner than
/// about 2.5 voxels of the still's volume (768), a film's two faces are so close that the volume's steps
/// show on it as rings (moiré); at this it is clean.
const WALL: f64 = 0.01;
/// While a slider moves (and while playing) the volume is 256: there films are drawn this thick instead,
/// fatter but whole, as a tree's twigs are.
const QUICK_WALL: f64 = 0.024;
/// How far the farthest film is from the middle, in the volume (a preset's `fit`, so that the films'
/// thickness stays as given). The double bubble is smaller: two big films that fill the picture, seen
/// through each other, cost more than the 11.1 ms a frame at 4K allows.
const SIZE: f64 = crate::FIT;
const DOUBLE_SIZE: f64 = 1.05;
fn wall(o: &CurveOpts) -> f64 {
    if o.quick { QUICK_WALL } else { WALL }
}

/// Seeded sizes: a log-normal spread of `variety`.
fn sizes(rng: &mut Rng, n: usize, variety: f64) -> Vec<f64> {
    (0..n).map(|_| (variety * rng.gauss()).clamp(-2.0, 2.0).exp()).collect()
}

fn cluster(v: &[f64], o: &CurveOpts) -> RawCurve {
    let (seed, n, variety) = (v[0], v[1].round().clamp(2.0, 30.0) as usize, v[2]);
    let mut rng = Rng::new(seed * 7919.0 + 11.0);
    let rs = sizes(&mut rng, n, variety);
    let spread = (n as f64).cbrt() * 0.8;
    let mut b: Vec<Bubble> = rs.iter().map(|&r| Bubble { c: mul(rng.dir(), spread * rng.next().cbrt()), r }).collect();
    relax(&mut b, false);
    finish(b, wall(o), false, SIZE)
}

fn double(v: &[f64], o: &CurveOpts) -> RawCurve {
    let (ratio, third) = (v[0], v[1] >= 0.5);
    let mut b = vec![Bubble { c: [0.0; 3], r: 1.0 }, Bubble { c: [contact(1.0, ratio), 0.0, 0.0], r: ratio }];
    if third {
        // a third bubble in the groove: the standard triple bubble
        let r = 0.5 * (1.0 + ratio);
        b.push(Bubble { c: [0.5 * contact(1.0, ratio), 0.6, 0.6], r });
        relax(&mut b, false);
    }
    finish(b, wall(o), false, DOUBLE_SIZE)
}

fn raft(v: &[f64], o: &CurveOpts) -> RawCurve {
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
    finish(b, wall(o), true, SIZE)
}

fn info(raw: &RawCurve, _v: &[f64]) -> String {
    let caps = raw.pieces.iter().filter(|p| matches!(p.terms.first(), Some(Term::Shell { .. }))).count();
    format!("{caps} bubbles, {} walls between them", raw.pieces.len() - caps)
}

fn presets() -> Vec<Preset> {
    let eq = vec!["neighbours: d² = ra² + rb² − ra·rb", "films meet three at a time, at 120°"];
    let common = |id, name, note, f: CurveFn, params: Vec<ParamDef>, rot: [f64; 3]| Preset {
        eq: eq.clone(),
        note,
        params,
        fixed_length: true,
        fit: Some(if id == "double" { DOUBLE_SIZE } else { SIZE }),
        rot: Some(rot),
        info: Some(Box::new(info)),
        // playing: the bubbles swell and shrink against each other
        animate: Some(vec![if id == "double" { "ratio" } else { "variety" }]),
        ..Preset::new(id, name, Shape::Curve(f))
    };
    vec![
        common(
            "cluster",
            "Bubble cluster",
            "Soap bubbles stuck together. Where two meet they share a wall; where three meet, the films are at 120° (Plateau's rules).",
            Box::new(cluster),
            vec![
                ParamDef::new("seed", 4.0, 1.0, 999.0, 1.0),
                ParamDef::labelled("count", "bubbles", 7.0, 2.0, 24.0, 1.0),
                ParamDef::labelled("variety", "size variety", 0.3, 0.0, 0.6, 0.01),
            ],
            [18.0, 30.0, 0.0],
        ),
        common(
            "double",
            "Double bubble",
            "Two bubbles share one wall; it is flat when they are the same size. The double bubble is the least film that holds two volumes.",
            Box::new(double),
            vec![ParamDef::labelled("ratio", "size ratio", 0.8, 0.4, 1.0, 0.01), ParamDef::labelled("third", "third bubble", 0.0, 0.0, 1.0, 1.0)],
            [12.0, 20.0, 0.0],
        ),
        common(
            "raft",
            "Bubble raft",
            "Half-bubbles on a surface of water: the walls stand upright and meet three at a time, at 120°, like a honeycomb.",
            Box::new(raft),
            vec![
                ParamDef::new("seed", 2.0, 1.0, 999.0, 1.0),
                ParamDef::labelled("count", "bubbles", 14.0, 2.0, 40.0, 1.0),
                ParamDef::labelled("variety", "size variety", 0.15, 0.0, 0.6, 0.01),
            ],
            [32.0, 0.0, 0.0],
        ),
    ]
}

pub fn family() -> Family {
    Family {
        presets: presets(),
        // (no melt: a melt would fill the corners where films meet, the Plateau borders, a little more
        // softly, but its volume holds true distances only near the films, so rays crossing the bubbles'
        // insides creep and a frame at 4K takes two to three times as long)
        melt: 0.0,
        defaults: vec![("ior", Setting::Num(1.33)), ("absorb", Setting::Vec3([0.02, 0.012, 0.006])), ("rot", Setting::Vec3([18.0, 30.0, 0.0])), ("dist", Setting::Num(6.0)), ("fov", Setting::Num(26.0))],
        lights: tube_lights(),
        ..Family::new(FAMILY, "Foam", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(id: &str, v: &[f64]) -> RawCurve {
        let f = family();
        let p = f.preset(id).unwrap();
        let Shape::Curve(c) = &p.shape else { panic!() };
        c(v, &CurveOpts { len: 1.0, quick: false, text: "" })
    }

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
    fn every_preset_gives_films_and_a_cluster_is_quick() {
        let f = family();
        for p in &f.presets {
            let t = std::time::Instant::now();
            let raw = build(p.id, &p.defaults());
            assert!(raw.pieces.len() > p.defaults().get(1).map_or(1.0, |n| n.min(2.0)) as usize, "{}", p.id);
            assert!(!raw.frame.is_empty());
            eprintln!("foam {}: {} pieces in {:?}", p.id, raw.pieces.len(), t.elapsed());
            assert!(t.elapsed().as_millis() < 200, "{} took {:?}", p.id, t.elapsed());
        }
        let t = std::time::Instant::now();
        let big = build("cluster", &[5.0, 24.0, 0.6]);
        eprintln!("foam cluster of 24: {} pieces in {:?}", big.pieces.len(), t.elapsed());
        assert!(big.pieces.len() > 24);
    }

    #[test]
    fn equal_double_bubble_has_a_flat_wall_through_the_middle() {
        let raw = build("double", &[1.0, 0.0]);
        let wall = raw.pieces.iter().find_map(|p| match p.terms.first() { Some(Term::Slab { n, d, .. }) => Some((*n, *d)), _ => None }).unwrap();
        // two equal bubbles: the wall is the plane halfway between them, at x = 0 once centred
        assert!((wall.0[0].abs() - 1.0).abs() < 1e-9 && wall.1.abs() < 1e-9, "{wall:?}");
    }
}
