//! The arithmetic that combines layers (core/src/layers.rs), checked without a GPU: what two layers reduce
//! to, that the order of the layers does not matter, that a surface is left exactly alone away from its
//! seams, how smooth each profile is across a seam, that each combination's options are its own, and how
//! steep the result can be.

use gf_core::layers::{bump, dip, mask_of, ramp, read_box, stage, Combine, Op, Opt, Profile, Stack, COMBINE_WIDTH, STAGE_R};

/// The same numbers every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[(self.next() * v.len() as f64) as usize % v.len()]
    }
}

/// A stack of `n` layers with every option of every combination drawn at random.
fn random_stack(rng: &mut Rng, n: usize, ops: &[Op], fills: bool) -> Stack {
    let mut s = Stack::new();
    for _ in 1..n {
        s.add();
    }
    for i in 0..n {
        s.layers[i].show = rng.next() > 0.2;
    }
    if s.layers.iter().all(|l| !l.show) { s.layers[0].show = true; }
    for m in s.combos() {
        if m.count_ones() == 2 {
            let p = s.pair_mut(m).unwrap();
            p.op = rng.pick(ops);
            p.radius = if rng.next() < 0.15 { 0.0 } else { rng.range(0.01, 0.5) };
            p.profile = rng.pick(&Profile::ALL);
            p.gap = rng.range(-0.05, 0.2);
            p.amount = rng.next();
        } else if fills {
            let j = s.junction_mut(m).unwrap();
            j.fill = rng.range(-1.0, 1.0);
            j.reach = rng.range(0.05, 0.5);
        }
    }
    s
}

fn two(op: Op, radius: f64, profile: Profile) -> Stack {
    let mut s = Stack::new();
    s.add();
    let p = s.pair_mut(0b11).unwrap();
    p.op = op;
    p.radius = radius;
    p.profile = profile;
    s
}

// the smooth minimum as it is ordinarily written (Quilez): quadratic and cubic
fn smin2(a: f64, b: f64, k: f64) -> f64 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}
fn smin3(a: f64, b: f64, k: f64) -> f64 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * h * k / 6.0
}

#[test]
fn two_layers_are_the_ordinary_smooth_booleans() {
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for _ in 0..4000 {
        let (a, b, k) = (rng.range(-0.6, 0.6), rng.range(-0.6, 0.6), rng.range(0.01, 0.5));
        let d = [a, b, 0.0, 0.0];
        // union
        assert!((two(Op::Join, k, Profile::Smooth).combine().eval(&d) - smin3(a, b, k)).abs() < 1e-14);
        assert!((two(Op::Join, k, Profile::Round).combine().eval(&d) - smin2(a, b, k)).abs() < 1e-14);
        // a bevel: the two surfaces and the flat between them
        assert!((two(Op::Join, k, Profile::Bevel).combine().eval(&d) - a.min(b).min(0.5 * (a + b) - 0.5 * k)).abs() < 1e-14);
        // intersection
        assert!((two(Op::Overlap, k, Profile::Smooth).combine().eval(&d) + smin3(-a, -b, k)).abs() < 1e-14);
        // subtraction, the cutter hidden: B less A, and A less B
        let mut s = two(Op::ACutsB, k, Profile::Smooth);
        s.layers[0].show = false;
        assert!((s.combine().eval(&d) + smin3(-b, a, k)).abs() < 1e-14);
        let mut s = two(Op::BCutsA, k, Profile::Smooth);
        s.layers[1].show = false;
        assert!((s.combine().eval(&d) + smin3(-a, b, k)).abs() < 1e-14);
        // the cutter showing as well: it sits in the hole it made
        let s = two(Op::ACutsB, k, Profile::Smooth);
        assert!((s.combine().eval(&d) - a.min(-smin3(-b, a, k))).abs() < 1e-14);
        // a clearance makes the hole that much larger than the cutter
        let mut s = two(Op::ACutsB, k, Profile::Smooth);
        s.layers[0].show = false;
        s.pair_mut(0b11).unwrap().gap = 0.07;
        assert!((s.combine().eval(&d) + smin3(-b, a - 0.07, k)).abs() < 1e-14);
        // a morph: so much of one, the rest of the other
        let mut s = two(Op::Morph, k, Profile::Smooth);
        let t = rng.next();
        s.pair_mut(0b11).unwrap().amount = t;
        assert!((s.combine().eval(&d) - ((1.0 - t) * a + t * b)).abs() < 1e-14);
        // apart, and a radius of nothing: a plain hard union
        assert_eq!(two(Op::Apart, k, Profile::Smooth).combine().eval(&d), a.min(b));
        assert_eq!(two(Op::Join, 0.0, Profile::Smooth).combine().eval(&d), a.min(b));
        assert_eq!(two(Op::Overlap, 0.0, Profile::Round).combine().eval(&d), a.max(b));
    }
}

#[test]
fn the_order_of_the_layers_does_not_matter() {
    let mut rng = Rng(0x2545f4914f6cdd1d);
    for round in 0..6000 {
        let n = 2 + round % 3;
        let s = random_stack(&mut rng, n, &Op::ALL, true);
        let mut d = [0.0; 4];
        for v in d.iter_mut().take(n) {
            *v = rng.range(-0.4, 0.4);
        }
        // every layer to a new place
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, (rng.next() * (i + 1) as f64) as usize % (i + 1));
        }
        let to: Vec<Option<usize>> = order.iter().map(|j| Some(*j)).collect();
        let mut t = s.clone();
        t.remap(&to);
        let mut e = [0.0; 4];
        for i in 0..n {
            e[order[i]] = d[i];
        }
        let (f, g) = (s.combine().eval(&d), t.combine().eval(&e));
        assert!((f - g).abs() < 1e-12, "{n} layers moved {order:?}: {f} against {g}\n{s:?}");
        // and moved back, the options are what they were
        let mut back = vec![None; n];
        for i in 0..n {
            back[order[i]] = Some(i);
        }
        t.remap(&back);
        assert_eq!(t.combine(), s.combine());
    }
}

#[test]
fn a_surface_is_left_alone_away_from_its_seams() {
    let mut rng = Rng(0x1234567887654321);
    // two layers: exactly the nearer one wherever the other is further off than the radius
    for _ in 0..4000 {
        let k = rng.range(0.01, 0.5);
        let a = rng.range(-0.5, 0.5);
        let b = a + k + rng.range(0.0, 0.5);
        for p in Profile::ALL {
            assert_eq!(two(Op::Join, k, p).combine().eval(&[a, b, 0.0, 0.0]), a);
            assert_eq!(two(Op::Join, k, p).combine().eval(&[b, a, 0.0, 0.0]), a);
            assert_eq!(two(Op::Overlap, k, p).combine().eval(&[a, b, 0.0, 0.0]), b);
        }
    }
    // three and four joined: exactly the nearest wherever every other is further off than the largest
    // radius plus the deepest the others' own seam can dip (a sixth of a smooth seam's radius for two
    // others, 0.27 for three; a bevel dips by half its radius)
    for round in 0..6000 {
        let n = 3 + round % 2;
        let smooth = round % 4 < 2;
        let mut s = random_stack(&mut rng, n, &[Op::Join, Op::Apart], false);
        let mut kmax = 0.0f64;
        for m in s.combos() {
            if m.count_ones() == 2 {
                if smooth { s.pair_mut(m).unwrap().profile = Profile::Smooth; }
                kmax = kmax.max(s.pair(m).radius);
            }
        }
        for l in s.layers.iter_mut() {
            l.show = true;
        }
        let margin = kmax * if smooth { if n == 3 { 1.0 + 1.0 / 6.0 } else { 1.27 } } else { 2.0 };
        let lead = (rng.next() * n as f64) as usize % n;
        let mut d = [0.0; 4];
        let low = rng.range(-0.3, 0.3);
        for (i, v) in d.iter_mut().enumerate().take(n) {
            *v = if i == lead { low } else { low + margin + rng.range(0.0, 0.4) };
        }
        assert_eq!(s.combine().eval(&d), low, "{n} layers, {d:?}");
    }
    // and with one layer further off than twice the largest radius, three behave exactly as the two that are near
    for _ in 0..3000 {
        let mut s = random_stack(&mut rng, 3, &[Op::Join], false);
        for l in s.layers.iter_mut() {
            l.show = true;
        }
        let (a, b) = (rng.range(-0.3, 0.3), rng.range(-0.3, 0.3));
        let pair = s.pair(0b011);
        let kmax = s.combos().iter().filter(|m| m.count_ones() == 2).map(|m| s.pair(*m).radius).fold(0.0, f64::max);
        let d = [a, b, a.max(b) + 2.0 * kmax + 1e-9 + rng.range(0.0, 0.5), 0.0];
        let f = s.combine().eval(&d);
        let g = a.min(b) - dip(pair.radius, pair.profile.power(), (a - b).abs());
        assert!((f - g).abs() < 1e-14, "{f} against {g}");
    }
}

/// f along a line through a combined field, sampled finely: the largest jump in its slope and in its
/// second difference between neighbouring samples, each divided by the step (so a smooth function gives
/// figures that stay put as the step shrinks, and a jump one that grows as 1 / step).
fn roughness(f: &dyn Fn(f64) -> f64, t0: f64, t1: f64, n: usize) -> (f64, f64) {
    let h = (t1 - t0) / n as f64;
    let v: Vec<f64> = (0..=n).map(|i| f(t0 + h * i as f64)).collect();
    let d1: Vec<f64> = v.windows(2).map(|w| (w[1] - w[0]) / h).collect();
    let d2: Vec<f64> = d1.windows(2).map(|w| (w[1] - w[0]) / h).collect();
    let j1 = d1.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
    let j2 = d2.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
    (j1, j2)
}

#[test]
fn the_smooth_profile_keeps_curvature_continuous_across_a_seam() {
    // two planes at right angles, a = x and b = y, walked across the whole seam along a line that
    // crosses both of its edges (|a - b| = k) and its middle (a = b)
    let k = 0.2;
    let line = |s: &Stack, t: f64| s.combine().eval(&[0.3 * t + 0.01, -0.2 * t + 0.02, 0.0, 0.0]);
    for (n, h) in [(4000usize, 1.0 / 4000.0), (16000, 1.0 / 16000.0)] {
        let smooth = two(Op::Join, k, Profile::Smooth);
        let round = two(Op::Join, k, Profile::Round);
        let bevel = two(Op::Join, k, Profile::Bevel);
        let (s1, s2) = roughness(&|t| line(&smooth, t), -0.5, 0.5, n);
        let (r1, r2) = roughness(&|t| line(&round, t), -0.5, 0.5, n);
        let (b1, _) = roughness(&|t| line(&bevel, t), -0.5, 0.5, n);
        // smooth: the second derivative changes by no more than its own slope allows between two samples
        // (the third derivative is at most (0.5)^3 / k^2 = 3.1 here): no jump at the seam's edges or middle
        assert!(s2 < 6.3 * h, "smooth: second derivative jumps by {s2} at step {h}");
        assert!(s1 < 2.0 * h);
        // round: the second derivative jumps by 0.5^2 / 2k = 0.625 where the seam begins, whatever the
        // step (seen here as two half jumps, the sample that straddles the edge being half one, half the other)
        assert!(r2 > 0.3 && r2 < 0.33, "round: {r2}");
        assert!(r1 < 2.0 * h);
        // bevel: the slope itself jumps (a hard edge), by 0.25 in all
        assert!(b1 > 0.12 && b1 < 0.26, "bevel: {b1}");
    }
}

#[test]
fn three_and_four_layers_join_as_smoothly_as_two() {
    // Lines through places where three and four layers are all within their seams of each other, with
    // different radii for every pair and junction fills: the second derivative never jumps.
    let mut rng = Rng(0xfeedc0ffee123457);
    let mut worst = 0.0f64;
    for round in 0..400 {
        let n = 3 + round % 2;
        let mut s = random_stack(&mut rng, n, &[Op::Join], true);
        for m in s.combos() {
            if m.count_ones() == 2 {
                let p = s.pair_mut(m).unwrap();
                p.profile = Profile::Smooth;
                p.radius = rng.range(0.15, 0.4);
            }
        }
        for l in s.layers.iter_mut() {
            l.show = true;
        }
        let c = s.combine();
        // each layer's distance runs linearly along the line, all of them near each other
        let (base, slope): (Vec<f64>, Vec<f64>) = (0..4).map(|_| (rng.range(-0.05, 0.05), rng.range(-0.5, 0.5))).unzip();
        let f = |t: f64| {
            let mut d = [0.0; 4];
            for i in 0..n {
                d[i] = base[i] + slope[i] * t;
            }
            c.eval(&d)
        };
        let (j1a, j2a) = roughness(&f, -0.4, 0.4, 2000);
        let (j1b, j2b) = roughness(&f, -0.4, 0.4, 8000);
        // a jump would stay the same size as the step shrinks; a smooth function's shrinks with the step
        assert!(j1b < 0.4 * j1a + 1e-9, "slope jumps: {j1a} then {j1b}");
        assert!(j2b < 0.4 * j2a + 1e-6, "{n} layers: second derivative jumps: {j2a} at 2000 steps, {j2b} at 8000\n{s:?}");
        worst = worst.max(j2b * 8000.0 / 0.8);
    }
    // (the largest third derivative met, for the record: a few hundred, from the tightest seams)
    assert!(worst.is_finite());
}

#[test]
fn each_combination_keeps_its_own_options() {
    // (four layers: every set of them is a combination of its own)
    let mut s = Stack::new();
    for _ in 1..COMBINE_WIDTH {
        s.add();
    }
    let all = s.combos();
    assert_eq!(all.len(), 11);
    // give every combination different numbers
    for (i, m) in all.iter().enumerate() {
        let v = 0.01 * (i + 1) as f64;
        if m.count_ones() == 2 {
            assert!(s.set_opt(*m, Opt::Radius, v) && s.set_opt(*m, Opt::Gap, v + 0.001) && s.set_opt(*m, Opt::Amount, v + 0.002));
            s.pair_mut(*m).unwrap().op = Op::ALL[i % 6];
            s.pair_mut(*m).unwrap().profile = Profile::ALL[i % 3];
            // a pair has no fill, a junction no radius
            assert!(!s.set_opt(*m, Opt::Fill, 1.0));
        } else {
            assert!(s.set_opt(*m, Opt::Fill, v) && s.set_opt(*m, Opt::Reach, v + 0.3));
            assert!(!s.set_opt(*m, Opt::Radius, 1.0));
        }
    }
    let before = s.clone();
    // changing any one leaves every other exactly as it was
    for m in &all {
        let mut t = before.clone();
        if m.count_ones() == 2 {
            t.set_opt(*m, Opt::Radius, 0.555);
            t.pair_mut(*m).unwrap().op = Op::Morph;
        } else {
            t.set_opt(*m, Opt::Fill, 0.0555);
        }
        for other in all.iter().filter(|o| *o != m) {
            assert_eq!(t.pair(*other), before.pair(*other));
            assert_eq!(t.junction(*other), before.junction(*other));
        }
        assert_ne!(t, before);
    }
    // hiding layers and showing them again changes nothing about any combination
    let mut t = before.clone();
    for l in t.layers.iter_mut() {
        l.show = false;
    }
    for l in t.layers.iter_mut() {
        l.show = true;
    }
    assert_eq!(t, before);
    // taking a layer away takes its combinations with it and leaves the rest as they were, renumbered
    let mut t = before.clone();
    assert!(t.remove(1));
    assert_eq!(t.combos().len(), 4);
    assert_eq!(t.pair(mask_of(&[0, 1])), before.pair(mask_of(&[0, 2])));
    assert_eq!(t.pair(mask_of(&[0, 2])), before.pair(mask_of(&[0, 3])));
    assert_eq!(t.pair(mask_of(&[1, 2])), before.pair(mask_of(&[2, 3])));
    assert_eq!(t.junction(0b0111), before.junction(0b1101));
    // a layer added afterwards starts with the defaults, not with what the one taken away had
    t.add();
    assert_eq!(t.pair(mask_of(&[0, 3])), Default::default());
    assert_eq!(t.junction(0b1111), Default::default());
    // a setup holds all of it
    assert_eq!(Stack::from_setup(&before.to_setup()), before);
}

/// The steepest the combined distance is with respect to its layers: the sum of the sizes of its
/// derivatives by each layer's distance (if every layer's distance changes by at most 1 per unit moved,
/// the result changes by at most this much).
fn steepness(c: &Combine, d: &[f64; 4]) -> f64 {
    let e = 1e-6;
    (0..c.n)
        .map(|i| {
            let (mut hi, mut lo) = (*d, *d);
            hi[i] += e;
            lo[i] -= e;
            ((c.eval(&hi) - c.eval(&lo)) / (2.0 * e)).abs()
        })
        .sum()
}

#[test]
fn the_result_is_never_steeper_than_the_bound_says() {
    let mut rng = Rng(0x0123456789abcdef);
    let (mut plain, mut mixed, mut filled) = (0.0f64, 0.0f64, 0.0f64);
    for round in 0..60000 {
        let n = 2 + round % 3;
        let kind = round % 3;
        let mut s = random_stack(&mut rng, n, &Op::ALL, kind == 2);
        if kind == 0 {
            // one radius for every pair, no fills: every step is a weighted mean, which steepens nothing
            for m in s.combos() {
                if m.count_ones() == 2 { s.pair_mut(m).unwrap().radius = 0.2; }
            }
        }
        let c = s.combine();
        let bound = c.slope_bound();
        // where the layers are near each other, which is where anything happens
        let mut d = [0.0; 4];
        let spread = rng.pick(&[0.05, 0.2, 0.6]);
        for v in d.iter_mut().take(n) {
            *v = rng.range(-spread, spread);
        }
        let st = steepness(&c, &d);
        assert!(st <= bound + 1e-4, "{st} against a bound of {bound}, {n} layers at {d:?}\n{s:?}");
        match kind {
            0 => plain = plain.max(st),
            1 => mixed = mixed.max(st),
            _ => filled = filled.max(st / bound),
        }
    }
    // with one radius and no fills nothing is steeper than its layers at all
    assert!(plain <= 1.0 + 1e-6, "{plain}");
    // (how close the search came: with mixed radii, and with fills as a share of their bound)
    println!("steepest met: one radius {plain:.4}, mixed radii {mixed:.4}, with fills {filled:.4} of the bound");
}

#[test]
fn a_junction_fill_acts_only_where_all_its_layers_meet() {
    let mut s = Stack::new();
    s.add();
    s.add();
    for m in s.combos() {
        if m.count_ones() == 2 { s.pair_mut(m).unwrap().radius = 0.1; }
    }
    let plain = s.combine();
    s.set_opt(0b111, Opt::Fill, 0.8);
    s.set_opt(0b111, Opt::Reach, 0.15);
    let filled = s.combine();
    // where all three are equal: the whole fill (0.8 of a tenth of the reach), as added glass
    let d = [0.02, 0.02, 0.02, 0.0];
    assert!((plain.eval(&d) - filled.eval(&d) - 0.8 * 0.1 * 0.15).abs() < 1e-12);
    // where one of the three is further off than the reach: nothing
    for d in [[0.0, 0.0, 0.4, 0.0], [0.3, -0.1, -0.1, 0.0], [0.0, 0.3, 0.6, 0.0]] {
        assert_eq!(plain.eval(&d), filled.eval(&d));
    }
    // a negative fill takes glass away there
    s.set_opt(0b111, Opt::Fill, -1.0);
    assert!((s.combine().eval(&[0.0; 4]) - plain.eval(&[0.0; 4]) - 0.015).abs() < 1e-12);
    // the pieces it is made of
    assert_eq!(bump(&[0.1, 0.1, 0.1], 0.2), 1.0);
    assert_eq!(bump(&[0.0, 0.3, 0.0], 0.2), 0.0);
    assert_eq!((ramp(-1.0), ramp(0.0), ramp(1.0)), (0.0, 0.5, 1.0));
}

#[test]
fn layers_held_together_by_booleans_make_bodies() {
    let tags = |s: &Stack| s.bodies().iter().map(|m| gf_core::layers::tag(*m)).collect::<Vec<_>>();
    let mut s = Stack::new();
    assert_eq!(s.bodies(), [0b1]);
    for _ in 1..4 {
        s.add();
    }
    // every pair joined (as they start): one body
    assert_eq!(tags(&s), ["1234"]);
    // one layer kept apart from all the others is a body by itself
    for m in [0b1001, 0b1010, 0b1100] {
        s.pair_mut(m).unwrap().op = Op::Apart;
    }
    assert_eq!(tags(&s), ["123", "4"]);
    // any operation but apart holds two layers together: a cut, an overlap, a morph
    for (i, op) in [Op::ACutsB, Op::Overlap, Op::Morph, Op::BCutsA].into_iter().enumerate() {
        s.pair_mut(0b1100).unwrap().op = op;
        assert_eq!(tags(&s), ["1234"], "{i}");
    }
    s.pair_mut(0b1100).unwrap().op = Op::Apart;
    // two that are apart from each other but both joined to a third are one body, through it
    s.pair_mut(0b0011).unwrap().op = Op::Apart;
    assert_eq!(tags(&s), ["123", "4"]);
    s.pair_mut(0b0101).unwrap().op = Op::Apart;
    assert_eq!(tags(&s), ["1", "23", "4"]);
    // bodies come in order of their lowest layer; hiding a layer changes nothing
    s.pair_mut(0b1001).unwrap().op = Op::Join;
    assert_eq!(tags(&s), ["14", "23"]);
    s.layers[0].show = false;
    assert_eq!(tags(&s), ["14", "23"]);
    let c = s.combine();
    assert_eq!((c.bodies, c.body, c.bodies_shown()), (2, [0, 1, 1, 0], 2));
    s.layers[3].show = false;
    assert_eq!(s.combine().bodies_shown(), 1);
}

#[test]
fn bodies_come_out_right_for_every_arrangement() {
    // every way of keeping pairs apart or not, for two, three and four layers (2 + 8 + 64 arrangements),
    // against a plain search for which layers can be reached from which
    let joined = [Op::Join, Op::ACutsB, Op::BCutsA, Op::Overlap, Op::Morph];
    let mut count = 0;
    for n in 2..=4usize {
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
        for pattern in 0..(1u32 << pairs.len()) {
            let mut s = Stack::new();
            for _ in 1..n {
                s.add();
            }
            let mut linked = [[false; 4]; 4];
            for (i, (a, b)) in pairs.iter().enumerate() {
                let together = pattern & (1 << i) != 0;
                // (any operation but apart holds two layers together: each in turn)
                s.pair_mut(mask_of(&[*a, *b])).unwrap().op = if together { joined[(i + pattern as usize) % joined.len()] } else { Op::Apart };
                linked[*a][*b] = together;
                linked[*b][*a] = together;
            }
            // which layers each layer reaches, by walking the links
            let mut want: Vec<Vec<usize>> = Vec::new();
            let mut seen = [false; 4];
            for start in 0..n {
                if seen[start] { continue; }
                let mut body = vec![start];
                seen[start] = true;
                let mut at = 0;
                while at < body.len() {
                    for j in 0..n {
                        if linked[body[at]][j] && !seen[j] {
                            seen[j] = true;
                            body.push(j);
                        }
                    }
                    at += 1;
                }
                body.sort_unstable();
                want.push(body);
            }
            let got: Vec<Vec<usize>> = s.bodies().iter().map(|m| gf_core::layers::members(*m)).collect();
            assert_eq!(got, want, "{n} layers, arrangement {pattern:b}");
            // in order of their lowest layer, every layer in exactly one, and the numbers the combining step is given agree
            assert!(got.windows(2).all(|w| w[0][0] < w[1][0]));
            let c = s.combine();
            assert_eq!(c.bodies, want.len());
            for (k, body) in want.iter().enumerate() {
                for i in body {
                    assert_eq!(c.body[*i], k);
                }
            }
            // a pair's seam never crosses from one body into another
            for a in 0..n {
                for b in 0..n {
                    if c.body[a] != c.body[b] { assert_eq!((c.ku[a][b], c.kc[a][b], c.term[a][b], if a == b { 0.0 } else { c.morph[a][b] }), (0.0, 0.0, 0.0, 0.0)); }
                }
            }
            count += 1;
        }
    }
    assert_eq!(count, 2 + 8 + 64);
}

#[test]
fn bodies_do_nothing_to_each_others_shape() {
    let mut rng = Rng(0x5151515151515151);
    for round in 0..4000 {
        // layers 1 and 2 joined, cut or overlapped; layer 3 (and 4) apart from both
        let n = 3 + round % 2;
        let mut s = random_stack(&mut rng, n, &[Op::Join, Op::ACutsB, Op::Overlap, Op::Morph], true);
        for m in s.combos() {
            if m.count_ones() == 2 && m != 0b0011 && m != 0b1100 { s.pair_mut(m).unwrap().op = Op::Apart; }
        }
        for l in s.layers.iter_mut() {
            l.show = true;
        }
        let c = s.combine();
        let mut d = [0.0; 4];
        for v in d.iter_mut().take(n) {
            *v = rng.range(-0.3, 0.3);
        }
        let b = c.eval_bodies(&d);
        // the first body is what its two layers make by themselves
        let mut pair = Stack::new();
        pair.add();
        *pair.pair_mut(0b11).unwrap() = s.pair(0b0011);
        assert_eq!(b[0], pair.combine().eval(&[d[0], d[1], 0.0, 0.0]), "{s:?}");
        // and stays the same wherever the others are
        let mut e = d;
        e[2] = rng.range(-0.3, 0.3);
        e[3] = rng.range(-0.3, 0.3);
        assert_eq!(c.eval_bodies(&e)[0], b[0]);
        // the scene is the nearest of the bodies; places beyond the last body read as far as can be
        assert_eq!(c.eval(&d), b.iter().take(c.bodies).fold(f64::MAX, |m, v| m.min(*v)));
        assert_eq!(c.bodies, if n == 3 { 2 } else if s.pair(0b1100).op == Op::Apart { 3 } else { 2 });
        for v in &b[c.bodies..] {
            assert_eq!(*v, gf_core::layers::ABSENT);
        }
        // cut to the stage, the scene's volume is exactly the smallest channel of the bodies' volume
        let p = [rng.range(-1.5, 1.5), rng.range(-1.5, 1.5), rng.range(-1.5, 1.5)];
        let at = c.eval_bodies_at(&d, p);
        assert_eq!(c.eval_at(&d, p), at.iter().fold(f64::MAX, |m, v| m.min(*v)));
    }
}

#[test]
fn the_stage_cuts_only_what_leaves_it() {
    // a surface (and anything near it) further than the rounding inside the stage's edge is exactly what it was
    for (f, r) in [(-0.2, 0.5), (0.0, 1.3), (0.3, 1.0), (0.0, 1.41), (-0.02, 1.38)] {
        assert_eq!(stage(f, [r, 0.0, 0.0]), f);
    }
    // glass that reaches beyond the stage ends on its ball
    assert!((stage(-0.5, [STAGE_R + 0.1, 0.0, 0.0]) - 0.1).abs() < 1e-12);
    assert_eq!(stage(-0.5, [0.0, STAGE_R, 0.0]), 0.0);
}

#[test]
fn outside_its_box_a_layer_reads_no_further_than_its_shape_is() {
    // a ball of radius 0.9 in a 48-voxel box, read from points in and around the box: never more than
    // the true distance (by more than the grid's own error), and exact at a voxel's centre
    let res = 48usize;
    let ball = |i: usize, j: usize, k: usize| {
        let c = |v: usize| ((v as f64 + 0.5) / res as f64 * 2.0 - 1.0) * 1.5;
        (c(i) * c(i) + c(j) * c(j) + c(k) * c(k)).sqrt() - 0.9
    };
    let mut rng = Rng(0xabcdef0123456789);
    for _ in 0..4000 {
        let q = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
        let truth = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - 0.9;
        let got = read_box(&ball, res, q);
        // (near the ball's centre the grid itself cannot follow the distance: left out)
        // (the B-spline near the surface averages a convex distance, so it reads a little over it: the
        // smoothing's own error, a few thousandths at 48 voxels, under what a march's step scale leaves)
        assert!(truth < -0.4 || got <= truth + 4e-3, "{got} at {q:?}, where the ball is {truth} away");
        // and not uselessly small: at least the distance to the box
        let out = q.iter().map(|v| (v.abs() - 1.5).max(0.0).powi(2)).sum::<f64>().sqrt();
        assert!(out == 0.0 || got >= out - 1e-9);
    }
    let c = |v: usize| ((v as f64 + 0.5) / res as f64 * 2.0 - 1.0) * 1.5;
    // exact at a voxel's centre away from the surface; near it (within 18 voxels) read through the
    // B-spline, as the combining shader reads it
    assert!((read_box(&ball, res, [c(0), c(0), c(0)]) - ball(0, 0, 0)).abs() < 1e-12);
    let near = [c(7), c(30), c(22)];
    assert!((read_box(&ball, res, near) - gf_core::layers::read_cubic(&ball, res, near)).abs() < 1e-12);
}
