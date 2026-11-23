//! The combining step on the GPU against the same arithmetic in plain Rust (gf_core::layers): known shapes
//! (balls, a box) are put up as layers' volumes, combined with a spread of options, read back, and every
//! voxel compared with what the reference gives for the same stored inputs.
//!
//! Where a layer is read exactly at its own voxels (not moved, same size of volume) the two must agree to
//! the last place of the 16-bit number stored (and three parts in ten million, the 32-bit arithmetic's own). Where a layer is moved, the GPU reads between voxels with
//! weights of 8 bits, so the agreement is to a few ten-thousandths.

use gf_core::layers::{read_layer, Combine, Op, Opt, Placement, Profile, Stack, WideUse};
use gf_core::HB;
use glassforms::bake::Volume;
use glassforms::combine::Combiner;
use glassforms::gpu::Gpu;
use half::f16;

fn centre(i: usize, res: usize) -> f64 {
    ((i as f64 + 0.5) / res as f64 * 2.0 - 1.0) * HB
}

/// A shape's distances at every voxel, as the 16-bit numbers a volume stores.
fn volume_of(res: usize, sd: &dyn Fn([f64; 3]) -> f64) -> Vec<f16> {
    let mut v = Vec::with_capacity(res * res * res);
    for k in 0..res {
        for j in 0..res {
            for i in 0..res {
                v.push(f16::from_f64(sd([centre(i, res), centre(j, res), centre(k, res)]).clamp(-1.0, 4.0)));
            }
        }
    }
    v
}

fn ball(c: [f64; 3], r: f64) -> impl Fn([f64; 3]) -> f64 {
    move |p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() - r
}
fn cuboid(half: [f64; 3]) -> impl Fn([f64; 3]) -> f64 {
    move |p| {
        let q = [p[0].abs() - half[0], p[1].abs() - half[1], p[2].abs() - half[2]];
        let out = (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2) + q[2].max(0.0).powi(2)).sqrt();
        out + q[0].max(q[1]).max(q[2]).min(0.0)
    }
}

fn read_back(gpu: &Gpu, vol: &Volume) -> Vec<f16> {
    let mut out = Vec::with_capacity((vol.res as usize).pow(3));
    for z in 0..vol.res {
        let bytes = gpu.read(&vol.tex, 0, 0, z, vol.res, vol.res, 2).unwrap();
        out.extend(bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|b| f16::from_bits(*b)));
    }
    out
}

/// The shader works in 32 bits and the reference in 64: sums of numbers up to 1 in size can differ by a
/// few parts in ten million, which near zero is more than the last place of the 16-bit number stored.
const SLACK: f64 = 3e-7;

/// The gap between a 16-bit number and its neighbours.
fn ulp16(x: f64) -> f64 {
    let b = f16::from_f64(x.abs()).to_bits();
    (f16::from_bits(b + 1).to_f64() - f16::from_bits(b).to_f64()).abs()
}

/// Combine on the GPU and in Rust; the largest difference, in units of the stored number's last place
/// (plus `slack`, an absolute allowance), and the largest in absolute terms.
fn compare(gpu: &Gpu, comb: &mut Combiner, data: &[(usize, Vec<f16>)], c: &Combine, res: usize, slack: f64) -> (f64, f64) {
    compare_wide(gpu, comb, data, &[], c, res, slack)
}

/// The same, with a wide volume for some of the layers (`wide[i]`: its voxels and how it is used).
fn compare_wide(gpu: &Gpu, comb: &mut Combiner, data: &[(usize, Vec<f16>)], wide: &[Option<(Vec<f16>, WideUse)>], c: &Combine, res: usize, slack: f64) -> (f64, f64) {
    let vols: Vec<Volume> = data.iter().map(|(r, d)| Volume::from_data(gpu, *r as u32, d)).collect();
    let refs: Vec<&Volume> = vols.iter().collect();
    let wvols: Vec<Option<(Volume, WideUse)>> = wide.iter().map(|w| w.as_ref().map(|(d, u)| (Volume::from_data(gpu, u.res, d), *u))).collect();
    let wrefs: Vec<Option<(&Volume, WideUse)>> = wvols.iter().map(|w| w.as_ref().map(|(v, u)| (v, *u))).collect();
    comb.run(&refs, &wrefs, c, res as u32).unwrap();
    let got = read_back(gpu, comb.volume().unwrap());
    let (mut worst, mut worst_abs) = (0.0f64, 0.0f64);
    for k in 0..res {
        for j in 0..res {
            for i in 0..res {
                // (a stage grown for layers placed far out spans box_k times the usual box)
                let p = [centre(i, res), centre(j, res), centre(k, res)].map(|x| x * c.box_k);
                let mut d = [0.0; 4];
                for l in 0..c.n {
                    let (r, v) = &data[l];
                    let at = |a: usize, b: usize, e: usize| v[(e * r + b) * r + a].to_f64();
                    d[l] = c.scale[l] * match wide.get(l).and_then(|w| w.as_ref()) {
                        Some((wv, u)) => {
                            let wr = u.res as usize;
                            let wat = |a: usize, b: usize, e: usize| wv[(e * wr + b) * wr + a].to_f64();
                            read_layer(&at, *r, Some((&wat, *u)), c.to_layer(l, p))
                        }
                        None => read_layer(&at, *r, None, c.to_layer(l, p)),
                    };
                }
                let want = c.eval_at(&d, p);
                let g = got[(k * res + j) * res + i].to_f64();
                let err = (g - want).abs();
                // Near a surface the shader reads each layer through a cubic B-spline built from the
                // hardware's linear taps, whose weights carry 8 bits below the texel (as the APIs allow):
                // it cannot match the f64 reference closer than about 1/256 of a texel's step. A 32nd of a
                // voxel is allowed there; everywhere else the reference holds to the last place. (The edge of
                // the band is decided on each side by a value read at its own precision, so it is widened
                // by a tenth: on a voxel at the edge one side may read cubic and the other linear.)
                let cubic = (0..c.n).any(|l| d[l].abs() < 1.1 * 18.0 * HB / data[l].0 as f64);
                let slack = if cubic { slack.max(2.0 * HB / data[0].0 as f64 / 32.0) } else { slack };
                if err > worst_abs && err > 3e-4 && std::env::var_os("LAYERS_DEBUG").is_some() { eprintln!("voxel {i},{j},{k} at {p:?}: got {g:.9} want {want:.9} from {d:?}; in layer 2: {:?}", c.to_layer(1.min(c.n - 1), p)); }
                worst_abs = worst_abs.max(err);
                let w = (err - slack).max(0.0) / ulp16(want.abs().max(g.abs()));
                if w > worst && std::env::var_os("LAYERS_DEBUG").is_some() { eprintln!("voxel {i},{j},{k}: got {g:.9} want {want:.9} from {d:?} ulp {:.3e}", ulp16(want.abs().max(g.abs()))); }
                worst = worst.max(w);
            }
        }
    }
    (worst, worst_abs)
}

fn stack(n: usize) -> Stack {
    let mut s = Stack::new();
    for _ in 1..n {
        s.add();
    }
    s
}

#[test]
fn the_shader_is_valid() {
    let src = include_str!("../src/shaders/combine.wgsl");
    for (name, text) in [("combine.wgsl", src), ("combine.wgsl (drawn)", src.split("//#compute").next().unwrap())] {
        let module = naga::front::wgsl::parse_str(text).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(text)));
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(text)));
    }
}

#[test]
fn the_shader_agrees_with_the_reference() {
    let gpu = Gpu::headless("vulkan").expect("a graphics device");
    let mut comb = Combiner::new(&gpu);
    let res = 48usize;
    let a = (res, volume_of(res, &ball([-0.25, 0.0, 0.0], 0.55)));
    let b = (res, volume_of(res, &ball([0.3, 0.1, 0.0], 0.5)));
    let bx = (res, volume_of(res, &cuboid([0.45, 0.3, 0.6])));
    let e = (res, volume_of(res, &ball([0.0, -0.3, 0.25], 0.45)));

    // ---- two layers, read at their own voxels: every operation, radius and profile, to the last place
    let mut count = 0;
    for (shapes, name) in [([&a, &b], "two balls"), ([&a, &bx], "a ball and a box")] {
        let data: Vec<(usize, Vec<f16>)> = shapes.iter().map(|s| (s.0, s.1.clone())).collect();
        for op in Op::ALL {
            for (radius, profile) in [(0.0, Profile::Smooth), (0.05, Profile::Smooth), (0.2, Profile::Smooth), (0.45, Profile::Smooth), (0.2, Profile::Round), (0.2, Profile::Bevel)] {
                for hide in [None, Some(0), Some(1)] {
                    let mut s = stack(2);
                    let p = s.pair_mut(0b11).unwrap();
                    p.op = op;
                    p.radius = radius;
                    p.profile = profile;
                    p.gap = 0.04;
                    p.amount = 0.3;
                    if let Some(h) = hide { s.layers[h].show = false; }
                    let (worst, abs) = compare(&gpu, &mut comb, &data, &s.combine(), res, SLACK);
                    assert!(worst <= 1.0, "{name}, {op:?} radius {radius} {profile:?} hidden {hide:?}: {worst} last places apart ({abs})");
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 216);

    // ---- three and four layers with different radii, profiles and junction fills, at their own voxels
    let all: Vec<(usize, Vec<f16>)> = vec![a.clone(), b.clone(), bx.clone(), e.clone()];
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for round in 0..40 {
        let n = 3 + round % 2;
        let mut s = stack(n);
        for m in s.combos() {
            if m.count_ones() == 2 {
                let p = s.pair_mut(m).unwrap();
                p.op = Op::ALL[(next() * 6.0) as usize % 6];
                p.radius = if next() < 0.2 { 0.0 } else { 0.03 + 0.4 * next() };
                p.profile = Profile::ALL[(next() * 3.0) as usize % 3];
                p.gap = 0.1 * next();
                p.amount = next();
            } else {
                s.set_opt(m, Opt::Fill, 2.0 * next() - 1.0);
                s.set_opt(m, Opt::Reach, 0.05 + 0.4 * next());
            }
        }
        for l in s.layers.iter_mut() {
            l.show = next() > 0.2;
        }
        let (worst, abs) = compare(&gpu, &mut comb, &all[..n], &s.combine(), res, SLACK);
        // long sums in 32 bits against 64: the odd voxel rounds the other way
        assert!(worst <= 2.0, "round {round}, {n} layers: {worst} last places apart ({abs})\n{s:?}");
    }

    // ---- layers that are moved, turned and resized, and whose volumes are a different size from the
    // one written: read between voxels, so to the precision of the GPU's 8-bit weights
    let fine = (72usize, volume_of(72, &cuboid([0.45, 0.3, 0.6])));
    for (i, place) in [
        Placement { pos: [0.2, -0.1, 0.15], rot: [0.0, 0.0, 0.0], size: 1.0 },
        Placement { pos: [0.0, 0.0, 0.0], rot: [20.0, -35.0, 60.0], size: 1.0 },
        Placement { pos: [-0.3, 0.2, 0.1], rot: [70.0, 15.0, -40.0], size: 0.6 },
        Placement { pos: [0.9, 0.9, -0.8], rot: [10.0, 0.0, 0.0], size: 1.7 },
    ]
    .into_iter()
    .enumerate()
    {
        let mut s = stack(2);
        s.layers[1].place = place;
        s.pair_mut(0b11).unwrap().radius = 0.15;
        s.pair_mut(0b11).unwrap().op = [Op::Join, Op::BCutsA, Op::Overlap, Op::Join][i];
        let data = vec![a.clone(), fine.clone()];
        // a voxel of the layer is 0.04 across (times its size): 8-bit weights place a reading to 1/512 of
        // that along each axis, 0.00035 in all with the box's own voxels; then the last places of the number stored
        let (worst, abs) = compare(&gpu, &mut comb, &data, &s.combine(), res, 3.5e-4 * place.size);
        assert!(worst <= 2.0, "placement {i}: {worst} last places beyond the weights' precision ({abs})");
    }
    // one layer alone, moved: the same
    let mut s = stack(1);
    s.layers[0].place = Placement { pos: [0.3, 0.0, -0.2], rot: [0.0, 45.0, 0.0], size: 0.8 };
    let (worst, abs) = compare(&gpu, &mut comb, &[bx.clone()], &s.combine(), res, 3e-4);
    assert!(worst <= 2.0, "one layer moved: {worst} last places beyond the weights' precision ({abs})");

    // ---- a layer with a wide volume: its own volume is right only close to its surface (a plateau beyond,
    // as a tube's is), the wide volume is right everywhere. The shader reads the two as the reference does,
    // and the seam comes out as it would from a volume that is right everywhere.
    let use_wide = WideUse { from: 0.04, to: 0.09, res: 40 };
    let near_only = (res, volume_of(res, &|p| ball([-0.25, 0.0, 0.0], 0.55)(p).min(0.1)));
    let far = volume_of(40, &ball([-0.25, 0.0, 0.0], 0.55));
    let mut s = stack(2);
    s.pair_mut(0b11).unwrap().radius = 0.35;
    for moved in [false, true] {
        if moved { s.layers[0].place = Placement { pos: [0.1, -0.05, 0.0], rot: [0.0, 20.0, 0.0], size: 0.9 }; }
        let c = s.combine();
        let (worst, abs) = compare_wide(&gpu, &mut comb, &[near_only.clone(), b.clone()], &[Some((far.clone(), use_wide)), None], &c, res, 1.2e-3);
        // the wide volume's voxels are 0.075 across, read with 8-bit weights at eight places
        assert!(worst <= 2.0, "a wide volume (moved: {moved}): {worst} last places beyond the weights' precision ({abs})");
        // against the same seam made from a volume that is right everywhere: the same to the wide volume's own fineness
        let with_wide = read_back(&gpu, comb.volume().unwrap());
        let vols = [Volume::from_data(&gpu, res as u32, &a.1), Volume::from_data(&gpu, res as u32, &b.1)];
        comb.run(&[&vols[0], &vols[1]], &[], &c, res as u32).unwrap();
        let whole = read_back(&gpu, comb.volume().unwrap());
        // (where there is anything to see: within 0.5 of the surface. At the edge of its box a smooth
        // reading has no neighbours to read and is off by a third of a voxel.)
        let near = |y: &f16| y.to_f64().abs() < 0.5;
        let apart = with_wide.iter().zip(&whole).filter(|(_, y)| near(y)).map(|(x, y)| (x.to_f64() - y.to_f64()).abs()).fold(0.0, f64::max);
        assert!(apart < 6e-3, "a wide volume against a whole one: {apart} apart");
        // and without it, the plateau shows: the seam is several times as far out
        let vols = [Volume::from_data(&gpu, res as u32, &near_only.1), Volume::from_data(&gpu, res as u32, &b.1)];
        comb.run(&[&vols[0], &vols[1]], &[], &c, res as u32).unwrap();
        let without = read_back(&gpu, comb.volume().unwrap());
        let off = without.iter().zip(&whole).filter(|(_, y)| near(y)).map(|(x, y)| (x.to_f64() - y.to_f64()).abs()).fold(0.0, f64::max);
        assert!(off > 4.0 * apart, "without the wide volume: only {off} apart");
    }

    // ---- bodies: layers kept apart are bodies of their own, each with a channel of the bodies' volume;
    // the scene's volume is the smallest of those channels, bit for bit. Two bodies take a volume of two
    // channels, three or four one of four (where a channel with no body reads as far as a volume can).
    for three in [false, true] {
        let mut s = stack(4);
        for m in s.combos() {
            if m.count_ones() == 2 {
                s.pair_mut(m).unwrap().op = if m == 0b0011 { Op::Join } else { Op::Apart };
                s.pair_mut(m).unwrap().radius = 0.2;
            }
        }
        // layers 3 and 4: one body (4 cuts 3 and is hidden), or two (the hidden one a body with nothing to show)
        if !three { s.pair_mut(0b1100).unwrap().op = Op::BCutsA; }
        s.layers[3].show = false;
        let c = s.combine();
        assert_eq!((c.bodies, c.bodies_shown()), (if three { 3 } else { 2 }, 2));
        let channels = if three { 4 } else { 2 };
        for use_compute in [true, false] {
            let g = if use_compute { gpu.clone() } else { Gpu { storage16: false, ..gpu.clone() } };
            let mut cb = Combiner::new(&g);
            let vols: Vec<Volume> = all.iter().map(|(r, d)| Volume::from_data(&gpu, *r as u32, d)).collect();
            let refs: Vec<&Volume> = vols.iter().collect();
            cb.run(&refs, &[], &c, res as u32).unwrap();
            let scene = read_back(&gpu, cb.volume().unwrap());
            let bv = cb.bodies_volume().expect("two bodies show: there is a bodies' volume");
            assert_eq!(cb.bodies_channels(), channels as u32);
            let mut bodies: Vec<f16> = Vec::with_capacity(scene.len() * channels);
            for z in 0..bv.res {
                let bytes = gpu.read(&bv.tex, 0, 0, z, bv.res, bv.res, 2 * channels as u32).unwrap();
                bodies.extend(bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|b| f16::from_bits(*b)));
            }
            let mut worst = 0.0f64;
            for k in 0..res {
                for j in 0..res {
                    for i in 0..res {
                        let at = (k * res + j) * res + i;
                        let p = [centre(i, res), centre(j, res), centre(k, res)];
                        let mut d = [0.0; 4];
                        // (each layer read as the shader reads it: through the B-spline near its surface)
                        for l in 0..4 {
                            let v = &all[l].1;
                            d[l] = read_layer(&|a: usize, b: usize, e: usize| v[(e * res + b) * res + a].to_f64(), res, None, p);
                        }
                        let want = c.eval_bodies_at(&d, p);
                        let got = &bodies[at * channels..(at + 1) * channels];
                        let cubic = d.iter().any(|v| v.abs() < 1.1 * 18.0 * HB / res as f64);
                        let slack = if cubic { SLACK.max(2.0 * HB / res as f64 / 32.0) } else { SLACK };
                        for b in 0..channels {
                            let err = (got[b].to_f64() - want[b]).abs();
                            worst = worst.max((err - slack).max(0.0) / ulp16(want[b].abs().max(got[b].to_f64().abs())));
                        }
                        // the scene's voxel is exactly the nearest body's
                        let nearest = got[..c.bodies].iter().map(|v| v.to_f32()).fold(f32::MAX, f32::min);
                        assert_eq!(scene[at].to_f32(), nearest, "voxel {i},{j},{k} (computed: {use_compute}, three bodies: {three})");
                        // a channel with no body reads as far as a volume can
                        if three { assert_eq!(got[3].to_f32(), 4.0); }
                    }
                }
            }
            assert!(worst <= 2.0, "bodies (computed: {use_compute}, three bodies: {three}): {worst} last places apart");
        }
    }
    // one body: no bodies' volume is made, and one made before is let go
    let mut one = stack(2);
    one.pair_mut(0b11).unwrap().radius = 0.1;
    let mut cb = Combiner::new(&gpu);
    let vols: Vec<Volume> = all[..2].iter().map(|(r, d)| Volume::from_data(&gpu, *r as u32, d)).collect();
    one.pair_mut(0b11).unwrap().op = Op::Apart;
    cb.run(&[&vols[0], &vols[1]], &[], &one.combine(), res as u32).unwrap();
    assert!(cb.bodies_volume().is_some());
    one.pair_mut(0b11).unwrap().op = Op::Join;
    cb.run(&[&vols[0], &vols[1]], &[], &one.combine(), res as u32).unwrap();
    assert!(cb.bodies_volume().is_none());
    // nor when it is not wanted (every body in the same material: nothing would read it)
    one.pair_mut(0b11).unwrap().op = Op::Apart;
    cb.bodies_wanted = false;
    cb.run(&[&vols[0], &vols[1]], &[], &one.combine(), res as u32).unwrap();
    assert!(cb.bodies_volume().is_none() && cb.bodies_channels() == 0);

    // ---- drawn a slice at a time (for devices whose compute shaders cannot write a 16-bit volume):
    // the same volume as computed
    let drawn_gpu = Gpu { storage16: false, ..gpu.clone() };
    let mut drawn = Combiner::new(&drawn_gpu);
    let mut s = stack(3);
    for m in s.combos() {
        if m.count_ones() == 2 { s.pair_mut(m).unwrap().radius = 0.2; }
    }
    s.pair_mut(0b101).unwrap().op = Op::BCutsA;
    let vols: Vec<Volume> = all[..3].iter().map(|(r, d)| Volume::from_data(&gpu, *r as u32, d)).collect();
    let refs: Vec<&Volume> = vols.iter().collect();
    // nothing moved: to the last place; one layer moved: to the precision of the weights it is read with
    for (moved, slack) in [(false, 0.0), (true, 3.5e-4)] {
        if moved { s.layers[2].place = Placement { pos: [0.1, 0.0, 0.2], rot: [15.0, 30.0, 0.0], size: 0.9 }; }
        let c = s.combine();
        comb.run(&refs, &[], &c, res as u32).unwrap();
        drawn.run(&refs, &[], &c, res as u32).unwrap();
        let (x, y) = (read_back(&gpu, comb.volume().unwrap()), read_back(&gpu, drawn.volume().unwrap()));
        let apart = x.iter().zip(&y).map(|(p, q)| ((p.to_f64() - q.to_f64()).abs() - slack).max(0.0) / ulp16(p.to_f64().abs().max(q.to_f64().abs()))).fold(0.0, f64::max);
        assert!(apart <= if moved { 2.0 } else { 1.0 }, "computed and drawn differ by {apart} last places (a layer moved: {moved})");
    }
}
