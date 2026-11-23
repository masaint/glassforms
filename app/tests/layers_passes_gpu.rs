//! Scenes of more layers than one combining pass takes: the multi-pass shader against the reference
//! (`Stack::pass_plan`, each pass's `Combine::eval_bodies_at`, bodies to their channels, the nearest).

use gf_core::layers::{mask_of, read_box, Op, Stack, MAX_LAYERS};

use gf_core::HB;
use glassforms::bake::Volume;
use glassforms::combine::{multi_alone_wgsl, multi_wgsl, Combiner};
use glassforms::gpu::Gpu;
use half::f16;

fn centre(i: usize, res: usize) -> f64 {
    ((i as f64 + 0.5) / res as f64 * 2.0 - 1.0) * HB
}

fn ball_volume(res: usize, c: [f64; 3], r: f64) -> Vec<f16> {
    let mut v = Vec::with_capacity(res * res * res);
    for k in 0..res {
        for j in 0..res {
            for i in 0..res {
                let p = [centre(i, res), centre(j, res), centre(k, res)];
                let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() - r;
                v.push(f16::from_f64(d.clamp(-1.0, 4.0)));
            }
        }
    }
    v
}

#[test]
fn the_multi_pass_shader_is_valid() {
    let (src, alone) = (multi_wgsl(), multi_alone_wgsl());
    for (name, text) in [("multi", src.as_str()), ("multi (drawn)", src.split("//#compute").next().unwrap()),
                         ("multi alone", alone.as_str()), ("multi alone (drawn)", alone.split("//#compute").next().unwrap())] {
        let module = naga::front::wgsl::parse_str(text).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(text)));
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(text)));
    }
}

#[test]
fn passes_agree_with_the_reference() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let res = 40usize;
    // (spread: the layers set far apart, so that most voxels leave most passes and layers out; bodies:
    // the bodies' volume written too, so that every channel must stay right, not only the nearest;
    // joined: two bodies of joined layers, else every layer alone)
    for (n, spread, bodies, joined) in [(5, 1.0, true, true), (6, 1.0, true, true), (8, 1.0, true, true), (5, 4.0, false, true),
                                        (8, 4.0, true, true), (5, 1.0, false, false), (8, 4.0, false, false), (8, 1.0, true, false)] {
        assert!(n <= MAX_LAYERS);
        let mut s = Stack::new();
        s.layers = vec![Default::default(); n];
        // two bodies of joined layers, the rest apart; every layer moved somewhere of its own
        for a in 0..n {
            for b in a + 1..n {
                let p = s.pair_mut(mask_of(&[a, b])).unwrap();
                p.op = if joined && ((a < 3 && b < 3) || (a == 3 && b == 4)) { Op::Join } else { Op::Apart };
                p.radius = 0.12;
            }
        }
        for (i, l) in s.layers.iter_mut().enumerate() {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            l.place.pos = [0.6 * spread * t.cos(), 0.6 * spread * t.sin(), 0.1 * i as f64 - 0.3];
            l.place.size = 0.5;
        }
        let data: Vec<Vec<f16>> = (0..n).map(|i| ball_volume(res, [0.0, 0.0, 0.0], 0.5 + 0.05 * (i % 3) as f64)).collect();
        let vols: Vec<Volume> = data.iter().map(|d| Volume::from_data(&gpu, res as u32, d)).collect();
        let refs: Vec<&Volume> = vols.iter().collect();
        let wides = vec![None; n];
        let plan = s.pass_plan(&s);
        assert!(plan.len() >= 2);
        let mut comb = Combiner::new(&gpu);
        comb.bodies_wanted = bodies;
        comb.run_passes(&refs, &wides, &plan, &[], 4, res as u32).unwrap();
        assert_eq!(comb.bodies_volume().is_some(), bodies);
        let vol = comb.volume().unwrap();
        let mut got = Vec::new();
        for z in 0..vol.res {
            let bytes = gpu.read(&vol.tex, 0, 0, z, vol.res, vol.res, 2).unwrap();
            got.extend(bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|b| f16::from_bits(*b).to_f64()));
        }
        // the reference: each pass by the one-pass combining step (combine.wgsl as it is), the nearest kept
        let read = |v: &Volume| {
            let mut out = Vec::new();
            for z in 0..v.res {
                let bytes = gpu.read(&v.tex, 0, 0, z, v.res, v.res, 2).unwrap();
                out.extend(bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|b| f16::from_bits(*b).to_f64()));
            }
            out
        };
        let read4 = |v: &Volume, channels: u32| {
            let mut out = Vec::new();
            for z in 0..v.res {
                let bytes = gpu.read(&v.tex, 0, 0, z, v.res, v.res, 2 * channels).unwrap();
                out.extend(bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|b| f16::from_bits(*b).to_f64()));
            }
            out
        };
        let voxels = res * res * res;
        let mut want = vec![f64::MAX; voxels];
        let mut want_b = vec![4.0f64; voxels * 4];
        let mut one = Combiner::new(&gpu);
        for pass in &plan {
            let local: Vec<&Volume> = pass.layers.iter().map(|&g| &vols[g]).collect();
            one.run(&local, &vec![None; local.len()], &pass.combine, res as u32).unwrap();
            let scene = read(one.volume().unwrap());
            for (w, v) in want.iter_mut().zip(&scene) { *w = w.min(*v); }
            // each of the pass's bodies to its channel (a pass of one body: the scene's volume is that body)
            match one.bodies_volume() {
                Some(b) => {
                    let got_b = read4(b, one.bodies_channels());
                    for (k, &ch) in pass.chan.iter().enumerate() {
                        if ch >= 4 || k >= one.bodies_channels() as usize { continue; }
                        for v in 0..voxels { want_b[v * 4 + ch as usize] = want_b[v * 4 + ch as usize].min(got_b[v * one.bodies_channels() as usize + k]); }
                    }
                }
                None => {
                    let ch = pass.chan[0] as usize;
                    for v in 0..voxels { want_b[v * 4 + ch] = want_b[v * 4 + ch].min(scene[v]); }
                }
            }
        }
        let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
        assert_eq!(worst, 0.0, "{n} layers (spread {spread}): the passes differ from the passes made one by one by up to {worst}");
        if bodies {
            let b = comb.bodies_volume().unwrap();
            assert_eq!(comb.bodies_channels(), 4);
            let got_b = read4(b, 4);
            let channels = s.body_channels();
            let mut worst_b = 0.0f64;
            for v in 0..voxels {
                for ch in 0..channels { worst_b = worst_b.max((got_b[v * 4 + ch] - want_b[v * 4 + ch]).abs()); }
            }
            assert_eq!(worst_b, 0.0, "{n} layers (spread {spread}): the bodies differ from the passes made one by one by up to {worst_b}");
        }
        // and the reference arithmetic agrees in the large (the shader reads near a surface more smoothly)
        let mut far = 0.0f64;
        for k in (0..res).step_by(5) {
            for j in (0..res).step_by(5) {
                for i in (0..res).step_by(5) {
                    let p = [centre(i, res), centre(j, res), centre(k, res)].map(|x| x * s.box_now());
                    let mut d_all = vec![0.0; n];
                    for pass in &plan {
                        for (l, &g) in pass.layers.iter().enumerate() {
                            let v = &data[g];
                            let at = |a: usize, b: usize, e: usize| v[(e * res + b) * res + a].to_f64();
                            d_all[g] = pass.combine.scale[l] * read_box(&at, res, pass.combine.to_layer(l, p));
                        }
                    }
                    let _ = &d_all;
                    let cpu = s.eval_passes(&d_all);
                    let g = got[(k * res + j) * res + i];
                    if g < 0.3 && g > -0.3 { far = far.max((g - gf_core::layers::stage_at(cpu, p, gf_core::layers::STAGE_R * s.box_now()).clamp(-1.0, 4.0)).abs()); }
                }
            }
        }
        // (placed layers are read between voxels with 8-bit weights: a few ten-thousandths, as layers_gpu.rs)
        assert!(far < 2e-2, "{n} layers: {far} from the reference arithmetic");
    }
}
