//! A grid part is only as safe to march through as the volume it is baked into: a step is safe when the value
//! read never says the surface is further than it is (nor puts a point on the wrong side). Union, subtract and
//! intersect (min, max and a change of sign), and their rounded seams (whose slope is a share of the two
//! slopes), all keep that: so the tree is safe wherever its parts' volumes are. These check the volumes the
//! bakers make against the exact distance, at many voxels: a tube (the curve baker, near its surface and in
//! the far field it fills from a coarse grid) and a ball read from a file (the model baker).
//!
//! Needs a GPU (through the queue).

use gf_core::state::{Detail, ShapeJob, State};
use gf_core::HB;
use glassforms::engine::Engine;
use glassforms::gpu::Gpu;

fn engine(gpu: &Gpu, pairs: &[(&str, &str)]) -> Engine {
    let args: Vec<(String, String)> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let mut e = Engine::new(gpu.clone(), State::with_args(&args, true), true);
    e.resize(64, 36);
    e.bake(Detail::Full).unwrap();
    e.state.want = None;
    e
}

/// The volume as floats, and its voxels a side.
fn values(e: &Engine) -> (usize, Vec<f32>) {
    let (res, bits) = e.read_volume().unwrap();
    (res as usize, bits.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect())
}

fn at(i: usize, res: usize) -> f64 {
    ((i as f64 + 0.5) / res as f64 * 2.0 - 1.0) * HB
}

/// Over every `stride`th voxel: the most a value says the surface is further than `exact` (in voxels), how many
/// voxels further than a voxel from the surface are on the wrong side, and the most the value is of the distance
/// further than two voxels from the surface (what a step taken by the value must be scaled down by to stay short).
fn worst(res: usize, v: &[f32], stride: usize, exact: impl Fn([f64; 3]) -> f64) -> (f64, usize, usize, f64) {
    let h = 2.0 * HB / res as f64;
    let (mut over, mut wrong, mut n, mut ratio) = (0.0f64, 0usize, 0usize, 0.0f64);
    for z in (0..res).step_by(stride) {
        for y in (0..res).step_by(stride) {
            for x in (0..res).step_by(stride) {
                let d = exact([at(x, res), at(y, res), at(z, res)]);
                let val = v[(z * res + y) * res + x] as f64;
                n += 1;
                // (a value more than the distance outside, or more negative than it inside: too long a step)
                let o = if d >= 0.0 { val - d } else { d - val };
                // (half floats round: a thousandth of the value is allowed)
                over = over.max((o - 1e-3 * d.abs()) / h);
                if d.abs() > h && (val > 0.0) != (d > 0.0) { wrong += 1; }
                if d.abs() > 2.0 * h { ratio = ratio.max(val / d); }
            }
        }
    }
    (over, wrong, n, ratio)
}

#[test]
fn a_tube_volume_never_says_the_surface_is_further_than_it_is() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        eprintln!("no graphics device: skipped");
        return;
    };
    // a knot of one radius, not melted (a melt is a smooth union, its surface not the tube's), at two sizes
    for res in ["128", "256"] {
        let mut e = engine(&gpu, &[("scene", "knots"), ("preset", "trefoil"), ("res", res), ("melt", "0")]);
        let ShapeJob::Curve { curve: c, .. } = e.state.shape_job(Detail::Full, None).unwrap().shape else { panic!("a curve") };
        let segs: Vec<([f64; 3], [f64; 3], f64)> = (0..c.n - 1).filter(|&i| c.brk[i] == 0).map(|i| {
            let p = |k: usize| [c.pts[k * 4] as f64, c.pts[k * 4 + 1] as f64, c.pts[k * 4 + 2] as f64];
            (p(i), p(i + 1), (c.pts[i * 4 + 3] as f64).max(c.pts[i * 4 + 7] as f64))
        }).collect();
        let r0 = segs[0].2;
        assert!(segs.iter().all(|s| (s.2 - r0).abs() < 1e-6), "the trefoil's tube is of one radius");
        let exact = |q: [f64; 3]| -> f64 {
            segs.iter().map(|(a, b, r)| {
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let aq = [q[0] - a[0], q[1] - a[1], q[2] - a[2]];
                let t = ((aq[0] * ab[0] + aq[1] * ab[1] + aq[2] * ab[2]) / (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).max(1e-30)).clamp(0.0, 1.0);
                let d = [aq[0] - t * ab[0], aq[1] - t * ab[1], aq[2] - t * ab[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - r
            }).fold(f64::MAX, f64::min)
        };
        let (res_n, v) = values(&e);
        let (over, wrong, n, ratio) = worst(res_n, &v, 3, exact);
        eprintln!("trefoil at {res}: at {n} voxels the value is at most {over:.3} voxel further than the surface, at most {ratio:.4} of the distance; {wrong} on the wrong side");
        assert!(ratio <= 1.001, "{ratio}");
        assert_eq!(wrong, 0);
        // (the curve is cut into steps of three quarters of a voxel: between its points the tube is a chain of
        // capsules, which bulges outside the smooth tube by less than a hundredth of a voxel)
        assert!(over < 0.05, "a value says the surface is {over:.3} voxel further than it is");
    }
}

#[test]
fn a_model_volume_never_says_the_surface_is_further_than_it_is() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        eprintln!("no graphics device: skipped");
        return;
    };
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("out").join("models").join("grid_safe");
    std::fs::create_dir_all(&dir).unwrap();
    // a ball of many facets (inscribed: its facets lie within a sagitta of the round ball)
    let n = 192;
    let rows = n / 2;
    let mut s = String::new();
    for j in 0..=rows {
        let th = std::f64::consts::PI * j as f64 / rows as f64;
        for i in 0..n {
            let ph = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            s += &format!("v {} {} {}\n", th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
        }
    }
    for j in 0..rows {
        for i in 0..n {
            let (a, b) = (j * n + i + 1, j * n + (i + 1) % n + 1);
            let (c, d) = (a + n, b + n);
            if j > 0 { s += &format!("f {a} {b} {c}\n"); }
            if j < rows - 1 { s += &format!("f {b} {d} {c}\n"); }
        }
    }
    let path = dir.join("ball.obj");
    std::fs::write(&path, s).unwrap();
    let path = path.to_string_lossy().replace('\\', "/");
    let e = engine(&gpu, &[("scene", "models"), ("preset", "file"), ("model", &path), ("res", "192")]);
    let (res, v) = values(&e);
    // The file is fitted to the volume: its radius there is where the value along the axes crosses zero.
    let h = 2.0 * HB / res as f64;
    let mid = res / 2;
    let cross = |line: &dyn Fn(usize) -> f32| -> f64 {
        let k = (mid..res - 1).find(|&k| line(k) <= 0.0 && line(k + 1) > 0.0).expect("the ball's surface");
        let (a, b) = (line(k) as f64, line(k + 1) as f64);
        at(k, res) + h * a / (a - b)
    };
    let rx = cross(&|k| v[(mid * res + mid) * res + k]);
    let ry = cross(&|k| v[(mid * res + k) * res + mid]);
    let r = 0.5 * (rx + ry);
    // (the facets: between the round ball and its inscribed polyhedron, the inradius cos(pi/n) of it)
    let sag = r * (1.0 - (std::f64::consts::PI / rows as f64).cos());
    // Outside, the distance to the facets is at most |p| - r (1 - sag): a value more than that says too much.
    // Inside it is at least as deep as |p| - r: a value deeper than that says too much.
    let (over_out, wrong_out, _, ratio) = worst(res, &v, 3, |q| {
        let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        if l >= r { l - (r - sag) } else { l - r }
    });
    eprintln!("ball (radius {r:.4} in the volume, facets within {:.4} voxel): at most {over_out:.3} voxel further than the surface, at most {ratio:.4} of the distance; {wrong_out} on the wrong side", sag / h);
    assert_eq!(wrong_out, 0);
    // The model baker's far field (read from a coarse grid) may say a little more than the distance: about a
    // voxel, a few hundredths of it. Its family marches by 0.85 of what it reads, which covers that; so must
    // any tree with a model in it (the tree marches as its most careful part's family does).
    let models = gf_core::presets::families().into_iter().find(|f| f.id == "models").unwrap().step_scale;
    assert!(ratio * models < 0.98, "a step of {models} of the value can pass the surface: the value is {ratio:.4} of the distance");
}
