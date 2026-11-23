//! The parts engine's GPU side (`glassforms::parts_volume`): its volume against the app's own merged volume of the same tree, a
//! drag that ends where a fresh build would, the value never over the true distance, and every refusal a sentence.
//!
//! GPU tests: run through tools/gpu_run.py, one job at a time, 1080p-scale only (volumes of 128 to 256 voxels here).
#[path = "support/boaty.rs"]
#[allow(dead_code, unused_mut)]
mod boaty;

use gf_core::csg::Part;
use gf_core::state::{Detail, State};
use glassforms::{bake::Volume, engine::Engine, gpu::Gpu, parts_volume::{PartsVolume, NOT_ENOUGH_MEMORY}};

fn kv(k: &str, v: impl ToString) -> (String, String) { (k.to_string(), v.to_string()) }

fn read(gpu: &Gpu, v: &Volume) -> Vec<f32> {
    let mut out = Vec::with_capacity((v.res as usize).pow(3));
    for z in 0..v.res {
        let bytes = gpu.read(&v.tex, 0, 0, z, v.res, v.res, 2).unwrap();
        out.extend(bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|h| half::f16::from_bits(*h).to_f32()));
    }
    out
}

fn state(args: &[(String, String)]) -> State { State::with_args(args, true) }

/// n formula parts scattered in a 1.4 cube (the csg-perf scenes' recipe).
fn tree_parts(n: usize, size: f64, smooth: f64, subtract_every: usize) -> String {
    const KINDS: [&str; 3] = ["sphere", "block", "cylinder"];
    let mut seed: u64 = 12345;
    let mut rnd = || { seed = (seed * 1103515245 + 12345) % 2147483648; seed as f64 / 2147483648.0 };
    let mut out = Vec::new();
    for i in 0..n {
        let pos: Vec<String> = (0..3).map(|_| format!("{:.3}", (rnd() * 2.0 - 1.0) * 0.7)).collect();
        let rot: Vec<String> = (0..3).map(|_| format!("{:.1}", rnd() * 180.0)).collect();
        let op = if subtract_every > 0 && i % subtract_every == subtract_every - 1 { "subtract" } else { "union" };
        out.push(format!("studio/{}@{}@{}@{:.3}@{op}@{smooth}@0", KINDS[i % 3], pos.join(","), rot.join(","), size * (0.7 + 0.6 * rnd())));
    }
    out.join(";")
}

/// The studio shapes the engine draws, one each, standing apart and turned, smooth-joined with one cut.
fn shapes_tree() -> String {
    let p = |k: &str, pos: &str, rot: &str, size: f64, op: &str, smooth: f64| format!("studio/{k}@{pos}@{rot}@{size}@{op}@{smooth}@0");
    [
        p("washer", "0.5,0.3,0", "20,0,0", 0.5, "union", 0.0),
        p("slab", "-0.5,0.4,0.1", "0,30,10", 0.5, "union", 0.04),
        p("bandring", "0.1,-0.5,0.4", "0,0,60", 0.45, "union", 0.0),
        p("rectframe", "-0.4,-0.4,-0.3", "10,40,0", 0.5, "union", 0.03),
        p("oval", "0.4,-0.3,-0.5", "0,10,70", 0.45, "union", 0.0),
        p("ellipsoid", "0,0.6,-0.5", "30,0,0", 0.5, "union", 0.05),
        p("capsule", "0.6,0,0.5", "0,0,40", 0.5, "union", 0.0),
        p("sphere", "0.1,0.1,0.1", "0,0,0", 0.35, "subtract", 0.02),
    ].join(";")
}

fn tree_args(base: &str, csg: &str) -> Vec<(String, String)> {
    vec![kv("scene", "studio"), kv("preset", base), kv("material", "glass"), kv("dist", "4.0"), kv("csg", csg), kv("res", "128")]
}

/// boaty64: one object, the hull's first part its own shape, the rest its tree (as the lab builds it).
fn boaty64_args() -> Vec<(String, String)> {
    let (mods, _) = boaty::build_with(64, false).unwrap();
    let prims: Vec<Part> = mods.iter().flatten().cloned().collect();
    let first = &prims[0];
    let mut a = vec![kv("scene", "studio"), kv("material", "glass"), kv("dist", "5.0"), kv("rx", "16"), kv("ry", "-36"), kv("preset", &first.preset), kv("select", 1)];
    for (id, v) in &first.values { a.push(kv(&format!("L1.p.{id}"), format!("{:.6}", if boaty::is_length(id) { v * first.place.size } else { *v }))); }
    a.push(kv("L1.trace", "csg"));
    let rest: Vec<Part> = prims[1..].iter().map(|p| { let mut p = p.clone(); for k in 0..3 { p.place.pos[k] -= first.place.pos[k]; } p }).collect();
    a.push(kv("L1.csgJson", serde_json::to_string(&rest).unwrap()));
    a.push(kv("select", 1));
    a.push(kv("res", "256"));
    a
}

/// The app's own volume of the tree, as its merge makes it.
fn app_volume(gpu: &Gpu, args: &[(String, String)]) -> (Vec<f32>, u32) {
    let mut a = args.to_vec();
    a.push(kv("trace", "csg"));
    a.push(kv("csgCoarse", "1"));
    let mut e = Engine::new(gpu.clone(), state(&a), true);
    e.bake(Detail::Full).expect("the app's own bake");
    let v = e.csg.layer_vols.first().and_then(|v| v.as_ref()).map(|v| &v.1).or_else(|| e.csg.volume()).or_else(|| e.scene_volume()).expect("the app's volume");
    (read(gpu, v), v.res)
}

struct Diff { worst: f64, near: usize, over: usize, excess: f64, signs: usize, inside: usize }

/// How far `mine` is from the app's volume `app` at the surface (where ours is within `near` of it: ours is exact there), in voxels; how
/// many voxels the two disagree on being inside; and the most `mine` goes over the app's value outside (informative: the app's tree volume
/// holds bounds, not distances, away from its surface).
fn compare(app: &[f32], mine: &[f32], h: f64, near: f64) -> Diff {
    let mut d = Diff { worst: 0.0, near: 0, over: 0, excess: 0.0, signs: 0, inside: 0 };
    for (x, y) in app.iter().zip(mine) {
        let (x, y) = (*x as f64, *y as f64);
        if (x < 0.0) != (y < 0.0) { d.signs += 1; }
        if y < 0.0 { d.inside += 1; }
        if y.abs() < near {
            d.near += 1;
            let e = (x - y).abs() / h;
            if e > d.worst { d.worst = e; }
            if e > 1.0 { d.over += 1; }
        } else if x > 0.0 {
            d.excess = d.excess.max((y - x) / h);
        }
    }
    d
}

fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::headless("vulkan") { Ok(g) if g.storage16 => Some(g), _ => { eprintln!("no GPU that writes r16f volumes: skipped"); None } }
}

#[test]
fn the_parts_volume_equals_the_apps_merged_volume() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scenes: Vec<(&str, Vec<(String, String)>)> = vec![
        ("random 24", tree_args("block", &tree_parts(24, 0.4, 0.05, 6))),
        ("random 60", tree_args("sphere", &tree_parts(60, 0.3, 0.04, 8))),
        ("shapes", tree_args("block", &shapes_tree())),
        ("boaty64", boaty64_args()),
    ];
    for (name, args) in scenes {
        let (app, res) = app_volume(&gpu, &args);
        let mut st = state(&args);
        // (the parts engine at the size the app measured the tree at)
        let pv = match PartsVolume::build(&gpu, &mut st, res, None) { Ok(p) => p, Err(e) => panic!("{name}: {e}") };
        let mine = read(&gpu, pv.volume());
        let h = 2.0 * st.scene_hb() / res as f64;
        let stat = |v: &[f32]| (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max), v.iter().filter(|x| **x < 0.0).count());
        eprintln!("{name}: app {:?} mine {:?}; hb {}", stat(&app), stat(&mine), st.scene_hb());
        let r = res as usize;
        for (label, v) in [("app", &app), ("mine", &mine)] {
            let row: String = (0..r).step_by(r / 48).map(|x| { let d = v[(r / 2 * r + r / 2) * r + x]; if d < 0.0 { '#' } else if d < 0.05 { '+' } else if d < 0.2 { '.' } else { ' ' } }).collect();
            eprintln!("  {label} centre row x: |{row}|");
        }
        if name == "boaty64" {
            let r = res as usize;
            let mut bad: Vec<(f64, usize)> = app.iter().zip(&mine).enumerate().filter(|(_, (_, b))| (b.abs() as f64) < 2.0 * h).map(|(k, (a, b))| (((*a - *b).abs() as f64) / h, k)).filter(|x| x.0 > 4.0).collect();
            eprintln!("  boaty: {} voxels over 4 voxels", bad.len());
            bad.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(e, k) in bad.iter().step_by(bad.len() / 8 + 1).take(8) {
                let (x, y, z) = (k % r, (k / r) % r, k / (r * r));
                let p = [-1.5 + (x as f64 + 0.5) * h, -1.5 + (y as f64 + 0.5) * h, -1.5 + (z as f64 + 0.5) * h];
                let mut near: Vec<(f64, usize)> = pv.scene.pieces.iter().enumerate().map(|(i, q)| (q.dist(p), i)).filter(|x| x.0.abs() < 0.12).collect();
                near.sort_by(|a, b| a.0.total_cmp(&b.0));
                let txt: Vec<String> = near.iter().take(6).map(|&(dd, i)| format!("#{i} k{} op{} seam{:.3} d{dd:.3}", pv.scene.pieces[i].a[0], pv.scene.pieces[i].a[2], pv.scene.pieces[i].a[3])).collect();
                let smin = |a: f64, b: f64, k: f64| if k <= 0.0 { a.min(b) } else { let h = (k - (a - b).abs()).max(0.0) / k; a.min(b) - h * h * k * 0.25 };
                let mut fold = 1e9f64;
                for q in &pv.scene.pieces { let e2 = q.dist(p); let k = q.a[3] as f64; fold = if q.is_cut() { -smin(-fold, e2, k) } else { smin(fold, e2, k) }; }
                let listed = pv.cell_list_at(p);
                let mut lf = 1e9f64;
                for &i in &listed { let q = &pv.scene.pieces[i as usize]; let e2 = q.dist(p); let k = q.a[3] as f64; lf = if q.is_cut() { -smin(-lf, e2, k) } else { smin(lf, e2, k) }; }
                eprintln!("    listed {:?} -> cpu fold of the listed {:.4}", listed, lf);
                eprintln!("  boaty err {e:.1} at {:.3},{:.3},{:.3}: app {:.4} mine {:.4} cpu fold {:.4} | {}", p[0], p[1], p[2], app[k], mine[k], fold, txt.join("; "));
            }
        }
        let d = compare(&app, &mine, h, 2.0 * h);
        eprintln!("{name}: {} parts, res {res}, {} voxels within 2 voxels of the surface, worst {:.2} voxels, {} over one voxel, {} of {} voxels disagree on inside, unbounded excess {:.2} voxels; {:.0} MB; laps {:?}", pv.scene.pieces.len(), d.near, d.worst, d.over, d.signs, d.inside, d.excess, pv.memory() as f64 / 1e6, pv.laps.iter().map(|l| format!("{} {:.1}", l.0.split(' ').next().unwrap_or(""), l.1)).collect::<Vec<_>>());
        assert!(d.near > 100, "{name}: nothing near the surface?");
        // (within one voxel; a few voxels of a sub-voxel thin ring, whose own text is a walk to the surface, may be a little over)
        assert!(d.over * 2000 <= d.near && d.worst <= 1.5, "{name}: {} of {} voxels differ by more than one voxel (worst {:.2})", d.over, d.near, d.worst);
        assert!(d.signs * 50 <= d.inside.max(1), "{name}: {} of {} voxels disagree on being inside", d.signs, d.inside);
        // (outside the band the app's own value is not a true distance either: whether ours is a lower bound of the true one is the other test's)
    }
}

#[test]
fn held_equals_settled() {
    let Some(gpu) = gpu_or_skip() else { return };
    let csg0 = tree_parts(30, 0.35, 0.05, 7);
    let moved = {
        // (part 4 of the tree moved along x: the tree's text again with its position changed)
        let mut parts: Vec<String> = csg0.split(';').map(str::to_string).collect();
        let f: Vec<&str> = parts[4].split('@').collect();
        let pos: Vec<f64> = f[1].split(',').map(|x| x.parse().unwrap()).collect();
        let mut g: Vec<String> = f.iter().map(|s| s.to_string()).collect();
        g[1] = format!("{:.3},{:.3},{:.3}", pos[0] + 0.3, pos[1] + 0.1, pos[2]);
        parts[4] = g.join("@");
        parts.join(";")
    };
    let a = tree_args("block", &csg0);
    let b = tree_args("block", &moved);
    let mut sa = state(&a);
    let sb = state(&b);
    let res = 128;
    let mut pv = PartsVolume::build(&gpu, &mut sa, res, None).unwrap();
    // (piece 0 is the object's own shape, the tree's parts follow: part 4 is piece 5)
    pv.start(&sa, &[5]).unwrap();
    for _ in 0..3 { pv.run_live(&sb).unwrap(); }
    // (the live frame is the rest joined with the live part: it reads like the settled picture near the part)
    let live = read(&gpu, pv.volume());
    pv.fold(&sb).unwrap();
    let held = read(&gpu, pv.volume());
    let mut sf = state(&b);
    let fresh_pv = PartsVolume::build(&gpu, &mut sf, res, None).unwrap();
    let fresh = read(&gpu, fresh_pv.volume());
    let diff = held.iter().zip(&fresh).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let h = 2.0 * sa.scene_hb() / res as f64;
    let d = compare(&fresh, &live, h, 2.0 * h);
    eprintln!("held vs fresh: {diff} voxels differ of {}; live vs settled near the surface: worst {:.2} voxels, {} over one voxel", held.len(), d.worst, d.over);
    assert_eq!(diff, 0, "after start, live and fold the volume is not the fresh build's");
    assert!(d.worst <= 1.0, "the live frame differs from the settled one by {:.2} voxels", d.worst);
}

/// The value never exceeds the true distance (a lower bound everywhere), and equals it within the band: for a union of exact pieces.
#[test]
fn the_volume_is_a_lower_bound_and_exact_in_the_band() {
    let Some(gpu) = gpu_or_skip() else { return };
    let csg: Vec<String> = [
        "studio/sphere@0.4,0.2,0@0,0,0@0.5@union@0@0", "studio/block@-0.5,0,0.3@10,20,30@0.5@union@0@0", "studio/cylinder@0,-0.5,-0.4@40,0,10@0.5@union@0@0",
        "studio/capsule@0.5,-0.4,0.4@0,0,50@0.5@union@0@0", "studio/ellipsoid@-0.4,0.5,-0.3@0,30,0@0.5@union@0@0", "studio/washer@0.2,0.6,0.5@20,0,0@0.4@union@0@0",
    ].iter().map(|s| s.to_string()).collect();
    let args = tree_args("sphere", &csg.join(";"));
    let mut st = state(&args);
    let res = 128u32;
    let pv = PartsVolume::build(&gpu, &mut st, res, None).unwrap();
    let vol = read(&gpu, pv.volume());
    let hb = st.scene_hb();
    let h = 2.0 * hb / res as f64;
    let (mut worst_over, mut worst_band, mut n_band) = (0.0f64, 0.0f64, 0usize);
    for (k, &v) in vol.iter().enumerate() {
        let (x, y, z) = (k % res as usize, (k / res as usize) % res as usize, k / (res as usize * res as usize));
        let p = [-hb + (x as f64 + 0.5) * h, -hb + (y as f64 + 0.5) * h, -hb + (z as f64 + 0.5) * h];
        // (the true distance of the union: the least of the pieces' own, exact for these kinds)
        let truth = pv.scene.pieces.iter().map(|q| q.dist(p)).fold(f64::MAX, f64::min);
        let over = v as f64 - truth;
        if over > worst_over { worst_over = over; }
        if truth.abs() < pv.band() * 0.9 {
            n_band += 1;
            let e = (v as f64 - truth).abs();
            if e > worst_band { worst_band = e; }
        }
    }
    eprintln!("lower bound: worst excess {worst_over:.5}; in the band ({n_band} voxels) worst error {worst_band:.5} (voxel {h:.4})");
    // (half-float precision of a value near the band is about 1e-3 of it; the quarter-voxel nothing: the engine reads no offset)
    assert!(worst_over < 0.004, "a value is over the true distance by {worst_over}");
    assert!(worst_band < 0.004, "in the band the value is off by {worst_band}");
}

#[test]
fn refusing_never_panics_and_says_why() {
    let Some(gpu) = gpu_or_skip() else { return };
    // a shape the engine does not draw, a bevel, a part with its own volume, and too little memory
    let mut lens = state(&[kv("scene", "studio"), kv("preset", "lens"), kv("material", "glass")]);
    let e = PartsVolume::build(&gpu, &mut lens, 64, None).err().expect("a lens is refused");
    assert!(e.starts_with("refused:"), "{e}");
    let mut gems = state(&[kv("scene", "gems"), kv("preset", "emerald"), kv("material", "glass")]);
    assert!(PartsVolume::build(&gpu, &mut gems, 64, None).is_err());
    let mut beveled = state(&[kv("scene", "studio"), kv("preset", "block"), kv("p.bevel", "0.03"), kv("material", "glass")]);
    assert!(PartsVolume::build(&gpu, &mut beveled, 64, None).err().unwrap().contains("bevel"));
    let mut ok = state(&tree_args("block", &tree_parts(12, 0.4, 0.05, 6)));
    let e = PartsVolume::build(&gpu, &mut ok, 128, Some(1 << 20)).err().expect("1 MB is not enough");
    assert_eq!(e, NOT_ENOUGH_MEMORY);
    // (and the same scene builds when the limit allows)
    assert!(PartsVolume::build(&gpu, &mut ok, 64, Some(1 << 30)).is_ok());
    // a drag of a part that is not there
    let mut pv = PartsVolume::build(&gpu, &mut ok, 64, None).unwrap();
    assert!(pv.start(&ok, &[9999]).is_err());
    assert!(pv.start(&ok, &[]).is_err());
}

#[test]
fn a_slider_and_an_added_part_are_built_where_they_are() {
    let Some(gpu) = gpu_or_skip() else { return };
    let csg = tree_parts(16, 0.4, 0.05, 5);
    let res = 128;
    let mut a = state(&tree_args("block", &csg));
    let mut pv = PartsVolume::build(&gpu, &mut a, res, None).unwrap();
    // a slider: the object's own shape's width
    let mut b = state(&tree_args("block", &csg));
    b.apply_arg("p.width", "0.9");
    pv.reshape(&b).unwrap();
    let held = read(&gpu, pv.volume());
    let mut fresh_state = state(&tree_args("block", &csg));
    fresh_state.apply_arg("p.width", "0.9");
    let fresh = read(&gpu, PartsVolume::build(&gpu, &mut fresh_state, res, None).unwrap().volume());
    let diff = held.iter().zip(&fresh).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    eprintln!("reshape vs fresh: {diff} voxels differ");
    assert_eq!(diff, 0);
    // one more part
    let more = format!("{csg};studio/sphere@0.2,0.1,-0.3@0,0,0@0.3@union@0.03@0");
    let mut c = state(&tree_args("block", &more));
    c.apply_arg("p.width", "0.9");
    pv.add(&c).unwrap();
    let held = read(&gpu, pv.volume());
    let mut fresh_state = state(&tree_args("block", &more));
    fresh_state.apply_arg("p.width", "0.9");
    let fresh = read(&gpu, PartsVolume::build(&gpu, &mut fresh_state, res, None).unwrap().volume());
    let diff = held.iter().zip(&fresh).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    eprintln!("add vs fresh: {diff} voxels differ");
    assert_eq!(diff, 0);
}

/// (diagnostic) where does the GPU's value leave the CPU fold of the same pieces, as parts are added one by one
#[test]
#[ignore]
fn bisect_boaty() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mods, _) = boaty::build_with(64, false).unwrap();
    let prims: Vec<Part> = mods.iter().flatten().cloned().collect();
    let first = prims[0].clone();
    let rest: Vec<Part> = prims[1..].iter().map(|p| { let mut p = p.clone(); for k in 0..3 { p.place.pos[k] -= first.place.pos[k]; } p }).collect();
    let res = 128u32;
    let smin = |a: f64, b: f64, k: f64| if k <= 0.0 { a.min(b) } else { let h = (k - (a - b).abs()).max(0.0) / k; a.min(b) - h * h * k * 0.25 };
    for n in [1usize, 2, 3, 5, 8, 11, 12, 13, 20, 30, 45, 63] {
        let mut a = vec![kv("scene", "studio"), kv("material", "glass"), kv("dist", "5.0"), kv("preset", &first.preset), kv("select", 1)];
        for (id, v) in &first.values { a.push(kv(&format!("L1.p.{id}"), format!("{:.6}", if boaty::is_length(id) { v * first.place.size } else { *v }))); }
        a.push(kv("L1.trace", "csg"));
        a.push(kv("L1.csgJson", serde_json::to_string(&rest[..n]).unwrap()));
        a.push(kv("res", "128"));
        let mut st = state(&a);
        let pv = PartsVolume::build(&gpu, &mut st, res, None).unwrap();
        let vol = read(&gpu, pv.volume());
        let h = 2.0 * 1.5 / res as f64;
        let (mut worst, mut at) = (0.0f64, 0usize);
        for (k, &v) in vol.iter().enumerate() {
            let (x, y, z) = (k % res as usize, (k / res as usize) % res as usize, k / (res as usize * res as usize));
            let p = [-1.5 + (x as f64 + 0.5) * h, -1.5 + (y as f64 + 0.5) * h, -1.5 + (z as f64 + 0.5) * h];
            let mut fold = 1e9f64;
            for q in &pv.scene.pieces { let e = q.dist(p); let s = q.a[3] as f64; fold = if q.is_cut() { -smin(-fold, e, s) } else { smin(fold, e, s) }; }
            if fold.abs() < 0.2 { let e = (fold - v as f64).abs(); if e > worst { worst = e; at = k; } }
        }
        let (x, y, z) = (at % res as usize, (at / res as usize) % res as usize, at / (res as usize * res as usize));
        if worst > 0.03 && n < 14 {
            let pp = [-1.5 + (x as f64 + 0.5) * h, -1.5 + (y as f64 + 0.5) * h, -1.5 + (z as f64 + 0.5) * h];
            for (i, q) in pv.scene.pieces.iter().enumerate() { eprintln!("    piece {i}: kind {} fam/preset {:?} dist {:.4} seam {} op {} scale {} b {:?} c {:?} rec v {:?}", q.a[0], pv.scene.kinds.get(pv.scene.part_kind[i] as usize), q.dist(pp), q.a[3], q.a[2], q.a[1], q.b, q.c, &pv.scene.recs[i][16..24]); }
            eprintln!("    gpu {} at {:?}; names {:?}", vol[at], pp, pv.scene.kinds.iter().map(|k| { let f = &st.families[k.0]; format!("{}/{}", f.id, f.presets[k.1].id) }).collect::<Vec<_>>());
        }
        eprintln!("parts {}: worst gpu-vs-cpu {:.4} at voxel {x},{y},{z}; kinds {:?}", pv.scene.pieces.len(), worst, pv.scene.kinds);
    }
}

/// (diagnostic) two objects joined: the parts volume against the app's combined volume
#[test]
#[ignore]
fn chamfer_pair_against_combine() {
    let Some(gpu) = gpu_or_skip() else { return };
    let args: Vec<(String, String)> = ["scene=studio", "preset=sphere", "material=glass", "layer=studio:sphere", "J12.op=join", "J12.profile=bevel", "J12.radius=0.4", "L1.x=-0.5", "L2.x=0.5", "L1.size=0.7", "L2.size=0.7", "dist=5.0", "rot=0,0,0", "res=128"].iter().map(|s| { let (k, v) = s.split_once('=').unwrap(); kv(k, v) }).collect();
    let mut e = Engine::new(gpu.clone(), state(&args), true);
    e.bake(Detail::Full).unwrap();
    let v = e.scene_volume().unwrap();
    let (app, res) = (read(&gpu, v), v.res);
    let mut st = state(&args);
    let pv = PartsVolume::build(&gpu, &mut st, res, None).unwrap();
    let mine = read(&gpu, pv.volume());
    let h = 2.0 * st.scene_hb() / res as f64;
    let d = compare(&app, &mine, h, 2.0 * h);
    eprintln!("chamfer: res {res}, worst {:.2} voxels, {} over one of {}; pair {:?}; inside app {} mine {}", d.worst, d.over, d.near, pv.scene.pair, app.iter().filter(|x| **x < 0.0).count(), mine.iter().filter(|x| **x < 0.0).count());
    let r = res as usize;
    for (label, vv) in [("app", &app), ("mine", &mine)] {
        let row: String = (0..r).step_by(r / 64).map(|x| { let d = vv[(r / 2 * r + r / 2) * r + x]; if d < -0.2 { '@' } else if d < 0.0 { '#' } else if d < 0.05 { '+' } else if d < 0.2 { '.' } else { ' ' } }).collect();
        eprintln!("  {label} centre row: |{row}|");
    }
    for p in &pv.scene.pieces { eprintln!("  piece kind {} scale {} seam {} x {:?} inv {:?}", p.a[0], p.a[1], p.a[3], p.x, p.inv); }
}

/// (tool) the app arguments of boatyN, as the lab builds them (N <= 512), written as JSON to the file in GF_DUMP
#[test]
#[ignore]
fn dump_boaty_args() {
    let Ok(path) = std::env::var("GF_DUMP") else { return };
    let n: usize = std::env::var("GF_N").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let (mods, _) = boaty::build_with(n, false).unwrap();
    let prims: Vec<Part> = mods.iter().flatten().cloned().collect();
    let objs: Vec<Vec<Part>> = if prims.len() <= 64 { vec![prims.clone()] } else { mods.clone() };
    let mut a = vec![kv("scene", "studio"), kv("material", "glass"), kv("dist", "5.0"), kv("rx", "16"), kv("ry", "-36")];
    for (j, ps) in objs.iter().enumerate() {
        let first = &ps[0];
        let l = format!("L{}", j + 1);
        if j == 0 { a.push(kv("preset", &first.preset)); } else {
            a.push(kv("layer", format!("studio:{}", first.preset)));
            for i in 0..j { a.push(kv(&format!("J{}{}.op", i + 1, j + 1), "apart")); }
        }
        a.push(kv("select", j + 1));
        for (id, v) in &first.values { a.push(kv(&format!("{l}.p.{id}"), format!("{:.6}", if boaty::is_length(id) { v * first.place.size } else { *v }))); }
        if objs.len() > 1 { for (ax, v) in ["x", "y", "z"].iter().zip(first.place.pos) { a.push(kv(&format!("{l}.{ax}"), format!("{v:.6}"))); } }
        a.push(kv(&format!("{l}.trace"), "csg"));
        if ps.len() > 1 {
            let rest: Vec<Part> = ps[1..].iter().map(|p| { let mut p = p.clone(); for k in 0..3 { p.place.pos[k] -= first.place.pos[k]; } p }).collect();
            a.push(kv(&format!("{l}.csgJson"), serde_json::to_string(&rest).unwrap()));
        }
    }
    a.push(kv("select", 1));
    if let Ok(setup) = std::env::var("GF_SETUP") { std::fs::write(setup, state(&a).setup_json()).unwrap(); }
    std::fs::write(path, serde_json::to_string(&a).unwrap()).unwrap();
}

#[test]
#[ignore]
fn setup_tree_shapes() {
    let Ok(path) = std::env::var("GF_SETUP_IN") else { return };
    let mut st = State::with_args(&[], true);
    st.apply_setup_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    for i in 0..st.stack.layers.len() {
        let (parts, joined, dropped) = st.csg_tree_layer(i);
        let own = st.layer_shape(i).map(|s| st.families[st.layer_fam(i)].presets[s.preset].id);
        eprintln!("layer {i}: own {:?}, tree {} parts, joined {joined}, dropped {dropped}, first {:?}", own, parts.len(), parts.first().map(|p| st.families[p.fam].presets[p.preset].id));
    }
}

#[test]
#[ignore]
fn setup_against_existing_path() {
    let Some(gpu) = gpu_or_skip() else { return };
    let Ok(path) = std::env::var("GF_SETUP_IN") else { return };
    let text = std::fs::read_to_string(path).unwrap();
    let mk = || { let mut st = State::with_args(&[kv("res", "256")], true); st.apply_setup_json(&text).unwrap(); st };
    let mut e = Engine::new(gpu.clone(), mk(), true);
    e.bake(Detail::Full).unwrap();
    let v = e.scene_volume().unwrap();
    let (app, res) = (read(&gpu, v), v.res);
    let mut st = mk();
    let pv = PartsVolume::build(&gpu, &mut st, res, None).unwrap();
    let mine = read(&gpu, pv.volume());
    let h = 2.0 * st.scene_hb() / res as f64;
    let d = compare(&app, &mine, h, 2.0 * h);
    eprintln!("setup: res {res}, worst {:.2} voxels, {} over one of {}; inside app {} mine {}", d.worst, d.over, d.near, app.iter().filter(|x| **x < 0.0).count(), mine.iter().filter(|x| **x < 0.0).count());
    let hb = st.scene_hb();
    for i in [0usize, 64, 128, 192] {
        let b = pv.scene.balls[i];
        let idx = |x: f64| (((x + hb) / h).floor() as usize).min(res as usize - 1);
        let k = (idx(b[2]) * res as usize + idx(b[1])) * res as usize + idx(b[0]);
        eprintln!("  piece {i} (object {}, kind {}): ball {:?}, hb {hb}; at its centre app {:.3} mine {:.3}; piece dist there {:.3}", pv.scene.owner[i].0, pv.scene.pieces[i].a[0], b, app[k], mine[k], pv.scene.pieces[i].dist([b[0], b[1], b[2]]));
    }
    let r = res as usize;
    let mut bad: Vec<(f64, usize)> = app.iter().zip(&mine).enumerate().filter(|(_, (a, b))| (a.abs() as f64) < 2.0 * h && (b.abs() as f64) >= 2.0 * h).map(|(k, (a, b))| (((*a - *b).abs() as f64) / h, k)).collect();
    bad.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!("  app-surface voxels where ours is not near: {}", bad.len());
    for &(e2, k) in bad.iter().step_by(bad.len() / 10 + 1).take(10) {
        let (x, y, z) = (k % r, (k / r) % r, k / (r * r));
        let p = [-1.5 + (x as f64 + 0.5) * h, -1.5 + (y as f64 + 0.5) * h, -1.5 + (z as f64 + 0.5) * h];
        let mut near: Vec<(f64, usize)> = pv.scene.pieces.iter().enumerate().map(|(i, q)| (q.dist(p), i)).filter(|x| x.0.abs() < 0.1).collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        eprintln!("  err {e2:.1} at {:.3},{:.3},{:.3}: app {:.4} mine {:.4} | {:?}", p[0], p[1], p[2], app[k], mine[k], near.iter().take(4).map(|&(dd, i)| (i, pv.scene.owner[i].0, pv.scene.pieces[i].a[0], (dd * 1000.0).round() / 1000.0, pv.scene.pieces[i].a[2])).collect::<Vec<_>>());
    }
}
