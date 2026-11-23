//! Listed arrays on the GPU: a ring read by listing its copies (forced by a blend too small to see) bakes the
//! same volume as the ring folded, alone and nested under a grid in one stack; an arc and copies along a path
//! bake a surface. Skipped without a Vulkan device.

use gf_core::mods::{Axes, Kind, Modifier};
use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

fn sample(e: &mut Engine) -> Vec<f32> {
    e.bake(Detail::Full).unwrap();
    e.render(1);
    let (_, v) = e.read_volume().unwrap();
    v.into_iter().map(|b| half::f16::from_bits(b).to_f32()).collect()
}

/// Each volume baked on an engine of its own (one engine swapping trees reuses a stale volume).
fn set(stages: &[Modifier]) -> Vec<f32> {
    let gpu = Gpu::headless("vulkan").expect("a device");
    let args = [("scene", "studio"), ("preset", "wedge"), ("trace", "csg"), ("res", "96")].map(|(k, v)| (k.into(), v.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(80, 60);
    e.state.csg.edit_construction().node_mut(&[0]).unwrap().mods.list = stages.to_vec();
    sample(&mut e)
}

#[test]
fn listed_copies_bake_as_the_fold_does() {
    if Gpu::headless("vulkan").is_err() {
        return;
    }
    let mut ring = Modifier::of(Kind::Repeat);
    ring.mode = 1;
    ring.a = Axes::new([7., 1., 1.], false);
    ring.detail = 0.9;
    let mut grid = Modifier::of(Kind::Repeat);
    grid.a = Axes::new([2., 2., 1.], false);
    grid.b = Axes::new([4.0, 4.0, 4.0], true);
    let mut listed = ring.clone();
    listed.array.blend = 1e-5;
    assert!(listed.listed() && !ring.listed());
    // (a ring of grids whose copies leave room, so the folded stack is traced exactly too, not baked)
    let mut wide = ring.clone();
    wide.a = Axes::new([5., 1., 1.], false);
    wide.detail = 2.4;
    let mut wide_listed = wide.clone();
    wide_listed.array.blend = 1e-5;
    let stack = gf_core::mods::Mods { list: vec![grid.clone(), wide.clone()] };
    assert!(stack.exact(&[-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]), "{:?}", stack.why_baked(&[-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]));
    for (name, folded, read) in [
        ("ring", vec![ring.clone()], vec![listed.clone()]),
        ("ring of grids", vec![grid.clone(), wide.clone()], vec![grid.clone(), wide_listed.clone()]),
    ] {
        let a = set(&folded);
        let b = set(&read);
        let occupied = a.iter().filter(|d| **d < 0.).count();
        let mismatch = a.iter().zip(&b).filter(|(x, y)| (**x < 0.) != (**y < 0.)).count();
        let worst = a.iter().zip(&b).filter(|(x, y)| x.abs().min(y.abs()) < 0.05).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        eprintln!("{name}: {occupied} occupied, {mismatch} signs differ, near the surface the two differ by {worst} at most");
        assert!(occupied > 100, "{name} lost its surface");
        assert!(mismatch <= occupied / 200, "{name}: {mismatch} of {occupied} signs differ");
    }
    let mut arc = ring.clone();
    arc.array.span = 150.0;
    let mut path = ring.clone();
    path.mode = 2;
    path.a = Axes::new([6., 1., 1.], false);
    path.array.step_grow = 0.08;
    for (name, stage) in [("arc", arc), ("path", path)] {
        let v = set(&[stage]);
        let occupied = v.iter().filter(|d| **d < 0.).count();
        eprintln!("{name}: {occupied} occupied");
        assert!(occupied > 100, "{name} has no surface");
    }
}
