use gf_core::mods::{Axes, Kind, Modifier};
use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn construction_patterns_match_the_same_whole_object_field() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "studio"),
        ("preset", "wedge"),
        ("trace", "csg"),
        ("res", "96"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(80, 60);
    let mut grid = Modifier::of(Kind::Repeat);
    grid.a = Axes::new([3., 2., 3.], false);
    grid.b = Axes::new([0.34, 0.31, 0.29], false);
    grid.stagger = 0.45;
    let mut ring = Modifier::of(Kind::Repeat);
    ring.mode = 1;
    ring.a = Axes::new([7., 1., 1.], false);
    ring.detail = 0.45;
    ring.flip = true;
    let mut wrap = Modifier::of(Kind::Wrap);
    wrap.a = Axes::new([0.82, 0.95, 0.48], false);
    wrap.size = 0.30;
    let mut turn = Modifier::of(Kind::Twist);
    turn.a = Axes::new([0., 32., 0.], false);
    let sample = |e: &mut Engine| {
        e.bake(Detail::Full).unwrap();
        e.render(1);
        let (_, v) = e.read_volume().unwrap();
        v.into_iter()
            .map(|b| half::f16::from_bits(b).to_f32())
            .collect::<Vec<_>>()
    };
    for (name, stages) in [
        ("staggered overlap", vec![grid]),
        ("mirrored polar overlap", vec![ring]),
        ("cone wrap", vec![wrap.clone()]),
        ("twist before wrap", vec![turn, wrap]),
    ] {
        e.state.csg.construction = None;
        e.state.shape_mut().mods.list = stages.clone();
        let source = sample(&mut e);
        e.state.shape_mut().mods.list.clear();
        let entry = e.state.csg.edit_construction().node_mut(&[0]).unwrap();
        entry.mods.list = stages;
        let node = sample(&mut e);
        let occupied = source.iter().filter(|d| **d < 0.).count();
        assert!(occupied > 100, "{name} lost its surface");
        let mismatch = source
            .iter()
            .zip(&node)
            .filter(|(a, b)| (**a < 0.) != (**b < 0.))
            .count();
        let deep = source
            .iter()
            .zip(&node)
            .filter(|(a, b)| (**a < 0.) != (**b < 0.) && a.abs().min(b.abs()) > 0.024)
            .count();
        eprintln!(
            "{name}: {occupied} occupied, {mismatch} different signs, {deep} in the interior"
        );
        assert!(
            mismatch < occupied / 20 && deep < 5,
            "{name}: {mismatch} of {occupied} signs differ, {deep} in the interior"
        );
    }
}
