//! The shader's joins agree with gf_core::join (f64): the volume baked on the GPU for a block joined by a
//! ball, voxel by voxel, against the f64 join of the volumes baked for the block alone and the ball alone.
//! Every profile with every operation, and every band.
use gf_core::csg::{Op, Part};
use gf_core::join::{Band, Profile};
use gf_core::layers::Placement;
use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

fn volume(e: &mut Engine) -> Vec<f32> {
    e.bake(Detail::Full).unwrap();
    let (_, v) = e.read_volume().unwrap();
    v.into_iter().map(|b| half::f16::from_bits(b).to_f32()).collect()
}

#[test]
fn baked_joins_agree_with_the_f64_formulas() {
    let settings: [(String, String); 4] = [("scene", "studio"), ("preset", "block"), ("trace", "csg"), ("res", "96")].map(|(k, v)| (k.into(), v.into()));
    let ball = Part { place: Placement { pos: [0.35, 0.4, 0.1], rot: [0.0; 3], size: 0.5 }, ..Part::default() };
    // each reading from an engine of its own: the block and the ball both in the tree, and as the
    // reference, the block's node alone (the ball's taken out) and the ball's alone (the block's taken out)
    let fresh = || -> Option<Engine> {
        let gpu = Gpu::headless("vulkan").ok()?;
        let mut e = Engine::new(gpu, State::with_args(&settings, true), true);
        e.resize(64, 48);
        Some(e)
    };
    let Some(mut e) = fresh() else { return };
    let path = e.state.csg.add_construction_part(&[], ball.clone()).unwrap();
    let only = |drop: usize| -> Vec<f32> {
        let mut r = fresh().unwrap();
        r.state.csg.add_construction_part(&[], ball.clone()).unwrap();
        assert!(r.state.csg.remove_construction_node(&[drop]));
        volume(&mut r)
    };
    let block = only(1);
    let alone = only(0);
    let mut cases: Vec<(Op, Profile, Band)> = Vec::new();
    for op in Op::ALL { for p in Profile::ALL { cases.push((op, p, Band::None)); } }
    for b in [Band::Pocket, Band::Groove, Band::Ridge, Band::Bead] { cases.push((b.op().unwrap(), Profile::Round, b)); }
    let (k, t, w) = (0.12, 0.09, 0.03);
    for (op, profile, band) in cases {
        let n = e.state.csg.edit_construction().node_mut(&path).unwrap();
        (n.op, n.blend, n.profile, n.band, n.depth, n.width) = (op, k, profile, band, t, w);
        let joined = volume(&mut e);
        assert_eq!(joined.len(), block.len());
        let (mut compared, mut worst, mut at) = (0usize, 0.0f64, (0usize, 0.0, 0.0, 0.0));
        for (i, ((&a, &b), &j)) in block.iter().zip(&alone).zip(&joined).enumerate() {
            if a.abs().max(b.abs()).max(j.abs()) > 0.25 { continue; }
            let (a, b) = (f64::from(a), f64::from(b));
            let want = if band.is_none() { gf_core::join::join(op, a, b, k, profile) } else { gf_core::join::band(band, a, b, t, w, k, profile) };
            if (want - f64::from(j)).abs() > worst { worst = (want - f64::from(j)).abs(); at = (i, a, b, f64::from(j)); }
            compared += 1;
        }
        println!("{op:?} {profile:?} {band:?}: {compared} voxels, worst {worst:.2e} at voxel {} (a {:.4}, b {:.4}, joined {:.4})", at.0, at.1, at.2, at.3);
        assert!(compared > 1000, "{op:?} {profile:?} {band:?}: only {compared} voxels near the surfaces");
        assert!(worst < 2e-3, "{op:?} {profile:?} {band:?}: the shader differs from f64 by {worst}");
    }
}
