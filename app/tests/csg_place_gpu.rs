//! An object traced as a tree (trace=csg) and placed off the origin must still be drawn: its tree's
//! volume, combined into the scene, holds its surface where it is placed.
use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

fn read_min(gpu: &Gpu, v: &glassforms::bake::Volume) -> (f32, usize) {
    let mut min = f32::MAX;
    let mut inside = 0;
    for z in 0..v.res {
        let bytes = gpu.read(&v.tex, 0, 0, z, v.res, v.res, 2).unwrap();
        for h in bytemuck::cast_slice::<u8, u16>(&bytes) {
            let d = half::f16::from_bits(*h).to_f32();
            min = min.min(d);
            if d < 0.0 { inside += 1; }
        }
    }
    (min, inside)
}

/// The scene's volume after a full build of this shape at this placement: its least value and how many
/// voxels are inside.
fn scene(gpu: &Gpu, scene: &str, preset: &str, csg: bool, pos: [f64; 3], rot: [f64; 3]) -> ((f32, usize), Option<(f32, usize)>) {
    let mut args = vec![("scene", scene), ("preset", preset), ("res", "128")];
    if csg { args.push(("trace", "csg")); }
    let args: Vec<(String, String)> = args.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let mut st = State::with_args(&args, true);
    st.stack.layers[0].place.pos = pos;
    st.stack.layers[0].place.rot = rot;
    let mut e = Engine::new(gpu.clone(), st, true);
    e.bake(Detail::Full).unwrap();
    let whole = read_min(&e.gpu, e.scene_volume().unwrap());
    let tree = e.csg.layer_vols.first().and_then(|v| v.as_ref()).map(|(_, v)| read_min(&e.gpu, v));
    (whole, tree)
}

#[test]
fn a_placed_object_traced_as_a_tree_keeps_its_surface() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    for (fam, preset) in [("attractors", "thomas"), ("solids", "sphere"), ("knots", "trefoil"), ("braids", "plait"), ("branching", "coral")] {
        for (pos, rot) in [([0.0; 3], [0.0; 3]), ([0.3, 0.0, 0.0], [0.0; 3]), ([0.0; 3], [20.0, 35.0, 0.0])] {
            let (volume, _) = scene(&gpu, fam, preset, false, pos, rot);
            let (whole, tree) = scene(&gpu, fam, preset, true, pos, rot);
            eprintln!("{fam}/{preset} at {pos:?} turned {rot:?}: as a volume {} inside; as a tree {} inside (its own volume: {tree:?})", volume.1, whole.1);
            // drawn as a tree, it holds about as much inside as drawn from its own volume (the formula is
            // finer: within a fifth)
            assert!(whole.1 as f64 > 0.8 * volume.1 as f64 && (whole.1 as f64) < 1.25 * volume.1 as f64, "{fam}/{preset} at {pos:?} turned {rot:?}: {} inside as a tree against {} as a volume", whole.1, volume.1);
            if let Some((_, inside)) = tree { assert!(inside > 0, "{fam}/{preset}: the tree's own volume is empty"); }
        }
    }
}
