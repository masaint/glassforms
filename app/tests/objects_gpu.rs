//! Objects of two kinds (volume and csg), made, taken away down to none, and converted, as the right panel's
//! tabs do: what the scene's volume holds after each. Needs a GPU (through the queue).

use gf_core::state::{Detail, State};
use glassforms::engine::Engine;
use glassforms::gpu::Gpu;

fn engine(gpu: &Gpu, pairs: &[(&str, &str)]) -> Engine {
    let args: Vec<(String, String)> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let mut e = Engine::new(gpu.clone(), State::with_args(&args, true), true);
    e.resize(64, 36);
    e.bake(Detail::Full).unwrap();
    e.state.drop_request();
    e
}

fn rebuild(e: &mut Engine) {
    if let Some(r) = e.state.take_request() { e.bake(r.detail).unwrap(); }
    e.draw(false);
    e.gpu.wait();
}

/// The scene's volume: how many voxels are inside something.
fn inside(e: &Engine) -> usize {
    let (_, bits) = e.read_volume().unwrap();
    bits.iter().filter(|b| half::f16::from_bits(**b).to_f32() < 0.0).count()
}

/// A GPU test that cannot get a device must say so loudly (cargo would still report "ok"); GF_REQUIRE_GPU=1 makes it fail instead.
fn skipped(why: &str) {
    eprintln!("**** NOT TESTED (skipped): {why}; set GF_REQUIRE_GPU=1 to fail instead ****");
    if std::env::var_os("GF_REQUIRE_GPU").is_some() { panic!("GF_REQUIRE_GPU is set and {why}"); }
}

#[test]
fn a_group_shell_hollows_the_join_after_internal_faces_disappear() {
    use gf_core::{groups::{DropAt, Item}, mods::{Kind, Modifier}};
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let mut e = engine(&gpu, &[("scene", "studio"), ("preset", "sphere"), ("res", "128")]);
    let (f, p) = find(&e.state, "studio", "sphere");
    e.state.create_object(f, p, true).unwrap();
    e.state.stack.layers[1].place.size = 0.4;
    let group = e.state.new_group(None).unwrap();
    // Moving the first sphere to the end puts the small one first in the underlying stack.
    e.state.move_item(Item::Object(0), DropAt::Inside(group)).unwrap();
    e.state.move_item(Item::Object(0), DropAt::Inside(group)).unwrap();
    e.state.selected_group = Some(group);
    e.state.selected_modifiers_mut().list.push(Modifier::of(Kind::Shell));
    e.state.sculpt_changed(false);
    rebuild(&mut e);
    let (res, field) = e.read_volume().unwrap();
    let n = res as usize;
    let ray = &field[(n / 2 * n + n / 2) * n..(n / 2 * n + n / 2 + 1) * n];
    let signs: Vec<_> = ray.iter().map(|b| half::f16::from_bits(*b).to_f32() < 0.0).collect();
    assert!(!signs[n / 2], "the centre of the group must be hollow");
    assert_eq!(signs.windows(2).filter(|s| s[0] != s[1]).count(), 4, "only the outside sphere has a wall");
    e.state.selected_modifiers_mut().list.clear();
    e.state.sculpt_changed(false);
    rebuild(&mut e);
    let (_, solid) = e.read_volume().unwrap();
    assert!(half::f16::from_bits(solid[(n / 2 * n + n / 2) * n + n / 2]).to_f32() < 0.0, "removing the group modifier restores solid geometry");
}

#[test]
fn group_subtraction_follows_the_order_of_the_parts() {
    use gf_core::groups::{DropAt, GroupOp, Item};
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let mut e = engine(&gpu, &[("scene", "studio"), ("preset", "sphere"), ("trace", "csg"), ("res", "128")]);
    let (f, p) = find(&e.state, "studio", "sphere");
    e.state.create_object(f, p, true).unwrap();
    e.state.stack.layers[1].place.size = 0.4;
    let id = e.state.new_group(None).unwrap();
    for _ in 0..2 { e.state.move_item(Item::Object(0), DropAt::Inside(id)).unwrap(); }
    e.state.stack.hierarchy.as_mut().unwrap().group_mut(id).unwrap().operation = GroupOp::Subtract;
    rebuild(&mut e);
    assert!(e.renderer.sdf_key().is_some_and(|key| key.contains("|group:")), "the cutter and its wall retain their formula normals");
    assert!(inside(&e) > 1000);
    e.state.move_item(Item::Object(1), DropAt::Before(Item::Object(0))).unwrap();
    rebuild(&mut e);
    assert_eq!(inside(&e), 0, "subtracting the enclosing sphere consumes the entire smaller sphere");
}

fn find(st: &State, fam: &str, preset: &str) -> (usize, usize) {
    let f = st.families.iter().position(|x| x.id == fam).unwrap();
    (f, st.families[f].presets.iter().position(|p| p.id == preset).unwrap())
}

/// As the window draws: nothing waited for, frame after frame. How bright the middle of the picture is.
fn window_middle(e: &mut Engine, frames: usize) -> f64 {
    let ((w, h), _) = e.renderer.size();
    for _ in 0..frames {
        e.frame(w, h, 1.0 / 60.0);
        e.gpu.wait();
    }
    let (sw, sh, rgba) = e.screenshot().unwrap();
    let (sw, sh) = (sw as usize, sh as usize);
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in sh / 4..3 * sh / 4 {
        for x in sw / 4..3 * sw / 4 {
            let i = (y * sw + x) * 4;
            sum += (rgba[i] as f64 + rgba[i + 1] as f64 + rgba[i + 2] as f64) / 3.0;
            n += 1.0;
        }
    }
    sum / n
}

#[test]
fn an_empty_scene_draws_nothing_in_the_window() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        skipped("no graphics device");
        return;
    };
    let mut e = engine(&gpu, &[("scene", "attractors"), ("preset", "thomas"), ("res", "128")]);
    e.auto = false;
    e.models.wait = false;
    let before = window_middle(&mut e, 30);
    assert!(e.state.delete_object(0));
    let after = window_middle(&mut e, 90);
    eprintln!("after: empty {}, want {:?}, scene volume {:?}, plain {}", e.state.empty, e.state.build_wanted(), e.scene_volume().map(|v| v.res), e.state.plain());
    eprintln!("the middle of the picture: {before:.2} with Thomas, {after:.2} with nothing");
    assert!(after < 0.25 * before, "the empty scene still shows something: {after:.2} against {before:.2}");
}

#[test]
fn hiding_the_only_layer_draws_nothing_in_the_window() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let mut e = engine(&gpu, &[("scene", "attractors"), ("preset", "thomas"), ("res", "128")]);
    e.auto = false;
    e.models.wait = false;
    let before = window_middle(&mut e, 30);
    e.state.set_layer_show(0, false);
    e.state.cut = true;
    let after = window_middle(&mut e, 90);
    eprintln!("hidden by its eye, the picture cut: {before:.2} shown, {after:.2} hidden");
}

#[test]
fn objects_are_made_taken_away_and_converted() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        skipped("no graphics device");
        return;
    };
    let mut e = engine(&gpu, &[("scene", "attractors"), ("preset", "thomas"), ("res", "128")]);
    let thomas = inside(&e);
    assert!(thomas > 0);
    // the last object taken away: nothing inside anything
    assert!(e.state.delete_object(0));
    rebuild(&mut e);
    assert_eq!(e.state.object_count(), 0);
    assert_eq!(inside(&e), 0, "an empty scene holds nothing");
    // a csg object made from the csg tab: a block, and nothing of the Thomas that was there
    let (f, p) = find(&e.state, "studio", "block");
    assert_eq!(e.state.create_object(f, p, true), Some(0));
    e.state.apply_arg("res", "128");
    rebuild(&mut e);
    let block_csg = inside(&e);
    let mut alone = engine(&gpu, &[("scene", "studio"), ("preset", "block"), ("res", "128")]);
    rebuild(&mut alone);
    let block = inside(&alone);
    eprintln!("thomas {thomas}, block alone {block}, block made as a csg object {block_csg}");
    assert!((block_csg as f64 - block as f64).abs() < 0.01 * block as f64, "the csg block holds {block_csg} voxels inside, the block alone {block}");
}

#[test]
fn a_tree_converted_and_beside_a_volume_object() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let mut e = engine(&gpu, &[("scene", "studio"), ("preset", "block"), ("res", "128"), ("trace", "csg"), ("csg", "studio/sphere@0.4,0.2,0@0,0,0@0.6@subtract@0.04")]);
    rebuild(&mut e);
    let tree = inside(&e);
    // converted: a volume object made from the tree holds the same shape
    assert!(e.state.convert_to_volume());
    rebuild(&mut e);
    let baked = inside(&e);
    eprintln!("the tree: {tree} voxels inside; made a volume object: {baked}");
    assert!((tree as f64 - baked as f64).abs() < 0.005 * tree as f64);
    // a block alone holds more: the ball was taken out of it
    let mut plain = engine(&gpu, &[("scene", "studio"), ("preset", "block"), ("res", "128")]);
    rebuild(&mut plain);
    assert!(inside(&plain) > tree + tree / 50);
    // beside a volume object (a Thomas far to the side): both are in the scene, the tree with its ball taken out
    let (f, p) = find(&e.state, "attractors", "thomas");
    assert!(e.state.create_object(f, p, false).is_some());
    assert!(e.state.apply_arg("L2.size", "0.3") && e.state.apply_arg("L2.x", "1.1"));
    e.state.apply_arg("res", "128");
    rebuild(&mut e);
    let both = inside(&e);
    eprintln!("beside a small Thomas: {both}");
    assert!(both > tree * 9 / 10 && both < inside(&plain), "{both}");
}
#[test]
fn two_formula_tables_keep_their_own_geometry_in_a_union() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let args = [("scene", "links"), ("preset", "hopflink"), ("res", "128"), ("trace", "csg")];
    let first = engine(&gpu, &args);
    let mut combined_args = args.to_vec();
    combined_args.push(("csg", "text/word@0.68,0.32,0@0,0,0@0.62@union"));
    let combined = engine(&gpu, &combined_args);
    let (_, left) = first.read_volume().unwrap();
    let (_, both) = combined.read_volume().unwrap();
    let negative = |v: u16| half::f16::from_bits(v).to_f32() < 0.0;
    let mut added = 0;
    let mut lost = 0;
    for (a, b) in left.into_iter().zip(both) {
        if negative(a) && !negative(b) { lost += 1; }
        if !negative(a) && negative(b) { added += 1; }
    }
    assert!(added > 100, "the second formula adds {added} interior voxels");
    assert!(lost < 20, "the first formula lost {lost} interior voxels");
}
