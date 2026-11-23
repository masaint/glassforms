use super::*;
use crate::gpu::Gpu;
use gf_core::{
    construction::CutMaterial,
    csg::{Op, Part},
    layers::Placement,
    state::State,
};
use std::collections::BTreeMap;

fn motion(e: &mut Engine) {
    e.state.set_playing(true);
    e.layers.force = true;
    e.bake(Detail::Anim).unwrap();
    e.layers.force = false;
    e.state.drop_request();
    e.draw(false);
    e.gpu.wait();
    e.state.set_playing(false);
}
fn next(e: &mut Engine) {
    e.frame(240, 180, 0.0);
    e.gpu.wait();
}
fn finish(e: &mut Engine) {
    for _ in 0..128 {
        next(e);
        if !e.has_layer_refinement() {
            return;
        }
    }
    panic!("the exact fields did not finish");
}
fn fits(e: &Engine) -> Vec<f64> {
    (0..2).map(|i| e.cast_thick(i).unwrap()).collect()
}
fn image(e: &mut Engine) -> Vec<u8> {
    e.renderer.reseed(0);
    e.render(12);
    e.screenshot().unwrap().2
}

#[test]
fn stopped_field_builds_publish_a_coherent_scene_and_cancel_on_edits() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "solids"),
        ("preset", "superellipsoid"),
        ("res", "512"),
        ("grain", "0"),
        ("layer", "solids:ring"),
        ("L2.x", "0.58"),
        ("L2.size", "0.5"),
        ("J12.op", "apart"),
        ("material", "satin"),
        ("B2.material", "iron"),
        ("fit", "1"),
        ("B2.fit", "1"),
        ("spin", "0"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(240, 180);
    e.bake(Detail::Full).unwrap();
    e.render(2);
    e.auto = false;
    motion(&mut e);
    next(&mut e);
    assert!(e.has_layer_refinement());
    // Reach the combined volume and then interrupt it with a new playback/edit.
    for _ in 0..32 {
        if e.layers
            .pending
            .as_ref()
            .is_some_and(|s| s.next_slice > 0 && s.layer == s.fields.len())
        {
            break;
        }
        next(&mut e);
    }
    assert!(e
        .layers
        .pending
        .as_ref()
        .is_some_and(|s| s.next_slice > 0 && s.layer == s.fields.len()));
    e.state.set_playing(true);
    assert!(e.state.apply_arg("L2.x", "0.31"));
    next(&mut e);
    assert!(!e.has_layer_refinement());
    e.state.set_playing(false);
    next(&mut e);
    assert!(e.has_layer_refinement());
    finish(&mut e);
    let thickness = fits(&e);
    let settled = image(&mut e);
    e.layers.force = true;
    e.bake(Detail::Full).unwrap();
    e.layers.force = false;
    e.settle_lighting();
    assert_eq!(thickness, fits(&e));
    assert_eq!(settled, image(&mut e));

    motion(&mut e);
    next(&mut e);
    assert!(e.has_layer_refinement());
    e.state.select_layer(0);
    let part = Part {
        place: Placement {
            pos: [0.0, 0.0, 0.68],
            size: 0.58,
            ..Default::default()
        },
        ..Default::default()
    };
    let path = e.state.csg.add_construction_part(&[], part).unwrap();
    let cut = e.state.csg.edit_construction().node_mut(&path).unwrap();
    cut.op = Op::Subtract;
    cut.cut_material = CutMaterial::Subtractor;
    cut.material = Some(BTreeMap::from([
        ("metal".into(), 3.),
        ("rough".into(), 0.1),
    ]));
    e.state.request(gf_core::build::BuildRequest::script(Detail::Full));
    next(&mut e);
    assert!(!e.has_layer_refinement());
    assert!(e.renderer.variant().world_faces);
    let dressed = image(&mut e);
    e.layers.force = true;
    e.bake(Detail::Full).unwrap();
    e.layers.force = false;
    assert_eq!(dressed, image(&mut e));

    // Playback retains only the large allocation already used by this assembly.
    motion(&mut e);
    assert!(e
        .layers
        .combiner
        .bodies_spare
        .as_ref()
        .is_some_and(|(v, _)| v.res > 384));
    let ordinary = [
        ("scene", "solids"),
        ("preset", "superellipsoid"),
        ("res", "64"),
        ("spin", "0"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    e.state = State::with_args(&ordinary, true);
    e.bake(Detail::Full).unwrap();
    e.render(1);
    assert!(!e.has_layer_refinement());
    assert!(e.layers.combiner.bodies_spare.is_none());
    assert!(e.layers.combiner.bodies_volume().is_none());
}

#[test]
fn more_objects_than_one_pass_settle_in_pieces_to_the_same_picture() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    // five objects apart (formula fields and tubes): the combining step goes in passes
    let mut args: Vec<(String, String)> = [
        ("scene", "solids"),
        ("preset", "superellipsoid"),
        ("res", "512"),
        ("grain", "0"),
        ("spin", "0"),
        ("layer", "solids:ring"),
        ("layer", "attractors:thomas"),
        ("layer", "solids:ring"),
        ("layer", "attractors:thomas"),
    ]
    .map(|(k, v)| (k.into(), v.into()))
    .to_vec();
    for i in 1..=5 {
        let x = (i as f64 - 3.0) * 0.36;
        args.push((format!("L{i}.x"), format!("{x}")));
        args.push((format!("L{i}.size"), "0.3".into()));
        for j in 1..i {
            args.push((format!("J{j}{i}.op"), "apart".into()));
        }
    }
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(240, 180);
    e.bake(Detail::Full).unwrap();
    assert_eq!(e.state.layer_count(), 5);
    e.render(2);
    e.auto = false;
    motion(&mut e);
    next(&mut e);
    assert!(e.has_layer_refinement());
    // pieces: at least one frame per layer built again and two bands of the combined volume
    let mut frames = 1;
    while e.has_layer_refinement() {
        next(&mut e);
        frames += 1;
        assert!(frames < 64, "the settled build did not finish");
    }
    assert!(frames >= 5, "only {frames} frames: the build was not split");
    assert!(e.layers.last_multi.is_some());
    let settled = image(&mut e);
    e.layers.force = true;
    e.bake(Detail::Full).unwrap();
    e.layers.force = false;
    assert_eq!(settled, image(&mut e));
}
