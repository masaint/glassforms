use gf_core::{
    construction::CutMaterial,
    csg::{Op, Part},
    layers::Placement,
    state::{Detail, State},
};
use glassforms::{engine::Engine, gpu::Gpu};
use std::collections::BTreeMap;

fn draw(engine: &mut Engine, first_surface_only: bool) -> Vec<u8> {
    engine.bake(Detail::Full).unwrap();
    engine.render(6);
    let (_, text) = engine
        .renderer
        .live_key_text()
        .expect("active material tracer");
    assert_eq!(
        text.contains("\n  return paintB_seen(k, p, n, rd, 1.0);\n"),
        first_surface_only
    );
    engine.screenshot().unwrap().2
}

#[test]
fn the_subsurface_meter_leaves_opaque_paint_at_the_scene_exposure() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let args = [
        ("scene", "studio"), ("preset", "sphere"), ("layer", "studio:sphere"),
        ("J12.op", "apart"), ("L1.x", "-0.45"), ("L2.x", "0.45"),
        ("L1.size", "0.4"), ("L2.size", "0.4"), ("res", "96"),
        ("material", "modelclay"), ("B2.material", "satinpaint"),
        ("rig", ""), ("ambient", "0.07,0.07,0.07"), ("envRoom", "0"),
        ("backdrop", "0"), ("meter", "1"), ("spin", "0"),
    ].map(|(key, value)| (key.into(), value.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(160, 112);
    for (material, metered) in [("satinpaint", false), ("crown", false), ("wax", true), ("porcelain", false)] {
        assert!(e.state.apply_arg("B2.material", material));
        e.bake(Detail::Full).unwrap();
        e.render(2);
        let inverse_meter = f32::from_bits(e.trace_uniforms().pad[0] as u32);
        if metered {
            assert!(inverse_meter < 0.9, "the wax exposure did not respond to the dark room");
        } else {
            assert_eq!(inverse_meter, 1.0, "{material} changed the exposure of the paint beside it");
        }
    }
}

#[test]
fn editing_a_body_or_a_cut_reselects_the_material_tracer() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "studio"),
        ("preset", "block"),
        ("trace", "csg"),
        ("res", "96"),
        ("grain", "0"),
        ("layer", "studio:sphere"),
        ("L2.x", "1.01"),
        ("L2.size", "0.4"),
        ("J12.op", "apart"),
        ("material", "lacquer"),
        ("B2.material", "lacquer"),
        ("bodyR", "0.7"),
        ("B2.bodyB", "0.8"),
        ("age", "0"),
        ("B2.age", "0"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    let state = State::with_args(&args, true);
    let mut e = Engine::new(gpu, state, true);
    e.resize(240, 180);
    let painted = draw(&mut e, true);
    assert!(e.state.apply_arg("B2.material", "crown"));
    let mixed = draw(&mut e, false);
    assert_ne!(painted, mixed);
    assert!(e.state.apply_arg("B2.material", "lacquer"));
    draw(&mut e, true);

    e.state.select_layer(0);
    let part = Part {
        place: Placement {
            pos: [0.0, 0.0, 0.71],
            size: 0.58,
            ..Default::default()
        },
        ..Default::default()
    };
    let path = e.state.csg.add_construction_part(&[], part).unwrap();
    let cut = e.state.csg.edit_construction().node_mut(&path).unwrap();
    cut.op = Op::Subtract;
    cut.cut_material = CutMaterial::Recipient;
    let inherited = draw(&mut e, true);
    assert!(!e.renderer.variant().world_faces);

    let cut = e.state.csg.edit_construction().node_mut(&path).unwrap();
    cut.cut_material = CutMaterial::Subtractor;
    cut.material = Some(BTreeMap::from([
        ("metal".to_string(), 3.0),
        ("rough".to_string(), 0.12),
    ]));
    let dressed = draw(&mut e, false);
    assert!(e.renderer.variant().world_faces);
    let changed = inherited
        .chunks_exact(4)
        .zip(dressed.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|index| a[index].abs_diff(b[index]) > 12))
        .count();
    assert!(
        changed > 24,
        "dressing an existing cut changed only {changed} pixels"
    );
}
