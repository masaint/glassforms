use gf_core::{
    groups::Item,
    modifier_pivot::Pivot,
    mods::{Axes, Kind, Modifier},
    state::{Detail, State},
    world_mods::Scope,
};
use glassforms::{engine::Engine, gpu::Gpu};

fn scene() -> State {
    let args = [
        ("scene", "studio"),
        ("preset", "sphere"),
        ("p.radius", "0.24"),
        ("res", "96"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    State::with_args(&args, true)
}

fn mirror() -> Modifier {
    let mut mirror = Modifier::of(Kind::Mirror);
    mirror.mode = 1;
    mirror.scope = Scope::SubObjects;
    mirror.pivot = Pivot::World;
    mirror.centre = [0.95, 0., 0.];
    mirror.cascade = true;
    mirror
}

fn sample(e: &Engine, x: f64) -> f32 {
    let field = e.scene_volume().unwrap();
    let side = field.res;
    let ix = ((x / e.state.scene_hb() + 1.) * 0.5 * f64::from(side)).floor() as u32;
    assert!(ix < side, "point {x} is outside the scene");
    let bytes = e
        .gpu
        .read(&field.tex, ix, side / 2, side / 2, 1, 1, 2)
        .unwrap();
    half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

#[test]
fn remote_world_mirror_preserves_all_occurrences_and_their_mass() {
    let source = Item::Object(0);
    let mut state = scene();
    let mut repeat = Modifier::of(Kind::Repeat);
    repeat.fit = false;
    repeat.a = Axes::new([2., 1., 1.], false);
    repeat.b = Axes::new([1.1, 1., 1.], false);
    state.stack.world_modifiers_mut(source).unwrap().list = vec![repeat, mirror()];
    state.stack.box_held = Some(3.);
    let mut e = Engine::new(Gpu::headless("vulkan").unwrap(), state, true);

    for held in [Some(3.), None] {
        e.state.stack.box_held = held;
        e.bake(Detail::Full).unwrap();
        let centres = [-0.55, 0.55, 1.35, 2.45].map(|x| sample(&e, x));
        let bounds = e.state.world_bounds(source).unwrap();
        eprintln!(
            "scene half-width {}, centres {centres:?}, x bounds {} to {}",
            e.state.scene_hb(),
            bounds[0],
            bounds[3]
        );
        assert!(e.state.scene_hb() > 2.69);
        assert!(centres.iter().all(|&d| d < 0.));
        assert!([0., 0.95, 1.9].iter().all(|&x| sample(&e, x) > 0.));
        assert!((bounds[0] + 0.79).abs() < 0.07 && (bounds[3] - 2.69).abs() < 0.07);
    }

    let mut turn = Modifier::of(Kind::Twist);
    turn.pivot = Pivot::CentreOfMass;
    e.state
        .stack
        .world_modifiers_mut(source)
        .unwrap()
        .list
        .push(turn);
    e.bake(Detail::Full).unwrap();
    let centre = e.state.stack.world_pivots[&(source, 2)];
    assert!((centre[0] - 0.95).abs() < 0.04, "{centre:?}");
}

#[test]
fn shared_remote_geometry_survives_save_and_returns_to_its_source_in_object_mode() {
    let source = Item::Object(0);
    let mut state = scene();
    state
        .stack
        .world_modifiers_mut(source)
        .unwrap()
        .list
        .push(mirror());
    let setup = state.setup();
    state.apply_setup(&setup).unwrap();
    let mut e = Engine::new(Gpu::headless("vulkan").unwrap(), state, true);
    e.bake(Detail::Full).unwrap();
    assert!(!e.state.stack.world_plan().has_distinct_fields(0));
    assert!(sample(&e, 0.) < 0. && sample(&e, 0.95) > 0. && sample(&e, 1.9) < 0.);
    assert!((e.state.world_bounds(source).unwrap()[3] - 2.14).abs() < 0.06);
    e.state.edit_scope.inside = Some(gf_core::edit_scope::ObjectScope::new(0));
    e.bake(Detail::Full).unwrap();
    assert!(sample(&e, 0.) < 0. && sample(&e, 0.95) > 0.);
    e.state.edit_scope.inside = None;
    e.bake(Detail::Full).unwrap();
    assert!(sample(&e, 1.9) < 0.);
}
