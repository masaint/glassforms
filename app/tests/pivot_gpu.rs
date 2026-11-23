use gf_core::groups::Item;
use gf_core::modifier_pivot::Pivot;
use gf_core::mods::{Axes, Kind, Modifier};
use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn later_mass_pivots_measure_the_actual_asymmetric_input() {
    let args = [("scene","studio"),("preset","sphere"),("trace","csg"),("res","128"),("p.radius","0.3")]
        .into_iter().map(|(k,v)| (k.to_owned(),v.to_owned())).collect::<Vec<_>>();
    let mut state = State::with_args(&args, true);
    // Reflect the positive half about a plane offset toward +x. A following
    // twist must see that changed solid, independently of its own angle.
    let mut mirror = Modifier::of(Kind::Mirror);
    mirror.centre = [0.15, 0., 0.];
    mirror.a = Axes::new([1., 0., 0.], false);
    let mut turn = Modifier::of(Kind::Twist);
    turn.pivot = Pivot::CentreOfMass;
    state.shape_mut().mods.list = vec![mirror, turn.clone()];
    let mut e = Engine::new(Gpu::headless("vulkan").unwrap(), state, true);
    e.bake(Detail::Full).unwrap();
    let first = e.state.modifier_pivots[&(Item::Object(0), 1)];
    assert!((first[0] - 0.15).abs() < 0.06, "incoming centre {first:?}");
    e.state.shape_mut().mods.list[1].amount = 0.5;
    e.bake(Detail::Full).unwrap();
    assert_eq!(first, e.state.modifier_pivots[&(Item::Object(0), 1)]);
    assert_eq!(e.state.shape().mods.list[1].pivot, Pivot::CentreOfMass);
}
