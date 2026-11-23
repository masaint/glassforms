use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

const W: u32 = 192;
const H: u32 = 168;

fn setup(gpu: &Gpu, pane: bool, index: &str) -> Engine {
    let options = [
        ("scene", "studio"),
        ("preset", "block"),
        ("studio", "none"),
        ("p.width", "1.3"),
        ("p.height", "1.12"),
        ("p.depth", ".2"),
        ("p.bevel", "0"),
        ("material", "crown"),
        ("ior", index),
        ("density", "0"),
        ("disp", "0"),
        ("age", "0"),
        ("L1.z", ".88"),
        ("layer", "studio:sphere"),
        ("L2.size", ".57"),
        ("L2.z", "-.12"),
        ("J12.op", "apart"),
        ("B2.material", "iron"),
        ("B2.age", "1"),
        ("B2.brush", "1"),
        ("B2.brushAngle", "0"),
        ("B2.rough", ".28"),
        ("B2.roughRef", "0"),
        ("B2.mottle", "0"),
        ("layer", "studio:sphere"),
        ("L3.x", "1.12"),
        ("L3.size", ".12"),
        ("J23.op", "apart"),
        ("B3.material", "crown"),
        ("B3.age", "0"),
        ("floor", "0"),
        ("meter", "0"),
        ("rx", "0"),
        ("ry", "0"),
        ("rz", "0"),
        ("dist", "18"),
        ("fov", "8.5"),
        ("nb", "12"),
        ("res", "144"),
        ("spin", "0"),
        ("grain", "0"),
    ]
    .map(|(key, value)| (key.to_owned(), value.to_owned()));
    let mut state = State::with_args(&options, true);
    state.stack.layers[0].show = pane;
    let mut engine = Engine::new(gpu.clone(), state, true);
    engine.resize(W, H);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    engine
}

fn rotation_pair(engine: &mut Engine, seed: u32) -> [Vec<f32>; 2] {
    ["0", "90"].map(|angle| {
        assert!(engine.state.apply_arg("B2.brushAngle", angle));
        engine.renderer.reseed(seed * 7919);
        engine.render(128);
        engine.renderer.read_hdr(0, 0, W, H).unwrap()
    })
}

#[test]
fn the_polishing_direction_survives_a_planar_clear_window() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let open = rotation_pair(&mut setup(&gpu, false, "1"), 3);
    for index in ["1", "1.5185"] {
        let mut covered = setup(&gpu, true, index);
        let glass = rotation_pair(&mut covered, 3);
        let repeated = rotation_pair(&mut covered, 11);
        let mut pixels = 0;
        let mut open_energy = 0.0;
        let mut window_energy = 0.0;
        let mut repeated_energy = 0.0;
        let mut direct_agreement = 0.0;
        let mut seed_agreement = 0.0;
        for y in H / 2 - 29..H / 2 + 29 {
            for x in W / 2 - 34..W / 2 + 34 {
                let i = (4 * (y * W + x)) as usize;
                if open[0][i + 3] <= 0.0 || glass[0][i + 3] + 0.15 >= open[0][i + 3] {
                    continue;
                }
                let difference = |images: &[Vec<f32>; 2], channel| {
                    f64::from(images[1][i + channel] - images[0][i + channel])
                };
                let before = [
                    difference(&open, 0),
                    difference(&open, 1),
                    difference(&open, 2),
                ];
                if before.iter().all(|v| v.abs() < 0.012) {
                    continue;
                }
                pixels += 1;
                for c in 0..3 {
                    let after = difference(&glass, c);
                    let check = difference(&repeated, c);
                    open_energy += before[c] * before[c];
                    window_energy += after * after;
                    repeated_energy += check * check;
                    direct_agreement += before[c] * after;
                    seed_agreement += after * check;
                }
            }
        }
        let direct_slope = direct_agreement / open_energy;
        let same_finish = seed_agreement / (window_energy * repeated_energy).sqrt();
        eprintln!("index {index}, {pixels} pixels, signed response {direct_slope:.3}, repeated response {same_finish:.3}");
        assert!(pixels > 500 && open_energy > 0.05);
        // Matching the pane to air keeps the signed brush signal registered. Crown bends
        // the rays, so confirm its reformed highlight in two independent sample sequences.
        if index == "1" {
            assert!(
                direct_slope > 0.6,
                "the index-matched pane lost the open highlight"
            );
        }
        assert!(
            seed_agreement > open_energy * 0.25 && same_finish > 0.7,
            "the transmitted brush response disappeared or became uncorrelated sampling noise"
        );
    }
}

#[test]
fn secondary_live_finish_is_only_compiled_for_an_active_conductor() {
    use gf_core::{material::Variant, settings::Settings};
    let metal = Settings {
        metal: 10.0,
        rough: 0.28,
        rough_ref: 0.0,
        brush: 1.0,
        ..Settings::default()
    };
    let glass = Settings::default();
    let sampled = Settings {
        rough_ref: 1.0,
        ..metal.clone()
    };
    let unmarked = Settings {
        age: 0.0,
        brush: 0.0,
        mottle: 0.0,
        ..metal.clone()
    };
    let only_paint_worn = Settings {
        metal: 0.0,
        ..metal.clone()
    };
    assert!(Variant::of_bodies(&[&metal, &glass]).live_conductor_finish);
    assert!(!Variant::of_bodies(&[&sampled, &glass]).live_conductor_finish);
    assert!(!Variant::of_bodies(&[&unmarked, &only_paint_worn]).live_conductor_finish);
}
