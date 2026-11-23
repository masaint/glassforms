use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 144;

fn ring(gpu: Gpu, sampled: bool, with_neighbour: bool) -> Vec<f32> {
    let mut options = vec![
        ("scene", "solids"),
        ("preset", "ring"),
        ("studio", "none"),
        ("rx", "58"),
        ("ry", "14"),
        ("rz", "0"),
        ("dist", "5.7"),
        ("fov", "31"),
        ("spin", "0"),
        ("grain", "0"),
        ("res", "144"),
        ("floor", "0"),
        ("L1.x", "-.32"),
        ("L1.size", ".69"),
        ("material", "copper"),
        ("rough", ".65"),
        ("age", "0"),
        ("brush", "0"),
        ("mottle", "0"),
        ("roughRef", if sampled { "1" } else { "0" }),
    ];
    if with_neighbour {
        options.extend([
            ("layer", "studio:sphere"),
            ("L2.x", ".85"),
            ("L2.y", ".14"),
            ("L2.z", "-.4"),
            ("L2.size", ".19"),
            ("J12.op", "apart"),
            ("B2.material", "crown"),
            ("B2.age", "0"),
        ]);
    }
    let settings: Vec<_> = options
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    let mut engine = Engine::new(gpu, State::with_args(&settings, true), true);
    engine.resize(WIDTH, HEIGHT);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    engine.render(384);
    engine.renderer.read_hdr(0, 0, WIDTH, HEIGHT).unwrap()
}

#[test]
fn a_separate_glass_body_preserves_the_copper_sampling_control() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let solo = ring(gpu.clone(), true, false);
    let sampled = ring(gpu.clone(), true, true);
    let live = ring(gpu, false, true);
    let mut squared = [0.0_f64; 2];
    let mut channels = 0;
    let width = WIDTH as usize;
    // Exclude the silhouette, where geometry resampling can move the edge.
    for y in 2..HEIGHT as usize - 2 {
        for x in 2..width - 2 {
            let i = 4 * (y * width + x);
            if [i, i - 8, i + 8, i - 8 * width, i + 8 * width]
                .iter()
                .any(|&j| solo[j + 3] == 0.0)
            {
                continue;
            }
            for c in 0..3 {
                squared[0] += f64::from(solo[i + c] - sampled[i + c]).powi(2);
                squared[1] += f64::from(solo[i + c] - live[i + c]).powi(2);
                channels += 1;
            }
        }
    }
    assert!(channels > 2_000);
    let [sampled_error, live_error] = squared.map(|v| (v / f64::from(channels)).sqrt());
    assert!(
        live_error > 0.035 && sampled_error < live_error * 0.65,
        "ordinary-room copper RMS for a sampled assembly {sampled_error}, live {live_error}"
    );
}
