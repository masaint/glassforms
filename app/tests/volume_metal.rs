use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

const WIDTH: usize = 480;
const HEIGHT: usize = 360;

fn pierced_metal(gpu: Gpu, trace: &str) -> Vec<f32> {
    let settings = [
        ("scene", "solids"),
        ("preset", "pierced"),
        ("trace", trace),
        ("material", "brushed"),
        ("body", "8"),
        ("bore", "2"),
        ("cavity", "0"),
        ("hole", ".48"),
        ("round", ".065"),
        ("rx", "58"),
        ("ry", "14"),
        ("rz", "0"),
        ("fov", "31"),
        ("dist", "5.7"),
        ("res", "192"),
        ("spin", "0"),
        ("meter", "0"),
        ("scale", "1"),
        ("grain", "0"),
        ("bloom", "0"),
    ]
    .map(|(k, v)| (k.to_owned(), v.to_owned()));
    let mut engine = Engine::new(gpu, State::with_args(&settings, true), true);
    engine.resize(WIDTH as u32, HEIGHT as u32);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    assert_eq!(engine.renderer.sdf_traced(), trace == "csg");
    engine.renderer.reseed(3);
    engine.render(128);
    engine
        .renderer
        .read_hdr(0, 0, WIDTH as u32, HEIGHT as u32)
        .unwrap()
}

#[test]
fn a_brushed_volume_tracks_the_reflections_on_the_analytic_solid() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let analytic = pierced_metal(gpu.clone(), "csg");
    let volume = pierced_metal(gpu, "volume");
    let on_both = |x: usize, y: usize| {
        let i = 4 * (y * WIDTH + x) + 3;
        analytic[i] > 0.0 && volume[i] > 0.0
    };
    let mut sum = 0.0;
    let mut pixels = 0;
    // Leave five pixels at the silhouette so image error measures the interior metal response.
    for y in 5..HEIGHT - 5 {
        for x in 5..WIDTH - 5 {
            if !(on_both(x, y)
                && on_both(x - 5, y)
                && on_both(x + 5, y)
                && on_both(x, y - 5)
                && on_both(x, y + 5))
            {
                continue;
            }
            for c in 0..3 {
                let i = 4 * (y * WIDTH + x) + c;
                sum += f64::from(analytic[i] - volume[i]).powi(2);
            }
            pixels += 1;
        }
    }
    assert!(pixels > 20_000);
    let rms = (sum / f64::from(3 * pixels)).sqrt();
    assert!(
        rms < 0.045,
        "linear RGB RMS {rms} across {pixels} interior pixels"
    );
}
