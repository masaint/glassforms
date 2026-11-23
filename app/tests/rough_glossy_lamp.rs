use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

fn fixture(gpu: Gpu, coat: &str, ambient: &str, lamp: &str, rough: &str, sampled: &str) -> Engine {
    let settings = [
        ("scene", "studio"),
        ("preset", "sphere"),
        ("material", "copper"),
        ("rough", rough),
        ("roughRef", sampled),
        ("age", "0"),
        ("brush", "0"),
        ("mottle", "0"),
        ("lightGain", "1"),
        ("backdrop", "0"),
        ("rig", ""),
        ("coat", coat),
        ("coatIor", "1.38"),
        ("ambient", ambient),
        ("lamp1", lamp),
        ("lamp1.seen", "0"),
        ("floor", "0"),
        ("dist", "4.8"),
        ("fov", "32"),
        ("spin", "0"),
        ("rx", "0"),
        ("ry", "0"),
        ("rz", "0"),
        ("res", "128"),
        ("grain", "0"),
        ("bloom", "0"),
        ("envRoom", "0"),
    ]
    .map(|(key, value)| (key.to_string(), value.to_string()));
    let mut engine = Engine::new(gpu, State::with_args(&settings, true), true);
    engine.resize(SIZE, SIZE);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    engine
}

const SIZE: u32 = 128;

fn image(engine: &mut Engine, frames: u32) -> Vec<f32> {
    engine.render(frames);
    engine.renderer.read_hdr(0, 0, SIZE, SIZE).unwrap()
}

fn rmse(low: &[f32], high: &[f32]) -> f64 {
    let mut squared_error = 0.0;
    let mut radiance = 0.0;
    let mut count = 0;
    for (a, b) in low.chunks_exact(4).zip(high.chunks_exact(4)) {
        if a[3] <= 0.0 || b[3] <= 0.0 {
            continue;
        }
        for c in 0..3 {
            squared_error += f64::from(a[c] - b[c]).powi(2);
            radiance += f64::from(b[c]);
            count += 1;
        }
    }
    assert!(count > 1000 && radiance / f64::from(count) > 0.01);
    (squared_error / f64::from(count)).sqrt()
}

#[test]
fn a_small_near_lamp_converges_on_rough_copper() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let mut engine = fixture(
        gpu,
        "0",
        "0,0,0",
        "ball,1.4,0.8,2.1,.2,0,1,1,1,35",
        ".65",
        "1",
    );
    let low = image(&mut engine, 16);
    let high = image(&mut engine, 256);
    let rmse = rmse(&low, &high);
    assert!(
        rmse < 0.055,
        "16-frame linear RGB RMSE {rmse} under finite sphere"
    );
}

#[test]
fn a_lamp_matching_its_surroundings_adds_no_grain_to_the_coating() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    // Compare the same 16 samples with and without a light that replaces equally bright sky.
    // Silhouette antialiasing and the finite-sample variance of the substrate are common to both.
    let mut empty = fixture(gpu.clone(), "1", ".25,.25,.25", "", ".65", "1");
    let baseline = image(&mut empty, 16);
    drop(empty);
    let mut matched = fixture(
        gpu,
        "1",
        ".25,.25,.25",
        "ball,.9,.4,2.1,.48,0,1,1,1,.25",
        ".65",
        "1",
    );
    let radiance = image(&mut matched, 16);
    let rmse = rmse(&baseline, &radiance);
    assert!(
        rmse < 0.001,
        "16-frame coated copper RMSE {rmse} in a uniform environment"
    );
}

#[test]
fn the_live_copper_highlight_tracks_the_finite_sphere() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let lamp = "ball,1.4,0.8,2.1,.2,0,1,1,1,35";
    for coat in ["0", "1"] {
        let mut live = fixture(gpu.clone(), coat, "0,0,0", lamp, ".35", "0");
        let live = image(&mut live, 64);
        let mut sampled = fixture(gpu.clone(), coat, "0,0,0", lamp, ".35", "1");
        let sampled = image(&mut sampled, 384);
        let error = rmse(&live, &sampled);
        let mut energy = [0.0_f64; 2];
        for (a, b) in live.chunks_exact(4).zip(sampled.chunks_exact(4)) {
            if a[3] > 0.0 && b[3] > 0.0 {
                for c in 0..3 {
                    energy[0] += f64::from(a[c]);
                    energy[1] += f64::from(b[c]);
                }
            }
        }
        let ratio = energy[0] / energy[1];
        assert!(
            (0.85..1.15).contains(&ratio) && error < 0.065,
            "coat {coat}, live/sample energy {ratio}, highlight RMS {error}"
        );
    }
}
