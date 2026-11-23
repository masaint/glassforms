use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn lossless_nested_frost_keeps_uniform_light_at_low_sample_counts() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let settings = [
        ("scene", "studio"),
        ("preset", "sphere"),
        ("L1.x", "-0.12"),
        ("L1.size", "0.80"),
        ("material", "glass"),
        ("ior", "1.333"),
        ("rough", ".20"),
        ("roughRef", "1"),
        ("disp", "0"),
        ("density", "0"),
        ("layer", "studio:sphere"),
        ("L2.x", "0.18"),
        ("L2.z", "-0.39"),
        ("L2.size", ".65"),
        ("J12.op", "apart"),
        ("B2.material", "glass"),
        ("B2.ior", "1.62"),
        ("B2.rough", ".32"),
        ("B2.roughRef", "1"),
        ("B2.disp", "0"),
        ("B2.density", "0"),
        ("rx", "0"),
        ("ry", "0"),
        ("rz", "0"),
        ("dist", "4.6"),
        ("res", "128"),
        ("rig", ""),
        ("ambient", ".25,.25,.25"),
        ("lightGain", "1"),
        ("envRoom", "0"),
        ("floor", "0"),
        ("backdrop", "0"),
        ("spin", "0"),
        ("grain", "0"),
        ("bloom", "0"),
        ("adapt", "0"),
    ]
    .map(|(key, value)| (key.to_string(), value.to_string()));
    const SIDE: u32 = 144;
    let mut engine = Engine::new(gpu, State::with_args(&settings, true), true);
    engine.resize(SIDE, SIDE);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    engine.render(16);
    let image = engine.renderer.read_hdr(0, 0, SIDE, SIDE).unwrap();
    let width = SIDE as usize;
    let at = |x: usize, y: usize| 4 * (y * width + x);
    let mut squared = 0.0;
    let mut count = 0;
    for y in 1..width - 1 {
        for x in 1..width - 1 {
            if [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .iter()
                .any(|&(u, v)| image[at(u, v) + 3] <= 0.0)
            {
                continue;
            }
            squared += f64::from(image[at(x, y)] - 0.25).powi(2);
            count += 1;
        }
    }
    assert!(count > 1000);
    let rmse = (squared / f64::from(count)).sqrt();
    assert!(
        rmse < 0.045,
        "16-frame RMSE {rmse} under uniform radiance 0.25"
    );
}
