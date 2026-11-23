use std::sync::OnceLock;

use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

fn wax_teapot() -> Option<&'static [f32]> {
    static IMAGE: OnceLock<Option<Vec<f32>>> = OnceLock::new();
    IMAGE
        .get_or_init(|| {
            let gpu = Gpu::headless("vulkan").ok()?;
            let model =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/teapot.obj");
            let scene = [
                ("scene", "models"),
                ("preset", "file"),
                ("model", model.to_str().unwrap()),
                ("rot", "18,-28,0"),
                ("material", "wax"),
                ("studio", "metal"),
                ("res", "256"),
                ("spin", "0"),
                ("grain", "0"),
                ("elev", "15"),
                ("dist", "4.8"),
            ]
            .map(|(key, value)| (key.to_string(), value.to_string()));
            assert!(
                model.is_file(),
                "missing teapot fixture at {}",
                model.display()
            );
            let mut engine = Engine::new(gpu, State::with_args(&scene, true), true);
            engine.resize(420, 330);
            engine.bake(Detail::Full).unwrap();
            engine.settle_lighting();
            engine.render(32);
            Some(engine.renderer.read_hdr(0, 0, 420, 330).unwrap())
        })
        .as_deref()
}

#[test]
fn a_grazing_ray_through_the_wax_spout_has_finite_radiance() {
    let Some(image) = wax_teapot() else { return };
    for (index, pixel) in image.chunks_exact(4).enumerate() {
        assert!(
            pixel[..3].iter().all(|v| v.is_finite()),
            "invalid radiance at ({}, {}): {pixel:?}",
            index % 420,
            index / 420
        );
    }
}

#[test]
fn thick_wax_attenuates_the_refracted_bottom_edge() {
    let Some(image) = wax_teapot() else { return };
    // This strip of viewing rays straddles the rear wall and the model's flat bottom.
    // Average across the width to distinguish the coherent edge from sampling noise.
    let strip = |offset: i32| {
        let mut light = 0.0;
        for x in 137..270 {
            let centre = (232.1_f32 - 0.159 * x as f32).round() as i32;
            for dy in -3..=3 {
                let y = (centre + offset + dy) as usize;
                let p = &image[4 * (420 * y + x)..];
                light += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            }
        }
        light / (133 * 7) as f32
    };
    let centre = strip(-2);
    let neighbours = 0.5 * (strip(-11) + strip(7));
    // The walk gives 0.8% here. The reduced-rate ballistic calculation gave 3.3%.
    assert!(
        (centre / neighbours - 1.0).abs() < 0.02,
        "the rear rim is too visible through the wax: centre {centre}, neighbours {neighbours}"
    );
}
