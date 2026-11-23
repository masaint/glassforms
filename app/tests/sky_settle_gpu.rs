//! The live view bakes the cloud dome coarse, a few pieces a frame, while the sky moves, and refines it
//! once the sky rests (clouds.rs). Once it has rested, what the tracer reads of the sky must be exactly
//! what a still made of that sky reads: the lights' map, the floor, the panorama, the sun's
//! transmittance, the dome and the shadow map. (A binary of its own: `GF_SKYHASH` is read once.)

use gf_core::state::{Detail, State};
use glassforms::clouds::TOP;
use glassforms::engine::Engine;
use glassforms::gpu::Gpu;

#[test]
fn a_sky_moved_then_left_settles_to_the_bake_a_still_makes() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    std::env::set_var("GF_SKYHASH", "1");
    let args = [
        ("scene", "solids"),
        ("preset", "sphere"),
        ("sky", "day"),
        ("floor", "1"),
        ("floorGloss", "0.3"),
        ("cloudCover", "0.5"),
        ("sunRays", "0.4"),
        ("sunPitch", "30"),
        ("res", "64"),
    ]
    .map(|(k, v)| (k.to_string(), v.to_string()));
    let (w, h) = (160u32, 120u32);
    let make = |gpu: Gpu| {
        let mut e = Engine::new(gpu, State::with_args(&args, true), true);
        e.renderer.resize(w, h, 1.0);
        e.bake(Detail::Full).unwrap();
        e
    };

    let mut live = make(gpu.clone());
    live.auto = false;
    let mut coarse_seen = false;
    for i in 0..16 {
        live.state.p.set("sunPitch", 30.0 + i as f64 * 0.5);
        live.frame(w, h, 1.0 / 90.0);
        live.gpu.wait();
        coarse_seen |= live.clouds.level() < TOP;
    }
    assert!(coarse_seen, "a moving sky must be shown from the coarse bakes");
    assert!(!live.sky_settled());
    let t0 = std::time::Instant::now();
    while !live.sky_settled() && t0.elapsed().as_secs() < 30 {
        live.frame(w, h, 1.0 / 90.0);
        live.gpu.wait();
    }
    live.frame(w, h, 1.0 / 90.0);
    live.gpu.wait();
    assert!(live.sky_settled() && live.clouds.level() == TOP);

    let mut still = make(gpu);
    still.state.p.set("sunPitch", 37.5);
    still.render(2);
    assert_eq!(live.sky_fingerprint(), still.sky_fingerprint());
}

fn live_engine(gpu: Gpu, extra: &[(&str, &str)]) -> Engine {
    let args: Vec<(String, String)> = [("scene", "solids"), ("preset", "sphere"), ("sky", "day"), ("floor", "1"), ("cloudCover", "0.5"), ("sunPitch", "30"), ("res", "64")]
        .iter()
        .chain(extra)
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.renderer.resize(160, 120, 1.0);
    e.bake(Detail::Full).unwrap();
    e.render(1);
    e.auto = false;
    e
}

/// A sky knob that changes every frame (the sun dragged or animated) must not hold the frame up for a
/// whole bake of the dome, as it once did (six faces, the panorama and a wait for the GPU, every frame):
/// each frame runs a few pieces of the bake and goes on.
#[test]
fn a_moving_sun_never_bakes_the_whole_dome_in_one_frame() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let mut e = live_engine(gpu, &[]);
    assert_eq!(e.clouds.level(), TOP);
    let mut partial = 0;
    for i in 0..24 {
        e.state.p.set("sunPitch", 31.0 + i as f64 * 0.25);
        e.frame(160, 120, 1.0 / 90.0);
        e.gpu.wait();
        if let Some((done, all)) = e.clouds.progress() {
            if done < all { partial += 1; }
        }
    }
    assert!(partial >= 12, "the bake must be spread over frames ({partial} frames with one half done)");
    // and the shown sky keeps following the sun: several coarse bakes landed
    assert!(e.clouds.bake_count() >= 3, "{}", e.clouds.bake_count());
    assert!(e.clouds.level() == 0);
}

/// While an environment picture is still being read the sky stays in use (the engine keeps the old
/// lighting until the picture is ready): the cloud dome must not drop out in the meantime.
#[test]
fn the_cloud_dome_stays_while_a_picture_loads() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let mut e = Engine::new(gpu, State::with_args(&[("scene", "solids"), ("preset", "sphere"), ("sky", "day"), ("cloudCover", "0.5"), ("res", "64")].map(|(k, v)| (k.to_string(), v.to_string())), true), false);
    e.renderer.resize(160, 120, 1.0);
    e.bake(Detail::Full).unwrap();
    for _ in 0..4 { e.frame(160, 120, 1.0 / 90.0); e.gpu.wait(); }
    assert!(e.clouds.is_active());
    // a large plain picture, so that reading it takes a while
    let (w, h) = (4096usize, 2048usize);
    let mut bytes = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n").into_bytes();
    bytes.extend(std::iter::repeat([128u8, 128, 128, 129]).take(w * h).flatten());
    let path = std::env::temp_dir().join(format!("gf_sky_loading_{}.hdr", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    e.state.hdri = Some(path.to_string_lossy().into_owned());
    let mut checked = 0;
    for _ in 0..2000 {
        e.frame(160, 120, 1.0 / 90.0);
        e.gpu.wait();
        if !e.hdri_loading() { break; }
        assert!(e.clouds.is_active(), "the dome dropped out while the picture was still loading");
        checked += 1;
    }
    let _ = std::fs::remove_file(&path);
    assert!(checked > 0, "the picture loaded within one frame: nothing was checked");
}
