//! Integration & GPU verification tests for the Volumetric Sky-Space Cloud Layer,
//! Dappled Ground Cloud Shadows, Crepuscular Sunrays, Atmospheric Mist, Cache Dirty-Tracking,
//! and Light Room / Settings / Setup / Animation parity.

use gf_core::limits;
use gf_core::math::axis_angle;
use gf_core::setup::Setup;
use gf_core::sky::{self, Sky};
use gf_core::state::{Detail, State};
use glassforms::engine::Engine;
use glassforms::gpu::Gpu;

#[test]
fn ui_knobs_limits_setup_and_animation_parity() {
    let st = State::new();
    let knobs = State::engine_knobs();
    for &name in sky::NAMES {
        assert!(st.p.get(name).is_some(), "missing setting `{name}` in Settings");
        assert!(limits::setting(name).is_some(), "missing hard limit for `{name}` in limits.rs");
    }
    for &(name, label, lo, hi, step, anim) in sky::KNOBS {
        assert!(
            label.chars().count() <= 16,
            "sky knob `{name}` label `{label}` too wide for Light Room column"
        );
        let (h_lo, h_hi) = limits::setting(name).expect("hard limit");
        assert!(
            lo >= h_lo - 1e-9 && hi <= h_hi + 1e-9,
            "knob `{name}` soft range [{lo}, {hi}] exceeds hard limits [{h_lo}, {h_hi}]"
        );
        let found = knobs.iter().find(|k| k.id == name).expect("registered in State::engine_knobs");
        assert!((found.step - step).abs() < 1e-12 && found.wave == anim);
    }

    // Default sky has zero cloud/atmosphere overhead.
    let mut clear = State::new();
    assert!(clear.sky_arg("day"));
    let sky_clear = Sky::of(&clear.p);
    assert!(sky_clear.active && !sky_clear.clouds_active() && !sky_clear.has_atmo_effects() && !sky_clear.needs_sky_space());

    // Setup JSON round-trip & animation track drift for cloudWindX / cloudWindZ.
    clear.p.set("cloudCover", 0.58);
    clear.p.set("cloudDensity", 1.35);
    clear.p.set("cloudWindX", -2.5);
    clear.p.set("cloudWindZ", 4.25);
    clear.p.set("atmoHaze", 0.35);
    clear.p.set("sunRays", 0.9);
    let json = serde_json::to_string(&clear.setup()).expect("serialize setup");
    let loaded: Setup = serde_json::from_str(&json).expect("deserialize setup");
    let mut restored = State::new();
    assert!(restored.apply_setup(&loaded).is_ok());
    assert!((restored.p.cloud_cover - 0.58).abs() < 1e-9);
    assert!((restored.p.cloud_density - 1.35).abs() < 1e-9);
    assert!((restored.p.cloud_wind_x + 2.5).abs() < 1e-9);
    assert!((restored.p.cloud_wind_z - 4.25).abs() < 1e-9);
    assert!((restored.p.atmo_haze - 0.35).abs() < 1e-9);
    assert!((restored.p.sun_rays - 0.9).abs() < 1e-9);
}

#[test]
fn cloud_cache_dirty_tracking_and_default_off_invariance() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let args = [
        ("scene", "solids"),
        ("preset", "sphere"),
        ("sky", "day"),
        ("floor", "1"),
        ("res", "64"),
    ]
    .map(|(k, v)| (k.to_string(), v.to_string()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.renderer.resize(160, 120, 1.0);
    e.bake(Detail::Full).unwrap();
    e.render(2);

    // 1. Default sky (cloudCover=0, atmo=0): cache must stay completely inactive with zero GPU bakes.
    assert!(!e.clouds.is_active(), "default clear sky must not activate volumetric sky-space");
    assert_eq!(e.clouds.bake_count(), 0, "default clear sky must perform zero cloud bakes");
    assert!(!e.renderer.variant().stage.sky_space && !e.renderer.variant().stage.atmo);

    // 2. Enable volumetric clouds + sunrays: exactly ONE GPU sky-space bake.
    e.state.p.set("cloudCover", 0.55);
    e.state.p.set("cloudDensity", 1.25);
    e.state.p.set("sunRays", 0.75);
    e.state.p.set("atmoHaze", 0.20);
    e.render(2);
    assert!(e.clouds.is_active());
    assert_eq!(e.clouds.bake_count(), 1, "enabling clouds + sunrays should bake sky-space once");
    assert!(e.renderer.variant().stage.sky_space && e.renderer.variant().stage.atmo);

    // 3. Camera orbit, pan, zoom, object rotation, rig turn, shadow-strength tweak, and multi-sample
    //    refinement MUST NOT trigger any cloud re-bake!
    e.state.p.set("camAzim", 42.0);
    e.state.p.set("camElev", 18.0);
    e.state.p.set("eyeX", 0.25);
    e.state.p.set("dist", 4.8);
    e.state.p.set("lyaw", 30.0);
    e.state.p.set("cloudShadow", 0.40);
    e.state.turn(&axis_angle(0.2, 1.0, 0.1, 0.45));
    e.render(6);
    assert_eq!(
        e.clouds.bake_count(),
        1,
        "camera orbit/pan/zoom, object turn, rig turn, cloudShadow, and refinement must pay zero cloud re-bake cost"
    );

    // 4. Changing cloud wind offset triggers exactly one re-bake.
    e.state.p.set("cloudWindX", 3.2);
    e.render(1);
    assert_eq!(e.clouds.bake_count(), 2, "changing cloudWindX must re-bake sky-space once");

    // 5. Changing sun elevation triggers exactly one re-bake.
    e.state.p.set("sunPitch", 14.0);
    e.render(1);
    assert_eq!(e.clouds.bake_count(), 3, "changing sunPitch must re-bake sky-space once");

    // 6. Turning clouds & atmosphere back off deactivates sky-space cleanly without baking.
    e.state.p.set("cloudCover", 0.0);
    e.state.p.set("sunRays", 0.0);
    e.state.p.set("atmoHaze", 0.0);
    e.render(1);
    assert!(!e.clouds.is_active());
    assert_eq!(e.clouds.bake_count(), 3);
    assert!(!e.renderer.variant().stage.sky_space && !e.renderer.variant().stage.atmo);
}

#[test]
fn ground_cloud_shadows_and_volumetric_atmosphere_render_cleanly() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let (w, h) = (192u32, 128u32);
    let args = [
        ("scene", "knots"),
        ("preset", "trefoil"),
        ("sky", "day"),
        ("sunPitch", "44"),
        ("sunYaw", "25"),
        ("floor", "1"),
        ("floorGloss", "0"),
        ("floorFade", "28"),
        ("camElev", "18"),
        ("dist", "6.2"),
        ("cloudCover", "0.56"),
        ("cloudDensity", "1.8"),
        ("cloudScale", "0.85"),
        ("cloudShadow", "0.0"),
        ("res", "64"),
    ]
    .map(|(k, v)| (k.to_string(), v.to_string()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.renderer.resize(w, h, 1.0);
    e.bake(Detail::Full).unwrap();

    // Render without ground cloud shadows first.
    e.render(4);
    let hdr_no_shadow = e.renderer.read_hdr(0, 0, w, h).unwrap();

    // Enable ground cloud shadows (note: does not even require a dome re-bake!).
    let bakes_before = e.clouds.bake_count();
    e.state.p.set("cloudShadow", 1.0);
    e.render(4);
    assert_eq!(e.clouds.bake_count(), bakes_before);
    let hdr_shadow = e.renderer.read_hdr(0, 0, w, h).unwrap();

    // Measure floor luminance modulation across lower-screen floor scanlines (avoiding center object).
    let lum = |px: &[f32], idx: usize| -> f32 {
        0.2126 * px[idx * 4] + 0.7152 * px[idx * 4 + 1] + 0.0722 * px[idx * 4 + 2]
    };
    let mut max_rel_diff = 0.0f32;
    let mut mean_abs_diff = 0.0f64;
    let mut samples = 0usize;
    for y in (h * 72 / 100)..(h * 94 / 100) {
        for x in 8..(w - 8) {
            if x > w * 35 / 100 && x < w * 65 / 100 {
                continue;
            }
            let idx = (y * w + x) as usize;
            let l0 = lum(&hdr_no_shadow, idx);
            let l1 = lum(&hdr_shadow, idx);
            if l0 > 0.02 {
                let rel = (l1 - l0).abs() / l0;
                max_rel_diff = max_rel_diff.max(rel);
                mean_abs_diff += (l1 - l0).abs() as f64;
                samples += 1;
            }
        }
    }
    assert!(samples > 200);
    assert!(
        max_rel_diff > 0.12 && (mean_abs_diff / samples as f64) > 0.005,
        "expected dappled ground cloud shadows to modulate floor luminance (max_rel_diff={max_rel_diff:.4})"
    );

    // Now test Golden Hour backlit clouds + volumetric sunrays + atmospheric haze for 100% finite output.
    e.state.sky_arg("golden");
    e.state.p.set("sunYaw", 8.0);
    e.state.p.set("sunPitch", 9.0);
    e.state.p.set("cloudCover", 0.52);
    e.state.p.set("cloudSilver", 1.65);
    e.state.p.set("atmoHaze", 0.0);
    e.state.p.set("atmoAerial", 0.0);
    e.state.p.set("sunRays", 0.0);
    e.render(4);
    let hdr_clear_air = e.renderer.read_hdr(0, 0, w, h).unwrap();

    e.state.p.set("atmoHaze", 0.48);
    e.state.p.set("atmoFogH", 1.4);
    e.state.p.set("atmoAerial", 0.42);
    e.state.p.set("sunRays", 1.35);
    e.state.p.set("sunRayG", 0.80);
    e.render(6);
    let hdr_atmo = e.renderer.read_hdr(0, 0, w, h).unwrap();

    for (i, chunk) in hdr_atmo.chunks_exact(4).enumerate() {
        let y = 0.2126 * chunk[0] + 0.7152 * chunk[1] + 0.0722 * chunk[2];
        assert!(
            chunk.iter().all(|v| v.is_finite() && v.abs() < 1e5) && y >= -1e-3,
            "non-finite or invalid HDR value at pixel {i}: {chunk:?}"
        );
        // Outside dispersive glass (`heroSettle` keeps single-wavelength sRGB gamut lobes signed in accum),
        // sky and floor radiance must be strictly non-negative in every channel.
        if chunk[3] <= 0.0 || chunk[3] > 8.0 {
            assert!(
                chunk[0] >= -1e-5 && chunk[1] >= -1e-5 && chunk[2] >= -1e-5,
                "negative sky/floor HDR channel at pixel {i}: {chunk:?}"
            );
        }
    }
    let mean_clear: f64 = (0..(w * h) as usize).map(|i| lum(&hdr_clear_air, i) as f64).sum::<f64>() / (w * h) as f64;
    let mean_atmo: f64 = (0..(w * h) as usize).map(|i| lum(&hdr_atmo, i) as f64).sum::<f64>() / (w * h) as f64;
    assert!(
        (mean_atmo - mean_clear).abs() > 0.01,
        "expected sunrays + haze to measurably alter backlit golden-hour radiance ({mean_clear:.4} vs {mean_atmo:.4})"
    );
}
