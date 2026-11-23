use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn a_thin_scattering_slab_transmits_the_uncollided_image_of_a_light() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "solids"),
        ("preset", "superellipsoid"),
        ("p.p", "8"),
        ("p.q", "8"),
        ("p.deep", ".3"),
        ("L1.size", ".36"),
        ("rx", "0"),
        ("ry", "0"),
        ("rz", "0"),
        ("material", "wax"),
        ("ior", "1"),
        ("fit", "0"),
        ("rough", "0"),
        ("coat", "0"),
        ("age", "0"),
        ("smudges", "0"),
        ("mottle", "0"),
        ("bodyR", "1"),
        ("bodyG", "1"),
        ("bodyB", "1"),
        ("absorb", "0,0,0"),
        ("sssDepth", ".6"),
        ("sssForward", "0"),
        ("sssHue", "0"),
        ("sssThrough", "0"),
        ("sssSpread", "0"),
        ("sssOpaque", "0"),
        ("sssDetail", "64"),
        ("dist", "3.5"),
        ("fov", "24"),
        ("rig", "0,0,6,6,10,10,10,0.15:180,0,6,6,10,10,10,0.15"),
        ("ambient", "0,0,0"),
        ("floor", "0"),
        ("backdrop", "0"),
        ("res", "128"),
        ("spin", "0"),
        ("meter", "0"),
        ("exposure", "1"),
        ("grain", "0"),
        ("bloom", "0"),
    ]
    .map(|(key, value)| (key.into(), value.into()));
    let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
    engine.resize(160, 120);
    let mut light_through = |amount: &str| {
        assert!(engine.state.apply_arg("sss", amount));
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(24);
        let values = engine.renderer.read_hdr(77, 57, 6, 6).unwrap();
        let pixels = values.chunks_exact(4);
        assert!(pixels.clone().all(|p| p[3] > 0.0 && p[1].is_finite()));
        pixels.map(|p| p[1]).sum::<f32>() / 36.0
    };
    let clear = light_through("0");
    let wax = light_through("1");
    // This normal ray spans less than half a mean free path. Even before the
    // scattered light is added, most rays have retained the bright panel.
    assert!(
        clear > 1.0 && wax > 0.5 * clear,
        "clear {clear}, scattering {wax}"
    );
}

#[test]
fn opaque_candle_wax_has_sculptural_contrast_and_warm_glow() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "solids"),
        ("preset", "staricosahedron"),
        ("material", "candle"),
        ("sky", "1"),
        ("skyPreset", "golden"),
        ("floor", "1"),
        ("res", "192"),
        ("spin", "0"),
        ("grain", "0"),
        ("bloom", "0"),
    ]
    .map(|(key, value)| (key.into(), value.into()));
    let (w, h) = (256usize, 256usize);
    let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
    engine.resize(w as u32, h as u32);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    engine.render(16);
    let hdr = engine.renderer.read_hdr(0, 0, w as u32, h as u32).unwrap();
    assert!(hdr.chunks_exact(4).all(|p| p[..3].iter().all(|v| v.is_finite() && *v >= 0.0)));

    let hit = |x: usize, y: usize| hdr[4 * (y * w + x) + 3] > 0.0;
    let mut lums = Vec::new();
    let mut r_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    for y in 4..(h - 4) {
        for x in 4..(w - 4) {
            if hit(x, y) && hit(x - 3, y) && hit(x + 3, y) && hit(x, y - 3) && hit(x, y + 3) {
                let p = &hdr[4 * (y * w + x)..];
                let lum = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                lums.push(lum);
                r_sum += p[0] as f64;
                b_sum += p[2] as f64;
            }
        }
    }
    assert!(lums.len() > 2000, "expected interior object pixels, got {}", lums.len());
    lums.sort_by(|a, b| a.total_cmp(b));
    let p10 = lums[lums.len() / 10];
    let p90 = lums[(lums.len() * 9) / 10];
    assert!(
        p90 > 1.75 * p10.max(1e-4),
        "candle wax washed out flat: p10={p10}, p90={p90}"
    );
    assert!(
        r_sum > 1.25 * b_sum.max(1e-6),
        "candle wax lost warm amber/ivory balance: r_sum={r_sum}, b_sum={b_sum}"
    );
}
