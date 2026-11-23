use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn a_rough_ideal_mirror_returns_uniform_light_inside_the_knot() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let settings = [
        ("scene", "knots"),
        ("preset", "trefoil"),
        ("material", "mirror"),
        ("rough", "0.95"),
        ("roughRef", "0"),
        ("age", "0"),
        ("rig", ""),
        ("ambient", "0.25,0.25,0.25"),
        ("lightGain", "1"),
        ("envRoom", "0"),
        ("floor", "0"),
        ("backdrop", "0"),
        ("spin", "0"),
        ("grain", "0"),
        ("res", "128"),
        ("dist", "4.8"),
        ("rx", "14"),
        ("ry", "25"),
    ]
    .map(|(key, value)| (key.to_string(), value.to_string()));
    let (width, height) = (144, 120);
    let mut engine = Engine::new(gpu, State::with_args(&settings, true), true);
    engine.resize(width, height);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    engine.render(32);
    let image = engine.renderer.read_hdr(0, 0, width, height).unwrap();
    let pixel = |x: usize, y: usize| 4 * (y * width as usize + x);
    let mut count = 0;
    let mut error = 0.0;
    for y in 1..height as usize - 1 {
        for x in 1..width as usize - 1 {
            if [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .iter()
                .all(|&(i, j)| image[pixel(i, j) + 3] > 0.0)
            {
                let red = image[pixel(x, y)];
                error += (red - 0.25).abs() as f64;
                count += 1;
            }
        }
    }
    assert!(count > 500, "only {count} metal pixels");
    assert!(
        error / f64::from(count) < 0.004,
        "mean error {} across {count} metal pixels",
        error / f64::from(count)
    );
}

#[test]
fn sampled_copper_and_its_live_lobe_see_the_same_studio() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let render = |reference: &str| {
        let settings = [
            ("scene", "solids"), ("preset", "ring"), ("studio", "none"),
            ("material", "copper"), ("rough", ".65"), ("roughRef", reference),
            ("rx", "58"), ("ry", "14"), ("rz", "0"), ("fov", "31"), ("dist", "5.7"),
            ("age", "0"), ("brush", "0"), ("mottle", "0"), ("meter", "0"),
            ("exposure", "1"), ("grain", "0"), ("res", "144"), ("spin", "0"),
        ].map(|(key, value)| (key.to_string(), value.to_string()));
        let mut engine = Engine::new(gpu.clone(), State::with_args(&settings, true), true);
        engine.resize(192, 144);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(256);
        engine.renderer.read_hdr(0, 0, 192, 144).unwrap()
    };
    let live = render("0");
    let sampled = render("1");
    let mut bias = [0.0f64; 3];
    let mut pixels = 0;
    for (a, b) in live.chunks_exact(4).zip(sampled.chunks_exact(4)) {
        if a[3] > 0.0 && b[3] > 0.0 {
            for c in 0..3 {
                bias[c] += f64::from(a[c] - b[c]);
            }
            pixels += 1;
        }
    }
    assert!(pixels > 3000);
    let bias = bias.map(|d| d / f64::from(pixels));
    assert!(bias.iter().all(|d| d.abs() < 0.075), "studio copper RGB bias {bias:?}");
}
