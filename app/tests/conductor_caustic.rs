use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn reflected_floor_light_carries_the_conductors_spectrum() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "gems"), ("preset", "radiant"), ("material", "mirror"),
        ("rx", "68"), ("ry", "12"), ("rz", "21"), ("elev", "13"),
        ("dist", "5"), ("fov", "32"), ("res", "144"), ("spin", "0"),
        ("rig", "0,78,62,40,1.55,1.55,1.62,0.7:-96,14,12,52,1.25,1.2,1.4,0.6:102,20,14,46,1.05,1.15,1.35,0.6:180,30,40,26,0.42,0.42,0.46,0.8"),
        ("ambient", ".05,.05,.06"), ("exposure", "1"), ("floor", "1"), ("floorFade", "16"),
        ("floorGloss", "0"), ("floorMirror", "0"), ("floorMatch", "0"),
        ("cast", "2"), ("castShadows", "0"), ("grain", "0"), ("meter", "0"),
    ].map(|(k,v)| (k.to_owned(),v.to_owned()));
    let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
    let (width, height) = (210, 232);
    engine.resize(width, height);
    engine.bake(Detail::Full).unwrap();
    let mut capture = |metal: &str, caustics: &str| {
        for (key, value) in [
            ("material", metal),
            ("age", "0"),
            ("rough", "0"),
            ("bodyR", "1"),
            ("bodyG", "1"),
            ("bodyB", "1"),
            ("castCaustics", caustics),
        ] {
            assert!(engine.state.apply_arg(key, value));
        }
        engine.settle_lighting();
        engine.render(32);
        engine.renderer.read_hdr(0, 0, width, height).unwrap()
    };
    let unlit = capture("mirror", "0");
    let mirror = capture("mirror", "1");
    let chrome_unlit = capture("chrome", "0");
    let chrome = capture("chrome", "1");
    let copper_unlit = capture("copper", "0");
    let copper = capture("copper", "1");
    // The bottom fifth of the frame is entirely matte floor.
    // Keep the pixels to which an ideal mirror sends measurable additional light.
    let mut sums = [[0.0_f64; 3]; 3];
    let mut pixels = 0;
    for y in height * 4 / 5..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            if (0..3).any(|c| mirror[i + c] - unlit[i + c] <= 0.025) {
                continue;
            }
            pixels += 1;
            for (sum, (lit, dark)) in sums.iter_mut().zip([(&mirror, &unlit), (&chrome, &chrome_unlit), (&copper, &copper_unlit)]) {
                for c in 0..3 {
                    sum[c] += f64::from(lit[i + c] - dark[i + c]);
                }
            }
        }
    }
    assert!(pixels > 800, "only {pixels} illuminated floor pixels");
    let chrome = [0, 1, 2].map(|c| sums[1][c] / sums[0][c]);
    let copper = [0, 1, 2].map(|c| sums[2][c] / sums[0][c]);
    eprintln!("{pixels} floor pixels, chrome {chrome:?}, copper {copper:?}");
    assert!(
        chrome.iter().all(|c| (0.42..0.72).contains(c)),
        "chrome {chrome:?}"
    );
    assert!(
        copper[0] > 0.80
            && copper[0] < 1.02
            && copper[0] - copper[1] > 0.12
            && copper[1] - copper[2] > 0.035,
        "copper {copper:?}"
    );
}
