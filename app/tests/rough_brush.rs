use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

fn fixture(gpu: Gpu, rough: &str, angle: &str) -> Engine {
    let settings = [
        ("scene", "studio"),
        ("preset", "sphere"),
        ("material", "iron"),
        ("rough", rough),
        ("roughRef", "1"),
        ("age", "1"),
        ("brush", "1"),
        ("brushAngle", angle),
        ("edgeWear", "0"),
        ("cavity", "0"),
        ("scratches", "0"),
        ("swirls", "0"),
        ("dust", "0"),
        ("smudges", "0"),
        ("peel", "0"),
        ("mottle", "0"),
        ("lightGain", "1"),
        ("backdrop", "0"),
        ("rig", ""),
        ("coat", "0"),
        ("coatIor", "1.38"),
        ("ambient", "0,0,0"),
        ("lamp1", "ball,1.4,.8,2.1,.2,0,1,1,1,35"),
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

// Measure the footprint of a compact lamp on the central face of the sphere. A weak
// threshold excludes silhouette antialiasing and broad multiple-scattering fill.
fn widths(pixels: &[f32]) -> [f64; 2] {
    let mut values = Vec::new();
    let mut peak = 0.0_f64;
    for (index, rgba) in pixels.chunks_exact(4).enumerate() {
        let x = (index % SIZE as usize) as f64;
        let y = (index / SIZE as usize) as f64;
        if rgba[3] <= 0.0 {
            continue;
        }
        let l = f64::from(rgba[0] * 0.2126 + rgba[1] * 0.7152 + rgba[2] * 0.0722);
        peak = peak.max(l);
        values.push((x, y, l));
    }
    let threshold = 0.035 * peak;
    let mut moments = [0.0; 5];
    for (x, y, light) in values {
        let weight = (light - threshold).max(0.0);
        for (sum, moment) in moments.iter_mut().zip([1.0, x, y, x * x, y * y]) {
            *sum += weight * moment;
        }
    }
    let [mass, x, y, xx, yy] = moments;
    assert!(mass > 1.0 && peak > 0.05);
    [
        (xx / mass - (x / mass).powi(2)).sqrt(),
        (yy / mass - (y / mass).powi(2)).sqrt(),
    ]
}

#[test]
fn sampled_brushed_iron_respects_roughness_and_polish_direction() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let render = |rough, angle| {
        let mut engine = fixture(gpu.clone(), rough, angle);
        widths(&image(&mut engine, 384))
    };
    let smooth = render(".22", "0");
    let broad = render(".60", "0");
    let turned = render(".60", "90");
    assert!(
        broad[0] > smooth[0] * 1.5 && broad[1] > smooth[1] * 1.5,
        "roughness {smooth:?} to {broad:?}"
    );
    assert!(
        broad[1] > broad[0] * 1.3 && turned[0] > turned[1] * 1.3,
        "polish {broad:?} to {turned:?}"
    );
}
