use gf_core::{
    material::{spec_w, METALS},
    state::{Detail, State},
};
use glassforms::{engine::Engine, gpu::Gpu};

const SIZE: u32 = 171;

fn interface(a: f64, real: f64, imaginary: f64) -> f64 {
    ((a - real).powi(2) + imaginary.powi(2)) / ((a + real).powi(2) + imaginary.powi(2))
}

// Incoherent power reflection from a parallel nonabsorbing coat. Follow the
// successive passes separately for s and p, then integrate the measured spectrum.
fn coated_copper(theta: f64) -> [f64; 3] {
    let copper = METALS.iter().find(|m| m.id == "copper").unwrap();
    let index = 1.5;
    let (sin, cos) = theta.sin_cos();
    let cc = (1.0 - (sin / index).powi(2)).sqrt();
    let mut sum = [0.0; 3];
    let mut sensor = [0.0; 3];
    for j in 0..16 {
        let (n, k) = (f64::from(copper.n[2 * j]), f64::from(copper.k[2 * j]));
        // Complex admittances of the absorbing substrate at the conserved
        // tangential wave vector sin(theta): s uses q, p uses (n+ik)^2/q.
        let (a, b) = (n * n - k * k - sin * sin, 2.0 * n * k);
        let norm = a.hypot(b);
        let qr = (0.5 * (norm + a)).sqrt();
        let qi = (0.5 * (norm - a)).sqrt();
        let pr = ((n * n - k * k) * qr + b * qi) / norm;
        let pi = (b * qr - (n * n - k * k) * qi) / norm;
        let mut power = 0.0;
        for (top, metal) in [
            (
                interface(cos, index * cc, 0.0),
                interface(index * cc, qr, qi),
            ),
            (
                interface(1.0 / cos, index / cc, 0.0),
                interface(index / cc, pr, pi),
            ),
        ] {
            let mut path = (1.0 - top).powi(2) * metal;
            let mut reflected = top;
            for _ in 0..256 {
                reflected += path;
                path *= top * metal;
                if path < 1e-14 {
                    break;
                }
            }
            power += 0.5 * reflected;
        }
        let w = spec_w(400.0 + 20.0 * j as f64);
        for c in 0..3 {
            sum[c] += w[c] * power;
            sensor[c] += w[c];
        }
    }
    std::array::from_fn(|c| (sum[c] / sensor[c]).clamp(0.0, 1.0))
}

fn render(gpu: Gpu, options: &[(&str, &str)], frames: u32) -> Vec<f32> {
    let args = options
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect::<Vec<_>>();
    let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
    engine.resize(SIZE, SIZE);
    engine.bake(Detail::Full).unwrap();
    engine.settle_lighting();
    engine.render(frames);
    engine.renderer.read_hdr(0, 0, SIZE, SIZE).unwrap()
}

fn plate(gpu: Gpu, angle: &str, alongside: bool, material: &str) -> Vec<f32> {
    let mut options = vec![
        ("scene", "studio"),
        ("preset", "block"),
        ("studio", "none"),
        ("rig", ""),
        ("ambient", ".25,.25,.25"),
        ("envRoom", "0"),
        ("floor", "0"),
        ("backdrop", "0"),
        ("meter", "0"),
        ("lightGain", "1"),
        ("exposure", "1"),
        ("grain", "0"),
        ("bloom", "0"),
        ("res", "192"),
        ("spin", "0"),
        ("rx", "0"),
        ("ry", angle),
        ("rz", "0"),
        ("dist", "5"),
        ("fov", "23"),
        ("width", "1.6"),
        ("height", "1.4"),
        ("depth", ".22"),
        ("bevel", ".04"),
        ("round", "0"),
        ("material", material),
        ("rough", "0"),
        ("age", "0"),
        ("mottle", "0"),
        ("brush", "0"),
        ("coat", "1"),
        ("coatIor", "1.5"),
    ];
    if alongside {
        options.extend([
            ("layer", "studio:sphere"),
            ("L2.x", "1.15"),
            ("L2.y", ".40"),
            ("L2.z", "-.35"),
            ("L2.size", ".14"),
            ("J12.op", "apart"),
            ("B2.material", "crown"),
            ("B2.density", "0"),
            ("B2.disp", "0"),
        ]);
    }
    render(gpu, &options, 64)
}

#[test]
fn a_clear_conductor_coat_returns_internal_reflections_at_normal_and_grazing_incidence() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    // At 70 degrees the image centre sees the adjoining broad face of the block.
    // Pixel 124 is inside the intended grazing face in both scene arrangements.
    for (angle, pixel) in [(0_u32, 85_u32), (70, 124)] {
        let offset =
            (2.0 * (f64::from(pixel) + 0.5) / f64::from(SIZE) - 1.0) * 11.5_f64.to_radians().tan();
        let expected = coated_copper(f64::from(angle).to_radians() + offset.atan());
        for alongside in [false, true] {
            for (material, target) in [("copper", expected), ("mirror", [1.0; 3])] {
                let hdr = plate(gpu.clone(), &angle.to_string(), alongside, material);
                let i = (4 * (85 * SIZE + pixel)) as usize;
                assert!(hdr[i + 3] > 0.0);
                for c in 0..3 {
                    let actual = 4.0 * f64::from(hdr[i + c]);
                    assert!(
                        (actual - target[c]).abs() < 0.004,
                        "{material}, {angle} degrees, alongside {alongside}, channel {c}: {actual} != {}",
                        target[c]
                    );
                }
            }
        }
    }
}

#[test]
fn a_pristine_coat_on_a_perfect_mirror_follows_the_same_recessed_reflection() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let ring = |coat, bounces| {
        let options = [
            ("scene", "solids"),
            ("preset", "ring"),
            ("studio", "none"),
            ("rx", "58"),
            ("ry", "14"),
            ("rz", "0"),
            ("dist", "5.7"),
            ("fov", "31"),
            ("material", "mirror"),
            ("metal", "1"),
            ("rough", "0"),
            ("age", "0"),
            ("brush", "0"),
            ("mottle", "0"),
            ("coat", coat),
            ("coatIor", "1.5"),
            ("nb", bounces),
            ("floor", "0"),
            ("grain", "0"),
            ("meter", "0"),
            ("exposure", "1"),
            ("res", "160"),
        ];
        render(gpu.clone(), &options, 32)
    };
    let once = ring("0", "1");
    let bare = ring("0", "6");
    let coated = ring("1", "6");
    let mut witnesses = 0;
    let mut square = 0.0;
    for ((short, metal), coat) in once
        .chunks_exact(4)
        .zip(bare.chunks_exact(4))
        .zip(coated.chunks_exact(4))
    {
        let difference = (0..3).map(|c| (short[c] - metal[c]).powi(2)).sum::<f32>();
        if metal[3] > 0.0 && difference > 0.08_f32.powi(2) {
            witnesses += 1;
            square += (0..3).map(|c| (metal[c] - coat[c]).powi(2)).sum::<f32>();
        }
    }
    assert!(
        witnesses > 75,
        "only {witnesses} secondary reflection pixels"
    );
    let rms = (square / (3 * witnesses) as f32).sqrt();
    assert!(
        rms < 0.0005,
        "adding the ideal coat changed a secondary reflection by {rms}"
    );
}
