use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn neighbouring_clay_transfers_colour_only_to_the_facing_side() {
    let capture = |red: &str, blue: &str| {
        let gpu = Gpu::headless("vulkan").ok()?;
        let args = [
            ("scene", "studio"),
            ("preset", "sphere"),
            ("material", "modelclay"),
            ("layer", "studio:sphere"),
            ("B2.material", "modelclay"),
            ("J12.op", "apart"),
            ("L1.x", "-.47"),
            ("L1.size", ".44"),
            ("L2.x", ".43"),
            ("L2.z", ".17"),
            ("L2.size", ".44"),
            ("bodyR", ".5"),
            ("bodyG", ".5"),
            ("bodyB", ".5"),
            ("B2.bodyR", red),
            ("B2.bodyG", ".025"),
            ("B2.bodyB", blue),
            ("ior", "1.5"),
            ("B2.ior", "1.5"),
            ("age", "0"),
            ("mottle", "0"),
            ("B2.age", "0"),
            ("B2.mottle", "0"),
            ("rig", ""),
            ("ambient", ".7,.7,.7"),
            ("envRoom", "0"),
            ("floor", "0"),
            ("backdrop", "0"),
            ("cast", "0"),
            ("rx", "0"),
            ("ry", "0"),
            ("rz", "0"),
            ("res", "192"),
            ("dist", "4.4"),
            ("fov", "36"),
            ("spin", "0"),
            ("grain", "0"),
            ("bloom", "0"),
            ("meter", "0"),
            ("nb", "2"),
        ]
        .map(|(key, value)| (key.to_owned(), value.to_owned()));
        let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
        engine.resize(208, 140);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(128);
        Some(engine.renderer.read_hdr(0, 0, 208, 140).unwrap())
    };
    let Some(red) = capture(".85", ".025") else {
        return;
    };
    let blue = capture(".025", ".85").unwrap();

    // Both patches are on the neutral sphere. Only its right side sees the donor.
    for (columns, expected_gain) in [(69..74, 0.0..=0.0001), (93..98, 0.025..=0.075)] {
        let mut gain = 0.0;
        let mut count = 0;
        for y in 67..73 {
            for x in columns.clone() {
                let i = 4 * (y * 208 + x);
                assert!(red[i + 3] > 0.0 && red[i + 3] == blue[i + 3]);
                let increase = red[i] - blue[i];
                let decrease = blue[i + 2] - red[i + 2];
                assert!((increase - decrease).abs() < 0.001);
                assert!((red[i + 1] - blue[i + 1]).abs() < 0.0001);
                gain += increase;
                count += 1;
            }
        }
        let mean = gain / count as f32;
        assert!(
            expected_gain.contains(&mean),
            "receiver patch {columns:?} gained {mean}"
        );
    }
}
