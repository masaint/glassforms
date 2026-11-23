use gf_core::{
    material::fresnel,
    state::{Detail, State},
};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn bare_paint_in_an_assembly_reflects_at_its_assigned_index() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "studio"),
        ("preset", "sphere"),
        ("L1.size", ".68"),
        ("rx", "0"),
        ("ry", "0"),
        ("rz", "0"),
        ("res", "128"),
        ("dist", "4"),
        ("fov", "28"),
        ("spin", "0"),
        ("material", "moulded"),
        ("bodyR", "0"),
        ("bodyG", "0"),
        ("bodyB", "0"),
        ("coat", "0"),
        ("age", "0"),
        ("mottle", "0"),
        ("brush", "0"),
        ("glazeHaze", "0"),
        ("layer", "studio:sphere"),
        ("L2.x", "1.1"),
        ("L2.z", "-.5"),
        ("L2.size", ".2"),
        ("J12.op", "apart"),
        ("B2.material", "crown"),
        ("rig", ""),
        ("ambient", ".5,.5,.5"),
        ("floor", "0"),
        ("backdrop", "0"),
        ("meter", "0"),
        ("exposure", "1"),
        ("bloom", "0"),
        ("grain", "0"),
    ]
    .map(|(key, value)| (key.into(), value.into()));
    let mut engine = Engine::new(gpu, State::with_args(&args, true), true);
    engine.resize(160, 120);
    let mut reflected = |ior: &str| {
        assert!(engine.state.apply_arg("ior", ior));
        engine.bake(Detail::Full).unwrap();
        engine.render(8);
        let center = engine.renderer.read_hdr(78, 58, 4, 4).unwrap();
        let pixels = center.chunks_exact(4);
        assert!(pixels.clone().all(|p| p[3] > 0.0 && p[0].is_finite()));
        pixels.map(|p| p[0]).sum::<f32>() / 16.0
    };
    let low = reflected("1.25");
    let high = reflected("1.8");
    let expected = (fresnel(1.0, 1.0, 1.8) / fresnel(1.0, 1.0, 1.25)) as f32;
    assert!(
        low > 0.0 && (high / low - expected).abs() < 0.04,
        "near-normal reflection ratio {} against Fresnel {expected}",
        high / low
    );
}

#[test]
fn pigment_on_a_subtractive_face_receives_the_scene_lighting() {
    use gf_core::{
        construction::CutMaterial,
        csg::{Op, Part},
        layers::Placement,
    };

    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "studio"),
        ("preset", "block"),
        ("trace", "csg"),
        ("res", "128"),
        ("material", "crown"),
        ("rx", "0"),
        ("ry", "0"),
        ("rz", "0"),
        ("rig", ""),
        ("ambient", ".5,.5,.5"),
        ("grain", "0"),
        ("bloom", "0"),
        ("spin", "0"),
    ]
    .map(|(key, value)| (key.into(), value.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.auto = false;
    e.resize(160, 120);
    let cutter = Part {
        place: Placement {
            pos: [0., 0., 0.71],
            size: 0.58,
            ..Default::default()
        },
        ..Default::default()
    };
    let path = e.state.csg.add_construction_part(&[], cutter).unwrap();
    let mut paint = e.state.p.clone();
    gf_core::material::NAMED
        .iter()
        .find(|preset| preset.id == "moulded")
        .unwrap()
        .apply(&mut paint);
    let mut render = |rgb: [f64; 3], owner| {
        [paint.body_r, paint.body_g, paint.body_b] = rgb;
        let face = e.state.csg.edit_construction().node_mut(&path).unwrap();
        face.op = Op::Subtract;
        face.cut_material = owner;
        face.material = Some(
            gf_core::material::body_names()
                .map(|key| (key.to_owned(), paint.get(key).unwrap()))
                .collect(),
        );
        e.bake(Detail::Full).unwrap();
        e.renderer.reseed(0);
        e.render(8);
        e.renderer.read_hdr(76, 56, 8, 8).unwrap()
    };
    let red = [0.72, 0.025, 0.025];
    let green = [0.025, 0.72, 0.025];
    let first = render(red, CutMaterial::Subtractor);
    let next = render(green, CutMaterial::Subtractor);
    for (a, b) in first.chunks_exact(4).zip(next.chunks_exact(4)) {
        assert_eq!(a[3], b[3]);
        assert!(
            a[3] > 0.0 && a[0] > b[0] + 0.015 && b[1] > a[1] + 0.015,
            "red and green cut at the same depth: {a:?}, {b:?}"
        );
    }
    let host_red = render(red, CutMaterial::Recipient);
    assert_eq!(host_red, render(green, CutMaterial::Recipient));
}

#[test]
fn painted_substrate_loses_the_light_its_interface_reflects() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "studio"),
        ("preset", "sphere"),
        ("material", "moulded"),
        ("coat", "0"),
        ("rig", ""),
        ("ambient", ".5,.5,.5"),
        ("res", "128"),
        ("age", "0"),
        ("mottle", "0"),
        ("glazeHaze", "0"),
        ("spin", "0"),
        ("exposure", "1"),
        ("bloom", "0"),
        ("grain", "0"),
    ]
    .map(|(key, value)| (key.into(), value.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(160, 120);
    let mut diffuse = |ior: &str, coat: &str| {
        for (key, value) in [("ior", ior), ("coat", coat), ("coatIor", "1.42")] {
            assert!(e.state.apply_arg(key, value));
        }
        let mut radiance = [0.0; 2];
        for (sample, grey) in radiance.iter_mut().zip(["0", ".5"]) {
            for channel in ["bodyR", "bodyG", "bodyB"] {
                assert!(e.state.apply_arg(channel, grey));
            }
            e.bake(Detail::Full).unwrap();
            e.render(8);
            let centre = e.renderer.read_hdr(77, 57, 6, 6).unwrap();
            assert!(centre
                .chunks_exact(4)
                .all(|p| p[0].is_finite() && p[3] > 0.0));
            *sample = centre.chunks_exact(4).map(|p| p[0]).sum::<f32>() / 36.0;
        }
        radiance[1] - radiance[0]
    };
    // Removing the black substrate cancels the specular return at each index.
    let uncoated = diffuse("1.8", "0") / diffuse("1.25", "0");
    let transmission = |n: f32| 1.0 - ((n - 1.0) / (n + 1.0)).powi(2);
    assert!(
        (uncoated - transmission(1.8) / transmission(1.25)).abs() < 0.004,
        "{uncoated}"
    );
    let coated = diffuse("1.8", "1") / diffuse("1.25", "1");
    let through_coat = |body: f32| {
        let top = 1.0 - transmission(1.42);
        let substrate = ((1.42 - body) / (1.42 + body)).powi(2);
        (1.0 - top) * (1.0 - substrate) / (1.0 - top * substrate)
    };
    assert!(
        (coated - through_coat(1.8) / through_coat(1.25)).abs() < 0.004,
        "{coated}"
    );
}
