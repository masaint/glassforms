use gf_core::{material, settings::Settings};
use glassforms::render::{tracer_source, March, Variant};

#[test]
fn only_compatible_selected_bodies_sample_glass_boundaries() {
    let base = Settings::default();
    let rough = Settings {
        rough: 0.28,
        ..base.clone()
    };
    let sampled = Settings {
        rough_ref: 1.,
        ..rough.clone()
    };
    let coat = Settings {
        coat: 1.,
        grating: 0.4,
        ..sampled.clone()
    };
    let metal = Settings {
        metal: 2.,
        ..sampled.clone()
    };
    let milk = Settings {
        sss: 0.6,
        sss_walk: 1.,
        ..sampled.clone()
    };
    let crystal = Settings {
        biref: 0.08,
        spectrum: 1.,
        cracks: 0.1,
        ..base.clone()
    };
    let gradient = Settings {
        grin: 0.05,
        ..base.clone()
    };
    for (mats, enabled) in [
        (vec![&rough, &metal], false),
        (vec![&sampled, &base], true),
        (vec![&base, &coat], true),
        (vec![&milk, &metal], false),
        (vec![&sampled, &milk, &metal], true),
        (vec![&sampled, &crystal, &gradient], true),
    ] {
        let flags = material::Variant::of_bodies(&mats);
        assert_eq!(
            flags.rough_ref, enabled,
            "materials {mats:?} flags {flags:?}"
        );
        assert_eq!(
            mats.iter().any(|p| material::body_bits(p) & 64 != 0),
            enabled
        );
        for world in [false, true] {
            let src = tracer_source(
                &March {
                    step_scale: 0.9,
                    steps_out: 180,
                    steps_in: 64,
                    layers: 2,
                },
                Variant {
                    mat: flags,
                    world_faces: world,
                    textured: world,
                    ..Default::default()
                },
                false,
            );
            assert_eq!(src.contains("fn sampledRadianceB("), enabled);
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        }
    }
}

#[test]
fn sampled_boundary_uses_the_adjacent_index_for_refraction_and_total_internal_reflection() {
    use glassforms::gpu::Gpu;
    let gpu = Gpu::headless("vulkan").unwrap();
    let shared = material::preprocess(
        include_str!("../src/shaders/frost_reference.wgsl"),
        &[
            ("FROST_BOUNDARY", true),
            ("FROST_COAT", true),
            ("BODIES", true),
        ],
    );
    let mut source = include_str!("../src/shaders/pbr/types.wgsl").to_owned();
    source += include_str!("../src/shaders/pbr/dielectric.wgsl");
    source += include_str!("../src/shaders/dielectric_fresnel.wgsl");
    source += include_str!("../src/shaders/pbr/hash.wgsl");
    source += include_str!("../src/shaders/pcg.wgsl");
    source += include_str!("../src/shaders/ggx_microsurface.wgsl");
    source += "struct Mat {surf:vec4f} var<private> mu:Mat;\n";
    source += &shared;
    source += r#"
@group(0) @binding(0) var output: texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn check(@builtin(global_invocation_id) id:vec3u) {
  let pair = id.z/2u;
  let coat = select(1.,1.42,(id.z&1u)!=0u);
  let outside=select(select(1.,1.333,pair==1u),1.5,pair==3u);
  let rough=select(0.0001,0.24,pair>=2u);
  mu.surf=vec4f(rough,0.,coat,0.);
  let c=0.65;
  let d=vec3f(sqrt(1.-c*c),0.,c);
  var rng=pcg(1u+id.x+128u*id.y);
  let v=frostBoundary(d,vec3f(0,0,1),true,outside,1.5,vec3f(0),550.,&rng);
  textureStore(output,id,v);
}"#;
    let tex = sample_cases(
        &gpu,
        "rough dielectric contacting another medium",
        &source,
        128,
        8,
    );
    for i in 0..8 {
        let bytes = gpu.read(&tex, 0, 0, i, 128, 128, 16).unwrap();
        let paths = bytemuck::cast_slice::<u8, [f32; 4]>(&bytes);
        let transmitted = paths.iter().filter(|p| p[3] > 0.).count() as f64 / paths.len() as f64;
        let lost = paths.iter().filter(|p| p[3] == 0.).count();
        assert!(lost < 20, "{i}: {lost} truncated");
        for ray in paths.iter().filter(|p| p[3] != 0.) {
            let length = ray[..3].iter().map(|x| x * x).sum::<f32>();
            assert!(
                (length - 1.).abs() < 0.003 && ray[2] * ray[3] > 0.,
                "{i} {ray:?}"
            );
        }
        eprintln!("configuration {i}: transmission {transmitted:.5}, truncated {lost}");
        if i < 2 {
            assert!(transmitted < 0.003);
        } else if i < 4 {
            assert!(transmitted > 0.94);
        } else if i == 4 {
            // Rough bare facets open the escape cone at this supercritical angle.
            assert!(transmitted > 0.10 && transmitted < 0.50);
        } else if i == 5 {
            // With a coat the outer smooth 1.42/air surface has its own critical angle.
            // The rare escapes follow a changed direction at the rough substrate.
            assert!(transmitted > 0. && transmitted < 0.02);
        } else if i == 6 {
            assert_eq!(transmitted, 1.);
            for ray in paths {
                assert!((ray[2] - 0.65).abs() < 0.0002);
            }
        } else {
            assert!(transmitted > 0.95);
        }
    }
}

fn sample_cases(
    gpu: &glassforms::gpu::Gpu,
    label: &str,
    source: &str,
    side: u32,
    cases: u32,
) -> wgpu::Texture {
    let module = gpu.shader(label, source, false);
    let pipe = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("check"),
            compilation_options: Default::default(),
            cache: None,
        });
    let tex = gpu.volume(
        "boundary samples",
        side,
        cases,
        wgpu::TextureFormat::Rgba32Float,
        true,
    );
    let view = tex.create_view(&Default::default());
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(side.div_ceil(8), side.div_ceil(8), cases);
    }
    gpu.queue.submit([enc.finish()]);
    tex
}

fn function(source: &str, name: &str) -> String {
    let start = source.find(&format!("fn {name}(")).unwrap();
    source[start..start + source[start..].find("\n}").unwrap() + 2].to_owned()
}

#[test]
fn only_transmitting_materials_require_chromatic_paths() {
    let gpu = glassforms::gpu::Gpu::headless("vulkan").unwrap();
    let frost = include_str!("../src/shaders/frost_bodies.wgsl");
    let mut source = String::from(
        "const COUNT: i32=2; struct MB { kind:vec4f,disp:vec4f } var<private> mb:MB;
",
    );
    source += &function(frost, "dispersiveFrostB");
    source += r#"
@group(0) @binding(0) var output: texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn check(@builtin(global_invocation_id) id:vec3u) {
    // All unused slots contain dispersive glass, as can happen after changing body count.
    mb.kind = vec4f(0.,f32(id.z),0.,0.);
    mb.disp = vec4f(0.,0.08,0.13,0.17);
    textureStore(output,id,vec4f(select(0.,1.,dispersiveFrostB())));
}"#;
    let tex = sample_cases(&gpu, "dispersion in physical bodies", &source, 8, 5);
    // Clear glass, conductor, volume-scattering body, thin wall, opaque paint.
    for (i, expected) in [1., 0., 1., 0., 0.].into_iter().enumerate() {
        let bytes = gpu.read(&tex, 0, 0, i as u32, 1, 1, 16).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&bytes)[0], expected);
    }
}

#[test]
fn a_clear_coat_keeps_both_boundaries_between_air_or_glass_and_paint() {
    let Ok(gpu) = glassforms::gpu::Gpu::headless("vulkan") else {
        return;
    };
    let source = concat!(
        include_str!("../src/shaders/pbr/types.wgsl"),
        include_str!("../src/shaders/pbr/dielectric.wgsl"),
        include_str!("../src/shaders/dielectric_fresnel.wgsl"),
        include_str!("../src/shaders/paint_interface.wgsl"),
        r#"
@group(0) @binding(0) var image: texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn check(@builtin(global_invocation_id) id: vec3u) {
    let c = (f32(id.x)+0.5)/128.0;
    let incoming = select(1.0,1.5,id.z>=3u);
    let index = select(array<f32,3>(1.0,1.42,1.8)[id.z%3u],1.5,id.z==6u);
    let body = select(1.5,1.8,id.z==6u);
    textureStore(image,id,vec4f(paintLayerWeights(c,incoming,body,index,id.y!=0u),1.0));
}
"#
    );
    let tex = sample_cases(&gpu, "painted dielectric top and substrate", source, 128, 7);
    // Follow successive reflected paths explicitly for each polarization. An
    // evanescent wave contributes no flux through that interface.
    fn reflect(c: f64, n1: f64, n2: f64, p: bool) -> (f64, f64) {
        if n1 == n2 {
            return (0.0, c);
        }
        let st = (1.0 - c * c).sqrt() * n1 / n2;
        if st >= 1.0 {
            return (1.0, 0.0);
        }
        let ct = (1.0 - st * st).sqrt();
        let (a, b) = if p {
            (n2 * c, n1 * ct)
        } else {
            (n1 * c, n2 * ct)
        };
        (((a - b) / (a + b)).powi(2), ct)
    }
    for case in 0..7 {
        let bytes = gpu.read(&tex, 0, 0, case, 128, 2, 16).unwrap();
        let data = bytemuck::cast_slice::<u8, [f32; 4]>(&bytes);
        let incoming = if case >= 3 { 1.5 } else { 1.0 };
        let coat = if case == 6 {
            1.5
        } else {
            [1.0, 1.42, 1.8][case as usize % 3]
        };
        let body = if case == 6 { 1.8 } else { 1.5 };
        for (i, sample) in data.iter().enumerate() {
            let c = ((i % 128) as f64 + 0.5) / 128.0;
            let coated = i >= 128;
            let mut expected = [0.0; 3];
            for pol in [false, true] {
                let (r1, cc) = reflect(c, incoming, if coated { coat } else { body }, pol);
                expected[0] += 0.5 * r1;
                if !coated {
                    expected[2] += 0.5 * (1.0 - r1);
                    continue;
                }
                if r1 == 1.0 {
                    continue;
                }
                let (r2, _) = reflect(cc, coat, body, pol);
                let mut down = 1.0 - r1;
                for _ in 0..4000 {
                    expected[2] += 0.5 * down * (1.0 - r2);
                    let up = down * r2;
                    expected[1] += 0.5 * up * (1.0 - r1);
                    down = up * r1;
                    if down < 1e-12 {
                        break;
                    }
                }
            }
            for (actual, reference) in sample[..3].iter().zip(expected) {
                assert!(
                    (f64::from(*actual) - reference).abs() < 2e-4,
                    "case {case}, cos {c}, coat {coated}: {sample:?} / {expected:?}"
                );
            }
        }
    }
}

#[test]
fn the_ordinary_rig_connects_at_one_outgoing_frosted_interface() {
    use glassforms::render::LightFrom;

    let rough = Settings {
        rough: 0.30,
        rough_ref: 1.0,
        ..Default::default()
    };
    let clear = Settings::default();
    for material in [
        material::Variant::of(&rough),
        material::Variant::of_bodies(&[&rough, &clear]),
    ] {
        for (light, near, once) in [
            (LightFrom::Lights, false, true),
            (LightFrom::Lights, true, false),
            (LightFrom::Picture, false, false),
        ] {
            let options = Variant {
                mat: material,
                light,
                near,
                ..Default::default()
            };
            let march = March {
                step_scale: 0.9,
                steps_out: 180,
                steps_in: 64,
                layers: 2,
            };
            let shader = tracer_source(&march, options, false);
            assert!(shader.contains("fn frostConnectLight("));
            assert_eq!(shader.contains("var roomSampled"), once);
            assert_eq!(shader.contains("frSceneRay(directFrost.point"), !once);
            let parsed = naga::front::wgsl::parse_str(&shader).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&parsed)
            .unwrap();
        }
    }
}

fn reference_paint_film(
    c: f64,
    lambda: f64,
    nm: f64,
    incident: f64,
    film: f64,
    outer: f64,
    body: f64,
    coated: bool,
) -> [f64; 3] {
    let tangential = incident * (1.0 - c * c).sqrt();
    let mut expected = [0.0; 3];
    for tm in [false, true] {
        let admittance = |n: f64| {
            let normal = (n * n - tangential * tangential).abs().sqrt();
            if tm {
                normal / (n * n)
            } else {
                normal
            }
        };
        let a = admittance(incident);
        let b = admittance(film);
        let h = admittance(outer);
        let phase =
            std::f64::consts::TAU * nm * (film * film - tangential * tangential).abs().sqrt()
                / lambda;
        let top = if tangential >= outer {
            1.0
        } else if tangential > film {
            // Identical exterior media with a low-index gap. Here T=1/(1+A sinh²δ).
            assert_eq!(incident, outer);
            let factor = (a * a + b * b) / (2.0 * a * b);
            1.0 - 1.0 / (1.0 + (factor * phase.sinh()).powi(2))
        } else {
            // Sum the complex reflected waves, keeping their phase through each film traversal.
            let r01 = (a - b) / (a + b);
            let r12 = (b - h) / (b + h);
            let real = (2.0 * phase).cos();
            let imag = (2.0 * phase).sin();
            ((r01 + r12 * real).powi(2) + (r12 * imag).powi(2))
                / ((1.0 + r01 * r12 * real).powi(2) + (r01 * r12 * imag).powi(2))
        };
        let base = if !coated {
            0.0
        } else if tangential >= body {
            1.0
        } else {
            let end = admittance(body);
            ((h - end) / (h + end)).powi(2)
        };
        expected[0] += 0.5 * top;
        let mut down = 1.0 - top;
        for _ in 0..2000 {
            expected[2] += 0.5 * down * (1.0 - base);
            let up = down * base;
            expected[1] += 0.5 * up * (1.0 - top);
            down = up * top;
            if down < 1e-11 {
                break;
            }
        }
    }
    expected
}

#[test]
fn opaque_paint_film_matches_independent_polarized_interface_amplitudes() {
    let Ok(gpu) = glassforms::gpu::Gpu::headless("vulkan") else {
        return;
    };
    let source = concat!(
        include_str!("../src/shaders/pbr/types.wgsl"),
        include_str!("../src/shaders/pbr/dielectric.wgsl"),
        include_str!("../src/shaders/dielectric_fresnel.wgsl"),
        include_str!("../src/shaders/paint_interface.wgsl"),
        include_str!("../src/shaders/paint_film.wgsl"),
        r#"
@group(0) @binding(0) var image: texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn check(@builtin(global_invocation_id) id:vec3u) {
  let c=(f32(id.x)+0.5)/48.0;
  let wavelength=400.0+(f32(id.y)+0.5)*6.25;
  let incoming=select(1.0,1.5,id.z>=3u);
  let coat=array<f32,3>(1.0,1.48,1.8)[id.z%3u];
  let film=select(1.93,1.0,id.z==6u);
  let body=select(1.64,1.5,id.z==6u);
  let nm=select(385.0,120.0,id.z==6u);
  textureStore(image,id,vec4f(paintFilmWeights(c,incoming,body,coat,id.z!=0u && id.z!=6u,film,nm,wavelength),1.0));
}"#
    );
    let tex = sample_cases(&gpu, "wavelength-dependent pigment and coat", source, 48, 7);
    for case in 0..7 {
        let bytes = gpu.read(&tex, 0, 0, case, 48, 48, 16).unwrap();
        let data = bytemuck::cast_slice::<u8, [f32; 4]>(&bytes);
        let incident = if case < 3 { 1.0_f64 } else { 1.5 };
        let coated = case != 0 && case != 6;
        let (film, body, nm) = if case == 6 {
            (1.0, 1.5, 120.0)
        } else {
            (1.93, 1.64, 385.0)
        };
        let outer = if coated {
            [1.0, 1.48, 1.8][case as usize % 3]
        } else {
            body
        };
        for (i, sample) in data.iter().enumerate() {
            let c = ((i % 48) as f64 + 0.5) / 48.;
            let lambda = 400. + ((i / 48) as f64 + 0.5) * 6.25;
            let expected = reference_paint_film(c, lambda, nm, incident, film, outer, body, coated);
            for (observed, reference) in sample[..3].iter().zip(expected) {
                assert!(
                    (f64::from(*observed) - reference).abs() < 0.00025,
                    "case {case}, cos={c}, wavelength={lambda}: {sample:?} / {expected:?}"
                );
            }
        }
    }
}

#[test]
fn paint_film_rgb_matches_a_dense_cie_integral_across_the_authored_thickness_range() {
    let Ok(gpu) = glassforms::gpu::Gpu::headless("vulkan") else {
        return;
    };
    let source = concat!(
        include_str!("../src/shaders/pbr/types.wgsl"),
        include_str!("../src/shaders/pbr/dielectric.wgsl"),
        include_str!("../src/shaders/dielectric_fresnel.wgsl"),
        include_str!("../src/shaders/paint_interface.wgsl"),
        include_str!("../src/shaders/paint_film.wgsl"),
        r#"
@group(0) @binding(0) var record: texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn check(@builtin(global_invocation_id) id:vec3u) {
  let c=(f32(id.x)+0.5)/8.0;
  let nm=array<f32,6>(120.,360.,530.,1800.,5000.,10000.)[id.z%6u];
  let film=select(1.9,4.0,id.z%12u>=6u);
  let body=select(1.64,1.05,id.z%12u>=6u);
  let value=paintFilmRGB(c,1.,body,1.5,id.z>=12u,film,nm);
  textureStore(record,id,vec4f(value[id.y%3u],1.0));
}"#
    );
    let tex = sample_cases(
        &gpu,
        "integrated pigment film across optical thickness",
        source,
        8,
        24,
    );
    let mut peak_top_error = 0.0_f64;
    let mut peak_base_error = 0.0_f64;
    for case in 0..24 {
        let bytes = gpu.read(&tex, 0, 0, case, 8, 3, 16).unwrap();
        let values = bytemuck::cast_slice::<u8, [f32; 4]>(&bytes);
        let nm = [120., 360., 530., 1800., 5000., 10000.][case as usize % 6];
        let (film, body) = if case % 12 >= 6 {
            (4.0, 1.05)
        } else {
            (1.9, 1.64)
        };
        for i in 0..8 {
            let c = (i as f64 + 0.5) / 8.;
            let mut reflected = [0.; 3];
            let mut substrate = [0.; 3];
            let coated = case >= 12;
            let mut white = [0.; 3];
            // 0.125 nm wavelength spacing resolves the fringes even at the maximum film thickness.
            for j in 0..2400 {
                let wavelength = 400. + (j as f64 + 0.5) * 0.125;
                let w = gf_core::material::spec_w(wavelength);
                let (top, base) = if coated {
                    let [top, base, _] =
                        reference_paint_film(c, wavelength, nm, 1.0, film, 1.5, body, true);
                    (top, base)
                } else {
                    (
                        gf_core::material::film_reflectance(
                            c,
                            wavelength,
                            nm,
                            1.0,
                            film,
                            (body, 0.0),
                        ),
                        0.0,
                    )
                };
                for channel in 0..3 {
                    reflected[channel] += top * w[channel];
                    substrate[channel] += base * w[channel];
                    white[channel] += w[channel];
                }
            }
            for channel in 0..3 {
                let expected = (reflected[channel] / white[channel]).clamp(0., 1.);
                let expected_base = (substrate[channel] / white[channel]).clamp(0., 1. - expected);
                let top = f64::from(values[i][channel]);
                let base = f64::from(values[8 + i][channel]);
                let diffuse = f64::from(values[16 + i][channel]);
                peak_top_error = peak_top_error.max((top - expected).abs());
                peak_base_error = peak_base_error.max((base - expected_base).abs());
                assert!(
                    (top - expected).abs() < 0.002,
                    "case {case}, {nm} nm, cos {c}, channel {channel}: {top} / {expected}"
                );
                assert!((base-expected_base).abs() < 0.002, "case {case}, {nm} nm, cos {c}, channel {channel}: buried {base} / {expected_base}");
                assert!((top + base + diffuse - 1.).abs() < 1e-5);
            }
        }
    }
    eprintln!("maximum film reflectance error {peak_top_error:.7}, buried return error {peak_base_error:.7}");
}
