use gf_core::material::Variant as Material;
use glassforms::render::{tracer_source, March, Variant};

const MATERIAL_SOURCE: &str = include_str!("../src/shaders/trace_material.wgsl");
const SMOOTH_INTERFACES: [&str; 7] = [
    "pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd).reflectance",
    "pbr_dielectric_split(traceDielectricHit(ne, airMedium(), resolvedSceneMedium(ior)), -rd).transmitted",
    "pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2).reflectance",
    "pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2).transmitted",
    "pbr_dielectric_split(traceDielectricHit(-mf, resolvedSceneMedium(iorIn), airMedium()), -dir).reflectance",
    "pbr_dielectric_split(traceDielectricHit(-mN, resolvedSceneMedium(iorIn), airMedium()), -dir).transmitted",
    // The shadeM entry deliberately has the same text as beyondM's smooth entry.
    "pbr_dielectric_split(traceDielectricHit(ne, airMedium(), resolvedSceneMedium(ior)), -rd).transmitted",
];

fn march() -> March {
    March {
        step_scale: 0.95,
        steps_out: 220,
        steps_in: 56,
        layers: 2,
    }
}

fn compose(mat: Material) -> String {
    tracer_source(
        &march(),
        Variant {
            mat,
            ..Default::default()
        },
        false,
    )
}

fn validate(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

fn functions(source: &str) -> (&str, &str) {
    let beyond = source
        .split_once("fn beyondM(")
        .unwrap()
        .1
        .split_once("\nfn surfaceK(")
        .unwrap()
        .0;
    let shade = source
        .split_once("\nfn shadeM(")
        .unwrap()
        .1
        .split_once("\nfn grazeM(")
        .unwrap()
        .0;
    (beyond, shade)
}

#[test]
fn smooth_material_interfaces_compose_through_the_resolved_ior_adapter() {
    let source = compose(Material {
        spectrum: true,
        ..Default::default()
    });
    validate(&source);
    assert!(source.contains("fn shadeM(ro: vec3f"));
    assert!(source.contains("PbrFrame(tangent, cross(normal, tangent), normal), incident, destination, 550.0"));
    // Five plain-tracer helpers remain beside the material tracer; the two functions in scope own seven.
    assert_eq!(
        source
            .matches("pbr_dielectric_split(traceDielectricHit(")
            .count(),
        12
    );
    for interface in SMOOTH_INTERFACES {
        assert!(source.contains(interface), "smooth source lost {interface}");
    }

    let (beyond, shade) = functions(&source);
    assert_eq!(
        beyond
            .matches("pbr_dielectric_split(traceDielectricHit(")
            .count(),
        4
    );
    assert_eq!(
        shade
            .matches("pbr_dielectric_split(traceDielectricHit(")
            .count(),
        3
    );
    assert!(!beyond.contains("refract("));
    assert!(!beyond.contains("fresnel(ci, 1.0, ior)"));
    assert!(!beyond.contains("fresnel(co, ior, 1.0)"));
    assert!(!shade.contains("refract("));
    assert!(!shade.contains("fresnel(cf, iorIn, 1.0)"));

    // The adapter supplies already-resolved indices; spectral path selection and transport policy stay here.
    assert!(source.contains("fn iorOf(i: i32, xi: f32, W: i32) -> f32"));
    assert!(source.contains("return iorAt(400.0 + 300.0 * (f32(i) + xi) / f32(W));"));
    assert!(MATERIAL_SOURCE.contains("var thr = 1.0 - top1(cosi);"));
    assert!(shade.contains("X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg));"));
    assert!(beyond.contains("thr *= (1.0 - F) * exp(-u.absorb.xyz * s);"));
    assert!(beyond.contains("if (F2 > 0.999)"));
    assert!(beyond.contains("ro = q + m * 0.006;"));
    assert!(shade.contains("let p0 = p - n * 0.004;"));
    assert!(shade.contains("if (bare < 0.999)"));
    assert!(shade.contains("let np = q - mN * 0.004;"));
    assert!(shade.contains("att *= exp(-u.absorb.xyz * s);"));
    assert!(shade.contains("if (b >= u.nb || thr < 0.02)"));
}

#[test]
fn layered_smooth_controls_remain_outside_the_bare_interface_leaf() {
    for (name, mat, required) in [
        (
            "film",
            Material {
                film: true,
                ..Default::default()
            },
            "filmRGB(ci, filmFar(p), 1.0, ior)",
        ),
        (
            "coat",
            Material {
                coat: true,
                ..Default::default()
            },
            "let F = fresnel(sqrt(max(1.0 - (1.0 - ci * ci)",
        ),
        (
            "hollow",
            Material {
                hollow: true,
                mono: true,
                ..Default::default()
            },
            "let F = wallR(ci);",
        ),
    ] {
        let source = compose(mat);
        validate(&source);
        assert!(
            source.contains(required),
            "{name} lost its specialised reflectance"
        );
        assert!(source.contains("pbr_dielectric_split(traceDielectricHit("));
        assert!(source.contains("var thr ="));
        let (beyond, shade) = functions(&source);
        assert!(
            !beyond.contains("pbr_dielectric_sample("),
            "{name} changed event sampling"
        );
        assert!(
            !shade.contains("pbr_dielectric_sample("),
            "{name} changed event sampling"
        );
    }

    let coated_film = compose(Material {
        coat: true,
        film: true,
        rough: true,
        rough_ref: true,
        ..Default::default()
    });
    validate(&coated_film);
    assert!(coated_film.contains("filmRGB(ci, filmFar(p), 1.0, mu.surf.z)"));
    assert!(coated_film.contains("frDir(refract(rd, ne, 1.0 / ior), n)"));
    let (beyond, shade) = functions(&coated_film);
    assert!(!beyond.contains("pbr_dielectric_split(traceDielectricHit("));
    assert!(!shade.contains("pbr_dielectric_split(traceDielectricHit("));

    let rough = compose(Material {
        rough: true,
        rough_ref: true,
        ..Default::default()
    });
    validate(&rough);
    assert!(rough.contains("od = frExit(dir, mN, mf, fw, iorIn);"));
    assert!(rough.contains("let fw = frWall(cf, fv, iorIn, 1.0, bend);"));
    let (beyond, shade) = functions(&rough);
    assert!(!beyond.contains("pbr_dielectric_split(traceDielectricHit("));
    assert!(!shade.contains("pbr_dielectric_split(traceDielectricHit("));
}

#[test]
fn birefringent_transport_keeps_wave_and_extraordinary_ray_interfaces() {
    let source = compose(Material {
        biref: true,
        ..Default::default()
    });
    validate(&source);
    let (beyond, shade) = functions(&source);
    assert!(!beyond.contains("pbr_dielectric_split("));
    assert!(!shade.contains("pbr_dielectric_split("));
    assert!(shade.contains("var kv = refract(rd, ne, 1.0 / ior) * ior;"));
    assert!(shade.contains("waveE(rd - dot(rd, ne) * ne, -ne, axisW(), ior, nE)"));
    assert!(shade.contains("let bare = fresnel(cf, iorIn, 1.0);"));
    assert!(shade.contains("var od = normalize(kv - dot(kv, mN) * mN"));
    assert!(shade.contains("if (extra && dot(kn, kn) > 0.01)"));
}

#[test]
fn split_migration_adds_no_random_draw_or_changes_walk_sampling() {
    for (mat, walk_calls) in [
        (
            Material {
                spectrum: true,
                ..Default::default()
            },
            0,
        ),
        (
            Material {
                sss: true,
                walk: true,
                mono: true,
                ..Default::default()
            },
            1,
        ),
    ] {
        let source = compose(mat);
        validate(&source);
        let (beyond, shade) = functions(&source);
        assert!(!beyond.contains("pbr_dielectric_sample("));
        assert!(!shade.contains("pbr_dielectric_sample("));
        assert!(!beyond.contains("interface_draw"));
        assert!(!shade.contains("interface_draw"));
        assert_eq!(beyond.matches("walkFlight(rates)").count(), walk_calls);
        assert_eq!(shade.matches("walkFlight(rates)").count(), walk_calls);
        assert_eq!(beyond.matches("wander(").count(), walk_calls);
        assert_eq!(shade.matches("wander(").count(), walk_calls);
        assert_eq!(shade.matches("rState = pcg(").count(), walk_calls);
    }

    // The two source-level sampling calls belong to wander; the deterministic split adds no others.
    assert_eq!(MATERIAL_SOURCE.matches("pbr_dielectric_sample(").count(), 2);
    assert_eq!(
        MATERIAL_SOURCE
            .matches("pbr_dielectric_split(traceDielectricHit(")
            .count(),
        7
    );
}
