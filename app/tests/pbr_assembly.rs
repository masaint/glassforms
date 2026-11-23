use glassforms::render::{tracer_source, March, Variant};

const BODY_SOURCE: &str = include_str!("../src/shaders/trace_bodies.wgsl");
const COMMON_SPLITS: [&str; 4] = [
    "pbr_dielectric_split(dielectricHit,-dir)",
    "pbr_dielectric_split(traceDielectricHit(nq, mediumB(m, na), mediumB(b, nb)), -rd).transmitted",
    "pbr_dielectric_split(traceDielectricHit(n, airMedium(), mediumB(k, n1)), -rd).transmitted",
    "pbr_dielectric_split(traceDielectricHit(nq, mediumB(m, na), mediumB(nx, nb)), -dir).transmitted",
];

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

#[test]
fn non_birefringent_body_refraction_uses_the_common_split_without_changing_transport_policy() {
    assert_eq!(BODY_SOURCE.matches("pbr_dielectric_split(").count(), 4);
    assert!(!BODY_SOURCE.contains("refract("));
    for split in COMMON_SPLITS {
        assert_eq!(
            BODY_SOURCE.matches(split).count(),
            1,
            "missing or duplicated {split}"
        );
    }

    // Bare walkers sample the resolved non-TIR interface without a second Fresnel weight. Every
    // layered, rough, opaque or extraordinary path keeps its established surfB/BIREF alternative.
    assert!(
        BODY_SOURCE.contains("let dielectricHit=traceDielectricHit(n,mediumB(m,iorB(m,lam)),mediumB(b,iorB(b,lam)));")
    );
    assert_eq!(BODY_SOURCE.matches("pbr_dielectric_sample(").count(), 1);
    assert!(BODY_SOURCE.contains("if(dielectricSplit.total_internal_reflection!=0u){"));
    assert!(BODY_SOURCE.contains("plainDielectric=plainDielectric && !sampled;"));
    assert!(BODY_SOURCE.contains("plainDielectric=plainDielectric && !biB(m) && !biB(b);"));
    assert!(!BODY_SOURCE.contains("interfaceSample.weight"));
    assert!(BODY_SOURCE.contains("let c=clamp(-dot(dir,n),0.0,1.0);let r=surfB(m,b,c,q,n,lam);"));
    assert!(BODY_SOURCE.contains("}else if(rnd()<prob){"));
    assert!(BODY_SOURCE.contains("keep*=(vec3f(1.0)-r)/max(1.0-prob,1e-6);"));
    assert!(BODY_SOURCE.contains("dir=refracted;pos=q-0.004*n;m=b;"));
    assert!(BODY_SOURCE.contains("let R = surfB(m, b, c, q, nq, lam);"));
    assert!(BODY_SOURCE.contains("thr *= 1.0 - R;"));
    assert!(BODY_SOURCE.contains("ro = q - nq * 0.006;"));
    assert!(BODY_SOURCE.contains("let R0 = surfB(-1, k, cosi, p, n, lam);"));
    assert!(BODY_SOURCE.contains("var att = 1.0 - R0;"));
    assert!(BODY_SOURCE.contains("var R = surfB(m, nx, c, q, nq, lam);"));
    assert!(BODY_SOURCE.contains("pos = q - nq * 0.004;"));
}

#[test]
fn assembled_body_tracers_validate_with_and_without_birefringence() {
    let cases = [
        ("smooth walker", false, false),
        ("rough deterministic walker", true, false),
        ("rough sampled walker", true, true),
    ];
    for biref in [false, true] {
        for (name, rough, rough_ref) in cases {
            let mat = gf_core::material::Variant {
                bodies: 3,
                biref,
                rough,
                rough_ref,
                sss: true,
                walk: true,
                ..Default::default()
            };
            let source = tracer_source(
                &March {
                    step_scale: 0.95,
                    steps_out: 220,
                    steps_in: 56,
                    layers: 2,
                },
                Variant {
                    mat,
                    ..Default::default()
                },
                false,
            );
            assert!(source
                .contains("PbrFrame(tangent, cross(normal, tangent), normal), incident, destination, 550.0"));
            assert_eq!(
                source
                    .matches("dielectricHit,-dir,PbrPathRng(vec2f(0.0),rnd())")
                    .count(),
                1,
                "{name}, birefringence {biref}: exactly one bare body sampling site"
            );
            if !biref {
                for split in COMMON_SPLITS {
                    assert!(
                        source.contains(split),
                        "{name}: non-birefringent source lost {split}"
                    );
                }
            }
            validate(&source);
        }
    }
}
