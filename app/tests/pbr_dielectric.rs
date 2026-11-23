use gf_core::material::fresnel as existing_fresnel;
use gf_core::settings::Settings;
use glassforms::gpu::Gpu;

const LEAF: &str = concat!(
    include_str!("../src/shaders/pbr/types.wgsl"),
    include_str!("../src/shaders/pbr/dielectric.wgsl"),
);
const EVENT_INVALID: u32 = 0;
const EVENT_REFLECTION: u32 = 1;
const EVENT_TRANSMISSION: u32 = 2;
const EVENT_TIR: u32 = 3;
const MEDIUM_STAY: u32 = 0;
const MEDIUM_ENTER_DESTINATION: u32 = 1;

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
fn smooth_dielectric_leaf_passes_naga_validation() {
    validate(LEAF);
    let optics = LEAF
        .split_once("fn pbr_medium_ior_at(")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    assert!(!optics.contains("identity") && !optics.contains("interface_priority"));
    assert!(include_str!("../src/shaders/trace.wgsl")
        .contains("fn airMedium() -> PbrMedium { return PbrMedium(1.0, 0.0, 0u, 0); }"));
}

#[test]
fn established_fresnel_entry_point_delegates_to_the_pure_leaf() {
    validate(&[
        LEAF,
        include_str!("../src/shaders/dielectric_fresnel.wgsl"),
    ].join("\n"));
    let forward = glassforms::cast_light::LIGHT_WGSL;
    assert!(forward.contains("fn pbr_dielectric_fresnel("));
    assert!(forward.contains("return pbr_medium_ior_at(PbrMedium(lu.body[j].a.x, lu.body[j].a.y, 0u, 0), lam);"));
    assert_eq!(forward.matches("fn fresnel(cosi: f32, n1: f32, n2: f32)").count(), 1);
}

#[test]
fn forward_clear_interfaces_use_the_common_split_without_changing_transport_policy() {
    let forward = glassforms::cast_light::LIGHT_WGSL;
    validate(forward);
    let follow = forward
        .split_once("fn follow(").unwrap().1
        .split_once("\n@compute @workgroup_size(8, 8, 1)").unwrap().0;
    assert!(forward.contains(
        "return pbr_medium_ior_at(PbrMedium(lu.body[j].a.x, lu.body[j].a.y, 0u, 0), lam);"
    ));
    assert!(forward.contains(
        "PbrFrame(tangent, cross(normal, tangent), normal), incident, destination, 550.0"
    ));
    assert_eq!(follow.matches(
        "pbr_dielectric_split(castDielectricHit(n, castMedium(m, n1), castMedium(next, n2)), -d)"
    ).count(), 1);
    assert!(!follow.contains("pbr_dielectric_sample("));
    assert!(follow.contains("for (var e = 0; e < EVENTS; e++)"));
    assert!(follow.contains("let leaving = j == m;"));
    assert!(follow.contains("if (k < m && dd[k] < 0.0) { next = k; }"));

    // Receiver/stopper and conductor ownership still precede the clear-body interface. Rough conductors
    // alone retain their RNG walk; smooth conductors and coats retain their established optics.
    assert!(follow.contains("if (next >= 0 && role == 4)"));
    assert!(follow.contains("if (next >= 0 && role == 3) { return out; }"));
    assert!(follow.contains("if (next >= 0 && role == 2)"));
    assert!(follow.contains("d = castConductorWalk(next,d,n,n1,rough*rough,&state,&w);"));
    assert!(follow.contains("w *= conductorReflectance(next, cosi, n1);"));

    let clear = follow
        .split_once("// a clear body, or the open: through, unless it is turned back whole").unwrap().1
        .split_once("if (max(w.x, max(w.y, w.z)) < 0.004) { return out; }").unwrap().0;
    assert!(follow.contains("let n1 = select(1.0, iorOf(max(m, 0), lam), m >= 0);"));
    assert!(clear.contains("let n2 = select(1.0, iorOf(max(next, 0), lam), next >= 0);"));
    assert!(clear.contains("let F = split.reflectance;"));
    assert!(clear.contains("if (F > 0.999)"));
    assert!(clear.contains("d = split.reflected;"));
    assert!(clear.contains("p += n * off;"));
    assert!(clear.contains("sig = sig * 16u + 10u;"));
    assert!(clear.contains("w *= split.transmission_weight;"));
    assert!(clear.contains("d = split.transmitted;"));
    assert!(clear.contains("p -= n * off;"));
    assert!(clear.contains("m = next;"));
    assert!(clear.contains("sig = sig * 16u + u32(next + 2);"));
    assert_eq!(clear.matches("travel = 0.0;").count(), 1);
    assert!(!clear.contains("fresnel(") && !clear.contains("refract("));

    assert!(follow.contains("if (max(w.x, max(w.y, w.z)) < 0.004) { return out; }"));
    assert!(follow.contains("let t = march(p, d, m, lu.gone1, &which);"));
    assert!(follow.contains("sig * 16u + u32(next) + 11u, pack(w), select(travel,-max(travel,1e-7),sampledMetal)"));
    assert!(follow.contains("select(travel+tf,-(travel+tf),sampledMetal)"));
    assert!(follow.contains("if (m >= 0) { w *= exp(-lu.body[m].absorb.rgb * t); }"));
    assert!(follow.contains("travel += t;"));
    assert_eq!(follow.matches("p += d * (sgn * smoothD(p, lu.body[which].mask));").count(), 2);
    assert!(follow.contains("j = which;"));
}

#[test]
fn plain_production_interfaces_use_the_common_split_without_changing_transport_policy() {
    let march = glassforms::render::March {
        step_scale: 0.95,
        steps_out: 220,
        steps_in: 56,
        layers: 2,
    };
    let source = glassforms::render::tracer_source(
        &march,
        glassforms::render::Variant::default(),
        false,
    );
    validate(&source);
    let free_camera = glassforms::render::eye_weave(&source);
    assert!(free_camera.find("  medium: vec4i,").unwrap() < free_camera.find("  eye: vec4f,").unwrap());
    validate(&free_camera);
    assert_eq!(source.matches("pbr_dielectric_split(traceDielectricHit(").count(), 5);
    assert!(source.contains("PbrFrame(tangent, cross(normal, tangent), normal), incident, destination, 550.0"));

    let beyond = source.split_once("fn beyond(").unwrap().1.split_once("\nfn shade(").unwrap().0;
    assert!(!beyond.contains("refract(") && !beyond.contains("fresnel("));
    assert!(beyond.contains("col += thr * F * env(entry.reflected);"));
    assert!(beyond.contains("let d2 = entry.transmitted;"));
    assert!(beyond.contains("traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2"));
    assert!(beyond.contains("if (F2 > 0.999)"));
    assert!(beyond.contains("thr *= (1.0 - F) * exp(-u.absorb.xyz * s);"));

    let shade = source.split_once("\nfn shade(").unwrap().1.split_once("\nfn primary(").unwrap().0;
    assert!(!shade.contains("refract(") && !shade.contains("fresnel("));
    assert!(shade.contains("traceDielectricHit(n, airMedium(), resolvedSceneMedium(u.ior)), -rd"));
    assert!(shade.contains("let Fl = F0;"));
    assert!(shade.contains("traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd"));
    assert!(shade.contains("traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -dir"));
    assert!(shade.contains("if (F < 0.999)"));
    assert!(shade.contains("att *= exp(-u.absorb.xyz * s);"));
    assert!(shade.contains("dir = exit.reflected;"));

    let hero = glassforms::hero::weave(
        &source,
        &glassforms::hero::Hero { mode: 1, k: 1, clamp: 3, ..Default::default() },
    ).expect("the production adapter must not break hero spectral exits");
    assert!(hero.contains("let oc = pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(iorAt(lc)), airMedium()), -dir).transmitted;"));
    assert!(!hero.contains("refract(dir, -m"));
    validate(&hero);

    let families = gf_core::presets::families();
    let (family, _) = families.iter().enumerate().find(|(_, family)| family.id == "algebraic").unwrap();
    let declaration = gf_core::csg::wgsl(&families, &[(family, 0)], gf_core::sdf::BINDING, false);
    let tree = format!(
        "{declaration}\nfn cnSample(p: vec3f) -> vec3f {{ return vec3f(csgDist(p),0.0,-1.0); }}\n// construction surface materials {}",
        serde_json::to_string(&[gf_core::material::Variant::default()]).unwrap(),
    );
    let construction = glassforms::sdf::weave(&source, &tree);
    assert_eq!(construction.matches("cnPick(p);\n    let ior = iorAt(lam);").count(), 1);
    let construction_shade = construction.split_once("\nfn shade(").unwrap().1.split_once("\nfn primary(").unwrap().0;
    assert!(!construction_shade.contains("cnPick(p);"));
    validate(&construction);
}

#[test]
fn random_walk_samples_each_oriented_interface_once_and_transport_applies_transitions() {
    let mut settings = Settings::default();
    assert!(settings.set("sss", 1.0));
    assert!(settings.set("sssWalk", 1.0));
    let march = glassforms::render::March {
        step_scale: 0.95,
        steps_out: 220,
        steps_in: 56,
        layers: 2,
    };
    let source = glassforms::render::tracer_source(
        &march,
        glassforms::render::Variant {
            mat: gf_core::material::Variant::of(&settings),
            ..Default::default()
        },
        false,
    );
    validate(&source);

    let wander = source
        .split_once("fn wander(")
        .unwrap()
        .1
        .split_once("\nfn iorOf(")
        .unwrap()
        .0;
    assert_eq!(wander.matches("pbr_dielectric_sample(").count(), 2);
    assert_eq!(
        wander
            .matches("PbrPathRng(vec2f(0.0), rnd())")
            .count(),
        2,
        "one existing PCG interface draw per sampled hit; no eager or substitute draw"
    );
    assert!(wander.contains(
        "traceDielectricHit(-m, resolvedSceneMedium(u.ior), airMedium()), -dir, PbrPathRng(vec2f(0.0), rnd())"
    ));
    assert!(wander.contains(
        "traceDielectricHit(n, airMedium(), resolvedSceneMedium(u.ior)), -dir, PbrPathRng(vec2f(0.0), rnd())"
    ));
    assert!(!wander.contains("rnd() < fresnel("));
    assert!(!wander.contains("refract(dir"));
    assert!(!wander.contains("interface_sample.weight"));
    assert_eq!(
        wander
            .matches("interface_sample.medium_transition == PBR_MEDIUM_ENTER_DESTINATION")
            .count(),
        2
    );
    assert!(wander.contains("pos += m * 0.006;"));
    assert!(wander.contains("pos -= m * 0.004;"));
    assert!(wander.contains("pos -= n * 0.004;"));
    assert!(wander.contains("pos += n * 0.004;"));
    assert_eq!(wander.matches("att *=").count(), 1);
    assert!(wander.contains(
        "att *= exp(-(u.absorb.xyz + mu.sssA.rgb) * got) * walkWeight(rates, got, !hit);"
    ));
    assert!(wander.contains("return lit + att * seen(dir, 0.1);"));

    let bodies = include_str!("../src/shaders/trace_bodies.wgsl");
    let walk_body = bodies
        .split_once("fn walkBodyPath(")
        .unwrap()
        .1
        .split_once("\nfn throughB(")
        .unwrap()
        .0;
    assert_eq!(walk_body.matches("pbr_dielectric_sample(").count(), 1);
    assert_eq!(walk_body.matches("PbrPathRng(").count(), 1);
    assert!(walk_body.contains(
        "let dielectricHit=traceDielectricHit(n,mediumB(m,iorB(m,lam)),mediumB(b,iorB(b,lam)));"
    ));
    assert!(walk_body.contains("let dielectricSplit=pbr_dielectric_split(dielectricHit,-dir);"));
    assert!(walk_body.contains("!opaqueB(b) && !wallB(interfaceOwner)"));
    assert!(walk_body.contains("mb.b[interfaceOwner].surf.x<=0.0"));
    assert!(walk_body.contains("mb.b[interfaceOwner].surf.z<=1.001"));
    assert!(walk_body.contains("plainDielectric=plainDielectric && mb.b[interfaceOwner].film.x<=0.0;"));
    assert!(walk_body.contains("plainDielectric=plainDielectric && !sampled;"));
    assert!(walk_body.contains("plainDielectric=plainDielectric && !biB(m) && !biB(b);"));

    let plain_branch = walk_body
        .split_once("    if(plainDielectric){\n")
        .unwrap()
        .1
        .split_once("      continue;\n    }\n")
        .unwrap()
        .0;
    let (tir, sampled) = plain_branch
        .split_once("if(dielectricSplit.total_internal_reflection!=0u){")
        .unwrap()
        .1
        .split_once("}else{")
        .unwrap();
    assert!(!tir.contains("rnd()") && !tir.contains("pbr_dielectric_sample("));
    assert!(tir.contains("dir=dielectricSplit.reflected;pos=q+0.004*n;"));
    assert_eq!(sampled.matches("rnd()").count(), 1);
    assert_eq!(sampled.matches("pbr_dielectric_sample(").count(), 1);
    assert!(sampled.contains("dielectricHit,-dir,PbrPathRng(vec2f(0.0),rnd())"));
    assert!(!plain_branch.contains("interfaceSample.weight"));
    assert!(sampled.contains(
        "if(interfaceSample.medium_transition==PBR_MEDIUM_ENTER_DESTINATION){\n          pos=q-0.004*n;m=b;\n        }else{\n          pos=q+0.004*n;\n        }"
    ));
    assert_eq!(plain_branch.matches("m=b;").count(), 1);

    assert!(walk_body.contains("for(var e=0;e<256;e++)"));
    assert!(walk_body.contains("if(max3(keep)<0.002){break;}"));
    assert!(walk_body.contains("if (sampled && e >= max(u.nb,8))"));
    assert!(walk_body.contains("keep*=exp(-(mb.absorb[m].rgb+mb.b[m].sssA.rgb)*fly)*walkWeight(rates,fly,true);"));
    assert!(walk_body.contains("dir=refracted;pos=q-0.004*n;m=b;"));
}

// Independent f64 optics. Reflectance uses the angular sine/tangent amplitude forms, rather than
// transcribing either the new WGSL or gf_core::material::fresnel. The latter is a second control.
type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] + b[i])
}
fn scale(a: V3, s: f64) -> V3 {
    a.map(|v| v * s)
}
fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: V3, b: V3) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn unit(v: V3) -> V3 {
    scale(v, dot(v, v).sqrt().recip())
}
fn close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{context}: {actual} / {expected}"
    );
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance * (1.0 + expected.abs()),
        "{context}: {actual} != {expected} (error {error})"
    );
}

#[derive(Clone, Copy)]
struct Medium {
    ior_550: f64,
    dispersion: f64,
    identity: u32,
    interface_priority: i32,
}

fn air() -> Medium {
    Medium {
        ior_550: 1.0,
        dispersion: 0.0,
        identity: 0,
        interface_priority: 0,
    }
}

fn crown() -> Medium {
    Medium {
        ior_550: 1.52,
        dispersion: 0.01,
        identity: 7,
        interface_priority: 23,
    }
}

// Same documented Cauchy convention as the legacy tracer, evaluated independently in f64.
fn ior_at(medium: Medium, wavelength_nm: f64) -> f64 {
    medium.ior_550 + medium.dispersion * (1.0e6 / wavelength_nm.powi(2) - 3.3058) / 4.2092
}

struct Reference {
    reflected: V3,
    transmitted: V3,
    reflectance: f64,
    transmission: f64,
    incident_ior: f64,
    destination_ior: f64,
    tir: bool,
}

fn independent_interface(
    wo: V3,
    normal: V3,
    incident: Medium,
    destination: Medium,
    lambda: f64,
) -> Reference {
    let n1 = ior_at(incident, lambda);
    let n2 = ior_at(destination, lambda);
    let cos_i = dot(wo, normal).clamp(0.0, 1.0);
    let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    let sin_t = n1 / n2 * sin_i;
    let reflected = add(scale(wo, -1.0), scale(normal, 2.0 * cos_i));
    if sin_t >= 1.0 {
        return Reference {
            reflected,
            transmitted: [0.0; 3],
            reflectance: 1.0,
            transmission: 0.0,
            incident_ior: n1,
            destination_ior: n2,
            tir: true,
        };
    }

    let theta_i = cos_i.acos();
    let theta_t = sin_t.asin();
    let cos_t = theta_t.cos();
    let reflectance = if cos_i > 1.0 - 1e-13 {
        ((n1 - n2) / (n1 + n2)).powi(2)
    } else {
        let rs = (theta_i - theta_t).sin() / (theta_i + theta_t).sin();
        let rp = (theta_i - theta_t).tan() / (theta_i + theta_t).tan();
        0.5 * (rs * rs + rp * rp)
    };

    // Construct the destination direction from the two angles and the incident tangential axis.
    // This does not call or transcribe WGSL refract.
    let tangent = sub(wo, scale(normal, cos_i));
    let transmitted = if dot(tangent, tangent) > 1e-24 {
        add(scale(unit(tangent), -sin_t), scale(normal, -cos_t))
    } else {
        scale(normal, -1.0)
    };
    Reference {
        reflected,
        transmitted,
        reflectance,
        transmission: 1.0 - reflectance,
        incident_ior: n1,
        destination_ior: n2,
        tir: false,
    }
}

fn normal_frame(normal: V3) -> (V3, V3) {
    let n = unit(normal);
    let seed = if n[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = unit(sub(seed, scale(n, dot(seed, n))));
    let bitangent = [
        n[1] * tangent[2] - n[2] * tangent[1],
        n[2] * tangent[0] - n[0] * tangent[2],
        n[0] * tangent[1] - n[1] * tangent[0],
    ];
    (n, add(scale(tangent, 0.6), scale(bitangent, 0.8)))
}

fn incident_direction(normal: V3, theta: f64) -> V3 {
    let (n, tangent) = normal_frame(normal);
    unit(add(scale(n, theta.cos()), scale(tangent, theta.sin())))
}

#[test]
fn identity_only_medium_metadata_is_inert_in_the_real_renderer() {
    let args = [
        ("scene", "studio"), ("preset", "sphere"), ("res", "64"),
        ("material", "crown"),
        ("layer", "studio:sphere"), ("L2.size", "0.55"), ("L2.x", "0.5"),
        ("J12.op", "apart"), ("B2.material", "diamond"),
        ("layer", "studio:sphere"), ("L3.size", "0.8"), ("L3.y", "-0.6"),
        ("J13.op", "apart"), ("J23.op", "apart"), ("B3.material", "satinpaint"),
        ("cast", "2"), ("castShadows", "0"), ("spin", "0"), ("grain", "0"),
    ].map(|(key, value)| (key.to_string(), value.to_string()));
    let configurations = [
        [("medium:z.scene", 0), ("medium:m.second", 0), ("medium:a.third", 0)],
        [("medium:a.shared", 0), ("medium:a.shared", 0), ("medium:z.third", 0)],
    ];
    let mut previous: Option<(gf_core::pbr::CompiledMedia, Vec<f32>, Vec<u8>)> = None;
    for configuration in configurations {
        let mut state = gf_core::state::State::with_args(&args, true);
        state.enable_canonical_materials().unwrap();
        {
            let owner = state.canonical.as_mut().unwrap();
            for (spec, (id, priority)) in std::iter::once(&mut owner.scene)
                .chain(owner.bodies.iter_mut().filter_map(|body| body.as_deref_mut()))
                .zip(configuration)
            {
                spec.transvolume.values.medium = gf_core::pbr::MediumDescriptor {
                    id: gf_core::pbr::MediumId::from(id),
                    interface_priority: priority,
                };
            }
        }
        let media = state.compiled_media();
        assert!(media.scene.identity > 0 && media.bodies.iter().all(|medium| medium.identity > 0));
        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        let mut engine = glassforms::engine::Engine::new(gpu, state, true);
        engine.resize(128, 96);
        engine.adapt = false;
        engine.bake(gf_core::state::Detail::Full).unwrap();
        assert_eq!(engine.trace_uniforms().medium, media.scene.raw());
        engine.render(3);
        let hdr = engine.renderer.read_hdr(0, 0, 128, 96).unwrap();
        let png = engine.screenshot().unwrap();
        assert_eq!((png.0, png.1), (128, 96));
        if let Some((old_media, old_hdr, old_png)) = previous {
            assert_ne!(media, old_media, "the fixture must exercise new identities with zero priorities");
            assert_eq!(media.bodies[0].identity, media.bodies[1].identity, "shared persistent IDs share one token");
            assert_eq!((media.scene.interface_priority, media.bodies[0].interface_priority, media.bodies[1].interface_priority), (0, 0, 0));
            let changed = hdr.iter().zip(&old_hdr).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let largest = hdr.iter().zip(&old_hdr).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(changed == 0, "medium metadata changed {changed} linear-HDR lanes (largest delta {largest})");
            let changed = png.2.iter().zip(&old_png).filter(|(a, b)| a != b).count();
            assert!(changed == 0, "medium metadata changed {changed} PNG bytes");
        }
        previous = Some((media, hdr, png.2));
    }
}

#[test]
fn independent_snell_fresnel_controls_cover_normal_grazing_and_critical_angles() {
    let normal = [0.0, 0.0, 1.0];
    for (incident, destination) in [(air(), crown()), (crown(), air()), (air(), air())] {
        for theta in [0.0, 0.2, 0.7, 1.2, 1.569] {
            let wo = incident_direction(normal, theta);
            let r = independent_interface(wo, normal, incident, destination, 550.0);
            let existing = existing_fresnel(dot(wo, normal), r.incident_ior, r.destination_ior);
            close(
                f64::from(r.reflectance),
                existing,
                3e-10,
                "independent angular Fresnel versus existing control",
            );
            close(
                f64::from(r.reflectance) + r.transmission,
                1.0,
                1e-12,
                "lossless interface energy",
            );
            close(
                dot(r.reflected, r.reflected),
                1.0,
                1e-12,
                "reflected direction unit length",
            );
            if !r.tir {
                close(
                    dot(r.transmitted, r.transmitted),
                    1.0,
                    1e-12,
                    "transmitted direction unit length",
                );
                let tangent = sub(wo, scale(normal, dot(wo, normal)));
                if dot(tangent, tangent) > 1e-20 {
                    let sin_t = -dot(r.transmitted, unit(tangent));
                    close(
                        r.incident_ior * (1.0 - dot(wo, normal).powi(2)).sqrt(),
                        r.destination_ior * sin_t,
                        2e-12,
                        "Snell's law",
                    );
                }
            }
        }
    }

    let r = independent_interface(
        [0.0, 0.0, 1.0],
        normal,
        air(),
        Medium {
            ior_550: 1.5,
            dispersion: 0.0,
            identity: 5,
            interface_priority: -9,
        },
        550.0,
    );
    close(
        f64::from(r.reflectance),
        0.04,
        1e-12,
        "normal-incidence glass reflectance",
    );
    let grazing = independent_interface(
        incident_direction(normal, 1.569),
        normal,
        air(),
        crown(),
        550.0,
    );
    let oblique = independent_interface(
        incident_direction(normal, 1.2),
        normal,
        air(),
        crown(),
        550.0,
    );
    close(
        f64::from(grazing.reflectance),
        existing_fresnel(
            1.569f64.cos(),
            grazing.incident_ior,
            grazing.destination_ior,
        ),
        3e-10,
        "independent grazing Fresnel expectation",
    );
    assert!(
        grazing.reflectance > oblique.reflectance && grazing.reflectance < 1.0,
        "finite-angle grazing reflectance approaches one without becoming TIR"
    );

    let critical = (1.0f64 / 1.52).asin();
    let below = independent_interface(
        incident_direction(normal, critical - 1e-5),
        normal,
        crown(),
        air(),
        550.0,
    );
    let above = independent_interface(
        incident_direction(normal, critical + 1e-5),
        normal,
        crown(),
        air(),
        550.0,
    );
    assert!(!below.tir && below.transmission > 0.0);
    assert!(above.tir && above.transmission == 0.0 && above.reflectance == 1.0);

    let n400 = ior_at(crown(), 400.0);
    let n550 = ior_at(crown(), 550.0);
    let n700 = ior_at(crown(), 700.0);
    close(n550, 1.52, 2e-7, "Cauchy reference wavelength");
    close(n400 - n700, 0.01, 2e-7, "Cauchy 400-700 nm dispersion");
    assert!(n400 > n550 && n550 > n700);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCase {
    wo_incident_ior: [f32; 4],
    normal_incident_dispersion: [f32; 4],
    destination_wavelength: [f32; 4],
    random: [f32; 4],
}

impl GpuCase {
    fn new(
        wo: V3,
        normal: V3,
        incident: Medium,
        destination: Medium,
        lambda: f64,
        random: f64,
    ) -> Self {
        Self {
            wo_incident_ior: [
                wo[0] as f32,
                wo[1] as f32,
                wo[2] as f32,
                incident.ior_550 as f32,
            ],
            normal_incident_dispersion: [
                normal[0] as f32,
                normal[1] as f32,
                normal[2] as f32,
                incident.dispersion as f32,
            ],
            destination_wavelength: [
                destination.ior_550 as f32,
                destination.dispersion as f32,
                lambda as f32,
                f32::from_bits(destination.identity),
            ],
            random: [
                random as f32,
                f32::from_bits(incident.identity),
                f32::from_bits(incident.interface_priority as u32),
                f32::from_bits(destination.interface_priority as u32),
            ],
        }
    }

    fn decode(&self) -> (V3, V3, Medium, Medium, f64, f64) {
        (
            std::array::from_fn(|i| f64::from(self.wo_incident_ior[i])),
            std::array::from_fn(|i| f64::from(self.normal_incident_dispersion[i])),
            Medium {
                ior_550: f64::from(self.wo_incident_ior[3]),
                dispersion: f64::from(self.normal_incident_dispersion[3]),
                identity: self.random[1].to_bits(),
                interface_priority: self.random[2].to_bits() as i32,
            },
            Medium {
                ior_550: f64::from(self.destination_wavelength[0]),
                dispersion: f64::from(self.destination_wavelength[1]),
                identity: self.destination_wavelength[3].to_bits(),
                interface_priority: self.random[3].to_bits() as i32,
            },
            f64::from(self.destination_wavelength[2]),
            f64::from(self.random[0]),
        )
    }
}

fn add_both_events(
    cases: &mut Vec<GpuCase>,
    normal: V3,
    theta: f64,
    incident: Medium,
    destination: Medium,
    lambda: f64,
) {
    let wo = incident_direction(normal, theta);
    let probe = GpuCase::new(wo, normal, incident, destination, lambda, 0.0);
    let (wo, normal, n1, n2, lambda, _) = probe.decode();
    let r = independent_interface(wo, normal, n1, n2, lambda);
    let f = f64::from(r.reflectance);
    // Do not place a draw at the boundary of an empty event interval. At equal indices, for example,
    // f64 gives F = 0 while f32 amplitude arithmetic can give a tiny positive F; a draw of exactly
    // zero would then test roundoff at an impossible event, not the sampler's event probabilities.
    if r.tir || r.transmission <= 1e-7 {
        cases.push(GpuCase::new(wo, normal, n1, n2, lambda, 0.75));
        return;
    }
    if f > 1e-7 {
        cases.push(GpuCase::new(wo, normal, n1, n2, lambda, 0.2 * f));
    }
    if r.transmission > 1e-7 {
        cases.push(GpuCase::new(wo, normal, n1, n2, lambda, f + 0.8 * (1.0 - f)));
    }
}

fn test_source() -> String {
    format!(
        r#"
{LEAF}
struct PbrTestCase {{
  wo_incident_ior: vec4f,
  normal_incident_dispersion: vec4f,
  destination_wavelength: vec4f,
  random: vec4f,
}}
@group(0) @binding(0) var<storage,read> cases: array<PbrTestCase>;
@group(0) @binding(1) var<storage,read_write> results: array<vec4f>;
@compute @workgroup_size(64)
fn contract(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.x];
  let normal = input.normal_incident_dispersion.xyz;
  let tangent = normalize(cross(normal,select(vec3f(0,0,1),vec3f(1,0,0),abs(normal.z)>0.8)));
  let hit = PbrSurfaceHit(
    PbrFrame(tangent,cross(normal,tangent),normal),
    PbrMedium(input.wo_incident_ior[3],input.normal_incident_dispersion[3],
      bitcast<u32>(input.random[1]),bitcast<i32>(input.random[2])),
    PbrMedium(input.destination_wavelength[0],input.destination_wavelength[1],
      bitcast<u32>(input.destination_wavelength[3]),bitcast<i32>(input.random[3])),
    input.destination_wavelength[2]);
  let wo = input.wo_incident_ior.xyz;
  let split = pbr_dielectric_split(hit,wo);
  let s = pbr_dielectric_sample(hit,wo,PbrPathRng(vec2f(0.0),input.random[0]));
  let wi = normalize(wo + vec3f(0.13,-0.21,0.17));
  let n1 = pbr_dielectric_ior_at(hit.incident,hit.wavelength_nm);
  let n2 = pbr_dielectric_ior_at(hit.destination,hit.wavelength_nm);
  results[7u*id.x] = vec4f(split.reflected,split.reflectance);
  results[7u*id.x+1u] = vec4f(split.transmitted,split.transmission_weight);
  results[7u*id.x+2u] = vec4f(s.wi,s.weight.x);
  results[7u*id.x+3u] = vec4f(f32(s.event),f32(s.delta),s.pdf,s.event_probability);
  results[7u*id.x+4u] = vec4f(pbr_dielectric_evaluate(hit,wo,wi),pbr_dielectric_pdf(hit,wo,wi),n1,n2);
  results[7u*id.x+5u] = vec4f(f32(split.valid),f32(split.total_internal_reflection),split.reflection_weight,f32(s.medium_transition));
  results[7u*id.x+6u] = vec4f(
    bitcast<f32>(hit.incident.identity),bitcast<f32>(hit.incident.interface_priority),
    bitcast<f32>(hit.destination.identity),bitcast<f32>(hit.destination.interface_priority));
}}
"#
    )
}

fn read_results(gpu: &Gpu, buffer: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR dielectric download"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    commands.copy_buffer_to_buffer(buffer, 0, &download, 0, bytes);
    gpu.queue.submit(Some(commands.finish()));
    let (send, recv) = std::sync::mpsc::channel();
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap();
        });
    gpu.wait();
    recv.recv().unwrap().unwrap();
    let mapped = download.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}

#[test]
fn wgsl_matches_independent_snell_fresnel_energy_directions_and_delta_probabilities() {
    let gpu = Gpu::headless("vulkan").expect("PBR dielectric contract requires Vulkan");
    let normal = [0.0, 0.0, 1.0];
    let tilted = unit([0.25, -0.4, 0.88]);
    let critical = (1.0f64 / 1.52).asin();
    let mut cases = Vec::new();

    // Normal, oblique and grazing entry; exit below, at and above the critical angle.
    for theta in [0.0, 0.31, 0.9, 1.569] {
        add_both_events(&mut cases, normal, theta, air(), crown(), 550.0);
    }
    for theta in [
        0.0,
        0.31,
        critical - 1e-4,
        critical,
        critical + 1e-4,
        1.2,
        1.569,
    ] {
        add_both_events(&mut cases, normal, theta, crown(), air(), 550.0);
    }
    // Dissimilar media, a non-axis frame, equal indices, and spectral wavelength variation.
    let water = Medium {
        ior_550: 1.33,
        dispersion: 0.004,
        identity: 3,
        interface_priority: -17,
    };
    let dense = Medium {
        ior_550: 1.9,
        dispersion: 0.018,
        identity: 11,
        interface_priority: 29,
    };
    add_both_events(&mut cases, tilted, 0.63, water, dense, 470.0);
    add_both_events(&mut cases, tilted, 0.63, dense, water, 470.0);
    add_both_events(&mut cases, tilted, 0.63, crown(), crown(), 550.0);
    for lambda in [400.0, 450.0, 550.0, 630.0, 700.0] {
        add_both_events(&mut cases, tilted, 0.48, air(), crown(), lambda);
    }

    // Stratified draws verify actual event frequencies, in addition to each reported event mass.
    let probability_start = cases.len();
    let probability_draws = 256usize;
    let probability_wo = incident_direction(normal, 0.55);
    for i in 0..probability_draws {
        cases.push(GpuCase::new(
            probability_wo,
            normal,
            air(),
            crown(),
            550.0,
            (i as f64 + 0.5) / probability_draws as f64,
        ));
    }

    let source = test_source();
    validate(&source);
    let shader = gpu.shader("pure smooth dielectric contract", &source, false);
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PBR smooth dielectric contract"),
            layout: None,
            module: &shader,
            entry_point: Some("contract"),
            compilation_options: Default::default(),
            cache: None,
        });
    let case_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR dielectric cases"),
        size: (cases.len() * std::mem::size_of::<GpuCase>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&case_buffer, 0, bytemuck::cast_slice(&cases));
    let result_bytes = (cases.len() * 7 * 16) as u64;
    let result_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR dielectric results"),
        size: result_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: case_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(cases.len().div_ceil(64) as u32, 1, 1);
    }
    gpu.queue.submit(Some(commands.finish()));
    let results = read_results(&gpu, &result_buffer, result_bytes);

    let mut event_counts = [0u32; 4];
    let mut empirical_reflections = 0u32;
    let mut wavelength_ior = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let (wo, normal, incident, destination, lambda, random) = case.decode();
        let r = independent_interface(wo, normal, incident, destination, lambda);
        let out = &results[28 * i..28 * (i + 1)];
        assert_eq!(
            [out[24].to_bits(), out[25].to_bits(), out[26].to_bits(), out[27].to_bits()],
            [incident.identity, incident.interface_priority as u32, destination.identity, destination.interface_priority as u32],
            "exact medium metadata roundtrip, case {i}"
        );
        assert_eq!(out[20] as u32, 1, "valid split, case {i}");
        assert_eq!(out[21] as u32, u32::from(r.tir), "TIR flag, case {i}");

        close(
            f64::from(out[18]),
            r.incident_ior,
            2e-6,
            "incident spectral IOR",
        );
        close(
            f64::from(out[19]),
            r.destination_ior,
            2e-6,
            "destination spectral IOR",
        );
        let existing = existing_fresnel(dot(wo, normal), r.incident_ior, r.destination_ior);
        close(
            f64::from(r.reflectance),
            existing,
            3e-9,
            "independent versus existing Fresnel",
        );
        close(
            f64::from(out[3]),
            f64::from(r.reflectance),
            3e-5,
            "WGSL Fresnel",
        );
        close(f64::from(out[7]), r.transmission, 3e-5, "WGSL transmission");
        close(
            f64::from(out[3] + out[7]),
            1.0,
            3e-5,
            "R + T energy conservation",
        );
        close(
            f64::from(out[22]),
            f64::from(r.reflectance),
            3e-5,
            "reflection split weight",
        );

        for axis in 0..3 {
            close(
                f64::from(out[axis]),
                r.reflected[axis],
                4e-5,
                "reflection direction",
            );
            close(
                f64::from(out[4 + axis]),
                r.transmitted[axis],
                5e-5,
                "transmission direction",
            );
        }

        let event = out[12] as u32;
        event_counts[event as usize] += 1;
        assert_eq!(out[13] as u32, 1, "all valid smooth events are delta");
        assert_eq!(out[14], 0.0, "delta sample has no continuous PDF");
        assert_eq!(
            out[16], 0.0,
            "delta evaluate is not a fabricated finite value"
        );
        assert_eq!(
            out[17], 0.0,
            "delta direct PDF is not a fabricated finite value"
        );
        close(
            f64::from(out[11]),
            1.0,
            1e-6,
            "discrete sampling throughput",
        );

        let reflection = event == EVENT_REFLECTION || event == EVENT_TIR;
        let expected_event = if r.tir {
            EVENT_TIR
        } else if random < f64::from(r.reflectance) {
            EVENT_REFLECTION
        } else {
            EVENT_TRANSMISSION
        };
        assert_eq!(
            event, expected_event,
            "selected event, case {i}: random={random:?}, F={:?}, TIR={}, output={out:?}",
            r.reflectance, r.tir
        );
        let expected_transition = if event == EVENT_TRANSMISSION {
            MEDIUM_ENTER_DESTINATION
        } else {
            MEDIUM_STAY
        };
        assert_eq!(
            out[23] as u32, expected_transition,
            "medium transition is reported, not applied, case {i}"
        );
        let probability = if reflection {
            f64::from(r.reflectance)
        } else {
            r.transmission
        };
        close(
            f64::from(out[15]),
            probability,
            3e-5,
            "discrete event probability",
        );
        close(
            f64::from(out[11]) * probability,
            probability,
            3e-5,
            "probability-compensated throughput",
        );
        let expected_direction = if reflection {
            r.reflected
        } else {
            r.transmitted
        };
        for axis in 0..3 {
            close(
                f64::from(out[8 + axis]),
                expected_direction[axis],
                5e-5,
                "sampled direction",
            );
        }

        if i >= probability_start && reflection {
            empirical_reflections += 1;
        }
        if (400.0..=700.0).contains(&lambda)
            && incident.ior_550 == 1.0
            && destination.ior_550 == 1.52
        {
            wavelength_ior.push((lambda, f64::from(out[18]), f64::from(out[19])));
        }
    }

    assert_eq!(event_counts[EVENT_INVALID as usize], 0);
    assert!(event_counts[EVENT_REFLECTION as usize] > 0);
    assert!(event_counts[EVENT_TRANSMISSION as usize] > 0);
    assert!(event_counts[EVENT_TIR as usize] > 0);

    let probability_case = cases[probability_start].decode();
    let probability_reference = independent_interface(
        probability_case.0,
        probability_case.1,
        probability_case.2,
        probability_case.3,
        probability_case.4,
    );
    let expected_reflections = (0..probability_draws)
        .filter(|&i| {
            let draw = f64::from(((i as f64 + 0.5) / probability_draws as f64) as f32);
            draw < f64::from(probability_reference.reflectance)
        })
        .count() as u32;
    assert_eq!(
        empirical_reflections, expected_reflections,
        "stratified reflection frequency"
    );
    close(
        f64::from(empirical_reflections) / probability_draws as f64,
        f64::from(probability_reference.reflectance),
        0.012,
        "empirical Fresnel event probability",
    );

    for pair in wavelength_ior.windows(2) {
        if pair[0].0 != pair[1].0 {
            assert!(
                pair[0].2 >= pair[1].2 || pair[0].0 > pair[1].0,
                "blue wavelengths have higher destination IOR"
            );
        }
    }
}
