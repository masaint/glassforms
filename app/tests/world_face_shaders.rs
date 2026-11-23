use gf_core::material;
use gf_core::settings::Settings;
use glassforms::render::{tracer_source, LightFrom, March, Variant};

#[test]
fn cut_face_transport_compiles_for_joined_and_separate_bodies() {
    let march = March {
        step_scale: 0.85,
        steps_out: 180,
        steps_in: 64,
        layers: 2,
    };
    let settings = [
        Settings::default(),
        Settings {
            metal: 2.0,
            grating: 0.7,
            rough: 0.15,
            ..Settings::default()
        },
        Settings {
            sss: 0.8,
            sss_walk: 1.0,
            ..Settings::default()
        },
        Settings {
            sss: 1.0,
            ceramic: 1.0,
            ceramic_through: 0.1,
            ..Settings::default()
        },
    ];
    let refs: Vec<_> = settings.iter().collect();
    for bodies in [1, 2, 4] {
        for light in [LightFrom::Lights, LightFrom::Picture] {
            let mut mat = material::Variant::of_bodies(&refs);
            mat.bodies = bodies;
            let src = tracer_source(
                &march,
                Variant {
                    world_faces: true,
                    mat,
                    light,
                    ..Default::default()
                },
                false,
            );
            assert!(src.contains("wfPick(k,p)"));
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
