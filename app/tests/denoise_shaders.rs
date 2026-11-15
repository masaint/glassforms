//! The tracers built for denoising parse and pass naga's validator, and nothing else changes. Needs no GPU.

use gf_core::material;
use gf_core::settings::Settings;
use glassforms::render::{tracer_source, tracer_source_lens, LightFrom, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

/// The materials drawn by chance, as named materials and settings choose them.
fn chance() -> Vec<(&'static str, material::Variant)> {
    let of = |f: &dyn Fn(&mut Settings)| {
        let mut p = Settings::default();
        f(&mut p);
        material::Variant::of(&p)
    };
    vec![
        ("true milk", of(&|p| { p.sss = 1.0; p.sss_walk = 1.0; p.disp = 0.0; })),
        ("true jade (rough)", of(&|p| { p.sss = 1.0; p.sss_walk = 1.0; p.rough = 0.1; p.disp = 0.0; })),
        ("true milk with a coat and a film", of(&|p| { p.sss = 1.0; p.sss_walk = 1.0; p.coat = 0.5; p.film = 400.0; })),
        ("rough by sampling alone", of(&|p| { p.rough = 0.6; p.rough_ref = 1.0; })),
        ("clear glass", material::Variant::default()),
    ]
}

#[test]
fn denoised_tracers_are_valid() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for (name, mat) in chance() {
        assert!(glassforms::denoise::wanted(&mat, false, 2.0), "{name}: not denoised at denoise=2");
        for light in [LightFrom::Lights, LightFrom::Picture] {
            for backdrop in [false, true] {
                let v = Variant { light, backdrop, mat, denoise: true, ..Default::default() };
                let src = tracer_source(&m, v, false);
                assert!(src.contains(glassforms::denoise::MARK), "{name}: not split");
                check(&format!("{name} {v:?}"), &src);
                check(&format!("{name} {v:?} through the lens"), &tracer_source_lens(&m, v, false));
                // counting lookups: as if it were not denoised
                assert_eq!(tracer_source(&m, v, true), tracer_source(&m, Variant { denoise: false, ..v }, true), "{name}: counting tracer changed");
            }
        }
    }
}

/// What is denoised: by default what is drawn by chance; clear glass only at `denoise=2`; never with the
/// setting off, under clay, or the materials that are clean already (the milky estimate, frost drawn
/// the live way, metals).
#[test]
fn what_is_denoised() {
    for (name, mat) in chance() {
        assert_eq!(glassforms::denoise::wanted(&mat, false, 1.0), name != "clear glass", "{name} at denoise=1");
        assert!(!glassforms::denoise::wanted(&mat, false, 0.0), "{name} at denoise=0");
        assert!(!glassforms::denoise::wanted(&mat, true, 2.0), "{name} under clay");
    }
    let clean: [&dyn Fn(&mut Settings); 3] = [&|p| p.sss = 1.0, &|p| p.rough = 0.6, &|p| { p.metal = 3.0; p.rough = 0.3; }];
    for f in clean {
        let mut p = Settings::default();
        f(&mut p);
        assert!(!glassforms::denoise::wanted(&material::Variant::of(&p), false, 2.0), "{:?}", material::Variant::of(&p));
    }
}
