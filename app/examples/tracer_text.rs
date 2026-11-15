//! Prints the tracer's source for a material given as settings (name=value ...), for reading.
use gf_core::settings::Settings;
use glassforms::render::{tracer_source, March, Variant};

fn main() {
    let mut p = Settings::default();
    for a in std::env::args().skip(1) {
        if let Some((k, v)) = a.split_once('=') { p.set(k, v.parse().unwrap()); }
    }
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    print!("{}", tracer_source(&m, Variant { mat: gf_core::material::Variant::of(&p), ..Variant::default() }, false));
}
