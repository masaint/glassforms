//! Triply periodic minimal surfaces, in glass, cut from a ball or a rounded cube. Mirrors
//! `scenes/surfaces.js`.
//!
//! Each is the zero set of a short trigonometric expression g(x, y, z) that repeats in all three
//! directions. "Solid" fills one side of the surface (a lattice of struts); otherwise the glass is a
//! thin wall following the surface itself.
use super::*;

// Every preset supplies g(q). Solid: glass where g > level. Otherwise a wall about the surface g = level.
const TAIL: &str = "
fn raw(p: vec3f) -> f32 {
  let v = g(p * 3.14159265 * pr.cells) - pr.level;        // \"cells\" repeats across the width of the piece
  return select(v, -v, pr.solid > 0.5);
}";
const CUBE: &str = "length(max(abs(p) - 0.66, vec3f(0.0))) + min(max(abs(p.x) - 0.66, max(abs(p.y) - 0.66, abs(p.z) - 0.66)), 0.0) - 0.14";

/// cells, solid (1 = fill one side, 0 = a wall about the surface), cube (1 = cut from a cube), note
struct Opt {
    cells: f64,
    solid: f64,
    cube: f64,
    note: &'static str,
}
impl Default for Opt {
    fn default() -> Self { Self { cells: 1.5, solid: 1.0, cube: 0.0, note: "" } }
}

fn tpms(id: &'static str, name: &'static str, eq: &[&'static str], g: &str, level: f64, opt: Opt) -> Preset {
    Preset {
        eq: eq.to_vec(),
        note: opt.note,
        animate: Some(vec![if opt.solid != 0.0 { "level" } else { "cells" }]),      // a lattice thickens and thins; a wall changes scale
        params: vec![
            ParamDef::new("cells", opt.cells, 0.6, 5.0, 0.05),
            ParamDef::new("level", level, -1.2, 1.2, 0.01),
            ParamDef::new("solid", opt.solid, 0.0, 1.0, 1.0),
            ParamDef::new("wall", 0.045, 0.012, 0.14, 0.002),
            ParamDef::labelled("box", "cube", opt.cube, 0.0, 1.0, 1.0),
        ],
        ..Preset::new(id, name, Shape::Field {
            wgsl: format!("fn g(q: vec3f) -> f32 {{ {g} }}{TAIL}"),
            clip: format!("mix(length(p) - 1.22, {CUBE}, pr.box)"),
            shell: "select(pr.wall, 0.0, pr.solid > 0.5)".to_string(),
        })
    }
}

pub fn family() -> Family {
    Family {
        step_scale: 0.9,
        layers: 2,
        presets: vec![
            tpms("gyroid", "Gyroid", &["sin x cos y + sin y cos z + sin z cos x = 0"],
                "return dot(sin(q), cos(q.yzx));", 0.0,
                Opt { solid: 0.0, note: "Found by Alan Schoen in 1970. It splits space into two identical intertwined labyrinths and contains no straight lines.", ..Opt::default() }),
            tpms("gyroidnet", "Gyroid lattice", &["sin x cos y + sin y cos z + sin z cos x > c"],
                "return dot(sin(q), cos(q.yzx));", 0.55,
                Opt { note: "One of the gyroid’s two labyrinths, filled in.", ..Opt::default() }),
            tpms("schwarzp", "Schwarz P", &["cos x + cos y + cos z = 0"],
                "return cos(q.x) + cos(q.y) + cos(q.z);", 0.25,
                Opt { note: "The \"primitive\" surface: a cubic lattice of chambers joined by necks.", ..Opt::default() }),
            tpms("schwarzd", "Schwarz D", &["sin x sin y sin z + sin x cos y cos z", "+ cos x sin y cos z + cos x cos y sin z = 0"],
                "let s = sin(q); let c = cos(q); return s.x*s.y*s.z + s.x*c.y*c.z + c.x*s.y*c.z + c.x*c.y*s.z;", 0.0,
                Opt { solid: 0.0, cells: 1.2, note: "The \"diamond\" surface: each of its two labyrinths is a diamond lattice.", ..Opt::default() }),
            tpms("neovius", "Neovius", &["3(cos x + cos y + cos z) + 4 cos x cos y cos z = 0"],
                "let c = cos(q); return (3.0*(c.x + c.y + c.z) + 4.0*c.x*c.y*c.z) / 3.0;", 0.0,
                Opt { cells: 1.0, solid: 0.0, cube: 1.0, ..Opt::default() }),
            tpms("iwp", "I-WP", &["2(cos x cos y + cos y cos z + cos z cos x)", "− (cos 2x + cos 2y + cos 2z) = 0"],
                "let c = cos(q); let d = cos(2.0*q); return (2.0*(c.x*c.y + c.y*c.z + c.z*c.x) - (d.x + d.y + d.z)) / 2.5;", 0.3,
                Opt { cells: 1.2, ..Opt::default() }),
            tpms("lidinoid", "Lidinoid", &["½(sin 2x cos y sin z + sin 2y cos z sin x + sin 2z cos x sin y)", "− ½(cos 2x cos 2y + cos 2y cos 2z + cos 2z cos 2x) + 0.15 = 0"],
                "let s = sin(q); let c = cos(q); let s2 = sin(2.0*q); let c2 = cos(2.0*q); return 0.5*(s2.x*c.y*s.z + s2.y*c.z*s.x + s2.z*c.x*s.y) - 0.5*(c2.x*c2.y + c2.y*c2.z + c2.z*c2.x) + 0.15;", 0.3,
                Opt { cells: 0.9, ..Opt::default() }),
            tpms("fischerkoch", "Fischer–Koch S", &["cos 2x sin y cos z + cos x cos 2y sin z", "+ sin x cos y cos 2z = 0"],
                "let s = sin(q); let c = cos(q); let c2 = cos(2.0*q); return c2.x*s.y*c.z + c.x*c2.y*s.z + s.x*c.y*c2.z;", 0.3,
                Opt { cells: 0.9, ..Opt::default() }),
        ],
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.30, 0.14, 0.06])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(3.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([22.0, 32.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        // Larger, softer lights than the tubes use: broad faces of glass want broad reflections.
        lights: surface_lights(),
        ..Family::new("surfaces", "Minimal surfaces", Kind::Field)
    }
}
