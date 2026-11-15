//! Algebraic surfaces: the points where one polynomial in x, y, z equals zero, filled with glass on
//! one side. Some are bounded on their own; the rest are cut off by a ball. Mirrors `scenes/algebraic.js`.
use super::*;
use crate::field::SHELL_DEFAULT;

/// What a preset may add to the plain recipe: its sliders, a note, the way it is turned to start with.
#[derive(Default)]
struct Opt {
    params: Vec<ParamDef>,
    note: &'static str,
    rot: Option<[f64; 3]>,
}

// Every preset supplies f(q) (glass where f < 0), a scale from the volume to the surface's own units,
// and the radius of the ball that clips it (in the volume's units).
fn alg(id: &'static str, name: &'static str, eq: &[&'static str], f: &str, scale: f64, ball_r: f64, opt: Opt) -> Preset {
    // animate the shape's own first smooth number, or failing that swell and shrink it
    let animate = opt.params.iter().find(|d| d.step < 1.0).map_or("@puff", |d| d.id);
    Preset {
        eq: eq.to_vec(),
        note: opt.note,
        rot: opt.rot,
        animate: Some(vec![animate]),
        params: opt.params,
        ..Preset::new(id, name, Shape::Field {
            wgsl: format!("fn f(q: vec3f) -> f32 {{ {f} }}\nfn raw(p: vec3f) -> f32 {{ return f(p * {scale:.4}); }}"),
            clip: format!("length(p) - {ball_r:.3}"),
            shell: SHELL_DEFAULT.to_string(),
        })
    }
}

pub fn family() -> Family {
    Family {
        step_scale: 0.9,
        layers: 2,
        presets: vec![
            alg("tanglecube", "Tanglecube", &["x⁴ − 5x² + y⁴ − 5y² + z⁴ − 5z² + c = 0"],
                "let a = q*q; return dot(a, a) - 5.0*(a.x + a.y + a.z) + pr.tangle;", 2.55, 1.42,
                Opt { params: vec![ParamDef::labelled("tangle", "c", 11.8, 7.5, 14.5, 0.02)],
                      note: "Eight blobs at the corners of a cube. Lower c and they reach out and join.", ..Opt::default() }),

            alg("chmutov4", "Chmutov quartic", &["T₄(x) + T₄(y) + T₄(z) = c", "T₄(u) = 8u⁴ − 8u² + 1"],
                "let a = q*q; let t = 8.0*a*a - 8.0*a + 1.0; return t.x + t.y + t.z - pr.cheb;", 1.33, 1.42,
                Opt { params: vec![ParamDef::labelled("cheb", "c", 1.2, -2.2, 2.2, 0.01)],
                      note: "Built from Chebyshev polynomials, the ones that wiggle evenly between −1 and 1. Take c below zero and it breaks into pebbles.", ..Opt::default() }),

            alg("chmutov6", "Chmutov sextic", &["T₆(x) + T₆(y) + T₆(z) = c", "T₆(u) = 32u⁶ − 48u⁴ + 18u² − 1"],
                "let a = q*q; let t = 32.0*a*a*a - 48.0*a*a + 18.0*a - 1.0; return t.x + t.y + t.z - pr.cheb;", 1.36, 1.42,
                Opt { params: vec![ParamDef::labelled("cheb", "c", 1.5, -2.2, 2.2, 0.01)], ..Opt::default() }),

            alg("decocube", "Decocube", &["((x² + y² − a²)² + (z² − 1)²) ·", "((y² + z² − a²)² + (x² − 1)²) ·", "((z² + x² − a²)² + (y² − 1)²) = b"],
                "let a = q*q; let c2: f32 = 0.64; return ((a.x + a.y - c2)*(a.x + a.y - c2) + (a.z - 1.0)*(a.z - 1.0)) * ((a.y + a.z - c2)*(a.y + a.z - c2) + (a.x - 1.0)*(a.x - 1.0)) * ((a.z + a.x - c2)*(a.z + a.x - c2) + (a.y - 1.0)*(a.y - 1.0)) - pr.deco;", 1.2, 1.42,
                Opt { params: vec![ParamDef::labelled("deco", "b", 0.02, 0.003, 0.12, 0.001)],
                      note: "A ring on every face of a cube, fused where they meet.", ..Opt::default() }),

            alg("orthocircles", "Orthocircles", &["((x² + y² − 1)² + z²) ·", "((y² + z² − 1)² + x²) ·", "((z² + x² − 1)² + y²) = a²(1 + b·r²)"],
                "let a = q*q; let r2 = a.x + a.y + a.z; return ((a.x + a.y - 1.0)*(a.x + a.y - 1.0) + a.z) * ((a.y + a.z - 1.0)*(a.y + a.z - 1.0) + a.x) * ((a.z + a.x - 1.0)*(a.z + a.x - 1.0) + a.y) - pr.ortho*pr.ortho*(1.0 + 3.0*r2);", 1.0, 1.42,
                Opt { params: vec![ParamDef::labelled("ortho", "a", 0.06, 0.02, 0.3, 0.002)],
                      note: "Three circles at right angles, thickened until they merge.", ..Opt::default() }),

            alg("pretzel", "Double torus", &["(x²(1 − x²) − y²)² + z² = c"],
                "let w = q.x*q.x*(1.0 - q.x*q.x) - q.y*q.y; return w*w + q.z*q.z - pr.holes;", 0.9, 1.45,
                Opt { params: vec![ParamDef::labelled("holes", "c", 0.03, 0.004, 0.08, 0.001)], rot: Some([18.0, 20.0, 0.0]),
                      note: "A surface with two holes: a doughnut for two." }),

            alg("heart", "Heart", &["(x² + 9⁄4·y² + z² − 1)³ − x²z³ − 9⁄80·y²z³ = 0"],
                // The same surface written as  a = z·(x² + 9/80·y²)^(1/3)  with a = x² + 9/4·y² + z² − 1. Cubed, that is the
                // equation shown. In this form the function crosses zero cleanly; the cubed form flattens out along the
                // heart's widest line, which left a seam in the glass.
                "let h = q.xzy; let a = h.x*h.x + 2.25*h.y*h.y + h.z*h.z - 1.0; return a - h.z * pow(h.x*h.x + 0.1125*h.y*h.y + 1e-9, 1.0/3.0);", 1.02, 1.42,
                Opt { rot: Some([8.0, 25.0, 0.0]), note: "Taubin’s heart surface: a sixth-degree polynomial.", ..Opt::default() }),

            alg("kummer", "Kummer surface", &["(x² + y² + z² − μ²)² = λ·p·q·r·s", "p, q, r, s: the four faces of a tetrahedron"],
                "let m2 = pr.mu; let lam = (3.0*m2 - 1.0) / (3.0 - m2); const R2: f32 = 1.41421356; let p1 = 1.0 - q.z - R2*q.x; let p2 = 1.0 - q.z + R2*q.x; let p3 = 1.0 + q.z + R2*q.y; let p4 = 1.0 + q.z - R2*q.y; let a = dot(q, q) - m2; return a*a - lam*p1*p2*p3*p4;", 2.3, 1.3,
                Opt { params: vec![ParamDef::labelled("mu", "μ²", 1.3, 1.02, 2.6, 0.01)],
                      note: "A quartic with sixteen cone points, the most a quartic can have.", ..Opt::default() }),

            alg("barth", "Barth sextic", &["4(φ²x² − y²)(φ²y² − z²)(φ²z² − x²)", "= (1 + 2φ)(x² + y² + z² − 1)²"],
                "const ph: f32 = 1.618034; let a = q*q; return pr.side * (4.0*(ph*ph*a.x - a.y)*(ph*ph*a.y - a.z)*(ph*ph*a.z - a.x) - (1.0 + 2.0*ph)*(a.x + a.y + a.z - 1.0)*(a.x + a.y + a.z - 1.0));", 1.9, 1.3,
                Opt { params: vec![ParamDef::new("side", -1.0, -1.0, 1.0, 2.0)],
                      note: "Sixty-five cone points, the most a sixth-degree surface can have, arranged with the symmetry of an icosahedron. φ is the golden ratio.", ..Opt::default() }),

            alg("clebsch", "Clebsch cubic", &["81(x³ + y³ + z³) − 189(x²y + x²z + y²x + y²z + z²x + z²y)", "+ 54xyz + 126(xy + yz + zx) − 9(x² + y² + z²) − 9(x + y + z) + 1 = 0"],
                "let x = q.x; let y = q.y; let z = q.z; return 81.0*(x*x*x + y*y*y + z*z*z) - 189.0*(x*x*y + x*x*z + y*y*x + y*y*z + z*z*x + z*z*y) + 54.0*x*y*z + 126.0*(x*y + y*z + z*x) - 9.0*(x*x + y*y + z*z) - 9.0*(x + y + z) + 1.0;", 1.1, 1.3,
                Opt { note: "A cubic surface containing 27 straight lines, all of them real.", ..Opt::default() }),
        ],
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.34, 0.16, 0.07])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(3.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([24.0, 34.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        lights: vec![
            Light { az:   38.0, el:  18.0, w: 14.0, h: 32.0, rgb: [5.0, 4.9, 4.7],   soft: 0.7 },
            Light { az:  -55.0, el:  -8.0, w: 10.0, h: 42.0, rgb: [0.5, 1.4, 4.0],   soft: 0.7 },
            Light { az: -140.0, el:   5.0, w:  8.0, h: 42.0, rgb: [5.0, 1.6, 0.25],  soft: 0.7 },
            Light { az:  150.0, el:  28.0, w: 24.0, h: 16.0, rgb: [3.2, 3.1, 3.0],   soft: 0.8 },
            Light { az:    0.0, el:  85.0, w: 34.0, h: 34.0, rgb: [0.8, 0.9, 1.1],   soft: 0.9 },
            Light { az:    0.0, el: -80.0, w: 30.0, h: 30.0, rgb: [0.5, 0.22, 0.05], soft: 0.9 },
        ],
        ..Family::new("algebraic", "Algebraic surfaces", Kind::Field)
    }
}
