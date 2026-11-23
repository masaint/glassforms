//! Directional hemispherical reflectance of a Smith GGX microsurface.
//! The two channels multiply the normal-incidence reflectance and grazing contribution of Schlick Fresnel.
//! They are independent of lights and material colour, so every conductor shares this integration.
use std::sync::OnceLock;

pub const ANGLES: usize = 16;
const ROUGHNESS: usize = 33;
type Angular = [[f32; 2]; ANGLES];

fn sample(alpha: f64, nv: f64, disk: &[[f64; 2]]) -> [f32; 2] {
    if alpha == 0.0 {
        let f = (1.0 - nv).powi(5);
        return [(1.0 - f) as f32, f as f32];
    }
    let vx = (1.0 - nv * nv).sqrt();
    let norm = (alpha * vx).hypot(nv);
    let (sx, sz) = (alpha * vx / norm, nv / norm);
    let mut sum = [0.0; 2];
    for &[x, y] in disk {
        // Sample the GGX visible-normal distribution, Heitz (2018), in the incident ray's plane.
        let s = 0.5 * (1.0 + sz);
        let y = (1.0 - s) * (1.0 - x*x).max(0.0).sqrt() + s*y;
        let z = (1.0 - x*x - y*y).max(0.0).sqrt();
        let (hx, hy, hz) = (alpha*(sx*z - sz*y), alpha*x, (sx*y + sz*z).max(0.0));
        let h = (hx*hx + hy*hy + hz*hz).sqrt();
        let vh = (vx*hx + nv*hz) / h;
        let nl = 2.0 * vh * hz / h - nv;
        if nl <= 0.0 { continue; }
        // Divide correlated Smith G2 by the incident G1 already in the visible-normal PDF.
        let lv = 0.5 * (norm / nv - 1.0);
        let ll = 0.5 * ((alpha*alpha + (1.0-alpha*alpha)*nl*nl).sqrt() / nl - 1.0);
        let visibility = (1.0 + lv) / (1.0 + lv + ll);
        let fresnel = (1.0 - vh.clamp(0.0, 1.0)).powi(5);
        sum[0] += visibility * (1.0 - fresnel);
        sum[1] += visibility * fresnel;
    }
    sum.map(|v| (v / disk.len() as f64) as f32)
}

/// Pack one roughness row into the material uniform. Cosines use a quadratic grid for extra grazing samples.
pub fn energy_row(roughness: f64) -> Angular {
    static GRID: OnceLock<[Angular; ROUGHNESS]> = OnceLock::new();
    let grid = GRID.get_or_init(|| {
        let samples = 1024u32;
        let disk: Vec<_> = (0..samples).map(|i| {
            let r = ((f64::from(i) + 0.5) / f64::from(samples)).sqrt();
            let phi = std::f64::consts::TAU * f64::from(i.reverse_bits()) / 4294967296.0;
            let (s, c) = phi.sin_cos();
            [r*c, r*s]
        }).collect();
        std::array::from_fn(|r| {
            let alpha = (r as f64 / (ROUGHNESS - 1) as f64).powi(2);
            std::array::from_fn(|i| sample(alpha, (i as f64 / (ANGLES - 1) as f64).powi(2).max(1e-4), &disk))
        })
    });
    let x = roughness.clamp(0.0, 1.0) * (ROUGHNESS - 1) as f64;
    let lo = x.floor() as usize;
    let hi = (lo + 1).min(ROUGHNESS - 1);
    let f = (x - lo as f64) as f32;
    std::array::from_fn(|i| std::array::from_fn(|j| grid[lo][i][j] + f * (grid[hi][i][j] - grid[lo][i][j])))
}

type BrushRow = [Angular; 3];

/// Directional GGX energy for the two authored slope widths. Choose the incident azimuth
/// across the grooves, between them, or along them. The shader interpolates in cosine squared.
pub fn brushed_energy(roughness: f64, brush: f64) -> BrushRow {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    type Cached = ((u64, u64), BrushRow);
    static ROWS: OnceLock<Mutex<VecDeque<Cached>>> = OnceLock::new();
    static DISK: OnceLock<Vec<[f64; 3]>> = OnceLock::new();
    let cache = ROWS.get_or_init(Mutex::default);
    let key = (roughness.to_bits(), brush.to_bits());
    if let Some((_, row)) = cache.lock().unwrap().iter().find(|item| item.0 == key) {
        return *row;
    }
    // The same cap sampler and correlated Smith masking as the sampled shader. Each
    // cap direction becomes a visible facet after adding the stretched incident ray.
    let samples = DISK.get_or_init(|| {
        (0..1024_u32)
            .map(|i| {
                let radial = (f64::from(i) + 0.5) / 1024.0;
                let phi = std::f64::consts::TAU * f64::from(i.reverse_bits()) / 4294967296.0;
                let (sin, cos) = phi.sin_cos();
                [radial, cos, sin]
            })
            .collect()
    });
    let alpha = roughness * roughness;
    let aspect = (1.0 - 0.60 * brush).sqrt();
    // brushStreak has mean 0.6 plus two zero-mean value-noise bands. Its RMS
    // lies near 0.61 with the fine band filtered and 0.63 with both bands present.
    let ax = (alpha / aspect).hypot(0.15 * 0.62 * brush);
    let ay = (alpha * aspect).hypot(0.006 * brush);
    let row = std::array::from_fn(|az| {
        let phi = std::f64::consts::FRAC_PI_4 * az as f64;
        let (sin, cos) = phi.sin_cos();
        std::array::from_fn(|i| {
            let nv = (i as f64 / (ANGLES - 1) as f64).powi(2).max(1e-4);
            let tangent = (1.0 - nv * nv).sqrt();
            let (vx, vy) = (cos * tangent, sin * tangent);
            let gv = (ax * vx).hypot(ay * vy).hypot(nv);
            let (sx, sy, sz) = (ax * vx / gv, ay * vy / gv, nv / gv);
            let mut ab = [0.0, 0.0];
            for &[u, c, s] in samples {
                let z = -sz + (1.0 + sz) * u;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let (x, y, z) = (ax * (sx + r * c), ay * (sy + r * s), (sz + z).max(1e-7));
                let inv = 1.0 / (x * x + y * y + z * z).sqrt();
                let vh = (vx * x + vy * y + nv * z) * inv;
                let k = 2.0 * vh * inv;
                let (lx, ly, nl) = (k * x - vx, k * y - vy, k * z - nv);
                if nl <= 0.0 {
                    continue;
                }
                let gl = (ax * lx).hypot(ay * ly).hypot(nl);
                let visibility = nl * (nv + gv) / (nl * gv + nv * gl);
                let fresnel = (1.0 - vh.clamp(0.0, 1.0)).powi(5);
                ab[0] += visibility * (1.0 - fresnel);
                ab[1] += visibility * fresnel;
            }
            ab.map(|v| (v / samples.len() as f64) as f32)
        })
    });
    let mut cache = cache.lock().unwrap();
    if cache.len() == 32 {
        cache.pop_front();
    }
    cache.push_back((key, row));
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    // The compensation in conductorEnergyFromAb (trace_material.wgsl:820-828) is not tested here:
    // with fAvg = f0 + (1-f0)/21, single = f0*a + b and m = 1-a-b, single + multiple = single/(1-m*fAvg).
    // For f0 = 1 that is (a+b)/(a+b) = 1 for any table, and for f0 <= 1 it lies in [single, 1]
    // whenever a, b >= 0 and a+b <= 1, which `table_is_a_bounded_split` already checks.

    /// A deterministic product-midpoint grid over the slope distribution of GGX, independent of the
    /// table's visible-normal Monte Carlo: u = t^2/(1+t^2) with t the stretched slope radius makes
    /// D(h) cos(theta_h) dω_h = du dφ / 2π exactly, so the integrand left is G2 F (v.h) / (n.v n.h).
    /// Same model as the table: anisotropic GGX D, height-correlated Smith G2, Schlick split
    /// F = f0 (1-(1-v.h)^5) + (1-v.h)^5. Returns the f0 = 1 term (a) and the f0 = 0 term (b).
    fn quadrature(ax: f64, ay: f64, v: [f64; 3]) -> [f64; 2] {
        const NU: usize = 1024;
        const NPHI: usize = 1024;
        let [vx, vy, nv] = v;
        let gv = (ax * vx).hypot(ay * vy).hypot(nv);
        let mut ab = [0.0; 2];
        for iu in 0..NU {
            let u = (iu as f64 + 0.5) / NU as f64;
            let t = (u / (1.0 - u)).sqrt();
            for ip in 0..NPHI {
                let phi = std::f64::consts::TAU * (ip as f64 + 0.5) / NPHI as f64;
                let (s, c) = phi.sin_cos();
                let (px, py) = (ax * t * c, ay * t * s);
                let hz = 1.0 / (1.0 + px * px + py * py).sqrt();
                let (hx, hy) = (-px * hz, -py * hz);
                let vh = vx * hx + vy * hy + nv * hz;
                if vh <= 0.0 { continue; }
                let (lx, ly, nl) = (2.0 * vh * hx - vx, 2.0 * vh * hy - vy, 2.0 * vh * hz - nv);
                if nl <= 0.0 { continue; }
                let gl = (ax * lx).hypot(ay * ly).hypot(nl);
                let g2 = 2.0 * nv * nl / (gv * nl + gl * nv);
                let w = g2 * vh / (nv * hz);
                let f = (1.0 - vh).powi(5);
                ab[0] += w * (1.0 - f);
                ab[1] += w * f;
            }
        }
        ab.map(|x| x / (NU * NPHI) as f64)
    }

    /// The shader's read (trace_material.wgsl:799-801): node i sits at cosine (i/15)^2, lerp in sqrt(cosine).
    fn shader_ab(row: &Angular, c: f64) -> [f64; 2] {
        let x = 15.0 * c.clamp(0.0, 1.0).sqrt();
        let i = x as usize;
        let j = (i + 1).min(ANGLES - 1);
        let f = x.fract();
        [0, 1].map(|k| row[i][k] as f64 + f * (row[j][k] as f64 - row[i][k] as f64))
    }

    fn node_row(r: usize) -> Angular { energy_row(r as f64 / (ROUGHNESS - 1) as f64) }

    /// Table vs quadrature, absolute per term. The table is 1024-point QMC stored as f32, read
    /// through two lerps (roughness rows 1/32 apart, cosine nodes in sqrt). Measured worst: 0.0078
    /// isotropic (cos 0.05), 0.0061 brushed. 0.01 is ~1% of energy, below what the compensation can show.
    const QUADRATURE_TOL: f64 = 0.01;
    /// Rounding slack for a+b <= 1 and the roughness ordering: table entries are f32 sums.
    const SUM_EPS: f32 = 1e-5;
    /// Below cosine ~0.13 the height-correlated Smith albedo itself rises with roughness once
    /// alpha*tan(theta) passes ~1 (the quadrature shows rises up to 0.027 at node 1, the table 0.029).
    /// Nodes from here on are strictly non-increasing; below it only the size of a rise is bounded.
    const FIRST_MONOTONE_NODE: usize = 6;
    const GRAZING_RISE: f32 = 0.035;

    #[test]
    fn table_matches_independent_quadrature() {
        let mut worst = 0.0f64;
        for rough in [0.1, 0.3, 0.5, 0.77, 1.0] {
            let row = energy_row(rough);
            let alpha = rough * rough;
            for c in [0.05f64, 0.3, 0.6, 1.0] {
                let reference = quadrature(alpha, alpha, [(1.0 - c * c).sqrt(), 0.0, c]);
                let table = shader_ab(&row, c);
                for k in 0..2 {
                    let e = (table[k] - reference[k]).abs();
                    worst = worst.max(e);
                    assert!(e < QUADRATURE_TOL, "rough {rough} cos {c} term {k}: table {} quadrature {}", table[k], reference[k]);
                }
            }
        }
        eprintln!("energy_row worst |table - quadrature| = {worst:.5}");
    }

    #[test]
    fn quadrature_reproduces_known_limit() {
        // alpha = 1 at normal incidence: the height-correlated Smith albedo is 1 - ln 2.
        let [a, b] = quadrature(1.0, 1.0, [0.0, 0.0, 1.0]);
        assert!((a + b - (1.0 - std::f64::consts::LN_2)).abs() < 1e-3, "{}", a + b);
    }

    #[test]
    fn table_is_a_bounded_split() {
        for r in 0..ROUGHNESS {
            for (i, [a, b]) in node_row(r).into_iter().enumerate() {
                assert!(a >= 0.0 && b >= 0.0 && a + b <= 1.0 + SUM_EPS, "row {r} node {i}: {a} {b}");
            }
        }
    }

    #[test]
    fn albedo_does_not_rise_with_roughness() {
        let mut worst = 0.0f32;
        for r in 0..ROUGHNESS - 1 {
            let (lo, hi) = (node_row(r), node_row(r + 1));
            for i in 0..ANGLES {
                let rise = (hi[i][0] + hi[i][1]) - (lo[i][0] + lo[i][1]);
                let limit = if i >= FIRST_MONOTONE_NODE { SUM_EPS } else { GRAZING_RISE };
                if i >= FIRST_MONOTONE_NODE { worst = worst.max(rise); }
                assert!(rise <= limit, "row {r}->{} node {i} rises by {rise}", r + 1);
            }
        }
        eprintln!("energy_row largest rise with roughness, cos >= 0.16: {worst:.6}");
    }

    #[test]
    fn smooth_limits() {
        // Exactly smooth: a mirror loses nothing and Schlick splits by (1-cos)^5.
        let smooth = energy_row(0.0);
        for [a, b] in smooth { assert!((a + b - 1.0).abs() <= SUM_EPS); }
        assert_eq!(smooth[ANGLES - 1], [1.0, 0.0]);
        // The first rough row converges to it: alpha ~ 0.001 only shadows the two grazing nodes
        // (cos 1e-4 and 0.0044 hold 0.980 and 0.973; quadrature gives 0.968 and 0.970).
        let near = node_row(1);
        for (i, [a, b]) in near.into_iter().enumerate().skip(2) {
            assert!(a + b > 0.99, "node {i}: {}", a + b);
        }
        let [a, b] = near[ANGLES - 1];
        assert!(a > 0.999 && b < 1e-3, "{a} {b}");
    }

    #[test]
    fn brushed_matches_quadrature_and_isotropic_table() {
        let mut worst = 0.0f64;
        for (rough, brush) in [(0.3, 0.5), (0.3, 1.0), (0.6, 1.0), (0.8, 0.3)] {
            let rows = brushed_energy(rough, brush);
            let alpha = rough * rough;
            let aspect = (1.0 - 0.60 * brush).sqrt();
            let ax = (alpha / aspect).hypot(0.15 * 0.62 * brush);
            let ay = (alpha * aspect).hypot(0.006 * brush);
            for (az, row) in rows.iter().enumerate() {
                let (s, cphi) = (std::f64::consts::FRAC_PI_4 * az as f64).sin_cos();
                for c in [0.05f64, 0.3, 0.6, 1.0] {
                    let t = (1.0 - c * c).sqrt();
                    let reference = quadrature(ax, ay, [cphi * t, s * t, c]);
                    let table = shader_ab(row, c);
                    for k in 0..2 {
                        let e = (table[k] - reference[k]).abs();
                        worst = worst.max(e);
                        assert!(e < QUADRATURE_TOL, "rough {rough} brush {brush} az {az} cos {c} term {k}: {} vs {}", table[k], reference[k]);
                    }
                }
                for (i, [a, b]) in row.iter().enumerate() {
                    assert!(*a >= 0.0 && *b >= 0.0 && a + b <= 1.0 + SUM_EPS, "az {az} node {i}: {a} {b}");
                }
            }
        }
        eprintln!("brushed_energy worst |table - quadrature| = {worst:.5}");
        // Unbrushed, every azimuth is the isotropic table at a grid roughness (a different sampler).
        let iso = node_row(16);
        for row in brushed_energy(0.5, 0.0) {
            for i in 0..ANGLES {
                for k in 0..2 {
                    assert!((row[i][k] - iso[i][k]).abs() < QUADRATURE_TOL as f32, "node {i} term {k}");
                }
            }
        }
    }
}
