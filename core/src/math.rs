//! 3D vector arithmetic, scalar helpers, and row-major 3x3 rotations (as the web engine keeps them).

pub type Vec3 = [f64; 3];
pub type Mat3 = [f64; 9];

pub const DEG: f64 = std::f64::consts::PI / 180.0;
pub const TAU: f64 = 2.0 * std::f64::consts::PI;

pub const I3: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub fn add(a: Vec3, b: Vec3) -> Vec3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn sub(a: Vec3, b: Vec3) -> Vec3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn scale(a: Vec3, s: f64) -> Vec3 { [a[0] * s, a[1] * s, a[2] * s] }
pub fn dot(a: Vec3, b: Vec3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn len(a: Vec3) -> f64 { dot(a, a).sqrt() }
pub fn unit(a: Vec3) -> Vec3 {
    let l = len(a);
    if l > 1e-300 { scale(a, 1.0 / l) } else { [0.0, 1.0, 0.0] }
}
pub fn norm(a: Vec3) -> Vec3 {
    let l = len(a);
    if l > 1e-300 { scale(a, 1.0 / l) } else { a }
}
pub fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 { add(a, scale(sub(b, a), t)) }

/// smoothstep(0, 1, x), the one copy (env, stage and cast lights use it).
pub fn sstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

pub fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    o
}

pub fn axis_angle(x: f64, y: f64, z: f64, a: f64) -> Mat3 {
    let l = (x * x + y * y + z * z).sqrt();
    let l = if l == 0.0 { 1.0 } else { l };
    let (x, y, z) = (x / l, y / l, z / l);
    let (c, s) = (a.cos(), a.sin());
    let t = 1.0 - c;
    [
        t * x * x + c, t * x * y - s * z, t * x * z + s * y,
        t * x * y + s * z, t * y * y + c, t * y * z - s * x,
        t * x * z - s * y, t * y * z + s * x, t * z * z + c,
    ]
}

pub fn transpose(m: &Mat3) -> Mat3 {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

pub fn apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Rotation about x, then y, then z (degrees).
pub fn euler(rx: f64, ry: f64, rz: f64) -> Mat3 {
    mul(&axis_angle(0.0, 0.0, 1.0, rz * DEG), &mul(&axis_angle(0.0, 1.0, 0.0, ry * DEG), &axis_angle(1.0, 0.0, 0.0, rx * DEG)))
}

/// The three angles (degrees) back out of such a matrix.
pub fn euler_of(m: &Mat3) -> [f64; 3] {
    [m[7].atan2(m[8]) / DEG, -m[6].clamp(-1.0, 1.0).asin() / DEG, m[3].atan2(m[0]) / DEG]
}

/// The lights' three angles (radians): roll about the line of sight, then pitch, then yaw about the vertical.
pub fn yxz(yaw: f64, pitch: f64, roll: f64) -> Mat3 {
    mul(&axis_angle(0.0, 1.0, 0.0, yaw), &mul(&axis_angle(1.0, 0.0, 0.0, pitch), &axis_angle(0.0, 0.0, 1.0, roll)))
}

pub fn yxz_of(m: &Mat3) -> [f64; 3] {
    [m[2].atan2(m[8]), -m[5].clamp(-1.0, 1.0).asin(), m[3].atan2(m[4])]
}

/// Square a matrix up again after many small turns have been multiplied into it.
pub fn orthonormalise(m: &Mat3) -> Mat3 {
    let n = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let x = n([m[0], m[3], m[6]]);
    let y = [m[1], m[4], m[7]];
    let d = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let y = n([y[0] - d * x[0], y[1] - d * x[1], y[2] - d * x[2]]);
    let z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
    [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
}

/// As 32-bit floats, row-major, for a shader.
pub fn f32s(m: &Mat3) -> [f32; 9] {
    let mut o = [0.0f32; 9];
    for i in 0..9 {
        o[i] = m[i] as f32;
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_round_trip() {
        let m = euler(12.0, 25.0, -40.0);
        let e = euler_of(&m);
        assert!((e[0] - 12.0).abs() < 1e-9 && (e[1] - 25.0).abs() < 1e-9 && (e[2] + 40.0).abs() < 1e-9, "{e:?}");
    }

    #[test]
    fn yxz_round_trip() {
        let a = yxz_of(&yxz(0.7, -0.3, 1.1));
        assert!((a[0] - 0.7).abs() < 1e-12 && (a[1] + 0.3).abs() < 1e-12 && (a[2] - 1.1).abs() < 1e-12, "{a:?}");
    }

    #[test]
    fn orthonormalise_keeps_a_rotation() {
        let m = euler(33.0, -71.0, 140.0);
        let o = orthonormalise(&m);
        for i in 0..9 {
            assert!((m[i] - o[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn vec3_and_smoothstep_helpers() {
        let a = [1.0, 2.0, -2.0];
        let b = [2.0, -1.0, 0.0];
        assert_eq!(add(a, b), [3.0, 1.0, -2.0]);
        assert_eq!(sub(a, b), [-1.0, 3.0, -2.0]);
        assert_eq!(scale(a, 0.5), [0.5, 1.0, -1.0]);
        assert_eq!(dot(a, b), 0.0);
        assert_eq!(len(a), 3.0);
        assert!((len(unit(a)) - 1.0).abs() < 1e-12);
        assert_eq!(unit([0.0; 3]), [0.0, 1.0, 0.0]);
        assert_eq!(norm([0.0; 3]), [0.0; 3]);
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(lerp(a, b, 0.5), [1.5, 0.5, -1.0]);
        assert_eq!(smoothstep(0.0, 2.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 2.0, 1.0), 0.5);
        assert_eq!(smoothstep(0.0, 2.0, 3.0), 1.0);
    }
}
