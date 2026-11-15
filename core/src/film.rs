//! Thin films: how a film drains. The tracer's `filmNm` (app/src/shaders/film.wgsl) is given the numbers
//! worked out here, and the same thickness is written out in plain arithmetic so that it can be tested
//! without a GPU. `docs/materials.md` says where the model comes from and what it leaves out.
//!
//! One slider, `filmDrain`, runs the whole life of a film of liquid: at 0 it is fresh and even; then
//! liquid runs down, so it is thinner the higher it is and thinnest of all on every summit; all the
//! while the whole film thins; at 1 nothing is left but black film.

use crate::settings::Settings;

/// Below this thickness (nm) a free film of liquid is black film or nearly so, and droplets of thicker
/// film are left standing in it. (A film this thin mirrors a sixth of what a thick one does at its
/// brightest; "common black film" is about 30 nm with ordinary soap.)
pub const BLACK_BELOW: f64 = 28.0;
/// The thickness a free film ends at: the thinnest black film (a "Newton black film" is 4 to 6 nm).
pub const BLACK_NM: f64 = 6.0;

/// How far a film has drained, as the tracer is told it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drain {
    /// thickness at the lowest point of the shape, nm (before the pattern is laid over it)
    pub bulk: f64,
    /// the share by which the film is thinner at the highest point than at the lowest
    pub lean: f64,
    /// how far round from a summit (radians) the sink reaches nothing
    pub hole: f64,
    /// one over the angle (radians) over which the thickness climbs back out of the sink
    pub climb: f64,
    /// how deep the sink is: 0 = there is none, 1 = down to nothing
    pub sink: f64,
    /// which way is up: 1, or -1 for a film set to drain upwards
    pub up: f64,
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A film that was `film_nm` thick when fresh, drained by `drain` (0 to 1; below 0: the same, upside
/// down).
///
/// The numbers are set against photographs of a soap film draining on a dome (the thickness of each
/// band read off its colour): the lowest point goes 1250, 950, 770, 540, 150, 70 nm, which is the
/// fresh thickness times (1 - drain)^1.2 at drain 0.15, 0.3, 0.45, 0.6, 0.85 and 0.95 for a film that
/// began at 1500; the top of the dome is 0.6 to 0.8 of the bottom throughout; the black spot on the
/// summit grows as the square of the time, and the bands round it crowd closer as it grows (a third
/// as wide by the time it is a sixth of a radian across) until the film is too thin to have any.
pub fn drain(film_nm: f64, drain: f64) -> Drain {
    let x = drain.abs().min(1.0);
    let on = smooth(0.0, 0.1, x);
    let width = 0.13 - 0.15 * x.min(0.6) + 0.4 * smooth(0.68, 0.88, x);
    Drain {
        bulk: film_nm.max(0.0) * (1.0 - x).powf(1.2),
        lean: 0.42 * on,
        hole: 0.03 + 0.36 * x * x,
        climb: 1.0 / width,
        sink: on,
        up: if drain < 0.0 { -1.0 } else { 1.0 },
    }
}

/// The film's thickness (nm) where the surface is at `height` (-1 at the lowest point of the shape to
/// 1 at the highest) and faces `round` radians away from straight up; `pattern`: the pattern there,
/// -1 to 1, laid on with strength `uneven`. `free`: a film of liquid with air on both sides, which
/// never gets thinner than black film. What `filmNm` works out, but for the droplets in black film.
pub fn thickness(d: &Drain, height: f64, round: f64, pattern: f64, uneven: f64, free: bool) -> f64 {
    let level = d.bulk * (1.0 + uneven * pattern) * (1.0 - d.lean * (0.5 * height.clamp(-1.0, 1.0) + 0.5));
    let nm = level * (1.0 - d.sink * (-(round - d.hole * (1.0 + 0.15 * pattern)).max(0.0) * d.climb).exp());
    if free { nm.max(BLACK_NM) } else { nm.max(0.0) }
}

/// The film's thickness taken over the whole shape, nm: what the pieces of a solid body seen through it
/// are given (they get the pattern but not the draining).
pub fn mean_nm(d: &Drain) -> f64 {
    d.bulk * (1.0 - 0.5 * d.lean)
}

/// The twelve numbers the tracer reads (`film3`, `film4`, `film5` in its block).
pub fn numbers(p: &Settings) -> [f64; 12] {
    let d = drain(p.film, p.film_drain);
    // (the tenth: 1 when the family's lights are kept lit beside an environment picture, so that a film
    // mirrors them as well as the picture)
    [d.bulk, d.lean, d.hole, d.climb, d.sink, BLACK_BELOW, 0.0, d.up, mean_nm(&d), if p.hdri_lights > 0.5 { 1.0 } else { 0.0 }, 0.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_film_is_even() {
        let d = drain(400.0, 0.0);
        for (h, r) in [(-1.0, 3.0), (0.0, 1.5), (1.0, 0.0)] {
            assert_eq!(thickness(&d, h, r, 0.0, 0.0, false), 400.0);
        }
    }

    #[test]
    fn a_draining_film_is_thinner_higher_up_and_thinnest_on_the_summit() {
        for x in [0.15, 0.3, 0.45, 0.6, 0.85] {
            let d = drain(1500.0, x);
            // down the side of a dome: height falls as the surface turns away from straight up
            let side: Vec<f64> = (0..=30).map(|i| { let r = i as f64 * 0.05; thickness(&d, r.cos(), r, 0.0, 0.0, false) }).collect();
            assert!(side.windows(2).all(|w| w[1] >= w[0]), "drain {x}: {side:?}");
            assert_eq!(side[0], 0.0, "the summit itself has thinned to nothing");
        }
    }

    #[test]
    fn the_film_thins_all_the_time_and_ends_as_black_film() {
        let at = |x: f64| thickness(&drain(1500.0, x), -0.5, 2.0, 0.0, 0.0, true);
        let run: Vec<f64> = (0..=20).map(|i| at(i as f64 / 20.0)).collect();
        assert!(run.windows(2).all(|w| w[1] <= w[0]), "{run:?}");
        assert_eq!(run[0], 1500.0);
        assert_eq!(run[20], BLACK_NM);
        // the photographs' thicknesses at the foot of the dome
        for (x, nm) in [(0.15, 1250.0), (0.3, 950.0), (0.45, 770.0), (0.6, 540.0), (0.85, 150.0)] {
            let foot = thickness(&drain(1500.0, x), -1.0, 2.5, 0.0, 0.0, true);
            assert!((foot - nm).abs() < 0.08 * nm + 15.0, "drain {x}: {foot} nm at the foot, the photograph shows {nm}");
        }
    }

    #[test]
    fn the_black_spot_grows() {
        // where down the side of a dome the film first exceeds black film
        let edge = |x: f64| {
            let d = drain(1500.0, x);
            (0..3000).map(|i| i as f64 * 0.001).find(|r| thickness(&d, r.cos(), *r, 0.0, 0.0, true) > BLACK_NM).unwrap_or(3.0)
        };
        let e: Vec<f64> = [0.15, 0.3, 0.45, 0.6, 0.85].iter().map(|x| edge(*x)).collect();
        assert!(e.windows(2).all(|w| w[1] > w[0]), "{e:?}");
        assert!(e[0] < 0.06 && e[4] > 0.25, "{e:?}");
    }

    #[test]
    fn draining_upwards_is_the_same_film_upside_down() {
        let (a, b) = (drain(900.0, 0.4), drain(900.0, -0.4));
        assert_eq!(a.up, 1.0);
        assert_eq!(b.up, -1.0);
        assert_eq!(Drain { up: 1.0, ..b }, a);
    }
}
