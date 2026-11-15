//! Animation: easing curves, the tracks that make a slider travel between two stops, and the plan that
//! makes everything fit a loop.

use serde::{Deserialize, Serialize};

/// The easing types, in the order the panel lists them. A setting holds an index into this.
pub const EASES: [&str; 11] = ["linear", "sine", "quad", "cubic", "quart", "quint", "expo", "circ", "back", "elastic", "bounce"];
/// Where an easing acts.
pub const EASE_AT: [&str; 3] = ["start", "finish", "both"];

fn bounce_out(mut x: f64) -> f64 {
    let (n, d) = (7.5625, 2.75);
    if x < 1.0 / d {
        n * x * x
    } else if x < 2.0 / d {
        x -= 1.5 / d;
        n * x * x + 0.75
    } else if x < 2.5 / d {
        x -= 2.25 / d;
        n * x * x + 0.9375
    } else {
        x -= 2.625 / d;
        n * x * x + 0.984375
    }
}

/// Each easing as a gentle START (0 at 0, 1 at 1); the finish and both-ends forms are made from it.
/// The functions are the browser's own, to the last digit: an eased slider can be a number that a chaotic
/// shape is built from, and there the last digit matters.
fn ease_in(ty: usize, x: f64) -> f64 {
    use crate::curve::jsmath::{cos, exp, sin, sqrt};
    use std::f64::consts::{LN_2, PI};
    match ty {
        0 => x,
        2 => x * x,
        3 => x * x * x,
        4 => x * x * x * x,
        5 => x * x * x * x * x,
        // 2 to the power 10x, rescaled to start from exactly 0 (the usual form begins with a small jump);
        // through exp, as the web version writes it, because pow is not the same number in every program
        6 => (exp(10.0 * x * LN_2) - 1.0) / 1023.0,
        7 => 1.0 - sqrt((1.0 - x * x).max(0.0)),
        // pulls back a little before setting off
        8 => 2.70158 * x * x * x - 1.70158 * x * x,
        9 => {
            if x <= 0.0 {
                0.0
            } else if x >= 1.0 {
                1.0
            } else {
                -(exp(10.0 * x * LN_2) - 1.0) / 1023.0 * sin((10.0 * x - 10.75) * (2.0 * PI) / 3.0)
            }
        }
        10 => 1.0 - bounce_out(1.0 - x),
        _ => 1.0 - cos(x * PI / 2.0),
    }
}

/// `at`: 0 = eased at the start, 1 = at the finish, 2 = at both ends. "sine, both" is a plain sinusoid.
pub fn eased(ty: usize, at: u8, x: f64) -> f64 {
    match at {
        0 => ease_in(ty, x),
        1 => 1.0 - ease_in(ty, 1.0 - x),
        _ => {
            if x < 0.5 {
                ease_in(ty, 2.0 * x) / 2.0
            } else {
                1.0 - ease_in(ty, 2.0 - 2.0 * x) / 2.0
            }
        }
    }
}

/// A track makes a slider travel from one stop to the other and back, once per period.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub on: bool,
    /// seconds for the trip there and back, before speed and loop snapping
    pub period: f64,
    pub from: f64,
    pub to: f64,
    /// where in its trip the track is at clock 0, in trips
    pub phase: f64,
    /// -1 = whatever is set for all, else an index into EASES, used with this track's own `at`
    pub ease: i32,
    pub at: u8,
}

impl Track {
    /// The easing this track uses, given the one set for all.
    pub fn easing(&self, all: (usize, u8)) -> (usize, u8) {
        if self.ease < 0 { all } else { (self.ease as usize, self.at) }
    }

    /// Where the slider is at time `clock`, the trip there and back taking `per` seconds.
    pub fn value(&self, clock: f64, per: f64, all: (usize, u8)) -> f64 {
        let u = ((clock / per + self.phase) % 1.0 + 1.0) % 1.0;
        let d = self.to - self.from;
        let (ty, at) = self.easing(all);
        if u < 0.5 { self.from + d * eased(ty, at, 2.0 * u) } else { self.to - d * eased(ty, at, 2.0 * u - 1.0) }
    }

    /// Choose where in its trip the track is, so that it carries on from `value` at time `clock`.
    pub fn rephase(&mut self, value: f64, clock: f64, per: f64, all: (usize, u8)) {
        let d = self.to - self.from;
        let (ty, at) = self.easing(all);
        let f = if d.abs() > 1e-12 { (value - self.from) / d } else { 0.0 };
        // at a stop (or outside): begin the trip from that stop
        let mut x = if f <= 0.0 { 0.0 } else { 1.0 };
        if f > 0.0 && f < 1.0 {
            // the earliest moment of the outward trip at which the easing passes this value
            // (some easings pass it more than once)
            const N: usize = 256;
            let (mut lo, mut hi) = (0.0, 1.0);
            for i in 1..=N {
                if eased(ty, at, i as f64 / N as f64) >= f {
                    lo = (i - 1) as f64 / N as f64;
                    hi = i as f64 / N as f64;
                    break;
                }
            }
            for _ in 0..30 {
                let m = (lo + hi) / 2.0;
                if eased(ty, at, m) < f { lo = m } else { hi = m }
            }
            x = (lo + hi) / 2.0;
        }
        self.phase = x / 2.0 - clock / per;
    }
}

/// JavaScript's Math.round: halves go up.
pub fn js_round(x: f64) -> f64 {
    crate::curve::jsmath::round(x)
}

/// The period nearest to `p` that fits a whole number of times into a loop of length `l`.
pub fn snap(l: f64, p: f64) -> f64 {
    l / js_round(l / p).max(1.0)
}

/// One thing that moves, in a loop plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanItem {
    pub label: String,
    /// its period as set, seconds
    pub was: f64,
    /// its period in the loop
    pub now: f64,
    /// how many times it goes round in one loop
    pub n: u32,
    /// it is a turning (spin, lights) rather than a swing
    pub turn: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoopPlan {
    /// length of the loop, seconds
    pub l: f64,
    pub items: Vec<PlanItem>,
}

/// Looping. Everything that moves has a period. A loop of length L repeats exactly only if each of those
/// fits into it a whole number of times, so each period p is replaced by the nearest that does. When the
/// length is not given (`fixed` <= 0), the shortest is looked for at which nothing has to change by more
/// than a trifle, so that slow things are not sped up just to fit.
pub fn loop_plan(moving: &[(String, f64, bool)], fixed: f64) -> LoopPlan {
    let mut l = fixed;
    if !(l > 0.0) {
        l = 8.0;
        if !moving.is_empty() {
            let lo = moving.iter().fold(2.0f64, |m, it| m.max(it.1));
            let mut best = f64::INFINITY;
            let mut c = lo;
            while c <= lo.max(90.0) + 1e-9 {
                let mut err = 0.0f64;
                for it in moving {
                    err = err.max((snap(c, it.1) - it.1).abs() / it.1);
                }
                // a second of extra length is worth a thirtieth of a percent
                let score = err + c / 3000.0;
                if score < best - 1e-9 {
                    best = score;
                    l = c;
                }
                c += 0.05;
            }
            l = js_round(l * 20.0) / 20.0;
        }
    }
    let items = moving
        .iter()
        .map(|(label, was, turn)| {
            let now = snap(l, *was);
            PlanItem { label: label.clone(), was: *was, now, n: js_round(l / now) as u32, turn: *turn }
        })
        .collect();
    LoopPlan { l, items }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn every_easing_runs_from_0_to_1_without_a_jump() {
        for ty in 0..EASES.len() {
            for at in 0..3u8 {
                let (e0, e1) = (eased(ty, at, 0.0), eased(ty, at, 1.0));
                assert!(e0.abs() < 1e-12 && (e1 - 1.0).abs() < 1e-12, "{} at {}: ends {e0}, {e1}", EASES[ty], at);
                let mut prev = e0;
                for i in 1..=1000 {
                    let v = eased(ty, at, i as f64 / 1000.0);
                    assert!(v.is_finite() && (v - prev).abs() < 0.05, "{} at {}: step at {i}", EASES[ty], at);
                    prev = v;
                }
                // a steep start is not a jump
                assert!((eased(ty, at, 1e-12) - e0).abs() < 1e-5 && (eased(ty, at, 1.0 - 1e-12) - e1).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn sine_both_is_a_sinusoid() {
        let tr = Track { on: true, period: 4.0, from: 1.3, to: 1.6, phase: 0.0, ease: -1, at: 2 };
        for i in 0..=40 {
            let t = i as f64 * 0.1;
            let want = 1.45 - 0.15 * (2.0 * PI * t / 4.0).cos();
            assert!((tr.value(t, 4.0, (1, 2)) - want).abs() < 1e-12);
        }
    }

    #[test]
    fn a_track_carries_on_from_its_slider() {
        for ty in 0..EASES.len() {
            for at in 0..3u8 {
                let mut tr = Track { on: true, period: 4.0, from: 1.3, to: 1.6, phase: 0.0, ease: ty as i32, at };
                tr.rephase(1.37, 2.5, 4.0, (1, 2));
                assert!((tr.value(2.5, 4.0, (1, 2)) - 1.37).abs() < 1e-8, "{} at {}", EASES[ty], at);
                tr.rephase(1.3, 0.0, 4.0, (1, 2));
                assert!((tr.value(0.0, 4.0, (1, 2)) - 1.3).abs() < 1e-12 && (tr.value(2.0, 4.0, (1, 2)) - 1.6).abs() < 1e-12);
            }
        }
    }

    // the numbers the web version gives for the same things (tools/ui_test.js, tools/feature_test.js)
    #[test]
    fn loop_plans_match_the_web_version() {
        let p = loop_plan(&[("index".into(), 14.0, false), ("spin".into(), 2.0 * PI / 0.3, true)], 0.0);
        assert!((p.l - 41.95).abs() < 1e-9, "{}", p.l);
        assert_eq!((p.items[0].n, p.items[1].n), (3, 2));
        let s = 1.7;
        let p = loop_plan(&[("index".into(), 4.0 / s, false), ("dispersion".into(), 5.5 / s, false), ("spin".into(), 2.0 * PI / (0.3 * s), true)], 0.0);
        assert!((p.l - 61.35).abs() < 1e-9, "{}", p.l);
        assert_eq!(p.items.iter().map(|i| i.n).collect::<Vec<_>>(), vec![26, 19, 5]);
        for it in &p.items {
            assert!((p.l / it.now - (p.l / it.now).round()).abs() < 1e-9);
        }
    }
}
