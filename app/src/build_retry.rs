//! Tracers whose build failed, and when each may be tried again.
//!
//! A failure is kept per tracer key so that the same broken shader is not compiled every frame, but it is
//! not final: the compiler thread can stop, a thread can be refused, the device can run short. Each key is
//! tried again after a wait that doubles with every failure in a row (`FIRST_WAIT` up to `LONGEST_WAIT`),
//! and a success forgets it.
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const FIRST_WAIT: Duration = Duration::from_secs(2);
pub const LONGEST_WAIT: Duration = Duration::from_secs(60);

struct Failure {
    why: String,
    at: Instant,
    /// failures in a row
    tries: u32,
}

#[derive(Default)]
pub struct Failures(HashMap<String, Failure>);

/// What a key's record says now.
#[derive(Debug, PartialEq, Eq)]
pub enum Standing<'a> {
    /// never failed, or failed and since built
    Clear,
    /// failed, and not yet due another try
    Waiting(&'a str),
    /// failed, and due another try (the message stays the one to show until it succeeds)
    Due(&'a str),
}

impl Failures {
    pub fn failed(&mut self, key: &str, why: String, now: Instant) {
        let tries = self.0.get(key).map_or(0, |f| f.tries) + 1;
        self.0.insert(key.to_owned(), Failure { why, at: now, tries });
    }
    pub fn built(&mut self, key: &str) { self.0.remove(key); }
    pub fn standing(&self, key: &str, now: Instant) -> Standing<'_> {
        let Some(f) = self.0.get(key) else { return Standing::Clear };
        let wait = FIRST_WAIT.saturating_mul(1 << (f.tries - 1).min(16)).min(LONGEST_WAIT);
        if now.duration_since(f.at) >= wait { Standing::Due(&f.why) } else { Standing::Waiting(&f.why) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_tracer_is_tried_again_after_a_wait_that_doubles() {
        let mut f = Failures::default();
        let t0 = Instant::now();
        assert_eq!(f.standing("k", t0), Standing::Clear);
        f.failed("k", "broke".into(), t0);
        assert_eq!(f.standing("k", t0), Standing::Waiting("broke"));
        assert_eq!(f.standing("k", t0 + FIRST_WAIT), Standing::Due("broke"));
        // tried and failed again: twice as long
        let t1 = t0 + FIRST_WAIT;
        f.failed("k", "broke again".into(), t1);
        assert_eq!(f.standing("k", t1 + FIRST_WAIT), Standing::Waiting("broke again"));
        assert_eq!(f.standing("k", t1 + FIRST_WAIT * 2), Standing::Due("broke again"));
        // other keys are not held up, and a build forgets the failure
        assert_eq!(f.standing("other", t1), Standing::Clear);
        f.built("k");
        assert_eq!(f.standing("k", t1), Standing::Clear);
    }

    #[test]
    fn the_wait_is_capped() {
        let mut f = Failures::default();
        let t = Instant::now();
        for _ in 0..40 { f.failed("k", "x".into(), t); }
        assert_eq!(f.standing("k", t + LONGEST_WAIT), Standing::Due("x"));
    }
}
