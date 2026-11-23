//! Split refinement: an expensive refining frame traced as two halves of its columns on two updates.
//! The first half is shown but not counted; the second completes the same sample and is counted.
//!
//! Which half comes next depends on which half was last *drawn*, not merely planned: a frame can be
//! called off after the plan (no tracer yet, no bind group), and a second half after a first that was
//! never drawn would count a sample whose other half of the columns was never traced.

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Half {
    /// the whole frame at once
    #[default]
    Whole = 0,
    First = 1,
    Second = 2,
}

impl Half {
    /// The half this frame would trace, after `self` was the last one drawn.
    pub fn after(self, split: bool) -> Half {
        match (split, self) {
            (false, _) => Half::Whole,
            (true, Half::First) => Half::Second,
            (true, _) => Half::First,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Half;

    #[test]
    fn halves_alternate_while_split() {
        let mut h = Half::Whole;
        let mut seen = Vec::new();
        for _ in 0..4 { h = h.after(true); seen.push(h); }
        assert_eq!(seen, [Half::First, Half::Second, Half::First, Half::Second]);
        assert_eq!(Half::First.after(false), Half::Whole);
    }

    #[test]
    fn a_first_half_called_off_is_planned_again_not_skipped() {
        // drawn: a whole frame. Planned: a first half, but the frame returns before tracing, so the
        // last drawn stays Whole; the next frame must plan a first half again.
        let drawn = Half::Whole;
        let planned = drawn.after(true);
        assert_eq!(planned, Half::First);
        assert_eq!(drawn.after(true), Half::First, "a second half with no first one drawn");
    }
}
