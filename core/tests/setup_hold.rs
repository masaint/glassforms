//! A setup read from a file is held to the same limits as a typed argument (R2-02), and the piezo track
//! is found under its knob key (R2-03).

use gf_core::anim::Track;
use gf_core::lamps::{Lamp, LampKind};
use gf_core::state::State;

fn track() -> Track {
    Track { on: true, period: 0.0, from: 0.0, to: 1.0, phase: f64::NAN, ease: 99, at: 7 }
}

#[test]
fn a_hand_edited_setup_is_held() {
    let mut s = State::with_args(&[], true);
    let mut setup = s.setup();
    setup.shape.bead_period = 0.0;
    setup.shape.beads = 2.0;
    setup.shape.thick = -3.0;
    setup.shape.melt = f64::NAN;
    setup.lamps = vec![Lamp { kind: LampKind::Tube, size: [-1.0, 0.0], power: -5.0, soft: 9.0, ..Lamp::default() }];
    setup.tracks.insert("thick".into(), track());
    let id = s.preset().params[0].id.to_string();
    setup.values.insert(id, 1e12);
    s.apply_setup(&setup).expect("loads");
    let st = s.shape();
    assert!(st.bead_period >= 0.1 && st.bead_period.is_finite());
    assert!(st.thick >= 0.05 && st.melt.is_finite());
    let d = &s.preset().params[0];
    assert!(st.vals[0] <= gf_core::limits::widen(d.lo, d.hi).1);
    let l = &s.lamps[0];
    assert!(l.size[0] > 0.0 && l.size[1] > 0.0 && l.power >= 0.0 && l.soft <= 1.0);
    assert!(l.reach_radius().is_finite());
    let t = &s.tracks["thick"];
    assert!(t.period >= 1e-3 && t.phase.is_finite() && t.ease < 11 && t.at <= 2);
}
