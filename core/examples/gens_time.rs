//! How long the newer families take to build on the CPU: each preset as it starts (the curve or pieces
//! worked out, placed and cut into steps at the still resolution), then again with its first slider
//! nudged (a slider change: nothing cached for it), and once more with nothing changed.
//!
//!     cargo run --release -p gf-core --example gens_time

use gf_core::state::{Detail, State};
use std::time::Instant;

fn main() {
    let mut s = State::new();
    for fam in ["links", "braids", "fieldlines", "crystals", "foam", "branching", "snowflakes", "text"] {
        let Some(f) = s.families.iter().position(|x| x.id == fam) else { continue };
        s.set_scene(f);
        for p in 0..s.family().presets.len() {
            s.choose(p);
            let t = Instant::now();
            let job = s.shape_job(Detail::Full, None).unwrap();
            let first = t.elapsed().as_secs_f64() * 1e3;
            // a slider change: the first slider moved by a step
            let d = &s.preset().params[0];
            let (i, v) = (0, (d.value + d.step).min(d.hi));
            s.shape_mut().vals[i] = v;
            let t = Instant::now();
            let _ = s.shape_job(Detail::Full, None).unwrap();
            let changed = t.elapsed().as_secs_f64() * 1e3;
            let n = match job.shape { gf_core::state::ShapeJob::Curve { curve, .. } => curve.n, _ => 0 };
            println!("{fam:11} {:16} first {first:7.1} ms   slider moved {changed:7.1} ms   {n} points", s.preset().id);
        }
    }
}
