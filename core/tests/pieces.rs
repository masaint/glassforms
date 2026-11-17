//! Solid pieces (`presets::Piece`) through `curve::add_pieces` and `Curve::moved`: what they hold stays
//! where the piece is, however the curve is placed and moved.

use gf_core::curve::{add_pieces, piece_len, Curve, Pose};
use gf_core::presets::{Piece, Term};

/// Is p inside every plane and ball term of the piece starting at point i?
fn inside(c: &Curve, i: usize, p: [f64; 3]) -> bool {
    let m = piece_len(&c.pts, i);
    let at = |k: usize| [c.pts[k * 4] as f64, c.pts[k * 4 + 1] as f64, c.pts[k * 4 + 2] as f64, c.pts[k * 4 + 3] as f64];
    let mut j = i + 2;
    while j < i + m {
        let (t, q) = (at(j), at(j + 1));
        let d = match t[0].round() as i32 {
            0 => q[0] * p[0] + q[1] * p[1] + q[2] * p[2] - q[3],
            1 => ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() - q[3],
            k => panic!("term {k} not looked at here"),
        };
        if d > 0.0 { return false; }
        j += 2;
    }
    true
}

#[test]
fn a_piece_moves_with_its_curve() {
    let pose = Pose { c: [0.5, 0.0, 0.0], sc: 2.0, ax: [0, 1, 2], e: None };
    let mut c = Curve { pts: Vec::new(), brk: Vec::new(), n: 0, rmax: 0.0, ds: 0.01, length: 0.0, amp: 1.0, pose, vary: true, cut: None };
    // a box from (0.4, -0.1, -0.1) to (0.8, 0.1, 0.1), cut by a ball round (0.8, 0, 0)
    let mut terms = Vec::new();
    for (n, d) in [([1.0, 0.0, 0.0], 0.8), ([-1.0, 0.0, 0.0], -0.4), ([0.0, 1.0, 0.0], 0.1), ([0.0, -1.0, 0.0], 0.1), ([0.0, 0.0, 1.0], 0.1), ([0.0, 0.0, -1.0], 0.1)] {
        terms.push(Term::Plane { n, d });
    }
    terms.push(Term::Ball { c: [0.8, 0.0, 0.0], r: 0.3 });
    add_pieces(&mut c, &[Piece { lo: [0.4, -0.1, -0.1], hi: [0.8, 0.1, 0.1], terms }], 1.0, 0.01);
    assert_eq!(piece_len(&c.pts, 0), c.n);
    // placed: (p - 0.5) * 2
    let place = |p: [f64; 3]| [(p[0] - 0.5) * 2.0, p[1] * 2.0, p[2] * 2.0];
    assert!(inside(&c, 0, place([0.7, 0.05, 0.0])));
    assert!(!inside(&c, 0, place([0.45, 0.0, 0.0])), "cut away by the ball");
    assert!(!inside(&c, 0, place([0.7, 0.15, 0.0])));
    // turned a quarter about y and moved
    let r = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
    let t = [0.3, -0.2, 0.1];
    let mv = |p: [f64; 3]| [r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0], r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1], r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2]];
    c.moved(&r, t);
    for (p, want) in [([0.7, 0.05, 0.0], true), ([0.45, 0.0, 0.0], false), ([0.7, 0.15, 0.0], false), ([0.75, -0.08, 0.08], true)] {
        assert_eq!(inside(&c, 0, mv(place(p))), want, "{p:?}");
    }
    // the box still holds the piece
    let mid = [c.pts[0] as f64, c.pts[1] as f64, c.pts[2] as f64];
    let half = [c.pts[4] as f64, c.pts[5] as f64, c.pts[6] as f64];
    let q = mv(place([0.75, -0.08, 0.08]));
    for k in 0..3 { assert!((q[k] - mid[k]).abs() <= half[k] + 1e-6); }
}
