//! The camera as one that can stand anywhere: orbiting the shape (as it always did) or flying free.
//!
//! The renderer's world is the camera's own frame with the object's centre at its origin: the object is
//! turned into it by `view_r` (the room's turn as the camera sees it, `room_m`, times the object's own
//! turn in the room, `r`), and the lens stands at `eye()` = (eyeX, eyeY, dist) looking down -z. With
//! eyeX = eyeY = 0 that is the camera the program always had, looking at the middle of the shape.
//!
//! So a free camera is nothing new to store: where it stands is `eye()`, which way it looks is the room's
//! turn (`azim` round the room's vertical, `elev` up or down, `roll`), all of them ordinary settings that
//! are kept in a setup and can be animated as before. Flying turns the camera where it stands (the
//! room's turn changes, and `eye()` with it so that the lens keeps its place in the room) and moves it
//! along its own axes (only `eye()` changes).
//!
//! Orbiting turns the view about a pivot: the object's centre, or what was last framed. The turn itself
//! is the old one (about the object's centre); `keep_pivot` then slides the camera so that the pivot
//! stays where it was in the picture. With the pivot at the centre that slide is nothing at all.

use crate::math::{apply, transpose, Mat3, DEG};
use crate::state::State;

/// How far a free camera may look up or down, degrees.
pub const PITCH_MAX: f64 = 89.5;

fn wrap(deg: f64) -> f64 {
    ((deg + 540.0) % 360.0 + 360.0) % 360.0 - 180.0
}

impl State {
    /// Where the lens stands, in the camera's frame (origin at the object's centre).
    pub fn eye(&self) -> [f64; 3] {
        [self.p.eye_x, self.p.eye_y, self.p.dist]
    }
    pub fn set_eye(&mut self, e: [f64; 3]) {
        if !e.iter().all(|x| x.is_finite()) { return; }
        (self.p.eye_x, self.p.eye_y, self.p.dist) = (e[0], e[1], e[2]);
    }
    /// How far the lens is from the object's centre.
    pub fn eye_reach(&self) -> f64 {
        let e = self.eye();
        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
    }
    /// The point the view turns about, in the object's frame.
    pub fn pivot(&self) -> [f64; 3] {
        [self.p.pivot_x, self.p.pivot_y, self.p.pivot_z]
    }
    pub fn set_pivot(&mut self, at: [f64; 3], r: f64) {
        (self.p.pivot_x, self.p.pivot_y, self.p.pivot_z, self.p.pivot_r) = (at[0], at[1], at[2], r.max(0.0));
    }
    /// The camera back on the line through the object's centre, turning about the centre.
    pub fn reset_eye(&mut self) {
        (self.p.eye_x, self.p.eye_y) = (0.0, 0.0);
        self.set_pivot([0.0; 3], 0.0);
    }

    /// The view was turned from `before` (object -> camera) to what it is now: slide the camera so that
    /// the pivot stays where it was in the picture.
    pub fn keep_pivot(&mut self, before: &Mat3) {
        let o = self.pivot();
        if o == [0.0; 3] { return; }
        let (a, b) = (apply(before, o), apply(&self.view_r(), o));
        let e = self.eye();
        self.set_eye([e[0] - a[0] + b[0], e[1] - a[1] + b[1], e[2] - a[2] + b[2]]);
    }

    /// The wheel while orbiting: closer to the pivot or further from it, along the line of sight, by the
    /// factor k. The nearest is a little outside what was framed (the whole shape: 1.5, as ever).
    pub fn dolly(&mut self, k: f64) {
        let s = apply(&self.view_r(), self.pivot());
        let a = self.p.dist - s[2];
        if a <= 1e-3 {
            // the pivot is level with the lens or behind it: plain steps along the line of sight
            self.p.dist += (k - 1.0) * 2.0;
            return;
        }
        let lo = if self.p.pivot_r > 0.0 { self.p.pivot_r * 1.05 } else { 1.5 };
        self.p.dist = s[2] + (a * k).clamp(lo.min(a), 60f64.max(a));
    }

    /// Fly: turn the camera where it stands, `yaw` degrees to the right and `pitch` degrees up. The room
    /// turns in the picture and the lens keeps its place in the room.
    pub fn fly_look(&mut self, yaw: f64, pitch: f64) {
        let c = apply(&transpose(&self.room_m()), self.eye());
        self.p.azim = wrap(self.p.azim + yaw);
        self.p.elev = (self.p.elev - pitch).clamp(-PITCH_MAX, PITCH_MAX);
        let e = apply(&self.room_m(), c);
        self.set_eye(e);
    }

    /// Fly: move by so much to the right and forward as the camera faces, and up as the room stands.
    pub fn fly_move(&mut self, right: f64, up: f64, forward: f64) {
        let u = apply(&self.room_m(), [0.0, 1.0, 0.0]);
        let e = self.eye();
        self.set_eye([e[0] + right + u[0] * up, e[1] + u[1] * up, e[2] - forward + u[2] * up]);
    }

    /// Back to orbiting after flying: the view turns about the point in the middle of the picture as deep
    /// as the object's centre (or a little ahead of the lens, if the centre is not in front of it).
    pub fn pivot_ahead(&mut self) {
        let e = self.eye();
        let s = if e[2] > 0.3 { [e[0], e[1], 0.0] } else { [e[0], e[1], e[2] - 3.0] };
        if e[0] == 0.0 && e[1] == 0.0 && e[2] > 0.3 && self.pivot() == [0.0; 3] { return; }
        let o = apply(&transpose(&self.view_r()), s);
        self.set_pivot(o, 0.0);
    }

    /// Point the camera at a ball (its centre in the object's frame, its radius): it keeps the way it
    /// looks and slides until the ball is in the middle of the picture and fills about two thirds of its
    /// height. What was framed becomes the pivot.
    pub fn frame_on(&mut self, centre: [f64; 3], r: f64) {
        let s = apply(&self.view_r(), centre);
        let t = (self.p.fov * DEG / 2.0).tan().max(1e-3);
        // (the ball's outline: the half angle it fills is asin(r / d))
        let d = (r.max(0.05) * 1.5 * (1.0 + t * t).sqrt() / t).max(r * 1.1);
        self.set_eye([s[0], s[1], s[2] + d]);
        self.set_pivot(centre, r);
        self.cut = true;
    }

    /// Look along one of the room's axes (0 x, 1 y, 2 z) from its positive or negative side, at the
    /// pivot, from as far as the camera is now: the views the corner cube snaps to.
    pub fn snap_view(&mut self, k: usize, positive: bool) {
        let o = self.pivot();
        let s = apply(&self.view_r(), o);
        let e = self.eye();
        let d = ((e[0] - s[0]).powi(2) + (e[1] - s[1]).powi(2) + (e[2] - s[2]).powi(2)).sqrt().max(0.3);
        let (azim, elev) = match (k, positive) {
            (0, true) => (-90.0, 0.0),
            (0, false) => (90.0, 0.0),
            (1, true) => (0.0, 90.0),
            (1, false) => (0.0, -90.0),
            (_, true) => (0.0, 0.0),
            (_, false) => (180.0, 0.0),
        };
        (self.p.azim, self.p.elev, self.p.roll) = (azim, elev, 0.0);
        let s = apply(&self.view_r(), o);
        self.set_eye([s[0], s[1], s[2] + d]);
    }
}
