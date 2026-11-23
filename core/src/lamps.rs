//! Lamps: lights that stand in the scene at a place, with a shape and a size (a ball, a tube, a panel),
//! beside the lights at infinity that a family's rig paints into the lights' map. None exist unless one
//! is added; with none, nothing anywhere changes.
//!
//! A lamp stands in the lights' own frame, as the rig's lights do: it turns with them (the light room's
//! ball, yaw, pitch and roll), stays in the room while the shape turns, or turns with the shape, by the
//! same `lights` setting. Its brightness is a radiance, in the same units as a rig light's: a lamp seen
//! in the glass is as bright as a rig light of the same numbers, and what it gives a surface falls off
//! with distance by itself, as the angle it covers shrinks.
//!
//! Each number of each lamp is a slider (`lamp<n>.x` and so on, n from 1), so lamps animate, loop and
//! are kept in a setup. On the command line: `lamp<n>=<kind>,x,y,z,a,b,r,g,b,power` (kind: ball, tube or
//! panel; a, b: its size, see `Lamp::size`), `lamp<n>=off`, `lamp<n>.<number>=`, `lamps=` (none).

use serde::{Deserialize, Serialize};

use crate::math::{apply, euler, Mat3};
use crate::state::{Knob, KnobRef, State};

/// The most lamps there can be.
pub const MAX_LAMPS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LampKind {
    /// a glowing ball
    #[default]
    Ball,
    /// a glowing tube, rounded at its ends (along the lamp's own x)
    Tube,
    /// a flat glowing rectangle that shines from its front (facing the lamp's own +z)
    Panel,
}
impl LampKind {
    pub fn id(self) -> &'static str {
        match self {
            LampKind::Ball => "ball",
            LampKind::Tube => "tube",
            LampKind::Panel => "panel",
        }
    }
    pub fn from_id(s: &str) -> Option<LampKind> {
        match s.trim() {
            "ball" | "sphere" => Some(LampKind::Ball),
            "tube" => Some(LampKind::Tube),
            "panel" | "rect" | "rectangle" => Some(LampKind::Panel),
            _ => None,
        }
    }
    pub fn number(self) -> f32 {
        match self {
            LampKind::Ball => 0.0,
            LampKind::Tube => 1.0,
            LampKind::Panel => 2.0,
        }
    }
}

/// One lamp.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Lamp {
    pub kind: LampKind,
    /// where it stands, in the lights' own frame (the scene's units: a shape fills about 1.3 from its middle)
    pub pos: [f64; 3],
    /// how it is turned: about x, then y, then z (degrees)
    pub rot: [f64; 3],
    /// its size: a ball's radius; a tube's radius and half its length; a panel's half width and half height
    pub size: [f64; 2],
    /// its colour (each 0 to 1) and how bright it is: its radiance is colour times power
    pub rgb: [f64; 3],
    pub power: f64,
    /// the share of its size over which its edge fades out (0 = a hard edge)
    pub soft: f64,
    /// the camera sees it, as a glowing shape
    pub seen: bool,
}
impl Default for Lamp {
    fn default() -> Self {
        Self { kind: LampKind::Ball, pos: [1.6, 1.4, 1.2], rot: [0.0; 3], size: [0.2, 0.6], rgb: [1.0, 1.0, 1.0], power: 8.0, soft: 0.3, seen: true }
    }
}

/// A lamp's numbers that are sliders: name, label, lowest, highest, step.
pub const LAMP_KNOBS: [(&str, &str, f64, f64, f64); 13] = [
    ("x", "lamp x", -6.0, 6.0, 0.005),
    ("y", "lamp y", -6.0, 6.0, 0.005),
    ("z", "lamp z", -6.0, 6.0, 0.005),
    ("rx", "lamp turn x °", -180.0, 180.0, 0.5),
    ("ry", "lamp turn y °", -180.0, 180.0, 0.5),
    ("rz", "lamp turn z °", -180.0, 180.0, 0.5),
    ("a", "lamp size", 0.01, 2.0, 0.005),
    ("b", "lamp length", 0.01, 3.0, 0.005),
    ("power", "lamp power", 0.0, 200.0, 0.05),
    ("r", "lamp red", 0.0, 1.0, 0.005),
    ("g", "lamp green", 0.0, 1.0, 0.005),
    ("bl", "lamp blue", 0.0, 1.0, 0.005),
    ("soft", "lamp edge", 0.0, 1.0, 0.01),
];

impl Lamp {
    pub fn get(&self, c: usize) -> f64 {
        match c {
            0..=2 => self.pos[c],
            3..=5 => self.rot[c - 3],
            6 | 7 => self.size[c - 6],
            8 => self.power,
            9..=11 => self.rgb[c - 9],
            _ => self.soft,
        }
    }
    pub fn set(&mut self, c: usize, v: f64) {
        match c {
            0..=2 => self.pos[c] = v,
            3..=5 => self.rot[c - 3] = v,
            6 | 7 => self.size[c - 6] = v.max(1e-3),
            8 => self.power = v.max(0.0),
            9..=11 => self.rgb[c - 9] = v.max(0.0),
            _ => self.soft = v.clamp(0.0, 1.0),
        }
    }
    /// Every number held to the lamp knobs' limits (a lamp read from a file may hold anything).
    pub fn hold(&mut self) {
        let mut v = [0.0; 13];
        v[..3].copy_from_slice(&self.pos);
        v[3..6].copy_from_slice(&self.rot);
        v[6..8].copy_from_slice(&self.size);
        v[8] = self.power;
        v[9..12].copy_from_slice(&self.rgb);
        v[12] = self.soft;
        for (c, (_, _, lo, hi, _)) in LAMP_KNOBS.iter().enumerate() {
            self.set(c, crate::limits::hold(v[c], (*lo, *hi)));
        }
    }
    /// Its radiance, by colour.
    pub fn radiance(&self) -> [f64; 3] {
        self.rgb.map(|c| c * self.power)
    }
    /// Its own axes (x along a tube and across a panel, y up a panel, z the way a panel faces), in the
    /// lights' frame.
    pub fn axes(&self) -> Mat3 {
        euler(self.rot[0], self.rot[1], self.rot[2])
    }
    /// The radius of a ball that sends about as much light: for the shadows it casts and how far it reaches.
    pub fn reach_radius(&self) -> f64 {
        match self.kind {
            LampKind::Ball => self.size[0],
            LampKind::Tube => (self.size[0] * (self.size[0] + 2.0 * self.size[1])).sqrt(),
            LampKind::Panel => (self.size[0] * self.size[1] / std::f64::consts::PI).sqrt() * 2.0,
        }
    }
    /// The turn (rx, ry, 0) that makes a panel at `pos` face the middle of the scene: its own z, turned
    /// about x and then about y, points from `pos` to the middle.
    pub fn facing_middle(pos: [f64; 3]) -> [f64; 3] {
        let l = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt().max(1e-9);
        let d = pos.map(|c| -c / l);
        [(-d[1]).clamp(-1.0, 1.0).asin().to_degrees(), d[0].atan2(d[2]).to_degrees(), 0.0]
    }
    /// As text: `kind,x,y,z,a,b,r,g,b,power`.
    pub fn text(&self) -> String {
        format!("{},{},{},{},{},{},{},{},{},{}", self.kind.id(), self.pos[0], self.pos[1], self.pos[2], self.size[0], self.size[1], self.rgb[0], self.rgb[1], self.rgb[2], self.power)
    }
    /// From such text (the numbers after the kind may stop early: the rest stay as a new lamp has them).
    pub fn from_text(text: &str) -> Option<Lamp> {
        let mut parts = text.split(',');
        let kind = LampKind::from_id(parts.next()?)?;
        let nums: Vec<f64> = parts.map(|s| s.trim().parse::<f64>().ok().filter(|x| x.is_finite())).collect::<Option<_>>()?;
        let mut l = Lamp { kind, ..Lamp::default() };
        if kind == LampKind::Panel { l.size = [0.5, 0.35]; }
        for (i, v) in nums.iter().enumerate() {
            match i {
                0..=2 => l.pos[i] = *v,
                3 | 4 => l.size[i - 3] = v.max(1e-3),
                5..=7 => l.rgb[i - 5] = v.max(0.0),
                8 => l.power = v.max(0.0),
                _ => return None,
            }
        }
        // (a panel faces the middle of the scene from where it stands; its turn sliders turn it from there)
        if kind == LampKind::Panel { l.rot = Lamp::facing_middle(l.pos); }
        Some(l)
    }
}

impl State {
    /// The lamps' sliders, every lamp's.
    pub fn lamp_knobs(&self) -> Vec<Knob> {
        (0..self.lamps.len()).flat_map(|i| self.lamp_knobs_of(i)).collect()
    }
    /// The sliders of lamp i.
    pub fn lamp_knobs_of(&self, i: usize) -> Vec<Knob> {
        LAMP_KNOBS
            .iter()
            .enumerate()
            .map(|(c, (id, label, lo, hi, step))| Knob {
                key: format!("lamp{}.{}", i + 1, id),
                id: (*id).into(),
                label: (*label).into(),
                lo: *lo,
                hi: *hi,
                step: *step,
                shape: false,
                wave: true,
                r: KnobRef::Lamp(i, c),
            })
            .collect()
    }
    pub(crate) fn lamp_get(&self, i: usize, c: usize) -> f64 {
        self.lamps.get(i).map_or(0.0, |l| l.get(c))
    }
    pub(crate) fn lamp_set(&mut self, i: usize, c: usize, v: f64) {
        if let Some(l) = self.lamps.get_mut(i) {
            if l.get(c) != v {
                l.set(c, v);
                self.lamps_rev += 1;
            }
        }
    }
    /// Add a lamp of this kind (None if there are as many as there can be): where a key light usually
    /// stands, up and to the right in front of the shape.
    pub fn add_lamp(&mut self, kind: LampKind) -> Option<usize> {
        if self.lamps.len() >= MAX_LAMPS { return None; }
        let mut l = Lamp { kind, ..Lamp::default() };
        match kind {
            LampKind::Ball => {}
            LampKind::Tube => { l.size = [0.06, 0.7]; l.rot = [0.0, 0.0, 90.0]; l.power = 10.0; }
            // a panel faces the middle of the scene from where it stands
            LampKind::Panel => {
                l.size = [0.6, 0.4];
                l.power = 4.0;
                l.rot = Lamp::facing_middle(l.pos);
            }
        }
        self.lamps.push(l);
        self.lamps_rev += 1;
        Some(self.lamps.len() - 1)
    }
    /// Take lamp i away (the tracks of its sliders go with it, and those of the lamps after it move up).
    pub fn remove_lamp(&mut self, i: usize) {
        if i >= self.lamps.len() { return; }
        self.lamps.remove(i);
        let old: Vec<String> = self.tracks.keys().filter(|k| k.starts_with("lamp")).cloned().collect();
        for k in old {
            let Some((n, rest)) = k[4..].split_once('.') else { continue };
            let Ok(n) = n.parse::<usize>() else { continue };
            let tr = self.tracks.remove(&k).expect("listed");
            if n == i + 1 { continue; }
            let n2 = if n > i + 1 { n - 1 } else { n };
            self.tracks.insert(format!("lamp{n2}.{rest}"), tr);
        }
        self.lamps_rev += 1;
    }
    /// Where lamp i stands as the camera sees it, and its axes (as columns of the matrix) likewise.
    pub fn lamp_in_view(&self, i: usize) -> Option<([f64; 3], Mat3)> {
        let l = self.lamps.get(i)?;
        let m = self.light_m();
        Some((apply(&m, l.pos), crate::math::mul(&m, &l.axes())))
    }

    /// `lamp<n>=...`, `lamp<n>=off`, `lamp<n>.<number>=`, `lamps=` (none). None if the name is not a lamp's.
    pub fn apply_lamp_arg(&mut self, name: &str, value: &str) -> Option<bool> {
        if name == "lamps" {
            if !value.trim().is_empty() && value.trim() != "none" { return Some(false); }
            self.lamps.clear();
            self.lamps_rev += 1;
            return Some(true);
        }
        let rest = name.strip_prefix("lamp")?;
        let (num, field) = rest.split_once('.').unwrap_or((rest, ""));
        let n = num.parse::<usize>().ok().filter(|n| *n >= 1)?;
        let i = n - 1;
        if field.is_empty() {
            if value.trim() == "off" {
                if i < self.lamps.len() { self.remove_lamp(i); }
                return Some(true);
            }
            let Some(l) = Lamp::from_text(value) else { return Some(false) };
            if i < self.lamps.len() {
                self.lamps[i] = l;
            } else if i == self.lamps.len() && i < MAX_LAMPS {
                self.lamps.push(l);
            } else {
                return Some(false);
            }
            self.lamps_rev += 1;
            return Some(true);
        }
        if i >= self.lamps.len() { return Some(false); }
        match field {
            "kind" => match LampKind::from_id(value) {
                Some(k) => { self.lamps[i].kind = k; self.lamps_rev += 1; Some(true) }
                None => Some(false),
            },
            "seen" => {
                self.lamps[i].seen = value.trim() != "0" && value.trim() != "off";
                self.lamps_rev += 1;
                Some(true)
            }
            _ => {
                let Some(c) = LAMP_KNOBS.iter().position(|k| k.0 == field) else { return Some(false) };
                let Ok(v) = value.trim().parse::<f64>() else { return Some(false) };
                if !v.is_finite() { return Some(false); }
                self.lamp_set(i, c, v);
                Some(true)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lamp_reads_back_from_its_text() {
        let l = Lamp { kind: LampKind::Tube, pos: [0.5, -1.0, 2.0], size: [0.1, 0.8], rgb: [1.0, 0.5, 0.25], power: 12.0, ..Lamp::default() };
        let back = Lamp::from_text(&l.text()).expect("reads");
        assert_eq!((back.kind, back.pos, back.size, back.rgb, back.power), (l.kind, l.pos, l.size, l.rgb, l.power));
        assert!(Lamp::from_text("lantern,1,2,3").is_none());
    }
}
