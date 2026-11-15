//! A whole setup as a file: which shape, every slider, the view, the lights, what is animated and how,
//! and the video options. Saved as JSON so that it can be read and edited by hand.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::anim::Track;
use crate::presets::Light;
use crate::settings::{RenderOpts, Settings};
use crate::state::{Detail, Rig, State, LIGHT_SOFT};

pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShapeSetup {
    pub thick: f64,
    pub len: f64,
    pub melt: f64,
    pub puff: f64,
    pub travel: f64,
    pub beads: f64,
    pub bead_period: f64,
    /// voxels per side: when still, and while the shape animates
    pub res: u32,
    pub anim_res: u32,
    /// the model file, when the shape is one the user chose (left out otherwise): relative to the
    /// program's folder when the file is in it
    #[serde(skip_serializing_if = "String::is_empty")]
    pub model: String,
}
impl Default for ShapeSetup {
    fn default() -> Self {
        Self { thick: 1.0, len: 1.0, melt: 0.006, puff: 0.0, travel: 0.0, beads: 0.0, bead_period: 2.0, res: crate::state::RES_FULL, anim_res: crate::state::RES_ANIM, model: String::new() }
    }
}

/// One light as a setup file has it (angles and sizes in degrees; `rgb` is colour and brightness together).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightSetup {
    pub az: f64,
    pub el: f64,
    pub w: f64,
    pub h: f64,
    pub rgb: [f64; 3],
    #[serde(default = "soft")]
    pub soft: f64,
}
fn soft() -> f64 { LIGHT_SOFT }

/// A family's lights, when they are not the ones it ships with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigSetup {
    #[serde(default)]
    pub lights: Vec<LightSetup>,
    #[serde(default)]
    pub ambient: [f64; 3],
}
impl From<&Rig> for RigSetup {
    fn from(r: &Rig) -> Self {
        Self { lights: r.lights.iter().map(|l| LightSetup { az: l.az, el: l.el, w: l.w, h: l.h, rgb: l.rgb, soft: l.soft }).collect(), ambient: r.ambient }
    }
}
impl From<&RigSetup> for Rig {
    fn from(r: &RigSetup) -> Self {
        Self { lights: r.lights.iter().map(|l| Light { az: l.az, el: l.el, w: l.w, h: l.h, rgb: l.rgb, soft: l.soft }).collect(), ambient: r.ambient }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Setup {
    pub version: u32,
    pub family: String,
    pub preset: String,
    /// the preset's own numbers, by name
    #[serde(default)]
    pub values: BTreeMap<String, f64>,
    #[serde(default)]
    pub shape: ShapeSetup,
    #[serde(default)]
    pub settings: Settings,
    /// the object's orientation as it is shown: object -> world, row by row
    pub orientation: [f64; 9],
    /// how far the lights have circled by themselves, radians
    #[serde(default)]
    pub light_turn: f64,
    /// the animation clock, seconds
    #[serde(default)]
    pub clock: f64,
    /// what is animated: knob key -> its track
    #[serde(default)]
    pub tracks: BTreeMap<String, Track>,
    #[serde(default)]
    pub render: RenderOpts,
    /// the lights, when they have been changed from the family's own (left out otherwise, and a file
    /// that leaves it out means the family's own)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig: Option<RigSetup>,
    /// the environment picture: its path, relative to the program's folder when the file is in it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdri: Option<String>,
    /// The layers: every one's placement, the shapes of those after the first, and the options of their
    /// combinations. Absent for a single shape (`family`, `preset`, `values` and `shape` above are the
    /// first layer's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<crate::layers::LayersSetup>,
    /// The materials of the bodies that have one of their own: the number of the body's lowest layer
    /// ("2" to "4") -> the settings that make up its material, by name. Absent while every body is made
    /// of the scene's material.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub bodies: BTreeMap<String, BTreeMap<String, f64>>,
}

impl State {
    /// Everything about the present setup that is worth keeping.
    pub fn setup(&self) -> Setup {
        let (f, p) = (self.layer_family(0), self.layer_preset(0));
        let st = self.layer_shape(0).expect("there is always a first layer");
        Setup {
            version: VERSION,
            family: f.id.to_string(),
            preset: p.id.to_string(),
            values: p.params.iter().zip(&st.vals).map(|(d, v)| (d.id.to_string(), *v)).collect(),
            shape: ShapeSetup { thick: st.thick, len: st.len, melt: st.melt, puff: st.puff, travel: st.travel, beads: st.beads, bead_period: st.bead_period, res: st.res, anim_res: st.anim_res, model: st.model.clone() },
            settings: self.p.clone(),
            orientation: self.r,
            light_turn: self.lphi,
            clock: self.clock,
            tracks: self.tracks.clone(),
            render: self.render.clone(),
            rig: (!self.rig_is_default()).then(|| RigSetup::from(self.rig())),
            hdri: self.hdri.clone(),
            layers: self.layers_setup(),
            bodies: (1..self.body_mats.len())
                .filter_map(|l| self.body_mats[l].as_ref().map(|m| ((l + 1).to_string(), crate::material::body_names().filter_map(|n| m.get(n).map(|v| (n.to_string(), v))).collect())))
                .collect(),
        }
    }

    /// Put a saved setup up. Fails, changing nothing, if it names a shape this version does not have.
    pub fn apply_setup(&mut self, s: &Setup) -> Result<(), String> {
        if s.version > VERSION {
            return Err(format!("this setup was saved by a newer version (format {}, this program reads up to {})", s.version, VERSION));
        }
        let fam = self.families.iter().position(|f| f.id == s.family).ok_or_else(|| format!("no family called \"{}\"", s.family))?;
        let preset = self.families[fam].presets.iter().position(|p| p.id == s.preset).ok_or_else(|| format!("no shape called \"{}\" among the {}", s.preset, self.families[fam].name))?;
        if let Some(l) = &s.layers { self.check_layers_setup(l)?; }
        self.one_layer();
        self.set_scene(fam);
        self.choose(preset);
        let ids: Vec<&'static str> = self.preset().params.iter().map(|d| d.id).collect();
        let st = self.shape_mut();
        for (i, id) in ids.iter().enumerate() {
            if let Some(v) = s.values.get(*id) { st.vals[i] = *v; }
        }
        let sh = &s.shape;
        st.thick = sh.thick;
        st.len = sh.len;
        st.melt = sh.melt;
        st.puff = sh.puff;
        st.travel = sh.travel;
        st.beads = sh.beads;
        st.bead_period = sh.bead_period;
        st.res = crate::state::valid_res(sh.res as f64);
        st.anim_res = crate::state::valid_res(sh.anim_res as f64);
        st.model = sh.model.clone();
        if let Some(l) = &s.layers { self.apply_layers_setup(l); }
        match &s.rig {
            Some(r) => *self.rig_mut() = Rig::from(r),
            None => self.reset_rig(),
        }
        self.set_hdri(s.hdri.clone());
        self.p = s.settings.clone();
        crate::camera::sync(&mut self.p);
        // each body's own material: the scene's settings with the body's laid over them
        self.body_mats = Default::default();
        for (layer, values) in &s.bodies {
            let Some(l) = layer.parse::<usize>().ok().filter(|l| (2..=self.body_mats.len()).contains(l)) else { continue };
            self.set_body_own(l - 1, true);
            for (name, v) in values { self.body_set(l - 1, name, *v); }
        }
        self.render = s.render.clone();
        self.tracks = s.tracks.clone();
        self.playing = false;
        self.clock = s.clock;
        self.lphi = s.light_turn;
        // the orientation exactly as saved, unless it is not a rotation (a file edited by hand): then the nearest one
        self.theta = 0.0;
        let square = crate::math::orthonormalise(&s.orientation);
        let exact = (0..9).all(|i| (square[i] - s.orientation[i]).abs() < 1e-9);
        self.r = if exact { s.orientation } else { square };
        self.r0 = self.r;
        self.want = Some(Detail::Full);
        self.cut = true;
        Ok(())
    }

    pub fn setup_json(&self) -> String {
        serde_json::to_string_pretty(&self.setup()).expect("a setup always serialises")
    }
    pub fn apply_setup_json(&mut self, text: &str) -> Result<(), String> {
        let s: Setup = serde_json::from_str(text).map_err(|e| format!("not a setup file: {e}"))?;
        self.apply_setup(&s)
    }
}
