//! Polygon models: shapes that are files. A mesh file is turned into the same distance volume every
//! other shape is, so it is glass, takes every material, and can be a layer like any generator.
//!
//! The family's presets are the sample models that ship in the program's `models` folder, and one more,
//! `file`, which stands for whatever file the user has chosen (its path is kept with the shape:
//! `ShapeState::model`). Every preset has the same sliders: the model's own controls.

use super::*;
use crate::mesh::{MeshJob, SMOOTH_STEP};
use crate::state::{Knob, KnobRef, ShapeState, State};

/// The family's id, and the id of the preset that stands for the user's own file.
pub const FAMILY: &str = "models";
pub const FILE: &str = "file";

/// What the `smooth` slider holds until a model has been read: "as the mesh itself suggests". How far a
/// model's detail should be melted depends on how it is cut into facets, which is known only once the
/// file has been examined; the amount found then is put on the slider in place of this.
pub const SMOOTH_UNSET: f64 = -1.0;

/// A sample model: id, name, its file below the program's folder, whether its up axis is z, whether it
/// starts as a shell whatever the examiner says, the orientation it first appears in, and what to say of it.
struct Sample {
    id: &'static str,
    name: &'static str,
    file: &'static str,
    up_z: bool,
    shell: bool,
    rot: [f64; 3],
    eq: [&'static str; 2],
    note: &'static str,
}

const SAMPLES: [Sample; 5] = [
    Sample {
        id: "teapot", name: "Utah teapot", file: "models/teapot.obj", up_z: false, shell: false, rot: [18.0, -28.0, 0.0],
        eq: ["Newell's 28 Bézier patches, 1975", "89,600 triangles · OBJ"],
        note: "As Newell left it: no bottom, a gap round the lid, spout and handle pushed into the body. As a solid the gaps are closed; as a shell it is the hollow pot.",
    },
    Sample {
        id: "owl", name: "Owl", file: "models/owl.ply", up_z: false, shell: false, rot: [8.0, -24.0, 0.0],
        eq: ["a figure as a scanner leaves one:", "four holes underneath · 80,483 triangles · PLY"],
        note: "Its surface stops at four ragged holes in its base, where a scanner could not have seen. They are found and closed.",
    },
    Sample {
        id: "goblet", name: "Goblet", file: "models/goblet.ply", up_z: false, shell: true, rot: [14.0, 20.0, 0.0],
        eq: ["one sheet turned about an axis,", "open at the mouth · 13,824 triangles · PLY"],
        note: "A surface with no inside: drawn as a shell, with the wall's thickness on a slider. As a solid the bowl is filled.",
    },
    Sample {
        id: "gear", name: "Gear", file: "models/gear.stl", up_z: false, shell: false, rot: [58.0, 20.0, 10.0],
        eq: ["an outline pushed through a thickness", "560 triangles · STL"],
        note: "Flat faces and sharp edges: what a part drawn in a CAD program looks like. Nothing of it is melted.",
    },
    Sample {
        id: "pawn", name: "Pawn", file: "models/pawn.glb", up_z: false, shell: false, rot: [16.0, 25.0, 0.0],
        eq: ["two closed pieces, one standing on the other", "2,060 triangles · GLB"],
        note: "The pawn and its plinth are separate pieces that touch: together they are one solid.",
    },
];

/// The sliders every model has. `mode` and `up` are choices rather than amounts: whole steps. `smooth`
/// starts unset: the amount the model's own facets call for is put there when the file has been read.
fn params(up_z: bool, shell: bool) -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("size", "size", 1.0, 0.25, 1.1, 0.005),
        ParamDef::labelled("smooth", "smooth", SMOOTH_UNSET, 0.0, 0.08, SMOOTH_STEP),
        ParamDef::labelled("wall", "wall thickness", 0.04, 0.01, 0.3, 0.002),
        ParamDef::labelled("mode", "solid or shell", if shell { 2.0 } else { 0.0 }, 0.0, 2.0, 1.0),
        ParamDef::labelled("up", "up axis", if up_z { 1.0 } else { 0.0 }, 0.0, 1.0, 1.0),
    ]
}

/// What the GPU side is to build for a model as its sliders stand.
pub fn job(p: &Preset, st: &ShapeState, file: &str) -> MeshJob {
    let v = |id: &str, or: f64| p.param_index(id).and_then(|i| st.vals.get(i).copied()).unwrap_or(or);
    MeshJob {
        file: if file.is_empty() { st.model.clone() } else { file.to_string() },
        up_z: v("up", 0.0) > 0.5,
        mode: (v("mode", 0.0) + 0.5).floor().clamp(0.0, 2.0) as u8,
        wall: v("wall", 0.04),
        size: v("size", 1.0).clamp(0.05, 2.0),
        smooth: v("smooth", SMOOTH_UNSET),
        inflate: st.puff,
        full_res: st.res,
    }
}

/// The model's family and its `file` preset, as places in the lists.
pub fn places(families: &[Family]) -> Option<(usize, usize)> {
    let f = families.iter().position(|f| f.id == FAMILY)?;
    Some((f, families[f].presets.iter().position(|p| p.id == FILE)?))
}

/// `model=<path>` given when the program starts: the shapes begin with that file up. Returns the
/// family to show.
pub fn start(families: &[Family], shapes: &mut [ShapeState], args: &[(String, String)]) -> Option<usize> {
    let path = crate::state::arg(args, "model").filter(|p| !p.is_empty())?;
    let (f, p) = places(families)?;
    let st = shapes.get_mut(f)?;
    // (a preset named as well keeps its own file; the path then waits with the `file` preset)
    if crate::state::arg(args, "preset").is_none_or(|id| id == FILE || families[f].preset(id).is_none()) {
        st.preset = p;
        st.vals = families[f].presets[p].params.iter().map(|d| crate::state::arg(args, &format!("p.{}", d.id)).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite()).unwrap_or(d.value)).collect();
    }
    st.model = path.trim().to_string();
    Some(f)
}

impl State {
    /// Show a model file as the selected layer's shape: the family of models, its `file` preset, and
    /// this path (a full one, or one below the program's folder). A file other than the one that was
    /// up starts with its own smoothing, whatever the slider said for the last.
    pub fn set_model(&mut self, path: &str) -> bool {
        let Some((f, p)) = places(&self.families) else { return false };
        if self.fam != f { self.set_scene(f); }
        if self.shape().preset != p { self.choose(p); }
        let path = path.trim().to_string();
        if self.shape().model != path {
            let smooth = self.families[f].presets[p].param_index("smooth");
            let st = self.shape_mut();
            if let Some(v) = smooth.and_then(|i| st.vals.get_mut(i)) { *v = SMOOTH_UNSET; }
            st.model = path;
        }
        self.want = Some(crate::state::Detail::Full);
        true
    }
    /// The file a layer's shape is, if it is a model: the path as the state has it ("" for the `file`
    /// shape before any file has been chosen).
    pub fn model_file(&self, layer: usize) -> Option<String> {
        let p = self.layer_preset(layer);
        match &p.shape {
            Shape::Mesh { file } if file.is_empty() => self.layer_shape(layer).map(|s| s.model.clone()),
            Shape::Mesh { file } => Some(file.to_string()),
            _ => None,
        }
    }
    /// Is the selected layer's shape a model?
    pub fn is_model(&self) -> bool {
        matches!(self.preset().shape, Shape::Mesh { .. })
    }
    /// Does a layer's shape wait for a file before anything of it can be built? (The `file` shape of
    /// the models, with no file chosen yet.)
    pub fn needs_file(&self, layer: usize) -> bool {
        self.model_file(layer).is_some_and(|f| f.trim().is_empty())
    }
    /// A layer's shape by name: the preset's, or for a model file of the user's the file's own name,
    /// without its folder and its ending.
    pub fn shape_name(&self, layer: usize) -> String {
        let p = self.layer_preset(layer);
        if let (Shape::Mesh { file }, Some(st)) = (&p.shape, self.layer_shape(layer)) {
            if file.is_empty() {
                let path = st.model.trim().replace('\\', "/");
                let name = path.rsplit('/').next().unwrap_or("");
                let stem = name.rsplit_once('.').map_or(name, |(s, _)| s);
                if !stem.is_empty() { return stem.to_string(); }
            }
        }
        p.name.to_string()
    }
    /// The names of a slider's settings, when it is a choice between a few things rather than an
    /// amount (a model's `mode` and `up`): what to write on the buttons, in the order of the slider's
    /// whole numbers from its lowest. None for an ordinary slider.
    pub fn knob_choices(&self, k: &Knob) -> Option<Vec<String>> {
        let layer = match &k.r {
            KnobRef::Of(layer, _) => *layer,
            KnobRef::Param(_) => self.stack.sel,
            _ => return None,
        };
        if self.layer_family(layer).id == super::trees::FAMILY { return super::trees::choices(&k.id); }
        if self.layer_family(layer).id == super::text3d::FAMILY { return super::text3d::choices(&k.id); }
        if self.layer_family(layer).id == super::branching::FAMILY { return super::branching::choices(&k.id); }
        if self.layer_family(layer).kind != Kind::Mesh { return None; }
        let names: &[&str] = match k.id.as_str() {
            "mode" => &["as found", "solid", "shell"],
            "up" => &["y up", "z up"],
            _ => return None,
        };
        Some(names.iter().map(|s| s.to_string()).collect())
    }
}

pub fn family() -> Family {
    let mut presets: Vec<Preset> = SAMPLES
        .iter()
        .map(|s| Preset { eq: s.eq.to_vec(), note: s.note, params: params(s.up_z, s.shell), rot: Some(s.rot), ..Preset::new(s.id, s.name, Shape::Mesh { file: s.file }) })
        .collect();
    presets.push(Preset {
        eq: vec!["any OBJ, STL, PLY, glTF or GLB file"],
        note: "Load a file, or drop one on the window. Gaps in its surface are found and closed; a surface with no inside becomes a shell.",
        params: params(false, false),
        rot: Some([16.0, 25.0, 0.0]),
        ..Preset::new(FILE, "Your file", Shape::Mesh { file: "" })
    });
    Family {
        step_scale: 0.85,
        steps_in: 64,
        layers: 2,
        presets,
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.24, 0.11, 0.05])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(4.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([16.0, 25.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        // Models have broad smooth faces and flat ones alike: broad soft lights that fade to their edges,
        // so that a face takes its tone from where in a light it looks, and two small bright ones for glints.
        lights: vec![
            Light { az:   40.0, el:  20.0, w: 24.0, h: 38.0, rgb: [3.6, 3.5, 3.3],   soft: 0.9 },
            Light { az:  -58.0, el:  -6.0, w: 18.0, h: 44.0, rgb: [0.45, 1.25, 3.6], soft: 0.9 },
            Light { az: -140.0, el:   5.0, w: 15.0, h: 44.0, rgb: [4.4, 1.45, 0.24], soft: 0.9 },
            Light { az:  150.0, el:  29.0, w: 28.0, h: 20.0, rgb: [2.6, 2.5, 2.45],  soft: 0.9 },
            Light { az:    0.0, el:  85.0, w: 38.0, h: 38.0, rgb: [0.8, 0.9, 1.1],   soft: 1.0 },
            Light { az:    0.0, el: -80.0, w: 34.0, h: 34.0, rgb: [0.5, 0.22, 0.05], soft: 1.0 },
            Light { az:   95.0, el:  46.0, w:  5.0, h:  5.0, rgb: [6.0, 6.0, 6.0],   soft: 0.5 },
            Light { az: -100.0, el: -36.0, w:  4.0, h:  8.0, rgb: [4.5, 5.0, 6.0],   soft: 0.5 },
        ],
        ..Family::new(FAMILY, "Models", Kind::Mesh)
    }
}
