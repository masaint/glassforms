//! A whole setup as a file: which shape, every slider, the view, the lights, what is animated and how,
//! and the video options. Saved as JSON so that it can be read and edited by hand.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::anim::Track;
use crate::pbr::MaterialSpec;
use crate::presets::Light;
use crate::settings::{RenderOpts, Settings};
use crate::state::{BuildRequest, Rig, State, LIGHT_SOFT};

pub const VERSION: u32 = 1;
pub const CANONICAL_VERSION: u32 = 2;

// Native setups saved before the optical-disc recipe stored positions 58..103
// for aluminium through bisque. The disc was inserted immediately before
// aluminium in the same release that added the grating controls. Use keys that
// the old writer included so a sparse, hand-authored setup keeps its indices.
fn predates_grating(has: impl Fn(&str) -> bool) -> bool {
    !has("grating") && ["mat", "metal", "brushAngle", "glazeHaze"].iter().all(|name| has(name))
}

fn update_legacy_recipe(index: &mut f64) {
    if (58.0..=103.0).contains(index) && index.fract() == 0.0 { *index += 1.0; }
}

fn read_scene_settings<'de, D: serde::Deserializer<'de>>(reader: D) -> Result<Settings, D::Error> {
    let object = serde_json::Map::deserialize(reader)?;
    let legacy = predates_grating(|name| object.contains_key(name));
    let mut settings: Settings = serde_json::from_value(serde_json::Value::Object(object))
        .map_err(serde::de::Error::custom)?;
    if legacy { update_legacy_recipe(&mut settings.mat); }
    Ok(settings)
}

type BodySettings = BTreeMap<String, BTreeMap<String, f64>>;

/// Lossless material ownership in a running state. Missing secondary bodies inherit `scene` live.
/// Legacy `Settings` and construction maps remain projections for the current renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalMaterials {
    pub scene: MaterialSpec,
    pub bodies: [Option<Box<MaterialSpec>>; crate::material::MAX_BODIES],
}
impl CanonicalMaterials {
    pub fn new(scene: MaterialSpec) -> Self {
        Self { scene, bodies: Default::default() }
    }
}

/// The same ownership as setup JSON: body keys are their lowest layer, counted from one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalMaterialsSetup {
    pub scene: MaterialSpec,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub bodies: BTreeMap<String, MaterialSpec>,
}

fn read_body_settings<'de, D: serde::Deserializer<'de>>(reader: D) -> Result<BodySettings, D::Error> {
    let mut bodies = BodySettings::deserialize(reader)?;
    for values in bodies.values_mut() {
        if predates_grating(|name| values.contains_key(name)) {
            if let Some(index) = values.get_mut("mat") { update_legacy_recipe(index); }
        }
    }
    Ok(bodies)
}

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
    /// how the shape is traced: from its volume (left out) or from its formula ("sdf")
    #[serde(default, skip_serializing_if = "crate::sdf::Trace::is_volume")]
    pub trace: crate::sdf::Trace,
    /// the surface studio, as setups of before kept it (read, never written: it is now a modifier)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sculpt: Option<crate::sculpt::Sculpt>,
    /// the modifiers, the surface lab among them (left out while there are none)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mods: Option<crate::mods::Mods>,
    /// the shape's strokes as a csg object's first part (left out while there are none)
    #[serde(default, skip_serializing_if = "crate::csg::Strokes::is_none")]
    pub strokes: crate::csg::Strokes,
}
impl ShapeSetup {
    /// The modifiers it holds: its own, or a setup of before's surface studio as a surface lab.
    pub fn mods_or_lab(&self) -> crate::mods::Mods {
        match &self.mods {
            Some(m) => m.clone(),
            None => crate::mods::Mods::with_lab(self.sculpt.clone()),
        }
    }
}
impl Default for ShapeSetup {
    fn default() -> Self {
        Self { thick: 1.0, len: 1.0, melt: 0.006, puff: 0.0, travel: 0.0, beads: 0.0, bead_period: 2.0, res: crate::state::RES_FULL, anim_res: crate::state::RES_ANIM, model: String::new(), trace: Default::default(), sculpt: None, mods: None, strokes: Default::default() }
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
    #[serde(default, deserialize_with = "read_scene_settings")]
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
    /// The original studio family when a hierarchy reorder changed the first layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig_family: Option<String>,
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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty", deserialize_with = "read_body_settings")]
    pub bodies: BTreeMap<String, BTreeMap<String, f64>>,
    /// Lossless scene and full-body materials. Present only in opt-in format 2; construction nodes
    /// carry their sparse `material_v2` patches beside their legacy renderer maps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical: Option<CanonicalMaterialsSetup>,
    /// The lamps standing in the scene (absent while there are none).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lamps: Vec<crate::lamps::Lamp>,
    /// The ten camera slots, empty or a stored view (absent while none is stored).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<Option<crate::history::ViewSlot>>,
    /// The tree of formula parts joined to the scene: its shape (the parts, in order), the object that
    /// places it, and whether it is traced from the volume it is measured into (each absent while unused).
    /// This is the tree itself; the volume it is measured into is only ever made from it, never read back.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub csg: Vec<crate::csg::Part>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction: Option<crate::construction::Construction>,
    #[serde(default, skip_serializing_if = "crate::layers::Placement::is_identity")]
    pub csg_place: crate::layers::Placement,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub csg_coarse: bool,
    /// the first object is a volume object made from a csg object's tree (Csg::baked)
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub csg_baked: bool,
    /// there is no object at all (the scene shows the backdrop and the floor alone)
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub empty: bool,
    /// how the viewport draws (view_mode.rs; absent while it is the full render with the default matcap)
    #[serde(default, skip_serializing_if = "crate::view_mode::ViewSettings::is_default")]
    pub view: crate::view_mode::ViewSettings,
}

fn tree_has_canonical(tree: &Option<crate::construction::Construction>) -> bool {
    fn any(nodes: &[crate::construction::Node]) -> bool {
        nodes.iter().any(|node| {
            node.material_v2.is_some()
                || matches!(&node.source, crate::construction::Source::Group(children) if any(children))
        })
    }
    tree.as_ref().is_some_and(|tree| any(&tree.roots))
}

fn setup_has_canonical_nodes(s: &Setup) -> bool {
    tree_has_canonical(&s.construction)
        || s.layers.as_ref().is_some_and(|layers| layers.more.iter().any(|layer| tree_has_canonical(&layer.construction)))
}

struct NodeProjection {
    layer: usize,
    path: Vec<usize>,
    patch: crate::pbr::MaterialPatch,
    values: BTreeMap<String, f64>,
}

struct PreparedCanonical {
    owner: CanonicalMaterials,
    settings: Settings,
    bodies: [Option<Box<Settings>>; crate::material::MAX_BODIES],
    nodes: Vec<NodeProjection>,
}

fn bridge_error(context: &str, error: impl std::fmt::Display) -> String {
    format!("canonical {context} cannot be rendered by the current legacy bridge: {error}")
}

fn material_projection(base: &Settings, exported: &Settings) -> Settings {
    let mut projected = base.clone();
    // RGB absorption is authoritative but is not one of the scalar `body_names` controls.
    projected.absorb = exported.absorb;
    for name in crate::material::body_names() {
        // Absorption is authoritative; writing its derived tint back would add an exp/ln round trip.
        if crate::settings::Settings::DERIVED.contains(&name) { continue; }
        if let Some(value) = exported.get(name) { projected.set(name, value); }
    }
    projected
}

fn effective_base<'a>(owner: &'a CanonicalMaterials, bodies: &[Vec<usize>], layer: usize) -> &'a MaterialSpec {
    let first = bodies.iter().find(|body| body.contains(&layer)).and_then(|body| body.first()).copied().unwrap_or(0);
    // (only the first MAX_BODIES layers can carry a material of their own, as `body_material_spec` refuses the rest: the shader reads at most
    // MAX_BODIES bodies, so a body that starts at a later layer inherits the scene's material rather than indexing past the array)
    if first == 0 { &owner.scene } else { owner.bodies.get(first).and_then(|b| b.as_deref()).unwrap_or(&owner.scene) }
}

fn clear_legacy_node_materials(nodes: &mut [crate::construction::Node]) {
    for node in nodes {
        node.material = None;
        if let crate::construction::Source::Group(children) = &mut node.source {
            clear_legacy_node_materials(children);
        }
    }
}

fn prepare_nodes(
    nodes: &[crate::construction::Node],
    base: &MaterialSpec,
    layer: usize,
    path: &mut Vec<usize>,
    reconcile: bool,
    out: &mut Vec<NodeProjection>,
) -> Result<(), String> {
    for (index, node) in nodes.iter().enumerate() {
        path.push(index);
        if let Some(saved) = &node.material_v2 {
            let where_ = format!("construction material at layer {}, node {}", layer + 1, path.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join("."));
            let patch = if reconcile {
                saved.clone().reconciled_legacy_activity(base, None).map_err(|error| bridge_error(&where_, error))?
            } else {
                saved.clone()
            };
            let values = patch.to_legacy_map(base).map_err(|error| bridge_error(&where_, error))?;
            out.push(NodeProjection { layer, path: path.clone(), patch, values });
        }
        if let crate::construction::Source::Group(children) = &node.source {
            prepare_nodes(children, base, layer, path, reconcile, out)?;
        }
        path.pop();
    }
    Ok(())
}

/// Convert or validate every sparse construction override during explicit opt-in.
fn convert_nodes(
    nodes: &[crate::construction::Node],
    base: &MaterialSpec,
    layer: usize,
    path: &mut Vec<usize>,
    out: &mut Vec<(usize, Vec<usize>, Option<crate::pbr::MaterialPatch>, BTreeMap<String, f64>)>,
) -> Result<(), String> {
    for (index, node) in nodes.iter().enumerate() {
        path.push(index);
        let where_ = format!("construction material at layer {}, node {}", layer + 1, path.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join("."));
        if let Some(patch) = &node.material_v2 {
            let values = patch.to_legacy_map(base).map_err(|error| bridge_error(&where_, error))?;
            out.push((layer, path.clone(), None, values));
        } else if let Some(values) = &node.material {
            let patch = crate::pbr::MaterialPatch::from_legacy_construction_map(base, values)
                .map_err(|error| format!("cannot convert legacy {where_}: {error}"))?;
            let projected = patch.to_legacy_map(base).map_err(|error| bridge_error(&where_, error))?;
            out.push((layer, path.clone(), Some(patch), projected));
        }
        if let crate::construction::Source::Group(children) = &node.source {
            convert_nodes(children, base, layer, path, out)?;
        }
        path.pop();
    }
    Ok(())
}

/// Export every canonical owner before `apply_setup` changes anything. The saved legacy material
/// fields are compatibility mirrors, never an additional source of canonical ownership.
fn prepare_canonical(s: &Setup) -> Result<Option<PreparedCanonical>, String> {
    let Some(saved) = &s.canonical else {
        if setup_has_canonical_nodes(s) {
            return Err("canonical construction materials require the canonical scene in setup format 2".into());
        }
        return Ok(None);
    };
    let settings = material_projection(&s.settings, &saved.scene.to_legacy_settings().map_err(|error| bridge_error("scene material", error))?);
    let mut owner = CanonicalMaterials::new(saved.scene.clone());
    let mut bodies: [Option<Box<Settings>>; crate::material::MAX_BODIES] = Default::default();
    let mut seen_bodies = std::collections::BTreeSet::new();
    for (key, spec) in &saved.bodies {
        let layer = key.parse::<usize>().ok().filter(|layer| (2..=crate::material::MAX_BODIES).contains(layer))
            .ok_or_else(|| format!("canonical body \"{key}\" is not a secondary body layer from 2 through {}", crate::material::MAX_BODIES))?;
        let index = layer - 1;
        if !seen_bodies.insert(index) {
            return Err(format!("canonical body {layer} is named more than once by textual aliases"));
        }
        let projected = spec.to_legacy_settings().map_err(|error| bridge_error(&format!("body {layer} material"), error))?;
        owner.bodies[index] = Some(Box::new(spec.clone()));
        bodies[index] = Some(Box::new(projected));
    }
    let body_layers = s.layers.as_ref().map(|layers| {
        let stack = crate::layers::Stack::from_setup(&layers.stack);
        stack.bodies().iter().map(|mask| crate::layers::members(*mask)).collect::<Vec<_>>()
    }).unwrap_or_else(|| vec![vec![0]]);
    let mut nodes = Vec::new();
    if let Some(tree) = &s.construction {
        prepare_nodes(&tree.roots, effective_base(&owner, &body_layers, 0), 0, &mut Vec::new(), false, &mut nodes)?;
    }
    if let Some(layers) = &s.layers {
        for (index, layer) in layers.more.iter().enumerate() {
            if let Some(tree) = &layer.construction {
                let layer = index + 1;
                prepare_nodes(&tree.roots, effective_base(&owner, &body_layers, layer), layer, &mut Vec::new(), false, &mut nodes)?;
            }
        }
    }
    Ok(Some(PreparedCanonical { owner, settings, bodies, nodes }))
}

impl State {
    /// Boolean topology can move a construction node to a different effective body owner without
    /// moving its layer. Re-resolve every sparse patch before the changed setup can be saved again.
    pub(crate) fn material_owners_changed(&mut self) {
        if let Some(owner) = self.canonical.clone() {
            let _ = self.install_canonical_owner(owner);
        }
    }

    /// Preflight a complete live owner against the strict renderer bridge and every dependent
    /// construction patch. Nothing is committed until all owners and nodes resolve successfully.
    pub(crate) fn install_canonical_owner(&mut self, owner: CanonicalMaterials) -> Result<(), String> {
        let settings = material_projection(&self.p, &owner.scene.to_legacy_settings().map_err(|error| bridge_error("scene material", error))?);
        let mut bodies: [Option<Box<Settings>>; crate::material::MAX_BODIES] = Default::default();
        for layer in 1..owner.bodies.len() {
            if let Some(spec) = &owner.bodies[layer] {
                bodies[layer] = Some(Box::new(spec.to_legacy_settings().map_err(|error| bridge_error(&format!("body {} material", layer + 1), error))?));
            }
        }
        let body_layers = self.bodies();
        let mut nodes = Vec::new();
        for layer in 0..self.layer_count() {
            if let Some(tree) = &self.layer_csg(layer).construction {
                prepare_nodes(&tree.roots, effective_base(&owner, &body_layers, layer), layer, &mut Vec::new(), true, &mut nodes)?;
            }
        }
        self.p = settings;
        self.body_mats = bodies;
        self.canonical = Some(owner);
        for projection in nodes {
            if let Some(node) = self.layer_csg_mut(projection.layer)
                .and_then(|csg| csg.construction.as_mut())
                .and_then(|tree| tree.node_mut(&projection.path))
            {
                node.material_v2 = Some(projection.patch);
                node.material = Some(projection.values);
            }
        }
        Ok(())
    }

    fn legacy_canonical_materials(&self) -> CanonicalMaterials {
        let mut owner = CanonicalMaterials::new(MaterialSpec::from_legacy(&self.p, crate::pbr::MaterialId::from("material:scene")));
        for layer in 1..self.body_mats.len() {
            if let Some(settings) = &self.body_mats[layer] {
                owner.bodies[layer] = Some(Box::new(MaterialSpec::from_legacy(settings, crate::pbr::MaterialId::from(format!("material:body:{}", layer + 1)))));
            }
        }
        owner
    }

    /// The effective full-body canonical material. An absent secondary owner inherits the scene live;
    /// a legacy-only state is converted with the same `MaterialSpec::from_legacy` used for opt-in.
    pub fn body_material_spec(&self, layer: usize) -> MaterialSpec {
        if let Some(owner) = &self.canonical {
            return if layer == 0 {
                owner.scene.clone()
            } else {
                owner.bodies.get(layer).and_then(Option::as_deref).unwrap_or(&owner.scene).clone()
            };
        }
        let id = if layer == 0 || !self.body_own(layer) {
            crate::pbr::MaterialId::from("material:scene")
        } else {
            crate::pbr::MaterialId::from(format!("material:body:{}", layer + 1))
        };
        MaterialSpec::from_legacy(self.body_settings(layer), id)
    }

    /// Apply a typed patch to a full body, creating canonical ownership on the first edit if needed.
    /// The complete candidate and all dependent sparse construction patches are preflighted before
    /// anything live changes. An inherited secondary body becomes a full owner only after success.
    pub fn apply_body_material_patch(&mut self, layer: usize, patch: crate::pbr::MaterialPatch) -> Result<(), String> {
        if layer >= crate::material::MAX_BODIES { return Err(format!("layer {} cannot have a material of its own: only the first {} can", layer + 1, crate::material::MAX_BODIES)); }
        if layer >= self.body_mats.len() {
            return Err(format!("no material body at layer {}", layer + 1));
        }
        let base = self.body_material_spec(layer);
        let mut edited = patch.apply(&base).map_err(|error| format!("cannot apply canonical body material: {error}"))?;
        if layer > 0 {
            edited.id = crate::pbr::MaterialId::from(format!("material:body:{}", layer + 1));
        }
        let mut owner = self.canonical.clone().unwrap_or_else(|| self.legacy_canonical_materials());
        if layer == 0 {
            owner.scene = edited;
        } else {
            owner.bodies[layer] = Some(Box::new(edited));
        }
        self.install_canonical_owner(owner)
    }

    /// Everything about the present setup that is worth keeping.
    pub fn setup(&self) -> Setup {
        let has_canonical_nodes = (0..self.layer_count()).any(|layer| tree_has_canonical(&self.layer_csg(layer).construction));
        let inferred = (self.canonical.is_none() && has_canonical_nodes).then(|| self.legacy_canonical_materials());
        let canonical_owner = self.canonical.as_ref().or(inferred.as_ref());
        // Canonical owners are authoritative on every v2 save, not only after a reopen. In particular,
        // perceptual roughness conversion may legitimately change the last bit of its legacy mirror.
        let canonical_settings = canonical_owner.map(|owner| {
            let exported = owner.scene.to_legacy_settings().expect("live canonical scene must remain renderable");
            material_projection(&self.p, &exported)
        });
        let canonical_bodies = canonical_owner.map(|owner| {
            (1..owner.bodies.len()).filter_map(|layer| owner.bodies[layer].as_ref().map(|spec| {
                let settings = spec.to_legacy_settings().expect("live canonical body must remain renderable");
                ((layer + 1).to_string(), crate::material::body_names().filter_map(|name| settings.get(name).map(|value| (name.to_string(), value))).collect())
            })).collect()
        });
        let canonical = canonical_owner.map(|owner| CanonicalMaterialsSetup {
            scene: owner.scene.clone(),
            bodies: (1..owner.bodies.len()).filter_map(|layer| owner.bodies[layer].as_ref().map(|spec| ((layer + 1).to_string(), spec.as_ref().clone()))).collect(),
        });
        let (f, p) = (self.layer_family(0), self.layer_preset(0));
        let st = self.layer_shape(0).expect("there is always a first layer");
        Setup {
            version: if canonical.is_some() { CANONICAL_VERSION } else { VERSION },
            family: f.id.to_string(),
            preset: p.id.to_string(),
            values: p.params.iter().zip(&st.vals).map(|(d, v)| (d.id.to_string(), *v)).collect(),
            shape: ShapeSetup { thick: st.thick, len: st.len, melt: st.melt, puff: st.puff, travel: st.travel, beads: st.beads, bead_period: st.bead_period, res: st.res, anim_res: st.anim_res, model: st.model.clone(), trace: st.trace, sculpt: None, mods: (!st.mods.is_empty()).then(|| st.mods.clone()), strokes: st.strokes },
            settings: canonical_settings.unwrap_or_else(|| self.p.clone()),
            orientation: self.r,
            light_turn: self.lphi,
            clock: self.clock,
            tracks: self.tracks.clone(),
            render: self.render.clone(),
            rig: (self.rig_family.is_some() || !self.rig_is_default()).then(|| RigSetup::from(self.rig())),
            rig_family: self.rig_family.map(|i| self.families[i].id.to_owned()),
            hdri: self.hdri.clone(),
            lamps: self.lamps.clone(),
            views: if self.views.iter().any(|v| v.is_some()) { self.views.clone() } else { Vec::new() },
            // (the first object's tree: another's is kept with its layer)
            construction: self.layer_csg(0).construction.clone(),
            csg: self.layer_csg(0).parts.clone(),
            csg_place: self.layer_csg(0).place,
            csg_coarse: self.layer_csg(0).coarse,
            csg_baked: self.layer_csg(0).baked,
            empty: self.empty,
            view: self.view.clone(),
            layers: self.layers_setup(),
            bodies: canonical_bodies.unwrap_or_else(|| (1..self.body_mats.len())
                .filter_map(|l| self.body_mats[l].as_ref().map(|m| ((l + 1).to_string(), crate::material::body_names().filter_map(|n| m.get(n).map(|v| (n.to_string(), v))).collect())))
                .collect()),
            canonical,
        }
    }

    /// Put a saved setup up. Fails, changing nothing, if it names a shape this version does not have.
    pub fn apply_setup(&mut self, s: &Setup) -> Result<(), String> {
        if s.version > CANONICAL_VERSION {
            return Err(format!("this setup was saved by a newer version (format {}, this program reads up to {})", s.version, CANONICAL_VERSION));
        }
        if s.version < CANONICAL_VERSION && (s.canonical.is_some() || setup_has_canonical_nodes(s)) {
            return Err(format!("canonical materials require setup format {CANONICAL_VERSION}"));
        }
        if s.version == CANONICAL_VERSION && s.canonical.is_none() {
            return Err(format!("setup format {CANONICAL_VERSION} requires canonical scene materials"));
        }
        let fam = self.families.iter().position(|f| f.id == s.family).ok_or_else(|| format!("no family called \"{}\"", s.family))?;
        let preset = self.families[fam].presets.iter().position(|p| p.id == s.preset).ok_or_else(|| format!("no shape called \"{}\" among the {}", s.preset, self.families[fam].name))?;
        if let Some(l) = &s.layers {
            self.check_layers_setup(l)?;
            if let Some(h) = &l.stack.hierarchy {
                let kinds: Vec<_> = std::iter::once(s.shape.trace).chain(l.more.iter().map(|m| m.shape.trace)).collect();
                h.validate(if s.empty { &[] } else { &kinds })?;
            }
        }
        let canonical = prepare_canonical(s)?;
        // Legacy setup import must not synchronise transitional settings into the previous v2 owner.
        // The fully preflighted owner is installed only after all scene/body projections are in place.
        self.canonical = None;
        self.edit_scope = Default::default();
        self.modifier_pivots.clear();
        self.construction_pivots.clear();
        self.one_layer();
        self.rig_family = None;
        self.set_scene(fam);
        self.choose(preset);
        let ids: Vec<(&'static str, f64, f64)> = self.preset().params.iter().map(|d| (d.id, d.lo, d.hi)).collect();
        let st = self.shape_mut();
        for (i, (id, lo, hi)) in ids.iter().enumerate() {
            if let Some(v) = s.values.get(*id) {
                st.vals[i] = crate::limits::hold(*v, crate::limits::widen(*lo, *hi));
            }
        }
        let sh = &s.shape;
        st.thick = sh.thick;
        st.len = sh.len;
        st.melt = sh.melt;
        st.puff = sh.puff;
        st.travel = sh.travel;
        st.beads = sh.beads;
        st.bead_period = sh.bead_period;
        st.hold();
        st.res = crate::state::valid_res(sh.res as f64);
        st.anim_res = crate::state::valid_res(sh.anim_res as f64);
        st.model = sh.model.clone();
        st.trace = sh.trace;
        st.mods = sh.mods_or_lab();
        st.strokes = sh.strokes;
        // (the first object's tree and object; which part the panel has selected stays, if there still is one)
        let sel = self.csg.sel.filter(|i| *i < s.csg.len());
        let whole = self.csg.whole && !s.csg.is_empty();
        self.csg = crate::csg::Csg { construction: s.construction.clone(), parts: s.csg.iter().take(crate::csg::MAX_PARTS).cloned().collect(), place: s.csg_place, coarse: s.csg_coarse, baked: s.csg_baked, sel, whole };
        if let Some(l) = &s.layers { self.apply_layers_setup(l); }
        // (no object: the one layer is hidden, standing for nothing)
        self.empty = s.empty;
        if s.empty { self.stack.layers[0].show = false; }
        self.rig_family = s.rig_family.as_deref().and_then(|id| self.families.iter().position(|f| f.id == id));
        match &s.rig {
            Some(r) => *self.rig_mut() = Rig::from(r),
            None => self.reset_rig(),
        }
        self.set_hdri(s.hdri.clone());
        self.lamps = s.lamps.iter().take(crate::lamps::MAX_LAMPS).copied().collect();
        for l in &mut self.lamps { l.hold(); }
        self.views = s.views.iter().take(crate::history::SLOTS).cloned().collect();
        self.view = s.view.clone();
        self.view.hold();
        self.lamps_rev += 1;
        self.p = canonical.as_ref().map_or_else(|| s.settings.clone(), |prepared| prepared.settings.clone());
        crate::camera::sync(&mut self.p);
        // (a file may hold anything: every setting to its hard limits, `limits`)
        self.p.hold_hard();
        // each body's own material: the scene's settings with the body's laid over them. In format 2
        // the full canonical bodies alone decide ownership; absent bodies inherit the canonical scene.
        self.body_mats = canonical.as_ref().map_or_else(Default::default, |prepared| prepared.bodies.clone());
        if canonical.is_none() {
            for (layer, values) in &s.bodies {
                let Some(l) = layer.parse::<usize>().ok().filter(|l| (2..=self.body_mats.len()).contains(l)) else { continue };
                self.set_body_own(l - 1, true);
                for (name, v) in values { self.body_set(l - 1, name, *v); }
            }
        }
        self.canonical = canonical.as_ref().map(|prepared| prepared.owner.clone());
        if let Some(prepared) = &canonical {
            for layer in 0..self.layer_count() {
                if let Some(tree) = self.layer_csg_mut(layer).and_then(|csg| csg.construction.as_mut()) {
                    clear_legacy_node_materials(&mut tree.roots);
                }
            }
            for projection in &prepared.nodes {
                if let Some(node) = self.layer_csg_mut(projection.layer)
                    .and_then(|csg| csg.construction.as_mut())
                    .and_then(|tree| tree.node_mut(&projection.path))
                {
                    node.material_v2 = Some(projection.patch.clone());
                    node.material = Some(projection.values.clone());
                }
            }
        }
        self.render = s.render.clone();
        self.tracks = s.tracks.clone();
        for t in self.tracks.values_mut() { t.hold(); }
        self.playing = false;
        self.clock = s.clock;
        self.lphi = s.light_turn;
        // the orientation exactly as saved, unless it is not a rotation (a file edited by hand): then the nearest one
        self.theta = 0.0;
        let square = crate::math::orthonormalise(&s.orientation);
        let exact = (0..9).all(|i| (square[i] - s.orientation[i]).abs() < 1e-9);
        self.r = if exact { s.orientation } else { square };
        self.r0 = self.r;
        self.request(BuildRequest::edit());
        self.cut = true;
        Ok(())
    }

    /// Opt the current materials into lossless canonical ownership without changing their rendering.
    /// Every legacy conversion and bridge export is checked before any canonical owner is installed.
    pub fn enable_canonical_materials(&mut self) -> Result<(), String> {
        let owner = self.canonical.clone().unwrap_or_else(|| self.legacy_canonical_materials());
        let scene_settings = owner.scene.to_legacy_settings().map_err(|error| bridge_error("scene material", error))?;
        let scene_settings = material_projection(&self.p, &scene_settings);
        let mut projected_bodies: [Option<Box<Settings>>; crate::material::MAX_BODIES] = Default::default();
        for (layer, spec) in owner.bodies.iter().enumerate().skip(1) {
            if let Some(spec) = spec {
                let settings = spec.to_legacy_settings().map_err(|error| bridge_error(&format!("body {} material", layer + 1), error))?;
                projected_bodies[layer] = Some(Box::new(settings));
            }
        }
        let body_layers = self.bodies();
        let mut converted = Vec::new();
        for layer in 0..self.layer_count() {
            if let Some(tree) = &self.layer_csg(layer).construction {
                convert_nodes(&tree.roots, effective_base(&owner, &body_layers, layer), layer, &mut Vec::new(), &mut converted)?;
            }
        }
        self.p = scene_settings;
        self.body_mats = projected_bodies;
        self.canonical = Some(owner);
        for (layer, path, patch, values) in converted {
            if let Some(node) = self.layer_csg_mut(layer)
                .and_then(|csg| csg.construction.as_mut())
                .and_then(|tree| tree.node_mut(&path))
            {
                if let Some(patch) = patch { node.material_v2 = Some(patch); }
                node.material = Some(values);
            }
        }
        Ok(())
    }

    /// Change a construction material through the current renderer's controls. In format 2 the typed
    /// patch is updated first, then its sparse legacy mirror; an invalid edit leaves both untouched.
    pub fn set_construction_material(
        &mut self,
        layer: usize,
        path: &[usize],
        material: Option<BTreeMap<String, f64>>,
    ) -> Result<(), String> {
        let mut patch = None;
        let mut projected = material.clone();
        if self.canonical.is_some() {
            if let Some(values) = &material {
                let bodies = self.bodies();
                let base = effective_base(self.canonical.as_ref().unwrap(), &bodies, layer).clone();
                let converted = crate::pbr::MaterialPatch::from_legacy_construction_map(&base, values)
                    .map_err(|error| format!("cannot convert legacy construction material: {error}"))?;
                projected = Some(converted.to_legacy_map(&base).map_err(|error| bridge_error("construction material", error))?);
                patch = Some(converted);
            }
        }
        let node = self.layer_csg_mut(layer)
            .and_then(|csg| csg.construction.as_mut())
            .and_then(|tree| tree.node_mut(path))
            .ok_or_else(|| format!("no construction material at layer {}, path {path:?}", layer + 1))?;
        node.material = projected;
        node.material_v2 = patch;
        Ok(())
    }

    /// Apply only controls actually changed by a construction UI edit. Unlike replacing the legacy
    /// mirror wholesale, this preserves untouched dormant values in an existing canonical patch.
    pub fn update_construction_material(
        &mut self,
        layer: usize,
        path: &[usize],
        values: BTreeMap<String, f64>,
    ) -> Result<(), String> {
        if values.is_empty() { return Ok(()); }
        if self.canonical.is_none() {
            let mut merged = self.layer_csg(layer).construction.as_ref()
                .and_then(|tree| tree.node(path)).and_then(|node| node.material.clone()).unwrap_or_default();
            merged.extend(values);
            return self.set_construction_material(layer, path, Some(merged));
        }
        let bodies = self.bodies();
        let base = effective_base(self.canonical.as_ref().unwrap(), &bodies, layer);
        let existing = self.layer_csg(layer).construction.as_ref()
            .and_then(|tree| tree.node(path)).and_then(|node| node.material_v2.clone())
            .unwrap_or_else(|| crate::pbr::MaterialPatch::new(base.id.clone()));
        let delta = crate::pbr::MaterialPatch::from_legacy_construction_map(base, &values)
            .map_err(|error| format!("cannot convert legacy construction material: {error}"))?;
        let previous = existing.apply(base).map_err(|error| bridge_error("construction material", error))?;
        let mut combined = existing;
        combined.properties.extend(delta.properties);
        combined.features.extend(delta.features);
        let patch = combined.reconciled_legacy_activity_from(base, &previous, Some(&values))
            .map_err(|error| bridge_error("construction material", error))?;
        let projected = patch.to_legacy_map(base).map_err(|error| bridge_error("construction material", error))?;
        let node = self.layer_csg_mut(layer)
            .and_then(|csg| csg.construction.as_mut())
            .and_then(|tree| tree.node_mut(path))
            .ok_or_else(|| format!("no construction material at layer {}, path {path:?}", layer + 1))?;
        node.material = Some(projected);
        node.material_v2 = Some(patch);
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

#[cfg(test)]
mod effective_base_tests {
    use super::*;

    #[test]
    fn a_body_starting_past_the_last_material_slot_inherits_the_scene() {
        let owner = CanonicalMaterials::new(MaterialSpec::default());
        let bodies = vec![vec![0], vec![5, 6], vec![7]];
        for layer in 0..8 {
            assert!(std::ptr::eq(effective_base(&owner, &bodies, layer), &owner.scene), "layer {layer}");
        }
    }
}
