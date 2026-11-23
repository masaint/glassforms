//! Canonical full-body inspection and the deliberately small, renderer-projectable editing subset.
//! The complete tree comes from `MaterialSpec`'s stable serialization, so dormant payloads, fields,
//! textures, compatibility and identity cannot be accidentally omitted by a parallel UI schema.

use eframe::egui::{self, Ui};
use gf_core::pbr::{self, Color, MaterialPatch, MaterialSpec, PropertyValue, ScalarInput};
use gf_core::state::State;
use serde_json::Value;

use super::super::widgets::{gap, note};
use super::super::Panels;

#[derive(Clone, Copy, Debug, PartialEq)]
enum AttributeKind {
    Scalar {
        lo: f64,
        hi: f64,
        step: f64,
        add: f64,
    },
    Colour,
    Feature,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Attribute {
    path: &'static str,
    group: &'static str,
    kind: AttributeKind,
}

const fn scalar(
    path: &'static str,
    group: &'static str,
    lo: f64,
    hi: f64,
    step: f64,
    add: f64,
) -> Attribute {
    Attribute {
        path,
        group,
        kind: AttributeKind::Scalar { lo, hi, step, add },
    }
}
const fn colour(path: &'static str, group: &'static str) -> Attribute {
    Attribute {
        path,
        group,
        kind: AttributeKind::Colour,
    }
}
const fn feature(path: &'static str, group: &'static str) -> Attribute {
    Attribute {
        path,
        group,
        kind: AttributeKind::Feature,
    }
}

/// Constant values for which the existing strict legacy projection has a representation. Linked
/// values are patched together below. Everything else remains visible in the complete read-only tree.
const ATTRIBUTES: &[Attribute] = &[
    scalar("specular.ior", "Surface", 1.01, 3.0, 0.01, 1.5),
    scalar("specular.roughness", "Surface", 0.0, 1.0, 0.01, 0.2),
    scalar(
        "base.ceramic.transmitted_fraction",
        "Ceramic",
        0.0,
        1.0,
        0.01,
        0.15,
    ),
    feature("coat", "Coat"),
    scalar("coat.weight", "Coat", 0.0, 1.0, 0.01, 1.0),
    scalar("coat.ior", "Coat", 1.01, 3.0, 0.01, 1.5),
    scalar("coat.roughness", "Coat", 0.0, 1.0, 0.01, 0.2),
    feature("film", "Film"),
    scalar("film.thickness_nm", "Film", 0.0, 2000.0, 1.0, 360.0),
    scalar("film.ior", "Film", 1.01, 3.0, 0.01, 1.33),
    scalar("film.thickness_variation", "Film", 0.0, 1.0, 0.01, 0.0),
    scalar("film.pattern_scale", "Film", 0.01, 100.0, 0.01, 1.0),
    scalar("film.flow", "Film", 0.0, 1.0, 0.01, 0.0),
    scalar("film.drainage", "Film", 0.0, 1.0, 0.01, 0.0),
    scalar("film.ambient_weight", "Film", 0.0, 1.0, 0.01, 0.5),
    colour("transvolume.absorption_coefficients", "Transmission"),
    scalar(
        "transvolume.absorption_scale",
        "Transmission",
        0.0,
        20.0,
        0.01,
        1.0,
    ),
    feature("subsurface", "Subsurface"),
    scalar("subsurface.weight", "Subsurface", 0.0, 1.0, 0.01, 0.5),
    scalar(
        "subsurface.mean_free_path",
        "Subsurface",
        0.001,
        20.0,
        0.01,
        0.2,
    ),
    scalar(
        "subsurface.phase_anisotropy",
        "Subsurface",
        -0.95,
        0.95,
        0.01,
        0.0,
    ),
    scalar(
        "subsurface.pigment.spectral_spread",
        "Subsurface",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    scalar(
        "subsurface.pigment.near_surface_hue",
        "Subsurface",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    scalar(
        "subsurface.pigment.unscattered_hue",
        "Subsurface",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    scalar(
        "subsurface.pigment.opaque_exit",
        "Subsurface",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    feature("emission", "Emission"),
    colour("emission.color", "Emission"),
    scalar(
        "emission.volume.strength",
        "Emission",
        0.0,
        100.0,
        0.05,
        2.0,
    ),
    scalar(
        "emission.volume.core_concentration",
        "Emission",
        0.0,
        1.0,
        0.01,
        0.5,
    ),
    feature("wear", "Wear"),
    scalar("wear.age", "Wear", 0.0, 1.0, 0.01, 0.0),
    scalar("wear.edge_wear", "Wear", 0.0, 1.0, 0.01, 0.25),
    scalar("wear.cavity", "Wear", 0.0, 1.0, 0.01, 0.0),
    colour("wear.cavity_color", "Wear"),
    scalar("wear.scratches", "Wear", 0.0, 1.0, 0.01, 0.0),
    scalar("wear.swirls", "Wear", 0.0, 1.0, 0.01, 0.0),
    scalar("wear.orange_peel", "Wear", 0.0, 1.0, 0.01, 0.0),
    scalar("wear.smudges", "Wear", 0.0, 1.0, 0.01, 0.0),
    scalar("wear.dust", "Wear", 0.0, 1.0, 0.01, 0.0),
    scalar("wear.brushing", "Wear", 0.0, 1.0, 0.01, 0.0),
    scalar(
        "wear.brushing_angle_radians",
        "Wear",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        0.0,
    ),
    scalar("wear.mottle", "Wear", 0.0, 1.0, 0.01, 0.0),
    feature("special_optics.grating", "Special optics"),
    scalar(
        "special_optics.grating.strength",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.5,
    ),
    scalar(
        "special_optics.grating.pitch_nm",
        "Special optics",
        1.0,
        10000.0,
        1.0,
        1600.0,
    ),
    scalar(
        "special_optics.grating.angle_radians",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.grating.radial",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    feature("special_optics.birefringence", "Special optics"),
    scalar(
        "special_optics.birefringence.delta_ior",
        "Special optics",
        0.0,
        0.2,
        0.001,
        0.009,
    ),
    scalar(
        "special_optics.birefringence.azimuth_radians",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.birefringence.elevation_radians",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        std::f64::consts::FRAC_PI_2,
    ),
    feature("special_optics.grin", "Special optics"),
    scalar(
        "special_optics.grin.strength",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.1,
    ),
    scalar(
        "special_optics.grin.depth",
        "Special optics",
        0.0,
        1.0,
        0.001,
        0.02,
    ),
    feature("special_optics.bubbles", "Special optics"),
    scalar(
        "special_optics.bubbles.amount",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.3,
    ),
    scalar(
        "special_optics.bubbles.size",
        "Special optics",
        0.001,
        1.0,
        0.001,
        0.03,
    ),
    scalar(
        "special_optics.bubbles.evenness",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.5,
    ),
    scalar(
        "special_optics.bubbles.shape",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.bubbles.seed",
        "Special optics",
        0.0,
        10000.0,
        1.0,
        1.0,
    ),
    feature("special_optics.gradient", "Special optics"),
    scalar(
        "special_optics.gradient.amount",
        "Special optics",
        0.0,
        1.0,
        0.01,
        1.0,
    ),
    colour("special_optics.gradient.target_tint", "Special optics"),
    scalar(
        "special_optics.gradient.azimuth_radians",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.gradient.elevation_radians",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        std::f64::consts::FRAC_PI_2,
    ),
    scalar(
        "special_optics.gradient.offset",
        "Special optics",
        -1.0,
        1.0,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.gradient.softness",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.6,
    ),
    scalar(
        "special_optics.gradient.bands",
        "Special optics",
        1.0,
        64.0,
        1.0,
        6.0,
    ),
    scalar(
        "special_optics.gradient.wave",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.3,
    ),
    feature("special_optics.cracks", "Special optics"),
    scalar(
        "special_optics.cracks.amount",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.4,
    ),
    scalar(
        "special_optics.cracks.fractures",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.cracks.size",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.25,
    ),
    scalar(
        "special_optics.cracks.depth",
        "Special optics",
        0.0,
        1.0,
        0.01,
        1.0,
    ),
    scalar(
        "special_optics.cracks.openness",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.8,
    ),
    scalar(
        "special_optics.cracks.growth",
        "Special optics",
        0.0,
        1.0,
        0.01,
        1.0,
    ),
    scalar(
        "special_optics.cracks.curvature",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.5,
    ),
    scalar(
        "special_optics.cracks.roughness",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.cracks.seed",
        "Special optics",
        0.0,
        10000.0,
        1.0,
        1.0,
    ),
    feature("special_optics.photonic", "Special optics"),
    scalar(
        "special_optics.photonic.amount",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.8,
    ),
    scalar(
        "special_optics.photonic.size",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.12,
    ),
    scalar(
        "special_optics.photonic.longest_wavelength",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.8,
    ),
    scalar(
        "special_optics.photonic.softness",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.3,
    ),
    scalar(
        "special_optics.photonic.roll",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.photonic.seed",
        "Special optics",
        0.0,
        10000.0,
        1.0,
        1.0,
    ),
    colour("special_optics.photonic.platelet_color", "Special optics"),
    feature("special_optics.silk", "Special optics"),
    scalar(
        "special_optics.silk.amount",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.8,
    ),
    scalar(
        "special_optics.silk.azimuth_radians",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.silk.elevation_radians",
        "Special optics",
        -std::f64::consts::PI,
        std::f64::consts::PI,
        0.01,
        0.0,
    ),
    scalar(
        "special_optics.silk.softness",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.3,
    ),
    scalar(
        "special_optics.silk.wave",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.0,
    ),
    feature("special_optics.pressure", "Special optics"),
    scalar(
        "special_optics.pressure.amount",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.5,
    ),
    scalar(
        "special_optics.pressure.color_bias",
        "Special optics",
        0.0,
        1.0,
        0.01,
        0.5,
    ),
];

fn canonical_json(spec: &MaterialSpec) -> Value {
    serde_json::to_value(spec).expect("a material spec always serializes")
}

fn property_pointer(path: &str) -> String {
    let parts: Vec<_> = path.split('.').collect();
    let mut out = Vec::new();
    match parts.as_slice() {
        ["base", "ceramic", ..] => out.extend(["base", "values", "ceramic", "values"]),
        ["special_optics", group, ..] => out.extend(["special_optics", group, "values"]),
        [group, ..] => out.extend([*group, "values"]),
        _ => {}
    }
    match parts.as_slice() {
        ["base", "ceramic", rest @ ..] | ["special_optics", _, rest @ ..] => {
            out.extend(rest.iter().copied())
        }
        [_, rest @ ..] => out.extend(rest.iter().copied()),
        _ => {}
    }
    format!("/{}", out.join("/"))
}

fn feature_pointer(path: &str) -> String {
    let parts: Vec<_> = path.split('.').collect();
    match parts.as_slice() {
        ["base", "ceramic"] => "/base/values/ceramic/enabled".to_owned(),
        ["special_optics", group] => format!("/special_optics/{group}/enabled"),
        [group] => format!("/{group}/enabled"),
        _ => String::new(),
    }
}

fn property_value<'a>(json: &'a Value, path: &str) -> Option<&'a Value> {
    json.pointer(&property_pointer(path))
}
fn feature_value(json: &Value, path: &str) -> Option<bool> {
    json.pointer(&feature_pointer(path))?.as_bool()
}
fn constant_scalar(spec: &MaterialSpec, path: &str) -> Option<f64> {
    let json = canonical_json(spec);
    let input = property_value(&json, path)?;
    (input.get("source")?.as_str()? == "constant")
        .then(|| input.get("value")?.as_f64())
        .flatten()
}
fn constant_colour(spec: &MaterialSpec, path: &str) -> Option<[f64; 3]> {
    let json = canonical_json(spec);
    let input = property_value(&json, path)?;
    if input.get("source")?.as_str()? != "constant" {
        return None;
    }
    let values = input.get("value")?.as_array()?;
    Some([
        values.first()?.as_f64()?,
        values.get(1)?.as_f64()?,
        values.get(2)?.as_f64()?,
    ])
}
fn input_is_constant(spec: &MaterialSpec, path: &str) -> bool {
    let json = canonical_json(spec);
    property_value(&json, path)
        .and_then(|input| input.get("source"))
        .and_then(Value::as_str)
        == Some("constant")
}

fn is_added(json: &Value, defaults: &Value, attribute: &Attribute) -> bool {
    match attribute.kind {
        AttributeKind::Feature => feature_value(json, attribute.path) == Some(true),
        _ => property_value(json, attribute.path) != property_value(defaults, attribute.path),
    }
}

fn scalar_override(patch: &mut MaterialPatch, path: &str, value: f64) {
    pbr::scalar_override(patch, path, ScalarInput::Constant(value));
}
fn colour_override(patch: &mut MaterialPatch, path: &str, value: [f64; 3]) {
    patch
        .properties
        .insert(path.into(), PropertyValue::Color(Color(value).into()));
}

fn property_patch(spec: &MaterialSpec, attribute: &Attribute, value: &[f64]) -> MaterialPatch {
    let mut patch = MaterialPatch::new(spec.id.clone());
    match attribute.kind {
        AttributeKind::Scalar { .. } => scalar_override(&mut patch, attribute.path, value[0]),
        AttributeKind::Colour => {
            colour_override(&mut patch, attribute.path, [value[0], value[1], value[2]])
        }
        AttributeKind::Feature => return patch,
    }
    // These are one value in legacy Settings. Patch both sides rather than relying on a lossy
    // categorical conversion or leaving a conflict for the renderer bridge to discover later.
    match attribute.path {
        "specular.ior" => scalar_override(&mut patch, "transvolume.ior", value[0]),
        "subsurface.weight" => {
            pbr::feature_override(&mut patch, "subsurface", value[0] > 0.0);
            let json = canonical_json(spec);
            let transmission =
                constant_scalar(spec, "base.ceramic.transmitted_fraction").unwrap_or(0.15);
            let ceramic = feature_value(&json, "base.ceramic") == Some(true);
            scalar_override(
                &mut patch,
                "transvolume.weight",
                if ceramic {
                    transmission
                } else {
                    1.0 - value[0]
                },
            );
        }
        "base.ceramic.transmitted_fraction" => {
            if feature_value(&canonical_json(spec), "base.ceramic") == Some(true) {
                scalar_override(&mut patch, "transvolume.weight", value[0]);
            }
        }
        "transvolume.absorption_coefficients" => {
            colour_override(
                &mut patch,
                "transvolume.color",
                [(-value[0]).exp(), (-value[1]).exp(), (-value[2]).exp()],
            );
        }
        _ => {}
    }
    patch
}

fn feature_patch(spec: &MaterialSpec, path: &str, enabled: bool) -> MaterialPatch {
    let mut patch = MaterialPatch::new(spec.id.clone());
    pbr::feature_override(&mut patch, path, enabled);
    if !enabled {
        if path == "subsurface" {
            let ceramic = feature_value(&canonical_json(spec), "base.ceramic") == Some(true);
            let transmission = if ceramic {
                constant_scalar(spec, "base.ceramic.transmitted_fraction").unwrap_or(0.15)
            } else {
                1.0
            };
            scalar_override(&mut patch, "transvolume.weight", transmission);
        }
        return patch;
    }
    let activate = |patch: &mut MaterialPatch, property: &str, value: f64| {
        if constant_scalar(spec, property).unwrap_or(0.0) == 0.0 {
            scalar_override(patch, property, value);
        }
    };
    match path {
        "coat" => activate(&mut patch, "coat.weight", 1.0),
        "film" => {
            scalar_override(&mut patch, "film.weight", 1.0);
            activate(&mut patch, "film.thickness_nm", 360.0);
        }
        "subsurface" => {
            activate(&mut patch, "subsurface.weight", 0.5);
            let weight = constant_scalar(spec, "subsurface.weight")
                .filter(|weight| *weight > 0.0)
                .unwrap_or(0.5);
            let ceramic = feature_value(&canonical_json(spec), "base.ceramic") == Some(true);
            let transmission = if ceramic {
                constant_scalar(spec, "base.ceramic.transmitted_fraction").unwrap_or(0.15)
            } else {
                1.0 - weight
            };
            scalar_override(&mut patch, "transvolume.weight", transmission);
        }
        "emission" => activate(&mut patch, "emission.volume.strength", 2.0),
        "wear" => {
            let value = |path| constant_scalar(spec, path).unwrap_or(0.0);
            let marks = [
                "wear.edge_wear",
                "wear.cavity",
                "wear.scratches",
                "wear.swirls",
                "wear.orange_peel",
                "wear.smudges",
                "wear.dust",
            ]
            .into_iter()
            .any(|path| value(path) > 0.0);
            let effective = value("wear.brushing") > 0.0
                || value("wear.mottle") > 0.0
                || (value("wear.age") > 0.0 && marks);
            if !effective {
                activate(&mut patch, "wear.age", 1.0);
                activate(&mut patch, "wear.edge_wear", 0.25);
            }
        }
        "special_optics.grating" => activate(&mut patch, "special_optics.grating.strength", 0.5),
        "special_optics.birefringence" => {
            activate(&mut patch, "special_optics.birefringence.delta_ior", 0.009)
        }
        "special_optics.grin" => activate(&mut patch, "special_optics.grin.strength", 0.1),
        "special_optics.bubbles" => activate(&mut patch, "special_optics.bubbles.amount", 0.3),
        "special_optics.gradient" => activate(&mut patch, "special_optics.gradient.amount", 1.0),
        "special_optics.cracks" => activate(&mut patch, "special_optics.cracks.amount", 0.4),
        "special_optics.photonic" => activate(&mut patch, "special_optics.photonic.amount", 0.8),
        "special_optics.silk" => activate(&mut patch, "special_optics.silk.amount", 0.8),
        "special_optics.pressure" => activate(&mut patch, "special_optics.pressure.amount", 0.5),
        _ => {}
    }
    patch
}

fn apply(st: &mut State, px: &mut Panels, layer: usize, patch: MaterialPatch) -> bool {
    match st.apply_body_material_patch(layer, patch) {
        Ok(()) => {
            px.canonical_error = None;
            st.cut = true;
            true
        }
        Err(error) => {
            px.canonical_error = Some((layer, error));
            false
        }
    }
}

fn add_attribute(st: &mut State, px: &mut Panels, layer: usize, attribute: &Attribute) {
    let spec = st.body_material_spec(layer);
    let patch = match attribute.kind {
        AttributeKind::Feature => feature_patch(&spec, attribute.path, true),
        AttributeKind::Scalar { add, .. } => property_patch(&spec, attribute, &[add]),
        AttributeKind::Colour => {
            let Some(rgb) = constant_colour(&spec, attribute.path) else {
                px.canonical_error = Some((
                    layer,
                    format!(
                        "{} is a field, texture, or animated input and is read-only",
                        attribute.path
                    ),
                ));
                return;
            };
            property_patch(&spec, attribute, &rgb)
        }
    };
    if apply(st, px, layer, patch) {
        px.canonical_added
            .insert((layer, attribute.path.to_owned()));
    }
}

fn structured_value(ui: &mut Ui, path: &str, name: &str, value: &Value, depth: usize) {
    match value {
        Value::Object(entries) => {
            egui::CollapsingHeader::new(format!("{name}  {{{}}}", entries.len()))
                .id_salt(("canonical material tree", path))
                .default_open(depth == 0)
                .show(ui, |ui| {
                    for (key, value) in entries {
                        let child = format!("{path}.{key}");
                        structured_value(ui, &child, key, value, depth + 1);
                    }
                });
        }
        Value::Array(entries) => {
            egui::CollapsingHeader::new(format!("{name}  [{}]", entries.len()))
                .id_salt(("canonical material tree", path))
                .show(ui, |ui| {
                    for (index, value) in entries.iter().enumerate() {
                        let child = format!("{path}[{index}]");
                        structured_value(ui, &child, &index.to_string(), value, depth + 1);
                    }
                });
        }
        _ => {
            let text = match value {
                Value::String(text) => format!("\"{text}\""),
                _ => value.to_string(),
            };
            ui.horizontal(|ui| {
                ui.monospace(name);
                ui.add_enabled(
                    false,
                    egui::Label::new(egui::RichText::new(text).monospace()),
                );
            });
        }
    }
}

fn draw_attribute(
    ui: &mut Ui,
    st: &mut State,
    px: &mut Panels,
    layer: usize,
    attribute: &Attribute,
) {
    let spec = st.body_material_spec(layer);
    match attribute.kind {
        AttributeKind::Feature => {
            let mut enabled =
                feature_value(&canonical_json(&spec), attribute.path).unwrap_or(false);
            let response = ui.checkbox(&mut enabled, attribute.path);
            px.note(
                format!("canonical:feature:{}", attribute.path),
                response.rect,
            );
            if response.changed() {
                apply(st, px, layer, feature_patch(&spec, attribute.path, enabled));
            }
        }
        AttributeKind::Scalar { lo, hi, step, .. } => {
            if !input_is_constant(&spec, attribute.path) {
                ui.horizontal(|ui| {
                    ui.monospace(attribute.path);
                    ui.label("field / texture / animated input · read-only");
                });
                return;
            }
            let mut value = constant_scalar(&spec, attribute.path).unwrap_or(0.0);
            let response = ui.add(
                egui::DragValue::new(&mut value)
                    .range(lo..=hi)
                    .speed(step)
                    .prefix(format!("{}  ", attribute.path)),
            );
            px.note(format!("canonical:edit:{}", attribute.path), response.rect);
            if response.changed() {
                apply(st, px, layer, property_patch(&spec, attribute, &[value]));
            }
        }
        AttributeKind::Colour => {
            if !input_is_constant(&spec, attribute.path) {
                ui.horizontal(|ui| {
                    ui.monospace(attribute.path);
                    ui.label("field / texture / animated input · read-only");
                });
                return;
            }
            let mut rgb = constant_colour(&spec, attribute.path).unwrap_or([0.0; 3]);
            ui.label(attribute.path);
            let before = rgb;
            for (axis, channel) in ["R", "G", "B"].into_iter().enumerate() {
                let response = ui.add(
                    egui::DragValue::new(&mut rgb[axis])
                        .speed(0.005)
                        .prefix(format!("{channel} ")),
                );
                px.note(
                    format!("canonical:edit:{}.{channel}", attribute.path),
                    response.rect,
                );
            }
            if rgb != before {
                apply(st, px, layer, property_patch(&spec, attribute, &rgb));
            }
        }
    }
}

/// Complete effective spec plus its safe authoring controls. This is deliberately in the full-body
/// path: construction targets return before this section and remain sparse `MaterialPatch` owners.
pub fn section(ui: &mut Ui, st: &mut State, px: &mut Panels, layer: usize, open: bool) {
    if !open {
        return;
    }
    let spec = st.body_material_spec(layer);
    let owned = layer == 0 || st.body_own(layer);
    let source = if st.canonical.is_some() {
        "existing canonical owner"
    } else {
        "MaterialSpec::from_legacy(selected Settings)"
    };
    let ownership = if layer == 0 {
        "scene full-body owner".to_owned()
    } else if owned {
        "secondary full-body owner".to_owned()
    } else {
        "inherited from the scene · no secondary body override".to_owned()
    };
    ui.monospace(format!("body {} · {ownership}", layer + 1));
    note(ui, &format!("Source: {source}. Editing an inherited body creates its full owner only after preflight succeeds."));
    note(ui, "Effective full-body MaterialSpec; sparse construction patches do not own this body. Disabled feature payloads remain authored and visible below.");

    let defaults = canonical_json(&MaterialSpec::default());
    let json = canonical_json(&spec);
    let available: Vec<_> = ATTRIBUTES
        .iter()
        .filter(|attribute| {
            !is_added(&json, &defaults, attribute)
                && !px
                    .canonical_added
                    .contains(&(layer, attribute.path.to_string()))
        })
        .copied()
        .collect();
    let response = ui.menu_button("Add attribute", |ui| {
        egui::ScrollArea::vertical()
            .id_salt("canonical add attribute")
            .max_height(420.0)
            .show(ui, |ui| {
                let mut group = "";
                for attribute in available {
                    if attribute.group != group {
                        group = attribute.group;
                        ui.separator();
                        ui.strong(group);
                    }
                    let response = ui.button(attribute.path);
                    px.note(format!("canonical:add:{}", attribute.path), response.rect);
                    if response.clicked() {
                        add_attribute(st, px, layer, &attribute);
                        ui.close();
                    }
                }
            });
    });
    px.note("canonical:add", response.response.rect);
    note(ui, "Add lists inactive features and properties still at schema defaults. Constant scalar/colour edits and projectable feature switches use MaterialPatch + full-owner preflight.");

    let spec = st.body_material_spec(layer);
    let defaults = canonical_json(&MaterialSpec::default());
    let json = canonical_json(&spec);
    let active: Vec<_> = ATTRIBUTES
        .iter()
        .filter(|attribute| {
            is_added(&json, &defaults, attribute)
                || px
                    .canonical_added
                    .contains(&(layer, attribute.path.to_string()))
        })
        .copied()
        .collect();
    if !active.is_empty() {
        ui.separator();
        ui.strong("Canonical attributes");
        for attribute in active {
            draw_attribute(ui, st, px, layer, &attribute);
        }
    }
    if let Some((at, error)) = &px.canonical_error {
        if *at == layer {
            ui.colored_label(
                super::super::theme::pal().error_text,
                format!("Edit rejected; live material unchanged: {error}"),
            );
        }
    }
    note(ui, "Read-only: animated/field/texture inputs, special enums, texture resources, field graphs, identity, schema, compatibility and medium ownership metadata. Canonical-only sheen, coverage, geometry and film flakes are not renderer-projectable.");
    gap(ui, 4.0);
    ui.strong("Effective MaterialSpec (read-only)");
    structured_value(
        ui,
        "material",
        "MaterialSpec",
        &canonical_json(&st.body_material_spec(layer)),
        0,
    );
}

pub fn told(st: &State, px: &Panels, layer: usize) -> Value {
    let spec = st.body_material_spec(layer);
    let json = canonical_json(&spec);
    let defaults = canonical_json(&MaterialSpec::default());
    let available: Vec<Value> = ATTRIBUTES.iter().filter(|attribute| {
        !is_added(&json, &defaults, attribute) && !px.canonical_added.contains(&(layer, attribute.path.to_string()))
    }).map(|attribute| serde_json::json!({
        "path": attribute.path,
        "group": attribute.group,
        "kind": match attribute.kind { AttributeKind::Scalar { .. } => "scalar", AttributeKind::Colour => "colour", AttributeKind::Feature => "feature" },
    })).collect();
    let active: Vec<_> = ATTRIBUTES
        .iter()
        .filter(|attribute| {
            is_added(&json, &defaults, attribute)
                || px
                    .canonical_added
                    .contains(&(layer, attribute.path.to_string()))
        })
        .map(|attribute| attribute.path)
        .collect();
    serde_json::json!({
        "open": !px.store.is_closed(super::CANONICAL),
        "body": layer + 1,
        "source": if st.canonical.is_some() { "canonical owner" } else { "MaterialSpec::from_legacy(settings)" },
        "owned": layer == 0 || st.body_own(layer),
        "inheritance": if layer > 0 && !st.body_own(layer) { "scene" } else { "full body" },
        "constructionOwnership": "sparse patches are separate and do not own a full body",
        "schemaVersion": spec.schema_version,
        "materialId": spec.id.as_str(),
        "availableAttributes": available,
        "activeAttributes": active,
        "addedAttributes": px.canonical_added.iter().filter(|(body, _)| *body == layer).map(|(_, path)| path).collect::<Vec<_>>(),
        "error": px.canonical_error.as_ref().filter(|(body, _)| *body == layer).map(|(_, error)| error),
        "readOnly": ["field and texture inputs", "special enums", "field graph", "texture resources", "identity and schema", "medium metadata", "renderer-unsupported features"],
        "spec": json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gf_core::settings::Settings;

    #[test]
    fn every_supported_attribute_is_a_typed_renderer_projectable_patch() {
        let spec = MaterialSpec::from_legacy(&Settings::default(), "material:scene".into());
        let mut paths = std::collections::BTreeSet::new();
        for attribute in ATTRIBUTES {
            assert!(
                paths.insert(attribute.path),
                "duplicate canonical editor path {}",
                attribute.path
            );
            let patch = match attribute.kind {
                AttributeKind::Feature => feature_patch(&spec, attribute.path, true),
                AttributeKind::Scalar { add, .. } => property_patch(&spec, attribute, &[add]),
                AttributeKind::Colour => property_patch(
                    &spec,
                    attribute,
                    &constant_colour(&spec, attribute.path).unwrap(),
                ),
            };
            let edited = patch
                .apply(&spec)
                .unwrap_or_else(|error| panic!("{}: {error}", attribute.path));
            edited
                .validate()
                .unwrap_or_else(|error| panic!("{}: {error}", attribute.path));
            edited.to_legacy_settings().unwrap_or_else(|error| {
                panic!("{} is not renderer-projectable: {error}", attribute.path)
            });
        }
        assert!(ATTRIBUTES.len() > 70);
    }

    #[test]
    fn subsurface_edits_reach_live_wax_and_ceramic_materials() {
        for name in ["wax", "moulded", "satinpaint"] {
            let mut state = State::new();
            let preset = gf_core::material::named_index(name).unwrap();
            assert!(state.set_material(preset));
            for (path, value) in [
                ("subsurface.weight", 0.42),
                ("subsurface.mean_free_path", 0.68),
                ("subsurface.phase_anisotropy", 0.29),
                ("subsurface.pigment.spectral_spread", 0.23),
                ("subsurface.pigment.near_surface_hue", 0.25),
                ("subsurface.pigment.unscattered_hue", 0.23),
                ("subsurface.pigment.opaque_exit", 0.64),
            ] {
                let attribute = ATTRIBUTES.iter().find(|attribute| attribute.path == path).unwrap();
                let patch = property_patch(&state.body_material_spec(0), attribute, &[value]);
                state.apply_body_material_patch(0, patch)
                    .unwrap_or_else(|error| panic!("{name} {path}: {error}"));
            }
            assert_eq!(state.p.sss, 0.42, "{name} subsurface weight");
            assert_eq!(state.p.sss_depth, 0.68, "{name} mean free path");
            assert_eq!(state.p.sss_forward, 0.29, "{name} phase anisotropy");
            assert_eq!(state.p.sss_spread, 0.23, "{name} pigment spectral spread");
            assert_eq!(state.p.sss_hue, 0.25, "{name} pigment near-surface hue");
            assert_eq!(state.p.sss_through, 0.23, "{name} pigment unscattered hue");
            assert_eq!(state.p.sss_opaque, 0.64, "{name} pigment opaque exit");

            let weight = ATTRIBUTES.iter().find(|attribute| attribute.path == "subsurface.weight").unwrap();
            for value in [0.0, 0.42] {
                let patch = property_patch(&state.body_material_spec(0), weight, &[value]);
                state.apply_body_material_patch(0, patch)
                    .unwrap_or_else(|error| panic!("{name} weight {value}: {error}"));
                assert_eq!(state.p.sss, value, "{name} live weight {value}");
                assert_eq!(state.body_material_spec(0).subsurface.enabled, value > 0.0);
            }

            let spec = state.body_material_spec(0);
            let ceramic = spec.base.values.ceramic.enabled;
            state.apply_body_material_patch(0, feature_patch(&spec, "subsurface", false))
                .unwrap_or_else(|error| panic!("{name} disable subsurface: {error}"));
            assert_eq!(state.p.sss, 0.0, "{name} disabled live weight");
            let saved = state.setup();
            let mut reopened = State::new();
            reopened.apply_setup(&saved).unwrap_or_else(|error| panic!("{name} reopen disabled SSS: {error}"));
            state = reopened;
            let spec = state.body_material_spec(0);
            state.apply_body_material_patch(0, feature_patch(&spec, "subsurface", true))
                .unwrap_or_else(|error| panic!("{name} re-enable subsurface: {error}"));
            assert_eq!(state.p.sss, 0.42, "{name} restored live weight");
            assert_eq!(state.body_material_spec(0).base.values.ceramic.enabled, ceramic,
                "{name} ceramic selection survives the SSS toggle");

            assert!(state.body_set(0, "sssWalk", 1.0), "{name} random walk edit");
            let spec = state.body_material_spec(0);
            assert_eq!(spec.subsurface.values.estimator, pbr::ScatteringEstimator::RandomWalk);
            assert_eq!(state.p.sss_walk, 1.0, "{name} live random walk");
            assert!(gf_core::material::Variant::of(&state.p).walk, "{name} active random walk");
        }
    }

    #[test]
    fn structured_metadata_is_the_complete_spec_not_a_second_schema() {
        let mut spec = MaterialSpec::from_legacy(&Settings::default(), "material:scene".into());
        spec.sheen.values.color = Color([0.2, 0.3, 0.4]).into();
        spec.fields
            .nodes
            .insert("f".into(), pbr::FieldNode::Time { scale: 2.0 });
        let json = canonical_json(&spec);
        assert_eq!(json["schema_version"], pbr::MATERIAL_SCHEMA_VERSION);
        assert_eq!(json["sheen"]["enabled"], false);
        assert_eq!(
            json["sheen"]["values"]["color"]["value"],
            serde_json::json!([0.2, 0.3, 0.4])
        );
        assert_eq!(json["fields"]["nodes"]["f"]["op"], "time");
        assert_eq!(json, serde_json::to_value(&spec).unwrap());
        assert_eq!(
            property_pointer("base.ceramic.transmitted_fraction"),
            "/base/values/ceramic/values/transmitted_fraction"
        );
        assert_eq!(
            property_pointer("special_optics.grin.depth"),
            "/special_optics/grin/values/depth"
        );
    }

    #[test]
    fn dormant_payloads_survive_feature_disable() {
        let spec = MaterialSpec::from_legacy(&Settings::default(), "material:scene".into());
        let coat = ATTRIBUTES
            .iter()
            .find(|attribute| attribute.path == "coat")
            .unwrap();
        let enabled = feature_patch(&spec, "coat", true).apply(&spec).unwrap();
        assert_eq!(constant_scalar(&enabled, "coat.weight"), Some(1.0));
        let disabled = feature_patch(&enabled, "coat", false)
            .apply(&enabled)
            .unwrap();
        assert_eq!(constant_scalar(&disabled, "coat.weight"), Some(1.0));
        assert_eq!(disabled.to_legacy_settings().unwrap().coat, 0.0);
        assert_eq!(
            feature_value(&canonical_json(&disabled), coat.path),
            Some(false)
        );
    }
}
