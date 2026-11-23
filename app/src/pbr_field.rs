//! Specialised WGSL generation for canonical, renderer-independent PBR field graphs.
//!
//! The generated source has no bindings and no renderer policy. It defines [`CONTEXT_TYPE`],
//! [`OUTPUT_TYPE`], and [`FUNCTION_NAME`]; the caller supplies an explicit hit-independent context
//! and a function-pointer output. Every reachable node is emitted once in dependency order. Helpers
//! for procedural families and finite checks are included only when used.
//!
//! # Numeric contract
//!
//! Canonical CPU evaluation uses f64 arithmetic and checked i64 lattice cells. WGSL uses f32
//! arithmetic, u32 wrapping PCG arithmetic, and checked i32 lattice cells whose neighbours must
//! also fit i32. Consequently the non-band lattice range is
//! `-2147483520..=2147483392` after scaling. Hashing still truncates each cell to its low 32 bits,
//! exactly like the CPU reference. f32 rounding, transcendental implementations, fused operations,
//! overflow, and the smaller lattice mean results are not bit-exact with f64/u64 CPU evaluation.
//! Authored f64 values must round to finite f32; distinct remap endpoints must remain distinct.
//!
//! Division by a zero component, normalising zero, a non-finite result, and an out-of-range lattice
//! return `false` without writing any output. Texture nodes are rejected: no sampler/footprint
//! binding or fallback value is invented here.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use gf_core::pbr::{
    BuiltinField, FieldGraph, FieldId, FieldNode, FieldValue, MathOp, NoiseKind, ValueType,
};
use gf_core::pbr::{SchemaError, TextureId};

pub const CONTEXT_TYPE: &str = "PbrFieldContext";
pub const OUTPUT_TYPE: &str = "PbrFieldOutputs";
pub const FUNCTION_NAME: &str = "pbr_field_evaluate";

#[derive(Debug)]
pub enum FieldWgslError {
    EmptyOutputs,
    DuplicateOutput(FieldId),
    UnknownOutput(FieldId),
    InvalidGraph(SchemaError),
    UnsupportedTexture { field: FieldId, texture: TextureId },
    F32Range { field: FieldId, value: f64 },
    F32RemapDomain(FieldId),
}

impl fmt::Display for FieldWgslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOutputs => f.write_str("field WGSL generation needs at least one output"),
            Self::DuplicateOutput(id) => write!(f, "duplicate field WGSL output `{id}`"),
            Self::UnknownOutput(id) => write!(f, "unknown field WGSL output `{id}`"),
            Self::InvalidGraph(error) => write!(f, "invalid field graph: {error}"),
            Self::UnsupportedTexture { field, texture } => write!(
                f,
                "field `{field}` texture `{texture}` has no WGSL sampler/footprint binding"
            ),
            Self::F32Range { field, value } => write!(
                f,
                "field `{field}` value {value} is not finite after conversion to f32"
            ),
            Self::F32RemapDomain(id) => write!(
                f,
                "field `{id}` remap input endpoints collapse to the same f32 value"
            ),
        }
    }
}

impl std::error::Error for FieldWgslError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidGraph(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SchemaError> for FieldWgslError {
    fn from(value: SchemaError) -> Self {
        Self::InvalidGraph(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldWgslOutput {
    pub id: FieldId,
    pub value_type: ValueType,
    /// Stable generated member name, indexed by the caller's output order.
    pub member: String,
}

#[derive(Clone, Debug)]
pub struct FieldWgsl {
    source: String,
    outputs: Vec<FieldWgslOutput>,
}

impl FieldWgsl {
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn outputs(&self) -> &[FieldWgslOutput] {
        &self.outputs
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Helper {
    Finite(usize),
    Lattice,
    Pcg,
    Hash,
    Gradient,
    Cells,
}

struct Compiler<'a> {
    graph: &'a FieldGraph,
    order: Vec<FieldId>,
    types: BTreeMap<FieldId, ValueType>,
    helpers: BTreeSet<Helper>,
    indices: BTreeMap<FieldId, usize>,
    body: String,
}

impl<'a> Compiler<'a> {
    fn visit(&mut self, id: &FieldId) -> Result<(), FieldWgslError> {
        if self.types.contains_key(id) {
            return Ok(());
        }
        let node = self.graph.nodes.get(id).cloned().ok_or_else(|| {
            FieldWgslError::InvalidGraph(SchemaError::new(
                format!("fields.{id}"),
                gf_core::pbr::ErrorKind::UnknownField(id.clone()),
            ))
        })?;
        if let FieldNode::Texture { texture, .. } = &node {
            return Err(FieldWgslError::UnsupportedTexture {
                field: id.clone(),
                texture: texture.clone(),
            });
        }
        for dependency in dependencies(&node) {
            self.visit(dependency)?;
        }
        // The canonical graph owns type checking. Cache its answer with the topological memo.
        let value_type = self.graph.output_type(id)?;
        self.types.insert(id.clone(), value_type);
        self.order.push(id.clone());
        Ok(())
    }

    fn emit(&mut self) -> Result<(), FieldWgslError> {
        let order = self.order.clone();
        for id in order {
            let index = self.indices.len();
            self.indices.insert(id.clone(), index);
            self.emit_node(&id, index)?;
        }
        Ok(())
    }

    fn reference(&self, id: &FieldId) -> String {
        format!("n{}", self.indices[id])
    }

    fn kind(&self, id: &FieldId) -> ValueType {
        self.types[id]
    }

    fn require_finite(&mut self, kind: ValueType) {
        self.helpers.insert(Helper::Finite(arity(kind)));
    }

    fn float(id: &FieldId, value: f64) -> Result<String, FieldWgslError> {
        let converted = value as f32;
        if !converted.is_finite() {
            return Err(FieldWgslError::F32Range {
                field: id.clone(),
                value,
            });
        }
        let mut text = format!("{converted:?}");
        if !text.contains(['.', 'e']) {
            text.push_str(".0");
        }
        Ok(text)
    }

    fn emit_node(&mut self, id: &FieldId, index: usize) -> Result<(), FieldWgslError> {
        let node = &self.graph.nodes[id];
        let kind = self.kind(id);
        let output_type = wgsl_type(kind);
        let name = format!("n{index}");
        let f = |value| Self::float(id, value);
        let expression = match node {
            FieldNode::Constant(value) => match value {
                FieldValue::Scalar(value) => f(*value)?,
                FieldValue::Vector2(values) => {
                    format!("vec2f({}, {})", f(values[0])?, f(values[1])?)
                }
                FieldValue::Vector3(values) | FieldValue::Color(values) => format!(
                    "vec3f({}, {}, {})",
                    f(values[0])?,
                    f(values[1])?,
                    f(values[2])?
                ),
            },
            FieldNode::Builtin(source) => format!("context.{}", builtin_name(*source)),
            FieldNode::Time { scale } => {
                format!("(context.time_seconds * {})", f(*scale)?)
            }
            FieldNode::Noise {
                kind: noise,
                position,
                scale,
                seed,
            } => {
                let position_ref = self.reference(position);
                let position_kind = self.kind(position);
                let vector = if position_kind == ValueType::Vector2 {
                    format!("vec3f({position_ref}, 0.0)")
                } else {
                    position_ref
                };
                let scaled = format!("{name}_position");
                self.body.push_str(&format!(
                    "  let {scaled}: vec3f = {vector} * {};\n",
                    f(*scale)?
                ));
                self.require_finite(ValueType::Vector3);
                self.body.push_str(&format!(
                    "  if (!pbr_field_finite_3({scaled})) {{ return false; }}\n"
                ));
                if *noise == NoiseKind::Bands {
                    format!(
                        "(0.5 + 0.5 * cos(6.283185307179586476925286766559 * {scaled}.x))"
                    )
                } else {
                    self.helpers.insert(Helper::Lattice);
                    self.body.push_str(&format!(
                        "  var {name}_base: vec3i;\n  var {name}_fraction: vec3f;\n  if (!pbr_field_lattice({scaled}, &{name}_base, &{name}_fraction)) {{ return false; }}\n"
                    ));
                    let lo = *seed as u32;
                    let hi = (*seed >> 32) as u32;
                    match noise {
                        NoiseKind::Noise | NoiseKind::Ridged | NoiseKind::Billow => {
                            self.helpers.insert(Helper::Pcg);
                            self.helpers.insert(Helper::Hash);
                            self.helpers.insert(Helper::Gradient);
                            let noise_value = format!(
                                "pbr_field_gradient({name}_base, {name}_fraction, {lo}u, {hi}u)"
                            );
                            match noise {
                                NoiseKind::Noise => noise_value,
                                NoiseKind::Ridged => {
                                    let ridge = format!("{name}_ridge");
                                    self.body.push_str(&format!(
                                        "  let {ridge}: f32 = 1.0 - abs({noise_value});\n"
                                    ));
                                    format!("({ridge} * {ridge})")
                                }
                                NoiseKind::Billow => format!("abs({noise_value})"),
                                _ => unreachable!(),
                            }
                        }
                        NoiseKind::Cells | NoiseKind::Veins => {
                            self.helpers.insert(Helper::Pcg);
                            self.helpers.insert(Helper::Hash);
                            self.helpers.insert(Helper::Cells);
                            let distances = format!("{name}_distances");
                            self.body.push_str(&format!(
                                "  let {distances}: vec2f = pbr_field_cells({name}_base, {name}_fraction, {lo}u, {hi}u);\n"
                            ));
                            match noise {
                                NoiseKind::Cells => {
                                    format!("min(sqrt({distances}.x), 1.0)")
                                }
                                NoiseKind::Veins => format!(
                                    "(1.0 - min(sqrt({distances}.y) - sqrt({distances}.x), 1.0))"
                                ),
                                _ => unreachable!(),
                            }
                        }
                        NoiseKind::Bands => unreachable!(),
                    }
                }
            }
            FieldNode::Math { op, inputs } => {
                let a = self.reference(&inputs[0]);
                let b = || self.reference(&inputs[1]);
                match op {
                    MathOp::Add => format!("({a} + {})", b()),
                    MathOp::Subtract => format!("({a} - {})", b()),
                    MathOp::Multiply => format!("({a} * {})", b()),
                    MathOp::Divide => {
                        let b = b();
                        let zero = match self.kind(&inputs[1]) {
                            ValueType::Scalar => format!("{b} == 0.0"),
                            ValueType::Vector2 => format!("any({b} == vec2f(0.0))"),
                            ValueType::Vector3 | ValueType::Color => {
                                format!("any({b} == vec3f(0.0))")
                            }
                        };
                        self.body.push_str(&format!(
                            "  if ({zero}) {{ return false; }}\n"
                        ));
                        format!("({a} / {b})")
                    }
                    MathOp::Min => format!("min({a}, {})", b()),
                    MathOp::Max => format!("max({a}, {})", b()),
                    MathOp::Power => format!("pow({a}, {})", b()),
                    MathOp::Dot => format!("dot({a}, {})", b()),
                    MathOp::Cross => format!("cross({a}, {})", b()),
                    MathOp::Length => {
                        let input_kind = self.kind(&inputs[0]);
                        let scale = format!("{name}_scale");
                        let scaled = format!("{name}_scaled");
                        let norm = format!("{name}_norm");
                        self.body.push_str(&format!(
                            "  let {scale}: f32 = {};\n  let {scaled}: {input_type} = {a} / select(1.0, {scale}, {scale} != 0.0);\n  let {norm}: f32 = sqrt({});\n",
                            max_component(input_kind, &a),
                            dot_self(input_kind, &scaled),
                            input_type = wgsl_type(input_kind),
                        ));
                        // The CPU's zero-length result is zero without dividing by the scale.
                        format!("select(0.0, {scale} * {norm}, {scale} != 0.0)")
                    }
                    MathOp::Normalize => {
                        let input_kind = self.kind(&inputs[0]);
                        let scale = format!("{name}_scale");
                        let scaled = format!("{name}_scaled");
                        let norm = format!("{name}_norm");
                        self.body.push_str(&format!(
                            "  let {scale}: f32 = {};\n  if ({scale} == 0.0) {{ return false; }}\n  let {scaled}: {input_type} = {a} / {scale};\n  let {norm}: f32 = sqrt({});\n",
                            max_component(input_kind, &a),
                            dot_self(input_kind, &scaled),
                            input_type = wgsl_type(input_kind),
                        ));
                        format!("({scaled} / {norm})")
                    }
                    MathOp::Sin => format!("sin({a})"),
                    MathOp::Cos => format!("cos({a})"),
                    MathOp::Abs => format!("abs({a})"),
                }
            }
            FieldNode::Remap {
                input,
                from,
                to,
                clamp,
            } => {
                let from_f32 = [from[0] as f32, from[1] as f32];
                if from_f32[0] == from_f32[1] {
                    return Err(FieldWgslError::F32RemapDomain(id.clone()));
                }
                let input_name = self.reference(input);
                if to[0] == to[1] {
                    f(to[0])?
                } else {
                    let value = format!("{name}_input");
                    let t = format!("{name}_t");
                    self.body.push_str(&format!(
                        "  var {value}: f32 = {input_name};\n  var {t}: f32;\n"
                    ));
                    if *clamp {
                        self.body.push_str(&format!(
                            "  {value} = clamp({value}, {}, {});\n",
                            f(from[0].min(from[1]))?,
                            f(from[0].max(from[1]))?
                        ));
                    }
                    self.body.push_str(&format!(
                        "  {t} = ({value} - {}) / ({} - {});\n",
                        f(from[0])?,
                        f(from[1])?,
                        f(from[0])?
                    ));
                    if *clamp {
                        self.body.push_str(&format!("  {t} = clamp({t}, 0.0, 1.0);\n"));
                    }
                    format!("({} + {t} * ({} - {}))", f(to[0])?, f(to[1])?, f(to[0])?)
                }
            }
            FieldNode::Compose { components, .. } => {
                let components = components
                    .iter()
                    .map(|id| self.reference(id))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{output_type}({components})")
            }
            FieldNode::Convert { input, .. } => self.reference(input),
            FieldNode::Texture { .. } => unreachable!("reachable textures fail during traversal"),
        };
        self.body.push_str(&format!(
            "  let {name}: {output_type} = {expression};\n"
        ));
        self.require_finite(kind);
        self.body.push_str(&format!(
            "  if (!pbr_field_finite_{}({name})) {{ return false; }}\n",
            arity(kind)
        ));
        Ok(())
    }
}

pub fn compile_field_wgsl(
    graph: &FieldGraph,
    outputs: &[FieldId],
) -> Result<FieldWgsl, FieldWgslError> {
    graph.validate()?;
    if outputs.is_empty() {
        return Err(FieldWgslError::EmptyOutputs);
    }
    let mut unique = BTreeSet::new();
    let mut metadata = Vec::with_capacity(outputs.len());
    for (index, id) in outputs.iter().enumerate() {
        if !unique.insert(id.clone()) {
            return Err(FieldWgslError::DuplicateOutput(id.clone()));
        }
        if !graph.nodes.contains_key(id) {
            return Err(FieldWgslError::UnknownOutput(id.clone()));
        }
        metadata.push(FieldWgslOutput {
            id: id.clone(),
            value_type: graph.output_type(id)?,
            member: format!("output_{index}"),
        });
    }
    let mut compiler = Compiler {
        graph,
        order: Vec::new(),
        types: BTreeMap::new(),
        helpers: BTreeSet::new(),
        indices: BTreeMap::new(),
        body: String::new(),
    };
    for id in outputs {
        compiler.visit(id)?;
    }
    compiler.emit()?;

    let mut source = String::from(
        "// Generated from a canonical FieldGraph. f32/u32; see pbr_field.rs numeric contract.\n",
    );
    source.push_str(CONTEXT_WGSL);
    source.push_str("\nstruct PbrFieldOutputs {\n");
    for output in &metadata {
        source.push_str(&format!(
            "  {}: {},\n",
            output.member,
            wgsl_type(output.value_type)
        ));
    }
    source.push_str("}\n\n");
    for helper in &compiler.helpers {
        source.push_str(helper_source(*helper));
        source.push('\n');
    }
    source.push_str(&format!(
        "fn {FUNCTION_NAME}(context: {CONTEXT_TYPE}, output: ptr<function, {OUTPUT_TYPE}>) -> bool {{\n"
    ));
    source.push_str(&compiler.body);
    for (index, output) in metadata.iter().enumerate() {
        source.push_str(&format!(
            "  (*output).{} = n{};\n",
            output.member, compiler.indices[&output.id]
        ));
        debug_assert_eq!(output.member, format!("output_{index}"));
    }
    source.push_str("  return true;\n}\n");
    Ok(FieldWgsl {
        source,
        outputs: metadata,
    })
}

fn dependencies(node: &FieldNode) -> Vec<&FieldId> {
    match node {
        FieldNode::Constant(_) | FieldNode::Builtin(_) | FieldNode::Time { .. } => Vec::new(),
        FieldNode::Noise { position, .. } => vec![position],
        FieldNode::Math { inputs, .. } | FieldNode::Compose { components: inputs, .. } => {
            inputs.iter().collect()
        }
        FieldNode::Remap { input, .. } | FieldNode::Convert { input, .. } => vec![input],
        FieldNode::Texture { .. } => Vec::new(),
    }
}

// Canonical graph IDs are metadata only and are never interpolated into WGSL identifiers.
fn builtin_name(source: BuiltinField) -> &'static str {
    match source {
        BuiltinField::ObjectPosition => "object_position",
        BuiltinField::WorldPosition => "world_position",
        BuiltinField::GeometricNormal => "geometric_normal",
        BuiltinField::ShadingNormal => "shading_normal",
        BuiltinField::Tangent => "tangent",
        BuiltinField::Uv => "uv",
        BuiltinField::Height => "height",
        BuiltinField::RadialDistance => "radial_distance",
        BuiltinField::AxialDistance => "axial_distance",
        BuiltinField::AuthoredDistance => "authored_distance",
        BuiltinField::SdfDistance => "sdf_distance",
        BuiltinField::Curvature => "curvature",
        BuiltinField::Edge => "edge",
        BuiltinField::Cavity => "cavity",
        BuiltinField::Shelter => "shelter",
        BuiltinField::DeterministicSeed => "deterministic_seed",
    }
}

fn wgsl_type(kind: ValueType) -> &'static str {
    match kind {
        ValueType::Scalar => "f32",
        ValueType::Vector2 => "vec2f",
        ValueType::Vector3 | ValueType::Color => "vec3f",
    }
}

fn arity(kind: ValueType) -> usize {
    match kind {
        ValueType::Scalar => 1,
        ValueType::Vector2 => 2,
        ValueType::Vector3 | ValueType::Color => 3,
    }
}

fn max_component(kind: ValueType, value: &str) -> String {
    match kind {
        ValueType::Vector2 => format!("max(abs({value}.x), abs({value}.y))"),
        ValueType::Vector3 => format!(
            "max(abs({value}.x), max(abs({value}.y), abs({value}.z)))"
        ),
        _ => unreachable!("length and normalise accept vectors only"),
    }
}

fn dot_self(kind: ValueType, value: &str) -> String {
    match kind {
        ValueType::Vector2 => format!("({value}.x * {value}.x + {value}.y * {value}.y)"),
        ValueType::Vector3 => format!(
            "({value}.x * {value}.x + {value}.y * {value}.y + {value}.z * {value}.z)"
        ),
        _ => unreachable!("length and normalise accept vectors only"),
    }
}

const CONTEXT_WGSL: &str = r#"struct PbrFieldContext {
  object_position: vec3f,
  world_position: vec3f,
  uv: vec2f,
  time_seconds: f32,
  deterministic_seed: f32,
  geometric_normal: vec3f,
  shading_normal: vec3f,
  tangent: vec3f,
  height: f32,
  radial_distance: f32,
  axial_distance: f32,
  authored_distance: f32,
  sdf_distance: f32,
  curvature: f32,
  edge: f32,
  cavity: f32,
  shelter: f32,
}
"#;

fn helper_source(helper: Helper) -> &'static str {
    match helper {
        Helper::Finite(1) => {
            "fn pbr_field_finite_1(v: f32) -> bool {\n  return v == v && abs(v) <= 0x1.fffffep+127;\n}\n"
        }
        Helper::Finite(2) => {
            "fn pbr_field_finite_2(v: vec2f) -> bool {\n  return all(v == v) && all(abs(v) <= vec2f(0x1.fffffep+127));\n}\n"
        }
        Helper::Finite(3) => {
            "fn pbr_field_finite_3(v: vec3f) -> bool {\n  return all(v == v) && all(abs(v) <= vec3f(0x1.fffffep+127));\n}\n"
        }
        Helper::Finite(_) => unreachable!(),
        Helper::Lattice => {
            r#"fn pbr_field_lattice(p: vec3f, base: ptr<function, vec3i>, fraction: ptr<function, vec3f>) -> bool {
  let lower = floor(p);
  if (any(lower < vec3f(-2147483520.0)) || any(lower >= vec3f(2147483520.0))) {
    return false;
  }
  *base = vec3i(lower);
  *fraction = p - lower;
  return true;
}
"#
        }
        Helper::Pcg => {
            r#"fn pbr_field_pcg3(v0: vec3u) -> vec3u {
  var v = v0 * vec3u(1664525u) + vec3u(1013904223u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}
"#
        }
        Helper::Hash => {
            r#"fn pbr_field_hash3(cell: vec3i, lo: u32, hi: u32) -> vec3f {
  let c = bitcast<vec3u>(cell);
  let hash = pbr_field_pcg3(vec3u(
    c.x + lo * 747796405u + hi * 2246822519u,
    c.y + lo * 2891336453u + ((hi << 13u) | (hi >> 19u)) + 17u,
    c.z + lo * 277803737u + hi * 3266489917u + 101u
  ));
  return vec3f(hash >> vec3u(8u)) * (1.0 / 16777216.0);
}
"#
        }
        Helper::Gradient => {
            r#"fn pbr_field_gradient(base: vec3i, fraction: vec3f, lo: u32, hi: u32) -> f32 {
  let smoothed = fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0);
  var corners: array<f32, 8>;
  for (var index = 0u; index < 8u; index++) {
    let offset = vec3u(index & 1u, (index >> 1u) & 1u, (index >> 2u) & 1u);
    let cell = base + vec3i(offset);
    let random = pbr_field_hash3(cell, lo, hi);
    let gradient = normalize(random * 2.0 - 1.0 + vec3f(0.0001));
    let delta = fraction - vec3f(offset);
    corners[index] = gradient.x * delta.x + gradient.y * delta.y + gradient.z * delta.z;
  }
  let x0 = corners[0] + (corners[1] - corners[0]) * smoothed.x;
  let x1 = corners[2] + (corners[3] - corners[2]) * smoothed.x;
  let x2 = corners[4] + (corners[5] - corners[4]) * smoothed.x;
  let x3 = corners[6] + (corners[7] - corners[6]) * smoothed.x;
  let y0 = x0 + (x1 - x0) * smoothed.y;
  let y1 = x2 + (x3 - x2) * smoothed.y;
  return 1.6 * (y0 + (y1 - y0) * smoothed.z);
}
"#
        }
        Helper::Cells => {
            r#"fn pbr_field_cells(base: vec3i, fraction: vec3f, lo: u32, hi: u32) -> vec2f {
  var first = 1.0e30;
  var second = 1.0e30;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let offset = vec3i(x, y, z);
        let random = pbr_field_hash3(base + offset, lo, hi);
        let point = vec3f(offset) + 0.5 + (random - 0.5) * 0.72;
        let delta = point - fraction;
        let distance = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (distance < first) {
          second = first;
          first = distance;
        } else if (distance < second) {
          second = distance;
        }
      }
    }
  }
  return vec2f(first, second);
}
"#
        }
    }
}
