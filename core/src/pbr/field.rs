use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::pbr::value::{finite, ErrorKind, SchemaError, TextureId};

pub const MAX_FIELD_NODES: usize = 1024;
pub const MAX_FIELD_DEPTH: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FieldId(pub String);

impl FieldId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    fn validate(&self, path: &str) -> Result<(), SchemaError> {
        if self.0.trim().is_empty() || self.0.chars().any(char::is_control) {
            Err(SchemaError::new(path, ErrorKind::InvalidIdentity))
        } else {
            Ok(())
        }
    }
}
impl From<&str> for FieldId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}
impl From<String> for FieldId {
    fn from(value: String) -> Self {
        Self(value)
    }
}
impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    #[default]
    Scalar,
    Vector2,
    Vector3,
    Color,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FieldGraph {
    /// Stable IDs and ordered storage make traversal, diagnostics, and JSON deterministic.
    pub nodes: BTreeMap<FieldId, FieldNode>,
}

impl FieldGraph {
    pub fn output_type(&self, id: &FieldId) -> Result<ValueType, SchemaError> {
        self.validate()?;
        self.types()?.get(id).copied().ok_or_else(|| {
            SchemaError::new(format!("fields.{id}"), ErrorKind::UnknownField(id.clone()))
        })
    }

    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.nodes.len() > MAX_FIELD_NODES {
            return Err(SchemaError::new(
                "fields",
                ErrorKind::FieldNodeLimit {
                    count: self.nodes.len(),
                    limit: MAX_FIELD_NODES,
                },
            ));
        }
        for id in self.nodes.keys() {
            id.validate(&format!("fields.{id}"))?;
        }
        self.types().map(|_| ())
    }

    fn types(&self) -> Result<BTreeMap<FieldId, ValueType>, SchemaError> {
        fn visit(
            graph: &FieldGraph,
            id: &FieldId,
            states: &mut BTreeMap<FieldId, VisitState>,
            types: &mut BTreeMap<FieldId, ValueType>,
            depths: &mut BTreeMap<FieldId, usize>,
            depth: usize,
        ) -> Result<ValueType, SchemaError> {
            if depth > MAX_FIELD_DEPTH {
                return Err(SchemaError::new(
                    format!("fields.{id}"),
                    ErrorKind::FieldDepthLimit {
                        limit: MAX_FIELD_DEPTH,
                    },
                ));
            }
            match states.get(id) {
                Some(VisitState::Done) => return Ok(types[id]),
                Some(VisitState::Visiting) => {
                    return Err(SchemaError::new(
                        format!("fields.{id}"),
                        ErrorKind::FieldCycle(id.clone()),
                    ));
                }
                None => {}
            }
            let node = graph.nodes.get(id).ok_or_else(|| {
                SchemaError::new(format!("fields.{id}"), ErrorKind::UnknownField(id.clone()))
            })?;
            states.insert(id.clone(), VisitState::Visiting);
            let mut node_depth = 0;
            let result = (|| {
                let mut dependency = |dependency: &FieldId| {
                    let kind = visit(graph, dependency, states, types, depths, depth + 1)?;
                    node_depth = node_depth.max(depths[dependency] + 1);
                    Ok(kind)
                };
                let mut expect = |input: &FieldId, expected: ValueType| {
                    let actual = dependency(input)?;
                    if actual == expected {
                        Ok(())
                    } else {
                        Err(SchemaError::new(
                            format!("fields.{id}.{input}"),
                            ErrorKind::FieldType { expected, actual },
                        ))
                    }
                };
                let kind = match node {
                    FieldNode::Constant(value) => {
                        value.validate(&format!("fields.{id}.constant"))?;
                        value.value_type()
                    }
                    FieldNode::Builtin(source) => source.value_type(),
                    FieldNode::Time { scale } => {
                        finite(*scale, &format!("fields.{id}.scale"))?;
                        ValueType::Scalar
                    }
                    FieldNode::Noise {
                        kind: _,
                        position,
                        scale,
                        seed: _,
                    } => {
                        let position_type = dependency(position)?;
                        if !matches!(position_type, ValueType::Vector2 | ValueType::Vector3) {
                            return Err(SchemaError::new(
                                format!("fields.{id}.position"),
                                ErrorKind::FieldType {
                                    expected: ValueType::Vector3,
                                    actual: position_type,
                                },
                            ));
                        }
                        finite(*scale, &format!("fields.{id}.scale"))?;
                        ValueType::Scalar
                    }
                    FieldNode::Math { op, inputs } => {
                        if inputs.len() != op.arity() {
                            return Err(SchemaError::new(
                                format!("fields.{id}.inputs"),
                                ErrorKind::InvalidFieldGraph(format!(
                                    "{op:?} needs {} inputs, found {}",
                                    op.arity(),
                                    inputs.len()
                                )),
                            ));
                        }
                        let input_types = inputs
                            .iter()
                            .map(dependency)
                            .collect::<Result<Vec<_>, _>>()?;
                        op.output_type(&input_types).ok_or_else(|| {
                            SchemaError::new(
                                format!("fields.{id}"),
                                ErrorKind::InvalidFieldGraph(format!(
                                    "{op:?} does not accept {input_types:?}"
                                )),
                            )
                        })?
                    }
                    FieldNode::Remap {
                        input,
                        from,
                        to,
                        clamp: _,
                    } => {
                        expect(input, ValueType::Scalar)?;
                        for (i, value) in from.iter().chain(to.iter()).enumerate() {
                            finite(*value, &format!("fields.{id}.remap[{i}]"))?;
                        }
                        if from[0] == from[1] {
                            return Err(SchemaError::new(
                                format!("fields.{id}.remap"),
                                ErrorKind::InvalidFieldGraph("remap input range is empty".into()),
                            ));
                        }
                        ValueType::Scalar
                    }
                    FieldNode::Compose { output, components } => {
                        let count = match output {
                            ValueType::Vector2 => 2,
                            ValueType::Vector3 | ValueType::Color => 3,
                            ValueType::Scalar => {
                                return Err(SchemaError::new(
                                    format!("fields.{id}.output"),
                                    ErrorKind::InvalidFieldGraph(
                                        "compose cannot output a scalar".into(),
                                    ),
                                ));
                            }
                        };
                        if components.len() != count {
                            return Err(SchemaError::new(
                                format!("fields.{id}.components"),
                                ErrorKind::InvalidFieldGraph(format!(
                                    "{output:?} needs {count} components"
                                )),
                            ));
                        }
                        for component in components {
                            expect(component, ValueType::Scalar)?;
                        }
                        *output
                    }
                    FieldNode::Convert { input, output } => {
                        let actual = dependency(input)?;
                        let convertible = matches!(
                            (actual, *output),
                            (ValueType::Vector3, ValueType::Color)
                                | (ValueType::Color, ValueType::Vector3)
                        );
                        if !convertible {
                            return Err(SchemaError::new(
                                format!("fields.{id}.output"),
                                ErrorKind::FieldType {
                                    expected: actual,
                                    actual: *output,
                                },
                            ));
                        }
                        *output
                    }
                    FieldNode::Texture { texture: _, output } => *output,
                };
                Ok(kind)
            })();
            match result {
                Ok(kind) => {
                    states.insert(id.clone(), VisitState::Done);
                    types.insert(id.clone(), kind);
                    depths.insert(id.clone(), node_depth);
                    Ok(kind)
                }
                Err(error) => {
                    states.remove(id);
                    Err(error)
                }
            }
        }

        #[derive(Clone, Copy)]
        enum VisitState {
            Visiting,
            Done,
        }

        let mut states = BTreeMap::new();
        let mut types = BTreeMap::new();
        let mut depths = BTreeMap::new();
        for id in self.nodes.keys() {
            visit(self, id, &mut states, &mut types, &mut depths, 0)?;
        }
        if let Some(id) = depths
            .iter()
            .find_map(|(id, depth)| (*depth > MAX_FIELD_DEPTH).then_some(id))
        {
            return Err(SchemaError::new(
                format!("fields.{id}"),
                ErrorKind::FieldDepthLimit {
                    limit: MAX_FIELD_DEPTH,
                },
            ));
        }
        Ok(types)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op", content = "args")]
pub enum FieldNode {
    Constant(FieldValue),
    Builtin(BuiltinField),
    Time {
        scale: f64,
    },
    Noise {
        kind: NoiseKind,
        position: FieldId,
        scale: f64,
        seed: u64,
    },
    Math {
        op: MathOp,
        inputs: Vec<FieldId>,
    },
    Remap {
        input: FieldId,
        from: [f64; 2],
        to: [f64; 2],
        clamp: bool,
    },
    Compose {
        output: ValueType,
        components: Vec<FieldId>,
    },
    /// Explicit numeric-vector/colour reinterpretation; no colour-space conversion is implied.
    Convert {
        input: FieldId,
        output: ValueType,
    },
    Texture {
        texture: TextureId,
        output: ValueType,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum FieldValue {
    Scalar(f64),
    Vector2([f64; 2]),
    Vector3([f64; 3]),
    Color([f64; 3]),
}

impl FieldValue {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::Scalar(_) => ValueType::Scalar,
            Self::Vector2(_) => ValueType::Vector2,
            Self::Vector3(_) => ValueType::Vector3,
            Self::Color(_) => ValueType::Color,
        }
    }

    pub(crate) fn validate(&self, path: &str) -> Result<(), SchemaError> {
        match self {
            Self::Scalar(value) => finite(*value, path),
            Self::Vector2(values) => {
                for (i, value) in values.iter().enumerate() {
                    finite(*value, &format!("{path}[{i}]"))?;
                }
                Ok(())
            }
            Self::Vector3(values) | Self::Color(values) => {
                for (i, value) in values.iter().enumerate() {
                    finite(*value, &format!("{path}[{i}]"))?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinField {
    ObjectPosition,
    WorldPosition,
    GeometricNormal,
    ShadingNormal,
    Tangent,
    Uv,
    Height,
    RadialDistance,
    AxialDistance,
    AuthoredDistance,
    SdfDistance,
    Curvature,
    Edge,
    Cavity,
    Shelter,
    DeterministicSeed,
}

impl BuiltinField {
    fn value_type(self) -> ValueType {
        match self {
            Self::ObjectPosition
            | Self::WorldPosition
            | Self::GeometricNormal
            | Self::ShadingNormal
            | Self::Tangent => ValueType::Vector3,
            Self::Uv => ValueType::Vector2,
            _ => ValueType::Scalar,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseKind {
    #[default]
    Noise,
    Ridged,
    Billow,
    Cells,
    Veins,
    Bands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
    Power,
    Dot,
    Cross,
    Length,
    Normalize,
    Sin,
    Cos,
    Abs,
}

impl MathOp {
    fn arity(self) -> usize {
        match self {
            Self::Length | Self::Normalize | Self::Sin | Self::Cos | Self::Abs => 1,
            _ => 2,
        }
    }

    fn output_type(self, inputs: &[ValueType]) -> Option<ValueType> {
        use MathOp::*;
        use ValueType::{Scalar, Vector2, Vector3};
        let same = inputs[0] == inputs[1.min(inputs.len() - 1)];
        let vector = |kind| matches!(kind, Vector2 | Vector3);
        match self {
            Add | Subtract | Min | Max if same => Some(inputs[0]),
            Multiply | Divide if same => Some(inputs[0]),
            Multiply | Divide if inputs.contains(&Scalar) && inputs.len() == 2 => {
                let other = if inputs[0] == Scalar {
                    inputs[1]
                } else {
                    inputs[0]
                };
                (other != Scalar).then_some(other)
            }
            Power if inputs == [Scalar, Scalar] => Some(Scalar),
            Dot if same && vector(inputs[0]) => Some(Scalar),
            Cross if inputs == [Vector3, Vector3] => Some(Vector3),
            Length if inputs.len() == 1 && vector(inputs[0]) => Some(Scalar),
            Normalize if inputs.len() == 1 && vector(inputs[0]) => Some(inputs[0]),
            Sin | Cos | Abs if inputs.len() == 1 => Some(inputs[0]),
            _ if inputs.len() == 1
                && matches!(
                    self,
                    Add | Subtract | Multiply | Divide | Min | Max | Power | Dot | Cross
                ) =>
            {
                None
            }
            _ => None,
        }
    }
}
