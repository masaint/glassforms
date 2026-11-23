use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::pbr::field::{FieldGraph, FieldId, ValueType};

macro_rules! stable_id {
    ($($name:ident),+ $(,)?) => {$(
        #[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self { Self(value.into()) }
            pub fn as_str(&self) -> &str { &self.0 }
            pub fn validate(&self, path: &str) -> Result<(), SchemaError> {
                if self.0.trim().is_empty() {
                    return Err(SchemaError::new(path, ErrorKind::EmptyIdentity));
                }
                if self.0.chars().any(char::is_control) {
                    return Err(SchemaError::new(path, ErrorKind::InvalidIdentity));
                }
                Ok(())
            }
        }
        impl From<&str> for $name { fn from(value: &str) -> Self { Self::new(value) } }
        impl From<String> for $name { fn from(value: String) -> Self { Self(value) } }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
        }
    )+};
}

stable_id!(
    MaterialId,
    PresetId,
    TextureId,
    MediumId,
    ConductorId,
    TrackId
);

#[derive(Clone, Debug, PartialEq)]
pub struct SchemaError {
    pub path: String,
    pub kind: ErrorKind,
}

impl SchemaError {
    pub fn new(path: impl Into<String>, kind: ErrorKind) -> Self {
        Self {
            path: path.into(),
            kind,
        }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.kind)
    }
}

impl std::error::Error for SchemaError {}

#[derive(Clone, Debug, PartialEq)]
pub enum ErrorKind {
    UnsupportedVersion {
        found: u32,
        supported: u32,
    },
    EmptyIdentity,
    InvalidIdentity,
    NonFinite,
    OutOfRange {
        min: f64,
        max: f64,
        value: f64,
    },
    UnknownField(FieldId),
    UnknownTexture(TextureId),
    FieldCycle(FieldId),
    FieldNodeLimit {
        count: usize,
        limit: usize,
    },
    FieldDepthLimit {
        limit: usize,
    },
    UnsupportedFieldEvaluation(&'static str),
    FieldEvaluation(String),
    FieldType {
        expected: ValueType,
        actual: ValueType,
    },
    InvalidFieldGraph(String),
    InvalidAnimation(String),
    UnknownProperty(String),
    PropertyType {
        expected: &'static str,
        actual: &'static str,
    },
    MissingBase(MaterialId),
    BaseIdentity {
        expected: MaterialId,
        actual: MaterialId,
    },
    InvalidLegacySetting {
        name: String,
        reason: String,
    },
    Incompatible(String),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "schema version {found} is newer than supported version {supported}"
            ),
            Self::EmptyIdentity => f.write_str("identity must not be empty"),
            Self::InvalidIdentity => f.write_str("identity contains a control character"),
            Self::NonFinite => f.write_str("value must be finite"),
            Self::OutOfRange { min, max, value } => write!(f, "{value} is outside {min}..={max}"),
            Self::UnknownField(id) => write!(f, "unknown field `{id}`"),
            Self::UnknownTexture(id) => write!(f, "unknown texture `{id}`"),
            Self::FieldCycle(id) => write!(f, "field graph cycles through `{id}`"),
            Self::FieldNodeLimit { count, limit } => {
                write!(f, "{count} field nodes exceed the limit of {limit}")
            }
            Self::FieldDepthLimit { limit } => {
                write!(f, "field dependency depth exceeds the limit of {limit}")
            }
            Self::UnsupportedFieldEvaluation(reason) => f.write_str(reason),
            Self::FieldEvaluation(reason) => f.write_str(reason),
            Self::FieldType { expected, actual } => {
                write!(f, "expected {expected:?}, found {actual:?}")
            }
            Self::InvalidFieldGraph(reason) => f.write_str(reason),
            Self::InvalidAnimation(reason) => f.write_str(reason),
            Self::UnknownProperty(path) => write!(f, "unknown canonical property `{path}`"),
            Self::PropertyType { expected, actual } => {
                write!(f, "expected {expected}, found {actual}")
            }
            Self::MissingBase(id) => write!(f, "sparse patch has no inherited material `{id}`"),
            Self::BaseIdentity { expected, actual } => {
                write!(f, "patch expects base `{expected}`, received `{actual}`")
            }
            Self::InvalidLegacySetting { name, reason } => {
                write!(f, "legacy setting `{name}`: {reason}")
            }
            Self::Incompatible(reason) => f.write_str(reason),
        }
    }
}

/// Validation context shared by all property groups. Dormant values are validated too.
pub(crate) struct ValidationContext<'a> {
    pub fields: &'a FieldGraph,
    pub textures: &'a BTreeMap<TextureId, TextureResource>,
}

pub(crate) trait Validate {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError>;
}

pub(crate) fn finite(value: f64, path: &str) -> Result<(), SchemaError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SchemaError::new(path, ErrorKind::NonFinite))
    }
}

pub(crate) fn range(value: f64, min: f64, max: f64, path: &str) -> Result<(), SchemaError> {
    finite(value, path)?;
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(SchemaError::new(
            path,
            ErrorKind::OutOfRange { min, max, value },
        ))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Color(pub [f64; 3]);

impl Color {
    pub const WHITE: Self = Self([1.0; 3]);
    pub const BLACK: Self = Self([0.0; 3]);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Vector2(pub [f64; 2]);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Vector3(pub [f64; 3]);

/// A type accepted by a typed material input.
pub trait MaterialValue: Clone + fmt::Debug + PartialEq {
    const VALUE_TYPE: ValueType;
    fn validate_finite(&self, path: &str) -> Result<(), SchemaError>;
}

impl MaterialValue for f64 {
    const VALUE_TYPE: ValueType = ValueType::Scalar;
    fn validate_finite(&self, path: &str) -> Result<(), SchemaError> {
        finite(*self, path)
    }
}

impl MaterialValue for Color {
    const VALUE_TYPE: ValueType = ValueType::Color;
    fn validate_finite(&self, path: &str) -> Result<(), SchemaError> {
        for (i, value) in self.0.iter().enumerate() {
            finite(*value, &format!("{path}[{i}]"))?;
        }
        Ok(())
    }
}

impl MaterialValue for Vector2 {
    const VALUE_TYPE: ValueType = ValueType::Vector2;
    fn validate_finite(&self, path: &str) -> Result<(), SchemaError> {
        for (i, value) in self.0.iter().enumerate() {
            finite(*value, &format!("{path}[{i}]"))?;
        }
        Ok(())
    }
}

impl MaterialValue for Vector3 {
    const VALUE_TYPE: ValueType = ValueType::Vector3;
    fn validate_finite(&self, path: &str) -> Result<(), SchemaError> {
        for (i, value) in self.0.iter().enumerate() {
            finite(*value, &format!("{path}[{i}]"))?;
        }
        Ok(())
    }
}

pub type ScalarInput = Input<f64>;
pub type ColorInput = Input<Color>;
pub type Vector3Input = Input<Vector3>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source", content = "value")]
pub enum Input<T> {
    Constant(T),
    Animated(AnimatedInput<T>),
    Field(FieldInput<T>),
    Texture(TextureInput<T>),
}

impl<T> From<T> for Input<T> {
    fn from(value: T) -> Self {
        Self::Constant(value)
    }
}

impl<T> Input<T> {
    pub fn constant(value: T) -> Self {
        Self::Constant(value)
    }

    pub fn as_constant(&self) -> Option<&T> {
        match self {
            Self::Constant(value) => Some(value),
            _ => None,
        }
    }

    /// Visit every authored constant, including dormant fallbacks and animation keys.
    pub fn for_each_authored(&self, mut visit: impl FnMut(&T)) {
        match self {
            Self::Constant(value) => visit(value),
            Self::Animated(track) => {
                visit(&track.fallback);
                for key in &track.keys {
                    visit(&key.value);
                }
            }
            Self::Field(input) => {
                visit(&input.fallback);
                if let Some(mapping) = &input.mapping {
                    mapping.for_each(&mut visit);
                }
            }
            Self::Texture(input) => {
                visit(&input.fallback);
                if let Some(mapping) = &input.mapping {
                    mapping.for_each(&mut visit);
                }
            }
        }
    }

    pub fn value_type(&self) -> ValueType
    where
        T: MaterialValue,
    {
        T::VALUE_TYPE
    }
}

impl<T: MaterialValue> Input<T> {
    pub(crate) fn validate_input(
        &self,
        cx: &ValidationContext<'_>,
        path: &str,
    ) -> Result<(), SchemaError> {
        match self {
            Self::Constant(value) => value.validate_finite(path),
            Self::Animated(track) => track.validate(path),
            Self::Field(input) => {
                input
                    .fallback
                    .validate_finite(&format!("{path}.fallback"))?;
                if let Some(mapping) = &input.mapping {
                    mapping.validate(&format!("{path}.mapping"))?;
                }
                if let Some(remap) = &input.remap {
                    remap.validate(&format!("{path}.remap"))?;
                }
                let actual = cx.fields.output_type(&input.field)?;
                if actual != T::VALUE_TYPE {
                    return Err(SchemaError::new(
                        path,
                        ErrorKind::FieldType {
                            expected: T::VALUE_TYPE,
                            actual,
                        },
                    ));
                }
                Ok(())
            }
            Self::Texture(input) => {
                input
                    .fallback
                    .validate_finite(&format!("{path}.fallback"))?;
                if let Some(mapping) = &input.mapping {
                    mapping.validate(&format!("{path}.mapping"))?;
                }
                if let Some(remap) = &input.remap {
                    remap.validate(&format!("{path}.remap"))?;
                }
                let resource = cx.textures.get(&input.texture).ok_or_else(|| {
                    SchemaError::new(path, ErrorKind::UnknownTexture(input.texture.clone()))
                })?;
                resource.validate(path)?;
                let channels_valid = match T::VALUE_TYPE {
                    ValueType::Scalar => matches!(
                        input.channels,
                        ChannelSelector::Red
                            | ChannelSelector::Green
                            | ChannelSelector::Blue
                            | ChannelSelector::Alpha
                    ),
                    ValueType::Vector2 => input.channels == ChannelSelector::Vector2,
                    ValueType::Vector3 => input.channels == ChannelSelector::Vector3,
                    ValueType::Color => {
                        matches!(input.channels, ChannelSelector::Rgb | ChannelSelector::Rgba)
                    }
                };
                if !channels_valid {
                    return Err(SchemaError::new(
                        path,
                        ErrorKind::InvalidFieldGraph(format!(
                            "{:?} channels cannot produce {:?}",
                            input.channels,
                            T::VALUE_TYPE
                        )),
                    ));
                }
                Ok(())
            }
        }
    }
}

impl<T: MaterialValue> Validate for Input<T> {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.validate_input(cx, path)
    }
}

impl<T: Validate> Validate for Feature<T> {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.values.validate(cx, &format!("{path}.values"))
    }
}

/// A group switch and its complete authored payload. Switching `enabled` never clears `values`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Feature<T> {
    pub enabled: bool,
    pub values: T,
}

impl<T: Default> Feature<T> {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            values: T::default(),
        }
    }
    pub fn enabled(values: T) -> Self {
        Self {
            enabled: true,
            values,
        }
    }
}

impl<T: Default> Default for Feature<T> {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimatedInput<T> {
    pub id: TrackId,
    /// Used before the first key and by tools that cannot evaluate the track.
    pub fallback: T,
    pub interpolation: Interpolation,
    /// Keys are ordered by strictly increasing seconds.
    pub keys: Vec<AnimationKey<T>>,
}

impl<T: MaterialValue> AnimatedInput<T> {
    fn validate(&self, path: &str) -> Result<(), SchemaError> {
        self.id.validate(&format!("{path}.id"))?;
        self.fallback.validate_finite(&format!("{path}.fallback"))?;
        if self.keys.is_empty() {
            return Err(SchemaError::new(
                path,
                ErrorKind::InvalidAnimation("a track needs at least one key".into()),
            ));
        }
        let mut previous = None;
        for (i, key) in self.keys.iter().enumerate() {
            let key_path = format!("{path}.keys[{i}]");
            finite(key.time_seconds, &format!("{key_path}.time_seconds"))?;
            key.value.validate_finite(&format!("{key_path}.value"))?;
            if previous.is_some_and(|time| key.time_seconds <= time) {
                return Err(SchemaError::new(
                    key_path,
                    ErrorKind::InvalidAnimation("key times must strictly increase".into()),
                ));
            }
            previous = Some(key.time_seconds);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationKey<T> {
    pub time_seconds: f64,
    pub value: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    Step,
    #[default]
    Linear,
    Cubic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldInput<T> {
    pub field: FieldId,
    pub fallback: T,
    pub mapping: Option<Affine<T>>,
    pub remap: Option<Remap>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureInput<T> {
    pub texture: TextureId,
    pub fallback: T,
    pub channels: ChannelSelector,
    pub mapping: Option<Affine<T>>,
    pub remap: Option<Remap>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Affine<T> {
    pub scale: T,
    pub offset: T,
}

impl<T> Affine<T> {
    fn for_each(&self, visit: &mut impl FnMut(&T)) {
        visit(&self.scale);
        visit(&self.offset);
    }
    fn validate(&self, path: &str) -> Result<(), SchemaError>
    where
        T: MaterialValue,
    {
        self.scale.validate_finite(&format!("{path}.scale"))?;
        self.offset.validate_finite(&format!("{path}.offset"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Remap {
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub clamp: bool,
}

impl Remap {
    pub(crate) fn validate(&self, path: &str) -> Result<(), SchemaError> {
        for (i, value) in self.from.iter().chain(self.to.iter()).enumerate() {
            finite(*value, &format!("{path}[{i}]"))?;
        }
        if self.from[0] == self.from[1] {
            return Err(SchemaError::new(
                path,
                ErrorKind::InvalidFieldGraph("remap input range is empty".into()),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelSelector {
    #[default]
    Rgb,
    Rgba,
    Red,
    Green,
    Blue,
    Alpha,
    Vector2,
    Vector3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureResource {
    pub uri: String,
    pub color_space: TextureColorSpace,
    pub sampling: TextureSampling,
}

impl TextureResource {
    pub(crate) fn validate(&self, path: &str) -> Result<(), SchemaError> {
        if self.uri.trim().is_empty() {
            return Err(SchemaError::new(
                format!("{path}.uri"),
                ErrorKind::InvalidIdentity,
            ));
        }
        self.sampling.validate(&format!("{path}.sampling"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureColorSpace {
    #[default]
    Srgb,
    Linear,
    NonColor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureSampling {
    pub uv_set: String,
    pub projection: Projection,
    pub transform: UvTransform,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub filter: FilterMode,
}

impl Default for TextureSampling {
    fn default() -> Self {
        Self {
            uv_set: "uv0".into(),
            projection: Projection::Uv,
            transform: UvTransform::default(),
            wrap_u: WrapMode::Repeat,
            wrap_v: WrapMode::Repeat,
            filter: FilterMode::LinearMip,
        }
    }
}

impl TextureSampling {
    fn validate(&self, path: &str) -> Result<(), SchemaError> {
        if self.uv_set.trim().is_empty() {
            return Err(SchemaError::new(
                format!("{path}.uv_set"),
                ErrorKind::EmptyIdentity,
            ));
        }
        self.transform.validate(&format!("{path}.transform"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    #[default]
    Uv,
    Object,
    World,
    Triplanar,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WrapMode {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterMode {
    Nearest,
    Linear,
    #[default]
    LinearMip,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UvTransform {
    pub offset: [f64; 2],
    pub scale: [f64; 2],
    pub rotation_radians: f64,
}

impl Default for UvTransform {
    fn default() -> Self {
        Self {
            offset: [0.0; 2],
            scale: [1.0; 2],
            rotation_radians: 0.0,
        }
    }
}

impl UvTransform {
    fn validate(&self, path: &str) -> Result<(), SchemaError> {
        for (i, value) in self.offset.iter().chain(self.scale.iter()).enumerate() {
            finite(*value, &format!("{path}[{i}]"))?;
        }
        finite(self.rotation_radians, &format!("{path}.rotation_radians"))?;
        Ok(())
    }
}
