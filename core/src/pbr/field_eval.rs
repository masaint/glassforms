//! Deterministic CPU reference evaluation for the canonical material field graph.
//!
//! This is deliberately independent of geometry construction and Surface Studio. The Studio's
//! gradient-noise folds and nearest/second-nearest cell distances inform the pattern semantics, but
//! these f64/u64 building blocks are not bit-compatible with its f32 shaders. There is one noise
//! scale per node; octaves, footprint filtering, image sampling, and shader generation remain later
//! integration work.

use std::collections::BTreeMap;

use crate::pbr::field::{
    BuiltinField, FieldGraph, FieldId, FieldNode, FieldValue, MathOp, NoiseKind, ValueType,
};
use crate::pbr::value::{ErrorKind, SchemaError};

/// Every builtin field is sourced from this explicit value; there is no global clock or hit state.
/// Scalar builtins other than time default to zero. The deterministic seed is an authored scalar,
/// separate from the exact `u64` seed stored by each procedural node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldContext {
    pub object_position: [f64; 3],
    pub world_position: [f64; 3],
    pub uv: [f64; 2],
    pub time_seconds: f64,
    pub deterministic_seed: f64,
    pub geometric_normal: [f64; 3],
    pub shading_normal: [f64; 3],
    pub tangent: [f64; 3],
    pub height: f64,
    pub radial_distance: f64,
    pub axial_distance: f64,
    pub authored_distance: f64,
    pub sdf_distance: f64,
    pub curvature: f64,
    pub edge: f64,
    pub cavity: f64,
    pub shelter: f64,
}

impl Default for FieldContext {
    fn default() -> Self {
        Self {
            object_position: [0.0; 3],
            world_position: [0.0; 3],
            uv: [0.0; 2],
            time_seconds: 0.0,
            deterministic_seed: 0.0,
            geometric_normal: [0.0, 0.0, 1.0],
            shading_normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            height: 0.0,
            radial_distance: 0.0,
            axial_distance: 0.0,
            authored_distance: 0.0,
            sdf_distance: 0.0,
            curvature: 0.0,
            edge: 0.0,
            cavity: 0.0,
            shelter: 0.0,
        }
    }
}

impl FieldGraph {
    /// Validate and evaluate one stable output ID. Shared dependencies are evaluated once per call.
    /// A non-finite context value is rejected if it reaches an output. Runtime math failures and
    /// unavailable image resources are errors rather than fallback values.
    pub fn evaluate(
        &self,
        output: &FieldId,
        context: &FieldContext,
    ) -> Result<FieldValue, SchemaError> {
        self.validate()?;
        let mut values = BTreeMap::new();
        self.evaluate_node(output, context, &mut values)
    }

    pub fn evaluate_scalar(
        &self,
        output: &FieldId,
        context: &FieldContext,
    ) -> Result<f64, SchemaError> {
        let value = self.evaluate(output, context)?;
        match value {
            FieldValue::Scalar(value) => Ok(value),
            other => Err(SchemaError::new(
                format!("fields.{output}"),
                ErrorKind::FieldType {
                    expected: ValueType::Scalar,
                    actual: other.value_type(),
                },
            )),
        }
    }

    fn evaluate_node(
        &self,
        id: &FieldId,
        context: &FieldContext,
        values: &mut BTreeMap<FieldId, FieldValue>,
    ) -> Result<FieldValue, SchemaError> {
        if let Some(value) = values.get(id) {
            return Ok(value.clone());
        }
        let node = self.nodes.get(id).cloned().ok_or_else(|| {
            SchemaError::new(format!("fields.{id}"), ErrorKind::UnknownField(id.clone()))
        })?;
        let path = format!("fields.{id}");
        let mut dependency = |dependency: &FieldId| self.evaluate_node(dependency, context, values);
        let value = match node {
            FieldNode::Constant(value) => value,
            FieldNode::Builtin(source) => builtin(source, context),
            FieldNode::Time { scale } => FieldValue::Scalar(context.time_seconds * scale),
            FieldNode::Noise {
                kind,
                position,
                scale,
                seed,
            } => {
                let position = dependency(&position)?;
                let position = vector_position(&position).ok_or_else(|| {
                    SchemaError::new(
                        format!("{path}.position"),
                        ErrorKind::FieldType {
                            expected: ValueType::Vector3,
                            actual: position.value_type(),
                        },
                    )
                })?;
                let scaled = [
                    position[0] * scale,
                    position[1] * scale,
                    position[2] * scale,
                ];
                if scaled.iter().any(|value| !value.is_finite()) {
                    return Err(SchemaError::new(
                        format!("{path}.position"),
                        ErrorKind::NonFinite,
                    ));
                }
                FieldValue::Scalar(pattern(kind, scaled, seed, &path)?)
            }
            FieldNode::Math { op, inputs } => {
                let inputs = inputs
                    .iter()
                    .map(dependency)
                    .collect::<Result<Vec<_>, _>>()?;
                evaluate_math(op, &inputs, &path)?
            }
            FieldNode::Remap {
                input,
                from,
                to,
                clamp,
            } => {
                let input = scalar(&dependency(&input)?).ok_or_else(|| {
                    SchemaError::new(
                        format!("{path}.input"),
                        ErrorKind::FieldEvaluation("remap input is not a scalar".into()),
                    )
                })?;
                FieldValue::Scalar(remap_scalar(input, from, to, clamp))
            }
            FieldNode::Compose { output, components } => {
                let components = components
                    .iter()
                    .map(dependency)
                    .collect::<Result<Vec<_>, _>>()?;
                let components = components
                    .iter()
                    .map(|value| {
                        scalar(value).ok_or_else(|| {
                            SchemaError::new(
                                format!("{path}.components"),
                                ErrorKind::FieldEvaluation(
                                    "compose component is not a scalar".into(),
                                ),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                match output {
                    ValueType::Vector2 => FieldValue::Vector2([components[0], components[1]]),
                    ValueType::Vector3 => {
                        FieldValue::Vector3([components[0], components[1], components[2]])
                    }
                    ValueType::Color => {
                        FieldValue::Color([components[0], components[1], components[2]])
                    }
                    ValueType::Scalar => {
                        return Err(SchemaError::new(
                            format!("{path}.output"),
                            ErrorKind::FieldEvaluation("compose cannot output a scalar".into()),
                        ));
                    }
                }
            }
            FieldNode::Convert { input, output } => {
                let input = dependency(&input)?;
                match (input, output) {
                    (FieldValue::Vector3(value), ValueType::Color) => FieldValue::Color(value),
                    (FieldValue::Color(value), ValueType::Vector3) => FieldValue::Vector3(value),
                    (other, _) => {
                        return Err(SchemaError::new(
                            format!("{path}.input"),
                            ErrorKind::FieldEvaluation(format!(
                                "cannot convert {:?} to {output:?}",
                                other.value_type()
                            )),
                        ));
                    }
                }
            }
            FieldNode::Texture { .. } => {
                return Err(SchemaError::new(
                    path,
                    ErrorKind::UnsupportedFieldEvaluation(
                        "texture nodes need an explicit image sampler/resource binding",
                    ),
                ));
            }
        };
        value.validate(&path)?;
        values.insert(id.clone(), value.clone());
        Ok(value)
    }
}

fn builtin(source: BuiltinField, context: &FieldContext) -> FieldValue {
    match source {
        BuiltinField::ObjectPosition => FieldValue::Vector3(context.object_position),
        BuiltinField::WorldPosition => FieldValue::Vector3(context.world_position),
        BuiltinField::Uv => FieldValue::Vector2(context.uv),
        BuiltinField::DeterministicSeed => FieldValue::Scalar(context.deterministic_seed),
        BuiltinField::GeometricNormal => FieldValue::Vector3(context.geometric_normal),
        BuiltinField::ShadingNormal => FieldValue::Vector3(context.shading_normal),
        BuiltinField::Tangent => FieldValue::Vector3(context.tangent),
        BuiltinField::Height => FieldValue::Scalar(context.height),
        BuiltinField::RadialDistance => FieldValue::Scalar(context.radial_distance),
        BuiltinField::AxialDistance => FieldValue::Scalar(context.axial_distance),
        BuiltinField::AuthoredDistance => FieldValue::Scalar(context.authored_distance),
        BuiltinField::SdfDistance => FieldValue::Scalar(context.sdf_distance),
        BuiltinField::Curvature => FieldValue::Scalar(context.curvature),
        BuiltinField::Edge => FieldValue::Scalar(context.edge),
        BuiltinField::Cavity => FieldValue::Scalar(context.cavity),
        BuiltinField::Shelter => FieldValue::Scalar(context.shelter),
    }
}

fn remap_scalar(input: f64, from: [f64; 2], to: [f64; 2], clamp: bool) -> f64 {
    if to[0] == to[1] {
        return to[0];
    }
    let input = if clamp {
        if from[0] < from[1] {
            input.clamp(from[0], from[1])
        } else {
            input.clamp(from[1], from[0])
        }
    } else {
        input
    };
    let numerator = scaled_difference(input, from[0]);
    let denominator = scaled_difference(from[1], from[0]);
    let output_range = scaled_difference(to[1], to[0]);
    let scaled_numerator = scale_binary(numerator.value, numerator.exponent - denominator.exponent);
    let mut t = scaled_numerator / denominator.value;
    if clamp {
        t = t.clamp(0.0, 1.0);
    }
    let interpolated = if output_range.exponent == 0 {
        to[0] + t * output_range.value
    } else {
        // A convex combination cannot overflow while t is in 0..=1. Extrapolation of an
        // output range that is itself unrepresentable is allowed to fail the finite-result check.
        to[0] * (1.0 - t) + to[1] * t
    };
    if ((t.abs() < f64::MIN_POSITIVE && numerator.value != 0.0)
        || (interpolated.abs() < f64::MIN_POSITIVE
            && numerator.value != 0.0
            && output_range.value != 0.0))
        || !interpolated.is_finite()
    {
        // The normalized ratio can underflow or overflow even though multiplying it by the output
        // range produces a finite answer. A rounded subnormal ratio can also double-round the final
        // product. Keep the three factors as mantissas and binary exponents until their product is
        // representable. Ordinary ratios retain the direct path above.
        let product = scaled_ratio_product(numerator, denominator, output_range);
        to[0] + product
    } else {
        interpolated
    }
}

/// A finite difference, halved with a binary exponent correction if the direct subtraction overflows.
#[derive(Clone, Copy)]
struct ScaledDifference {
    value: f64,
    exponent: i32,
}

fn scaled_difference(a: f64, b: f64) -> ScaledDifference {
    let direct = a - b;
    if direct.is_finite() {
        ScaledDifference {
            value: direct,
            exponent: 0,
        }
    } else {
        ScaledDifference {
            value: a * 0.5 - b * 0.5,
            exponent: 1,
        }
    }
}

/// The sign and absolute mantissa/exponent of a finite nonzero f64. Normal values have a
/// mantissa in [0.5, 1); the smallest subnormals retain a 0.25 mantissa with a corrected exponent.
fn binary_parts(value: f64) -> (f64, i32) {
    let bits = value.abs().to_bits();
    let stored_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (significand, exponent) = if stored_exponent == 0 {
        let shift = fraction.leading_zeros() - 12;
        (fraction << shift, -1021 - shift as i32)
    } else {
        ((1u64 << 52) | fraction, stored_exponent - 1022)
    };
    (
        (significand as f64 / 9_007_199_254_740_992.0).copysign(value),
        exponent,
    )
}

fn power_of_two(exponent: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn scale_binary(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 1023 {
        value *= power_of_two(1023);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= power_of_two(-1022);
        exponent += 1022;
    }
    value * power_of_two(exponent.clamp(-1022, 1023))
}

fn scaled_ratio_product(
    numerator: ScaledDifference,
    denominator: ScaledDifference,
    output: ScaledDifference,
) -> f64 {
    if numerator.value == 0.0 || output.value == 0.0 {
        return 0.0;
    }
    let (numerator_mantissa, numerator_exponent) = binary_parts(numerator.value);
    let (denominator_mantissa, denominator_exponent) = binary_parts(denominator.value);
    let (output_mantissa, output_exponent) = binary_parts(output.value);
    let mantissa = numerator_mantissa / denominator_mantissa * output_mantissa;
    scale_binary(
        mantissa,
        numerator_exponent + numerator.exponent - denominator_exponent - denominator.exponent
            + output_exponent
            + output.exponent,
    )
}

fn scalar(value: &FieldValue) -> Option<f64> {
    match value {
        FieldValue::Scalar(value) => Some(*value),
        _ => None,
    }
}

fn vector_position(value: &FieldValue) -> Option<[f64; 3]> {
    match value {
        FieldValue::Vector2(value) => Some([value[0], value[1], 0.0]),
        FieldValue::Vector3(value) => Some(*value),
        _ => None,
    }
}

fn map_unary(value: &FieldValue, f: impl Fn(f64) -> f64) -> FieldValue {
    match value {
        FieldValue::Scalar(a) => FieldValue::Scalar(f(*a)),
        FieldValue::Vector2(a) => FieldValue::Vector2([f(a[0]), f(a[1])]),
        FieldValue::Vector3(a) => FieldValue::Vector3([f(a[0]), f(a[1]), f(a[2])]),
        FieldValue::Color(a) => FieldValue::Color([f(a[0]), f(a[1]), f(a[2])]),
    }
}

fn map_same(a: &FieldValue, b: &FieldValue, f: impl Fn(f64, f64) -> f64) -> Option<FieldValue> {
    match (a, b) {
        (FieldValue::Scalar(a), FieldValue::Scalar(b)) => Some(FieldValue::Scalar(f(*a, *b))),
        (FieldValue::Vector2(a), FieldValue::Vector2(b)) => {
            Some(FieldValue::Vector2([f(a[0], b[0]), f(a[1], b[1])]))
        }
        (FieldValue::Vector3(a), FieldValue::Vector3(b)) => Some(FieldValue::Vector3([
            f(a[0], b[0]),
            f(a[1], b[1]),
            f(a[2], b[2]),
        ])),
        (FieldValue::Color(a), FieldValue::Color(b)) => Some(FieldValue::Color([
            f(a[0], b[0]),
            f(a[1], b[1]),
            f(a[2], b[2]),
        ])),
        _ => None,
    }
}

fn map_scalar_vector(
    a: &FieldValue,
    b: &FieldValue,
    f: impl Fn(f64, f64) -> f64 + Copy,
) -> Option<FieldValue> {
    match (a, b) {
        (FieldValue::Scalar(a), FieldValue::Vector2(b)) => {
            Some(FieldValue::Vector2([f(*a, b[0]), f(*a, b[1])]))
        }
        (FieldValue::Vector2(a), FieldValue::Scalar(b)) => {
            Some(FieldValue::Vector2([f(a[0], *b), f(a[1], *b)]))
        }
        _ => map_scalar_vector3(a, b, f),
    }
}

// Kept separate to make the scalar/vector operand order explicit for non-commutative division.
fn map_scalar_vector3(
    a: &FieldValue,
    b: &FieldValue,
    f: impl Fn(f64, f64) -> f64 + Copy,
) -> Option<FieldValue> {
    fn apply3(
        scalar: f64,
        vector: &[f64; 3],
        scalar_first: bool,
        f: impl Fn(f64, f64) -> f64,
    ) -> [f64; 3] {
        let apply = |v| {
            if scalar_first {
                f(scalar, v)
            } else {
                f(v, scalar)
            }
        };
        [apply(vector[0]), apply(vector[1]), apply(vector[2])]
    }
    match (a, b) {
        (FieldValue::Scalar(a), FieldValue::Vector3(b)) => {
            Some(FieldValue::Vector3(apply3(*a, b, true, f)))
        }
        (FieldValue::Vector3(a), FieldValue::Scalar(b)) => {
            Some(FieldValue::Vector3(apply3(*b, a, false, f)))
        }
        (FieldValue::Scalar(a), FieldValue::Color(b)) => {
            Some(FieldValue::Color(apply3(*a, b, true, f)))
        }
        (FieldValue::Color(a), FieldValue::Scalar(b)) => {
            Some(FieldValue::Color(apply3(*b, a, false, f)))
        }
        _ => None,
    }
}

fn components(value: &FieldValue) -> Vec<f64> {
    match value {
        FieldValue::Scalar(a) => vec![*a],
        FieldValue::Vector2(a) => a.to_vec(),
        FieldValue::Vector3(a) | FieldValue::Color(a) => a.to_vec(),
    }
}

fn stable_length(value: &FieldValue) -> Option<f64> {
    if !matches!(value, FieldValue::Vector2(_) | FieldValue::Vector3(_)) {
        return None;
    }
    let values = components(value);
    let scale = values.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
    if scale == 0.0 {
        return Some(0.0);
    }
    let scaled_norm = values
        .iter()
        .map(|value| {
            let scaled = *value / scale;
            scaled * scaled
        })
        .sum::<f64>()
        .sqrt();
    Some(scale * scaled_norm)
}

fn invalid_math(path: &str) -> SchemaError {
    SchemaError::new(
        path,
        ErrorKind::FieldEvaluation("validated math operands could not be evaluated".into()),
    )
}

fn evaluate_math(op: MathOp, values: &[FieldValue], path: &str) -> Result<FieldValue, SchemaError> {
    let invalid = || invalid_math(path);
    match op {
        MathOp::Add | MathOp::Subtract | MathOp::Min | MathOp::Max => {
            let f: fn(f64, f64) -> f64 = match op {
                MathOp::Add => |a, b| a + b,
                MathOp::Subtract => |a, b| a - b,
                MathOp::Min => f64::min,
                MathOp::Max => f64::max,
                _ => unreachable!(),
            };
            map_same(&values[0], &values[1], f).ok_or_else(invalid)
        }
        MathOp::Multiply | MathOp::Divide => {
            if op == MathOp::Divide && components(&values[1]).contains(&0.0) {
                return Err(SchemaError::new(
                    path,
                    ErrorKind::FieldEvaluation("division by zero".into()),
                ));
            }
            let f: fn(f64, f64) -> f64 = if op == MathOp::Multiply {
                |a, b| a * b
            } else {
                |a, b| a / b
            };
            map_same(&values[0], &values[1], f)
                .or_else(|| map_scalar_vector(&values[0], &values[1], f))
                .ok_or_else(invalid)
        }
        MathOp::Power => match (&values[0], &values[1]) {
            (FieldValue::Scalar(a), FieldValue::Scalar(b)) => {
                Ok(FieldValue::Scalar(libm::pow(*a, *b)))
            }
            _ => Err(invalid()),
        },
        MathOp::Dot => match (&values[0], &values[1]) {
            (FieldValue::Vector2(a), FieldValue::Vector2(b)) => {
                Ok(FieldValue::Scalar(a[0] * b[0] + a[1] * b[1]))
            }
            (FieldValue::Vector3(a), FieldValue::Vector3(b)) => {
                Ok(FieldValue::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
            }
            _ => Err(invalid()),
        },
        MathOp::Cross => match (&values[0], &values[1]) {
            (FieldValue::Vector3(a), FieldValue::Vector3(b)) => Ok(FieldValue::Vector3([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])),
            _ => Err(invalid()),
        },
        MathOp::Length => stable_length(&values[0])
            .map(FieldValue::Scalar)
            .ok_or_else(invalid),
        MathOp::Normalize => {
            if !matches!(&values[0], FieldValue::Vector2(_) | FieldValue::Vector3(_)) {
                return Err(invalid());
            }
            let vector = components(&values[0]);
            let scale = vector.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
            if scale == 0.0 || !scale.is_finite() {
                return Err(SchemaError::new(
                    path,
                    ErrorKind::FieldEvaluation(
                        "cannot normalize a zero-length or non-finite vector".into(),
                    ),
                ));
            }
            let scaled_norm = vector
                .iter()
                .map(|value| {
                    let scaled = *value / scale;
                    scaled * scaled
                })
                .sum::<f64>()
                .sqrt();
            // Dividing by the largest component first also normalizes vectors whose actual
            // length exceeds f64::MAX, without changing their representable direction.
            Ok(map_unary(&values[0], |value| (value / scale) / scaled_norm))
        }
        MathOp::Sin => Ok(map_unary(&values[0], libm::sin)),
        MathOp::Cos => Ok(map_unary(&values[0], libm::cos)),
        MathOp::Abs => Ok(map_unary(&values[0], f64::abs)),
    }
}

/// `Noise` is signed gradient noise (approximately -1..1). `Ridged` is `(1-|n|)^2`,
/// `Billow` is `|n|`, `Cells` is clamped nearest-feature distance, and `Veins` is one
/// minus clamped F2-F1 distance, so veins are high at cell boundaries. `Bands` is an
/// unphased 0..1 cosine along the first scaled coordinate and does not consume `seed`.
fn pattern(kind: NoiseKind, p: [f64; 3], seed: u64, path: &str) -> Result<f64, SchemaError> {
    if kind == NoiseKind::Bands {
        let phase = std::f64::consts::TAU * p[0];
        // Preserve ordinary-coordinate rounding; reduce only when the original phase overflows.
        let phase = if phase.is_finite() {
            phase
        } else {
            std::f64::consts::TAU * p[0].rem_euclid(1.0)
        };
        return Ok(0.5 + 0.5 * libm::cos(phase));
    }
    let (base, fraction) = pattern_lattice(p, path)?;
    match kind {
        NoiseKind::Noise | NoiseKind::Ridged | NoiseKind::Billow => {
            let noise = gradient_noise(base, fraction, seed);
            Ok(match kind {
                NoiseKind::Noise => noise,
                NoiseKind::Ridged => {
                    let ridge = 1.0 - noise.abs();
                    ridge * ridge
                }
                NoiseKind::Billow => noise.abs(),
                _ => unreachable!(),
            })
        }
        NoiseKind::Cells | NoiseKind::Veins => {
            let (first, second) = cell_distances(base, fraction, seed);
            if kind == NoiseKind::Cells {
                Ok(first.sqrt().min(1.0))
            } else {
                Ok(1.0 - (second.sqrt() - first.sqrt()).min(1.0))
            }
        }
        NoiseKind::Bands => unreachable!(),
    }
}

fn pattern_lattice(p: [f64; 3], path: &str) -> Result<([i64; 3], [f64; 3]), SchemaError> {
    let floor = p.map(f64::floor);
    let mut base = [0_i64; 3];
    for axis in 0..3 {
        // `i64::MAX as f64` rounds to 2^63, an exclusive upper bound. Staying one cell
        // away from either endpoint makes all neighbouring lattice arithmetic representable.
        if floor[axis] <= i64::MIN as f64 || floor[axis] >= i64::MAX as f64 {
            return Err(SchemaError::new(
                format!("{path}.position"),
                ErrorKind::FieldEvaluation(format!(
                    "pattern coordinate {} on axis {axis} is outside the checked i64 lattice",
                    p[axis]
                )),
            ));
        }
        base[axis] = floor[axis] as i64;
        if base[axis].checked_sub(1).is_none() || base[axis].checked_add(1).is_none() {
            return Err(SchemaError::new(
                format!("{path}.position"),
                ErrorKind::FieldEvaluation(format!(
                    "pattern coordinate {} on axis {axis} has no representable lattice neighbours",
                    p[axis]
                )),
            ));
        }
    }
    Ok((base, [p[0] - floor[0], p[1] - floor[1], p[2] - floor[2]]))
}

fn pcg3(mut value: [u32; 3]) -> [u32; 3] {
    for lane in &mut value {
        *lane = lane.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    value[0] = value[0].wrapping_add(value[1].wrapping_mul(value[2]));
    value[1] = value[1].wrapping_add(value[2].wrapping_mul(value[0]));
    value[2] = value[2].wrapping_add(value[0].wrapping_mul(value[1]));
    for lane in &mut value {
        *lane ^= *lane >> 16;
    }
    value[0] = value[0].wrapping_add(value[1].wrapping_mul(value[2]));
    value[1] = value[1].wrapping_add(value[2].wrapping_mul(value[0]));
    value[2] = value[2].wrapping_add(value[0].wrapping_mul(value[1]));
    value
}

fn hash3(cell: [i64; 3], seed: u64) -> [f64; 3] {
    // Low 32-bit lanes are a well-defined lattice hash for every i64 cell and retain the
    // original PCG inputs (and therefore values) throughout the former i32 coordinate range.
    let lo = seed as u32;
    let hi = (seed >> 32) as u32;
    let hash = pcg3([
        (cell[0] as u32)
            .wrapping_add(lo.wrapping_mul(747_796_405))
            .wrapping_add(hi.wrapping_mul(2_246_822_519)),
        (cell[1] as u32)
            .wrapping_add(lo.wrapping_mul(2_891_336_453))
            .wrapping_add(hi.rotate_left(13))
            .wrapping_add(17),
        (cell[2] as u32)
            .wrapping_add(lo.wrapping_mul(277_803_737))
            .wrapping_add(hi.wrapping_mul(3_266_489_917))
            .wrapping_add(101),
    ]);
    hash.map(|value| (value >> 8) as f64 * (1.0 / 16_777_216.0))
}

fn gradient_noise(base: [i64; 3], fraction: [f64; 3], seed: u64) -> f64 {
    let smooth =
        fraction.map(|value| value * value * value * (value * (value * 6.0 - 15.0) + 10.0));
    let mut corners = [0.0; 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        let offset = [index & 1, (index >> 1) & 1, (index >> 2) & 1];
        let cell = [
            base[0] + offset[0] as i64,
            base[1] + offset[1] as i64,
            base[2] + offset[2] as i64,
        ];
        let random = hash3(cell, seed);
        let mut gradient = [
            random[0] * 2.0 - 1.0 + 1e-4,
            random[1] * 2.0 - 1.0 + 1e-4,
            random[2] * 2.0 - 1.0 + 1e-4,
        ];
        let length = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        for value in &mut gradient {
            *value /= length;
        }
        let delta = [
            fraction[0] - offset[0] as f64,
            fraction[1] - offset[1] as f64,
            fraction[2] - offset[2] as f64,
        ];
        *corner = gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2];
    }
    let mix = |a, b, t| a + (b - a) * t;
    let x0 = mix(corners[0], corners[1], smooth[0]);
    let x1 = mix(corners[2], corners[3], smooth[0]);
    let x2 = mix(corners[4], corners[5], smooth[0]);
    let x3 = mix(corners[6], corners[7], smooth[0]);
    1.6 * mix(mix(x0, x1, smooth[1]), mix(x2, x3, smooth[1]), smooth[2])
}

fn cell_distances(base: [i64; 3], fraction: [f64; 3], seed: u64) -> (f64, f64) {
    let mut first = f64::INFINITY;
    let mut second = f64::INFINITY;
    for z in -1_i64..=1 {
        for y in -1_i64..=1 {
            for x in -1_i64..=1 {
                let cell = [base[0] + x, base[1] + y, base[2] + z];
                let random = hash3(cell, seed);
                // Distances are relative to the base cell, never differences of huge absolute
                // coordinates. The checked lattice guarantees these additions cannot overflow.
                let point = [
                    x as f64 + 0.5 + (random[0] - 0.5) * 0.72,
                    y as f64 + 0.5 + (random[1] - 0.5) * 0.72,
                    z as f64 + 0.5 + (random[2] - 0.5) * 0.72,
                ];
                let delta = [
                    point[0] - fraction[0],
                    point[1] - fraction[1],
                    point[2] - fraction[2],
                ];
                let distance = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                if distance < first {
                    second = first;
                    first = distance;
                } else if distance < second {
                    second = distance;
                }
            }
        }
    }
    (first, second)
}
