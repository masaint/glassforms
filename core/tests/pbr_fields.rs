//! CPU-only persistence, validation, and evaluation contracts for canonical typed fields.
use std::collections::BTreeMap;

use gf_core::pbr::*;

fn id(value: &str) -> FieldId {
    FieldId::from(value)
}

fn field_graph(nodes: impl IntoIterator<Item = (&'static str, FieldNode)>) -> FieldGraph {
    FieldGraph {
        nodes: nodes
            .into_iter()
            .map(|(name, node)| (id(name), node))
            .collect(),
    }
}

fn constant(value: f64) -> FieldNode {
    FieldNode::Constant(FieldValue::Scalar(value))
}

fn math(op: MathOp, inputs: &[&str]) -> FieldNode {
    FieldNode::Math {
        op,
        inputs: inputs.iter().map(|input| id(input)).collect(),
    }
}

fn scalar_at(graph: &FieldGraph, output: &str, context: &FieldContext) -> f64 {
    graph.evaluate_scalar(&id(output), context).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, found {actual}"
    );
}

#[test]
fn material_field_graph_round_trips_with_stable_identity_and_results() {
    let mut spec = MaterialSpec::default();
    spec.id = MaterialId::from("material:field-roundtrip");
    spec.fields = field_graph([
        ("object", FieldNode::Builtin(BuiltinField::ObjectPosition)),
        (
            "noise",
            FieldNode::Noise {
                kind: NoiseKind::Ridged,
                position: id("object"),
                scale: 3.25,
                seed: 0x1234_5678_9abc_def0,
            },
        ),
        (
            "roughness",
            FieldNode::Remap {
                input: id("noise"),
                from: [0.0, 1.0],
                to: [0.1, 0.7],
                clamp: true,
            },
        ),
        (
            "unused:vector",
            FieldNode::Constant(FieldValue::Vector3([1.0, 2.0, 3.0])),
        ),
        (
            "unused:colour",
            FieldNode::Convert {
                input: id("unused:vector"),
                output: ValueType::Color,
            },
        ),
    ]);
    spec.base.values.metalness = Input::Field(FieldInput {
        field: id("roughness"),
        fallback: 0.25,
        mapping: None,
        remap: None,
    });
    spec.validate().unwrap();

    let json = serde_json::to_string_pretty(&spec).unwrap();
    assert!(json.find("noise").unwrap() < json.find("object").unwrap());
    assert_eq!(json, serde_json::to_string_pretty(&spec).unwrap());
    let reopened: MaterialSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(reopened, spec);
    assert_eq!(reopened.id.as_str(), "material:field-roundtrip");
    assert_eq!(
        reopened.fields.nodes[&id("noise")],
        spec.fields.nodes[&id("noise")]
    );

    let context = FieldContext {
        object_position: [0.13, -0.41, 0.79],
        ..FieldContext::default()
    };
    assert_eq!(
        spec.fields.evaluate(&id("roughness"), &context).unwrap(),
        reopened
            .fields
            .evaluate(&id("roughness"), &context)
            .unwrap()
    );
    assert_eq!(
        serde_json::to_string_pretty(&reopened).unwrap(),
        json,
        "loading or evaluating mutated canonical persistence"
    );
}

#[test]
fn every_builtin_and_time_come_from_the_explicit_context() {
    let context = FieldContext {
        object_position: [1.0, 2.0, 3.0],
        world_position: [4.0, 5.0, 6.0],
        uv: [0.25, 0.75],
        time_seconds: 1.25,
        deterministic_seed: 91.0,
        geometric_normal: [0.0, 1.0, 0.0],
        shading_normal: [0.0, 0.0, -1.0],
        tangent: [1.0, 0.0, 0.0],
        height: 2.0,
        radial_distance: 3.0,
        axial_distance: 4.0,
        authored_distance: 5.0,
        sdf_distance: 6.0,
        curvature: 7.0,
        edge: 8.0,
        cavity: 9.0,
        shelter: 10.0,
    };
    let vector_cases = [
        (BuiltinField::ObjectPosition, context.object_position),
        (BuiltinField::WorldPosition, context.world_position),
        (BuiltinField::GeometricNormal, context.geometric_normal),
        (BuiltinField::ShadingNormal, context.shading_normal),
        (BuiltinField::Tangent, context.tangent),
    ];
    for (index, (source, expected)) in vector_cases.into_iter().enumerate() {
        let graph = field_graph([("value", FieldNode::Builtin(source))]);
        assert_eq!(
            graph.evaluate(&id("value"), &context).unwrap(),
            FieldValue::Vector3(expected),
            "vector builtin {index}"
        );
    }
    let uv = field_graph([("value", FieldNode::Builtin(BuiltinField::Uv))]);
    assert_eq!(
        uv.evaluate(&id("value"), &context).unwrap(),
        FieldValue::Vector2(context.uv)
    );
    let scalar_cases = [
        (BuiltinField::DeterministicSeed, context.deterministic_seed),
        (BuiltinField::Height, context.height),
        (BuiltinField::RadialDistance, context.radial_distance),
        (BuiltinField::AxialDistance, context.axial_distance),
        (BuiltinField::AuthoredDistance, context.authored_distance),
        (BuiltinField::SdfDistance, context.sdf_distance),
        (BuiltinField::Curvature, context.curvature),
        (BuiltinField::Edge, context.edge),
        (BuiltinField::Cavity, context.cavity),
        (BuiltinField::Shelter, context.shelter),
    ];
    for (source, expected) in scalar_cases {
        let graph = field_graph([("value", FieldNode::Builtin(source))]);
        close(scalar_at(&graph, "value", &context), expected);
        assert_eq!(graph.output_type(&id("value")).unwrap(), ValueType::Scalar);
    }
    let time = field_graph([("time", FieldNode::Time { scale: -2.0 })]);
    close(scalar_at(&time, "time", &context), -2.5);
    let later = FieldContext {
        time_seconds: 2.0,
        ..context
    };
    close(scalar_at(&time, "time", &later), -4.0);
}

#[test]
fn math_compose_convert_and_remap_have_typed_reference_semantics() {
    let graph = field_graph([
        ("a", constant(12.0)),
        ("b", constant(3.0)),
        ("zero", constant(0.0)),
        ("two", constant(2.0)),
        (
            "v",
            FieldNode::Constant(FieldValue::Vector3([1.0, 2.0, 3.0])),
        ),
        (
            "w",
            FieldNode::Constant(FieldValue::Vector3([4.0, 5.0, 6.0])),
        ),
        ("uv", FieldNode::Constant(FieldValue::Vector2([3.0, 4.0]))),
        ("divide", math(MathOp::Divide, &["a", "b"])),
        ("power", math(MathOp::Power, &["two", "b"])),
        ("scale", math(MathOp::Multiply, &["v", "two"])),
        ("inverse", math(MathOp::Divide, &["a", "v"])),
        ("dot", math(MathOp::Dot, &["v", "w"])),
        ("cross", math(MathOp::Cross, &["v", "w"])),
        ("length", math(MathOp::Length, &["uv"])),
        ("normalize", math(MathOp::Normalize, &["uv"])),
        ("sine", math(MathOp::Sin, &["zero"])),
        (
            "compose",
            FieldNode::Compose {
                output: ValueType::Color,
                components: vec![id("a"), id("b"), id("divide")],
            },
        ),
        (
            "convert",
            FieldNode::Convert {
                input: id("compose"),
                output: ValueType::Vector3,
            },
        ),
        (
            "remap",
            FieldNode::Remap {
                input: id("a"),
                from: [0.0, 24.0],
                to: [-1.0, 1.0],
                clamp: false,
            },
        ),
        (
            "clamped",
            FieldNode::Remap {
                input: id("high"),
                from: [0.0, 24.0],
                to: [0.0, 10.0],
                clamp: true,
            },
        ),
        ("high", constant(36.0)),
        (
            "unclamped",
            FieldNode::Remap {
                input: id("high"),
                from: [0.0, 24.0],
                to: [0.0, 2.0],
                clamp: false,
            },
        ),
        (
            "reversed",
            FieldNode::Remap {
                input: id("zero"),
                from: [1.0, -1.0],
                to: [0.0, 10.0],
                clamp: true,
            },
        ),
    ]);
    let context = FieldContext::default();
    close(scalar_at(&graph, "divide", &context), 4.0);
    close(scalar_at(&graph, "power", &context), 8.0);
    close(scalar_at(&graph, "dot", &context), 32.0);
    close(scalar_at(&graph, "length", &context), 5.0);
    close(scalar_at(&graph, "sine", &context), 0.0);
    close(scalar_at(&graph, "remap", &context), 0.0);
    close(scalar_at(&graph, "clamped", &context), 10.0);
    close(scalar_at(&graph, "unclamped", &context), 3.0);
    close(scalar_at(&graph, "reversed", &context), 5.0);
    assert_eq!(
        graph.evaluate(&id("scale"), &context).unwrap(),
        FieldValue::Vector3([2.0, 4.0, 6.0])
    );
    assert_eq!(
        graph.evaluate(&id("inverse"), &context).unwrap(),
        FieldValue::Vector3([12.0, 6.0, 4.0])
    );
    assert_eq!(
        graph.evaluate(&id("cross"), &context).unwrap(),
        FieldValue::Vector3([-3.0, 6.0, -3.0])
    );
    assert_eq!(
        graph.evaluate(&id("normalize"), &context).unwrap(),
        FieldValue::Vector2([0.6, 0.8])
    );
    assert_eq!(
        graph.evaluate(&id("convert"), &context).unwrap(),
        FieldValue::Vector3([12.0, 3.0, 4.0])
    );
    assert!(matches!(
        graph.evaluate_scalar(&id("convert"), &context),
        Err(SchemaError {
            kind: ErrorKind::FieldType {
                expected: ValueType::Scalar,
                actual: ValueType::Vector3,
            },
            ..
        })
    ));
}

#[test]
fn evaluation_math_domains_and_non_finite_context_fail_cleanly() {
    let divide = field_graph([
        ("a", constant(1.0)),
        ("zero", constant(0.0)),
        ("bad", math(MathOp::Divide, &["a", "zero"])),
    ]);
    assert!(matches!(
        divide.evaluate(&id("bad"), &FieldContext::default()),
        Err(SchemaError {
            kind: ErrorKind::FieldEvaluation(reason),
            ..
        }) if reason.contains("division by zero")
    ));
    let normalize = field_graph([
        ("zero", FieldNode::Constant(FieldValue::Vector3([0.0; 3]))),
        ("bad", math(MathOp::Normalize, &["zero"])),
    ]);
    assert!(matches!(
        normalize.evaluate(&id("bad"), &FieldContext::default()),
        Err(SchemaError {
            kind: ErrorKind::FieldEvaluation(_),
            ..
        })
    ));
    let power = field_graph([
        ("negative", constant(-1.0)),
        ("half", constant(0.5)),
        ("bad", math(MathOp::Power, &["negative", "half"])),
    ]);
    assert!(matches!(
        power.evaluate(&id("bad"), &FieldContext::default()),
        Err(SchemaError {
            kind: ErrorKind::NonFinite,
            ..
        })
    ));
    let position = field_graph([("position", FieldNode::Builtin(BuiltinField::ObjectPosition))]);
    let context = FieldContext {
        object_position: [f64::NAN, 0.0, 0.0],
        ..FieldContext::default()
    };
    assert!(matches!(
        position.evaluate(&id("position"), &context),
        Err(SchemaError {
            kind: ErrorKind::NonFinite,
            ..
        })
    ));
}

#[test]
fn extreme_remaps_and_vector_lengths_do_not_overflow_or_underflow_intermediates() {
    let remap = field_graph([
        ("zero", constant(0.0)),
        ("half", constant(0.5)),
        (
            "input_range",
            FieldNode::Remap {
                input: id("zero"),
                from: [-1e308, 1e308],
                to: [0.0, 1.0],
                clamp: false,
            },
        ),
        (
            "reversed_input_range",
            FieldNode::Remap {
                input: id("zero"),
                from: [1e308, -1e308],
                to: [0.0, 1.0],
                clamp: true,
            },
        ),
        (
            "output_range",
            FieldNode::Remap {
                input: id("half"),
                from: [0.0, 1.0],
                to: [-1e308, 1e308],
                clamp: true,
            },
        ),
        ("tiny_input", constant(1e-200)),
        ("huge_input", constant(1e200)),
        ("maximum", constant(1e308)),
        ("subnormal_input", constant(1.25 * 2.0_f64.powi(-500))),
        (
            "subnormal_ratio",
            FieldNode::Remap {
                input: id("subnormal_input"),
                from: [0.0, 2.0_f64.powi(574)],
                to: [0.0, 0.5],
                clamp: false,
            },
        ),
        (
            "overflowing_numerator",
            FieldNode::Remap {
                input: id("maximum"),
                from: [-1e308, 0.0],
                to: [0.0, 1.0],
                clamp: false,
            },
        ),
        (
            "underflowing_ratio",
            FieldNode::Remap {
                input: id("tiny_input"),
                from: [0.0, 1e200],
                to: [0.0, 1e200],
                clamp: false,
            },
        ),
        (
            "overflowing_ratio",
            FieldNode::Remap {
                input: id("huge_input"),
                from: [0.0, 1e-200],
                to: [0.0, 1e-200],
                clamp: false,
            },
        ),
    ]);
    let context = FieldContext::default();
    close(scalar_at(&remap, "input_range", &context), 0.5);
    close(scalar_at(&remap, "overflowing_numerator", &context), 2.0);
    assert_eq!(
        scalar_at(&remap, "subnormal_ratio", &context),
        f64::from_bits(1)
    );
    close(scalar_at(&remap, "reversed_input_range", &context), 0.5);
    close(scalar_at(&remap, "output_range", &context), 0.0);
    for output in ["underflowing_ratio", "overflowing_ratio"] {
        let expected = if output == "underflowing_ratio" {
            1e-200
        } else {
            1e200
        };
        let actual = scalar_at(&remap, output, &context);
        assert!(
            (actual / expected - 1.0).abs() < 1e-15,
            "{output}: expected {expected}, found {actual}"
        );
    }

    let vectors = field_graph([
        (
            "tiny",
            FieldNode::Constant(FieldValue::Vector2([1e-300, 0.0])),
        ),
        (
            "huge",
            FieldNode::Constant(FieldValue::Vector2([1e308, 0.0])),
        ),
        (
            "wide",
            FieldNode::Constant(FieldValue::Vector3([1e308, 1e308, 0.0])),
        ),
        (
            "max",
            FieldNode::Constant(FieldValue::Vector3([f64::MAX, -f64::MAX, f64::MAX])),
        ),
        ("tiny_length", math(MathOp::Length, &["tiny"])),
        ("huge_length", math(MathOp::Length, &["huge"])),
        ("wide_length", math(MathOp::Length, &["wide"])),
        ("tiny_normal", math(MathOp::Normalize, &["tiny"])),
        ("huge_normal", math(MathOp::Normalize, &["huge"])),
        ("max_normal", math(MathOp::Normalize, &["max"])),
    ]);
    assert_eq!(scalar_at(&vectors, "tiny_length", &context), 1e-300);
    close(scalar_at(&vectors, "huge_length", &context), 1e308);
    let wide = scalar_at(&vectors, "wide_length", &context);
    assert!((wide / (2.0_f64.sqrt() * 1e308) - 1.0).abs() < 1e-15);
    assert_eq!(
        vectors.evaluate(&id("tiny_normal"), &context).unwrap(),
        FieldValue::Vector2([1.0, 0.0])
    );
    assert_eq!(
        vectors.evaluate(&id("huge_normal"), &context).unwrap(),
        FieldValue::Vector2([1.0, 0.0])
    );
    let FieldValue::Vector3(normal) = vectors.evaluate(&id("max_normal"), &context).unwrap() else {
        panic!("normalization changed the vector type")
    };
    let expected = 1.0 / 3.0_f64.sqrt();
    close(normal[0], expected);
    close(normal[1], -expected);
    close(normal[2], expected);
}

#[test]
fn pattern_lattices_are_checked_and_large_cell_distances_stay_relative() {
    let patterns = field_graph([
        ("object", FieldNode::Builtin(BuiltinField::ObjectPosition)),
        (
            "noise",
            FieldNode::Noise {
                kind: NoiseKind::Noise,
                position: id("object"),
                scale: 1.0,
                seed: 73,
            },
        ),
        (
            "cells",
            FieldNode::Noise {
                kind: NoiseKind::Cells,
                position: id("object"),
                scale: 1.0,
                seed: 73,
            },
        ),
        (
            "veins",
            FieldNode::Noise {
                kind: NoiseKind::Veins,
                position: id("object"),
                scale: 1.0,
                seed: 73,
            },
        ),
        (
            "bands",
            FieldNode::Noise {
                kind: NoiseKind::Bands,
                position: id("object"),
                scale: 1.0,
                seed: 73,
            },
        ),
    ]);
    let boundary = FieldContext {
        object_position: [2_147_483_648.5, 0.25, -0.75],
        ..FieldContext::default()
    };
    let noise = scalar_at(&patterns, "noise", &boundary);
    let cells = scalar_at(&patterns, "cells", &boundary);
    let veins = scalar_at(&patterns, "veins", &boundary);
    close(noise, 0.195_384_573_291_080_9);
    close(cells, 0.393_372_681_582_830_4);
    close(veins, 0.778_181_004_375_939_8);

    let large = FieldContext {
        object_position: [1e18, -1e18, 0.0],
        ..FieldContext::default()
    };
    let large_cells = scalar_at(&patterns, "cells", &large);
    assert!(large_cells.is_finite());
    assert!(
        (0.0..1.0).contains(&large_cells),
        "large-cell distance lost its fractional position: {large_cells}"
    );
    assert_eq!(large_cells, scalar_at(&patterns, "cells", &large));
    close(scalar_at(&patterns, "noise", &large), 0.0);

    let unrepresentable = FieldContext {
        object_position: [1e300, 0.0, 0.0],
        ..FieldContext::default()
    };
    for output in ["noise", "cells", "veins"] {
        assert!(matches!(
            patterns.evaluate(&id(output), &unrepresentable),
            Err(SchemaError {
                kind: ErrorKind::FieldEvaluation(reason),
                ..
            }) if reason.contains("checked i64 lattice")
        ));
    }
    assert!(scalar_at(&patterns, "bands", &unrepresentable).is_finite());
    let overflowing_phase = FieldContext {
        object_position: [f64::MAX, 0.0, 0.0],
        ..FieldContext::default()
    };
    close(scalar_at(&patterns, "bands", &overflowing_phase), 1.0);
}

#[test]
fn graph_cycles_types_unknown_nodes_and_limits_are_typed_errors() {
    let cycle = field_graph([
        ("a", math(MathOp::Abs, &["b"])),
        ("b", math(MathOp::Abs, &["a"])),
    ]);
    assert!(matches!(
        cycle.validate(),
        Err(SchemaError {
            kind: ErrorKind::FieldCycle(_),
            ..
        })
    ));
    let unknown = field_graph([("a", math(MathOp::Abs, &["missing"]))]);
    assert!(matches!(
        unknown.validate(),
        Err(SchemaError {
            kind: ErrorKind::UnknownField(_),
            ..
        })
    ));
    let wrong_type = field_graph([
        ("scalar", constant(1.0)),
        (
            "noise",
            FieldNode::Noise {
                kind: NoiseKind::Noise,
                position: id("scalar"),
                scale: 1.0,
                seed: 0,
            },
        ),
    ]);
    assert!(matches!(
        wrong_type.validate(),
        Err(SchemaError {
            kind: ErrorKind::FieldType {
                expected: ValueType::Vector3,
                actual: ValueType::Scalar,
            },
            ..
        })
    ));

    let too_many = FieldGraph {
        nodes: (0..=MAX_FIELD_NODES)
            .map(|index| (id(&format!("constant:{index:04}")), constant(index as f64)))
            .collect::<BTreeMap<_, _>>(),
    };
    assert!(matches!(
        too_many.validate(),
        Err(SchemaError {
            kind: ErrorKind::FieldNodeLimit { count, limit },
            ..
        }) if count == MAX_FIELD_NODES + 1 && limit == MAX_FIELD_NODES
    ));

    let mut nodes = BTreeMap::from([(id("node:0000"), constant(1.0))]);
    for index in 1..=MAX_FIELD_DEPTH + 1 {
        nodes.insert(
            id(&format!("node:{index:04}")),
            math(MathOp::Abs, &[&format!("node:{:04}", index - 1)]),
        );
    }
    let mut deep = FieldGraph { nodes };
    assert!(matches!(
        deep.validate(),
        Err(SchemaError {
            kind: ErrorKind::FieldDepthLimit { limit },
            ..
        }) if limit == MAX_FIELD_DEPTH
    ));
    deep.nodes
        .remove(&id(&format!("node:{:04}", MAX_FIELD_DEPTH + 1)));
    deep.validate().unwrap();
}

#[test]
fn seeded_patterns_are_repeatable_varied_and_bounded() {
    let kinds = [
        NoiseKind::Noise,
        NoiseKind::Ridged,
        NoiseKind::Billow,
        NoiseKind::Cells,
        NoiseKind::Veins,
        NoiseKind::Bands,
    ];
    let mut nodes = BTreeMap::from([(
        id("object"),
        FieldNode::Builtin(BuiltinField::ObjectPosition),
    )]);
    for kind in kinds {
        nodes.insert(
            id(&format!("{kind:?}")),
            FieldNode::Noise {
                kind,
                position: id("object"),
                scale: 2.75,
                seed: 0x1234_5678_9abc_def0,
            },
        );
    }
    let graph = FieldGraph { nodes };
    graph.validate().unwrap();
    for kind in kinds {
        let output = id(&format!("{kind:?}"));
        let mut values = Vec::new();
        for z in 0..2 {
            for y in 0..4 {
                for x in 0..4 {
                    let context = FieldContext {
                        object_position: [
                            0.13 + x as f64 * 0.37,
                            -0.21 + y as f64 * 0.29,
                            0.07 + z as f64 * 0.43,
                        ],
                        ..FieldContext::default()
                    };
                    let first = graph.evaluate_scalar(&output, &context).unwrap();
                    let second = graph.evaluate_scalar(&output, &context).unwrap();
                    assert_eq!(first, second, "{kind:?} was not repeatable");
                    assert!(first.is_finite());
                    match kind {
                        NoiseKind::Noise | NoiseKind::Billow => {
                            assert!((0.0..=1.5).contains(&first.abs()))
                        }
                        _ => assert!((0.0..=1.0).contains(&first)),
                    }
                    values.push(first);
                }
            }
        }
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max - min > 1e-4, "{kind:?} had no representative variation");
    }

    let bands = field_graph([
        ("object", FieldNode::Builtin(BuiltinField::ObjectPosition)),
        (
            "bands",
            FieldNode::Noise {
                kind: NoiseKind::Bands,
                position: id("object"),
                scale: 1.0,
                seed: 123,
            },
        ),
    ]);
    for (x, expected) in [(0.0, 1.0), (0.25, 0.5), (0.5, 0.0)] {
        let context = FieldContext {
            object_position: [x, 19.0, -4.0],
            ..FieldContext::default()
        };
        close(scalar_at(&bands, "bands", &context), expected);
    }
}

#[test]
fn object_world_and_uv_attachment_and_full_width_seeds_are_deterministic() {
    let graph = field_graph([
        ("object", FieldNode::Builtin(BuiltinField::ObjectPosition)),
        ("world", FieldNode::Builtin(BuiltinField::WorldPosition)),
        ("uv", FieldNode::Builtin(BuiltinField::Uv)),
        (
            "object_noise",
            FieldNode::Noise {
                kind: NoiseKind::Noise,
                position: id("object"),
                scale: 4.0,
                seed: 17,
            },
        ),
        (
            "world_noise",
            FieldNode::Noise {
                kind: NoiseKind::Noise,
                position: id("world"),
                scale: 4.0,
                seed: 17,
            },
        ),
        (
            "uv_cells",
            FieldNode::Noise {
                kind: NoiseKind::Cells,
                position: id("uv"),
                scale: 5.0,
                seed: 18,
            },
        ),
    ]);
    let a = FieldContext {
        object_position: [0.13, 0.41, -0.27],
        world_position: [3.13, -2.0, 0.8],
        uv: [0.19, 0.72],
        ..FieldContext::default()
    };
    let b = FieldContext {
        world_position: [8.0, 1.0, -3.0],
        ..a
    };
    assert_eq!(
        scalar_at(&graph, "object_noise", &a),
        scalar_at(&graph, "object_noise", &b)
    );
    assert_ne!(
        scalar_at(&graph, "world_noise", &a),
        scalar_at(&graph, "world_noise", &b)
    );
    assert_eq!(
        scalar_at(&graph, "uv_cells", &a),
        scalar_at(&graph, "uv_cells", &b)
    );

    let seeded = |seed| {
        field_graph([
            ("object", FieldNode::Builtin(BuiltinField::ObjectPosition)),
            (
                "noise",
                FieldNode::Noise {
                    kind: NoiseKind::Noise,
                    position: id("object"),
                    scale: 4.0,
                    seed,
                },
            ),
        ])
    };
    let low = scalar_at(&seeded(17), "noise", &a);
    let same = scalar_at(&seeded(17), "noise", &a);
    let high = scalar_at(&seeded(17 + (1_u64 << 32)), "noise", &a);
    assert_eq!(low, same);
    assert_ne!(low, high, "the upper 32 seed bits were discarded");
}

#[test]
fn texture_nodes_are_persistent_but_require_an_explicit_sampler() {
    let graph = field_graph([(
        "image",
        FieldNode::Texture {
            texture: TextureId::from("texture:base"),
            output: ValueType::Color,
        },
    )]);
    graph.validate().unwrap();
    let json = serde_json::to_string(&graph).unwrap();
    let reopened: FieldGraph = serde_json::from_str(&json).unwrap();
    assert_eq!(reopened, graph);
    assert!(matches!(
        reopened.evaluate(&id("image"), &FieldContext::default()),
        Err(SchemaError {
            kind: ErrorKind::UnsupportedFieldEvaluation(_),
            ..
        })
    ));
}
