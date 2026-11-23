use std::collections::BTreeMap;

use gf_core::pbr::*;
use glassforms::pbr_field::{compile_field_wgsl, FieldWgslError, FUNCTION_NAME};

fn id(value: &str) -> FieldId {
    FieldId::from(value)
}

fn graph(nodes: impl IntoIterator<Item = (&'static str, FieldNode)>) -> FieldGraph {
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

fn validate(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}\n{source}", error.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{}\n{source}", error.emit_to_string(source)));
}

#[test]
fn specialised_source_memoises_shared_nodes_and_omits_unreachable_operations() {
    let fields = graph([
        ("shared", constant(0.375)),
        ("left", math(MathOp::Abs, &["shared"])),
        ("right", math(MathOp::Sin, &["shared"])),
        ("out", math(MathOp::Add, &["left", "right"])),
        ("unused:position", FieldNode::Builtin(BuiltinField::WorldPosition)),
        (
            "unused:noise",
            FieldNode::Noise {
                kind: NoiseKind::Cells,
                position: id("unused:position"),
                scale: 2.0,
                seed: 9,
            },
        ),
        ("unused:cos", math(MathOp::Cos, &["shared"])),
    ]);
    let compiled = compile_field_wgsl(&fields, &[id("out")]).unwrap();
    let source = compiled.source();
    validate(source);
    assert!(source.contains(&format!("fn {FUNCTION_NAME}(")));
    assert!(source.contains("abs(n0)"));
    assert!(source.contains("sin(n0)"));
    assert_eq!(source.matches("let n0: f32 = 0.375;").count(), 1);
    assert_eq!(source.matches("let n").count(), 4);
    assert!(!source.contains("cos("));
    assert!(!source.contains("pbr_field_hash3"));
    assert!(!source.contains("= context.world_position"));
    assert_eq!(compiled.outputs()[0].value_type, ValueType::Scalar);
    assert_eq!(compiled.outputs()[0].member, "output_0");
}

#[test]
fn every_math_shape_and_procedural_family_passes_naga() {
    let mut nodes = BTreeMap::from([
        (id("a"), constant(1.25)),
        (id("b"), constant(2.0)),
        (id("v"), FieldNode::Constant(FieldValue::Vector2([3.0, 4.0]))),
        (id("w"), FieldNode::Constant(FieldValue::Vector2([-2.0, 5.0]))),
        (id("x"), FieldNode::Constant(FieldValue::Vector3([1.0, -2.0, 3.0]))),
        (id("y"), FieldNode::Constant(FieldValue::Vector3([4.0, 5.0, -6.0]))),
        (id("c"), FieldNode::Constant(FieldValue::Color([0.2, 0.4, 0.8]))),
        (id("position"), FieldNode::Builtin(BuiltinField::ObjectPosition)),
        (id("uv"), FieldNode::Builtin(BuiltinField::Uv)),
    ]);
    for (name, op, inputs) in [
        ("add", MathOp::Add, &["a", "b"][..]),
        ("subtract", MathOp::Subtract, &["v", "w"][..]),
        ("multiply", MathOp::Multiply, &["c", "b"][..]),
        ("divide", MathOp::Divide, &["x", "b"][..]),
        ("min", MathOp::Min, &["v", "w"][..]),
        ("max", MathOp::Max, &["x", "y"][..]),
        ("power", MathOp::Power, &["a", "b"][..]),
        ("dot", MathOp::Dot, &["v", "w"][..]),
        ("cross", MathOp::Cross, &["x", "y"][..]),
        ("length2", MathOp::Length, &["v"][..]),
        ("length3", MathOp::Length, &["x"][..]),
        ("normalize2", MathOp::Normalize, &["v"][..]),
        ("normalize3", MathOp::Normalize, &["x"][..]),
        ("sin", MathOp::Sin, &["c"][..]),
        ("cos", MathOp::Cos, &["a"][..]),
        ("abs", MathOp::Abs, &["x"][..]),
    ] {
        nodes.insert(id(name), math(op, inputs));
    }
    nodes.insert(
        id("compose"),
        FieldNode::Compose {
            output: ValueType::Color,
            components: vec![id("a"), id("b"), id("dot")],
        },
    );
    nodes.insert(
        id("convert"),
        FieldNode::Convert {
            input: id("compose"),
            output: ValueType::Vector3,
        },
    );
    nodes.insert(
        id("remap"),
        FieldNode::Remap {
            input: id("a"),
            from: [-2.0, 3.0],
            to: [0.25, 0.75],
            clamp: true,
        },
    );
    nodes.insert(
        id("reversed"),
        FieldNode::Remap {
            input: id("a"),
            from: [3.0, -2.0],
            to: [1.0, 1.0],
            clamp: false,
        },
    );
    for kind in [
        NoiseKind::Noise,
        NoiseKind::Ridged,
        NoiseKind::Billow,
        NoiseKind::Cells,
        NoiseKind::Veins,
        NoiseKind::Bands,
    ] {
        nodes.insert(
            id(&format!("{kind:?}")),
            FieldNode::Noise {
                kind,
                position: id(if kind == NoiseKind::Cells { "uv" } else { "position" }),
                scale: 2.75,
                seed: 0x1234_5678_9abc_def0,
            },
        );
    }
    let fields = FieldGraph { nodes };
    let outputs = fields.nodes.keys().cloned().collect::<Vec<_>>();
    let compiled = compile_field_wgsl(&fields, &outputs).unwrap();
    validate(compiled.source());
    assert_eq!(compiled.source().matches("fn pbr_field_pcg3(").count(), 1);
    assert_eq!(compiled.source().matches("fn pbr_field_hash3(").count(), 1);
    assert_eq!(compiled.source().matches("fn pbr_field_gradient(").count(), 1);
    assert_eq!(compiled.source().matches("fn pbr_field_cells(").count(), 1);
    assert_eq!(compiled.outputs().len(), outputs.len());
}

#[test]
fn invalid_graphs_outputs_textures_and_unrepresentable_constants_fail_clearly() {
    let cycle = graph([
        ("a", math(MathOp::Abs, &["b"])),
        ("b", math(MathOp::Abs, &["a"])),
    ]);
    assert!(matches!(
        compile_field_wgsl(&cycle, &[id("a")]),
        Err(FieldWgslError::InvalidGraph(_))
    ));

    let ordinary = graph([("a", constant(1.0))]);
    assert!(matches!(
        compile_field_wgsl(&ordinary, &[]),
        Err(FieldWgslError::EmptyOutputs)
    ));
    assert!(matches!(
        compile_field_wgsl(&ordinary, &[id("missing")]),
        Err(FieldWgslError::UnknownOutput(_))
    ));
    assert!(matches!(
        compile_field_wgsl(&ordinary, &[id("a"), id("a")]),
        Err(FieldWgslError::DuplicateOutput(_))
    ));

    let texture = graph([(
        "image",
        FieldNode::Texture {
            texture: TextureId::from("texture:base"),
            output: ValueType::Color,
        },
    )]);
    let error = compile_field_wgsl(&texture, &[id("image")]).unwrap_err();
    assert!(matches!(error, FieldWgslError::UnsupportedTexture { .. }));
    assert!(error.to_string().contains("sampler/footprint"));
    assert!(error.to_string().contains("texture:base"));

    // An unreachable texture does not poison an otherwise specialised numeric output.
    let with_unused_texture = graph([
        ("a", constant(1.0)),
        (
            "image",
            FieldNode::Texture {
                texture: TextureId::from("texture:unused"),
                output: ValueType::Scalar,
            },
        ),
    ]);
    validate(compile_field_wgsl(&with_unused_texture, &[id("a")]).unwrap().source());

    let too_large = graph([("a", constant(f64::MAX))]);
    assert!(matches!(
        compile_field_wgsl(&too_large, &[id("a")]),
        Err(FieldWgslError::F32Range { .. })
    ));
    let collapsed = graph([
        ("a", constant(1.0)),
        (
            "remap",
            FieldNode::Remap {
                input: id("a"),
                from: [1.0, 1.0 + f64::EPSILON],
                to: [0.0, 1.0],
                clamp: false,
            },
        ),
    ]);
    assert!(matches!(
        compile_field_wgsl(&collapsed, &[id("remap")]),
        Err(FieldWgslError::F32RemapDomain(_))
    ));
}
