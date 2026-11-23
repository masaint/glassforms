use std::collections::BTreeMap;

use gf_core::pbr::*;
use glassforms::gpu::Gpu;
use glassforms::pbr_field::{compile_field_wgsl, FieldWgsl};

const CONTEXT_STRIDE: usize = 128;

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

fn f32_value(value: f64) -> f64 {
    f32::from_bits((value as f32).to_bits()) as f64
}

fn quantize(context: &FieldContext) -> FieldContext {
    let vector2 = |v: [f64; 2]| v.map(f32_value);
    let vector3 = |v: [f64; 3]| v.map(f32_value);
    FieldContext {
        object_position: vector3(context.object_position),
        world_position: vector3(context.world_position),
        uv: vector2(context.uv),
        time_seconds: f32_value(context.time_seconds),
        deterministic_seed: f32_value(context.deterministic_seed),
        geometric_normal: vector3(context.geometric_normal),
        shading_normal: vector3(context.shading_normal),
        tangent: vector3(context.tangent),
        height: f32_value(context.height),
        radial_distance: f32_value(context.radial_distance),
        axial_distance: f32_value(context.axial_distance),
        authored_distance: f32_value(context.authored_distance),
        sdf_distance: f32_value(context.sdf_distance),
        curvature: f32_value(context.curvature),
        edge: f32_value(context.edge),
        cavity: f32_value(context.cavity),
        shelter: f32_value(context.shelter),
    }
}

fn put_f32(bytes: &mut [u8], offset: usize, value: f64) {
    bytes[offset..offset + 4].copy_from_slice(&(value as f32).to_bits().to_le_bytes());
}

fn put_vector(bytes: &mut [u8], offset: usize, values: &[f64]) {
    for (lane, value) in values.iter().enumerate() {
        put_f32(bytes, offset + lane * 4, *value);
    }
}

fn context_bytes(context: &FieldContext) -> Vec<u8> {
    let mut bytes = vec![0_u8; CONTEXT_STRIDE];
    put_vector(&mut bytes, 0, &context.object_position);
    put_vector(&mut bytes, 16, &context.world_position);
    put_vector(&mut bytes, 32, &context.uv);
    put_f32(&mut bytes, 40, context.time_seconds);
    put_f32(&mut bytes, 44, context.deterministic_seed);
    put_vector(&mut bytes, 48, &context.geometric_normal);
    put_vector(&mut bytes, 64, &context.shading_normal);
    put_vector(&mut bytes, 80, &context.tangent);
    for (offset, value) in [
        (92, context.height),
        (96, context.radial_distance),
        (100, context.axial_distance),
        (104, context.authored_distance),
        (108, context.sdf_distance),
        (112, context.curvature),
        (116, context.edge),
        (120, context.cavity),
        (124, context.shelter),
    ] {
        put_f32(&mut bytes, offset, value);
    }
    bytes
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

fn execute(gpu: &Gpu, compiled: &FieldWgsl, contexts: &[FieldContext]) -> Vec<Vec<[f32; 4]>> {
    let output_count = compiled.outputs().len();
    let mut wrapper = format!(
        "{}\n@group(0) @binding(0) var<storage, read> contexts: array<PbrFieldContext>;\n@group(0) @binding(1) var<storage, read_write> results: array<vec4f>;\n@compute @workgroup_size(64)\nfn evaluate(@builtin(global_invocation_id) invocation: vec3u) {{\n  let case_index = invocation.x;\n  if (case_index >= arrayLength(&contexts)) {{ return; }}\n  var values: PbrFieldOutputs;\n  let ok = pbr_field_evaluate(contexts[case_index], &values);\n  if (!ok) {{\n",
        compiled.source()
    );
    for output in 0..output_count {
        wrapper.push_str(&format!(
            "    results[case_index * {output_count}u + {output}u] = vec4f(0.0);\n"
        ));
    }
    wrapper.push_str("    return;\n  }\n");
    for (output, metadata) in compiled.outputs().iter().enumerate() {
        let value = format!("values.{}", metadata.member);
        let vector = match metadata.value_type {
            ValueType::Scalar => format!("vec4f({value}, 0.0, 0.0, 1.0)"),
            ValueType::Vector2 => format!("vec4f({value}, 0.0, 1.0)"),
            ValueType::Vector3 | ValueType::Color => format!("vec4f({value}, 1.0)"),
        };
        wrapper.push_str(&format!(
            "  results[case_index * {output_count}u + {output}u] = {vector};\n"
        ));
    }
    wrapper.push_str("}\n");
    validate(&wrapper);

    let shader = gpu.shader("canonical PBR field numerical contract", &wrapper, false);
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("canonical PBR field numerical contract"),
            layout: None,
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
    let input_bytes = contexts
        .iter()
        .flat_map(|context| context_bytes(&quantize(context)))
        .collect::<Vec<_>>();
    let inputs = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR field contexts"),
        size: input_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&inputs, 0, &input_bytes);
    let result_bytes = (contexts.len() * output_count * 16) as u64;
    let results = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR field results"),
        size: result_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR field result download"),
        size: result_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PBR field contract"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: results.as_entire_binding(),
            },
        ],
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(contexts.len().div_ceil(64) as u32, 1, 1);
    }
    commands.copy_buffer_to_buffer(&results, 0, &download, 0, result_bytes);
    gpu.queue.submit(Some(commands.finish()));
    let (send, receive) = std::sync::mpsc::channel();
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap();
        });
    gpu.wait();
    receive.recv().unwrap().unwrap();
    let mapped = download.slice(..).get_mapped_range().unwrap();
    let floats: &[f32] = bytemuck::cast_slice(&mapped);
    floats
        .chunks_exact(4)
        .map(|v| [v[0], v[1], v[2], v[3]])
        .collect::<Vec<_>>()
        .chunks(output_count)
        .map(|case| case.to_vec())
        .collect()
}

fn components(value: &FieldValue) -> Vec<f64> {
    match value {
        FieldValue::Scalar(value) => vec![*value],
        FieldValue::Vector2(value) => value.to_vec(),
        FieldValue::Vector3(value) | FieldValue::Color(value) => value.to_vec(),
    }
}

fn compare(
    graph: &FieldGraph,
    compiled: &FieldWgsl,
    contexts: &[FieldContext],
    actual: &[Vec<[f32; 4]>],
    tolerance: f64,
) -> usize {
    let mut comparisons = 0;
    for (case_index, context) in contexts.iter().enumerate() {
        let context = quantize(context);
        for (output_index, metadata) in compiled.outputs().iter().enumerate() {
            let observed = actual[case_index][output_index];
            assert_eq!(observed[3], 1.0, "{} case {case_index} failed", metadata.id);
            let expected = graph.evaluate(&metadata.id, &context).unwrap();
            assert_eq!(expected.value_type(), metadata.value_type);
            for (lane, expected) in components(&expected).into_iter().enumerate() {
                let observed = observed[lane] as f64;
                let error = (observed - expected).abs();
                let allowed = tolerance * expected.abs().max(1.0);
                assert!(
                    error <= allowed,
                    "case {case_index} output {} lane {lane}: CPU {expected}, GPU {observed}, error {error} > {allowed}",
                    metadata.id
                );
                comparisons += 1;
            }
        }
    }
    comparisons
}

fn typed_graph() -> FieldGraph {
    let mut nodes = BTreeMap::from([
        (id("a"), constant(1.25)),
        (id("b"), constant(2.0)),
        (id("zero"), constant(0.0)),
        (id("v"), FieldNode::Constant(FieldValue::Vector2([3.0, -4.0]))),
        (id("w"), FieldNode::Constant(FieldValue::Vector2([-2.0, 5.0]))),
        (id("x"), FieldNode::Constant(FieldValue::Vector3([1.0, -2.0, 3.0]))),
        (id("y"), FieldNode::Constant(FieldValue::Vector3([4.0, 5.0, -6.0]))),
        (id("c"), FieldNode::Constant(FieldValue::Color([0.2, 0.4, 0.8]))),
        (id("zero_vector"), FieldNode::Constant(FieldValue::Vector2([0.0; 2]))),
    ]);
    for source in [
        BuiltinField::ObjectPosition,
        BuiltinField::WorldPosition,
        BuiltinField::GeometricNormal,
        BuiltinField::ShadingNormal,
        BuiltinField::Tangent,
        BuiltinField::Uv,
        BuiltinField::Height,
        BuiltinField::RadialDistance,
        BuiltinField::AxialDistance,
        BuiltinField::AuthoredDistance,
        BuiltinField::SdfDistance,
        BuiltinField::Curvature,
        BuiltinField::Edge,
        BuiltinField::Cavity,
        BuiltinField::Shelter,
        BuiltinField::DeterministicSeed,
    ] {
        nodes.insert(id(&format!("{source:?}")), FieldNode::Builtin(source));
    }
    nodes.insert(id("time"), FieldNode::Time { scale: -2.5 });
    for (name, op, inputs) in [
        ("add", MathOp::Add, &["a", "b"][..]),
        ("subtract", MathOp::Subtract, &["v", "w"][..]),
        ("multiply", MathOp::Multiply, &["c", "b"][..]),
        ("multiply_left", MathOp::Multiply, &["a", "v"][..]),
        ("divide", MathOp::Divide, &["x", "b"][..]),
        ("divide_left", MathOp::Divide, &["a", "y"][..]),
        ("min", MathOp::Min, &["v", "w"][..]),
        ("max", MathOp::Max, &["x", "y"][..]),
        ("power", MathOp::Power, &["a", "b"][..]),
        ("dot", MathOp::Dot, &["v", "w"][..]),
        ("dot3", MathOp::Dot, &["x", "y"][..]),
        ("cross", MathOp::Cross, &["x", "y"][..]),
        ("length2", MathOp::Length, &["v"][..]),
        ("length3", MathOp::Length, &["x"][..]),
        ("length_zero", MathOp::Length, &["zero_vector"][..]),
        ("normalize2", MathOp::Normalize, &["v"][..]),
        ("normalize3", MathOp::Normalize, &["x"][..]),
        ("sin", MathOp::Sin, &["c"][..]),
        ("cos", MathOp::Cos, &["a"][..]),
        ("abs", MathOp::Abs, &["x"][..]),
        ("shared", MathOp::Add, &["a", "b"][..]),
        ("shared_again", MathOp::Multiply, &["shared", "b"][..]),
    ] {
        nodes.insert(id(name), math(op, inputs));
    }
    for (name, output, components) in [
        ("compose2", ValueType::Vector2, &["a", "b"][..]),
        ("compose3", ValueType::Vector3, &["a", "b", "dot"][..]),
        ("compose_color", ValueType::Color, &["a", "b", "dot"][..]),
    ] {
        nodes.insert(
            id(name),
            FieldNode::Compose {
                output,
                components: components.iter().map(|input| id(input)).collect(),
            },
        );
    }
    nodes.insert(
        id("to_vector"),
        FieldNode::Convert {
            input: id("compose_color"),
            output: ValueType::Vector3,
        },
    );
    nodes.insert(
        id("to_color"),
        FieldNode::Convert {
            input: id("to_vector"),
            output: ValueType::Color,
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
            input: id("zero"),
            from: [1.0, -1.0],
            to: [0.0, 10.0],
            clamp: true,
        },
    );
    nodes.insert(
        id("flat_remap"),
        FieldNode::Remap {
            input: id("a"),
            from: [0.0, 2.0],
            to: [0.75, 0.75],
            clamp: false,
        },
    );
    FieldGraph { nodes }
}

fn contexts() -> Vec<FieldContext> {
    vec![
        FieldContext::default(),
        FieldContext {
            object_position: [0.13, -0.41, 0.79],
            world_position: [4.0, -5.0, 6.0],
            uv: [0.25, 0.75],
            time_seconds: 1.25,
            deterministic_seed: 91.0,
            geometric_normal: [0.0, 1.0, 0.0],
            shading_normal: [0.0, 0.0, -1.0],
            tangent: [1.0, 0.5, 0.0],
            height: 2.0,
            radial_distance: 3.0,
            axial_distance: 4.0,
            authored_distance: 5.0,
            sdf_distance: 6.0,
            curvature: 7.0,
            edge: 8.0,
            cavity: 9.0,
            shelter: 10.0,
        },
        FieldContext {
            object_position: [-1.0, 0.0, 1.0],
            world_position: [-100.5, 1000.25, 0.125],
            uv: [-0.5, 1.5],
            time_seconds: -3.75,
            deterministic_seed: -17.0,
            geometric_normal: [-1.0, 0.0, 0.0],
            shading_normal: [0.5, -0.5, 0.75],
            tangent: [-2.0, 3.0, -4.0],
            height: -2.0,
            radial_distance: 0.0,
            axial_distance: -4.0,
            authored_distance: 0.5,
            sdf_distance: -6.0,
            curvature: -7.0,
            edge: 1.0,
            cavity: 0.25,
            shelter: 0.75,
        },
        FieldContext {
            object_position: [1024.25, -2048.5, 0.999],
            world_position: [0.0; 3],
            uv: [1024.0, -1024.0],
            time_seconds: 1000.0,
            deterministic_seed: 4294967297.0,
            geometric_normal: [1.0, 1.0, 1.0],
            shading_normal: [1.0, -1.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            height: 1.0e-20,
            radial_distance: 1.0e20,
            axial_distance: -1.0e10,
            authored_distance: 1.0e-10,
            sdf_distance: 0.0,
            curvature: 1.0e5,
            edge: 1.0e-5,
            cavity: 1.0,
            shelter: 0.0,
        },
    ]
}

#[test]
fn vulkan_matches_cpu_typed_math_contexts_shared_dependencies_and_multiple_outputs() {
    let gpu = Gpu::headless("vulkan").expect("field WGSL qualification requires Vulkan");
    let fields = typed_graph();
    let outputs = fields.nodes.keys().cloned().collect::<Vec<_>>();
    let compiled = compile_field_wgsl(&fields, &outputs).unwrap();
    let shared = outputs.iter().position(|output| output.as_str() == "shared").unwrap();
    let shared_member = &compiled.outputs()[shared].member;
    let source = compiled.source();
    let declaration = format!("(*output).{shared_member} = ");
    let assignment = source.split_once(&declaration).unwrap().1;
    let shared_node = assignment.split(';').next().unwrap();
    assert_eq!(source.matches(&format!("let {shared_node}:")).count(), 1);
    let contexts = contexts();
    let actual = execute(&gpu, &compiled, &contexts);
    let comparisons = compare(&fields, &compiled, &contexts, &actual, 3.0e-5);
    assert!(comparisons > 400, "only {comparisons} typed comparisons");
    println!("typed/context/shared/multi-output component comparisons: {comparisons}");
}

fn pattern_graph(scale: f64) -> FieldGraph {
    let mut nodes = BTreeMap::from([
        (id("object"), FieldNode::Builtin(BuiltinField::ObjectPosition)),
        (id("uv"), FieldNode::Builtin(BuiltinField::Uv)),
    ]);
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
                position: id("object"),
                scale,
                seed: 0x1234_5678_9abc_def0,
            },
        );
    }
    nodes.insert(
        id("NoiseRepeat"),
        FieldNode::Noise {
            kind: NoiseKind::Noise,
            position: id("object"),
            scale,
            seed: 0x1234_5678_9abc_def0,
        },
    );
    nodes.insert(
        id("OtherSeed"),
        FieldNode::Noise {
            kind: NoiseKind::Noise,
            position: id("object"),
            scale,
            seed: 0x1234_5678_9abc_def1,
        },
    );
    nodes.insert(
        id("UvCells"),
        FieldNode::Noise {
            kind: NoiseKind::Cells,
            position: id("uv"),
            scale,
            seed: 0xffff_ffff_ffff_fffe,
        },
    );
    FieldGraph { nodes }
}

#[test]
fn vulkan_matches_all_cpu_patterns_seeds_and_checked_lattice_boundaries() {
    let gpu = Gpu::headless("vulkan").expect("field WGSL qualification requires Vulkan");
    let fields = pattern_graph(2.75);
    let outputs = [
        "object", "uv", "Noise", "NoiseRepeat", "OtherSeed", "Ridged", "Billow",
        "Cells", "Veins", "Bands", "UvCells",
    ]
    .map(id);
    let compiled = compile_field_wgsl(&fields, &outputs).unwrap();
    assert_eq!(compiled.source().matches("let n0: vec3f = context.object_position;").count(), 1);
    assert_eq!(compiled.source().matches("fn pbr_field_hash3(").count(), 1);
    let contexts = (0..18)
        .map(|index| FieldContext {
            object_position: [
                -1.25 + index as f64 * 0.173,
                (index % 5) as f64 - 2.0 + 0.03125,
                (index % 3) as f64 * -0.5 + 0.25,
            ],
            uv: [index as f64 * 0.125 - 0.75, index as f64 * -0.0625 + 0.5],
            ..FieldContext::default()
        })
        .chain([
            FieldContext { object_position: [-1.0, 0.0, 1.0], ..FieldContext::default() },
            FieldContext { object_position: [65536.0, -65536.0, 256.0], ..FieldContext::default() },
        ])
        .collect::<Vec<_>>();
    let actual = execute(&gpu, &compiled, &contexts);
    let comparisons = compare(&fields, &compiled, &contexts, &actual, 4.0e-4);
    for case in &actual {
        assert_eq!(case[2], case[3], "identical seeds were not repeatable");
    }
    assert!(
        actual.iter().any(|case| case[2] != case[4]),
        "the seed low bit did not affect any non-lattice noise sample"
    );
    println!("seeded procedural component comparisons: {comparisons}");

    // Exact f32 integers at both checked i32-neighbour limits retain the CPU's low-u32 hash.
    let boundary_fields = pattern_graph(1.0);
    let boundary_outputs = ["object", "Noise", "Cells", "Veins", "Bands"].map(id);
    let boundary = compile_field_wgsl(&boundary_fields, &boundary_outputs).unwrap();
    let boundaries = [
        [-2147483520.0, 0.25, -0.75],
        [2147483392.0, -0.25, 0.75],
    ]
    .map(|object_position| FieldContext {
        object_position,
        ..FieldContext::default()
    });
    let actual = execute(&gpu, &boundary, &boundaries);
    let boundary_comparisons = compare(
        &boundary_fields,
        &boundary,
        &boundaries,
        &actual,
        4.0e-4,
    );
    println!("checked lattice boundary component comparisons: {boundary_comparisons}");
}

#[test]
fn vulkan_reports_math_non_finite_and_lattice_domain_failures() {
    let gpu = Gpu::headless("vulkan").expect("field WGSL qualification requires Vulkan");
    let cases = [
        graph([
            ("a", constant(1.0)),
            ("zero", constant(0.0)),
            ("bad", math(MathOp::Divide, &["a", "zero"])),
        ]),
        graph([
            ("a", constant(1.0)),
            ("zero", FieldNode::Constant(FieldValue::Vector3([1.0, 0.0, 1.0]))),
            ("bad", math(MathOp::Divide, &["a", "zero"])),
        ]),
        graph([
            ("zero", FieldNode::Constant(FieldValue::Vector2([0.0; 2]))),
            ("bad", math(MathOp::Normalize, &["zero"])),
        ]),
        graph([
            ("negative", constant(-1.0)),
            ("half", constant(0.5)),
            ("bad", math(MathOp::Power, &["negative", "half"])),
        ]),
        graph([
            ("large", constant(f32::MAX as f64)),
            ("two", constant(2.0)),
            ("bad", math(MathOp::Multiply, &["large", "two"])),
        ]),
    ];
    for (index, fields) in cases.into_iter().enumerate() {
        let compiled = compile_field_wgsl(&fields, &[id("bad")]).unwrap();
        if index < 4 {
            assert!(fields.evaluate(&id("bad"), &FieldContext::default()).is_err());
        } else {
            assert!(fields.evaluate(&id("bad"), &FieldContext::default()).is_ok());
        }
        let actual = execute(&gpu, &compiled, &[FieldContext::default()]);
        assert_eq!(actual[0][0][3], 0.0);
    }

    let position = graph([("value", FieldNode::Builtin(BuiltinField::ObjectPosition))]);
    let position = compile_field_wgsl(&position, &[id("value")]).unwrap();
    let non_finite = FieldContext {
        object_position: [f64::NAN, 0.0, 0.0],
        ..FieldContext::default()
    };
    assert_eq!(execute(&gpu, &position, &[non_finite])[0][0][3], 0.0);

    let lattice = pattern_graph(1.0);
    let lattice = compile_field_wgsl(&lattice, &[id("Noise")]).unwrap();
    let outside = FieldContext {
        object_position: [2147483648.0, 0.0, 0.0],
        ..FieldContext::default()
    };
    assert_eq!(execute(&gpu, &lattice, &[outside])[0][0][3], 0.0);
    println!("math/non-finite/lattice failure cases: 7");
}
