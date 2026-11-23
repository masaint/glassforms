use gf_core::pbr::{
    dielectric_coat_evaluate, dielectric_coat_pdf, dielectric_coat_sample,
    diffuse_substrate_evaluate, diffuse_substrate_pdf, diffuse_substrate_sample,
    opaque_layered_coat_probability, opaque_layered_evaluate, opaque_layered_pdf,
    opaque_layered_sample, DielectricCoat, DiffuseSubstrate, OpaqueEvent, OpaqueFrame,
    OpaqueMediumTransition, OpaqueSample, OpaqueVector,
};
use glassforms::gpu::Gpu;

const TYPES: &str = include_str!("../src/shaders/pbr/types.wgsl");
const CONDUCTOR_FRESNEL: &str = include_str!("../src/shaders/conductor_fresnel.wgsl");
const DIELECTRIC: &str = include_str!("../src/shaders/pbr/dielectric.wgsl");
const GGX: &str = concat!(include_str!("../src/shaders/pbr/ggx.wgsl"), include_str!("../src/shaders/pbr/rough_conductor.wgsl"));
const DIFFUSE: &str = include_str!("../src/shaders/pbr/diffuse_substrate.wgsl");
const COAT: &str = include_str!("../src/shaders/pbr/dielectric_coat.wgsl");
const PAINT_ADAPTER: &str = include_str!("../src/shaders/pbr/opaque_paint.wgsl");

fn leaf_source() -> String {
    [TYPES, CONDUCTOR_FRESNEL, DIELECTRIC, GGX, DIFFUSE, COAT].join("\n")
}

fn validate(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn opaque_lobes_pass_naga_validation_without_measured_optics_replacement() {
    validate(&leaf_source());
    assert!(COAT.contains("pbr_dielectric_fresnel("));
    assert!(COAT.contains("pbr_ggx_evaluate("));
    assert!(COAT.contains("pbr_ggx_pdf("));
    assert!(!COAT.contains("fresMetal("));
    assert!(!COAT.contains("pow(1.0 -"));
}

fn unit(v: OpaqueVector) -> OpaqueVector {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.map(|x| x / norm)
}

fn direction(x: f64, y: f64, z: f64) -> OpaqueVector {
    unit([x, y, z])
}

fn close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{context}: {actual} / {expected}"
    );
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance * (1.0 + expected.abs()),
        "{context}: {actual} != {expected} (error {error})"
    );
}

fn close_representable(actual: f32, expected: f64, tolerance: f64, context: &str) {
    if expected.is_finite() && expected.abs() > f64::from(f32::MAX) {
        assert!(expected.is_sign_positive(), "{context}: unexpected negative density");
        assert_eq!(actual, f32::MAX, "{context}: incorrect saturated output");
    } else {
        close(f64::from(actual), expected, tolerance, context);
    }
}

fn close_rgb_representable(actual: &[f32], expected: [f64; 3], tolerance: f64, context: &str) {
    for channel in 0..3 {
        close_representable(
            actual[channel],
            expected[channel],
            tolerance,
            &format!("{context}, channel {channel}"),
        );
    }
}

fn close_rgb(actual: &[f32], expected: [f64; 3], tolerance: f64, context: &str) {
    for channel in 0..3 {
        close(
            f64::from(actual[channel]),
            expected[channel],
            tolerance,
            &format!("{context}, channel {channel}"),
        );
    }
}

fn check_flags(actual: &[f32], expected: &OpaqueSample, context: &str) {
    assert_eq!(
        actual[0] as u32,
        u32::from(expected.delta),
        "{context} delta"
    );
    close(
        f64::from(actual[1]),
        expected.event_probability,
        2e-6,
        &format!("{context} event probability"),
    );
    assert_eq!(actual[2] as u32, 0, "{context} stays in incident medium");
    assert_eq!(expected.medium_transition, OpaqueMediumTransition::Stay);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCase {
    wo_substrate_weight: [f32; 4],
    wi_coat_weight: [f32; 4],
    media: [f32; 4],
    rng: [f32; 4],
    albedo: [f32; 4],
}

fn gpu_case(
    substrate_weight: f64,
    coat_weight: f64,
    incident: f64,
    coat_ior: f64,
    roughness: f64,
    wo: OpaqueVector,
    wi: OpaqueVector,
    rng: [f64; 3],
) -> GpuCase {
    GpuCase {
        wo_substrate_weight: [
            wo[0] as f32,
            wo[1] as f32,
            wo[2] as f32,
            substrate_weight as f32,
        ],
        wi_coat_weight: [wi[0] as f32, wi[1] as f32, wi[2] as f32, coat_weight as f32],
        media: [incident as f32, coat_ior as f32, roughness as f32, 0.0],
        rng: [rng[0] as f32, rng[1] as f32, rng[2] as f32, 0.0],
        albedo: [0.65, 0.3, 0.12, 0.0],
    }
}

fn reference(
    case: &GpuCase,
) -> (
    DiffuseSubstrate,
    DielectricCoat,
    OpaqueVector,
    OpaqueVector,
    f64,
) {
    let substrate = DiffuseSubstrate {
        albedo: std::array::from_fn(|i| f64::from(case.albedo[i])),
        weight: f64::from(case.wo_substrate_weight[3]),
    };
    let coat = DielectricCoat {
        weight: f64::from(case.wi_coat_weight[3]),
        ior: f64::from(case.media[1]),
        roughness: f64::from(case.media[2]),
    };
    let wo = std::array::from_fn(|i| f64::from(case.wo_substrate_weight[i]));
    let wi = std::array::from_fn(|i| f64::from(case.wi_coat_weight[i]));
    (substrate, coat, wo, wi, f64::from(case.media[0]))
}

fn test_source(direction_count: usize, phi_steps: usize, z_steps: usize) -> String {
    format!(
        r#"
{leaf}
struct PbrTestCase {{
  wo_substrate_weight: vec4f,
  wi_coat_weight: vec4f,
  media: vec4f,
  rng: vec4f,
  albedo: vec4f,
}}
fn testHit(input: PbrTestCase) -> PbrSurfaceHit {{
  return PbrSurfaceHit(PbrFrame(vec3f(1.0,0.0,0.0),vec3f(0.0,1.0,0.0),vec3f(0.0,0.0,1.0)),
    PbrMedium(input.media[0],0.0,9u,31),PbrMedium(1.0,0.0,0u,0),550.0);
}}
fn testSubstrate(input: PbrTestCase) -> PbrDiffuseSubstrate {{
  return PbrDiffuseSubstrate(input.albedo.xyz,input.wo_substrate_weight[3]);
}}
fn testCoat(input: PbrTestCase) -> PbrDielectricCoat {{
  return PbrDielectricCoat(input.wi_coat_weight[3],input.media[1],input.media[2]);
}}
@group(0) @binding(0) var<storage,read> cases: array<PbrTestCase>;
@group(0) @binding(1) var<storage,read_write> results: array<vec4f>;
@compute @workgroup_size(64)
fn contract(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.x];
  let hit = testHit(input);
  let substrate = testSubstrate(input);
  let coat = testCoat(input);
  let wo = input.wo_substrate_weight.xyz;
  let wi = input.wi_coat_weight.xyz;
  let rng = PbrPathRng(input.rng.xy,input.rng.z);
  let f = pbr_opaque_layered_evaluate(hit,substrate,coat,wo,wi);
  let cf = pbr_dielectric_coat_evaluate(hit,coat,wo,wi);
  let df = pbr_diffuse_substrate_evaluate(hit,substrate,wo,wi);
  let s = pbr_opaque_layered_sample(hit,substrate,coat,wo,rng);
  let cs = pbr_dielectric_coat_sample(hit,coat,wo,rng);
  let ds = pbr_diffuse_substrate_sample(hit,substrate,wo,rng);
  results[12u*id.x] = vec4f(f,pbr_opaque_layered_pdf(hit,substrate,coat,wo,wi));
  results[12u*id.x+1u] = vec4f(cf,pbr_dielectric_coat_pdf(hit,coat,wo,wi));
  results[12u*id.x+2u] = vec4f(df,pbr_diffuse_substrate_pdf(hit,substrate,wo,wi));
  results[12u*id.x+3u] = vec4f(s.weight,s.pdf);
  results[12u*id.x+4u] = vec4f(s.wi,f32(s.event));
  results[12u*id.x+5u] = vec4f(f32(s.delta),s.event_probability,f32(s.medium_transition),
    pbr_opaque_layered_coat_probability(hit,substrate,coat,wo));
  results[12u*id.x+6u] = vec4f(cs.weight,cs.pdf);
  results[12u*id.x+7u] = vec4f(cs.wi,f32(cs.event));
  results[12u*id.x+8u] = vec4f(f32(cs.delta),cs.event_probability,f32(cs.medium_transition),0.0);
  results[12u*id.x+9u] = vec4f(ds.weight,ds.pdf);
  results[12u*id.x+10u] = vec4f(ds.wi,f32(ds.event));
  results[12u*id.x+11u] = vec4f(f32(ds.delta),ds.event_probability,f32(ds.medium_transition),0.0);
}}
const TEST_DIRECTION_COUNT: u32 = {direction_count}u;
const TEST_PHI_STEPS: u32 = {phi_steps}u;
const TEST_Z_STEPS: u32 = {z_steps}u;
@compute @workgroup_size(64)
fn quadrature(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= TEST_DIRECTION_COUNT || id.y >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.y];
  let hit = testHit(input);
  let substrate = testSubstrate(input);
  let coat = testCoat(input);
  let wo = input.wo_substrate_weight.xyz;
  // Midpoints in cos(theta) make dOmega = dz*dphi and integrate a cosine PDF exactly.
  let z = -1.0+2.0*(f32(id.x/TEST_PHI_STEPS)+0.5)/f32(TEST_Z_STEPS);
  let phi = PBR_TAU*(f32(id.x%TEST_PHI_STEPS)+0.5)/f32(TEST_PHI_STEPS);
  let radius = sqrt(max(0.0,1.0-z*z));
  let wi = vec3f(radius*cos(phi),radius*sin(phi),z);
  let f = pbr_opaque_layered_evaluate(hit,substrate,coat,wo,wi);
  let cf = pbr_dielectric_coat_evaluate(hit,coat,wo,wi);
  results[id.y*TEST_DIRECTION_COUNT+id.x] = vec4f(
    pbr_opaque_layered_pdf(hit,substrate,coat,wo,wi),dot(f,vec3f(1.0/3.0))*max(z,0.0),
    pbr_dielectric_coat_pdf(hit,coat,wo,wi),dot(cf,vec3f(1.0/3.0))*max(z,0.0));
}}
"#,
        leaf = leaf_source(),
    )
}

fn read_results(gpu: &Gpu, buffer: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("opaque lobes download"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    commands.copy_buffer_to_buffer(buffer, 0, &download, 0, bytes);
    gpu.queue.submit(Some(commands.finish()));
    let (send, recv) = std::sync::mpsc::channel();
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap();
        });
    gpu.wait();
    recv.recv().unwrap().unwrap();
    let mapped = download.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}

fn dispatch(gpu: &Gpu, pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup, x: u32, y: u32) {
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    gpu.queue.submit(Some(commands.finish()));
}

fn storage(gpu: &Gpu, label: &str, bytes: u64, copy_dst: bool) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | if copy_dst {
                wgpu::BufferUsages::COPY_DST
            } else {
                wgpu::BufferUsages::COPY_SRC
            },
        mapped_at_creation: false,
    })
}

fn group(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    })
}

fn pipeline(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    label: &str,
    entry: &str,
) -> wgpu::ComputePipeline {
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
}

#[test]
fn wgsl_matches_f64_and_pdf_plus_delta_normalises_with_bounded_energy() {
    let gpu = Gpu::headless("vulkan").expect("opaque lobe WGSL contract requires Vulkan");
    let views = [
        direction(0.0, 0.0, 1.0),
        direction(0.65, -0.3, 0.7),
        direction(2.0, 0.2, 0.22),
    ];
    let lights = [
        direction(-0.2, 0.4, 1.0),
        direction(0.8, -0.1, 0.3),
        direction(0.2, 0.3, -0.4),
        direction(-1.0, -0.5, 0.05),
        direction(0.1, 0.8, 0.25),
    ];
    let randoms = [
        [0.13, 0.77, 0.001],
        [0.81, 0.22, 0.45],
        [0.5, 0.5, 0.8],
        [0.997, 0.003, 0.99],
    ];
    let mut cases = Vec::new();
    for (r, roughness) in [0.0, 0.08, 0.35, 0.8].into_iter().enumerate() {
        for (c, coat_weight) in [0.0, 0.55, 1.0].into_iter().enumerate() {
            for (s, substrate_weight) in [0.0, 0.8].into_iter().enumerate() {
                for (v, &wo) in views.iter().enumerate() {
                    let index = r + 4 * (c + 3 * (s + 2 * v));
                    cases.push(gpu_case(
                        substrate_weight,
                        coat_weight,
                        if index % 5 == 0 { 1.33 } else { 1.0 },
                        if index % 7 == 0 { 1.62 } else { 1.46 },
                        roughness,
                        wo,
                        lights[index % lights.len()],
                        randoms[index % randoms.len()],
                    ));
                }
            }
        }
    }
    cases.push(gpu_case(
        0.0,
        1.0,
        1.0,
        1.5,
        1.0e-8,
        views[0],
        views[0],
        [0.2, 0.3, 0.8],
    ));
    cases.push(gpu_case(
        0.0,
        1.0,
        1.0,
        1.5,
        2.0e-19,
        views[0],
        views[0],
        [0.2, 0.3, 0.8],
    ));
    cases.push(gpu_case(
        0.0,
        1.0e-30,
        1.0,
        1.5,
        1.0e-10,
        views[0],
        views[0],
        [0.2, 0.3, 0.8],
    ));
    cases.push(gpu_case(
        0.8,
        1.0e-30,
        1.0,
        1.5,
        1.0e-10,
        views[0],
        views[0],
        [0.2, 0.3, 0.8],
    ));
    cases.push(gpu_case(
        0.8,
        0.7,
        0.0,
        1.5,
        0.35,
        views[0],
        lights[0],
        [0.2, 0.3, 0.8],
    ));
    cases.push(gpu_case(
        0.8,
        0.7,
        1.0,
        1.5,
        0.35,
        views[1].map(|x| -x),
        lights[0],
        [0.2, 0.3, 0.8],
    ));
    cases.push(gpu_case(
        0.8,
        0.7,
        1.0,
        1.5,
        0.35,
        views[0],
        lights[0],
        [0.2, 0.3, 1.0],
    ));
    for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let mut albedo = gpu_case(
            0.8,
            0.7,
            1.0,
            1.5,
            0.35,
            views[1],
            lights[0],
            [0.2, 0.3, 0.8],
        );
        albedo.albedo[0] = bad;
        cases.push(albedo);
        cases.push(gpu_case(
            0.8,
            0.7,
            f64::from(bad),
            1.5,
            0.35,
            views[1],
            lights[0],
            [0.2, 0.3, 0.8],
        ));
        cases.push(gpu_case(
            0.8,
            0.7,
            1.0,
            f64::from(bad),
            0.35,
            views[1],
            lights[0],
            [0.2, 0.3, 0.8],
        ));
        let mut bad_wo = views[1];
        bad_wo[0] = f64::from(bad);
        cases.push(gpu_case(
            0.8,
            0.7,
            1.0,
            1.5,
            0.35,
            bad_wo,
            lights[0],
            [0.2, 0.3, 0.8],
        ));
        let mut bad_wi = lights[0];
        bad_wi[1] = f64::from(bad);
        cases.push(gpu_case(
            0.8,
            0.7,
            1.0,
            1.5,
            0.35,
            views[1],
            bad_wi,
            [0.2, 0.3, 0.8],
        ));
        cases.push(gpu_case(
            0.8,
            0.7,
            1.0,
            1.5,
            0.35,
            views[1],
            lights[0],
            [f64::from(bad), 0.3, 0.8],
        ));
        cases.push(gpu_case(
            0.8,
            0.7,
            1.0,
            1.5,
            0.35,
            views[1],
            lights[0],
            [0.2, 0.3, f64::from(bad)],
        ));
    }
    let mut energy_amplifying = gpu_case(
        0.8,
        0.7,
        1.0,
        1.5,
        0.35,
        views[1],
        lights[0],
        [0.2, 0.3, 0.8],
    );
    energy_amplifying.albedo[1] = 1.000_000_1;
    cases.push(energy_amplifying);

    let quadrature_cases = [
        gpu_case(
            1.0,
            0.0,
            1.0,
            1.5,
            0.0,
            views[1],
            lights[0],
            [0.13, 0.77, 0.0],
        ),
        gpu_case(
            0.0,
            1.0,
            1.0,
            1.5,
            0.0,
            views[2],
            lights[0],
            [0.13, 0.77, 0.0],
        ),
        gpu_case(
            0.0,
            1.0,
            1.0,
            1.5,
            0.45,
            views[0],
            lights[0],
            [0.13, 0.77, 0.0],
        ),
        gpu_case(
            1.0,
            0.8,
            1.0,
            1.46,
            0.0,
            views[1],
            lights[0],
            [0.13, 0.77, 0.0],
        ),
        gpu_case(
            1.0,
            0.8,
            1.33,
            1.46,
            0.45,
            views[1],
            lights[0],
            [0.13, 0.77, 0.0],
        ),
        gpu_case(
            1.0,
            1.0,
            1.0,
            1.62,
            0.8,
            views[2],
            lights[0],
            [0.13, 0.77, 0.0],
        ),
    ];
    let (phi_steps, z_steps) = (128_usize, 256_usize);
    let direction_count = phi_steps * z_steps;
    let source = test_source(direction_count, phi_steps, z_steps);
    validate(&source);
    let shader = gpu.shader(
        "pure opaque diffuse and dielectric coat contract",
        &source,
        false,
    );
    let contract_pipeline = pipeline(
        &gpu,
        &shader,
        "opaque evaluate/pdf/sample contract",
        "contract",
    );
    let quadrature_pipeline =
        pipeline(&gpu, &shader, "opaque full-sphere quadrature", "quadrature");

    let case_buffer = storage(
        &gpu,
        "opaque contract cases",
        (cases.len() * std::mem::size_of::<GpuCase>()) as u64,
        true,
    );
    gpu.queue
        .write_buffer(&case_buffer, 0, bytemuck::cast_slice(&cases));
    let contract_bytes = (cases.len() * 12 * 16) as u64;
    let contract_buffer = storage(&gpu, "opaque contract results", contract_bytes, false);
    let contract_group = group(&gpu, &contract_pipeline, &case_buffer, &contract_buffer);
    dispatch(
        &gpu,
        &contract_pipeline,
        &contract_group,
        cases.len().div_ceil(64) as u32,
        1,
    );
    let contract_results = read_results(&gpu, &contract_buffer, contract_bytes);

    let frame = OpaqueFrame::default();
    let mut event_counts = [0_u32; 5];
    let mut delta_cases = 0;
    for (i, case) in cases.iter().enumerate() {
        let (substrate, coat, wo, wi, incident) = reference(case);
        let random = std::array::from_fn(|j| f64::from(case.rng[j]));
        let expected_f = opaque_layered_evaluate(&frame, incident, &substrate, &coat, wo, wi);
        let expected_p = opaque_layered_pdf(&frame, incident, &substrate, &coat, wo, wi);
        let expected_cf = dielectric_coat_evaluate(&frame, incident, &coat, wo, wi);
        let expected_cp = dielectric_coat_pdf(&frame, incident, &coat, wo, wi);
        let expected_df = diffuse_substrate_evaluate(&frame, &substrate, wo, wi);
        let expected_dp = diffuse_substrate_pdf(&frame, &substrate, wo, wi);
        let expected = opaque_layered_sample(&frame, incident, &substrate, &coat, wo, random);
        let expected_cs =
            dielectric_coat_sample(&frame, incident, &coat, wo, [random[0], random[1]]);
        let expected_ds = diffuse_substrate_sample(&frame, &substrate, wo, [random[0], random[1]]);
        let output = &contract_results[48 * i..48 * (i + 1)];
        close_rgb_representable(
            &output[0..3],
            expected_f,
            5e-4,
            &format!("layered evaluate case {i}"),
        );
        close_representable(output[3], expected_p, 4e-5, "layered PDF");
        close_rgb_representable(&output[4..7], expected_cf, 5e-4, "coat evaluate");
        close_representable(output[7], expected_cp, 4e-5, "coat PDF");
        close_rgb(&output[8..11], expected_df, 2e-6, "diffuse evaluate");
        close(f64::from(output[11]), expected_dp, 2e-6, "diffuse PDF");
        close_rgb(
            &output[12..15],
            expected.weight,
            8e-4,
            &format!("layered sample weight case {i}"),
        );
        close_representable(output[15], expected.pdf, 4e-5, "layered sample PDF");
        close_rgb(&output[16..19], expected.wi, 4e-5, "layered sample wi");
        assert_eq!(
            output[19] as u32, expected.event as u32,
            "layered event case {i}"
        );
        check_flags(&output[20..23], &expected, "layered");
        let expected_probability =
            opaque_layered_coat_probability(&frame, incident, &substrate, &coat, wo);
        if expected.event != OpaqueEvent::Invalid && expected_probability.is_finite() {
            close(
                f64::from(output[23]),
                expected_probability,
                2e-6,
                "coat selection mass",
            );
        }
        close_rgb(
            &output[24..27],
            expected_cs.weight,
            8e-4,
            "pure coat sample weight",
        );
        close_representable(output[27], expected_cs.pdf, 4e-5, "pure coat sample PDF");
        close_rgb(&output[28..31], expected_cs.wi, 4e-5, "pure coat sample wi");
        assert_eq!(
            output[31] as u32, expected_cs.event as u32,
            "pure coat event case {i}"
        );
        check_flags(&output[32..35], &expected_cs, "pure coat");
        close_rgb(
            &output[36..39],
            expected_ds.weight,
            3e-5,
            "pure diffuse sample weight",
        );
        close(
            f64::from(output[39]),
            expected_ds.pdf,
            2e-6,
            "pure diffuse sample PDF",
        );
        close_rgb(
            &output[40..43],
            expected_ds.wi,
            4e-5,
            "pure diffuse sample wi",
        );
        assert_eq!(
            output[43] as u32, expected_ds.event as u32,
            "pure diffuse event case {i}"
        );
        check_flags(&output[44..47], &expected_ds, "pure diffuse");
        event_counts[expected.event as usize] += 1;
        delta_cases += usize::from(expected.delta);
        if expected.event == OpaqueEvent::Reflection && !expected.delta {
            let reconstructed =
                opaque_layered_evaluate(&frame, incident, &substrate, &coat, wo, expected.wi);
            for channel in 0..3 {
                close(
                    expected.weight[channel],
                    reconstructed[channel] * expected.wi[2].abs() / expected.pdf,
                    1e-12,
                    "continuous weight identity",
                );
            }
        }
        if expected.delta {
            assert_eq!(expected.pdf, 0.0);
        }
    }
    assert!(delta_cases > 0, "smooth coat delta coverage");
    assert!(
        [
            OpaqueEvent::Invalid,
            OpaqueEvent::Reflection,
            OpaqueEvent::BelowHemisphere
        ]
        .iter()
        .all(|event| event_counts[*event as usize] > 0),
        "layered event coverage: {event_counts:?}"
    );

    let quadrature_case_buffer = storage(
        &gpu,
        "opaque quadrature cases",
        (quadrature_cases.len() * std::mem::size_of::<GpuCase>()) as u64,
        true,
    );
    gpu.queue.write_buffer(
        &quadrature_case_buffer,
        0,
        bytemuck::cast_slice(&quadrature_cases),
    );
    let quadrature_bytes = (quadrature_cases.len() * direction_count * 16) as u64;
    let quadrature_buffer = storage(&gpu, "opaque quadrature results", quadrature_bytes, false);
    let quadrature_group = group(
        &gpu,
        &quadrature_pipeline,
        &quadrature_case_buffer,
        &quadrature_buffer,
    );
    dispatch(
        &gpu,
        &quadrature_pipeline,
        &quadrature_group,
        direction_count.div_ceil(64) as u32,
        quadrature_cases.len() as u32,
    );
    let quadrature_results = read_results(&gpu, &quadrature_buffer, quadrature_bytes);

    for (i, case) in quadrature_cases.iter().enumerate() {
        let (substrate, coat, wo, _, incident) = reference(case);
        let mut shader_pdf = 0.0;
        let mut reference_pdf = 0.0;
        let mut shader_energy = 0.0;
        let mut reference_energy = 0.0;
        let mut shader_coat_pdf = 0.0;
        let mut shader_coat_energy = 0.0;
        for z in 0..z_steps {
            let cosine = -1.0 + 2.0 * (z as f64 + 0.5) / z_steps as f64;
            let radius = (1.0 - cosine * cosine).max(0.0).sqrt();
            let solid_angle = 2.0 * std::f64::consts::TAU / (z_steps * phi_steps) as f64;
            for phi in 0..phi_steps {
                let azimuth = std::f64::consts::TAU * (phi as f64 + 0.5) / phi_steps as f64;
                let wi = [radius * azimuth.cos(), radius * azimuth.sin(), cosine];
                let index = i * direction_count + z * phi_steps + phi;
                let output = &quadrature_results[4 * index..4 * index + 4];
                let expected_p = opaque_layered_pdf(&frame, incident, &substrate, &coat, wo, wi);
                let expected_f =
                    opaque_layered_evaluate(&frame, incident, &substrate, &coat, wo, wi);
                let expected_cp = dielectric_coat_pdf(&frame, incident, &coat, wo, wi);
                let expected_cf = dielectric_coat_evaluate(&frame, incident, &coat, wo, wi);
                close(
                    f64::from(output[0]),
                    expected_p,
                    2e-4,
                    "quadrature layered PDF",
                );
                close(
                    f64::from(output[2]),
                    expected_cp,
                    2e-4,
                    "quadrature coat PDF",
                );
                shader_pdf += f64::from(output[0]) * solid_angle;
                reference_pdf += expected_p * solid_angle;
                shader_energy += f64::from(output[1]) * solid_angle;
                reference_energy +=
                    expected_f.iter().sum::<f64>() / 3.0 * cosine.max(0.0) * solid_angle;
                shader_coat_pdf += f64::from(output[2]) * solid_angle;
                shader_coat_energy += f64::from(output[3]) * solid_angle;
                assert!(expected_cf.iter().all(|x| x.is_finite() && *x >= 0.0));
            }
        }
        let sample =
            opaque_layered_sample(&frame, incident, &substrate, &coat, wo, [0.13, 0.77, 0.0]);
        let coat_sample = dielectric_coat_sample(&frame, incident, &coat, wo, [0.13, 0.77]);
        let delta_pdf = if sample.delta {
            sample.event_probability
        } else {
            0.0
        };
        let delta_energy = if sample.delta {
            sample.weight.iter().sum::<f64>() / 3.0 * sample.event_probability
        } else {
            0.0
        };
        let coat_delta_pdf = if coat_sample.delta { 1.0 } else { 0.0 };
        let coat_delta_energy = if coat_sample.delta {
            coat_sample.weight.iter().sum::<f64>() / 3.0
        } else {
            0.0
        };
        println!(
            "opaque quadrature case {i}: layered PDF {:.8} + delta {:.8}, energy {:.8}; coat PDF {:.8} + delta {:.8}, energy {:.8}",
            shader_pdf, delta_pdf, shader_energy + delta_energy,
            shader_coat_pdf, coat_delta_pdf, shader_coat_energy + coat_delta_energy,
        );
        close(
            shader_pdf + delta_pdf,
            1.0,
            0.012,
            "WGSL layered PDF plus delta",
        );
        close(
            reference_pdf + delta_pdf,
            1.0,
            0.012,
            "f64 layered PDF plus delta",
        );
        close(
            shader_coat_pdf + coat_delta_pdf,
            1.0,
            0.012,
            "WGSL coat PDF plus delta",
        );
        assert!(
            reference_energy + delta_energy >= 0.0 && reference_energy + delta_energy <= 1.0 + 2e-9
        );
        assert!(shader_energy + delta_energy >= 0.0 && shader_energy + delta_energy <= 1.003);
        assert!(
            shader_coat_energy + coat_delta_energy >= 0.0
                && shader_coat_energy + coat_delta_energy <= 1.003
        );
    }
}

#[test]
fn paint_adapter_preserves_the_legacy_irradiance_over_pi_convention() {
    let gpu = Gpu::headless("vulkan").expect("opaque paint adapter contract requires Vulkan");
    let source = format!(
        r#"
{types}
{diffuse}
{adapter}
struct PbrTestCase {{
  wo_substrate_weight: vec4f,
  wi_coat_weight: vec4f,
  media: vec4f,
  rng: vec4f,
  albedo: vec4f,
}}
@group(0) @binding(0) var<storage,read> cases: array<PbrTestCase>;
@group(0) @binding(1) var<storage,read_write> results: array<vec4f>;
@compute @workgroup_size(64)
fn contract(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.x];
  let hit = PbrSurfaceHit(PbrFrame(vec3f(1.0,0.0,0.0),vec3f(0.0,1.0,0.0),vec3f(0.0,0.0,1.0)),
    PbrMedium(input.media[0],0.0,9u,31),PbrMedium(1.5,0.0,0u,0),550.0);
  let substrate = PbrDiffuseSubstrate(input.albedo.xyz,input.wo_substrate_weight[3]);
  let wo = input.wo_substrate_weight.xyz;
  let gather = input.rng.xyz;
  let response = pbr_opaque_paint_response(hit,substrate,wo,gather);
  let leaf = pbr_diffuse_substrate_evaluate(hit,substrate,wo,hit.frame.normal);
  results[id.x] = vec4f(response,dot(leaf,vec3f(1.0/3.0))*PBR_PI);
}}
"#,
        types = TYPES,
        diffuse = DIFFUSE,
        adapter = PAINT_ADAPTER,
    );
    validate(&source);
    let shader = gpu.shader("assembly paint adapter contract", &source, false);
    let pipeline = pipeline(&gpu, &shader, "assembly paint adapter", "contract");
    let views = [
        direction(0.0, 0.0, 1.0),
        direction(0.65, -0.3, 0.7),
        direction(2.0, 0.2, 0.22),
        [0.4, -0.2, 0.0],
        direction(0.3, 0.1, -0.8),
    ];
    let weights = [0.73, 0.41, 0.96, 0.58, 0.62, 1.2];
    let gathers = [
        [0.4, 0.7, 1.1],
        [1.3, 0.2, 0.8],
        [0.05, 0.9, 0.3],
        [0.6, 0.45, 0.25],
        [0.8, 0.35, 1.2],
        [0.5, 0.6, 0.7],
    ];
    let cases: Vec<GpuCase> = (0..weights.len())
        .map(|i| {
            let mut case = gpu_case(
                weights[i],
                0.0,
                1.0,
                1.5,
                0.0,
                views[i % views.len()],
                direction(0.0, 0.0, 1.0),
                gathers[i],
            );
            case.albedo = [0.65, 0.3, 0.12, 0.0];
            case
        })
        .collect();
    let bytes = (cases.len() * std::mem::size_of::<GpuCase>()) as u64;
    let input = storage(&gpu, "paint adapter cases", bytes, true);
    gpu.queue.write_buffer(&input, 0, bytemuck::cast_slice(&cases));
    let output_bytes = (cases.len() * 16) as u64;
    let output = storage(&gpu, "paint adapter results", output_bytes, false);
    let group = group(&gpu, &pipeline, &input, &output);
    dispatch(&gpu, &pipeline, &group, cases.len().div_ceil(64) as u32, 1);
    let results = read_results(&gpu, &output, output_bytes);
    for (i, case) in cases.iter().enumerate() {
        let result = &results[4 * i..4 * i + 4];
        for channel in 0..3 {
            let expected = case.albedo[channel] * case.wo_substrate_weight[3] * case.rng[channel];
            close(
                f64::from(result[channel]),
                f64::from(expected),
                2e-6,
                &format!("paint adapter legacy response case {i}, channel {channel}"),
            );
        }
        if case.wo_substrate_weight[2] > 0.0 && case.wo_substrate_weight[3] <= 1.0 {
            // The fourth lane is the mean pure-leaf evaluate times pi, not another response.
            // Comparing it to unit gather independently catches a missing or duplicated pi.
            let expected = case.albedo.iter().sum::<f32>() / 3.0 * case.wo_substrate_weight[3];
            close(f64::from(result[3]), f64::from(expected), 2e-6, "pure diffuse leaf times pi");
        }
    }
}
