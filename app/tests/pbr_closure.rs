use gf_core::material::{fresnel_metal, spec_w, METALS};
use glassforms::gpu::Gpu;

const TYPES: &str = include_str!("../src/shaders/pbr/types.wgsl");
const LEAF: &str = concat!(include_str!("../src/shaders/pbr/ggx.wgsl"), include_str!("../src/shaders/pbr/rough_conductor.wgsl"));
const FRESNEL: &str = include_str!("../src/shaders/conductor_fresnel.wgsl");
const EVENT_INVALID: u32 = 0;
const EVENT_REFLECTION: u32 = 1;
const EVENT_BELOW_HEMISPHERE: u32 = 4;
const EVENT_UNSUPPORTED_DELTA: u32 = 5;

fn leaf_source() -> String {
    [TYPES, FRESNEL, LEAF].join("\n")
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
fn rough_conductor_leaf_passes_naga_validation() {
    validate(&leaf_source());
}

// Independent f64 contract arithmetic. The Fresnel reference uses complex admittances rather than
// transcribing fresMetalPolar; the production metalRGB and frGlossyFrame equations are controls.
type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] + b[i])
}
fn scale(a: V3, s: f64) -> V3 {
    a.map(|v| v * s)
}
fn dot(a: V3, b: V3) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(v: V3) -> V3 {
    scale(v, dot(v, v).sqrt().recip())
}
fn direction(x: f64, y: f64, z: f64) -> V3 {
    unit([x, y, z])
}

#[derive(Clone)]
struct Frame {
    tangent: V3,
    bitangent: V3,
    normal: V3,
}

impl Frame {
    fn new(tangent: V3) -> Self {
        let normal = [0.0, 0.0, 1.0];
        let tangent = unit(tangent);
        Self {
            tangent,
            bitangent: cross(normal, tangent),
            normal,
        }
    }

    fn local(&self, v: V3) -> V3 {
        [
            dot(v, self.tangent),
            dot(v, self.bitangent),
            dot(v, self.normal),
        ]
    }

    fn world(&self, v: V3) -> V3 {
        add(
            add(scale(self.tangent, v[0]), scale(self.bitangent, v[1])),
            scale(self.normal, v[2]),
        )
    }
}

#[derive(Clone)]
struct Material {
    alpha: [f64; 2],
    nk: [[f64; 2]; 16],
    spectral: [V3; 16],
}

fn alpha(roughness: f64, anisotropy: f64) -> [f64; 2] {
    let aspect = (1.0 - 0.9 * anisotropy).sqrt();
    [
        roughness * roughness / aspect,
        roughness * roughness * aspect,
    ]
}

fn copper(roughness: f64, anisotropy: f64) -> Material {
    let copper = METALS.iter().find(|m| m.id == "copper").unwrap();
    Material {
        alpha: alpha(roughness, anisotropy),
        nk: std::array::from_fn(|lane| {
            [f64::from(copper.n[2 * lane]), f64::from(copper.k[2 * lane])]
        }),
        spectral: std::array::from_fn(|lane| spec_w(400.0 + 20.0 * lane as f64)),
    }
}

fn interface(incident: f64, real: f64, imaginary: f64) -> f64 {
    ((incident - real).powi(2) + imaginary.powi(2))
        / ((incident + real).powi(2) + imaginary.powi(2))
}

fn conductor_power(cosine: f64, incident_ior: f64, n: f64, k: f64) -> f64 {
    let c = cosine.clamp(0.0, 1.0);
    let sin2 = 1.0 - c * c;
    let (n, k) = (n / incident_ior, k / incident_ior);
    let (a, b) = (n * n - k * k - sin2, 2.0 * n * k);
    let norm = a.hypot(b);
    let qr = (0.5 * (norm + a)).sqrt();
    let qi = (0.5 * (norm - a)).sqrt();
    let pr = ((n * n - k * k) * qr + b * qi) / norm;
    let pi = (b * qr - (n * n - k * k) * qi) / norm;
    0.5 * (interface(c, qr, qi) + interface(c.recip(), pr, pi))
}

fn measured_rgb(cosine: f64, incident_ior: f64, material: &Material) -> V3 {
    let mut sum = [0.0; 3];
    let mut sensor = [0.0; 3];
    for lane in 0..16 {
        let [n, k] = material.nk[lane];
        let power = conductor_power(cosine, incident_ior, n, k);
        // Also guard the independent admittance calculation against the existing public helper.
        let existing = fresnel_metal(cosine, incident_ior, n, k);
        assert!((power - existing).abs() < 2e-12, "Fresnel controls differ");
        for channel in 0..3 {
            sum[channel] += material.spectral[lane][channel] * power;
            sensor[channel] += material.spectral[lane][channel];
        }
    }
    std::array::from_fn(|channel| sum[channel] / sensor[channel])
}

fn half_vector(wo: V3, wi: V3) -> Option<V3> {
    let h = add(wo, wi);
    (dot(h, h) > 1e-24).then(|| unit(h))
}

fn lambda(v: V3, alpha: [f64; 2]) -> f64 {
    let projected = (alpha[0] * v[0]).powi(2) + (alpha[1] * v[1]).powi(2);
    0.5 * ((1.0 + projected / v[2].powi(2)).sqrt() - 1.0)
}

fn density(h: V3, alpha: [f64; 2]) -> f64 {
    let squared = (h[0] / alpha[0]).powi(2) + (h[1] / alpha[1]).powi(2) + h[2].powi(2);
    1.0 / (std::f64::consts::PI * alpha[0] * alpha[1] * squared * squared)
}

fn evaluate(frame: &Frame, incident_ior: f64, material: &Material, wo: V3, wi: V3) -> V3 {
    if material.alpha.iter().any(|&a| a <= 0.0) || incident_ior <= 0.0 {
        return [0.0; 3];
    }
    let (v, l) = (frame.local(wo), frame.local(wi));
    if v[2] <= 0.0 || l[2] <= 0.0 {
        return [0.0; 3];
    }
    let Some(h_world) = half_vector(wo, wi) else {
        return [0.0; 3];
    };
    let h = frame.local(h_world);
    let vh = dot(v, h);
    if h[2] <= 0.0 || vh <= 0.0 {
        return [0.0; 3];
    }
    let g2 = 1.0 / (1.0 + lambda(v, material.alpha) + lambda(l, material.alpha));
    let geometry = density(h, material.alpha) * g2 / (4.0 * v[2] * l[2]);
    scale(measured_rgb(vh, incident_ior, material), geometry)
}

fn pdf(frame: &Frame, incident_ior: f64, material: &Material, wo: V3, wi: V3) -> f64 {
    if material.alpha.iter().any(|&a| a <= 0.0) || incident_ior <= 0.0 {
        return 0.0;
    }
    let v = frame.local(wo);
    if v[2] <= 0.0 {
        return 0.0;
    }
    let Some(h_world) = half_vector(wo, wi) else {
        return 0.0;
    };
    let h = frame.local(h_world);
    if h[2] <= 0.0 || dot(v, h) <= 0.0 {
        return 0.0;
    }
    let g1 = 1.0 / (1.0 + lambda(v, material.alpha));
    density(h, material.alpha) * g1 / (4.0 * v[2])
}

struct Sample {
    wi: V3,
    weight: V3,
    pdf: f64,
    event: u32,
}

fn sample(
    frame: &Frame,
    incident_ior: f64,
    material: &Material,
    wo: V3,
    random: [f64; 2],
) -> Sample {
    let invalid = |event| Sample {
        wi: [0.0; 3],
        weight: [0.0; 3],
        pdf: 0.0,
        event,
    };
    if material.alpha.iter().any(|&a| a <= 0.0) {
        return invalid(EVENT_UNSUPPORTED_DELTA);
    }
    if incident_ior <= 0.0
        || random.iter().any(|&u| !(0.0..1.0).contains(&u))
        || frame.local(wo)[2] <= 0.0
    {
        return invalid(EVENT_INVALID);
    }

    // This is the standard stretched-hemisphere VNDF construction, written independently in f64.
    let v = unit([
        material.alpha[0] * frame.local(wo)[0],
        material.alpha[1] * frame.local(wo)[1],
        frame.local(wo)[2],
    ]);
    let lensq = v[0] * v[0] + v[1] * v[1];
    let t1 = if lensq > 0.0 {
        scale([-v[1], v[0], 0.0], lensq.sqrt().recip())
    } else {
        [1.0, 0.0, 0.0]
    };
    let t2 = cross(v, t1);
    let r = random[0].sqrt();
    let phi = std::f64::consts::TAU * random[1];
    let p1 = r * phi.cos();
    let hemisphere = 0.5 * (1.0 + v[2]);
    let p2 = (1.0 - hemisphere) * (1.0 - p1 * p1).max(0.0).sqrt() + hemisphere * r * phi.sin();
    let nh = add(
        add(scale(t1, p1), scale(t2, p2)),
        scale(v, (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt()),
    );
    let h = frame.world(unit([
        material.alpha[0] * nh[0],
        material.alpha[1] * nh[1],
        nh[2].max(0.0),
    ]));
    let wi = add(scale(wo, -1.0), scale(h, 2.0 * dot(wo, h)));
    let probability = pdf(frame, incident_ior, material, wo, wi);
    if probability <= 0.0 {
        return invalid(EVENT_INVALID);
    }
    if frame.local(wi)[2] <= 0.0 {
        return Sample {
            wi,
            weight: [0.0; 3],
            pdf: probability,
            event: EVENT_BELOW_HEMISPHERE,
        };
    }
    let f = evaluate(frame, incident_ior, material, wo, wi);
    Sample {
        wi,
        weight: scale(f, frame.local(wi)[2].abs() / probability),
        pdf: probability,
        event: EVENT_REFLECTION,
    }
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

fn legacy_glossy(material: &Material, wo: V3, wi: V3) -> (f64, f64) {
    // The non-folded, no-light-proposal branch of frGlossyFrame in frost.wgsl.
    let (nv, nl) = (wo[2], wi[2]);
    let h = unit(add(wo, wi));
    let a2 = material.alpha[0] * material.alpha[0];
    let nh = h[2];
    let d = a2 / (std::f64::consts::PI * ((1.0 - nh * nh) + a2 * nh * nh).powi(2));
    let gv = (a2 + (1.0 - a2) * nv * nv).sqrt();
    let gl = (a2 + (1.0 - a2) * nl * nl).sqrt();
    let proposal = d / (2.0 * (nv + gv));
    let cosine_response_per_reflectance = d * nl / (2.0 * (nl * gv + nv * gl));
    (proposal, cosine_response_per_reflectance)
}

#[test]
fn independent_reference_rejects_invalid_hemispheres_and_matches_the_glossy_leaf() {
    let frame = Frame::new([1.0, 0.0, 0.0]);
    for (roughness, anisotropy) in [(0.05, 0.0), (0.3, 0.0), (0.75, 0.0)] {
        let material = copper(roughness, anisotropy);
        for wo in [
            direction(0.0, 0.0, 1.0),
            direction(0.7, -0.2, 0.4),
            direction(2.0, 0.0, 0.25),
        ] {
            for wi in [
                direction(0.1, 0.2, 1.0),
                direction(-0.8, 0.4, 0.3),
                direction(1.0, 0.0, 0.15),
            ] {
                let f = evaluate(&frame, 1.0, &material, wo, wi);
                let p = pdf(&frame, 1.0, &material, wo, wi);
                let (old_pdf, old_response) = legacy_glossy(&material, wo, wi);
                close(p, old_pdf, 2e-12, "frGlossyFrame PDF");
                close(
                    f[0] * wi[2] / measured_rgb(dot(wo, unit(add(wo, wi))), 1.0, &material)[0],
                    old_response,
                    2e-12,
                    "frGlossyFrame cosine response",
                );
            }
        }
    }

    let material = copper(0.4, 0.65);
    let wo = direction(0.4, 0.1, 0.8);
    for wi in [direction(0.2, 0.4, -0.3), direction(-0.2, 0.1, -1.0)] {
        assert_eq!(evaluate(&frame, 1.0, &material, wo, wi), [0.0; 3]);
    }
    let below_wo = scale(wo, -1.0);
    assert_eq!(evaluate(&frame, 1.0, &material, below_wo, wo), [0.0; 3]);
    assert_eq!(pdf(&frame, 1.0, &material, below_wo, wo), 0.0);

    let delta = copper(0.0, 0.0);
    let smooth = sample(&frame, 1.0, &delta, wo, [0.2, 0.7]);
    assert_eq!(smooth.event, EVENT_UNSUPPORTED_DELTA);
    assert_eq!(smooth.pdf, 0.0);
    assert_eq!(pdf(&frame, 1.0, &delta, wo, wo), 0.0);
    assert_eq!(evaluate(&frame, 1.0, &delta, wo, wo), [0.0; 3]);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCase {
    tangent_ior: [f32; 4],
    wo_roughness: [f32; 4],
    wi_anisotropy: [f32; 4],
    rng: [f32; 4],
}

fn gpu_case(
    roughness: f64,
    anisotropy: f64,
    incident_ior: f64,
    tangent: V3,
    wo: V3,
    wi: V3,
    rng: [f64; 2],
) -> GpuCase {
    GpuCase {
        tangent_ior: [
            tangent[0] as f32,
            tangent[1] as f32,
            tangent[2] as f32,
            incident_ior as f32,
        ],
        wo_roughness: [wo[0] as f32, wo[1] as f32, wo[2] as f32, roughness as f32],
        wi_anisotropy: [wi[0] as f32, wi[1] as f32, wi[2] as f32, anisotropy as f32],
        rng: [rng[0] as f32, rng[1] as f32, 0.0, 0.0],
    }
}

fn test_source(direction_count: usize, phi_steps: usize, z_steps: usize) -> String {
    let material = copper(1.0, 0.0);
    let nk = material
        .nk
        .iter()
        .map(|nk| format!("vec2f({:.9e},{:.9e})", nk[0], nk[1]))
        .collect::<Vec<_>>()
        .join(",");
    let spectral = material
        .spectral
        .iter()
        .map(|w| format!("vec3f({:.9e},{:.9e},{:.9e})", w[0], w[1], w[2]))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"
{leaf}
fn testMaterial(roughness: f32, anisotropy: f32) -> PbrConductor {{
  let aspect = sqrt(1.0 - 0.9 * anisotropy);
  return PbrConductor(vec2f(roughness*roughness/aspect,roughness*roughness*aspect),
    array<vec2f,16>({nk}), array<vec3f,16>({spectral}));
}}
fn testHit(input: PbrTestCase) -> PbrSurfaceHit {{
  let t = normalize(input.tangent_ior.xyz);
  let n = vec3f(0.0,0.0,1.0);
  return PbrSurfaceHit(PbrFrame(t,cross(n,t),n),
    PbrMedium(input.tangent_ior[3],0.0,17u,-23),PbrMedium(1.0,0.0,0u,0),550.0);
}}
struct PbrTestCase {{
  tangent_ior: vec4f,
  wo_roughness: vec4f,
  wi_anisotropy: vec4f,
  rng: vec4f,
}}
@group(0) @binding(0) var<storage,read> cases: array<PbrTestCase>;
@group(0) @binding(1) var<storage,read_write> results: array<vec4f>;
@compute @workgroup_size(64)
fn contract(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.x];
  let hit = testHit(input);
  let material = testMaterial(input.wo_roughness[3],input.wi_anisotropy[3]);
  let wo = input.wo_roughness.xyz;
  let wi = input.wi_anisotropy.xyz;
  let f = pbr_conductor_evaluate(hit,material,wo,wi);
  let p = pbr_conductor_pdf(hit,material,wo,wi);
  let s = pbr_conductor_sample(hit,material,wo,PbrPathRng(input.rng.xy,0.5));
  results[4u*id.x] = vec4f(f,p);
  results[4u*id.x+1u] = vec4f(s.weight,s.pdf);
  results[4u*id.x+2u] = vec4f(s.wi,f32(s.event));
  results[4u*id.x+3u] = vec4f(f32(s.delta),s.event_probability,f32(s.medium_transition),
    pbrConductorD(vec3f(1.0e-10,0.0,1.0),vec2f(1.0e-20)));
}}
const TEST_DIRECTION_COUNT: u32 = {direction_count}u;
const TEST_PHI_STEPS: u32 = {phi_steps}u;
const TEST_Z_STEPS: u32 = {z_steps}u;
@compute @workgroup_size(64)
fn quadrature(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= TEST_DIRECTION_COUNT || id.y >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.y];
  let hit = testHit(input);
  let material = testMaterial(input.wo_roughness[3],input.wi_anisotropy[3]);
  let theta = PBR_PI*(f32(id.x/TEST_PHI_STEPS)+0.5)/f32(TEST_Z_STEPS);
  let phi = PBR_TAU*(f32(id.x%TEST_PHI_STEPS)+0.5)/f32(TEST_PHI_STEPS);
  let z = cos(theta);
  let radius = sin(theta);
  let wi = vec3f(radius*cos(phi),radius*sin(phi),z);
  let p = pbr_conductor_pdf(hit,material,input.wo_roughness.xyz,wi);
  results[id.y*TEST_DIRECTION_COUNT+id.x] = vec4f(p,0.0,0.0,0.0);
}}
"#,
        leaf = leaf_source(),
    )
}

fn read_results(gpu: &Gpu, buffer: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR contract download"),
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

#[test]
fn wgsl_contract_matches_f64_and_pdf_normalizes_over_proposals() {
    let gpu = Gpu::headless("vulkan").expect("PBR WGSL contract requires Vulkan");
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
    ];
    let randoms = [[0.13, 0.77], [0.81, 0.22], [0.5, 0.5], [0.997, 0.003]];
    let mut cases = Vec::new();
    for (r, roughness) in [0.05, 0.2, 0.55, 0.9].into_iter().enumerate() {
        for (a, anisotropy) in [0.0, 0.4, 0.85].into_iter().enumerate() {
            for (m, incident_ior) in [1.0, 1.33, 1.9].into_iter().enumerate() {
                for (v, &wo) in views.iter().enumerate() {
                    let index = r + 4 * (a + 3 * (m + 3 * v));
                    let wi = lights[index % lights.len()];
                    let tangent = if index % 2 == 0 {
                        [1.0, 0.0, 0.0]
                    } else {
                        direction(0.3, 1.0, 0.0)
                    };
                    cases.push(gpu_case(
                        roughness,
                        anisotropy,
                        incident_ior,
                        tangent,
                        wo,
                        wi,
                        randoms[index % randoms.len()],
                    ));
                }
            }
        }
    }
    // Explicit delta and invalid-incident cases exercise flags, not a fabricated continuous PDF.
    cases.push(gpu_case(
        0.0,
        0.0,
        1.0,
        [1.0, 0.0, 0.0],
        views[0],
        lights[0],
        [0.2, 0.3],
    ));
    cases.push(gpu_case(
        0.3,
        0.0,
        0.0,
        [1.0, 0.0, 0.0],
        views[0],
        lights[0],
        [0.2, 0.3],
    ));

    cases.push(gpu_case(
        1.0e-8,
        0.0,
        1.0,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.2, 0.3],
    ));
    cases.push(gpu_case(
        1.0e-5,
        0.0,
        1.0,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.96, 0.0, 0.28],
        [0.2, 0.3],
    ));
    cases.push(gpu_case(
        1.0e-10,
        0.0,
        1.0,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.2, 0.3],
    ));
    cases.push(gpu_case(
        2.0e-19,
        0.0,
        1.0,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.2, 0.3],
    ));

    let normalisation_cases = [
        (0.15, 0.0, 1.0),
        (0.35, 0.8, 1.33),
        (0.5, 0.65, 1.9),
        (0.65, 0.9, 1.0),
        (0.9, 0.0, 1.5),
    ];
    let (phi_steps, z_steps) = (256_usize, 512_usize);
    let direction_count = phi_steps * z_steps;
    let mut quadrature_cases = Vec::new();
    for (i, &(roughness, anisotropy, ior)) in normalisation_cases.iter().enumerate() {
        quadrature_cases.push(gpu_case(
            roughness,
            anisotropy,
            ior,
            if i % 2 == 0 {
                [1.0, 0.0, 0.0]
            } else {
                direction(0.3, 1.0, 0.0)
            },
            views[i % views.len()],
            [0.0, 0.0, 1.0],
            [0.3, 0.4],
        ));
    }

    let source = test_source(direction_count, phi_steps, z_steps);
    validate(&source);
    let shader = gpu.shader("pure rough conductor contract", &source, false);
    let contract_pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PBR evaluate/pdf/sample contract"),
            layout: None,
            module: &shader,
            entry_point: Some("contract"),
            compilation_options: Default::default(),
            cache: None,
        });
    let quadrature_pipeline =
        gpu.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PBR full-sphere PDF quadrature"),
                layout: None,
                module: &shader,
                entry_point: Some("quadrature"),
                compilation_options: Default::default(),
                cache: None,
            });

    let case_bytes = (cases.len() * std::mem::size_of::<GpuCase>()) as u64;
    let case_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR contract cases"),
        size: case_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&case_buffer, 0, bytemuck::cast_slice(&cases));
    let contract_bytes = (cases.len() * 4 * 16) as u64;
    let contract_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR contract results"),
        size: contract_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let contract_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &contract_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: case_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: contract_buffer.as_entire_binding(),
            },
        ],
    });
    dispatch(
        &gpu,
        &contract_pipeline,
        &contract_group,
        cases.len().div_ceil(64) as u32,
        1,
    );
    let contract_results = read_results(&gpu, &contract_buffer, contract_bytes);

    let mut event_counts = [0_u32; 6];
    for (i, case) in cases.iter().enumerate() {
        let roughness = f64::from(case.wo_roughness[3]);
        let anisotropy = f64::from(case.wi_anisotropy[3]);
        let ior = f64::from(case.tangent_ior[3]);
        let tangent = [
            f64::from(case.tangent_ior[0]),
            f64::from(case.tangent_ior[1]),
            f64::from(case.tangent_ior[2]),
        ];
        let frame = Frame::new(tangent);
        let material = copper(roughness, anisotropy);
        let wo = std::array::from_fn(|j| f64::from(case.wo_roughness[j]));
        let wi = std::array::from_fn(|j| f64::from(case.wi_anisotropy[j]));
        let random = [f64::from(case.rng[0]), f64::from(case.rng[1])];
        let expected_f = evaluate(&frame, ior, &material, wo, wi);
        let expected_pdf = pdf(&frame, ior, &material, wo, wi);
        let expected_sample = sample(&frame, ior, &material, wo, random);
        let output = &contract_results[16 * i..16 * (i + 1)];
        for channel in 0..3 {
            close_representable(
                output[channel],
                expected_f[channel],
                3e-4,
                &format!("WGSL evaluate case {i}"),
            );
            close(
                f64::from(output[4 + channel]),
                expected_sample.weight[channel],
                5e-4,
                "WGSL sample weight",
            );
            close(
                f64::from(output[8 + channel]),
                expected_sample.wi[channel],
                3e-5,
                "WGSL sample wi",
            );
        }
        close_representable(
            output[3],
            expected_pdf,
            3e-5,
            &format!("WGSL direct PDF case {i}"),
        );
        close_representable(
            output[7],
            expected_sample.pdf,
            3e-5,
            &format!("WGSL sample PDF case {i}"),
        );
        assert_eq!(
            output[11] as u32, expected_sample.event,
            "sample event, case {i}"
        );
        assert_eq!(output[12] as u32, 0, "conductor events are continuous");
        assert_eq!(output[13], 1.0, "one conductor lobe has unit selection mass");
        assert_eq!(output[14] as u32, 0, "reflection stays in the incident medium");
        close(
            f64::from(output[15]),
            1.0 / std::f64::consts::PI,
            3e-4,
            "scaled off-axis density avoids inf/inf",
        );
        event_counts[expected_sample.event as usize] += 1;
        if expected_sample.event == EVENT_BELOW_HEMISPHERE {
            assert!(
                expected_sample.pdf > 0.0,
                "rejected proposal lost its real PDF"
            );
        }
        if expected_sample.event == EVENT_REFLECTION {
            let reconstructed = evaluate(&frame, ior, &material, wo, expected_sample.wi);
            for channel in 0..3 {
                close(
                    expected_sample.weight[channel],
                    reconstructed[channel] * frame.local(expected_sample.wi)[2].abs()
                        / expected_sample.pdf,
                    1e-12,
                    "sample.weight = evaluate*cos/pdf",
                );
            }
        } else {
            assert!(output[4..7].iter().all(|&v| v == 0.0));
        }
    }
    assert!(
        [EVENT_INVALID, EVENT_REFLECTION, EVENT_BELOW_HEMISPHERE, EVENT_UNSUPPORTED_DELTA]
            .iter().all(|&event| event_counts[event as usize] > 0),
        "event coverage (invalid, reflection, below, delta): {event_counts:?}"
    );

    let quadrature_case_bytes = (quadrature_cases.len() * std::mem::size_of::<GpuCase>()) as u64;
    let quadrature_case_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR PDF cases"),
        size: quadrature_case_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(
        &quadrature_case_buffer,
        0,
        bytemuck::cast_slice(&quadrature_cases),
    );
    let quadrature_bytes = (quadrature_cases.len() * direction_count * 16) as u64;
    let quadrature_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PBR PDF quadrature results"),
        size: quadrature_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let quadrature_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &quadrature_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: quadrature_case_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: quadrature_buffer.as_entire_binding(),
            },
        ],
    });
    dispatch(
        &gpu,
        &quadrature_pipeline,
        &quadrature_group,
        direction_count.div_ceil(64) as u32,
        quadrature_cases.len() as u32,
    );
    let quadrature_results = read_results(&gpu, &quadrature_buffer, quadrature_bytes);
    let theta_step = std::f64::consts::PI / z_steps as f64;
    let phi_step = std::f64::consts::TAU / phi_steps as f64;

    for (i, case) in quadrature_cases.iter().enumerate() {
        let frame = Frame::new(std::array::from_fn(|j| f64::from(case.tangent_ior[j])));
        let material = copper(
            f64::from(case.wo_roughness[3]),
            f64::from(case.wi_anisotropy[3]),
        );
        let ior = f64::from(case.tangent_ior[3]);
        let wo = std::array::from_fn(|j| f64::from(case.wo_roughness[j]));
        let mut shader_integral = 0.0;
        let mut reference_integral = 0.0;
        for z in 0..z_steps {
            let theta = std::f64::consts::PI * (z as f64 + 0.5) / z_steps as f64;
            let (radius, cos_theta) = theta.sin_cos();
            let solid_angle = radius * theta_step * phi_step;
            for phi in 0..phi_steps {
                let azimuth = std::f64::consts::TAU * (phi as f64 + 0.5) / phi_steps as f64;
                let wi = [radius * azimuth.cos(), radius * azimuth.sin(), cos_theta];
                let index = i * direction_count + z * phi_steps + phi;
                let shader_pdf = f64::from(quadrature_results[4 * index]);
                let reference_pdf = pdf(&frame, ior, &material, wo, wi);
                assert!(shader_pdf.is_finite() && shader_pdf >= 0.0);
                close(shader_pdf, reference_pdf, 2e-4, "WGSL quadrature PDF");
                shader_integral += shader_pdf * solid_angle;
                reference_integral += reference_pdf * solid_angle;
            }
        }
        println!(
            "PDF quadrature case {i}: shader {shader_integral:.8}, reference {reference_integral:.8}"
        );
        close(shader_integral, 1.0, 0.012, "full-sphere PDF normalization");
    }
}
