use glassforms::gpu::Gpu;

const TYPES: &str = include_str!("../src/shaders/pbr/types.wgsl");
const DIELECTRIC: &str = include_str!("../src/shaders/pbr/dielectric.wgsl");
const CONDUCTOR_FRESNEL: &str = include_str!("../src/shaders/conductor_fresnel.wgsl");
const GGX: &str = concat!(include_str!("../src/shaders/pbr/ggx.wgsl"), include_str!("../src/shaders/pbr/rough_conductor.wgsl"));
const IMPORTED: &str = include_str!("../src/shaders/pbr/imported_opaque.wgsl");
const EVENT_INVALID: u32 = 0;
const EVENT_REFLECTION: u32 = 1;
const EVENT_BELOW_HEMISPHERE: u32 = 4;

fn leaf_source() -> String {
    [TYPES, CONDUCTOR_FRESNEL, DIELECTRIC, GGX, IMPORTED].join("\n")
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
fn imported_opaque_leaf_passes_naga_validation() {
    validate(&leaf_source());
}

// Independent f64 arithmetic. Dielectric Fresnel uses the trigonometric amplitude equations rather
// than transcribing the shader; no imported material is assigned measured n,k data.
type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] + b[i])
}
fn scale(a: V3, s: f64) -> V3 {
    a.map(|v| v * s)
}
fn mul(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] * b[i])
}
fn mix(a: V3, b: V3, t: f64) -> V3 {
    add(scale(a, 1.0 - t), scale(b, t))
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
        [dot(v, self.tangent), dot(v, self.bitangent), dot(v, self.normal)]
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
    base: V3,
    metal: f64,
    rough: f64,
}

fn valid(frame: &Frame, incident: f64, destination: f64, wo: V3) -> bool {
    incident > 0.0 && destination > 0.0 && frame.local(wo)[2] > 0.0
}

fn dielectric_fresnel(cosine: f64, incident: f64, destination: f64) -> f64 {
    let c = cosine.clamp(0.0, 1.0);
    if c <= 1e-14 {
        return 1.0;
    }
    if 1.0 - c <= 1e-14 {
        return ((incident - destination) / (incident + destination)).powi(2);
    }
    let theta_i = c.acos();
    let transmitted_sine = incident / destination * theta_i.sin();
    if transmitted_sine >= 1.0 {
        return 1.0;
    }
    let theta_t = transmitted_sine.asin();
    let rs = (theta_i - theta_t).sin() / (theta_i + theta_t).sin();
    let rp = (theta_i - theta_t).tan() / (theta_i + theta_t).tan();
    0.5 * (rs * rs + rp * rp)
}

fn metal_fresnel(material: &Material, cosine: f64) -> V3 {
    let edge = (1.0 - cosine.clamp(0.0, 1.0)).powi(5);
    material.base.map(|f0| f0 + (1.0 - f0) * edge)
}

fn specular_fresnel(incident: f64, destination: f64, material: &Material, cosine: f64) -> V3 {
    mix(
        [dielectric_fresnel(cosine, incident, destination); 3],
        metal_fresnel(material, cosine),
        material.metal,
    )
}

fn diffuse_albedo(frame: &Frame, incident: f64, destination: f64, material: &Material, wo: V3) -> V3 {
    if !valid(frame, incident, destination, wo) {
        return [0.0; 3];
    }
    scale(
        material.base,
        (1.0 - material.metal)
            * (1.0 - dielectric_fresnel(frame.local(wo)[2], incident, destination)),
    )
}

fn half_vector(wo: V3, wi: V3) -> Option<V3> {
    let h = add(wo, wi);
    (dot(h, h) > 1e-24).then(|| unit(h))
}

fn lambda(v: V3, alpha: f64) -> f64 {
    0.5 * ((1.0 + alpha * alpha * (v[0] * v[0] + v[1] * v[1]) / (v[2] * v[2])).sqrt() - 1.0)
}

fn density(h: V3, alpha: f64) -> f64 {
    let stretched = h[0] * h[0] + h[1] * h[1] + alpha * alpha * h[2] * h[2];
    alpha * alpha / (std::f64::consts::PI * stretched * stretched)
}

fn ggx_evaluate(frame: &Frame, rough: f64, wo: V3, wi: V3) -> f64 {
    if rough <= 0.0 {
        return 0.0;
    }
    let alpha = rough * rough;
    let (v, l) = (frame.local(wo), frame.local(wi));
    if v[2] <= 0.0 || l[2] <= 0.0 {
        return 0.0;
    }
    let Some(h_world) = half_vector(wo, wi) else {
        return 0.0;
    };
    let h = frame.local(h_world);
    if h[2] <= 0.0 || dot(v, h) <= 0.0 {
        return 0.0;
    }
    let g2 = 1.0 / (1.0 + lambda(v, alpha) + lambda(l, alpha));
    density(h, alpha) * g2 / (4.0 * v[2] * l[2])
}

fn ggx_pdf(frame: &Frame, rough: f64, wo: V3, wi: V3) -> f64 {
    if rough <= 0.0 {
        return 0.0;
    }
    let alpha = rough * rough;
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
    density(h, alpha) / ((1.0 + lambda(v, alpha)) * 4.0 * v[2])
}

fn diffuse_probability(material: &Material) -> f64 {
    0.5 * (1.0 - material.metal)
}

fn evaluate(frame: &Frame, incident: f64, destination: f64, material: &Material, wo: V3, wi: V3) -> V3 {
    if !valid(frame, incident, destination, wo) || frame.local(wi)[2] <= 0.0 {
        return [0.0; 3];
    }
    let diffuse = scale(
        diffuse_albedo(frame, incident, destination, material, wo),
        std::f64::consts::FRAC_1_PI,
    );
    if material.rough <= 0.0 {
        return diffuse;
    }
    let Some(h) = half_vector(wo, wi) else {
        return diffuse;
    };
    add(
        diffuse,
        scale(
            specular_fresnel(incident, destination, material, dot(wo, h)),
            ggx_evaluate(frame, material.rough, wo, wi),
        ),
    )
}

fn pdf(frame: &Frame, incident: f64, destination: f64, material: &Material, wo: V3, wi: V3) -> f64 {
    if !valid(frame, incident, destination, wo) {
        return 0.0;
    }
    let diffuse = diffuse_probability(material) * frame.local(wi)[2].max(0.0) * std::f64::consts::FRAC_1_PI;
    if material.rough <= 0.0 {
        return diffuse;
    }
    diffuse
        + (1.0 - diffuse_probability(material)) * ggx_pdf(frame, material.rough, wo, wi)
}

fn response(
    frame: &Frame,
    incident: f64,
    destination: f64,
    material: &Material,
    wo: V3,
    diffuse_gather: V3,
    specular_cosine_integral: V3,
) -> V3 {
    if !valid(frame, incident, destination, wo) {
        return [0.0; 3];
    }
    add(
        mul(
            diffuse_albedo(frame, incident, destination, material, wo),
            diffuse_gather,
        ),
        mul(
            specular_fresnel(incident, destination, material, frame.local(wo)[2]),
            specular_cosine_integral,
        ),
    )
}

struct Sample {
    wi: V3,
    weight: V3,
    pdf: f64,
    event: u32,
    delta: bool,
    event_probability: f64,
}

fn sample(
    frame: &Frame,
    incident: f64,
    destination: f64,
    material: &Material,
    wo: V3,
    random: [f64; 3],
) -> Sample {
    let invalid = || Sample {
        wi: [0.0; 3],
        weight: [0.0; 3],
        pdf: 0.0,
        event: EVENT_INVALID,
        delta: false,
        event_probability: 1.0,
    };
    if !valid(frame, incident, destination, wo) || random.iter().any(|&u| !(0.0..1.0).contains(&u)) {
        return invalid();
    }
    let p_diffuse = diffuse_probability(material);
    let wi;
    if random[2] < p_diffuse || material.rough <= 0.0 {
        if random[2] >= p_diffuse {
            let weight = scale(
                specular_fresnel(incident, destination, material, frame.local(wo)[2]),
                1.0 / (1.0 - p_diffuse),
            );
            return Sample {
                wi: add(scale(wo, -1.0), scale(frame.normal, 2.0 * frame.local(wo)[2])),
                weight,
                pdf: 0.0,
                event: EVENT_REFLECTION,
                delta: true,
                event_probability: 1.0 - p_diffuse,
            };
        }
        let radius = random[0].sqrt();
        let phi = std::f64::consts::TAU * random[1];
        wi = frame.world([
            radius * phi.cos(),
            radius * phi.sin(),
            (1.0 - random[0]).max(0.0).sqrt(),
        ]);
    } else {
        // Independent stretched-hemisphere VNDF construction for the shared GGX proposal.
        let alpha = material.rough * material.rough;
        let local = frame.local(wo);
        let v = unit([alpha * local[0], alpha * local[1], local[2]]);
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
        let p2 = (1.0 - hemisphere) * (1.0 - p1 * p1).max(0.0).sqrt()
            + hemisphere * r * phi.sin();
        let nh = add(
            add(scale(t1, p1), scale(t2, p2)),
            scale(v, (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt()),
        );
        let h = frame.world(unit([alpha * nh[0], alpha * nh[1], nh[2].max(0.0)]));
        wi = add(scale(wo, -1.0), scale(h, 2.0 * dot(wo, h)));
    }
    let probability = pdf(frame, incident, destination, material, wo, wi);
    if probability <= 0.0 {
        return invalid();
    }
    if frame.local(wi)[2] <= 0.0 {
        return Sample {
            wi,
            weight: [0.0; 3],
            pdf: probability,
            event: EVENT_BELOW_HEMISPHERE,
            delta: false,
            event_probability: 1.0,
        };
    }
    let f = evaluate(frame, incident, destination, material, wo, wi);
    Sample {
        wi,
        weight: scale(f, frame.local(wi)[2].abs() / probability),
        pdf: probability,
        event: EVENT_REFLECTION,
        delta: false,
        event_probability: 1.0,
    }
}

fn close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(actual.is_finite() && expected.is_finite(), "{context}: {actual} / {expected}");
    let error = (actual - expected).abs();
    assert!(error <= tolerance * (1.0 + expected.abs()), "{context}: {actual} != {expected} (error {error})");
}

#[test]
fn independent_reference_covers_smooth_rough_metalness_cosine_and_domains() {
    let frame = Frame::new([1.0, 0.0, 0.0]);
    let wo = direction(0.4, -0.2, 0.8);
    let wi = direction(-0.3, 0.5, 0.6);
    let base = [0.65, 0.3, 0.12];
    close(dielectric_fresnel(1.0, 1.0, 1.5), 0.04, 1e-14, "normal Fresnel");
    close(dielectric_fresnel(0.0, 1.0, 1.5), 1.0, 1e-14, "grazing Fresnel");
    close(dielectric_fresnel(0.2, 1.5, 1.0), 1.0, 1e-14, "total internal reflection");

    let dielectric = Material { base, metal: 0.0, rough: 0.0 };
    let other_wi = direction(0.7, 0.1, 0.2);
    assert_eq!(
        evaluate(&frame, 1.0, 1.5, &dielectric, wo, wi),
        evaluate(&frame, 1.0, 1.5, &dielectric, wo, other_wi),
        "smooth dielectric continuous response is cosine-free Lambertian f, not f*cos"
    );
    assert_eq!(pdf(&frame, 1.0, 1.5, &dielectric, wo, scale(wi, -1.0)), 0.0);
    assert_eq!(evaluate(&frame, 1.0, 1.5, &dielectric, wo, scale(wi, -1.0)), [0.0; 3]);
    assert_eq!(evaluate(&frame, 1.0, 1.5, &dielectric, scale(wo, -1.0), wi), [0.0; 3]);
    assert_eq!(pdf(&frame, 1.0, 1.5, &dielectric, scale(wo, -1.0), wi), 0.0);

    let metal = Material { base, metal: 1.0, rough: 0.45 };
    assert_eq!(diffuse_albedo(&frame, 1.0, 1.5, &metal, wo), [0.0; 3]);
    let smooth_metal = Material { rough: 0.0, ..metal.clone() };
    assert_eq!(evaluate(&frame, 1.0, 1.5, &smooth_metal, wo, wi), [0.0; 3]);
    let delta = sample(&frame, 1.0, 1.5, &smooth_metal, wo, [0.2, 0.7, 0.3]);
    assert!(delta.delta && delta.pdf == 0.0 && delta.event_probability == 1.0);

    for rough in [0.0, 0.35, 0.8] {
        let d = Material { base, metal: 0.0, rough };
        let m = Material { base, metal: 1.0, rough };
        let mixed = Material { base, metal: 0.37, rough };
        let fd = evaluate(&frame, 1.33, 1.62, &d, wo, wi);
        let fm = evaluate(&frame, 1.33, 1.62, &m, wo, wi);
        let actual = evaluate(&frame, 1.33, 1.62, &mixed, wo, wi);
        let expected = mix(fd, fm, 0.37);
        for channel in 0..3 {
            close(actual[channel], expected[channel], 2e-12, "continuous metalness mixes final responses");
        }
    }

    let gathers = ([0.31, 0.47, 0.59], [0.23, 0.37, 0.53]);
    let actual = response(&frame, 1.0, 1.5, &mixed_for_response(), wo, gathers.0, gathers.1);
    let albedo = diffuse_albedo(&frame, 1.0, 1.5, &mixed_for_response(), wo);
    let f = specular_fresnel(1.0, 1.5, &mixed_for_response(), frame.local(wo)[2]);
    for channel in 0..3 {
        let expected = albedo[channel] * gathers.0[channel] + f[channel] * gathers.1[channel];
        close(actual[channel], expected, 1e-14, "finite-lamp response has one cosine and one Fresnel partition");
    }
}

fn mixed_for_response() -> Material {
    Material { base: [0.65, 0.3, 0.12], metal: 0.37, rough: 0.45 }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCase {
    tangent_ior: [f32; 4],
    wo_roughness: [f32; 4],
    wi_metal: [f32; 4],
    rng: [f32; 4],
}

fn gpu_case(rough: f64, metal: f64, incident: f64, tangent: V3, wo: V3, wi: V3, rng: [f64; 3]) -> GpuCase {
    GpuCase {
        tangent_ior: [tangent[0] as f32, tangent[1] as f32, tangent[2] as f32, incident as f32],
        wo_roughness: [wo[0] as f32, wo[1] as f32, wo[2] as f32, rough as f32],
        wi_metal: [wi[0] as f32, wi[1] as f32, wi[2] as f32, metal as f32],
        rng: [rng[0] as f32, rng[1] as f32, rng[2] as f32, 0.0],
    }
}

fn test_source(direction_count: usize, phi_steps: usize, z_steps: usize) -> String {
    format!(
        r#"
{leaf}
struct PbrTestCase {{
  tangent_ior: vec4f,
  wo_roughness: vec4f,
  wi_metal: vec4f,
  rng: vec4f,
}}
fn testHit(input: PbrTestCase) -> PbrSurfaceHit {{
  let t = normalize(input.tangent_ior.xyz);
  let n = vec3f(0.0,0.0,1.0);
  return PbrSurfaceHit(PbrFrame(t,cross(n,t),n),
    PbrMedium(input.tangent_ior[3],0.0,0u,0),PbrMedium(1.5,0.0,0u,0),550.0);
}}
fn testMaterial(input: PbrTestCase) -> PbrImportedOpaque {{
  return PbrImportedOpaque(vec3f(0.65,0.3,0.12),input.wi_metal[3],input.wo_roughness[3]);
}}
// The source-composition test binds this adapter policy to fileSurface itself. The pure response
// remains strict; only the accepted opaque hit's legacy grazing fallback is exercised here.
fn testAdapterResponse(hit: PbrSurfaceHit, material: PbrImportedOpaque, wo: vec3f,
                       diffuse: vec3f, specular: vec3f) -> vec3f {{
  if (dot(wo,hit.frame.normal)<=0.0) {{ return specular; }}
  return pbr_imported_opaque_response(hit,material,wo,diffuse,specular);
}}
@group(0) @binding(0) var<storage,read> cases: array<PbrTestCase>;
@group(0) @binding(1) var<storage,read_write> results: array<vec4f>;
@compute @workgroup_size(64)
fn contract(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.x];
  let hit = testHit(input);
  let material = testMaterial(input);
  let wo = input.wo_roughness.xyz;
  let wi = input.wi_metal.xyz;
  let f = pbr_imported_opaque_evaluate(hit,material,wo,wi);
  let p = pbr_imported_opaque_pdf(hit,material,wo,wi);
  let direct = pbr_imported_opaque_response(hit,material,wo,vec3f(0.31,0.47,0.59),vec3f(0.23,0.37,0.53));
  let adapter = testAdapterResponse(hit,material,wo,vec3f(0.31,0.47,0.59),vec3f(0.23,0.37,0.53));
  let s = pbr_imported_opaque_sample(hit,material,wo,PbrPathRng(input.rng.xy,input.rng.z));
  results[6u*id.x] = vec4f(f,p);
  results[6u*id.x+1u] = vec4f(direct,pbr_imported_opaque_diffuse_probability(material));
  results[6u*id.x+2u] = vec4f(s.weight,s.pdf);
  results[6u*id.x+3u] = vec4f(s.wi,f32(s.event));
  results[6u*id.x+4u] = vec4f(f32(s.delta),s.event_probability,f32(s.medium_transition),0.0);
  results[6u*id.x+5u] = vec4f(adapter,0.0);
}}
const TEST_DIRECTION_COUNT: u32 = {direction_count}u;
const TEST_PHI_STEPS: u32 = {phi_steps}u;
const TEST_Z_STEPS: u32 = {z_steps}u;
@compute @workgroup_size(64)
fn quadrature(@builtin(global_invocation_id) id: vec3u) {{
  if (id.x >= TEST_DIRECTION_COUNT || id.y >= arrayLength(&cases)) {{ return; }}
  let input = cases[id.y];
  let hit = testHit(input);
  let material = testMaterial(input);
  let theta = PBR_PI*(f32(id.x/TEST_PHI_STEPS)+0.5)/f32(TEST_Z_STEPS);
  let phi = PBR_TAU*(f32(id.x%TEST_PHI_STEPS)+0.5)/f32(TEST_PHI_STEPS);
  let z = cos(theta);
  let radius = sin(theta);
  let wi = vec3f(radius*cos(phi),radius*sin(phi),z);
  let p = pbr_imported_opaque_pdf(hit,material,input.wo_roughness.xyz,wi);
  results[id.y*TEST_DIRECTION_COUNT+id.x] = vec4f(p,0.0,0.0,0.0);
}}
"#,
        leaf = leaf_source(),
    )
}

fn read_results(gpu: &Gpu, buffer: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imported opaque contract download"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    commands.copy_buffer_to_buffer(buffer, 0, &download, 0, bytes);
    gpu.queue.submit(Some(commands.finish()));
    let (send, recv) = std::sync::mpsc::channel();
    download.slice(..).map_async(wgpu::MapMode::Read, move |result| {
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
            | if copy_dst { wgpu::BufferUsages::COPY_DST } else { wgpu::BufferUsages::COPY_SRC },
        mapped_at_creation: false,
    })
}

fn group(gpu: &Gpu, pipeline: &wgpu::ComputePipeline, input: &wgpu::Buffer, output: &wgpu::Buffer) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    })
}

#[test]
fn wgsl_contract_matches_f64_response_and_pdf_normalizes() {
    let gpu = Gpu::headless("vulkan").expect("imported opaque WGSL contract requires Vulkan");
    let views = [direction(0.0, 0.0, 1.0), direction(0.65, -0.3, 0.7), direction(2.0, 0.2, 0.22)];
    let lights = [
        direction(-0.2, 0.4, 1.0),
        direction(0.8, -0.1, 0.3),
        direction(0.2, 0.3, -0.4),
        direction(-1.0, -0.5, 0.05),
        direction(0.1, 0.8, 0.25),
    ];
    let randoms = [[0.13, 0.77, 0.05], [0.81, 0.22, 0.45], [0.5, 0.5, 0.8], [0.997, 0.003, 0.99]];
    let mut cases = Vec::new();
    for (r, rough) in [0.0, 0.08, 0.35, 0.8].into_iter().enumerate() {
        for (m, metal) in [0.0, 0.37, 1.0].into_iter().enumerate() {
            for (ior, incident) in [1.0, 1.33].into_iter().enumerate() {
                for (v, &wo) in views.iter().enumerate() {
                    let index = r + 4 * (m + 3 * (ior + 2 * v));
                    cases.push(gpu_case(
                        rough,
                        metal,
                        incident,
                        if index % 2 == 0 { [1.0, 0.0, 0.0] } else { direction(0.3, 1.0, 0.0) },
                        wo,
                        lights[index % lights.len()],
                        randoms[index % randoms.len()],
                    ));
                }
            }
        }
    }
    cases.push(gpu_case(0.4, 0.5, 0.0, [1.0, 0.0, 0.0], views[0], lights[0], [0.2, 0.3, 0.8]));
    // Exercise the production adapter's compatibility domain with valid media. Frame::new keeps
    // world Z as its normal, so the second view is exactly grazing even after normalization to f32.
    cases.push(gpu_case(0.4, 0.5, 1.0, [1.0, 0.0, 0.0], direction(0.4, -0.2, -0.8), lights[0], [0.2, 0.3, 0.8]));
    cases.push(gpu_case(0.4, 0.5, 1.0, [1.0, 0.0, 0.0], direction(0.4, -0.2, 0.0), lights[0], [0.2, 0.3, 0.8]));

    let normalisation_cases = [
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.33),
        (0.15, 0.0, 1.0),
        (0.35, 0.37, 1.33),
        (0.65, 1.0, 1.0),
        (0.9, 0.2, 1.5),
    ];
    let (phi_steps, z_steps) = (128_usize, 256_usize);
    let direction_count = phi_steps * z_steps;
    let quadrature_cases = normalisation_cases.iter().enumerate().map(|(i, &(rough, metal, incident))| {
        gpu_case(
            rough,
            metal,
            incident,
            if i % 2 == 0 { [1.0, 0.0, 0.0] } else { direction(0.3, 1.0, 0.0) },
            // Resolve narrow low-roughness lobes on the fixed solid-angle grid. Grazing views remain
            // in the contract matrix and in the broad-lobe normalization case below.
            if i == 2 { views[0] } else { views[i % views.len()] },
            [0.0, 0.0, 1.0],
            [0.3, 0.4, 0.8],
        )
    }).collect::<Vec<_>>();

    let source = test_source(direction_count, phi_steps, z_steps);
    validate(&source);
    let shader = gpu.shader("pure imported opaque contract", &source, false);
    let contract_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("imported opaque evaluate/pdf/response/sample"),
        layout: None,
        module: &shader,
        entry_point: Some("contract"),
        compilation_options: Default::default(),
        cache: None,
    });
    let quadrature_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("imported opaque full-sphere PDF quadrature"),
        layout: None,
        module: &shader,
        entry_point: Some("quadrature"),
        compilation_options: Default::default(),
        cache: None,
    });

    let case_buffer = storage(&gpu, "imported opaque cases", (cases.len() * std::mem::size_of::<GpuCase>()) as u64, true);
    gpu.queue.write_buffer(&case_buffer, 0, bytemuck::cast_slice(&cases));
    let contract_bytes = (cases.len() * 6 * 16) as u64;
    let contract_buffer = storage(&gpu, "imported opaque results", contract_bytes, false);
    let contract_group = group(&gpu, &contract_pipeline, &case_buffer, &contract_buffer);
    dispatch(&gpu, &contract_pipeline, &contract_group, cases.len().div_ceil(64) as u32, 1);
    let contract_results = read_results(&gpu, &contract_buffer, contract_bytes);

    let mut event_counts = [0_u32; 5];
    let mut adapter_fallback_coverage = [0_u32; 2];
    for (i, case) in cases.iter().enumerate() {
        let rough = f64::from(case.wo_roughness[3]);
        let metal = f64::from(case.wi_metal[3]);
        let incident = f64::from(case.tangent_ior[3]);
        let tangent = std::array::from_fn(|j| f64::from(case.tangent_ior[j]));
        let frame = Frame::new(tangent);
        let material = Material { base: [0.65, 0.3, 0.12], metal, rough };
        let wo = std::array::from_fn(|j| f64::from(case.wo_roughness[j]));
        let wi = std::array::from_fn(|j| f64::from(case.wi_metal[j]));
        let random = std::array::from_fn(|j| f64::from(case.rng[j]));
        let expected_f = evaluate(&frame, incident, 1.5, &material, wo, wi);
        let expected_pdf = pdf(&frame, incident, 1.5, &material, wo, wi);
        let expected_direct = response(&frame, incident, 1.5, &material, wo, [0.31, 0.47, 0.59], [0.23, 0.37, 0.53]);
        let expected_sample = sample(&frame, incident, 1.5, &material, wo, random);
        let output = &contract_results[24 * i..24 * (i + 1)];
        for channel in 0..3 {
            close(f64::from(output[channel]), expected_f[channel], 4e-4, "WGSL imported evaluate");
            close(f64::from(output[4 + channel]), expected_direct[channel], 3e-5, "WGSL direct finite-lamp response");
            close(f64::from(output[8 + channel]), expected_sample.weight[channel], 8e-4,
                &format!("WGSL sample weight case {i} channel {channel}"));
            close(f64::from(output[12 + channel]), expected_sample.wi[channel], 3e-5,
                &format!("WGSL sample wi case {i} channel {channel}"));
        }
        close(f64::from(output[3]), expected_pdf, 3e-5, "WGSL direct PDF");
        close(f64::from(output[7]), diffuse_probability(&material), 0.0, "diffuse selection probability");
        close(f64::from(output[11]), expected_sample.pdf, 3e-5, "WGSL sample PDF");
        assert_eq!(output[15] as u32, expected_sample.event, "sample event case {i}");
        assert_eq!(output[16] as u32, u32::from(expected_sample.delta), "delta flag case {i}");
        close(f64::from(output[17]), expected_sample.event_probability, 1e-6, "event probability");
        assert_eq!(output[18] as u32, 0, "opaque reflection stays in incident medium");
        let local_z = frame.local(wo)[2];
        let expected_adapter = if local_z <= 0.0 { [0.23, 0.37, 0.53] } else { expected_direct };
        for channel in 0..3 {
            close(f64::from(output[20 + channel]), expected_adapter[channel], 3e-5,
                "accepted opaque hit keeps the complete nonzero grazing specular gather");
        }
        if incident > 0.0 && local_z <= 0.0 {
            assert_eq!(expected_direct, [0.0; 3], "pure closure rejects a valid-media nonpositive-Z view");
            for channel in 0..3 {
                close(f64::from(output[4 + channel]), 0.0, 0.0,
                    "pure direct response remains zero at the adapter fallback");
                close(f64::from(output[20 + channel]), [0.23, 0.37, 0.53][channel], 3e-5,
                    "adapter returns the complete nonzero specular gather at the fallback");
            }
            adapter_fallback_coverage[usize::from(local_z == 0.0)] += 1;
        }
        event_counts[expected_sample.event as usize] += 1;
        if expected_sample.event == EVENT_REFLECTION && !expected_sample.delta {
            for channel in 0..3 {
                close(
                    expected_sample.weight[channel],
                    evaluate(&frame, incident, 1.5, &material, wo, expected_sample.wi)[channel]
                        * frame.local(expected_sample.wi)[2].abs() / expected_sample.pdf,
                    1e-12,
                    "sample.weight = evaluate*cos/pdf",
                );
            }
        }
    }
    assert!(
        [EVENT_INVALID, EVENT_REFLECTION, EVENT_BELOW_HEMISPHERE].iter().all(|&event| event_counts[event as usize] > 0),
        "event coverage (invalid, reflection, below): {event_counts:?}"
    );
    assert!(cases.iter().enumerate().any(|(i, case)| {
        case.wo_roughness[3] == 0.0 && contract_results[20 * i + 16] as u32 == 1
    }), "smooth specular delta coverage");

    let quadrature_case_buffer = storage(&gpu, "imported opaque PDF cases", (quadrature_cases.len() * std::mem::size_of::<GpuCase>()) as u64, true);
    gpu.queue.write_buffer(&quadrature_case_buffer, 0, bytemuck::cast_slice(&quadrature_cases));
    let quadrature_bytes = (quadrature_cases.len() * direction_count * 16) as u64;
    let quadrature_buffer = storage(&gpu, "imported opaque PDF quadrature results", quadrature_bytes, false);
    let quadrature_group = group(&gpu, &quadrature_pipeline, &quadrature_case_buffer, &quadrature_buffer);
    dispatch(&gpu, &quadrature_pipeline, &quadrature_group, direction_count.div_ceil(64) as u32, quadrature_cases.len() as u32);
    let quadrature_results = read_results(&gpu, &quadrature_buffer, quadrature_bytes);
    let theta_step = std::f64::consts::PI / z_steps as f64;
    let phi_step = std::f64::consts::TAU / phi_steps as f64;

    for (i, case) in quadrature_cases.iter().enumerate() {
        let frame = Frame::new(std::array::from_fn(|j| f64::from(case.tangent_ior[j])));
        let material = Material {
            base: [0.65, 0.3, 0.12],
            metal: f64::from(case.wi_metal[3]),
            rough: f64::from(case.wo_roughness[3]),
        };
        let incident = f64::from(case.tangent_ior[3]);
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
                let reference_pdf = pdf(&frame, incident, 1.5, &material, wo, wi);
                assert!(shader_pdf.is_finite() && shader_pdf >= 0.0);
                close(shader_pdf, reference_pdf, 2e-4, "WGSL imported quadrature PDF");
                shader_integral += shader_pdf * solid_angle;
                reference_integral += reference_pdf * solid_angle;
            }
        }
        let delta_mass = if material.rough <= 0.0 { 1.0 - diffuse_probability(&material) } else { 0.0 };
        println!("imported PDF quadrature case {i}: shader {shader_integral:.8}, reference {reference_integral:.8}, delta {delta_mass:.8}");
        close(shader_integral + delta_mass, 1.0, 0.012, "full-sphere PDF plus smooth delta normalization");
        close(reference_integral + delta_mass, 1.0, 0.012, "independent PDF plus smooth delta normalization");
    }
}
