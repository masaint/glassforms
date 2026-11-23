//! Real-GPU numerical checks of the specialized body and forward-caustic ownership helpers.
//! Queue this test through native/tools/gpu_run.py.

use gf_core::material::medium_priority_bodies_wgsl;
use glassforms::cast_light::{medium_priority_cast_wgsl, LIGHT_WGSL};
use glassforms::gpu::Gpu;

fn source() -> String {
    let bodies = medium_priority_bodies_wgsl(include_str!("../src/shaders/trace_bodies.wgsl"));
    let body_helpers = bodies
        .split_once("fn bodyPrecedesB").unwrap().1
        .split_once("// Body k's surface normal").unwrap().0;
    let cast = medium_priority_cast_wgsl(LIGHT_WGSL);
    let cast_helpers = cast
        .split_once("fn castPrecedes").unwrap().1
        .split_once("// March from ro along rd").unwrap().0;
    format!(r#"
const COUNT: i32 = 4;
const BIG: f32 = 1000.0;
struct MB {{ medium: array<vec4i, 4> }}
@group(0) @binding(0) var<uniform> mb: MB;
struct BodyL {{ medium: vec4i }}
struct LU {{ body: array<BodyL, 4> }}
@group(0) @binding(1) var<uniform> lu: LU;
fn bodyPrecedesB{body_helpers}
fn castPrecedes{cast_helpers}
@group(0) @binding(2) var<storage, read_write> results: array<vec4i>;
@compute @workgroup_size(1)
fn contract() {{
  let both = vec4f(-0.2, -0.3, 1.0, 1.0);
  let nearest = vec4f(-0.1, -0.1, BIG, BIG);
  let tied = vec4f(-0.1, BIG, BIG, -0.1);
  let negative = vec4f(BIG, BIG, -0.1, BIG);
  let visible = within(vec4f(0.25, -0.5, 0.25, 0.25), 1);
  results[0] = vec4i(
    containingB(both, -1), containingB(both, 0), containingB(both, 1),
    interfaceOwnerB(1, 0));
  results[1] = vec4i(
    interfaceOwnerB(0, 1), nearestOf(nearest), nearestOf(tied),
    containingB(vec4f(1.0, 1.0, -0.2, 1.0), -1));
  results[2] = vec4i(
    i32(visible.x == 0.25), i32(visible.y == 0.5),
    i32(visible.z == 0.25), i32(visible.w == 0.25));
  results[3] = vec4i(
    containingB(vec4f(1.0), -1), interfaceOwnerB(-1, 2), interfaceOwnerB(2, -1),
    i32(bodyPrecedesB(0, 1)));
  results[4] = vec4i(castWhich(nearest, -0.1), castWhich(tied, -0.1), castWhich(negative, -0.1), 0);
}}
"#)
}

fn read(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<i32> {
    let bytes = 20 * 4;
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("medium-priority download"), size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    commands.copy_buffer_to_buffer(buffer, 0, &download, 0, bytes);
    gpu.queue.submit(Some(commands.finish()));
    let (send, recv) = std::sync::mpsc::channel();
    download.slice(..).map_async(wgpu::MapMode::Read, move |result| { send.send(result).unwrap(); });
    gpu.wait();
    recv.recv().unwrap().unwrap();
    let mapped = download.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}

#[test]
fn specialized_wgsl_ownership_matches_numerical_contract() {
    let gpu = Gpu::headless("vulkan").expect("medium-priority GPU contract requires Vulkan");
    let source = source();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module).unwrap();
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("medium-priority numerical contract"), source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("medium-priority numerical contract"), layout: None, module: &shader,
        entry_point: Some("contract"), compilation_options: Default::default(), cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("medium-priority results"), size: 20 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for (priorities, expected) in [
        ([10, 0, -4, 10], [
            0, 1, 0, 0,
            0, 0, 3, 2,
            1, 1, 0, 1,
            -1, 2, 2, 1,
            0, 3, 2, 0,
        ]),
        ([0, 0, 0, 0], [
            1, 1, 0, 1,
            1, 1, 3, 2,
            0, 1, 1, 1,
            -1, 2, 2, 0,
            1, 3, 2, 0,
        ]),
    ] {
        let mut raw = [0i32; 16];
        for k in 0..4 { raw[4 * k + 1] = priorities[k]; }
        let body = gpu.uniform_buffer("medium-priority body metadata", 64);
        let cast = gpu.uniform_buffer("medium-priority caustic metadata", 64);
        gpu.queue.write_buffer(&body, 0, bytemuck::cast_slice(&raw));
        gpu.queue.write_buffer(&cast, 0, bytemuck::cast_slice(&raw));
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("medium-priority numerical contract"), layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: body.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cast.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        let mut commands = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = commands.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit(Some(commands.finish()));
        assert_eq!(read(&gpu, &output), expected, "priorities {priorities:?}");
    }
}
