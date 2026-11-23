//! Transport the authored surface ID through the same coordinate and copy
//! selection as each voxel deformation. Geometry-only sculpting and smoothing
//! extend the source labels in place. The final surface supplies the distance.
use crate::{gpu::Gpu, world_faces::LocalGrid};
use wgpu::util::DeviceExt;

pub(crate) struct Passes {
    warp: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

const SHADER: &str = r#"
@group(0) @binding(4) var<storage,read> priorFaces:array<u32>;
@group(0) @binding(5) var<storage,read_write> nextFaces:array<u32>;
@group(0) @binding(6) var<uniform> faceDims:vec4u;
fn faceIndex(id:vec3u)->u32 {
  return id.x+faceDims.x*(id.y+faceDims.x*id.z);
}
@compute @workgroup_size(8,8,4)
fn warpFaces(@builtin(global_invocation_id) cell:vec3u) {
  let n=faceDims.x;
  if(any(cell>=vec3u(n))){return;}
  let p=((vec3f(cell)+0.5)/f32(n)*2.0-1.0)*mu.head.y;
  let back=modRead(p).source/(2.0*mu.head.y)+0.5;
  let prior = faceDims.y;
  let origin = vec3u(clamp(back * f32(prior), vec3f(0.0), vec3f(f32(prior - 1u))));
  nextFaces[faceIndex(cell)] = priorFaces[origin.x + prior * (origin.y + prior * origin.z)];
}
@compute @workgroup_size(8,8,4)
fn finishFaces(@builtin(global_invocation_id) cell:vec3u) {
  let n=faceDims.x;
  if(any(cell>=vec3u(n))){return;}
  let uv=(vec3f(cell)+0.5)/f32(n);
  let d=textureSampleLevel(srcVol,smp,uv,0.).r;
  let i=faceIndex(cell);
  nextFaces[i]=(nextFaces[i]&0xffff0000u)|(pack2x16float(vec2f(d,0.0))&65535u);
}
"#;

impl Passes {
    pub fn new(gpu: &Gpu) -> Self {
        let src = gf_core::mods::wgsl_baked() + SHADER;
        let shader = gpu.shader("construction faces through baked modifiers", &src, true);
        let pipeline = |name| {
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &shader,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        Self {
            warp: pipeline("warpFaces"),
            finish: pipeline("finishFaces"),
        }
    }
    pub fn dims(gpu: &Gpu, res: u32, prior: u32) -> wgpu::Buffer {
        gpu.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("construction face grid size"),
                contents: bytemuck::cast_slice(&[res, prior, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn warp(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        face: &mut LocalGrid,
        target: u32,
        dims: &wgpu::Buffer,
        src: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        uniform: &wgpu::Buffer,
        offset: u64,
    ) {
        let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deformed construction faces"),
            size: u64::from(target).pow(3) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset,
                    size: wgpu::BufferSize::new((gf_core::mods::BAKED_FLOATS * 4) as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: face.grid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: dims.as_entire_binding(),
            },
        ];
        dispatch(gpu, enc, &self.warp, &entries, target);
        face.res = target;
        face.grid = out;
    }
    pub fn finish(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        face: &LocalGrid,
        dims: &wgpu::Buffer,
        src: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        let entries = [
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: face.grid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: dims.as_entire_binding(),
            },
        ];
        dispatch(gpu, enc, &self.finish, &entries, face.res);
    }
}
fn dispatch(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    pipe: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry],
    res: u32,
) {
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries,
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("deformed construction faces"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), res.div_ceil(4));
}
