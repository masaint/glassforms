//! Keep a baked group's exact R16 distance and its transported surface IDs in
//! one grid binding. Its first slices are a bitwise copy of the geometry.
//! Remaining slices contain two exact bytes per surface-table index.
use crate::{gpu::Gpu, world_faces::LocalGrid};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[derive(Default)]
pub(crate) struct Atlases {
    pipe: Option<wgpu::ComputePipeline>,
    fields: HashMap<usize, (String, wgpu::TextureView)>,
}
const SHADER: &str = r#"
@group(0) @binding(0) var<storage,read> face:array<u32>;
@group(0) @binding(1) var volume:texture_storage_3d<r16float,write>;
@group(0) @binding(2) var<uniform> dims:vec4u;
@compute @workgroup_size(8,8,4)
fn pack(@builtin(global_invocation_id) p:vec3u) {
    let r=dims.x;let n=dims.y;
    let i=p.x+r*(p.y+r*p.z);
    if(any(p.xy>=vec2u(r)) || i>=2u*n*n*n) {return;}
    let label=face[i/2u]>>16u;
    let byte=(label>>(8u*(i&1u)))&255u;
    textureStore(volume,p+vec3u(0,0,r),vec4f(f32(byte)));
}
"#;
impl Atlases {
    pub fn get(&self, index: usize, signature: &str) -> Option<wgpu::TextureView> {
        self.fields
            .get(&index)
            .filter(|(key, _)| key == signature)
            .map(|(_, v)| v.clone())
    }
    pub fn retain(&mut self, range: std::ops::Range<usize>, active: &[usize]) {
        self.fields
            .retain(|index, _| !range.contains(index) || active.contains(index));
    }
    pub fn forget(&mut self, index: usize) {
        self.fields.remove(&index);
    }
    pub fn store(
        &mut self,
        gpu: &Gpu,
        index: usize,
        signature: String,
        res: u32,
        geometry: &wgpu::Texture,
        faces: &LocalGrid,
    ) {
        let pipe = self.pipe.get_or_insert_with(|| {
            let shader = gpu.shader("cached construction face atlas", SHADER, false);
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("pack construction face IDs"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("pack"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        });
        let n = faces.res;
        let extra = (2 * n.pow(3)).div_ceil(res * res);
        let atlas = gpu.volume(
            "construction surface and face atlas",
            res,
            res + extra,
            crate::bake::VOL_FORMAT,
            true,
        );
        let view = atlas.create_view(&Default::default());
        let dims = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("construction atlas dimensions"),
                usage: wgpu::BufferUsages::UNIFORM,
                contents: bytemuck::cast_slice(&[res, n, 0, 0]),
            });
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: faces.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims.as_entire_binding(),
                },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_texture(
            geometry.as_image_copy(),
            atlas.as_image_copy(),
            wgpu::Extent3d {
                width: res,
                height: res,
                depth_or_array_layers: res,
            },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack baked construction face IDs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), extra.div_ceil(4));
        }
        gpu.queue.submit([enc.finish()]);
        self.fields.insert(index, (signature, view));
    }
}
