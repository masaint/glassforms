//! Bubbles: hollows cut into a finished distance field (shaders/bubbles.wgsl says how).
//!
//! The shape's own volume is left as it is and the bubbled copy is written into a second one, so the
//! bubbles can be changed without building the shape again. That second volume is the cost: as much
//! video memory again as the shape's.

use bytemuck::{Pod, Zeroable};
use gf_core::material::Bubbles;
use gf_core::HB;

use crate::bake::{Volume, VOL_FORMAT};
use crate::gpu::*;

const BUBBLES_WGSL: &str = include_str!("shaders/bubbles.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BubbleU {
    res: f32,
    hb: f32,
    cell: f32,
    size: f32,
    even: f32,
    shape: f32,
    seed: f32,
    margin: f32,
}

/// How the bubbled copy gets written: by one compute dispatch, or a slice at a time.
enum Way {
    Computed(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    Drawn(wgpu::RenderPipeline, wgpu::BindGroupLayout),
}

pub struct Bubbler {
    gpu: Gpu,
    lin: wgpu::Sampler,
    u: wgpu::Buffer,
    way: Way,
    /// bubbled volumes kept for reuse, and which of them was written last
    vols: Vec<Volume>,
    cur: usize,
}

impl Bubbler {
    pub fn new(gpu: &Gpu) -> Self {
        let tex = |vis| wgpu::BindGroupLayoutEntry { binding: 1, visibility: vis, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None };
        let smp = |vis| wgpu::BindGroupLayoutEntry { binding: 2, visibility: vis, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
        let way = if gpu.storage16 {
            let c = wgpu::ShaderStages::COMPUTE;
            let out = wgpu::BindGroupLayoutEntry { binding: 3, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: VOL_FORMAT, view_dimension: wgpu::TextureViewDimension::D3 }, count: None };
            let layout = gpu.bind_layout("bubbles (computed)", &[uniform_entry(0, c, false), tex(c), smp(c), out]);
            let module = gpu.shader("bubbles (computed)", BUBBLES_WGSL, false);
            let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("bubbles (computed)"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipe = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("bubbles (computed)"), layout: Some(&pl), module: &module, entry_point: Some("cs"), compilation_options: Default::default(), cache: None });
            Way::Computed(pipe, layout)
        } else {
            let f = wgpu::ShaderStages::FRAGMENT;
            let layout = gpu.bind_layout("bubbles", &[uniform_entry(0, f, false), tex(f), smp(f)]);
            let drawn = BUBBLES_WGSL.split("//#compute").next().unwrap();
            let module = gpu.shader("bubbles", drawn, false);
            let pipe = gpu.pipeline(&PipeDesc { label: "bubbles", module: &module, vs: "vs", fs: "fs", layouts: &[&layout], format: VOL_FORMAT, blend: None, strip: false, buffers: &[] });
            Way::Drawn(pipe, layout)
        };
        Self {
            gpu: gpu.clone(),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            u: gpu.uniform_buffer("bubbles", std::mem::size_of::<BubbleU>() as u64),
            way, vols: Vec::new(), cur: 0,
        }
    }

    /// The bubbled volume written last.
    pub fn volume(&self) -> Option<&Volume> {
        self.vols.get(self.cur)
    }

    /// A volume of this size to write into: one kept from before, or a new one (of the big ones only
    /// the newest is kept, as the shapes' own are).
    fn take(&mut self, res: u32) -> usize {
        if let Some(i) = self.vols.iter().position(|v| v.res == res) {
            self.cur = i;
            return i;
        }
        if res > 384 { self.vols.retain(|v| v.res <= 384); }
        let tex = self.gpu.volume("volume with bubbles", res, res, VOL_FORMAT, self.gpu.storage16);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.vols.push(Volume { tex, view, res });
        self.cur = self.vols.len() - 1;
        self.cur
    }

    /// Write `src` with the bubbles cut into it, and return that volume.
    pub fn carve(&mut self, src: &Volume, b: &Bubbles) -> &Volume {
        let res = src.res;
        let vox = 2.0 * HB as f32 / res as f32;
        // The surface is read through a smooth lookup four voxels wide, so that a bubble's edge must stay
        // a few voxels inside it if the outside of the shape is not to move.
        let u = BubbleU { res: res as f32, hb: HB as f32, cell: b.cell, size: b.size, even: b.even, shape: b.shape, seed: b.seed, margin: 3.5 * vox };
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&u));
        let i = self.take(res);
        let dst = &self.vols[i];
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bubbles") });
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) },
        ];
        match &self.way {
            Way::Computed(pipe, layout) => {
                entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dst.view) });
                let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries });
                let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                c.set_pipeline(pipe);
                c.set_bind_group(0, &g, &[]);
                c.dispatch_workgroups(res / 8, res / 8, res / 4);
            }
            Way::Drawn(pipe, layout) => {
                let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries });
                for z in 0..res {
                    let mut rp = pass(&mut enc, &dst.view, Some(z), true);
                    rp.set_pipeline(pipe);
                    rp.set_bind_group(0, &g, &[]);
                    rp.draw(0..3, z..z + 1);
                }
            }
        }
        self.gpu.queue.submit([enc.finish()]);
        &self.vols[i]
    }
}
