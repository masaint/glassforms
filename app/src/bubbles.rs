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
use crate::combine::bodies_format;

const BUBBLES_WGSL: &str = concat!(include_str!("shaders/pbr/hash.wgsl"), include_str!("shaders/bubbles.wgsl"));

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
    /// for several bodies: the numbers, the layout and (made when first wanted) the passes for a bodies'
    /// volume of two and of four channels, and the bodies' volume and the scene's last written
    bu: wgpu::Buffer,
    body_layout: wgpu::BindGroupLayout,
    body_pipes: Vec<(u32, wgpu::RenderPipeline)>,
    body_vols: Option<(Volume, Volume, u32)>,
    /// the volumes cut span [-hb, hb] (larger than ever when the layers are placed far out)
    pub hb: f64,
}

/// The numbers of the pass that cuts each body's bubbles into the bodies' volume (`BB` in bubbles.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BodiesU {
    bub: [BubbleU; 4],
    on: [f32; 4],
    count: [f32; 4],
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
            way, vols: Vec::new(), cur: 0, hb: HB,
            bu: gpu.uniform_buffer("bubbles in each body", std::mem::size_of::<BodiesU>() as u64),
            body_layout: {
                let f = wgpu::ShaderStages::FRAGMENT;
                let tex = wgpu::BindGroupLayoutEntry { binding: 1, visibility: f, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None };
                let smp = wgpu::BindGroupLayoutEntry { binding: 2, visibility: f, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
                gpu.bind_layout("bubbles in each body", &[tex, smp, uniform_entry(4, f, false)])
            },
            body_pipes: Vec::new(),
            body_vols: None,
        }
    }

    /// The pass that writes a bodies' volume of this many channels (2 or 4) and the scene's volume beside it.
    fn body_pipe(&mut self, channels: u32) -> usize {
        if let Some(i) = self.body_pipes.iter().position(|p| p.0 == channels) { return i; }
        let drawn = BUBBLES_WGSL.split("//#compute").next().unwrap();
        let module = self.gpu.shader("bubbles in each body", drawn, false);
        let layout = self.gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("bubbles in each body"), bind_group_layouts: &[Some(&self.body_layout)], immediate_size: 0 });
        let target = |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let pipe = self.gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bubbles in each body"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs_bodies"), compilation_options: Default::default(), targets: &[target(bodies_format(channels)), target(VOL_FORMAT)] }),
            multiview_mask: None,
            cache: None,
        });
        self.body_pipes.push((channels, pipe));
        self.body_pipes.len() - 1
    }

    /// Cut each body's bubbles (`per[k]`: body k's, or None) into its own channel of the bodies' volume
    /// `src` (`channels` of them, `count` bodies), and make the scene's volume afresh from the result.
    /// Returns (the bodies' volume, the scene's volume), both kept until the next call: as much video
    /// memory again as the two volumes the scene was built into.
    pub fn carve_bodies(&mut self, src: &Volume, channels: u32, count: u32, per: &[Option<Bubbles>]) -> (&Volume, &Volume) {
        let res = src.res;
        let vox = 2.0 * self.hb as f32 / res as f32;
        let mut u = BodiesU { bub: [BubbleU::zeroed(); 4], on: [0.0; 4], count: [count as f32, 0.0, 0.0, 0.0] };
        for (k, b) in per.iter().enumerate().take(4) {
            if let Some(b) = b {
                u.bub[k] = BubbleU { res: res as f32, hb: self.hb as f32, cell: b.cell, size: b.size, even: b.even, shape: b.shape, seed: b.seed, margin: 3.5 * vox };
                u.on[k] = 1.0;
            }
        }
        self.gpu.queue.write_buffer(&self.bu, 0, bytemuck::bytes_of(&u));
        let pi = self.body_pipe(channels);
        if self.body_vols.as_ref().is_none_or(|v| v.0.res != res || v.2 != channels) {
            // (the old pair goes before the new is made, so that the two are never held at once)
            self.body_vols = None;
            let make = |label, format| {
                let tex = self.gpu.volume(label, res, res, format, false);
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                Volume { tex, view, res }
            };
            self.body_vols = Some((make("bodies with bubbles", bodies_format(channels)), make("scene with each body's bubbles", VOL_FORMAT), channels));
        }
        let (bodies, scene, _) = self.body_vols.as_ref().unwrap();
        let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.body_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) },
                wgpu::BindGroupEntry { binding: 4, resource: self.bu.as_entire_binding() },
            ],
        });
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bubbles in each body") });
        let attach = |view, z| Some(wgpu::RenderPassColorAttachment { view, depth_slice: Some(z), resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
        for z in 0..res {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[attach(&bodies.view, z), attach(&scene.view, z)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.body_pipes[pi].1);
            rp.set_bind_group(0, &g, &[]);
            rp.draw(0..3, z..z + 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        let (bodies, scene, _) = self.body_vols.as_ref().unwrap();
        (bodies, scene)
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
        let vox = 2.0 * self.hb as f32 / res as f32;
        // The surface is read through a smooth lookup four voxels wide, so that a bubble's edge must stay
        // a few voxels inside it if the outside of the shape is not to move.
        let u = BubbleU { res: res as f32, hb: self.hb as f32, cell: b.cell, size: b.size, even: b.even, shape: b.shape, seed: b.seed, margin: 3.5 * vox };
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
