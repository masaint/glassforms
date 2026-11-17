//! Adaptive refining: while a still picture refines, pixels whose mean is already certain are left out of
//! the tracer's passes, and the time saved goes to the pixels still noisy (more passes a frame within the
//! same budget: see `Engine::frame`).
//!
//! - Each pixel keeps its own count of samples and a second mean of its odd-numbered samples only
//!   (adapt.wgsl `fold`, which does the still picture's running mean in place of accum.wgsl's still
//!   branch, weighting each pixel by its own count). How far the two means differ, as a step of the
//!   screen's brightness, says how far the mean may still be off.
//! - Before each pass of the tracer, a pass writes a stencil (adapt.wgsl `mask`): a 2 x 2 block whose
//!   pixels and the ring around them are all certain, with at least `nmin` samples, is marked. Every
//!   tracer's pipeline tests the stencil (`tracer_stencil`), so a marked pixel never runs the tracer at
//!   all, and the pixels that do are packed together. The tracer's text is not touched.
//! - Every block is traced anyway once in `RECHECK` passes (in a scattered order, so no pass is dearer
//!   than another), so that a pixel that looked certain before a rare bright path turned up is found out.
//! - A frame's budget seldom holds a whole number of passes: what is left of it goes to a pass for that
//!   share of the blocks, chosen afresh each time (`share`). Each pixel's own count keeps its mean right.
//! - Anything that starts the picture afresh, or moves it, starts the counts afresh too: the counts are
//!   only carried on from a frame that was itself folded in here.
//!
//! Only for tracers whose frame goes straight into the picture (not the stage for rough glass, which
//! blurs the frame across pixels, and not a denoised tracer, whose noisy part keeps a running mean of its
//! own): those are folded in by accum.wgsl as before and never masked.

use bytemuck::{Pod, Zeroable};

use crate::gpu::*;

const WGSL: &str = include_str!("shaders/adapt.wgsl");
/// The stencil the tracer's passes test.
pub const STENCIL: wgpu::TextureFormat = wgpu::TextureFormat::Stencil8;
/// Every block of pixels is traced at least once in this many passes.
pub const RECHECK: u32 = 8;
/// Samples a pixel must have before its mean may count as certain.
pub const NMIN: f32 = 32.0;
/// How far a pixel's two means may differ and it count as certain: a step of the screen (0..1; 1/255 is
/// one level of an 8-bit screen).
pub const TOL: f32 = 1.0 / 255.0;

/// How every tracer's pipeline treats the stencil: it runs only where the stencil is 0 (left as it is).
pub fn tracer_stencil() -> wgpu::DepthStencilState {
    let face = wgpu::StencilFaceState { compare: wgpu::CompareFunction::Equal, fail_op: wgpu::StencilOperation::Keep, depth_fail_op: wgpu::StencilOperation::Keep, pass_op: wgpu::StencilOperation::Keep };
    wgpu::DepthStencilState::stencil(STENCIL, wgpu::StencilState { front: face, back: face, read_mask: 0xff, write_mask: 0 })
}

/// A tracer's pipeline: one triangle over these colour targets, run only where the stencil is 0.
pub fn tracer_pipeline(gpu: &Gpu, label: &str, module: &wgpu::ShaderModule, layouts: &[&wgpu::BindGroupLayout], formats: &[wgpu::TextureFormat]) -> wgpu::RenderPipeline {
    let groups: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &groups, immediate_size: 0 });
    let targets: Vec<Option<wgpu::ColorTargetState>> = formats.iter().map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })).collect();
    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        vertex: wgpu::VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(tracer_stencil()),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState { module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &targets }),
        multiview_mask: None,
        cache: None,
    })
}

/// Draw a tracer's one triangle over these targets (the first cleared to `first`, the others to nothing),
/// testing the stencil as the mask pass left it (`masked`) or with every pixel traced.
#[allow(clippy::too_many_arguments)]
pub fn trace_pass(enc: &mut wgpu::CommandEncoder, views: &[&wgpu::TextureView], first: wgpu::Color, stencil: &wgpu::TextureView, masked: bool, pipe: &wgpu::RenderPipeline, group: &wgpu::BindGroup, second: Option<&wgpu::BindGroup>) {
    let atts: Vec<Option<wgpu::RenderPassColorAttachment>> = views
        .iter()
        .enumerate()
        .map(|(i, &view)| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(if i == 0 { first } else { wgpu::Color::TRANSPARENT }), store: wgpu::StoreOp::Store } }))
        .collect();
    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("trace"), color_attachments: &atts, depth_stencil_attachment: Some(attachment(stencil, masked)), timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
    rp.set_pipeline(pipe);
    rp.set_bind_group(0, group, &[]);
    if let Some(g) = second { rp.set_bind_group(1, g, &[]); }
    rp.draw(0..3, 0..1);
}

/// The stencil attachment for a tracer's pass: kept as the mask pass wrote it (`masked`), or cleared, so
/// that every pixel is traced.
pub fn attachment(view: &wgpu::TextureView, masked: bool) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: None,
        stencil_ops: Some(wgpu::Operations { load: if masked { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(0) }, store: wgpu::StoreOp::Discard }),
    }
}

/// Adaptive refining is on unless GLASSFORMS_ADAPT=off (for comparing; `adapt=0` on the command line too).
pub fn enabled() -> bool {
    std::env::var("GLASSFORMS_ADAPT").map_or(true, |v| v != "off")
}

/// The colour a masked pass clears the tracer's frame to: alpha below zero says "not traced" (the tracer
/// writes a distance of 0 or more).
pub const NOT_TRACED: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: -1.0 };

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AU {
    res: [f32; 2],
    tol: f32,
    nmin: f32,
    expo: f32,
    base: f32,
    pass: f32,
    recheck: f32,
    share: f32,
    pad: [f32; 3],
}

struct Sized {
    w: u32,
    h: u32,
    stats: wgpu::Buffer,
    err: wgpu::Buffer,
    stencil: wgpu::TextureView,
    /// fold: reading accum[i] and the frame, writing the other picture
    g_fold: [wgpu::BindGroup; 2],
    g_mask: wgpu::BindGroup,
}

pub struct Adapt {
    u: wgpu::Buffer,
    l_fold: wgpu::BindGroupLayout,
    l_mask: wgpu::BindGroupLayout,
    p_fold: wgpu::ComputePipeline,
    p_mask: wgpu::RenderPipeline,
    s: Option<Sized>,
    /// leave out pixels whose mean is certain (set by the engine for each frame: only while refining)
    pub on: bool,
    pub tol: f32,
    pub nmin: f32,
    /// the counts hold this picture's: the last frame was folded in here
    pub valid: bool,
    /// passes folded in since the counts started
    pass: u32,
    /// the share of the blocks of pixels the next pass is for (set by the engine: 1 = all of them; less,
    /// for a pass that fills what is left of a frame's budget)
    pub share: f32,
}

impl Adapt {
    pub fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let f = wgpu::ShaderStages::FRAGMENT;
        let uniform = |binding, visibility| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let tex = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None };
        let buf = |binding, visibility, read_only| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let l_fold = gpu.bind_layout("adapt: fold", &[
            uniform(0, c), tex(1), tex(2),
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: crate::render::ACC, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
            buf(4, c, false), buf(5, c, false),
        ]);
        let l_mask = gpu.bind_layout("adapt: mask", &[uniform(0, f), buf(6, f, true)]);
        let module = gpu.shader("adapt", WGSL, false);
        let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("adapt: fold"), bind_group_layouts: &[Some(&l_fold)], immediate_size: 0 });
        let p_fold = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("adapt: fold"), layout: Some(&pl), module: &module, entry_point: Some("fold"), compilation_options: Default::default(), cache: None });
        let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("adapt: mask"), bind_group_layouts: &[Some(&l_mask)], immediate_size: 0 });
        let write = wgpu::StencilFaceState { compare: wgpu::CompareFunction::Always, fail_op: wgpu::StencilOperation::Keep, depth_fail_op: wgpu::StencilOperation::Keep, pass_op: wgpu::StencilOperation::Replace };
        let p_mask = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("adapt: mask"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState::stencil(STENCIL, wgpu::StencilState { front: write, back: write, read_mask: 0xff, write_mask: 0xff })),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("mask"), compilation_options: Default::default(), targets: &[] }),
            multiview_mask: None,
            cache: None,
        });
        Self { u: gpu.uniform_buffer("adapt", std::mem::size_of::<AU>() as u64), l_fold, l_mask, p_fold, p_mask, s: None, on: false, tol: TOL, nmin: NMIN, valid: false, pass: 0, share: 1.0 }
    }

    /// Make the counts and the stencil fit pictures of this size (`accum`: the two pictures so far, which
    /// must be storage-capable; `frame`: the tracer's frame).
    pub fn resize(&mut self, gpu: &Gpu, w: u32, h: u32, accum: [&wgpu::TextureView; 2], frame: &wgpu::TextureView) {
        let n = (w as u64) * (h as u64);
        let buffer = |label, size| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let stats = buffer("adapt: counts", n * 16);
        let err = buffer("adapt: uncertainty", n * 4);
        let stencil = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("adapt: stencil"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: STENCIL,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let g_fold = [0, 1].map(|i| gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adapt: fold"),
            layout: &self.l_fold,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(accum[i]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(frame) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(accum[1 - i]) },
                wgpu::BindGroupEntry { binding: 4, resource: stats.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: err.as_entire_binding() },
            ],
        }));
        let g_mask = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adapt: mask"),
            layout: &self.l_mask,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() }, wgpu::BindGroupEntry { binding: 6, resource: err.as_entire_binding() }],
        });
        self.s = Some(Sized { w, h, stats, err, stencil: stencil.create_view(&wgpu::TextureViewDescriptor::default()), g_fold, g_mask });
        self.valid = false;
    }

    /// The stencil the tracer's passes test.
    pub fn stencil(&self) -> &wgpu::TextureView {
        &self.s.as_ref().expect("the counts are sized with the pictures").stencil
    }

    /// Set this frame's numbers and, when pixels are to be left out of it, write the stencil that says
    /// which (those whose mean is certain, and, for a pass for a share of them, the blocks not chosen). `base`: the count every pixel starts from when the counts start afresh (None: they carry
    /// on). Returns whether the tracer's pass is masked.
    pub fn begin(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, base: Option<u32>, expo: f32) -> bool {
        let Some(s) = &self.s else { return false };
        if base.is_some() { self.pass = 0; }
        let masked = self.on && base.is_none();
        let au = AU { res: [s.w as f32, s.h as f32], tol: self.tol, nmin: self.nmin, expo, base: base.map_or(-1.0, |b| b as f32), pass: self.pass as f32, recheck: RECHECK as f32, share: if masked { self.share.clamp(0.0, 1.0) } else { 1.0 }, pad: [0.0; 3] };
        gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&au));
        if masked {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("adapt: mask"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &s.stencil,
                    depth_ops: None,
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Store }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.p_mask);
            rp.set_bind_group(0, &s.g_mask, &[]);
            rp.set_stencil_reference(1);
            rp.draw(0..3, 0..1);
        }
        masked
    }

    /// Each pixel's count and how uncertain its mean is (as `fold` last left them), top row first, for
    /// looking at (`adaptShow` on the command line). Waits for the GPU.
    pub fn read(&self, gpu: &Gpu) -> Option<(u32, u32, Vec<[f32; 4]>, Vec<f32>)> {
        let s = self.s.as_ref()?;
        let get = |b: &wgpu::Buffer| -> Option<Vec<u8>> {
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("adapt: read"), size: b.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(b, 0, &staging, 0, b.size());
            gpu.queue.submit([enc.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r.is_ok()); });
            gpu.wait();
            if !rx.recv().ok()? { return None; }
            let v = staging.slice(..).get_mapped_range().ok()?.to_vec();
            Some(v)
        };
        let stats = get(&s.stats)?;
        let err = get(&s.err)?;
        Some((s.w, s.h, bytemuck::cast_slice::<u8, [f32; 4]>(&stats).to_vec(), bytemuck::cast_slice::<u8, f32>(&err).to_vec()))
    }

    /// Fold the traced frame into the picture so far: read picture `idx`, write the other.
    pub fn fold(&mut self, enc: &mut wgpu::CommandEncoder, idx: usize) {
        let Some(s) = &self.s else { return };
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("adapt: fold"), timestamp_writes: None });
        c.set_pipeline(&self.p_fold);
        c.set_bind_group(0, &s.g_fold[idx], &[]);
        c.dispatch_workgroups(s.w.div_ceil(8), s.h.div_ceil(8), 1);
        drop(c);
        self.valid = true;
        self.pass += 1;
    }
}
