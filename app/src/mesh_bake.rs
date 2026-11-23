//! Polygon models into distance volumes.
//!
//! A model file becomes the same kind of volume every other shape is, in three steps:
//!
//!   1. on another thread (`work`): the file is read and examined (`gf_core::mesh`), and everything a
//!      volume needs that only the CPU gives exactly is prepared (`gf_core::mesh::solid`): which voxels
//!      are inside, which triangles are the solid's outside, the distance far from the surface;
//!   2. the model's base volume is built on the GPU (`MeshBaker::base`, shaders/bake_mesh.wgsl): the
//!      exact distance to the triangles within a few voxels of the surface, gathered a slab of slices at
//!      a time on one sheet as tubes are, the coarse grid beyond, the CPU's word for which side;
//!   3. the volume the tracer reads is made from the base by the model's sliders (`MeshBaker::finish`,
//!      shaders/bake_mesh_finish.wgsl): size, smoothing, a shell's wall, inflation. None of them reads
//!      the file or the triangles again, so they are quick enough to animate.
//!
//! With nothing resized or smoothed, step 2 writes the tracer's volume directly and there is no base.
//! Otherwise the base is kept while sliders move (and through a video), and given back after a full build.
//!
//! `Models` is the engine's side: the files read so far, what has been prepared, what is being read on
//! other threads, and what the window is told about each model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use gf_core::mesh::solid::{self, Prep};
use gf_core::mesh::{self, Examined, MeshJob, Report, Treatment};
use gf_core::state::Detail;
use gf_core::HB;

use crate::bake::{BakeStats, Baker, GROUP, VOL_FORMAT};
use crate::engine::Engine;
use crate::gpu::*;
use crate::hdri;

const MESH_WGSL: &str = include_str!("shaders/bake_mesh.wgsl");
const FINISH_WGSL: &str = include_str!("shaders/bake_mesh_finish.wgsl");
/// Bytes between one block of uniforms and the next where several share a buffer.
const STRIDE: u32 = 256;
const MAX_SLABS: u32 = 256;
/// How far a shell's wall or an inflation may move the surface before the volume is made by way of
/// the base volume (which cuts it to the stage) rather than directly.
const DIRECT_REACH: f64 = 0.05;
/// How many prepared models are kept (each with its base volume, if it has one).
const KEPT: usize = 5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshU {
    hb: f32,
    res: f32,
    band: f32,
    lo: f32,
    mres: f32,
    mcell: f32,
    offset: f32,
    pad0: f32,
    slab: u32,
    stiles: u32,
    shell: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FinishU {
    hb: f32,
    res: f32,
    base_res: f32,
    size: f32,
    offset: f32,
    lod: f32,
    soft_lod: f32,
    soft: f32,
    exact: u32,
    pad: [u32; 3],
}

/// A model's base volume: the distances as the model was built, with smaller, softer levels of itself.
struct Base {
    _tex: wgpu::Texture,
    all: wgpu::TextureView,
    levels: Vec<wgpu::TextureView>,
    res: u32,
}

/// What the volume the tracer reads is made from the base with.
struct Finish {
    size: f64,
    offset: f64,
    smooth: f64,
}

/// What writes volumes by compute shader, where that is possible.
struct Computed {
    resolve: wgpu::ComputePipeline,
    l_resolve: wgpu::BindGroupLayout,
    slabs: wgpu::Buffer,
    g_slabs: wgpu::BindGroup,
    finish: wgpu::ComputePipeline,
    smaller: wgpu::ComputePipeline,
    l_finish: wgpu::BindGroupLayout,
}

/// The pipelines that turn a prepared model into volumes.
struct MeshBaker {
    lin: wgpu::Sampler,
    u: wgpu::Buffer,
    g0: wgpu::BindGroup,
    l_tris: wgpu::BindGroupLayout,
    l_resolve: wgpu::BindGroupLayout,
    p_splat: wgpu::RenderPipeline,
    p_resolve: wgpu::RenderPipeline,
    sheet_format: wgpu::TextureFormat,
    tris: GrowBuf,
    pairs: GrowBuf,
    lines: GrowBuf,
    flips: GrowBuf,
    fu: wgpu::Buffer,
    l_finish: wgpu::BindGroupLayout,
    p_finish: wgpu::RenderPipeline,
    p_smaller: wgpu::RenderPipeline,
    computed: Option<Computed>,
}

impl MeshBaker {
    fn new(gpu: &Gpu) -> Self {
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let c = wgpu::ShaderStages::COMPUTE;
        let fr = wgpu::ShaderStages::FRAGMENT;
        let drawn = gpu.shader("bake mesh", MESH_WGSL.split("//#compute").next().unwrap(), false);
        let l0 = gpu.bind_layout("mesh params", &[uniform_entry(0, vf | c, false)]);
        let l_tris = gpu.bind_layout("mesh triangles", &[storage_entry(0, wgpu::ShaderStages::VERTEX)]);
        let l_resolve = gpu.bind_layout("mesh resolve", &[
            texture_entry(1, wgpu::TextureViewDimension::D3, true), sampler_entry(2), texture_entry(3, wgpu::TextureViewDimension::D2, false),
            storage_entry(4, fr), storage_entry(5, fr),
        ]);
        // a pixel of the sheet keeps the least of what is drawn on it: 32 bits where they can be blended into
        let sheet_format = if gpu.blend32 { wgpu::TextureFormat::R32Float } else { wgpu::TextureFormat::R16Float };
        // one instance = one triangle in one slice
        let pair_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 4, shader_location: 1 },
            ],
        })];
        let one = wgpu::BlendFactor::One;
        let least = wgpu::BlendComponent { src_factor: one, dst_factor: one, operation: wgpu::BlendOperation::Min };
        let p_splat = gpu.pipeline(&PipeDesc { label: "mesh splat", module: &drawn, vs: "vs_splat", fs: "fs_splat", layouts: &[&l0, &l_tris], format: sheet_format, blend: Some(wgpu::BlendState { color: least, alpha: least }), strip: true, buffers: &pair_layout });
        let p_resolve = gpu.pipeline(&PipeDesc { label: "mesh resolve", module: &drawn, vs: "vs_full", fs: "fs_resolve", layouts: &[&l0, &l_resolve], format: VOL_FORMAT, blend: None, strip: false, buffers: &[] });
        let u = gpu.uniform_buffer("mesh params", std::mem::size_of::<MeshU>() as u64);
        let g0 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l0, entries: &[wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }] });

        // finishing: one block of uniforms for the volume, one for each smaller level of a base
        let fin_drawn = gpu.shader("finish mesh", FINISH_WGSL.split("//#compute").next().unwrap(), false);
        let l_finish = gpu.bind_layout("mesh finish", &[uniform_entry(0, vf, true), texture_entry(1, wgpu::TextureViewDimension::D3, true), sampler_entry(2)]);
        let p_finish = gpu.pipeline(&PipeDesc { label: "mesh finish", module: &fin_drawn, vs: "vs_full", fs: "fs_finish", layouts: &[&l_finish], format: VOL_FORMAT, blend: None, strip: false, buffers: &[] });
        let p_smaller = gpu.pipeline(&PipeDesc { label: "mesh levels", module: &fin_drawn, vs: "vs_full", fs: "fs_smaller", layouts: &[&l_finish], format: VOL_FORMAT, blend: None, strip: false, buffers: &[] });
        let fu = gpu.uniform_buffer("mesh finish", (STRIDE * 16) as u64);

        let computed = gpu.storage16.then(|| {
            let tex = |binding, view_dimension, filterable| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension, multisampled: false }, count: None };
            let smp = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
            let out = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: VOL_FORMAT, view_dimension: wgpu::TextureViewDimension::D3 }, count: None };
            let l_resolve = gpu.bind_layout("mesh resolve (computed)", &[tex(1, wgpu::TextureViewDimension::D3, true), smp(2), tex(3, wgpu::TextureViewDimension::D2, false), storage_entry(4, c), storage_entry(5, c), out(6)]);
            let l_slabs = gpu.bind_layout("mesh slab", &[uniform_entry(0, c, true)]);
            let slabs = gpu.uniform_buffer("mesh slabs", (STRIDE * MAX_SLABS) as u64);
            let g_slabs = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &l_slabs,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &slabs, offset: 0, size: wgpu::BufferSize::new(16) }) }],
            });
            let whole = gpu.shader("bake mesh (computed)", MESH_WGSL, false);
            let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mesh resolve (computed)"), bind_group_layouts: &[Some(&l0), Some(&l_resolve), Some(&l_slabs)], immediate_size: 0 });
            let resolve = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("mesh resolve (computed)"), layout: Some(&layout), module: &whole, entry_point: Some("cs_resolve"), compilation_options: Default::default(), cache: None });
            let l_finish = gpu.bind_layout("mesh finish (computed)", &[uniform_entry(0, c, true), tex(1, wgpu::TextureViewDimension::D3, true), smp(2), out(3)]);
            let fin = gpu.shader("finish mesh (computed)", FINISH_WGSL, false);
            let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mesh finish (computed)"), bind_group_layouts: &[Some(&l_finish)], immediate_size: 0 });
            let pipe = |entry| gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &fin, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
            Computed { resolve, l_resolve, slabs, g_slabs, finish: pipe("cs_finish"), smaller: pipe("cs_smaller"), l_finish }
        });
        let lin = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            lin, u, g0, l_tris, l_resolve, p_splat, p_resolve, sheet_format,
            tris: GrowBuf::new(gpu, "mesh triangles", wgpu::BufferUsages::STORAGE, 1 << 20),
            pairs: GrowBuf::new(gpu, "mesh slices", wgpu::BufferUsages::VERTEX, 1 << 20),
            lines: GrowBuf::new(gpu, "mesh lines", wgpu::BufferUsages::STORAGE, 1 << 16),
            flips: GrowBuf::new(gpu, "mesh flips", wgpu::BufferUsages::STORAGE, 1 << 16),
            fu, l_finish, p_finish, p_smaller, computed,
        }
    }

    /// Build a prepared model's distances into `out`, a volume of `p.res` voxels a side, with `offset`
    /// taken off every one of them. Returns how many triangle-slices were drawn.
    fn base(&mut self, gpu: &Gpu, p: &Prep, out: &wgpu::TextureView, offset: f64) -> R<()> {
        let res = p.res;
        if res % GROUP != 0 { return Err(format!("a volume's size must be a multiple of {GROUP}, not {res}")); }
        // as many slices across the sheet as the largest texture allows, eight at most; a whole number of eights to a slab
        let tiles = (gpu.device.limits().max_texture_dimension_2d / res).min(8);
        let slab = tiles * tiles / GROUP * GROUP;
        if slab == 0 || res.div_ceil(slab.max(1)) > MAX_SLABS {
            return Err(format!("a volume of {res} voxels a side is too large for this graphics device"));
        }
        // the triangles (three corners of four numbers each), the slices each reaches, and the solid's inside
        let corners: Vec<[f32; 4]> = p.tris.iter().flat_map(|t| [[t[0], t[1], t[2], 0.0], [t[3], t[4], t[5], 0.0], [t[6], t[7], t[8], 0.0]]).collect();
        self.tris.write(gpu, bytemuck::cast_slice(&corners));
        drop(corners);
        self.pairs.write(gpu, bytemuck::cast_slice(&p.pairs));
        if p.shell {
            self.lines.write(gpu, bytemuck::cast_slice(&[0u32; 4]));
            self.flips.write(gpu, bytemuck::cast_slice(&[0u32; 4]));
        } else {
            self.lines.write(gpu, bytemuck::cast_slice(&p.start));
            self.flips.write(gpu, bytemuck::cast_slice(if p.flips.is_empty() { &[0u32; 4][..] } else { &p.flips[..] }));
        }
        let m = p.mid_res;
        let mid = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh coarse grid"),
            size: wgpu::Extent3d { width: m, height: m, depth_or_array_layers: m },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &mid, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&p.mid),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(m * 4), rows_per_image: Some(m) },
            wgpu::Extent3d { width: m, height: m, depth_or_array_layers: m },
        );
        let mid_view = mid.create_view(&wgpu::TextureViewDescriptor::default());
        let side = tiles * res;
        let sheet = gpu.target("mesh sheet", side, side, self.sheet_format, &[]).create_view(&wgpu::TextureViewDescriptor::default());
        let u = MeshU {
            hb: HB as f32, res: res as f32, band: p.band as f32, lo: p.lo as f32, mres: m as f32, mcell: (2.0 * HB / m as f64) as f32,
            offset: offset as f32, pad0: 0.0, slab, stiles: tiles, shell: p.shell as u32, pad1: 0,
        };
        gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&u));
        let g_tris = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_tris, entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.tris.buf.as_entire_binding() }] });
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&mid_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sheet) },
            wgpu::BindGroupEntry { binding: 4, resource: self.lines.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: self.flips.buf.as_entire_binding() },
        ];
        let g_resolve = match &self.computed {
            Some(cp) => {
                // each slab's first slice and size, one block per slab
                let mut data = vec![0u8; (STRIDE * res.div_ceil(slab)) as usize];
                for (i, z0) in (0..res).step_by(slab as usize).enumerate() {
                    let at = i * STRIDE as usize;
                    data[at..at + 4].copy_from_slice(&z0.to_le_bytes());
                    data[at + 4..at + 8].copy_from_slice(&slab.min(res - z0).to_le_bytes());
                }
                gpu.queue.write_buffer(&cp.slabs, 0, &data);
                entries.push(wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(out) });
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cp.l_resolve, entries: &entries })
            }
            None => gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_resolve, entries: &entries }),
        };
        // A slab of slices at a time: gather the triangles on the sheet, then write the volume. Each slab is
        // handed over by itself, so that a model with a great deal of surface is many short pieces of work
        // for the graphics device rather than one long one.
        let mut z0 = 0;
        while z0 < res {
            let z1 = (z0 + slab).min(res);
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake mesh") });
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &sheet, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 60000.0, g: 0.0, b: 0.0, a: 0.0 }), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let (a, b) = (p.slices[z0 as usize], p.slices[z1 as usize]);
                if b > a {
                    rp.set_pipeline(&self.p_splat);
                    rp.set_bind_group(0, &self.g0, &[]);
                    rp.set_bind_group(1, &g_tris, &[]);
                    rp.set_vertex_buffer(0, self.pairs.buf.slice(..));
                    rp.draw(0..4, a..b);
                }
            }
            match &self.computed {
                Some(cp) => {
                    let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    c.set_pipeline(&cp.resolve);
                    c.set_bind_group(0, &self.g0, &[]);
                    c.set_bind_group(1, &g_resolve, &[]);
                    c.set_bind_group(2, &cp.g_slabs, &[z0 / slab * STRIDE]);
                    c.dispatch_workgroups(res / 8, res / 8, (z1 - z0).div_ceil(4));
                }
                None => {
                    for z in z0..z1 {
                        let mut rp = pass(&mut enc, out, Some(z), true);
                        rp.set_pipeline(&self.p_resolve);
                        rp.set_bind_group(0, &self.g0, &[]);
                        rp.set_bind_group(1, &g_resolve, &[]);
                        rp.draw(0..3, z..z + 1);
                    }
                }
            }
            gpu.queue.submit([enc.finish()]);
            z0 = z1;
        }
        Ok(())
    }

    /// An empty base volume of `res` voxels a side, with levels each half the one before down to 8 or so
    /// (the coarsest are what the most smoothing reads).
    fn new_base(&self, gpu: &Gpu, res: u32) -> Base {
        let mut n = 1;
        while (res >> n) >= 8 && n < 15 { n += 1; }
        let extra = if self.computed.is_some() { wgpu::TextureUsages::STORAGE_BINDING } else { wgpu::TextureUsages::empty() };
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh base"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            mip_level_count: n,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: VOL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | extra,
            view_formats: &[],
        });
        let all = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let levels = (0..n).map(|l| tex.create_view(&wgpu::TextureViewDescriptor { base_mip_level: l, mip_level_count: Some(1), ..Default::default() })).collect();
        Base { _tex: tex, all, levels, res }
    }

    /// One pass over a whole volume of `res` voxels a side through the finishing shader: reads `from`,
    /// writes `to`, with the block of uniforms at `block`.
    fn sweep(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, smaller: bool, from: &wgpu::TextureView, to: &wgpu::TextureView, res: u32, block: u32) {
        let uniforms = wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.fu, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<FinishU>() as u64) });
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: uniforms },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(from) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) },
        ];
        match &self.computed {
            Some(cp) => {
                entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(to) });
                let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &cp.l_finish, entries: &entries });
                let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                c.set_pipeline(if smaller { &cp.smaller } else { &cp.finish });
                c.set_bind_group(0, &g, &[block * STRIDE]);
                c.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), res.div_ceil(4));
            }
            None => {
                let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_finish, entries: &entries });
                for z in 0..res {
                    let mut rp = pass(enc, to, Some(z), true);
                    rp.set_pipeline(if smaller { &self.p_smaller } else { &self.p_finish });
                    rp.set_bind_group(0, &g, &[block * STRIDE]);
                    rp.draw(0..3, z..z + 1);
                }
            }
        }
    }

    /// Make a base volume's smaller levels from its finest.
    fn levels(&self, gpu: &Gpu, b: &Base) {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mesh levels") });
        for l in 1..b.levels.len() {
            let res = (b.res >> l).max(1);
            let u = FinishU { hb: HB as f32, res: res as f32, base_res: b.res as f32, size: 1.0, offset: 0.0, lod: 0.0, soft_lod: 0.0, soft: 0.0, exact: 0, pad: [0; 3] };
            gpu.queue.write_buffer(&self.fu, (l as u32 * STRIDE) as u64, bytemuck::bytes_of(&u));
            self.sweep(gpu, &mut enc, true, &b.levels[l - 1], &b.levels[l], res, l as u32);
        }
        gpu.queue.submit([enc.finish()]);
    }

    /// The volume the tracer reads (`res` voxels a side, written through `out`) from a base volume.
    fn finish(&self, gpu: &Gpu, b: &Base, out: &wgpu::TextureView, res: u32, f: &Finish) {
        let voxel = 2.0 * HB / b.res as f64;
        // the level whose voxels are the size of the volume's, when the volume is the coarser
        let lod = (b.res as f64 / res as f64).log2().max(0.0);
        // A level read smoothly softens about as a bell 0.6 of its voxels wide: the level for the softening asked for.
        let (soft, soft_lod) = if f.smooth <= 0.0 {
            (0.0, 0.0)
        } else {
            let l = (f.smooth / (0.6 * voxel)).log2();
            if l < 0.0 { (f.smooth / (0.6 * voxel), 0.0) } else { (1.0, l.max(lod).min(b.levels.len() as f64 - 2.0).max(0.0)) }
        };
        let exact = f.size == 1.0 && f.smooth <= 0.0 && res == b.res;
        let u = FinishU {
            hb: HB as f32, res: res as f32, base_res: b.res as f32, size: f.size as f32, offset: f.offset as f32, lod: lod as f32,
            soft_lod: soft_lod as f32, soft: soft as f32, exact: exact as u32, pad: [0; 3],
        };
        gpu.queue.write_buffer(&self.fu, 0, bytemuck::bytes_of(&u));
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("finish mesh") });
        self.sweep(gpu, &mut enc, false, &b.all, out, res, 0);
        gpu.queue.submit([enc.finish()]);
    }
}

// ---- the engine's side ---------------------------------------------------------------------------------

/// A model file read and examined.
pub struct Loaded {
    appearance_notice: Option<String>,
    /// Canonical owners retained beside their renderer projection and packed decoded pixels.
    pub canonical: Option<Arc<gf_core::pbr::CanonicalImport>>,
    pub surface: Option<Arc<Vec<u32>>>,
    pub ex: Examined,
    /// the file's name, its size, and how long reading and examining took
    pub file: String,
    pub bytes: u64,
    pub read_ms: f64,
    pub examine_ms: f64,
}

/// What a prepared model is prepared for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    path: PathBuf,
    up_z: bool,
    shell: bool,
    res: u32,
}

struct Entry {
    prep: Arc<Prep>,
    prep_ms: f64,
    base: Option<Base>,
    /// when it was last used (a count of builds)
    used: u64,
}

/// A model being read or prepared on another thread: for which file and how, and where its result arrives.
struct Running {
    path: PathBuf,
    up_z: bool,
    mode: u8,
    res: u32,
    rx: Receiver<R<Done>>,
}
struct Done {
    loaded: Arc<Loaded>,
    key: Key,
    prep: Prep,
    prep_ms: f64,
}

/// What the window is told about a layer's model.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    /// the file's name
    pub file: String,
    /// the examiner's result in a line: "closed", "3 small gaps, closed automatically", ...
    pub line: String,
    /// drawn as a shell (by the examiner's finding or by choice)
    pub shell: bool,
    /// what the examiner found it to be, whatever it is drawn as
    pub treatment: Treatment,
    pub report: Report,
    /// triangles the distances are measured to (the solid's outside, or the whole surface of a shell)
    pub triangles: usize,
    /// parts of triangles left out because they are inside the solid; voxel lines along which the closed
    /// surface was not closed after all (none, unless something is wrong); voxels of exact distance
    /// either side of the surface
    pub hidden: usize,
    pub unclosed: usize,
    pub band: f64,
    /// how far its detail was melted in the last build (`report.smooth` is what its own facets call for)
    pub smooth: f64,
    /// milliseconds: reading the file, examining it, preparing it for its volume
    pub read_ms: f64,
    pub examine_ms: f64,
    pub prepare_ms: f64,
    /// milliseconds the graphics device took over the last build (to the end only when stills are made;
    /// in the window nothing waits for it)
    pub gpu_ms: f64,
    /// the volume was written directly, with no base volume kept
    pub direct: bool,
}

/// Is a model drawn as a shell? As it is asked for, or as the examiner found it.
fn as_shell(mode: u8, report: &Report) -> bool {
    match mode {
        1 => false,
        2 => true,
        _ => report.treatment == Treatment::Shell,
    }
}

fn load(path: &Path) -> R<Loaded> {
    let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let t = Instant::now();
    let asset = mesh::read::read_asset(path)?;
    let read = asset.mesh;
    let read_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let ex = mesh::examine(&read).map_err(|e| format!("{file}: {e}"))?;
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut canonical = None;
    let surface = asset.appearance.map(|a| {
        let fit = mesh::solid::Fit::of(&ex.surface, false);
        let source_id = path.to_string_lossy();
        let (import, words) = a.canonical_shader_words(&source_id, &read, &fit)
            .map_err(|error| format!("{file}: {error}"))?;
        canonical = Some(Arc::new(import));
        Ok::<_, String>(Arc::new(words))
    }).transpose()?;
    Ok(Loaded { ex, canonical, surface, appearance_notice: asset.appearance_notice, file, bytes, read_ms, examine_ms: t.elapsed().as_secs_f64() * 1e3 })
}

/// Everything that needs no GPU: read and examine the file (unless that has been done), and prepare
/// it for a volume of `res` voxels a side.
fn work(path: &Path, have: Option<Arc<Loaded>>, up_z: bool, mode: u8, res: u32) -> R<Done> {
    let loaded = match have {
        Some(l) => l,
        None => Arc::new(load(path)?),
    };
    let shell = as_shell(mode, &loaded.ex.report);
    let t = Instant::now();
    let prep = solid::prepare(&loaded.ex, up_z, shell, res).map_err(|e| format!("{}: {e}", loaded.file))?;
    Ok(Done { loaded, key: Key { path: path.to_path_buf(), up_z, shell, res }, prep, prep_ms: t.elapsed().as_secs_f64() * 1e3 })
}

pub struct Models {
    /// read files there and then instead of on another thread (stills, videos, timings)
    pub wait: bool,
    baker: Option<MeshBaker>,
    loaded: HashMap<PathBuf, Arc<Loaded>>,
    entries: Vec<(Key, Entry)>,
    running: Vec<Running>,
    /// files that could not be read, and why (asked for again only when the user loads them again)
    failed: HashMap<PathBuf, String>,
    clock: u64,
    /// the last build stopped because its model is still being read
    pending: bool,
    /// a file the window asked for, as given, and whether as a new layer: put up once it has been read
    /// without error
    asked: Option<(String, bool)>,
    /// why the last file the window asked for could not be used
    pub error: Option<String>,
    info: Vec<Option<ModelInfo>>,
    /// each layer's model path as it was last seen and tidied
    tidied: Vec<String>,
}

impl Models {
    pub fn new(wait: bool) -> Self {
        Self { wait, baker: None, loaded: HashMap::new(), entries: Vec::new(), running: Vec::new(), failed: HashMap::new(), clock: 0, pending: false, asked: None, error: None, info: Vec::new(), tidied: Vec::new() }
    }

    /// The last build stopped because its model is still being read: it is not an error.
    pub fn waiting(&self) -> bool { self.pending }
    /// Something is being read, or the window is waiting for a file.
    pub fn busy(&self) -> bool { !self.running.is_empty() || self.asked.is_some() }

    /// Start reading and preparing on another thread, unless that very thing is on its way already.
    fn start(&mut self, path: &Path, up_z: bool, mode: u8, res: u32) {
        if self.running.iter().any(|r| r.path == path && r.up_z == up_z && r.mode == mode && r.res == res) { return; }
        let (tx, rx) = channel();
        let have = self.loaded.get(path).cloned();
        let p = path.to_path_buf();
        let spawned = std::thread::Builder::new().name("model".into()).spawn(move || {
            let _ = tx.send(work(&p, have, up_z, mode, res));
        });
        if let Err(e) = spawned {
            self.failed.insert(path.to_path_buf(), format!("could not start reading the model: {e}"));
            return;
        }
        self.running.push(Running { path: path.to_path_buf(), up_z, mode, res, rx });
    }

    fn keep(&mut self, d: Done) {
        self.loaded.insert(d.key.path.clone(), d.loaded);
        self.entries.retain(|e| e.0 != d.key);
        self.entries.push((d.key, Entry { prep: Arc::new(d.prep), prep_ms: d.prep_ms, base: None, used: self.clock }));
    }

    /// Take in whatever the other threads have finished. True if anything arrived.
    fn poll(&mut self) -> bool {
        let mut arrived = false;
        let mut i = 0;
        while i < self.running.len() {
            let r = match self.running[i].rx.try_recv() {
                Ok(r) => r,
                Err(TryRecvError::Empty) => { i += 1; continue; }
                Err(TryRecvError::Disconnected) => Err("reading the model stopped unexpectedly".into()),
            };
            let run = self.running.swap_remove(i);
            match r {
                Ok(d) => self.keep(d),
                Err(e) => { self.failed.insert(run.path, e); }
            }
            arrived = true;
        }
        arrived
    }

    /// Read and examine a file here and now, unless that has been done (or tried and failed).
    fn read_now(&mut self, path: &Path) {
        if self.loaded.contains_key(path) || self.failed.contains_key(path) { return; }
        match load(path) {
            Ok(l) => { self.loaded.insert(path.to_path_buf(), Arc::new(l)); }
            Err(e) => { self.failed.insert(path.to_path_buf(), e); }
        }
    }

    /// Forget everything about a file, so that it is read afresh the next time it is asked for.
    pub fn forget(&mut self, path: &Path) {
        self.loaded.remove(path);
        self.failed.remove(path);
        self.entries.retain(|e| e.0.path != path);
    }

    /// Build a layer's model into that layer's volume, `res` voxels a side. If the file has not been read
    /// yet it is read on another thread and this returns an error for now (`waiting()` says it is that).
    pub fn build(&mut self, gpu: &Gpu, volumes: &mut Baker, layer: usize, job: &MeshJob, res: u32, mode: Detail) -> R<BakeStats> {
        self.pending = false;
        if job.file.trim().is_empty() { return Err("no model file has been chosen: load one, or drop one on the window".into()); }
        let path = hdri::resolve(&job.file);
        if let Some(e) = self.failed.get(&path) { return Err(e.clone()); }
        let full = job.full_res.max(GROUP) / GROUP * GROUP;
        let ready = self.loaded.get(&path).and_then(|l| {
            let key = Key { path: path.clone(), up_z: job.up_z, shell: as_shell(job.mode, &l.ex.report), res: full };
            self.entries.iter().position(|e| e.0 == key)
        });
        let at = match ready {
            Some(i) => i,
            None if self.wait => {
                let d = work(&path, self.loaded.get(&path).cloned(), job.up_z, job.mode, full).inspect_err(|e| { self.failed.insert(path.clone(), e.clone()); })?;
                self.keep(d);
                self.entries.len() - 1
            }
            None => {
                self.start(&path, job.up_z, job.mode, full);
                if let Some(e) = self.failed.get(&path) { return Err(e.clone()); }
                self.pending = true;
                return Err(format!("reading {}", path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
            }
        };
        self.clock += 1;
        // (smoothing that has not been set: what the model's own facets call for)
        let smooth = if job.smooth < 0.0 { self.loaded.get(&path).map_or(0.0, |l| l.ex.report.smooth) } else { job.smooth };
        if self.baker.is_none() { self.baker = Some(MeshBaker::new(gpu)); }
        let baker = self.baker.as_mut().expect("just made");
        let (key, e) = &mut self.entries[at];
        e.used = self.clock;
        let p = e.prep.clone();
        // a shell's wall is never thinner than the volume can show
        let voxel = 2.0 * HB / full as f64;
        let offset = job.inflate + if key.shell { 0.5 * job.wall.max(2.5 * voxel) } else { 0.0 };
        let plain = job.size == 1.0 && smooth <= 0.0 && res == full;
        // While a slider moves or a video is made the base volume is wanted again at once and is kept;
        // a full build is the last for a while, and gives the base's memory back.
        let keep = mode.keeps_model_base();
        // (GLASSFORMS_MODEL_BASE=1 goes by way of the base volume always, to check the two ways against each other)
        let direct = plain && offset.abs() <= DIRECT_REACH && (e.base.is_none() || !keep) && std::env::var_os("GLASSFORMS_MODEL_BASE").is_none();
        let (out, _) = volumes.target(res)?;
        let t = Instant::now();
        if direct {
            baker.base(gpu, &p, &out, offset)?;
        } else {
            if e.base.is_none() {
                let b = baker.new_base(gpu, full);
                baker.base(gpu, &p, &b.levels[0], 0.0)?;
                baker.levels(gpu, &b);
                e.base = Some(b);
            }
            baker.finish(gpu, e.base.as_ref().expect("just made"), &out, res, &Finish { size: job.size, offset, smooth });
        }
        if !keep { e.base = None; }
        // (timed to the end only where nothing is waiting on the picture)
        if self.wait { gpu.wait(); }
        let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
        // what the window is told
        let l = &self.loaded[&path];
        let info = ModelInfo {
            file: l.file.clone(), line: l.appearance_notice.as_ref().map_or_else(|| l.ex.report.line.clone(), |why| format!("{}. Source materials could not be loaded. {why}", l.ex.report.line)), shell: key.shell, treatment: l.ex.report.treatment, report: l.ex.report.clone(), triangles: p.tris.len(),
            hidden: p.hidden, unclosed: p.unclosed, band: p.band / voxel, smooth,
            read_ms: l.read_ms, examine_ms: l.examine_ms, prepare_ms: e.prep_ms, gpu_ms, direct,
        };
        if self.info.len() <= layer { self.info.resize(layer + 1, None); }
        self.info[layer] = Some(info);
        // only the models used of late are kept
        if self.entries.len() > KEPT {
            self.entries.sort_by_key(|e| std::cmp::Reverse(e.1.used));
            self.entries.truncate(KEPT);
        }
        Ok(BakeStats { segments: p.tris.len(), splats: p.pairs.len() })
    }
}

impl Engine {
    /// Keep the models in step with the state, before anything is built: take in what other threads have
    /// finished reading (and ask for the shape to be built again if something arrived); put up a file
    /// the window asked for (`load_model`) if it read without error, or keep why it did not; keep each
    /// chosen file's path the way a setup should; and put on every model's `smooth` slider that has not
    /// been set the amount its own facets call for, once its file has been examined.
    pub(crate) fn model_upkeep(&mut self) {
        self.models.pending = false;
        let arrived = self.models.poll();
        if let Some((given, new_layer)) = self.models.asked.take() {
            let path = hdri::resolve(&given);
            if let Some(e) = self.models.failed.get(&path) {
                self.models.error = Some(e.clone());
            } else if self.models.loaded.contains_key(&path) {
                self.models.error = None;
                self.put_up(&given, new_layer);
            } else {
                self.models.asked = Some((given, new_layer));
            }
        }
        if arrived { self.state.request_if_idle(gf_core::build::BuildRequest::model_ready()); }
        for layer in 0..self.state.layer_count() {
            let Some(file) = self.state.model_file(layer).filter(|f| !f.trim().is_empty()) else { continue };
            // A file the user chose is kept relative to the program's folder when it is in it. (Looked
            // at only when a layer's path is one not seen before.)
            if self.state.layer_shape(layer).is_some_and(|st| !st.model.is_empty()) && self.models.tidied.get(layer).is_none_or(|t| *t != file) {
                let tidy = hdri::tidy(&file);
                if self.models.tidied.len() <= layer { self.models.tidied.resize(layer + 1, String::new()); }
                self.models.tidied[layer] = tidy.clone();
                if let Some(st) = self.state.layer_shape_mut(layer) { st.model = tidy; }
            }
            // smoothing not set yet: what the mesh calls for, as soon as that is known
            let Some(i) = self.state.layer_preset(layer).param_index("smooth") else { continue };
            if !self.state.layer_shape(layer).is_some_and(|st| st.vals.get(i).is_some_and(|v| *v < 0.0)) { continue; }
            let path = hdri::resolve(&file);
            if self.models.wait { self.models.read_now(&path); }
            let Some(own) = self.models.loaded.get(&path).map(|l| l.ex.report.smooth) else { continue };
            if let Some(v) = self.state.layer_shape_mut(layer).and_then(|st| st.vals.get_mut(i)) { *v = own; }
        }
    }

    /// Make a file that has been read the shape of the selected layer, or of a new layer if one was
    /// asked for and there is room (with four layers already, the selected layer's shape is replaced).
    fn put_up(&mut self, given: &str, new_layer: bool) {
        if new_layer {
            if let Some((fam, preset)) = gf_core::presets::models::places(&self.state.families) {
                self.state.add_layer_of(fam, preset);
            }
        }
        self.state.set_model(&hdri::tidy(given));
    }

    /// Show a model file as the selected layer's shape (as choosing a shape does). The file is read and
    /// examined on another thread while the picture goes on as it is; the shape changes when it is
    /// ready. A file that cannot be read changes nothing: `model_error()` then says why. `path`: a full
    /// path, or one below the program's folder.
    pub fn load_model(&mut self, path: &str) {
        self.load_model_as(path, false);
    }

    /// The same, with the choice of adding the model as a new layer (if there are fewer than four)
    /// instead of replacing the selected layer's shape.
    pub fn load_model_as(&mut self, path: &str, new_layer: bool) {
        let full = hdri::resolve(path);
        self.models.forget(&full);
        self.models.error = None;
        let res = self.state.shape().res;
        if self.models.wait {
            match work(&full, None, false, 0, res) {
                Ok(d) => {
                    self.models.keep(d);
                    self.put_up(path, new_layer);
                }
                Err(e) => self.models.error = Some(e),
            }
            return;
        }
        self.models.start(&full, false, 0, res);
        self.models.asked = Some((path.to_string(), new_layer));
    }

    /// File-authored surface data and placements for the visible model instances.
    pub(crate) fn imported_surfaces(&self) -> Vec<crate::model_surface::Instance> {
        if self.models.loaded.values().all(|model| model.surface.is_none()) { return Vec::new(); }
        let mut surfaces = Vec::new();
        let edited = self.state.edit_scope.object();
        let world = edited.is_none().then(|| self.state.stack.world_plan());
        for (layer, source) in self.state.stack.layers.iter().enumerate() {
            // Object Mode presents the source at its authored placement, even if hidden in World Mode.
            // Keep the triangle lookup in the same scope as the displayed distance field.
            let placements: &[_] = match &world {
                Some(plan) if source.show => &plan.placements[layer],
                None if edited == Some(layer) => std::slice::from_ref(&source.place),
                _ => continue,
            };
            let Some(file) = self.state.model_file(layer).filter(|p| !p.is_empty()) else { continue };
            let Some(data) = self.models.loaded.get(&hdri::resolve(&file)).and_then(|m| m.surface.as_ref()) else { continue };
            let Some(shape) = self.state.layer_shape(layer) else { continue };
            let preset = self.state.layer_preset(layer);
            if preset.param_index("fileSurface").and_then(|i|shape.vals.get(i)).copied().unwrap_or(1.0) < 0.5 { continue; }
            let job = gf_core::presets::models::job(preset, shape, &file);
            for &place in placements {
                let reach = 0.012 + job.smooth.max(0.0) + job.inflate.abs() + if job.mode == 2 { job.wall } else { 0.0 };
                surfaces.push(crate::model_surface::Instance {
                    data: data.clone(), place, up_z: job.up_z, size: job.size, reach,
                });
            }
        }
        surfaces
    }

    /// Why the last file given to `load_model` could not be used (None once one loads).
    pub fn model_error(&self) -> Option<&str> {
        self.models.error.as_deref()
    }
    /// A model is being read: the shape will change when it is ready.
    pub fn model_loading(&self) -> bool {
        self.models.busy()
    }
    /// What there is to say about a layer's model: its file, the examiner's line, whether it is drawn
    /// as a shell. None if the layer's shape is not a model, or its model has not been built yet.
    pub fn model_info(&self, layer: usize) -> Option<&ModelInfo> {
        if layer >= self.state.layer_count() || self.state.model_file(layer).is_none() { return None; }
        self.models.info.get(layer)?.as_ref()
    }

    /// The models of the scene as a JSON member to add to a report (nothing when no layer is a model).
    pub fn models_json(&self) -> String {
        let rows: Vec<String> = (0..self.state.layer_count())
            .filter_map(|i| {
                let m = self.model_info(i)?;
                Some(format!(
                    "{{\"layer\":{},\"file\":\"{}\",\"line\":\"{}\",\"shell\":{},\"triangles\":{},\"hidden\":{},\"unclosed\":{},\"band\":{:.1},\"smooth\":{:.4},\"direct\":{},\"readMs\":{:.1},\"examineMs\":{:.1},\"prepareMs\":{:.1},\"gpuMs\":{:.1},\"report\":{}}}",
                    i + 1, m.file.replace('"', "'"), m.line.replace('"', "'"), m.shell, m.triangles, m.hidden, m.unclosed, m.band, m.smooth, m.direct, m.read_ms, m.examine_ms, m.prepare_ms, m.gpu_ms, m.report.json()
                ))
            })
            .collect();
        if rows.is_empty() { String::new() } else { format!(",\"models\":[{}]", rows.join(",")) }
    }
}

#[cfg(test)]
mod surface_tests {
    use super::*;
    use gf_core::groups::Item;
    use gf_core::mods::{Axes, Kind, Modifier};
    use gf_core::state::State;
    use serde_json::json;

    fn write_loader_gltf(dir: &Path, material: serde_json::Value, textured: bool) -> PathBuf {
        let mut buffer = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            buffer.extend(value.to_le_bytes());
        }
        for value in [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0] {
            buffer.extend(value.to_le_bytes());
        }
        std::fs::write(dir.join("loader.bin"), &buffer).unwrap();
        let mut doc = json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "TEXCOORD_0": 1
            }, "material": 0}]}],
            "materials": [material],
            "buffers": [{"uri": "loader.bin", "byteLength": buffer.len()}],
            "bufferViews": [
                {"buffer": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 24}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}
            ]
        });
        if textured {
            let file = std::fs::File::create(dir.join("loader.png")).unwrap();
            let mut image = png::Encoder::new(file, 1, 1);
            image.set_color(png::ColorType::Rgba);
            image.write_header().unwrap().write_image_data(&[40, 80, 120, 255]).unwrap();
            doc["images"] = json!([{"uri": "loader.png"}]);
            doc["textures"] = json!([{"source": 0}]);
        }
        let path = dir.join("loader.gltf");
        std::fs::write(&path, doc.to_string()).unwrap();
        path
    }

    #[test]
    fn imported_surfaces_follow_world_visibility_and_object_isolation() {
        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        let params = vec![("res".into(), "64".into())];
        let mut e = Engine::new(gpu, State::with_args(&params, true), true);
        let dir = std::env::temp_dir().join(format!("glassforms-surface-scope-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tetra.obj");
        std::fs::write(&path, "mtllib paint.mtl\nv 0 1 0\nv -1 -1 1\nv 1 -1 1\nv 0 -1 -1\nvt 0 0\nusemtl paint\nf 1/1 2/1 3/1\nf 1/1 3/1 4/1\nf 1/1 4/1 2/1\nf 2/1 4/1 3/1\n").unwrap();
        std::fs::write(dir.join("paint.mtl"), "newmtl paint\nmap_Kd paint.png\n").unwrap();
        let file = std::fs::File::create(dir.join("paint.png")).unwrap();
        let mut image = png::Encoder::new(file, 1, 1);
        image.set_color(png::ColorType::Rgb);
        image.write_header().unwrap().write_image_data(&[210, 30, 10]).unwrap();

        e.models.wait = true;
        e.load_model_as(&path.to_string_lossy(), true);
        assert_eq!(e.model_error(), None);
        let shape = Item::Object(1);
        let mut repeat = Modifier::of(Kind::Repeat);
        repeat.a = Axes::new([3., 1., 1.], false);
        e.state.stack.world_modifiers_mut(shape).unwrap().list.push(repeat);
        assert_eq!(e.imported_surfaces().len(), 3);
        e.state.set_layer_show(1, false);
        assert!(e.imported_surfaces().is_empty());
        assert!(e.state.enter_world_item(shape));
        let selected = e.imported_surfaces();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].place, e.state.stack.layers[1].place);
        e.state.leave_object();
        e.state.set_layer_show(1, true);
        assert!(e.state.enter_world_item(Item::Object(0)));
        assert!(e.imported_surfaces().is_empty());
        e.state.leave_object();
        assert_eq!(e.imported_surfaces().len(), 3);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn factor_only_load_retains_canonical_owner_and_renderable_surface_payload() {
        let dir = std::env::temp_dir().join(format!("glassforms-factor-surface-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("factor.obj");
        std::fs::write(&path, "mtllib factor.mtl\nv 0 1 0\nv -1 -1 1\nv 1 -1 1\nv 0 -1 -1\nusemtl paint\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3\n").unwrap();
        std::fs::write(dir.join("factor.mtl"), "newmtl paint\nKd 0.1 0.3 0.7\nPr 0.42\nPm 0.23\n").unwrap();
        let loaded = load(&path).unwrap();
        let canonical = loaded.canonical.expect("canonical import owner was dropped");
        assert_eq!(canonical.source_id, path.to_string_lossy());
        assert_eq!(canonical.materials.len(), 1);
        assert!(canonical.materials[0].spec.textures.is_empty());
        let words = loaded.surface.expect("factor-only material has no renderer payload");
        let material = words[2] as usize;
        let values: Vec<f32> = words[material..material + 6]
            .iter().map(|word| f32::from_bits(*word)).collect();
        assert_eq!(&values[..4], &[0.1, 0.3, 0.7, 1.0]);
        assert_eq!(&values[4..], &[0.42, 0.23]);
        assert_eq!(words[3] as usize, words.len());
        let repeated = load(&path).unwrap();
        assert_eq!(repeated.canonical.as_deref(), Some(canonical.as_ref()));
        assert_eq!(repeated.surface.as_deref(), Some(words.as_ref()));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_scale_gltf_uv_transform_retains_canonical_owner_and_projection() {
        let dir = std::env::temp_dir().join(format!("glassforms-zero-scale-uv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_loader_gltf(&dir, json!({
            "pbrMetallicRoughness": {"baseColorTexture": {
                "index": 0,
                "extensions": {"KHR_texture_transform": {
                    "scale": [0.0, 1.0], "offset": [0.25, 0.0]
                }}
            }}
        }), true);
        let loaded = load(&path).unwrap();
        assert!(loaded.appearance_notice.is_none());
        let canonical = loaded.canonical.expect("zero-scale UV transform lost its canonical owner");
        assert_eq!(canonical.materials.len(), 1);
        let words = loaded.surface.unwrap();
        let uv = words[1] as usize + 9;
        let actual: Vec<f32> = words[uv..uv + 6].iter().map(|word| f32::from_bits(*word)).collect();
        assert_eq!(actual, [0.25, 0.0, 0.25, 0.0, 0.25, 1.0]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn factor_only_gltf_alpha_cutoff_above_one_retains_canonical_owner_and_payload() {
        let dir = std::env::temp_dir().join(format!("glassforms-alpha-cutoff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_loader_gltf(&dir, json!({
            "alphaMode": "MASK",
            "alphaCutoff": 2.0,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.16, 0.42, 0.78, 1.0],
                "roughnessFactor": 0.29,
                "metallicFactor": 0.37
            }
        }), false);
        let loaded = load(&path).unwrap();
        assert!(loaded.appearance_notice.is_none());
        let canonical = loaded.canonical.expect("valid alpha cutoff lost its canonical owner");
        assert_eq!(canonical.materials[0].spec.coverage.values.alpha_cutoff.as_constant(), Some(&2.0));
        let words = loaded.surface.unwrap();
        let material = words[2] as usize;
        let actual: Vec<f32> = words[material..material + 6]
            .iter().map(|word| f32::from_bits(*word)).collect();
        assert_eq!(actual, [0.16, 0.42, 0.78, 1.0, 0.29, 0.37]);
        assert_eq!(words[3] as usize, words.len());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
