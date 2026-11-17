//! The surface studio on the GPU (the data and the starter stacks are gf_core::sculpt).
//!
//! The studio works on a layer's baked distance volume, in place: the volume the baker built is kept
//! aside untouched (the "pristine" copy), and the studio's result is written back into the baker's
//! volume, so that everything downstream (the tracer, the combining of layers, bubbles, thickness)
//! reads the sculpted shape and knows nothing of the studio. When only the studio changes, it runs again
//! from the pristine copy and the shape is not built again.
//!
//! A run: the stack is evaluated at every voxel (shaders/sculpt.wgsl); the field it gives is made a true
//! distance again by jump flooding in a band about the surface (shaders/sculpt_fix.wgsl); then, as
//! asked, edges are rounded and thin walls taken away (an opening, and a closing, each an offset of the
//! distance and a new distance), and pieces left floating free are found and removed. The last step
//! always leaves a distance field the tracer can step through safely.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use gf_core::sculpt::{Sculpt, MAX_IMAGES, UNIFORM_FLOATS};
use gf_core::HB;

use crate::bake::{Volume, VOL_FORMAT};
use crate::gpu::*;

const EVAL_WGSL: &str = include_str!("shaders/sculpt.wgsl");
const FIX_WGSL: &str = include_str!("shaders/sculpt_fix.wgsl");
/// Bytes between one block of the distance passes' numbers and the next.
const STRIDE: u64 = 256;
const BLOCKS: u64 = 96;
use gf_core::sculpt::BAND_VOXELS as BAND;
/// Pictures are made this size before they are handed to the GPU.
const IMAGE_SIDE: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FixU {
    res: u32,
    step: i32,
    cres: u32,
    thresh: u32,
    h: f32,
    pre: f32,
    post: f32,
    reach: f32,
    slope: f32,
    hb: f32,
    pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CcU {
    res: u32,
    cres: u32,
    thresh: u32,
    pad: u32,
    h: f32,
    pad1: [f32; 3],
}

/// The volumes a run needs beside the shape's own, for one size.
struct Work {
    res: u32,
    tmp: Volume,
    /// two sets of seeds: the flooding reads one and writes the other
    seeds: [wgpu::TextureView; 2],
    /// the pieces' grid: its size, each cell's label, each piece's count (and the largest at the end)
    cc: Option<(u32, wgpu::Buffer, wgpu::Buffer)>,
}

/// A layer's shape as the baker built it, before the studio.
struct Pristine {
    vol: Volume,
    /// the shape it is (the layer's key, its size and care)
    key: String,
    /// when it was last kept (to keep only the newest of a size)
    age: u64,
}

/// What the last run came to (for the window and the timings).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ran {
    pub res: u32,
    pub ms: f64,
    pub layers: usize,
    pub cleaned: bool,
    pub pieces: bool,
}

/// The studio's pipelines and what they read, made the first time the studio runs (a program that never
/// uses the studio never compiles its shaders).
struct Pipes {
    lin: wgpu::Sampler,
    wrap: wgpu::Sampler,
    u_eval: wgpu::Buffer,
    l_eval: wgpu::BindGroupLayout,
    p_eval: wgpu::ComputePipeline,
    u_fix: wgpu::Buffer,
    l_fix: wgpu::BindGroupLayout,
    p_seed: wgpu::ComputePipeline,
    p_jfa: wgpu::ComputePipeline,
    p_resolve: wgpu::ComputePipeline,
    p_blur: wgpu::ComputePipeline,
    u_cc: wgpu::Buffer,
    l_cc: wgpu::BindGroupLayout,
    p_cc: [wgpu::ComputePipeline; 5],
    /// the pictures bound now, by path, and the array they are in
    images: (Vec<String>, wgpu::TextureView),
}

pub struct Sculptor {
    gpu: Gpu,
    ok: bool,
    pipes: Option<Pipes>,
    /// small and large: an animation alternates between the two
    work: Vec<Work>,
    /// the shapes kept aside, known by their keys (for each layer with a studio, its newest small one and its newest large)
    pristine: Vec<Pristine>,
    /// what each layer's volume holds now: the studio's key over the pristine key ("" = the pristine shape)
    pub applied: Vec<String>,
    pub last: Ran,
    /// a picture that could not be read, and why
    pub image_error: Option<String>,
    /// one pixel's width at the shape for the present view and picture (gf_core::sculpt::footprint), which
    /// the pre-filter works to; 0 before there is a picture
    pub footprint: f64,
    /// counts the shapes kept aside
    age: u64,
}

fn compute_pipe(gpu: &Gpu, label: &str, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, entry: &str) -> wgpu::ComputePipeline {
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
    gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
}

impl Sculptor {
    pub fn new(gpu: &Gpu) -> Self {
        Self { gpu: gpu.clone(), ok: gpu.storage16, pipes: None, work: Vec::new(), pristine: Vec::new(), applied: Vec::new(), last: Ran::default(), image_error: None, footprint: 0.0, age: 0 }
    }
}

impl Pipes {
    fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let tex = |binding, dim, sample_type| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Texture { sample_type, view_dimension: dim, multisampled: false }, count: None };
        let float = |filterable| wgpu::TextureSampleType::Float { filterable };
        let store = |binding, format, access| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::StorageTexture { access, format, view_dimension: wgpu::TextureViewDimension::D3 }, count: None };
        let samp = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
        let buf = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let l_eval = gpu.bind_layout("sculpt", &[
            uniform_entry(0, c, false),
            tex(1, wgpu::TextureViewDimension::D3, float(true)),
            samp(2),
            tex(3, wgpu::TextureViewDimension::D2Array, float(true)),
            samp(4),
            store(5, VOL_FORMAT, wgpu::StorageTextureAccess::WriteOnly),
        ]);
        let l_fix = gpu.bind_layout("sculpt distance", &[
            uniform_entry(0, c, true),
            tex(1, wgpu::TextureViewDimension::D3, float(false)),
            store(2, wgpu::TextureFormat::R32Uint, wgpu::StorageTextureAccess::ReadWrite),
            store(3, VOL_FORMAT, wgpu::StorageTextureAccess::WriteOnly),
            store(4, wgpu::TextureFormat::R32Uint, wgpu::StorageTextureAccess::WriteOnly),
        ]);
        let l_cc = gpu.bind_layout("sculpt pieces", &[
            uniform_entry(0, c, false),
            tex(1, wgpu::TextureViewDimension::D3, float(false)),
            buf(2),
            buf(3),
            store(4, VOL_FORMAT, wgpu::StorageTextureAccess::WriteOnly),
        ]);
        let eval = gpu.shader("sculpt", EVAL_WGSL, false);
        let mut parts = FIX_WGSL.split("//#pieces");
        let dist = gpu.shader("sculpt distance", parts.next().unwrap(), false);
        let pieces = gpu.shader("sculpt pieces", &format!("const NONE: u32 = 0xffffffffu;\n{}", parts.next().unwrap()), false);
        let p_eval = compute_pipe(gpu, "sculpt", &eval, &l_eval, "cs_eval");
        let p_seed = compute_pipe(gpu, "sculpt seeds", &dist, &l_fix, "cs_seed");
        let p_jfa = compute_pipe(gpu, "sculpt flood", &dist, &l_fix, "cs_jfa");
        let p_resolve = compute_pipe(gpu, "sculpt distance", &dist, &l_fix, "cs_resolve");
        let p_blur = compute_pipe(gpu, "sculpt round", &dist, &l_fix, "cs_blur");
        let p_cc = ["cs_cc_init", "cs_cc_merge", "cs_cc_compress", "cs_cc_largest", "cs_cc_remove"].map(|e| compute_pipe(gpu, e, &pieces, &l_cc, e));
        let u_eval = gpu.uniform_buffer("sculpt", (UNIFORM_FLOATS * 4) as u64);
        let u_fix = gpu.uniform_buffer("sculpt distance", STRIDE * BLOCKS);
        let u_cc = gpu.uniform_buffer("sculpt pieces", 32);
        let images = (Vec::new(), Self::blank_images(gpu));
        Self {
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            wrap: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat),
            u_eval, l_eval, p_eval, u_fix, l_fix, p_seed, p_jfa, p_resolve, p_blur, u_cc, l_cc, p_cc, images,
        }
    }

    fn blank_images(gpu: &Gpu) -> wgpu::TextureView {
        let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sculpt pictures"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[128, 128, 128, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
    }

    /// The pictures these paths name, read, made square and put in one array (kept while they stay the same).
    /// (Some(error) for a picture that could not be read; it is used as plain grey.)
    fn ensure_images(&mut self, gpu: &Gpu, paths: &[String]) -> Option<Option<String>> {
        if self.images.0 == paths { return None; }
        let mut error = None;
        if paths.is_empty() {
            self.images = (Vec::new(), Self::blank_images(gpu));
            return Some(None);
        }
        let n = paths.len().min(MAX_IMAGES) as u32;
        let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sculpt pictures"),
            size: wgpu::Extent3d { width: IMAGE_SIDE, height: IMAGE_SIDE, depth_or_array_layers: n },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (i, p) in paths.iter().take(n as usize).enumerate() {
            let px = match read_picture(p) {
                Ok(px) => px,
                Err(e) => {
                    error = Some(e);
                    vec![128u8; (IMAGE_SIDE * IMAGE_SIDE * 4) as usize]
                }
            };
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 }, aspect: wgpu::TextureAspect::All },
                &px,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(IMAGE_SIDE * 4), rows_per_image: Some(IMAGE_SIDE) },
                wgpu::Extent3d { width: IMAGE_SIDE, height: IMAGE_SIDE, depth_or_array_layers: 1 },
            );
        }
        let view = t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        self.images = (paths.to_vec(), view);
        Some(error)
    }
}

impl Sculptor {
    /// Can the studio run on this device (it writes volumes by compute shader)?
    pub fn available(&self) -> bool {
        self.ok
    }

    fn work_for(&mut self, res: u32) -> usize {
        if let Some(i) = self.work.iter().position(|w| w.res == res) { return i; }
        // keep a small one (an animation's) and the newest large one
        if res > 384 { self.work.retain(|w| w.res <= 384); } else { self.work.retain(|w| w.res > 384); }
        let mk = || self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sculpt seeds"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        }).create_view(&Default::default());
        let seeds = [mk(), mk()];
        let w = Work { res, tmp: Volume::new(&self.gpu, res, true), seeds, cc: None };
        self.work.push(w);
        self.work.len() - 1
    }

    fn cc_for(&mut self, wi: usize) {
        if self.work[wi].cc.is_some() { return; }
        let cres = self.work[wi].res / 2;
        let n = cres as u64 * cres as u64 * cres as u64;
        let mk = |label| self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (n + 4) * 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let (a, b) = (mk("sculpt labels"), mk("sculpt pieces"));
        self.work[wi].cc = Some((cres, a, b));
    }

    fn applied_slot(&mut self, layer: usize) -> &mut String {
        while self.applied.len() <= layer { self.applied.push(String::new()); }
        &mut self.applied[layer]
    }

    /// Is the shape with this key kept aside (so that the studio alone can run again)? The shapes kept
    /// are known by what they are, not by which layer they belong to: a layer that moves up the list when
    /// another is taken away still finds its own.
    pub fn has(&self, key: &str) -> bool {
        self.pristine.iter().any(|p| p.key == key)
    }

    /// The baker has just built a layer's shape into `vol` (the layer at `layer`): keep it aside, under `key`.
    pub fn keep(&mut self, layer: usize, vol: &Volume, key: &str) {
        let gpu = self.gpu.clone();
        let res = vol.res;
        self.age += 1;
        let i = match self.pristine.iter().position(|p| p.key == key && p.vol.res == res) {
            Some(i) => i,
            None => {
                self.pristine.push(Pristine { vol: Volume::new(&gpu, res, false), key: key.to_string(), age: 0 });
                self.pristine.len() - 1
            }
        };
        self.pristine[i].age = self.age;
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sculpt keep") });
        copy_volume(&mut enc, &vol.tex, &self.pristine[i].vol.tex, res);
        gpu.queue.submit([enc.finish()]);
        self.applied_slot(layer).clear();
    }

    /// Put the pristine shape back into `vol`, the volume of the layer at `layer` (its studio has been
    /// emptied). False if it is not kept.
    pub fn restore(&mut self, layer: usize, vol: &Volume, key: &str) -> bool {
        let Some(p) = self.pristine.iter().find(|p| p.key == key && p.vol.res == vol.res) else { return false };
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sculpt restore") });
        copy_volume(&mut enc, &p.vol.tex, &vol.tex, vol.res);
        self.gpu.queue.submit([enc.finish()]);
        self.applied_slot(layer).clear();
        true
    }

    /// The layer at `layer` now holds its shape as built, untouched by any studio.
    pub fn forget(&mut self, layer: usize) {
        if let Some(a) = self.applied.get_mut(layer) { a.clear(); }
    }
    /// What the volumes of the layers from this one on hold is no longer known (they have been taken away).
    pub fn drop_layers(&mut self, from: usize) {
        self.applied.truncate(from.max(1));
    }
    /// Let go of every shape kept aside that no layer with a studio is now made of (`shapes`: those
    /// layers' keys, without their size), keeping for each of those the newest of the small size and the
    /// newest of the large; and of the working volumes, once nothing is kept at all.
    pub fn keep_only(&mut self, shapes: &[String]) {
        let shape_of = |key: &str| key.rsplitn(3, '|').nth(2).unwrap_or("").to_string();
        let mut newest: Vec<(String, bool, u64)> = Vec::new();
        for p in &self.pristine {
            let k = (shape_of(&p.key), p.vol.res > 384);
            match newest.iter_mut().find(|n| n.0 == k.0 && n.1 == k.1) {
                Some(n) => n.2 = n.2.max(p.age),
                None => newest.push((k.0, k.1, p.age)),
            }
        }
        self.pristine.retain(|p| {
            let s = shape_of(&p.key);
            shapes.contains(&s) && newest.iter().any(|n| n.0 == s && n.1 == (p.vol.res > 384) && n.2 == p.age)
        });
        if self.pristine.is_empty() { self.work.clear(); }
    }

    /// Run the studio for this layer: from its pristine shape (kept under `key`) into `vol`.
    /// `phases`: each studio layer's phase of evolving. `timed`: wait for the GPU, so the time is true.
    /// `moving`: a build while something moves or is dragged (loose pieces are left).
    pub fn run(&mut self, layer: usize, vol: &Volume, key: &str, s: &Sculpt, phases: &[f64], timed: bool, moving: bool) -> R<Ran> {
        if !self.ok { return Err("the surface studio needs a graphics device that can write volumes by compute shader".into()); }
        let t0 = Instant::now();
        let res = vol.res;
        let gpu = self.gpu.clone();
        if self.pipes.is_none() { self.pipes = Some(Pipes::new(&gpu)); }
        if let Some(e) = self.pipes.as_mut().unwrap().ensure_images(&gpu, &s.images()) { self.image_error = e; }
        let wi = self.work_for(res);
        // (loose pieces are looked for only in a still build: while things move it costs too much)
        let pieces = s.crumbs > 0.0 && !moving;
        if pieces { self.cc_for(wi); }
        let pi = self.pristine.iter().position(|p| p.key == key && p.vol.res == res).ok_or("the shape before the studio is not kept")?;
        self.applied_slot(layer);
        let src = &self.pristine[pi].vol;
        let w = &self.work[wi];
        let g = self.pipes.as_ref().unwrap();
        let h = 2.0 * HB / res as f64;

        // the numbers: the stack's, and one block for every pass of the distance steps
        let uni = s.uniform(phases, res, HB, &g.images.0, self.footprint);
        gpu.queue.write_buffer(&g.u_eval, 0, bytemuck::cast_slice(&uni));
        let mut blocks: Vec<FixU> = Vec::new();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sculpt") });

        // 1. the stack, from the pristine shape into the spare volume
        let g_eval = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &g.l_eval,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: g.u_eval.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&g.lin) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&g.images.1) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&g.wrap) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&w.tmp.view) },
            ],
        });
        {
            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sculpt layers"), timestamp_writes: None });
            c.set_pipeline(&g.p_eval);
            c.set_bind_group(0, &g_eval, &[]);
            c.dispatch_workgroups(res / 8, res / 8, res / 4);
        }

        // 2. a true distance again; then the cleanups, each an offset and a new distance
        // (`at_tmp`: the field as it stands is in the spare volume, not in the shape's)
        let (vol_v, tmp_v) = (&vol.view, &w.tmp.view);
        let mut at_tmp = true;
        let redist = |enc: &mut wgpu::CommandEncoder, blocks: &mut Vec<FixU>, at_tmp: &mut bool, pre: f64, post: f64, slope: f64, reach_vox: f64| {
            let (from, to) = if *at_tmp { (tmp_v, vol_v) } else { (vol_v, tmp_v) };
            distance(&gpu, g, enc, blocks, from, to, &w.seeds, res, h, pre, post, slope, reach_vox);
            *at_tmp = !*at_tmp;
        };
        let slope = s.slope();
        redist(&mut enc, &mut blocks, &mut at_tmp, 0.0, 0.0, slope, BAND);
        let open = 0.5 * s.wall;
        if open > 0.0 {
            // an opening: in by half the wall and out again. Walls thinner than that go, cleanly.
            redist(&mut enc, &mut blocks, &mut at_tmp, open, -open, 1.05, BAND + open / h);
        }
        if s.round > 0.0 {
            // rounding: the distance smoothed over the radius in all three directions, then made a
            // distance again. Edges come out round; slivers thinner than the radius melt away.
            let rv = (s.round / h).round().clamp(1.0, 24.0) as u32;
            let reach = BAND + rv as f64;
            for axis in 0..3 {
                let (from, to) = if at_tmp { (tmp_v, vol_v) } else { (vol_v, tmp_v) };
                blur(&gpu, g, &mut enc, &mut blocks, from, to, &w.seeds, res, h, axis, rv, reach);
                at_tmp = !at_tmp;
            }
            redist(&mut enc, &mut blocks, &mut at_tmp, 0.0, 0.0, 1.05, reach);
        }
        if pieces {
            // 3. pieces floating free, smaller than asked for, become empty space; then a distance again
            let (cres, labels, sizes) = w.cc.as_ref().unwrap();
            let cell = 2.0 * h;
            // (a piece's measured extent is between half its width and its width: one under half the size
            // asked for is surely narrower than it)
            let thresh = ((0.5 * s.crumbs / cell).ceil() as u32).max(1);
            let cu = CcU { res, cres: *cres, thresh, pad: 0, h: h as f32, pad1: [0.0; 3] };
            gpu.queue.write_buffer(&g.u_cc, 0, bytemuck::bytes_of(&cu));
            let (from, to) = if at_tmp { (tmp_v, vol_v) } else { (vol_v, tmp_v) };
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &g.l_cc,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: g.u_cc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(from) },
                    wgpu::BindGroupEntry { binding: 2, resource: labels.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: sizes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(to) },
                ],
            });
            for (i, p) in g.p_cc.iter().enumerate() {
                let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sculpt pieces"), timestamp_writes: None });
                c.set_pipeline(p);
                c.set_bind_group(0, &bg, &[]);
                if i == 4 { c.dispatch_workgroups(res / 8, res / 8, res / 4); } else { c.dispatch_workgroups(cres.div_ceil(4), cres.div_ceil(4), cres.div_ceil(4)); }
            }
            at_tmp = !at_tmp;
            redist(&mut enc, &mut blocks, &mut at_tmp, 0.0, 0.0, 1.05, BAND);
        }
        if at_tmp {
            copy_volume(&mut enc, &w.tmp.tex, &vol.tex, res);
        }
        if blocks.len() as u64 > BLOCKS { return Err("the surface studio asked for too many passes".into()); }
        let mut data = vec![0u8; blocks.len() * STRIDE as usize];
        for (i, b) in blocks.iter().enumerate() {
            data[i * STRIDE as usize..i * STRIDE as usize + std::mem::size_of::<FixU>()].copy_from_slice(bytemuck::bytes_of(b));
        }
        if !data.is_empty() { gpu.queue.write_buffer(&g.u_fix, 0, &data); }
        gpu.queue.submit([enc.finish()]);
        if timed { gpu.wait(); }
        if let Some(a) = self.applied.get_mut(layer) { *a = format!("{key}#{}", s.key(phases, self.footprint)); }
        self.last = Ran { res, ms: t0.elapsed().as_secs_f64() * 1e3, layers: s.live().count(), cleaned: open > 0.0 || s.round > 0.0, pieces };
        Ok(self.last)
    }

}

/// One distance step: the zero of (field + pre) in `from`, as a distance (+ post) into `to`.
#[allow(clippy::too_many_arguments)]
fn distance(gpu: &Gpu, g: &Pipes, enc: &mut wgpu::CommandEncoder, blocks: &mut Vec<FixU>, from: &wgpu::TextureView, to: &wgpu::TextureView, seeds: &[wgpu::TextureView; 2], res: u32, h: f64, pre: f64, post: f64, slope: f64, reach_vox: f64) {
    // (bg[i] reads seeds[i] and writes the other set)
    let bg = [0usize, 1].map(|i| gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.l_fix,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &g.u_fix, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<FixU>() as u64) }) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(from) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&seeds[i]) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(to) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&seeds[1 - i]) },
        ],
    }));
    let base = FixU { res, step: 0, cres: 0, thresh: 0, h: h as f32, pre: pre as f32, post: post as f32, reach: (reach_vox * h) as f32, slope: slope as f32, hb: HB as f32, pad: [0.0; 2] };
    // the flooding's steps: one of 1, halving from the first power of two at least half the band, then one more of 1
    let mut steps = vec![1];
    let mut k = ((reach_vox / 2.0).ceil() as u32).max(1).next_power_of_two();
    while k >= 1 {
        steps.push(k as i32);
        k /= 2;
    }
    steps.push(1);
    let groups = (res / 8, res / 8, res / 4);
    let mut run = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, step: i32, read: usize| {
        let at = blocks.len() as u64 * STRIDE;
        blocks.push(FixU { step, ..base });
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sculpt distance"), timestamp_writes: None });
        c.set_pipeline(pipe);
        c.set_bind_group(0, &bg[read], &[at as u32]);
        c.dispatch_workgroups(groups.0, groups.1, groups.2);
    };
    // seeds into set 1; each flooding step reads the set the last one wrote; the distance reads the last
    run(enc, &g.p_seed, 0, 0);
    let mut read = 1;
    for s in steps {
        run(enc, &g.p_jfa, s, read);
        read = 1 - read;
    }
    run(enc, &g.p_resolve, 0, read);
}

/// One pass of the rounding: the distance in `from` smoothed along one axis into `to`.
#[allow(clippy::too_many_arguments)]
fn blur(gpu: &Gpu, g: &Pipes, enc: &mut wgpu::CommandEncoder, blocks: &mut Vec<FixU>, from: &wgpu::TextureView, to: &wgpu::TextureView, seeds: &[wgpu::TextureView; 2], res: u32, h: f64, axis: i32, radius: u32, reach_vox: f64) {
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.l_fix,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &g.u_fix, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<FixU>() as u64) }) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(from) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&seeds[0]) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(to) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&seeds[1]) },
        ],
    });
    let at = blocks.len() as u64 * STRIDE;
    blocks.push(FixU { res, step: axis, cres: radius, thresh: 0, h: h as f32, pre: 0.0, post: 0.0, reach: (reach_vox * h) as f32, slope: 1.05, hb: HB as f32, pad: [0.0; 2] });
    let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sculpt round"), timestamp_writes: None });
    c.set_pipeline(&g.p_blur);
    c.set_bind_group(0, &bg, &[at as u32]);
    c.dispatch_workgroups(res / 8, res / 8, res / 4);
}

fn copy_volume(enc: &mut wgpu::CommandEncoder, from: &wgpu::Texture, to: &wgpu::Texture, res: u32) {
    enc.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo { texture: from, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyTextureInfo { texture: to, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
    );
}

/// A PNG file as IMAGE_SIDE x IMAGE_SIDE RGBA bytes (stretched to the square, smoothly).
pub fn read_picture(path: &str) -> R<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|e| format!("could not open {path}: {e}"))?;
    let mut dec = png::Decoder::new(std::io::BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| format!("{path} is not a PNG picture this can read: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or("the picture is too large")?];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("could not read {path}: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("indexed pictures are not read".into()),
    };
    let at = |x: usize, y: usize| -> [f32; 3] {
        let o = (y * w + x) * ch;
        if ch < 3 { [buf[o] as f32; 3] } else { [buf[o] as f32, buf[o + 1] as f32, buf[o + 2] as f32] }
    };
    let side = IMAGE_SIDE as usize;
    let mut out = vec![0u8; side * side * 4];
    for y in 0..side {
        let fy = ((y as f64 + 0.5) / side as f64 * h as f64 - 0.5).clamp(0.0, (h - 1) as f64);
        let (y0, ty) = (fy.floor() as usize, (fy - fy.floor()) as f32);
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..side {
            let fx = ((x as f64 + 0.5) / side as f64 * w as f64 - 0.5).clamp(0.0, (w - 1) as f64);
            let (x0, tx) = (fx.floor() as usize, (fx - fx.floor()) as f32);
            let x1 = (x0 + 1).min(w - 1);
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            for k in 0..3 {
                let v = (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty;
                out[(y * side + x) * 4 + k] = v.round().clamp(0.0, 255.0) as u8;
            }
            out[(y * side + x) * 4 + 3] = 255;
        }
    }
    Ok(out)
}
