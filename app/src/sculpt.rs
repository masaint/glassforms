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
use gf_core::mods::{Bake, Stage};
use gf_core::sculpt::{Sculpt, MAX_IMAGES, UNIFORM_FLOATS};
use gf_core::HB;

use crate::bake::{Volume, VOL_FORMAT};
use crate::gpu::*;

const EVAL_WGSL: &str = include_str!("shaders/sculpt.wgsl");
const FIX_WGSL: &str = include_str!("shaders/sculpt_fix.wgsl");
/// Bytes between one block of the distance passes' numbers and the next.
const STRIDE: u64 = 256;
const BLOCKS: u64 = 96;
/// Runs of modifiers one application may hold (each a block of numbers).
const WARP_BLOCKS: u64 = 8;
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
    /// when it was last worked in
    used: std::time::Instant,
}

/// Working volumes larger than this many voxels a side are let go of once unused for `WORK_IDLE`: at the
/// still detail they are 11 bytes a voxel (two 32-bit seed volumes, a 16-bit volume, the pieces' grid),
/// 5 GB at 768 and 12 GB at 1024, held for nothing while the picture stands still. Small ones (an
/// animation's, a drag's) are kept.
pub const WORK_KEPT_RES: u32 = 384;
pub const WORK_IDLE: std::time::Duration = std::time::Duration::from_secs(2);

/// Is a working volume of `res` voxels a side, unused for `idle`, still kept?
pub fn keep_work(res: u32, idle: std::time::Duration) -> bool {
    res <= WORK_KEPT_RES || idle < WORK_IDLE
}

/// A layer's shape as the baker built it, before the studio.
struct Pristine {
    vol: Volume,
    /// the shape it is (the layer's key, its size and care)
    key: String,
    /// when it was last kept (to keep only the newest of a size)
    age: u64,
    /// the shape's own size (`gf_core::sculpt::Sculpt::size_of`), once a stack with sizes as shares of it
    /// has needed it
    size: Option<f64>,
}

/// How deep below the surface the points inside a shape lie, on average (scene units): from eight slices
/// of its volume read back (cheaply enough; it is done once for a shape, not for every change of the
/// studio). NaN if nothing of it is inside.
fn mean_depth(gpu: &Gpu, vol: &Volume) -> f64 {
    let (mut sum, mut count) = (0.0f64, 0u64);
    for i in 0..8 {
        let z = (((i as f64 + 0.5) / 8.0) * vol.res as f64) as u32;
        let Ok(bytes) = gpu.read(&vol.tex, 0, 0, z.min(vol.res - 1), vol.res, vol.res, 2) else { continue };
        for h in bytemuck::cast_slice::<u8, half::f16>(&bytes) {
            let d = h.to_f32();
            if d < 0.0 {
                sum -= d as f64;
                count += 1;
            }
        }
    }
    if count > 0 { sum / count as f64 } else { f64::NAN }
}

/// What the last run came to (for the window and the timings).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ran {
    pub res: u32,
    pub ms: f64,
    pub layers: usize,
    pub cleaned: bool,
    pub pieces: bool,
    /// the shape's own size, which sizes given as shares are taken of
    pub size: f64,
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
    warp_distance: [wgpu::ComputePipeline; 3],
    u_cc: wgpu::Buffer,
    l_cc: wgpu::BindGroupLayout,
    p_cc: [wgpu::ComputePipeline; 5],
    /// the modifiers' pass (gf_core::mods::wgsl_baked): a run of them read at every voxel
    u_mods: wgpu::Buffer,
    l_mods: wgpu::BindGroupLayout,
    p_mods: wgpu::ComputePipeline,
    /// the pictures bound now, by path, and the array they are in
    images: (Vec<String>, wgpu::TextureView),
}

pub struct Sculptor {
    face_passes: Option<crate::baked_faces::Passes>,
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
    /// each layer's shape's own size, as last measured (0: not yet)
    pub sizes: Vec<f64>,
    /// the keys of the trees kept aside this build for their objects' modifiers (an object read as a tree
    /// has them applied to the volume its tree is measured into, in the slot TREE_SLOT + its layer)
    pub trees: Vec<String>,
    /// a volume was made a distance again this build (`redistance`): the working volumes stay
    pub hold_work: bool,
}

/// Where an object read as a tree keeps what its modifiers made: past any layer's own slot.
pub const TREE_SLOT: usize = 32;
/// The other slots past the layers' trees, one range each (MAX_LAYERS slots), so that none overlaps:
/// a group's bodies, World fields by source, the pivot-mass probe, then each layer's construction grids.
pub const GROUP_SLOT: usize = TREE_SLOT + gf_core::layers::MAX_LAYERS;
pub const WORLD_SLOT: usize = GROUP_SLOT + gf_core::layers::MAX_LAYERS;
pub const PIVOT_SLOT: usize = WORLD_SLOT + gf_core::layers::MAX_LAYERS;
pub const GRID_SLOT: usize = PIVOT_SLOT + 1;
const _: () = assert!(gf_core::layers::MAX_LAYERS <= TREE_SLOT);

fn compute_pipe(gpu: &Gpu, label: &str, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout, entry: &str) -> wgpu::ComputePipeline {
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
    gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
}

impl Sculptor {
    pub fn new(gpu: &Gpu) -> Self {
        Self { gpu: gpu.clone(), ok: gpu.storage16, pipes: None, face_passes: None, work: Vec::new(), pristine: Vec::new(), applied: Vec::new(), last: Ran::default(), image_error: None, footprint: 0.0, age: 0, sizes: Vec::new(), trees: Vec::new(), hold_work: false }
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
        let l_mods = gpu.bind_layout("modifiers", &[
            uniform_entry(0, c, true),
            tex(1, wgpu::TextureViewDimension::D3, float(true)),
            samp(2),
            store(3, VOL_FORMAT, wgpu::StorageTextureAccess::WriteOnly),
        ]);
        let p_mods = compute_pipe(gpu, "modifiers", &gpu.shader("modifiers", &gf_core::mods::wgsl_baked(), false), &l_mods, "cs_mods");
        let u_mods = gpu.uniform_buffer("modifiers", gf_core::mods::BAKED_STRIDE as u64 * WARP_BLOCKS);
        let eval = gpu.shader("sculpt", EVAL_WGSL, false);
        let mut parts = FIX_WGSL.split("//#pieces");
        let dist = gpu.shader("sculpt distance", parts.next().unwrap(), false);
        let pieces = gpu.shader("sculpt pieces", &format!("const NONE: u32 = 0xffffffffu;\n{}", parts.next().unwrap()), false);
        let p_eval = compute_pipe(gpu, "sculpt", &eval, &l_eval, "cs_eval");
        let p_seed = compute_pipe(gpu, "sculpt seeds", &dist, &l_fix, "cs_seed");
        let p_jfa = compute_pipe(gpu, "sculpt flood", &dist, &l_fix, "cs_jfa");
        let p_resolve = compute_pipe(gpu, "sculpt distance", &dist, &l_fix, "cs_resolve");
        let warp_distance = ["cs_warp_cells", "cs_warp_axis", "cs_warp_distance"]
            .map(|e| compute_pipe(gpu, e, &dist, &l_fix, e));
        let p_blur = compute_pipe(gpu, "sculpt round", &dist, &l_fix, "cs_blur");
        let p_cc = ["cs_cc_init", "cs_cc_merge", "cs_cc_compress", "cs_cc_largest", "cs_cc_remove"].map(|e| compute_pipe(gpu, e, &pieces, &l_cc, e));
        let u_eval = gpu.uniform_buffer("sculpt", (UNIFORM_FLOATS * 4) as u64);
        let u_fix = gpu.uniform_buffer("sculpt distance", STRIDE * BLOCKS);
        let u_cc = gpu.uniform_buffer("sculpt pieces", 32);
        let images = (Vec::new(), Self::blank_images(gpu));
        Self {
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            wrap: gpu.sampler_uv(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat, wgpu::AddressMode::Repeat),
            u_eval, l_eval, p_eval, u_fix, l_fix, p_seed, p_jfa, p_resolve, p_blur, warp_distance, u_cc, l_cc, p_cc, u_mods, l_mods, p_mods, images,
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

    fn work_for(&mut self, res: u32) -> R<usize> {
        if let Some(i) = self.work.iter().position(|w| w.res == res) {
            self.work[i].used = std::time::Instant::now();
            return Ok(i);
        }
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
        let voxels = res as u64 * res as u64 * res as u64;
        let seeds = self.gpu.allocate("the surface studio's working volumes", voxels * 8, || [mk(), mk()])?;
        let w = Work { res, tmp: Volume::try_new(&self.gpu, res, true)?, seeds, cc: None, used: std::time::Instant::now() };
        self.work.push(w);
        Ok(self.work.len() - 1)
    }

    fn cc_for(&mut self, wi: usize) -> R<()> {
        if self.work[wi].cc.is_some() { return Ok(()); }
        let cres = self.work[wi].res / 2;
        let n = cres as u64 * cres as u64 * cres as u64;
        let mk = |label| self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (n + 4) * 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let (a, b) = self.gpu.allocate("the surface studio's piece labels", (n + 4) * 8, || (mk("sculpt labels"), mk("sculpt pieces")))?;
        self.work[wi].cc = Some((cres, a, b));
        Ok(())
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
    pub fn keep(&mut self, layer: usize, vol: &Volume, key: &str) -> R<()> {
        let gpu = self.gpu.clone();
        let res = vol.res;
        self.age += 1;
        let i = match self.pristine.iter().position(|p| p.key == key && p.vol.res == res) {
            Some(i) => i,
            None => {
                self.pristine.push(Pristine { vol: Volume::try_new(&gpu, res, false)?, key: key.to_string(), age: 0, size: None });
                self.pristine.len() - 1
            }
        };
        self.pristine[i].age = self.age;
        // (kept afresh: the shape may be another, so its size is measured again when next wanted)
        self.pristine[i].size = None;
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sculpt keep") });
        copy_volume(&mut enc, &vol.tex, &self.pristine[i].vol.tex, res);
        gpu.queue.submit([enc.finish()]);
        self.applied_slot(layer).clear();
        Ok(())
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
    /// The trees kept aside this build (`trees`) are kept too.
    pub fn keep_only(&mut self, shapes: &[String]) {
        let shape_of = |key: &str| key.rsplitn(3, '|').nth(2).unwrap_or("").to_string();
        let mut shapes = shapes.to_vec();
        shapes.extend(self.trees.iter().map(|k| shape_of(k)));
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
        if self.pristine.is_empty() && !self.hold_work { self.work.clear(); }
        self.hold_work = false;
    }

    /// Let go of the large working volumes not worked in lately (`keep_work`).
    pub fn trim_work(&mut self) {
        self.work.retain(|w| keep_work(w.res, w.used.elapsed()));
    }

    /// Make `vol` a true distance again about its surface (the band the studio makes exact; beyond it the
    /// reading over `slope`). For a csg object's merged volume read through exact modifiers: what it holds is
    /// the distance shortened by their bound, which would shorten every step of the march by as much.
    pub fn redistance(&mut self, vol: &Volume, slope: f64) -> R<()> {
        if !self.ok { return Ok(()); }
        let gpu = self.gpu.clone();
        if self.pipes.is_none() { self.pipes = Some(Pipes::new(&gpu)); }
        self.hold_work = true;
        let res = vol.res;
        let wi = self.work_for(res)?;
        let w = &self.work[wi];
        let g = self.pipes.as_ref().unwrap();
        let h = 2.0 * HB / res as f64;
        let mut blocks: Vec<FixU> = Vec::new();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("modifiers distance") });
        // (all through the volume, not only in a band about the surface: beyond a band the field read is divided by
        // the warps' bound, which can be tens, and a march, a shadow or the surface pass reading so small a
        // number takes it for the surface itself)
        distance(&gpu, g, &mut enc, &mut blocks, &vol.view, &w.tmp.view, &w.seeds, res, h, 0.0, 0.0, slope.abs(), res as f64);
        copy_volume(&mut enc, &w.tmp.tex, &vol.tex, res);
        let mut data = vec![0u8; blocks.len() * STRIDE as usize];
        for (i, b) in blocks.iter().enumerate() {
            data[i * STRIDE as usize..i * STRIDE as usize + std::mem::size_of::<FixU>()].copy_from_slice(bytemuck::bytes_of(b));
        }
        gpu.queue.write_buffer(&g.u_fix, 0, &data);
        gpu.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Apply an object's modifiers to its volume: from its pristine shape (kept under `key`) into `vol`, stage
    /// by stage (`gf_core::mods::Bake`: runs of warps, the surface lab, smoothings), each followed by a new
    /// distance. `phases`: each studio layer's phase of evolving. `timed`: wait for the GPU, so the time is
    /// true. `moving`: a build while something moves or is dragged (loose pieces are left). `object`: where the
    /// object stands in the volume's frame (identity for a volume object, whose volume is its own frame).
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, layer: usize, vol: &Volume, key: &str, bake: &Bake, phases: &[f64], timed: bool, moving: bool, object: &gf_core::layers::Placement) -> R<Ran> {
        self.run_with_faces(layer,vol,key,bake,phases,timed,moving,object,None)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_with_faces(&mut self, layer: usize, vol: &Volume, key: &str, bake: &Bake, phases: &[f64], timed: bool, moving: bool, object: &gf_core::layers::Placement, faces: Option<&mut crate::world_faces::LocalGrid>) -> R<Ran> {
        self.run_in_domain(layer,vol,key,bake,phases,timed,moving,object,faces,None)
    }

    /// World fields may need a wider destination cube than the pristine object.
    /// The first warp reads the original texture directly at full source precision.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_world_field(&mut self, layer: usize, vol: &Volume, key: &str, mut bake: Bake,
        moving: bool, scale: f64, source_res: u32, mut faces: Option<&mut crate::world_faces::LocalGrid>) -> R<Ran> {
        let object = gf_core::layers::Placement { size: 1.0 / scale, ..Default::default() };
        if scale != 1.0 || source_res != vol.res {
            // A filtering stage requires an input in its own voxel coordinates.
            if !matches!(bake.stages.first(), Some(Stage::Warp(_))) {
                bake.stages.insert(0, Stage::Warp(Vec::new()));
            }
            for stage in &mut bake.stages {
                if let Stage::Smooth(radius) = stage { *radius = radius.map(|r| r / scale); }
            }
        }
        if !matches!(bake.stages.first(), Some(Stage::Warp(_))) {
            if let Some(face) = faces.as_deref_mut() { face.detach(&self.gpu); }
        }
        self.run_in_domain(layer, vol, key, &bake, &[], false, moving, &object, faces, Some((scale, source_res)))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_in_domain(&mut self, layer: usize, vol: &Volume, key: &str, bake: &Bake, phases: &[f64], timed: bool, moving: bool, object: &gf_core::layers::Placement, mut faces: Option<&mut crate::world_faces::LocalGrid>, world_domain: Option<(f64, u32)>) -> R<Ran> {
        let (source_scale, source_res) = world_domain.unwrap_or((1.0, vol.res));
        if !self.ok { return Err("the surface studio needs a graphics device that can write volumes by compute shader".into()); }
        let t0 = Instant::now();
        let res = vol.res;
        let gpu = self.gpu.clone();
        if self.pipes.is_none() { self.pipes = Some(Pipes::new(&gpu)); }
        if faces.is_some() && self.face_passes.is_none() {
            self.face_passes = Some(crate::baked_faces::Passes::new(&gpu));
        }
        let face_res = faces.as_ref().map_or(0, |f| {
            if world_domain.is_some() && source_scale > 1.0 {
                // Only the current distinct field owns these temporary labels.
                ((f64::from(f.res) * source_scale).ceil() as u32).min(res.max(f.res)).min(256)
            } else { f.res }
        });
        let face_dims = faces.as_ref().map(|f| (
            crate::baked_faces::Passes::dims(&gpu, face_res, f.res),
            crate::baked_faces::Passes::dims(&gpu, face_res, face_res),
        ));
        let no_lab = Sculpt::default();
        let lab = bake.lab().unwrap_or(&no_lab);
        if let Some(e) = self.pipes.as_mut().unwrap().ensure_images(&gpu, &lab.images()) { self.image_error = e; }
        let wi = self.work_for(res)?;
        // (loose pieces are looked for only in a still build: while things move it costs too much)
        let pieces = bake.lab().is_some_and(|l| l.crumbs > 0.0) && !moving;
        if pieces { self.cc_for(wi)?; }
        let pi = self.pristine.iter().position(|p| p.key == key && p.vol.res == source_res).ok_or("the shape before the studio is not kept")?;
        // The shape's own size, as it is before the studio (measured once for a shape, and kept for the
        // window, which switches a layer between the two ways of giving sizes without changing its look);
        // sizes given as shares of it become lengths.
        let size = match self.pristine[pi].size {
            Some(v) => v,
            // There is no artist-facing size control on a projected occurrence.
            // A warp-only World field never needs a synchronous depth reduction.
            None if world_domain.is_some() && bake.lab().is_none() => 0.0,
            None => {
                let v = Sculpt::size_of(mean_depth(&gpu, &self.pristine[pi].vol));
                self.pristine[pi].size = Some(v);
                v
            }
        };
        if self.sizes.len() <= layer { self.sizes.resize(layer + 1, 0.0); }
        self.sizes[layer] = size;
        let resolved = lab.any_relative().then(|| lab.resolved(size));
        let s = resolved.as_ref().unwrap_or(lab);
        self.applied_slot(layer);
        let src = &self.pristine[pi].vol;
        let w = &self.work[wi];
        let g = self.pipes.as_ref().unwrap();
        let h = 2.0 * HB / res as f64;

        // the numbers: the lab's, each run of warps', and one block for every pass of the distance steps
        if bake.lab().is_some() {
            let uni = s.uniform(phases, res, HB, &g.images.0, self.footprint);
            gpu.queue.write_buffer(&g.u_eval, 0, bytemuck::cast_slice(&uni));
        }
        let mut blocks: Vec<FixU> = Vec::new();
        let mut warps: Vec<u8> = Vec::new();
        let runs = bake.stages.iter().filter(|stage| matches!(stage, Stage::Warp(_))).count() as u64;
        let bytes = runs.saturating_mul(gf_core::mods::BAKED_STRIDE as u64);
        if bytes > gpu.device.limits().max_buffer_size.min(u32::MAX as u64) {
            return Err("the modifier stack exceeds this device's available buffer range".into());
        }
        let large = (runs > WARP_BLOCKS).then(|| gpu.uniform_buffer("long modifier stack", bytes));
        let warp_uniform = large.as_ref().unwrap_or(&g.u_mods);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sculpt") });

        // Where the field stands as the stages go: the pristine copy (untouched), the shape's own volume or
        // the spare one. A stage reads where it stands and writes the other of the two; its new distance
        // goes back the other way.
        #[derive(Clone, Copy, PartialEq)]
        enum At { Pristine, Vol, Tmp }
        let view = |a: At| match a { At::Pristine => &src.view, At::Vol => &vol.view, At::Tmp => &w.tmp.view };
        let other = |a: At| if a == At::Tmp { At::Vol } else { At::Tmp };
        let mut at = At::Pristine;
        let (mut cleaned, mut lab_ran) = (false, false);
        let redist = |enc: &mut wgpu::CommandEncoder, blocks: &mut Vec<FixU>, from: At, pre: f64, post: f64, slope: f64, reach_vox: f64| -> At {
            let to = other(from);
            distance(&gpu, g, enc, blocks, view(from), view(to), &w.seeds, res, h, pre, post, slope, reach_vox);
            to
        };
        for stage in &bake.stages {
            match stage {
                Stage::Warp(run) => {
                    // every voxel read through the run, from where the field stands
                    let to = other(at);
                    let at_block = (warps.len() / gf_core::mods::BAKED_STRIDE) as u64;
                    let mut nums = gf_core::mods::baked_numbers(run, res, HB, object);
                    if at == At::Pristine {
                        nums[gf_core::mods::BAKED_FLOATS - 4] = source_scale as f32;
                    }
                    let mut block = vec![0u8; gf_core::mods::BAKED_STRIDE];
                    block[..nums.len() * 4].copy_from_slice(bytemuck::cast_slice(&nums));
                    warps.extend(block);
                    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &g.l_mods,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: warp_uniform, offset: 0, size: wgpu::BufferSize::new((gf_core::mods::BAKED_FLOATS * 4) as u64) }) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view(at)) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&g.lin) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(view(to)) },
                        ],
                    });
                    {
                        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("modifiers"), timestamp_writes: None });
                        c.set_pipeline(&g.p_mods);
                        c.set_bind_group(0, &bg, &[(at_block * gf_core::mods::BAKED_STRIDE as u64) as u32]);
                        c.dispatch_workgroups(res / 8, res / 8, res / 4);
                    }
                    if let (Some(pipe),Some(face),Some(dims))=(&self.face_passes,faces.as_deref_mut(),&face_dims) {
                        let dims = if at == At::Pristine { &dims.0 } else { &dims.1 };
                        pipe.warp(&gpu,&mut enc,face,face_res,dims,view(at),&g.lin,warp_uniform,at_block*gf_core::mods::BAKED_STRIDE as u64);
                    }
                    // (all through the volume: beyond a band the warped field is divided by the warps' bound, which can be
                    // tens, and so small a number would read as the surface itself)
                    at = redist(&mut enc, &mut blocks, to, 0.0, 0.0, 1.0, res as f64);
                }
                Stage::Lab(_) => {
                    // 1. the stack, from where the field stands into the other volume
                    lab_ran = true;
                    let to = other(at);
                    let g_eval = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &g.l_eval,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: g.u_eval.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view(at)) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&g.lin) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&g.images.1) },
                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&g.wrap) },
                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(view(to)) },
                        ],
                    });
                    {
                        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sculpt layers"), timestamp_writes: None });
                        c.set_pipeline(&g.p_eval);
                        c.set_bind_group(0, &g_eval, &[]);
                        c.dispatch_workgroups(res / 8, res / 8, res / 4);
                    }
                    // 2. a true distance again; then the cleanups, each an offset and a new distance
                    at = redist(&mut enc, &mut blocks, to, 0.0, 0.0, s.slope(), BAND);
                    let open = 0.5 * s.wall;
                    if open > 0.0 {
                        // an opening: in by half the wall and out again. Walls thinner than that go, cleanly.
                        at = redist(&mut enc, &mut blocks, at, open, -open, 1.05, BAND + open / h);
                        cleaned = true;
                    }
                    if s.round > 0.0 {
                        // rounding: the distance smoothed over the radius in all three directions, then made a
                        // distance again. Edges come out round; slivers thinner than the radius melt away.
                        let rv = (s.round / h).round().clamp(1.0, 24.0) as u32;
                        let reach = BAND + rv as f64;
                        for axis in 0..3 {
                            blur(&gpu, g, &mut enc, &mut blocks, view(at), view(other(at)), &w.seeds, res, h, axis, rv, reach);
                            at = other(at);
                        }
                        at = redist(&mut enc, &mut blocks, at, 0.0, 0.0, 1.05, reach);
                        cleaned = true;
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
                        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &g.l_cc,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: g.u_cc.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view(at)) },
                                wgpu::BindGroupEntry { binding: 2, resource: labels.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: sizes.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(view(other(at))) },
                            ],
                        });
                        for (i, p) in g.p_cc.iter().enumerate() {
                            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sculpt pieces"), timestamp_writes: None });
                            c.set_pipeline(p);
                            c.set_bind_group(0, &bg, &[]);
                            if i == 4 { c.dispatch_workgroups(res / 8, res / 8, res / 4); } else { c.dispatch_workgroups(cres.div_ceil(4), cres.div_ceil(4), cres.div_ceil(4)); }
                        }
                        at = other(at);
                        at = redist(&mut enc, &mut blocks, at, 0.0, 0.0, 1.05, BAND);
                    }
                }
                Stage::Smooth(radius) => {
                    // the distance smoothed along each axis by its own radius, then made a distance again
                    if at == At::Pristine {
                        copy_volume(&mut enc, &src.tex, &vol.tex, res);
                        at = At::Vol;
                    }
                    let mut most = 0u32;
                    for axis in 0..3 {
                        let rv = (radius[axis] / h).round().clamp(0.0, 24.0) as u32;
                        if rv == 0 { continue; }
                        most = most.max(rv);
                        blur(&gpu, g, &mut enc, &mut blocks, view(at), view(other(at)), &w.seeds, res, h, axis as i32, rv, BAND + rv as f64);
                        at = other(at);
                    }
                    if most > 0 { at = redist(&mut enc, &mut blocks, at, 0.0, 0.0, 1.05, BAND + most as f64); }
                }
            }
        }
        match at {
            At::Tmp => copy_volume(&mut enc, &w.tmp.tex, &vol.tex, res),
            At::Pristine => copy_volume(&mut enc, &src.tex, &vol.tex, res),
            At::Vol => {}
        }
        if let (Some(pipe),Some(face),Some(dims))=(&self.face_passes,faces,&face_dims) {
            pipe.finish(&gpu,&mut enc,face,&dims.1,&vol.view,&g.lin);
        }
        if blocks.len() as u64 > BLOCKS { return Err("the surface studio asked for too many passes".into()); }
        let mut data = vec![0u8; blocks.len() * STRIDE as usize];
        for (i, b) in blocks.iter().enumerate() {
            data[i * STRIDE as usize..i * STRIDE as usize + std::mem::size_of::<FixU>()].copy_from_slice(bytemuck::bytes_of(b));
        }
        if !data.is_empty() { gpu.queue.write_buffer(&g.u_fix, 0, &data); }
        if !warps.is_empty() { gpu.queue.write_buffer(warp_uniform, 0, &warps); }
        gpu.queue.submit([enc.finish()]);
        if timed { gpu.wait(); }
        if let Some(a) = self.applied.get_mut(layer) { *a = format!("{key}#{}", bake.key(phases, self.footprint)); }
        self.last = Ran { res, ms: t0.elapsed().as_secs_f64() * 1e3, layers: if lab_ran { s.live().count() } else { 0 }, cleaned, pieces, size };
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
    let groups = (res / 8, res / 8, res / 4);
    let mut run = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, step: i32, read: usize, groups: (u32, u32, u32)| {
        let at = blocks.len() as u64 * STRIDE;
        blocks.push(FixU { step, ..base });
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sculpt distance"), timestamp_writes: None });
        c.set_pipeline(pipe);
        c.set_bind_group(0, &bg[read], &[at as u32]);
        c.dispatch_workgroups(groups.0, groups.1, groups.2);
    };
    if res <= 384 && reach_vox >= res as f64 && pre == 0.0 && post == 0.0 && slope > 0.0 {
        // Every fine sample contributes to an 8³ cell's signs. An exact separable
        // squared-distance transform on those cells bounds long steps in empty space.
        let cells = res.div_ceil(8);
        let coarse_groups = (cells.div_ceil(8), cells.div_ceil(8), cells.div_ceil(4));
        run(enc, &g.warp_distance[0], 0, 0, (cells, cells, cells));
        for axis in 0..3 {
            run(enc, &g.warp_distance[1], axis, (1 + axis as usize) % 2, coarse_groups);
        }
        run(enc, &g.warp_distance[2], 0, 0, groups);
        return;
    }
    // the flooding's steps: one of 1, halving from the first power of two at least half the band, then one more of 1
    let mut steps = vec![1];
    let mut k = ((reach_vox / 2.0).ceil() as u32).max(1).next_power_of_two();
    while k >= 1 {
        steps.push(k as i32);
        k /= 2;
    }
    steps.push(1);
    // seeds into set 1; each flooding step reads the set the last one wrote; the distance reads the last
    run(enc, &g.p_seed, 0, 0, groups);
    let mut read = 1;
    for s in steps {
        run(enc, &g.p_jfa, s, read, groups);
        read = 1 - read;
    }
    run(enc, &g.p_resolve, 0, read, groups);
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

#[cfg(test)]
mod tests {
    use super::*;
    use gf_core::sculpt::{Kind, Layer, Mode};

    fn validate_wgsl(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        v.validate(&module).unwrap_or_else(|e| panic!("{name} validation error:\n{}", e.emit_to_string(src)));
    }

    #[test]
    fn sculpt_shaders_validate_and_pack_coordinates_up_to_res_max() {
        validate_wgsl("sculpt.wgsl", EVAL_WGSL);
        let mut parts = FIX_WGSL.split("//#pieces");
        let dist = parts.next().unwrap();
        let pieces = format!("const NONE: u32 = 0xffffffffu;\n{}", parts.next().unwrap());
        validate_wgsl("sculpt_fix.wgsl (distance)", dist);
        validate_wgsl("sculpt_fix.wgsl (pieces)", &pieces);

        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        let head = dist.split("@compute").next().unwrap();
        let probe_wgsl = format!(
            "{head}\n@group(0) @binding(10) var<storage, read_write> probe: array<u32>;\n\
             @compute @workgroup_size(1)\n\
             fn cs_probe() {{\n\
               let pts = array<vec3i, 5>(\n\
                 vec3i(0, 0, 0),\n\
                 vec3i(1023, 1024, 1535),\n\
                 vec3i(1535, 1025, 1024),\n\
                 vec3i(1200, 1300, 1400),\n\
                 vec3i(1535, 1535, 1535),\n\
               );\n\
               for (var i = 0u; i < 5u; i++) {{\n\
                 let p = pack(pts[i]);\n\
                 probe[i] = select(0u, 1u, p != NONE && all(unpack(p) == pts[i]));\n\
               }}\n\
             }}\n"
        );
        validate_wgsl("sculpt_fix probe", &probe_wgsl);
        let module = gpu.shader("sculpt_fix probe", &probe_wgsl, false);
        let c = wgpu::ShaderStages::COMPUTE;
        let layout = gpu.bind_layout("sculpt_fix probe", &[
            uniform_entry(0, c, false),
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: c,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ]);
        let pipe = compute_pipe(&gpu, "sculpt_fix probe", &module, &layout, "cs_probe");
        let ubuf = gpu.uniform_buffer("sculpt_fix probe u", std::mem::size_of::<FixU>() as u64);
        let fu = FixU { res: gf_core::state::RES_MAX, step: 0, cres: 0, thresh: 0, h: 0.001, pre: 0.0, post: 0.0, reach: 0.01, slope: 1.0, hb: HB as f32, pad: [0.0; 2] };
        gpu.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&fu));
        let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe out"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe readback"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &readback, 0, 20);
        gpu.queue.submit([enc.finish()]);
        readback.slice(..20).map_async(wgpu::MapMode::Read, |_| {});
        gpu.wait();
        let view = readback.slice(..20).get_mapped_range().unwrap();
        let got: &[u32] = bytemuck::cast_slice(&view);
        assert_eq!(got, &[1, 1, 1, 1, 1], "linear voxel pack/unpack failed at RES_MAX = {}", gf_core::state::RES_MAX);
    }

    #[test]
    fn surface_lab_image_tiles_vertically_and_handles_zero_gradient() {
        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        if !gpu.storage16 { return; }

        let png_path = std::env::temp_dir().join(format!("gf_sculpt_tile_{}.png", std::process::id()));
        {
            let file = std::fs::File::create(&png_path).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 4, 4);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            // Top two rows white (1.0), bottom two rows black (0.0).
            let mut px = [0u8; 4 * 4 * 3];
            px[..2 * 4 * 3].fill(255);
            writer.write_image_data(&px).unwrap();
        }
        let path_str = png_path.to_string_lossy().into_owned();

        let mut sculptor = Sculptor::new(&gpu);

        // 1. Constant zero field -> gradient is identically zero everywhere inside the active band.
        // Triplanar (proj = 0) image displacement must remain finite (no 0/0 = NaN).
        let res_flat = 32u32;
        let vol_flat = Volume::new(&gpu, res_flat, true);
        let zeros = vec![half::f16::from_f32(0.0); (res_flat * res_flat * res_flat) as usize];
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &vol_flat.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&zeros),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(res_flat * 2), rows_per_image: Some(res_flat) },
            wgpu::Extent3d { width: res_flat, height: res_flat, depth_or_array_layers: res_flat },
        );
        let mut layer0 = Layer::of(Kind::Image);
        layer0.relative = false;
        layer0.scale = 0.35;
        layer0.depth = 0.05;
        layer0.mode = Mode::Raise;
        layer0.a = 0.0; // triplanar
        layer0.b = 1.0;
        layer0.image = path_str.clone();
        let mut lab0 = Sculpt::default();
        lab0.layers = vec![layer0];
        sculptor.keep(0, &vol_flat, "flat").unwrap();
        sculptor.run(0, &vol_flat, "flat", &Bake { stages: vec![Stage::Lab(lab0)] }, &[], true, true, &Default::default()).unwrap();
        let flat_slice = gpu.read(&vol_flat.tex, 0, 0, res_flat / 2, res_flat, res_flat, 2).unwrap();
        for h in bytemuck::cast_slice::<u8, half::f16>(&flat_slice) {
            assert!(h.to_f32().is_finite(), "triplanar image displacement produced NaN/Inf on zero-gradient voxel");
        }

        // 2. Planar YZ plane (d = x) with front planar projection (proj = 1, uv = s.xy * (1, -1) + 0.5).
        // Verify V repeats outside [0, 1] rather than clamping vertically.
        let res = 64u32;
        let vol = Volume::new(&gpu, res, true);
        let mut plane = Vec::with_capacity((res * res * res) as usize);
        for _iz in 0..res {
            for _iy in 0..res {
                for ix in 0..res {
                    let x = ((ix as f64 + 0.5) / res as f64 * 2.0 - 1.0) * HB;
                    plane.push(half::f16::from_f32(x as f32));
                }
            }
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &vol.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&plane),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(res * 2), rows_per_image: Some(res) },
            wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
        );
        let mut layer = Layer::of(Kind::Image);
        layer.relative = false;
        layer.scale = 0.4;
        layer.depth = 0.08;
        layer.mode = Mode::Raise;
        layer.a = 1.0; // planar front: v = -y / scale + 0.5
        layer.b = 2.0;
        layer.image = path_str;
        let mut lab = Sculpt::default();
        lab.layers = vec![layer];
        sculptor.keep(0, &vol, "plane").unwrap();
        sculptor.run(0, &vol, "plane", &Bake { stages: vec![Stage::Lab(lab)] }, &[], true, true, &Default::default()).unwrap();
        let slice = gpu.read(&vol.tex, 0, 0, res / 2, res, res, 2).unwrap();
        let f16s = bytemuck::cast_slice::<u8, half::f16>(&slice);
        let sample_y = |target_y: f64| -> f32 {
            let iy = (((target_y / HB + 1.0) * 0.5 * res as f64 - 0.5).round() as usize).clamp(0, res as usize - 1);
            let ix = (res / 2) as usize;
            f16s[iy * res as usize + ix].to_f32()
        };
        // Both pairs lie strictly outside the central v in [0, 1] tile (|y| > 0.2):
        // y = +0.50 -> v = -0.75 (fract 0.25, white -> raised), y = +0.30 -> v = -0.25 (fract 0.75, black -> unraised)
        // y = -0.30 -> v = +1.25 (fract 0.25, white -> raised), y = -0.50 -> v = +1.75 (fract 0.75, black -> unraised)
        let (d_pos_hi, d_pos_lo) = (sample_y(0.50), sample_y(0.30));
        let (d_neg_hi, d_neg_lo) = (sample_y(-0.30), sample_y(-0.50));
        let _ = std::fs::remove_file(&png_path);
        assert!(d_pos_hi < d_pos_lo - 0.03, "expected vertical repeat for v < 0: d(+0.50)={d_pos_hi}, d(+0.30)={d_pos_lo}");
        assert!(d_neg_hi < d_neg_lo - 0.03, "expected vertical repeat for v > 1: d(-0.30)={d_neg_hi}, d(-0.50)={d_neg_lo}");
    }
}

#[cfg(test)]
mod work_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn large_working_volumes_go_once_idle_small_ones_stay() {
        assert!(keep_work(768, Duration::from_millis(100)), "in use (an animation, a refinement)");
        assert!(!keep_work(768, WORK_IDLE), "the picture stands still: let go");
        assert!(!keep_work(1024, Duration::from_secs(60)));
        assert!(keep_work(256, Duration::from_secs(600)), "a drag's are kept");
    }
}
