//! Shapes into distance fields. Every shape, however it is defined, ends up as one 3D texture holding the
//! signed distance to its surface; the tracer then needs a single lookup per step, whether the shape is a
//! ball or a tangle. (bakeCurve and bakeField of the web version's forms.js.)
//!
//! The arithmetic is the web version's. What differs is how the work is handed to the GPU: the web page
//! draws one slice of the volume per pass (thousands of small passes); here a slab of slices is gathered
//! on one sheet in a single draw, and the volume is written by a compute shader, a whole slab (or, for an
//! equation, the whole volume) per dispatch. Where a compute shader cannot write a 16-bit float volume,
//! the volume is drawn one slice per pass instead.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use gf_core::curve::Curve;
use gf_core::presets::Preset;
use gf_core::{field, BAND, HB};

use crate::gpu::*;

pub const VOL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// A volume's size is a multiple of this: compute shaders work in blocks of 8 x 8 x 4 voxels.
pub const GROUP: u32 = 8;
/// Bytes between one slab's uniform block and the next.
const SLAB_STRIDE: u32 = 256;
const MAX_SLABS: u32 = 256;
const BAKE_WGSL: &str = include_str!("shaders/bake_curve.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BakeU {
    hb: f32,
    res: f32,
    band: f32,
    rmax: f32,
    k: f32,
    amp: f32,
    safety: f32,
    floor: f32,
    cres: f32,
    cloud_n: u32,
    slab: u32,
    tiles: u32,
    stiles: u32,
    melted: u32,
    pad: [u32; 2],
}

pub struct Volume {
    pub tex: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub res: u32,
}
impl Volume {
    /// An empty volume of `res` voxels a side; `storage`: one a compute shader can write.
    pub fn new(gpu: &Gpu, res: u32, storage: bool) -> Self {
        let tex = gpu.volume("volume", res, res, VOL_FORMAT, storage);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self { tex, view, res }
    }
    /// A volume holding these distances: voxel (i, j, k) at `data[(k * res + j) * res + i]`.
    pub fn from_data(gpu: &Gpu, res: u32, data: &[half::f16]) -> Self {
        let v = Self::new(gpu, res, false);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &v.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(res * 2), rows_per_image: Some(res) },
            wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
        );
        v
    }
}

/// The volumes kept for one layer, and which of them holds its shape now.
#[derive(Default)]
struct Kept {
    vols: Vec<Volume>,
    cur: usize,
}
impl Kept {
    /// A volume of this size to build into: one kept from before, or a new one. Big ones are not hoarded.
    fn take(&mut self, gpu: &Gpu, res: u32, storage: bool) -> usize {
        if let Some(i) = self.vols.iter().position(|v| v.res == res) {
            self.cur = i;
            return i;
        }
        // keep the small ones (an animation alternates between a small volume and the full one); of the
        // big ones only the newest
        if res > 384 {
            self.vols.retain(|v| v.res <= 384);
        }
        self.vols.push(Volume::new(gpu, res, storage));
        self.cur = self.vols.len() - 1;
        self.cur
    }
}

/// How far from a tube its wide volume holds the exact distance, in the tube's own units. Beyond, it
/// holds the distance to a thinned-out cloud of the tube's points: close, and smooth.
pub const WIDE_REACH: f64 = 0.12;

#[derive(Clone, Copy, Debug, Default)]
pub struct BakeStats {
    pub segments: usize,
    pub splats: usize,
}

/// The coarse far-field volume of one size, what reads it, and the sheet it is worked out on.
#[derive(Clone)]
struct Coarse {
    tex: wgpu::Texture,
    sheet: wgpu::Texture,
    sheet_view: wgpu::TextureView,
    view: wgpu::TextureView,
}

/// The sheet a slab of slices is gathered on, for one size of volume.
#[derive(Clone)]
struct Sheet {
    view: wgpu::TextureView,
    /// slices across the sheet, and slices gathered at a time
    tiles: u32,
    slab: u32,
    /// reads the sheet together with the coarse volume, when the volume is drawn
    group: wgpu::BindGroup,
}

/// What builds an equation's volume: a pass per slice, or one compute dispatch.
enum FieldPipe {
    Drawn(wgpu::RenderPipeline),
    Computed(wgpu::ComputePipeline),
}

/// What writes volumes by compute shader, where that is possible.
struct Computed {
    resolve: wgpu::ComputePipeline,
    /// coarse volume, sampler, sheet, and the volume written
    l_resolve: wgpu::BindGroupLayout,
    /// each slab's first slice and size, one uniform block per slab
    slabs: wgpu::Buffer,
    g_slabs: wgpu::BindGroup,
    l_field: wgpu::BindGroupLayout,
}

pub struct Baker {
    gpu: Gpu,
    lin: wgpu::Sampler,
    u: wgpu::Buffer,
    g0: wgpu::BindGroup,
    l_cloud: wgpu::BindGroupLayout,
    l_resolve: wgpu::BindGroupLayout,
    l_pts: wgpu::BindGroupLayout,
    p_coarse: wgpu::RenderPipeline,
    p_splat: wgpu::RenderPipeline,
    p_melt: wgpu::RenderPipeline,
    p_resolve: wgpu::RenderPipeline,
    pts: GrowBuf,
    cloud: GrowBuf,
    segs: GrowBuf,
    g_pts: wgpu::BindGroup,
    g_cloud: wgpu::BindGroup,
    coarse: HashMap<u32, Coarse>,
    /// by (voxels a side of the volume, of its coarse volume)
    sheets: Vec<((u32, u32), Sheet)>,
    sheet_format: wgpu::TextureFormat,
    l_field: wgpu::BindGroupLayout,
    field_u: wgpu::Buffer,
    g_field: wgpu::BindGroup,
    field_pipes: HashMap<String, FieldPipe>,
    computed: Option<Computed>,
    /// volumes kept for reuse, one set for each layer, and the layer that is being built
    kept: Vec<Kept>,
    layer: usize,
    /// each layer's wide volume, where it has one (see `curve_wide`)
    wide: Vec<Kept>,
}

impl Baker {
    /// What is kept between builds, counted (for the window's diagnostic log).
    pub fn kept(&self) -> String {
        format!("coarse={} sheets={} fields={} vols={}", self.coarse.len(), self.sheets.len(), self.field_pipes.len(), self.kept.iter().chain(&self.wide).map(|k| k.vols.len()).sum::<usize>())
    }

    pub fn new(gpu: &Gpu) -> Self {
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let all = vf | wgpu::ShaderStages::COMPUTE;
        // the part of the shader that draws; the rest needs volumes that compute shaders can write
        let drawn = BAKE_WGSL.split("//#compute").next().unwrap();
        let module = gpu.shader("bake curve", drawn, false);
        let l0 = gpu.bind_layout("bake params", &[uniform_entry(0, all, false)]);
        let l_cloud = gpu.bind_layout("bake cloud", &[storage_entry(0, wgpu::ShaderStages::FRAGMENT)]);
        let l_resolve = gpu.bind_layout("bake resolve", &[texture_entry(1, wgpu::TextureViewDimension::D3, true), sampler_entry(2), texture_entry(3, wgpu::TextureViewDimension::D2, false)]);
        let l_pts = gpu.bind_layout("bake points", &[storage_entry(4, wgpu::ShaderStages::VERTEX)]);
        // the melt's sums need 32 bits; 16 will do, coarsely, where 32-bit targets cannot be blended into
        let sheet_format = if gpu.blend32 { wgpu::TextureFormat::R32Float } else { wgpu::TextureFormat::R16Float };
        // one instance = one segment in one slice
        let seg_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 4, shader_location: 1 },
            ],
        })];
        let one = wgpu::BlendFactor::One;
        let blend = |operation| Some(wgpu::BlendState { color: wgpu::BlendComponent { src_factor: one, dst_factor: one, operation }, alpha: wgpu::BlendComponent { src_factor: one, dst_factor: one, operation } });
        let p_coarse = gpu.pipeline(&PipeDesc { label: "bake coarse", module: &module, vs: "vs_full", fs: "fs_coarse", layouts: &[&l0, &l_cloud], format: VOL_FORMAT, blend: None, strip: false, buffers: &[] });
        let seg_pipe = |label, fs, op| gpu.pipeline(&PipeDesc { label, module: &module, vs: "vs_splat", fs, layouts: &[&l0, &l_pts], format: sheet_format, blend: blend(op), strip: true, buffers: &seg_layout });
        let p_splat = seg_pipe("bake splat", "fs_splat", wgpu::BlendOperation::Min);
        let p_melt = seg_pipe("bake melt", "fs_melt", wgpu::BlendOperation::Add);
        let p_resolve = gpu.pipeline(&PipeDesc { label: "bake resolve", module: &module, vs: "vs_full", fs: "fs_resolve", layouts: &[&l0, &l_resolve], format: VOL_FORMAT, blend: None, strip: false, buffers: &[] });

        let u = gpu.uniform_buffer("bake params", std::mem::size_of::<BakeU>() as u64);
        let g0 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l0, entries: &[wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }] });
        let pts = GrowBuf::new(gpu, "curve points", wgpu::BufferUsages::STORAGE, 1 << 20);
        let cloud = GrowBuf::new(gpu, "curve cloud", wgpu::BufferUsages::STORAGE, 4096 * 16);
        let segs = GrowBuf::new(gpu, "segment lists", wgpu::BufferUsages::VERTEX, 1 << 20);
        let g_pts = Self::buf_group(gpu, &l_pts, 4, &pts.buf);
        let g_cloud = Self::buf_group(gpu, &l_cloud, 0, &cloud.buf);

        let l_field = gpu.bind_layout("field params", &[uniform_entry(0, vf, false)]);
        let field_u = gpu.uniform_buffer("field params", field::UNIFORM_SIZE as u64);
        let g_field = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_field, entries: &[wgpu::BindGroupEntry { binding: 0, resource: field_u.as_entire_binding() }] });

        // writing volumes by compute shader
        let computed = gpu.storage16.then(|| {
            let c = wgpu::ShaderStages::COMPUTE;
            let tex = |binding, view_dimension, filterable| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension, multisampled: false }, count: None };
            let out = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: VOL_FORMAT, view_dimension: wgpu::TextureViewDimension::D3 }, count: None };
            let l_resolve = gpu.bind_layout("bake resolve (computed)", &[
                tex(1, wgpu::TextureViewDimension::D3, true),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: c, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                tex(3, wgpu::TextureViewDimension::D2, false),
                out(5),
            ]);
            let l_slabs = gpu.bind_layout("bake slab", &[uniform_entry(0, c, true)]);
            let slabs = gpu.uniform_buffer("bake slabs", (SLAB_STRIDE * MAX_SLABS) as u64);
            let g_slabs = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &l_slabs,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &slabs, offset: 0, size: wgpu::BufferSize::new(16) }) }],
            });
            let whole = gpu.shader("bake curve (computed)", BAKE_WGSL, false);
            let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("bake resolve (computed)"), bind_group_layouts: &[Some(&l0), Some(&l_resolve), Some(&l_slabs)], immediate_size: 0 });
            let resolve = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("bake resolve (computed)"), layout: Some(&layout), module: &whole, entry_point: Some("cs_resolve"), compilation_options: Default::default(), cache: None });
            let l_field = gpu.bind_layout("field (computed)", &[uniform_entry(0, c, false), out(1)]);
            Computed { resolve, l_resolve, slabs, g_slabs, l_field }
        });
        Self {
            gpu: gpu.clone(), lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge), u, g0,
            l_cloud, l_resolve, l_pts, p_coarse, p_splat, p_melt, p_resolve,
            pts, cloud, segs, g_pts, g_cloud, coarse: HashMap::new(), sheets: Vec::new(), sheet_format,
            l_field, field_u, g_field, field_pipes: HashMap::new(), computed, kept: vec![Kept::default()], layer: 0, wide: Vec::new(),
        }
    }

    fn buf_group(gpu: &Gpu, layout: &wgpu::BindGroupLayout, binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &[wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }] })
    }

    /// The volume holding the shape now (of the layer being built: the only one, unless there are layers).
    pub fn volume(&self) -> Option<&Volume> {
        self.layer_volume(self.layer)
    }
    /// The volume holding a layer's shape now.
    pub fn layer_volume(&self, layer: usize) -> Option<&Volume> {
        self.kept.get(layer).and_then(|k| k.vols.get(k.cur))
    }
    /// Build into this layer's volumes from now on. Every layer keeps its own, so that one can be built
    /// again while the others stay as they are.
    pub fn set_layer(&mut self, layer: usize) {
        while self.kept.len() <= layer {
            self.kept.push(Kept::default());
        }
        self.layer = layer;
    }
    /// Let go of the volumes of the layers from this one on (they have been taken away).
    pub fn drop_layers(&mut self, from: usize) {
        self.kept.truncate(from.max(1));
        self.wide.truncate(from.max(1));
        self.layer = self.layer.min(self.kept.len() - 1);
    }
    /// A layer's wide volume, if one has been built for it.
    pub fn wide_volume(&self, layer: usize) -> Option<&Volume> {
        self.wide.get(layer).and_then(|k| k.vols.get(k.cur))
    }
    /// Forget a layer's wide volume (its shape is no longer a tube, or no seam reaches that far).
    pub fn drop_wide(&mut self, layer: usize) {
        if let Some(k) = self.wide.get_mut(layer) { k.vols.clear(); }
    }

    /// A volume of `res` voxels a side for the layer being built, for a shape that is built outside
    /// this file (a polygon model): the view to write it through, and whether a compute shader can.
    pub fn target(&mut self, res: u32) -> (wgpu::TextureView, bool) {
        let storage = self.computed.is_some();
        let k = &mut self.kept[self.layer];
        let vi = k.take(&self.gpu, res, storage);
        (k.vols[vi].view.clone(), storage)
    }

    /// How many slices across the sheet the coarse volume of this size is worked out on.
    fn coarse_tiles(cres: u32) -> u32 {
        (cres as f64).sqrt().ceil() as u32
    }

    fn coarse_for(&mut self, cres: u32) -> Coarse {
        if let Some(c) = self.coarse.get(&cres) {
            return c.clone();
        }
        let tex = self.gpu.volume("coarse", cres, cres, VOL_FORMAT, false);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let side = Self::coarse_tiles(cres) * cres;
        let sheet = self.gpu.target("coarse sheet", side, side, VOL_FORMAT, &[]);
        let sheet_view = sheet.create_view(&wgpu::TextureViewDescriptor::default());
        let c = Coarse { tex, sheet, sheet_view, view };
        self.coarse.insert(cres, c.clone());
        c
    }

    /// The sheet slabs are gathered on, for a volume of this size (read together with a coarse volume of
    /// that size). The small ones are kept; of the large ones only the newest.
    fn sheet_for(&mut self, res: u32, cres: u32, coarse: &wgpu::TextureView) -> R<Sheet> {
        if let Some((_, s)) = self.sheets.iter().find(|(k, _)| *k == (res, cres)) {
            return Ok(s.clone());
        }
        // as many slices across as the largest texture allows, eight at most; a whole number of eights to a slab
        let tiles = (self.gpu.device.limits().max_texture_dimension_2d / res).min(8);
        let slab = tiles * tiles / GROUP * GROUP;
        if slab == 0 || res.div_ceil(slab.max(1)) > MAX_SLABS {
            return Err(format!("a volume of {res} voxels a side is too large for this graphics device"));
        }
        let side = tiles * res;
        let view = self.gpu.target("gather sheet", side, side, self.sheet_format, &[]).create_view(&wgpu::TextureViewDescriptor::default());
        let group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.l_resolve,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(coarse) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        let s = Sheet { view, tiles, slab, group };
        if res > 384 { self.sheets.retain(|(k, _)| k.0 <= 384); }
        self.sheets.push(((res, cres), s.clone()));
        Ok(s)
    }

    /// A curve: `c.pts` are x,y,z,radius; `c.brk[i]` = 1 means no segment from point i to the next.
    /// melt: strands closer than about this flow into each other (0 = hard union).
    /// `res` must be a multiple of eight.
    pub fn curve(&mut self, c: &Curve, res: u32, melt: f64) -> R<BakeStats> {
        self.build_curve(c, res, melt, false)
    }

    /// The same tube into the layer's wide volume: a small volume (`res` voxels a side) in which the
    /// distance is exact out to `WIDE_REACH` from the tube rather than a few voxels, and close to exact
    /// beyond (the distance to the thinned-out cloud of its points, with nothing taken off for safety).
    /// A seam between layers reads a tube's distance well away from the tube; the tube's own volume holds
    /// only a safe lower bound out there, with steps in it, which would show in the seam.
    pub fn curve_wide(&mut self, c: &Curve, res: u32, melt: f64) -> R<BakeStats> {
        // a melt too slight to matter at this distance is left out (its sums would not reach)
        let melt = if melt >= 0.003 { melt } else { 0.0 };
        // the tube's own points are closer together than this small volume can use: every few will do
        let h = 2.0 * HB / res as f64;
        let stride = ((0.75 * h / c.ds).floor() as usize).max(1);
        if stride == 1 { return self.build_curve(c, res, melt, true); }
        let (mut pts, mut brk) = (Vec::with_capacity(c.n / stride * 4 + 64), Vec::with_capacity(c.n / stride + 16));
        let mut first = 0;
        while first < c.n {
            // one strand: from `first` to the point after which there is no segment
            let last = (first..c.n).find(|&i| c.brk[i] != 0).unwrap_or(c.n - 1);
            let mut i = first;
            loop {
                pts.extend_from_slice(&c.pts[i * 4..i * 4 + 4]);
                brk.push(0u8);
                if i == last { break; }
                i = (i + stride).min(last);
            }
            *brk.last_mut().unwrap() = 1;
            first = last + 1;
        }
        // with a melt: what one straight strand cut into these longer steps sums to on its own surface
        let ds = c.ds * stride as f64;
        let mut amp = 1.0;
        if melt > 0.0 {
            let (k, r) = (1.0 / melt, c.rmax);
            let mut sum = 1.0;
            for m in 1..6000 {
                let t = (-k * ((r * r + ((m as f64 - 0.5) * ds).powi(2)).sqrt() - r)).exp();
                sum += 2.0 * t;
                if t < 1e-8 { break; }
            }
            amp = 1.0 / sum;
        }
        let n = brk.len();
        let thin = Curve { pts, brk, n, rmax: c.rmax, ds, length: c.length, amp, pose: c.pose.clone() };
        self.build_curve(&thin, res, melt, true)
    }

    fn build_curve(&mut self, c: &Curve, res: u32, melt: f64, wide: bool) -> R<BakeStats> {
        if res % GROUP != 0 { return Err(format!("a volume's size must be a multiple of {GROUP}, not {res}")); }
        let gpu = self.gpu.clone();
        let n = c.n;
        // the points, and a thinned-out cloud of them for the far field
        if self.pts.write(&gpu, bytemuck::cast_slice(&c.pts[..n * 4])) {
            self.g_pts = Self::buf_group(&gpu, &self.l_pts, 4, &self.pts.buf);
        }
        let stride = n.div_ceil(4096).max(1);
        let cn = n.div_ceil(stride);
        let mut cd = vec![0f32; 4096 * 4];
        for i in 0..cn {
            let s = (i * stride).min(n - 1) * 4;
            cd[i * 4..i * 4 + 3].copy_from_slice(&c.pts[s..s + 3]);
        }
        if self.cloud.write(&gpu, bytemuck::cast_slice(&cd)) {
            self.g_cloud = Self::buf_group(&gpu, &self.l_cloud, 0, &self.cloud.buf);
        }

        // (a wide volume leans on its far field, so that is worked out more finely)
        let cres = if wide { (res >> 1).clamp(48, 96) } else { (res >> 3).max(32) };
        let coarse = self.coarse_for(cres);
        let sheet = self.sheet_for(res, cres, &coarse.view)?;
        let h = 2.0 * HB / res as f64;
        // Melted: a bundle of n strands shares a surface about melt*ln(n) outside each strand's own, and the
        // smooth lookup reaches 2 voxels further. Contributions are exact out to `clear` and fade out by `band`.
        // (wide: the melt's sums can be turned back into distances only out to some 18 melts)
        let clear = if wide { WIDE_REACH.min(17.0 * melt) } else { 2.2 * melt + 2.5 * h };
        let melted = melt > 0.0;
        let band = if melted { clear / 0.65 } else if wide { WIDE_REACH } else { BAND * h };
        // (a wide volume is not marched through: its far field is kept as close to the truth as it comes)
        let safety = if wide { 0.0 } else { 0.9 * (2.0 * HB / cres as f64) + 0.5 * stride as f64 * c.ds + 1e-3 };
        let u = BakeU {
            hb: HB as f32, res: res as f32, band: band as f32, rmax: c.rmax as f32,
            k: if melted { (1.0 / melt) as f32 } else { 0.0 }, amp: c.amp as f32, safety: safety as f32,
            // far field: never below the band's edge; melted: never below where sums are still exact
            floor: if melted { (clear - 0.5 * h) as f32 } else { band as f32 },
            cres: cres as f32, cloud_n: cn as u32, slab: sheet.slab, tiles: Self::coarse_tiles(cres), stiles: sheet.tiles, melted: melted as u32, pad: [0; 2],
        };
        gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&u));

        // which segments reach which slice
        let p = &c.pts;
        let mut count = vec![0u32; res as usize + 1];
        let mut k0 = vec![1i32; n];
        let mut k1 = vec![0i32; n];
        let mut total = 0usize;
        for i in 0..n.saturating_sub(1) {
            if c.brk[i] != 0 { continue; }
            let (za, zb) = (p[i * 4 + 2] as f64, p[i * 4 + 6] as f64);
            let reach = (p[i * 4 + 3] as f64).max(p[i * 4 + 7] as f64) + band;
            let a = ((((za.min(zb) - reach) / HB + 1.0) * 0.5 * res as f64 - 0.5).ceil() as i32).max(0);
            let b = ((((za.max(zb) + reach) / HB + 1.0) * 0.5 * res as f64 - 0.5).floor() as i32).min(res as i32 - 1);
            k0[i] = a;
            k1[i] = b;
            for k in a..=b { count[k as usize] += 1; }
            total += (b - a + 1).max(0) as usize;
        }
        let mut start = vec![0u32; res as usize + 1];
        for k in 0..res as usize { start[k + 1] = start[k] + count[k]; }
        let mut fill: Vec<u32> = start[..res as usize].to_vec();
        // (segment, slice) pairs, in order of slice
        let mut list = vec![[0u32; 2]; total.max(1)];
        for i in 0..n.saturating_sub(1) {
            for k in k0[i]..=k1[i] {
                list[fill[k as usize] as usize] = [i as u32, k as u32];
                fill[k as usize] += 1;
            }
        }
        self.segs.write(&gpu, bytemuck::cast_slice(&list));

        let storage = self.computed.is_some();
        let vol_view = if wide {
            while self.wide.len() <= self.layer {
                self.wide.push(Kept::default());
            }
            let k = &mut self.wide[self.layer];
            let vi = k.take(&gpu, res, storage);
            k.vols[vi].view.clone()
        } else {
            let k = &mut self.kept[self.layer];
            let vi = k.take(&gpu, res, storage);
            k.vols[vi].view.clone()
        };
        // computed: what each slab's dispatch reads and writes, and each slab's first slice and size
        let g_out = self.computed.as_ref().map(|cp| {
            let mut data = vec![0u8; (SLAB_STRIDE * res.div_ceil(sheet.slab)) as usize];
            for (i, z0) in (0..res).step_by(sheet.slab as usize).enumerate() {
                let at = i * SLAB_STRIDE as usize;
                data[at..at + 4].copy_from_slice(&z0.to_le_bytes());
                data[at + 4..at + 8].copy_from_slice(&sheet.slab.min(res - z0).to_le_bytes());
            }
            gpu.queue.write_buffer(&cp.slabs, 0, &data);
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &cp.l_resolve,
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&coarse.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sheet.view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&vol_view) },
                ],
            })
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake curve") });
        // 1. far field: distance to the cloud at low resolution, every slice at once on one sheet, then each
        // slice copied to its place in the coarse volume
        {
            let mut rp = pass(&mut enc, &coarse.sheet_view, None, true);
            rp.set_pipeline(&self.p_coarse);
            rp.set_bind_group(0, &self.g0, &[]);
            rp.set_bind_group(1, &self.g_cloud, &[]);
            rp.draw(0..3, 0..1);
        }
        let tiles = Self::coarse_tiles(cres);
        for z in 0..cres {
            enc.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: &coarse.sheet, mip_level: 0, origin: wgpu::Origin3d { x: (z % tiles) * cres, y: (z / tiles) * cres, z: 0 }, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &coarse.tex, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z }, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: cres, height: cres, depth_or_array_layers: 1 },
            );
        }
        // 2 and 3, a slab of slices at a time: gather the segments on the sheet, then write the volume.
        // Melted: every segment adds exp(k (r - d)), and the sum is turned back into a distance.
        // Hard union: every segment writes its exact distance, keeping the minimum.
        let empty = if melted { wgpu::Color::TRANSPARENT } else { wgpu::Color { r: 60000.0, g: 0.0, b: 0.0, a: 0.0 } };
        let mut z0 = 0;
        while z0 < res {
            let z1 = (z0 + sheet.slab).min(res);
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &sheet.view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(empty), store: wgpu::StoreOp::Store } })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let (a, b) = (start[z0 as usize], start[z1 as usize]);
                if b > a {
                    rp.set_pipeline(if melted { &self.p_melt } else { &self.p_splat });
                    rp.set_bind_group(0, &self.g0, &[]);
                    rp.set_bind_group(1, &self.g_pts, &[]);
                    rp.set_vertex_buffer(0, self.segs.buf.slice(..));
                    rp.draw(0..4, a..b);
                }
            }
            match (&self.computed, &g_out) {
                (Some(cp), Some(g_out)) => {
                    let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    c.set_pipeline(&cp.resolve);
                    c.set_bind_group(0, &self.g0, &[]);
                    c.set_bind_group(1, g_out, &[]);
                    c.set_bind_group(2, &cp.g_slabs, &[z0 / sheet.slab * SLAB_STRIDE]);
                    c.dispatch_workgroups(res / 8, res / 8, (z1 - z0).div_ceil(4));
                }
                _ => {
                    for z in z0..z1 {
                        let mut rp = pass(&mut enc, &vol_view, Some(z), true);
                        rp.set_pipeline(&self.p_resolve);
                        rp.set_bind_group(0, &self.g0, &[]);
                        rp.set_bind_group(1, &sheet.group, &[]);
                        rp.draw(0..3, z..z + 1);
                    }
                }
            }
            z0 = z1;
        }
        gpu.queue.submit([enc.finish()]);
        Ok(BakeStats { segments: n.saturating_sub(1), splats: total })
    }

    /// An equation: every voxel walks to the surface and stores how far it went.
    /// `res` must be a multiple of eight.
    pub fn field(&mut self, key: &str, p: &Preset, uniforms: &[f32; 12], res: u32) -> R<BakeStats> {
        if res % GROUP != 0 { return Err(format!("a volume's size must be a multiple of {GROUP}, not {res}")); }
        let gpu = self.gpu.clone();
        if !self.field_pipes.contains_key(key) {
            let pipe = match &self.computed {
                Some(cp) => {
                    let module = gpu.shader(key, &field::bake_shader_compute(p), false);
                    let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(key), bind_group_layouts: &[Some(&cp.l_field)], immediate_size: 0 });
                    FieldPipe::Computed(gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(key), layout: Some(&layout), module: &module, entry_point: Some("cs"), compilation_options: Default::default(), cache: None }))
                }
                None => {
                    let module = gpu.shader(key, &field::bake_shader(p), false);
                    FieldPipe::Drawn(gpu.pipeline(&PipeDesc { label: key, module: &module, vs: "vs", fs: "fs", layouts: &[&self.l_field], format: VOL_FORMAT, blend: None, strip: false, buffers: &[] }))
                }
            };
            self.field_pipes.insert(key.to_string(), pipe);
        }
        gpu.queue.write_buffer(&self.field_u, 0, bytemuck::cast_slice(uniforms));
        let storage = self.computed.is_some();
        let k = &mut self.kept[self.layer];
        let vi = k.take(&gpu, res, storage);
        let view = k.vols[vi].view.clone();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake field") });
        match (&self.field_pipes[key], &self.computed) {
            (FieldPipe::Computed(pipe), Some(cp)) => {
                // the whole volume in one dispatch
                let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cp.l_field,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.field_u.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) }],
                });
                let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                c.set_pipeline(pipe);
                c.set_bind_group(0, &g, &[]);
                c.dispatch_workgroups(res / 8, res / 8, res / 4);
            }
            (FieldPipe::Drawn(pipe), _) => {
                // one slice per pass
                for z in 0..res {
                    let mut rp = pass(&mut enc, &view, Some(z), true);
                    rp.set_pipeline(pipe);
                    rp.set_bind_group(0, &self.g_field, &[]);
                    rp.draw(0..3, z..z + 1);
                }
            }
            _ => unreachable!(),
        }
        gpu.queue.submit([enc.finish()]);
        Ok(BakeStats::default())
    }
}
