//! Layers on the GPU: each layer's shape is built into its own volume by the baker, exactly as a single
//! shape is, and a combining step writes the one volume the tracer reads (shaders/combine.wgsl, the
//! arithmetic of gf_core::layers).
//!
//! A layer's volume stays good while only its placement or the way it combines changes, so those cost
//! the combining step alone; a layer is built again only when its own shape changes.

#[path="world_fields.rs"]
mod world_fields;
#[path = "staged_layers.rs"]
mod staging;

use std::time::Instant;

use gf_core::layers::{Combine, Pass, Placement, WideUse, COMBINE_WIDTH as W, MAX_LAYERS, UNIFORM_FLOATS};
use gf_core::math::euler;
use gf_core::state::{Detail, Job, ShapeJob};
use gf_core::HB;

use crate::bake::{BakeStats, Volume, VOL_FORMAT};
use crate::engine::Engine;
use crate::gpu::*;

const COMBINE_WGSL: &str = include_str!("shaders/combine.wgsl");

/// The bodies' volume: a distance a voxel for each body. Two channels when there are two bodies, four
/// when there are three or four (there is no three-channel format).
pub fn bodies_format(channels: u32) -> wgpu::TextureFormat {
    if channels == 2 { wgpu::TextureFormat::Rg16Float } else { wgpu::TextureFormat::Rgba16Float }
}

/// The pipelines of a combining shader (`text`): computed where the device can write 16-bit volumes
/// from compute shaders, drawn a slice at a time otherwise.
fn make_pipe(gpu: &Gpu, text: &str, l_in: &wgpu::BindGroupLayout) -> Pipe {
    if gpu.storage16 {
        let c = wgpu::ShaderStages::COMPUTE;
        let out = |binding, format| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: c,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: wgpu::TextureViewDimension::D3 },
            count: None,
        };
        let l_scene = gpu.bind_layout("combine (computed)", &[out(0, VOL_FORMAT)]);
        let l_four = gpu.bind_layout("combine with bodies (computed)", &[out(0, VOL_FORMAT), out(1, bodies_format(4))]);
        // the last part of the shader writes a two-channel volume, which not every device can
        let two_ok = can_store(gpu, bodies_format(2));
        let text = if two_ok { text } else { text.split("//#compute2").next().unwrap() };
        let module = gpu.shader("combine (computed)", text, false);
        let make = |label, l_out: &wgpu::BindGroupLayout, entry| {
            let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(l_in), Some(l_out)], immediate_size: 0 });
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let scene = make("combine (computed)", &l_scene, "cs");
        let four = Both { pipe: make("combine with bodies (computed)", &l_four, "cs_bodies"), layout: l_four, binding: 1 };
        let two = two_ok.then(|| {
            let l_two = gpu.bind_layout("combine with two bodies (computed)", &[out(0, VOL_FORMAT), out(2, bodies_format(2))]);
            Both { pipe: make("combine with two bodies (computed)", &l_two, "cs_bodies2"), layout: l_two, binding: 2 }
        });
        Pipe::Computed { scene, l_scene, four, two }
    } else {
        // the part of the shader that draws
        let module = gpu.shader("combine", text.split("//#compute").next().unwrap(), false);
        let make = |label, fs, format| gpu.pipeline(&PipeDesc { label, module: &module, vs: "vs", fs, layouts: &[l_in], format, blend: None, strip: false, buffers: &[] });
        Pipe::Drawn { scene: make("combine", "fs", VOL_FORMAT), four: make("combine bodies", "fs_bodies", bodies_format(4)), two: make("combine two bodies", "fs_bodies", bodies_format(2)) }
    }
}

/// What writes the scene's volume with the bodies' beside it, by compute shader.
struct Both {
    pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// where the bodies' volume is bound
    binding: u32,
}

/// What writes the combined volume: one compute dispatch, or a pass per slice. Each in three kinds: the
/// scene's volume alone, and the scene's with the bodies' beside it in four channels or in two.
enum Pipe {
    Computed { scene: wgpu::ComputePipeline, l_scene: wgpu::BindGroupLayout, four: Both, two: Option<Both> },
    Drawn { scene: wgpu::RenderPipeline, four: wgpu::RenderPipeline, two: wgpu::RenderPipeline },
}

/// Can a compute shader write a volume of this format on this device? Asked by trying a tiny one.
fn can_store(gpu: &Gpu, format: wgpu::TextureFormat) -> bool {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _ = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe"),
        size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 8 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    pollster::block_on(scope.pop()).is_none()
}

pub struct Combiner {
    gpu: Gpu,
    lin: wgpu::Sampler,
    u: wgpu::Buffer,
    l_in: wgpu::BindGroupLayout,
    pipe: Pipe,
    /// stands in for the layers there are not
    blank: Volume,
    instance_table: crate::gpu::GrowBuf,
    /// volumes kept for reuse, and which of them holds the scene now
    vols: Vec<Volume>,
    cur: usize,
    /// the bodies' volume, while there is more than one body to tell apart, and how many channels it has
    bodies: Option<Volume>,
    bodies_channels: u32,
    /// One spare: the bodies' volume shown before this one (`keep_sizes` says which sizes may stay).
    bodies_spare: Option<(Volume, u32)>,
    /// Make the bodies' volume when more than one body shows. Whoever gives bodies their materials can
    /// switch this off while every body has the same one (nothing then reads it, and it is two or four
    /// volumes of memory), and must ask for the volume to be made again (`State::request`) on switching it on.
    pub bodies_wanted: bool,
    /// the combining step in passes, for scenes of more than COMBINE_WIDTH layers
    multi: Option<Multi>,
}

impl Combiner {
    pub fn new(gpu: &Gpu) -> Self {
        let vis = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
            count: None,
        };
        let l_in = gpu.bind_layout("combine", &[
            uniform_entry(0, vis | wgpu::ShaderStages::VERTEX, false),
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: vis, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            tex(2), tex(3), tex(4), tex(5), tex(6), tex(7), tex(8), tex(9),
            wgpu::BindGroupLayoutEntry { binding: 10, visibility: vis, ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None
            }, count: None },
        ]);
        let pipe = make_pipe(gpu, COMBINE_WGSL, &l_in);
        Self {
            gpu: gpu.clone(),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            u: gpu.uniform_buffer("combine", ((UNIFORM_FLOATS + 4) * 4) as u64),
            l_in, pipe,
            blank: Volume::new(gpu, 8, false),
            instance_table: crate::gpu::GrowBuf::new(gpu, "world occurrences", wgpu::BufferUsages::STORAGE, 64),
            vols: Vec::new(), cur: 0, bodies: None, bodies_channels: 0, bodies_spare: None, bodies_wanted: true, multi: None,
        }
    }

    /// Bytes of the volumes kept beside the ones in use: combined volumes of other sizes, the spare bodies'.
    pub(crate) fn spare_bytes(&self) -> u64 {
        let vox = |r: u32| (r as u64).pow(3) * 2;
        let vols: u64 = self.vols.iter().enumerate().filter(|(i, _)| *i != self.cur).map(|(_, v)| vox(v.res)).sum();
        vols + self.bodies_spare.as_ref().map_or(0, |(v, c)| vox(v.res) * u64::from(*c))
    }
    /// Let go of every volume kept beside the ones in use.
    pub(crate) fn drop_spares(&mut self) {
        if let Some(i) = Some(self.cur).filter(|&i| i < self.vols.len()) {
            let v = self.vols.swap_remove(i);
            self.vols = vec![v];
            self.cur = 0;
        }
        self.bodies_spare = None;
    }

    /// Let go of the kept volumes (and the spare bodies' volume) whose size is none of `sizes`, the sizes
    /// the scene is built at now (settled and drafted). The ones shown stay. Between a draft and a settled
    /// build each keeps the other's volumes, which are then written again rather than made anew: a new
    /// volume costs the GPU a clearing of all its memory before its first use (tens of milliseconds at 1024).
    pub(crate) fn keep_sizes(&mut self, sizes: &[u32]) {
        // (one volume of a size at most: `take` sees to it)
        let cur = self.vols.get(self.cur).map(|v| v.res);
        self.vols.retain(|v| v.res <= 384 || sizes.contains(&v.res) || Some(v.res) == cur);
        self.cur = cur.and_then(|r| self.vols.iter().position(|v| v.res == r)).unwrap_or(0);
        if self.bodies_spare.as_ref().is_some_and(|(v, _)| v.res > 384 && !sizes.contains(&v.res)) {
            self.bodies_spare = None;
        }
    }

    /// The combined volume.
    pub fn volume(&self) -> Option<&Volume> {
        self.vols.get(self.cur)
    }
    pub(crate) fn take_output(&mut self) -> Volume { self.vols.swap_remove(self.cur) }

    /// The bodies' volume: the same size as the combined volume, channel k the distance to body k. With
    /// two bodies it has two channels (`Rg16Float`: read as a texture, its third and fourth read 0 and 1
    /// and mean nothing); with three or four, four (`Rgba16Float`, a body that does not exist reading 4,
    /// the most a volume stores). The combined volume is, voxel for voxel, the smallest of the channels
    /// that stand for bodies. None unless more than one body shows.
    pub fn bodies_volume(&self) -> Option<&Volume> {
        self.bodies.as_ref()
    }
    /// How many channels the bodies' volume has (2 or 4; 0 when there is none).
    pub fn bodies_channels(&self) -> u32 {
        if self.bodies.is_some() { self.bodies_channels } else { 0 }
    }
    /// Let go of every volume (there are no layers to combine any more).
    pub fn clear(&mut self) {
        self.vols.clear();
        self.bodies = None;
        self.bodies_spare = None;
    }

    /// A volume of this size to write: one kept from before, or a new one. Big ones are not hoarded: one
    /// big one is kept beside the one asked for (the full volume while a handle is dragged or the scene
    /// plays at a coarser size: making a volume of a gigabyte or two again costs tens of milliseconds).
    fn take(&mut self, res: u32) -> R<usize> {
        if let Some(i) = self.vols.iter().position(|v| v.res == res) {
            self.cur = i;
            return Ok(i);
        }
        if res > 384 {
            let keep = self.vols.iter().map(|v| v.res).filter(|&r| r > 384).max();
            self.vols.retain(|v| v.res <= 384 || Some(v.res) == keep);
        }
        self.vols.push(Volume::try_new(&self.gpu, res, matches!(self.pipe, Pipe::Computed { .. }))?);
        self.cur = self.vols.len() - 1;
        Ok(self.cur)
    }

    /// Combine the layers' volumes (`layers[i]` is layer i's) into a volume of `res` voxels a side.
    /// `wide[i]`: layer i's wide volume and how it is used, for the layers that have one.
    pub fn run(&mut self, layers: &[&Volume], wide: &[Option<(&Volume, WideUse)>], c: &Combine, res: u32) -> R<()> {
        self.run_slices(layers, wide, c, res, 0..res)
    }

    pub(crate) fn run_slices(&mut self, layers: &[&Volume], wide: &[Option<(&Volume, WideUse)>], c: &Combine, res: u32, slices: std::ops::Range<u32>) -> R<()> {
        if res % 8 != 0 { return Err(format!("a volume's size must be a multiple of 8, not {res}")); }
        if layers.len() < c.n { return Err(format!("{} layers to combine, but only {} have been built", c.n, layers.len())); }
        let gpu = self.gpu.clone();
        // each layer's volume holds values out to the centres of its outermost voxels
        let mut lim = [HB; W];
        for (i, v) in layers.iter().enumerate().take(W) {
            lim[i] = HB - HB / v.res as f64;
        }
        let mut uses = [None; W];
        for (i, w) in wide.iter().enumerate().take(c.n.min(W)) {
            uses[i] = w.map(|w| w.1);
        }
        gpu.queue.write_buffer(&self.u, 0, bytemuck::cast_slice(&c.uniform(res, lim, uses)));
        gpu.queue.write_buffer(&self.u, (UNIFORM_FLOATS * 4) as u64, bytemuck::cast_slice(&[0u32, 0, slices.start, 0]));
        let vi = self.take(res)?;
        let view = |i: usize| &layers.get(i).filter(|_| i < c.n).copied().unwrap_or(&self.blank).view;
        let far = |i: usize| &wide.get(i).filter(|_| i < c.n).copied().flatten().map_or(&self.blank, |w| w.0).view;
        if !c.occurrences.is_empty() {
            let records = gf_core::world_mods::occurrence_table(&c.occurrences);
            self.instance_table.write(&gpu, bytemuck::cast_slice(&records));
        }
        let g_in = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("combine"),
            layout: &self.l_in,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.lin) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(view(0)) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(view(1)) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(view(2)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(view(3)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(far(0)) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(far(1)) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(far(2)) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(far(3)) },
                wgpu::BindGroupEntry { binding: 10, resource: self.instance_table.buf.as_entire_binding() },
            ],
        });
        let two_ok = !matches!(self.pipe, Pipe::Computed { two: None, .. });
        self.prepare_bodies(res, c.needs_body_channels() && self.bodies_wanted, c.bodies == 2 && two_ok)?;
        dispatch(&gpu, &self.pipe, self.bodies.as_ref(), self.bodies_channels, &self.vols[vi].view, &g_in, res, slices);
        Ok(())
    }

    fn multi(&mut self) -> &Multi {
        if self.multi.is_none() {
            let gpu = &self.gpu;
            let vis = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
            let storage = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None
            }, count: None };
            let mut e = vec![
                uniform_entry(0, vis | wgpu::ShaderStages::VERTEX, false),
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: vis, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                storage(10), uniform_entry(11, vis, false),
            ];
            e.extend((0..2 * MAX_LAYERS).map(|k| wgpu::BindGroupLayoutEntry {
                binding: (MULTI_TEX + k) as u32,
                visibility: vis,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                count: None,
            }));
            let l_in = gpu.bind_layout("combine in passes", &e);
            let pipe = make_pipe(gpu, &multi_wgsl(), &l_in);
            // (a uniform block, not storage: every voxel reads these numbers, and from storage that cost
            // more than the reading of the layers did)
            let passes = gpu.uniform_buffer("combine passes", (PASS_WORDS * 4 * MAX_LAYERS) as u64);
            self.multi = Some(Multi { l_in, pipe, alone: std::cell::OnceCell::new(), passes });
        }
        self.multi.as_ref().unwrap()
    }

    /// Combine a scene of any number of layers in passes (`Stack::pass_plan`): `layers[g]` is layer g's
    /// volume, `wide[g]` its wide volume and how it is used, `occurrences[g]` its World copies. The bodies'
    /// volume has `channels` channels (2 or 4) when it is wanted. One dispatch: each voxel runs every pass.
    pub fn run_passes(&mut self, layers: &[&Volume], wide: &[Option<(&Volume, WideUse)>], passes: &[Pass], occurrences: &[Vec<Placement>], channels: usize, res: u32) -> R<()> {
        self.run_passes_slices(layers, wide, passes, occurrences, channels, res, 0..res)
    }

    /// The same, slices `slices` of the volume only (a staged build writes it over several frames).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_passes_slices(&mut self, layers: &[&Volume], wide: &[Option<(&Volume, WideUse)>], passes: &[Pass], occurrences: &[Vec<Placement>], channels: usize, res: u32, slices: std::ops::Range<u32>) -> R<()> {
        if res % 8 != 0 { return Err(format!("a volume's size must be a multiple of 8, not {res}")); }
        if passes.is_empty() || passes.len() > MAX_LAYERS { return Err(format!("{} combining passes", passes.len())); }
        if layers.len() > MAX_LAYERS { return Err(format!("{} layers to combine, at most {MAX_LAYERS}", layers.len())); }
        let gpu = self.gpu.clone();
        let has_occurrences = occurrences.iter().any(|o| !o.is_empty());
        if has_occurrences {
            let records = gf_core::world_mods::occurrence_table(occurrences);
            self.instance_table.write(&gpu, bytemuck::cast_slice(&records));
        }
        let want = self.bodies_wanted && passes.iter().any(|p| (0..p.combine.n).any(|j| p.combine.show[j] && p.chan.get(p.combine.body[j]).is_some_and(|&ch| ch != 0 && ch != u32::MAX)));
        let two_ok = !matches!(self.pipe, Pipe::Computed { two: None, .. });
        self.prepare_bodies(res, want, channels == 2 && two_ok)?;
        // (with the bodies' volume written every channel must be right; without, only the nearest)
        let per_channel = self.bodies.is_some();
        let mut words: Vec<u32> = Vec::with_capacity(PASS_WORDS * passes.len());
        let mut head = Vec::new();
        let mut all_alone = true;
        for (k, p) in passes.iter().enumerate() {
            let mut lim = [HB; W];
            let mut uses = [None; W];
            let mut slot = [0u32; 4];
            for (j, &g) in p.layers.iter().enumerate() {
                let v = layers.get(g).ok_or_else(|| format!("layer {} has not been built", g + 1))?;
                lim[j] = HB - HB / v.res as f64;
                uses[j] = wide.get(g).copied().flatten().map(|w| w.1);
                slot[j] = g as u32;
            }
            let u = p.combine.uniform(res, lim, uses);
            if k == 0 { head = u.to_vec(); }
            words.extend(u.iter().map(|f| f.to_bits()));
            words.extend([0, 0, 0, 0]);
            words.extend(slot);
            words.extend(p.chan);
            let (solo, cull) = cull_words(&p.combine, has_occurrences, per_channel);
            all_alone &= cull[3] > 0.5;
            words.extend(solo);
            words.extend(cull.map(f32::to_bits));
        }
        gpu.queue.write_buffer(&self.u, 0, bytemuck::cast_slice(&head));
        gpu.queue.write_buffer(&self.u, (UNIFORM_FLOATS * 4) as u64, bytemuck::cast_slice(&[0u32, 0, slices.start, passes.len() as u32]));
        let vi = self.take(res)?;
        self.multi();
        gpu.queue.write_buffer(&self.multi.as_ref().unwrap().passes, 0, bytemuck::cast_slice(&words));
        let m = self.multi.as_ref().unwrap();
        let view = |g: usize| &layers.get(g).copied().unwrap_or(&self.blank).view;
        let far = |g: usize| &wide.get(g).copied().flatten().map_or(&self.blank, |w| w.0).view;
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.lin) },
            wgpu::BindGroupEntry { binding: 10, resource: self.instance_table.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: m.passes.as_entire_binding() },
        ];
        for g in 0..MAX_LAYERS {
            entries.push(wgpu::BindGroupEntry { binding: (MULTI_TEX + g) as u32, resource: wgpu::BindingResource::TextureView(view(g)) });
            entries.push(wgpu::BindGroupEntry { binding: (MULTI_TEX + MAX_LAYERS + g) as u32, resource: wgpu::BindingResource::TextureView(far(g)) });
        }
        let g_in = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("combine in passes"), layout: &m.l_in, entries: &entries });
        // (a scene of objects that all stand alone: the shader without the general arithmetic, much smaller)
        let pipe = if all_alone { m.alone.get_or_init(|| make_pipe(&gpu, &multi_alone_wgsl(), &m.l_in)) } else { &m.pipe };
        dispatch(&gpu, pipe, self.bodies.as_ref(), self.bodies_channels, &self.vols[vi].view, &g_in, res, slices);
        Ok(())
    }

    /// Make or keep the bodies' volume (`want`): two channels (`two`) or four, at `res` voxels a side.
    fn prepare_bodies(&mut self, res: u32, want: bool, two: bool) -> R<()> {
        let gpu = self.gpu.clone();
        // The bodies' volume: only while more than one body shows. Two channels do for two bodies (twice
        // the size of the scene's volume); three or four bodies take four (four times its size).
        if want {
            let channels = if two { 2 } else { 4 };
            if self.bodies.as_ref().is_none_or(|b| b.res != res) || self.bodies_channels != channels {
                let reused = self.bodies_spare.take_if(|(volume, stored)| volume.res == res && *stored == channels);
                // (the one shown until now is kept as the spare: a draft's and the settled build's take
                // turns, and a new volume costs a clearing of its memory: `keep_sizes`)
                if let Some(current) = self.bodies.take() { self.bodies_spare = Some((current, self.bodies_channels)); }
                self.bodies = Some(match reused {
                    Some((volume, _)) => volume,
                    None => {
                        let tex = gpu.try_volume("bodies", res, res, bodies_format(channels), matches!(self.pipe, Pipe::Computed { .. }))?;
                        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                        Volume { tex, view, res }
                    }
                });
                self.bodies_channels = channels;
            }
        } else {
            self.bodies = None;
            self.bodies_spare = None;
        }
        Ok(())
    }
}

/// Write the combined volume (`out`) and the bodies' (`bodies`) with a combining pipeline, slice range
/// `slices` of a volume `res` voxels a side.
#[allow(clippy::too_many_arguments)]
fn dispatch(gpu: &Gpu, pipe: &Pipe, bodies: Option<&Volume>, bodies_channels: u32, out: &wgpu::TextureView, g_in: &wgpu::BindGroup, res: u32, slices: std::ops::Range<u32>) {
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("combine") });
    fn view_of(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(v)
    }
    match (pipe, bodies) {
        (Pipe::Computed { scene, l_scene, four, two }, bodies) => {
            let (pipe, g_out) = match bodies {
                Some(b) => {
                    let both = two.as_ref().filter(|_| bodies_channels == 2).unwrap_or(four);
                    (&both.pipe, gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &both.layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: view_of(out) }, wgpu::BindGroupEntry { binding: both.binding, resource: view_of(&b.view) }] }))
                }
                None => (scene, gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l_scene, entries: &[wgpu::BindGroupEntry { binding: 0, resource: view_of(out) }] })),
            };
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(pipe);
            p.set_bind_group(0, g_in, &[]);
            p.set_bind_group(1, &g_out, &[]);
            p.dispatch_workgroups(res / 8, res / 8, (slices.end - slices.start) / 4);
        }
        (Pipe::Drawn { scene, four, two }, bodies) => {
            let draw_bodies = if bodies_channels == 2 { two } else { four };
            for z in slices.clone() {
                let mut rp = pass(&mut enc, out, Some(z), true);
                rp.set_pipeline(scene);
                rp.set_bind_group(0, g_in, &[]);
                rp.draw(0..3, z..z + 1);
            }
            if let Some(b) = bodies {
                for z in slices.clone() {
                    let mut rp = pass(&mut enc, &b.view, Some(z), true);
                    rp.set_pipeline(draw_bodies);
                    rp.set_bind_group(0, g_in, &[]);
                    rp.draw(0..3, z..z + 1);
                }
            }
        }
    }
    gpu.queue.submit([enc.finish()]);
}

/// The combining shader for a scene of more than COMBINE_WIDTH layers: combine.wgsl as it is, with every
/// layer's volume bound (MAX_LAYERS of them), and each voxel running every pass (`Stack::pass_plan`) in
/// turn, each pass's numbers copied into `c` and its layers' places into `SLOT`; each pass's bodies go
/// to their channels, the nearest kept. A scene of COMBINE_WIDTH layers or fewer uses combine.wgsl itself.
pub fn multi_wgsl() -> String {
    multi_text(false)
}

/// The same for a scene whose every layer stands alone (every pass's `cull_words` all-alone): each pass's
/// bodies made straight from its layers' distances, the general arithmetic left out of the shader (it
/// would hold registers that every voxel pays for).
pub fn multi_alone_wgsl() -> String {
    multi_text(true)
}

fn multi_text(alone: bool) -> String {
    let n = MAX_LAYERS;
    assert!(n <= 8, "the multi-pass shader keeps eight layers' bounds");
    let mut t = COMBINE_WGSL.replace("\r\n", "\n");
    let mut swap = |a: &str, b: &str, all: bool| {
        assert!(t.contains(a), "combine.wgsl no longer has {a:?}");
        t = if all { t.replace(a, b) } else { t.replacen(a, b, 1) };
    };
    swap("@group(0) @binding(0) var<uniform> c: C;", &format!("@group(0) @binding(0) var<uniform> c0: C;\nvar<private> PI: u32;\nvar<private> SLOT: vec4u;\nstruct P {{ c: C, slot: vec4u, chan: vec4u, solo: vec4u, cull: vec4f, }}\n@group(0) @binding(11) var<uniform> passes: array<P, {n}>;"), false);
    for k in 0..W {
        swap(&format!("@group(0) @binding({}) var vol{k}: texture_3d<f32>;\n", 2 + k), "", false);
        swap(&format!("@group(0) @binding({}) var wide{k}: texture_3d<f32>;\n", 6 + k), "", false);
    }
    let mut decl = String::new();
    for g in 0..n {
        decl += &format!("@group(0) @binding({}) var vol{g}: texture_3d<f32>;\n@group(0) @binding({}) var wide{g}: texture_3d<f32>;\n", MULTI_TEX + g, MULTI_TEX + n + g);
    }
    let occ = "@group(0) @binding(10) var<storage, read> occurrences: array<vec4f>;\n";
    swap(occ, &format!("{occ}{decl}"), false);
    // (the layers read in one loop, so that the reading is written out once: the pass's nearest lone
    // layer first when it is the one `combineAll` chose, each through `layerCull`)
    swap("  d.x = layer(vol0, wide0, 0u, p);\n  if (n > 1u) { d.y = layer(vol1, wide1, 1u, p); }\n  if (n > 2u) { d.z = layer(vol2, wide2, 2u, p); }\n  if (n > 3u) { d.w = layer(vol3, wide3, 3u, p); }\n",
         "  for (var t = 0u; t < n; t++) {\n    var j = t;\n    if (PI == PRE.x) { j = (t + PRE.y) % n; }\n    d[j] = layerCull(j, p);\n  }\n  if (passes[PI].cull.w > 0.5) { return aloneBodies(d, n); }\n", false);
    swap("let range = occurrences[i];", "let range = occurrences[SLOT[i]];", false);
    for (a, b) in [("c.q.w", "c0.q.w"), ("c.stage.z", "c0.stage.z"), ("c.dispatch_offset", "c0.dispatch_offset"),
                   ("nearest(combine(point(", "nearest(combineAll(point("), ("= combine(point(", "= combineAll(point("),
                   ("return combine(point(", "return combineAll(point(")] {
        swap(a, b, true);
    }
    let cases: String = (0..n).map(|g| format!("    case {g}u: {{ return layer(vol{g}, wide{g}, i, p); }}\n")).collect();
    let extra = format!("
// layer i of the pass, whose volume is the scene's layer g
fn layerAt(g: u32, i: u32, p: vec3f) -> f32 {{
  switch g {{
{cases}    default: {{ return EMPTY; }}
  }}
}}

// Leaving out what cannot change a voxel. A layer's distance is at least the distance to its placed box
// (outside the box the reading adds that distance on), and a body is cut to the stage, so neither layer
// nor body is nearer than max(box distance, stage distance). With the passes' numbers (`cull_words`):
// a lone layer whose bound is no nearer than the best found for its channel is not read (its bound
// stands in: its body then comes out no nearer than that best, so the nearest is the same), and a pass
// none of whose bodies can come nearer than the best is not run. The pass with the nearest lone layer
// runs first, that layer read first, so the best is known early. (No arrays in private memory: they
// cost every voxel more than the work they would save.)
var<private> LB0: vec4f;             // each of the scene's layers' bounds, layers 0 to 3
var<private> LB1: vec4f;             // and 4 to 7
var<private> BEST: vec4f;            // the nearest each channel can be so far (an upper bound)
var<private> PRE: vec2u;             // the nearest lone layer: its pass, its place in it
var<private> S: f32;                 // the stage's distance
const FAR: f32 = 1e9;

fn lbOf(g: u32) -> f32 {{
  if (g < 4u) {{ return LB0[g]; }}
  return LB1[g - 4u];
}}

fn bestFor(ch: u32) -> f32 {{
  if (passes[0].cull.z > 0.5) {{ return BEST[ch]; }}
  return min(min(BEST.x, BEST.y), min(BEST.z, BEST.w));
}}

// layer i of the pass (PI): no nearer than this at p
fn boxBound(i: u32, p: vec3f) -> f32 {{
  if (c.wide[i].w > 0.0) {{ return -FAR; }}
  let r0 = c.rows[3u * i];
  let r1 = c.rows[3u * i + 1u];
  let r2 = c.rows[3u * i + 2u];
  let q = vec3f(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
  let lim = c.info[i].y;
  let o = q - clamp(q, vec3f(-lim), vec3f(lim));
  let o2 = dot(o, o);
  var lb = -FAR;
  if (o2 > 0.0) {{ lb = c.info[i].x * sqrt(o2); }}
  return max(lb, S);
}}

// can a body of the pass (PI) come nearer than the best of its channel?
fn passMatters() -> bool {{
  if (passes[PI].cull.y < 0.5) {{ return true; }}
  let n = u32(c.q.z);
  let nb = u32(c.stage.w);
  let slack = passes[PI].cull.x;
  for (var b = 0u; b < nb; b++) {{
    let ch = passes[PI].chan[b];
    if (ch >= 4u) {{ continue; }}
    var m = FAR;
    var any = false;
    for (var j = 0u; j < n; j++) {{
      if (c.info[j].z > 0.5 && u32(c.info[j].w) == b) {{ m = min(m, lbOf(SLOT[j])); any = true; }}
    }}
    let f = select(max(EMPTY, S), max(m - slack, S), any);
    if (f < bestFor(ch)) {{ return true; }}
  }}
  return false;
}}

// a body of a lone layer at distance d, as `combine` makes it
fn aloneBody(d: f32) -> f32 {{
  let f = max(d, S) + dip(c.stage.y, 3.0, abs(d - S));
  return clamp(f, -1.0, 4.0);
}}

// the bodies of a pass whose every layer stands alone, as `combine` makes them (a hidden one's: EMPTY)
fn aloneBodies(d: vec4f, n: u32) -> vec4f {{
  var bodies = vec4f(ABSENT);
  for (var j = 0u; j < n; j++) {{
    bodies[u32(c.info[j].w)] = aloneBody(select(EMPTY, d[j], c.info[j].z > 0.5));
  }}
  return bodies;
}}

// layer j of the pass (PI), or its bound where it cannot matter
fn layerCull(j: u32, p: vec3f) -> f32 {{
  let solo = passes[PI].solo[j];
  let lb = lbOf(SLOT[j]);
  let pre = PI == PRE.x && j == PRE.y;
  let ch = passes[PI].chan[u32(c.info[j].w)];
  if (solo == 2u || (solo == 1u && !pre && ch < 4u && lb >= bestFor(ch))) {{ return lb; }}
  let d = layerAt(SLOT[j], j, p);
  if (pre && ch < 4u) {{ BEST[ch] = min(BEST[ch], aloneBody(d)); }}
  return d;
}}

// every pass's bodies in their channels: the nearest of each channel's (c0.dispatch_offset.w passes)
fn combineAll(p: vec3f) -> vec4f {{
  let np = c0.dispatch_offset.w;
  S = length(p) - c0.stage.x;
  BEST = vec4f(ABSENT);
  PRE = vec2u(np, 0u);
  LB0 = vec4f(FAR);
  LB1 = vec4f(FAR);
  var near = FAR;
  for (var k = 0u; k < np; k++) {{
    PI = k;
    let n = u32(c.q.z);
    for (var j = 0u; j < n; j++) {{
      let g = passes[k].slot[j];
      let lb = boxBound(j, p);
      if (g < 4u) {{ LB0[g] = lb; }} else {{ LB1[g - 4u] = lb; }}
      if (passes[k].solo[j] == 1u && lb < near) {{ near = lb; PRE = vec2u(k, j); }}
    }}
  }}
  var out = vec4f(ABSENT);
  for (var it = 0u; it < np; it++) {{
    // the nearest lone layer's pass first, then the rest in order
    var k = it;
    if (PRE.x < np) {{
      if (it == 0u) {{ k = PRE.x; }} else if (it <= PRE.x) {{ k = it - 1u; }}
    }}
    PI = k;
    SLOT = passes[k].slot;
    if (k != PRE.x && !passMatters()) {{ continue; }}
    let ch = passes[k].chan;
    let b = combine(p);
    for (var j = 0u; j < 4u; j++) {{
      if (ch[j] < 4u) {{ out[ch[j]] = min(out[ch[j]], b[j]); BEST[ch[j]] = min(BEST[ch[j]], out[ch[j]]); }}
    }}
  }}
  return out;
}}
");
    swap("// the scene's distance: the nearest of the bodies", &format!("{extra}// the scene's distance: the nearest of the bodies"), false);
    // Every pass's numbers are read where they lie (`passes[PI].c.`): a private copy of the whole block
    // per voxel and pass would not stay in registers (it made the step hundreds of times slower).
    if alone {
        // (every pass's bodies straight from the distances: the general arithmetic is cut out of `combine`)
        let end = "  return bodies;\n}\n";
        let a = t.find("  // 1. morph").expect("combine.wgsl no longer has its morph step");
        let b = a + t[a..].find(end).expect("combine.wgsl's combine no longer ends as it did") + end.len();
        t.replace_range(a..b, "  return aloneBodies(d, n);\n}\n");
    }
    let mut out = String::with_capacity(t.len() * 2);
    let mut prev = ' ';
    let mut it = t.chars().peekable();
    while let Some(ch) = it.next() {
        if ch == 'c' && it.peek() == Some(&'.') && !(prev.is_alphanumeric() || prev == '_') {
            out.push_str("passes[PI].c");
        } else {
            out.push(ch);
        }
        prev = ch;
    }
    out
}

/// Where the multi-pass shader's layer volumes are bound: MAX_LAYERS layer volumes from here, then their
/// wide volumes.
const MULTI_TEX: usize = 12;
/// u32 words of one pass in the multi-pass shader's table: the combine block, then the layers' places,
/// the bodies' channels, which layers stand alone, and the culling numbers (`cull_words`).
const PASS_WORDS: usize = UNIFORM_FLOATS + 4 + 4 + 4 + 4 + 4;

/// What the multi-pass shader needs to leave out the work that cannot change a voxel (`multi_wgsl`).
///
/// `solo[j]`: 1 when layer j of the pass is a body on its own that shows (no morph, no carve to or from
/// another layer), 2 when it is that but hidden (nothing reads it), 0 otherwise. Such a layer's body is
/// its distance cut to the stage, at least the distance to the layer's placed box; where that is no
/// nearer than the best already found the layer is not read at all.
///
/// `cull`: how far below the nearest of its layers' boxes a body of the pass can reach (its seams and
/// fills, generously), 1 when the pass can be left out by its boxes (no morph, no World copies), 1
/// when each channel must be right (the bodies' volume is written) rather than only the nearest, and 1
/// when every layer of the pass stands alone (its bodies are then made straight from the distances).
fn cull_words(c: &Combine, occurrences: bool, per_channel: bool) -> ([u32; 4], [f32; 4]) {
    let n = c.n.min(W);
    let plain_morph = (0..n).all(|i| (0..n).all(|j| c.morph[i][j] == if i == j { 1.0 } else { 0.0 }));
    let cullable = plain_morph && !occurrences && (0..n).all(|i| c.occurrences.get(i).is_none_or(|o| o.is_empty()));
    let mut solo = [0u32; 4];
    if cullable {
        for i in 0..n {
            let alone = (0..n).all(|j| j == i || (c.term[i][j] == 0.0 && c.term[j][i] == 0.0 && c.body[j] != c.body[i]));
            if alone { solo[i] = if c.show[i] { 1 } else { 2 }; }
        }
    }
    // A union of several takes at most 1.5 seam radii below the nearest (each dip is at most half a
    // radius, and four deep at most three of them count); a carve's or a join's fill at most its size
    // each time it is used.
    let mut k = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j { k = k.max(c.ku[i][j].abs()); }
        }
    }
    let fill = c.tfill.iter().take(n).fold(0.0f64, |a, f| a.max(f.abs())) + c.qfill.abs();
    let slack = 2.0 * k + 4.0 * fill;
    let all_alone = n > 0 && solo[..n].iter().all(|&s| s != 0);
    (solo, [slack as f32, cullable as u8 as f32, per_channel as u8 as f32, all_alone as u8 as f32])
}

/// The multi-pass combining step (`multi_wgsl`), made the first time a scene needs it.
struct Multi {
    l_in: wgpu::BindGroupLayout,
    pipe: Pipe,
    /// the same for scenes whose every layer stands alone (`multi_alone_wgsl`), made when first needed
    alone: std::cell::OnceCell<Pipe>,
    /// every pass's numbers (`PASS_WORDS` each, room for MAX_LAYERS passes)
    passes: wgpu::Buffer,
}

/// What a layer's volume was built from.
struct Built {
    key: String,
    res: u32,
    /// how carefully: a full build is exact, one made while a slider was dragged or the shape animated is not
    mode: Detail,
    /// A tube: how far from its surface its own volume is exact (the reading passes to the wide volume
    /// between these two distances). None for an equation, whose volume is right as far as seams reach.
    exact_to: Option<(f64, f64)>,
    /// voxels per side of the wide volume built for this shape, once one has been needed
    wide: Option<u32>,
}
impl Built {
    fn exact(&self) -> bool {
        self.mode.is_final()
    }
}

/// Voxels per side of the wide volume that goes with a tube's own volume of this size.
fn wide_res(res: u32) -> u32 {
    if res >= 512 { 192 } else if res >= 256 { 128 } else { 64 }
}

/// The engine's side of the layers: the combining step, what each layer's volume holds, and what the
/// last build did.
pub struct LayerWork {
    pending: Option<staging::FullRefinement>,
    pub combiner: Combiner,
    pub(crate) world_fields: world_fields::Fields,
    group_scratch: Option<Combiner>,
    group_key: Option<String>,
    built: Vec<Option<Built>>,
    /// what was last combined: the numbers, the size, each layer's own size and the use of its wide volume
    last: Option<(Combine, u32, Vec<u32>, Vec<Option<WideUse>>)>,
    /// the same for a scene combined in passes
    last_multi: Option<(Vec<Pass>, u32, Vec<u32>, Vec<Option<WideUse>>)>,
    /// the tracer reads the combined volume (the last build was of layers)
    pub combined: bool,
    /// which layers the last build built again
    pub rebuilt: Vec<usize>,
    /// milliseconds the last combining step took (to the end, when builds are waited for)
    pub combine_ms: f64,
    /// since the start: how often each layer was built, how often the layers were combined, and the
    /// combining steps' milliseconds added up
    pub builds: [u32; MAX_LAYERS],
    /// and how often its wide volume
    pub wide_builds: [u32; MAX_LAYERS],
    pub combines: u32,
    pub combine_ms_sum: f64,
    /// build every layer and combine again even if nothing has changed (to time it)
    pub force: bool,
    /// The tracer's stride through a combined scene as a share of what it would be, with as many more
    /// strides: below 1 to check that the usual stride misses nothing.
    pub stride: f64,
    /// Build a tube that has been moved or turned (not resized) where it is placed, instead of reading
    /// its volume between voxels: to measure what that reading costs in sharpness.
    pub in_place: bool,
    /// how the tracer was last told to march through the combined scene: stride, strides outside and
    /// inside, and how many further pieces of glass a ray leaving one may enter
    pub march: (f64, u32, u32, u32),
    /// Do without the tubes' wide volumes: to see what they are for.
    pub no_wide: bool,
}

impl LayerWork {
    pub(crate) fn projected_face_field(&self, source: usize) -> Option<(&Volume, f64, &[Placement])> {
        let (field, scale) = self.world_fields.volume(source)?;
        let placements = self.world_fields.plan.as_ref()?.placements.get(source)?;
        Some((field, scale, placements))
    }

    /// Voxels a side of layer i's volume as last built (its tree's is made at this size).
    pub(crate) fn built_res(&self, i: usize) -> Option<u32> {
        self.built.get(i).and_then(|b| b.as_ref()).map(|b| b.res)
    }

    pub(crate) fn face_layout(&self) -> Option<&gf_core::layers::Combine> {
        self.last.as_ref().map(|last| &last.0)
    }

    pub fn new(gpu: &Gpu) -> Self {
        Self { pending: None, combiner: Combiner::new(gpu), world_fields: Default::default(), group_scratch: None, group_key: None, built: Vec::new(), last: None, last_multi: None, combined: false, rebuilt: Vec::new(), combine_ms: 0.0, builds: [0; MAX_LAYERS], wide_builds: [0; MAX_LAYERS], combines: 0, combine_ms_sum: 0.0, force: false, stride: 1.0, in_place: false, march: (0.0, 0, 0, 0), no_wide: false }
    }
    /// A single shape has just been built the plain way: remember what the first layer's volume holds,
    /// so that it need not be built again when a second layer arrives, and let go of everything that
    /// was kept for layers that are no longer there.
    /// The last build was of one plain layer (its volume is the one the tracer reads).
    /// Everything kept of what was built is let go (another engine drew the scene meanwhile): the next build builds afresh.
    pub(crate) fn forget(&mut self) {
        self.built.clear();
        self.last = None;
        self.last_multi = None;
        self.combined = false;
    }
    pub fn built_plain_still(&self) -> bool {
        !self.combined && self.built.len() == 1
    }
    pub fn built_plain(&mut self, baker: &mut crate::bake::Baker, key: String, res: u32, mode: Detail, melt: Option<f64>) {
        baker.drop_layers(1);
        baker.drop_wide(0);
        self.combiner.clear();
        self.world_fields=Default::default();
        self.group_key = None;
        self.group_scratch = None;
        self.built.clear();
        self.built.push(Some(Built { key, res, mode, exact_to: melt.map(|m| exact_to(res, m)), wide: None }));
        self.combined = false;
        self.last = None;
        self.rebuilt = vec![0];
        self.builds[0] += 1;
    }
}

/// The bodies' volume and what is in it (see `Combiner::bodies_volume`).
pub struct Bodies<'a> {
    pub volume: &'a Volume,
    /// how many bodies there are: channel k of the volume is body k, for k below this
    pub count: u32,
    /// how many channels the volume has: 2 or 4
    pub channels: u32,
}

/// How far from a tube's surface its own volume holds the exact distance, as the baker builds it: the
/// reading passes to the wide volume between the two distances given (both well inside that).
fn exact_to(res: u32, melt: f64) -> (f64, f64) {
    let h = 2.0 * HB / res as f64;
    // melted: exact to 2.2 melts and 2.5 voxels, the far field never below half a voxel less; hard: 6 voxels
    if melt > 0.0 { (0.4 * (2.2 * melt + 1.5 * h), 2.2 * melt + 1.5 * h) } else { (2.0 * h, 5.0 * h) }
}

impl Engine {
    /// The bodies' volume (see `Combiner::bodies_volume`): there only while the scene is of layers and
    /// more than one body shows. `State::bodies()` says which layers each body holds.
    pub fn bodies_volume(&self) -> Option<&Volume> {
        if let Some(stage) = &self.layers.pending { return stage.visible_body.as_ref(); }
        if self.layers.combined { self.layers.combiner.bodies_volume() } else { None }
    }

    /// The bodies as whatever draws them needs them: the volume, how many bodies there are (channels
    /// 0 to count - 1 are theirs; no other channel is to be read), and how many channels the volume has.
    pub fn bodies(&self) -> Option<Bodies<'_>> {
        let volume = self.bodies_volume()?;
        let channels = match &self.layers.pending { Some(stage) => stage.visible_channels, None => self.layers.combiner.bodies_channels() };
        Some(Bodies { volume, count: self.state.stack.body_channels() as u32, channels })
    }

    /// The bodies' volume read back: voxels per side, channels per voxel (2 or 4), and every voxel's
    /// distances as 16-bit floats' bits.
    pub fn read_bodies(&self) -> R<(u32, u32, Vec<u16>)> {
        let b = self.bodies().ok_or("there is one body: no bodies' volume has been made")?;
        let vol = b.volume;
        let mut out = Vec::with_capacity((vol.res as usize).pow(3) * b.channels as usize);
        for z in 0..vol.res {
            let bytes = self.gpu.read(&vol.tex, 0, 0, z, vol.res, vol.res, 2 * b.channels)?;
            out.extend_from_slice(bytemuck::cast_slice::<u8, u16>(&bytes));
        }
        Ok((vol.res, b.channels, out))
    }

    /// Bytes of video memory the scene's volumes hold as they stand: every layer's, the tubes' wide
    /// volumes, the scene's and the bodies'. (`gf_core::layers::volumes_bytes` says what a scene would hold.)
    pub fn volumes_bytes(&self) -> u64 {
        let size = |v: Option<&Volume>, channels: u64| v.map_or(0, |v| (v.res as u64).pow(3) * 2 * channels);
        let n = if self.layers.combined { self.state.layer_count() } else { 1 };
        let layers: u64 = (0..n).map(|i| size(self.baker.layer_volume(i), 1) + size(self.baker.wide_volume(i), 1)).sum();
        // (and each object's tree's volume, which stands for it in the combining step)
        let trees: u64 = if self.layers.combined { (0..n).filter(|&i| self.state.has_tree(i)).map(|i| size(self.csg.layer_vols.get(i).and_then(|v| v.as_ref()).map(|v| &v.1), 1)).sum() } else { 0 };
        let scene = if self.layers.combined { size(self.layers.combiner.volume(), 1) } else { 0 };
        layers + trees + scene + size(self.bodies_volume(), self.layers.combiner.bodies_channels() as u64)
    }

    /// The volumes of the other detail kept beside the ones in use (`Combiner::keep_sizes`,
    /// `CsgWork::layer_spares`) are let go when, with them, the volumes held would pass
    /// `budget::TOTAL_MAX`: the next change between draft and settled detail then makes its volumes anew
    /// (their clearing costs a stall). Said once on stderr each time it happens.
    pub(crate) fn trim_spares(&mut self) {
        let spares = self.layers.combiner.spare_bytes() + self.csg.spare_bytes();
        if spares == 0 { return; }
        let held = self.volumes_bytes();
        if held + spares <= gf_core::budget::TOTAL_MAX { return; }
        self.layers.combiner.drop_spares();
        self.csg.drop_layer_spares();
        eprintln!("[memory] {:.1} GB of volumes kept for the other detail let go: with the {:.1} GB in use they would pass the {} GB budget", spares as f64 / 1e9, held as f64 / 1e9, gf_core::budget::TOTAL_MAX / 1_000_000_000);
    }

    /// The volume the tracer reads: the one shape's, or the layers combined.
    pub fn scene_volume(&self) -> Option<&Volume> {
        if self.parts_active { if let Some(p) = &self.parts { return Some(p.volume()); } }
        if let Some(stage) = &self.layers.pending { return Some(&stage.visible_scene); }
        if self.state.empty { self.csg.empty_vol.as_ref() } else if self.layers.combined { self.layers.combiner.volume() } else if let Some(v) = self.csg.volume() { Some(v) } else { self.baker.layer_volume(0) }
    }

    /// Object i's own volume is never read, so not made: its tree's volume stands for it
    /// (`State::layer_volume_unread`), on a device that can merge trees.
    pub(crate) fn own_volume_unread(&self, i: usize) -> bool {
        self.csg.available() && self.state.layer_volume_unread(i)
    }

    /// An object read as a tree whose first part is its own formula: its tree reads the same table
    /// (`csg_formula_tables`), so the one its own job has just made is kept for it rather than made twice.
    /// (`res`: the size the layer is built at, `State::layer_build_res`)
    pub(crate) fn share_formula_table(&mut self, i: usize, mode: Detail, job: &Job, res: u32) {
        if let ShapeJob::Field { table, .. } = &job.shape {
            if !table.is_empty() && self.state.has_tree(i) && self.state.layer_preset(i).has_table() && job.res == res {
                self.csg.keep_formula(crate::csg::CsgWork::formula_key(&self.state.layer_key(i), mode, job.res), table.clone());
            }
        }
    }

    /// Build what has changed of the layers, this carefully, combine them, and point the tracer at the result.
    pub(crate) fn bake_layers(&mut self, mode: Detail) -> R<()> {
        // (the volumes kept for the sizes the scene is built at now, settled and drafted; the others are
        // let go, even when this build skips the combining step. Worked out on settled builds only: the
        // sizes cost a detail plan each, which a drag would pay every frame)
        if mode.is_final() {
            let sizes = [Detail::Full, Detail::Anim, Detail::Quick].map(|d| self.state.scene_res(d));
            self.layers.combiner.keep_sizes(&sizes);
            let per = [Detail::Full, Detail::Anim, Detail::Quick].map(|d| self.state.layer_build_sizes(d));
            let layer_sizes: Vec<[u32; 3]> = (0..self.state.layer_count()).map(|i| per.each_ref().map(|s| s.get(i).copied().unwrap_or(0))).collect();
            self.csg.keep_layer_spares(&layer_sizes);
        }
        let t0 = Instant::now();
        let mut laps = crate::engine::Laps::new("layers");
        let n = self.state.layer_count();
        let exact = mode.is_final();
        self.layers.built.resize_with(n, || None);
        self.baker.drop_layers(n);
        let mut rebuilt = Vec::new();
        let mut jobs: Vec<Option<Job>> = (0..n).map(|_| None).collect();
        let (mut work, mut hand) = (0.0, 0.0);
        let mut todo = Vec::new();
        // (the sizes the layers are built at, within the budget, worked out once)
        let sizes = self.state.layer_build_sizes(mode);
        for i in 0..n {
            // A layer's volume will do if it holds the shape as it stands; for a full build it must also
            // be the full size and exact. (While something else moves, a still layer keeps its fine volume.)
            let place = self.state.stack.layers[i].place;
            // (not an object with a tree: its tree is measured into a volume of its own, in its own frame)
            let in_place = !self.state.stack.has_world_modifiers() && self.layers.in_place && place.size == 1.0 && !place.is_identity() && self.state.layer_family(i).kind == gf_core::presets::Kind::Curve && !self.state.layer_preset(i).exact_tubes && !self.state.has_tree(i);
            let mut key = self.state.layer_key(i);
            if in_place { key.push_str(&format!("|at{:?}{:?}", place.pos, place.rot)); }
            // (the size the layer's job builds at, as `State::shape_job` sets it: past COMBINE_WIDTH layers
            // `layer_res` is the detail plan's, which the job does not use, and comparing against it
            // built every layer again on every full build)
            let res = sizes[i];
            let good = !self.layers.force && self.layers.built[i].as_ref().is_some_and(|b| b.key == key && (!exact || (b.exact() && b.res == res)));
            if good { continue; }
            todo.push((i, key, in_place, place));
        }
        // the jobs of the layers to build (a tube's points and tables: milliseconds of CPU each), worked out
        // side by side when there are several
        let t = Instant::now();
        let ready: Vec<R<Job>> = if todo.len() > 1 {
            let layers: Vec<usize> = todo.iter().map(|w| w.0).collect();
            self.state.layer_jobs(&layers, mode)
        } else {
            Vec::new()
        };
        work += t.elapsed().as_secs_f64() * 1e3;
        let mut ready = ready.into_iter();
        for (i, key, in_place, place) in todo {
            let t = Instant::now();
            let mut job = match ready.next() {
                Some(job) => job?,
                None => {
                    let mut job = self.state.layer_job(i, mode, None)?;
                    job.prepare_formula(&self.state.families)?;
                    job
                }
            };
            self.share_formula_table(i, mode, &job, sizes[i]);
            if in_place {
                if let ShapeJob::Curve { curve, .. } = &mut job.shape {
                    let r = euler(place.rot[0], place.rot[1], place.rot[2]);
                    // (each point as what it is: a solid piece's planes and directions turn without moving)
                    curve.moved(&r, place.pos);
                    // a flat cut goes up and down with the layer; turned, it would no longer be level
                    let turned = place.rot.iter().any(|&a| a != 0.0);
                    curve.cut = if turned { None } else { curve.cut.map(|y| y + place.pos[1]) };
                }
            }
            let t1 = Instant::now();
            self.baker.set_layer(i);
            let fam = &self.state.families[job.family];
            let (stats, melt): (BakeStats, Option<f64>) = match &job.shape {
                // (a csg object whose tree stands for it: its own volume would never be read; its job is
                // still made, for its tree's table: `share_formula_table`)
                _ if self.own_volume_unread(i) => {
                    self.baker.clear_layer(i);
                    (BakeStats::default(), None)
                }
                ShapeJob::Curve { curve, melt } => (self.baker.curve(curve, job.res, *melt)?, Some(*melt)),
                ShapeJob::Field { uniforms, table } => {
                    let p = &fam.presets[job.preset];
                    debug_assert!(p.formula().is_some());
                    (self.baker.field(&format!("{}/{}", fam.id, p.id), p, uniforms, table, job.res)?, None)
                }
                ShapeJob::Mesh(model) => (self.models.build(&self.gpu, &mut self.baker, i, model, job.res, mode)?, None),
            };
            work += (t1 - t).as_secs_f64() * 1e3;
            hand += t1.elapsed().as_secs_f64() * 1e3;
            self.layers.built[i] = Some(Built { key, res: job.res, mode, exact_to: melt.map(|m| exact_to(job.res, m)), wide: None });
            self.layers.builds[i] += 1;
            if let Some(st) = self.state.layer_shape_mut(i) {
                st.meta.res = job.res;
                st.meta.ms = t.elapsed().as_secs_f64() * 1e3;
                st.meta.segments = stats.segments;
                st.meta.splats = stats.splats;
            }
            jobs[i] = Some(job);
            rebuilt.push(i);
        }
        laps.lap(&format!("rebuilt{rebuilt:?} jobs {work:.1} bakes"));
        // The surface studio of each layer: run on a layer just built (its shape kept aside first), or on
        // one whose studio alone has changed (from the shape kept aside); a layer whose studio was emptied
        // gets its shape back.
        self.sculptor.drop_layers(n);
        self.sculptor.trees.clear();
        for i in 0..n {
            let Some(b) = self.layers.built[i].as_ref() else { continue };
            let skey = format!("{}|{}|{}", b.key, b.res, b.mode.quality().tag());
            let fresh = rebuilt.contains(&i);
            let s = self.state.layer_bake(i).filter(|_| self.sculptor.available());
            // (no own volume: a csg object whose tree stands for it, which has no studio of its own)
            let Some(vol) = self.baker.layer_volume(i).filter(|_| !self.own_volume_unread(i)) else {
                self.sculptor.forget(i);
                continue;
            };
            let applied = self.sculptor.applied.get(i).cloned().unwrap_or_default();
            match s {
                Some(s) => {
                    let phases = self.state.sculpt_phases(i);
                    let want = format!("{skey}#{}", s.key(&phases, self.sculptor.footprint));
                    if fresh || (applied.is_empty() && !self.sculptor.has(&skey)) { self.sculptor.keep(i, vol, &skey)?; }
                    if fresh || applied != want {
                        if !self.sculptor.has(&skey) {
                            // (the shape kept aside has gone: build the layer again next time)
                            self.layers.built[i] = None;
                            self.state.request(gf_core::build::BuildRequest::again(mode));
                            continue;
                        }
                        self.sculptor.run(i, vol, &skey, &s, &phases, mode.is_settle() || self.time_bakes, mode.is_draft(), &Placement::default())?;
                        if !fresh { rebuilt.push(i); }
                    }
                }
                None => {
                    if !fresh && !applied.is_empty() {
                        if !self.sculptor.restore(i, vol, &skey) {
                            self.layers.built[i] = None;
                            self.state.request(gf_core::build::BuildRequest::again(mode));
                            continue;
                        }
                        rebuilt.push(i);
                    }
                    self.sculptor.forget(i);
                }
            }
        }
        // (the shapes kept aside for layers that no longer have modifiers, or are gone, are let go: once the
        // trees below have said which of theirs they keep)
        let kept: Vec<String> = (0..n).filter(|&i| self.state.layer_bake(i).is_some()).filter_map(|i| self.layers.built[i].as_ref().map(|b| b.key.clone())).collect();
        laps.lap("studio");
        let mut c = self.state.stack.combine();
        // More layers than one combining pass takes: the passes (`Stack::pass_plan`), each with its own
        // numbers; `c` (the first COMBINE_WIDTH layers) is then not used for the combining.
        let multi = n > W;
        if multi && (self.state.stack.has_world_modifiers() || self.state.stack.hierarchy.as_ref().is_some_and(|h| h.changes_geometry())) {
            return Err(format!("World and group modifiers work on scenes of up to {W} objects for now: this one has {n}"));
        }
        let mut plan: Vec<Pass> = Vec::new();
        if multi {
            let authored = &self.state.stack;
            plan = match self.state.edit_scope.object().filter(|i| *i < n) {
                Some(layer) => {
                    let mut solo = authored.clone();
                    for i in 0..n {
                        solo.layers[i].show = i == layer;
                        for j in 0..i { solo.pair_mut(gf_core::layers::mask_of(&[i, j])).unwrap().op = gf_core::layers::Op::Apart; }
                    }
                    solo.pass_plan(authored)
                }
                None => authored.pass_plan(authored),
            };
        }
        // where each layer is in the passes: (pass, place in it)
        let at: Vec<(usize, usize)> = (0..n).map(|i| plan.iter().enumerate()
            .find_map(|(p, pass)| pass.layers.iter().position(|&g| g == i).map(|k| (p, k))).unwrap_or((0, i))).collect();
        if let Some(plan)=&self.layers.world_fields.plan {
            c.occurrences=plan.placements.clone();
        }
        if let Some(layer) = self.state.edit_scope.object().filter(|i| *i < n) {
            // Build a presentation copy. Object isolation never alters a saved
            // show flag, Boolean relationship or material assignment.
            let mut solo = self.state.stack.clone();
            solo.world_mods.clear();
            solo.hierarchy = None;
            for i in 0..n {
                solo.layers[i].show = i == layer;
                for j in 0..i {
                    solo.pair_mut(gf_core::layers::mask_of(&[i,j])).unwrap().op = gf_core::layers::Op::Apart;
                }
            }
            let authored_bodies = (c.body, c.bodies);
            c = solo.combine();
            c.occurrences.clear();
            (c.body, c.bodies) = authored_bodies;
        }
        // A tube that takes part in a seam wider than its own volume is exact needs its wide volume: built
        // once for the shape as it stands, the first time such a seam is asked for.
        let mut uses: Vec<Option<WideUse>> = vec![None; n.max(W)];
        for i in 0..n {
            let Some(b) = self.layers.built[i].as_ref() else { continue };
            let Some((from, to)) = b.exact_to else {
                self.baker.drop_wide(i);
                continue;
            };
            // (a sculpted layer's wide volume would be of the shape before the studio: it goes without)
            let sculpted = self.state.layer_bake(i).is_some();
            // (an object with a tree is read from the tree's volume: its own tube's wide volume is not the tree)
            let reach = if multi { plan[at[i].0].combine.reach(at[i].1) } else { c.reach(i) };
            if self.layers.no_wide || sculpted || reach <= to || b.key.contains("|at") || self.state.has_tree(i) { continue; }
            let (res, built_as, wide) = (b.res, b.mode, b.wide);
            let wres = wide_res(res);
            if wide != Some(wres) {
                let t = Instant::now();
                // the tube as its own volume was built from it (the one just made, or made again the same way)
                let job = match jobs[i].take() {
                    Some(j) => j,
                    None => self.state.layer_job(i, built_as, Some(res))?,
                };
                let t1 = Instant::now();
                if let ShapeJob::Curve { curve, melt } = &job.shape {
                    self.baker.set_layer(i);
                    self.baker.curve_wide(curve, wres, *melt)?;
                }
                work += (t1 - t).as_secs_f64() * 1e3;
                hand += t1.elapsed().as_secs_f64() * 1e3;
                if let Some(b) = self.layers.built[i].as_mut() { b.wide = Some(wres); }
                self.layers.wide_builds[i] += 1;
            }
            uses[i] = Some(WideUse { from, to, res: wres });
        }
        laps.lap("plan+wide");
        self.baker.set_layer(0);
        for i in 0..n {
            // a layer built where it is placed is read as it is
            if self.layers.built[i].as_ref().is_some_and(|b| b.key.contains("|at")) {
                if multi {
                    let (p, k) = at[i];
                    plan[p].combine.inv[k] = Placement::default().inverse();
                    plan[p].combine.scale[k] = 1.0;
                } else {
                    c.inv[i] = Placement::default().inverse();
                    c.scale[i] = 1.0;
                }
            }
        }
        // Each object with a tree: the tree measured into a volume of its own stands for it in the combining
        // step (made again only when the object or its tree has changed).
        let mut trees = vec![false; n];
        for i in 0..n {
            if let Some(made) = self.csg_layer_volume(i, rebuilt.contains(&i), mode)? {
                trees[i] = true;
                if made && !rebuilt.contains(&i) { rebuilt.push(i); }
            }
        }
        self.csg_grids_done(mode);
        self.sculptor.keep_only(&kept);
        // the combining step, unless nothing at all has changed since the last one
        let res = self.state.scene_res(mode);
        let originals: Vec<Volume> = (0..n).map(|i| {
            let tree = if trees[i] { self.csg.layer_vols.get(i).and_then(|v| v.as_ref()).map(|v| &v.1) } else { None };
            tree.or_else(|| self.baker.layer_volume(i)).cloned().ok_or_else(|| format!("layer {} has no volume", i + 1))
        }).collect::<R<_>>()?;
        let groups_consume_world=self.state.edit_scope.object().is_none() && self.state.stack.hierarchy.as_ref().is_some_and(|h|h.changes_geometry());
        let fields_changed=self.layers.world_fields.prepare(&self.gpu,&originals,&self.csg.faces,&rebuilt,&self.state.stack.world_source_bounds,res,c.box_k,groups_consume_world,mode.quality(),&mut self.sculptor)?;
        if fields_changed {self.state.stack.world_field_bounds=self.layers.world_fields.measured_bounds();}
        let mut vols=Vec::with_capacity(n);
        let mut projected=[false;MAX_LAYERS];
        for (i,base) in originals.iter().enumerate() {
            if let Some((field,scale))=self.layers.world_fields.volume(i) {
                vols.push(field);
                projected[i]=true;
                c.inv[i]=Placement {size:scale,..Default::default()}.inverse();
                c.scale[i]=scale;
                if let Some(instances)=c.occurrences.get_mut(i) {instances.clear();}
                uses[i]=None;
            } else {vols.push(base);}
        }
        let wides: Vec<Option<(&Volume, WideUse)>> = (0..n).map(|i| uses[i].and_then(|u| self.baker.wide_volume(i).map(|v| (v, u)))).collect();
        let sizes: Vec<u32> = vols.iter().map(|v| v.res).collect();
        let groups = self.state.stack.hierarchy.as_ref().filter(|h| self.state.edit_scope.object().is_none() && h.changes_geometry());
        let group_key = groups.map(|h| {
            let phases: Vec<_> = h.rows().into_iter().filter_map(|(_, n)| match n {
                gf_core::groups::Node::Group(g) => g.mods.lab_live(), _ => None,
            }).flat_map(|s| s.periods()).map(|p| self.state.cycles(p)).collect();
            format!("{}|{phases:?}|{}|{exact}", serde_json::to_string(h).unwrap_or_default(), self.sculptor.footprint)
        });
        let same = if multi {
            !fields_changed && self.layers.combined && rebuilt.is_empty() && !self.layers.force && self.layers.last.is_none()
                && self.layers.last_multi.as_ref().is_some_and(|l| l.0 == plan && l.1 == res && l.2 == sizes && l.3 == uses)
        } else {
            !fields_changed && group_key == self.layers.group_key && self.layers.combined && rebuilt.is_empty() && !self.layers.force && self.layers.last.as_ref().is_some_and(|l| l.0 == c && l.1 == res && l.2 == sizes && l.3 == uses)
                && self.layers.combiner.bodies_volume().is_some() == (c.needs_body_channels() && self.layers.combiner.bodies_wanted)
        };
        laps.lap("trees+fields");
        // (in the window nothing waits for the GPU: see `bake`)
        let timed = self.waits_for(mode);
        let t2 = Instant::now();
        if timed { self.gpu.wait(); }
        let t3 = Instant::now();
        if !same {
            if multi {
                self.layers.group_scratch = None;
                let none: Vec<Vec<Placement>> = Vec::new();
                let channels = if self.state.stack.body_channels() == 2 { 2 } else { 4 };
                self.layers.combiner.run_passes(&vols, &wides, &plan, &none, channels, res)?;
            } else if groups.is_some() {
                crate::group_bake::run(&self.state, &self.gpu, &vols, &wides, &projected, &c, res, !exact,
                    &mut self.sculptor, &mut self.layers.group_scratch, &mut self.layers.combiner)?;
            } else {
                self.layers.group_scratch = None;
                self.layers.combiner.run(&vols, &wides, &c, res)?;
            }
            self.layers.group_key = group_key;
            if timed { self.gpu.wait(); }
            self.layers.combine_ms = t3.elapsed().as_secs_f64() * 1e3;
            self.layers.combine_ms_sum += self.layers.combine_ms;
            self.layers.combines += 1;
            if multi {
                self.layers.last = None;
                self.layers.last_multi = Some((plan, res, sizes, uses));
            } else {
                self.layers.last = Some((c, res, sizes, uses));
                self.layers.last_multi = None;
            }
        }
        laps.lap(if same { "same" } else { "combine" });
        self.layers.combined = true;
        self.layers.rebuilt = rebuilt;
        self.last_bake_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.last_bake_parts = [work, hand, (t3 - t2).as_secs_f64() * 1e3];
        let (mut step, mut out, mut inn, pieces) = self.state.march();
        if self.layers.stride > 0.0 && self.layers.stride != 1.0 {
            step *= self.layers.stride;
            out = (out as f64 / self.layers.stride).ceil() as u32;
            inn = (inn as f64 / self.layers.stride).ceil() as u32;
        }
        self.layers.march = (step, out, inn, pieces);
        self.renderer.set_bodies(self.layers.combiner.bodies_volume().map(|v| (v, self.state.stack.body_channels() as u32)));
        // (several pieces are not one cut stone: the tracer marches through them)
        self.renderer.set_gem(None);
        // (a scene with a tree in it reads the tree's volume: its normals guarded, sdf.rs guard_normals)
        if !self.set_group_formula(res, mode)? {
            if trees.iter().any(|t| *t) { self.renderer.set_sdf(Some((crate::sdf::GUARD_KEY, Some(crate::sdf::GUARD_DECL.to_string()), &[0.0; 4]))); } else { self.renderer.set_sdf(None); }
        }
        self.csg.active = false;
        let vol = self.layers.combiner.volume().ok_or("no volume")?;
        self.renderer.set_scene(&self.lights_view, vol, step, out, inn, pieces);
        laps.done("tracer");
        self.trim_spares();
        self.dirty = true;
        Ok(())
    }

    /// Time the combining step alone: `n` runs with the layers as they are, each waited for. Milliseconds a run.
    pub fn bench_combine(&mut self, n: u32, mode: Detail) -> R<f64> {
        if self.state.plain() { return Err("there is one plain layer: nothing to combine".into()); }
        // (a scene of more layers than one pass takes is timed in its passes)
        let (c, plan, uses) = match (self.layers.last.clone(), self.layers.last_multi.clone()) {
            (Some((c, _, _, uses)), _) => (Some(c), Vec::new(), uses),
            (None, Some((plan, _, _, uses))) => (None, plan, uses),
            _ => return Err("the layers have not been combined yet".into()),
        };
        let res = self.state.scene_res(mode);
        let count = self.state.layer_count();
        let vols: Vec<&Volume> = (0..count).map(|i| {
            self.layers.world_fields.volume(i).map(|(v,_)|v)
                .or_else(||self.csg.layer_vols.get(i).and_then(|slot|slot.as_ref()).filter(|_|self.state.has_tree(i)).map(|(_,v)|v))
                .or_else(||self.baker.layer_volume(i)).ok_or_else(||format!("layer {} has no volume",i+1))
        }).collect::<R<_>>()?;
        let wides: Vec<Option<(&Volume, WideUse)>> = (0..count).map(|i| uses[i].and_then(|u| self.baker.wide_volume(i).map(|v| (v, u)))).collect();
        let channels = if self.state.stack.body_channels() == 2 { 2 } else { 4 };
        let once = |combiner: &mut Combiner| match &c {
            Some(c) => combiner.run(&vols, &wides, c, res),
            None => combiner.run_passes(&vols, &wides, &plan, &[], channels, res),
        };
        for _ in 0..3 {
            once(&mut self.layers.combiner)?;
        }
        self.gpu.wait();
        let t = Instant::now();
        for _ in 0..n {
            once(&mut self.layers.combiner)?;
            self.gpu.wait();
        }
        Ok(t.elapsed().as_secs_f64() * 1e3 / n.max(1) as f64)
    }

    /// What the layers came to, as JSON members to add to a report (nothing for one plain layer):
    /// how many layers, the combined volume's size, which layers the last build built again and how long
    /// its combining step took, and since the start how often each layer was built and the layers combined.
    pub fn layers_json(&self) -> String {
        if !self.layers.combined { return String::new(); }
        let n = self.state.layer_count();
        let l = &self.layers;
        let list = |v: &[u32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let rebuilt: Vec<u32> = l.rebuilt.iter().map(|i| *i as u32 + 1).collect();
        let (step, out, inn, pieces) = l.march;
        let bodies: Vec<String> = self.state.bodies().iter().map(|b| format!("[{}]", b.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join(","))).collect();
        format!(
            ",\"layers\":{n},\"sceneVolume\":{},\"rebuilt\":[{}],\"combineMs\":{:.3},\"builds\":[{}],\"wideBuilds\":[{}],\"combines\":{},\"combineMsMean\":{:.3},\"step\":{step:.2},\"steps\":[{out},{inn}],\"pieces\":{pieces},\"bodies\":[{}],\"bodiesVolume\":{},\"bodiesChannels\":{},\"volumesMB\":{:.0}",
            l.combiner.volume().map_or(0, |v| v.res), list(&rebuilt), l.combine_ms, list(&l.builds[..n]), list(&l.wide_builds[..n]), l.combines, l.combine_ms_sum / l.combines.max(1) as f64, bodies.join(","), self.bodies_volume().is_some(), self.layers.combiner.bodies_channels(), self.volumes_bytes() as f64 / 1048576.0
        )
    }
}
