//! Layers on the GPU: each layer's shape is built into its own volume by the baker, exactly as a single
//! shape is, and a combining step writes the one volume the tracer reads (shaders/combine.wgsl, the
//! arithmetic of gf_core::layers).
//!
//! A layer's volume stays good while only its placement or the way it combines changes, so those cost
//! the combining step alone; a layer is built again only when its own shape changes.

use std::time::Instant;

use gf_core::layers::{Combine, Placement, WideUse, UNIFORM_FLOATS};
use gf_core::math::{apply, euler};
use gf_core::presets::Shape;
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
    /// volumes kept for reuse, and which of them holds the scene now
    vols: Vec<Volume>,
    cur: usize,
    /// the bodies' volume, while there is more than one body to tell apart, and how many channels it has
    bodies: Option<Volume>,
    bodies_channels: u32,
    /// a small one kept from the last time things moved, with its channels
    bodies_small: Option<(Volume, u32)>,
    /// Make the bodies' volume when more than one body shows. Whoever gives bodies their materials can
    /// switch this off while every body has the same one (nothing then reads it, and it is two or four
    /// volumes of memory), and must ask for the volume to be made again (`state.want`) on switching it on.
    pub bodies_wanted: bool,
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
        ]);
        let pipe = if gpu.storage16 {
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
            let text = if two_ok { COMBINE_WGSL } else { COMBINE_WGSL.split("//#compute2").next().unwrap() };
            let module = gpu.shader("combine (computed)", text, false);
            let make = |label, l_out: &wgpu::BindGroupLayout, entry| {
                let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(&l_in), Some(l_out)], immediate_size: 0 });
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
            let module = gpu.shader("combine", COMBINE_WGSL.split("//#compute").next().unwrap(), false);
            let make = |label, fs, format| gpu.pipeline(&PipeDesc { label, module: &module, vs: "vs", fs, layouts: &[&l_in], format, blend: None, strip: false, buffers: &[] });
            Pipe::Drawn { scene: make("combine", "fs", VOL_FORMAT), four: make("combine bodies", "fs_bodies", bodies_format(4)), two: make("combine two bodies", "fs_bodies", bodies_format(2)) }
        };
        Self {
            gpu: gpu.clone(),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            u: gpu.uniform_buffer("combine", (UNIFORM_FLOATS * 4) as u64),
            l_in, pipe,
            blank: Volume::new(gpu, 8, false),
            vols: Vec::new(), cur: 0, bodies: None, bodies_channels: 0, bodies_small: None, bodies_wanted: true,
        }
    }

    /// The combined volume.
    pub fn volume(&self) -> Option<&Volume> {
        self.vols.get(self.cur)
    }
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
        self.bodies_small = None;
    }

    /// A volume of this size to write: one kept from before, or a new one. Big ones are not hoarded.
    fn take(&mut self, res: u32) -> usize {
        if let Some(i) = self.vols.iter().position(|v| v.res == res) {
            self.cur = i;
            return i;
        }
        if res > 384 {
            self.vols.retain(|v| v.res <= 384);
        }
        self.vols.push(Volume::new(&self.gpu, res, matches!(self.pipe, Pipe::Computed { .. })));
        self.cur = self.vols.len() - 1;
        self.cur
    }

    /// Combine the layers' volumes (`layers[i]` is layer i's) into a volume of `res` voxels a side.
    /// `wide[i]`: layer i's wide volume and how it is used, for the layers that have one.
    pub fn run(&mut self, layers: &[&Volume], wide: &[Option<(&Volume, WideUse)>], c: &Combine, res: u32) -> R<()> {
        if res % 8 != 0 { return Err(format!("a volume's size must be a multiple of 8, not {res}")); }
        if layers.len() < c.n { return Err(format!("{} layers to combine, but only {} have been built", c.n, layers.len())); }
        let gpu = self.gpu.clone();
        // each layer's volume holds values out to the centres of its outermost voxels
        let mut lim = [HB; 4];
        for (i, v) in layers.iter().enumerate().take(4) {
            lim[i] = HB - HB / v.res as f64;
        }
        let mut uses = [None; 4];
        for (i, w) in wide.iter().enumerate().take(c.n.min(4)) {
            uses[i] = w.map(|w| w.1);
        }
        gpu.queue.write_buffer(&self.u, 0, bytemuck::cast_slice(&c.uniform(res, lim, uses)));
        let vi = self.take(res);
        let view = |i: usize| &layers.get(i).filter(|_| i < c.n).copied().unwrap_or(&self.blank).view;
        let far = |i: usize| &wide.get(i).filter(|_| i < c.n).copied().flatten().map_or(&self.blank, |w| w.0).view;
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
            ],
        });
        // The bodies' volume: only while more than one body shows. Two channels do for two bodies (twice
        // the size of the scene's volume); three or four bodies take four (four times its size).
        if c.bodies_shown() > 1 && self.bodies_wanted {
            let two = c.bodies == 2 && !matches!(self.pipe, Pipe::Computed { two: None, .. });
            let channels = if two { 2 } else { 4 };
            if self.bodies.as_ref().is_none_or(|b| b.res != res) || self.bodies_channels != channels {
                // A small one is kept while the full one is in use and the other way round is not (an
                // animation alternates between the two; a large one given up first, so that two large
                // ones never exist together).
                if let Some(old) = self.bodies.take() {
                    if old.res <= 384 { self.bodies_small = Some((old, self.bodies_channels)); }
                }
                self.bodies = match self.bodies_small.take() {
                    Some((v, ch)) if v.res == res && ch == channels => Some(v),
                    kept => {
                        self.bodies_small = kept.filter(|_| res > 384);
                        let tex = gpu.volume("bodies", res, res, bodies_format(channels), matches!(self.pipe, Pipe::Computed { .. }));
                        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                        Some(Volume { tex, view, res })
                    }
                };
                self.bodies_channels = channels;
            }
        } else {
            self.bodies = None;
            self.bodies_small = None;
        }
        let out = &self.vols[vi].view;
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("combine") });
        fn view_of(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::TextureView(v)
        }
        match (&self.pipe, &self.bodies) {
            (Pipe::Computed { scene, l_scene, four, two }, bodies) => {
                let (pipe, g_out) = match bodies {
                    Some(b) => {
                        let both = two.as_ref().filter(|_| self.bodies_channels == 2).unwrap_or(four);
                        (&both.pipe, gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &both.layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: view_of(out) }, wgpu::BindGroupEntry { binding: both.binding, resource: view_of(&b.view) }] }))
                    }
                    None => (scene, gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: l_scene, entries: &[wgpu::BindGroupEntry { binding: 0, resource: view_of(out) }] })),
                };
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                p.set_pipeline(pipe);
                p.set_bind_group(0, &g_in, &[]);
                p.set_bind_group(1, &g_out, &[]);
                p.dispatch_workgroups(res / 8, res / 8, res / 4);
            }
            (Pipe::Drawn { scene, four, two }, bodies) => {
                let draw_bodies = if self.bodies_channels == 2 { two } else { four };
                for z in 0..res {
                    let mut rp = pass(&mut enc, out, Some(z), true);
                    rp.set_pipeline(scene);
                    rp.set_bind_group(0, &g_in, &[]);
                    rp.draw(0..3, z..z + 1);
                }
                if let Some(b) = bodies {
                    for z in 0..res {
                        let mut rp = pass(&mut enc, &b.view, Some(z), true);
                        rp.set_pipeline(draw_bodies);
                        rp.set_bind_group(0, &g_in, &[]);
                        rp.draw(0..3, z..z + 1);
                    }
                }
            }
        }
        gpu.queue.submit([enc.finish()]);
        Ok(())
    }
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
        matches!(self.mode, Detail::Full | Detail::Hq)
    }
}

/// Voxels per side of the wide volume that goes with a tube's own volume of this size.
fn wide_res(res: u32) -> u32 {
    if res >= 512 { 192 } else if res >= 256 { 128 } else { 64 }
}

/// The engine's side of the layers: the combining step, what each layer's volume holds, and what the
/// last build did.
pub struct LayerWork {
    pub combiner: Combiner,
    built: Vec<Option<Built>>,
    /// what was last combined: the numbers, the size, each layer's own size and the use of its wide volume
    last: Option<(Combine, u32, Vec<u32>, [Option<WideUse>; 4])>,
    /// the tracer reads the combined volume (the last build was of layers)
    pub combined: bool,
    /// which layers the last build built again
    pub rebuilt: Vec<usize>,
    /// milliseconds the last combining step took (to the end, when builds are waited for)
    pub combine_ms: f64,
    /// since the start: how often each layer was built, how often the layers were combined, and the
    /// combining steps' milliseconds added up
    pub builds: [u32; 4],
    /// and how often its wide volume
    pub wide_builds: [u32; 4],
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
    pub fn new(gpu: &Gpu) -> Self {
        Self { combiner: Combiner::new(gpu), built: Vec::new(), last: None, combined: false, rebuilt: Vec::new(), combine_ms: 0.0, builds: [0; 4], wide_builds: [0; 4], combines: 0, combine_ms_sum: 0.0, force: false, stride: 1.0, in_place: false, march: (0.0, 0, 0, 0), no_wide: false }
    }
    /// A single shape has just been built the plain way: remember what the first layer's volume holds,
    /// so that it need not be built again when a second layer arrives, and let go of everything that
    /// was kept for layers that are no longer there.
    /// The last build was of one plain layer (its volume is the one the tracer reads).
    pub fn built_plain_still(&self) -> bool {
        !self.combined && self.built.len() == 1
    }
    pub fn built_plain(&mut self, baker: &mut crate::bake::Baker, key: String, res: u32, mode: Detail, melt: Option<f64>) {
        baker.drop_layers(1);
        baker.drop_wide(0);
        self.combiner.clear();
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
        if self.layers.combined { self.layers.combiner.bodies_volume() } else { None }
    }

    /// The bodies as whatever draws them needs them: the volume, how many bodies there are (channels
    /// 0 to count - 1 are theirs; no other channel is to be read), and how many channels the volume has.
    pub fn bodies(&self) -> Option<Bodies<'_>> {
        let volume = self.bodies_volume()?;
        Some(Bodies { volume, count: self.state.stack.bodies().len() as u32, channels: self.layers.combiner.bodies_channels() })
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
        let scene = if self.layers.combined { size(self.layers.combiner.volume(), 1) } else { 0 };
        layers + scene + size(self.bodies_volume(), self.layers.combiner.bodies_channels() as u64)
    }

    /// The volume the tracer reads: the one shape's, or the layers combined.
    pub fn scene_volume(&self) -> Option<&Volume> {
        if self.state.empty { self.csg.empty_vol.as_ref() } else if self.layers.combined { self.layers.combiner.volume() } else if let Some(v) = self.csg.volume() { Some(v) } else { self.baker.layer_volume(0) }
    }

    /// Build what has changed of the layers, this carefully, combine them, and point the tracer at the result.
    pub(crate) fn bake_layers(&mut self, mode: Detail) -> R<()> {
        let t0 = Instant::now();
        let n = self.state.layer_count();
        let exact = matches!(mode, Detail::Full | Detail::Hq);
        self.layers.built.resize_with(n, || None);
        self.baker.drop_layers(n);
        let mut rebuilt = Vec::new();
        let mut jobs: Vec<Option<Job>> = (0..n).map(|_| None).collect();
        let (mut work, mut hand) = (0.0, 0.0);
        for i in 0..n {
            // A layer's volume will do if it holds the shape as it stands; for a full build it must also
            // be the full size and exact. (While something else moves, a still layer keeps its fine volume.)
            let place = self.state.stack.layers[i].place;
            // (not an object with a tree: its tree is measured into a volume of its own, in its own frame)
            let in_place = self.layers.in_place && place.size == 1.0 && !place.is_identity() && self.state.layer_family(i).kind == gf_core::presets::Kind::Curve && !self.state.has_tree(i);
            let mut key = self.state.layer_key(i);
            if in_place { key.push_str(&format!("|at{:?}{:?}", place.pos, place.rot)); }
            let res = self.state.layer_res(i, mode);
            let good = !self.layers.force && self.layers.built[i].as_ref().is_some_and(|b| b.key == key && (!exact || (b.exact() && b.res == res)));
            if good { continue; }
            let t = Instant::now();
            let mut job = self.state.layer_job(i, mode, None)?;
            if in_place {
                if let ShapeJob::Curve { curve, .. } = &mut job.shape {
                    let r = euler(place.rot[0], place.rot[1], place.rot[2]);
                    for q in curve.pts.chunks_exact_mut(4) {
                        let w = apply(&r, [q[0] as f64, q[1] as f64, q[2] as f64]);
                        for a in 0..3 {
                            q[a] = (w[a] + place.pos[a]) as f32;
                        }
                    }
                    // a flat cut goes up and down with the layer; turned, it would no longer be level
                    let turned = place.rot.iter().any(|&a| a != 0.0);
                    curve.cut = if turned { None } else { curve.cut.map(|y| y + place.pos[1]) };
                }
            }
            let t1 = Instant::now();
            self.baker.set_layer(i);
            let fam = &self.state.families[job.family];
            let (stats, melt): (BakeStats, Option<f64>) = match &job.shape {
                ShapeJob::Curve { curve, melt } => (self.baker.curve(curve, job.res, *melt)?, Some(*melt)),
                ShapeJob::Field { uniforms } => {
                    let p = &fam.presets[job.preset];
                    debug_assert!(matches!(p.shape, Shape::Field { .. }));
                    (self.baker.field(&format!("{}/{}", fam.id, p.id), p, uniforms, job.res)?, None)
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
        // The surface studio of each layer: run on a layer just built (its shape kept aside first), or on
        // one whose studio alone has changed (from the shape kept aside); a layer whose studio was emptied
        // gets its shape back.
        self.sculptor.drop_layers(n);
        for i in 0..n {
            let Some(b) = self.layers.built[i].as_ref() else { continue };
            let skey = format!("{}|{}|{}", b.key, b.res, b.exact());
            let fresh = rebuilt.contains(&i);
            let s = self.state.layer_sculpt(i).filter(|s| !s.is_empty()).cloned().filter(|_| self.sculptor.available());
            let vol = self.baker.layer_volume(i).ok_or_else(|| format!("layer {} has no volume", i + 1))?;
            let applied = self.sculptor.applied.get(i).cloned().unwrap_or_default();
            match s {
                Some(s) => {
                    let phases = self.state.sculpt_phases(i);
                    let want = format!("{skey}#{}", s.key(&phases, self.sculptor.footprint));
                    if fresh || (applied.is_empty() && !self.sculptor.has(&skey)) { self.sculptor.keep(i, vol, &skey); }
                    if fresh || applied != want {
                        if !self.sculptor.has(&skey) {
                            // (the shape kept aside has gone: build the layer again next time)
                            self.layers.built[i] = None;
                            self.state.want = Some(mode);
                            continue;
                        }
                        self.sculptor.run(i, vol, &skey, &s, &phases, matches!(mode, Detail::Full) || self.time_bakes, !matches!(mode, Detail::Full | Detail::Hq))?;
                        if !fresh { rebuilt.push(i); }
                    }
                }
                None => {
                    if !fresh && !applied.is_empty() {
                        if !self.sculptor.restore(i, vol, &skey) {
                            self.layers.built[i] = None;
                            self.state.want = Some(mode);
                            continue;
                        }
                        rebuilt.push(i);
                    }
                    self.sculptor.forget(i);
                }
            }
        }
        // (the shapes kept aside for layers that no longer have a studio, or are gone, are let go)
        let kept: Vec<String> = (0..n).filter(|&i| self.state.layer_sculpt(i).is_some_and(|s| !s.is_empty())).filter_map(|i| self.layers.built[i].as_ref().map(|b| b.key.clone())).collect();
        self.sculptor.keep_only(&kept);
        let mut c = self.state.stack.combine();
        // A tube that takes part in a seam wider than its own volume is exact needs its wide volume: built
        // once for the shape as it stands, the first time such a seam is asked for.
        let mut uses: [Option<WideUse>; 4] = [None; 4];
        for i in 0..n {
            let Some(b) = self.layers.built[i].as_ref() else { continue };
            let Some((from, to)) = b.exact_to else {
                self.baker.drop_wide(i);
                continue;
            };
            // (a sculpted layer's wide volume would be of the shape before the studio: it goes without)
            let sculpted = self.state.layer_sculpt(i).is_some_and(|s| !s.is_empty());
            // (an object with a tree is read from the tree's volume: its own tube's wide volume is not the tree)
            if self.layers.no_wide || sculpted || c.reach(i) <= to || b.key.contains("|at") || self.state.has_tree(i) { continue; }
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
        self.baker.set_layer(0);
        for i in 0..n {
            // a layer built where it is placed is read as it is
            if self.layers.built[i].as_ref().is_some_and(|b| b.key.contains("|at")) {
                c.inv[i] = Placement::default().inverse();
                c.scale[i] = 1.0;
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
        // the combining step, unless nothing at all has changed since the last one
        let res = self.state.scene_res(mode);
        let vols: Vec<&Volume> = (0..n).map(|i| {
            let tree = if trees[i] { self.csg.layer_vols.get(i).and_then(|v| v.as_ref()).map(|v| &v.1) } else { None };
            tree.or_else(|| self.baker.layer_volume(i)).ok_or_else(|| format!("layer {} has no volume", i + 1))
        }).collect::<R<_>>()?;
        let wides: Vec<Option<(&Volume, WideUse)>> = (0..n).map(|i| uses[i].and_then(|u| self.baker.wide_volume(i).map(|v| (v, u)))).collect();
        let sizes: Vec<u32> = vols.iter().map(|v| v.res).collect();
        let same = self.layers.combined && rebuilt.is_empty() && !self.layers.force && self.layers.last.as_ref().is_some_and(|l| l.0 == c && l.1 == res && l.2 == sizes && l.3 == uses)
            && self.layers.combiner.bodies_volume().is_some() == (c.bodies_shown() > 1 && self.layers.combiner.bodies_wanted);
        // (in the window nothing waits for the GPU: see `bake`)
        let timed = (matches!(mode, Detail::Full) && self.blocking()) || self.time_bakes;
        let t2 = Instant::now();
        if timed { self.gpu.wait(); }
        let t3 = Instant::now();
        if !same {
            self.layers.combiner.run(&vols, &wides, &c, res)?;
            if timed { self.gpu.wait(); }
            self.layers.combine_ms = t3.elapsed().as_secs_f64() * 1e3;
            self.layers.combine_ms_sum += self.layers.combine_ms;
            self.layers.combines += 1;
            self.layers.last = Some((c, res, sizes, uses));
        }
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
        self.renderer.set_bodies(self.layers.combiner.bodies_volume().map(|v| (v, self.state.stack.bodies().len() as u32)));
        // (several pieces are not one cut stone: the tracer marches through them)
        self.renderer.set_gem(None);
        self.renderer.set_sdf(None);
        self.csg.active = false;
        let vol = self.layers.combiner.volume().ok_or("no volume")?;
        self.renderer.set_scene(&self.lights_view, vol, step, out, inn, pieces);
        self.dirty = true;
        Ok(())
    }

    /// Time the combining step alone: `n` runs with the layers as they are, each waited for. Milliseconds a run.
    pub fn bench_combine(&mut self, n: u32, mode: Detail) -> R<f64> {
        if self.state.plain() { return Err("there is one plain layer: nothing to combine".into()); }
        let Some((c, _, _, uses)) = self.layers.last.clone() else { return Err("the layers have not been combined yet".into()) };
        let res = self.state.scene_res(mode);
        let count = self.state.layer_count();
        let vols: Vec<&Volume> = (0..count).map(|i| self.baker.layer_volume(i).ok_or_else(|| format!("layer {} has no volume", i + 1))).collect::<R<_>>()?;
        let wides: Vec<Option<(&Volume, WideUse)>> = (0..count).map(|i| uses[i].and_then(|u| self.baker.wide_volume(i).map(|v| (v, u)))).collect();
        for _ in 0..3 {
            self.layers.combiner.run(&vols, &wides, &c, res)?;
        }
        self.gpu.wait();
        let t = Instant::now();
        for _ in 0..n {
            self.layers.combiner.run(&vols, &wides, &c, res)?;
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
