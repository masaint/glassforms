//! The light inside a body that scatters it, worked out into a small volume for the tracer to read
//! (shaders/scatter.wgsl says how; gf_core::scatter what the lighting and the body are told as).
//!
//! The volume does not depend on the camera. It is worked out again whenever the lights, the shape or
//! the material change, from nothing each time: a small volume first, then one twice as fine that
//! starts from it, and so on up to the size asked for.

use bytemuck::{Pod, Zeroable};
use gf_core::scatter::{Lighting, Medium, MAX_BEAMS};
use gf_core::HB;

use crate::bake::Volume;
use crate::gpu::*;

#[path = "scatter_sparse.rs"]
mod sparse;

pub const SCATTER_WGSL: &str = include_str!("shaders/scatter.wgsl");
/// What the tracer reads, and what the beams' own light is kept in: 16-bit floats, interpolated.
const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// What the relaxation works in: its steps are small differences of large numbers.
const WORK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// The radius of the sphere every shape fits in (the tracer's BOUND).
const BOUND: f32 = 1.47;
/// How far a relaxation step moves a cell past what its neighbours say, on a volume `res` cells a side.
/// Going past by the right amount is what lets a small volume reach equilibrium in a number of steps like
/// its size (successive over-relaxation; the amount is the textbook one for a body half the volume
/// across). The larger volumes only have fine detail left to settle, which a plain step does best.
fn over(res: u32) -> f64 {
    if res > 80 { return 1.0; }
    2.0 / (1.0 + (std::f64::consts::PI / (0.5 * res as f64).max(2.0)).sin())
}
/// The beams march through the shape's own volume up to this size; a larger one through a copy this size.
const SHAPE_COPY: u32 = 256;
/// The smallest volume a fresh start begins from, in cells a side.
const COARSEST: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BeamU {
    dir: [f32; 4],
    rgb: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterU {
    res: f32,
    hb: f32,
    nb: u32,
    steps: u32,
    st: [f32; 4],
    ss: [f32; 4],
    k: [f32; 4],
    zb: [f32; 4],
    ab: [f32; 4],
    even: [f32; 4],
    lean: [[f32; 4]; 3],
    beams: [BeamU; MAX_BEAMS],
    /// which channel of the volume marched through is the body's distance; and which the copy is made from
    pick: [f32; 4],
    pick_src: [f32; 4],
    shading: Shading,
    /// x: how uneven the body's colour is (0 = even)
    mottle: [f32; 4],
}

/// What the bodies cast on the body whose light is worked out (scatter.wgsl: castThrough, sdfKeep): the
/// numbers cast.wgsl's shadow march is given, with the body itself left out. All nothing: nothing is cast.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug, Default)]
pub struct Shading {
    /// x: 1 = the other bodies' shadows are marched for (a body among several, each with its own light
    /// volume); y: 1 = several bodies of one material, a beam darkened only by the body its cell is in and
    /// those that cast; z: how far off its start a march first looks; w: its shortest stride
    pub cast: [f32; 4],
    /// for each body: it casts a shadow here; it lets nothing through; a large distance if it does not cast
    pub casts: [f32; 4],
    pub solid: [f32; 4],
    pub far: [f32; 4],
    /// each body: what a unit length of it takes from light (rgb), what its surfaces let through (w)
    pub body: [[f32; 4]; 4],
    /// how strong shadows are; how strong caustics are; cells a side of the volume of light arriving by
    /// way of the bodies (0: none); how far off the surface that light was landed
    pub caus: [f32; 4],
}

/// What the bodies cast, as the light volume reads it: the numbers, the distances to every body (one a
/// channel, cast.rs's modest copy), and the light arriving by way of the bodies, if any is made.
pub struct Casting {
    pub numbers: Shading,
    pub bodies: wgpu::TextureView,
    pub light: Option<wgpu::TextureView>,
}

/// One channel of four, as the numbers that pick it out of a texel.
fn one_hot(k: usize) -> [f32; 4] {
    let mut v = [0.0; 4];
    v[k.min(3)] = 1.0;
    v
}

/// One size of light volume and everything that reads and writes it.
struct Level {
    res: u32,
    u: wgpu::Buffer,
    /// the beams' own light in each cell, and each cell's distance to the surface
    src: wgpu::TextureView,
    /// scattered light: the relaxation reads one and writes the other, turn about
    phi: [wgpu::TextureView; 2],
    /// what the tracer reads
    out: wgpu::TextureView,
    /// the same body's light volume under even white light from everywhere, with nothing absorbed (see
    /// `Scatter::solve`): what this way of working the light out gives where the true answer is known
    calib: wgpu::TextureView,
    /// the scene's volume, the light volume of the size below, and what the bodies cast (their distances,
    /// the light arriving by way of them) that this level's first step was last bound to
    g_inject: Option<([wgpu::TextureView; 4], wgpu::BindGroup)>,
    g_begin: wgpu::BindGroup,
    /// read phi[i], write phi[1 - i]
    g_diffuse: [wgpu::BindGroup; 2],
    sparse: Option<sparse::Grid>,
    /// read phi[i], write the tracer's volume
    g_finish: [wgpu::BindGroup; 2],
    /// the body's balance: sums over blocks of cells of phi[i]; over blocks of those, twice; and
    /// phi[i] brought to the level the last sums give, written to phi[1 - i]
    g_sum: [wgpu::BindGroup; 2],
    g_fold: [wgpu::BindGroup; 2],
    g_level: [wgpu::BindGroup; 2],
    /// cells a side of the three sizes of sums
    sums: [u32; 3],
    /// which of the two holds the scattered light now
    cur: usize,
}

pub struct Scatter {
    gpu: Gpu,
    lin: wgpu::Sampler,
    p_inject: wgpu::ComputePipeline,
    p_inject_coarse: wgpu::ComputePipeline,
    sparse: Option<sparse::Passes>,
    /// makes the small copy of the shape's volume that the beams march through, where it can be made
    p_shape: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// that copy, and which shape it was made from (the volume, the count of its changes, the channel)
    copy: Option<(wgpu::TextureView, u64, usize, Volume)>,
    p_begin: wgpu::ComputePipeline,
    p_refine: wgpu::ComputePipeline,
    /// a relaxation step for the cells of each colour of the chessboard
    p_relax: [wgpu::ComputePipeline; 2],
    p_finish: wgpu::ComputePipeline,
    p_sum: wgpu::ComputePipeline,
    p_fold: wgpu::ComputePipeline,
    p_level: wgpu::ComputePipeline,
    l_inject: wgpu::BindGroupLayout,
    l_refine: wgpu::BindGroupLayout,
    l_step: wgpu::BindGroupLayout,
    l_finish: wgpu::BindGroupLayout,
    l_sum: wgpu::BindGroupLayout,
    l_fold: wgpu::BindGroupLayout,
    l_level: wgpu::BindGroupLayout,
    levels: Vec<Level>,
    /// the size of the volume the tracer reads now (0: none yet)
    pub res: u32,
    /// Relaxation steps (each moves the cells of one colour of the chessboard) on each size of volume,
    /// counted from the finest (the last is for every size smaller still), before the body is brought
    /// into balance as a whole; and how many after.
    pub steps: [u32; 5],
    pub steps_after: u32,
    /// stands in for the light volume until there is one
    blank: wgpu::TextureView,
    /// what the calibration in the finest volume was worked out for (see `solve`)
    calibrated: Option<Calibrated>,
    /// false: the light volume is used as it is worked out, without the calibration (for comparing)
    pub calibrate: bool,
    /// the box the shape's volume (and so the light volume) spans: [-hb, hb] (larger than ever when the
    /// layers are placed far out)
    pub hb: f64,
    /// the lamps standing in the scene, for the light worked out (not the furnace's): where each stands in
    /// the lights' own frame, its radius, and the light it sends (radiance times the area of its disc)
    pub lamps: Vec<[f64; 7]>,
}

/// What a calibration depends on: the shape (its volume, its changes, the channel), the cells a side, and
/// the body as a medium apart from what it absorbs.
#[derive(Clone, PartialEq)]
struct Calibrated {
    view: wgpu::TextureView,
    shape_no: u64,
    chan: usize,
    res: u32,
    medium: [u64; 6],
    hb: u64,
}

/// The lighting of a furnace: white light of brightness 1 from every direction, as the light volume
/// takes lighting (beams and the rest), made once.
fn furnace() -> &'static Lighting {
    static L: std::sync::OnceLock<Lighting> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        let (w, h) = (gf_core::env::ENV_W, gf_core::env::ENV_H);
        gf_core::scatter::beams(&vec![1.0f32; w * h * 4], w, h, None)
    })
}
/// What the light volume holds, in a body that absorbs nothing, under the furnace's light: the light
/// inside is the same everywhere, the light outside times the square of the index, from all 4 pi of
/// directions (the tracer turns it back into what is seen by the reverse of both).
fn furnace_level(ior: f64) -> f64 {
    4.0 * std::f64::consts::PI * ior * ior
}

/// The sizes a fresh start climbs through, smallest first, ending at `res`.
pub fn sizes(res: u32) -> Vec<u32> {
    let mut v = vec![res];
    while v[0] % 2 == 0 && v[0] / 2 >= COARSEST { v.insert(0, v[0] / 2); }
    v
}

fn entry(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
}

impl Scatter {
    pub fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let d3 = wgpu::TextureViewDimension::D3;
        let tex = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: d3, multisampled: false }, count: None };
        let smp = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
        let out = |binding, format| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: d3 }, count: None };
        let l_inject = gpu.bind_layout("scatter: beams", &[uniform_entry(0, c, false), tex(1), smp(2), out(3, OUT_FORMAT), tex(13), tex(14), tex(15)]);
        let l_step = gpu.bind_layout("scatter: relax", &[uniform_entry(0, c, false), tex(4), tex(5), out(6, WORK_FORMAT)]);
        let l_refine = gpu.bind_layout("scatter: refine", &[uniform_entry(0, c, false), smp(2), tex(4), tex(5), out(6, WORK_FORMAT)]);
        let l_finish = gpu.bind_layout("scatter: finish", &[uniform_entry(0, c, false), tex(4), tex(5), out(7, OUT_FORMAT), tex(16)]);
        let l_sum = gpu.bind_layout("scatter: sum", &[uniform_entry(0, c, false), tex(4), tex(5), out(10, WORK_FORMAT), out(11, WORK_FORMAT)]);
        let l_fold = gpu.bind_layout("scatter: fold", &[tex(8), tex(9), out(10, WORK_FORMAT), out(11, WORK_FORMAT)]);
        let l_level = gpu.bind_layout("scatter: level", &[tex(5), out(6, WORK_FORMAT), tex(8), tex(9)]);
        let module = gpu.shader("scatter", SCATTER_WGSL, false);
        // (a compute shader can write a one-channel 16-bit float volume only on some devices)
        let p_shape = gpu.storage16.then(|| {
            let l = gpu.bind_layout("scatter: shape", &[uniform_entry(0, c, false), tex(1), smp(2), out(12, crate::bake::VOL_FORMAT)]);
            let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scatter: shape"), bind_group_layouts: &[Some(&l)], immediate_size: 0 });
            (gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("scatter: shape"), layout: Some(&pl), module: &module, entry_point: Some("cs_shape"), compilation_options: Default::default(), cache: None }), l)
        });
        let pipe = |label: &str, layout: &wgpu::BindGroupLayout, entry: &str| {
            let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        Self {
            gpu: gpu.clone(),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            p_inject_coarse: pipe("scatter: coarse beams", &l_inject, "cs_inject_coarse"),
            p_inject: pipe("scatter: beams", &l_inject, "cs_inject"),
            p_shape, copy: None,
            p_begin: pipe("scatter: begin", &l_step, "cs_begin"),
            p_refine: pipe("scatter: refine", &l_refine, "cs_refine"),
            p_relax: [pipe("scatter: relax even", &l_step, "cs_even"), pipe("scatter: relax odd", &l_step, "cs_odd")],
            p_finish: pipe("scatter: finish", &l_finish, "cs_finish"),
            p_sum: pipe("scatter: sum", &l_sum, "cs_sum"),
            p_fold: pipe("scatter: fold", &l_fold, "cs_fold"),
            p_level: pipe("scatter: level", &l_level, "cs_level"),
            l_inject, l_refine, l_step, l_finish, l_sum, l_fold, l_level,
            sparse: None, levels: Vec::new(), res: 0, steps: [8, 12, 24, 24, 24], steps_after: 4,
            blank: blank(gpu), calibrated: None, calibrate: true, hb: HB, lamps: Vec::new(),
        }
    }

    /// The volume the tracer reads: the light inside the body, as last worked out.
    pub fn view(&self) -> &wgpu::TextureView {
        self.levels.iter().find(|l| l.res == self.res).map_or(&self.blank, |l| &l.out)
    }

    /// Video memory the light volumes hold, in bytes.
    pub fn bytes(&self) -> u64 {
        self.levels.iter().map(|l| (l.res as u64).pow(3) * (8 + 8 + 16 + 16)).sum()
    }

    fn level(&mut self, res: u32) -> usize {
        if let Some(i) = self.levels.iter().position(|l| l.res == res) { return i; }
        let gpu = &self.gpu;
        let make = |label: &str, format, res: u32| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }).create_view(&wgpu::TextureViewDescriptor::default())
        };
        let src = make("scatter: beams' light", OUT_FORMAT, res);
        let phi = [make("scatter: scattered light a", WORK_FORMAT, res), make("scatter: scattered light b", WORK_FORMAT, res)];
        let out = make("light volume", OUT_FORMAT, res);
        let calib = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light volume: calibration"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: OUT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default());
        // sums over blocks of 4 cells a side, then of 8 of those, then of 8 of those (one block in all up
        // to 256 cells a side)
        let sums = [res.div_ceil(4), res.div_ceil(32), res.div_ceil(256)];
        let sum: Vec<[wgpu::TextureView; 2]> = sums.iter().map(|&n| [make("scatter: added", WORK_FORMAT, n), make("scatter: lost", WORK_FORMAT, n)]).collect();
        let u = gpu.uniform_buffer("scatter", std::mem::size_of::<ScatterU>() as u64);
        let group = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries });
        let uniform = || wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() };
        let g_diffuse = [0, 1].map(|i| group(&self.l_step, &[uniform(), entry(4, &src), entry(5, &phi[i]), entry(6, &phi[1 - i])]));
        let g_finish = [0, 1].map(|i| group(&self.l_finish, &[uniform(), entry(4, &src), entry(5, &phi[i]), entry(7, &out), entry(16, &calib)]));
        // (beginning writes phi[0]; it reads phi[1] only because the layout says so)
        let g_begin = group(&self.l_step, &[uniform(), entry(4, &src), entry(5, &phi[1]), entry(6, &phi[0])]);
        let g_sum = [0, 1].map(|i| group(&self.l_sum, &[uniform(), entry(4, &src), entry(5, &phi[i]), entry(10, &sum[0][0]), entry(11, &sum[0][1])]));
        let g_fold = [0, 1].map(|i| group(&self.l_fold, &[entry(8, &sum[i][0]), entry(9, &sum[i][1]), entry(10, &sum[i + 1][0]), entry(11, &sum[i + 1][1])]));
        let g_level = [0, 1].map(|i| group(&self.l_level, &[entry(5, &phi[i]), entry(6, &phi[1 - i]), entry(8, &sum[2][0]), entry(9, &sum[2][1])]));
        let sparse = sparse::worthwhile(res).then(|| {
            self.sparse.get_or_insert_with(|| sparse::Passes::new(gpu)).grid(gpu,res,&u,&src,&phi,&sum[0])
        });
        self.levels.push(Level { res, u, src, phi, out, calib, g_inject: None, g_begin, g_diffuse, sparse, g_finish, g_sum, g_fold, g_level, sums, cur: 0 });
        self.levels.len() - 1
    }

    #[allow(clippy::too_many_arguments)]
    fn numbers(res: u32, light: &Lighting, m: &Medium, to_shape: &[f64; 9], pick: [[f32; 4]; 2], shading: Shading, correct: bool, hb: f64, lamps: &[[f64; 7]]) -> ScatterU {
        let beams = &light.beams;
        let h = 2.0 * hb / res as f64;
        let bound = BOUND as f64 * hb / HB;
        let (st, d, zb) = (m.sig_t(), m.diffusion(), m.beyond());
        let f = |v: [f64; 3], w: f64| [v[0] as f32, v[1] as f32, v[2] as f32, w as f32];
        let mut u = ScatterU {
            res: res as f32, hb: hb as f32, nb: (beams.len() + lamps.len()).min(MAX_BEAMS) as u32, steps: 64,
            st: f(st, bound * bound),
            ss: f(m.sig_s, over(res)),
            k: f([0, 1, 2].map(|i| h * h / d[i]), m.ior),
            zb: f([0, 1, 2].map(|i| h / zb[i]), m.enters),
            // (the last: what the calibration should read, or 0 where it is not applied)
            ab: f([0, 1, 2].map(|i| m.sig_a[i] * h * h / d[i]), if correct { furnace_level(m.ior) } else { 0.0 }),
            // (the last: the body lights itself where one part of it faces another)
            even: [0.0, 0.0, 0.0, 1.0],
            lean: [[0.0; 4]; 3],
            beams: [BeamU::zeroed(); MAX_BEAMS],
            pick: pick[0],
            pick_src: pick[1],
            shading,
            mottle: [m.mottle as f32, 0.0, 0.0, 0.0],
        };
        // The rest of the lighting: its brightness around direction d is  even + lean . d  (the lean is
        // turned from the lights' frame into the shape's, like the beams).
        for c in 0..3 {
            u.even[c] = light.rest.even[c] as f32;
            let l = light.rest.lean[c];
            for a in 0..3 {
                u.lean[a][c] = (to_shape[a * 3] * l[0] + to_shape[a * 3 + 1] * l[1] + to_shape[a * 3 + 2] * l[2]) as f32;
            }
        }
        for (i, b) in beams.iter().take(MAX_BEAMS).enumerate() {
            // from the lights' own frame into the shape's
            let r = to_shape;
            let d = [0, 1, 2].map(|k| r[k * 3] * b.dir[0] + r[k * 3 + 1] * b.dir[1] + r[k * 3 + 2] * b.dir[2]);
            u.beams[i] = BeamU { dir: [d[0] as f32, d[1] as f32, d[2] as f32, b.tan as f32], rgb: [b.rgb[0] as f32, b.rgb[1] as f32, b.rgb[2] as f32, 0.0] };
        }
        // The lamps, after the beams: a beam whose direction, width and strength each cell works out from
        // where the lamp stands (scatter.wgsl). dir: where it stands, in the shape's frame, and its radius;
        // rgb: the light it sends, and 1 to say it is a lamp.
        for (i, l) in lamps.iter().enumerate() {
            let k = beams.len().min(MAX_BEAMS) + i;
            if k >= MAX_BEAMS { break; }
            let r = to_shape;
            let p = [0, 1, 2].map(|a| r[a * 3] * l[0] + r[a * 3 + 1] * l[1] + r[a * 3 + 2] * l[2]);
            u.beams[k] = BeamU { dir: [p[0] as f32, p[1] as f32, p[2] as f32, l[3] as f32], rgb: [l[4] as f32, l[5] as f32, l[6] as f32, 1.0] };
        }
        u
    }

    /// Work the light volume out. `vol`: the shape's distance field, in channel `chan` of its texels (the
    /// bodies' volume holds one body a channel). `beams`: the lighting, in the lights' own frame;
    /// `to_shape`: that frame's directions in the shape's (a rotation, by rows). `res`: cells a side.
    /// `shape_no`: a number that changes whenever the volume holds another shape. `cast`: what the bodies
    /// cast on this one, if anything casts.
    ///
    /// The volume is calibrated. However it is worked out, a grid of cells a little finer than the pieces
    /// of the body, and light that spreads as heat does, are approximations, and they lose light where a
    /// piece is thin (a tube of Thomas held 0.67 of the light it should). Where the answer is known their
    /// error can be measured: in a furnace (white light from everywhere on a body that absorbs nothing)
    /// the light inside is the same everywhere. So the same body is worked out once in a furnace, with
    /// nothing absorbed, and every cell of the finest volume is divided by what that cell held there, as
    /// a share of what it should have held. The furnace does not depend on where the lights are or how
    /// the shape is turned: it is worked out again only when the shape, its detail or the body's
    /// scattering changes.
    #[allow(clippy::too_many_arguments)]
    pub fn solve(&mut self, vol: &Volume, shape_no: u64, light: &Lighting, m: &Medium, to_shape: &[f64; 9], res: u32, chan: usize, cast: Option<&Casting>) {
        let key = Calibrated {
            view: vol.view.clone(),
            shape_no,
            chan,
            res,
            medium: [m.sig_s[0], m.sig_s[1], m.sig_s[2], m.hold, m.enters, m.ior].map(f64::to_bits),
            hb: self.hb.to_bits(),
        };
        if self.calibrate && self.calibrated.as_ref() != Some(&key) {
            let clear = Medium { sig_a: [0.0; 3], mottle: 0.0, ..*m };
            // (the furnace has no lamps: its light is white from everywhere)
            let lamps = std::mem::take(&mut self.lamps);
            self.run(vol, shape_no, furnace(), &clear, to_shape, res, chan, None, false);
            self.lamps = lamps;
            let l = self.levels.iter().find(|l| l.res == res).expect("just worked out");
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scatter: calibration") });
            enc.copy_texture_to_texture(l.out.texture().as_image_copy(), l.calib.texture().as_image_copy(), wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res });
            self.gpu.queue.submit([enc.finish()]);
            self.calibrated = Some(key);
        }
        let correct = self.calibrate && self.calibrated.is_some();
        self.run(vol, shape_no, light, m, to_shape, res, chan, cast, correct);
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&mut self, vol: &Volume, shape_no: u64, light: &Lighting, m: &Medium, to_shape: &[f64; 9], res: u32, chan: usize, cast: Option<&Casting>, correct: bool) {
        let sizes: Vec<u32> = sizes(res);
        // (volumes of sizes no longer wanted are let go: the largest hold a few hundred megabytes)
        self.levels.retain(|l| sizes.contains(&l.res));
        if !self.levels.iter().any(|l| l.res == res) { self.calibrated = None; }
        let ids: Vec<usize> = sizes.iter().map(|&r| self.level(r)).collect();
        let gpu = self.gpu.clone();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scatter") });
        // what the beams march through: the shape's volume, or a small copy of a large one (made when
        // the volume in hand is another than the copy was made from)
        let mut fresh_copy = None;
        if vol.res <= SHAPE_COPY || self.p_shape.is_none() {
            self.copy = None;
        } else if self.copy.as_ref().is_none_or(|(view, no, ch, _)| *view != vol.view || *no != shape_no || *ch != chan) {
            let (pipe, layout) = self.p_shape.as_ref().expect("looked");
            let small = match self.copy.take() {
                Some((_, _, _, v)) => v,
                None => Volume::new(&gpu, SHAPE_COPY, true),
            };
            // (the copy reads which channel to take from the smallest level's numbers, written below
            // before anything is run)
            let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.levels[ids[0]].u.as_entire_binding() }, entry(1, &vol.view), wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) }, entry(12, &small.view)],
            });
            fresh_copy = Some((pipe, g, SHAPE_COPY.div_ceil(4)));
            self.copy = Some((vol.view.clone(), shape_no, chan, small));
        }
        // the channel the beams march through: the copy holds the body's distance alone
        let pick = [if self.copy.is_some() { one_hot(0) } else { one_hot(chan) }, one_hot(chan)];
        let shading = cast.map_or(Shading::default(), |c| c.numbers);
        for (n, &i) in ids.iter().enumerate() {
            // (only the finest volume is calibrated: the smaller ones serve as its start, and as the body's
            // own light, in the furnace as in the light worked out)
            gpu.queue.write_buffer(&self.levels[i].u, 0, bytemuck::bytes_of(&Self::numbers(sizes[n], light, m, to_shape, pick, shading, correct && n + 1 == sizes.len(), self.hb, &self.lamps)));
        }
        let cast_views = [cast.map_or(&self.blank, |c| &c.bodies).clone(), cast.and_then(|c| c.light.as_ref()).unwrap_or(&self.blank).clone()];
        if let Some((pipe, g, n)) = &fresh_copy {
            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(pipe);
            c.set_bind_group(0, g, &[]);
            c.dispatch_workgroups(*n, *n, *n);
        }
        let shape = self.copy.as_ref().map_or(&vol.view, |(_, _, _, small)| &small.view).clone();
        let mut below: Option<usize> = None;
        for (n, &i) in ids.iter().enumerate() {
            let r = sizes[n];
            // (the light volume of the size below: what the body's own light is read from)
            let lower = below.map_or(&self.blank, |b| &self.levels[b].out).clone();
            let bound = [shape.clone(), lower, cast_views[0].clone(), cast_views[1].clone()];
            if self.levels[i].g_inject.as_ref().is_none_or(|(v, _)| *v != bound) {
                let l = &self.levels[i];
                let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.l_inject,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: l.u.as_entire_binding() }, entry(1, &bound[0]), wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) },
                        entry(3, &l.src), entry(13, &bound[1]), entry(14, &bound[2]), entry(15, &bound[3]),
                    ],
                });
                self.levels[i].g_inject = Some((bound, g));
            }
            // where this level starts from: the level below, read with interpolation
            let g_refine = below.map(|b| {
                let (l, lo) = (&self.levels[i], &self.levels[b]);
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.l_refine,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: l.u.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) }, entry(4, &l.src), entry(5, &lo.phi[lo.cur]), entry(6, &l.phi[0])],
                })
            });
            // (the small volumes cost next to nothing and settle the large features of the light, which
            // the fine ones would take thousands of steps over)
            let (before, after) = (self.steps[(ids.len() - 1 - n).min(4)], self.steps_after);
            let l = &mut self.levels[i];
            let cells = r.div_ceil(4);
            // The storage accesses of compute dispatches are ordered inside a pass.
            // Keep every relaxation step and reduction in the original order.
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("scatter: level"), timestamp_writes: None });
            let run = |pass: &mut wgpu::ComputePass, pipe: &wgpu::ComputePipeline, group: &wgpu::BindGroup, n: u32| {
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(n, n, n);
            };
            let (inject, groups) = if r <= 40 {
                (&self.p_inject_coarse, r)
            } else {
                (&self.p_inject, cells)
            };
            run(&mut pass, inject, &l.g_inject.as_ref().unwrap().1, groups);
            match &g_refine {
                Some(g) => run(&mut pass, &self.p_refine, g, cells),
                None => run(&mut pass, &self.p_begin, &l.g_begin, cells),
            }
            let compact = self.sparse.as_ref().zip(l.sparse.as_ref()).filter(|_| before >= 3);
            if let Some((passes, grid)) = compact { passes.classify(&gpu, &mut pass, grid, cells); }
            let relax = |pass: &mut wgpu::ComputePass, current: &mut usize, steps: u32| {
                for k in 0..steps {
                    let colour = (k % 2) as usize;
                    if let Some((passes, grid)) = compact {
                        passes.relax(pass, grid, colour, *current);
                    } else {
                        run(pass, &self.p_relax[colour], &l.g_diffuse[*current], cells);
                    }
                    *current = 1 - *current;
                }
            };
            l.cur = 0;
            relax(&mut pass, &mut l.cur, before);
            // the body brought into balance as a whole
            if let Some((passes,grid))=compact {
                passes.sum(&mut pass,grid,l.cur,l.sums[0]);
            } else {
                run(&mut pass, &self.p_sum, &l.g_sum[l.cur], l.sums[0]);
            }
            run(&mut pass, &self.p_fold, &l.g_fold[0], l.sums[1]);
            run(&mut pass, &self.p_fold, &l.g_fold[1], l.sums[2]);
            run(&mut pass, &self.p_level, &l.g_level[l.cur], cells);
            l.cur = 1 - l.cur;
            relax(&mut pass, &mut l.cur, after);
            // (every size is finished: the next one up reads it for the body's own light)
            run(&mut pass, &self.p_finish, &l.g_finish[l.cur], cells);
            below = Some(i);
        }
        gpu.queue.submit([enc.finish()]);
        self.res = res;
    }

    /// The light volume as it stands: cells a side, and every cell's red, green, blue and distance to the
    /// surface, slice by slice. For checks.
    pub fn read(&self) -> R<(u32, Vec<f32>)> {
        let l = self.levels.iter().find(|l| l.res == self.res).ok_or("no light volume has been worked out")?;
        let mut out = Vec::with_capacity((l.res as usize).pow(3) * 4);
        for z in 0..l.res {
            let bytes = self.gpu.read(l.out.texture(), 0, 0, z, l.res, l.res, 8)?;
            out.extend(bytemuck::cast_slice::<u8, half::f16>(&bytes).iter().map(|v| v.to_f32()));
        }
        Ok((l.res, out))
    }
}

/// A light volume with nothing in it: what the tracer is given to read until there is a real one.
pub fn blank(gpu: &Gpu) -> wgpu::TextureView {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("no light volume"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: OUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chromatic_sss_calibrates_each_channel_and_updates_when_spread_changes() {
        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        let vres = 64u32;
        let vol = Volume::new(&gpu, vres, gpu.storage16);
        let mut sdf = Vec::with_capacity((vres * vres * vres) as usize);
        for iz in 0..vres {
            let z = ((iz as f64 + 0.5) / vres as f64 * 2.0 - 1.0) * HB;
            for iy in 0..vres {
                let y = ((iy as f64 + 0.5) / vres as f64 * 2.0 - 1.0) * HB;
                for ix in 0..vres {
                    let x = ((ix as f64 + 0.5) / vres as f64 * 2.0 - 1.0) * HB;
                    // Thin slab along z so coarse-grid diffusion leakage differs strongly across chromatic sig_s.
                    let d = (x.abs() - 0.85).max(y.abs() - 0.85).max(z.abs() - 0.08);
                    sdf.push(half::f16::from_f32(d as f32));
                }
            }
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &vol.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&sdf),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(vres * 2), rows_per_image: Some(vres) },
            wgpu::Extent3d { width: vres, height: vres, depth_or_array_layers: vres },
        );

        let ior = 1.5;
        let enters = 1.0 - gf_core::scatter::diffuse_reflectance(1.0, ior);
        let back = 1.0 - enters / (ior * ior);
        let hold = (1.0 + back) / (1.0 - back);
        let chrom = Medium { sig_s: [5.0, 22.0, 80.0], sig_a: [0.0; 3], hold, enters, ior, mottle: 0.0 };
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // res = 24 climbs [6, 12, 24], also exercising COARSEST = 6 in cs_level.
        let res = 24u32;
        assert_eq!(sizes(res), vec![6, 12, 24]);

        let mut sc = Scatter::new(&gpu);
        let centre_rgb = |sc: &Scatter| -> [f32; 3] {
            let (r, px) = sc.read().unwrap();
            let c = (r / 2) as usize;
            let o = ((c * r as usize + c) * r as usize + c) * 4;
            [px[o], px[o + 1], px[o + 2]]
        };

        sc.calibrate = false;
        sc.solve(&vol, 1, furnace(), &chrom, &id, res, 0, None);
        let raw = centre_rgb(&sc);
        assert!((raw[2] - raw[0]).abs() > 0.5, "expected thin-slab chromatic spread before calibration: {raw:?}");

        sc.calibrate = true;
        sc.solve(&vol, 1, furnace(), &chrom, &id, res, 0, None);
        let cal = centre_rgb(&sc);
        let want = furnace_level(ior) as f32;
        for (ch, &v) in cal.iter().enumerate() {
            assert!((v - want).abs() / want < 0.02, "channel {ch} furnace calibration off: got {v}, want {want} ({cal:?})");
        }

        // Keep sig_s[0] unchanged and alter sig_s[1..3]: calibration cache must invalidate and recalibrate G/B.
        let achrom = Medium { sig_s: [5.0, 5.0, 5.0], ..chrom };
        sc.solve(&vol, 1, furnace(), &achrom, &id, res, 0, None);
        let eq = centre_rgb(&sc);
        assert!((eq[1] - eq[0]).abs() < 0.02 && (eq[2] - eq[0]).abs() < 0.02, "stale calibration when sig_s[1..3] changed: {eq:?}");
    }
}
