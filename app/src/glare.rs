//! The lens's glare: what replaces the bloom pyramid (`bloomKind` 0; 1 keeps the old pyramid).
//!
//! The finished picture is the picture convolved with the lens's point spread function, as a mix and
//! not an addition: the sharp picture keeps 1 - E of its light and the spread carries E. So only light
//! far above its surroundings spreads visibly; a white surface at 1.0 gets back from its neighbours
//! nearly all it gives away, and does not glow.
//!
//! The spread comes from the camera (shaders/glare.wgsl draws it):
//!   - diffraction at the lens opening, a polygon of `blades` sides turned by `bokehRot` (round when
//!     fewer than 3): a spike along each side's normal (so as many spikes as sides when the count is
//!     even, twice as many when odd) falling as 1 / r^2, plus a faint round halo falling as 1 / r^3.
//!     Its share of the light beyond a distance r is about 0.2 x lambda x N / r (the Airy pattern's
//!     far field), so it grows with the f-number and with the wavelength: red is drawn larger than
//!     blue, which fringes the spikes' ends;
//!   - a long power-law veil (scatter in the glass and on its surfaces), the same for every colour;
//!   - and a tight core: everything within one grid texel, which is the sharp picture itself.
//! `bloom` is the strength: it scales the diffraction (from what the lens would really give, which
//! only the sun would show) and the veil's share together.
//!
//! How: an FFT convolution in compute shaders on a fixed grid of 2048 x 1024, onto which the picture
//! is drawn at most 720 texels high (a third of 4K); the rest of the grid is the room the spread needs
//! so that nothing wraps. The kernel's spectrum is made by the same transform, only when the lens or
//! the size changes.

use bytemuck::{Pod, Zeroable};
use gf_core::settings::Settings;

use crate::gpu::*;

const WGSL: &str = include_str!("shaders/glare.wgsl");
const NX: u32 = 2048;
const NY: u32 = 1024;
const BINS: u32 = NX / 2 + 1;
/// the picture's height on the grid at most (720: the grid's spare 304 rows are 0.42 of it)
const MAX_H: u32 = 720;

/// Each colour's wavelength over green's: red ~620 nm, green ~550, blue ~460.
const SCALE: [f64; 3] = [620.0 / 550.0, 1.0, 460.0 / 550.0];
/// How much the diffraction is made stronger than a real lens's at `bloom` 1 (a real lens's spikes need
/// the sun to be seen; the lamps here are a few dozen times white).
const GAIN_DIFFRACTION: f64 = 64.0;
/// The veil's share of the light at `bloom` 1.
const GAIN_VEIL: f64 = 0.075;
/// The diffraction's core: what falls within this many grid texels stays in the sharp picture.
const CORE: f64 = 3.0;
/// The most light the spread may carry, all of it together.
const MAX_SHARE: f64 = 0.6;

/// The lens as the glare needs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlareLens {
    /// 0 = the lens's glare (this file), 1 = the old bloom pyramid
    pub kind: u32,
    pub strength: f32,
    /// the f-number (8 when the lens has no opening: everything sharp)
    pub fstop: f32,
    /// sides of the opening (round below 3) and its turn, radians
    pub blades: u32,
    pub rot: f32,
    /// the sensor's height, mm
    pub sensor: f32,
}

impl GlareLens {
    pub fn of(p: &Settings) -> Self {
        let n = gf_core::camera::get(p, "fstop").unwrap_or(0.0);
        let n = if n > 0.0 && n.is_finite() { n.clamp(1.0, 64.0) } else { 8.0 };
        let sides = (p.blades + 0.5).floor().max(0.0) as u32;
        GlareLens {
            kind: if p.bloom_kind >= 0.5 { 1 } else { 0 },
            strength: p.bloom.max(0.0) as f32,
            fstop: n as f32,
            blades: if sides >= 3 { sides.min(16) } else { 0 },
            rot: (p.bokeh_rot * std::f64::consts::PI / 180.0) as f32,
            sensor: if p.sensor > 0.0 { p.sensor as f32 } else { 24.0 },
        }
    }
    pub fn pyramid(&self) -> bool { self.kind == 1 }
}

/// The picture's size on the grid and the spread's reach, for a picture `tw` x `th`.
fn grid(tw: u32, th: u32) -> (u32, u32, f64) {
    let (tw, th) = (tw.max(2) as f64, th.max(2) as f64);
    let mut h = th.min(MAX_H as f64);
    let mut w = (tw * h / th).round();
    if w > 1400.0 {
        w = 1400.0;
        h = (th * w / tw).round().max(2.0);
    }
    let r = (NY as f64 - h).min(NX as f64 - w).min(0.45 * h).min(500.0).max(4.0);
    (w.max(2.0) as u32, h as u32, r)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
struct GlareU {
    img: [f32; 4],
    lens: [f32; 4],
    a_s: [f32; 4],
    a_h: [f32; 4],
    scale: [f32; 4],
    energy: [f32; 4],
}

/// The window the shader puts on the spread (1 out to 0.55 R, 0 at R).
fn win(r: f64, big_r: f64) -> f64 {
    let s = |a: f64, b: f64, x: f64| { let t = ((x - a) / (b - a)).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
    1.0 - s(0.55 * big_r, big_r, r)
}
fn inner(r: f64, rc: f64) -> f64 {
    let t = (r - rc + 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
/// ∫ f(r) dr from 0 to `to` (midpoint rule, fine enough: the integrands are smooth beyond a texel)
fn integrate(to: f64, f: impl Fn(f64) -> f64) -> f64 {
    let n = 4000;
    let dr = to / n as f64;
    (0..n).map(|i| f((i as f64 + 0.5) * dr)).sum::<f64>() * dr
}

/// The shader's numbers for this lens and size: the uniform, and the share of light the spread carries
/// for each colour.
fn uniform(lens: &GlareLens, tw: u32, th: u32) -> GlareU {
    let (w, h, big_r) = grid(tw, th);
    let hf = h as f64;
    // lambda N in grid texels, for green
    let psi = 0.55e-3 * lens.fstop as f64 / lens.sensor as f64 * hf;
    let k = lens.strength as f64;
    let env = |name: &str, v: f64| std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(v);
    let (gain_d, gain_v, rc) = (env("GLARE_GD", GAIN_DIFFRACTION), env("GLARE_GV", GAIN_VEIL), env("GLARE_RC", CORE));
    let knee = 0.005 * hf;
    let i_veil = integrate(big_r, |r| 2.0 * std::f64::consts::PI * r * (knee / (r + knee)).powf(2.5) * inner(r, 1.0) * win(r, big_r));
    let ev = gain_v * k;
    let a_v = if i_veil > 0.0 { ev / i_veil } else { 0.0 };
    let spikes = if lens.blades >= 3 { 0.75 } else { 0.0 };
    let mut a_s = [0.0f64; 3];
    let mut a_h = [0.0f64; 3];
    let mut e = [0.0f64; 3];
    for c in 0..3 {
        let s = SCALE[c];
        // the diffraction's strength: its share beyond r is d / r (unwindowed)
        let d = 0.2 * psi * s * gain_d * k;
        a_s[c] = spikes * d;
        a_h[c] = (1.0 - spikes) * d;
        let i_spike = integrate(big_r / s, |a| if a > rc - 0.5 { inner(a, rc) / (a * a) * win(a * s, big_r) } else { 0.0 });
        let i_halo = integrate(big_r / s, |r| r * inner(r, rc) / r.max(rc).powi(3) * win(r * s, big_r));
        e[c] = a_s[c] * i_spike + a_h[c] * i_halo + ev;
    }
    let most = e.iter().cloned().fold(0.0, f64::max);
    let cut = if most > MAX_SHARE { MAX_SHARE / most } else { 1.0 };
    let (half_lines, turn) = match lens.blades {
        0 => (0.0, 0.0),
        // each side's normal, and for an odd count its opposite too: evenly spaced half-lines
        n if n % 2 == 0 => (n as f32, lens.rot + std::f32::consts::PI / n as f32),
        n => (2.0 * n as f32, lens.rot),
    };
    let f = |v: [f64; 3], wv: f64| [(v[0] * cut) as f32, (v[1] * cut) as f32, (v[2] * cut) as f32, wv as f32];
    GlareU {
        img: [w as f32, h as f32, tw as f32, th as f32],
        lens: [half_lines, turn, 0.6, big_r as f32],
        a_s: f(a_s, knee),
        a_h: f(a_h, a_v * cut),
        // (GLASSFORMS_GLARE_POINT=v: the picture replaced by one point of brightness v, to see the spread)
        scale: [SCALE[0] as f32, SCALE[1] as f32, SCALE[2] as f32, std::env::var("GLASSFORMS_GLARE_POINT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)],
        energy: f(e, rc),
    }
}

/// The share of its light the sharp picture keeps, per colour, at this size (1 - E): what the final
/// pass multiplies the picture by before adding the glare.
pub fn keep(lens: &GlareLens, tw: u32, th: u32) -> [f32; 3] {
    let e = uniform(lens, tw, th).energy;
    [1.0 - e[0], 1.0 - e[1], 1.0 - e[2]]
}

struct Out {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    groups: [wgpu::BindGroup; 3],
    tw: u32,
    th: u32,
}

pub struct Glare {
    gpu: Gpu,
    layout: wgpu::BindGroupLayout,
    p_rows: wgpu::ComputePipeline,
    p_rows_kernel: wgpu::ComputePipeline,
    p_cols_kernel: wgpu::ComputePipeline,
    p_cols: wgpu::ComputePipeline,
    p_rows_inverse: wgpu::ComputePipeline,
    u: wgpu::Buffer,
    spec: wgpu::Buffer,
    kern: wgpu::Buffer,
    smp: wgpu::Sampler,
    out: Option<Out>,
    /// what the kernel's spectrum was last made for
    built: Option<GlareU>,
    /// the last lens and size worked out, and what came of it (the integrals are not free)
    memo: std::cell::RefCell<Option<(GlareLens, u32, u32, GlareU)>>,
}

impl Glare {
    pub fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let rw = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let layout = gpu.bind_layout("glare", &[
            uniform_entry(0, c, false),
            rw(1),
            rw(2),
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: c, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
        ]);
        let module = gpu.shader("glare", WGSL, false);
        let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("glare"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipe = |entry: &str| gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
        let storage = |label, size: u64| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let n = 3 * NY as u64 * BINS as u64;
        Glare {
            gpu: gpu.clone(),
            p_rows: pipe("cs_rows"),
            p_rows_kernel: pipe("cs_rows_kernel"),
            p_cols_kernel: pipe("cs_cols_kernel"),
            p_cols: pipe("cs_cols"),
            p_rows_inverse: pipe("cs_rows_inverse"),
            layout,
            u: gpu.uniform_buffer("glare", std::mem::size_of::<GlareU>() as u64),
            spec: storage("glare: spectrum", n * 8),
            kern: storage("glare: the kernel's spectrum", n * 4),
            smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            out: None,
            built: None,
            memo: std::cell::RefCell::new(None),
        }
    }

    /// Size the glare for a picture `tw` x `th` read from any of `sources` (the two pictures so far and
    /// the focused one, as the bloom reads them).
    pub fn resize(&mut self, tw: u32, th: u32, sources: [&wgpu::TextureView; 3]) {
        let (w, h, _) = grid(tw, th);
        let tex = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glare"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let groups = sources.map(|src| {
            self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("glare"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.spec.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.kern.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.smp) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&view) },
                ],
            })
        });
        self.out = Some(Out { tex, view, groups, tw, th });
        self.built = None;
    }

    /// The share of its light the sharp picture keeps, per colour, at the size the glare is made for.
    pub fn keep_for(&self, lens: &GlareLens) -> [f32; 3] {
        self.out.as_ref().map_or([1.0; 3], |o| { let e = self.uniform(lens, o.tw, o.th).energy; [1.0 - e[0], 1.0 - e[1], 1.0 - e[2]] })
    }

    fn uniform(&self, lens: &GlareLens, tw: u32, th: u32) -> GlareU {
        let mut m = self.memo.borrow_mut();
        if let Some((l, w, h, u)) = *m { if l == *lens && w == tw && h == th { return u; } }
        let u = uniform(lens, tw, th);
        *m = Some((*lens, tw, th, u));
        u
    }

    /// The glare picture (what the final pass adds to the sharp picture), once sized.
    pub fn view(&self) -> Option<&wgpu::TextureView> { self.out.as_ref().map(|o| &o.view) }
    pub fn texture(&self) -> Option<&wgpu::Texture> { self.out.as_ref().map(|o| &o.tex) }

    /// Convolve picture `src` (0, 1: the pictures so far; 2: the focused one) with the lens's spread,
    /// making the kernel's spectrum first when the lens or the size has changed. Returns the share of
    /// light the sharp picture keeps, per colour.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, src: usize, lens: &GlareLens) -> [f32; 3] {
        let Some(o) = self.out.as_ref() else { return [1.0; 3] };
        let u = self.uniform(lens, o.tw, o.th);
        let (h, group) = (u.img[1] as u32, &o.groups[src]);
        let fresh = self.built != Some(u);
        if fresh {
            self.gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&u));
            self.built = Some(u);
            if std::env::var_os("GLARE_PRINT").is_some() { eprintln!("glare {u:?}"); }
        }
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("glare"), timestamp_writes: None });
        c.set_bind_group(0, group, &[]);
        if fresh {
            c.set_pipeline(&self.p_rows_kernel);
            c.dispatch_workgroups(NY, 1, 1);
            c.set_pipeline(&self.p_cols_kernel);
            c.dispatch_workgroups(BINS, 3, 1);
        }
        // (GLARE_REPEAT=n: the convolution done n times, to time it)
        let repeat = std::env::var("GLARE_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1u32).max(1);
        for _ in 0..repeat {
            c.set_pipeline(&self.p_rows);
            c.dispatch_workgroups(h, 1, 1);
            c.set_pipeline(&self.p_cols);
            c.dispatch_workgroups(BINS, 3, 1);
            c.set_pipeline(&self.p_rows_inverse);
            c.dispatch_workgroups(h, 1, 1);
        }
        [1.0 - u.energy[0], 1.0 - u.energy[1], 1.0 - u.energy[2]]
    }
}
