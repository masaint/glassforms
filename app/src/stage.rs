//! The studio on the GPU: what the tracer is told about the backdrop and the floor, the text that draws
//! them, and how high the floor stands (measured from the shape itself).
//!
//! What is drawn where, and why (core/src/stage.rs has the maths):
//!
//!   glass       sees the studio through the lights' map, into which backdrop and floor are painted
//!               before anything is traced. The tracer's marching and shading are not touched, and
//!               every material sees the studio through the lookups it already makes.
//!   camera      rays that miss the shape draw the backdrop and the floor themselves (`stage.wgsl`, put
//!               into the tracer where it says what lies behind the shape): the backdrop exactly, not
//!               through the texels of a small map; the floor by meeting its plane, so that it has a
//!               place and a distance.
//!   depth       a floor pixel stores how far away the floor is, like glass. Focus therefore blurs the
//!               floor by its real distance, a click on it focuses there, and the antialiasing can
//!               tell floor from glass (the stored distance is the plane's along that pixel's ray) and
//!               follow it through the room instead of round with the shape.
//!   height      with `floorAuto` the floor touches the lowest point of the shape. That point is found on
//!               the GPU from the shape's distance volume (`Lowest`), once for each pose: a shape turning
//!               about the room's vertical keeps its lowest point, so the floor stands still under it.

use bytemuck::{Pod, Zeroable};
use gf_core::math::{f32s, DEG};
use gf_core::stage::{Stage, FADE_FROM};
use gf_core::state::State;

use crate::bake::Volume;
use crate::engine::Engine;
use crate::gpu::*;

const STAGE_WGSL: &str = include_str!("shaders/stage.wgsl");
const LOWEST_WGSL: &str = include_str!("shaders/stage_lowest.wgsl");

/// Where the studio's numbers are bound in the tracer's group 0.
pub const BINDING: u32 = 40;

/// What of the studio a tracer is built to draw where a ray from the camera misses the shape.
/// Nothing (the default) leaves the tracer's text exactly as it is without a studio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Seen {
    /// the studio's backdrop is shown behind the shape
    pub backdrop: bool,
    /// there is a floor
    pub floor: bool,
    /// the shape is mirrored in the floor
    pub mirror: bool,
    /// shadows and caustics fall on the floor (the tracer then has a `castFloor`)
    pub cast: bool,
}

impl Seen {
    pub fn any(&self) -> bool {
        self.backdrop || self.floor
    }
}

/// The studio's numbers as the tracer reads them (see stage.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct StageU {
    pub room: [[f32; 4]; 3],
    pub top: [f32; 4],
    pub mid: [f32; 4],
    pub low: [f32; 4],
    pub glow: [f32; 4],
    pub glowd: [f32; 4],
    pub floor: [f32; 4],
    pub k: [f32; 4],
    pub tint: [f32; 4],
}

impl StageU {
    /// `height`: where the floor stands in the room; `light`: its colour where nothing shadows it.
    pub fn of(state: &State, stage: &Stage, height: f64, light: &gf_core::stage::FloorLight) -> StageU {
        let m = f32s(&state.room_m());
        let mut u: StageU = Zeroable::zeroed();
        u.room = [[m[0], m[1], m[2], 0.0], [m[3], m[4], m[5], 0.0], [m[6], m[7], m[8], 0.0]];
        u.top[3] = 1.0;
        u.glow[3] = 1.0;
        (u.k[2], u.k[3]) = (-2.0, -3.0);
        (u.glowd[3], u.floor[3]) = (2.0, 3.0);
        if let Some(b) = &stage.backdrop {
            let v = |c: [f64; 3], w: f64| [c[0] as f32, c[1] as f32, c[2] as f32, w as f32];
            u.top = v(b.top, 1.0 / (b.spread * DEG));
            u.mid = v(b.mid, 0.0);
            u.low = v(b.low, 0.0);
            u.glow = v(b.glow, b.glow_k());
            u.glowd = v(b.glow_dir, 0.0);
            let (whole, gone) = b.wrap_cos();
            (u.k[2], u.k[3]) = (whole as f32, gone as f32);
            let (under, over) = b.high_sin();
            (u.glowd[3], u.floor[3]) = (under as f32, over as f32);
        }
        u.mid[3] = height as f32;
        if let Some(f) = &stage.floor {
            u.low[3] = f.fade as f32;
            (u.floor[0], u.floor[1], u.floor[2], u.room[0][3]) = (light.base[0] as f32, light.base[1] as f32, light.base[2] as f32, f.gloss as f32);
            u.tint = [light.tint[0] as f32, light.tint[1] as f32, light.tint[2] as f32, 0.0];
            // an image in a perfectly glossy floor does not fade; in a duller one it is gone a shape's height up
            (u.k[0], u.k[1]) = ((FADE_FROM * f.fade) as f32, (2.2 * (1.0 - f.gloss)) as f32);
        }
        u
    }
}

/// What the tracer shows where a ray from the camera misses the shape: an expression in the terms of
/// `sampleAt` in trace.wgsl. `picture`: an environment picture is shown there.
pub fn background(seen: Seen, picture: bool) -> &'static str {
    match (seen.floor, seen.mirror, seen.backdrop, picture) {
        (true, true, ..) => "stageFloor(ro, rd, &t, &g, &depth, xi, W, fc)",
        (true, false, ..) => "stageFloor(ro, rd, &t, &g, &depth)",
        (false, _, true, _) => "stageBackdrop(rd)",
        (false, _, false, true) => "backdrop(rd)",
        (false, _, false, false) => "vec3f(0.0)",
    }
}

/// The studio's part of the tracer's declarations: nothing unless a backdrop is shown or there is a floor.
/// `shade`: how the tracer shades the shape (the same text it uses for a ray from the camera).
pub fn tracer_decl(seen: Seen, picture: bool, shade: &str) -> String {
    if !seen.any() { return String::new(); }
    let mut out = String::with_capacity(STAGE_WGSL.len());
    for line in STAGE_WGSL.trim_end().lines() {
        let kept = if let Some(rest) = line.strip_prefix("//#floor ") {
            seen.floor.then_some(rest)
        } else if let Some(rest) = line.strip_prefix("//#mirror ") {
            (seen.floor && seen.mirror).then_some(rest)
        } else {
            Some(line)
        };
        if let Some(l) = kept {
            out.push_str(l);
            out.push('\n');
        }
    }
    out.replace("{{STAGE_BINDING}}", &BINDING.to_string())
        .replace("{{MIRROR_PARAMS}}", if seen.mirror { ", xi: f32, W: i32, fc: vec2f" } else { "" })
        .replace("{{BEHIND}}", if seen.backdrop { "backdropAt(d)" } else if picture { "backdrop(rd)" } else { "vec3f(0.0)" })
        .replace("{{CAST_FLOOR}}", if seen.cast { "castFloor(p)" } else { "vec3f(1.0)" })
        .replace("{{MIRROR}}", if seen.mirror { "stageMirror(ro + rd * tf, r, xi, W, fc)" } else { "env(r)" })
        .replace("{{SHADE}}", shade)
}

// ---- the lowest point of a shape ---------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LowU {
    /// the direction looked along, in the shape's own frame; samples along a side
    down: [f32; 4],
    /// distances counted: below this, above this; the width of the volume; unused
    k: [f32; 4],
}

/// The most samples along a side of the volume that are looked at; and while the shape keeps tipping.
const LOW_N: u32 = 256;
const LOW_N_MOVING: u32 = 128;
/// Columns along a side of a block in the second pass (stage_lowest.wgsl: BLOCK).
const LOW_BLOCK: u32 = 16;

/// Finds how far a shape reaches in one direction: the largest of p . d over its surface (what a flat
/// plate pushed against it from that side would stop at).
///
/// The volume is looked at on a grid. A sample near the surface with distance s at place p says that
/// the surface comes to p . d - s at least: a ball of radius |s| about p is all glass (inside) or just
/// touches the glass (outside). The largest of that over the samples near the surface is the reach, and
/// it is far finer than the grid: a sample's own distance makes up for its not lying on the surface,
/// and what is left is the grid's spacing squared over the surface's curvature. Three passes: every
/// column of the grid, the largest of each block of columns, the largest of the blocks; one number is
/// read back.
pub struct Lowest {
    gpu: Gpu,
    l_rows: wgpu::BindGroupLayout,
    l_from: wgpu::BindGroupLayout,
    p_rows: wgpu::RenderPipeline,
    p_blocks: wgpu::RenderPipeline,
    p_all: wgpu::RenderPipeline,
    rows_view: wgpu::TextureView,
    blocks_view: wgpu::TextureView,
    out: wgpu::Texture,
    out_view: wgpu::TextureView,
    smp: wgpu::Sampler,
}

impl Lowest {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let l_rows = gpu.bind_layout("lowest: columns", &[uniform_entry(0, frag, false), texture_entry(1, wgpu::TextureViewDimension::D3, true), sampler_entry(2)]);
        let l_from = gpu.bind_layout("lowest: largest", &[uniform_entry(0, frag, false), texture_entry(3, wgpu::TextureViewDimension::D2, false)]);
        let module = gpu.shader("lowest", LOWEST_WGSL, false);
        let fmt = wgpu::TextureFormat::R32Float;
        let pipe = |label, fs, layout: &wgpu::BindGroupLayout| gpu.pipeline(&PipeDesc { label, module: &module, vs: "vs", fs, layouts: &[layout], format: fmt, blend: None, strip: false, buffers: &[] });
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let out = gpu.target("lowest", 1, 1, fmt, &[]);
        Self {
            gpu: gpu.clone(),
            p_rows: pipe("lowest: columns", "fs_rows", &l_rows),
            p_blocks: pipe("lowest: blocks", "fs_blocks", &l_from),
            p_all: pipe("lowest: all", "fs_all", &l_from),
            l_rows, l_from,
            rows_view: view(&gpu.target("lowest: columns", LOW_N, LOW_N, fmt, &[])),
            blocks_view: view(&gpu.target("lowest: blocks", LOW_N / LOW_BLOCK, LOW_N / LOW_BLOCK, fmt, &[])),
            out_view: view(&out),
            out,
            smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
        }
    }

    /// How far the shape in `vol` reaches along `d` (a unit vector in the shape's own frame), from the
    /// middle of the volume. None if the volume holds no shape. Waits for the GPU.
    pub fn reach(&self, vol: &Volume, d: [f64; 3], hb: f64) -> Option<f64> {
        let asked = self.ask(vol, d, false, hb);
        self.gpu.wait();
        asked.answer(&self.gpu).flatten()
    }

    /// The same, asked for now and answered later (see `Asked::answer`): the GPU is handed the work and
    /// nothing waits for it. `quick`: on a grid half as fine (an eighth of the work), for a shape that
    /// is tipping and will be measured again in a moment.
    /// `hb`: the volume spans [-hb, hb] (larger than ever when the layers are placed far out).
    pub fn ask(&self, vol: &Volume, d: [f64; 3], quick: bool, hb: f64) -> Asked {
        let gpu = &self.gpu;
        let n = vol.res.min(if quick { LOW_N_MOVING } else { LOW_N });
        let width = 2.0 * hb;
        let spacing = width / n as f64;
        // samples up to one and a half spacings outside the surface and two inside: there is always one
        // of them beside any point of the surface, and that near the surface every volume's distances
        // are true ones
        let u = LowU { down: [d[0] as f32, d[1] as f32, d[2] as f32, n as f32], k: [(1.5 * spacing) as f32, (-2.0 * spacing) as f32, width as f32, 0.0] };
        // (a block of numbers of its own for every asking: several may be on their way at once)
        let ub = gpu.uniform_buffer("lowest", std::mem::size_of::<LowU>() as u64);
        gpu.queue.write_buffer(&ub, 0, bytemuck::bytes_of(&u));
        let g_rows = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lowest: columns"),
            layout: &self.l_rows,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&vol.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.smp) },
            ],
        });
        let from = |view: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lowest: largest"),
                layout: &self.l_from,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(view) }],
            })
        };
        let (g_blocks, g_all) = (from(&self.rows_view), from(&self.blocks_view));
        let blocks = n.div_ceil(LOW_BLOCK);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lowest") });
        for (target, pipe, group, size) in [(&self.rows_view, &self.p_rows, &g_rows, n), (&self.blocks_view, &self.p_blocks, &g_blocks, blocks), (&self.out_view, &self.p_all, &g_all, 1)] {
            let mut rp = pass(&mut enc, target, None, true);
            rp.set_viewport(0.0, 0.0, size as f32, size as f32, 0.0, 1.0);
            rp.set_scissor_rect(0, 0, size, size);
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, group, &[]);
            rp.draw(0..3, 0..1);
        }
        // the one number, copied to where it can be read from
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lowest: read back"),
            size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT), rows_per_image: Some(1) } },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([enc.finish()]);
        let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
        let flag = state.clone();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(if r.is_ok() { 1 } else { 2 }, std::sync::atomic::Ordering::SeqCst));
        Asked { buf, state }
    }
}

/// A reach that has been asked for (`Lowest::ask`).
pub struct Asked {
    buf: wgpu::Buffer,
    /// 0 = the GPU has not got there yet, 1 = the number can be read, 2 = it could not be read back
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

impl Asked {
    /// The answer, if the GPU has got there: Some(None) = the volume holds no shape (or the number could
    /// not be read back). None = not yet; ask again later. Never waits.
    pub fn answer(&self, gpu: &Gpu) -> Option<Option<f64>> {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        match self.state.load(std::sync::atomic::Ordering::SeqCst) {
            0 => None,
            1 => {
                let v = match self.buf.slice(..).get_mapped_range() {
                    Ok(view) => f32::from_le_bytes([view[0], view[1], view[2], view[3]]) as f64,
                    Err(_) => return Some(None),
                };
                self.buf.unmap();
                Some((v > -1e8).then_some(v))
            }
            _ => Some(None),
        }
    }
}

// ---- the studio as the engine keeps it ---------------------------------------------------------------

/// What the engine keeps of the studio from frame to frame.
#[derive(Default)]
pub struct StageWork {
    lowest: Option<Lowest>,
    /// counts the times the shape was built
    shape_no: u64,
    /// the reach last measured: for which build of the shape, along which direction, and what it was
    measured: Option<(u64, [i64; 3], Option<f64>)>,
    /// that measurement was a quick one (a coarser grid): to be done again with care once the shape rests
    rough: bool,
    /// what the reach was wanted for at the frame before
    wanted: Option<(u64, [i64; 3])>,
    /// reaches asked for and not yet answered, oldest first: what each was asked for, whether quickly
    asked: Vec<((u64, [i64; 3]), bool, Asked)>,
    /// wait for every reach to be measured before drawing on (stills, videos, timings): the floor then
    /// stands exactly under the shape in every frame. A live view does not wait: while the shape tips,
    /// the floor follows it a frame or two behind, and stands exactly once the shape is at rest.
    pub wait: bool,
    /// what the floor came to when the studio was last painted into the lights' map
    light: gf_core::stage::FloorLight,
    /// where the floor stands in the room (None: there is no floor)
    height: Option<f64>,
    /// what the tracer was last told
    told: Option<StageU>,
    /// how many times the reach has been measured, and how long the last one took, milliseconds
    pub measures: u32,
    pub measure_ms: f64,
}

impl StageWork {
    /// `wait`: every measurement of the floor's height is waited for (see the field).
    /// (GLASSFORMS_FLOOR_LIVE=1 has a still or a timing not wait either: for measuring what a live view pays.)
    pub fn new(wait: bool) -> Self {
        Self { wait: wait && std::env::var_os("GLASSFORMS_FLOOR_LIVE").is_none(), ..Default::default() }
    }

    /// The shape has been built again: its lowest point is not known any more.
    pub fn rebuilt(&mut self) {
        self.shape_no += 1;
    }

    /// What of the studio the tracer draws itself, as the settings stand.
    pub fn seen(state: &State) -> Seen {
        let s = Stage::of(&state.p);
        Seen { backdrop: s.show, floor: s.floor.is_some(), mirror: s.floor.as_ref().is_some_and(|f| f.mirror), cast: false }
    }

    /// What the painting of the lights' map depends on, of the studio (nothing when there is none).
    pub fn paint_key(state: &State) -> Vec<u64> {
        let s = Stage::of(&state.p);
        if s.off() { Vec::new() } else { s.paint_key(&state.lights_to_room()) }
    }

    /// Paint the studio into the lights' map (see `Stage::paint`); remembers the floor's own colour.
    pub fn paint(&mut self, state: &State, px: &mut [f32], w: usize, h: usize, extra: [f64; 3]) {
        self.light = Stage::of(&state.p).paint(px, w, h, &state.lights_to_room(), extra);
    }

    /// The same without remembering anything: for a copy of the map made for another purpose.
    pub fn paint_copy(state: &State, px: &mut [f32], w: usize, h: usize) {
        Stage::of(&state.p).paint(px, w, h, &state.lights_to_room(), [0.0; 3]);
    }

    /// Where the floor stands in the room, if there is one.
    pub fn height(&self) -> Option<f64> {
        self.height
    }
    /// The floor's height is still being measured for the shape as it now stands (a live view only):
    /// the picture will change when the answer arrives.
    pub fn pending(&self) -> bool {
        !self.asked.is_empty()
    }
    /// The floor's own colour where nothing shadows it.
    pub fn floor_rgb(&self) -> [f64; 3] {
        self.light.own
    }
}

/// The floor, for whoever works out the light that reaches it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageFloor {
    /// the room (y up, the floor level in it) -> what the camera sees
    pub room: gf_core::math::Mat3,
    /// the floor is the plane y = height of the room
    pub height: f64,
    /// what a white matte floor gives back of the backdrop's light alone, by colour (already over pi):
    /// light that no shadow of the lights takes away
    pub backdrop: [f64; 3],
    /// up to this distance from the room's middle the floor is wholly floor; by `fade` it has melted
    /// into the backdrop and nothing on it shows any more
    pub reach: f64,
    pub fade: f64,
}

/// The light a white matte floor gives back from an environment picture (given as its nine smooth
/// terms, in the lights' frame), `gain` times as bright: for `StageWork::paint`'s `extra`.
pub fn picture_on_floor(sh: &gf_core::env::Sh9, gain: f64, state: &State) -> [f64; 3] {
    if state.p.floor < 0.5 { return [0.0; 3]; }
    let up = gf_core::math::apply(&gf_core::math::transpose(&state.lights_to_room()), [0.0, 1.0, 0.0]);
    gf_core::env::sh_eval(&gf_core::env::sh_terms(sh, gf_core::env::SH_MATTE), up).map(|c| c.max(0.0) * gain)
}

impl Engine {
    /// Keep what the tracer is told of the studio in step with the settings, the shape and its pose:
    /// where the floor stands (measured when the shape or its pose has changed), the room, the colours.
    /// True if anything the picture shows changed.
    pub(crate) fn stage_upkeep(&mut self) -> bool {
        let stage = Stage::of(&self.state.p);
        if stage.off() {
            self.stage.height = None;
            if self.stage.told.take().is_some() {
                self.renderer.set_stage(&Zeroable::zeroed());
                return true;
            }
            return false;
        }
        let mut height = None;
        if stage.floor.is_some() {
            let p = &self.state.p;
            let mut h = p.floor_y;
            if p.floor_auto > 0.5 {
                // "down" as the shape has it: the room's down taken into the shape's own frame
                let r = &self.state.r;
                let down = [-r[3], -r[4], -r[5]];
                let key = (self.stage.shape_no, down.map(|c| (c * 1e9).round() as i64));
                // answers that have arrived: the newest of them is what is known now
                let gpu = self.gpu.clone();
                while let Some((k, quick, a)) = self.stage.asked.first() {
                    let Some(reach) = a.answer(&gpu) else { break };
                    self.stage.measured = Some((k.0, k.1, reach));
                    self.stage.rough = *quick;
                    self.stage.asked.remove(0);
                }
                // The shape is tipping if its "down" is not what it was a frame ago. While it tips, a live
                // view measures quickly and does not wait; once it rests, it is measured with care.
                let tipping = self.stage.wanted.is_some_and(|w| w != key);
                self.stage.wanted = Some(key);
                let known = self.stage.measured.as_ref().is_some_and(|m| (m.0, m.1) == key) && (tipping || !self.stage.rough);
                if !known && !self.stage.asked.iter().any(|(k, quick, _)| *k == key && (tipping || !*quick)) {
                    let t0 = std::time::Instant::now();
                    if self.stage.lowest.is_none() { self.stage.lowest = Some(Lowest::new(&self.gpu)); }
                    // (a frame of a video is waited for, like a still; and so is the first measurement of
                    // all, when there is nothing to go on meanwhile)
                    let wait = self.stage.wait || self.state.rendering || self.stage.measured.is_none();
                    let hb = self.state.scene_hb();
                    match (self.scene_volume(), &self.stage.lowest) {
                        (Some(vol), Some(low)) if wait => {
                            self.stage.measured = Some((key.0, key.1, low.reach(vol, down, hb)));
                            self.stage.rough = false;
                        }
                        // (no more than a few on their way at once: a shape that tips every frame asks every frame)
                        (Some(vol), Some(low)) => if self.stage.asked.len() < 4 { self.stage.asked.push((key, tipping, low.ask(vol, down, tipping, hb))); },
                        _ => self.stage.measured = Some((key.0, key.1, None)),
                    }
                    self.stage.measures += 1;
                    self.stage.measure_ms = t0.elapsed().as_secs_f64() * 1e3;
                }
                // (a volume with nothing in it: the floor stands where a shape that fills the volume would rest)
                h -= self.stage.measured.as_ref().and_then(|m| m.2).unwrap_or(gf_core::FIT);
            }
            height = Some(h);
        }
        self.stage.height = height;
        let u = StageU::of(&self.state, &stage, height.unwrap_or(0.0), &self.stage.light);
        if self.stage.told == Some(u) { return false; }
        self.renderer.set_stage(&u);
        self.stage.told = Some(u);
        true
    }

    /// The nearest the floor comes to the lens within the picture, measured along the line of sight
    /// (0 = there is no floor, or none of it is in the picture). The focus stage sizes itself by it.
    pub(crate) fn floor_nearest(&self) -> f64 {
        let Some(h) = self.stage.height() else { return 0.0 };
        let p = &self.state.p;
        let m = self.state.room_m();
        // the room's up as the camera sees it; the camera stands at (0, 0, dist)
        let n = [m[1], m[4], m[7]];
        let ((w, hh), _) = self.renderer.size();
        let aspect = w.max(1) as f64 / hh.max(1) as f64;
        let tf = (p.fov * DEG / 2.0).tan();
        // The floor shows in a part of the picture cut off by a straight line (the horizon), so if any
        // of it shows, a corner of the picture shows it, and the nearest of it is at a corner.
        let mut best = f64::MAX;
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            let dir = [(sx * aspect + p.shift[0]) * tf, (sy + p.shift[1]) * tf, -1.0];
            let t = (h - n[2] * p.dist) / (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2]);
            if t > 0.0 && t.is_finite() { best = best.min(t); }
        }
        if best == f64::MAX { 0.0 } else { best }
    }

    /// Move layer i straight up or down in the room until its lowest point touches the floor (measured
    /// from the layer's own volume, as the floor is from the scene's). False if there is no floor, or
    /// nothing to measure.
    pub fn layer_to_floor(&mut self, i: usize) -> bool {
        let Some(h) = self.stage.height() else { return false };
        if i >= self.state.layer_count() { return false; }
        if self.stage.lowest.is_none() { self.stage.lowest = Some(Lowest::new(&self.gpu)); }
        let pl = self.state.stack.layers[i].place;
        let size = if pl.size != 0.0 { pl.size } else { 1.0 };
        let r = self.state.r;
        let rt = gf_core::math::transpose(&r);
        let turn = gf_core::math::euler(pl.rot[0], pl.rot[1], pl.rot[2]);
        // the room's down, in the layer's own frame
        let down = gf_core::math::apply(&gf_core::math::transpose(&turn), gf_core::math::apply(&rt, [0.0, -1.0, 0.0]));
        let (Some(vol), Some(low)) = (self.baker.layer_volume(i), self.stage.lowest.as_ref()) else { return false };
        let Some(reach) = low.reach(vol, down, gf_core::HB) else { return false };
        // its lowest point's height in the room now, and the move up the room's vertical that sets it on the floor
        let centre = gf_core::math::apply(&r, pl.pos)[1];
        let dy = h - (centre - reach * size);
        if std::env::var("GLASSFORMS_FLOOR_DEBUG").is_ok() { eprintln!("to floor: layer {i} reach {reach} size {size} centre {centre} floor {h} move {dy} down {down:?}"); }
        let up = gf_core::math::apply(&rt, [0.0, 1.0, 0.0]);
        let pos = [pl.pos[0] + up[0] * dy, pl.pos[1] + up[1] * dy, pl.pos[2] + up[2] * dy];
        for (c, k) in self.state.place_knobs(i).into_iter().enumerate().take(3) {
            let v = pos[c].clamp(k.lo, k.hi);
            self.state.user_set(&k, v, false);
        }
        true
    }

    /// Where the floor stands in the room (its height above the middle of the shape, so below it is
    /// negative); None when there is no floor.
    pub fn floor_height(&self) -> Option<f64> {
        self.stage.height()
    }

    /// The floor as whoever puts light on it needs to know it (None when there is no floor). The floor's
    /// own colour is worked out here; shadows and caustics are a number per colour that it is multiplied
    /// by, asked for in the tracer as `castFloor(p)` with p a point of the floor in the room's frame.
    pub fn stage_floor(&self) -> Option<StageFloor> {
        let height = self.stage.height()?;
        let floor = gf_core::stage::Floor::of(&self.state.p)?;
        Some(StageFloor { room: self.state.room_m(), height, backdrop: self.stage.light.backdrop, reach: FADE_FROM * floor.fade, fade: floor.fade })
    }

    /// What a set of settings costs: turning frames with them and without, in turn, `rounds` times over,
    /// `n` frames a go. Returns (ms a frame without, ms a frame with), each the quickest of its goes:
    /// other programs on the GPU can only make a go slower, and taking the two in turn in one run keeps
    /// what they do to the one close to what they do to the other.
    pub fn bench_with(&mut self, with: &[(String, f64)], n: u32, rounds: u32) -> (f64, f64) {
        let turn = gf_core::math::axis_angle(0.3, 1.0, 0.2, 0.6 / 120.0);
        let without: Vec<(String, f64)> = with.iter().map(|(k, _)| (k.clone(), self.state.p.get(k).unwrap_or(0.0))).collect();
        let mut best = [f64::MAX; 2];
        for _ in 0..rounds {
            for (k, set) in [&without, &with.to_vec()].into_iter().enumerate() {
                for (name, v) in set { self.state.p.set(name, *v); }
                // (the first frames after a change build what is not built yet, and start the picture afresh)
                for _ in 0..6 { self.state.turn(&turn); self.draw(false); }
                self.gpu.wait();
                let t0 = std::time::Instant::now();
                for i in 0..n {
                    self.state.turn(&turn);
                    self.draw(false);
                    if i % 4 == 3 { self.gpu.wait(); }
                }
                self.gpu.wait();
                best[k] = best[k].min(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
            }
        }
        for (name, v) in &without { self.state.p.set(name, *v); }
        self.dirty = true;
        (best[0], best[1])
    }

    /// The studio as a JSON member to add to a report (nothing when there is no studio).
    pub fn stage_json(&self) -> String {
        let s = Stage::of(&self.state.p);
        if s.off() { return String::new(); }
        let own = self.stage.floor_rgb();
        format!(
            ",\"stage\":{{\"backdrop\":{},\"shown\":{},\"floor\":{},\"floorRgb\":[{:.5},{:.5},{:.5}],\"measures\":{},\"measureMs\":{:.2}}}",
            s.backdrop.is_some(), s.show, self.stage.height().map_or("null".into(), |h| format!("{h:.6}")), own[0], own[1], own[2], self.stage.measures, self.stage.measure_ms
        )
    }
}
