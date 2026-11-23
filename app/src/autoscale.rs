//! Keeping the frame rate (docs/autoscale.md): an optional controller that lowers how sharp and how
//! detailed MOVING frames are, so that the window keeps up with a frame rate the owner picks, on any GPU.
//! Settled pictures are never touched: once the view is still everything goes back to full quality and
//! the picture converges to exactly what it would have been. Stills, videos and benchmarks ignore it.
//!
//! What it may change, least visible first (each only while moving):
//!   1. the internal render scale of moving frames (dynamic resolution, upscaled by the antialiasing that
//!      follows motion: render.rs `dyn_scale`), in steps of 5% down to `SCALE_MIN`;
//!   2. the voxel detail of a shape rebuilt every frame while it animates (`State::anim_cap`);
//!   3. the bounces inside the glass, then wavelengths beyond three;
//! and, separately, while settling:
//!   4. the refining budget (how many passes a settling frame holds; never what a pass is).
//!
//! The controller works from the GPU's own time for each frame (timestamp queries where the device has
//! them; otherwise frames timed to the end now and then, as refining does), smoothed, with hysteresis: it
//! steps down as soon as moving frames are over the budget (by a jump in proportion when far over, so a
//! weak GPU lands at a sane scale within a few frames), and steps up only after a run of frames well
//! under it. A step up that is followed straight away by a step down holds further steps up off for a
//! while, doubling each time, so it never pumps between two levels.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::gpu::Gpu;

/// Frame rates offered; 0 = the display's refresh rate.
pub const TARGETS: [u32; 6] = [30, 60, 90, 120, 144, 0];
/// The target until the owner picks another.
pub const TARGET_DEFAULT: u32 = 90;
/// The share of a frame's time the GPU work of a frame may take (the rest: the panels, the compositor).
pub const HEADROOM: f64 = 0.9;
/// The lowest render scale while moving, and the size of a step.
pub const SCALE_MIN: f64 = 0.5;
pub const SCALE_STEP: f64 = 0.05;
/// Moving frames are stepped down when their smoothed time is over the budget, and up when it has been
/// under this share of it for `UP_RUN` measured frames in a row.
const UP_AT: f64 = 0.8;
const UP_RUN: u32 = 24;
/// Measured frames at a level before it may be stepped down from (the first frames after a change can be
/// the old level's, still in flight).
const DOWN_AFTER: u32 = 4;
/// Over this many times the budget, the step down is a jump in proportion to the overshoot.
const JUMP_AT: f64 = 1.25;
/// A step down within this many seconds of a step up says the step up was a mistake.
const REGRET_S: f64 = 2.0;
const COOLDOWN_MAX_S: f64 = 16.0;
/// How quickly the smoothed time follows each frame, and the most one frame may count as (a lone hitch,
/// a shape rebuilt, must not throw the scale down).
const ALPHA: f64 = 0.3;
const CLIP: f64 = 2.5;
/// The settling budget is scaled by `settle_k`, kept between these.
const SETTLE_K_MIN: f64 = 0.3;
/// Settling frames are judged a run of this many at a time.
const SETTLE_RUN: usize = 15;

/// GLASSFORMS_SLOW=k (2 to 16): the plain tracer's pass is traced k times over, to stand in for a GPU k
/// times slower at tracing while trying the controller out. 1 (as always) otherwise.
pub fn slow() -> u32 {
    static SLOW: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *SLOW.get_or_init(|| std::env::var("GLASSFORMS_SLOW").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(1).clamp(1, 16))
}

/// GLASSFORMS_KEEP_SCALE=s: moving frames traced at share s whatever the controller says (for looking at
/// what a given scale looks like). None (the controller decides) otherwise.
pub fn fixed_scale() -> Option<f64> {
    static FIXED: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *FIXED.get_or_init(|| std::env::var("GLASSFORMS_KEEP_SCALE").ok().and_then(|v| v.trim().parse::<f64>().ok()).map(|s| s.clamp(0.25, 1.0)))
}

/// The owner's choice, kept in the window's own file (ui/store.rs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct KeepRate {
    pub on: bool,
    /// frames a second; 0 = the display's refresh rate
    pub target: u32,
}
impl Default for KeepRate {
    fn default() -> Self {
        Self { on: false, target: TARGET_DEFAULT }
    }
}

/// What a frame is: moving (the antialiasing that follows motion folds it in) or settling (refining).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Moving,
    Settling,
}

/// What a measured frame was: moving or settling, the controller's level when it began, and how long the
/// frame thread took over it (ms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tag {
    pub phase: Phase,
    pub gen: u32,
    pub cpu: f64,
}

/// What can be lowered for the scene as it is now.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Avail {
    /// moving frames can be traced smaller (render.rs `dyn_ok`)
    pub scale: bool,
    /// voxels a side the shape could be rebuilt at while it animates, below what it is set to, finest first
    /// (empty unless the shape is animating)
    pub details: Vec<u32>,
    /// the bounces and wavelengths set
    pub nb: i32,
    pub nw: i32,
}

/// One step past the render scale, in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Extra {
    Detail(u32),
    /// the share of the bounces kept
    Bounces(f64),
    Wavelengths(i32),
}

fn extras(a: &Avail) -> Vec<Extra> {
    let mut out: Vec<Extra> = a.details.iter().map(|&r| Extra::Detail(r)).collect();
    // (bounces: three quarters, then half, never under two; only steps that change something)
    let mut last = a.nb;
    for k in [0.75, 0.5] {
        let nb = ((a.nb as f64 * k).round() as i32).max(2);
        if nb < last {
            out.push(Extra::Bounces(k));
            last = nb;
        }
    }
    if a.nw > 3 { out.push(Extra::Wavelengths(3)); }
    out
}

/// What a moving frame is drawn with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Levers {
    /// the traced picture's size, per side, of the full one
    pub scale: f64,
    /// the most voxels a side an animating shape is rebuilt at
    pub detail: Option<u32>,
    pub nb: i32,
    pub nw: i32,
}

/// The bounces kept of `nb` at share `k`.
pub fn bounces(nb: i32, k: f64) -> i32 {
    if k >= 1.0 { nb } else { ((nb as f64 * k).round() as i32).max(2).min(nb) }
}

/// The controller. Pure arithmetic: fed the time each frame took, it says what moving frames are drawn
/// with (`levers`) and how the settling budget is scaled (`settle_k`).
#[derive(Clone, Debug)]
pub struct Controller {
    /// what a frame's GPU work may take, ms
    pub budget: f64,
    /// the render scale of moving frames (applied when the scene allows it)
    pub scale: f64,
    /// the steps past the scale taken, in order. Kept by what they lower, not by place in `extras`:
    /// that list is made from the scene each frame (no detail step once playback stops), so a place
    /// would change meaning under it. A step the scene no longer offers is kept but does nothing.
    pub extra: Vec<Extra>,
    /// goes up with every change: a time measured at an older level is not counted
    pub gen: u32,
    ema: Option<f64>,
    n: u32,
    over: u32,
    under: u32,
    last_up: Option<f64>,
    cooldown: f64,
    block_until: f64,
    /// the settling budget is multiplied by this (1: as without the controller, at 90 frames a second)
    pub settle_k: f64,
    settle_win: Vec<f64>,
    /// the last smoothed times, moving and settling, for the readout
    pub moving_ms: f64,
    pub settle_ms: f64,
}

impl Controller {
    pub fn new(budget: f64) -> Self {
        Self { budget, scale: 1.0, extra: Vec::new(), gen: 0, ema: None, n: 0, over: 0, under: 0, last_up: None, cooldown: 0.0, block_until: 0.0, settle_k: 1.0, settle_win: Vec::new(), moving_ms: 0.0, settle_ms: 0.0 }
    }

    /// The budget for a target rate.
    pub fn budget_for(fps: f64) -> f64 {
        1000.0 / fps.max(1.0) * HEADROOM
    }

    pub fn set_budget(&mut self, budget: f64) {
        if (budget - self.budget).abs() > 1e-9 {
            self.budget = budget;
            self.restart();
        }
    }

    fn restart(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        self.ema = None;
        self.n = 0;
        self.over = 0;
        self.under = 0;
    }

    /// What moving frames are drawn with, for the scene as it is now.
    pub fn levers(&self, a: &Avail) -> Levers {
        let ex = extras(a);
        let mut lv = Levers { scale: if a.scale { self.scale } else { 1.0 }, detail: None, nb: a.nb, nw: a.nw };
        for e in ex.iter().filter(|e| self.extra.contains(e)) {
            match *e {
                Extra::Detail(r) => lv.detail = Some(r),
                Extra::Bounces(k) => lv.nb = bounces(a.nb, k),
                Extra::Wavelengths(w) => lv.nw = w.min(a.nw),
            }
        }
        lv
    }

    /// A frame's time on the GPU (from its first piece of work to its last), measured for frame `tag`;
    /// `now` in seconds (any steady clock).
    pub fn measured(&mut self, tag: Tag, ms: f64, now: f64, a: &Avail) {
        let Tag { phase, gen, cpu } = tag;
        if !ms.is_finite() || ms <= 0.0 { return; }
        let b = self.budget;
        match phase {
            Phase::Settling => {
                // (a still picture: the next movement starts its own count)
                self.ema = None;
                self.n = 0;
                self.under = 0;
                // (judged by the 80th percentile of a run of frames, as the frame rate is: settling with one
                // wavelength inside has a whole-comb frame every few, which no budget makes cheaper)
                self.settle_win.push(ms);
                if self.settle_win.len() < SETTLE_RUN { return; }
                let mut w = std::mem::take(&mut self.settle_win);
                w.sort_by(f64::total_cmp);
                let p80 = w[(w.len() * 4) / 5 - 1];
                self.settle_ms = p80;
                if p80 > b {
                    self.settle_k = (self.settle_k * 0.85).max(SETTLE_K_MIN);
                } else if p80 < UP_AT * b && self.settle_k < 1.0 {
                    self.settle_k = (self.settle_k * 1.1).min(1.0);
                }
            }
            Phase::Moving => {
                self.settle_win.clear();
                if gen != self.gen { return; }
                let s = ms.min(CLIP * b);
                let e = self.ema.map_or(s, |e| e + ALPHA * (s - e));
                self.ema = Some(e);
                self.moving_ms = e;
                self.n += 1;
                // (two smoothed times over in a row: one hitch lifts the mean over for a frame at most)
                // (a frame the frame thread itself took nearly as long over is held up by the CPU, not by
                // tracing: nothing the GPU draws with would help, so it is not counted as over)
                let cpu_bound = cpu > 0.85 * ms;
                self.over = if e > b && !cpu_bound { self.over + 1 } else { 0 };
                if self.n >= DOWN_AFTER && self.over >= 2 {
                    self.down(e, now, a);
                } else if e < UP_AT * b {
                    self.under += 1;
                    if self.under >= UP_RUN && now >= self.block_until { self.up(now, a); }
                } else {
                    self.under = 0;
                }
            }
        }
    }

    fn down(&mut self, ms: f64, now: f64, a: &Avail) {
        // a step down soon after a step up: that step was one too many; hold the next one off for longer
        if self.last_up.is_some_and(|t| now - t < REGRET_S) {
            self.cooldown = (self.cooldown * 2.0).clamp(1.0, COOLDOWN_MAX_S);
            self.block_until = now + self.cooldown;
        } else {
            self.cooldown *= 0.5;
        }
        self.last_up = None;
        let next = extras(a).into_iter().find(|e| !self.extra.contains(e));
        if a.scale && self.scale > SCALE_MIN + 1e-9 {
            let mut s = self.scale - SCALE_STEP;
            if ms > JUMP_AT * self.budget {
                // most of a frame's cost goes with its pixels: aim a little under the budget at once
                let aim = self.scale * (0.95 * self.budget / ms).sqrt();
                s = s.min((aim / SCALE_STEP).floor() * SCALE_STEP);
            }
            self.scale = snap(s.max(SCALE_MIN));
        } else if let Some(e) = next {
            self.extra.push(e);
        } else {
            // nothing left to lower: keep measuring
            self.ema = None;
            self.n = 0;
            return;
        }
        self.restart();
    }

    fn up(&mut self, now: f64, a: &Avail) {
        // (the latest step the scene still offers goes back first)
        let ex = extras(a);
        if let Some(k) = self.extra.iter().rposition(|e| ex.contains(e)) {
            self.extra.remove(k);
        } else if a.scale && self.scale < 1.0 - 1e-9 {
            self.scale = snap((self.scale + SCALE_STEP).min(1.0));
        } else {
            self.under = 0;
            return;
        }
        self.last_up = Some(now);
        self.restart();
    }

    /// What it is doing, in a few words (the quality section's readout).
    pub fn says(&self, a: &Avail) -> String {
        let lv = self.levers(a);
        let mut parts = Vec::new();
        if lv.scale < 0.999 { parts.push(format!("{:.0}% scale", lv.scale * 100.0)); }
        if let Some(r) = lv.detail { parts.push(format!("{r}³")); }
        if lv.nb < a.nb { parts.push(format!("{} bounces", lv.nb)); }
        if lv.nw < a.nw { parts.push(format!("{} wavelengths", lv.nw)); }
        let moving = if parts.is_empty() { "moving at full quality".to_owned() } else { format!("moving at {}", parts.join(", ")) };
        let settle = if self.settle_k < 0.999 { format!(" · settling at {:.0}% budget", self.settle_k * 100.0) } else { String::new() };
        format!("{moving}{settle}")
    }

    /// Something is lowered now.
    pub fn active(&self, a: &Avail) -> bool {
        let lv = self.levers(a);
        lv.scale < 0.999 || lv.detail.is_some() || lv.nb < a.nb || lv.nw < a.nw || self.settle_k < 0.999
    }
}

/// A scale put on the 5% grid (so steps up and down meet the same values).
fn snap(s: f64) -> f64 {
    ((s / SCALE_STEP).round() * SCALE_STEP).clamp(SCALE_MIN, 1.0)
}

/// The display's refresh rate, worked out from how often the window's frames come while the GPU has time
/// to spare (with vsync they come once a refresh): the commonest rates, nearest.
#[derive(Clone, Debug, Default)]
pub struct Refresh {
    gaps: Vec<f64>,
    pub hz: Option<f64>,
}
impl Refresh {
    const RATES: [f64; 10] = [30.0, 50.0, 60.0, 75.0, 90.0, 100.0, 120.0, 144.0, 165.0, 240.0];
    /// The time between two frames, and whether that frame's GPU work was well within it.
    pub fn gap(&mut self, dt: f64, spare: bool) {
        if !(spare && dt > 1.0 / 400.0 && dt < 0.1) { return; }
        self.gaps.push(dt);
        if self.gaps.len() >= 60 {
            let mut g = std::mem::take(&mut self.gaps);
            g.sort_by(f64::total_cmp);
            let hz = 1.0 / g[g.len() / 2];
            self.hz = Self::RATES.iter().copied().min_by(|a, b| (a - hz).abs().total_cmp(&(b - hz).abs()));
        }
    }
}

/// The GPU's own time for a frame: a timestamp written before its first piece of work and after its last,
/// read back a few frames later without waiting. None where the device cannot write timestamps.
pub struct Clock {
    set: wgpu::QuerySet,
    slots: Vec<Slot>,
    next: usize,
    period_ns: f64,
}
struct Slot {
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    /// 0 free, 1 begun, 2 on its way back, 3 arrived
    state: Arc<AtomicU8>,
    tag: Tag,
}
const SLOTS: usize = 8;

impl Clock {
    pub fn new(gpu: &Gpu) -> Option<Clock> {
        let need = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !gpu.device.features().contains(need) || std::env::var_os("GLASSFORMS_KEEP_NO_TIMESTAMPS").is_some() { return None; }
        let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("keep rate"), ty: wgpu::QueryType::Timestamp, count: 2 * SLOTS as u32 });
        let slots = (0..SLOTS)
            .map(|_| Slot {
                resolve: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("keep rate: resolve"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
                read: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("keep rate: read"), size: 16, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false }),
                state: Arc::new(AtomicU8::new(0)),
                tag: Tag { phase: Phase::Moving, gen: 0, cpu: 0.0 },
            })
            .collect();
        Some(Clock { set, slots, next: 0, period_ns: gpu.queue.get_timestamp_period() as f64 })
    }

    /// Mark the start of a frame's GPU work. None if every slot is still on its way back (that frame is
    /// not measured).
    pub fn begin(&mut self, gpu: &Gpu) -> Option<usize> {
        let i = self.next;
        if self.slots[i].state.load(Ordering::Acquire) != 0 { return None; }
        self.next = (i + 1) % SLOTS;
        self.slots[i].state.store(1, Ordering::Release);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("keep rate: begin") });
        enc.write_timestamp(&self.set, 2 * i as u32);
        gpu.queue.submit([enc.finish()]);
        Some(i)
    }

    /// Mark the end of that frame's GPU work, `tag`ged with what the frame was. `drew`: false when the
    /// frame drew nothing after all (the slot is given back unmeasured).
    pub fn end(&mut self, gpu: &Gpu, i: usize, tag: Option<Tag>) {
        let s = &mut self.slots[i];
        let Some(tag) = tag else {
            // (the begin mark is in the queue already; it is simply never read)
            s.state.store(0, Ordering::Release);
            return;
        };
        s.tag = tag;
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("keep rate: end") });
        enc.write_timestamp(&self.set, 2 * i as u32 + 1);
        enc.resolve_query_set(&self.set, 2 * i as u32..2 * i as u32 + 2, &s.resolve, 0);
        enc.copy_buffer_to_buffer(&s.resolve, 0, &s.read, 0, 16);
        gpu.queue.submit([enc.finish()]);
        s.state.store(2, Ordering::Release);
        let flag = s.state.clone();
        s.read.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(if r.is_ok() { 3 } else { 0 }, Ordering::Release));
    }

    /// The frames whose times have come back since last asked: (what the frame was, GPU ms). Never waits.
    pub fn arrived(&mut self, gpu: &Gpu) -> Vec<(Tag, f64)> {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let mut out = Vec::new();
        for s in &mut self.slots {
            if s.state.load(Ordering::Acquire) != 3 { continue; }
            let ticks = {
                let v = s.read.slice(..).get_mapped_range();
                v.ok().map(|v| {
                    let a = u64::from_le_bytes(v[0..8].try_into().expect("8 bytes"));
                    let b = u64::from_le_bytes(v[8..16].try_into().expect("8 bytes"));
                    b.saturating_sub(a)
                })
            };
            s.read.unmap();
            s.state.store(0, Ordering::Release);
            if let Some(t) = ticks.filter(|t| *t > 0) { out.push((s.tag, t as f64 * self.period_ns / 1e6)); }
        }
        out
    }
}

/// The controller with its clock, as the engine keeps it while the owner has it on.
pub struct Keep {
    pub ctl: Controller,
    pub clock: Option<Clock>,
    /// frames a second aimed at
    pub fps: f64,
    /// what can be lowered now (set each frame by the engine)
    pub avail: Avail,
    /// the slot of the frame being drawn, and when the clock is missing, frames counted for timing the
    /// odd one to its end instead
    pub(crate) slot: Option<usize>,
    pub(crate) count: u32,
    pub(crate) t0: Option<std::time::Instant>,
    pub(crate) started: std::time::Instant,
    /// the frame being drawn is a moving or a settling one (None: nothing drawn yet)
    pub(crate) phase: Option<Phase>,
    /// the controller's level when the frame began (a time measured for it counts only at that level)
    pub(crate) gen: u32,
    /// when the frame thread began the frame
    pub(crate) cpu0: Option<std::time::Instant>,
    /// GLASSFORMS_KEEP_LOG: one line per measured frame
    pub(crate) log: Option<std::fs::File>,
}

impl Keep {
    pub fn new(gpu: &Gpu, fps: f64) -> Keep {
        let log = std::env::var_os("GLASSFORMS_KEEP_LOG").and_then(|p| std::fs::File::create(p).ok());
        Keep { ctl: Controller::new(Controller::budget_for(fps)), clock: Clock::new(gpu), fps, avail: Avail::default(), slot: None, count: 0, t0: None, started: std::time::Instant::now(), phase: None, gen: 0, cpu0: None, log }
    }

    pub fn set_fps(&mut self, fps: f64) {
        self.fps = fps;
        self.ctl.set_budget(Controller::budget_for(fps));
    }

    pub fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// What it is doing, for the quality section.
    pub fn says(&self) -> String {
        format!("{} · aiming at {:.0} fps", self.ctl.says(&self.avail), self.fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avail() -> Avail {
        Avail { scale: true, details: vec![], nb: 4, nw: 3 }
    }

    /// A pretend GPU: a fixed cost plus one in proportion to the pixels, plus bounces.
    fn cost(lv: &Levers, base_px: f64, fixed: f64) -> f64 {
        fixed + base_px * lv.scale * lv.scale * (0.5 + 0.5 * lv.nb as f64 / 4.0)
    }

    fn run(ctl: &mut Controller, a: &Avail, frames: usize, base_px: f64, fixed: f64, t: &mut f64, log: &mut Vec<f64>) {
        // (times come back three frames late, as the timestamps do)
        let mut queue: std::collections::VecDeque<(u32, f64)> = Default::default();
        for _ in 0..frames {
            let lv = ctl.levers(a);
            queue.push_back((ctl.gen, cost(&lv, base_px, fixed)));
            if queue.len() > 3 {
                let (g, ms) = queue.pop_front().unwrap();
                ctl.measured(Tag { phase: Phase::Moving, gen: g, cpu: 1.0 }, ms, *t, a);
            }
            log.push(lv.scale);
            *t += 1.0 / 90.0;
        }
    }

    #[test]
    fn a_weak_gpu_lands_under_the_budget_quickly_and_stays() {
        let a = avail();
        let mut c = Controller::new(10.0);
        let (mut t, mut log) = (0.0, Vec::new());
        // four times too slow at full size
        run(&mut c, &a, 90, 39.0, 1.0, &mut t, &mut log);
        let lv = c.levers(&a);
        let ms = cost(&lv, 39.0, 1.0);
        assert!(ms <= 10.0, "still over after a second: {ms} at {lv:?}");
        // and within a couple of frames of a second's worth it no longer changes
        let tail: Vec<f64> = log[60..].to_vec();
        assert!(tail.windows(2).all(|w| w[0] == w[1]), "it kept changing: {tail:?}");
        // then, for five seconds more, it does not pump
        let mut more = Vec::new();
        run(&mut c, &a, 450, 39.0, 1.0, &mut t, &mut more);
        let changes = more.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(changes <= 4, "{changes} changes in five seconds: {more:?}");
        assert!(cost(&c.levers(&a), 39.0, 1.0) <= 10.0);
    }

    #[test]
    fn a_fast_gpu_stays_at_full_quality() {
        let a = avail();
        let mut c = Controller::new(10.0);
        let (mut t, mut log) = (0.0, Vec::new());
        run(&mut c, &a, 600, 6.0, 1.0, &mut t, &mut log);
        assert!(log.iter().all(|s| *s == 1.0));
        assert_eq!(c.levers(&a), Levers { scale: 1.0, detail: None, nb: 4, nw: 3 });
    }

    #[test]
    fn it_climbs_back_when_the_scene_gets_lighter() {
        let a = avail();
        let mut c = Controller::new(10.0);
        let (mut t, mut log) = (0.0, Vec::new());
        run(&mut c, &a, 120, 20.0, 1.0, &mut t, &mut log);
        assert!(c.scale < 0.8);
        run(&mut c, &a, 900, 5.0, 1.0, &mut t, &mut log);
        assert_eq!(c.scale, 1.0, "did not climb back");
    }

    #[test]
    fn past_the_least_scale_it_lowers_detail_then_bounces() {
        let a = Avail { scale: true, details: vec![192], nb: 8, nw: 3 };
        let mut c = Controller::new(10.0);
        let (mut t, mut log) = (0.0, Vec::new());
        run(&mut c, &a, 400, 200.0, 1.0, &mut t, &mut log);
        let lv = c.levers(&a);
        assert_eq!(lv.scale, SCALE_MIN);
        assert_eq!(lv.detail, Some(192));
        assert_eq!(lv.nb, 4);
        // with no scale to lower, the other steps come first
        let b = Avail { scale: false, ..a.clone() };
        let mut d = Controller::new(10.0);
        run(&mut d, &b, 60, 30.0, 1.0, &mut t, &mut log);
        assert_eq!(d.levers(&b).scale, 1.0);
        assert!(!d.extra.is_empty());
    }

    #[test]
    fn settling_lowers_only_its_budget() {
        let a = avail();
        let mut c = Controller::new(10.0);
        for i in 0..20 { c.measured(Tag { phase: Phase::Settling, gen: 0, cpu: 1.0 }, 14.0, i as f64 * 0.01, &a); }
        assert!(c.settle_k < 1.0);
        assert_eq!(c.levers(&a).scale, 1.0);
        for i in 0..200 { c.measured(Tag { phase: Phase::Settling, gen: 0, cpu: 1.0 }, 5.0, i as f64 * 0.01, &a); }
        assert_eq!(c.settle_k, 1.0);
    }

    #[test]
    fn a_lone_hitch_does_not_lower_anything() {
        let a = avail();
        let mut c = Controller::new(10.0);
        for i in 0..60 {
            let ms = if i == 30 { 80.0 } else { 6.0 };
            c.measured(Tag { phase: Phase::Moving, gen: c.gen, cpu: 1.0 }, ms, i as f64 / 90.0, &a);
        }
        assert_eq!(c.scale, 1.0);
    }

    #[test]
    fn a_frame_held_up_by_the_cpu_lowers_nothing() {
        let a = avail();
        let mut c = Controller::new(10.0);
        for i in 0..60 { c.measured(Tag { phase: Phase::Moving, gen: c.gen, cpu: 19.0 }, 20.0, i as f64 / 90.0, &a); }
        assert_eq!(c.levers(&a).scale, 1.0);
    }

    #[test]
    fn the_refresh_rate_is_read_from_frame_gaps() {
        let mut r = Refresh::default();
        for i in 0..60 { r.gap(1.0 / 143.7 + if i % 7 == 0 { 0.003 } else { 0.0 }, true); }
        assert_eq!(r.hz, Some(144.0));
    }
    #[test]
    fn a_lowered_step_keeps_its_meaning_when_the_scene_changes() {
        // playing: the shape is rebuilt every frame, so its detail can be lowered first
        let playing = Avail { scale: false, details: vec![192], nb: 8, nw: 3 };
        let mut c = Controller::new(10.0);
        let mut i = 0;
        while c.levers(&playing).detail.is_none() {
            c.measured(Tag { phase: Phase::Moving, gen: c.gen, cpu: 1.0 }, 30.0, i as f64 / 90.0, &playing);
            i += 1;
            assert!(i < 100, "never lowered the detail");
        }
        assert_eq!(c.levers(&playing).nb, 8);
        // stopped: there is no detail to lower any more, and nothing else was measured as needed
        let stopped = Avail { details: vec![], ..playing.clone() };
        let lv = c.levers(&stopped);
        assert_eq!((lv.nb, lv.nw, lv.detail), (8, 3, None), "the bounces dropped without a measurement");
        // playing again: the detail it lowered is lowered again
        assert_eq!(c.levers(&playing).detail, Some(192));
    }
}
