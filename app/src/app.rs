//! The live window: the picture filling it, the panels floating over it, the mouse and the keys.
//!
//! The window and the renderer share one graphics device. Each frame the controls change the state, the
//! engine draws what the state now says, and its picture is shown pixel for pixel behind the panels. When
//! the picture has settled and nothing moves, nothing is drawn and the program sits idle.
//!
//! On the command line, besides the settings of the picture:
//!   --setup <file>        start from a saved setup (settings given as name=value are laid over it)
//!   w=<pixels> h=<pixels> size the picture exactly;  backend=vulkan|dx12  and  vsync=0  choose how it is drawn
//!   fps= samples= lensRays= shutter= codec= crf=      how videos are made, as for --video
//!
//! For tests the window can be driven by a script and made to write a screenshot and its state:
//!   --ui-do=<actions>     run these first (see ui/script.rs); written with the "=", so that an action
//!                         with a "=" of its own is not taken for a setting
//!   --ui-shot <png>       then save a picture of the whole window, and leave
//!   --ui-frames <n>       ... once the picture has had this many frames (30)
//!   --ui-dump <json>      write the state as it ends up, and leave
//!   --ui-scale <s>        pixels per point (default: the display's own scaling)
//!   --ui-out <folder>     where the script's videos go
//!   --ui-store <file>     remember things in this file (a scripted run otherwise remembers nothing)
//!   --ui-timeout <s>      give up after this long (180)
//!   --ui-show             put the window on the desktop. Without it a scripted run's window opens past the
//!                         edge of the desktop, where it is drawn and photographed like any other but is in
//!                         nobody's way; it never takes the keyboard or shows in the taskbar either way

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Align2, Color32, CursorIcon, Id, PointerButton, Pos2, Rect, Sense};
use eframe::egui_wgpu;
use gf_core::math::euler;
use gf_core::state::State;

use crate::engine::Engine;
use crate::gpu::{self, Gpu};
use crate::ui::script::{self, Act, Hold};
use crate::ui::store::{self, Store};
use crate::ui::theme::{self, EDGE, LEFT_W, RIGHT_W};
use crate::ui::{animate, gizmo, knob, left, material, overlay, right, room, widgets, Bench, Panels, Request};
use crate::video::{Codec, VideoJob, VideoOpts, VideoResult};

/// Things given as name=value that are about the window or the video rather than settings of the picture.
const NOT_SETTINGS: [&str; 14] = ["w", "h", "frames", "anim", "backend", "vsync", "fps", "samples", "lensRays", "shutter", "codec", "crf", "detail", "play"];

/// What the command line asked of the window.
struct Opts {
    pairs: Vec<(String, String)>,
    /// the picture's size in pixels, if one was asked for
    size_px: Option<(u32, u32)>,
    scale: Option<f32>,
    script: Vec<Act>,
    shot: Option<PathBuf>,
    frames: u32,
    dump: Option<PathBuf>,
    out: Option<PathBuf>,
    store: Option<PathBuf>,
    /// a saved setup to start from
    setup: Option<PathBuf>,
    timeout: f64,
    /// a script, a screenshot or a dump was asked for: real input is ignored and nothing is remembered
    scripted: bool,
    /// a scripted run's window is wanted on the desktop, where it can be watched
    show: bool,
}

impl Opts {
    fn read(pairs: Vec<(String, String)>, flags: &[(String, Option<String>)]) -> Result<Opts, String> {
        let flag = |k: &str| flags.iter().rev().find(|f| f.0 == k).map(|f| f.1.clone().unwrap_or_default());
        let pair = |k: &str| pairs.iter().rev().find(|p| p.0 == k).map(|p| p.1.as_str());
        let number = |k: &str| -> Result<Option<f64>, String> {
            match flag(k) {
                Some(v) => v.trim().parse::<f64>().map(Some).map_err(|_| format!("--{k}: \"{v}\" is not a number")),
                None => Ok(None),
            }
        };
        let mut text = flag("ui-do");
        if text.as_deref() == Some("") {
            // "--ui-do set:ior=1.7" reaches us as the setting "set:ior" = "1.7": put it back together
            let found = pairs.iter().find(|(k, _)| k.contains(':'));
            text = found.map(|(k, v)| format!("{k}={v}"));
        }
        let script = match &text {
            Some(t) => script::parse(t).map_err(|e| format!("--ui-do: {e}"))?,
            None => Vec::new(),
        };
        let path = |k: &str| flag(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let (shot, dump) = (path("ui-shot"), path("ui-dump"));
        let px = |k: &str| pair(k).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| *v >= 2.0).map(|v| v as u32);
        Ok(Opts {
            size_px: px("w").zip(px("h")),
            scale: number("ui-scale")?.map(|s| s as f32).filter(|s| *s > 0.1),
            scripted: text.is_some() || shot.is_some() || dump.is_some(),
            show: flag("ui-show").is_some(),
            frames: number("ui-frames")?.map_or(30, |n| n.max(1.0) as u32),
            timeout: number("ui-timeout")?.unwrap_or(180.0),
            out: path("ui-out"),
            store: path("ui-store"),
            setup: path("setup"),
            script, shot, dump, pairs,
        })
    }
    fn pair(&self, k: &str) -> Option<&str> {
        self.pairs.iter().rev().find(|p| p.0 == k).map(|p| p.1.as_str())
    }
}

/// Open the window and run until it is closed. `pairs`: settings as name=value; `flags`: the command
/// line's --flags with their values.
pub fn run(pairs: Vec<(String, String)>, flags: Vec<(String, Option<String>)>) -> Result<(), String> {
    let opts = Opts::read(pairs, &flags)?;
    // a scripted run remembers nothing unless it is given a file to remember in
    let store_path = if opts.scripted { opts.store.clone() } else { Some(opts.store.clone().unwrap_or_else(Store::default_path)) };
    let store = store_path.as_deref().map(Store::load).unwrap_or_default();
    let no_terminal = !opts.scripted && leave_own_console();

    let mut viewport = egui::ViewportBuilder::default().with_title("Glass Forms").with_min_inner_size([480.0, 320.0]).with_icon(theme::icon());
    viewport = match opts.size_px {
        // the scaling of the display is not known yet: a first guess, put right once the window is up
        Some((w, h)) => viewport.with_inner_size([w as f32, h as f32]),
        None => viewport.with_inner_size(store.window.unwrap_or([1440.0, 900.0])).with_maximized(store.maximized),
    };
    if opts.scripted {
        // a test's window should not take the keyboard from whoever is at the machine, nor get in their way
        viewport = viewport.with_active(false).with_taskbar(false);
        if !opts.show { viewport = viewport.with_position(out_of_sight()); }
    }
    let native = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: wgpu_options(opts.pair("backend").unwrap_or("auto"), opts.pair("vsync") != Some("0")),
        persist_window: false,
        dithering: false,
        centered: !opts.scripted || opts.show,
        ..Default::default()
    };
    if opts.scripted {
        // a test must never be left running
        let limit = opts.timeout;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(limit));
            eprintln!("glassforms: the scripted run did not finish in {limit} s");
            std::process::exit(3);
        });
    }
    let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let report = failure.clone();
    eframe::run_native(
        "Glass Forms",
        native,
        Box::new(move |cc| match App::new(cc, opts, store, store_path, report) {
            Ok(app) => Ok(Box::new(app) as Box<dyn eframe::App>),
            Err(e) => Err(e.into()),
        }),
    )
    .map_err(|e| {
        let why = format!("could not open the window: {e}");
        if no_terminal {
            // nowhere to print it: say it in a box
            rfd::MessageDialog::new().set_title("Glass Forms").set_description(why.as_str()).set_level(rfd::MessageLevel::Error).show();
        }
        why
    })?;
    let failed = failure.lock().ok().and_then(|mut f| f.take());
    failed.map_or(Ok(()), Err)
}

/// A place for a window where nobody will see it: just past the right edge of the whole desktop. The
/// window is as real there as anywhere (it is drawn, and a picture can be taken of it); it is only not
/// in front of whoever is using the machine. In points, which are never smaller than pixels, so that the
/// place is past the edge whatever the scaling.
/// With GF_PERFLOG=<file> in the environment, a line a second is appended to that file: seconds since the
/// window opened, frames drawn in that second, the mean and longest time between drawn frames, the mean
/// and longest time the engine's own call took, and what the engine keeps (see `Engine::kept`).
fn perf_log(engine: &Engine, drew: bool, call_ms: f64) {
    struct Log { file: std::fs::File, t0: Instant, at: Instant, last: Option<Instant>, n: u32, gap: (f64, f64), call: (f64, f64) }
    static LOG: Mutex<Option<Option<Log>>> = Mutex::new(None);
    let mut guard = LOG.lock().unwrap_or_else(|e| e.into_inner());
    let log = guard.get_or_insert_with(|| {
        let path = std::env::var("GF_PERFLOG").ok()?;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).ok()?;
        let now = Instant::now();
        Some(Log { file, t0: now, at: now, last: None, n: 0, gap: (0.0, 0.0), call: (0.0, 0.0) })
    });
    let Some(l) = log.as_mut() else { return };
    let now = Instant::now();
    if drew {
        if let Some(t) = l.last {
            let g = (now - t).as_secs_f64() * 1e3;
            l.gap = (l.gap.0 + g, l.gap.1.max(g));
        }
        l.last = Some(now);
        l.n += 1;
        l.call = (l.call.0 + call_ms, l.call.1.max(call_ms));
    }
    if (now - l.at).as_secs_f64() >= 1.0 {
        use std::io::Write as _;
        let n = l.n.max(1) as f64;
        let _ = writeln!(l.file, "{:.1} {} {:.2} {:.1} {:.2} {:.1} {}", (now - l.t0).as_secs_f64(), l.n, l.gap.0 / n, l.gap.1, l.call.0 / n, l.call.1, engine.kept());
        (l.at, l.n, l.gap, l.call) = (now, 0, (0.0, 0.0), (0.0, 0.0));
    }
}

fn out_of_sight() -> [f32; 2] {
    #[cfg(windows)]
    {
        #[link(name = "user32")]
        unsafe extern "system" {
            fn GetSystemMetrics(index: i32) -> i32;
        }
        // Safety: asks only for two numbers. 76 and 78: where the whole desktop begins, and how wide it is.
        let right = unsafe { GetSystemMetrics(76) + GetSystemMetrics(78) };
        return [right.clamp(0, 12000) as f32 + 64.0, 0.0];
    }
    #[allow(unreachable_code)]
    [12000.0, 0.0]
}

/// Started by a double click, the program is given a console window of its own, which is of no use to a
/// program with a window: let it go. Started from a terminal, the terminal is kept, for messages.
/// True if there is now nowhere to print.
fn leave_own_console() -> bool {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetConsoleProcessList(list: *mut u32, count: u32) -> u32;
            fn FreeConsole() -> i32;
        }
        let mut ids = [0u32; 2];
        // Safety: the list has room for the two ids asked for; both calls only concern this process's console.
        unsafe {
            if GetConsoleProcessList(ids.as_mut_ptr(), 2) == 1 {
                FreeConsole();
                return true;
            }
        }
    }
    false
}

/// How the graphics device is made: the fastest adapter, through Vulkan where there is a choice
/// (measured faster than DirectX 12 for this renderer), with the features and limits the renderer wants.
fn wgpu_options(backend: &str, vsync: bool) -> egui_wgpu::WgpuConfiguration {
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = match backend {
        "dx12" => wgpu::Backends::DX12,
        "vulkan" => wgpu::Backends::VULKAN,
        _ => wgpu::Backends::VULKAN | wgpu::Backends::DX12,
    };
    setup.power_preference = wgpu::PowerPreference::HighPerformance;
    setup.native_adapter_selector = Some(Arc::new(|adapters, surface| {
        let rank = |a: &wgpu::Adapter| {
            let info = a.get_info();
            let kind = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 3,
                wgpu::DeviceType::IntegratedGpu => 2,
                wgpu::DeviceType::Cpu => 0,
                _ => 1,
            };
            kind * 2 + (info.backend == wgpu::Backend::Vulkan) as i32
        };
        adapters
            .iter()
            .filter(|a| surface.is_none_or(|s| a.is_surface_supported(s)))
            .max_by_key(|a| rank(a))
            .cloned()
            .ok_or_else(|| "no graphics adapter can draw to this window".to_owned())
    }));
    setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
        label: Some("glassforms"),
        required_features: gpu::wanted_features(adapter),
        required_limits: gpu::wanted_limits(adapter),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    });
    egui_wgpu::WgpuConfiguration {
        surface: egui_wgpu::SurfaceConfig {
            present_mode: if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync },
            desired_maximum_frame_latency: Some(1),
        },
        wgpu_setup: setup.into(),
        ..Default::default()
    }
}

/// The engine's picture as egui knows it.
struct Shown {
    texture: wgpu::Texture,
    id: egui::TextureId,
    /// drawn smoothed (a video frame fitted into the window) rather than pixel for pixel
    smooth: bool,
}

/// What kind of drag is going on over the picture.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gesture {
    Turn,
    Pan,
    Roll,
    /// one of the handles over the picture has been taken hold of
    Handle,
}
struct Drag {
    kind: Gesture,
    /// how far the pointer has travelled, in points: a press that hardly moves is a click
    moved: f32,
    at: Pos2,
}

/// One step of a scripted run.
enum Step {
    Do(Act),
    /// hand these to the window as input at the start of the next frame, with these keys held
    Input(Vec<egui::Event>, egui::Modifiers),
    /// drop this file on the window at the start of the next frame, with these keys held
    Dropped(PathBuf, egui::Modifiers),
    /// let frames go by
    Frames(u32),
    /// wait for the video to end
    VideoDone,
    /// wait until a picture or a model being read, or a tracer being built, is in use
    Lit,
    /// wait until the window has the size asked for and the picture has had this many frames, then hold it there
    Settle(u32),
    /// save a picture of the whole window; `true` once it has been asked for
    Shot(PathBuf, bool),
    /// wait until nothing more is being drawn, then count the frames drawn in this many seconds
    Idle(f32, bool),
    Dump(PathBuf),
    Leave,
}

/// A scripted run in progress.
struct Test {
    steps: VecDeque<Step>,
    /// frames still to let go by
    waiting: u32,
    /// frames drawn since the settling began
    settled: u32,
    /// where a screenshot lands when the window system hands it over
    picture: Arc<Mutex<Option<Arc<egui::ColorImage>>>>,
    /// frames counted while idle: (seconds, frames)
    idle: Arc<Mutex<Option<(f32, u64)>>>,
    idle_found: Option<(f32, u64)>,
    out: PathBuf,
    /// the next video stops after this many frames
    video_frames: Option<u32>,
    /// keys held during scripted input, and whether the window has been told of them
    held: egui::Modifiers,
    told: Option<egui::Modifiers>,
    /// what the script noted each time it asked: what the line at the foot of the window said, which shape
    /// was up and how it was built, and its own sliders
    noted: Vec<serde_json::Value>,
}

struct App {
    engine: Engine,
    rs: egui_wgpu::RenderState,
    shown: Option<Shown>,
    /// the engine's small picture of the lighting, as egui knows it
    preview: Option<Shown>,
    px: Panels,
    opts: Opts,
    /// the size in pixels the window is still being asked to take (given on the command line)
    seeking: Option<(u32, u32)>,
    /// the picture's size in pixels and the window's inside in points, as of this frame
    canvas: (u32, u32),
    full: Rect,
    /// when the engine last drew, and whether it then had nothing more to do
    last: Option<Instant>,
    rested: bool,
    /// frames a second, as last worked out; and the frames drawn since then, and when that was
    fps: f64,
    drawn: (u32, Instant),
    stat_at: Instant,
    drag: Option<Drag>,
    video: Option<VideoJob>,
    last_video: Option<VideoResult>,
    /// frames until the benchmark runs (its button says "running..." for a frame first)
    bench_in: u8,
    /// the picture is held as it is (a screenshot is being taken)
    frozen: bool,
    test: Option<Test>,
    /// input waiting to be handed to the window
    inject: Vec<egui::Event>,
    /// files a script drops on the window, waiting likewise
    dropped: Vec<PathBuf>,
    /// the model file last asked for, by name, while it is being read; what there is to say of how it
    /// was taken in; and, if it was to be one more layer, the layer that was selected and how many
    /// there were when it was asked for
    model_asked: Option<String>,
    model_said: String,
    adding: Option<(usize, usize)>,
    /// why the last model file could not be used, and when that was first seen
    model_refused: Option<(String, Instant)>,
    failure: Arc<Mutex<Option<String>>>,
    store_path: Option<PathBuf>,
    saved: Store,
    changed_at: Option<Instant>,
    /// frames drawn since the window opened
    frames: Arc<AtomicU64>,
    /// times the program has been run since then: once for every frame, and also whenever the window
    /// system was late asking for a frame (as it may be for a window nobody can see)
    passes: u64,
    ffmpeg_checked: Instant,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, opts: Opts, store: Store, store_path: Option<PathBuf>, failure: Arc<Mutex<Option<String>>>) -> Result<App, String> {
        let rs = cc.wgpu_render_state.clone().ok_or("the window has no graphics device")?;
        let gpu = Gpu::from_parts(rs.device.clone(), rs.queue.clone(), &rs.adapter.get_info())?;
        let mut state = State::with_args(&opts.pairs, false);
        state.render = store.video.clone();
        if let Some(path) = &opts.setup {
            // a saved setup first, then whatever else the command line gave laid over it
            let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            state.apply_setup_json(&text)?;
            for (k, v) in &opts.pairs {
                if !NOT_SETTINGS.contains(&k.as_str()) && !state.apply_arg(k, v) { eprintln!("ignoring {k}={v}: no such setting"); }
            }
        }
        // how videos are made, by the names the command line's --video takes
        let number = |k: &str| opts.pair(k).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite() && *v >= 0.0);
        if let Some(v) = number("fps") { state.render.fps = (v as u32).max(1); }
        if let Some(v) = number("samples") { state.render.samples = (v as u32).max(1); }
        if let Some(v) = number("lensRays") { state.render.lens = v as u32; }
        if let Some(v) = number("shutter") { state.render.shutter = v.min(1.0); }
        if let Some(v) = number("crf") { state.render.crf = v as u32; }
        if let Some(c) = opts.pair("codec").and_then(Codec::from_id) { state.render.codec = c.id().into(); }
        let engine = Engine::new(gpu, state, false);
        theme::install_fonts(&cc.egui_ctx);
        theme::install_style(&cc.egui_ctx);

        let test = opts.scripted.then(|| {
            let mut steps: VecDeque<Step> = opts.script.iter().cloned().map(Step::Do).collect();
            if opts.shot.is_some() || opts.dump.is_some() {
                steps.push_back(Step::Settle(opts.frames));
                if let Some(p) = &opts.shot { steps.push_back(Step::Shot(p.clone(), false)); }
                if let Some(p) = &opts.dump { steps.push_back(Step::Dump(p.clone())); }
                steps.push_back(Step::Leave);
            }
            // videos made by a script go where they were asked to, or to the workspace's "out" when run from a build
            let out = opts.out.clone().unwrap_or_else(|| {
                let exe = crate::ui::store::program_folder();
                match exe.parent().and_then(Path::parent) {
                    Some(root) if root.join("Cargo.toml").exists() => root.join("out"),
                    _ => exe.join("renders"),
                }
            });
            Test {
                steps, waiting: 0, settled: 0, picture: Arc::new(Mutex::new(None)), idle: Arc::new(Mutex::new(None)), idle_found: None,
                out, video_frames: None, held: egui::Modifiers::NONE, told: None, noted: Vec::new(),
            }
        });
        let mut px = Panels::new(store.clone());
        px.ffmpeg = crate::video::ffmpeg().is_some();
        px.record = opts.scripted;
        Ok(App {
            engine, rs, shown: None, preview: None, px, seeking: opts.size_px, canvas: (0, 0), full: Rect::ZERO, last: None, rested: false, fps: 0.0, drawn: (0, Instant::now()), stat_at: Instant::now(),
            drag: None, video: None, last_video: None, bench_in: 0, frozen: false, test, inject: Vec::new(), dropped: Vec::new(), model_asked: None, model_said: String::new(), adding: None, model_refused: None, failure, store_path,
            saved: store, changed_at: None, frames: Arc::new(AtomicU64::new(0)), passes: 0, ffmpeg_checked: Instant::now(), opts,
        })
    }

    fn fail(&mut self, ctx: &egui::Context, why: String) {
        if let Ok(mut f) = self.failure.lock() { f.get_or_insert(why); }
        if let Some(t) = &mut self.test { t.steps.clear(); }
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    // ---- video ---------------------------------------------------------------------------------
    fn start_video(&mut self, frames: Option<u32>, dir: PathBuf) {
        let r = self.engine.state.render.clone();
        let size = if r.size == "window" {
            Some(self.canvas)
        } else {
            r.size.split_once('x').and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
        };
        let Some((width, height)) = size else {
            self.px.video_text = format!("could not render: \"{}\" is not a size", r.size);
            return;
        };
        let opts = VideoOpts {
            width, height, fps: r.fps.max(1), samples: r.samples.max(1), lens: r.lens, shutter: r.shutter, frames,
            codec: Codec::from_id(&r.codec).unwrap_or(Codec::H264), crf: r.crf, out: None, dir,
        };
        match VideoJob::start(&mut self.engine, opts) {
            Ok(job) => {
                self.px.video_text = format!("frame 0 of {}", job.total);
                self.video = Some(job);
                self.px.rendering = true;
                self.drag = None;
            }
            Err(e) => self.px.video_text = format!("could not render: {e}"),
        }
    }

    /// Render the next frame of the video; when the last has gone, close the file.
    fn step_video(&mut self) {
        let Some(mut job) = self.video.take() else { return };
        match job.step(&mut self.engine) {
            Ok(false) => {
                let (per, left) = job.pace();
                self.px.video_text = format!("frame {} of {} · {per:.2} s each · {:.0} s left", job.done, job.total, left.ceil());
                self.video = Some(job);
            }
            Ok(true) => {
                self.px.rendering = false;
                match job.finish(&mut self.engine) {
                    Ok(r) => {
                        let name = r.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        self.px.video_text = format!("{name}\n{} frames, {:.1} MB, took {:.0} s", r.frames, r.bytes as f64 / 1048576.0, r.seconds);
                        self.last_video = Some(r);
                    }
                    Err(e) => self.px.video_text = format!("could not render: {e}"),
                }
            }
            Err(e) => {
                job.cancel(&mut self.engine);
                self.px.rendering = false;
                self.px.video_text = format!("could not render: {e}");
            }
        }
    }

    fn cancel_video(&mut self) {
        if let Some(job) = self.video.take() {
            let done = job.done;
            job.cancel(&mut self.engine);
            self.px.rendering = false;
            self.px.video_text = format!("cancelled after {done} frames");
        }
    }

    // ---- setups and stills ---------------------------------------------------------------------
    /// Keep the whole setup (shape, sliders, view, lights, animation, video choices) in a file.
    fn save_setup(&mut self, path: &Path) {
        let name = file_name(path);
        let saved = make_folder(path).and_then(|()| std::fs::write(path, self.engine.state.setup_json()).map_err(|e| e.to_string()));
        if saved.is_ok() { self.px.store.setup_folder = store::remembered(path, "setups"); }
        self.px.file_text = match saved {
            Ok(()) => format!("saved {name}"),
            Err(e) => format!("could not save {name}: {e}"),
        };
    }

    /// Put a saved setup up. If it cannot be, everything stays as it is and the panel says why.
    fn load_setup(&mut self, path: &Path) {
        let name = file_name(path);
        let loaded = std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|text| self.engine.state.apply_setup_json(&text));
        if loaded.is_ok() { self.px.store.setup_folder = store::remembered(path, "setups"); }
        self.px.file_text = match loaded {
            Ok(()) => format!("loaded {name}"),
            Err(e) => format!("could not load {name}: {e}"),
        };
    }

    /// Where the "save" button puts a still: the stills folder, named like the videos, never over an older one.
    fn next_still(&self) -> PathBuf {
        // a script's stills go where its videos go
        let dir = self.test.as_ref().map_or_else(|| self.px.store.stills_folder(), |t| t.out.clone());
        let (screen, traced) = self.engine.renderer.size();
        let (w, h) = if self.px.store.still_16 { traced } else { screen };
        let base = format!("glass-{}-{w}x{h}", self.engine.state.preset().id);
        (1u32..).map(|n| dir.join(if n == 1 { format!("{base}.png") } else { format!("{base}-{n}.png") })).find(|p| !p.exists()).unwrap_or_else(|| dir.join(format!("{base}.png")))
    }

    /// Save the picture as it stands. With 8 bits a channel it is exactly what is on screen; with 16 it is
    /// the picture so far tone-mapped as the screen is, without grain and without rounding to 8 bits on
    /// the way (it has the size the picture is traced at). `keep`: later stills go to the same folder.
    fn save_still(&mut self, path: &Path, keep: bool) -> bool {
        let (name, deep) = (file_name(path), self.px.store.still_16);
        let e = &self.engine;
        let saved = make_folder(path).and_then(|()| {
            if deep {
                let (_, (w, h)) = e.renderer.size();
                let rgb = crate::render::tone_map16(&e.renderer.read_hdr(0, 0, w, h)?, &e.renderer.read_bloom()?, w, h, &e.state.p);
                // PNG keeps the high byte first
                let bytes: Vec<u8> = rgb.iter().flat_map(|v| v.to_be_bytes()).collect();
                write_png(path, w, h, png::ColorType::Rgb, png::BitDepth::Sixteen, &bytes).map(|()| (w, h))
            } else {
                let (w, h, rgba) = e.screenshot()?;
                write_png(path, w, h, png::ColorType::Rgba, png::BitDepth::Eight, &rgba).map(|()| (w, h))
            }
        });
        if keep && saved.is_ok() { self.px.store.still_folder = store::remembered(path, "stills"); }
        self.px.file_text = match &saved {
            Ok((w, h)) => format!("saved {name}\n{w} × {h}, {} bits a channel", if deep { 16 } else { 8 }),
            Err(e) => format!("could not save {name}: {e}"),
        };
        saved.is_ok()
    }

    // ---- models ----------------------------------------------------------------------------------
    /// Show a model's file: as the selected layer's shape or, when `new_layer` is asked for, as one more
    /// layer. With four layers there is no room for another: it takes the selected layer's place, and
    /// the panel says so. The file is read in the background; the shape changes when it is ready.
    fn load_model(&mut self, path: &Path, new_layer: bool) {
        let st = &self.engine.state;
        self.model_said = if new_layer && st.layer_count() >= gf_core::layers::MAX_LAYERS {
            format!("four layers already: it took the place of {}", st.layer_title(st.stack.sel))
        } else {
            String::new()
        };
        self.model_asked = Some(file_name(path));
        self.adding = new_layer.then_some((st.stack.sel, st.layer_count()));
        self.engine.load_model_as(&path.display().to_string(), new_layer);
    }

    /// What the panels are told of the selected layer's model: what the engine found on reading it,
    /// whether a file is being read, and why the last one could not be used.
    fn model_told(&mut self) -> crate::ui::Model {
        use gf_core::mesh::Treatment;
        let loading = self.engine.model_loading();
        if !loading { self.model_asked = None; }
        let st = &self.engine.state;
        // (what was said of the last file stands for as long as a model is up; while one is on its way it waits)
        if !loading && !st.is_model() { self.model_said.clear(); }
        let sel = st.stack.sel;
        let error = self.engine.model_error().map(str::to_owned);
        match &error {
            Some(why) if self.model_refused.as_ref().is_some_and(|(seen, _)| seen == why) => {}
            Some(why) => self.model_refused = Some((why.clone(), Instant::now())),
            None => self.model_refused = None,
        }
        // (the name of what is being read: the file asked for, or the one the shape already is)
        let reading = loading.then(|| self.model_asked.clone().or_else(|| st.model_file(sel).map(|f| file_name(Path::new(&f)))).unwrap_or_default());
        let said = if loading || error.is_some() { String::new() } else { self.model_said.clone() };
        let Some(i) = self.engine.model_info(sel) else { return crate::ui::Model { reading, error, said, ..Default::default() } };
        let r = &i.report;
        let found_shell = i.treatment == Treatment::Shell;
        let mut line = i.line.clone();
        if i.shell != found_shell { line += if i.shell { " · drawn as a shell by choice" } else { " · drawn as a solid by choice" }; }
        let mut tip = format!("as read: {} triangles, {} corners; kept: {} and {}", right::with_commas(r.triangles_in), right::with_commas(r.vertices_in), right::with_commas(r.triangles), right::with_commas(r.vertices));
        if !r.gaps.is_empty() {
            let some: Vec<String> = r.gaps.iter().take(6).map(|g| format!("{} edges round, {:.0}% of the model's size", g.edges, 100.0 * g.size)).collect();
            tip += &format!("\ngaps: {}{}", some.join("; "), if r.gaps.len() > 6 { format!("; and {} more", r.gaps.len() - 6) } else { String::new() });
        }
        for (n, what) in [(r.rewound, "had faces the wrong way round, put right"), (r.turned, "were inside out, turned"), (r.hollows, "are hollows inside others"), (r.one_sided, "have one side only")] {
            if n > 0 { tip += &format!("\n{n} piece{} {what}", if n == 1 { "" } else { "s" }); }
        }
        tip += &format!("\nread in {:.0} ms, examined in {:.0} ms, prepared in {:.0} ms", i.read_ms, i.examine_ms, i.prepare_ms);
        crate::ui::Model {
            known: true,
            file: format!("{} · {} triangles · {} piece{}", i.file, right::with_commas(r.triangles), r.pieces, if r.pieces == 1 { "" } else { "s" }),
            line, tip, shell: i.shell, found_shell, own_smooth: r.smooth, facet: r.facet, reading, error, said,
        }
    }

    // ---- what the panels ask for ---------------------------------------------------------------
    fn serve(&mut self, frame: &eframe::Frame) {
        for req in std::mem::take(&mut self.px.requests) {
            match req {
                Request::RenderVideo if self.video.is_none() => {
                    // a scripted video goes to the tests' folder, and is cut short if the script said so
                    let (frames, dir) = match &mut self.test {
                        Some(t) => (t.video_frames.take(), t.out.clone()),
                        None => (None, self.px.store.video_folder()),
                    };
                    self.start_video(frames, dir);
                }
                Request::RenderVideo => {}
                Request::CancelVideo => self.cancel_video(),
                Request::ToFloor(i) => { self.engine.layer_to_floor(i); }
                Request::Benchmark => {
                    self.px.bench = Bench::Asked;
                    self.bench_in = 2;
                }
                Request::OpenFolder => {
                    let dir = self.px.store.video_folder();
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = std::process::Command::new("explorer.exe").arg(&dir).spawn();
                }
                Request::ChooseFolder => {
                    let dir = self.px.store.video_folder();
                    let mut dialog = rfd::FileDialog::new().set_title("Folder for videos").set_parent(frame);
                    if dir.is_dir() { dialog = dialog.set_directory(&dir); }
                    if let Some(chosen) = dialog.pick_folder() {
                        self.px.store.folder = Some(chosen.display().to_string());
                    }
                }
                Request::SaveStill => {
                    let path = self.next_still();
                    self.save_still(&path, false);
                }
                // a script cannot answer a dialog
                Request::SaveSetup | Request::LoadSetup | Request::SaveStillAs if self.test.is_some() => {
                    self.px.file_text = "a scripted run opens no dialogs".into();
                }
                Request::LoadPicture | Request::LoadModel if self.test.is_some() => {}
                Request::LoadModel => {
                    // it opens where the selected layer's file is, if that is a file of the user's; otherwise
                    // among the models that come with the program
                    let st = &self.engine.state;
                    let own = st.model_file(st.stack.sel).filter(|f| !f.trim().is_empty() && st.preset().id == gf_core::presets::models::FILE);
                    let here = own.and_then(|f| crate::hdri::resolve(&f).parent().map(Path::to_path_buf)).filter(|d| d.is_dir());
                    let dir = here.unwrap_or_else(|| crate::hdri::home().join("models"));
                    let dialog = rfd::FileDialog::new().set_title("A model").add_filter("Models (.obj, .stl, .ply, .gltf, .glb)", &gf_core::mesh::read::EXTENSIONS).set_parent(frame);
                    if let Some(path) = in_folder(dialog, &dir).pick_file() { self.load_model(&path, false); }
                }
                Request::LoadSculptPicture(_) if self.test.is_some() => {}
                Request::LoadSculptPicture(i) => {
                    let dialog = rfd::FileDialog::new().set_title("A picture for the surface").add_filter("PNG pictures", &["png"]).set_parent(frame);
                    if let Some(path) = dialog.pick_file() {
                        let st = &mut self.engine.state;
                        if let Some(l) = st.sculpt_mut().layers.get_mut(i) { l.image = path.display().to_string(); }
                        st.sculpt_changed(false);
                    }
                }
                Request::LoadPicture => {
                    // it opens where the picture in use is, or in the folder kept for them beside the program
                    let here = self.engine.hdri_details().and_then(|i| i.path.parent().map(Path::to_path_buf)).filter(|d| d.is_dir());
                    let dir = here.unwrap_or_else(|| crate::hdri::home().join("hdri"));
                    let dialog = rfd::FileDialog::new().set_title("Environment picture").add_filter("Panoramas (.exr, .hdr)", &["exr", "hdr"]).set_parent(frame);
                    if let Some(path) = in_folder(dialog, &dir).pick_file() { self.engine.state.set_hdri(Some(path.display().to_string())); }
                }
                Request::SaveSetup => {
                    let dir = self.px.store.setups_folder();
                    let _ = std::fs::create_dir_all(&dir);
                    let name = format!("glass-{}.json", self.engine.state.preset().id);
                    let dialog = rfd::FileDialog::new().set_title("Save this setup").set_file_name(name).add_filter("Glass Forms setup", &["json"]).set_parent(frame);
                    if let Some(path) = in_folder(dialog, &dir).save_file() { self.save_setup(&with_ending(path, "json")); }
                }
                Request::LoadSetup => {
                    let dialog = rfd::FileDialog::new().set_title("Load a setup").add_filter("Glass Forms setup", &["json"]).set_parent(frame);
                    if let Some(path) = in_folder(dialog, &self.px.store.setups_folder()).pick_file() { self.load_setup(&path); }
                }
                Request::SaveStillAs => {
                    let usual = self.next_still();
                    let dir = self.px.store.stills_folder();
                    let _ = std::fs::create_dir_all(&dir);
                    let dialog = rfd::FileDialog::new().set_title("Save the picture").set_file_name(file_name(&usual)).add_filter("PNG picture", &["png"]).set_parent(frame);
                    if let Some(path) = in_folder(dialog, &dir).save_file() { self.save_still(&with_ending(path, "png"), true); }
                }
            }
        }
    }

    // ---- the mouse and the keys on the picture -------------------------------------------------
    // left drag: turn the shape.   right drag or shift+drag: slide the view.   ctrl+drag or middle drag: roll.
    // wheel: move closer or away.  shift+wheel: change the lens, moving the camera to keep the shape the same size.
    // click (with a lens aperture set): focus there.   double click: put the view back.
    fn gestures(&mut self, ctx: &egui::Context, picture: &egui::Response) {
        let st = &mut self.engine.state;
        let (down, delta, at, mods) = ctx.input(|i| (picture.is_pointer_button_down_on(), i.pointer.delta(), i.pointer.latest_pos(), i.modifiers));
        if down && self.drag.is_none() {
            let (right, middle) = ctx.input(|i| (i.pointer.button_down(PointerButton::Secondary), i.pointer.button_down(PointerButton::Middle)));
            // (a press on one of the handles takes hold of it, whatever keys are held: they mean steps or finely)
            let handle = if right || middle { None } else { at.and_then(|p| self.px.handles.hit(st, self.full, p)) };
            let kind = if let (Some(part), Some(p)) = (handle, at) {
                self.px.handles.begin(st, self.full, p, part);
                Gesture::Handle
            } else if right || mods.shift { Gesture::Pan } else if middle || mods.ctrl { Gesture::Roll } else { Gesture::Turn };
            self.drag = Some(Drag { kind, moved: 0.0, at: at.unwrap_or(self.full.center()) });
        } else if let Some(d) = &mut self.drag {
            if down {
                d.moved += delta.x.abs() + delta.y.abs();
                if let Some(p) = at { d.at = p; }
                let (dx, dy) = (delta.x as f64, delta.y as f64);
                if dx != 0.0 || dy != 0.0 {
                    match d.kind {
                        Gesture::Turn => st.drag_turn(dx, dy),
                        Gesture::Pan => st.drag_pan(dx, dy, self.full.height() as f64),
                        Gesture::Roll => st.drag_roll(dx),
                        Gesture::Handle => self.px.handles.drag(st, self.full, d.at, mods.ctrl, mods.shift),
                    }
                }
            } else {
                // let go: a press that hardly moved is a click, which focuses where it landed (while the
                // handles show, it selects the layer whose shape is there; with Alt held it focuses still)
                let (fx, fy) = ((d.at.x / self.full.width()) as f64, (d.at.y / self.full.height()) as f64);
                if d.kind == Gesture::Handle {
                    self.px.handles.end(st);
                } else if d.kind == Gesture::Turn && d.moved < 4.0 && self.px.handles.on && !mods.alt {
                    // a lamp under the pointer first (it is known exactly), then a layer's shape
                    if let Some(i) = crate::ui::handles::pick_lamp(&self.engine.state, self.full, d.at) {
                        self.px.handles.lamp = Some(i);
                        self.px.lamp = i;
                    } else if let Some(i) = crate::ui::handles::pick_layer(&self.engine, fx, fy) {
                        self.px.handles.lamp = None;
                        if i != self.engine.state.stack.sel { self.engine.state.select_layer(i); }
                    }
                } else if d.kind == Gesture::Turn && d.moved < 4.0 && st.p.aperture > 0.0 {
                    self.engine.pick_focus(fx, fy);
                }
                self.drag = None;
            }
        }
        let st = &mut self.engine.state;
        if picture.double_clicked() { st.reset_view(); }
        if let Some(d) = &self.drag {
            ctx.set_cursor_icon(if d.kind == Gesture::Pan { CursorIcon::Move } else { CursorIcon::Grabbing });
        } else if picture.hovered() && self.px.handles.hover.is_some() {
            ctx.set_cursor_icon(CursorIcon::PointingHand);
        } else if picture.hovered() {
            ctx.set_cursor_icon(CursorIcon::Grab);
        }
        if picture.contains_pointer() {
            for ev in ctx.input(|i| i.events.clone()) {
                if let egui::Event::MouseWheel { unit, delta, modifiers, .. } = ev {
                    // with shift held some systems report the wheel sideways
                    let turned = if delta.y != 0.0 { delta.y } else { delta.x };
                    // the browser's units: about 100 for one notch, positive toward the user
                    let d = -turned as f64 * match unit {
                        egui::MouseWheelUnit::Line => 100.0,
                        egui::MouseWheelUnit::Page => 800.0,
                        egui::MouseWheelUnit::Point => 1.0,
                    };
                    st.wheel(d, modifiers.shift);
                }
            }
        }
    }

    // space: spin.  P: animate.  H: hide the panels.  L: the light room.  F: fullscreen.  R: reset the view.
    // C: clay or glass.  F1: the help.  1-9: family.  arrows: step through the shapes.
    fn keys(&mut self, ctx: &egui::Context) {
        if self.video.is_some() || ctx.egui_wants_keyboard_input() || egui::Popup::is_any_open(ctx) { return; }
        let st = &mut self.engine.state;
        for ev in ctx.input(|i| i.events.clone()) {
            let egui::Event::Key { key, pressed: true, repeat, modifiers, .. } = ev else { continue };
            if modifiers.ctrl || modifiers.alt || modifiers.command { continue; }
            use egui::Key::*;
            match key {
                ArrowRight | ArrowDown => st.step_preset(1),
                ArrowLeft | ArrowUp => st.step_preset(-1),
                _ if repeat => {}
                Space => st.toggle_spin(),
                P => st.set_playing(!st.playing),
                H => {
                    // hidden if either is showing, shown again only when both are hidden
                    let hide = !(self.px.hide_left && self.px.hide_right);
                    (self.px.hide_left, self.px.hide_right) = (hide, hide);
                }
                L => self.px.store.room_open = !self.px.store.room_open,
                // the material and the camera: their sections opened or folded
                M => self.px.store.fold("material"),
                K => self.px.store.fold("camera"),
                C => st.p.clay = if st.p.clay > 0.5 { 0.0 } else { 1.0 },
                F1 => self.px.help_open = !self.px.help_open,
                R => st.reset_view(),
                F => {
                    let full = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!full));
                }
                // G: the handles over the picture; T: what they do (move, turn, size)
                G => self.px.handles.toggle(),
                T => {
                    if self.px.handles.on { self.px.handles.mode = self.px.handles.mode.next(); } else { self.px.handles.on = true; }
                }
                // Esc puts away the nearest thing: a drag of a handle (the layer goes back), the help if it
                // is open, the handles, otherwise the fullscreen
                Escape if self.px.handles.cancel(st) => {}
                Escape if self.px.help_open => self.px.help_open = false,
                Escape if self.px.handles.on => self.px.handles.toggle(),
                Escape => {
                    if ctx.input(|i| i.viewport().fullscreen.unwrap_or(false)) {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                    }
                }
                Num1 | Num2 | Num3 | Num4 | Num5 | Num6 | Num7 | Num8 | Num9 => {
                    let n = [Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9].iter().position(|k| *k == key).unwrap_or(0);
                    if n < st.families.len() { st.set_scene(n); }
                }
                _ => {}
            }
        }
    }

    // ---- the panels ----------------------------------------------------------------------------
    fn panels(&mut self, ctx: &egui::Context) {
        let st = &self.engine.state;
        self.px.bright = st.hdri.is_some() && st.p.hdri_show > 0.5 && self.engine.hdri_info().is_some();
        let frame = egui::Frame::new()
            .fill(theme::panel_fill(self.px.bright))
            .stroke(theme::panel_stroke())
            .corner_radius(theme::PANEL_RADIUS)
            .inner_margin(egui::Margin { left: 12, right: 2, top: 10, bottom: 9 });
        // a panel taller than the window scrolls; its scroll bar sits in the margin at its right
        // (as far from the foot of the window as from its top: the box around it adds 21 points to its height)
        let max_h = (self.full.height() - 2.0 * EDGE - 21.0).max(60.0);
        // (returns how tall the contents turned out to be)
        let scroller = |ui: &mut egui::Ui, width: f32, to: Option<f32>, salt: &str, body: &mut dyn FnMut(&mut egui::Ui)| -> f32 {
            let space = Rect::from_min_size(ui.cursor().min, egui::vec2(width + 10.0, max_h));
            ui.scope_builder(egui::UiBuilder::new().max_rect(space), |ui| {
                let mut area = egui::ScrollArea::vertical().id_salt(salt).max_height(max_h).auto_shrink([false, true]);
                if let Some(y) = to { area = area.vertical_scroll_offset(y); }
                area.show(ui, |ui| {
                    ui.set_width(width);
                    body(ui);
                }).content_size.y
            }).inner
        };
        self.px.rects.clear();
        widgets::forget_menu(ctx);
        // what the light room shows of the engine's doing (the picture of the lighting is only asked for, and
        // so only drawn, while the room is there to show it)
        let room = self.px.store.room_open && !(self.px.hide_left && self.px.hide_right);
        self.px.lighting = crate::ui::Lighting {
            preview: if room { self.preview_texture() } else { None },
            picture: self.engine.hdri_info(),
            error: self.engine.hdri_error.clone(),
            loading: self.engine.hdri_loading(),
        };
        self.px.sculpted = (self.engine.sculptor.last.res > 0).then(|| (self.engine.sculptor.last.res, self.engine.sculptor.last.ms));
        self.px.sculpt_error = self.engine.sculptor.image_error.clone().or_else(|| (!self.engine.sculptor.available()).then(|| "the surface studio needs a graphics device that can write volumes by compute shader".to_owned()));
        self.px.joined = if self.engine.layers.combined { self.engine.scene_volume().map(|v| (v.res, self.engine.layers.combine_ms)) } else { None };
        self.px.model = self.model_told();
        let cast = &self.engine.cast;
        self.px.casting = crate::ui::Casting {
            on: cast.on, receivers: self.engine.cast_receivers(), focusing: cast.focusing, lights: cast.lights,
            casters: cast.casters.iter().take(self.engine.state.bodies().len()).map(|c| (c.shadow, c.caustic)).collect(),
            caustics: cast.focusing.then(|| cast.caustics.size()), floor: self.engine.floor_height(),
        };
        let ((cw, ch), _) = self.engine.renderer.size();
        if cw > 0 && ch > 0 { self.px.aspect = cw as f64 / ch as f64; }
        // a model that arrived as one more layer: its join with the layer it was added beside is opened, as
        // when a layer is added with the "+"
        if let Some((beside, were)) = self.adding {
            let st = &self.engine.state;
            if st.layer_count() > were {
                self.px.combo_open = Some(gf_core::layers::mask_of(&[beside, st.stack.sel]));
                if self.px.store.is_closed("joins") { self.px.store.fold("joins"); }
                self.adding = None;
            } else if !self.engine.model_loading() {
                self.adding = None;
            }
        }
        let (st, px) = (&mut self.engine.state, &mut self.px);
        px.room_h = max_h;
        (px.left_at, px.right_at) = (None, None);
        if !px.hide_left {
            let to = px.scroll_to[0].take();
            let shown = egui::Area::new(Id::unique("left panel")).order(egui::Order::Middle).fixed_pos(Pos2::new(EDGE, EDGE)).constrain(false).fade_in(false).show(ctx, |ui| {
                px.place = "left";
                frame.show(ui, |ui| scroller(ui, LEFT_W, to, "left", &mut |ui| left::show(ui, st, px))).inner
            });
            px.needs[0] = shown.inner;
            px.left_at = Some(shown.response.rect);
            px.note("left", shown.response.rect);
        }
        if !px.hide_right {
            let to = px.scroll_to[1].take();
            let shown = egui::Area::new(Id::unique("right panel")).order(egui::Order::Middle).anchor(Align2::RIGHT_TOP, egui::vec2(-EDGE, EDGE)).constrain(false).fade_in(false).show(ctx, |ui| {
                px.place = "right";
                frame.show(ui, |ui| scroller(ui, RIGHT_W, to, "right", &mut |ui| right::show(ui, st, px))).inner
            });
            px.needs[1] = shown.inner;
            px.right_at = Some(shown.response.rect);
            px.note("right", shown.response.rect);
        }
        // the windows that float over both: hidden with the panels, but not closed
        if !(px.hide_left && px.hide_right) {
            room::show(ctx, st, px, self.full);
            crate::ui::sculpt::show(ctx, st, px, self.full);
            overlay::help(ctx, px, st.families.len(), self.full);
        }
        // something went wrong building the shape: say so over the picture (the shape before stays up).
        // A model's file that could not be used is said there too, for a while, when the shape that is up
        // is not a model: its panel, where that is otherwise said and stays, is not there to say it.
        let refused = self.model_refused.as_ref().filter(|(_, at)| !self.engine.state.is_model() && at.elapsed() < REFUSED_FOR).map(|(why, at)| {
            ctx.request_repaint_after(REFUSED_FOR.saturating_sub(at.elapsed()) + Duration::from_millis(30));
            why.clone()
        });
        if let Some(err) = self.engine.error.as_ref().or(refused.as_ref()) {
            let width = (self.full.width() - LEFT_W - RIGHT_W - 150.0).clamp(240.0, 640.0);
            egui::Area::new(Id::unique("error")).order(egui::Order::Foreground).anchor(Align2::CENTER_TOP, egui::vec2(0.0, EDGE)).interactable(false).fade_in(false).show(ctx, |ui| {
                egui::Frame::new()
                    .fill(theme::ERROR_FILL.gamma_multiply(0.94))
                    .stroke(egui::Stroke::new(1.0, theme::ERROR_TEXT.gamma_multiply(0.45)))
                    .corner_radius(theme::PANEL_RADIUS)
                    .inner_margin(egui::Margin::symmetric(16, 12))
                    .show(ui, |ui| {
                        ui.set_width(width);
                        widgets::paragraph(ui, err, theme::mono(12.0), theme::ERROR_TEXT, theme::LINE, 0.0);
                    });
            });
        }
    }

    // ---- the picture ---------------------------------------------------------------------------
    /// Tell egui about the engine's picture if it is not the one egui has (the engine makes a new one
    /// whenever its size changes).
    fn texture(&mut self, smooth: bool) -> Option<egui::TextureId> {
        let screen = self.engine.renderer.screen()?;
        if let Some(s) = &self.shown {
            if s.texture == *screen && s.smooth == smooth { return Some(s.id); }
        }
        // The picture's bytes are ready for the display as they stand. egui reads textures without
        // converting them and writes its own colours to the window the same way, so the plain view of
        // the texture (not the sRGB one) is the view that carries the bytes through unchanged.
        let view = screen.create_view(&wgpu::TextureViewDescriptor::default());
        let filter = if smooth { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
        let mut renderer = self.rs.renderer.write();
        let id = match &self.shown {
            Some(s) => {
                renderer.update_egui_texture_from_wgpu_texture(&self.rs.device, &view, filter, s.id);
                s.id
            }
            None => renderer.register_native_texture(&self.rs.device, &view, filter),
        };
        self.shown = Some(Shown { texture: screen.clone(), id, smooth });
        Some(id)
    }

    /// Tell egui about the engine's picture of the lighting, if there is one yet.
    fn preview_texture(&mut self) -> Option<egui::TextureId> {
        let picture = self.engine.env_preview()?;
        if let Some(s) = &self.preview {
            if s.texture == *picture { return Some(s.id); }
        }
        // (its bytes are ready for the display as they stand, like the screen's: the plain view carries them through)
        let view = picture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut renderer = self.rs.renderer.write();
        let id = match &self.preview {
            Some(s) => {
                renderer.update_egui_texture_from_wgpu_texture(&self.rs.device, &view, wgpu::FilterMode::Linear, s.id);
                s.id
            }
            None => renderer.register_native_texture(&self.rs.device, &view, wgpu::FilterMode::Linear),
        };
        self.preview = Some(Shown { texture: picture.clone(), id, smooth: true });
        Some(id)
    }

    /// Bring the line at the foot of the window up to date: the size the picture is traced at; frames a
    /// second while it moves, or how far it has refined while it is still; how long the shape took to build.
    /// True if it now says something else.
    fn perf(&mut self) -> bool {
        let n = self.engine.renderer.accum_n;
        let (_, (tw, th)) = self.engine.renderer.size();
        let moving = n as f64 <= (1.0 / (1.0 - self.engine.state.p.smooth)).round() + 1.0;
        let pace = if self.video.is_some() {
            // a video is being made: how far along it is
            format!("video {}", self.px.video_text.lines().next().unwrap_or_default())
        } else if moving && self.fps > 0.0 {
            format!("{:.0} fps", self.fps)
        } else if n >= self.engine.max_accum() {
            "refined".to_owned()
        } else {
            format!("refining {n}/{}", self.engine.max_accum())
        };
        // how long the selected layer's shape took to build, and with several layers the joining of them
        let ms = self.engine.state.shape().meta.ms;
        let mut text = format!("{tw}×{th} · {pace} · shape {ms:.*} ms", if ms < 9.95 { 1 } else { 0 });
        if let Some((_, join)) = self.px.joined { text += &format!(" · join {join:.*} ms", if join < 9.95 { 1 } else { 0 }); }
        // a model's file being read (it may take seconds, and the picture carries on as it was meanwhile)
        if let Some(name) = &self.px.model.reading { text += &format!(" · reading {name}…"); }
        let changed = text != self.px.perf;
        self.px.perf = text;
        changed
    }

    // ---- remembering ---------------------------------------------------------------------------
    fn remember(&mut self, ctx: &egui::Context) {
        let Some(path) = self.store_path.clone() else { return };
        if self.px.store.video != self.engine.state.render { self.px.store.video = self.engine.state.render.clone(); }
        // the window's size, unless it was fixed from the command line or is not its ordinary self just now
        if self.opts.size_px.is_none() && self.opts.scale.is_none() {
            let (rect, max, full, min) = ctx.input(|i| {
                let v = i.viewport();
                (v.inner_rect, v.maximized.unwrap_or(false), v.fullscreen.unwrap_or(false), v.minimized.unwrap_or(false))
            });
            if !full && !min {
                self.px.store.maximized = max;
                if let (false, Some(r)) = (max, rect) {
                    if r.width() > 100.0 && r.height() > 100.0 { self.px.store.window = Some([r.width().round(), r.height().round()]); }
                }
            }
        }
        if self.px.store == self.saved {
            self.changed_at = None;
            return;
        }
        // written once things have stopped changing, not on every step of a window being dragged larger
        let since = *self.changed_at.get_or_insert_with(Instant::now);
        if since.elapsed() > Duration::from_millis(400) && !ctx.input(|i| i.pointer.any_down()) {
            if let Err(e) = self.px.store.save(&path) { eprintln!("glassforms: {e}"); }
            self.saved = self.px.store.clone();
            self.changed_at = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(450));
        }
    }

    // ---- scripted runs -------------------------------------------------------------------------
    /// The window has the size and the scaling the command line asked for (or none was asked for).
    fn sized(&self, ctx: &egui::Context) -> bool {
        self.seeking.is_none() && self.opts.scale.is_none_or(|s| (ctx.pixels_per_point() - s).abs() < 1e-3)
    }

    /// Carry out the script as far as it goes this frame.
    fn script(&mut self, ctx: &egui::Context) {
        // nothing is done until the window has settled at its size (or plainly never will)
        if self.test.is_some() && !self.sized(ctx) { return; }
        loop {
            let sized = self.sized(ctx);
            let Some(t) = &mut self.test else { return };
            if t.waiting > 0 {
                t.waiting -= 1;
                return;
            }
            let Some(step) = t.steps.pop_front() else { return };
            match step {
                Step::Do(act) => {
                    if let Err(e) = self.act(ctx, act) { return self.fail(ctx, e); }
                    return;
                }
                Step::Input(events, held) => {
                    self.inject.extend(events);
                    t.held = held;
                    return;
                }
                Step::Dropped(file, held) => {
                    self.dropped.push(file);
                    t.held = held;
                    return;
                }
                Step::Frames(n) => t.waiting = n,
                Step::VideoDone => {
                    if self.video.is_some() || self.px.requests.contains(&Request::RenderVideo) {
                        t.steps.push_front(Step::VideoDone);
                        return;
                    }
                }
                Step::Lit => {
                    if self.engine.lighting_pending() || self.engine.model_loading() {
                        t.steps.push_front(Step::Lit);
                        return;
                    }
                }
                Step::Settle(n) => {
                    if !sized || self.opts.size_px.is_some_and(|s| s != self.canvas) {
                        let (want, have) = (self.opts.size_px.unwrap_or(self.canvas), self.canvas);
                        return self.fail(ctx, format!("the window could not be made {} x {} pixels (it is {} x {})", want.0, want.1, have.0, have.1));
                    }
                    let st = &self.engine.state;
                    let moving = st.playing || st.p.spin > 0.0 || st.p.lspin > 0.0;
                    t.settled += 1;
                    let had = if moving { t.settled } else { self.engine.renderer.accum_n };
                    // (clay settles after fewer frames than glass; a picture or a model being read and a tracer
                    // being built are waited for, so that what is photographed is what was asked for)
                    if had < n.min(self.engine.max_accum()) || self.video.is_some() || self.engine.lighting_pending() || self.engine.model_loading() {
                        t.steps.push_front(Step::Settle(n));
                        return;
                    }
                    // hold the picture exactly as it is while it is photographed and described
                    self.frozen = true;
                    self.perf();
                }
                Step::Shot(path, asked) => {
                    if !asked {
                        let slot = t.picture.clone();
                        let again = ctx.clone();
                        ctx.request_screenshot(move |image| {
                            if let Ok(mut s) = slot.lock() { *s = Some(image); }
                            again.request_repaint();
                        });
                        t.steps.push_front(Step::Shot(path, true));
                        return;
                    }
                    let got = t.picture.lock().ok().and_then(|mut s| s.take());
                    match got {
                        Some(image) => {
                            if let Err(e) = save_png(&path, &image) { return self.fail(ctx, e); }
                        }
                        None => {
                            t.steps.push_front(Step::Shot(path, true));
                            return;
                        }
                    }
                }
                Step::Idle(seconds, started) => {
                    if !started {
                        // wait for the picture to finish refining, and for the panels to have caught up
                        if !self.rested || self.video.is_some() {
                            t.steps.push_front(Step::Idle(seconds, false));
                            return;
                        }
                        let (frames, found, wake) = (self.frames.clone(), t.idle.clone(), ctx.clone());
                        std::thread::spawn(move || {
                            std::thread::sleep(Duration::from_millis(700));
                            let before = frames.load(Ordering::SeqCst);
                            std::thread::sleep(Duration::from_secs_f32(seconds));
                            let drawn = frames.load(Ordering::SeqCst) - before;
                            if let Ok(mut f) = found.lock() { *f = Some((seconds, drawn)); }
                            wake.request_repaint();
                        });
                        t.steps.push_front(Step::Idle(seconds, true));
                        return;
                    }
                    match t.idle.lock().ok().and_then(|mut f| f.take()) {
                        Some(found) => t.idle_found = Some(found),
                        None => {
                            t.steps.push_front(Step::Idle(seconds, true));
                            return;
                        }
                    }
                }
                Step::Dump(path) => {
                    let text = serde_json::to_string_pretty(&self.describe(ctx)).unwrap_or_default();
                    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) { let _ = std::fs::create_dir_all(dir); }
                    if let Err(e) = std::fs::write(&path, text) { return self.fail(ctx, format!("{}: {e}", path.display())); }
                }
                Step::Leave => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    return;
                }
            }
        }
    }

    /// True while the script is counting idle frames: nothing may ask for a frame then.
    fn counting_idle(&self) -> bool {
        self.test.as_ref().is_some_and(|t| matches!(t.steps.front(), Some(Step::Idle(_, true))))
    }

    /// One action of a script.
    fn act(&mut self, ctx: &egui::Context, act: Act) -> Result<(), String> {
        let st = &mut self.engine.state;
        let centre = self.full.center();
        let button = |pos: Pos2, button: PointerButton, pressed: bool, modifiers: egui::Modifiers| egui::Event::PointerButton { pos, button, pressed, modifiers };
        let mut steps: Vec<Step> = Vec::new();
        let none = egui::Modifiers::NONE;
        match act {
            Act::Wave(name) => {
                let k = script::find_knob(st, &name)?;
                if !k.wave { return Err(format!("wave: \"{name}\" cannot be animated")); }
                st.toggle_wave(&k);
            }
            Act::Set(name, v) => {
                let k = script::find_knob(st, &name)?;
                st.user_set(&k, v, true);
                st.after_user(&k, false);
            }
            Act::Stops(name, from, to) => {
                let k = script::find_knob(st, &name)?;
                if !st.tracks.contains_key(&k.key) { return Err(format!("stops: \"{name}\" has no track (light its wave first)")); }
                knob::set_stops(st, &k, from, to);
                st.after_user(&k, false);
            }
            Act::Every(name, seconds) => {
                let k = script::find_knob(st, &name)?;
                knob::set_period(st, &k, seconds);
            }
            Act::Ease(name, ease, at) => {
                let k = script::find_knob(st, &name)?;
                let was = st.tracks.get(&k.key).map(|t| t.at).ok_or_else(|| format!("ease: \"{name}\" has no track (light its wave first)"))?;
                knob::set_ease(st, &k, ease, at.unwrap_or(was));
            }
            Act::Setting(name, v) => {
                let keeps = st.r;
                if !st.p.set(&name, v) { return Err(format!("p: there is no setting called \"{name}\"")); }
                match name.as_str() {
                    "ease" | "easeAt" => st.rephase_all(),
                    "spinMode" => st.set_r(&keeps),
                    _ => {}
                }
            }
            Act::Fold(group) => {
                // the groups of the panels and the light room, and what folds of the material: the chooser of
                // named materials and the layers of the lab
                let lab = material::folds();
                if !store::GROUPS.contains(&group.as_str()) && !lab.contains(&group) {
                    return Err(format!("fold: there is no group called \"{group}\" (there are: {}, {})", store::GROUPS.join(", "), lab.join(", ")));
                }
                self.px.store.fold(&group);
            }
            Act::Note => {
                self.perf();
                let st = &self.engine.state;
                let sliders: serde_json::Map<String, serde_json::Value> =
                    st.shape_knobs().into_iter().map(|k| (k.id.clone(), serde_json::json!([k.lo, k.hi, k.step, st.get(&k.r)]))).collect();
                let note = serde_json::json!({
                    "perf": self.px.perf, "family": st.family().id, "preset": st.preset().id, "error": self.engine.error,
                    "built": st.shape().meta.res, "sliders": sliders, "material": material::id(st), "needsFile": st.needs_file(st.stack.sel),
                });
                if let Some(t) = &mut self.test { t.noted.push(note); }
            }
            Act::Room(open) => self.px.store.room_open = open,
            Act::Sculpt(open) => self.px.store.sculpt_open = open,
            // (the material and the camera are sections of the left panel now: these open or fold them)
            Act::Lab(open) => {
                if open == self.px.store.is_closed("material") { self.px.store.fold("material"); }
                // (open, as the lab was: every part of the material in view, its sub-groups open too)
                if open {
                    for g in ["lab-in it", "lab-special optics", "lab-casts"] {
                        if self.px.store.is_closed(g) { self.px.store.fold(g); }
                    }
                }
            }
            Act::Camera(open) => { if open == self.px.store.is_closed("camera") { self.px.store.fold("camera"); } }
            Act::Light(n) => {
                if n > st.rig().lights.len() { return Err(format!("light: there are only {} lights", st.rig().lights.len())); }
                self.px.light = n - 1;
            }
            Act::SculptSet(json) => {
                *st.sculpt_mut() = gf_core::sculpt::Sculpt::from_arg(&json).ok_or_else(|| format!("sculpt-set: \"{json}\" is not a surface studio"))?;
                st.sculpt_changed(false);
            }
            Act::RemoveLayer(i) => {
                if i >= st.layer_count() || st.layer_count() < 2 { return Err(format!("remove-layer: there is no layer {} to take away", i + 1)); }
                st.remove_layer(i);
                self.px.combo_open = None;
            }
            Act::SelectLayer(i) => {
                if i >= st.layer_count() { return Err(format!("select-layer: there is no layer {}", i + 1)); }
                st.select_layer(i);
            }
            Act::AddLayer => {
                if st.layer_count() >= gf_core::layers::MAX_LAYERS { return Err("add-layer: there are four layers already".into()); }
                crate::ui::layers::add(st, &mut self.px);
            }
            Act::Join(tag) => {
                let mask = if tag.is_empty() { None } else { Some(gf_core::layers::parse_tag(&tag).filter(|m| st.stack.combos().contains(m)).ok_or_else(|| format!("join: there is no combination \"{tag}\""))?) };
                self.px.combo_open = mask;
            }
            Act::Arg(name, value) => {
                if NOT_SETTINGS.contains(&name.as_str()) || !st.apply_arg(&name, &value) { return Err(format!("arg: \"{name}={value}\" means nothing here")); }
            }
            // (the picture is read in the background: what follows in the script finds it in use, or refused)
            Act::Hdri(file) => {
                st.set_hdri((!file.is_empty()).then_some(file));
                steps = vec![Step::Frames(1), Step::Lit, Step::Frames(2)];
            }
            Act::Ready => steps = vec![Step::Frames(1), Step::Lit, Step::Frames(2)],
            // (the file is read in the background: what follows in the script finds it up, or refused)
            Act::Drop(file, shift) => {
                let held = if shift { egui::Modifiers::SHIFT } else { none };
                steps = vec![Step::Dropped(PathBuf::from(file), held), Step::Input(Vec::new(), none), Step::Frames(2), Step::Lit, Step::Frames(2)];
            }
            Act::Help(open) => self.px.help_open = open,
            Act::Family(id) => {
                let i = st.families.iter().position(|f| f.id == id).ok_or_else(|| format!("family: there is no family called \"{id}\""))?;
                st.set_scene(i);
            }
            Act::Preset(id) => {
                let i = st.family().presets.iter().position(|p| p.id == id).ok_or_else(|| format!("preset: {} has no shape called \"{id}\"", st.family().name))?;
                st.choose(i);
            }
            Act::Detail(n) => st.set_detail(n),
            Act::Motion(n) => st.set_anim_detail(n),
            Act::Play => st.set_playing(!st.playing),
            Act::View(name) => {
                let (rx, ry) = match name.as_str() {
                    "front" => (0.0, 0.0),
                    "back" => (0.0, 180.0),
                    "left" => (0.0, 90.0),
                    "right" => (0.0, -90.0),
                    "top" => (90.0, 0.0),
                    "under" => (-90.0, 0.0),
                    _ => return Err(format!("view: there is no view called \"{name}\"")),
                };
                st.set_r(&euler(rx, ry, 0.0));
            }
            Act::Reset(what) => match what.as_str() {
                "view" => st.reset_view(),
                "object" => st.reset_orientation(),
                "lights" => st.reset_lights(),
                _ => return Err(format!("reset: \"{what}\" is not view, object or lights")),
            },
            Act::Key(name) => {
                let key = script::key(&name)?;
                let press = |pressed: bool| egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: none };
                steps = vec![Step::Input(vec![press(true)], none), Step::Input(vec![press(false)], none), Step::Frames(1)];
            }
            Act::Drag(dx, dy, how) => {
                let (which, held) = match how {
                    Hold::Left => (PointerButton::Primary, none),
                    Hold::Right => (PointerButton::Secondary, none),
                    Hold::Middle => (PointerButton::Middle, none),
                    Hold::Shift => (PointerButton::Primary, egui::Modifiers::SHIFT),
                    Hold::Ctrl => (PointerButton::Primary, egui::Modifiers::CTRL),
                };
                steps.push(Step::Input(vec![egui::Event::PointerMoved(centre)], held));
                steps.push(Step::Input(vec![button(centre, which, true, held)], held));
                let n = 5;
                for i in 1..=n {
                    steps.push(Step::Input(vec![egui::Event::PointerMoved(centre + egui::vec2(dx, dy) * (i as f32 / n as f32))], held));
                }
                steps.push(Step::Input(vec![button(centre + egui::vec2(dx, dy), which, false, held)], held));
                steps.push(Step::Input(Vec::new(), none));
                steps.push(Step::Frames(1));
            }
            Act::Wheel(d, shift, place) => {
                let held = if shift { egui::Modifiers::SHIFT } else { none };
                let centre = place.map_or(centre, |(x, y)| Pos2::new(x, y));
                let wheel = egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line, delta: egui::vec2(0.0, -d / 100.0), phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel, modifiers: held,
                };
                steps = vec![Step::Input(vec![egui::Event::PointerMoved(centre)], held), Step::Input(vec![wheel], held), Step::Input(Vec::new(), none), Step::Frames(1)];
            }
            Act::Type(text) => steps = vec![Step::Input(vec![egui::Event::Text(text)], none), Step::Frames(1)],
            Act::Click(fx, fy, twice) => {
                let at = Pos2::new(fx * self.full.width(), fy * self.full.height());
                steps.push(Step::Input(vec![egui::Event::PointerMoved(at)], none));
                for _ in 0..if twice { 2 } else { 1 } {
                    steps.push(Step::Input(vec![button(at, PointerButton::Primary, true, none)], none));
                    steps.push(Step::Input(vec![button(at, PointerButton::Primary, false, none)], none));
                }
                steps.push(Step::Frames(2));
            }
            Act::Ball(dx, dy) => match self.px.rects.get("ball").filter(|r| self.full.contains_rect(**r)) {
                // the ball is in view: take hold of it as a mouse would
                Some(ball) => {
                    let at = ball.center();
                    steps.push(Step::Input(vec![egui::Event::PointerMoved(at)], none));
                    steps.push(Step::Input(vec![button(at, PointerButton::Primary, true, none)], none));
                    for i in 1..=4 {
                        steps.push(Step::Input(vec![egui::Event::PointerMoved(at + egui::vec2(dx, dy) * (i as f32 / 4.0))], none));
                    }
                    steps.push(Step::Input(vec![button(at + egui::vec2(dx, dy), PointerButton::Primary, false, none)], none));
                    steps.push(Step::Frames(1));
                }
                None => gizmo::drag(st, dx as f64, dy as f64),
            },
            Act::PointerDown(x, y, shift) => {
                let (at, held) = (Pos2::new(x, y), if shift { egui::Modifiers::SHIFT } else { none });
                steps = vec![Step::Input(vec![egui::Event::PointerMoved(at)], held), Step::Input(vec![button(at, PointerButton::Primary, true, held)], held)];
            }
            // (whatever key the press was made with stays held until the pointer is let go)
            Act::PointerMove(x, y) => {
                let held = self.test.as_ref().map_or(none, |t| t.held);
                steps = vec![Step::Input(vec![egui::Event::PointerMoved(Pos2::new(x, y))], held)];
            }
            Act::PointerUp => {
                let at = ctx.input(|i| i.pointer.latest_pos()).unwrap_or(centre);
                let held = self.test.as_ref().map_or(none, |t| t.held);
                steps = vec![Step::Input(vec![button(at, PointerButton::Primary, false, held)], held), Step::Input(Vec::new(), none), Step::Frames(1)];
            }
            Act::VideoOpt(name, v) => {
                let n = || v.parse::<u32>().map_err(|_| format!("vopt: \"{v}\" is not a number"));
                match name.as_str() {
                    "size" => st.render.size = v,
                    "fps" => st.render.fps = n()?,
                    "samples" => st.render.samples = n()?,
                    "lens" => st.render.lens = n()?,
                    "codec" => st.render.codec = Codec::from_id(&v).ok_or_else(|| format!("vopt: there is no codec called \"{v}\""))?.id().into(),
                    "quality" => st.render.crf = n()?,
                    "shutter" => st.render.shutter = v.parse::<f64>().map_err(|_| format!("vopt: \"{v}\" is not a number"))?,
                    _ => return Err(format!("vopt: there is no video choice called \"{name}\"")),
                }
            }
            Act::Video(frames, wait) => {
                if !self.px.ffmpeg { return Err("video: ffmpeg was not found".into()); }
                if let Some(t) = &mut self.test { t.video_frames = Some(frames.max(1)); }
                self.px.requests.push(Request::RenderVideo);
                if wait { steps = vec![Step::VideoDone, Step::Frames(1)]; }
            }
            Act::Cancel => self.px.requests.push(Request::CancelVideo),
            Act::Bench => {
                self.px.requests.push(Request::Benchmark);
                steps = vec![Step::Frames(4)];
            }
            Act::Hide(which) => {
                (self.px.hide_left, self.px.hide_right) = match which.as_str() {
                    "left" => (true, false),
                    "right" => (false, true),
                    "both" => (true, true),
                    "none" => (false, false),
                    _ => return Err(format!("hide: \"{which}\" is not left, right, both or none")),
                };
            }
            // (no further than the panel's contents go: a large number means "to the foot")
            Act::Scroll(which, y) => match which.as_str() {
                "left" => self.px.scroll_to[0] = Some(y.clamp(0.0, (self.px.needs[0] - self.px.room_h).max(0.0))),
                "right" => self.px.scroll_to[1] = Some(y.clamp(0.0, (self.px.needs[1] - self.px.room_h).max(0.0))),
                _ => return Err(format!("scroll: \"{which}\" is not left or right")),
            },
            Act::Wait(n) => steps = vec![Step::Frames(n)],
            Act::Shot(path) => steps = vec![Step::Shot(PathBuf::from(path), false)],
            Act::Picture(path) => {
                let (w, h, rgba) = self.engine.screenshot()?;
                save_png(Path::new(&path), &egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba))?;
            }
            Act::Error(words) => self.engine.error = Some(words),
            Act::Hold => self.frozen = true,
            Act::SaveSetup(path) => {
                self.save_setup(Path::new(&path));
                if !self.px.file_text.starts_with("saved") { return Err(self.px.file_text.clone()); }
            }
            Act::LoadSetup(path) => self.load_setup(Path::new(&path)),
            Act::Still(path, bits) => {
                if let Some(bits) = bits { self.px.store.still_16 = bits == 16; }
                let saved = if path.is_empty() {
                    let usual = self.next_still();
                    self.save_still(&usual, false)
                } else {
                    self.save_still(Path::new(&path), true)
                };
                if !saved { return Err(self.px.file_text.clone()); }
            }
            Act::StillBits(bits) => self.px.store.still_16 = bits == 16,
            Act::Scrub(seconds) => {
                animate::scrub(st, seconds);
                animate::scrub_done(st);
            }
            Act::Idle(seconds) => steps = vec![Step::Idle(seconds, false)],
        }
        if let Some(t) = &mut self.test {
            for s in steps.into_iter().rev() { t.steps.push_front(s); }
        }
        Ok(())
    }

    /// Everything a test might want to check, as it stands.
    fn describe(&self, ctx: &egui::Context) -> serde_json::Value {
        use serde_json::json;
        let st = &self.engine.state;
        let knobs: serde_json::Map<String, serde_json::Value> =
            st.knob_list().into_iter().chain(State::orient_knobs()).map(|k| (k.key.clone(), json!(st.get(&k.r)))).collect();
        let ranges: serde_json::Map<String, serde_json::Value> =
            st.knob_list().into_iter().chain(State::orient_knobs()).map(|k| (k.key.clone(), json!([k.lo, k.hi, k.step]))).collect();
        let tracks: serde_json::Map<String, serde_json::Value> = st.tracks.iter().map(|(k, t)| (k.clone(), json!(t))).collect();
        let rects: serde_json::Map<String, serde_json::Value> =
            self.px.rects.iter().map(|(k, r)| (k.clone(), json!([r.min.x, r.min.y, r.max.x, r.max.y]))).collect();
        let (sh, m) = (st.shape(), &st.shape().meta);
        let ((cw, ch), (tw, th)) = self.engine.renderer.size();
        let folder = self.px.store.video_folder();
        let mut told = json!({
            "family": st.family().id,
            "preset": st.preset().id,
            "playing": st.playing,
            "clock": st.clock,
            "knobs": knobs,
            "ranges": ranges,
            "tracks": tracks,
            "settings": st.p,
            "orientation": { "r": st.r, "r0": st.r0, "theta": st.theta, "euler": gf_core::math::euler_of(&st.r) },
            "lights": { "yaw": st.p.lyaw, "pitch": st.p.lpitch, "roll": st.p.lroll, "turned": st.lphi, "fixedTo": st.p.lights, "matrix": st.light_m() },
            "shape": {
                "values": sh.vals, "thick": sh.thick, "len": sh.len, "melt": sh.melt, "puff": sh.puff, "travel": sh.travel, "beads": sh.beads,
                "beadPeriod": sh.bead_period, "res": sh.res, "animRes": sh.anim_res,
                "built": { "res": m.res, "ms": m.ms, "segments": m.segments, "info": m.info },
            },
            "says": { "file": self.px.file_text, "time": format!("{:.2}", st.clock.rem_euclid(st.loop_plan().l)), "perf": self.px.perf, "built": right::built_text(st, self.px.joined), "moves": animate::moves_text(st), "plan": animate::plan_text(st),
                      "frames": left::frames_text(st), "video": self.px.video_text,
                      "bench": match &self.px.bench { Bench::Done(s) => s.as_str(), Bench::Asked => "running...", Bench::Idle => "" } },
            "closed": self.px.store.closed,
            "arrangement": self.px.store.arrangement,
            "hidden": { "left": self.px.hide_left, "right": self.px.hide_right },
            "room": { "open": self.px.store.room_open, "at": self.px.store.room_at, "light": self.px.light + 1 },
            "rig": { "lights": st.rig().lights.iter().map(|l| json!({ "az": l.az, "el": l.el, "w": l.w, "h": l.h, "rgb": l.rgb, "soft": l.soft })).collect::<Vec<_>>(),
                     "ambient": st.rig().ambient, "own": st.rig_is_default(), "lit": room::lights_lit(st) },
            "hdri": { "path": st.hdri, "name": self.engine.hdri_info().map(|i| i.0), "size": self.engine.hdri_info().map(|i| [i.1, i.2]), "error": self.engine.hdri_error,
                      "loading": self.engine.hdri_loading(), "preview": self.preview.is_some() },
            "settles": self.engine.max_accum(),
            "layers": {
                "selected": st.stack.sel + 1,
                "list": (0..st.layer_count()).map(|i| {
                    let (l, sh) = (&st.stack.layers[i], st.layer_shape(i));
                    json!({ "title": st.layer_title(i), "family": st.layer_family(i).id, "preset": st.layer_preset(i).id, "show": l.show, "body": st.body_of(i) + 1,
                            "pos": l.place.pos, "rot": l.place.rot, "size": l.place.size, "res": sh.map(|s| s.res), "animRes": sh.map(|s| s.anim_res) })
                }).collect::<Vec<_>>(),
                "bodies": st.bodies().iter().map(|b| b.iter().map(|i| i + 1).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "combos": st.stack.combos().iter().map(|m| {
                    let (p, j) = (st.stack.pair(*m), st.stack.junction(*m));
                    let mut c = json!({ "tag": gf_core::layers::tag(*m), "title": st.combo_title(*m), "says": st.combo_summary(*m), "oneBody": st.one_body(*m) });
                    if m.count_ones() == 2 {
                        c["op"] = json!(p.op.id());
                        c["profile"] = json!(p.profile.id());
                        c["radius"] = json!(p.radius);
                        c["gap"] = json!(p.gap);
                        c["amount"] = json!(p.amount);
                    } else {
                        c["fill"] = json!(j.fill);
                        c["reach"] = json!(j.reach);
                    }
                    c
                }).collect::<Vec<_>>(),
                "open": self.px.combo_open.map(gf_core::layers::tag),
                "joined": self.px.joined.map(|j| j.0),
                "plain": st.plain(),
            },
            "panels": { "room": self.px.room_h, "left": self.px.needs[0], "right": self.px.needs[1], "rightRest": self.px.right_rest },
            "help": self.px.help_open,
            "noted": self.test.as_ref().map(|t| t.noted.clone()),
            "families": st.families.iter().map(|f| f.id).collect::<Vec<_>>(),
            "videoChoices": { "size": st.render.size, "fps": st.render.fps, "samples": st.render.samples, "lens": st.render.lens, "shutter": st.render.shutter,
                              "codec": st.render.codec, "quality": st.render.crf, "folder": folder.display().to_string() },
            "video": self.last_video.as_ref().map(|v| json!({
                "path": v.path.display().to_string(), "frames": v.frames, "loop": v.loop_s, "fps": v.fps, "width": v.width, "height": v.height,
                "seconds": v.seconds, "bytes": v.bytes, "lens": v.lens,
            })),
            "stillBits": if self.px.store.still_16 { 16 } else { 8 },
            "lamps": st.lamps.iter().map(|l| json!({ "kind": l.kind.id(), "pos": l.pos, "rot": l.rot, "size": l.size, "rgb": l.rgb, "power": l.power, "seen": l.seen })).collect::<Vec<_>>(),
            "handles": { "on": self.px.handles.on, "mode": self.px.handles.mode.word(), "lamp": self.px.handles.lamp },
            "folders": { "videos": folder.display().to_string(), "setups": self.px.store.setups_folder().display().to_string(), "stills": self.px.store.stills_folder().display().to_string() },
            "rendering": self.video.is_some(),
            "ffmpeg": self.px.ffmpeg,
            "window": { "pixels": [self.canvas.0, self.canvas.1], "points": [self.full.width(), self.full.height()], "pixelsPerPoint": ctx.pixels_per_point() },
            "picture": { "size": [cw, ch], "traced": [tw, th], "frames": self.engine.renderer.accum_n },
            "framesShown": self.frames.load(Ordering::SeqCst),
            "undrawn": self.passes - self.frames.load(Ordering::SeqCst),
            "idle": self.test.as_ref().and_then(|t| t.idle_found).map(|(s, n)| json!({ "seconds": s, "frames": n })),
            "error": self.engine.error,
            "adapter": self.engine.gpu.name,
            "windowFormat": format!("{:?}", self.rs.target_format),
            "rects": rects,
        });
        // (what follows is added afterwards: one list of everything is more than the macro that makes it will take)
        let r = st.camera().readout(self.px.aspect);
        let camera = json!({ "focal": r.focal, "fstop": r.fstop, "field": [r.field_v, r.field_h], "focusMm": r.focus_mm, "distanceMm": r.distance_mm, "proj": st.camera().proj.id(),
                             "says": crate::ui::camera::summary(st, self.px.aspect) });
        let more = json!({
            "lab": { "open": !self.px.store.is_closed("material") },
            "cameraWindow": { "open": !self.px.store.is_closed("camera") },
            "model": { "is": st.is_model(), "needsFile": st.needs_file(st.stack.sel), "path": st.model_file(st.stack.sel), "known": self.px.model.known, "file": self.px.model.file,
                       "line": self.px.model.line, "tip": self.px.model.tip, "shell": self.px.model.shell, "foundShell": self.px.model.found_shell, "ownSmooth": self.px.model.own_smooth,
                       "reading": self.px.model.reading, "error": self.px.model.error, "said": self.px.model.said,
                       "overPicture": self.model_refused.as_ref().is_some_and(|(_, at)| !st.is_model() && at.elapsed() < REFUSED_FOR) },
            "material": material::told(st, &self.px),
            "casting": { "on": self.px.casting.on, "receivers": self.px.casting.receivers, "focusing": self.px.casting.focusing, "lights": self.px.casting.lights, "casters": self.px.casting.casters, "floor": self.px.casting.floor },
            "camera": camera,
            "menu": widgets::menu_places(ctx).iter().map(|r| json!([r.min.x, r.min.y, r.max.x, r.max.y])).collect::<Vec<_>>(),
        });
        if let (Some(all), serde_json::Value::Object(more)) = (told.as_object_mut(), more) { all.extend(more); }
        told
    }
}

/// How long a model's file that could not be used is said over the picture, when no model is up.
const REFUSED_FOR: Duration = Duration::from_secs(10);

/// A file a script drops on the window: what the window is handed when a real one is dropped on it.
#[derive(Debug)]
struct Dropped(PathBuf);
impl egui::DroppedFile for Dropped {
    fn path(&self) -> &Path {
        &self.0
    }
    fn bytes(&self) -> Result<Vec<u8>, String> {
        std::fs::read(&self.0).map_err(|e| format!("{}: {e}", self.0.display()))
    }
}

/// The name of a file, for telling the user about it.
fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Make sure the folder a file is to go into exists.
fn make_folder(file: &Path) -> Result<(), String> {
    match file.parent().filter(|d| !d.as_os_str().is_empty()) {
        Some(dir) => std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display())),
        None => Ok(()),
    }
}

/// A name typed into a dialog without an ending is given the usual one.
fn with_ending(path: PathBuf, ending: &str) -> PathBuf {
    if path.extension().is_none() { path.with_extension(ending) } else { path }
}

/// Have a dialog open in a folder, if the folder is there.
fn in_folder(dialog: rfd::FileDialog, dir: &Path) -> rfd::FileDialog {
    if dir.is_dir() { dialog.set_directory(dir) } else { dialog }
}

/// Write pixels as a PNG: top row first, 16-bit values high byte first.
fn write_png(path: &Path, w: u32, h: u32, colour: png::ColorType, depth: png::BitDepth, pixels: &[u8]) -> Result<(), String> {
    make_folder(path)?;
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(colour);
    enc.set_depth(depth);
    let mut wr = enc.write_header().map_err(|e| format!("{}: {e}", path.display()))?;
    wr.write_image_data(pixels).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write a picture of the window as a PNG.
fn save_png(path: &Path, image: &egui::ColorImage) -> Result<(), String> {
    let mut rgba: Vec<u8> = image.as_raw().to_vec();
    // the window is opaque whatever was drawn into its alpha
    for px in rgba.chunks_exact_mut(4) { px[3] = 255; }
    write_png(path, image.size[0] as u32, image.size[1] as u32, png::ColorType::Rgba, png::BitDepth::Eight, &rgba)
}

impl eframe::App for App {
    /// A scripted run sees only the input its script gives it, so that a mouse that happens to be over
    /// the window cannot change what a test finds.
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        let Some(t) = &mut self.test else { return };
        raw.events.retain(|e| {
            use egui::Event::*;
            !matches!(
                e,
                Key { .. } | Text(_) | ModifiersChanged(_) | PointerMoved(_) | MouseMoved(_) | PointerButton { .. } | PointerGone | MouseWheel { .. } | Zoom(_) | Touch { .. }
                    | WindowFocused(_) | Copy | Cut | Paste(_)
            )
        });
        raw.focused = true;
        // The script's input stays queued until a pass that draws has taken it (see `ui`). A pass that
        // only runs the logic hands its input on to the next pass, which comes through here again: what
        // was put in then has just been taken out with the rest, and is put in afresh, once.
        if t.told != Some(t.held) { raw.events.push(egui::Event::ModifiersChanged(t.held)); }
        raw.events.extend(self.inject.iter().cloned());
        raw.dropped_files.extend(self.dropped.iter().map(|p| Arc::new(Dropped(p.clone())) as egui::DroppedFileHandle));
    }

    /// What goes on whether or not the window is being drawn: the benchmark, a video.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.passes += 1;
        if self.bench_in > 0 {
            self.bench_in -= 1;
            if self.bench_in == 0 && self.video.is_none() {
                let ms = self.engine.bench(60);
                self.px.bench = Bench::Done(format!("{ms:.2} ms  =  {:.0} fps", 1000.0 / ms));
                self.last = None;
            }
            ctx.request_repaint();
        }
        if self.video.is_some() {
            self.step_video();
            ctx.request_repaint();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.frames.fetch_add(1, Ordering::SeqCst);
        // this pass has taken the script's input; the script moves on, one step for every frame drawn
        self.inject.clear();
        self.dropped.clear();
        if let Some(t) = &mut self.test { t.told = Some(t.held); }
        self.script(&ctx);
        if let Some(s) = self.opts.scale {
            if (ctx.pixels_per_point() - s).abs() > 1e-3 { ctx.set_pixels_per_point(s); }
        }
        // the picture is as large as the window's inside, counted in real pixels
        let ppp = ctx.pixels_per_point();
        self.canvas = match frame.winit_window() {
            Some(w) => (w.inner_size().width, w.inner_size().height),
            None => ((ctx.content_rect().width() * ppp).round() as u32, (ctx.content_rect().height() * ppp).round() as u32),
        };
        self.full = Rect::from_min_size(Pos2::ZERO, egui::vec2(self.canvas.0 as f32 / ppp, self.canvas.1 as f32 / ppp));
        // a size asked for on the command line: ask the window for it until it has it, or plainly never will;
        // after that the window is the user's to resize
        if let Some((w, h)) = self.seeking {
            let tries = self.frames.load(Ordering::SeqCst);
            if (w, h) == self.canvas || tries > 300 {
                self.seeking = None;
            } else if tries % 8 == 1 {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w as f32 / ppp, h as f32 / ppp)));
            }
            ctx.request_repaint();
        }
        if self.video.is_none() && !self.px.ffmpeg && self.ffmpeg_checked.elapsed() > Duration::from_secs(3) {
            self.ffmpeg_checked = Instant::now();
            self.px.ffmpeg = crate::video::ffmpeg().is_some();
        }

        // a setup file dropped on the window is opened; a panorama dropped on it lights the shape; a model
        // dropped on it becomes the selected layer's shape, or with shift held one more layer
        if self.video.is_none() {
            let dropped = |ends: &[&str]| ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).find(|p| p.extension().is_some_and(|e| ends.iter().any(|x| e.eq_ignore_ascii_case(x)))));
            if let Some(path) = dropped(&["json"]) { self.load_setup(&path); }
            if let Some(path) = dropped(&["exr", "hdr"]) { self.engine.state.set_hdri(Some(path.display().to_string())); }
            if let Some(path) = dropped(&gf_core::mesh::read::EXTENSIONS) {
                let shift = ctx.input(|i| i.modifiers.shift);
                self.load_model(&path, shift);
            }
        }

        // the controls change the state ...
        let picture = ui.interact(self.full, Id::unique("picture"), Sense::click_and_drag());
        if self.video.is_none() {
            self.gestures(&ctx, &picture);
            self.keys(&ctx);
        }
        self.panels(&ctx);
        self.serve(frame);

        // ... and the engine draws what the state now says
        let mut again = false;
        let ready = self.canvas.0 >= 2 && self.canvas.1 >= 2 && self.seeking.is_none();
        if self.video.is_none() && !self.frozen && ready {
            let now = Instant::now();
            let mut dt = self.last.map_or(0.0, |t| (now - t).as_secs_f64()).min(0.1);
            // after a rest the clock starts afresh: nothing was moving, so no time has to be made up
            if self.rested { dt = dt.min(1.0 / 60.0); }
            self.last = Some(now);
            let t_frame = Instant::now();
            let did = self.engine.frame(self.canvas.0, self.canvas.1, dt);
            perf_log(&self.engine, did.drew, t_frame.elapsed().as_secs_f64() * 1e3);
            // frames a second: those drawn over the last fifth of a second or more, counted
            if did.drew { self.drawn.0 += 1; }
            let span = (now - self.drawn.1).as_secs_f64();
            if span >= 0.2 {
                if self.drawn.0 > 1 { self.fps = self.drawn.0 as f64 / span; }
                self.drawn = (0, now);
            }
            again = did.again;
            // the line at the foot is brought up to date a few times a second, and when the picture settles
            if !did.again || self.px.perf.is_empty() || self.stat_at.elapsed() > Duration::from_millis(250) {
                self.stat_at = now;
                if self.perf() { ctx.request_repaint(); }
            }
            self.rested = !did.again;
        } else if self.video.is_some() && self.stat_at.elapsed() > Duration::from_millis(250) {
            self.stat_at = Instant::now();
            self.perf();
        }

        // the picture: pixel for pixel, or a frame of the video fitted into the window
        let rendering = self.video.is_some();
        if let Some(id) = self.texture(rendering) {
            let ((w, h), _) = self.engine.renderer.size();
            let rect = if rendering && w > 0 && h > 0 {
                let k = (self.full.width() / w as f32).min(self.full.height() / h as f32).min(1.0 / ppp);
                Rect::from_center_size(self.full.center(), egui::vec2(w as f32 * k, h as f32 * k))
            } else {
                self.full
            };
            ui.painter().image(id, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        // over it: the handles, if they are asked for
        if self.video.is_none() {
            let pointer = ctx.input(|i| i.pointer.hover_pos());
            self.px.handles.paint(ui.painter(), &self.engine.state, self.full, pointer);
        }
        // over it, at the foot: the quiet line of figures, hidden with the panels
        if !(self.px.hide_left && self.px.hide_right) { overlay::foot(ui, &mut self.px, self.full); }

        // draw again only while something is going on
        let busy = again || rendering || self.drag.is_some() || ctx.input(|i| i.pointer.any_down());
        let scripted = self.test.as_ref().is_some_and(|t| !t.steps.is_empty() || t.waiting > 0);
        if (busy || scripted) && !self.counting_idle() { ctx.request_repaint(); }
        self.remember(&ctx);
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }

    fn on_exit(&mut self) {
        // an unfinished video is thrown away, and what is remembered is written down
        if let Some(job) = self.video.take() { job.cancel(&mut self.engine); }
        if let Some(path) = &self.store_path {
            if self.px.store != self.saved {
                if let Err(e) = self.px.store.save(path) { eprintln!("glassforms: {e}"); }
            }
        }
    }
}
