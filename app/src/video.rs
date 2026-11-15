//! Rendering one whole loop to a video file.
//!
//! Every period is first snapped so that the last frame leads back into the first (see `State::loop_plan`).
//! Each frame is the average of `samples` traced frames, with the shape rebuilt at full detail whenever it
//! changes. The frames are handed, uncompressed, to ffmpeg, which compresses them and writes the file.
//! A second thread does the handing over, so that rendering the next frame and compressing the last one
//! go on at the same time.
//!
//! A render is driven one frame at a time (`start`, then `step` until it says done, then `finish`), so a
//! window can keep answering while it goes on, and it can be cancelled between any two frames.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::JoinHandle;
use std::time::Instant;

use gf_core::anim::LoopPlan;
use gf_core::state::{Detail, VideoKeep};

use crate::engine::Engine;
use crate::gpu::R;

/// How the video is compressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    /// H.264, 8 bits: plays everywhere
    H264,
    /// H.265, 10 bits: smoother dark gradients, smaller files, plays on most recent things
    H265,
    /// ProRes 422 HQ, 10 bits: for editing; very large
    ProRes,
}
impl Codec {
    pub const ALL: [Codec; 3] = [Codec::H264, Codec::H265, Codec::ProRes];
    pub fn name(self) -> &'static str {
        match self {
            Codec::H264 => "H.264 (plays everywhere)",
            Codec::H265 => "H.265 10-bit",
            Codec::ProRes => "ProRes 422 HQ (for editing)",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::H265 => "h265",
            Codec::ProRes => "prores",
        }
    }
    pub fn from_id(s: &str) -> Option<Codec> {
        Codec::ALL.into_iter().find(|c| c.id() == s)
    }
    pub fn extension(self) -> &'static str {
        if self == Codec::ProRes { "mov" } else { "mp4" }
    }
    /// The file keeps more than 8 bits a channel, so the frames are handed over with 16.
    pub fn deep(self) -> bool {
        self != Codec::H264
    }
}

#[derive(Clone, Debug)]
pub struct VideoOpts {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// traced frames averaged into each frame of the video
    pub samples: u32,
    /// rays through the lens (0 = as live); only matters with an aperture set
    pub lens: u32,
    /// motion blur: the share of each frame's time the shutter is open (0 = none). The samples of a frame
    /// are spread over that time; a shape that is itself changing is rebuilt `SHAPE_STEPS` times within it.
    pub shutter: f64,
    /// stop after this many frames instead of the whole loop
    pub frames: Option<u32>,
    pub codec: Codec,
    /// constant quality for H.264 / H.265: lower is better and larger (16 is hard to tell from the original)
    pub crf: u32,
    /// the file to write; None = a name made from the shape and size, in `dir`
    pub out: Option<PathBuf>,
    pub dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct VideoResult {
    pub path: PathBuf,
    pub frames: u32,
    pub loop_s: f64,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub seconds: f64,
    pub bytes: u64,
    /// lens rays used (0 = no lens)
    pub lens: u32,
    pub plan: LoopPlan,
}

/// How many times within one blurred frame a changing shape is rebuilt.
const SHAPE_STEPS: u32 = 4;

pub struct VideoJob {
    opts: VideoOpts,
    keep: Option<VideoKeep>,
    child: Child,
    /// frames on their way to ffmpeg: at most two wait, so the renderer cannot run far ahead of it
    frames: Option<SyncSender<Vec<u8>>>,
    writer: Option<JoinHandle<std::io::Result<()>>>,
    errors: Option<JoinHandle<String>>,
    path: PathBuf,
    /// frames in the whole loop, and how many of them will be rendered
    whole: u32,
    pub total: u32,
    pub done: u32,
    lens: u32,
    was_deep: bool,
    t0: Instant,
}

/// Where ffmpeg is: beside the program, or on the PATH.
pub fn ffmpeg() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    if let Some(here) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join(exe))) {
        if here.exists() { return Some(here); }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe)).find(|p| p.exists())
}

fn ffmpeg_args(o: &VideoOpts, path: &Path) -> Vec<String> {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    // frames arrive as raw pixels: 8 bits a channel, or 16 for a file that keeps more than 8
    let mut a = s(&["-hide_banner", "-loglevel", "error", "-nostats", "-xerror", "-y", "-f", "rawvideo", "-pix_fmt", if o.codec.deep() { "rgba64le" } else { "rgba" }]);
    a.extend(["-s".into(), format!("{}x{}", o.width, o.height), "-framerate".into(), o.fps.to_string(), "-i".into(), "-".into(), "-an".into()]);
    // The frames are sRGB. Video is stored as brightness and two colour differences, by the BT.709 rule
    // and in the narrow "TV" range; say so both to the converter (which otherwise uses an older rule and
    // shifts the colours) and in the file (so that players undo it the same way). The tags are set on the
    // frames themselves, as the last filter: given as options after, ffmpeg drops two of the three.
    // 16-bit frames go through zscale, which dithers on the way down to 10 bits; the plain scaler does not.
    let (pix, trc) = match o.codec {
        Codec::H264 => ("yuv420p", "iec61966-2-1"),
        Codec::H265 => ("yuv420p10le", "iec61966-2-1"),
        // editing programs expect the plain BT.709 tag
        Codec::ProRes => ("yuv422p10le", "bt709"),
    };
    let convert = if o.codec.deep() {
        "zscale=rangein=full:matrix=709:range=limited:dither=error_diffusion".to_string()
    } else {
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int".to_string()
    };
    a.extend(["-vf".into(), format!("{convert},format={pix},setparams=colorspace=bt709:color_primaries=bt709:color_trc={trc}:range=tv")]);
    // A file whose job is to loop: a key frame every second so any player can start cleanly, and no
    // B-frames, which make players that ignore an MP4's edit list hitch at the seam.
    let g = o.fps.to_string();
    match o.codec {
        Codec::H264 => a.extend(s(&["-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-crf", &o.crf.to_string(), "-g", &g, "-bf", "0", "-movflags", "+faststart"])),
        Codec::H265 => a.extend(s(&[
            "-c:v", "libx265", "-preset", "medium", "-crf", &o.crf.to_string(), "-tag:v", "hvc1",
            "-x265-params", &format!("log-level=error:keyint={g}:open-gop=0:bframes=0:aq-mode=3"), "-movflags", "+faststart",
        ])),
        Codec::ProRes => a.extend(s(&["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-qscale:v", "4"])),
    }
    a.push(path.to_string_lossy().into_owned());
    a
}

impl VideoJob {
    /// Begin: fit everything to the loop, size the picture, start ffmpeg.
    pub fn start(e: &mut Engine, mut opts: VideoOpts) -> R<VideoJob> {
        let ff = ffmpeg().ok_or("ffmpeg was not found (put ffmpeg.exe beside the program or on the PATH)")?;
        opts.width = (opts.width & !1).max(2);
        opts.height = (opts.height & !1).max(2);
        let whole = e.state.video_frames(opts.fps);
        let total = opts.frames.map_or(whole, |f| f.min(whole)).max(1);
        let path = match &opts.out {
            Some(p) => p.clone(),
            None => opts.dir.join(format!("glass-{}-{}x{}-{:.1}s.{}", e.state.preset().id, opts.width, opts.height, total as f64 / opts.fps as f64, opts.codec.extension())),
        };
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|err| format!("{}: {err}", dir.display()))?;
        }
        let mut cmd = Command::new(ff);
        cmd.args(ffmpeg_args(&opts, &path)).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);            // no console window of its own
        }
        let mut child = cmd.spawn().map_err(|err| format!("could not start ffmpeg: {err}"))?;
        // one thread hands the frames over (ffmpeg takes a frame only as fast as it compresses) ...
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no input")?;
        let (tx, rx) = sync_channel::<Vec<u8>>(2);
        let writer = std::thread::spawn(move || -> std::io::Result<()> {
            for frame in rx {
                stdin.write_all(&frame)?;
            }
            Ok(())                                      // the pipe closes here: that tells ffmpeg the video is complete
        });
        // ... and another collects whatever ffmpeg complains about, so a failure can be explained
        let mut stderr = child.stderr.take();
        let errors = std::thread::spawn(move || {
            let mut s = String::new();
            if let Some(st) = stderr.as_mut() { let _ = st.read_to_string(&mut s); }
            s
        });
        let lens = e.state.video_lens(opts.lens);
        let keep = e.state.video_begin(opts.lens);
        let was_deep = e.renderer.deep;
        e.renderer.deep = opts.codec.deep();
        e.size_override = Some((opts.width, opts.height));
        e.resize(opts.width, opts.height);
        Ok(VideoJob { opts, keep: Some(keep), child, frames: Some(tx), writer: Some(writer), errors: Some(errors), path, whole, total, done: 0, lens, was_deep, t0: Instant::now() })
    }

    /// Render the next frame and hand it to ffmpeg. True when the last frame has gone.
    pub fn step(&mut self, e: &mut Engine) -> R<bool> {
        if self.done >= self.total { return Ok(true); }
        let keep = self.keep.as_ref().ok_or("the render has ended")?;
        let n = self.opts.samples.max(1);
        let shutter = self.opts.shutter.clamp(0.0, 1.0);
        if e.state.video_frame(keep, self.done as f64, self.whole) {
            e.bake(Detail::Hq)?;
        }
        e.state.want = None;
        e.restart();
        e.averaging = shutter > 0.0;
        for k in 0..n {
            if shutter > 0.0 && k > 0 {
                // the moment within the frame this sample looks at; the shape is rebuilt only every so often
                let at = self.done as f64 + shutter * k as f64 / n as f64;
                let moving = e.state.video_frame(keep, at, self.whole);
                let every = (n / SHAPE_STEPS).max(1);
                if moving && k % every == 0 {
                    if let Err(err) = e.bake(Detail::Hq) {
                        e.averaging = false;
                        return Err(err);
                    }
                }
                e.state.want = None;
            }
            e.draw(k + 1 < n);
            if k % 16 == 15 { e.gpu.wait(); }
        }
        e.averaging = false;
        let frame: Vec<u8> = if self.opts.codec.deep() {
            let (_, _, deep) = e.renderer.read_screen16()?;
            bytemuck::cast_slice(&deep).to_vec()
        } else {
            e.screenshot()?.2
        };
        let sent = self.frames.as_ref().is_some_and(|tx| tx.send(frame).is_ok());
        if !sent {
            return Err(self.ffmpeg_failed());
        }
        self.done += 1;
        Ok(self.done >= self.total)
    }

    /// ffmpeg stopped taking frames: find out what it said.
    fn ffmpeg_failed(&mut self) -> String {
        self.frames = None;
        let wrote = self.writer.take().and_then(|h| h.join().ok());
        let _ = self.child.wait();
        let said = self.errors.take().and_then(|h| h.join().ok()).unwrap_or_default();
        let why = match wrote {
            Some(Err(err)) => format!(" ({err})"),
            _ => String::new(),
        };
        format!("ffmpeg stopped{why}{}", if said.trim().is_empty() { String::new() } else { format!(": {}", said.trim()) })
    }

    /// Seconds per frame so far, and seconds left at that rate.
    pub fn pace(&self) -> (f64, f64) {
        let per = self.t0.elapsed().as_secs_f64() / self.done.max(1) as f64;
        (per, per * (self.total - self.done) as f64)
    }

    fn restore(&mut self, e: &mut Engine) {
        if let Some(keep) = self.keep.take() {
            e.state.video_end(keep);
        }
        e.renderer.deep = self.was_deep;
        e.size_override = None;
        e.restart();
    }

    /// The last frame has gone: close the file and put everything back the way it was.
    pub fn finish(mut self, e: &mut Engine) -> R<VideoResult> {
        let plan = self.keep.as_ref().map(|k| k.plan.clone()).ok_or("the render has ended")?;
        self.restore(e);
        self.frames = None;                             // no more frames: the writer finishes and closes the pipe
        let wrote = self.writer.take().and_then(|h| h.join().ok());
        let status = self.child.wait().map_err(|err| format!("ffmpeg: {err}"))?;
        let said = self.errors.take().and_then(|h| h.join().ok()).unwrap_or_default();
        if !status.success() || !matches!(wrote, Some(Ok(()))) {
            let _ = std::fs::remove_file(&self.path);
            return Err(format!("ffmpeg failed: {}", said.trim()));
        }
        let bytes = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        Ok(VideoResult {
            path: self.path.clone(), frames: self.done, loop_s: plan.l, fps: self.opts.fps, width: self.opts.width, height: self.opts.height,
            seconds: self.t0.elapsed().as_secs_f64(), bytes, lens: self.lens, plan,
        })
    }

    /// Stop now: put everything back and throw the unfinished file away.
    pub fn cancel(mut self, e: &mut Engine) {
        self.restore(e);
        let _ = self.child.kill();
        self.frames = None;
        if let Some(h) = self.writer.take() { let _ = h.join(); }
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Render a whole video in one go (for the command line and tests). `progress(done, total)` is called after every frame.
pub fn render(e: &mut Engine, opts: VideoOpts, mut progress: impl FnMut(u32, u32)) -> R<VideoResult> {
    let mut job = VideoJob::start(e, opts)?;
    loop {
        match job.step(e) {
            Ok(done) => {
                progress(job.done, job.total);
                if done { break; }
            }
            Err(err) => {
                job.cancel(e);
                return Err(err);
            }
        }
    }
    job.finish(e)
}
