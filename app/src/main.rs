//! Glass Forms.
//!
//!   glassforms [name=value ...]                       the live app
//!   glassforms --still out.png [name=value ...]       render a still and exit
//!   glassforms --video out.mp4 [name=value ...]       render one whole loop to a video and exit
//!                                                     (fps, samples, lensRays, shutter, frames, codec=h264|h265|prores, crf)
//!   glassforms --bench [name=value ...]               time frames of a turning view and exit
//!   glassforms --bench-anim [name=value ...]          time frames of the animation playing and exit
//!   glassforms --list                                 print every family and its presets
//!   glassforms --list-materials                       print every named material (material=<id> chooses one)
//!
//!   --setup file.json        start from a saved setup (settings given as name=value are laid over it)
//!   --save-setup file.json   with --still: also save the setup the still was made from
//!   --16                     with --still: a 16-bit PNG, tone-mapped as on screen but without the grain
//!   --preview file.png       with --still: also save the small picture of the lighting a light room shows
//!   glassforms --hdri-info picture.exr                what an environment picture is, and how long it takes to read
//!   glassforms --load-test picture.exr                frame times of a turning view while that picture is read
//!                                                     in the background, as the window reads one
//!                                                     (--load-test clay: while the tracer for clay is built)
//!
//! Every setting can be given as name=value, with the names the web version takes in its URL: scene, preset,
//! any setting (ior, disp, dist, fov, aperture, ...), thick, len, melt, puff, travel, beads, res, p.<name>,
//! rx, ry, rz. For the modes that make no window: w, h (size of the picture), frames (how many are
//! averaged), anim (seconds into the animation), backend (vulkan, dx12, auto), seed (a whole number: the
//! same picture made from other rays, to see how far two renders of it agree), drift (name:step, for
//! --still: after the still, as many frames again are drawn with that setting moving on by the step
//! each frame, or with drift=turn:degrees the shape turning: the picture as it looks in motion).
//!
//! The lights (of the family that is up; angles and sizes in degrees; r, g, b are colour and brightness in one):
//!   rig=<light>:<light>:...     all the lights at once, each as az,el,w,h,r,g,b or az,el,w,h,r,g,b,soft
//!                               (az 0 = behind the shape, 90 = to the right, 180 = at the camera; el = up;
//!                               w, h = half-sizes; soft = the share of the half-size that fades out, 0.6 if
//!                               left out). rig= (nothing) is no lights; rig=default is the family's own
//!   light<n>=<light>            light n (counted from 1) replaced; one more than there are adds a light
//!   light<n>=off                light n taken away
//!   light<n>.az=  .el=  .w=  .h=  .soft=  .rgb=r,g,b      one part of light n
//!   ambient=r,g,b               the light that comes from everywhere
//!   lightGain=                  all of the above made brighter or dimmer together
//! An environment picture (HDRI), an .exr or .hdr panorama:
//!   hdri=<file>                 light the shape with it (a path that is not a full one is looked for in the
//!                               program's folder: hdri=hdri/suburban_garden_4k.exr); hdri=none for none
//!   hdriGain=  hdriBlur=        its brightness; how much it is softened for the glass (0 sharp .. 1 very soft)
//!   hdriShow=1                  show it behind the shape as well
//!   hdriLights=1                keep the family's lights lit with it
//! The lights' turning (lights, lyaw, lpitch, lroll, lspin) turns the picture with them.
//!   envRoom=0                   with no picture, a metal or milky body mirrors the lights on black instead of
//!                               the dim studio room the program makes round them (core/src/room.rs)
//!   clay=1                      draw the shape as matte clay instead of glass
//! The camera in a photographer's terms (the old names fov, aperture, fnear, ffar mean what they did;
//! of a lens said both ways the later one counts):
//!   focal=<mm>  sensor=<mm>     the lens's focal length, on a sensor this high (24 = full frame)
//!   fstop=<N>   subject=<mm>    the f-stop; how big the subject is (what makes the blur right for its size)
//!   focus=  band=  focusDist=   the plane in focus (depth from the object's centre), how deep a range is
//!                               held sharp (0 = a true lens), or the distance focused at in millimetres
//!   dollyZoom=<mm>              change the lens and move the camera so that the subject keeps its size
//!   proj=<kind>                 perspective, orthographic, fisheye, equisolid, stereographic, panorama, cylindrical
//!   distort=  fringe=  falloff=  squeeze=  bokeh=  catseye=  tiltX=  tiltY=  shiftX=  shiftY=  ev=
//!   pick=<x>,<y>                with --still: focus on the glass under that point of the picture (fractions
//!                               of its width and height from the top left), as a click in the window does
//!   glassforms --camera [name=value ...]              print the camera as a photographer would read it, and exit
//! A polygon model (an OBJ, STL, PLY, glTF or GLB file) as the shape:
//!   scene=models preset=teapot  one of the models that ship (glassforms --list names them)
//!   model=<file>                any file (a path that is not a full one is looked for in the program's folder)
//!   p.size=  p.smooth=  p.wall= puff=      its size, how far detail is melted, a shell's wall, inflation
//!   p.mode=0|1|2                solid or shell as the file was found to be, a solid, a shell
//!   p.up=0|1                    the file's up axis: y, z
//!   glassforms --model-info <file> [res=768] [p.mode=] [p.up=]      what a file is, with no graphics device

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use glassforms::core::state::{Detail, State};
use glassforms::engine::Engine;
use glassforms::gpu::Gpu;

/// Names given as name=value that are not settings of the picture.
const NOT_SETTINGS: [&str; 21] = ["w", "h", "frames", "anim", "backend", "fps", "samples", "lensRays", "shutter", "codec", "crf", "detail", "play", "seed", "drift", "turn", "rounds", "rebuild", "stride", "inPlace", "noWide"];

struct Args {
    flags: Vec<(String, Option<String>)>,
    pairs: Vec<(String, String)>,
}
impl Args {
    fn parse() -> Self {
        let mut flags: Vec<(String, Option<String>)> = Vec::new();
        let mut pairs = Vec::new();
        for a in std::env::args().skip(1) {
            if let Some(f) = a.strip_prefix("--") {
                match f.split_once('=') {
                    Some((k, v)) => flags.push((k.to_string(), Some(v.to_string()))),
                    None => flags.push((f.to_string(), None)),
                }
            } else if let Some((k, v)) = a.split_once('=') {
                pairs.push((k.to_string(), v.to_string()));
            } else if let Some(last) = flags.last_mut().filter(|f| f.1.is_none()) {
                // a bare word after a flag is that flag's value:  --still out.png
                last.1 = Some(a);
            } else {
                eprintln!("ignoring \"{a}\": settings are given as name=value");
            }
        }
        Self { flags, pairs }
    }
    fn flag(&self, k: &str) -> Option<&Option<String>> {
        self.flags.iter().find(|f| f.0 == k).map(|f| &f.1)
    }
    fn num(&self, k: &str) -> Option<f64> {
        self.pairs.iter().rev().find(|p| p.0 == k).and_then(|p| p.1.parse().ok())
    }
    fn text(&self, k: &str) -> Option<&str> {
        self.pairs.iter().rev().find(|p| p.0 == k).map(|p| p.1.as_str())
    }
}

fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let file = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| format!("{path}: {e}"))?;
    wr.write_image_data(rgba).map_err(|e| format!("{path}: {e}"))
}

fn write_png16(path: &str, w: u32, h: u32, rgb: &[u16]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let file = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Sixteen);
    let mut wr = enc.write_header().map_err(|e| format!("{path}: {e}"))?;
    // PNG keeps the high byte first
    let bytes: Vec<u8> = rgb.iter().flat_map(|v| v.to_be_bytes()).collect();
    wr.write_image_data(&bytes).map_err(|e| format!("{path}: {e}"))
}

/// An engine with no window, the shape built and the pictures sized.
fn headless(args: &Args) -> Result<Engine, String> {
    let gpu = Gpu::headless(args.text("backend").unwrap_or("vulkan"))?;
    let mut state = State::with_args(&args.pairs, true);
    if let Some(Some(path)) = args.flag("setup") {
        // a saved setup first, then whatever was given on the command line over it
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        state.apply_setup_json(&text)?;
        for (k, v) in &args.pairs {
            if !NOT_SETTINGS.contains(&k.as_str()) && !state.apply_arg(k, v) {
                eprintln!("ignoring {k}={v}: no such setting");
            }
        }
    }
    let mut e = Engine::new(gpu, state, true);
    // (a check that reads the bodies' volume wants it made whatever the bodies are made of)
    e.bodies_always = args.flag("bodies-out").is_some();
    e.layers.stride = args.num("stride").unwrap_or(1.0);
    e.layers.in_place = args.num("inPlace").unwrap_or(0.0) > 0.5;
    e.layers.no_wide = args.num("noWide").unwrap_or(0.0) > 0.5;
    let (w, h) = (args.num("w").unwrap_or(1280.0) as u32, args.num("h").unwrap_or(720.0) as u32);
    e.size_override = None;
    e.resize(w, h);
    if let Some(mode) = e.state.want.take() {
        e.bake(mode)?;
    }
    // an environment picture, if one was asked for, is read before anything is drawn; one that cannot
    // be read is said so, and the lighting stays as it was
    e.settle_lighting();
    if let Some(why) = &e.hdri_error {
        eprintln!("glassforms: the environment picture was not used: {why}");
    }
    Ok(e)
}

/// The environment picture in use, as a piece of JSON for the line a still or a benchmark prints.
fn hdri_json(e: &Engine) -> String {
    let text = |s: &str| s.replace('\\', "/").replace('"', "'");
    let picture = match e.hdri_details() {
        Some(i) => format!(
            "{{\"file\":\"{}\",\"size\":[{},{}],\"face\":{},\"levels\":{},\"megabytes\":{:.1},\"scale\":{:.5},\"peak\":{:.1},\"takenOff\":{:.4},\"givenBack\":{:.4},\"glows\":{},\"readMs\":{:.0},\"tameMs\":{:.0},\"uploadMs\":{:.0},\"gpuMs\":{:.0}}}",
            text(&i.file), i.w, i.h, i.face, i.levels, i.bytes as f64 / 1048576.0, i.scale, i.peak, i.taken_off, i.given_back, i.glows, i.read_ms, i.tame_ms, i.upload_ms, i.gpu_ms
        ),
        None => "null".into(),
    };
    let path = e.state.hdri.as_ref().map_or("null".into(), |p| format!("\"{}\"", text(p)));
    let error = e.hdri_error.as_ref().map_or("null".into(), |w| format!("\"{}\"", text(w)));
    format!("\"hdri\":{picture},\"hdriPath\":{path},\"hdriError\":{error}{}{}", e.models_json(), e.stage_json())
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    if args.flag("list").is_some() {
        // every family and its presets, for scripts
        let fams = glassforms::core::presets::families();
        let rows: Vec<String> = fams
            .iter()
            .map(|f| format!("{{\"id\":\"{}\",\"name\":\"{}\",\"presets\":[{}]}}", f.id, f.name, f.presets.iter().map(|p| format!("\"{}\"", p.id)).collect::<Vec<_>>().join(",")))
            .collect();
        println!("[{}]", rows.join(","));
        return Ok(());
    }
    if let Some(file) = args.flag("model-info") {
        // read a model file and examine it as the program does (and, with res= given, prepare it for a
        // volume of that size): what was found and how long each part took. No graphics device is used.
        let file = file.clone().ok_or("--model-info needs a file")?;
        let shell = match args.text("p.mode") { Some("1") => Some(false), Some("2") => Some(true), _ => None };
        let json = glassforms::core::mesh::describe(&glassforms::hdri::resolve(&file), args.num("p.up").unwrap_or(0.0) > 0.5, args.num("res").map(|r| glassforms::core::state::valid_res(r)), shell)?;
        println!("{json}");
        return Ok(());
    }
    if let Some(file) = args.flag("hdri-info") {
        // read an environment picture as the program does, and say what it is and how long each part took
        let file = file.clone().ok_or("--hdri-info needs a file")?;
        let gpu = Gpu::headless(args.text("backend").unwrap_or("vulkan"))?;
        let mut state = State::with_args(&args.pairs, true);
        state.set_hdri(Some(file));
        let mut e = Engine::new(gpu, state, true);
        e.settle_lighting();
        println!("{{{}}}", hdri_json(&e));
        return match e.hdri_error.take() {
            Some(why) => Err(why),
            None => Ok(()),
        };
    }
    if let Some(file) = args.flag("load-test") {
        // What the window goes through when a picture is chosen: frames of a turning view go on being
        // drawn while the picture is read on another thread, then the lighting switches over.
        let file = file.clone().ok_or("--load-test needs a file")?;
        let mut e = headless(&args)?;
        e.hdri_wait = false;
        let turn = glassforms::core::math::axis_angle(0.3, 1.0, 0.2, 0.005);
        let frame = |e: &mut Engine| {
            let t = Instant::now();
            e.state.turn(&turn);
            e.draw(false);
            e.gpu.wait();
            t.elapsed().as_secs_f64() * 1e3
        };
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let mut before: Vec<f64> = (0..40).map(|_| frame(&mut e)).collect();
        // (--load-test clay times the other thing that is built in the background: a tracer not used before)
        // (and --load-test lens the same for the tracer that traces through any lens: a fisheye is put on)
        if file == "lens" { e.state.apply_arg("proj", "equisolid"); } else if file == "clay" { e.state.p.clay = 1.0; } else { e.state.set_hdri(Some(file)); }
        let t0 = Instant::now();
        let (mut during, mut switch, mut took) = (Vec::new(), 0.0, 0.0);
        while t0.elapsed().as_secs_f64() < 60.0 {
            let was = e.lighting_pending() || during.is_empty();
            let ms = frame(&mut e);
            if was && !e.lighting_pending() {
                // the frame in which the picture arrived: it also builds the tracer for the new lighting
                switch = ms;
                took = t0.elapsed().as_secs_f64() * 1e3;
                break;
            }
            during.push(ms);
        }
        let mut after: Vec<f64> = (0..40).map(|_| frame(&mut e)).collect();
        let longest = during.iter().cloned().fold(0.0, f64::max);
        println!(
            "{{\"frameMsBefore\":{:.2},\"whileReading\":{{\"frames\":{},\"medianMs\":{:.2},\"longestMs\":{:.2}}},\"switchFrameMs\":{:.1},\"readyAfterMs\":{:.0},\"frameMsAfter\":{:.2},{}}}",
            median(&mut before), during.len(), median(&mut during.clone()), longest, switch, took, median(&mut after), hdri_json(&e)
        );
        return match e.hdri_error.take() {
            Some(why) => Err(why),
            None => Ok(()),
        };
    }
    if args.flag("list-materials").is_some() {
        // every named material, for scripts: its id (what `material=` takes), name, group and whether it is costly
        let rows: Vec<String> = glassforms::core::material::NAMED
            .iter()
            .map(|m| format!("{{\"id\":\"{}\",\"name\":\"{}\",\"group\":\"{}\",\"costly\":{}}}", m.id, m.name, m.group, m.costly))
            .collect();
        println!("[{}]", rows.join(","));
        return Ok(());
    }
    if args.flag("camera").is_some() {
        // the camera as the settings come to, in a photographer's terms: no picture is made
        let mut state = State::with_args(&args.pairs, true);
        if let Some(Some(path)) = args.flag("setup") {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            state.apply_setup_json(&text)?;
            for (k, v) in &args.pairs {
                if !NOT_SETTINGS.contains(&k.as_str()) { state.apply_arg(k, v); }
            }
        }
        let aspect = args.num("w").unwrap_or(1280.0) / args.num("h").unwrap_or(720.0);
        println!("{}", glassforms::camera::readout_json(&state, aspect));
        return Ok(());
    }
    if let Some(out) = args.flag("still") {
        let out = out.clone().unwrap_or_else(|| "still.png".into());
        let t0 = Instant::now();
        let mut e = headless(&args)?;
        let setup = t0.elapsed().as_secs_f64() * 1e3;
        let frames = args.num("frames").unwrap_or(64.0).max(1.0) as u32;
        if let Some(t) = args.num("anim") {
            // show everything as it is that long into the animation
            e.state.clock = t;
            e.state.apply_clock(true);
            e.bake(Detail::Hq)?;
            e.state.want = None;
        }
        if let Some(Some(path)) = args.flag("preview") {
            // the small picture of the lighting: asking for it is what gets it drawn
            e.env_preview();
            e.settle_lighting();
            let tex = e.env_preview().ok_or("the picture of the lighting was not drawn")?;
            let rgba = e.gpu.read(tex, 0, 0, 0, tex.width(), tex.height(), 4)?;
            write_png(path, tex.width(), tex.height(), &rgba)?;
        }
        let t1 = Instant::now();
        // turn=N: first draw N frames of a turning view. Alone, the still is the last of them (what a
        // moving view shows); with frames given as well, that view is then rendered afresh and averaged.
        let turn = args.num("turn").map(|n| n.max(0.0) as u32);
        if let Some(n) = turn { e.turned(n); }
        if turn.is_none() || args.num("frames").is_some() { e.render(frames); }
        if let Some((fx, fy)) = args.text("pick").and_then(|p| p.split_once(',')).and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?))) {
            // pick=x,y: focus on the glass under that point of the picture (fractions of its width and
            // height from the top left), as a click in the window does, and render again
            if e.pick_focus(fx, fy).is_some() { e.render(frames); }
        }
        if let Some((name, step)) = args.text("drift").and_then(|d| d.split_once(':')).and_then(|(n, s)| Some((n, s.parse::<f64>().ok()?))) {
            // the picture as it is while something moves: every frame a setting moves on by a step
            // (or, with the name "turn", the shape turns by that many degrees) and the frame is folded
            // in by the antialiasing that follows motion, as in the window
            for _ in 0..frames {
                if name == "turn" {
                    e.state.turn(&glassforms::core::math::axis_angle(0.3, 1.0, 0.2, step.to_radians()));
                } else if name == "spin" {
                    // (the spin itself, a step at a time: about the room's vertical when it is a turntable)
                    e.state.theta += step.to_radians();
                    e.state.update_r();
                } else if let Some(v) = glassforms::core::camera::get(&e.state.p, name).or_else(|| e.state.p.get(name)) {
                    // (the camera's names go through the camera, so that what follows from them follows)
                    if !glassforms::core::camera::set(&mut e.state.p, name, v + step) { e.state.p.set(name, v + step); }
                }
                e.draw(false);
            }
            e.gpu.wait();
        }
        if let Some(seed) = args.num("seed").filter(|s| *s >= 1.0) {
            // the same picture from other rays: start the sequences elsewhere and render again
            e.renderer.reseed(seed as u32 * 7919);
            e.render(frames);
        }
        let per = t1.elapsed().as_secs_f64() * 1e3 / frames as f64;
        let (w, h, rgba) = e.screenshot()?;
        if let Some(Some(path)) = args.flag("hdr") {
            // the picture before bloom and tone mapping: width, height, then RGBA 32-bit floats, top row first
            let ((_, _), (tw, th)) = e.renderer.size();
            let hdr = e.renderer.read_hdr(0, 0, tw, th)?;
            let mut f = BufWriter::new(File::create(path).map_err(|e| format!("{path}: {e}"))?);
            f.write_all(&tw.to_le_bytes()).and_then(|_| f.write_all(&th.to_le_bytes())).and_then(|_| f.write_all(bytemuck::cast_slice(&hdr))).map_err(|e| format!("{path}: {e}"))?;
        }
        if let Some(Some(path)) = args.flag("glow-out") {
            // the light inside a milky body as the tracer reads it: cells a side, then red, green, blue
            // and distance to the surface of every cell as 32-bit floats
            let (res, cells) = e.scatter.read()?;
            let mut f = BufWriter::new(File::create(path).map_err(|e| format!("{path}: {e}"))?);
            f.write_all(&res.to_le_bytes()).and_then(|_| f.write_all(bytemuck::cast_slice(&cells))).map_err(|e| format!("{path}: {e}"))?;
        }
        if args.flag("16").is_some() {
            // sixteen bits a channel: the picture so far, tone-mapped as the screen is, with no grain and no
            // rounding to 8 bits in between (for further work in an editor)
            let ((_, _), (tw, th)) = e.renderer.size();
            let hdr = e.renderer.read_hdr(0, 0, tw, th)?;
            let bloom = e.renderer.read_bloom()?;
            write_png16(&out, tw, th, &glassforms::render::tone_map16(&hdr, &bloom, tw, th, &e.state.p))?;
        } else {
            write_png(&out, w, h, &rgba)?;
        }
        if let Some(Some(path)) = args.flag("save-setup") {
            std::fs::write(path, e.state.setup_json()).map_err(|e| format!("{path}: {e}"))?;
        }
        let m = &e.state.shape().meta;
        println!(
            "{{\"scene\":\"{}\",\"preset\":\"{}\",\"adapter\":\"{}\",\"size\":[{w},{h}],\"frames\":{frames},\"msPerFrame\":{per:.3},\"setupMs\":{setup:.0},\"volume\":{},\"volumeMs\":{:.1},\"segments\":{}{},\"out\":\"{}\",{}}}",
            e.state.family().id, e.state.preset().id, e.gpu.name, m.res, m.ms, m.segments, format!("{}{}", e.layers_json(), e.cast_json()), out.replace('\\', "/"), hdri_json(&e)
        );
        return Ok(());
    }
    if let Some(n) = args.flag("bench") {
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(100u32);
        let mut e = headless(&args)?;
        let ms = e.bench(n);
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"adapter\":\"{}\",\"frames\":{n},\"ms\":{ms:.3},\"fps\":{:.1}}}", e.state.family().id, e.state.preset().id, e.gpu.name, 1000.0 / ms);
        return Ok(());
    }
    if let Some(out) = args.flag("video") {
        // render one whole loop to a video file, exactly as the window's "render video" does
        use glassforms::video::{self, Codec, VideoOpts};
        let mut e = headless(&args)?;
        // by default a still does not spin; a video of nothing moving would be no use, so honour what was asked
        if args.num("play").unwrap_or(1.0) > 0.5 { e.state.set_playing(true); e.state.want = None; }
        let opts = VideoOpts {
            width: args.num("w").unwrap_or(1920.0) as u32,
            height: args.num("h").unwrap_or(1080.0) as u32,
            fps: args.num("fps").unwrap_or(60.0) as u32,
            samples: args.num("samples").unwrap_or(32.0) as u32,
            lens: args.num("lensRays").unwrap_or(4.0) as u32,
            // (also as a film camera's shutter angle, or as an exposure time: shutterSpeed=120 is 1/120 s)
            shutter: glassforms::camera::shutter_arg(args.num("shutter"), args.num("shutterAngle"), args.num("shutterSpeed"), args.num("fps").unwrap_or(60.0)),
            frames: args.num("frames").map(|f| f as u32),
            codec: args.text("codec").and_then(Codec::from_id).unwrap_or(Codec::H264),
            crf: args.num("crf").unwrap_or(16.0) as u32,
            out: out.clone().map(std::path::PathBuf::from),
            dir: std::path::PathBuf::from("."),
        };
        let quiet = args.flag("quiet").is_some();
        let r = video::render(&mut e, opts, |done, total| {
            if !quiet && (done % 10 == 0 || done == total) { eprintln!("frame {done} of {total}"); }
        })?;
        let plan: Vec<String> = r.plan.items.iter().map(|i| format!("\"{}: {:.2} s -> {:.2} s, {} per loop\"", i.label, i.was, i.now, i.n)).collect();
        println!("{{\"file\":\"{}\",\"frames\":{},\"loop\":{:.4},\"fps\":{},\"size\":[{},{}],\"took\":{:.1},\"bytes\":{},\"lens\":{},\"plan\":[{}]}}",
            r.path.to_string_lossy().replace('\\', "/"), r.frames, r.loop_s, r.fps, r.width, r.height, r.seconds, r.bytes, r.lens, plan.join(","));
        return Ok(());
    }
    if args.flag("vol-hash").is_some() {
        // a fingerprint of the distance field, to compare two ways of building the same shape:
        // a hash of every bit, and the mean and extremes of the distances
        let mut e = headless(&args)?;
        if let Some(mode) = match args.text("detail") { Some("anim") => Some(Detail::Anim), Some("quick") => Some(Detail::Quick), Some("hq") => Some(Detail::Hq), _ => None } {
            e.bake(mode)?;
        }
        let (res, bits) = e.read_volume()?;
        let mut h = 0xcbf29ce484222325u64;
        let (mut sum, mut lo, mut hi) = (0.0f64, f32::MAX, f32::MIN);
        for &b in &bits {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            let v = half::f16::from_bits(b).to_f32();
            sum += v as f64;
            lo = lo.min(v);
            hi = hi.max(v);
        }
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"volume\":{res},\"hash\":\"{h:016x}\",\"mean\":{:.9},\"min\":{lo},\"max\":{hi}{}{}}}", e.state.family().id, e.state.preset().id, sum / bits.len() as f64, e.layers_json(), e.models_json());
        if let Some(Some(path)) = args.flag("vol-out") {
            std::fs::write(path, bytemuck::cast_slice::<u16, u8>(&bits)).map_err(|e| format!("{path}: {e}"))?;
        }
        if let Some(Some(path)) = args.flag("bodies-out") {
            // the bodies' volume: two or four 16-bit floats a voxel (see combine.rs)
            let (_, _, bodies) = e.read_bodies()?;
            std::fs::write(path, bytemuck::cast_slice::<u16, u8>(&bodies)).map_err(|e| format!("{path}: {e}"))?;
        }
        return Ok(());
    }
    if let Some(n) = args.flag("bench-bake") {
        // build the shape over and over, each build timed to the end, in parts
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(20u32);
        let mut e = headless(&args)?;
        e.time_bakes = true;
        e.layers.force = args.num("rebuild").unwrap_or(0.0) > 0.5;
        let mode = match args.text("detail") { Some("anim") => Detail::Anim, Some("quick") => Detail::Quick, Some("hq") => Detail::Hq, _ => Detail::Full };
        e.bake(mode)?;
        let mut sum = [0.0f64; 3];
        for i in 0..n {
            // nudge the clock so that an animated shape really changes between builds
            e.state.clock = i as f64 / 60.0;
            if mode == Detail::Anim { e.state.apply_clock(false); e.state.want = None; }
            e.bake(mode)?;
            for k in 0..3 { sum[k] += e.last_bake_parts[k] / n as f64; }
        }
        let m = &e.state.shape().meta;
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"volume\":{},\"segments\":{},\"splats\":{},\"workOutMs\":{:.2},\"handOverMs\":{:.2},\"gpuMs\":{:.2},\"totalMs\":{:.2}{}{}}}",
            e.state.family().id, e.state.preset().id, m.res, m.segments, m.splats, sum[0], sum[1], sum[2], sum[0] + sum[1] + sum[2], e.layers_json(), e.models_json());
        return Ok(());
    }
    if let Some(n) = args.flag("bench-combine") {
        // the layers' combining step alone, each run timed to the end
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(20u32);
        let mut e = headless(&args)?;
        let mode = match args.text("detail") { Some("anim") => Detail::Anim, Some("quick") => Detail::Quick, _ => Detail::Full };
        if mode != Detail::Full { e.bake(mode)?; }
        let ms = e.bench_combine(n, mode)?;
        println!("{{\"runs\":{n},\"combineOnlyMs\":{ms:.3}{}}}", e.layers_json());
        return Ok(());
    }
    if args.flag("profile").is_some() {
        // how many volume lookups a pixel costs: the tracer built to count them
        let gpu = Gpu::headless(args.text("backend").unwrap_or("vulkan"))?;
        let mut e = Engine::new(gpu, State::with_args(&args.pairs, true), true);
        e.renderer.profile = true;
        e.state.p.smooth = 0.0;
        e.resize(args.num("w").unwrap_or(1280.0) as u32, args.num("h").unwrap_or(720.0) as u32);
        if let Some(mode) = e.state.want.take() { e.bake(mode)?; }
        e.render(1);
        let ((_, _), (tw, th)) = e.renderer.size();
        let hdr = e.renderer.read_hdr(0, 0, tw, th)?;
        let (mut plain, mut smooth, mut hit, mut plain_hit, mut smooth_hit, mut far) = (0.0f64, 0.0f64, 0u64, 0.0f64, 0.0f64, 0.0f64);
        for px in hdr.chunks(4) {
            plain += px[0] as f64;
            smooth += px[1] as f64;
            far += px[2] as f64;
            if px[3] > 0.0 { hit += 1; plain_hit += px[0] as f64; smooth_hit += px[1] as f64; }
        }
        let n = (tw * th) as f64;
        let (miss, h) = (n - hit as f64, (hit as f64).max(1.0));
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"farShareOfPlain\":{:.3},\"pixelsOnGlass\":{:.3},\"perPixel\":{{\"plain\":{:.1},\"smooth\":{:.1}}},\"onGlass\":{{\"plain\":{:.1},\"smooth\":{:.1}}},\"offGlass\":{{\"plain\":{:.1},\"smooth\":{:.1}}}}}",
            e.state.family().id, e.state.preset().id, far / plain.max(1.0), hit as f64 / n, plain / n, smooth / n, plain_hit / h, smooth_hit / h, (plain - plain_hit) / miss.max(1.0), (smooth - smooth_hit) / miss.max(1.0));
        return Ok(());
    }
    if let Some(n) = args.flag("bench-focus") {
        // what focus costs a turning frame: the lens as set against no lens, in turn, the quickest of each
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(40u32);
        let mut e = headless(&args)?;
        let (without, with) = e.bench_focus(n, args.num("rounds").unwrap_or(8.0) as u32);
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"noLensMs\":{without:.3},\"lensMs\":{with:.3},\"focusMs\":{:.3}}}", e.state.family().id, e.state.preset().id, with - without);
        return Ok(());
    }
    if let Some(n) = args.flag("bench-with") {
        // what settings cost a turning frame: with=name:value,name:value against the same view without
        // them, in turn, the quickest go of each
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(40u32);
        let with: Vec<(String, f64)> = args.text("with").unwrap_or("").split(',').filter_map(|s| s.split_once(':')).filter_map(|(k, v)| Some((k.trim().to_string(), v.trim().parse().ok()?))).collect();
        let mut e = headless(&args)?;
        let (without, ms) = e.bench_with(&with, n, args.num("rounds").unwrap_or(6.0) as u32);
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"withoutMs\":{without:.3},\"withMs\":{ms:.3},\"times\":{:.3}}}", e.state.family().id, e.state.preset().id, ms / without);
        return Ok(());
    }
    if let Some(n) = args.flag("bench-parts") {
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(100u32);
        let mut e = headless(&args)?;
        let (trace, post) = e.bench_parts(n);
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"traceAndFoldMs\":{trace:.3},\"bloomAndScreenMs\":{post:.3}}}", e.state.family().id, e.state.preset().id);
        return Ok(());
    }
    if let Some(Some(dir)) = args.flag("motion-seq") {
        // The live view through a change of motion, frame by frame, with a settled picture of each frame
        // to measure its grain against. The picture first settles; then `move=play` starts the animation
        // (`move=turn` starts the view spinning instead, `move=both` does both), `play=` frames are drawn as the window draws them
        // (a 60th of a second apart), the motion stops and `after=` more are drawn. Every frame is written
        // as <dir>/live_NNN.png; frames listed in `refs=` (a,b,c; numbered on from the first moving frame)
        // also get <dir>/ref_NNN.png: the same moment settled over `refFrames=` frames. A line of JSON a
        // frame goes to <dir>/frames.jsonl.
        let mut e = headless(&args)?;
        e.auto = false;
        let ((w, h), _) = e.renderer.size();
        let (play, after) = (args.num("play").unwrap_or(90.0) as usize, args.num("after").unwrap_or(90.0) as usize);
        // (move=both: spinning and playing at once, as the window does by default when play is pressed)
        let turn = matches!(args.text("move"), Some("turn" | "both"));
        let anim = !matches!(args.text("move"), Some("turn"));
        let spin = args.num("spin").unwrap_or(0.25);
        std::fs::create_dir_all(dir).map_err(|err| format!("{dir}: {err}"))?;
        let save = |e: &Engine, name: &str| -> Result<(), String> {
            let (w, h, rgba) = e.screenshot()?;
            write_png(&format!("{dir}/{name}.png"), w, h, &rgba)
        };
        e.render(args.num("frames").unwrap_or(64.0) as u32);
        save(&e, "settled")?;
        let mut log = String::new();
        // (clock, angle, the detail it was built at) for every frame
        let mut when: Vec<(f64, f64, Detail)> = Vec::new();
        let mut detail = Detail::Full;
        if turn { e.state.p.spin = spin; }
        if anim { e.state.set_playing(true); }
        for i in 0..play + after {
            if i == play {
                e.state.p.spin = 0.0;
                e.state.set_playing(false);
            }
            e.state.tick(1.0 / 60.0);
            if let Some(d) = e.state.want { detail = d; }
            let t0 = Instant::now();
            e.frame(w, h, 0.0);
            e.gpu.wait();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            when.push((e.state.clock, e.state.theta, detail));
            log += &format!("{{\"i\":{i},\"moving\":{},\"accum\":{},\"detail\":\"{detail:?}\",\"ms\":{ms:.2}}}
", i < play, e.renderer.accum_n);
            save(&e, &format!("live_{i:03}"))?;
        }
        std::fs::write(format!("{dir}/frames.jsonl"), log).map_err(|err| format!("{dir}: {err}"))?;
        let refs: Vec<usize> = args.text("refs").unwrap_or("0,1,2,4,8,16,32,64,89,90,91,92,94,98,106,122,150,179").split(',').filter_map(|x| x.trim().parse().ok()).filter(|i| *i < when.len()).collect();
        let n_ref = args.num("refFrames").unwrap_or(256.0) as u32;
        e.state.set_playing(false);
        e.state.p.spin = 0.0;
        for i in refs {
            let (clock, theta, d) = when[i];
            e.state.clock = clock;
            e.state.theta = theta;
            e.state.update_r();
            e.state.apply_time();
            e.state.want = None;
            e.bake(d)?;
            e.render(n_ref);
            save(&e, &format!("ref_{i:03}"))?;
        }
        return Ok(());
    }
    if let Some(n) = args.flag("session") {
        // A live session timed frame by frame, as the window runs it: `n` frames of the view being turned
        // by hand, `n` of the animation playing, then `n` after everything stops (the picture settling,
        // then idling once it has). Every frame is waited for, so each time is that frame's own.
        // Prints the 50th, 80th and 95th percentile and the slowest, for each part and for all.
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(240usize);
        let mut e = headless(&args)?;
        e.auto = false;
        let ((w, h), _) = e.renderer.size();
        let turn = gf_core::math::axis_angle(0.3, 1.0, 0.2, 0.6 / 120.0);
        let dt = 1.0 / 90.0;
        // (the first picture, and whatever the scene builds once, before the clock starts)
        e.render(4);
        let mut parts: Vec<(&str, Vec<f64>)> = Vec::new();
        for part in ["turn", "play", "settle"] {
            if part == "play" { e.state.set_playing(true); }
            if part == "settle" { e.state.set_playing(false); }
            let mut ms = Vec::with_capacity(n);
            for _ in 0..n {
                let t0 = std::time::Instant::now();
                if part == "turn" { e.state.turn(&turn); }
                e.frame(w, h, dt);
                e.gpu.wait();
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            parts.push((part, ms));
        }
        let pct = |v: &[f64], q: f64| -> f64 {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[((q * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)]
        };
        let all: Vec<f64> = parts.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let mut out = format!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"w\":{w},\"h\":{h}", e.state.family().id, e.state.preset().id);
        for (name, v) in parts.iter().map(|(a, b)| (*a, b.as_slice())).chain(std::iter::once(("all", all.as_slice()))) {
            out += &format!(",\"{name}\":{{\"p50\":{:.2},\"p80\":{:.2},\"p95\":{:.2},\"max\":{:.2}}}", pct(v, 0.5), pct(v, 0.8), pct(v, 0.95), pct(v, 1.0));
        }
        println!("{out}}}");
        return Ok(());
    }
    if let Some(n) = args.flag("bench-anim") {
        let n = n.as_ref().and_then(|v| v.parse().ok()).unwrap_or(60u32);
        let mut e = headless(&args)?;
        let (ms, bake) = e.bench_anim(n);
        let m = &e.state.shape().meta;
        println!("{{\"scene\":\"{}\",\"preset\":\"{}\",\"frames\":{n},\"ms\":{ms:.3},\"shapeMs\":{bake:.3},\"volume\":{}{}}}", e.state.family().id, e.state.preset().id, m.res, e.layers_json());
        return Ok(());
    }
    glassforms::app::run(args.pairs, args.flags)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("glassforms: {e}");
        std::process::exit(1);
    }
}
