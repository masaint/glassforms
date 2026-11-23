//! The lens's arithmetic as the shaders do it (shaders/camera.wgsl) against the same in plain Rust
//! (gf_core::camera), on the GPU, for every kind of lens and with straight lines bowed either way:
//!   - every pixel's ray is the ray the Rust camera gives;
//!   - a point along that ray, taken back to the picture, lands on the pixel it came from (what the pass
//!     that follows the surface from frame to frame relies on), and is as far along as it was put;
//!   - the blur of glass at that point is the blur the Rust camera gives (what the focus stage draws from).

use gf_core::camera::{Camera, Projection};
use gf_core::settings::Settings;
use glassforms::camera::LENS_WGSL;
use glassforms::gpu::*;

const W: u32 = 96;
const H: u32 = 64;

const SHADER: &str = "
struct T {
  cam: Cam,
  res: vec2f,
  t: f32,          // how far along each ray the point is put
  k: f32,          // lens radius over pixel size
  band: vec2f,     // the band in focus
  pad: vec2f,
}
@group(0) @binding(0) var<uniform> tu: T;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// rgb: the pixel's ray; a: how far (in pixels) the point along it lands from the pixel when taken back,
// or how far off its distance along the ray is (whichever is worse); -1 where the lens draws nothing
@fragment
fn fs_ray(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let fc = vec2f(pos.x, tu.res.y - pos.y);
  let film = (2.0 * fc - tu.res) / tu.res.y + tu.cam.b.xy;
  var ro: vec3f;
  var rd: vec3f;
  if (!camRay(film, tu.cam, &ro, &rd)) { return vec4f(0.0, 0.0, 0.0, -1.0); }
  let p = ro + rd * tu.t;
  let f = camFilm(p, tu.cam);
  let back = ((f.xy - tu.cam.b.xy) * tu.res.y + tu.res) * 0.5;
  let miss = select(1e6, length(back - fc), f.z > 0.5);
  return vec4f(rd, max(miss, abs(camAlong(p, tu.cam) - tu.t)));
}

// r: the blur of glass at that point, in pixels; g: its depth the way focus measures it; b: where the ray starts (x)
@fragment
fn fs_blur(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let fc = vec2f(pos.x, tu.res.y - pos.y);
  let film = (2.0 * fc - tu.res) / tu.res.y + tu.cam.b.xy;
  var ro: vec3f;
  var rd: vec3f;
  if (!camRay(film, tu.cam, &ro, &rd)) { return vec4f(0.0, 0.0, 0.0, -1.0); }
  return vec4f(tu.k * camBlur(ro, rd, tu.t, tu.band.x, tu.band.y, tu.cam), camDepth(ro + rd * tu.t, tu.cam), ro.x, 1.0);
}
";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TU {
    cam: [[f32; 4]; 4],
    res: [f32; 2],
    t: f32,
    k: f32,
    band: [f32; 2],
    pad: [f32; 2],
}

struct Rig {
    gpu: Gpu,
    u: wgpu::Buffer,
    group: wgpu::BindGroup,
    ray: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    tex: wgpu::Texture,
}

impl Rig {
    fn new() -> Rig {
        let gpu = Gpu::headless("vulkan").expect("a graphics device");
        let layout = gpu.bind_layout("lens test", &[uniform_entry(0, wgpu::ShaderStages::FRAGMENT, false)]);
        let module = gpu.shader("lens test", &format!("{SHADER}{LENS_WGSL}"), false);
        let format = wgpu::TextureFormat::Rgba32Float;
        let pipe = |fs| gpu.pipeline(&PipeDesc { label: "lens test", module: &module, vs: "vs", fs, layouts: &[&layout], format, blend: None, strip: false, buffers: &[] });
        let u = gpu.uniform_buffer("lens test", std::mem::size_of::<TU>() as u64);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }] });
        let (ray, blur) = (pipe("fs_ray"), pipe("fs_blur"));
        let tex = gpu.target("lens test", W, H, format, &[]);
        Rig { gpu, u, group, ray, blur, tex }
    }

    /// One of the two pictures for a camera: `W` x `H` RGBA floats, top row first.
    fn draw(&self, c: &Camera, t: f64, blur: bool) -> Vec<f32> {
        let tu = TU { cam: c.uniform(1.0, 0.0, 0.0), res: [W as f32, H as f32], t: t as f32, k: (c.lens / c.pixel(H as f64)) as f32, band: [c.near as f32, c.far as f32], pad: [0.0; 2] };
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&tu));
        let view = self.tex.create_view(&Default::default());
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut rp = pass(&mut enc, &view, None, true);
            rp.set_pipeline(if blur { &self.blur } else { &self.ray });
            rp.set_bind_group(0, &self.group, &[]);
            rp.draw(0..3, 0..1);
        }
        self.gpu.queue.submit([enc.finish()]);
        bytemuck::pod_collect_to_vec(&self.gpu.read(&self.tex, 0, 0, 0, W, H, 16).unwrap())
    }
}

/// The middle of a pixel, across and up from the picture's middle in half-heights.
fn uv_of(x: u32, y: u32) -> [f64; 2] {
    let (fx, fy) = (x as f64 + 0.5, H as f64 - (y as f64 + 0.5));
    [(2.0 * fx - W as f64) / H as f64, (2.0 * fy - H as f64) / H as f64]
}

fn settings(proj: Projection, distort: f64) -> Settings {
    let mut p = Settings::default();
    (p.dist, p.fov, p.shift, p.distort) = (4.0, 50.0, [0.06, -0.03], distort);
    p.proj = proj.index() as f64;
    (p.aperture, p.fnear, p.ffar) = (0.08, -0.2, 0.3);
    p
}

#[test]
fn the_shaders_lens_is_the_rust_lens() {
    let rig = Rig::new();
    for proj in Projection::ALL {
        for (distort, tilt) in [(0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 30.0)] {
            let mut p = settings(proj, distort);
            (p.tilt_x, p.tilt_y) = (tilt, -tilt / 2.0);
            let c = Camera::of(&p);
            for t in [1.3, 4.2, 7.5] {
                let rays = rig.draw(&c, t, false);
                let blurs = rig.draw(&c, t, true);
                let (mut seen, mut worst_ray, mut worst_miss, mut worst_blur) = (0, 0.0f64, 0.0f64, 0.0f64);
                for y in 0..H {
                    for x in 0..W {
                        let i = ((y * W + x) * 4) as usize;
                        let uv = uv_of(x, y);
                        let Some((o, d)) = c.ray(uv) else {
                            assert!(rays[i + 3] < 0.0, "{proj:?} {distort}: the shader draws at ({x}, {y}) where the lens has no picture");
                            continue;
                        };
                        // (right at the edge of a fisheye's circle the two may disagree on which side a pixel is)
                        if rays[i + 3] < 0.0 { continue; }
                        seen += 1;
                        for k in 0..3 { worst_ray = worst_ray.max((rays[i + k] as f64 - d[k]).abs()); }
                        // a panorama's poles and the point straight behind a fisheye are whole rows or rings
                        // of the picture at once: nothing there has one pixel to come back to
                        let pole = (proj == Projection::Panorama && d[1].abs() > 0.995) || (proj.fisheye() && d[2] > 0.99);
                        // the very edge of what a lens with bowed lines can show does not come back either
                        let edge = c.project(c.point(uv, t).unwrap()).is_none();
                        if !pole && !edge { worst_miss = worst_miss.max(rays[i + 3] as f64); }
                        let want = c.blur(uv, t, H as f64).unwrap();
                        worst_blur = worst_blur.max((blurs[i] as f64 - want).abs() / (1.0 + want.abs()));
                        assert!((blurs[i + 1] as f64 - c.depth_at(uv, t).unwrap()).abs() < 2e-5, "{proj:?}: depth");
                        assert!((blurs[i + 2] as f64 - o[0]).abs() < 1e-5, "{proj:?}: where the ray starts");
                    }
                }
                assert!(seen > (W * H / 3) as usize, "{proj:?} distort {distort}: only {seen} pixels have a picture");
                assert!(worst_ray < 2e-5, "{proj:?} distort {distort} tilt {tilt}: a ray is {worst_ray} off");
                assert!(worst_miss < 0.02, "{proj:?} distort {distort} tilt {tilt} t {t}: a point comes back {worst_miss} pixels from where it started");
                assert!(worst_blur < 2e-3, "{proj:?} distort {distort} tilt {tilt} t {t}: a blur is {worst_blur} off");
            }
        }
    }
}

#[test]
fn off_axis_tilted_camera_matches_and_negative_reach_stays_finite() {
    let rig = Rig::new();
    // 1. Off-axis camera (`eye != [0, 0]`) + tilted plane of focus (`#25` #8): CPU and GPU depth & blur agree.
    for proj in Projection::ALL {
        let mut st = gf_core::state::State::new();
        st.p = settings(proj, 0.15);
        (st.p.tilt_x, st.p.tilt_y) = (26.0, -17.0);
        (st.p.eye_x, st.p.eye_y) = (0.85, -0.55);
        let c = st.camera();
        let blurs = rig.draw(&c, 3.8, true);
        let mut worst_blur = 0.0f64;
        for y in 0..H {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                if blurs[i + 3] < 0.0 { continue; }
                let uv = uv_of(x, y);
                let Some(want) = c.blur(uv, 3.8, H as f64) else { continue };
                worst_blur = worst_blur.max((blurs[i] as f64 - want).abs() / (1.0 + want.abs()));
                assert!((blurs[i + 1] as f64 - c.depth_at(uv, 3.8).unwrap()).abs() < 2e-5, "{proj:?}: off-axis tilted depth");
            }
        }
        assert!(worst_blur < 2e-3, "{proj:?}: off-axis tilted blur {worst_blur}");
    }

    // 2. Planar `camReach` when focus band lies at/behind the camera or tilts past the lens (`#25` #9):
    // must never divide by zero or flip blur sign to negative for glass behind the focus plane.
    for proj in [Projection::Perspective, Projection::Orthographic, Projection::Equidistant] {
        for (near, far, tilt) in [(-5.5, -4.5, 0.0), (-4.0, -4.0, 0.0), (-3.8, -3.8, 72.0)] {
            let mut p = settings(proj, 0.0);
            (p.fnear, p.ffar, p.tilt_x) = (near, far, tilt);
            let c = Camera::of(&p);
            let blurs = rig.draw(&c, 2.5, true);
            for y in 0..H {
                for x in 0..W {
                    let i = ((y * W + x) * 4) as usize;
                    for k in 0..4 {
                        assert!(blurs[i + k].is_finite(), "{proj:?} ({near}, {far}, {tilt}) at ({x},{y}): channel {k} is {}", blurs[i + k]);
                    }
                    if blurs[i + 3] > 0.0 && near == -5.5 {
                        assert!(blurs[i] > 0.0, "{proj:?}: glass at t=2.5 in front of camera lies behind focus band (-5.5..-4.5), got blur {}", blurs[i]);
                    }
                }
            }
        }
    }
}

#[test]
fn denoiser_clamps_f16_variance_and_reconstructs_orthographic_planes() {
    use glassforms::denoise::{Denoise, Tune};

    const WRITE_NOISY: &str = "
struct Mode { kind: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> m: Mode;
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  if (m.kind < 0.5) {
    // Mode 0: extreme single-pixel firefly (lum=1000) on a 3x3 island at t=4.0 surrounded by a step to t=8.0
    if (c.x == 32 && c.y == 32) { return vec4f(1000.0, 1000.0, 1000.0, 4.0); }
    if (abs(c.x - 32) <= 1 && abs(c.y - 32) <= 1) { return vec4f(1.0, 1.0, 1.0, 4.0); }
    return vec4f(0.2, 0.3, 0.4, 8.0);
  }
  // Mode 1: noisy checkerboard on a flat orthographic front plane at constant t = 2.0 (z > 0 relative to origin when dist = 5.0)
  let hi = f32((c.x + c.y) & 1);
  let val = select(0.2, 0.8, hi > 0.5);
  return vec4f(val, val, val, 2.0);
}
";

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestAccumU {
        head: [[f32; 4]; 24],
        lens: [[f32; 4]; 4],
        lens_p: [[f32; 4]; 4],
    }

    let gpu = Gpu::headless("vulkan").expect("a graphics device");
    let lin = gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge);
    let pic = [
        gpu.target("pic0", W, H, wgpu::TextureFormat::Rgba32Float, &[]),
        gpu.target("pic1", W, H, wgpu::TextureFormat::Rgba32Float, &[]),
    ];
    let views = [pic[0].create_view(&Default::default()), pic[1].create_view(&Default::default())];
    let accum_u = gpu.uniform_buffer("accum_u", std::mem::size_of::<TestAccumU>() as u64);

    let mode_u = gpu.uniform_buffer("mode_u", 16);
    let write_layout = gpu.bind_layout("write noisy", &[uniform_entry(0, wgpu::ShaderStages::FRAGMENT, false)]);
    let write_mod = gpu.shader("write noisy", WRITE_NOISY, false);
    let write_pipe = gpu.pipeline(&PipeDesc {
        label: "write noisy", module: &write_mod, vs: "vs", fs: "fs", layouts: &[&write_layout],
        format: wgpu::TextureFormat::Rgba32Float, blend: None, strip: false, buffers: &[],
    });
    let write_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &write_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: mode_u.as_entire_binding() }],
    });

    let mut p = Settings::default();
    p.proj = Projection::Orthographic.index() as f64;
    (p.dist, p.fov) = (5.0, 48.0);
    let c_ortho = Camera::of(&p);
    let mut au = TestAccumU { head: [[0.0; 4]; 24], lens: c_ortho.uniform(1.0, 0.0, 0.0), lens_p: c_ortho.uniform(1.0, 0.0, 0.0) };
    // res = (W, H), cam = (dist, t) in head[0]; cam_p = (dist, t) in head[1]; w = 1.0, moving = 0.0 in head[2]
    au.head[0] = [W as f32, H as f32, c_ortho.dist as f32, c_ortho.t as f32];
    au.head[1] = [c_ortho.dist as f32, c_ortho.t as f32, 0.0, 0.0];
    au.head[2] = [0.0, 0.0, 1.0, 0.0];
    gpu.queue.write_buffer(&accum_u, 0, bytemuck::bytes_of(&au));

    let tune = Tune { keep: 32.0, clampk: 0.0, sigma_l: 8.0, sigma_z: 1.0, sigma_n: 16.0, young: 4.0, passes: 5, chroma: false, moving_only: false };
    let lens_temporal = || glassforms::camera::accum(&glassforms::denoise::temporal_source(glassforms::render::ACCUM_SOURCE));

    // Case A (`#25` #2): extreme firefly variance > f16::MAX next to a depth step must produce zero NaN/Inf pixels.
    let mut dn = Denoise::new(&gpu);
    dn.resize(W, H, [&views[0], &views[1]], &accum_u);
    gpu.queue.write_buffer(&mode_u, 0, bytemuck::bytes_of(&[0.0f32, 0.0, 0.0, 0.0]));
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let fv = dn.frame_view(&lin).unwrap().clone();
        let _ = pass(&mut enc, &views[0], None, true);
        let mut rp = pass(&mut enc, &fv, None, true);
        rp.set_pipeline(&write_pipe);
        rp.set_bind_group(0, &write_bg, &[]);
        rp.draw(0..3, 0..1);
    }
    dn.time(&mut enc, true, false, [c_ortho.dist as f32, c_ortho.t as f32, 0.0, 0.0], [0.5, 0.5], tune, None);
    assert_eq!(dn.filter(&mut enc, 0, &views[1]), Some(1));
    gpu.queue.submit([enc.finish()]);
    let out: Vec<f32> = bytemuck::pod_collect_to_vec(&gpu.read(&pic[1], 0, 0, 0, W, H, 16).unwrap());
    for (idx, &val) in out.iter().enumerate() {
        assert!(val.is_finite(), "pixel {} channel {} became non-finite ({val}) after extreme firefly variance", idx / 4, idx % 4);
    }
    let bg_px = ((10 * W + 10) * 4) as usize;
    assert!((out[bg_px] - 0.2).abs() < 1e-2, "step-separated background surface should stay near 0.2, got {}", out[bg_px]);

    // Case B (`#25` #10): orthographic flat plane at t = 2.0 (< dist = 5.0) denoises smoothly across the view.
    let mut dn_ortho = Denoise::new(&gpu);
    dn_ortho.resize(W, H, [&views[0], &views[1]], &accum_u);
    gpu.queue.write_buffer(&mode_u, 0, bytemuck::bytes_of(&[1.0f32, 0.0, 0.0, 0.0]));
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let fv = dn_ortho.frame_view(&lin).unwrap().clone();
        let _ = pass(&mut enc, &views[0], None, true);
        let mut rp = pass(&mut enc, &fv, None, true);
        rp.set_pipeline(&write_pipe);
        rp.set_bind_group(0, &write_bg, &[]);
        rp.draw(0..3, 0..1);
    }
    dn_ortho.time(&mut enc, true, false, [c_ortho.dist as f32, c_ortho.t as f32, 0.0, 0.0], [0.5, 0.5], tune, Some(&lens_temporal));
    assert_eq!(dn_ortho.filter(&mut enc, 0, &views[1]), Some(1));
    gpu.queue.submit([enc.finish()]);
    let out_ortho: Vec<f32> = bytemuck::pod_collect_to_vec(&gpu.read(&pic[1], 0, 0, 0, W, H, 16).unwrap());
    for y in 16..(H - 16) {
        for x in 16..(W - 16) {
            let v = out_ortho[((y * W + x) * 4) as usize];
            assert!(v.is_finite() && (v - 0.5).abs() < 0.05, "orthographic flat plane at ({x},{y}) was not smoothed to ~0.5: got {v}");
        }
    }
}
