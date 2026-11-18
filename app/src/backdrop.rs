//! The frosted glass the panels and the floating windows are made of: a small, blurred and toned copy of
//! the picture, made once a frame (only when the picture or the look has changed) and shared by every
//! panel, window and bar, each of which shows the part of it that lies under it.
//!
//! The blur is a dual-Kawase one (shaders/backdrop.wgsl): the picture is halved `levels` times, then
//! doubled back up to a quarter of its size, where it is kept. The last pass also sets the theme's tone,
//! pressing the picture's range into a band dark enough (or, in the light theme, light enough) for the
//! words over it to stay readable over the brightest picture.

use bytemuck::{Pod, Zeroable};

use crate::gpu::*;

const WGSL: &str = include_str!("shaders/backdrop.wgsl");
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How the frost looks: how far the picture is blurred, and the tone it is given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// how many times the picture is halved (2: a quarter of its size, the least; each one more doubles
    /// the blur's reach)
    pub levels: u32,
    /// how far apart the taps are, in texels of each level (1 is the usual Kawase spread)
    pub spread: f32,
    /// the darkest and the brightest colour the frost comes to (display values, 0 to 1)
    pub lo: [f32; 3],
    pub hi: [f32; 3],
    /// saturation: 1 leaves the colours as they are
    pub sat: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct U {
    texel: [f32; 4],
    lo: [f32; 4],
    hi: [f32; 4],
}

/// The textures and bind groups for one size of picture and one number of levels.
struct Chain {
    /// the picture they were made for, and its size
    src: wgpu::Texture,
    levels: u32,
    /// level k (from 1) is the picture at 1 / 2^k
    tex: Vec<wgpu::Texture>,
    out: wgpu::Texture,
    /// one for each pass, down and then up, with its own uniforms
    passes: Vec<(wgpu::BindGroup, wgpu::Buffer, wgpu::TextureView, [f32; 2])>,
}

pub struct Backdrop {
    gpu: Gpu,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    chain: Option<Chain>,
    /// what the copy was last made from: which look (the picture is told apart by the caller)
    pub look: Option<Look>,
    /// bumped whenever the copy is a new texture, so that whoever shows it knows to take it up again
    pub generation: u64,
}

fn size_at(w: u32, h: u32, k: u32) -> (u32, u32) {
    ((w >> k).max(1), (h >> k).max(1))
}

impl Backdrop {
    pub fn new(gpu: &Gpu) -> Self {
        let module = gpu.shader("backdrop", WGSL, true);
        let layout = gpu.bind_layout("backdrop", &[
            texture_entry(0, wgpu::TextureViewDimension::D2, true),
            sampler_entry(1),
            uniform_entry(2, wgpu::ShaderStages::FRAGMENT, false),
        ]);
        let pipe = |fs: &str| gpu.pipeline(&PipeDesc { label: "backdrop", module: &module, vs: "vs", fs, layouts: &[&layout], format: FORMAT, blend: None, strip: false, buffers: &[] });
        Backdrop {
            down: pipe("down"),
            up: pipe("up"),
            sampler: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            layout,
            gpu: gpu.clone(),
            chain: None,
            look: None,
            generation: 0,
        }
    }

    /// The frosted copy, a quarter of the picture's size, once one has been made.
    pub fn texture(&self) -> Option<&wgpu::Texture> {
        self.chain.as_ref().map(|c| &c.out)
    }

    fn build(&mut self, src: &wgpu::Texture, levels: u32) {
        let gpu = &self.gpu;
        let (w, h) = (src.width(), src.height());
        let tex: Vec<wgpu::Texture> = (1..=levels).map(|k| { let (a, b) = size_at(w, h, k); gpu.target("backdrop level", a, b, FORMAT, &[]) }).collect();
        let (ow, oh) = size_at(w, h, 2);
        let out = gpu.target("backdrop", ow, oh, FORMAT, &[]);
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let mut passes = Vec::new();
        let mut pass = |from: &wgpu::Texture, to: &wgpu::Texture| {
            let buf = gpu.uniform_buffer("backdrop", std::mem::size_of::<U>() as u64);
            let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("backdrop"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view(from)) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
                ],
            });
            passes.push((bind, buf, view(to), [1.0 / from.width() as f32, 1.0 / from.height() as f32]));
        };
        // down: the picture, then each level into the next
        pass(src, &tex[0]);
        for k in 1..levels as usize { pass(&tex[k - 1], &tex[k]); }
        // up: from the smallest level back to a quarter of the picture's size, each into the level above
        // (no longer needed once the way down is done); the last into the copy that is kept
        let mut k = levels as usize;
        while k > 2 {
            if k - 1 == 2 { pass(&tex[k - 1], &out); } else { pass(&tex[k - 1], &tex[k - 2]); }
            k -= 1;
        }
        if levels <= 2 { pass(&tex[levels as usize - 1], &out); }
        self.chain = Some(Chain { src: src.clone(), levels, tex, out, passes });
        self.generation += 1;
    }

    /// Make the frosted copy of `src` (the picture as the window shows it) with this look. Returns false
    /// if the picture is too small to frost.
    pub fn make(&mut self, src: &wgpu::Texture, look: &Look) -> bool {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        if !self.encode(&mut enc, src, look) { return false; }
        self.gpu.queue.submit([enc.finish()]);
        self.look = Some(*look);
        true
    }

    /// The same, written into an encoder the caller submits.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, src: &wgpu::Texture, look: &Look) -> bool {
        if src.width() < 16 || src.height() < 16 { return false; }
        let levels = look.levels.clamp(2, 6);
        if !self.chain.as_ref().is_some_and(|c| c.src == *src && c.levels == levels) { self.build(src, levels); }
        let Some(c) = &self.chain else { return false };
        let n = c.passes.len();
        let downs = levels as usize;
        for (i, (bind, buf, to, texel)) in c.passes.iter().enumerate() {
            let last = i + 1 == n;
            let u = U {
                texel: [texel[0], texel[1], look.spread, 0.0],
                lo: [look.lo[0], look.lo[1], look.lo[2], if last { 1.0 } else { 0.0 }],
                hi: [look.hi[0], look.hi[1], look.hi[2], look.sat],
            };
            self.gpu.queue.write_buffer(buf, 0, bytemuck::bytes_of(&u));
            let mut p = pass(enc, to, None, false);
            p.set_pipeline(if i < downs { &self.down } else { &self.up });
            p.set_bind_group(0, bind, &[]);
            p.draw(0..3, 0..1);
        }
        let _ = &c.tex;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look(levels: u32) -> Look {
        Look { levels, spread: 1.0, lo: [0.1, 0.1, 0.1], hi: [0.5, 0.5, 0.5], sat: 1.0 }
    }

    /// The copy is a quarter of the picture's size, and a white picture comes out as the top of the band,
    /// whatever the number of levels.
    #[test]
    fn a_white_picture_comes_out_at_the_top_of_the_band() {
        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        let src = gpu.target("white", 256, 128, wgpu::TextureFormat::Rgba8Unorm, &[]);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let view = src.create_view(&Default::default());
        {
            let _p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        gpu.queue.submit([enc.finish()]);
        let mut b = Backdrop::new(&gpu);
        for levels in [2, 3, 5] {
            assert!(b.make(&src, &look(levels)));
            let out = b.texture().unwrap();
            assert_eq!((out.width(), out.height()), (64, 32));
            let bytes = gpu.read(out, 0, 0, 0, 64, 32, 8).unwrap();
            let px: Vec<f32> = bytes.chunks(2).map(|h| half::f16::from_le_bytes([h[0], h[1]]).to_f32()).collect();
            for p in px.chunks(4) {
                for c in &p[..3] { assert!((c - 0.5).abs() < 1e-2, "levels {levels}: {p:?}"); }
            }
        }
    }

    /// What making the copy costs at 4K, the GPU's own time for many copies in a row, divided. Run alone:
    /// `cargo test --release -p glassforms backdrop::tests::cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cost() {
        let gpu = Gpu::headless("vulkan").expect("a GPU");
        let src = gpu.target("picture", 3840, 2160, wgpu::TextureFormat::Rgba8Unorm, &[]);
        let mut b = Backdrop::new(&gpu);
        for levels in [3, 4, 5] {
            let l = Look { levels, ..look(levels) };
            for _ in 0..20 { b.make(&src, &l); }
            gpu.wait();
            let n = 400;
            let t = std::time::Instant::now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            for _ in 0..n { b.encode(&mut enc, &src, &l); }
            gpu.queue.submit([enc.finish()]);
            gpu.wait();
            println!("backdrop at 3840x2160, {levels} levels: {:.4} ms a copy", t.elapsed().as_secs_f64() * 1e3 / n as f64);
        }
    }
}
