//! A shape exported as a mesh (gf_core::export makes it): the volume the tracer draws, copied aside on
//! the graphics device, read back a slice at a time and contoured on a thread of its own, so that the
//! window carries on meanwhile. The mesh is written and forgotten; nothing reads it back.

use std::sync::mpsc::{channel, Receiver};
use std::time::Instant;

use gf_core::export::{check, surface_net, write, Check};

use crate::bake::Volume;
use crate::gpu::*;

/// What an export came to.
#[derive(Clone, Debug)]
pub struct Done {
    pub path: String,
    pub res: u32,
    pub check: Check,
    pub ms: f64,
}

/// An export under way.
pub struct Exporting {
    pub path: String,
    rx: Receiver<Result<Done, String>>,
}
impl Exporting {
    /// Its result once it has one (only once).
    pub fn poll(&self) -> Option<Result<Done, String>> {
        self.rx.try_recv().ok()
    }
    /// Wait for it.
    pub fn wait(self) -> Result<Done, String> {
        self.rx.recv().map_err(|_| "the export stopped".to_string())?
    }
}

/// Start exporting this volume (`res` voxels a side, spanning [-hb, hb]) to `path` (.obj or .stl). The
/// volume is copied first, so that the shape may change while the copy is read.
pub fn start(gpu: &Gpu, vol: &Volume, hb: f64, path: &str) -> Exporting {
    let res = vol.res;
    let copy = gpu.volume("export", res, res, crate::bake::VOL_FORMAT, false);
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export copy") });
    enc.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo { texture: &vol.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyTextureInfo { texture: &copy, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
    );
    gpu.queue.submit([enc.finish()]);
    let (tx, rx) = channel();
    let (gpu, path_owned) = (gpu.clone(), path.to_string());
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let mut failed: Option<String> = None;
        let mesh = surface_net(res as usize, hb, |z| match gpu.read(&copy, 0, 0, z as u32, res, res, 2) {
            Ok(bytes) => bytemuck::cast_slice::<u8, half::f16>(&bytes).iter().map(|h| h.to_f32()).collect(),
            Err(e) => {
                failed.get_or_insert(e);
                Vec::new()
            }
        });
        copy.destroy();
        let out = match failed {
            Some(e) => Err(e),
            None => write(&mesh, &path_owned).map(|_| Done { path: path_owned, res, check: check(&mesh), ms: t0.elapsed().as_secs_f64() * 1e3 }),
        };
        let _ = tx.send(out);
    });
    Exporting { path: path.to_string(), rx }
}

/// What an export came to, in a line.
pub fn says(d: &Done) -> String {
    let whole = if d.check.open == 0 { "closed".to_string() } else { format!("{} open edges", d.check.open) };
    format!("{} triangles, {} points, {whole}, from {} voxels a side, in {:.1} s: {}", d.check.tris, d.check.points, d.res, d.ms / 1e3, d.path)
}
