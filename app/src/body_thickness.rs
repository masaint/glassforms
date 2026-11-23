use crate::bake::Volume;
use crate::gpu::Gpu;
use std::sync::OnceLock;

/// A fitted material uses five times its mean interior distance. Reduce the
/// texture before readback so the CPU receives tile totals instead of voxels.
#[derive(Default)]
pub(crate) struct BodyThickness(OnceLock<wgpu::ComputePipeline>);

impl BodyThickness {
    pub fn measure(&self, gpu: &Gpu, volume: &Volume, bodies: usize) -> Option<Vec<f64>> {
        let device = &gpu.device;
        let pipe = self.0.get_or_init(|| {
            let module = gpu.shader("body thickness", include_str!("shaders/body_thickness.wgsl"), false);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("body thickness"), layout: None, module: &module,
                entry_point: Some("measure"), compilation_options: Default::default(), cache: None,
            })
        });
        let side = volume.res.div_ceil(8);
        let size = u64::from(side).pow(2) * 32;
        let buffer = |usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body thickness tile totals"), size, usage, mapped_at_creation: false,
        });
        let rows = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let read = buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&volume.view) },
                wgpu::BindGroupEntry { binding: 1, resource: rows.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("measure body thickness"), timestamp_writes: None });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(side, side, 1);
        }
        enc.copy_buffer_to_buffer(&rows, 0, &read, 0, size);
        gpu.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        gpu.wait();
        rx.recv().ok()?.ok()?;
        let mapped = read.slice(..).get_mapped_range().ok()?;
        let mut depth = [0.0; 4];
        let mut count = [0u64; 4];
        for row in bytemuck::cast_slice::<u8, [u32; 8]>(&mapped) {
            for k in 0..bodies {
                depth[k] += f64::from(f32::from_bits(row[k]));
                count[k] += u64::from(row[k + 4]);
            }
        }
        drop(mapped);
        read.unmap();
        Some((0..bodies).map(|k| 5.0 * depth[k] / count[k].max(1) as f64).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_rows_preserve_each_bodys_measured_absorption_distance() {
        let Ok(gpu) = Gpu::headless("vulkan") else { return };
        let n = 73u32;
        let tex = gpu.volume("fitted bodies", n, n, wgpu::TextureFormat::Rgba16Float, false);
        let volume = Volume { view: tex.create_view(&Default::default()), tex, res: n };
        let mut voxels = Vec::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let slab = if z == 3 { -0.123 } else { 1.0 };
                    let channels = [(x as f32 - y as f32) * 0.013, (z as f32 - 32.) * 0.008, slab, 1.0];
                    voxels.push(channels.map(half::f16::from_f32));
                }
            }
        }
        gpu.queue.write_texture(volume.tex.as_image_copy(), bytemuck::cast_slice(&voxels),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 8), rows_per_image: Some(n) },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n });
        let measured = BodyThickness::default().measure(&gpu, &volume, 4).unwrap();
        for k in 0..4 {
            let mut sum = 0.;
            let mut count = 0u64;
            for voxel in &voxels {
                let d = voxel[k].to_f64();
                if d < 0. { sum -= d; count += 1; }
            }
            let reference = 5. * sum / count.max(1) as f64;
            assert!((measured[k] - reference).abs() < 1e-7, "channel {k}: {} vs {reference}", measured[k]);
        }
    }
}
