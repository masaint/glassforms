use crate::gpu::Gpu;
use std::io::Write;

/// Optional per-pass GPU timings for a diagnostic session. Only a sampled
/// frame waits for readback. Run the normal frame-time gate separately.
pub(crate) struct PassTimes {
    queries: wgpu::QuerySet,
    result: wgpu::Buffer,
    read: wgpu::Buffer,
    log: std::fs::File,
}

impl PassTimes {
    pub fn from_env(gpu: &Gpu) -> Option<Self> {
        let path = std::env::var_os("GLASSFORMS_PASS_LOG")?;
        let flags = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !gpu.device.features().contains(flags) { return None; }
        let mut log = std::fs::File::create(path).ok()?;
        writeln!(log, "frame\tmoving\tscale\ttrace\tsurface\tfrost\taccum\tpost").ok()?;
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass timings"), ty: wgpu::QueryType::Timestamp, count: 6
        });
        let buffer = |usage| gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass timings"), size: 48, usage, mapped_at_creation: false
        });
        Some(Self { queries, log,
            result: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST) })
    }

    pub fn mark(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        encoder.write_timestamp(&self.queries, index);
        if index == 5 {
            encoder.resolve_query_set(&self.queries, 0..6, &self.result, 0);
            encoder.copy_buffer_to_buffer(&self.result, 0, &self.read, 0, 48);
        }
    }

    pub fn record(&mut self, gpu: &Gpu, frame: u32, moving: bool, scale: f64) {
        let slice = self.read.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        gpu.wait();
        if !matches!(rx.recv(), Ok(Ok(()))) { return; }
        if let Ok(bytes) = slice.get_mapped_range() {
            let ticks: &[u64] = bytemuck::cast_slice(&bytes);
            let _ = write!(self.log, "{frame}\t{moving}\t{scale:.2}");
            for t in ticks.windows(2) {
                let ms = t[1].saturating_sub(t[0]) as f64 * f64::from(gpu.queue.get_timestamp_period()) / 1e6;
                let _ = write!(self.log, "\t{ms:.3}");
            }
            let _ = writeln!(self.log);
        }
        self.read.unmap();
    }
}
