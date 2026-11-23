use glassforms::gpu::Gpu;

#[test]
fn rough_forward_mirror_preserves_and_reciprocates_diffuse_irradiance() {
    let gpu = Gpu::headless("vulkan").expect("optical qualification requires Vulkan");
    let forward = include_str!("../src/shaders/cast_light.wgsl");
    let start = forward.find("fn castConductorWalk(").unwrap();
    let end = forward[start..].find("// What a ray inside body").unwrap() + start;
    let mirror = "fn conductorReflectance(j:i32,c:f32,eta:f32)->vec3f { return vec3f(1); }";
    let source = [
        include_str!("../src/shaders/pbr/hash.wgsl"),
        include_str!("../src/shaders/pcg.wgsl"),
        include_str!("../src/shaders/ggx_microsurface.wgsl"),
        mirror,
        &forward[start..end],
    ]
    .join("\n");
    let source = source
        + r#"
@group(0) @binding(0) var<storage,read_write> events: array<atomic<u32>>;
@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) id: vec3u) {
  let i=id.x;
  let roughness=array<f32,4>(0.05,0.35,0.65,0.9)[id.y];
  var state=pcg(i+14689117u*id.y);
  let ci=sqrt(ggxUniform(&state));
  let phi=6.283185307*ggxUniform(&state);
  let xy=sqrt(1.0-ci*ci);
  let incident=vec3f(xy*cos(phi),xy*sin(phi),-ci);
  var weight=vec3f(1.0);
  let outgoing=castConductorWalk(0,incident,vec3f(0,0,1),1.0,roughness*roughness,&state,&weight);
  if(weight.x==1.0 && outgoing.z>=0.0 && abs(length(outgoing)-1.0)<0.0005) {
    let src=min(15u,u32(16.0*ci*ci));
    let dst=min(15u,u32(16.0*outgoing.z*outgoing.z));
    atomicAdd(&events[256u*id.y+16u*src+dst],1u);
  }
}"#;
    let shader = gpu.shader("rough reflected photon transport", &source, false);
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("measure"),
            compilation_options: Default::default(),
            cache: None,
        });
    let size = 4096;
    let counts = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: counts.as_entire_binding(),
        }],
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(16384, 4, 1);
    }
    commands.copy_buffer_to_buffer(&counts, 0, &download, 0, size);
    gpu.queue.submit(Some(commands.finish()));
    let (send, recv) = std::sync::mpsc::channel();
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap()
        });
    gpu.wait();
    recv.recv().unwrap().unwrap();
    let mapped = download.slice(..).get_mapped_range().unwrap();
    let counts: &[u32] = bytemuck::cast_slice(&mapped);
    for (rough, bins) in [0.05, 0.35, 0.65, 0.9].iter().zip(counts.chunks_exact(256)) {
        let launched = 1_048_576;
        let accounted: u32 = bins.iter().sum();
        let output: Vec<u32> = (0..16)
            .map(|j| (0..16).map(|i| bins[i * 16 + j]).sum())
            .collect();
        let max_bias = output
            .iter()
            .map(|&v| (f64::from(v) / 65536.0 - 1.0).abs())
            .fold(0.0, f64::max);
        // Equal-area cosine-weighted bins exchange equal flux in either direction.
        let mut chi = 0.0;
        let mut pairs = 0;
        for i in 0..16 {
            for j in (i + 1)..16 {
                let (a, b) = (f64::from(bins[i * 16 + j]), f64::from(bins[j * 16 + i]));
                if a + b >= 20.0 {
                    chi += (a - b) * (a - b) / (a + b);
                    pairs += 1;
                }
            }
        }
        println!("roughness {rough}, accounted {accounted}/{launched}, angular bin deviation {max_bias:.5}, reciprocity chi {chi:.2}/{pairs}");
        assert!(
            accounted >= launched - 2
                && max_bias < 0.025
                && chi < f64::from(pairs) + 6.0 * (2.0 * f64::from(pairs)).sqrt() + 6.0
        );
    }
}
