use glassforms::gpu::Gpu;

#[test]
fn polishing_arcs_and_fingerprints_cross_cells_continuously() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let source = concat!(
        "fn mapW(p: vec3f) -> f32 { return length(p)-1.0; }\n",
        include_str!("../src/shaders/wear_pigment.wgsl"),
        include_str!("../src/shaders/wear_marks.wgsl"),
        r#"
@group(0) @binding(0) var result: texture_storage_3d<rgba32float,write>;
fn pointOnSphere(axis: u32, x: f32, angle: f32) -> vec3f {
    let r = sqrt(1.0-x*x);
    var p = vec3f(0.0);
    p[axis] = x;
    p[(axis+1u)%3u] = r*cos(angle);
    p[(axis+2u)%3u] = r*sin(angle);
    return p;
}
@compute @workgroup_size(8)
fn check(@builtin(global_invocation_id) id: vec3u) {
    let a = 6.2831853*f32(id.x)/256.0;
    let axis = id.y/3u;
    let plane = f32(i32(id.y%3u)-1);
    let l = pointOnSphere(axis, plane*0.7-0.000001, a);
    let r = pointOnSphere(axis, plane*0.7+0.000001, a);
    let nl = normalize(l + 0.03*polishingPads(l,l,0.001));
    let nr = normalize(r + 0.03*polishingPads(r,r,0.001));
    let f = pointOnSphere(axis, plane*0.45-0.000001, a);
    let g = pointOnSphere(axis, plane*0.45+0.000001, a);
    let mf = fingerMarks(f,f,0.001,1.0);
    let mg = fingerMarks(g,g,0.001,1.0);
    textureStore(result,id,vec4f(length(nl-nr),abs(mf-mg),length(nl-l),mf));
}"#,
    );
    let module = gpu.shader("wear at cell boundaries", source, false);
    let pipe = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("check"),
            compilation_options: Default::default(),
            cache: None,
        });
    let image = gpu.volume(
        "wear samples",
        256,
        1,
        wgpu::TextureFormat::Rgba32Float,
        true,
    );
    let view = image.create_view(&Default::default());
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(32, 9, 1);
    }
    gpu.queue.submit([enc.finish()]);
    let values = gpu.read(&image, 0, 0, 0, 256, 9, 16).unwrap();
    let pixels: &[f32] = bytemuck::cast_slice(&values);
    let mut maximum = [0_f32; 4];
    for pixel in pixels.chunks_exact(4) {
        for i in 0..4 {
            assert!(pixel[i].is_finite());
            maximum[i] = maximum[i].max(pixel[i]);
        }
    }
    assert!(
        maximum[0] < 0.001 && maximum[1] < 0.001,
        "discontinuous marks: {maximum:?}"
    );
    assert!(
        maximum[2] > 0.008 && maximum[3] > 0.2,
        "missing marks: {maximum:?}"
    );
}

#[test]
fn mottled_clay_colours_the_body_under_uniform_illumination() {
    use gf_core::state::{Detail, State};
    use glassforms::engine::Engine;

    let width = 270;
    let height = 215;
    let render = |mottle| {
        let gpu = Gpu::headless("vulkan").ok()?;
        let scene = [
            ("scene", "studio"),
            ("preset", "sphere"),
            ("material", "modelclay"),
            ("L1.size", ".82"),
            ("rx", "13"),
            ("ry", "-21"),
            ("rz", "6"),
            ("dist", "4"),
            ("fov", "28"),
            ("spin", "0"),
            ("res", "256"),
            ("rig", ""),
            ("ambient", ".65,.65,.65"),
            ("backdrop", "0"),
            ("floor", "0"),
            ("exposure", "1"),
            ("meter", "0"),
            ("grain", "0"),
            ("bloom", "0"),
            ("mottle", mottle),
            ("age", "0"),
            ("brush", "0"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()));
        let mut engine = Engine::new(gpu, State::with_args(&scene, true), true);
        engine.resize(width, height);
        engine.bake(Detail::Full).unwrap();
        engine.settle_lighting();
        engine.render(32);
        Some(engine.renderer.read_hdr(0, 0, width, height).unwrap())
    };
    let Some(clean) = render("0") else {
        return;
    };
    let mottled = render("1").unwrap();
    let mut squared = 0.0;
    let mut count = 0;
    for y in 80..137 {
        for x in 88..158 {
            let i = 4 * (y * width + x) as usize;
            assert!(clean[i + 3] > 0.0);
            assert!(mottled[i].is_finite() && clean[i] > 0.0);
            squared += ((mottled[i] - clean[i]) / clean[i]).powi(2);
            count += 1;
        }
    }
    let rms = (squared / count as f32).sqrt();
    // Red is the dominant pigment. Tinting only the faint clear glaze gave ~0.24% here.
    assert!(
        (0.012..0.05).contains(&rms),
        "relative red pigment variation {rms}"
    );
}

#[test]
fn wear_from_obj_inverts_wear_local_and_keeps_brushed_tangents_finite_when_orbited() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let material_src = include_str!("../src/shaders/trace_material.wgsl");
    let wear_head = material_src
        .split_once("fn wearLocal(")
        .unwrap()
        .1
        .split_once("fn mEdge(")
        .unwrap()
        .0;
    let brush_surf = material_src
        .split_once("struct BrushSurface {")
        .unwrap()
        .1
        .split_once("fn brushN(")
        .unwrap()
        .0;
    let source = format!(
        r#"
struct UTest {{
  obj0: vec4f,
  obj1: vec4f,
  obj2: vec4f,
}}
struct MUTest {{
  gratingFrame: array<vec4f, 3>,
  more: vec4f,
  tab: array<vec4f, 32>,
}}
var<private> u: UTest;
var<private> mu: MUTest;
const BR_ACROSS: f32 = 0.15;
const BR_ALONG: f32 = 0.006;
fn toObj(p: vec3f) -> vec3f {{ return p.x * u.obj0.xyz + p.y * u.obj1.xyz + p.z * u.obj2.xyz; }}
fn brushStreak(p: vec3f) -> f32 {{ return 1.0; }}
fn wearLocal({wear_head}
struct BrushSurface {{{brush_surf}

@group(0) @binding(0) var<storage, read_write> out: array<vec4f>;

fn rotAxis(axis: vec3f, ang: f32) -> mat3x3f {{
  let a = normalize(axis);
  let c = cos(ang);
  let s = sin(ang);
  let t = 1.0 - c;
  return mat3x3f(
    vec3f(t*a.x*a.x + c,       t*a.x*a.y + s*a.z,   t*a.x*a.z - s*a.y),
    vec3f(t*a.x*a.y - s*a.z,   t*a.y*a.y + c,       t*a.y*a.z + s*a.x),
    vec3f(t*a.x*a.z + s*a.y,   t*a.y*a.z - s*a.x,   t*a.z*a.z + c)
  );
}}

@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) id: vec3u) {{
  let idx = id.x;
  if (idx >= 64u) {{ return; }}
  let t = f32(idx);
  // Include exact 45-deg / 90-deg orbits where R^T != R previously aligned w.b with w.n -> NaN.
  let r_view = rotAxis(vec3f(1.0, 0.0, 0.0), 0.78539816 + 0.17 * t)
             * rotAxis(vec3f(0.0, 1.0, 0.0), 0.52359878 * t)
             * rotAxis(vec3f(0.0, 0.0, 1.0), 1.57079633 + 0.11 * t);
  // Rows of R (object -> world), matching engine.rs TraceU::obj.
  u.obj0 = vec4f(r_view[0].x, r_view[1].x, r_view[2].x, 0.0);
  u.obj1 = vec4f(r_view[0].y, r_view[1].y, r_view[2].y, 0.0);
  u.obj2 = vec4f(r_view[0].z, r_view[1].z, r_view[2].z, 0.0);

  let inv_s = 1.35;
  let r_owner = rotAxis(vec3f(0.3, 0.8, -0.5), 0.37 * t);
  // Rows of inv_s * R_owner^T, matching inverse placement in MU::gratingFrame.
  mu.gratingFrame[0] = vec4f(inv_s * r_owner[0], 0.2);
  mu.gratingFrame[1] = vec4f(inv_s * r_owner[1], -0.1);
  mu.gratingFrame[2] = vec4f(inv_s * r_owner[2], 0.4);
  mu.more = vec4f(0.0, 0.0, 0.0, 1.0);
  let brush_ang = 0.23 * t;
  mu.tab[17] = vec4f(0.0, 0.0, 0.0, sin(brush_ang));
  mu.tab[18] = vec4f(0.0, 0.0, 0.0, cos(brush_ang) - 1.0);

  let n = normalize(vec3f(sin(0.41 * t + 0.7), cos(0.78539816 + 0.19 * t), cos(0.53 * t - 0.3)));
  let roundtrip = wearFromObj(wearLocal(n, 0.0) / inv_s);
  let rt_err = length(roundtrip - n);

  let nq = wearLocal(n, 0.0) / inv_s;
  let sine = mu.tab[17].w;
  let brushedNormal = brushCoordinates(nq, sine);
  var bq = vec3f(0.0, 1.0, 0.0) - brushedNormal * brushedNormal.y;
  let lb = length(bq);
  bq = mix(vec3f(1.0, 0.0, 0.0) - brushedNormal * brushedNormal.x, bq, smoothstep(0.1, 0.3, lb));
  var w: Wear;
  w.n = n;
  w.b = wearFromObj(brushCoordinates(normalize(bq + vec3f(1e-6)), -sine));
  w.aniso = 0.08;
  let bs = brushSurface(w, n * 0.5, 0.08);
  out[idx] = vec4f(rt_err, abs(dot(w.b, w.n)), length(bs.across), bs.slopes.x);
}}
"#
    );
    let module = gpu.shader("wearFromObj inverse rotation check", &source, false);
    let pipe = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("check"),
            compilation_options: Default::default(),
            cache: None,
        });
    let size = (64 * 16) as u64;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wearFromObj out"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wearFromObj readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf, 0, &readback, 0, size);
    gpu.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    gpu.wait();
    rx.recv().unwrap().unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    let rows: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    for (i, row) in rows.iter().enumerate() {
        assert!(
            row.iter().all(|x| x.is_finite()),
            "case {i} emitted non-finite value: {row:?}"
        );
        assert!(
            row[0] < 1e-5,
            "case {i} wearFromObj failed to invert wearLocal: err={}",
            row[0]
        );
        assert!(
            row[1] < 1e-4,
            "case {i} brushed direction left tangent plane: |dot(w.b, w.n)|={}",
            row[1]
        );
        assert!(
            (row[2] - 1.0).abs() < 1e-4,
            "case {i} brushSurface.across not unit length: {}",
            row[2]
        );
    }
}
