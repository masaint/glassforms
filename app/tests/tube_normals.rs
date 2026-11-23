//! Analytic swept-tube distances and normals (exact sphere-envelope SDF, joints, and melt).
//! Needs a GPU (through the queue).

use gf_core::presets::exact::{write_tubes, Written};

/// A table of tubes as the program makes one for a settled 768 volume.
fn write(pieces: Vec<Written>, ground: Option<f64>, melt: f64) -> Vec<[f32; 4]> {
    write_tubes(pieces, ground, melt, gf_core::presets::tubes::clear(melt, false, 768), 6.0)
}
use gf_core::state::{Detail, ShapeJob, State};
use glassforms::gpu::Gpu;

fn evaluate(gpu: &Gpu, table: &[[f32; 4]], points: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let source = format!(
        "@group(0) @binding(0) var<storage, read> tbl: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4f>;
@group(0) @binding(2) var<storage, read> points: array<vec4f>;
{}
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) invocation: vec3u) {{
  let i = invocation.x;
  if (i >= arrayLength(&points)) {{ return; }}
  results[i] = rawGrad(points[i].xyz);
}}",
        include_str!("../../core/src/presets/tubes.wgsl")
    );
    let shader = gpu.shader("tube joint normal regression", &source, false);
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tube joint normal regression"),
            layout: None,
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
    let buffer = |label, bytes: &[u8], usage| {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, bytes);
        buffer
    };
    let table = buffer(
        "tube table",
        bytemuck::cast_slice(table),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let sample_points = buffer(
        "tube sample points",
        bytemuck::cast_slice(points),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let result_size = (points.len() * 16) as u64;
    let results = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tube normals"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let download = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tube normal download"),
        size: result_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tube joint normal regression"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: table.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: results.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sample_points.as_entire_binding(),
            },
        ],
    });
    let mut commands = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(points.len().div_ceil(64) as u32, 1, 1);
    }
    commands.copy_buffer_to_buffer(&results, 0, &download, 0, result_size);
    gpu.queue.submit(Some(commands.finish()));
    let (send, receive) = std::sync::mpsc::channel();
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap()
        });
    gpu.wait();
    receive.recv().unwrap().unwrap();
    let mapped = download.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec()
}

fn piece(kind: f32, start: f32, end: f32, s0: f32, s1: f32) -> Written {
    let a = [start, 0.0, 0.0, 0.03 + 0.2 * start];
    let d = [end, 0.0, 0.0, 0.03 + 0.2 * end];
    let b = std::array::from_fn(|k| (2.0 * a[k] + d[k]) / 3.0);
    let c = std::array::from_fn(|k| (a[k] + 2.0 * d[k]) / 3.0);
    let cv = |v: [f32; 4]| v;
    let tag = [kind, s0, s1, 0.08];
    let e = if kind == 111.0 {
        vec![tag, cv(a), cv(d)]
    } else {
        vec![tag, cv(a), cv(b), cv(c), cv(d)]
    };
    Written {
        lo: [(start - 0.06).into(), -0.06, -0.06],
        hi: [(end + 0.06).into(), 0.06, 0.06],
        e,
        meta: -1.0,
    }
}

/// Sample point and exact unit outward normal on the sphere-envelope surface for a cubic span.
fn envelope_sample(
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
    t: f64,
    offset_h: f64,
) -> Option<([f32; 4], [f64; 3])> {
    let u = 1.0 - t;
    let p: [f64; 4] = std::array::from_fn(|k| {
        u * u * u * f64::from(a[k])
            + 3.0 * u * u * t * f64::from(b[k])
            + 3.0 * u * t * t * f64::from(c[k])
            + t * t * t * f64::from(d[k])
    });
    let v: [f64; 4] = std::array::from_fn(|k| {
        3.0 * u * u * f64::from(b[k] - a[k])
            + 6.0 * u * t * f64::from(c[k] - b[k])
            + 3.0 * t * t * f64::from(d[k] - c[k])
    });
    let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if speed < 1e-8 {
        return None;
    }
    let tangent = [v[0] / speed, v[1] / speed, v[2] / speed];
    let slope = v[3] / speed;
    if slope.abs() >= 0.99 {
        return None;
    }
    let mut radial = [0.0_f64, 0.0, 1.0];
    let proj = tangent[2];
    for k in 0..3 {
        radial[k] -= proj * tangent[k];
    }
    let rlen = radial.iter().map(|x| x * x).sum::<f64>().sqrt();
    if rlen < 1e-8 {
        return None;
    }
    radial = radial.map(|x| x / rlen);
    let perp = (1.0 - slope * slope).sqrt();
    let normal: [f64; 3] = std::array::from_fn(|k| perp * radial[k] - slope * tangent[k]);
    let acc: [f64; 4] = std::array::from_fn(|k| {
        6.0 * u * f64::from(a[k] - 2.0 * b[k] + c[k])
            + 6.0 * t * f64::from(b[k] - 2.0 * c[k] + d[k])
    });
    let d_surf = p[3] + offset_h;
    if d_surf <= 1e-4 {
        return None;
    }
    let n_dot_acc = normal[0] * acc[0] + normal[1] * acc[1] + normal[2] * acc[2];
    let fpp = speed * speed * (1.0 - slope * slope) / d_surf - n_dot_acc - acc[3];
    if fpp <= 0.1 * speed * speed / d_surf {
        return None;
    }
    let mut point = [0.0_f32; 4];
    for k in 0..3 {
        point[k] = (p[k] + d_surf * normal[k]) as f32;
    }
    Some((point, normal))
}

#[test]
fn generated_thomas_pieces_converge_to_their_own_surface() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene".into(), "attractors".into()),
        ("preset".into(), "thomas".into()),
        ("p.b".into(), "0.19".into()),
        ("beads".into(), "0.8".into()),
        ("melt".into(), "0".into()),
    ];
    let mut state = State::with_args(&args, false);
    let mut job = state.shape_job(Detail::Full, Some(192)).unwrap();
    job.prepare_formula(&state.families).unwrap();
    let ShapeJob::Field { table, .. } = job.shape else {
        panic!("no analytic tube table")
    };
    let table: Vec<[f32; 4]> = table
        .chunks_exact(4)
        .map(|v| v.try_into().unwrap())
        .collect();
    let mut points = Vec::new();
    let mut out_points = Vec::new();
    let mut expected = Vec::new();
    let mut offsets = Vec::new();
    let mut locations = Vec::new();
    let mut at = table[2][3] as usize;
    while at < table.len() {
        let kind = table[at + 2][0];
        let size = if kind == 110.0 {
            let (a, b, c, d) = (table[at + 3], table[at + 4], table[at + 5], table[at + 6]);
            for h in [0.0, 0.002, 0.005, -0.002] {
                for j in 1..20 {
                    let t = f64::from(j) / 20.0;
                    if let Some((point, normal)) = envelope_sample(a, b, c, d, t, h) {
                        out_points.push([
                            point[0] + 0.002 * normal[0] as f32,
                            point[1] + 0.002 * normal[1] as f32,
                            point[2] + 0.002 * normal[2] as f32,
                            0.0,
                        ]);
                        points.push(point);
                        expected.push(normal);
                        offsets.push(h as f32);
                        locations.push((at, t, h));
                    }
                }
            }
            7
        } else if kind == 111.0 {
            4
        } else if kind == 108.0 {
            6
        } else if kind == 101.0 {
            3
        } else {
            panic!("unknown tube record {kind} at {at}")
        };
        at += size;
    }
    let mut eval_points = points.clone();
    eval_points.extend_from_slice(&out_points);
    let actual = evaluate(&gpu, &table, &eval_points);
    let (actual_at, actual_out) = actual.split_at(points.len());
    let mut checked = 0;
    let mut worst_n = 0.0_f32;
    let mut worst_d = 0.0_f32;
    let mut detail = String::new();
    for (((((&point, &value), &out_value), wanted), &h), location) in points
        .iter()
        .zip(actual_at)
        .zip(actual_out)
        .zip(expected)
        .zip(&offsets)
        .zip(locations)
    {
        // Skip points where another strand is closer or lies across the medial axis.
        if value[3] < h - 2e-6 || out_value[3] < h + 0.002 - 2e-6 {
            continue;
        }
        checked += 1;
        let d_err = (value[3] - h).abs();
        worst_d = worst_d.max(d_err);
        let length = value[..3].iter().map(|x| x * x).sum::<f32>().sqrt();
        let error = (0..3)
            .map(|k| (value[k] / length - wanted[k] as f32).abs())
            .fold(0.0_f32, f32::max);
        if error > worst_n {
            worst_n = error;
            detail = format!(
                "piece {}, t {}, h {}: point {point:?}, value {value:?}, expected {wanted:?}",
                location.0, location.1, location.2
            );
        }
    }
    assert!(
        checked > 4000,
        "only {checked} non-intersected surface/shell samples"
    );
    assert!(worst_d < 0.0002, "worst distance error {worst_d}");
    assert!(
        worst_n < 0.002,
        "worst generated-sweep normal error {worst_n}: {detail}"
    );
}

#[test]
fn curved_tapered_tube_normals_match_the_surface_derivative() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let a = [-0.4, 0.0, 0.0, 0.025];
    let b = [-0.1, 0.35, 0.12, 0.065];
    let c = [0.25, -0.2, -0.1, 0.018];
    let d = [0.5, 0.1, 0.2, 0.05];
    let tag = [110.0, 0.0, 1.0, 0.5];
    let table = write(
        vec![Written {
            lo: [-0.6, -0.5, -0.4],
            hi: [0.7, 0.6, 0.5],
            e: vec![tag, a, b, c, d],
            meta: -1.0,
        }],
        None,
        0.0,
    );
    let mut points = Vec::new();
    let mut expected = Vec::new();
    let mut offsets = Vec::new();
    for h in [0.0, 0.003, -0.002] {
        for i in 1..20 {
            let t = f64::from(i) / 20.0;
            if let Some((point, normal)) = envelope_sample(a, b, c, d, t, h) {
                points.push(point);
                expected.push(normal);
                offsets.push(h as f32);
            }
        }
    }
    let actual = evaluate(&gpu, &table, &points);
    for (((point, value), wanted), h) in points.iter().zip(actual).zip(expected).zip(offsets) {
        if value[3] < h - 0.0002 {
            continue;
        }
        assert!(
            (value[3] - h).abs() < 0.0002,
            "point {point:?}: dist {}, expected {h}",
            value[3]
        );
        let length = value[..3].iter().map(|x| x * x).sum::<f32>().sqrt();
        let error = (0..3)
            .map(|k| (value[k] / length - wanted[k] as f32).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            error < 0.0005,
            "point {point:?}: normal {:?}, expected {wanted:?}, error {error}",
            value
        );
    }
}

#[test]
fn fitted_joints_keep_the_tapered_tube_normal_on_and_off_surface() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let slope = 0.2_f32;
    let perp = (1.0 - slope * slope).sqrt();
    let expected = [-slope, perp, 0.0];
    for kind in [110.0, 111.0] {
        let table = write(
            vec![
                piece(kind, -0.1, 0.0, 0.0, 0.1),
                piece(kind, 0.0, 0.1, 0.1, 0.2),
            ],
            None,
            0.0,
        );
        let mut points = Vec::new();
        let mut expected_d = Vec::new();
        for h in [0.0_f32, 0.002, 0.005, -0.002] {
            for x in [-0.05_f32, -0.0001, 0.0, 0.0001, 0.05] {
                let r = 0.03 + slope * x;
                let px = x + (r + h) * expected[0];
                let py = (r + h) * expected[1];
                points.push([px, py, 0.0, 0.0]);
                expected_d.push(h);
            }
        }
        let actual = evaluate(&gpu, &table, &points);
        for ((point, value), want_d) in points.iter().zip(actual).zip(expected_d) {
            assert!(
                (value[3] - want_d).abs() < 0.00015,
                "kind {kind}, point {point:?}: dist {}, expected {want_d}",
                value[3]
            );
            let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
            let normal = [value[0] / length, value[1] / length, value[2] / length];
            let error = (0..3)
                .map(|k| (normal[k] - expected[k]).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                error < 0.0005,
                "kind {kind}, point {point:?}: normal {normal:?}, expected {expected:?}"
            );
        }
    }
}

#[test]
fn melted_tube_strands_are_continuous_across_joints_and_crossings() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let strand_piece = |ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, r: f32, s0: f32, s1: f32, meta: f32| {
        let a = [ax, ay, az, r];
        let d = [bx, by, bz, r];
        let b = std::array::from_fn(|k| (2.0 * a[k] + d[k]) / 3.0);
        let c = std::array::from_fn(|k| (a[k] + 2.0 * d[k]) / 3.0);
        Written {
            lo: [f64::from(ax.min(bx) - r), f64::from(ay.min(by) - r), f64::from(az.min(bz) - r)],
            hi: [f64::from(ax.max(bx) + r), f64::from(ay.max(by) + r), f64::from(az.max(bz) + r)],
            e: vec![[110.0, s0, s1, r + 0.001], a, b, c, d],
            meta,
        }
    };
    let pieces = vec![
        // Strand 1 along X with a joint at x = 0.0 (no self-bulge).
        strand_piece(-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.3, -1.0),
        strand_piece(0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.04, 0.3, 0.6, -1.0),
        // Strand 2 along Y at z = 0.09 (crosses Strand 1 and melts smoothly).
        strand_piece(0.0, -0.3, 0.09, 0.0, 0.3, 0.09, 0.04, 0.0, 0.6, -2.0),
    ];
    let melt = 0.02;
    let table_fwd = write(pieces.clone(), None, melt);
    let table_rev = write(pieces.into_iter().rev().collect(), None, melt);
    let probe = vec![
        [0.0, 0.0, -0.04, 0.0],     // on Strand 1 opposite the crossing (far from Strand 2)
        [-0.001, 0.0, -0.04, 0.0],  // just left of joint
        [0.001, 0.0, -0.04, 0.0],   // just right of joint
        [0.0, 0.0, 0.045, 0.0],     // midpoint between crossing strands (smooth melt)
        [0.15, 0.0, 0.04, 0.0],     // isolated Strand 1 away from Strand 2 and from its own end: no self-bulge
        [0.01, 0.02, 0.046, 0.0],   // off-saddle crossing point (non-zero blended normal)
    ];
    let fwd = evaluate(&gpu, &table_fwd, &probe);
    let rev = evaluate(&gpu, &table_rev, &probe);
    for (idx, (a, b)) in fwd.iter().zip(&rev).enumerate() {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() < 1e-5,
                "melt order dependence at probe {idx} lane {k}: {a:?} vs {b:?}"
            );
        }
    }
    // At x = 0.15 (on Strand 1 surface away from Strand 2), distance must equal 0 (no self-melt).
    assert!(fwd[4][3].abs() < 0.0005, "isolated joint swelled: {:?}", fwd[4]);
    // At the crossing midpoint (z = 0.045, each strand d = 0.005), smooth-min must lower d below 0.005.
    assert!(
        fwd[3][3] < 0.005 - 0.1 * melt as f32,
        "crossing strands failed to smooth-blend: {:?}",
        fwd[3]
    );
}

#[test]
fn melted_tube_normals_are_c1_across_grid_cells_and_cutoff_and_not_stale_cached() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let strand_piece = |ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, r: f32, meta: f32| {
        let a = [ax, ay, az, r];
        let d = [bx, by, bz, r];
        let b = std::array::from_fn(|k| (2.0 * a[k] + d[k]) / 3.0);
        let c = std::array::from_fn(|k| (a[k] + 2.0 * d[k]) / 3.0);
        Written {
            lo: [f64::from(ax.min(bx) - r), f64::from(ay.min(by) - r), f64::from(az.min(bz) - r)],
            hi: [f64::from(ax.max(bx) + r), f64::from(ay.max(by) + r), f64::from(az.max(bz) + r)],
            e: vec![[110.0, 0.0, 1.0, r + 0.001], a, b, c, d],
            meta,
        }
    };
    let melt = 0.025;
    let table = write(
        vec![
            strand_piece(-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.04, -1.0),
            strand_piece(0.0, -0.5, 0.10, 0.0, 0.5, 0.10, 0.04, -2.0),
        ],
        None,
        melt,
    );
    // Sweep along Strand 1's upper surface (z = 0.04, y = 0.005) across the cells' boundaries and across
    // where Strand 2's part of the melt fades in and out (about 0.1 from its surface, either side of x = 0).
    let mut scan = Vec::new();
    for i in 0..=800 {
        let x = -0.16 + 0.32 * (i as f32 / 800.0); // step = 0.0004
        scan.push([x, 0.005, 0.04, 0.0]);
    }
    let values = evaluate(&gpu, &table, &scan);
    let unit = |v: [f32; 4]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    // The normal turns steadily along the scan (fastest in the fillet under the crossing, where the melt
    // of 0.025 bends the surface to a radius of about 0.05): no step in it, and no step in how fast it turns.
    let mut max_jump = 0.0_f32;
    let mut max_change = 0.0_f32;
    let mut worst_x = 0.0_f32;
    let mut last = 0.0_f32;
    for i in 0..scan.len() - 1 {
        let n0 = unit(values[i]);
        let n1 = unit(values[i + 1]);
        let jump = ((n1[0] - n0[0]).powi(2) + (n1[1] - n0[1]).powi(2) + (n1[2] - n0[2]).powi(2)).sqrt();
        if i > 0 && (jump - last).abs() > max_change {
            max_change = (jump - last).abs();
            worst_x = scan[i][0];
        }
        last = jump;
        max_jump = max_jump.max(jump);
    }
    eprintln!("normal: turns by at most {max_jump} a step, and that changes by at most {max_change} (near x = {worst_x})");
    assert!(max_jump < 0.012, "normal jump {max_jump} across cell/melt boundary");
    assert!(
        max_change < 0.001,
        "the normal's turning steps by {max_change} across a cell or the melt's reach near x = {worst_x}"
    );

    // Also verify rawGrad cache does not return a stale gradient after a sub-millimeter Newton settle step (0.00035).
    let cache_src = format!(
        "@group(0) @binding(0) var<storage, read> tbl: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4f>;
{}
@compute @workgroup_size(1)
fn check_cache() {{
  let p0 = vec3f(0.02, 0.036, 0.02);
  let p1 = p0 + vec3f(0.0, 0.00035, 0.00025);
  let _d0 = raw(p0);
  results[0] = vec4f(normalize(rawGrad(p1).xyz), 0.0);
  results[1] = vec4f(normalize(tube_eval(p1).xyz), 0.0);
  results[2] = vec4f(normalize(tube_eval(p0).xyz), 0.0);
}}",
        include_str!("../../core/src/presets/tubes.wgsl")
    );
    let shader = gpu.shader("tube cache regression", &cache_src, false);
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tube cache regression"),
        layout: None,
        module: &shader,
        entry_point: Some("check_cache"),
        compilation_options: Default::default(),
        cache: None,
    });
    let tbl_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tbl"),
        size: (table.len() * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&tbl_buf, 0, bytemuck::cast_slice(&table));
    let res_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("res"),
        size: 48,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dl_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dl"),
        size: 48,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tbl_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: res_buf.as_entire_binding() },
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&res_buf, 0, &dl_buf, 0, 48);
    gpu.queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    dl_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    gpu.wait();
    rx.recv().unwrap().unwrap();
    let mapped = dl_buf.slice(..).get_mapped_range().unwrap();
    let out: Vec<[f32; 4]> = bytemuck::cast_slice(&mapped).to_vec();
    let err_exact = ((out[0][0] - out[1][0]).powi(2) + (out[0][1] - out[1][1]).powi(2) + (out[0][2] - out[1][2]).powi(2)).sqrt();
    let diff_p0 = ((out[0][0] - out[2][0]).powi(2) + (out[0][1] - out[2][1]).powi(2) + (out[0][2] - out[2][2]).powi(2)).sqrt();
    assert!(err_exact < 1e-6 && diff_p0 > 1e-4, "rawGrad(p1) returned stale cached normal from p0: {out:?}");
}

/// The shader against the CPU's copy of it (`tubes::cpu`, which the field's tests in core measure): the
/// same distance and the same slope, near the surface, in the melt, and far off, for a playing attractor
/// and for a tree with its leaves and its ground.
#[test]
fn the_shader_reads_what_the_cpu_copy_reads() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    for (args, detail, res) in [
        (vec![("scene", "attractors"), ("preset", "thomas")], Detail::Full, 768u32),
        (vec![("scene", "attractors"), ("preset", "thomas")], Detail::Anim, 256),
        (vec![("scene", "attractors"), ("preset", "thomas"), ("melt", "0")], Detail::Full, 768),
        (vec![("scene", "trees")], Detail::Full, 768),
        (vec![("scene", "fieldlines")], Detail::Full, 768),
    ] {
        let pairs: Vec<(String, String)> = args.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let mut state = State::with_args(&pairs, false);
        let mut job = state.shape_job(detail, Some(res)).unwrap();
        job.prepare_formula(&state.families).unwrap();
        let ShapeJob::Field { table, .. } = job.shape else { panic!("no analytic tube table") };
        let table: Vec<[f32; 4]> = table.chunks_exact(4).map(|v| v.try_into().unwrap()).collect();
        // points all through the grid, and a little beyond it
        let (lo, size) = (table[1], table[1][3] * table[2][0]);
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 40) as f32 / (1u64 << 24) as f32 };
        let points: Vec<[f32; 4]> = (0..60000).map(|_| [lo[0] + size * (1.1 * next() - 0.05), lo[1] + size * (1.1 * next() - 0.05), lo[2] + size * (1.1 * next() - 0.05), 0.0]).collect();
        let actual = evaluate(&gpu, &table, &points);
        let (mut worst_d, mut worst_n, mut near) = (0.0f64, 0.0f64, 0);
        for (p, got) in points.iter().zip(&actual) {
            let want = gf_core::presets::tubes::cpu::eval(&table, [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
            let d = (f64::from(got[3]) - want[3]).abs();
            assert!(d < 2e-5 + 2e-4 * want[3].abs(), "{args:?} at {p:?}: the GPU reads {} and the CPU {}", got[3], want[3]);
            worst_d = worst_d.max(d);
            // (the slope where it is read for a normal: by the surface)
            if want[3].abs() < 0.01 {
                near += 1;
                let (gl, wl) = ((0..3).map(|k| f64::from(got[k]).powi(2)).sum::<f64>().sqrt(), (0..3).map(|k| want[k].powi(2)).sum::<f64>().sqrt());
                worst_n = worst_n.max((0..3).map(|k| (f64::from(got[k]) / gl - want[k] / wl).abs()).fold(0.0, f64::max));
            }
        }
        eprintln!("{args:?} {detail:?}: worst distance {worst_d:.2e}, worst slope {worst_n:.2e} over {near} points by the surface");
        assert!(near > 200, "{args:?}: only {near} points by the surface");
        assert!(worst_n < 2e-3, "{args:?}: the GPU's slope is {worst_n} from the CPU's");
    }
}
