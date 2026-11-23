use crate::{engine::Engine, gpu::Gpu};
use gf_core::{
    construction::CutMaterial,
    csg::{Op, Part},
    groups::Item,
    layers::Placement,
    modifier_pivot::Pivot,
    mods::{Axes, Kind, Modifier},
    state::{Detail, State},
    world_mods::Scope,
};
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

/// Run the render shader's own face selection against the packed scene buffer.
fn records(e: &Engine, points: &[[f32; 4]]) -> Vec<u32> {
    let gpu = &e.gpu;
    let src = include_str!("shaders/world_faces.wgsl");
    let declarations = src.split_once("fn wfMat").unwrap().0;
    let dec = declarations
        .lines()
        .filter(|l| !l.contains("var<private>"))
        .collect::<Vec<_>>()
        .join("\n");
    let pick = src.split_once("fn wfProjected(").unwrap().1;
    let src=format!("{dec}
var<private> record:u32;
fn wfAssign(k:i32,row:u32,frame:u32){{record=row;}}
fn toObj(p:vec3f)->vec3f{{return p;}}
fn wfProjected({pick}
@group(0) @binding(0) var<storage,read> points:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3u){{wfPick(0,points[id.x].xyz);result[id.x]=record;}}")
        .replace("{{RECORD}}", &(gf_core::material::MU_FLOATS+8).to_string());
    let shader = gpu.shader("World cut probe", &src, true);
    let pipe = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let input = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(points),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let bytes = points.len() as u64 * 4;
    let buf = |usage| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let out = buf(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let read = buf(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let blank = gpu
        .volume(
            "empty projected field",
            1,
            1,
            wgpu::TextureFormat::Rgba16Float,
            false,
        )
        .create_view(&Default::default());
    let entries = [
        (0, &input),
        (1, &out),
        (super::BINDING, &e.renderer.world_faces.buffer),
    ]
    .map(|(binding, b)| wgpu::BindGroupEntry {
        binding,
        resource: b.as_entire_binding(),
    })
    .into_iter()
    .chain(
        e.renderer
            .world_faces
            .projected
            .iter()
            .enumerate()
            .map(|(i, v)| wgpu::BindGroupEntry {
                binding: super::PROJECTED_BINDING + i as u32,
                resource: wgpu::BindingResource::TextureView(v.as_ref().unwrap_or(&blank)),
            }),
    )
    .collect::<Vec<_>>();
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(points.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
    gpu.queue.submit([enc.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    read.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
    gpu.wait();
    recv.recv().unwrap().unwrap();
    let mapped = read.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, u32>(&mapped).to_vec()
}

#[test]
fn remote_mirrors_transport_asymmetric_cut_surface_records() {
    let args = [
        ("scene", "studio"),
        ("preset", "sphere"),
        ("p.radius", "0.24"),
        ("trace", "csg"),
        ("res", "128"),
        ("layer", "studio:sphere"),
        ("L2.x", "-1.40"),
        ("L2.size", "0.17"),
        ("J12.op", "apart"),
        ("B2.material", "chrome"),
        ("rx", "0"),
        ("ry", "0"),
        ("rz", "0"),
        ("dist", "6.3"),
        ("bloom", "0"),
        ("grain", "0"),
        ("disp", "0"),
    ]
    .map(|(a, b)| (a.into(), b.into()));
    let mut state = State::with_args(&args, true);
    state.select_layer(0);
    let cut = Part {
        place: Placement {
            pos: [0.075, 0.035, 0.223],
            size: 0.15,
            ..Default::default()
        },
        ..Default::default()
    };
    let path = state.csg.add_construction_part(&[], cut).unwrap();
    let node = state.csg.edit_construction().node_mut(&path).unwrap();
    node.op = Op::Subtract;
    node.material = Some(BTreeMap::from([
        ("metal".into(), 3.),
        ("rough".into(), 0.08),
    ]));
    let mut repeat = Modifier::of(Kind::Repeat);
    repeat.fit = false;
    repeat.a = Axes::new([2., 1., 1.], false);
    repeat.b = Axes::new([1.1, 1., 1.], false);
    let mut mirror = Modifier::of(Kind::Mirror);
    mirror.mode = 1;
    mirror.scope = Scope::SubObjects;
    mirror.pivot = Pivot::World;
    mirror.centre = [0.95, 0., 0.];
    mirror.cascade = true;
    state
        .stack
        .world_modifiers_mut(Item::Object(0))
        .unwrap()
        .list = vec![repeat, mirror];
    state.stack.box_held = Some(3.);
    let mut e = Engine::new(Gpu::headless("vulkan").unwrap(), state, true);
    e.auto = false;
    e.resize(960, 400);
    e.bake(Detail::Full).unwrap();
    e.renderer.reseed(0);
    e.render(12);
    let photo = e.screenshot().unwrap().2;
    if let Some(dir) = std::env::var_os("GF_CONSTRUCTION_EVIDENCE") {
        std::fs::write(
            std::path::PathBuf::from(dir).join("subtractor.json"),
            e.state.setup_json(),
        )
        .unwrap();
    }
    let (res, volume) = e.read_volume().unwrap();
    let res = res as usize;
    let hb = e.state.scene_hb();
    let step = 2. * hb / res as f64;
    let centres = [-0.55, 0.55, 1.35, 2.45];
    // Trace three central cut samples and three intact exterior samples per occurrence.
    let mut points = Vec::new();
    for (which, &cx) in centres.iter().enumerate() {
        let sign = if which < 2 { 1. } else { -1. };
        for offset in [0.055, 0.075, 0.095, -0.08, -0.11, -0.14] {
            let ix = ((cx + sign * offset + hb) / step).floor() as usize;
            let iy = ((0.035 + hb) / step).floor() as usize;
            let d = |iz: usize| half::f16::from_bits(volume[ix + res * (iy + res * iz)]).to_f64();
            let z = (1..res - 1)
                .rev()
                .find(|&z| d(z) <= 0. && d(z + 1) > 0.)
                .unwrap();
            let alpha = -d(z) / (d(z + 1) - d(z));
            points.push([
                ((ix as f64 + 0.5) * step - hb) as f32,
                ((iy as f64 + 0.5) * step - hb) as f32,
                ((z as f64 + 0.5 + alpha) * step - hb) as f32,
                0.,
            ]);
        }
    }
    for hits in points.chunks_exact(6) {
        assert!(hits[..3].iter().all(|p| p[2] < 0.12));
        assert!(hits[3..].iter().all(|p| p[2] > 0.17));
    }
    let ids = records(&e, &points);
    // Compare with the authored source at each of the same physical hit positions.
    let mut layout = e.state.stack.combine();
    layout.occurrences.clear();
    layout.inv[0] = Placement::default().inverse();
    let mut authored = super::SceneFaces::new(&e.gpu);
    authored.update(&e.gpu, &e.csg.faces, &layout, [None; 4]);
    let actual = std::mem::replace(&mut e.renderer.world_faces, authored);
    let local_points: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let side = i / 6;
            [
                (p[0] - centres[side] as f32) * if side < 2 { 1. } else { -1. },
                p[1],
                p[2],
                0.,
            ]
        })
        .collect();
    let reference = records(&e, &local_points);
    e.renderer.world_faces = actual;
    for authored in reference.chunks_exact(6) {
        assert!(authored[..3].iter().all(|&id| id != 0));
        assert!(authored[4..].iter().all(|&id| id == 0));
    }
    let cavity_z: Vec<_> = points
        .chunks_exact(6)
        .map(|pts| {
            pts.iter()
                .map(|p| (p[2] * 1000.).round() as i32)
                .collect::<Vec<_>>()
        })
        .collect();
    eprintln!("surface depth in thousandths {cavity_z:?}, coating records {ids:?}");
    e.state
        .csg
        .edit_construction()
        .node_mut(&path)
        .unwrap()
        .cut_material = CutMaterial::Recipient;
    e.bake(Detail::Full).unwrap();
    e.renderer.reseed(0);
    e.render(12);
    let recipient = e.screenshot().unwrap().2;
    if let Some(dir) = std::env::var_os("GF_CONSTRUCTION_EVIDENCE") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::write(dir.join("recipient.json"), e.state.setup_json()).unwrap();
        for (name, image) in [
            ("mirror_cut_subtractor.png", &photo),
            ("mirror_cut_recipient.png", &recipient),
        ] {
            let file = std::fs::File::create(dir.join(name)).unwrap();
            let mut encoder = png::Encoder::new(file, 960, 400);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(image)
                .unwrap();
        }
    }
    let mask = centres.map(|x| {
        let camera = e.state.camera();
        let screen = camera.project([x, 0., 0.]).unwrap();
        let cx = ((960. + 400. * screen[0]) * 0.5) as usize;
        let radius = (camera.project([x + 0.25, 0., 0.]).unwrap()[0] - screen[0]) * 200.;
        let r = radius.ceil() as usize;
        (200 - r..=200 + r)
            .flat_map(|y| (cx - r..=cx + r).map(move |x| 4 * (x + 960 * y)))
            .filter(|&p| (0..3).any(|c| photo[p + c].abs_diff(recipient[p + c]) > 8))
            .count()
    });
    eprintln!("changed photo pixels by occurrence {mask:?}");
    if !e.renderer.world_faces.sources.is_empty() {
        assert!(records(&e, &points).iter().all(|&r| r == 0));
    }
    assert!(mask.iter().all(|&n| n >= 12), "{mask:?}");

    assert!(e.state.remove_layer(1));
    e.state
        .csg
        .edit_construction()
        .node_mut(&path)
        .unwrap()
        .cut_material = CutMaterial::Subtractor;
    e.bake(Detail::Full).unwrap();
    e.renderer.reseed(0);
    e.render(2);
    let solo = records(&e, &points);
    assert!(ids
        .iter()
        .zip(solo)
        .all(|(&left, right)| (left > 0) == (right > 0)));
    assert!(
        ids.iter()
            .zip(reference)
            .all(|(&world, local)| (world > 0) == (local > 0)),
        "{ids:?}"
    );
}
