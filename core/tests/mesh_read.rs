//! Model files read into triangles. Each format is written here from a small mesh and has to come back
//! as that mesh; and nothing that can be wrong with a file may make the reader panic, hang or ask for
//! memory the file does not justify. Nothing is wrapped in catch_unwind: a panic fails the test.

use gf_core::mesh::{read::*, Mesh};
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};

// ---- how much memory was asked for at once ----

// Every request for memory is noted, a thread at a time (each test runs on its own thread), so that a
// test can say how much the reader asked for in one go.
struct Noting;

thread_local! { static LARGEST: Cell<usize> = const { Cell::new(0) }; }

fn note(size: usize) {
    let _ = LARGEST.try_with(|c| c.set(c.get().max(size)));
}

unsafe impl GlobalAlloc for Noting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Noting = Noting;

// What `f` gives, and the largest single piece of memory asked for while it ran.
fn largest_request<T>(f: impl FnOnce() -> T) -> (T, usize) {
    LARGEST.with(|c| c.set(0));
    let out = f();
    (out, LARGEST.with(|c| c.get()))
}

// A file that lies about how much is in it has to be turned away without memory being set aside for the lie.
fn refused_cheaply(what: &str, bytes: &[u8], format: Format) -> String {
    // (the measure itself is checked first: it sees a request of a known size)
    assert!(largest_request(|| Vec::<u8>::with_capacity(3 << 20)).1 >= 3 << 20);
    let (result, largest) = largest_request(|| parse(bytes, format, None));
    let error = result.expect_err(what);
    assert!(largest < 1 << 20, "{what}: {largest} bytes were asked for at once before it was refused ({error})");
    error
}

// ---- the meshes written into every format ----

// Polygons as triangles, fanned from each one's first corner (the way every reader fans them).
fn fanned(polys: &[&[u32]]) -> Vec<[u32; 3]> {
    polys.iter().flat_map(|p| (1..p.len() - 1).map(|k| [p[0], p[k], p[k + 1]])).collect()
}

// A box of 8 vertices and 12 triangles, facing outwards: 1 by 2 by 3 and off-centre, so that two axes
// swapped or a sign lost would show.
fn cube() -> Mesh {
    let corner = |x: f32, y: f32, z: f32| [x - 0.5, 2.0 * y + 0.25, 3.0 * z + 1.5];
    let pos = vec![
        corner(0.0, 0.0, 0.0), corner(1.0, 0.0, 0.0), corner(1.0, 1.0, 0.0), corner(0.0, 1.0, 0.0),
        corner(0.0, 0.0, 1.0), corner(1.0, 0.0, 1.0), corner(1.0, 1.0, 1.0), corner(0.0, 1.0, 1.0),
    ];
    let tri = fanned(&[&[0, 3, 2, 1], &[4, 5, 6, 7], &[0, 1, 5, 4], &[2, 3, 7, 6], &[0, 4, 7, 3], &[1, 2, 6, 5]]);
    Mesh { pos, tri }
}

fn tetra() -> Mesh {
    let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    Mesh { pos, tri: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] }
}

// The mesh's triangles joined back into quads wherever two in a row are a fan of four corners.
fn polys(m: &Mesh) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < m.tri.len() {
        let [a, b, c] = m.tri[i];
        match m.tri.get(i + 1) {
            Some(&[a2, c2, d]) if a2 == a && c2 == c => { out.push(vec![a, b, c, d]); i += 2; }
            _ => { out.push(vec![a, b, c]); i += 1; }
        }
    }
    out
}

fn moved(m: &Mesh, f: impl Fn([f32; 3]) -> [f32; 3]) -> Mesh {
    Mesh { pos: m.pos.iter().map(|&p| f(p)).collect(), tri: m.tri.clone() }
}

fn joined(a: &Mesh, b: &Mesh) -> Mesh {
    let base = a.pos.len() as u32;
    let mut out = a.clone();
    out.pos.extend(&b.pos);
    out.tri.extend(b.tri.iter().map(|t| t.map(|i| i + base)));
    out
}

// The triangles as sets of corner positions, rounded and sorted: the same whatever the indexing.
fn triangles(m: &Mesh) -> Vec<[[i64; 3]; 3]> {
    let mut out: Vec<_> = m.tri.iter().map(|t| {
        let mut corners = t.map(|i| m.pos[i as usize].map(|c| (c as f64 * 1e4).round() as i64));
        corners.sort();
        corners
    }).collect();
    out.sort();
    out
}

// The volume inside, which changes sign if the triangles are turned inside out.
fn volume(m: &Mesh) -> f64 {
    m.tri.iter().map(|t| {
        let [a, b, c] = t.map(|i| m.pos[i as usize].map(|v| v as f64));
        (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    }).sum()
}

// What every mesh that is read has to be.
fn sound(m: &Mesh) {
    assert!(!m.tri.is_empty(), "no triangles");
    assert!(m.tri.len() <= MAX_TRIANGLES && m.pos.len() <= MAX_VERTICES);
    assert!(m.pos.iter().flatten().all(|c| c.is_finite() && c.abs() <= 1e18), "a position that is not finite");
    assert!(m.tri.iter().flatten().all(|&i| (i as usize) < m.pos.len()), "an index out of range");
}

// ---- the mesh written into each format ----

fn obj(m: &Mesh) -> Vec<u8> {
    let count = m.pos.len() as i64;
    let mut out = b"# a caf\xe9 comment that is not UTF-8\r\nmtllib box.mtl\no box\n".to_vec();
    for (i, p) in m.pos.iter().enumerate() {
        // plain, with a weight, with a colour
        let tail = ["\n", " 1.0\r\n", " 0.5 0.25 1\n"][i % 3];
        out.extend(format!("v {} {} {}{tail}", p[0], p[1], p[2]).bytes());
    }
    out.extend(b"vt 0 0\nvt 1 0\nvn 0 0 1\nvn 0 1 0\n\ng side\nusemtl glass\ns off\n");
    for (k, poly) in polys(m).iter().enumerate() {
        // every third face counts back from the end; the four forms of an entry take turns
        let words: Vec<String> = poly.iter().map(|&i| {
            let v = if k % 3 == 2 { i as i64 - count } else { i as i64 + 1 };
            match k % 4 { 0 => format!("{v}"), 1 => format!("{v}/1"), 2 => format!("{v}//2"), _ => format!("{v}/2/1") }
        }).collect();
        if k == 1 {
            out.extend(format!("f {} \\\r\n  {}\n", words[0], words[1..].join(" ")).bytes()); // carried over a line end
        } else {
            out.extend(format!("f {}{}", words.join("\t"), if k % 2 == 0 { "\n" } else { "\r\n" }).bytes());
        }
    }
    out.extend(b"f 1 2\nl 1 2 3\np 1\n");
    out
}

fn stl_text(m: &Mesh) -> Vec<u8> {
    let mut s = String::from("solid a vertex and a facet in its name\n");
    for t in &m.tri {
        s += "  facet normal 0 0 0\n    outer loop\n";
        for &i in t {
            let p = m.pos[i as usize];
            s += &format!("      vertex {:e} {} {}\n", p[0], p[1], p[2]);
        }
        s += "    endloop\n  endfacet\n";
    }
    (s + "endsolid a vertex and a facet in its name\n").into_bytes()
}

fn stl_binary(m: &Mesh) -> Vec<u8> {
    let mut out = b"solid, it says, though this one is binary".to_vec();
    out.resize(80, b' ');
    out.extend((m.tri.len() as u32).to_le_bytes());
    for t in &m.tri {
        out.extend([0u8; 12]);
        for &i in t { out.extend(m.pos[i as usize].iter().flat_map(|c| c.to_le_bytes())); }
        out.extend([0u8; 2]);
    }
    out
}

// Extra vertex properties (a list among them) around x, z and y, two elements that are not wanted between
// the vertices and the faces, and a property before the corners of a face.
fn ply_text(m: &Mesh) -> Vec<u8> {
    let faces = polys(m);
    let mut s = format!(
        "ply\nformat ascii 1.0\ncomment written by a test\nobj_info nothing much\nelement vertex {}\nproperty float nx\nproperty float x\nproperty uchar red\nproperty float z\nproperty list uchar int near\nproperty float y\nelement edge 3\nproperty int a\nproperty list uchar float weights\nelement material 2\nproperty uchar r\nproperty float shine\nelement face {}\nproperty uchar flag\nproperty list uchar int vertex_indices\nend_header\n",
        m.pos.len(), faces.len()
    );
    for (i, p) in m.pos.iter().enumerate() {
        s += &format!("0.5 {} {} {} 2 {} {} {}\n", p[0], 200 + i, p[2], i, i + 1, p[1]);
    }
    s += "1 0\n2 1 0.5\n3 2 0.5 0.25\n255 0.5\n0 1e3\n"; // the edges, each with a list, and the materials
    for f in &faces {
        s += &format!("7 {} {}\n", f.len(), f.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
    }
    s.into_bytes()
}

// Coordinates as doubles among other properties, two elements that are not wanted (one with a list, to be
// walked; one without, to be stepped over whole), and the elements in a different order in each byte order.
fn ply_binary(m: &Mesh, big: bool) -> Vec<u8> {
    let faces = polys(m);
    let vertex = format!(
        "element vertex {}\nproperty double x\nproperty float confidence\nproperty double y\nproperty list uchar short tags\nproperty double z\nproperty ushort id\n",
        m.pos.len()
    );
    let face = format!("element face {}\nproperty list uchar uint vertex_index\nproperty float quality\n", faces.len());
    let edge = "element edge 3\nproperty int a\nproperty list ushort double weights\nproperty char kind\n".to_string();
    let material = "element material 2\nproperty uchar r\nproperty double shine\n".to_string();
    let order = if big { [&edge, &face, &material, &vertex] } else { [&vertex, &edge, &material, &face] };
    let header = format!(
        "ply\nformat binary_{}_endian 1.0\r\ncomment in a different order in each byte order\n{}{}{}{}end_header\n",
        if big { "big" } else { "little" }, order[0], order[1], order[2], order[3]
    );
    let mut out = header.into_bytes();
    macro_rules! put {
        ($v:expr) => { if big { out.extend($v.to_be_bytes()) } else { out.extend($v.to_le_bytes()) } };
    }
    for part in order {
        if part.starts_with("element vertex") {
            for (i, p) in m.pos.iter().enumerate() {
                put!(p[0] as f64);
                put!(0.75f32);
                put!(p[1] as f64);
                put!((i % 3) as u8);
                for k in 0..i % 3 { put!(k as i16); }
                put!(p[2] as f64);
                put!(i as u16);
            }
        } else if part.starts_with("element face") {
            for f in &faces {
                put!(f.len() as u8);
                for &i in f { put!(i); }
                put!(1.5f32);
            }
        } else if part.starts_with("element edge") {
            for e in 0..3u16 {
                put!(-(e as i32));
                put!(e);
                for k in 0..e { put!(k as f64 * 0.5); }
                put!(-3i8);
            }
        } else {
            for k in 0..2u8 {
                put!(k);
                put!(0.25f64);
            }
        }
    }
    out
}

fn b64(bytes: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().fold(0u32, |n, &b| n << 8 | b as u32) << (8 * (3 - chunk.len()));
        for k in 0..4 {
            out.push(if k <= chunk.len() { ABC[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

fn data_uri(bytes: &[u8]) -> String {
    format!("data:application/octet-stream;base64,{}", b64(bytes))
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// A .gltf with its buffer inside it as base64: float positions and u32 indices, the mesh on a child node
// whose parent is moved, turned a quarter turn about x and doubled in size.
fn gltf_doc(m: &Mesh, uri: String) -> (Value, Vec<u8>) {
    let mut buffer = f32s(&m.pos.concat());
    let indices_at = buffer.len();
    buffer.extend(m.tri.iter().flatten().flat_map(|i| i.to_le_bytes()));
    let doc = json!({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"children": [1], "translation": [1, 2, 3], "rotation": [0.7071067811865476, 0, 0, 0.7071067811865476], "scale": [2, 2, 2]},
            {"mesh": 0, "translation": [0, 0, 1], "name": "the box"}
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"name": "glass"}],
        "buffers": [{"byteLength": buffer.len(), "uri": uri}],
        "bufferViews": [
            {"buffer": 0, "byteLength": indices_at},
            {"buffer": 0, "byteOffset": indices_at, "byteLength": buffer.len() - indices_at}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": m.pos.len(), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": m.tri.len() * 3, "type": "SCALAR"}
        ]
    });
    (doc, buffer)
}

fn gltf(m: &Mesh) -> Vec<u8> {
    let (_, buffer) = gltf_doc(m, String::new());
    serde_json::to_vec(&gltf_doc(m, data_uri(&buffer)).0).unwrap()
}

// Where the .gltf's two nodes put the mesh: (p + (0, 0, 1)) doubled, turned so that y goes to z and z
// to -y, then moved by (1, 2, 3).
fn gltf_expected(m: &Mesh) -> Mesh {
    moved(m, |[x, y, z]| [2.0 * x + 1.0, -2.0 * (z + 1.0) + 2.0, 2.0 * y + 3.0])
}

// A .glb: u16 indices, then (after bytes that belong to nothing) positions with a normal between each
// and the next, 24 bytes apart. Two nodes use the one mesh: one moves it, one has a matrix.
fn glb_parts(m: &Mesh) -> (Value, Vec<u8>) {
    let mut bin: Vec<u8> = m.tri.iter().flatten().flat_map(|&i| (i as u16).to_le_bytes()).collect();
    let indices_len = bin.len();
    while bin.len() % 4 != 0 { bin.push(0); }
    let view_at = bin.len();
    bin.extend([0xAAu8; 8]);
    for p in &m.pos { bin.extend(f32s(&[p[0], p[1], p[2], 0.0, 0.0, 1.0])); }
    let doc = json!({
        "asset": {"version": "2.0", "generator": "a test"},
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0, "translation": [10, 0, 0]},
            {"mesh": 0, "matrix": [0, 2, 0, 0, -2, 0, 0, 0, 0, 0, 2, 0, 0, 5, 0, 1]}
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "NORMAL": 2}, "indices": 0, "mode": 4}]}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": indices_len, "target": 34963},
            {"buffer": 0, "byteOffset": view_at, "byteLength": bin.len() - view_at, "byteStride": 24, "target": 34962}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": m.tri.len() * 3, "type": "SCALAR"},
            {"bufferView": 1, "byteOffset": 8, "componentType": 5126, "count": m.pos.len(), "type": "VEC3"},
            {"bufferView": 1, "byteOffset": 20, "componentType": 5126, "count": m.pos.len(), "type": "VEC3"}
        ]
    });
    (doc, bin)
}

// The container: 12 bytes of header, the JSON chunk filled out with spaces, the binary chunk with zeros.
fn glb_packed(doc: &Value, bin: &[u8]) -> Vec<u8> {
    let mut json = serde_json::to_vec(doc).unwrap();
    while json.len() % 4 != 0 { json.push(b' '); }
    let mut bin = bin.to_vec();
    while bin.len() % 4 != 0 { bin.push(0); }
    let mut out = b"glTF".to_vec();
    out.extend(2u32.to_le_bytes());
    out.extend(((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    out.extend((json.len() as u32).to_le_bytes());
    out.extend(0x4E4F534Au32.to_le_bytes());
    out.extend(&json);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(0x004E4942u32.to_le_bytes());
    out.extend(&bin);
    out
}

fn glb(m: &Mesh) -> Vec<u8> {
    let (doc, bin) = glb_parts(m);
    glb_packed(&doc, &bin)
}

// The first node moves the mesh by (10, 0, 0); the second's matrix doubles it, turns x towards y and
// lifts it by 5.
fn glb_expected(m: &Mesh) -> Mesh {
    joined(&moved(m, |[x, y, z]| [x + 10.0, y, z]), &moved(m, |[x, y, z]| [-2.0 * y, 2.0 * x + 5.0, 2.0 * z]))
}

// One mesh in one format, and what reading it should give.
struct Sample { name: &'static str, format: Format, bytes: Vec<u8>, expect: Mesh }

fn samples(m: &Mesh) -> Vec<Sample> {
    let sample = |name, format, bytes, expect| Sample { name, format, bytes, expect };
    let mut out = indexed_samples(m);
    out.push(sample("text STL", Format::Stl, stl_text(m), m.clone()));
    out.push(sample("binary STL", Format::Stl, stl_binary(m), m.clone()));
    out
}

// The formats that have indices in them (an STL has none: each triangle carries its own corners).
fn indexed_samples(m: &Mesh) -> Vec<Sample> {
    let sample = |name, format, bytes, expect| Sample { name, format, bytes, expect };
    vec![
        sample("OBJ", Format::Obj, obj(m), m.clone()),
        sample("text PLY", Format::Ply, ply_text(m), m.clone()),
        sample("little-endian PLY", Format::Ply, ply_binary(m, false), m.clone()),
        sample("big-endian PLY", Format::Ply, ply_binary(m, true), m.clone()),
        sample("glTF", Format::Gltf, gltf(m), gltf_expected(m)),
        sample("GLB", Format::Glb, glb(m), glb_expected(m)),
    ]
}

const FORMATS: [Format; 5] = [Format::Obj, Format::Stl, Format::Ply, Format::Gltf, Format::Glb];

fn text(s: &str, format: Format) -> Result<Mesh, String> {
    parse(s.as_bytes(), format, None)
}

// `bytes` with the first `from` in it turned into `to`.
fn swapped(bytes: &[u8], from: &str, to: &str) -> Vec<u8> {
    let at = bytes.windows(from.len()).position(|w| w == from.as_bytes()).unwrap_or_else(|| panic!("no \"{from}\" to swap"));
    [&bytes[..at], to.as_bytes(), &bytes[at + from.len()..]].concat()
}

fn patched(bytes: &[u8], at: usize, with: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[at..at + with.len()].copy_from_slice(with);
    out
}

// ---- 1 and 2: every format gives the mesh back, and glTF's transforms put it where they should ----

#[test]
fn every_format_gives_back_the_mesh() {
    for source in [cube(), tetra()] {
        for s in samples(&source) {
            let got = parse(&s.bytes, s.format, None).unwrap_or_else(|e| panic!("{}: {e}", s.name));
            sound(&got);
            assert_eq!(got.tri.len(), s.expect.tri.len(), "{}: triangles", s.name);
            assert_eq!(triangles(&got), triangles(&s.expect), "{}: the triangles' corners", s.name);
            assert!((volume(&got) - volume(&s.expect)).abs() < 1e-4, "{}: the triangles face the wrong way", s.name);
            match s.format {
                // nothing is shared between an STL's triangles
                Format::Stl => assert_eq!(got.pos.len(), 3 * got.tri.len(), "{}", s.name),
                // these come back exactly as written: nothing welded, nothing reordered
                Format::Obj | Format::Ply => assert_eq!(got, s.expect, "{}", s.name),
                Format::Gltf | Format::Glb => {}
            }
        }
    }
    assert!((volume(&cube()) - 6.0).abs() < 1e-5 && (volume(&tetra()) - 1.0 / 6.0).abs() < 1e-6);
}

#[test]
fn gltf_transforms_put_the_positions_where_they_should_be() {
    let source = cube();
    let close = |a: &Mesh, b: &Mesh| a.tri == b.tri && a.pos.len() == b.pos.len()
        && a.pos.iter().flatten().zip(b.pos.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-5);

    // parent (translation, rotation, scale) times child (translation)
    let got = parse(&gltf(&source), Format::Gltf, None).unwrap();
    assert!(close(&got, &gltf_expected(&source)), "{:?}", got.pos);
    assert_eq!(got.pos[0], [0.0, -3.0, 3.5]); // the corner (-0.5, 0.25, 1.5), worked by hand

    // the mesh once for each node that uses it: moved, then through a matrix
    let got = parse(&glb(&source), Format::Glb, None).unwrap();
    assert_eq!((got.pos.len(), got.tri.len()), (16, 24));
    assert!(close(&got, &glb_expected(&source)), "{:?}", got.pos);
    assert_eq!((got.pos[0], got.pos[8]), ([9.5, 0.25, 1.5], [-0.5, 4.0, 3.0]));
    assert!((volume(&got) - (6.0 + 48.0)).abs() < 1e-3);

    // a matrix on the parent and translation, rotation and scale on the child, three levels deep
    let (mut doc, buffer) = gltf_doc(&tetra(), String::new());
    doc["buffers"][0]["uri"] = json!(data_uri(&buffer));
    doc["scenes"] = json!([{"nodes": [2]}]);
    doc["nodes"] = json!([
        {"children": [1], "scale": [1, 2, 4]},
        {"mesh": 0, "rotation": [0, 0, 1, 0]},
        {"children": [0], "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 200, 300, 1]}
    ]);
    // half a turn about z (x, y -> -x, -y), then the scale, then the move
    let expect = moved(&tetra(), |[x, y, z]| [100.0 - x, 200.0 - 2.0 * y, 300.0 + 4.0 * z]);
    let got = parse(&serde_json::to_vec(&doc).unwrap(), Format::Gltf, None).unwrap();
    assert!(close(&got, &expect), "{:?}", got.pos);
}

// A small .gltf of one mesh: the buffer inside it, and the views, accessors and primitives as given.
fn small_gltf(buffer: &[u8], views: Value, accessors: Value, primitives: Value) -> Value {
    json!({
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "buffers": [{"byteLength": buffer.len(), "uri": data_uri(buffer)}],
        "bufferViews": views,
        "accessors": accessors
    })
}

fn read_doc(doc: &Value) -> Result<Mesh, String> {
    parse(&serde_json::to_vec(doc).unwrap(), Format::Gltf, None)
}

const SQUARE: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

// Four corners of a square as floats, then `indices` as single bytes.
fn square_gltf(indices: &[u8], primitives: Value) -> Value {
    let buffer = [f32s(&SQUARE), indices.to_vec()].concat();
    small_gltf(
        &buffer,
        json!([{"buffer": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": indices.len()}]),
        json!([
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": indices.len(), "type": "SCALAR"}
        ]),
        primitives,
    )
}

#[test]
fn gltf_strips_fans_and_the_rest() {
    let flat = |m: &Mesh| volume(m) == 0.0 && m.tri.iter().all(|t| {
        // every triangle of the square faces +z
        let [a, b, c] = t.map(|i| m.pos[i as usize]);
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0
    });
    // a strip with no indices: the vertices in order, every other triangle turned round
    let strip = read_doc(&square_gltf(&[], json!([{"attributes": {"POSITION": 0}, "mode": 5}]))).unwrap();
    assert_eq!(strip.tri, [[0, 1, 2], [2, 1, 3]]);
    assert!(flat(&strip));
    // a fan, through single-byte indices (and a buffer whose length is not a multiple of three, so its
    // base64 ends in "=")
    let fan = read_doc(&square_gltf(&[0, 1, 3, 2], json!([{"attributes": {"POSITION": 0}, "indices": 1, "mode": 6}]))).unwrap();
    assert_eq!(fan.tri, [[0, 1, 3], [0, 3, 2]]);
    assert!(flat(&fan));
    for extra in 0..3 {
        let indices = [&[0u8, 1, 2, 2, 1, 3][..], &[0u8; 2][..extra]].concat();
        let doc = square_gltf(&indices, json!([{"attributes": {"POSITION": 0}, "indices": 1}]));
        assert_eq!(read_doc(&doc).unwrap().tri.len(), indices.len() / 3, "base64 with {extra} bytes over");
    }
    // points and lines are passed over; two primitives that share their positions share them in the mesh
    let both = read_doc(&square_gltf(&[0, 1, 2, 2, 1, 3], json!([
        {"attributes": {"POSITION": 0}, "mode": 0},
        {"attributes": {"POSITION": 0}, "indices": 1, "mode": 1},
        {"attributes": {"POSITION": 0}, "indices": 1},
        {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
    ]))).unwrap();
    assert_eq!((both.pos.len(), both.tri.len()), (4, 4));
    let only_lines = read_doc(&square_gltf(&[0, 1, 2], json!([{"attributes": {"POSITION": 0}, "indices": 1, "mode": 1}])));
    assert_eq!(only_lines.unwrap_err(), "no triangles");

    // whole-number positions (KHR_mesh_quantization): normalized ones are fractions, the others as they are
    let shorts: Vec<u8> = [32767i16, 0, -32767, -32768, 16384, 0, 0, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let quantized = |normalized: bool| small_gltf(
        &shorts,
        json!([{"buffer": 0, "byteLength": 18}]),
        json!([{"bufferView": 0, "componentType": 5122, "normalized": normalized, "count": 3, "type": "VEC3"}]),
        json!([{"attributes": {"POSITION": 0}}]),
    );
    let mut doc = quantized(true);
    doc["extensionsRequired"] = json!(["KHR_mesh_quantization", "KHR_materials_transmission", "KHR_texture_transform", "KHR_lights_punctual"]);
    assert_eq!(read_doc(&doc).unwrap().pos, [[1.0, 0.0, -1.0], [-1.0, 16384.0 / 32767.0, 0.0], [0.0, 0.0, 0.0]]);
    assert_eq!(read_doc(&quantized(false)).unwrap().pos[1], [-32768.0, 16384.0, 0.0]);
    // for floats "normalized" means nothing
    let mut doc = square_gltf(&[0, 1, 2], json!([{"attributes": {"POSITION": 0}, "indices": 1}]));
    doc["accessors"][0]["normalized"] = json!(true);
    doc["buffers"][0]["uri"] = json!(data_uri(&[f32s(&[-5.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]), vec![0, 1, 2]].concat()));
    assert_eq!(read_doc(&doc).unwrap().pos[..2], [[-5.0, 0.0, 0.0], [7.0, 0.0, 0.0]]);

    // no scenes at all: every node that is no other's child, here node 1 and through it node 0
    let mut doc = square_gltf(&[0, 1, 2], json!([{"attributes": {"POSITION": 0}, "indices": 1}]));
    doc.as_object_mut().unwrap().remove("scenes");
    doc["nodes"] = json!([{"mesh": 0}, {"children": [0], "translation": [0, 0, 7]}]);
    assert_eq!(read_doc(&doc).unwrap().pos[3], [1.0, 1.0, 7.0]);
    // the scene the file names is the one that is read
    doc["nodes"] = json!([{"mesh": 0}, {"mesh": 0, "translation": [0, 0, 7]}]);
    doc["scenes"] = json!([{"nodes": [0]}, {"nodes": [1]}]);
    assert_eq!(read_doc(&doc).unwrap().pos[0], [0.0, 0.0, 0.0]);
    doc["scene"] = json!(1);
    let second = read_doc(&doc).unwrap();
    assert_eq!((second.pos.len(), second.pos[0]), (4, [0.0, 0.0, 7.0]));
    doc["scene"] = json!(2);
    assert!(read_doc(&doc).is_err());

    // a chain of nodes far deeper than any stack would stand, each moving up by one
    let depth = 100_000;
    let nodes: Vec<Value> = (0..depth).map(|n| if n + 1 < depth { json!({"children": [n + 1], "translation": [0, 0, 1]}) } else { json!({"mesh": 0}) }).collect();
    doc["nodes"] = Value::Array(nodes);
    doc["scene"] = json!(0);
    assert_eq!(read_doc(&doc).unwrap().pos[0], [0.0, 0.0, (depth - 1) as f32]);
}

// ---- 3: bad input is an error, never a panic ----

#[test]
fn empty_input_is_an_error() {
    for format in FORMATS {
        assert!(parse(&[], format, None).is_err(), "{}", format.name());
        assert!(parse(b" ", format, None).is_err(), "{}", format.name());
        assert!(parse(b"\n\n\n", format, None).is_err(), "{}", format.name());
    }
}

#[test]
fn every_prefix_of_a_good_file_is_an_error_or_a_smaller_mesh() {
    for source in [cube(), tetra()] {
        for s in samples(&source) {
            let mut errors = 0;
            for n in 0..s.bytes.len() {
                match parse(&s.bytes[..n], s.format, None) {
                    Ok(mesh) => {
                        sound(&mesh);
                        assert!(mesh.tri.len() <= s.expect.tri.len(), "{} cut at {n}: more triangles than the whole file", s.name);
                    }
                    Err(e) => {
                        assert!(!e.is_empty());
                        errors += 1;
                    }
                }
            }
            // a text file cut at the right place is a smaller file; a binary one, or JSON, never is
            let all = matches!(s.name, "binary STL" | "little-endian PLY" | "big-endian PLY" | "glTF" | "GLB");
            if all {
                assert_eq!(errors, s.bytes.len(), "{}: a cut file was read", s.name);
            } else {
                assert!(errors * 2 > s.bytes.len(), "{}: only {errors} of {} cuts were errors", s.name, s.bytes.len());
            }
        }
    }
}

#[test]
fn one_format_read_as_another_is_an_error() {
    for s in samples(&cube()) {
        for format in FORMATS.into_iter().filter(|f| *f != s.format) {
            assert!(parse(&s.bytes, format, None).is_err(), "{} read as {}", s.name, format.name());
        }
    }
    // the two kinds of glTF say which they are
    assert!(parse(&glb(&cube()), Format::Gltf, None).unwrap_err().contains(".glb"));
    assert!(parse(&gltf(&cube()), Format::Glb, None).unwrap_err().contains(".gltf"));
}

#[test]
fn counts_the_file_cannot_hold_are_refused_without_asking_for_memory() {
    let source = cube();
    let max = u32::MAX.to_le_bytes();

    // binary STL: a hundred bytes that say they are four thousand million triangles
    let mut tiny = vec![b' '; 100];
    tiny[80..84].copy_from_slice(&max);
    let e = refused_cheaply("tiny STL", &tiny, Format::Stl);
    assert!(e.contains("4294967295") && e.contains("214748364834") && e.contains("100 bytes"), "{e}");
    refused_cheaply("STL count", &patched(&stl_binary(&source), 80, &max), Format::Stl);
    refused_cheaply("STL count", &patched(&stl_binary(&source), 80, &11u32.to_le_bytes()), Format::Stl);
    refused_cheaply("STL count", &patched(&stl_binary(&source), 80, &13u32.to_le_bytes()), Format::Stl);

    // PLY: each element's count, set past anything and set to something the limits would allow
    for big in [false, true] {
        let good = ply_binary(&source, big);
        for (element, count) in [("vertex", 8), ("face", 6), ("edge", 3), ("material", 2)] {
            for lie in ["4294967295", "18446744073709551615", "20000000", "9000000"] {
                let bad = swapped(&good, &format!("element {element} {count}"), &format!("element {element} {lie}"));
                refused_cheaply(&format!("binary PLY, {lie} of {element}"), &bad, Format::Ply);
            }
        }
    }
    for (element, count) in [("vertex", 8), ("face", 6)] {
        for lie in ["4294967295", "20000000", "9000000"] {
            let bad = swapped(&ply_text(&source), &format!("element {element} {count}"), &format!("element {element} {lie}"));
            refused_cheaply(&format!("text PLY, {lie} of {element}"), &bad, Format::Ply);
        }
    }
    // a count with nothing in each row has nothing to read, however large it is
    let empty = swapped(&ply_text(&source), "element face", "element nothing 4000000000\nelement face");
    assert_eq!(parse(&empty, Format::Ply, None).unwrap(), source);

    // GLB: the length of the file, of the JSON chunk and of the binary chunk
    let good = glb(&source);
    let json_len = u32::from_le_bytes(good[12..16].try_into().unwrap()) as usize;
    for at in [8, 12, 20 + json_len] {
        refused_cheaply(&format!("GLB length at {at}"), &patched(&good, at, &max), Format::Glb);
        refused_cheaply(&format!("GLB length at {at}"), &patched(&good, at, &0x7FFF_FFF0u32.to_le_bytes()), Format::Glb);
    }
    assert!(parse(&patched(&good, 4, &1u32.to_le_bytes()), Format::Glb, None).unwrap_err().contains("version 1"));

    // glTF: how many items an accessor says it has, with data behind it and with none
    let (doc, bin) = glb_parts(&source);
    for accessor in 0..2 {
        for lie in [u32::MAX as u64, 20_000_000, u64::MAX] {
            let mut bad = doc.clone();
            bad["accessors"][accessor]["count"] = json!(lie);
            refused_cheaply(&format!("GLB accessor {accessor} of {lie}"), &glb_packed(&bad, &bin), Format::Glb);
            bad["accessors"][accessor].as_object_mut().unwrap().remove("bufferView");
            refused_cheaply(&format!("GLB accessor {accessor} of {lie}, no data"), &glb_packed(&bad, &bin), Format::Glb);
        }
        let (mut bad, buffer) = gltf_doc(&source, String::new());
        bad["buffers"][0]["uri"] = json!(data_uri(&buffer));
        bad["accessors"][accessor]["count"] = json!(u32::MAX);
        refused_cheaply(&format!("glTF accessor {accessor}"), &serde_json::to_vec(&bad).unwrap(), Format::Gltf);
    }
    // ranges that run off the end of what holds them
    for (path, value) in [
        (("bufferViews", 1, "byteLength"), 1 << 40), (("bufferViews", 1, "byteOffset"), 1 << 40), (("bufferViews", 1, "byteStride"), 250),
        (("bufferViews", 1, "byteStride"), 4), (("accessors", 1, "byteOffset"), 21), (("accessors", 1, "byteOffset"), u64::MAX),
        (("bufferViews", 0, "byteOffset"), u64::MAX), (("bufferViews", 0, "buffer"), 1), (("accessors", 0, "bufferView"), 9),
    ] {
        let mut bad = doc.clone();
        bad[path.0][path.1][path.2] = json!(value);
        refused_cheaply(&format!("GLB {path:?} = {value}"), &glb_packed(&bad, &bin), Format::Glb);
    }
}

#[test]
fn positions_that_are_not_finite_are_errors() {
    for (value, says) in [(f32::NAN, "NaN"), (f32::INFINITY, "infinite"), (f32::NEG_INFINITY, "infinite"), (1e19, "larger than 1e18"), (-3e30, "larger than 1e18")] {
        let mut bad = cube();
        bad.pos[5][1] = value;
        for s in samples(&bad) {
            let e = parse(&s.bytes, s.format, None).expect_err(s.name);
            // (a glTF's transform can make a NaN of an infinity: nothing times infinity is not a number)
            let turned = matches!(s.format, Format::Gltf | Format::Glb) && value.is_infinite() && e.contains("NaN");
            assert!(e.contains(says) || turned, "{} with {value}: {e}", s.name);
        }
    }
    // a number too large for a float is infinite
    assert!(text("v 0 0 0\nv 1e39 0 0\nv 0 1 0\nf 1 2 3\n", Format::Obj).unwrap_err().contains("infinite"));
    assert!(text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", Format::Obj).is_ok());
    // a position that no triangle uses does not spoil the file, and is not left as it was
    let spare = text("v 0 0 0\nv 1 0 0\nv nan inf 1e30\nv 0 1 0\nf 1 2 4\n", Format::Obj).unwrap();
    sound(&spare);
    assert_eq!((spare.pos.len(), spare.tri[0]), (4, [0, 1, 3]));
    // a matrix that sends everything off to infinity
    let (mut doc, bin) = glb_parts(&cube());
    doc["nodes"][0]["translation"] = json!([1e300, 0, 0]);
    assert!(parse(&glb_packed(&doc, &bin), Format::Glb, None).is_err());
}

#[test]
fn indices_out_of_range_are_errors() {
    let mut bad = cube();
    bad.tri[7][2] = 8;
    for s in indexed_samples(&bad) {
        let e = parse(&s.bytes, s.format, None).expect_err(s.name);
        assert!(e.contains('8') || e.contains('9'), "{}: {e}", s.name);
    }
    let triangle = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
    let obj = |faces: &str| text(&format!("{triangle}{faces}"), Format::Obj);
    assert_eq!(obj("f 1 2 9\n").unwrap_err(), "line 4: a face names vertex 9 but there are 3");
    assert_eq!(obj("f 1 2 3\nf 1 0 3\n").unwrap_err(), "line 5: a face names vertex 0 (they count from 1)");
    assert!(obj("f -1 -2 -4\n").unwrap_err().contains("line 4"));
    assert!(obj("f 1 2 99999999999999999999\n").is_err());
    assert!(obj("f 1 2 4000000000\n").is_err());
    assert!(obj("f 1 2 -9223372036854775808\n").is_err());
    assert!(obj("f 1 2 x\n").is_err());
    assert!(obj("f 1/1/1 2/2/2 /3\n").is_err());
    assert!(obj("v 1 2\n").unwrap_err().contains("line 4"));
    assert_eq!(obj("").unwrap_err(), "no faces");
    assert_eq!(obj("f 1 2\nl 1 2 3\n").unwrap_err(), "no faces");

    // negative numbers count back from the vertices read so far, and a face may name a vertex that
    // comes after it in the file
    let mesh = text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nv 0 0 1\nf -1 -4 -3\nf 1 2 5\nv 9 9 9\n", Format::Obj).unwrap();
    assert_eq!(mesh.tri, [[0, 1, 2], [3, 0, 1], [0, 1, 4]]);

    // PLY: below nothing, past the end, and not a whole number
    for index in ["8", "-1", "1.5", "4294967296", "nan"] {
        let bad = swapped(&ply_text(&cube()), "7 4 0 3 2 1", &format!("7 4 0 3 2 {index}"));
        assert!(parse(&bad, Format::Ply, None).unwrap_err().contains("face 0"), "PLY index {index}");
    }
    // glTF: the limit is the number of positions the primitive has
    let doc = square_gltf(&[0, 1, 4], json!([{"attributes": {"POSITION": 0}, "indices": 1}]));
    assert!(read_doc(&doc).unwrap_err().contains("an index of 4"));
}

#[test]
fn stl_is_told_apart_by_its_size() {
    let source = tetra();
    // binary although it begins "solid"; and still binary with the word "facet" in its header
    let binary = stl_binary(&source);
    assert!(binary.starts_with(b"solid"));
    assert_eq!(parse(&binary, Format::Stl, None).unwrap().tri.len(), 4);
    let mut named = binary.clone();
    named[..11].copy_from_slice(b"solid facet");
    assert_eq!(parse(&named, Format::Stl, None).unwrap().tri.len(), 4);
    // one byte over: neither the size its count gives, nor text. Both numbers are in the error.
    let mut over = binary.clone();
    over.push(0);
    let e = parse(&over, Format::Stl, None).unwrap_err();
    assert!(e.contains("284 bytes") && e.contains("285 bytes") && e.contains("4 triangles"), "{e}");
    // text, in capitals, with space before it
    let shout = format!("\r\n  {}", String::from_utf8(stl_text(&source)).unwrap().to_uppercase());
    assert_eq!(triangles(&text(&shout, Format::Stl).unwrap()), triangles(&source));
    // facets that are not three corners
    let good = String::from_utf8(stl_text(&source)).unwrap();
    assert!(text(&good.replacen("      vertex 0e0 0 0\n", "", 1), Format::Stl).unwrap_err().contains("three corners"));
    assert!(text(&good.replacen("endfacet", "", 1), Format::Stl).is_err());
    assert!(text(&good.replacen("vertex 0e0 0 0", "vertex 0e0 0", 1), Format::Stl).is_err());
    assert!(text("solid nothing\nendsolid nothing\n", Format::Stl).is_err());
}

#[test]
fn ply_says_what_is_missing() {
    let good = String::from_utf8(ply_text(&cube())).unwrap();
    let ply = |s: String| text(&s, Format::Ply);
    assert!(ply(good.replace("property float z\n", "property float depth\n")).unwrap_err().contains("no z property"));
    assert!(ply(good.replace("property float x\n", "property list uchar float x\n")).unwrap_err().contains("no x property"));
    assert!(ply(good.replace("element face 6", "element strip 6")).unwrap_err().contains("no face element"));
    assert!(ply(good.replace("vertex_indices", "corners")).unwrap_err().contains("vertex_indices"));
    assert!(ply(good.replace("element vertex 8", "element point 8")).unwrap_err().contains("no vertex element"));
    assert!(ply(good.replace("format ascii 1.0\n", "")).unwrap_err().contains("format"));
    assert!(ply(good.replace("format ascii", "format binary_middle_endian")).is_err());
    assert!(ply(good.replace("property uchar red", "property colour red")).unwrap_err().contains("line 8"));
    assert!(ply(good.replace("end_header\n", "")).is_err());
    assert!(ply(good.replace("obj_info nothing much", "element vertex 3")).unwrap_err().contains("second vertex"));
    // faces of fewer than three corners make nothing
    let header = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_index\nend_header\n0 0 0\n1 0 0\n0 1 0\n";
    assert_eq!(ply(format!("{header}2 0 1\n")).unwrap_err(), "no faces");
    assert_eq!(ply(format!("{header}3 0 1 2\n")).unwrap().tri, [[0, 1, 2]]);
    assert_eq!(ply(format!("{header}3 0 1 2").replace('\n', "\r\n")).unwrap().tri, [[0, 1, 2]]);
}

#[test]
fn gltf_says_what_it_does_not_read() {
    let (doc, bin) = glb_parts(&cube());
    let with = |change: &dyn Fn(&mut Value)| {
        let mut doc = doc.clone();
        change(&mut doc);
        parse(&glb_packed(&doc, &bin), Format::Glb, None)
    };
    assert!(with(&|_| {}).is_ok());

    // a node that is its own descendant, directly and by way of another
    assert!(with(&|d| d["nodes"][0]["children"] = json!([0])).unwrap_err().contains("node 0"));
    assert!(with(&|d| { d["nodes"][0]["children"] = json!([1]); d["nodes"][1]["children"] = json!([0]); }).unwrap_err().contains("reached through itself"));
    // nodes that are not there, or not numbers
    assert!(with(&|d| d["scenes"][0]["nodes"] = json!([0, 2])).is_err());
    assert!(with(&|d| d["nodes"][0]["children"] = json!(["one"])).is_err());
    assert!(with(&|d| d["nodes"][1]["mesh"] = json!(3)).is_err());
    assert!(with(&|d| d["nodes"][1]["matrix"] = json!([1, 0, 0])).unwrap_err().contains("16 numbers"));
    assert!(with(&|d| d["nodes"][0]["translation"] = json!("left")).is_err());

    // every node used twice over by the one above it: forty levels of that would be a million million visits
    let twice: Vec<Value> = (0..40).map(|n| if n < 39 { json!({"children": [n + 1, n + 1]}) } else { json!({"mesh": 0}) }).collect();
    assert!(with(&|d| { d["nodes"] = json!(twice); d["scenes"] = json!([{"nodes": [0]}]); }).unwrap_err().contains("more than 2000000 nodes"));
    // and thousands of nodes each using a mesh of thousands of primitives that come to nothing
    let points: Vec<Value> = (0..3000).map(|_| json!({"attributes": {"POSITION": 1}, "mode": 0})).collect();
    let users: Vec<Value> = (0..3000).map(|_| json!({"mesh": 0})).collect();
    let all: Vec<usize> = (0..3000).collect();
    let e = with(&|d| { d["meshes"][0]["primitives"] = json!(points); d["nodes"] = json!(users); d["scenes"] = json!([{"nodes": all}]); });
    assert!(e.unwrap_err().contains("more than 2000000 nodes and primitives"));

    // compression, and anything else the file says it cannot be read without
    let e = with(&|d| d["extensionsRequired"] = json!(["KHR_draco_mesh_compression"])).unwrap_err();
    assert!(e.contains("KHR_draco_mesh_compression") && e.contains("Draco"), "{e}");
    assert!(with(&|d| d["extensionsRequired"] = json!(["KHR_materials_ior", "EXT_meshopt_compression"])).unwrap_err().contains("EXT_meshopt_compression"));
    assert!(with(&|d| d["extensionsRequired"] = json!(["VENDOR_something"])).unwrap_err().contains("VENDOR_something"));
    assert!(with(&|d| d["extensionsRequired"] = json!(["KHR_materials_volume", "KHR_mesh_quantization"])).is_ok());
    let draco = with(&|d| {
        d["meshes"][0]["primitives"][0]["extensions"] = json!({"KHR_draco_mesh_compression": {"bufferView": 0}});
        d["accessors"][1].as_object_mut().unwrap().remove("bufferView");
    });
    assert!(draco.unwrap_err().contains("Draco"));

    // sparse accessors, for the positions or for the indices
    for accessor in 0..2 {
        let e = with(&|d| d["accessors"][accessor]["sparse"] = json!({"count": 1})).unwrap_err();
        assert!(e.contains("sparse") && e.contains(["indices", "POSITION"][accessor]), "{e}");
    }
    // kinds of accessor that positions and indices cannot be
    assert!(with(&|d| d["accessors"][1]["type"] = json!("VEC2")).is_err());
    assert!(with(&|d| d["accessors"][1]["componentType"] = json!(5125)).is_err());
    assert!(with(&|d| d["accessors"][0]["componentType"] = json!(5126)).is_err());
    assert!(with(&|d| d["accessors"][0]["type"] = json!("VEC3")).is_err());
    assert!(with(&|d| { d["meshes"][0]["primitives"][0]["attributes"].as_object_mut().unwrap().remove("POSITION"); }).unwrap_err().contains("POSITION"));
    // an accessor with no data is all zeros, which makes no surface
    assert_eq!(with(&|d| { d["accessors"][1].as_object_mut().unwrap().remove("bufferView"); }).unwrap_err(), "no triangles");
    // other versions
    assert!(with(&|d| d["asset"]["version"] = json!("1.0")).unwrap_err().contains("version 1.0"));
    // a GLB with no binary chunk, whose buffer therefore means nothing
    let mut json = serde_json::to_vec(&doc).unwrap();
    while json.len() % 4 != 0 { json.push(b' '); }
    let mut bare = b"glTF".to_vec();
    for word in [2, 20 + json.len() as u32, json.len() as u32, 0x4E4F534A] { bare.extend(word.to_le_bytes()); }
    bare.extend(&json);
    assert!(parse(&bare, Format::Glb, None).unwrap_err().contains("buffer 0"));

    // JSON that is not a scene, or nests without end
    for junk in ["[]", "3", "null", "{}", "{\"scenes\": 4, \"nodes\": {}}", "{\"asset\": {\"version\": \"2.0\"}, \"scenes\": [{\"nodes\": [0]}]}"] {
        assert!(text(junk, Format::Gltf).is_err(), "{junk}");
    }
    assert!(text(&"[".repeat(100_000), Format::Gltf).is_err());
    assert!(text(&format!("{}{}", "{\"a\":".repeat(50_000), "1"), Format::Gltf).is_err());
}

#[test]
fn gltf_buffers_are_only_read_from_beside_the_model() {
    let dir = test_dir("uris");
    let (doc, buffer) = gltf_doc(&cube(), "box data.bin".to_string());
    std::fs::write(dir.join("box data.bin"), &buffer).unwrap();
    std::fs::create_dir_all(dir.join("inner")).unwrap();
    std::fs::write(dir.join("inner").join("deep.bin"), &buffer).unwrap();
    let inner = dir.join("inner");
    let with = |uri: &str, dir: Option<&Path>| {
        let mut doc = doc.clone();
        doc["buffers"][0]["uri"] = json!(uri);
        parse(&serde_json::to_vec(&doc).unwrap(), Format::Gltf, dir)
    };
    // beside the model, with a space as it is and as %20, and in a folder below
    for uri in ["box data.bin", "box%20data.bin", "inner/deep.bin", "./box%20data.bin", "inner%2Fdeep.bin"] {
        assert_eq!(triangles(&with(uri, Some(&dir)).unwrap()), triangles(&gltf_expected(&cube())), "{uri}");
    }
    assert!(with("box data.bin", None).unwrap_err().contains("no folder"));
    assert!(with("missing.bin", Some(&dir)).unwrap_err().contains("missing.bin"));
    // the file is really there one folder up, and is still not read
    let absolute = dir.join("box data.bin").to_string_lossy().into_owned();
    for uri in [
        "../box data.bin", "..\\box data.bin", "%2e%2e/box data.bin", "..%2Fbox%20data.bin", "deeper/../../box data.bin", ".. /box data.bin",
        "C:/x.bin", "c:x.bin", "/x.bin", "\\x.bin", "\\\\server\\share\\x.bin", "file:///C:/x.bin", "https://example.com/x.bin",
        "NUL", "con.bin", "inner/COM1", absolute.as_str(),
    ] {
        let e = with(uri, Some(&inner)).expect_err(uri);
        assert!(e.contains("which is not read"), "{uri}: {e}");
    }
    assert!(with("../x.bin", Some(&inner)).unwrap_err().contains(".."));
    // data: URIs that are not base64, or not good base64
    for uri in ["data:,hello", "data:application/octet-stream;base64", "data:application/octet-stream;base64,AAA*AAAA", "data:;base64,A", "data:;base64,AA AA"] {
        assert!(with(uri, Some(&dir)).is_err(), "{uri}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- 4: random bytes and damaged files never panic ----

// xorshift, with a fixed seed so that every run tries the same things
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 16) % n.max(1) as u64) as usize
    }
}

// Read `bytes` as every format. Nothing may panic, and whatever is read has to be a sound mesh.
fn read_as_everything(bytes: &[u8]) -> usize {
    FORMATS.into_iter().filter(|&format| parse(bytes, format, None).map(|mesh| sound(&mesh)).is_ok()).count()
}

#[test]
fn random_bytes_never_panic() {
    let mut random = Random(0x9E37_79B9_7F4A_7C15);
    let good: Vec<Vec<u8>> = samples(&tetra()).into_iter().map(|s| s.bytes).collect();
    let words: Vec<&[u8]> = vec![
        b"v", b"f", b"vn", b"\\", b"\n", b"\n", b" ", b" ", b"/", b"//", b"-", b"1", b"2", b"3", b"0", b"9", b".", b"e", b"nan", b"inf",
        b"solid", b"facet", b"normal", b"outer", b"loop", b"vertex", b"endloop", b"endfacet", b"endsolid", b"ply", b"format", b"ascii",
        b"binary_little_endian", b"binary_big_endian", b"1.0", b"element", b"face", b"property", b"list", b"uchar", b"int", b"float",
        b"double", b"x", b"y", b"z", b"vertex_indices", b"end_header", b"comment", b"4294967295", b"\r\n", b"\x00", b"\xff",
    ];
    let mut read = 0;
    for round in 0..6000 {
        let mut bytes = Vec::new();
        // some begin as a good file does, so that more than the first check is reached
        if round % 3 != 0 {
            let start = &good[random.below(good.len())];
            bytes.extend(&start[..random.below(start.len())]);
        }
        let more = random.below(400);
        if round % 2 == 0 {
            bytes.extend((0..more).map(|_| random.next() as u8));
        } else {
            for _ in 0..more / 3 {
                bytes.extend(words[random.below(words.len())]);
                if random.below(3) > 0 { bytes.push(b' '); }
            }
        }
        read += read_as_everything(&bytes);
    }
    assert!(read > 0, "none of the random files was readable, which says the test is not reaching far");
}

#[test]
fn damaged_files_never_panic() {
    let mut random = Random(0x2545_F491_4F6C_DD1D);
    for source in [cube(), tetra()] {
        for s in samples(&source) {
            // every byte in turn: cleared, filled, its top bit turned over, and made a digit
            for at in 0..s.bytes.len() {
                for change in 0..4 {
                    let mut bytes = s.bytes.clone();
                    bytes[at] = match change { 0 => 0, 1 => 0xFF, 2 => bytes[at] ^ 0x80, _ => b'9' };
                    read_as_everything(&bytes);
                }
            }
            // then at random: one byte, or a few, or a piece cut out, or a piece said twice
            for round in 0..4000 {
                let mut bytes = s.bytes.clone();
                match round % 8 {
                    0..=4 => { let at = random.below(bytes.len()); bytes[at] = random.next() as u8; }
                    5 => for _ in 0..1 + random.below(6) { let at = random.below(bytes.len()); bytes[at] = random.next() as u8; },
                    6 => { let at = random.below(bytes.len()); let len = random.below(bytes.len() - at).min(40); bytes.drain(at..at + len); }
                    _ => { let at = random.below(bytes.len()); let len = random.below(bytes.len() - at).min(40); let piece = bytes[at..at + len].to_vec(); bytes.splice(at..at, piece); }
                }
                read_as_everything(&bytes);
            }
        }
    }
}

// Every place in a JSON document, as the pointer that leads to it ("/nodes/0/mesh").
fn places(v: &Value, here: String, out: &mut Vec<String>) {
    match v {
        Value::Array(items) => for (i, item) in items.iter().enumerate() { places(item, format!("{here}/{i}"), out); },
        Value::Object(map) => for (key, item) in map { places(item, format!("{here}/{key}"), out); },
        _ => {}
    }
    if !here.is_empty() { out.push(here); }
}

#[test]
fn gltf_with_any_value_changed_or_removed_never_panics() {
    let swaps = [
        json!(0), json!(1), json!(2), json!(3), json!(7), json!(255), json!(65536), json!(u32::MAX), json!(u64::MAX), json!(-1), json!(0.5),
        json!(1e300), json!(-1e300), json!(null), json!(true), json!("x"), json!([]), json!({}), json!([0, 1]), json!([[0]]),
        json!({"sparse": {}}), json!("data:;base64,AAAA"), json!([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    ];
    let source = tetra();
    let (packed, bin) = glb_parts(&source);
    let (mut inside, buffer) = gltf_doc(&source, String::new());
    inside["buffers"][0]["uri"] = json!(data_uri(&buffer));
    let mut read = 0;
    for (doc, bin) in [(&packed, Some(&bin)), (&inside, None)] {
        let mut check = |changed: &Value| {
            let result = match bin {
                Some(bin) => parse(&glb_packed(changed, bin), Format::Glb, None),
                None => parse(&serde_json::to_vec(changed).unwrap(), Format::Gltf, None),
            };
            if let Ok(mesh) = result {
                sound(&mesh);
                read += 1;
            }
        };
        let mut all = Vec::new();
        places(doc, String::new(), &mut all);
        for place in &all {
            for swap in &swaps {
                let mut changed = doc.clone();
                *changed.pointer_mut(place).unwrap() = swap.clone();
                check(&changed);
            }
            let (parent, key) = place.rsplit_once('/').unwrap();
            let mut changed = doc.clone();
            match changed.pointer_mut(parent).unwrap() {
                Value::Object(map) => { map.remove(key); }
                Value::Array(items) => { items.remove(key.parse().unwrap()); }
                _ => unreachable!(),
            }
            check(&changed);
        }
    }
    assert!(read > 100, "only {read} of the changed files could be read, which says the changes are not reaching far");
}

#[test]
fn ply_headers_with_any_word_changed_never_panic() {
    let swaps = [
        "0", "1", "3", "255", "65536", "4294967295", "18446744073709551616", "-1", "1.5", "list", "uchar", "char", "ushort", "double", "float",
        "int", "x", "vertex", "face", "vertex_indices", "element", "property", "end_header", "comment", "format", "ascii", "binary_big_endian",
        "binary_little_endian", "", "\n",
    ];
    let source = tetra();
    let mut read = 0;
    for bytes in [ply_text(&source), ply_binary(&source, false), ply_binary(&source, true)] {
        let header = bytes.windows(10).position(|w| w == b"end_header").unwrap() + 10;
        let mut start = 0;
        for at in 0..=header {
            if at < header && !bytes[at].is_ascii_whitespace() { continue; }
            for swap in swaps.iter().filter(|_| start < at) {
                let changed = [&bytes[..start], swap.as_bytes(), &bytes[at..]].concat();
                if let Ok(mesh) = parse(&changed, Format::Ply, None) {
                    sound(&mesh);
                    read += 1;
                }
            }
            start = at + 1;
        }
    }
    assert!(read > 20, "only {read} of the changed files could be read");
}

// ---- 5: files on disk ----

// A new folder under out\models\test_read for one test's files.
fn test_dir(tag: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("out").join("models").join("test_read");
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = root.join(format!("{tag}_{}_{stamp}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn format_goes_by_the_extension() {
    for (name, format) in [("a.obj", Format::Obj), ("A.STL", Format::Stl), ("dir.gltf/b.Ply", Format::Ply), ("c.gltf", Format::Gltf), ("d.e.GlB", Format::Glb)] {
        assert_eq!(Format::from_path(Path::new(name)), Some(format), "{name}");
    }
    for name in ["a.fbx", "obj", "a.obj.bak", "a.", "", ".stl/readme", "a.objx"] {
        assert_eq!(Format::from_path(Path::new(name)), None, "{name}");
    }
    assert_eq!(FORMATS.map(Format::name), ["OBJ", "STL", "PLY", "glTF", "GLB"]);
    assert_eq!(EXTENSIONS, ["obj", "stl", "ply", "gltf", "glb"]);
    for (extension, format) in EXTENSIONS.iter().zip(FORMATS) {
        assert_eq!(Format::from_path(&Path::new("model").with_extension(extension)), Some(format));
    }
}

#[test]
fn files_are_read_from_disk() {
    let dir = test_dir("files");
    let source = cube();

    // a path that is not there, and names that are not model files: the error begins with the file's name
    let e = read_file(&dir.join("nothing.obj")).unwrap_err();
    assert!(e.starts_with("nothing.obj: ") && !e.contains("test_read"), "{e}");
    std::fs::write(dir.join("model.fbx"), b"Kaydara FBX Binary").unwrap();
    for name in ["model.fbx", "missing.fbx", "model", "notes.txt"] {
        let e = read_file(&dir.join(name)).unwrap_err();
        assert!(e.starts_with(&format!("{name}: ")) && e.contains("obj, stl, ply, gltf, glb"), "{e}");
    }
    // a folder with a model's name, and a file with nothing in it
    std::fs::create_dir_all(dir.join("folder.stl")).unwrap();
    assert!(read_file(&dir.join("folder.stl")).unwrap_err().starts_with("folder.stl: "));
    std::fs::write(dir.join("empty.ply"), b"").unwrap();
    assert_eq!(read_file(&dir.join("empty.ply")).unwrap_err(), "empty.ply: the file is empty");
    // what is wrong inside a file, after its name
    std::fs::write(dir.join("teapot.obj"), "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n").unwrap();
    assert_eq!(read_file(&dir.join("teapot.obj")).unwrap_err(), "teapot.obj: line 4: a face names vertex 9 but there are 3");

    // every format, by its extension in whatever case
    let (apart, buffer) = gltf_doc(&source, "apart%20data.bin".to_string());
    std::fs::write(dir.join("apart data.bin"), &buffer).unwrap();
    let files = [
        ("box.obj", obj(&source), source.clone()),
        ("text.stl", stl_text(&source), source.clone()),
        ("BOX.STL", stl_binary(&source), source.clone()),
        ("box.ply", ply_text(&source), source.clone()),
        ("little.Ply", ply_binary(&source, false), source.clone()),
        ("big.PLY", ply_binary(&source, true), source.clone()),
        ("inside.gltf", gltf(&source), gltf_expected(&source)),
        ("apart.gltf", serde_json::to_vec(&apart).unwrap(), gltf_expected(&source)),
        ("box.glb", glb(&source), glb_expected(&source)),
    ];
    for (name, bytes, expect) in &files {
        std::fs::write(dir.join(name), bytes).unwrap();
        let got = read_file(&dir.join(name)).unwrap_or_else(|e| panic!("{e}"));
        sound(&got);
        assert_eq!(triangles(&got), triangles(expect), "{name}");
    }
    // the same file under another format's name is an error that names it
    std::fs::write(dir.join("wrong.glb"), &files[0].1).unwrap();
    assert!(read_file(&dir.join("wrong.glb")).unwrap_err().starts_with("wrong.glb: not a GLB file"));
    // a .gltf whose buffer file has gone
    std::fs::remove_file(dir.join("apart data.bin")).unwrap();
    let e = read_file(&dir.join("apart.gltf")).unwrap_err();
    assert!(e.starts_with("apart.gltf: ") && e.contains("apart data.bin"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- how long a large file takes (run with --ignored --nocapture) ----

#[test]
#[ignore = "a timing, not a check: about a million triangles in each format"]
fn a_million_triangles() {
    // a sheet of 708 by 708 squares, gently rippled
    let side = 708u32;
    let mut sheet = Mesh::default();
    for j in 0..=side {
        for i in 0..=side {
            sheet.pos.push([i as f32 * 0.01, j as f32 * 0.01, ((i * 7 + j * 13) % 100) as f32 * 0.001]);
        }
    }
    for j in 0..side {
        for i in 0..side {
            let a = j * (side + 1) + i;
            sheet.tri.extend([[a, a + 1, a + side + 2], [a, a + side + 2, a + side + 1]]);
        }
    }
    let mut ply = ply_binary(&Mesh { pos: sheet.pos.clone(), tri: vec![] }, false);
    ply = swapped(&ply, "element face 0", &format!("element face {}", sheet.tri.len()));
    for t in &sheet.tri {
        ply.push(3);
        ply.extend(t.iter().flat_map(|i| i.to_le_bytes()));
        ply.extend(1.5f32.to_le_bytes());
    }
    let (mut doc, _) = glb_parts(&tetra());
    let bin = [f32s(&sheet.pos.concat()), sheet.tri.iter().flatten().flat_map(|i| i.to_le_bytes()).collect()].concat();
    doc["bufferViews"] = json!([{"buffer": 0, "byteLength": sheet.pos.len() * 12}, {"buffer": 0, "byteOffset": sheet.pos.len() * 12, "byteLength": sheet.tri.len() * 12}]);
    doc["accessors"] = json!([
        {"bufferView": 1, "componentType": 5125, "count": sheet.tri.len() * 3, "type": "SCALAR"},
        {"bufferView": 0, "componentType": 5126, "count": sheet.pos.len(), "type": "VEC3"}
    ]);
    doc["meshes"][0]["primitives"][0]["attributes"] = json!({"POSITION": 1});
    doc["scenes"] = json!([{"nodes": [0]}]);
    let files = [
        ("binary STL", Format::Stl, stl_binary(&sheet)),
        ("text STL", Format::Stl, stl_text(&sheet)),
        ("OBJ", Format::Obj, obj(&sheet)),
        ("text PLY", Format::Ply, ply_text(&sheet)),
        ("binary PLY", Format::Ply, ply),
        ("GLB", Format::Glb, glb_packed(&doc, &bin)),
    ];
    for (name, format, bytes) in &files {
        let start = std::time::Instant::now();
        let mesh = parse(bytes, *format, None).unwrap_or_else(|e| panic!("{name}: {e}"));
        let took = start.elapsed();
        assert_eq!(mesh.tri.len(), sheet.tri.len(), "{name}");
        eprintln!("{name:>10}: {:>4} MB, {} triangles read in {:.0} ms", bytes.len() >> 20, mesh.tri.len(), took.as_secs_f64() * 1000.0);
    }
}
