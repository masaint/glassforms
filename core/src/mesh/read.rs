//! Model files turned into a plain list of triangles.
//!
//! A file is untrusted data. Whatever is wrong with one comes back as an `Err` that says so in plain
//! English, with the line or the byte where that helps; nothing here panics on bad data, and no count
//! found in a file is believed (or has memory set aside for it) until it has been checked against the
//! bytes that are really there.
//!
//! What each reader takes, and what it leaves out:
//!   OBJ    `v` and `f` lines: the `v`, `v/vt`, `v//vn` and `v/vt/vn` forms, negative numbers counting
//!          back, polygons fanned from their first corner, `\` carrying a line on. Texture
//!          coordinates and MTL material slots are available through `read_asset`. Groups, lines and points are passed over.
//!   STL    binary and text. Three positions for every triangle, as the file has them (nothing is
//!          shared between triangles); the file's normals are ignored.
//!   PLY    text and both binary byte orders. The `vertex` element's x, y and z and the `face`
//!          element's list of corners, polygons fanned; every other property and element is stepped over.
//!   glTF   2.0, as `.gltf` and `.glb`: the triangles, strips and fans of the default scene, each node's
//!          transform applied. Buffers come from the GLB itself, from base64 `data:` URIs, or from files
//!          beside the model. The appearance reader takes PBR factors, UVs, tangents, PNG/JPEG colour and
//!          normal and metallic-roughness maps. Skins, animations and morphs are omitted. Sparse accessors and compressed geometry are refused
//!          by name.

use super::Mesh;
mod appearance;
pub use appearance::{Appearance, CanonicalFaceUvs, FaceAppearance, ImageMap, ImportedAlphaMode, ImportedCompatibility, ImportedMaterial, MaterialMap, ModelAsset};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

/// Limits, stated in the error when a file goes over one.
pub const MAX_FILE_BYTES: u64 = 1 << 30; // 1 GB
pub const MAX_TRIANGLES: usize = 10_000_000;
pub const MAX_VERTICES: usize = 30_000_000;

// A coordinate further out than this would overflow the arithmetic done on it later.
const MAX_COORD: f32 = 1e18;
// The most nodes and primitives a glTF scene is followed through. Its nodes are meant to be a tree, each
// met once; a file that uses them over and over could otherwise ask for more steps than there is time for.
const MAX_SCENE_PARTS: usize = 2_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { Obj, Stl, Ply, Gltf, Glb }

/// The extensions that can be read, lower case, for file choosers and error messages.
pub const EXTENSIONS: [&str; 5] = ["obj", "stl", "ply", "gltf", "glb"];

impl Format {
    /// From a file name's extension, any case: obj, stl, ply, gltf, glb. None for anything else.
    pub fn from_path(path: &Path) -> Option<Format> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        let formats = [Format::Obj, Format::Stl, Format::Ply, Format::Gltf, Format::Glb];
        EXTENSIONS.iter().zip(formats).find(|(e, _)| **e == ext).map(|(_, format)| format)
    }

    pub fn name(self) -> &'static str {
        match self { Format::Obj => "OBJ", Format::Stl => "STL", Format::Ply => "PLY", Format::Gltf => "glTF", Format::Glb => "GLB" }
    }
}

/// Read a model file. The format goes by the extension, and the file's size is checked against
/// MAX_FILE_BYTES before any of it is read. Every error begins with the file's name.
pub fn read_file(path: &Path) -> Result<Mesh, String> {
    read_model(path, false).map(|asset| asset.mesh)
}

/// Read authored materials and per-corner texture coordinates alongside the geometry.
pub fn read_asset(path: &Path) -> Result<ModelAsset, String> {
    match read_model(path, true) {
        Ok(asset) => Ok(asset),
        Err(appearance_error) => {
            let mut asset = read_model(path, false)?;
            asset.appearance_notice = Some(appearance_error);
            Ok(asset)
        }
    }
}

fn read_model(path: &Path, materials: bool) -> Result<ModelAsset, String> {
    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    let named = |e: String| format!("{name}: {e}");
    let format = Format::from_path(path)
        .ok_or_else(|| named(format!("not a kind of model file that can be read (these can: {})", EXTENSIONS.join(", "))))?;
    let size = std::fs::metadata(path).map_err(|e| named(format!("the file cannot be opened ({e})")))?.len();
    if size > MAX_FILE_BYTES { return Err(named(too_big(size))); }
    let bytes = std::fs::read(path).map_err(|e| named(format!("the file cannot be read ({e})")))?;
    parse_model(&bytes, format, path.parent(), materials).map_err(named)
}

/// The same from bytes already in memory. `dir` is the folder a .gltf's separate buffer files are looked
/// for in (None: a .gltf that needs one is an error).
pub fn parse(bytes: &[u8], format: Format, dir: Option<&Path>) -> Result<Mesh, String> {
    parse_model(bytes, format, dir, false).map(|asset| asset.mesh)
}

fn parse_model(bytes: &[u8], format: Format, dir: Option<&Path>, materials: bool) -> Result<ModelAsset, String> {
    let mut surface = materials.then(Appearance::default);
    if bytes.len() as u64 > MAX_FILE_BYTES { return Err(too_big(bytes.len() as u64)); }
    if bytes.is_empty() { return Err("the file is empty".into()); }
    let mesh = match format {
        Format::Obj => read_obj(bytes, dir, surface.as_mut()),
        Format::Stl => read_stl(bytes),
        Format::Ply => read_ply(bytes),
        Format::Gltf => read_gltf(bytes, dir, surface.as_mut()),
        Format::Glb => read_glb(bytes, dir, surface.as_mut()),
    }?;
    let mesh = checked(mesh)?;
    Ok(ModelAsset { mesh, appearance_notice: None, appearance: surface.filter(|s| !s.materials.is_empty() && s.faces.iter().any(|f| f.material.is_some())) })
}

fn too_big(size: u64) -> String { format!("the file is {size} bytes; the most that can be read is {MAX_FILE_BYTES} (1 GB)") }
fn too_many(what: &str, most: usize) -> String { format!("more than {most} {what}, the most that can be read") }

// What every reader's result has to pass before it is handed on: some triangles, counts within the
// limits, every index a real position, and every position that a triangle uses a finite number of a
// sensible size. A bad position that no triangle uses does no harm, and is moved onto one that is used
// so that nothing later has to look out for it.
fn checked(mut mesh: Mesh) -> Result<Mesh, String> {
    let count = mesh.pos.len();
    if mesh.tri.is_empty() { return Err("no triangles".into()); }
    if mesh.tri.len() > MAX_TRIANGLES { return Err(too_many("triangles", MAX_TRIANGLES)); }
    if count > MAX_VERTICES { return Err(too_many("vertices", MAX_VERTICES)); }
    let mut used = vec![false; count];
    for &i in mesh.tri.iter().flatten() {
        match used.get_mut(i as usize) {
            Some(u) => *u = true,
            None => return Err(format!("a triangle names vertex {i} but there are {count}")),
        }
    }
    let good = |p: &[f32; 3]| p.iter().all(|c| c.abs() <= MAX_COORD);
    let mut spare = None;
    for (i, (p, _)) in mesh.pos.iter().zip(&used).enumerate().filter(|(_, (_, used))| **used) {
        if !good(p) {
            let why = if p.iter().any(|c| c.is_nan()) { "is not a number (NaN)" }
                else if p.iter().any(|c| c.is_infinite()) { "is infinite" }
                else { "is larger than 1e18" };
            return Err(format!("vertex {} of {count} has a coordinate that {why}", i + 1));
        }
        spare.get_or_insert(*p);
    }
    if let Some(spare) = spare {
        for (p, used) in mesh.pos.iter_mut().zip(&used) {
            if !*used && !good(p) { *p = spare; }
        }
    }
    Ok(mesh)
}

fn push_pos(mesh: &mut Mesh, p: [f32; 3]) -> Result<(), String> {
    if mesh.pos.len() >= MAX_VERTICES { return Err(too_many("vertices", MAX_VERTICES)); }
    mesh.pos.push(p);
    Ok(())
}

fn push_tri(mesh: &mut Mesh, t: [u32; 3]) -> Result<(), String> {
    if mesh.tri.len() >= MAX_TRIANGLES { return Err(too_many("triangles", MAX_TRIANGLES)); }
    mesh.tri.push(t);
    Ok(())
}

// ---- pieces the readers share ----

// `N` bytes of `b` from `at` on, None when they are not all there.
fn take<const N: usize>(b: &[u8], at: usize) -> Option<[u8; N]> { b.get(at..at.checked_add(N)?)?.try_into().ok() }

// A word of text as a number, None when it is not one.
fn num<T: std::str::FromStr>(word: &[u8]) -> Option<T> { std::str::from_utf8(word).ok()?.parse().ok() }

// A number from a file as a place in a list of `count`, None when it is not a whole number inside it.
fn index(v: f64, count: usize) -> Option<u32> {
    (v >= 0.0 && v < count as f64 && v.fract() == 0.0).then_some(v as u32)
}

// A piece of a file as text for an error message, cut short when it is long.
fn show(word: &[u8]) -> String { String::from_utf8_lossy(word.get(..40).unwrap_or(word)).into_owned() }

// The next line of `rest` without its line ending, and `rest` moved past it.
fn next_line<'a>(rest: &mut &'a [u8]) -> &'a [u8] {
    let all = *rest;
    let end = all.iter().position(|&b| b == b'\n').unwrap_or(all.len());
    let line = all.get(..end).unwrap_or(all);
    *rest = all.get(end + 1..).unwrap_or(&[]);
    line.strip_suffix(b"\r").unwrap_or(line)
}

// The words of a text file one after another, whatever space or line ends lie between them, keeping
// count of the line for error messages.
struct Words<'a> { b: &'a [u8], at: usize, line: usize }

impl<'a> Words<'a> {
    fn next(&mut self) -> Option<&'a [u8]> {
        while let Some(&c) = self.b.get(self.at) {
            if !c.is_ascii_whitespace() { break; }
            if c == b'\n' { self.line += 1; }
            self.at += 1;
        }
        let start = self.at;
        while self.b.get(self.at).is_some_and(|c| !c.is_ascii_whitespace()) { self.at += 1; }
        self.b.get(start..self.at).filter(|word| !word.is_empty())
    }

    fn skip_line(&mut self) { while self.b.get(self.at).is_some_and(|&c| c != b'\n') { self.at += 1; } }
}

// The kinds of number a PLY property or a glTF accessor can hold.
#[derive(Clone, Copy)]
enum Num { I8, U8, I16, U16, I32, U32, F32, F64 }

impl Num {
    // A PLY type name, under either of its two spellings.
    fn named(word: &[u8]) -> Option<Num> {
        Some(match word {
            b"char" | b"int8" => Num::I8, b"uchar" | b"uint8" => Num::U8,
            b"short" | b"int16" => Num::I16, b"ushort" | b"uint16" => Num::U16,
            b"int" | b"int32" => Num::I32, b"uint" | b"uint32" => Num::U32,
            b"float" | b"float32" => Num::F32, b"double" | b"float64" => Num::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self { Num::I8 | Num::U8 => 1, Num::I16 | Num::U16 => 2, Num::I32 | Num::U32 | Num::F32 => 4, Num::F64 => 8 }
    }

    // The number at `at` in `b`, in either byte order. None when its bytes are not all there.
    fn read(self, b: &[u8], at: usize, big: bool) -> Option<f64> {
        macro_rules! get {
            ($t:ty) => {{
                let raw = take(b, at)?;
                (if big { <$t>::from_be_bytes(raw) } else { <$t>::from_le_bytes(raw) }) as f64
            }};
        }
        Some(match self {
            Num::I8 => get!(i8), Num::U8 => get!(u8), Num::I16 => get!(i16), Num::U16 => get!(u16),
            Num::I32 => get!(i32), Num::U32 => get!(u32), Num::F32 => get!(f32), Num::F64 => get!(f64),
        })
    }
}

// ---- OBJ ----

fn read_obj(bytes: &[u8], dir: Option<&Path>, mut surface: Option<&mut Appearance>) -> Result<Mesh, String> {
    let mut obj = appearance::ObjMaterials::default();
    let mut mesh = Mesh::default();
    let (mut rest, mut line_no) = (bytes, 0usize);
    let mut joined: Vec<u8> = Vec::new();
    // The highest vertex number a face has named before that vertex was read, and the line it was on.
    // Whether there is such a vertex is only known at the end of the file.
    let mut ahead = (0usize, 0usize);
    while !rest.is_empty() {
        let mut line = next_line(&mut rest);
        line_no += 1;
        let at = line_no;
        // a line that ends in a backslash carries on into the next one
        if let Some(mut head) = line.strip_suffix(b"\\") {
            joined.clear();
            loop {
                joined.extend_from_slice(head);
                joined.push(b' ');
                if rest.is_empty() { break; }
                let more = next_line(&mut rest);
                line_no += 1;
                match more.strip_suffix(b"\\") {
                    Some(h) => head = h,
                    None => { joined.extend_from_slice(more); break; }
                }
            }
            line = &joined;
        }
        let mut words = line.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty());
        match words.next() {
            Some(b"vt") if surface.is_some() => obj.texcoord(words, at)?,
            Some(b"mtllib") if surface.is_some() => {
                let rest = std::str::from_utf8(line).map_err(|_| "the material filename is not UTF-8")?;
                obj.library(dir, rest.trim_start().strip_prefix("mtllib").unwrap_or("").trim(), surface.as_deref_mut().unwrap())?;
            }
            Some(b"usemtl") if surface.is_some() => {
                let rest = std::str::from_utf8(line).map_err(|_| "the material name is not UTF-8")?;
                obj.select(rest.trim_start().strip_prefix("usemtl").unwrap_or("").trim());
            }
            Some(b"v") => {
                // x y z; a weight or a colour after them is not wanted
                let mut p = [0f32; 3];
                for c in &mut p {
                    *c = words.next().and_then(num).ok_or_else(|| format!("line {at}: a vertex needs three numbers"))?;
                }
                push_pos(&mut mesh, p)?;
            }
            Some(b"f") => {
                let (mut first, mut prev) = (0u32, 0u32);
                let mut uv_corner = [[0.0; 2]; 3];
                for (corner, word) in words.enumerate() {
                    // of v/vt/vn only the first, the position, is wanted
                    let text = word.split(|&b| b == b'/').next().unwrap_or(word);
                    let i: i64 = num(text).ok_or_else(|| format!("line {at}: \"{}\" in a face is not a vertex number", show(word)))?;
                    // 1 is the first vertex of the file; -1 is the last one read so far
                    let (have, size) = (mesh.pos.len(), i.unsigned_abs());
                    let place = if i > 0 && size <= MAX_VERTICES as u64 {
                        if size as usize > have.max(ahead.0) { ahead = (size as usize, at); }
                        size as usize - 1
                    } else if i < 0 && size <= have as u64 {
                        have - size as usize
                    } else if i == 0 {
                        return Err(format!("line {at}: a face names vertex 0 (they count from 1)"));
                    } else {
                        return Err(format!("line {at}: a face names vertex {i} but only {have} have been read"));
                    };
                    let v = place as u32;
                    if let Some(s) = surface.as_deref_mut() {
                        if corner == 0 { s.faces.resize(mesh.tri.len(), FaceAppearance::default()); }
                        let uv = obj.corner(word);
                        match corner {
                            0 => uv_corner[0] = uv,
                            1 => uv_corner[1] = uv,
                            _ => {
                                uv_corner[2] = uv;
                                s.faces.push(FaceAppearance { base_uv: uv_corner, normal_uv: uv_corner, canonical_uv: CanonicalFaceUvs { base: uv_corner, normal: uv_corner, ..Default::default() }, material: obj.current, ..Default::default() });
                                uv_corner[1] = uv;
                            }
                        }
                    }
                    // a polygon is fanned out from its first corner; fewer than three corners make nothing
                    match corner {
                        0 => first = v,
                        1 => prev = v,
                        _ => { push_tri(&mut mesh, [first, prev, v])?; prev = v; }
                    }
                }
            }
            _ => {}
        }
    }
    if ahead.0 > mesh.pos.len() {
        return Err(format!("line {}: a face names vertex {} but there are {}", ahead.1, ahead.0, mesh.pos.len()));
    }
    if mesh.tri.is_empty() { return Err("no faces".into()); }
    Ok(mesh)
}

// ---- STL ----

fn read_stl(bytes: &[u8]) -> Result<Mesh, String> {
    let size = bytes.len() as u64;
    let count = take(bytes, 80).map(u32::from_le_bytes);
    let binary_size = |n: u32| 84 + 50 * n as u64;
    // The size settles it: some binary files begin with "solid" too, so the first word alone cannot.
    if let Some(n) = count.filter(|&n| binary_size(n) == size) { return stl_binary(bytes, n as usize); }
    let is_text = bytes.trim_ascii_start().get(..5).is_some_and(|s| s.eq_ignore_ascii_case(b"solid"))
        && bytes.windows(5).any(|w| w.eq_ignore_ascii_case(b"facet"));
    if is_text { return stl_text(bytes); }
    Err(match count.map(|n| (n, binary_size(n))) {
        Some((n, want)) => format!("the size does not match the count: its {n} triangles would make a binary STL of {want} bytes, but the file is {size} bytes (and it is not a text STL)"),
        None => format!("{size} bytes is too short for a binary STL (84 at least), and it is not a text STL"),
    })
}

// 80 bytes of header, the count, then 50 bytes a triangle: a normal, three corners, two spare bytes.
// The count has already been checked against the size of the file.
fn stl_binary(bytes: &[u8], count: usize) -> Result<Mesh, String> {
    if count > MAX_TRIANGLES { return Err(too_many("triangles", MAX_TRIANGLES)); }
    let mut mesh = Mesh { pos: Vec::with_capacity(count * 3), tri: Vec::with_capacity(count) };
    for record in bytes.get(84..).unwrap_or(&[]).chunks_exact(50) {
        let base = mesh.pos.len() as u32;
        for corner in 0..3 {
            let mut p = [0f32; 3];
            for (k, c) in p.iter_mut().enumerate() {
                *c = f32::from_le_bytes(take(record, 12 + corner * 12 + k * 4).ok_or("a triangle is cut short")?);
            }
            mesh.pos.push(p);
        }
        mesh.tri.push([base, base + 1, base + 2]);
    }
    Ok(mesh)
}

// `facet normal .. outer loop / vertex x y z (three times) / endloop / endfacet`, between `solid` and
// `endsolid` lines. Only the facets and their vertices matter here; the other words are passed over.
fn stl_text(bytes: &[u8]) -> Result<Mesh, String> {
    let mut mesh = Mesh::default();
    let mut words = Words { b: bytes, at: 0, line: 1 };
    // how many corners the facet being read has so far; None between facets
    let mut corners: Option<usize> = None;
    while let Some(word) = words.next() {
        let is = |name: &[u8]| word.eq_ignore_ascii_case(name);
        if is(b"solid") || is(b"endsolid") {
            words.skip_line(); // the rest of the line is the solid's name, which could be any words at all
        } else if is(b"facet") {
            if corners.is_some() { return Err(format!("line {}: a facet begins before the one before it has ended", words.line)); }
            corners = Some(0);
        } else if is(b"vertex") {
            let Some(n) = corners.as_mut() else { return Err(format!("line {}: a vertex outside any facet", words.line)) };
            let mut p = [0f32; 3];
            for c in &mut p {
                *c = words.next().and_then(num).ok_or_else(|| format!("line {}: a vertex needs three numbers", words.line))?;
            }
            push_pos(&mut mesh, p)?;
            *n += 1;
        } else if is(b"endfacet") {
            if corners.take() != Some(3) { return Err(format!("line {}: a facet that does not have three corners", words.line)); }
            let base = mesh.pos.len().saturating_sub(3) as u32;
            push_tri(&mut mesh, [base, base + 1, base + 2])?;
        }
    }
    if corners.is_some() { return Err(format!("the file ends inside a facet (line {})", words.line)); }
    Ok(mesh)
}

// ---- PLY ----

struct Prop<'a> { name: &'a [u8], ty: Num, list: Option<Num> } // `list`: the type of a list's length, when it is a list
struct Element<'a> { name: &'a [u8], count: usize, props: Vec<Prop<'a>> }

// What a property is wanted for.
enum Role { Skip, Axis(usize), Corners }

// Where a PLY's numbers come from: words of text, or bytes in one of the two byte orders.
enum Source<'a> { Text(Words<'a>), Binary { b: &'a [u8], at: usize, big: bool } }

impl Source<'_> {
    fn left(&self) -> usize {
        match self { Source::Text(w) => w.b.len().saturating_sub(w.at), Source::Binary { b, at, .. } => b.len().saturating_sub(*at) }
    }

    fn num(&mut self, ty: Num) -> Result<f64, String> {
        match self {
            Source::Text(words) => {
                let word = words.next().ok_or_else(|| format!("the file ends early, at line {}", words.line))?;
                num(word).ok_or_else(|| format!("line {}: \"{}\" is not a number", words.line, show(word)))
            }
            Source::Binary { b, at, big } => {
                let v = ty.read(b, *at, *big).ok_or_else(|| format!("the file ends early (a number is wanted at byte {at} of {})", b.len()))?;
                *at += ty.size();
                Ok(v)
            }
        }
    }

    // Step over `n` numbers of one type.
    fn skip(&mut self, ty: Num, n: usize) -> Result<(), String> {
        match self {
            Source::Text(words) => {
                for _ in 0..n { words.next().ok_or_else(|| format!("the file ends early, at line {}", words.line))?; }
            }
            Source::Binary { b, at, .. } => {
                *at = n.checked_mul(ty.size()).and_then(|len| at.checked_add(len)).filter(|&end| end <= b.len())
                    .ok_or_else(|| format!("the file ends early ({} bytes are wanted at byte {at} of {})", n.saturating_mul(ty.size()), b.len()))?;
            }
        }
        Ok(())
    }

    // How many items the list that starts here has.
    fn length(&mut self, ty: Num) -> Result<usize, String> {
        let v = self.num(ty)?;
        if v >= 0.0 && v <= u32::MAX as f64 && v.fract() == 0.0 { Ok(v as usize) } else { Err(format!("a list that says it has {v} items")) }
    }
}

fn read_ply(bytes: &[u8]) -> Result<Mesh, String> {
    if !bytes.starts_with(b"ply") { return Err("not a PLY file (it does not begin with \"ply\")".into()); }
    let (mut rest, mut line_no) = (bytes, 0usize);
    let mut format: Option<&[u8]> = None;
    let mut elements: Vec<Element> = Vec::new();
    // the header: lines of text, up to and including the line that says end_header
    loop {
        if rest.is_empty() { return Err("the header has no end_header line".into()); }
        let line = next_line(&mut rest);
        line_no += 1;
        let bad = || format!("line {line_no}: the header line \"{}\" is not understood", show(line));
        let mut words = line.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty());
        match words.next() {
            Some(b"ply") if line_no == 1 => {}
            Some(b"comment") | Some(b"obj_info") | None => {}
            Some(b"format") => format = Some(words.next().ok_or_else(bad)?),
            Some(b"element") => {
                let name = words.next().ok_or_else(bad)?;
                let count = words.next().and_then(num).ok_or_else(bad)?;
                if matches!(name, b"vertex" | b"face") && elements.iter().any(|e| e.name == name) {
                    return Err(format!("line {line_no}: a second {} element", show(name)));
                }
                elements.push(Element { name, count, props: Vec::new() });
            }
            Some(b"property") => {
                let element = elements.last_mut().ok_or_else(bad)?;
                let mut kind = words.next().ok_or_else(bad)?;
                let mut list = None;
                if kind == b"list" {
                    list = Some(words.next().and_then(Num::named).ok_or_else(bad)?);
                    kind = words.next().ok_or_else(bad)?;
                }
                let ty = Num::named(kind).ok_or_else(bad)?;
                element.props.push(Prop { name: words.next().ok_or_else(bad)?, ty, list });
            }
            Some(b"end_header") => break,
            _ => return Err(bad()),
        }
    }
    let at = bytes.len() - rest.len();
    let mut source = match format {
        Some(b"ascii") => Source::Text(Words { b: bytes, at, line: line_no + 1 }),
        Some(b"binary_little_endian") => Source::Binary { b: bytes, at, big: false },
        Some(b"binary_big_endian") => Source::Binary { b: bytes, at, big: true },
        Some(other) => return Err(format!("a PLY format of \"{}\" is not one of ascii, binary_little_endian, binary_big_endian", show(other))),
        None => return Err("the header has no format line".into()),
    };

    let vertex = elements.iter().find(|e| e.name == b"vertex").ok_or("no vertex element")?;
    for axis in ["x", "y", "z"] {
        if !vertex.props.iter().any(|p| p.list.is_none() && p.name == axis.as_bytes()) {
            return Err(format!("the vertex element has no {axis} property"));
        }
    }
    if vertex.count > MAX_VERTICES { return Err(too_many("vertices", MAX_VERTICES)); }
    let face = elements.iter().find(|e| e.name == b"face").ok_or("no face element (points alone have no surface to show)")?;
    if !face.props.iter().any(|p| p.list.is_some() && matches!(p.name, b"vertex_indices" | b"vertex_index")) {
        return Err("the face element has no vertex_indices list".into());
    }

    let mut mesh = Mesh::default();
    for e in &elements {
        // A row cannot be shorter than this, so a count that the bytes left could not hold is caught
        // here, before anything is set aside for it or walked.
        let text = matches!(source, Source::Text(_));
        let least: usize = e.props.iter().map(|p| if text { 1 } else { p.list.unwrap_or(p.ty).size() }).sum();
        if e.count.checked_mul(least).is_none_or(|need| need > source.left()) {
            return Err(format!("the header says there are {} of \"{}\", but only {} bytes are left for them", e.count, show(e.name), source.left()));
        }
        ply_rows(&mut source, e, vertex.count, &mut mesh).map_err(|err| format!("in the {} data: {err}", show(e.name)))?;
    }
    if mesh.tri.is_empty() { return Err("no faces".into()); }
    Ok(mesh)
}

// One element's rows: positions out of `vertex`, triangles out of `face`, anything else stepped over.
fn ply_rows(src: &mut Source, e: &Element, vertices: usize, mesh: &mut Mesh) -> Result<(), String> {
    let roles: Vec<Role> = e.props.iter().map(|p| match (e.name, p.name, p.list.is_some()) {
        (b"vertex", b"x", false) => Role::Axis(0),
        (b"vertex", b"y", false) => Role::Axis(1),
        (b"vertex", b"z", false) => Role::Axis(2),
        (b"face", b"vertex_indices" | b"vertex_index", true) => Role::Corners,
        _ => Role::Skip,
    }).collect();
    // An element with nothing wanted in it and no lists is so many numbers of each type: stepped over
    // whole. (One with no properties at all has nothing in the file, however many it says there are.)
    if e.props.iter().zip(&roles).all(|(p, role)| p.list.is_none() && matches!(role, Role::Skip)) {
        for p in &e.props { src.skip(p.ty, e.count)?; }
        return Ok(());
    }
    let is_vertex = e.name == b"vertex";
    if is_vertex { mesh.pos.reserve(e.count); }
    for row in 0..e.count {
        let mut p = [0f32; 3];
        for (prop, role) in e.props.iter().zip(&roles) {
            match (prop.list, role) {
                (None, Role::Axis(axis)) => {
                    let v = src.num(prop.ty)? as f32;
                    if let Some(c) = p.get_mut(*axis) { *c = v; }
                }
                (None, _) => src.skip(prop.ty, 1)?,
                (Some(len), Role::Corners) => {
                    // a polygon is fanned out from its first corner; fewer than three corners make nothing
                    let (mut first, mut prev) = (0u32, 0u32);
                    for corner in 0..src.length(len)? {
                        let v = src.num(prop.ty)?;
                        let i = index(v, vertices)
                            .ok_or_else(|| format!("face {row} names vertex {v} but there are {vertices} (they count from 0)"))?;
                        match corner {
                            0 => first = i,
                            1 => prev = i,
                            _ => { push_tri(mesh, [first, prev, i])?; prev = i; }
                        }
                    }
                }
                (Some(len), _) => {
                    let n = src.length(len)?;
                    src.skip(prop.ty, n)?;
                }
            }
        }
        if is_vertex { push_pos(mesh, p)?; }
    }
    Ok(())
}

// ---- glTF ----

const GLB_JSON: u32 = 0x4E4F_534A;
const GLB_BIN: u32 = 0x004E_4942;

// A transform as glTF writes it: sixteen numbers, a column at a time.
type Mat = [f64; 16];
const IDENTITY: Mat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

// a then b's transform as one: b is applied to a point first
fn mul(a: &Mat, b: &Mat) -> Mat {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
        }
    }
    out
}

// A whole number in the JSON that can serve as a count or a place in a list.
fn uint(v: &Value) -> Option<usize> {
    let whole = v.as_u64().or_else(|| v.as_f64().filter(|f| *f >= 0.0 && *f < 9e15 && f.fract() == 0.0).map(|f| f as u64));
    usize::try_from(whole?).ok()
}

fn read_gltf(bytes: &[u8], dir: Option<&Path>, surface: Option<&mut Appearance>) -> Result<Mesh, String> {
    if bytes.starts_with(b"glTF") { return Err("this is a binary glTF, not a text one (its name should end .glb)".into()); }
    gltf_scene(bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes), None, dir, surface)
}

// The binary container: 12 bytes of header, then chunks of (length, type, data). The first chunk is the
// JSON; a binary chunk after it is the buffer that has no uri.
fn read_glb(bytes: &[u8], dir: Option<&Path>, surface: Option<&mut Appearance>) -> Result<Mesh, String> {
    if !bytes.starts_with(b"glTF") {
        let text = bytes.trim_ascii_start().starts_with(b"{");
        return Err(if text { "this is a text glTF, not a binary one (its name should end .gltf)" } else { "not a GLB file (it does not begin with \"glTF\")" }.into());
    }
    let word = |at: usize| take(bytes, at).map(u32::from_le_bytes);
    let (version, total) = word(4).zip(word(8)).ok_or("the file ends inside its 12-byte header")?;
    if version != 2 { return Err(format!("GLB version {version} is not read (only version 2)")); }
    let file = bytes.get(..total as usize).ok_or_else(|| format!("the header says the file is {total} bytes but it is {}", bytes.len()))?;
    let (mut json, mut bin) = (None, None);
    let mut at = 12usize;
    while let (Some(len), Some(kind), Some(rest)) = (word(at), word(at + 4), file.get(at + 8..)) {
        let data = rest.get(..len as usize)
            .ok_or_else(|| format!("the chunk at byte {at} says it is {len} bytes, more than the {} left in the file", rest.len()))?;
        // (the loop ends where there is no room left for a chunk's eight bytes of length and type)
        match kind {
            GLB_JSON if at == 12 => json = Some(data),
            _ if at == 12 => return Err("the first chunk is not the JSON".into()),
            GLB_BIN if bin.is_none() => bin = Some(data),
            _ => {} // chunks of other kinds are allowed, and not wanted
        }
        at += 8 + data.len();
    }
    gltf_scene(json.ok_or("there is no JSON chunk")?, bin, dir, surface)
}

// The scene the JSON describes, as one mesh: every node of the default scene walked, each mesh a node
// uses put through the transforms of the nodes above it.
fn gltf_scene(json: &[u8], bin: Option<&[u8]>, dir: Option<&Path>, mut surface: Option<&mut Appearance>) -> Result<Mesh, String> {
    let doc: Value = serde_json::from_slice(json).map_err(|e| format!("its JSON cannot be read ({e})"))?;
    if !doc.is_object() { return Err("its JSON does not describe a glTF scene".into()); }
    if let Some(version) = doc["asset"]["version"].as_str().filter(|v| !v.starts_with("2.")) {
        return Err(format!("glTF version {version} is not read (only version 2)"));
    }
    // Extensions the file says it cannot be read without. The ones let through change how a surface
    // looks or add lights, or (mesh quantization) only allow whole-number positions, which are read.
    for ext in doc["extensionsRequired"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        let readable = ext == "KHR_mesh_quantization" || ext.starts_with("KHR_materials_") || ext == "KHR_texture_transform" || ext == "KHR_lights_punctual";
        if !readable {
            let what = match ext {
                "KHR_draco_mesh_compression" => "Draco compression (KHR_draco_mesh_compression)".to_string(),
                "EXT_meshopt_compression" | "KHR_meshopt_compression" => format!("meshopt compression ({ext})"),
                _ => format!("the extension {}", show(ext.as_bytes())),
            };
            return Err(format!("the file needs {what}, which is not read"));
        }
    }

    let none = Vec::new();
    let nodes = doc["nodes"].as_array().unwrap_or(&none);
    // Where to start: the scene the file names, else its first; in a file with no scenes, every node
    // that is no other node's child.
    let roots: Vec<Value> = match doc["scenes"].as_array().filter(|scenes| !scenes.is_empty()) {
        Some(scenes) => {
            let which = match doc.get("scene") { Some(v) => uint(v).ok_or("its \"scene\" is not a number")?, None => 0 };
            let scene = scenes.get(which).ok_or_else(|| format!("its scene is number {which} but it has {} (they count from 0)", scenes.len()))?;
            scene["nodes"].as_array().cloned().unwrap_or_default()
        }
        None => {
            let mut child = vec![false; nodes.len()];
            for kid in nodes.iter().filter_map(|node| node["children"].as_array()).flatten().filter_map(uint) {
                if let Some(c) = child.get_mut(kid) { *c = true; }
            }
            child.iter().enumerate().filter(|(_, c)| !**c).map(|(n, _)| Value::from(n)).collect()
        }
    };

    // The way down from the scene to the node being read: one level for each node on it, with the
    // transform down to there and how far through its children it has got. No recursion, so a file
    // cannot run the stack out however deep it nests.
    struct Level<'a> { node: Option<usize>, m: Mat, kids: &'a [Value], next: usize }
    let mut path = vec![Level { node: None, m: IDENTITY, kids: &roots, next: 0 }];
    let mut on_path = vec![false; nodes.len()];
    let mut parts = 0usize;
    let too_long = || format!("the scene has more than {MAX_SCENE_PARTS} nodes and primitives, the most that can be followed");
    let mut bufs = Buffers { doc: &doc, bin, dir, loaded: Vec::new(), owned: 0 };
    let default_material = if let Some(s) = surface.as_deref_mut() {
        appearance::gltf_materials(&mut bufs, s)?
    } else {
        None
    };
    let mut mesh = Mesh::default();
    while let Some(top) = path.last_mut() {
        let (kids, parent) = (top.kids, top.m);
        let Some(kid) = kids.get(top.next) else {
            // all its children done: back up a level, and the node is no longer on the way down
            if let Some(flag) = top.node.and_then(|n| on_path.get_mut(n)) { *flag = false; }
            path.pop();
            continue;
        };
        top.next += 1;
        let n = uint(kid).unwrap_or(usize::MAX);
        let (Some(node), Some(here)) = (nodes.get(n), on_path.get_mut(n)) else {
            return Err(format!("the scene uses node {kid}, but there are {} nodes (they count from 0)", nodes.len()));
        };
        if *here { return Err(format!("node {n} is reached through itself: its children lead back round to it")); }
        *here = true;
        parts += 1;
        let m = mul(&parent, &node_matrix(node).map_err(|e| format!("node {n}: {e}"))?);
        if let Some(mesh_no) = node.get("mesh") {
            let prims = uint(mesh_no).and_then(|i| doc["meshes"][i]["primitives"].as_array())
                .ok_or_else(|| format!("node {n} uses mesh {mesh_no}, which the file does not have"))?;
            // where each POSITION accessor's positions went, so that primitives sharing one share them
            let mut shared = HashMap::new();
            parts = parts.saturating_add(prims.len());
            if parts > MAX_SCENE_PARTS { return Err(too_long()); }
            for (p, prim) in prims.iter().enumerate() {
                gltf_primitive(&mut bufs, prim, &m, &mut shared, &mut mesh, surface.as_deref_mut(), default_material).map_err(|e| format!("mesh {mesh_no}, primitive {p}: {e}"))?;
            }
        }
        if parts > MAX_SCENE_PARTS { return Err(too_long()); }
        let kids = node["children"].as_array().map_or(&[][..], Vec::as_slice);
        path.push(Level { node: Some(n), m, kids, next: 0 });
    }
    Ok(mesh)
}

// A node's own transform: its `matrix`, or its translation, rotation and scale put together.
fn node_matrix(node: &Value) -> Result<Mat, String> {
    fn numbers<const N: usize>(node: &Value, key: &str, or: [f64; N]) -> Result<[f64; N], String> {
        let Some(given) = node.get(key) else { return Ok(or) };
        let wrong = || format!("its {key} is not {N} numbers");
        let mut out = or;
        for (o, v) in out.iter_mut().zip(given.as_array().filter(|a| a.len() == N).ok_or_else(wrong)?) {
            *o = v.as_f64().ok_or_else(wrong)?;
        }
        Ok(out)
    }
    if node.get("matrix").is_some() { return numbers(node, "matrix", IDENTITY); }
    let t = numbers(node, "translation", [0.0; 3])?;
    let q = numbers(node, "rotation", [0.0, 0.0, 0.0, 1.0])?;
    let s = numbers(node, "scale", [1.0; 3])?;
    // the rotation is a quaternion x, y, z, w, meant to be of length one: made so if it is not
    let len = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    let [x, y, z, w] = if len > 0.0 && len.is_finite() { q.map(|c| c / len) } else { q };
    Ok([
        (1.0 - 2.0 * (y * y + z * z)) * s[0], 2.0 * (x * y + w * z) * s[0], 2.0 * (x * z - w * y) * s[0], 0.0,
        2.0 * (x * y - w * z) * s[1], (1.0 - 2.0 * (x * x + z * z)) * s[1], 2.0 * (y * z + w * x) * s[1], 0.0,
        2.0 * (x * z + w * y) * s[2], 2.0 * (y * z - w * x) * s[2], (1.0 - 2.0 * (x * x + y * y)) * s[2], 0.0,
        t[0], t[1], t[2], 1.0,
    ])
}

// One primitive of a mesh, added to `out` with the transform `m` applied to its positions.
fn gltf_primitive(bufs: &mut Buffers, prim: &Value, m: &Mat, shared: &mut HashMap<usize, usize>, out: &mut Mesh, mut surface: Option<&mut Appearance>, default_material: Option<usize>) -> Result<(), String> {
    let mode = match prim.get("mode") { Some(v) => uint(v).ok_or("its mode is not a whole number")?, None => 4 };
    if !(4..=6).contains(&mode) { return Ok(()); } // points and lines have no surface
    let pos_no = uint(&prim["attributes"]["POSITION"]).ok_or("it has no POSITION")?;
    let pos = bufs.view(pos_no, "POSITION", "VEC3", &[5126, 5120, 5121, 5122, 5123])?;
    let idx = match prim.get("indices") {
        Some(v) => Some(bufs.view(uint(v).ok_or("its indices are not the number of an accessor")?, "indices", "SCALAR", &[5121, 5123, 5125])?),
        None => None,
    };
    // An accessor with no bufferView is all zeros: every position at one point, or every triangle one
    // corner three times. Either way there is no surface, and nothing is made of it (which also means
    // that a count with no bytes behind it never has memory set aside for it).
    let Some(pos) = pos else {
        if !prim["extensions"]["KHR_draco_mesh_compression"].is_null() {
            return Err("it is compressed with Draco (KHR_draco_mesh_compression), which is not read".into());
        }
        return Ok(());
    };
    let idx = match idx { Some(None) => return Ok(()), Some(Some(view)) => Some(view), None => None };

    // with no indices the vertices are taken in order
    let corners = idx.as_ref().map_or(pos.count, |view| view.count);
    let triangles = if mode == 4 { corners / 3 } else { corners.saturating_sub(2) };
    if triangles == 0 { return Ok(()); }
    if out.tri.len().saturating_add(triangles) > MAX_TRIANGLES { return Err(too_many("triangles", MAX_TRIANGLES)); }

    let base = match shared.get(&pos_no) {
        Some(&base) => base,
        None => {
            let base = out.pos.len();
            if base.saturating_add(pos.count) > MAX_VERTICES { return Err(too_many("vertices", MAX_VERTICES)); }
            let data = bufs.data(&pos);
            out.pos.reserve(pos.count);
            for k in 0..pos.count {
                let mut p = [0f64; 3];
                for (c, v) in p.iter_mut().enumerate() {
                    let raw = pos.ty.read(data, k * pos.stride + c * pos.ty.size(), false).ok_or("POSITION runs past the end of its data")?;
                    // a normalized whole number stands for a fraction: of 1 unsigned, of -1 to 1 signed
                    *v = pos.unit.map_or(raw, |unit| (raw / unit).max(-1.0));
                }
                let [x, y, z] = p;
                out.pos.push([
                    (m[0] * x + m[4] * y + m[8] * z + m[12]) as f32,
                    (m[1] * x + m[5] * y + m[9] * z + m[13]) as f32,
                    (m[2] * x + m[6] * y + m[10] * z + m[14]) as f32,
                ]);
            }
            shared.insert(pos_no, base);
            base
        }
    };

    let slot = match prim.get("material") {
        Some(value) => Some(uint(value).ok_or("its material is not a whole number")?),
        None => default_material,
    };
    if let (Some(s), Some(i)) = (surface.as_ref(), slot) {
        if i >= s.materials.len() { return Err(format!("material slot {i} does not exist")); }
    }
    let uvs = if let Some(mat) = surface.as_ref().and_then(|s| slot.and_then(|i| s.materials.get(i))).filter(|m| m.colour.is_some() || m.normal.is_some() || m.metallic_roughness.is_some()) {
        Some(appearance::read_uvs(bufs, prim, mat, pos.count)?)
    } else { None };
    let tangent = if uvs.is_some() { appearance::tangents(bufs, prim, m, pos.count)? } else { None };
    let indices = idx.as_ref().map(|view| (view, bufs.data(view)));
    let corner = |k: usize| -> Result<u32, String> {
        let i = match indices {
            Some((view, data)) => view.ty.read(data, k * view.stride, false).ok_or("the indices run past the end of their data")?,
            None => k as f64,
        };
        let i = index(i, pos.count).ok_or_else(|| format!("an index of {i}, but POSITION has {} (they count from 0)", pos.count))?;
        Ok(base as u32 + i)
    };
    out.tri.reserve(triangles);
    for t in 0..triangles {
        let (a, b, c) = match mode {
            4 => (3 * t, 3 * t + 1, 3 * t + 2),
            5 if t % 2 == 1 => (t + 1, t, t + 2), // every other triangle of a strip is turned round, so that all face one way
            5 => (t, t + 1, t + 2),
            _ => (0, t + 1, t + 2),
        };
        let indices = [corner(a)?, corner(b)?, corner(c)?];
        out.tri.push(indices);
        if let Some(s) = surface.as_deref_mut() {
            let face = if let Some(uvs) = &uvs {
                let local = indices.map(|i| i as usize - base);
                FaceAppearance { material: slot, tangent: tangent.as_ref().map_or([[0.;4];3],|t|local.map(|i|t[i])), ..uvs.face(local) }
            } else { FaceAppearance {material:slot, ..Default::default()} };
            s.faces.push(face);
        }
    }
    Ok(())
}

// Where an accessor's numbers are: which buffer, the byte the first is at, how far apart they are.
struct View { buffer: usize, start: usize, stride: usize, count: usize, ty: Num, unit: Option<f64> }

// The file's buffers, each fetched the first time something wants it.
struct Buffers<'a> {
    doc: &'a Value,
    bin: Option<&'a [u8]>,
    dir: Option<&'a Path>,
    loaded: Vec<Option<Cow<'a, [u8]>>>,
    owned: u64, // bytes decoded or read from other files so far, held to the same limit as a file
}

impl<'a> Buffers<'a> {
    // Accessor `no` as a view of its buffer, every range checked against the bytes that are there.
    // Ok(None): it has no bufferView, which means it is all zeros. `what` names it in errors; `shape`
    // and `types` are the type and the component types it is allowed to be.
    fn view(&mut self, no: usize, what: &str, shape: &str, types: &[u64]) -> Result<Option<View>, String> {
        let doc = self.doc;
        let acc = &doc["accessors"][no];
        let what = format!("{what}: accessor {no}");
        if !acc.is_object() { return Err(format!("{what} is not in the file")); }
        if acc.get("sparse").is_some() { return Err(format!("{what} is sparse, which is not read")); }
        let (kind, code) = (acc["type"].as_str().unwrap_or("of no type"), acc["componentType"].as_u64().unwrap_or(0));
        if kind != shape || !types.contains(&code) {
            return Err(format!("{what} is {} with component type {code}, which is not read (it has to be {shape} with one of {types:?})", show(kind.as_bytes())));
        }
        let ty = match code { 5120 => Num::I8, 5121 => Num::U8, 5122 => Num::I16, 5123 => Num::U16, 5125 => Num::U32, _ => Num::F32 };
        let count = uint(&acc["count"]).ok_or_else(|| format!("{what} does not say how many items it has"))?;
        let Some(bv_no) = acc.get("bufferView") else { return Ok(None) };
        let bv = uint(bv_no).map(|i| &doc["bufferViews"][i]).filter(|bv| bv.is_object())
            .ok_or_else(|| format!("{what} uses bufferView {bv_no}, which is not in the file"))?;
        let field = |of: &Value, name: &str| match of.get(name) {
            Some(v) => uint(v).ok_or_else(|| format!("{what}: a {name} that is not a whole number")),
            None => Ok(0),
        };
        let (buffer, bv_at, bv_len) = (field(bv, "buffer")?, field(bv, "byteOffset")?, field(bv, "byteLength")?);
        let width = ty.size() * match shape { "VEC2" => 2, "VEC3" => 3, "VEC4" => 4, _ => 1 };
        let at = field(acc, "byteOffset")?;
        let stride = match field(bv, "byteStride")? { 0 => width, stride => stride }; // not given: one straight after another
        let have = self.load(buffer)?.len();
        // The bufferView has to lie inside its buffer, and the accessor's last item inside the bufferView.
        let need = match count.checked_sub(1) {
            Some(last) => last.checked_mul(stride).and_then(|n| n.checked_add(width)).and_then(|n| n.checked_add(at)),
            None => Some(0),
        };
        let fits = stride >= width && bv_at.checked_add(bv_len).is_some_and(|end| end <= have) && need.is_some_and(|n| n <= bv_len);
        if !fits {
            return Err(format!("{what} ({count} items of {width} bytes, {stride} apart, from byte {at}) does not fit in bufferView {bv_no} ({bv_len} bytes from byte {bv_at} of a buffer of {have})"));
        }
        // what a normalized whole number is a fraction of (for the other kinds the word means nothing)
        let unit = match ty { Num::I8 => Some(127.0), Num::U8 => Some(255.0), Num::I16 => Some(32767.0), Num::U16 => Some(65535.0), _ => None };
        let unit = unit.filter(|_| acc["normalized"].as_bool() == Some(true));
        Ok(Some(View { buffer, start: bv_at.saturating_add(at), stride, count, ty, unit }))
    }

    // The bytes a view starts at, to the end of its buffer.
    fn data(&self, view: &View) -> &[u8] {
        self.loaded.get(view.buffer).and_then(|b| b.as_deref()).and_then(|b| b.get(view.start..)).unwrap_or(&[])
    }

    // Buffer `no`'s bytes: the GLB's binary chunk for one with no uri, the decoded data of a base64
    // `data:` URI, or a file beside the model.
    fn load(&mut self, no: usize) -> Result<&[u8], String> {
        let buffer = &self.doc["buffers"][no];
        if !buffer.is_object() { return Err(format!("buffer {no} is used, but the file does not have it")); }
        if self.loaded.len() <= no { self.loaded.resize(no + 1, None); }
        if !matches!(self.loaded.get(no), Some(Some(_))) {
            let bytes = match buffer.get("uri").map(Value::as_str) {
                None => Cow::Borrowed(self.bin.ok_or_else(|| format!("buffer {no} has no uri, and the file has no binary part for it to mean"))?),
                Some(None) => return Err(format!("buffer {no}'s uri is not text")),
                Some(Some(uri)) => Cow::Owned(match uri.strip_prefix("data:") {
                    Some(data) => {
                        let (kind, text) = data.split_once(',').ok_or_else(|| format!("buffer {no}'s data: URI has no comma before its data"))?;
                        if !kind.ends_with(";base64") { return Err(format!("buffer {no}'s data: URI is not base64, the only kind that is read")); }
                        base64(text.as_bytes()).map_err(|e| format!("buffer {no}: {e}"))?
                    }
                    None => self.file(no, uri)?,
                }),
            };
            if let Cow::Owned(b) = &bytes { self.owned = self.owned.saturating_add(b.len() as u64); }
            if self.owned > MAX_FILE_BYTES { return Err(format!("the file's buffers come to more than {MAX_FILE_BYTES} bytes (1 GB), the most that can be read")); }
            if let Some(slot) = self.loaded.get_mut(no) { *slot = Some(bytes); }
        }
        Ok(self.loaded.get(no).and_then(|b| b.as_deref()).unwrap_or(&[]))
    }

    // A buffer kept in a file of its own. Only a file in the model's folder or below it is read: the
    // name comes out of the model, and must not be able to point anywhere else on the machine.
    fn file(&self, no: usize, uri: &str) -> Result<Vec<u8>, String> {
        relative_bytes(self.dir, uri, &format!("buffer {no}"))
    }
}

fn relative_bytes(dir: Option<&Path>, uri: &str, label: &str) -> Result<Vec<u8>, String> {
    let refuse = |why: &str| format!("{label} is at \"{}\", which is not read: {why}", show(uri.as_bytes()));
    let name = percent_decoded(uri).ok_or_else(|| refuse("the name is not readable text"))?;
    let parts = || name.split(['/', '\\']);
    if name.contains(':') { return Err(refuse("only files beside the model are read, not drive letters or web addresses")); }
    if name.starts_with(['/', '\\']) { return Err(refuse("it is an absolute path, and only files beside the model are read")); }
    if parts().any(|part| part.len() > 1 && part.trim_matches(['.', ' ']).is_empty()) {
        return Err(refuse("it climbs out of the model's folder with \"..\""));
    }
    // names that Windows takes for a device (the console, a serial port) in whatever folder they are
    let device = |part: &str| {
        let stem = part.split('.').next().unwrap_or("").trim_end().to_ascii_uppercase();
        let port = stem.len() == 4 && (stem.starts_with("COM") || stem.starts_with("LPT")) && stem.ends_with(|c: char| c.is_ascii_digit());
        port || matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$")
    };
    if parts().any(device) { return Err(refuse("it names a device, not a file")); }
    let dir = dir.ok_or_else(|| format!("{label} is in a file of its own (\"{}\"), and there is no folder to look for it in", show(name.as_bytes())))?;
    let path = dir.join(&name);
    let cannot = |e: std::io::Error| format!("{label}'s file \"{}\" cannot be read ({e})", show(name.as_bytes()));
    let size = std::fs::metadata(&path).map_err(cannot)?.len();
    if size > MAX_FILE_BYTES { return Err(format!("{label}'s file \"{}\": {}", show(name.as_bytes()), too_big(size))); }
    std::fs::read(&path).map_err(cannot)
}

// A URI with its %XX escapes turned back into the characters they stand for (%20 a space, and so on).
fn percent_decoded(uri: &str) -> Option<String> {
    let b = uri.as_bytes();
    let hex = |at: usize| b.get(at).and_then(|&c| (c as char).to_digit(16));
    let (mut out, mut at) = (Vec::with_capacity(b.len()), 0usize);
    while let Some(&c) = b.get(at) {
        match (c, hex(at + 1), hex(at + 2)) {
            (b'%', Some(high), Some(low)) => { out.push((high * 16 + low) as u8); at += 3; }
            _ => { out.push(c); at += 1; }
        }
    }
    String::from_utf8(out).ok()
}

// Base64 as a `data:` URI has it: A-Z a-z 0-9 + /, six bits a character, `=` filling out the end.
fn base64(text: &[u8]) -> Result<Vec<u8>, String> {
    let text = text.strip_suffix(b"=").unwrap_or(text);
    let text = text.strip_suffix(b"=").unwrap_or(text);
    if text.len() % 4 == 1 { return Err("its base64 data is cut short".into()); }
    let mut out = Vec::with_capacity(text.len() / 4 * 3 + 3);
    let (mut held, mut bits) = (0u32, 0u32);
    for (at, &c) in text.iter().enumerate() {
        let six = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("its base64 data has a character that does not belong in it, at {at}")),
        };
        held = (held << 6 | six as u32) & 0xFFF; // never more than six bits waiting and six coming in
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((held >> bits) as u8);
        }
    }
    Ok(out)
}
