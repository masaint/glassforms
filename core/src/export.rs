//! A shape as a mesh, for other programs: its surface found in its distance volume, written as OBJ or
//! binary STL.
//!
//! The mesh is a consequence of the shape, never a source of it. It is made from the volume the tracer
//! draws (every layer, the surface studio and all), written to a file and forgotten; nothing in the
//! program reads it back as the model.
//!
//! The surface is found by surface nets, a dual way of contouring: each cell of the grid whose corners
//! are not all on one side of the surface gets one point, the mean of the places where the surface
//! crosses the cell's edges, and every edge of the grid that the surface crosses gets a quad joining the
//! points of the four cells round it, split into two triangles along its shorter diagonal. Every quad's
//! edge is an edge of the next, so the mesh has no open edges: it is watertight wherever the shape is
//! closed. Against marching cubes it gives about half the triangles, no slivers, and no table of cases.
//! Where four cells meet across a face that the surface crosses twice (a saddle finer than the grid), one
//! mesh edge is shared by four triangles; those are counted (`Check::shared`).
//!
//! The volume is read a slice at a time, so a large one never has to be held whole: only two slices and
//! the points of two layers of cells are kept.

use std::collections::HashMap;
use std::io::Write;

/// A mesh: points, and triangles as three indices into them, each turning anticlockwise seen from
/// outside the shape.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub points: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
}

const NONE: u32 = u32::MAX;

/// The mesh of a volume of `res` voxels a side spanning [-hb, hb], its distances read a slice at a time
/// by `slice(z)` (res x res values, row by row, x fastest; negative inside). Everything outside the
/// volume counts as outside the shape, so the mesh is closed even where the shape touches the volume's
/// edge.
pub fn surface_net(res: usize, hb: f64, mut slice: impl FnMut(usize) -> Vec<f32>) -> Mesh {
    // the grid padded by one voxel of outside all round: n voxels, n - 1 cells a side
    let n = res + 2;
    let outside = 1.0f32;
    let padded = |z: usize, slice: &mut dyn FnMut(usize) -> Vec<f32>| -> Vec<f32> {
        let mut out = vec![outside; n * n];
        if z == 0 || z == n - 1 { return out; }
        let s = slice(z - 1);
        for y in 0..res {
            for x in 0..res {
                let v = s.get(y * res + x).copied().unwrap_or(outside);
                out[(y + 1) * n + (x + 1)] = if v.is_finite() { v } else { outside };
            }
        }
        out
    };
    let place = |i: f64| ((i - 1.0 + 0.5) / res as f64 * 2.0 - 1.0) * hb;
    let cells = n - 1;
    let mut mesh = Mesh::default();
    // the points of the cells of the layer below and of this one
    let mut below = vec![NONE; cells * cells];
    let mut here = vec![NONE; cells * cells];
    let mut s0 = padded(0, &mut slice);
    for z in 0..cells {
        let s1 = padded(z + 1, &mut slice);
        // the points of this layer of cells (between slices z and z + 1)
        here.fill(NONE);
        for y in 0..cells {
            for x in 0..cells {
                let v = |dx: usize, dy: usize, dz: usize| if dz == 0 { s0[(y + dy) * n + x + dx] } else { s1[(y + dy) * n + x + dx] };
                let corners = [v(0, 0, 0), v(1, 0, 0), v(0, 1, 0), v(1, 1, 0), v(0, 0, 1), v(1, 0, 1), v(0, 1, 1), v(1, 1, 1)];
                let inside = corners.iter().filter(|c| **c < 0.0).count();
                if inside == 0 || inside == 8 { continue; }
                // the mean of the crossings on the cell's twelve edges
                let mut sum = [0.0f64; 3];
                let mut k = 0.0;
                for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)] {
                    let (va, vb) = (corners[a], corners[b]);
                    if (va < 0.0) == (vb < 0.0) { continue; }
                    let t = (va / (va - vb)) as f64;
                    let pa = [(a & 1) as f64, ((a >> 1) & 1) as f64, ((a >> 2) & 1) as f64];
                    let pb = [(b & 1) as f64, ((b >> 1) & 1) as f64, ((b >> 2) & 1) as f64];
                    for c in 0..3 { sum[c] += pa[c] + (pb[c] - pa[c]) * t; }
                    k += 1.0;
                }
                let p = [place(x as f64 + sum[0] / k), place(y as f64 + sum[1] / k), place(z as f64 + sum[2] / k)];
                here[y * cells + x] = mesh.points.len() as u32;
                mesh.points.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            }
        }
        // Quads. For an edge of the grid that the surface crosses, the four cells round it; the quad turns
        // so that seen from the outside end of the edge it goes anticlockwise.
        let quad = |a: u32, b: u32, c: u32, d: u32, out_first: bool, points: &Vec<[f32; 3]>, tris: &mut Vec<[u32; 3]>| {
            if a == NONE || b == NONE || c == NONE || d == NONE { return; }
            let (a, b, c, d) = if out_first { (a, d, c, b) } else { (a, b, c, d) };
            let dist = |i: u32, j: u32| {
                let (p, q) = (points[i as usize], points[j as usize]);
                (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)
            };
            if dist(a, c) <= dist(b, d) {
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            } else {
                tris.push([a, b, d]);
                tris.push([b, c, d]);
            }
        };
        let at = |layer: &Vec<u32>, x: usize, y: usize| layer[y * cells + x];
        // edges along z, from slice z to slice z + 1: the cells round them are all in this layer
        for y in 1..cells {
            for x in 1..cells {
                let (va, vb) = (s0[y * n + x], s1[y * n + x]);
                if (va < 0.0) == (vb < 0.0) { continue; }
                quad(at(&here, x - 1, y - 1), at(&here, x, y - 1), at(&here, x, y), at(&here, x - 1, y), va >= 0.0, &mesh.points, &mut mesh.tris);
            }
        }
        // edges along x and y in slice z: the cells round them are in the layer below and this one
        if z > 0 {
            for y in 1..cells {
                for x in 0..cells {
                    let (va, vb) = (s0[y * n + x], s0[y * n + x + 1]);
                    if (va < 0.0) == (vb < 0.0) { continue; }
                    quad(at(&below, x, y - 1), at(&below, x, y), at(&here, x, y), at(&here, x, y - 1), va >= 0.0, &mesh.points, &mut mesh.tris);
                }
            }
            for y in 0..cells {
                for x in 1..cells {
                    let (va, vb) = (s0[y * n + x], s0[(y + 1) * n + x]);
                    if (va < 0.0) == (vb < 0.0) { continue; }
                    quad(at(&below, x - 1, y), at(&here, x - 1, y), at(&here, x, y), at(&below, x, y), va >= 0.0, &mesh.points, &mut mesh.tris);
                }
            }
        }
        std::mem::swap(&mut below, &mut here);
        s0 = s1;
    }
    mesh
}

/// What a mesh is, as a check that it is whole: edges used by one triangle only (open: a hole), by two
/// (as they should be), by more (`shared`: where the surface touches itself between grid points); and
/// the volume it encloses (positive when its triangles face outwards).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Check {
    pub points: usize,
    pub tris: usize,
    pub open: usize,
    pub shared: usize,
    pub volume: f64,
}

pub fn check(m: &Mesh) -> Check {
    let mut edges: HashMap<(u32, u32), u32> = HashMap::with_capacity(m.tris.len() * 3 / 2 + 16);
    let mut volume = 0.0f64;
    for t in &m.tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
        let p = |i: u32| m.points[i as usize].map(|v| v as f64);
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
    }
    Check {
        points: m.points.len(),
        tris: m.tris.len(),
        open: edges.values().filter(|&&k| k == 1).count(),
        shared: edges.values().filter(|&&k| k > 2).count(),
        volume,
    }
}

/// Write a mesh as OBJ (text: points and triangles) or binary STL, by the file's ending (.stl, else OBJ).
pub fn write(m: &Mesh, path: &str) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut w = std::io::BufWriter::new(file);
    let err = |e: std::io::Error| format!("{path}: {e}");
    if path.to_ascii_lowercase().ends_with(".stl") {
        let mut head = [0u8; 80];
        let says = b"Glass Forms: a shape's surface, from its distance volume";
        head[..says.len()].copy_from_slice(says);
        w.write_all(&head).map_err(err)?;
        w.write_all(&(m.tris.len() as u32).to_le_bytes()).map_err(err)?;
        for t in &m.tris {
            let (a, b, c) = (m.points[t[0] as usize], m.points[t[1] as usize], m.points[t[2] as usize]);
            let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            let mut nrm = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let l = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            if l > 0.0 { nrm = nrm.map(|x| x / l); }
            for x in nrm.iter().chain(&a).chain(&b).chain(&c) { w.write_all(&x.to_le_bytes()).map_err(err)?; }
            w.write_all(&[0, 0]).map_err(err)?;
        }
    } else {
        writeln!(w, "# Glass Forms: a shape's surface, from its distance volume ({} points, {} triangles)", m.points.len(), m.tris.len()).map_err(err)?;
        for p in &m.points { writeln!(w, "v {} {} {}", p[0], p[1], p[2]).map_err(err)?; }
        for t in &m.tris { writeln!(w, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).map_err(err)?; }
    }
    w.flush().map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ball's distances, sliced.
    fn ball(res: usize, hb: f64, r: f64, at: [f64; 3]) -> impl FnMut(usize) -> Vec<f32> {
        move |z| {
            let c = |i: usize| ((i as f64 + 0.5) / res as f64 * 2.0 - 1.0) * hb;
            let mut s = Vec::with_capacity(res * res);
            for y in 0..res {
                for x in 0..res {
                    let d = ((c(x) - at[0]).powi(2) + (c(y) - at[1]).powi(2) + (c(z) - at[2]).powi(2)).sqrt() - r;
                    s.push(d as f32);
                }
            }
            s
        }
    }

    #[test]
    fn a_ball_is_closed_faces_out_and_holds_its_volume() {
        let m = surface_net(48, 1.5, ball(48, 1.5, 0.9, [0.0; 3]));
        let c = check(&m);
        assert_eq!((c.open, c.shared), (0, 0), "{c:?}");
        // a sphere: points - edges + faces = 2, with every edge in two triangles
        assert_eq!(c.points as i64 - (c.tris as i64 * 3 / 2) + c.tris as i64, 2);
        let v = 4.0 / 3.0 * std::f64::consts::PI * 0.9f64.powi(3);
        assert!((c.volume / v - 1.0).abs() < 0.03, "{} against {v}", c.volume);
    }

    #[test]
    fn a_shape_cut_by_the_volumes_edge_is_still_closed() {
        let m = surface_net(32, 1.0, ball(32, 1.0, 0.8, [0.5, 0.0, 0.0]));
        let c = check(&m);
        assert_eq!(c.open, 0);
        assert!(c.volume > 0.0);
    }

    #[test]
    fn it_writes_both_kinds_of_file() {
        let m = surface_net(16, 1.0, ball(16, 1.0, 0.6, [0.0; 3]));
        let dir = std::env::temp_dir();
        let (obj, stl) = (dir.join("gf_export_test.obj"), dir.join("gf_export_test.stl"));
        write(&m, obj.to_str().unwrap()).unwrap();
        write(&m, stl.to_str().unwrap()).unwrap();
        let text = std::fs::read_to_string(&obj).unwrap();
        assert_eq!(text.lines().filter(|l| l.starts_with("f ")).count(), m.tris.len());
        assert_eq!(std::fs::metadata(&stl).unwrap().len(), 84 + 50 * m.tris.len() as u64);
    }
}
