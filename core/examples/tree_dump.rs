//! A tree's strands as text, and how long growing it takes: for looking at the growth without the GPU.
//!
//!   cargo run --release -p gf-core --example tree_dump -- out.json [p.<slider>=value ...]
use gf_core::presets::{families, CurveOpts, Shape};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().cloned().unwrap_or_else(|| "tree.json".into());
    let fams = families();
    let f = fams.iter().find(|f| f.id == "trees").unwrap();
    let p = &f.presets[0];
    let mut v = p.defaults();
    for a in &args[1..] {
        if let Some((k, x)) = a.strip_prefix("p.").and_then(|a| a.split_once('=')) {
            if let Some(i) = p.param_index(k) { v[i] = x.parse().unwrap(); }
        }
    }
    let Shape::Curve(c) = &p.shape else { panic!() };
    let t = Instant::now();
    let raw = c(&v, &CurveOpts { len: 1.0, quick: false });
    let grow_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let _again = c(&v, &CurveOpts { len: 1.0, quick: false });
    let cut_ms = t.elapsed().as_secs_f64() * 1e3;
    for res in [256.0, 768.0] {
        let t = Instant::now();
        let lines: Vec<&[f64]> = raw.lines.iter().map(|l| &l[..]).collect();
        let radii: Vec<&[f64]> = raw.radii.iter().map(|l| &l[..]).collect();
        let pose = gf_core::curve::placement(&[&raw.frame[..]], &gf_core::curve::PlaceOpts { centre: true, ..Default::default() });
        let c = gf_core::curve::prep_strands(&lines, &radii, &gf_core::curve::StrandOpts { thick: 1.0, ds: 0.75 * 2.0 * gf_core::HB / res, place: Default::default(), pose: Some(pose), ground: raw.ground });
        eprintln!("prep at {res}: {} points in {:.1} ms", c.n, t.elapsed().as_secs_f64() * 1e3);
    }
    let pts: usize = raw.lines.iter().map(|l| l.len() / 3).sum();
    eprintln!("{} strands, {} points, grown in {grow_ms:.1} ms, cut again in {cut_ms:.2} ms", raw.lines.len(), pts);
    let mut s = String::from("{\"lines\":[");
    for (i, (l, r)) in raw.lines.iter().zip(&raw.radii).enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[{:?},{:?}]", l, r));
    }
    s.push_str("]}");
    std::fs::write(out, s).unwrap();
}
