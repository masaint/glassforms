//! Measuring only (`cargo test --release -p gf-core --test tube_cells -- --ignored --nocapture`): how many
//! pieces a cell of Thomas's exact tube table lists, which is what each exact reading of it loops over.

use gf_core::state::{Detail, ShapeJob, State};

#[test]
#[ignore]
fn thomas_cell_lists() {
    let mut s = State::new();
    let mut job = s.shape_job(Detail::Full, None).unwrap();
    job.prepare_formula(&s.families).unwrap();
    let ShapeJob::Field { table, .. } = &job.shape else { panic!("thomas is a formula") };
    // (GF_TUBE_TABLE=<file>: the table written out as little-endian f32s, for reading elsewhere)
    if let Some(path) = std::env::var_os("GF_TUBE_TABLE") {
        std::fs::write(path, table.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    let t: Vec<[f32; 4]> = table.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let (header, grid, sizes) = (t[0], t[1], t[2]);
    let n = sizes[0] as usize;
    let cells = &t[3..3 + n * n * n];
    let mut counts: Vec<u32> = cells.iter().map(|c| c[1] as u32).filter(|&c| c > 0).collect();
    counts.sort();
    let melted = cells.iter().filter(|c| c[1] > 0.0 && c[3] >= 0.5).count();
    let q = |f: f64| counts[((counts.len() - 1) as f64 * f) as usize];
    let mean = counts.iter().map(|&c| c as f64).sum::<f64>() / counts.len() as f64;
    println!("header {header:?} grid {grid:?} sizes {sizes:?}");
    println!("n {n} cell {:.4} entries {} | non-empty cells {} of {} (melt branch {melted})", grid[3], t.len(), counts.len(), n * n * n);
    println!("pieces per non-empty cell: mean {mean:.1} p50 {} p90 {} p99 {} max {}", q(0.5), q(0.9), q(0.99), counts[counts.len() - 1]);
}
