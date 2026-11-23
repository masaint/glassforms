use glassforms::{bake::Volume, gpu::Gpu, stage::Lowest};
use half::f16;

#[test]
fn floor_support_matches_every_grid_sample_for_each_dominant_axis() {
    let gpu = Gpu::headless("vulkan").unwrap();
    let solver = Lowest::new(&gpu);
    let n = 33usize;
    let points = (0..n.pow(3)).map(|i| {
        [i % n, i / n % n, i / (n*n)].map(|v| (v as f32 + 0.5) * (2.0 / n as f32) - 1.0)
    }).collect::<Vec<_>>();
    let field = points.iter().map(|&[x,y,z]| {
        f16::from_f32(((x - 0.23).powi(2) + (y + 0.31).powi(2) + (z - 0.08).powi(2)).sqrt() - 0.43)
    }).collect::<Vec<_>>();
    let vol = Volume::from_data(&gpu, n as u32, &field);
    let step = 2.0 / n as f32;
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut d = [0.2_f32, -0.3, 0.15];
            d[axis] = sign;
            let length = d.iter().map(|v| v*v).sum::<f32>().sqrt();
            d = d.map(|c| c / length);
            let expected = points.iter().zip(&field).filter_map(|(p,s)| {
                let s = s.to_f32();
                (s < 1.5 * step && s > -2.0 * step).then(|| p.iter().zip(d).map(|(a,b)| a*b).sum::<f32>() - s)
            }).reduce(f32::max).unwrap();
            let actual = solver.reach(&vol, d.map(f64::from), 1.0).unwrap() as f32;
            assert!((actual - expected).abs() < 2e-5, "{d:?} {actual} vs {expected}");
        }
    }
    let empty = Volume::from_data(&gpu, n as u32, &vec![f16::ONE; n.pow(3)]);
    assert!(solver.reach(&empty, [0.,-1.,0.], 1.0).is_none());
}
