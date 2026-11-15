//! Prints the beams a rig of lights comes to (gf_core::scatter::beams), for reading.
//! usage: scatter_beams [az,el,w,h,r,g,b[,soft] ...] [ambient=r,g,b]
use gf_core::env;
use gf_core::presets::Light;
use gf_core::scatter;

fn main() {
    let mut lights = Vec::new();
    let mut ambient = [0.0; 3];
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("ambient=") {
            let v: Vec<f64> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            ambient = [v[0], v[1], v[2]];
        } else {
            let v: Vec<f64> = a.split(',').filter_map(|x| x.parse().ok()).collect();
            lights.push(Light { az: v[0], el: v[1], w: v[2], h: v[3], rgb: [v[4], v[5], v[6]], soft: v.get(7).copied().unwrap_or(0.6) });
        }
    }
    let px = env::paint(&lights, ambient, env::ENV_W, env::ENV_H);
    let light = scatter::beams(&px, env::ENV_W, env::ENV_H, None);
    let total = scatter::beams_total(&light);
    println!("{} beams, light in all {:.3} {:.3} {:.3}", light.beams.len(), total[0], total[1], total[2]);
    println!("the rest, from all round: {:.4} {:.4} {:.4}, leaning (green) {:.4} {:.4} {:.4}", light.rest.even[0], light.rest.even[1], light.rest.even[2], light.rest.lean[1][0], light.rest.lean[1][1], light.rest.lean[1][2]);
    for b in &light.beams {
        println!("from {:6.3} {:6.3} {:6.3}  half-width {:5.1} deg  light {:.4} {:.4} {:.4}", b.dir[0], b.dir[1], b.dir[2], b.tan.atan().to_degrees(), b.rgb[0], b.rgb[1], b.rgb[2]);
    }
}
