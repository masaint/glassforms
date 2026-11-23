//! A click picks by the pixel under it: the shape where the picture shows it, nothing a pixel or two
//! beside its silhouette, nothing on the background.
use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu, ui::handles::pick_occurrence};

#[test]
fn picking_follows_the_drawn_silhouette_to_the_pixel() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let mut e = Engine::new(gpu, State::new(), true);
    e.bake(Detail::Full).unwrap();
    let (w, h) = (480u32, 270u32);
    e.resize(w, h);
    e.render(2);
    let frac = |x: u32, y: u32| ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
    let at = |x: u32, y: u32| { let (fx, fy) = frac(x, y); pick_occurrence(&e, fx, fy) };
    let (cx, cy) = (w / 2, h / 2);
    assert!(at(cx, cy).is_some(), "the middle of the shape");
    assert!(at(2, 2).is_none(), "the background");
    // walk out from the middle along the row to the silhouette, as the picture's distances show it
    let row = e.renderer.read_hdr(0, cy, w, 1).unwrap();
    let edge = (cx..w).find(|&x| row[x as usize * 4 + 3] <= 0.0).expect("the shape ends inside the picture");
    assert!(at(edge - 1, cy).is_some(), "the last pixel the shape covers");
    assert!(at(edge, cy).is_none(), "the first pixel beside it");
    assert!(at(edge + 2, cy).is_none(), "two pixels beside it");
}
