//! A still's background, apart from its lighting: as seen, one flat colour, or transparent (a cut-out).
//!
//! Where the shape is comes from the picture's own depth (the fourth channel the tracer keeps: 0 where a
//! ray from the camera met nothing of the shape). That is one sample a pixel, so its edge is softened over
//! the pixels round it (3 x 3) to give an alpha without stair steps. Everything the shape does not cover
//! goes: the sky, the studio, the floor and what is cast on it. Through glass the world stays as it was
//! seen, as glass shows it. The picture on screen does not change; only the file written.

/// What a still shows where the shape is not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Background {
    Seen,
    Plate([u8; 3]),
    Clear,
}

impl Background {
    /// From the settings' words ("seen", "plate", "clear") and the plate's colour.
    pub fn of(name: &str, plate: [u8; 3]) -> Background {
        match name {
            "plate" => Background::Plate(plate),
            "clear" => Background::Clear,
            _ => Background::Seen,
        }
    }
}

/// How much of each pixel the shape covers (0 to 1), from the picture's depths (RGBA floats, depth in
/// the fourth): the share of the 3 x 3 pixels round it where a ray met the shape.
pub fn coverage(rgba: &[f32], w: usize, h: usize) -> Vec<f32> {
    let hit = |x: usize, y: usize| (rgba[(y * w + x) * 4 + 3] > 0.0) as u8 as f32;
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut n) = (0.0, 0.0);
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    sum += hit(xx, yy);
                    n += 1.0;
                }
            }
            // (a pixel the shape itself covers keeps all of it: the softening is only outward)
            out[y * w + x] = if hit(x, y) > 0.0 { 1.0 } else { sum / n };
        }
    }
    out
}

/// The coverage at a pixel of a picture `w` x `h`, from coverage worked out at `cw` x `ch` (the two
/// differ when the picture is traced smaller or larger than the screen).
fn at(cov: &[f32], cw: usize, ch: usize, x: usize, y: usize, w: usize, h: usize) -> f32 {
    let cx = ((x as f64 + 0.5) * cw as f64 / w as f64) as usize;
    let cy = ((y as f64 + 0.5) * ch as f64 / h as f64) as usize;
    cov[cy.min(ch - 1) * cw + cx.min(cw - 1)]
}

/// Put the background into an 8-bit RGBA picture in place.
pub fn apply8(rgba: &mut [u8], w: usize, h: usize, cov: &[f32], cw: usize, ch: usize, bg: Background) {
    if bg == Background::Seen { return; }
    for y in 0..h {
        for x in 0..w {
            let a = at(cov, cw, ch, x, y, w, h);
            let p = &mut rgba[(y * w + x) * 4..(y * w + x) * 4 + 4];
            match bg {
                Background::Plate(c) => for k in 0..3 { p[k] = (c[k] as f32 + (p[k] as f32 - c[k] as f32) * a).round() as u8 },
                Background::Clear => p[3] = (a * 255.0).round() as u8,
                Background::Seen => {}
            }
        }
    }
}

/// The same for a 16-bit RGB picture; a clear background gives back RGBA (four channels), the rest RGB.
pub fn apply16(rgb: &[u16], w: usize, h: usize, cov: &[f32], cw: usize, ch: usize, bg: Background) -> (Vec<u16>, bool) {
    match bg {
        Background::Seen => (rgb.to_vec(), false),
        Background::Plate(c) => {
            let mut out = rgb.to_vec();
            for y in 0..h {
                for x in 0..w {
                    let a = at(cov, cw, ch, x, y, w, h);
                    for k in 0..3 {
                        let v = &mut out[(y * w + x) * 3 + k];
                        let plate = c[k] as f32 * 257.0;
                        *v = (plate + (*v as f32 - plate) * a).round() as u16;
                    }
                }
            }
            (out, false)
        }
        Background::Clear => {
            let mut out = Vec::with_capacity(w * h * 4);
            for y in 0..h {
                for x in 0..w {
                    out.extend_from_slice(&rgb[(y * w + x) * 3..(y * w + x) * 3 + 3]);
                    out.push((at(cov, cw, ch, x, y, w, h) * 65535.0).round() as u16);
                }
            }
            (out, true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4 x 3 picture with the shape in its middle two pixels of the middle row.
    fn depths() -> Vec<f32> {
        let mut d = vec![0.0f32; 4 * 3 * 4];
        for x in [1, 2] { d[(4 + x) * 4 + 3] = 2.5; }
        d
    }

    #[test]
    fn the_shape_keeps_its_pixels_and_its_edge_is_soft() {
        let c = coverage(&depths(), 4, 3);
        assert_eq!((c[5], c[6]), (1.0, 1.0));
        // a corner pixel sees one hit among its four
        assert_eq!(c[0], 0.25);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn a_plate_replaces_the_background_and_clear_makes_it_transparent() {
        let c = coverage(&depths(), 4, 3);
        let mut px = vec![100u8; 4 * 3 * 4];
        apply8(&mut px, 4, 3, &c, 4, 3, Background::Plate([200, 0, 50]));
        assert_eq!(&px[5 * 4..5 * 4 + 3], &[100, 100, 100], "the shape is untouched");
        assert_eq!(&px[0..3], &[175, 25, 63], "a quarter covered: three quarters plate");
        let mut px = vec![100u8; 4 * 3 * 4];
        apply8(&mut px, 4, 3, &c, 4, 3, Background::Clear);
        assert_eq!((px[5 * 4 + 3], px[3]), (255, 64));
        let mut px = vec![100u8; 4 * 3 * 4];
        apply8(&mut px, 4, 3, &c, 4, 3, Background::Seen);
        assert!(px.iter().all(|v| *v == 100));
        // the 16-bit form: clear adds a fourth channel; a picture twice as wide reads the coverage scaled
        let (out, alpha) = apply16(&vec![1000u16; 8 * 3 * 3], 8, 3, &c, 4, 3, Background::Clear);
        assert!(alpha && out.len() == 8 * 3 * 4);
        assert_eq!(out[(8 + 3) * 4 + 3], 65535);
        assert_eq!(out[7 * 4 + 3], 16384, "the top right corner: a quarter");
    }

    #[test]
    fn the_words_name_the_backgrounds() {
        assert_eq!(Background::of("clear", [0; 3]), Background::Clear);
        assert_eq!(Background::of("plate", [1, 2, 3]), Background::Plate([1, 2, 3]));
        assert_eq!(Background::of("anything", [0; 3]), Background::Seen);
    }
}
