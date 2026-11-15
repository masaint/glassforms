//! The look: colours, fonts and sizes, and egui's own style brought into line with them.
//! Everything is taken from the web version's style sheet, so the two read as the same program.

use std::sync::Arc;

use eframe::egui::{self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Stroke};

/// Ordinary text.
pub const TEXT: Color32 = Color32::from_rgb(0xc9, 0xce, 0xd9);
/// The name of a control.
pub const DIM: Color32 = Color32::from_rgb(0x8c, 0x93, 0xa3);
/// A value, and anything lit.
pub const BRIGHT: Color32 = Color32::from_rgb(0xdf, 0xe5, 0xf2);
/// Group titles and notes.
pub const FAINT: Color32 = Color32::from_rgb(0x7a, 0x81, 0x92);
/// The hint at the foot of a panel.
pub const HINT: Color32 = Color32::from_rgb(0x63, 0x6a, 0x79);
/// The words on a button.
pub const BUTTON: Color32 = Color32::from_rgb(0xaa, 0xb1, 0xc0);
/// The words on a lit button.
pub const INK: Color32 = Color32::from_rgb(0x0b, 0x0c, 0x10);
/// A slider's handle and the filled part of its rail.
pub const ACCENT: Color32 = Color32::from_rgb(0xb9, 0xc4, 0xe6);
/// The same for the small sliders under an animated knob.
pub const ACCENT_MINI: Color32 = Color32::from_rgb(0x7f, 0x8d, 0xb5);
/// The second handle of a range.
pub const HANDLE_TO: Color32 = Color32::from_rgb(0x8f, 0xa0, 0xd0);
/// An error's words and the box behind them.
pub const ERROR_TEXT: Color32 = Color32::from_rgb(0xff, 0x9a, 0x9a);
pub const ERROR_FILL: Color32 = Color32::from_rgb(0x12, 0x06, 0x08);
/// Menus and hover notes.
pub const MENU_FILL: Color32 = Color32::from_rgb(0x14, 0x16, 0x1d);
/// The mark of something that costs noticeably more to draw.
pub const COSTLY: Color32 = Color32::from_rgb(0xe6, 0xb4, 0x6a);

/// White at this opacity (0 to 1).
pub fn white(a: f32) -> Color32 {
    Color32::from_white_alpha((a * 255.0).round() as u8)
}
/// A colour at this opacity (0 to 1).
pub fn fade(c: Color32, a: f32) -> Color32 {
    c.gamma_multiply(a)
}
/// The box behind a panel: dark, with a little of the picture showing through. `bright`: the picture
/// behind it is a bright one from edge to edge (an environment picture shown behind the shape), and the
/// box lets much less of it through, so that the dimmer words stay easy to read.
pub fn panel_fill(bright: bool) -> Color32 {
    Color32::from_rgba_unmultiplied(9, 10, 15, if bright { 236 } else { 200 })
}
pub fn panel_stroke() -> Stroke {
    Stroke::new(1.0, white(0.09))
}
/// The line of figures at the foot of the window: there to be read when looked for, not to be noticed.
pub fn quiet() -> Color32 {
    fade(BRIGHT, 0.46)
}
/// The box behind a window that floats over the picture and may lie over a panel: more solid than a
/// panel, so that what is under it does not show through its words.
pub fn float_fill() -> Color32 {
    Color32::from_rgba_unmultiplied(11, 12, 18, 242)
}

pub const PANEL_RADIUS: u8 = 9;
pub const BUTTON_RADIUS: u8 = 5;
/// Width inside the left and the right panel, and the gap between a panel and the window's edge.
pub const LEFT_W: f32 = 244.0;
pub const RIGHT_W: f32 = 270.0;
pub const EDGE: f32 = 14.0;
/// Width inside a window that floats over the picture (the light room, the help).
pub const FLOAT_W: f32 = 286.0;
/// The materials lab is wider, and so is its column for names: its controls have names like "double refraction".
pub const LAB_W: f32 = 310.0;
pub const LAB_LABEL: f32 = 120.0;
/// The columns of a slider's row: name, slider, value, wave button, and the gap between them. In the
/// right panel the names get more room: a shape's own sliders have names like "pavilion angle °".
pub const COL_LABEL: f32 = 84.0;
pub const COL_LABEL_WIDE: f32 = 100.0;
pub const COL_VALUE: f32 = 44.0;
pub const COL_WAVE: f32 = 18.0;
pub const COL_GAP: f32 = 6.0;
pub const ROW_H: f32 = 19.0;
/// Size of ordinary text, and the height of one line of it.
pub const SIZE: f32 = 11.0;
pub const SMALL: f32 = 10.5;
pub const LINE: f32 = 1.45;

pub fn mono(size: f32) -> FontId {
    FontId::new(size, FontFamily::Monospace)
}
pub fn sans(size: f32) -> FontId {
    FontId::new(size, FontFamily::Proportional)
}

/// How tall a font stands above and hangs below its baseline, and the extra gap it asks for between lines,
/// as fractions of its size; read from the font file by the same rule egui uses.
fn line_metrics(font: &[u8]) -> Option<(f32, f32, f32)> {
    let u16_at = |o: usize| font.get(o..o + 2).map(|b| u16::from_be_bytes([b[0], b[1]]));
    let i16_at = |o: usize| u16_at(o).map(|v| v as i16 as f32);
    let u32_at = |o: usize| font.get(o..o + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let table = |tag: &[u8; 4]| -> Option<usize> {
        (0..u16_at(4)? as usize).map(|i| 12 + i * 16).find(|&rec| font.get(rec..rec + 4) == Some(&tag[..])).and_then(|rec| u32_at(rec + 8))
    };
    let em = u16_at(table(b"head")? + 18)? as f32;
    let (mut up, mut down, mut gap) = (0.0, 0.0, 0.0);
    let os2 = table(b"OS/2");
    let typo = |o: usize| Some((i16_at(o + 68)?, i16_at(o + 70)?, i16_at(o + 72)?));
    // some fonts say their "typographic" figures are the ones to use; otherwise the older table's are
    let wants_typo = os2.and_then(|o| u16_at(o + 62)).is_some_and(|flags| flags & 0x80 != 0);
    if let (true, Some(o)) = (wants_typo, os2) {
        (up, down, gap) = typo(o)?;
    } else if let Some(h) = table(b"hhea") {
        (up, down, gap) = (i16_at(h + 4)?, i16_at(h + 6)?, i16_at(h + 8)?);
    }
    if up == 0.0 && down == 0.0 { (up, down, gap) = typo(os2?)?; }
    (em > 0.0).then_some((up / em, down / em, gap / em))
}

/// Use the fonts the web version asks for where the system has them: Cascadia Mono, then Consolas, for
/// everything; Segoe UI for notes; Segoe UI Symbol for the few signs the others lack. egui's built-in
/// fonts stay at the end of each list, so nothing is ever left without a glyph.
pub fn install_fonts(ctx: &egui::Context) {
    let mut defs = FontDefinitions::default();
    let dir = std::env::var_os("WINDIR").map(std::path::PathBuf::from).unwrap_or_else(|| r"C:\Windows".into()).join("Fonts");
    let files: Vec<(&str, Vec<u8>)> = [("cascadia", "CascadiaMono.ttf"), ("consolas", "consola.ttf"), ("segoe", "segoeui.ttf"), ("symbols", "seguisym.ttf")]
        .into_iter()
        .filter_map(|(name, file)| std::fs::read(dir.join(file)).ok().map(|bytes| (name, bytes)))
        .collect();
    for (family, wanted) in [(FontFamily::Monospace, &["cascadia", "consolas", "segoe", "symbols"][..]), (FontFamily::Proportional, &["segoe", "symbols"][..])] {
        let mut list: Vec<String> = Vec::new();
        // the first font of a family sets its baseline
        let mut first: Option<(f32, f32)> = None;
        for (name, bytes) in files.iter().filter(|(n, _)| wanted.contains(n)) {
            let mut data = FontData::from_owned(bytes.clone());
            if let Some((up, down, gap)) = line_metrics(bytes) {
                let height = up - down + gap;
                match first {
                    None => first = Some((up, height)),
                    // egui centres a stand-in glyph in the line by its own font's height, which leaves it
                    // sitting above or below the line of text around it: move it onto the family's baseline
                    Some((first_up, first_height)) => data.tweak.y_offset_factor = first_up - up - 0.5 * (first_height - height),
                }
            }
            let key = format!("{name} in {family:?}");
            defs.font_data.insert(key.clone(), Arc::new(data));
            list.push(key);
        }
        let builtin = defs.families.entry(family).or_default();
        list.append(builtin);
        *builtin = list;
    }
    ctx.set_fonts(defs);
}

/// Dark, quiet, monospaced: the parts of egui's own style that still show through (menus, hover notes,
/// scroll bars, text fields).
pub fn install_style(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|s| {
        use egui::TextStyle::*;
        s.text_styles = [(Body, mono(SIZE)), (Button, mono(SIZE)), (Monospace, mono(SIZE)), (Small, mono(10.0)), (Heading, mono(12.0))].into();
        s.spacing.item_spacing = egui::vec2(0.0, 0.0);
        s.spacing.button_padding = egui::vec2(8.0, 3.0);
        s.spacing.interact_size = egui::vec2(18.0, 17.0);
        s.spacing.menu_margin = egui::Margin::same(4);
        s.spacing.tooltip_width = 260.0;
        s.spacing.scroll = egui::style::ScrollStyle::thin();
        s.spacing.scroll.bar_width = 6.0;
        s.spacing.scroll.floating_allocated_width = 0.0;
        s.interaction.selectable_labels = false;
        s.interaction.tooltip_delay = 0.4;
        s.animation_time = 0.08;
        let v = &mut s.visuals;
        *v = egui::Visuals::dark();
        v.override_text_color = Some(TEXT);
        v.window_fill = MENU_FILL;
        v.panel_fill = Color32::BLACK;
        v.window_stroke = Stroke::new(1.0, white(0.14));
        v.window_corner_radius = CornerRadius::same(BUTTON_RADIUS);
        v.menu_corner_radius = CornerRadius::same(BUTTON_RADIUS);
        v.popup_shadow = egui::Shadow { offset: [0, 4], blur: 12, spread: 0, color: Color32::from_black_alpha(120) };
        v.extreme_bg_color = Color32::from_rgba_unmultiplied(255, 255, 255, 13);
        v.selection.bg_fill = fade(ACCENT, 0.35);
        v.selection.stroke = Stroke::new(1.0, BRIGHT);
        v.text_cursor.stroke = Stroke::new(1.0, BRIGHT);
        for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
            w.corner_radius = CornerRadius::same(BUTTON_RADIUS);
            w.bg_stroke = Stroke::new(1.0, white(0.1));
            w.fg_stroke = Stroke::new(1.0, BRIGHT);
            w.bg_fill = white(0.05);
            w.weak_bg_fill = white(0.05);
            w.expansion = 0.0;
        }
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, white(0.3));
        v.widgets.active.bg_stroke = Stroke::new(1.0, white(0.3));
    });
    // one pass per frame: a click must never be acted on twice
    ctx.options_mut(|o| o.max_passes = std::num::NonZeroUsize::MIN);
}

/// The program's icon: a ring of glass on black, its rim split into colours.
pub fn icon() -> egui::IconData {
    let n = 64usize;
    let mut rgba = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (u, v) = ((x as f32 + 0.5) / n as f32 * 2.0 - 1.0, (y as f32 + 0.5) / n as f32 * 2.0 - 1.0);
            let r = (u * u + v * v).sqrt();
            // a rounded dark tile
            let corner = ((u.abs() - 0.72).max(0.0).powi(2) + (v.abs() - 0.72).max(0.0).powi(2)).sqrt();
            let tile = (1.0 - (corner - 0.24) * 16.0).clamp(0.0, 1.0);
            // the ring, drawn three times at slightly different sizes: red outermost, blue innermost
            let band = |r0: f32| (1.0 - ((r - r0).abs() - 0.085) * 14.0).clamp(0.0, 1.0);
            let light = 0.55 + 0.45 * (-(u + v) * 0.7).clamp(-1.0, 1.0);
            let c = [band(0.60), band(0.565), band(0.53)];
            let i = (y * n + x) * 4;
            for k in 0..3 {
                let base = [0.035, 0.04, 0.06][k];
                rgba[i + k] = ((base + c[k] * light * [0.95, 0.97, 1.0][k]).min(1.0) * tile * 255.0) as u8;
            }
            rgba[i + 3] = (tile * 255.0) as u8;
        }
    }
    egui::IconData { rgba, width: n as u32, height: n as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_font_file_gives_up_its_line_metrics() {
        // the fonts built into egui will do as files to read
        let defs = FontDefinitions::default();
        for (name, data) in &defs.font_data {
            let bytes: &[u8] = (*data.font).as_ref();
            let (up, down, gap) = line_metrics(bytes).unwrap_or_else(|| panic!("{name}: no metrics"));
            assert!(up > 0.5 && up < 1.5 && down < 0.0 && down > -0.6 && (0.0..0.5).contains(&gap), "{name}: {up} {down} {gap}");
        }
        assert_eq!(line_metrics(&[0, 1, 0, 0]), None);
        assert_eq!(line_metrics(&[]), None);
    }

    #[test]
    fn the_icon_is_a_square_of_pixels() {
        let icon = icon();
        assert_eq!(icon.rgba.len() as u32, icon.width * icon.height * 4);
        assert!(icon.rgba.chunks(4).any(|p| p[3] == 255) && icon.rgba.chunks(4).any(|p| p[3] == 0));
    }
}
