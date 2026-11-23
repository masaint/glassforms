//! The look: colours, fonts and sizes, and egui's own style brought into line with them.
//! Everything is taken from the web version's style sheet, so the two read as the same program.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui::{self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Pos2, Rect, Shape, Stroke};

/// The two looks the window comes in: dark (the picture's own world, quiet around it) and light (clean
/// and bright). Which one is shown is chosen in the settings (dark, light, or as Windows has it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Dark,
    Light,
}

/// The two settings of the panels' glass, each 0 to 1. Blur: how soft the picture under them is (0: none,
/// the plain boxes of before, and nothing is made for them). Opacity: how much of the picture they let
/// through, from most (0) to nearly none (1). The defaults are the look the panels had before either could
/// be set: fully frosted, the tone band as it was.
pub const BLUR_DEFAULT: f32 = 0.75;
pub const OPACITY_DEFAULT: f32 = 0.5;

/// The blur and opacity the old named frosts stand for ("off", "subtle", "full"), for settings files and
/// scripts written before there were sliders.
pub fn frost_named(name: &str) -> (f32, f32) {
    match name {
        "off" => (0.0, OPACITY_DEFAULT),
        "subtle" => (0.5, 0.7),
        _ => (BLUR_DEFAULT, OPACITY_DEFAULT),
    }
}

/// The opacity as a lean either way from the default: -1 lets the most through, 1 the least.
fn lean() -> f32 {
    ((opacity() - OPACITY_DEFAULT) * 2.0).clamp(-1.0, 1.0)
}

/// An alpha leaned by the opacity: `base` at the default, `low` of it at the least, `top` at the most.
fn leaned_alpha(base: u8, low: f32, top: u8) -> u8 {
    let k = lean();
    let b = base as f32;
    let a = if k < 0.0 { b * (1.0 + (1.0 - low) * k) } else { b + (top as f32 - b).max(0.0) * k };
    a.round().clamp(0.0, 255.0) as u8
}

/// The colours of one look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    /// ordinary text
    pub text: Color32,
    /// the name of a control
    pub dim: Color32,
    /// a value, and anything lit
    pub bright: Color32,
    /// group titles and notes
    pub faint: Color32,
    /// the hint at the foot of a panel
    pub hint: Color32,
    /// the words on a button
    pub button: Color32,
    /// the words on a lit button
    pub ink: Color32,
    /// a lit button itself
    pub lit: Color32,
    /// the words and marks of a control under the pointer
    pub hot: Color32,
    /// a slider's handle and the filled part of its rail
    pub accent: Color32,
    /// the same for the small sliders under an animated knob
    pub accent_mini: Color32,
    /// the second handle of a range
    pub handle_to: Color32,
    /// an error's words and the box behind them
    pub error_text: Color32,
    pub error_fill: Color32,
    /// menus and hover notes
    pub menu_fill: Color32,
    /// a slider's value taken past the slider's own ends (with alt, or typed): out of bounds
    pub past: Color32,
    /// the mark of something that costs noticeably more to draw
    pub costly: Color32,
    /// what the faint fills and lines on a surface are made of (`white` gives it at an opacity): white
    /// over the dark look, a dark slate over the light one; and how strong it is made
    pub over: [u8; 3],
    pub over_gain: f32,
    /// the boxes when the frost is off: a panel (over an ordinary picture, and over one bright from edge
    /// to edge) and a window that floats over the panels
    pub solid: [u8; 3],
    pub solid_alpha: [u8; 2],
    pub float: [u8; 4],
    /// the soft bed under the line of figures at the foot of the window, and its alpha (ordinary, bright)
    pub bed: [u8; 3],
    pub bed_alpha: [u8; 2],
    /// the frost's own veil over a floating window (it is nearer than the panels, and a little more solid)
    pub veil: [u8; 4],
    /// the rim of glass round a surface, and the brighter line along its top where the light catches it
    pub rim: [u8; 4],
    pub rim_top: [u8; 4],
    /// the shadow under a surface: its darkness for a bar, a panel and a floating window
    pub shadow: [u8; 3],
    pub dark: bool,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color32 {
    Color32::from_rgb(r, g, b)
}

/// Today's look, made airier: the same quiet greys over frosted glass, the names a little lighter so
/// that they hold up over the brighter glass.
pub const DARK: Palette = Palette {
    text: rgb(0xcf, 0xd4, 0xdf),
    dim: rgb(0x9a, 0xa1, 0xb1),
    bright: rgb(0xe6, 0xeb, 0xf6),
    faint: rgb(0x90, 0x97, 0xa8),
    hint: rgb(0x80, 0x87, 0x99),
    button: rgb(0xb2, 0xb9, 0xc8),
    ink: rgb(0x0b, 0x0c, 0x10),
    lit: rgb(0xe6, 0xeb, 0xf6),
    hot: Color32::WHITE,
    accent: rgb(0xb9, 0xc4, 0xe6),
    accent_mini: rgb(0x8a, 0x98, 0xc0),
    handle_to: rgb(0x8f, 0xa0, 0xd0),
    error_text: rgb(0xff, 0x9a, 0x9a),
    error_fill: rgb(0x12, 0x06, 0x08),
    menu_fill: rgb(0x16, 0x18, 0x20),
    past: rgb(0x5f, 0xe3, 0xc0),
    costly: rgb(0xe6, 0xb4, 0x6a),
    over: [255, 255, 255],
    over_gain: 1.0,
    solid: [9, 10, 15],
    solid_alpha: [200, 236],
    float: [11, 12, 18, 242],
    bed: [0, 0, 0],
    bed_alpha: [70, 150],
    veil: [6, 7, 11, 56],
    rim: [255, 255, 255, 22],
    rim_top: [255, 255, 255, 40],
    shadow: [50, 70, 96],
    dark: true,
};

/// Clean and bright: dark slate words on pale frosted glass, a deep blue for what is lit and moved.
pub const LIGHT: Palette = Palette {
    text: rgb(0x2a, 0x2f, 0x3a),
    dim: rgb(0x52, 0x59, 0x69),
    bright: rgb(0x12, 0x15, 0x1d),
    faint: rgb(0x5f, 0x66, 0x77),
    hint: rgb(0x74, 0x7b, 0x8b),
    button: rgb(0x38, 0x3e, 0x4b),
    ink: rgb(0xf7, 0xf8, 0xfb),
    lit: rgb(0x2c, 0x3d, 0x6e),
    hot: rgb(0x00, 0x00, 0x00),
    accent: rgb(0x3a, 0x55, 0xa4),
    accent_mini: rgb(0x6f, 0x83, 0xbd),
    handle_to: rgb(0x5a, 0x72, 0xb8),
    error_text: rgb(0xb4, 0x23, 0x1f),
    error_fill: rgb(0xfd, 0xec, 0xea),
    menu_fill: rgb(0xf5, 0xf6, 0xf9),
    past: rgb(0x0a, 0x86, 0x68),
    costly: rgb(0xb0, 0x6c, 0x12),
    over: [22, 27, 38],
    over_gain: 0.9,
    solid: [244, 245, 248],
    solid_alpha: [232, 246],
    float: [248, 249, 251, 248],
    bed: [250, 251, 253],
    bed_alpha: [150, 205],
    veil: [255, 255, 255, 64],
    rim: [255, 255, 255, 150],
    rim_top: [255, 255, 255, 235],
    shadow: [18, 26, 40],
    dark: false,
};

static LIGHT_ON: AtomicBool = AtomicBool::new(false);
/// the blur and the opacity in use (f32 bits)
static BLUR: AtomicU32 = AtomicU32::new(0x3f40_0000); // 0.75
static OPACITY: AtomicU32 = AtomicU32::new(0x3f00_0000); // 0.5
/// The frosted copy of the picture, as egui knows it, and the place in the window the picture fills.
static BACKDROP: Mutex<Option<(egui::TextureId, Rect)>> = Mutex::new(None);

/// The colours of the look in use.
pub fn pal() -> &'static Palette {
    if LIGHT_ON.load(Ordering::Relaxed) { &LIGHT } else { &DARK }
}
pub fn mode() -> Mode {
    if LIGHT_ON.load(Ordering::Relaxed) { Mode::Light } else { Mode::Dark }
}
pub fn blur() -> f32 {
    f32::from_bits(BLUR.load(Ordering::Relaxed))
}
pub fn opacity() -> f32 {
    f32::from_bits(OPACITY.load(Ordering::Relaxed))
}
/// The panels are frosted glass (the blur is above nothing).
pub fn frosted() -> bool {
    blur() > 0.0
}
pub fn set_glass(blur: f32, opacity: f32) {
    BLUR.store(blur.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    OPACITY.store(opacity.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}
/// The frosted copy of the picture the surfaces are drawn on, and where the picture is (none: plain boxes).
pub fn set_backdrop(b: Option<(egui::TextureId, Rect)>) {
    if let Ok(mut g) = BACKDROP.lock() { *g = b; }
}
fn backdrop() -> Option<(egui::TextureId, Rect)> {
    BACKDROP.lock().ok().and_then(|g| *g)
}

/// The look a setting names: "dark", "light", or "system" (as Windows has it, dark if it does not say).
pub fn mode_named(name: &str, system: Option<egui::Theme>) -> Mode {
    match name {
        "light" => Mode::Light,
        "system" => if system == Some(egui::Theme::Light) { Mode::Light } else { Mode::Dark },
        _ => Mode::Dark,
    }
}
/// How far the picture is blurred for a blur setting, as levels of halving and the spread of the taps
/// within a level: the reach doubles every quarter of the slider, from a quarter of the picture's size
/// (2 levels) up; one level more for a picture of more pixels, so that it is as soft at every size. The
/// default is exactly the frost of before (5 levels at 4K, 4 below, spread 1).
pub fn blur_reach(blur: f32, height_px: u32) -> (u32, f32) {
    let big = height_px >= 1800;
    let s = (if big { 2.0 } else { 1.0 } + 4.0 * blur.clamp(0.0, 1.0)).max(2.0);
    let levels = (s - 1e-4).ceil().clamp(2.0, 6.0);
    (levels as u32, 2f32.powf((s - levels).min(0.0)))
}

/// The band of tone the frost presses the picture into, for a look, a picture bright from edge to edge or
/// not, and an opacity lean (-1 to 1, see `lean`). The band is set by the brightest picture there is: over
/// a white room the frost's brightest is still dark enough, or in the light look light enough, that the
/// faintest words over it read; letting more through widens it only so far, letting less through closes
/// it toward the panel's own colour.
fn band(dark: bool, bright: bool, k: f32) -> ([f32; 3], [f32; 3], f32) {
    let towards = |a: [f32; 3], b: [f32; 3], t: f32| [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t);
    let shut = 0.8 * k.max(0.0);
    let sat = |s: f32| s + (1.0 - s) * k.max(0.0);
    if dark {
        let lo = [0.035, 0.04, 0.055];
        let wide = 1.0 + 0.15 * (-k).max(0.0);
        let hi = [0.255, 0.27, 0.305].map(|v| v * if bright { 0.8 } else { 1.0 } * wide);
        (lo, towards(hi, lo, shut), sat(1.25))
    } else {
        let hi = [0.975, 0.978, 0.985];
        let lift = if bright { 0.04 } else { 0.0 };
        let lo = [0.77, 0.79, 0.83].map(|v| v + lift - 0.08 * (-k).max(0.0));
        (towards(lo, hi, shut), hi, sat(1.3))
    }
}

/// How the frost is made for the look in use: its blur (`blur_reach`) and its band of tone (`band`).
/// None: no frost (the blur is off).
pub fn frost_look(bright: bool, height_px: u32) -> Option<crate::backdrop::Look> {
    if !frosted() { return None; }
    let (levels, spread) = blur_reach(blur(), height_px);
    let (lo, hi, sat) = band(pal().dark, bright, lean());
    Some(crate::backdrop::Look { levels, spread, lo, hi, sat })
}

/// What a surface is: a bar over the picture, a panel, or a window floating over both (each nearer the
/// eye than the last, with a longer shadow).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lift {
    Bar,
    Panel,
    Float,
}

/// A surface of frosted glass over the picture: its soft shadow, the blurred picture under it (or a plain
/// box while the frost is off), and its rim, brighter along the top. Painted under whatever goes on it.
pub fn glass(rect: Rect, radius: u8, lift: Lift, bright: bool) -> Shape {
    let p = pal();
    let mut v: Vec<Shape> = Vec::with_capacity(5);
    let (off, blur, k) = match lift { Lift::Bar => (2, 10, 0), Lift::Panel => (4, 22, 1), Lift::Float => (8, 30, 2) };
    let shadow = egui::Shadow { offset: [0, off], blur, spread: 0, color: Color32::from_black_alpha(p.shadow[k]) };
    v.push(shadow.as_shape(rect, radius).into());
    let a = |c: [u8; 4]| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
    match backdrop().filter(|_| frosted()) {
        Some((id, full)) => {
            let uv = |q: Pos2| Pos2::new((q.x - full.left()) / full.width().max(1.0), (q.y - full.top()) / full.height().max(1.0));
            v.push(egui::epaint::RectShape::filled(rect, radius, Color32::WHITE).with_texture(id, Rect::from_min_max(uv(rect.min), uv(rect.max))).into());
            if lift == Lift::Float { v.push(Shape::rect_filled(rect, radius, a([p.veil[0], p.veil[1], p.veil[2], leaned_alpha(p.veil[3], 0.5, 170)]))); }
        }
        None => v.push(Shape::rect_filled(rect, radius, if lift == Lift::Float { float_fill() } else { panel_fill(bright) })),
    }
    v.push(Shape::rect_stroke(rect, radius, Stroke::new(1.0, a(p.rim)), egui::StrokeKind::Inside));
    // the light caught along the top edge, fading out at the corners' start
    let r = radius as f32;
    if rect.width() > 2.0 * r + 4.0 {
        let y = rect.top() + 0.5;
        v.push(Shape::line_segment([Pos2::new(rect.left() + r, y), Pos2::new(rect.right() - r, y)], Stroke::new(1.0, a(p.rim_top))));
    }
    Shape::Vec(v)
}

/// Draw `body` on a surface of frosted glass (see `glass`), with this much room round it inside.
pub fn glass_frame<R>(ui: &mut egui::Ui, margin: egui::Margin, radius: u8, lift: Lift, bright: bool, body: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
    let behind = ui.painter().add(Shape::Noop);
    let shown = egui::Frame::new().inner_margin(margin).show(ui, body);
    ui.painter().set(behind, glass(shown.response.rect, radius, lift, bright));
    shown
}

/// The faint stuff of a surface at this opacity (0 to 1): white over the dark look, a dark slate over
/// the light one. (Named for the dark look, where it began.)
pub fn white(a: f32) -> Color32 {
    let p = pal();
    Color32::from_rgba_unmultiplied(p.over[0], p.over[1], p.over[2], (a * p.over_gain * 255.0).round().clamp(0.0, 255.0) as u8)
}
/// A colour at this opacity (0 to 1).
pub fn fade(c: Color32, a: f32) -> Color32 {
    c.gamma_multiply(a)
}
/// The box behind a panel while the frost is off: a little of the picture showing through. `bright`: the
/// picture behind it is a bright one from edge to edge (an environment picture shown behind the shape),
/// and the box lets much less of it through, so that the dimmer words stay easy to read.
pub fn panel_fill(bright: bool) -> Color32 {
    let p = pal();
    // (letting the most through still keeps seven tenths of the box, so the dimmer words read)
    Color32::from_rgba_unmultiplied(p.solid[0], p.solid[1], p.solid[2], leaned_alpha(p.solid_alpha[bright as usize], 0.7, 252))
}
/// The line of figures at the foot of the window: there to be read when looked for, not to be noticed.
pub fn quiet() -> Color32 {
    fade(pal().bright, if pal().dark { 0.46 } else { 0.62 })
}
/// The soft bed under the line of figures (and the "?"), `bright` over a picture bright from edge to edge.
pub fn bed(bright: bool) -> Color32 {
    let p = pal();
    Color32::from_rgba_unmultiplied(p.bed[0], p.bed[1], p.bed[2], p.bed_alpha[bright as usize])
}
/// The box behind a window that floats over the picture and may lie over a panel, while the frost is
/// off: more solid than a panel, so that what is under it does not show through its words.
pub fn float_fill() -> Color32 {
    let f = pal().float;
    Color32::from_rgba_unmultiplied(f[0], f[1], f[2], leaned_alpha(f[3], 0.8, 252))
}

/// Put this look in use: the colours, and egui's own style (menus, hover notes, scroll bars, text fields).
/// Does nothing if it is the one in use already.
pub fn set_mode(ctx: &egui::Context, m: Mode) {
    if LIGHT_ON.swap(m == Mode::Light, Ordering::Relaxed) == (m == Mode::Light) && ctx.data(|d| d.get_temp::<bool>(egui::Id::unique("theme installed"))).is_some() { return; }
    install_style(ctx);
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
pub const LAB_W: f32 = 330.0;
pub const LAB_LABEL: f32 = 138.0;
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

/// Quiet and monospaced, in the look in use: the parts of egui's own style that still show through (menus,
/// hover notes, scroll bars, text fields). Menus and hover notes are the same glass as the panels, made
/// nearly solid: they are small, and appear over anything.
pub fn install_style(ctx: &egui::Context) {
    let p = *pal();
    ctx.set_theme(if p.dark { egui::Theme::Dark } else { egui::Theme::Light });
    ctx.data_mut(|d| d.insert_temp(egui::Id::unique("theme installed"), true));
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
        *v = if p.dark { egui::Visuals::dark() } else { egui::Visuals::light() };
        let over = |a: f32| Color32::from_rgba_unmultiplied(p.over[0], p.over[1], p.over[2], (a * p.over_gain * 255.0).round() as u8);
        v.override_text_color = Some(p.text);
        v.window_fill = p.menu_fill.gamma_multiply(if p.dark { 0.97 } else { 0.98 });
        v.panel_fill = if p.dark { Color32::BLACK } else { Color32::WHITE };
        v.window_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(p.rim[0], p.rim[1], p.rim[2], if p.dark { 36 } else { 255 }));
        v.window_corner_radius = CornerRadius::same(BUTTON_RADIUS + 2);
        v.menu_corner_radius = CornerRadius::same(BUTTON_RADIUS + 2);
        v.popup_shadow = egui::Shadow { offset: [0, 5], blur: 16, spread: 0, color: Color32::from_black_alpha(p.shadow[2]) };
        v.window_shadow = v.popup_shadow;
        v.extreme_bg_color = over(0.05);
        v.faint_bg_color = over(0.03);
        v.selection.bg_fill = fade(p.accent, 0.35);
        v.selection.stroke = Stroke::new(1.0, p.bright);
        v.text_cursor.stroke = Stroke::new(1.0, p.bright);
        v.hyperlink_color = p.accent;
        for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
            w.corner_radius = CornerRadius::same(BUTTON_RADIUS);
            w.bg_stroke = Stroke::new(1.0, over(0.1));
            w.fg_stroke = Stroke::new(1.0, p.bright);
            w.bg_fill = over(0.05);
            w.weak_bg_fill = over(0.05);
            w.expansion = 0.0;
        }
        v.widgets.inactive.fg_stroke = Stroke::new(1.0, p.button);
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, over(0.3));
        v.widgets.active.bg_stroke = Stroke::new(1.0, over(0.3));
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

    /// The defaults are the frost of before, exactly: 5 levels at 4K and 4 below, spread 1, the old band.
    #[test]
    fn the_default_glass_is_the_look_of_before() {
        assert_eq!(blur_reach(BLUR_DEFAULT, 2160), (5, 1.0));
        assert_eq!(blur_reach(BLUR_DEFAULT, 900), (4, 1.0));
        assert_eq!(band(true, false, 0.0), ([0.035, 0.04, 0.055], [0.255, 0.27, 0.305], 1.25));
        assert_eq!(band(false, false, 0.0), ([0.77, 0.79, 0.83], [0.975, 0.978, 0.985], 1.3));
        assert_eq!(frost_named("full"), (BLUR_DEFAULT, OPACITY_DEFAULT));
        assert_eq!(frost_named("off").0, 0.0);
        // the reach grows with the slider and stays within the levels the backdrop makes
        let mut last = 0.0;
        for i in 0..=40 {
            let (l, s) = blur_reach(i as f32 / 40.0, 2160);
            assert!((2..=6).contains(&l) && s > 0.49 && s <= 1.0);
            let reach = 2f32.powi(l as i32) * s;
            assert!(reach >= last, "{i}: {reach} < {last}");
            last = reach;
        }
    }

    /// The faintest words (group titles, notes) keep a contrast of at least 2.5 to 1 against the frost's
    /// worst tone at every opacity, in both looks (WCAG ratio of relative luminances).
    #[test]
    fn the_faintest_words_read_at_every_opacity() {
        let lin = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
        let lum = |c: [f32; 3]| 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]);
        let of = |c: Color32| [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0];
        for i in 0..=20 {
            let k = -1.0 + i as f32 / 10.0;
            for bright in [false, true] {
                let (_, hi, _) = band(true, bright, k);
                let r = (lum(of(DARK.faint)) + 0.05) / (lum(hi) + 0.05);
                assert!(r >= 2.5, "dark, lean {k}: {r}");
                let (lo, _, _) = band(false, bright, k);
                let r = (lum(lo) + 0.05) / (lum(of(LIGHT.faint)) + 0.05);
                assert!(r >= 2.5, "light, lean {k}: {r}");
            }
        }
    }

    #[test]
    fn the_icon_is_a_square_of_pixels() {
        let icon = icon();
        assert_eq!(icon.rgba.len() as u32, icon.width * icon.height * 4);
        assert!(icon.rgba.chunks(4).any(|p| p[3] == 255) && icon.rgba.chunks(4).any(|p| p[3] == 0));
    }
}
