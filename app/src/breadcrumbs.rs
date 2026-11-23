//! What led up to a crash: the last changes the user made, how the scene stood after each, and the
//! video memory held then. Written into crash.log by the panic hook (main.rs), in plain text.
//!
//! Recorded only when a change settles into the undo history (a few times a minute at most), never per
//! frame. The ring holds the last `KEPT` entries; the panic hook only tries the lock, so a panic raised
//! while it is held still writes the log.
use std::sync::Mutex;

use gf_core::{mods::Mods, state::{Detail, State}};

pub const KEPT: usize = 30;

struct Trail {
    lines: Vec<String>,
    next: usize,
    /// window size and how the picture is drawn: set when they change
    window: String,
}

static TRAIL: Mutex<Trail> = Mutex::new(Trail { lines: Vec::new(), next: 0, window: String::new() });

fn clock() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02} UTC", s / 3600 % 24, s / 60 % 60, s % 60)
}

/// Keep one entry (the oldest goes once `KEPT` are held).
pub fn note(line: String) {
    let mut t = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let line = format!("{} {line}", clock());
    if t.lines.len() < KEPT { t.lines.push(line); } else { let i = t.next; t.lines[i] = line; }
    t.next = (t.next + 1) % KEPT;
}

/// The window's size and how its picture is drawn.
pub fn window(words: String) {
    let mut t = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if t.window != words { t.window = words; }
}

/// Everything kept, oldest first, for crash.log (empty if the trail is busy).
pub fn dump() -> String {
    let Ok(t) = TRAIL.try_lock() else { return "(the trail was being written when the program stopped)\n".into() };
    let mut out = format!("window: {}\nlast {} changes, oldest first:\n", t.window, t.lines.len());
    let n = t.lines.len();
    let start = if n < KEPT { 0 } else { t.next };
    for k in 0..n { out.push_str("  "); out.push_str(&t.lines[(start + k) % n]); out.push('\n'); }
    out
}

fn mods(m: &Mods) -> String {
    m.live().map(|m| {
        let a = m.a.get();
        format!("{:?}[a {:.3},{:.3},{:.3} size {:.3} amount {:.3}]", m.kind, a[0], a[1], a[2], m.size, m.amount)
    }).collect::<Vec<_>>().join(" ")
}

/// How the scene stands: each object's shape, detail and modifiers, the world modifiers, the level being
/// edited, playing or not, and the detail the memory budget allows.
pub fn scene(st: &State) -> String {
    let mut out = Vec::new();
    for i in 0..st.layer_count() {
        let Some(s) = st.layer_shape(i) else { continue };
        let own = mods(&s.mods);
        out.push(format!("object {}: {}/{} detail {} (moving {}){}", i + 1, st.layer_family(i).id, st.layer_preset(i).id, s.res, s.anim_res, if own.is_empty() { String::new() } else { format!(" mods {own}") }));
    }
    for (i, m) in &st.stack.world_mods {
        let w = mods(m);
        if !w.is_empty() { out.push(format!("world mods on object {}: {w}", i + 1)); }
    }
    let level = match st.edit_scope.object() { Some(i) => format!("inside object {}", i + 1), None => "world".into() };
    let budget = st.memory_scene();
    out.push(format!("{level}; {}; built at {} (still) / {} (moving); budget {:?}", if st.playing { "playing" } else { "still" }, st.scene_res(Detail::Full), st.scene_res(Detail::Anim), budget));
    out.join(" | ")
}

/// Video memory the device's allocator holds, where it can say (Vulkan and DX12 do).
pub fn gpu_memory(gpu: &crate::gpu::Gpu) -> String {
    match gpu.device.generate_allocator_report() {
        Some(r) => format!("GPU memory {:.0} MB in {} allocations", r.total_allocated_bytes as f64 / 1e6, r.allocations.len()),
        None => "GPU memory not reported".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trail_keeps_the_last_entries_oldest_first() {
        for i in 0..KEPT + 5 { note(format!("change {i}")); }
        window("1920x1080".into());
        let d = dump();
        assert!(d.contains("window: 1920x1080"));
        let lines: Vec<&str> = d.lines().filter(|l| l.contains("change ")).collect();
        assert_eq!(lines.len(), KEPT);
        assert!(lines[0].ends_with("change 5") && lines[KEPT - 1].ends_with(&format!("change {}", KEPT + 4)), "{d}");
    }

    #[test]
    fn the_scene_says_its_shapes_modifiers_and_level() {
        let mut st = State::new();
        st.selected_modifiers_mut().list.push(gf_core::mods::Modifier::of(gf_core::mods::Kind::Twist));
        let s = scene(&st);
        assert!(s.contains("object 1:") && s.contains("Twist[a 0.000,180.000,0.000") && s.contains("world;"), "{s}");
    }
}
