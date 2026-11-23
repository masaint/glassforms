//! Video memory while a twist is dragged: what the device holds after each step of a slider drag, and
//! after it is let go. Run by hand (it prints a table): `cargo test --test twist_memory_gpu -- --ignored --nocapture`.
use gf_core::{mods::{Axes, Kind, Modifier}, state::{Detail, State}};
use glassforms::{engine::Engine, gpu::Gpu};

fn held(e: &Engine) -> (u64, usize, Vec<(String, u64)>) {
    let Some(r) = e.gpu.device.generate_allocator_report() else { return (0, 0, vec![]) };
    let mut big: Vec<(String, u64)> = r.allocations.iter().map(|a| (a.name.clone(), a.size)).collect();
    big.sort_by(|a, b| b.1.cmp(&a.1));
    big.truncate(8);
    (r.total_allocated_bytes, r.allocations.len(), big)
}

/// This process's dedicated video memory, from Windows' own per-process GPU counter (MB).
fn smi() -> String {
    let q = format!("(Get-Counter '\\GPU Process Memory(pid_{}_*)\\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum | ForEach-Object {{ [math]::Round($_.Sum / 1e6) }}", std::process::id());
    std::process::Command::new("powershell").args(["-NoProfile", "-Command", &q]).output()
        .map(|o| format!("{} MB", String::from_utf8_lossy(&o.stdout).trim())).unwrap_or_default()
}

fn drag(scene: &str, preset: &str, world: bool) {
    let Ok(gpu) = Gpu::headless("vulkan") else { eprintln!("no device"); return };
    let args = [("scene", scene), ("preset", preset)].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut st = State::with_args(&args, true);
    let twist = Modifier::of(Kind::Twist);
    if world { st.stack.world_modifiers_mut(gf_core::groups::Item::Object(0)).unwrap().list.push(twist); } else { st.selected_modifiers_mut().list.push(twist); }
    let mut e = Engine::new(gpu, st, true);
    let (w, h) = (1920, 1080);
    e.state.request(gf_core::build::BuildRequest::script(Detail::Full));
    for _ in 0..3 { e.frame(w, h, 1.0 / 60.0); }
    let (b0, n0, _) = held(&e);
    println!("{scene}/{preset} world={world}: settled {:.0} MB in {n0} allocations (process {})", b0 as f64 / 1e6, smi());
    for i in 0..120 {
        let deg = 180.0 + 540.0 * ((i as f64) * 0.11).sin();
        let a = Axes::new([0.0, deg, 0.0], false);
        if world { e.state.stack.world_modifiers_mut(gf_core::groups::Item::Object(0)).unwrap().list.last_mut().unwrap().a = a; }
        else { e.state.selected_modifiers_mut().list.last_mut().unwrap().a = a; }
        e.state.request(gf_core::build::BuildRequest::script(Detail::Quick));
        e.frame(w, h, 1.0 / 60.0);
        if i % 20 == 19 {
            let (b, n, _) = held(&e);
            println!("  drag step {:3}: {:.0} MB in {n} allocations (process {})", i + 1, b as f64 / 1e6, smi());
        }
    }
    e.state.request(gf_core::build::BuildRequest::script(Detail::Full));
    for _ in 0..3 { e.frame(w, h, 1.0 / 60.0); }
    let (b, n, _) = held(&e);
    println!("  let go: {:.0} MB in {n} allocations (process {})", b as f64 / 1e6, smi());
    std::thread::sleep(std::time::Duration::from_millis(2500));
    for _ in 0..2 { e.frame(w, h, 1.0 / 60.0); }
    e.gpu.wait();
    let (b, n, big) = held(&e);
    println!("  still for 2.5 s: {:.0} MB in {n} allocations (process {})", b as f64 / 1e6, smi());
    for (name, size) in big { println!("    {:7.1} MB  {name}", size as f64 / 1e6); }
}

/// A world modifier of this kind on the scene's one object, built once at full detail: what is held.
fn world_kind(scene: &str, preset: &str, kind: Kind, turn: f64) {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let args = [("scene", scene), ("preset", preset)].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut st = State::with_args(&args, true);
    let mut m = Modifier::of(kind);
    if turn != 0.0 { m.a = Axes::new([0.0, turn, 0.0], false); }
    st.stack.world_modifiers_mut(gf_core::groups::Item::Object(0)).unwrap().list.push(m);
    let k = st.stack.box_k();
    let mut e = Engine::new(gpu, st, true);
    e.state.request(gf_core::build::BuildRequest::script(Detail::Full));
    for _ in 0..3 { e.frame(1920, 1080, 1.0 / 60.0); }
    let (b, n, big) = held(&e);
    println!("{scene}/{preset} world {kind:?} {turn}: box {k}, scene res {}, {:.0} MB in {n} allocations (process {}){}", e.state.scene_res(Detail::Full), b as f64 / 1e6, smi(), e.error.as_deref().map(|x| format!(" error: {x}")).unwrap_or_default());
    for (name, size) in big.into_iter().take(4) { println!("    {:7.1} MB  {name}", size as f64 / 1e6); }
}

#[test]
#[ignore]
fn world_modifier_memory() {
    for (kind, turn) in [(Kind::Twist, 0.0), (Kind::Twist, 720.0), (Kind::Bend, 0.0), (Kind::Swirl, 0.0), (Kind::Taper, 0.0), (Kind::Repeat, 0.0), (Kind::Wrap, 0.0), (Kind::Offset, 0.0), (Kind::Mirror, 0.0)] {
        world_kind("attractors", "thomas", kind, turn);
    }
}

#[test]
#[ignore]
fn twist_drag_memory() {
    drag("attractors", "thomas", false);
    drag("attractors", "thomas", true);
    drag("solids", "sphere", false);
}

/// The owner's steps: in the world view a modifier on an object, then cascade on. Built once at full
/// detail, then each later frame: what is held, and what went wrong.
fn cascade(kind: Kind, scope: gf_core::world_mods::Scope, objects: usize) {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let args = [("scene", "attractors"), ("preset", "thomas")].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut st = State::with_args(&args, true);
    for _ in 1..objects { st.create_world_object(st.fam, st.shape().preset, false, None).unwrap(); }
    let mut m = Modifier::of(kind);
    m.scope = scope;
    st.stack.world_modifiers_mut(gf_core::groups::Item::Object(0)).unwrap().list.push(m);
    let mut e = Engine::new(gpu, st, true);
    e.state.request(gf_core::build::BuildRequest::script(Detail::Full));
    for _ in 0..3 { e.frame(1920, 1080, 1.0 / 60.0); }
    let (b0, _, _) = held(&e);
    let plan0 = e.state.stack.world_plan();
    println!("{kind:?} {scope:?} x{objects} cascade off: {:.0} MB, {} fields, box {}, scene res {}", b0 as f64 / 1e6, plan0.fields.len(), e.state.stack.box_k(), e.state.scene_res(Detail::Full));
    e.state.stack.world_modifiers_mut(gf_core::groups::Item::Object(0)).unwrap().list[0].cascade = true;
    e.state.sculpt_changed(false);
    e.state.request(gf_core::build::BuildRequest::script(Detail::Full));
    for i in 0..3 {
        e.frame(1920, 1080, 1.0 / 60.0);
        let (b, n, big) = held(&e);
        let plan = e.state.stack.world_plan();
        println!("  cascade on, frame {i}: {:.0} MB in {n} allocations, {} fields, box {}, scene res {} (process {}){}", b as f64 / 1e6, plan.fields.len(), e.state.stack.box_k(), e.state.scene_res(Detail::Full), smi(), e.error.as_deref().map(|x| format!(" ERROR {x}")).unwrap_or_default());
        if i == 2 { for (name, size) in big.into_iter().take(5) { println!("    {:7.1} MB  {name}", size as f64 / 1e6); } }
    }
    println!("{}", glassforms::breadcrumbs::dump());
}

#[test]
#[ignore]
fn world_cascade_memory() {
    use gf_core::world_mods::Scope;
    for (kind, scope, objects) in [(Kind::Twist, Scope::World, 1), (Kind::Twist, Scope::SubObjects, 1), (Kind::Twist, Scope::World, 2), (Kind::Repeat, Scope::World, 1), (Kind::Bend, Scope::SubObjects, 2)] {
        cascade(kind, scope, objects);
    }
}
