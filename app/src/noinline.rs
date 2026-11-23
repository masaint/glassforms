//! Shaders with functions kept out of line. A tracer reads a shape's formula at a dozen places (every
//! settle and every normal), and the driver puts the whole formula in at each of them: Thomas traced
//! exactly came to 760 KB of GPU code (11 s to compile), ten times the volume tracer's 71 KB, and ran
//! slower everywhere, marching included (docs/project-notes/plans/csg_perf.md). WGSL cannot ask for a
//! function not to be inlined, but SPIR-V can (FunctionControl DontInline), and the NVIDIA driver keeps
//! such a function as one copy that each place calls. So a text that names functions in a line
//! `// noinline: name name ...` is turned into SPIR-V here, the same way wgpu turns WGSL into it for
//! Vulkan with every run-time check left out (`Gpu::shader` with `hot`), the named functions marked, and
//! handed to the driver as it is (`Features::PASSTHROUGH_SHADERS`). Any failure gives None, and the text
//! is compiled the ordinary way.

/// The bindings of the bind group layouts a passthrough module may be used with, by label. wgpu's Vulkan
/// backend numbers each layout's bindings 0, 1, 2... in the order of their numbers and rewrites the
/// shader's to match; SPIR-V handed over as it is must already be numbered so (`spirv`).
static LAYOUTS: std::sync::Mutex<Vec<(String, Vec<u32>)>> = std::sync::Mutex::new(Vec::new());

/// Keep a layout's bindings (Gpu::bind_layout calls this for every layout it makes).
pub fn remember(label: &str, entries: &[wgpu::BindGroupLayoutEntry]) {
    let mut b: Vec<u32> = entries.iter().map(|e| e.binding).collect();
    b.sort_unstable();
    let Ok(mut l) = LAYOUTS.lock() else { return };
    // (two layouts of one label with other bindings: neither can be trusted)
    match l.iter_mut().find(|(n, _)| n == label) {
        Some((_, old)) if *old != b => old.clear(),
        Some(_) => {}
        None => l.push((label.to_string(), b)),
    }
}

/// The marker a text names its out-of-line functions with.
pub const MARK: &str = "// noinline: ";

/// The functions a text asks to keep out of line (the marker's names; GF_NOINLINE=<names> overrides it,
/// for measuring: "-" for none).
pub fn names(src: &str) -> Vec<String> {
    if let Ok(v) = std::env::var("GF_NOINLINE") {
        return v.split([',', ' ']).filter(|s| !s.is_empty() && *s != "-").map(str::to_string).collect();
    }
    let mut out: Vec<String> = Vec::new();
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix(MARK) {
            for n in rest.split_whitespace() { if !out.iter().any(|o| o == n) { out.push(n.to_string()); } }
        }
    }
    out
}

/// The SPIR-V of the text's entry point `entry` (vertex `vs`, fragment `fs`, compute `cs`...: all are
/// written into one module), with the named functions marked DontInline. None if the text does not parse,
/// or none of the names is found.
pub fn spirv(src: &str, names: &[String], groups: &[&str]) -> Option<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(src).ok()?;
    // the bindings as wgpu's Vulkan backend numbers them (see LAYOUTS): every one the text declares must be known
    let mut map = naga::back::spv::BindingMap::default();
    {
        let l = LAYOUTS.lock().ok()?;
        for (g, label) in groups.iter().enumerate() {
            let Some((_, b)) = l.iter().find(|(n, _)| n == label) else { continue };
            for (rank, &binding) in b.iter().enumerate() {
                map.insert(naga::ResourceBinding { group: g as u32, binding }, naga::back::spv::BindingInfo { descriptor_set: g as u32, binding: rank as u32, binding_array_size: None });
            }
        }
    }
    for (_, v) in module.global_variables.iter() {
        if let Some(rb) = &v.binding { if !map.contains_key(rb) { return None; } }
    }
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).ok()?;
    let mut options = naga::back::spv::Options::default();
    options.lang_version = (1, 6);
    // (as wgpu's Vulkan backend writes it, with the names kept: they are how the functions are found)
    options.flags = naga::back::spv::WriterFlags::LABEL_VARYINGS | naga::back::spv::WriterFlags::FORCE_POINT_SIZE | naga::back::spv::WriterFlags::DEBUG;
    options.capabilities = None;
    options.bounds_check_policies = naga::proc::BoundsCheckPolicies::default();
    options.force_loop_bounding = false;
    options.binding_map = map;
    let words = naga::back::spv::write_vec(&module, &info, &options, None).ok()?;
    let (words, n) = dont_inline(words, names);
    (n > 0).then_some(words)
}

/// `words` with FunctionControl DontInline set on every function whose OpName is one of `names`, and how
/// many were marked.
pub fn dont_inline(mut words: Vec<u32>, names: &[String]) -> (Vec<u32>, usize) {
    let mut ids = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        if len == 0 { break; }
        // OpName (5): the id, then a nul-terminated string
        if op == 5 && len > 2 {
            let bytes: Vec<u8> = words[i + 2..i + len].iter().flat_map(|w| w.to_le_bytes()).collect();
            let s = String::from_utf8_lossy(&bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())]).to_string();
            if names.iter().any(|n| *n == s) { ids.push(words[i + 1]); }
        }
        i += len;
    }
    let mut n = 0;
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        if len == 0 { break; }
        // OpFunction (54): result type, result id, function control (Inline 1, DontInline 2), type
        if op == 54 && ids.contains(&words[i + 2]) {
            words[i + 3] = (words[i + 3] & !1) | 2;
            n += 1;
        }
        i += len;
    }
    (words, n)
}

#[cfg(test)]
mod tests {
    #[test]
    fn marks_a_named_function() {
        let src = "// noinline: heavy
fn heavy(x: f32) -> f32 { var y = x; for (var i = 0; i < 8; i++) { y = sin(y) * x; } return y; }
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f { return vec4f(heavy(p.x), heavy(p.y), 0.0, 1.0); }
";
        let names = super::names(src);
        assert_eq!(names, vec!["heavy".to_string()]);
        assert!(super::spirv(src, &names, &[]).is_some());
        assert!(super::spirv(src, &["none".to_string()], &[]).is_none());
    }
}
