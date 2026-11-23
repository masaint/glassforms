//! Measuring only. `regcount <tracer.wgsl> [more.wgsl ...]`: compiles each tracer text (as dumped by
//! GF_TRACER_DUMP) the way wgpu does (naga to SPIR-V, unchecked), builds a Vulkan graphics pipeline from
//! its `vs` and `fs` with VK_KHR_pipeline_executable_properties, and prints what the driver reports for
//! each stage (on NVIDIA: registers per thread, binary size, local memory / spills when it says so).

use ash::vk;
use std::ffi::CStr;

fn spirv(module: &naga::Module, info: &naga::valid::ModuleInfo, stage: naga::ShaderStage, entry: &str) -> Vec<u32> {
    let mut options = naga::back::spv::Options::default();
    options.lang_version = (1, 3);
    if std::env::var_os("REGCOUNT_NOINLINE").is_none() { options.flags.remove(naga::back::spv::WriterFlags::DEBUG); } else { options.flags.insert(naga::back::spv::WriterFlags::DEBUG); }
    options.bounds_check_policies = naga::proc::BoundsCheckPolicies::default();
    let po = naga::back::spv::PipelineOptions { shader_stage: stage, entry_point: entry.to_string() };
    let words = naga::back::spv::write_vec(module, info, &options, Some(&po)).expect("spir-v");
    match std::env::var("REGCOUNT_NOINLINE") { Ok(names) => dont_inline(words, &names.split(',').collect::<Vec<_>>()), Err(_) => words }
}

/// The SPIR-V with FunctionControl DontInline set on the functions named (by OpName, which naga writes
/// for functions whatever the flags).
pub fn dont_inline(mut words: Vec<u32>, names: &[&str]) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        if op == 5 {
            // OpName: id, then a nul-terminated string
            let bytes: Vec<u8> = words[i + 2..i + len].iter().flat_map(|w| w.to_le_bytes()).collect();
            let s = String::from_utf8_lossy(&bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())]).to_string();
            if names.contains(&s.as_str()) { ids.push(words[i + 1]); }
        }
        i += len.max(1);
    }
    let mut i = 5;
    let mut n = 0;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        // OpFunction: result type, result id, function control, function type
        if op == 54 && ids.contains(&words[i + 2]) { words[i + 3] = (words[i + 3] & !1) | 2; n += 1; }
        i += len.max(1);
    }
    eprintln!("dont-inline on {n} functions");
    words
}

fn descriptor_type(module: &naga::Module, g: &naga::GlobalVariable) -> vk::DescriptorType {
    match g.space {
        naga::AddressSpace::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
        naga::AddressSpace::Storage { .. } => vk::DescriptorType::STORAGE_BUFFER,
        _ => match &module.types[g.ty].inner {
            naga::TypeInner::Sampler { .. } => vk::DescriptorType::SAMPLER,
            naga::TypeInner::Image { class: naga::ImageClass::Storage { .. }, .. } => vk::DescriptorType::STORAGE_IMAGE,
            naga::TypeInner::Image { .. } => vk::DescriptorType::SAMPLED_IMAGE,
            other => panic!("binding of type {other:?}"),
        },
    }
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    unsafe {
        let entry = ash::Entry::load().expect("vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).expect("instance");
        let pd = instance.enumerate_physical_devices().unwrap().into_iter()
            .max_by_key(|&d| instance.get_physical_device_properties(d).device_type == vk::PhysicalDeviceType::DISCRETE_GPU).unwrap();
        let props = instance.get_physical_device_properties(pd);
        eprintln!("device: {:?}", CStr::from_ptr(props.device_name.as_ptr()));
        // every feature the device has, plus the executable statistics
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut fx = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12).push_next(&mut f13).push_next(&mut fx);
        instance.get_physical_device_features2(pd, &mut f2);

        let qf = instance.get_physical_device_queue_family_properties(pd).iter().position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS)).unwrap() as u32;
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let exts = [ash::khr::pipeline_executable_properties::NAME.as_ptr()];
        let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_extension_names(&exts).push_next(&mut f2);
        let device = instance.create_device(pd, &dci, None).expect("device");
        let pep = ash::khr::pipeline_executable_properties::Device::new(&instance, &device);

        for file in &files {
            let src = std::fs::read_to_string(file).expect("read");
            let module = match naga::front::wgsl::parse_str(&src) { Ok(m) => m, Err(e) => { eprintln!("{file}: {}", e.emit_to_string(&src)); continue; } };
            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("valid");
            // the bind group layouts the text declares
            let mut groups: Vec<Vec<vk::DescriptorSetLayoutBinding>> = Vec::new();
            for (_, g) in module.global_variables.iter() {
                let Some(b) = &g.binding else { continue };
                let gi = b.group as usize;
                while groups.len() <= gi { groups.push(Vec::new()); }
                groups[gi].push(vk::DescriptorSetLayoutBinding::default().binding(b.binding).descriptor_type(descriptor_type(&module, g)).descriptor_count(1).stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS));
            }
            let sets: Vec<vk::DescriptorSetLayout> = groups.iter().map(|b| device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(b), None).unwrap()).collect();
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets), None).unwrap();
            let vs = spirv(&module, &info, naga::ShaderStage::Vertex, "vs");
            let fs = spirv(&module, &info, naga::ShaderStage::Fragment, "fs");
            let vm = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs), None).unwrap();
            let fm = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fs), None).unwrap();
            let stages = [
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vm).name(c"vs"),
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fm).name(c"fs"),
            ];
            // (as many colour outputs as `fs` writes)
            let outputs = module.entry_points.iter().find(|e| e.name == "fs").map_or(1, |e| match &e.function.result {
                Some(r) => match &module.types[r.ty].inner { naga::TypeInner::Struct { members, .. } => members.len(), _ => 1 },
                None => 0,
            });
            let formats = vec![vk::Format::R16G16B16A16_SFLOAT; outputs];
            let blends = vec![vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA); outputs];
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
            let vi = vk::PipelineVertexInputStateCreateInfo::default();
            let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let rs = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
            let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
            let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let ds = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
            let ci = vk::GraphicsPipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR)
                .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia).viewport_state(&vp)
                .rasterization_state(&rs).multisample_state(&ms).color_blend_state(&cb).dynamic_state(&ds)
                .layout(layout).push_next(&mut rendering);
            let t0 = std::time::Instant::now();
            let pipe = match device.create_graphics_pipelines(vk::PipelineCache::null(), &[ci], None) { Ok(p) => p[0], Err((_, e)) => { eprintln!("{file}: pipeline {e:?}"); continue; } };
            let ms_compile = t0.elapsed().as_secs_f64() * 1e3;
            let execs = pep.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipe)).unwrap();
            let mut line = format!("{} | compile {ms_compile:.0} ms | spirv words fs {}", std::path::Path::new(file).file_name().unwrap().to_string_lossy(), fs.len());
            for (i, e) in execs.iter().enumerate() {
                let name = CStr::from_ptr(e.name.as_ptr()).to_string_lossy();
                let stats = pep.get_pipeline_executable_statistics(&vk::PipelineExecutableInfoKHR::default().pipeline(pipe).executable_index(i as u32)).unwrap();
                line += &format!(" || {name}:");
                for s in stats {
                    let n = CStr::from_ptr(s.name.as_ptr()).to_string_lossy();
                    let v = match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => format!("{}", s.value.b32),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => format!("{}", s.value.i64),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => format!("{}", s.value.u64),
                        _ => format!("{:.3}", s.value.f64),
                    };
                    line += &format!(" {n}={v};");
                }
            }
            println!("{line}");
            device.destroy_pipeline(pipe, None);
            device.destroy_shader_module(vm, None);
            device.destroy_shader_module(fm, None);
            device.destroy_pipeline_layout(layout, None);
            for s in sets { device.destroy_descriptor_set_layout(s, None); }
        }
    }
}
