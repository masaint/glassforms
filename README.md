# Glass Forms

Mathematical shapes rendered live as dispersive glass, metal and other materials. It's a real-time renderer written in Rust on `wgpu` (Vulkan).

## Building

You need Rust (stable) and a Vulkan-capable GPU. Development and testing are on an NVIDIA RTX 5090.

```
cargo build --release -p glassforms
```

Alternatively, run `run.bat`. It builds the program, keeps a copy as `glassforms.exe` in this folder, and starts it. Renders, stills, setups and UI settings are saved beside the program. Video output needs `ffmpeg` on the PATH.

DLSS is optional. To use it, build with `--features dlss` and supply NVIDIA's runtime DLL yourself; it isn't included here.

## Layout

| Folder | Contents |
|---|---|
| `app/` | the window, GPU passes, WGSL shaders and UI |
| `core/` | shapes, settings, materials and scene state; no GPU code |
| `models/` | sample meshes |
| `tools/` | Python scripts for stills, checks and performance gates |

## Tests

```
cargo test --release
```

The GPU tests need a Vulkan device.
