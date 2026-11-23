# Unified PBR qualification tools

## Focused light on painted assemblies

`focused_light_matrix.py` renders unfilmed and filmed satin paint beneath a separate Crown caster, with caustics off and on. It uses the established aligned fixture copied into the phase output folder; it never writes into the original checkout's `material_review` directory.

Run from any directory through the shared queue:

```powershell
$env:GF_GPU_QUEUE = 'D:\glassforms\_gpu_queue'
python native/tools/gpu_run.py --name pbr-focused-before -- python -I native/tools/pbr/focused_light_matrix.py `
  --exe D:\glassforms\native\glassforms-gpt-42a015e-dlss.exe `
  --output native/out/unified_pbr/01-focused-light --label before
```

Then use the new executable and `--label after --compare-to before`.

Each invocation records the executable SHA-256, exact commands, full stdout/stderr logs, saved resolved setups, linear RGBA/depth, PNGs, and numerical analysis. A paired after-run also creates `native/out/unified_pbr/review/focused-light-comparison.png`.

The depth-range receiver mask follows the original focused-light fixture analysis. The images remain necessary: a non-zero pixel count alone does not establish correct caustic shape, placement, colour, or absence of double counting.

## Cut ownership

`cut_owner_matrix.py` captures saved, restored, recipient and inherited states, then compares saved/restored and recipient/inherited linear RGBA/depth exactly. Use the same `--exe`, `--output`, `--label` and `--compare-to` pattern, targeting `native/out/unified_pbr/04-cut-owner`.

Complete the baseline before launching the paired after-run: ordinary queue jobs may overlap, and the comparison needs all four baseline files. A paired run creates `review/cut-owner-comparison.png`.
