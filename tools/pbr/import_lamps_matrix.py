"""Compare imported rough-metal finite-lamp response with matched controls.

Run the outer command through native/tools/gpu_run.py. All fixture copies, logs,
renders and reports are written below the explicitly selected phase directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

W, H, FRAMES = 360, 270, 32
FINISHES = {
    "mirror": (0.0, 1.0),
    "rough": (1.0, 1.0),
    "pigment": (1.0, 0.0),
}
RIGS = ("dark", "ball", "room")
LIMITATIONS = [
    "The finite-ball minus dark response measures radiance changes on one box fixture; it is not an independent highlight-shape qualification.",
    "The fixture uses a single finite ball, fixed camera, exposure, seed and sample count; panels, other geometries, occlusion configurations and animation are not qualified.",
    "Mirror and pigment are regression controls, not references for energy conservation or physically correct imported BRDF response.",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rgba(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        width, height = np.fromfile(f, "<u4", count=2)
        pixels = np.fromfile(f, "<f4").reshape(height, width, 4)
    if (width, height) != (W, H):
        raise ValueError(f"unexpected dimensions in {path}: {width}x{height}")
    if not np.isfinite(pixels).all():
        raise ValueError(f"non-finite pixel in {path}")
    return pixels


def prepare_fixtures(phase: Path, source: Path) -> None:
    fixtures = phase / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    for name in ("box.bin", "finish.png", "white.png"):
        shutil.copy2(source / name, fixtures / name)
    template = json.loads((source / "rough.gltf").read_text(encoding="utf-8"))
    for name, (roughness, metallic) in FINISHES.items():
        doc = dict(template)
        doc["materials"] = [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "baseColorFactor": [0.82, 0.42, 0.18, 1],
            "roughnessFactor": roughness,
            "metallicFactor": metallic,
        }}]
        (fixtures / f"{name}.gltf").write_text(json.dumps(doc), encoding="utf-8")
    provenance = {
        "source": str(source.resolve()),
        "source_files_sha256": {
            name: sha256(source / name)
            for name in ("box.bin", "finish.png", "white.png", "rough.gltf")
        },
        "fixture_sha256": {
            path.name: sha256(path)
            for path in sorted(fixtures.iterdir()) if path.is_file()
        },
    }
    (phase / "reports").mkdir(parents=True, exist_ok=True)
    (phase / "reports" / "fixtures.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def capture(exe: Path, label: str, phase: Path, native: Path) -> dict:
    fixtures = phase / "fixtures"
    captures = []
    for finish in FINISHES:
        folder = phase / "renders" / label / finish
        folder.mkdir(parents=True, exist_ok=True)
        for rig in RIGS:
            name = f"{finish}-{rig}"
            stem = folder / name
            args = [
                f"model={fixtures / f'{finish}.gltf'}",
                f"w={W}", f"h={H}", "res=128", f"frames={FRAMES}",
                "rx=0", "ry=0", "rz=0", "spin=0", "grain=0", "bloom=0",
                "floor=0", "backdrop=0", "meter=0", "exposure=1", "seed=5",
            ]
            if rig != "room":
                args += ["rig=", "ambient=0,0,0"]
            if rig == "ball":
                args += ["lamp1=ball,.95,.65,2.5,.22,0,1,1,1,16"]
            cmd = [str(exe), "--still", str(stem.with_suffix(".png")),
                   "--hdr", str(stem.with_suffix(".rgba")), *args]
            run = subprocess.run(
                cmd, cwd=native, env=dict(os.environ, GLASSFORMS_NO_WARM="1"),
                capture_output=True, text=True, timeout=300,
            )
            log = phase / "logs" / f"{label}-{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                json.dumps(cmd) + "\n\nSTDOUT\n" + run.stdout + "\nSTDERR\n" + run.stderr,
                encoding="utf-8",
            )
            if run.returncode:
                raise RuntimeError(f"{label}/{name} failed with {run.returncode}; see {log}")
            read_rgba(stem.with_suffix(".rgba"))
            result = json.loads(run.stdout.splitlines()[-1])
            stem.with_suffix(".report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            captures.append({"finish": finish, "rig": rig, "command": cmd,
                             "returncode": run.returncode, "log": str(log)})
    manifest = {
        "label": label,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(exe),
        "executable_sha256": sha256(exe),
        "width": W, "height": H, "frames": FRAMES,
        "captures": captures,
    }
    (phase / "reports" / f"capture-{label}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def response(pixels: dict[str, np.ndarray]) -> dict:
    mask = binary_erosion(pixels["dark"][:, :, 3] > 0, iterations=4)
    delta = (pixels["ball"] - pixels["dark"])[mask, :3]
    centre = (pixels["ball"] - pixels["dark"])[122:148, 166:194, :3]
    return {
        "pixels": int(mask.sum()),
        "mean_linear_rgb": delta.mean(axis=0, dtype=float).tolist(),
        "p90_linear_rgb": np.percentile(delta, 90, axis=0).tolist(),
        "central_mean_linear_rgb": centre.mean(axis=(0, 1), dtype=float).tolist(),
        "peak_linear_rgb": delta.max(axis=0).tolist(),
        "bright_red_pixels_over_0_001": int((delta[:, 0] > 0.001).sum()),
    }


def analyse(phase: Path, manifests: dict[str, dict]) -> dict:
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executables": {
            label: {"path": manifest["executable"], "sha256": manifest["executable_sha256"]}
            for label, manifest in manifests.items()
        },
        "settings": {"width": W, "height": H, "frames": FRAMES, "seed": 5, "exposure": 1},
        "responses": {},
        "baseline_vs_candidate": {},
        "limitations": LIMITATIONS,
    }
    for label in manifests:
        report["responses"][label] = {}
        for finish in FINISHES:
            pixels = {
                rig: read_rgba(phase / "renders" / label / finish / f"{finish}-{rig}.rgba")
                for rig in RIGS
            }
            report["responses"][label][finish] = response(pixels)
    for finish in FINISHES:
        report["baseline_vs_candidate"][finish] = {}
        for rig in RIGS:
            name = f"{finish}-{rig}.rgba"
            before = read_rgba(phase / "renders" / "baseline" / finish / name)
            after = read_rgba(phase / "renders" / "candidate" / finish / name)
            delta = after[:, :, :3] - before[:, :, :3]
            report["baseline_vs_candidate"][finish][rig] = {
                "max_abs_linear_rgb": float(np.abs(delta).max()),
                "linear_rgb_rms": float(np.sqrt(np.mean(delta ** 2))),
                "mean_abs_linear_rgb": float(np.abs(delta).mean()),
                "max_abs_depth": float(np.abs(after[:, :, 3] - before[:, :, 3]).max()),
            }
    (phase / "reports" / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def review_sheet(phase: Path) -> Path:
    caption = 28
    columns = ["baseline dark", "baseline ball", "candidate dark", "candidate ball",
               "candidate ball-dark x8", "baseline room", "candidate room", "candidate-baseline ball x16"]
    sheet = Image.new("RGB", (W * len(columns), 42 + (H + caption) * len(FINISHES)), (20, 23, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), "Imported glTF: matched finite-lamp response and controls; linear deltas are amplified for inspection", fill=(235, 238, 242))
    for row, finish in enumerate(FINISHES):
        before = {rig: phase / "renders" / "baseline" / finish / f"{finish}-{rig}.png" for rig in RIGS}
        after = {rig: phase / "renders" / "candidate" / finish / f"{finish}-{rig}.png" for rig in RIGS}
        before_hdr = read_rgba(phase / "renders" / "baseline" / finish / f"{finish}-ball.rgba")
        dark = read_rgba(phase / "renders" / "candidate" / finish / f"{finish}-dark.rgba")
        ball = read_rgba(phase / "renders" / "candidate" / finish / f"{finish}-ball.rgba")
        response_image = np.clip((ball[:, :, :3] - dark[:, :, :3]) * 8.0, 0, 1) ** (1 / 2.2)
        comparison = np.clip(np.abs(ball[:, :, :3] - before_hdr[:, :, :3]) * 16.0, 0, 1) ** (1 / 2.2)
        images = [Image.open(before["dark"]).convert("RGB"), Image.open(before["ball"]).convert("RGB"),
                  Image.open(after["dark"]).convert("RGB"), Image.open(after["ball"]).convert("RGB"),
                  Image.fromarray((response_image * 255).astype(np.uint8), "RGB"),
                  Image.open(before["room"]).convert("RGB"), Image.open(after["room"]).convert("RGB"),
                  Image.fromarray((comparison * 255).astype(np.uint8), "RGB")]
        y = 42 + row * (H + caption)
        for col, (label, image) in enumerate(zip(columns, images)):
            x = col * W
            draw.text((x + 8, y + 7), f"{finish} | {label}", fill=(220, 225, 232))
            sheet.paste(image, (x, y + caption))
    review = phase / "review"
    review.mkdir(parents=True, exist_ok=True)
    path = review / "imported-lamps-comparison.png"
    sheet.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-exe", type=Path, required=True)
    parser.add_argument("--candidate-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-source", type=Path, required=True)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    phase = args.output.resolve()
    native = Path(__file__).resolve().parents[2]
    (phase / "reports").mkdir(parents=True, exist_ok=True)
    invocation = {
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "cwd": os.getcwd(),
        "gpu_queue": os.environ.get("GF_GPU_QUEUE"),
        "inside_gpu_queue": os.environ.get("GLASSFORMS_GPU_QUEUED") == "1",
        "analyse_only": args.analyse_only,
    }
    (phase / "reports" / "tool-invocation.json").write_text(json.dumps(invocation, indent=2), encoding="utf-8")
    if not args.analyse_only:
        prepare_fixtures(phase, args.fixture_source.resolve())
        manifests = {}
        for label, exe in (("baseline", args.baseline_exe), ("candidate", args.candidate_exe)):
            manifests[label] = capture(exe.resolve(), label, phase, native)
    else:
        manifests = {
            label: json.loads((phase / "reports" / f"capture-{label}.json").read_text(encoding="utf-8"))
            for label in ("baseline", "candidate")
        }
    report = analyse(phase, manifests)
    review = review_sheet(phase)
    print(json.dumps({"review": str(review), "rough_candidate_response": report["responses"]["candidate"]["rough"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"import_lamps_matrix: {exc}", file=sys.stderr)
        raise
