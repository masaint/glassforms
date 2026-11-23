"""Fixed-frame imported-surface comparison for baseline-a/candidate/baseline-b.

Queue the outer capture command through native/tools/gpu_run.py. Each label has an
explicit executable and may be captured independently. Raw PNG/float RGBA, exact
commands, per-command logs, fixture provenance and renderer reports are retained.
The consolidated contact sheet is produced only after all three labels exist.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

LABELS = ("baseline-a", "candidate", "baseline-b")
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FRAMES = 360, 270, 32
SOURCE_FILES = ("rough.gltf", "box.bin", "finish.png", "white.png")
FINISHES = {
    "mirror": (0.0, 1.0),
    "rough": (1.0, 1.0),
    "dielectric": (0.82, 0.0),
}
SCENE_NAMES = (
    "curved-pawn-oblique",
    "faceted-recess-oblique",
    "factor-only-faceted",
    "continuous-mix",
    "normal-detail",
    "height-detail",
    "nested-under-glass",
    "secondary-reflection",
    "secondary-transmission",
    "assembly-frost",
    "assembly-walk",
    "lamp-mirror-dark",
    "lamp-mirror-focused",
    "lamp-rough-dark",
    "lamp-rough-focused",
    "lamp-dielectric-dark",
    "lamp-dielectric-focused",
)
COVERAGE = {
    "curved-pawn-oblique": "Shipped imported curved pawn at an oblique fixed camera",
    "faceted-recess-oblique": "Generated closed faceted part with a recessed pocket",
    "factor-only-faceted": "The same closed faceted part with only canonical constant material inputs",
    "continuous-mix": "Textured continuous perceptual-roughness and metallic coverage ramp",
    "normal-detail": "glTF tangent-space normal-map detail with authored tangents",
    "height-detail": "OBJ height-map derivative detail using the same geometry and base colour",
    "nested-under-glass": "Small imported textured part nested apart inside a glass sphere",
    "secondary-reflection": "Imported textured part beside a separate smooth silver reflector",
    "secondary-transmission": "Imported textured part behind a separate transmissive glass sphere",
    "assembly-frost": "Imported part in an apart assembly with sampled rough glass",
    "assembly-walk": "Imported part in an apart assembly with random-walk wax",
    "lamp-mirror-dark": "Mirror imported box without finite lamps",
    "lamp-mirror-focused": "Mirror imported box with focused finite ball and tube lamps",
    "lamp-rough-dark": "Rough-metal imported box without finite lamps",
    "lamp-rough-focused": "Rough-metal imported box with focused finite ball and tube lamps",
    "lamp-dielectric-dark": "Rough-dielectric imported box without finite lamps",
    "lamp-dielectric-focused": "Rough-dielectric imported box with focused finite ball and tube lamps",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_rgba(path: Path, width: int, height: int) -> np.ndarray:
    with path.open("rb") as f:
        dimensions = np.fromfile(f, "<u4", count=2)
        if dimensions.size != 2:
            raise ValueError(f"missing RGBA dimensions in {path}")
        actual_width, actual_height = map(int, dimensions)
        pixels = np.fromfile(f, "<f4")
    if (actual_width, actual_height) != (width, height):
        raise ValueError(f"unexpected dimensions in {path}: {actual_width}x{actual_height}")
    expected = width * height * 4
    if pixels.size != expected:
        raise ValueError(f"unexpected float count in {path}: {pixels.size}, expected {expected}")
    pixels = pixels.reshape(height, width, 4)
    if not np.isfinite(pixels).all():
        raise ValueError(f"non-finite pixel in {path}")
    return pixels


def add_quad(positions: list, uvs: list, tangents: list, points: list, uv: list) -> None:
    """Append two independent triangles with a consistent authored tangent frame."""
    a, b, c = (np.asarray(points[i], dtype=float) for i in (0, 1, 2))
    tangent = b - a
    tangent /= np.linalg.norm(tangent)
    for index in (0, 1, 2, 0, 2, 3):
        positions.append(tuple(float(x) for x in points[index]))
        uvs.append(tuple(float(x) for x in uv[index]))
        tangents.append((*[float(x) for x in tangent], 1.0))


def write_gltf(path: Path, positions: list, uvs: list, tangents: list, material: dict,
               images: list | None = None, textures: list | None = None) -> None:
    arrays = (positions, uvs, tangents)
    types = ("VEC3", "VEC2", "VEC4")
    blob = bytearray()
    views, accessors = [], []
    for values, kind in zip(arrays, types):
        offset = len(blob)
        flat = [component for value in values for component in value]
        blob.extend(struct.pack("<" + "f" * len(flat), *flat))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob) - offset})
        accessors.append({
            "bufferView": len(views) - 1, "componentType": 5126,
            "type": kind, "count": len(values),
        })
    binary = path.with_suffix(".bin")
    binary.write_bytes(blob)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": 0, "TEXCOORD_0": 1, "TANGENT": 2,
        }, "material": 0}]}],
        "buffers": [{"uri": binary.name, "byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
        "materials": [material],
    }
    if images is not None:
        doc["images"] = [{"uri": name} for name in images]
        doc["samplers"] = [{"wrapS": 10497, "wrapT": 10497}]
        doc["textures"] = textures or [{"source": i, "sampler": 0} for i in range(len(images))]
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def box_arrays(source: Path) -> tuple[list, list]:
    data = (source / "box.bin").read_bytes()
    positions = np.frombuffer(data[:432], dtype="<f4").reshape(-1, 3)
    uvs = np.frombuffer(data[432:], dtype="<f4").reshape(-1, 2)
    return positions.tolist(), uvs.tolist()


def box_tangents(positions: list, uvs: list) -> list:
    result = []
    for start in range(0, len(positions), 3):
        p = [np.asarray(positions[start + i], dtype=float) for i in range(3)]
        uv = [np.asarray(uvs[start + i], dtype=float) for i in range(3)]
        e1, e2 = p[1] - p[0], p[2] - p[0]
        du, dv = uv[1] - uv[0], uv[2] - uv[0]
        determinant = du[0] * dv[1] - du[1] * dv[0]
        tangent = e1 if abs(determinant) < 1e-9 else (e1 * dv[1] - e2 * du[1]) / determinant
        length = np.linalg.norm(tangent)
        tangent = np.asarray((1.0, 0.0, 0.0)) if length < 1e-9 else tangent / length
        result.extend([(*tangent.tolist(), 1.0)] * 3)
    return result


def make_textures(fixtures: Path) -> None:
    size = 128
    y, x = np.mgrid[0:size, 0:size]
    u, v = x / (size - 1), y / (size - 1)
    base = np.empty((size, size, 4), dtype=np.uint8)
    base[:, :, 0] = np.rint(45 + 170 * u).astype(np.uint8)
    base[:, :, 1] = np.rint(170 - 70 * u + 25 * np.sin(2 * np.pi * v)).astype(np.uint8)
    base[:, :, 2] = np.rint(205 - 120 * u).astype(np.uint8)
    base[:, :, 3] = 255
    Image.fromarray(base, "RGBA").save(fixtures / "continuous-base.png")
    finish = np.full((size, size, 4), 255, dtype=np.uint8)
    finish[:, :, 1] = np.rint(255 * (0.10 + 0.72 * (0.5 + 0.5 * np.sin(2 * np.pi * u)))).astype(np.uint8)
    finish[:, :, 2] = np.rint(255 * u).astype(np.uint8)
    Image.fromarray(finish, "RGBA").save(fixtures / "continuous-finish.png")
    wave = 0.5 + 0.24 * np.sin(6 * np.pi * u) * np.cos(4 * np.pi * v)
    height = np.rint(np.clip(wave, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(height, "L").save(fixtures / "detail-height.png")
    dx = 0.24 * 6 * np.pi * np.cos(6 * np.pi * u) * np.cos(4 * np.pi * v) / size
    dy = -0.24 * 4 * np.pi * np.sin(6 * np.pi * u) * np.sin(4 * np.pi * v) / size
    normal = np.stack((-0.7 * dx, -0.7 * dy, np.ones_like(dx)), axis=2)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    Image.fromarray(np.rint(np.clip(normal * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8), "RGB").save(
        fixtures / "detail-normal.png")


def prepare_fixtures(phase: Path, source: Path, pawn_source: Path) -> dict:
    fixtures = phase / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(source / name, fixtures / name)
    shutil.copy2(pawn_source, fixtures / "pawn.glb")
    make_textures(fixtures)
    positions, uvs = box_arrays(source)
    tangents = box_tangents(positions, uvs)

    for name, (roughness, metallic) in FINISHES.items():
        doc = json.loads((source / "rough.gltf").read_text(encoding="utf-8"))
        doc["materials"] = [{"name": name, "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "baseColorFactor": [0.82, 0.42, 0.18, 1],
            "roughnessFactor": roughness,
            "metallicFactor": metallic,
        }}]
        (fixtures / f"lamp-{name}.gltf").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    write_gltf(fixtures / "continuous.gltf", positions, uvs, tangents, {
        "name": "continuous metal dielectric",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 1},
            "roughnessFactor": 1.0, "metallicFactor": 1.0,
        },
    }, ["continuous-base.png", "continuous-finish.png"])
    write_gltf(fixtures / "normal.gltf", positions, uvs, tangents, {
        "name": "normal detail",
        "normalTexture": {"index": 1, "scale": 2.5},
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "roughnessFactor": 0.38, "metallicFactor": 0.25,
        },
    }, ["continuous-base.png", "detail-normal.png"])

    with (fixtures / "height.obj").open("w", encoding="utf-8", newline="\n") as f:
        f.write("mtllib height.mtl\n")
        for p in positions:
            f.write("v %.9g %.9g %.9g\n" % tuple(p))
        for uv in uvs:
            f.write("vt %.9g %.9g\n" % (uv[0], 1.0 - uv[1]))
        f.write("usemtl height\n")
        for i in range(0, len(positions), 3):
            f.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(i + 1, i + 2, i + 3))
    (fixtures / "height.mtl").write_text(
        "newmtl height\nKd 0.65 0.72 0.82\nPr 0.38\nPm 0.25\n"
        "map_Kd continuous-base.png\nmap_Bump -bm 45 detail-height.png\n", encoding="utf-8")

    pocket_positions, pocket_uvs, pocket_tangents = [], [], []
    square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    inner = [(-0.48, -0.48), (0.48, -0.48), (0.48, 0.48), (-0.48, 0.48)]
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i in range(4):
        j = (i + 1) % 4
        # Front annulus, pocket walls and floor, all explicitly faceted.
        add_quad(pocket_positions, pocket_uvs, pocket_tangents,
                 [(*square[i], 1), (*square[j], 1), (*inner[j], 1), (*inner[i], 1)], uv)
        add_quad(pocket_positions, pocket_uvs, pocket_tangents,
                 [(*inner[i], 1), (*inner[j], 1), (*inner[j], 0.55), (*inner[i], 0.55)], uv)
        add_quad(pocket_positions, pocket_uvs, pocket_tangents,
                 [(*square[j], -1), (*square[i], -1), (*square[i], 1), (*square[j], 1)], uv)
        add_quad(pocket_positions, pocket_uvs, pocket_tangents,
                 [(*inner[j], 0.55), (*inner[i], 0.55),
                  (inner[i][0], inner[i][1], 0.55), (inner[j][0], inner[j][1], 0.55)], uv)
    # Replace the degenerate pocket-floor quads above with one nontrivial floor.
    pocket_positions = pocket_positions[:-4 * 6]
    pocket_uvs = pocket_uvs[:-4 * 6]
    pocket_tangents = pocket_tangents[:-4 * 6]
    add_quad(pocket_positions, pocket_uvs, pocket_tangents,
             [(*inner[0], 0.55), (*inner[1], 0.55), (*inner[2], 0.55), (*inner[3], 0.55)], uv)
    add_quad(pocket_positions, pocket_uvs, pocket_tangents,
             [(*square[3], -1), (*square[2], -1), (*square[1], -1), (*square[0], -1)], uv)
    write_gltf(fixtures / "faceted-recess.gltf", pocket_positions, pocket_uvs, pocket_tangents, {
        "name": "faceted recessed ochre metal",
        "normalTexture": {"index": 1, "scale": 0.65},
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.78, 0.36, 0.10, 1],
            "metallicRoughnessTexture": {"index": 0},
            "roughnessFactor": 0.9, "metallicFactor": 1.0,
        },
    }, ["continuous-finish.png", "detail-normal.png"])
    write_gltf(fixtures / "factor-only.gltf", pocket_positions, pocket_uvs, pocket_tangents, {
        "name": "factor-only blue metal",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.16, 0.42, 0.78, 1],
            "roughnessFactor": 0.29, "metallicFactor": 0.37,
        },
    })

    provenance = {
        "source": str(source.resolve()),
        "source_files_sha256": {name: sha256(source / name) for name in SOURCE_FILES},
        "pawn_source": str(pawn_source.resolve()),
        "pawn_source_sha256": sha256(pawn_source),
        "fixture_sha256": {path.name: sha256(path) for path in sorted(fixtures.iterdir()) if path.is_file()},
        "generation": "Deterministic local geometry/textures; canonical import readers are not modified",
    }
    write_json(phase / "reports" / "fixtures.json", provenance)
    return provenance


def scene_command_args(phase: Path, name: str) -> list[str]:
    fixtures = phase / "fixtures"
    common = [
        "rx=20", "ry=-30", "rz=6", "spin=0", "grain=0", "bloom=0",
        "floor=0", "backdrop=0", "meter=0", "exposure=1", "res=128",
    ]
    if name == "curved-pawn-oblique":
        return [f"model={fixtures / 'pawn.glb'}", *common]
    if name == "faceted-recess-oblique":
        return [f"model={fixtures / 'faceted-recess.gltf'}", *common]
    if name == "factor-only-faceted":
        return [f"model={fixtures / 'factor-only.gltf'}", *common]
    if name == "continuous-mix":
        return [f"model={fixtures / 'continuous.gltf'}", *common]
    if name == "normal-detail":
        return [f"model={fixtures / 'normal.gltf'}", *common, "dist=4.2"]
    if name == "height-detail":
        return [f"model={fixtures / 'height.obj'}", *common, "dist=4.2"]
    if name == "nested-under-glass":
        return [f"model={fixtures / 'continuous.gltf'}", "L1.size=0.52",
                "layer=studio:sphere", "L2.size=1.05", "J12.op=apart",
                "B2.material=glass", "B2.ior=1.5", "B2.rough=0.025", "B2.disp=0", "B2.density=0", *common]
    if name == "secondary-reflection":
        return [f"model={fixtures / 'normal.gltf'}", "L1.size=0.65", "L1.x=-0.65",
                "layer=studio:sphere", "L2.size=0.62", "L2.x=0.75", "J12.op=apart",
                "B2.material=silver", "B2.rough=0.035", *common]
    if name == "secondary-transmission":
        return [f"model={fixtures / 'continuous.gltf'}", "L1.size=0.62", "L1.x=-0.35", "L1.z=-0.55",
                "layer=studio:sphere", "L2.size=0.62", "L2.x=0.35", "L2.z=0.5", "J12.op=apart",
                "B2.material=glass", "B2.rough=0.04", "B2.disp=0", "B2.density=0", *common]
    if name == "assembly-frost":
        return [f"model={fixtures / 'faceted-recess.gltf'}", "L1.size=0.68", "L1.x=-0.5",
                "layer=algebraic:heart", "L2.size=0.62", "L2.x=0.65", "J12.op=apart",
                "B2.material=glass", "B2.rough=0.32", "B2.roughRef=1", "B2.disp=0", *common]
    if name == "assembly-walk":
        return [f"model={fixtures / 'continuous.gltf'}", "L1.size=0.68", "L1.x=-0.5",
                "layer=algebraic:heart", "L2.size=0.62", "L2.x=0.65", "J12.op=apart",
                "B2.material=wax", "B2.sssWalk=1", *common]
    for finish in FINISHES:
        dark, focused = f"lamp-{finish}-dark", f"lamp-{finish}-focused"
        if name in (dark, focused):
            args = [f"model={fixtures / f'lamp-{finish}.gltf'}", *common,
                    "rx=0", "ry=0", "rz=0", "rig=", "ambient=0,0,0"]
            if name == focused:
                args += ["lamp1=ball,.95,.65,2.5,.18,0,1,1,1,24",
                         "lamp2=tube,-1.2,.3,1.8,.06,.65,.75,.9,1,18"]
            return args
    raise KeyError(name)


def image_metrics(pixels: np.ndarray) -> dict:
    rgb, alpha = pixels[:, :, :3], pixels[:, :, 3]
    return {
        "finite": bool(np.isfinite(pixels).all()),
        "mean_linear_rgb": rgb.mean(axis=(0, 1), dtype=float).tolist(),
        "min_linear_rgb": rgb.min(axis=(0, 1)).tolist(),
        "max_linear_rgb": rgb.max(axis=(0, 1)).tolist(),
        "linear_rgb_rms": float(np.sqrt(np.mean(rgb.astype(np.float64) ** 2))),
        "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
        "alpha_mean": float(alpha.mean(dtype=float)),
        "nonzero_alpha_pixels": int(np.count_nonzero(alpha > 0)),
    }


def validate_renderer_report(report: dict, seed: int, frames: int) -> None:
    """The explicit pre-seed path must perform exactly one final still render."""
    if report.get("frames") != frames or report.get("preseed") != seed:
        raise ValueError(f"renderer did not honor preseed={seed} and frames={frames}: {report}")
    if report.get("renderPasses") != 1:
        raise ValueError(f"expected one still render pass, got {report.get('renderPasses')}: {report}")


def validate_comparison_settings(a: dict, b: dict) -> dict:
    for key in ("width", "height", "frames", "fixed_seed", "adaptive", "render_mode"):
        if a.get(key) != b.get(key):
            raise ValueError(f"capture {key} differs for {a['label']} and {b['label']}: {a.get(key)} / {b.get(key)}")
    if a.get("fixtures_sha256") != b.get("fixtures_sha256"):
        raise ValueError(f"capture fixtures differ for {a['label']} and {b['label']}")
    if a["render_mode"] != "preseed-single-render":
        raise ValueError("cross-label comparisons require the explicit single-render preseed mode")
    return {
        "fixture_hashes_identical": True,
        "width": a["width"], "height": a["height"], "frames": a["frames"],
        "fixed_seed": a["fixed_seed"], "adaptive": a["adaptive"],
        "single_preseeded_render": a["render_mode"] == "preseed-single-render",
    }


def capture_label(executable: Path, label: str, phase: Path, native: Path,
                  width: int, height: int, frames: int) -> dict:
    manifest = {
        "label": label, "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(executable), "executable_sha256": sha256(executable),
        "width": width, "height": height, "frames": frames,
        "fixed_seed": 5, "adaptive": False, "timing_verdict": False,
        "render_mode": "preseed-single-render",
        "fixtures_sha256": {path.name: sha256(path) for path in sorted((phase / "fixtures").iterdir()) if path.is_file()},
        "captures": [],
    }
    manifest_path = phase / "reports" / f"capture-{label}.json"
    write_json(manifest_path, manifest)
    try:
        for name in SCENE_NAMES:
            folder = phase / "renders" / label / name
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / name
            png, rgba = stem.with_suffix(".png"), stem.with_suffix(".rgba")
            report_path = stem.with_suffix(".report.json")
            quarantined = []
            for old in (png, rgba, report_path):
                if old.exists():
                    destination = phase / "superseded" / manifest["started_utc"].replace(":", "-") / label / name / old.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old), str(destination))
                    quarantined.append(str(destination))
            log = phase / "logs" / label / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(executable), "--still", str(png), "--hdr", str(rgba),
                f"w={width}", f"h={height}", f"frames={frames}", "adapt=0", "preseed=5",
                *scene_command_args(phase, name),
            ]
            item = {
                "scene": name, "coverage": COVERAGE[name], "command": command,
                "cwd": str(native), "environment": {"GLASSFORMS_NO_WARM": "1"},
                "png": str(png), "rgba": str(rgba), "log": str(log),
                "quarantined_outputs": quarantined,
            }
            try:
                run = subprocess.run(command, cwd=native, env=dict(os.environ, GLASSFORMS_NO_WARM="1"),
                                     capture_output=True, text=True, timeout=600)
                stdout, stderr, returncode = run.stdout, run.stderr, run.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                returncode = None
                item["timeout_seconds"] = 600
            log.write_text(json.dumps(command) + "\n\nSTDOUT\n" + stdout + "\nSTDERR\n" + stderr,
                           encoding="utf-8")
            item["returncode"] = returncode
            manifest["captures"].append(item)
            if returncode != 0:
                raise RuntimeError(f"{label}/{name} failed with return code {returncode}; see {log}")
            if not png.is_file() or not rgba.is_file():
                raise RuntimeError(f"{label}/{name} did not produce both PNG and RGBA")
            pixels = read_rgba(rgba, width, height)
            with Image.open(png) as image:
                if image.size != (width, height):
                    raise ValueError(f"unexpected PNG dimensions for {label}/{name}: {image.size}")
            lines = [line for line in stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(f"{label}/{name} returned no renderer report")
            renderer_report = json.loads(lines[-1])
            validate_renderer_report(renderer_report, manifest["fixed_seed"], frames)
            write_json(report_path, renderer_report)
            item.update({
                "renderer_report": str(report_path),
                "png_sha256": sha256(png), "rgba_sha256": sha256(rgba),
                "metrics": image_metrics(pixels),
            })
            write_json(manifest_path, manifest)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = str(exc)
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        raise


def load_manifests(phase: Path, selected: list[str] | None = None) -> dict[str, dict]:
    labels = selected or list(LABELS)
    manifests = {}
    for label in labels:
        path = phase / "reports" / f"capture-{label}.json"
        if not path.is_file():
            if selected:
                raise FileNotFoundError(path)
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(f"capture manifest {path} is not complete")
        manifests[label] = manifest
    if not manifests:
        raise RuntimeError(f"no complete selected capture manifests in {phase / 'reports'}")
    return manifests


def analyse(phase: Path, manifests: dict[str, dict]) -> dict:
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed-frame image comparisons only; no timing verdicts",
        "labels": {},
        "comparisons": {},
        "scenes": {name: COVERAGE[name] for name in SCENE_NAMES},
        "cross_label_validation": {},
    }
    for label, manifest in manifests.items():
        width, height = manifest["width"], manifest["height"]
        result["labels"][label] = {
            "executable": manifest["executable"], "executable_sha256": manifest["executable_sha256"],
            "settings": {"width": width, "height": height, "frames": manifest["frames"],
                         "seed": manifest["fixed_seed"], "adapt": 0},
            "metrics": {},
        }
        for item in manifest["captures"]:
            pixels = read_rgba(Path(item["rgba"]), width, height)
            result["labels"][label]["metrics"][item["scene"]] = image_metrics(pixels)
        result["labels"][label]["focused_lamp_response"] = {}
        for finish in FINISHES:
            dark = read_rgba(phase / "renders" / label / f"lamp-{finish}-dark" / f"lamp-{finish}-dark.rgba",
                             width, height)
            lit = read_rgba(phase / "renders" / label / f"lamp-{finish}-focused" / f"lamp-{finish}-focused.rgba",
                            width, height)
            mask = (dark[:, :, 3] > 0) & (lit[:, :, 3] > 0)
            delta = (lit[:, :, :3] - dark[:, :, :3]).astype(np.float64)
            luminance = delta @ np.array([0.2126, 0.7152, 0.0722])
            result["labels"][label]["focused_lamp_response"][finish] = {
                "object_pixels": int(mask.sum()),
                "mean_linear_rgb_gain": delta[mask].mean(axis=0).tolist(),
                "p90_linear_rgb_gain": np.percentile(delta[mask], 90, axis=0).tolist(),
                "peak_linear_rgb_gain": delta[mask].max(axis=0).tolist(),
                "mean_luminance_gain": float(luminance[mask].mean()),
                "positive_luminance_pixels_over_0_001": int(np.count_nonzero(mask & (luminance > 0.001))),
            }
    for left, right in itertools.combinations(LABELS, 2):
        if left not in manifests or right not in manifests:
            continue
        a_manifest, b_manifest = manifests[left], manifests[right]
        if (a_manifest["width"], a_manifest["height"], a_manifest["frames"]) != (
                b_manifest["width"], b_manifest["height"], b_manifest["frames"]):
            raise ValueError(f"capture settings differ for {left} and {right}")
        key = f"{left}__{right}"
        result["cross_label_validation"][key] = validate_comparison_settings(a_manifest, b_manifest)
        result["comparisons"][key] = {}
        for name in SCENE_NAMES:
            a = read_rgba(phase / "renders" / left / name / f"{name}.rgba",
                          a_manifest["width"], a_manifest["height"])
            b = read_rgba(phase / "renders" / right / name / f"{name}.rgba",
                          b_manifest["width"], b_manifest["height"])
            delta = b[:, :, :3].astype(np.float64) - a[:, :, :3].astype(np.float64)
            depth = b[:, :, 3].astype(np.float64) - a[:, :, 3].astype(np.float64)
            object_pixels = (a[:, :, 3] > 0) & (b[:, :, 3] > 0)
            nonzero_to_zero = object_pixels & np.any(a[:, :, :3] != 0, axis=2) & np.all(b[:, :, :3] == 0, axis=2)
            result["comparisons"][key][name] = {
                "nonzero_to_zero_rgb_pixels": int(np.count_nonzero(nonzero_to_zero)),
                "mean_abs_linear_rgb": float(np.abs(delta).mean()),
                "linear_rgb_rms": float(np.sqrt(np.mean(delta ** 2))),
                "max_abs_linear_rgb": float(np.abs(delta).max()),
                "different_rgb_pixels_over_1e-6": int(np.count_nonzero(np.any(np.abs(delta) > 1e-6, axis=2))),
                "mean_abs_alpha_or_depth": float(np.abs(depth).mean()),
                "max_abs_alpha_or_depth": float(np.abs(depth).max()),
                "rgba_identical": bool(np.array_equal(a, b)),
            }
    write_json(phase / "reports" / "analysis.json", result)
    return result


def contact_sheet(phase: Path, manifests: dict[str, dict]) -> Path:
    missing = [label for label in LABELS if label not in manifests]
    if missing:
        raise RuntimeError(f"consolidated contact sheet requires all labels; missing {', '.join(missing)}")
    settings = {(m["width"], m["height"], m["frames"]) for m in manifests.values()}
    if len(settings) != 1:
        raise ValueError("all three labels need matched width, height and frames for the contact sheet")
    width, height, frames = next(iter(settings))
    caption, header = 28, 42
    sheet = Image.new("RGB", (width * 3, header + (height + caption) * len(SCENE_NAMES)), (20, 23, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"Imported surfaces | fixed {width}x{height}, {frames} frames | baseline-a / candidate / baseline-b",
              fill=(235, 238, 242))
    for row, name in enumerate(SCENE_NAMES):
        y = header + row * (height + caption)
        for column, label in enumerate(LABELS):
            path = phase / "renders" / label / name / f"{name}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            x = column * width
            draw.text((x + 7, y + 7), f"{name} | {label}", fill=(220, 225, 232))
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB"), (x, y + caption))
    review = phase / "review"
    review.mkdir(parents=True, exist_ok=True)
    path = review / "import-surface-comparison.png"
    sheet.save(path)
    write_json(phase / "reports" / "contact-sheet.json", {
        "path": str(path), "sha256": sha256(path), "labels": list(LABELS),
        "scenes": list(SCENE_NAMES), "width": width, "height": height, "frames": frames,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    return path


def selected_labels(selected: list[str] | None) -> list[str]:
    """Keep an explicit selection narrow; a capture invocation defaults to baseline A only."""
    return [label for label in LABELS if label in (selected or ["baseline-a"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-a-exe", type=Path)
    parser.add_argument("--candidate-exe", type=Path)
    parser.add_argument("--baseline-b-exe", type=Path)
    parser.add_argument("--labels", nargs="+", choices=LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-source", type=Path)
    parser.add_argument("--pawn-source", type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--contact-sheet-only", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare_only, args.analyse_only, args.contact_sheet_only)) > 1:
        parser.error("choose only one of prepare-only, analyse-only and contact-sheet-only")
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        parser.error("width, height and frames must be positive")
    phase = args.output.resolve()
    native = Path(__file__).resolve().parents[2]
    (phase / "reports").mkdir(parents=True, exist_ok=True)
    invocation = {
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "cwd": os.getcwd(), "gpu_queue": os.environ.get("GF_GPU_QUEUE"),
        "inside_gpu_queue": os.environ.get("GLASSFORMS_GPU_QUEUED") == "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(phase / "reports" / "tool-invocation.json", invocation)

    if args.analyse_only or args.contact_sheet_only:
        selected = args.labels
        manifests = load_manifests(phase, selected)
        report = analyse(phase, manifests)
        review = contact_sheet(phase, manifests) if args.contact_sheet_only or set(manifests) == set(LABELS) else None
        print(json.dumps({"analysis": str(phase / "reports" / "analysis.json"),
                          "contact_sheet": str(review) if review else None,
                          "labels": list(manifests)}, indent=2))
        return

    if not args.fixture_source:
        parser.error("--fixture-source is required when preparing or capturing")
    provenance = prepare_fixtures(
        phase, args.fixture_source.resolve(strict=True),
        (args.pawn_source or native / "models" / "pawn.glb").resolve(strict=True))
    if args.prepare_only:
        print(json.dumps({"fixtures": str(phase / "reports" / "fixtures.json"),
                          "fixture_count": len(provenance["fixture_sha256"])}, indent=2))
        return
    labels = selected_labels(args.labels)
    if not labels:
        parser.error("select at least one label")
    for label in labels:
        executable = getattr(args, label.replace("-", "_") + "_exe")
        if executable is None:
            parser.error(f"--{label}-exe is required when capturing {label}")
        capture_label(executable.resolve(strict=True), label, phase, native, args.width, args.height, args.frames)
    manifests = load_manifests(phase)
    report = analyse(phase, manifests)
    review = contact_sheet(phase, manifests) if set(manifests) == set(LABELS) else None
    print(json.dumps({"captured": labels, "analysis": str(phase / "reports" / "analysis.json"),
                      "contact_sheet": str(review) if review else None,
                      "contact_sheet_pending_labels": [] if review else [x for x in LABELS if x not in manifests]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"import_surface_matrix: {exc}", file=sys.stderr)
        raise
