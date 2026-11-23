"""Render and analyse focused Crown light on painted assembly receivers.

Run inside native/tools/gpu_run.py. The executable and all outputs are explicit;
there are no writes to the original checkout or the shared material_review folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

W, H, FRAMES = 384, 300, 64
FILMS = {"unfilmed": 0.0, "filmed": 385.0}


def read_hdr(path: Path) -> np.ndarray:
    return np.fromfile(path, "<f4", offset=8).reshape(H, W, 4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture(args: argparse.Namespace) -> None:
    phase = args.output.resolve()
    native = Path(__file__).resolve().parents[2]
    fixture = phase / "fixtures" / "crown-focused.setup.json"
    exe = args.exe.resolve()
    variants = []
    for film_name, film in FILMS.items():
        for enabled in (0, 1):
            name = f"{film_name}-caustics-{enabled}"
            folder = phase / "renders" / args.label / film_name
            folder.mkdir(parents=True, exist_ok=True)
            log_dir = phase / "logs"
            report_dir = phase / "reports"
            log_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)
            stem = folder / name
            cmd = [
                str(exe), "--setup", str(fixture),
                "material=satinpaint", f"film={film}", "filmIor=1.33",
                "filmVar=0", "filmScale=1", "filmFlow=0", "filmDrain=0", "filmSky=0",
                "cast=2", "castShadows=0", f"castCaustics={enabled}",
                f"w={W}", f"h={H}", f"frames={FRAMES}", "grain=0",
                "--save-setup", str(stem.with_suffix(".json")),
                "--hdr", str(stem.with_suffix(".rgba")),
                "--still", str(stem.with_suffix(".png")),
            ]
            run = subprocess.run(cmd, cwd=native, capture_output=True, text=True, timeout=300)
            (log_dir / f"{args.label}-{name}.log").write_text(
                "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + run.stdout + "\nSTDERR\n" + run.stderr,
                encoding="utf-8",
            )
            if run.returncode:
                raise RuntimeError(f"{name} failed with {run.returncode}; see {log_dir}")
            report = json.loads(run.stdout.splitlines()[-1])
            stem.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            variants.append({"name": name, "film": film_name, "caustics": enabled,
                             "png": str(stem.with_suffix(".png")), "hdr": str(stem.with_suffix(".rgba")),
                             "command": cmd})
    manifest = {
        "label": args.label,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(exe),
        "executable_sha256": sha256(exe),
        "fixture": str(fixture),
        "width": W, "height": H, "frames": FRAMES,
        "variants": variants,
    }
    (phase / "reports" / f"capture-{args.label}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def analyse(args: argparse.Namespace) -> dict:
    phase = args.output.resolve()
    result = {"label": args.label, "films": {}}
    for film in FILMS:
        folder = phase / "renders" / args.label / film
        off = read_hdr(folder / f"{film}-caustics-0.rgba")
        on = read_hdr(folder / f"{film}-caustics-1.rgba")
        # The block front behind the foremost sphere, matching the established fixture analysis.
        mask = binary_erosion((on[:, :, 3] > 4.78) & (on[:, :, 3] < 5.06), iterations=3)
        delta = (on - off)[:, :, :3]
        lum = delta @ np.array([0.2126, 0.7152, 0.0722])
        gained = mask & (lum > 0.002)
        result["films"][film] = {
            "receiver_pixels": int(mask.sum()),
            "gained_pixels": int(gained.sum()),
            "mean_linear_rgb_gain": delta[mask].mean(axis=0).tolist(),
            "peak_luminance_gain": float(lum[mask].max()),
            "mean_luminance_gain": float(lum[mask].mean()),
            "max_abs_depth_change": float(np.abs((on - off)[:, :, 3]).max()),
            "off_on_rms": float(np.sqrt(np.mean(delta[mask] ** 2))),
        }
    if args.compare_to:
        before = args.compare_to
        result["comparison_to"] = before
        result["controls"] = {}
        for film in FILMS:
            for enabled in (0, 1):
                name = f"{film}-caustics-{enabled}.rgba"
                old = read_hdr(phase / "renders" / before / film / name)
                new = read_hdr(phase / "renders" / args.label / film / name)
                result["controls"][f"{film}-{enabled}"] = {
                    "max_abs_rgba_change": float(np.abs(new - old).max()),
                    "linear_rgb_rms_change": float(np.sqrt(np.mean((new[:, :, :3] - old[:, :, :3]) ** 2))),
                    "max_abs_depth_change": float(np.abs(new[:, :, 3] - old[:, :, 3]).max()),
                }
    (phase / "reports" / f"analysis-{args.label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def review_sheet(args: argparse.Namespace) -> None:
    if not args.compare_to:
        return
    phase = args.output.resolve()
    labels = ["before: caustics off", "before: caustics on", "after: caustics off", "after: caustics on", "after focused gain x8"]
    tile_w, tile_h, caption = W, H, 30
    sheet = Image.new("RGB", (tile_w * len(labels), (tile_h + caption) * len(FILMS) + 42), (20, 23, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), "Focused Crown light on satin paint assembly; matched camera, exposure, fixture and samples", fill=(235, 238, 242))
    for row, film in enumerate(FILMS):
        before0 = read_hdr(phase / "renders" / args.compare_to / film / f"{film}-caustics-0.rgba")
        paths = [
            phase / "renders" / args.compare_to / film / f"{film}-caustics-0.png",
            phase / "renders" / args.compare_to / film / f"{film}-caustics-1.png",
            phase / "renders" / args.label / film / f"{film}-caustics-0.png",
            phase / "renders" / args.label / film / f"{film}-caustics-1.png",
        ]
        after0 = read_hdr(phase / "renders" / args.label / film / f"{film}-caustics-0.rgba")
        after1 = read_hdr(phase / "renders" / args.label / film / f"{film}-caustics-1.rgba")
        delta = np.clip(np.abs(after1[:, :, :3] - after0[:, :, :3]) * 8.0, 0, 1) ** (1 / 2.2)
        images = [Image.open(p).convert("RGB") for p in paths]
        images.append(Image.fromarray((delta * 255).astype(np.uint8), "RGB"))
        for col, (label, image) in enumerate(zip(labels, images)):
            x, y = col * tile_w, 42 + row * (tile_h + caption)
            draw.text((x + 8, y + 8), f"{film} | {label}", fill=(220, 225, 232))
            sheet.paste(image, (x, y + caption))
    review = phase.parent / "review"
    review.mkdir(parents=True, exist_ok=True)
    sheet.save(review / "focused-light-comparison.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", choices=["before", "after"], required=True)
    p.add_argument("--compare-to", choices=["before"])
    p.add_argument("--analyse-only", action="store_true")
    args = p.parse_args()
    if not args.analyse_only:
        capture(args)
    result = analyse(args)
    review_sheet(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"focused_light_matrix: {exc}", file=sys.stderr)
        raise
