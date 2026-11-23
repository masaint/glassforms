"""Baseline/candidate/baseline image checks for shared forward-caustic dielectric optics.

Queue the outer invocation through tools/gpu_run.py. The established focused-Crown fixture supplies
real diffuse receivers. Both caustics-off controls and caustics-on images are retained in PNG and
linear RGBA. This is an image-equivalence check, not timing qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageStat


NATIVE = Path(__file__).resolve().parents[2]
LABELS = ["baseline-a", "candidate", "baseline-b"]
FILMS = {"unfilmed": 0.0, "filmed": 385.0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(executable, fixture, label, root, width, height, frames):
    runs = {}
    for film_name, film in FILMS.items():
        for enabled in (0, 1):
            name = f"{film_name}-caustics-{enabled}"
            folder = root / name / label
            folder.mkdir(parents=True, exist_ok=False)
            stem = folder / name
            command = [
                str(executable), "--setup", str(fixture),
                "material=satinpaint", f"film={film}", "filmIor=1.33",
                "filmVar=0", "filmScale=1", "filmFlow=0", "filmDrain=0", "filmSky=0",
                "cast=2", "castShadows=0", f"castCaustics={enabled}",
                f"w={width}", f"h={height}", f"frames={frames}", "grain=0",
                "--hdr", str(stem.with_suffix(".rgba")), "--still", str(stem.with_suffix(".png")),
            ]
            with (folder / "session.json").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=NATIVE, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=900)
            runs[name] = {
                "command": command,
                "executable": str(executable),
                "executableSha256": digest(executable),
                "pngSha256": digest(stem.with_suffix(".png")),
                "linearRgbaSha256": digest(stem.with_suffix(".rgba")),
            }
    return runs


def statistics(before, after):
    with Image.open(before) as a, Image.open(after) as b:
        if a.size != b.size:
            raise ValueError(f"Dimensions differ: {before}={a.size}, {after}={b.size}")
        diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
        pixels = np.frombuffer(diff.tobytes(), dtype=np.uint8).reshape(-1, 3)
        return {
            "dimensions": list(a.size),
            "meanAbsoluteRGB": ImageStat.Stat(diff).mean,
            "maxRGB": [high for _, high in diff.getextrema()],
            "differentPixels": int(np.count_nonzero(np.any(pixels != 0, axis=1))),
        }


def linear_statistics(before, after):
    raw_a, raw_b = before.read_bytes(), after.read_bytes()
    if len(raw_a) < 8 or len(raw_b) < 8:
        raise ValueError("Linear RGBA capture is shorter than its dimension header")
    width, height = np.frombuffer(raw_a[:8], dtype="<u4")
    if raw_a[:8] != raw_b[:8]:
        raise ValueError(f"Linear dimensions differ: {before} and {after}")
    expected = 8 + int(width) * int(height) * 4 * 4
    if len(raw_a) != expected or len(raw_b) != expected:
        raise ValueError("Linear RGBA capture size does not match its dimension header")
    shape = (int(height), int(width), 4)
    a = np.frombuffer(raw_a, dtype="<f4", offset=8).reshape(shape).astype(np.float64)
    b = np.frombuffer(raw_b, dtype="<f4", offset=8).reshape(shape).astype(np.float64)
    diff = np.abs(a - b)
    changed = np.any(diff[:, :, :3] != 0.0, axis=2)
    coordinates = np.argwhere(changed)
    bounds = None
    if len(coordinates):
        lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
        bounds = [int(lo[1]), int(lo[0]), int(hi[1]), int(hi[0])]
    return {
        "dimensions": [int(width), int(height)],
        "meanAbsoluteRGB": diff[:, :, :3].mean(axis=(0, 1)).tolist(),
        "maxRGB": diff[:, :, :3].max(axis=(0, 1)).tolist(),
        "differentPixels": int(np.count_nonzero(changed)),
        "differentChannels": int(np.count_nonzero(diff)),
        "differentPixelBoundsXY": bounds,
        "alphaDifferentValues": int(np.count_nonzero(diff[:, :, 3])),
    }


def review_sheet(root):
    cell_w, cell_h, title_h = 384, 300, 28
    variants = [(film_name, enabled) for film_name in FILMS for enabled in (0, 1)]
    sheet = Image.new("RGB", (3 * cell_w, len(variants) * (cell_h + title_h)), "#242424")
    draw = ImageDraw.Draw(sheet)
    for row, (film_name, enabled) in enumerate(variants):
        name = f"{film_name}-caustics-{enabled}"
        for col, label in enumerate(LABELS):
            path = root / name / label / f"{name}.png"
            with Image.open(path) as image:
                x, y = col * cell_w, row * (cell_h + title_h)
                draw.text((x + 8, y + 7), f"{label} — {film_name}, caustics {'on' if enabled else 'off'}", fill="white")
                sheet.paste(image.convert("RGB"), (x, y + title_h))
    path = root.parent.parent / "review" / f"{root.name}-review.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def analyse(root):
    result = {"variants": {}}
    for film_name in FILMS:
        for enabled in (0, 1):
            name = f"{film_name}-caustics-{enabled}"
            comparisons = {}
            for left, right in [("baseline-a", "candidate"),
                                ("baseline-a", "baseline-b"),
                                ("baseline-b", "candidate")]:
                a = root / name / left / f"{name}.png"
                b = root / name / right / f"{name}.png"
                ah, bh = a.with_suffix(".rgba"), b.with_suffix(".rgba")
                comparisons[f"{left}__{right}"] = {
                    "png": statistics(a, b),
                    "linearRgbaIdentical": ah.read_bytes() == bh.read_bytes(),
                    "linearRgbaSha256": [digest(ah), digest(bh)],
                    "linearRgba": linear_statistics(ah, bh),
                }
            result["variants"][name] = {"comparisons": comparisons}
    result["reviewSheet"] = str(review_sheet(root))
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height, args.frames) <= 0:
        parser.error("width, height and frames must be positive")
    root = args.output.resolve()
    fixture = args.fixture.resolve(strict=True)
    if not args.analyse_only:
        baseline = args.baseline.resolve(strict=True)
        candidate = args.candidate.resolve(strict=True)
        root.mkdir(parents=True, exist_ok=False)
        provenance = {
            "fixture": str(fixture), "fixtureSha256": digest(fixture),
            "width": args.width, "height": args.height, "frames": args.frames, "runs": {},
        }
        for label, executable in [("baseline-a", baseline), ("candidate", candidate),
                                  ("baseline-b", baseline)]:
            provenance["runs"][label] = capture(
                executable, fixture, label, root, args.width, args.height, args.frames)
            (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(analyse(root), indent=2))


if __name__ == "__main__":
    main()
