"""Fixed-frame qualification for priority-directed overlapping media.

Queue the outer invocation through tools/gpu_run.py. Two oblique curved bodies overlap. Zero and equal
priorities are invariance controls; the +10/0 case is an intentional ownership change. Both camera transport
and a representative forward-caustic test-floor render retain PNG, linear RGBA, commands and session logs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

NATIVE = Path(__file__).resolve().parents[2]
MODES = {"camera": False, "forward-caustic": True}
RUNS = [
    ("zero", "baseline-a", "baseline"),
    ("zero", "candidate", "candidate"),
    ("zero", "baseline-b", "baseline"),
    ("priority", "baseline", "baseline"),
    ("priority", "candidate", "candidate"),
    ("tie", "baseline", "baseline"),
    ("tie", "candidate", "candidate"),
]
COMPARISONS = [
    ("zero/baseline-a", "zero/candidate", "invariance: zero-priority baseline-A/candidate"),
    ("zero/baseline-a", "zero/baseline-b", "invariance: zero-priority baseline A/B"),
    ("zero/baseline-b", "zero/candidate", "invariance: zero-priority baseline-B/candidate"),
    ("tie/baseline", "tie/candidate", "invariance: equal-priority baseline/candidate"),
    ("zero/candidate", "tie/candidate", "invariance: candidate zero/equal priorities"),
    ("priority/baseline", "priority/candidate", "EXPECTED CHANGE: +10 body 0 versus 0 body 1"),
    ("zero/candidate", "priority/candidate", "EXPECTED CHANGE: candidate zero versus +10/0"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(executable, fixture, mode, label, root, width, height, frames):
    folder = root / mode / fixture / label
    folder.mkdir(parents=True, exist_ok=False)
    stem = folder / f"{fixture}-{label}"
    command = [
        str(executable), "--setup", str(NATIVE / "out/unified_pbr/18-medium-priority/fixtures" / f"overlap-{fixture}.json"),
        f"w={width}", f"h={height}", f"frames={frames}", "grain=0", "spin=0",
        "--hdr", str(stem.with_suffix(".rgba")), "--still", str(stem.with_suffix(".png")),
    ]
    env = dict(os.environ)
    if MODES[mode]:
        command.extend(["cast=2", "castShadows=0", "castCaustics=1", "castDetail=1"])
        env["GLASSFORMS_CAST_FLOOR"] = "-1.3,35,4"
    with (folder / "session.json").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=NATIVE, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=900)
    return {
        "command": command,
        "floorEnvironment": env.get("GLASSFORMS_CAST_FLOOR"),
        "fixtureSha256": digest(NATIVE / "out/unified_pbr/18-medium-priority/fixtures" / f"overlap-{fixture}.json"),
        "executable": str(executable), "executableSha256": digest(executable),
        "pngSha256": digest(stem.with_suffix(".png")),
        "linearRgbaSha256": digest(stem.with_suffix(".rgba")),
    }


def png_stats(a, b):
    with Image.open(a) as left, Image.open(b) as right:
        diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
        values = np.frombuffer(diff.tobytes(), dtype=np.uint8).reshape(-1, 3)
        return {
            "differentPixels": int(np.count_nonzero(np.any(values != 0, axis=1))),
            "meanAbsoluteRGB": ImageStat.Stat(diff).mean,
            "maxRGB": [high for _, high in diff.getextrema()],
            "byteIdentical": a.read_bytes() == b.read_bytes(),
        }


def linear_stats(a, b):
    raw_a, raw_b = a.read_bytes(), b.read_bytes()
    if raw_a[:8] != raw_b[:8] or len(raw_a) < 8:
        raise ValueError(f"Linear dimensions differ: {a}, {b}")
    width, height = np.frombuffer(raw_a[:8], dtype="<u4")
    expected = 8 + int(width) * int(height) * 16
    if len(raw_a) != expected or len(raw_b) != expected:
        raise ValueError("Linear RGBA capture has an unexpected size")
    shape = (int(height), int(width), 4)
    left = np.frombuffer(raw_a, dtype="<f4", offset=8).reshape(shape)
    right = np.frombuffer(raw_b, dtype="<f4", offset=8).reshape(shape)
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    changed = np.any(diff[:, :, :3] != 0, axis=2)
    return {
        "differentPixels": int(np.count_nonzero(changed)),
        "differentChannels": int(np.count_nonzero(diff)),
        "meanAbsoluteRGB": diff[:, :, :3].mean(axis=(0, 1)).tolist(),
        "maxRGB": diff[:, :, :3].max(axis=(0, 1)).tolist(),
        "byteIdentical": raw_a == raw_b,
    }


def paths(root, mode, run):
    fixture, label = run.split("/")
    stem = root / mode / fixture / label / f"{fixture}-{label}"
    return stem.with_suffix(".png"), stem.with_suffix(".rgba")


def analyse(root):
    result = {"comparisons": {}, "expectations": {}}
    for mode in MODES:
        for left, right, kind in COMPARISONS:
            ap, ah = paths(root, mode, left)
            bp, bh = paths(root, mode, right)
            key = f"{mode}:{left}__{right}"
            result["comparisons"][key] = {"kind": kind, "png": png_stats(ap, bp), "linearRgba": linear_stats(ah, bh)}
            expected_change = kind.startswith("EXPECTED CHANGE")
            comparison = result["comparisons"][key]
            rgb_changed = comparison["linearRgba"]["differentPixels"] > 0
            invariant = comparison["png"]["byteIdentical"] and comparison["linearRgba"]["byteIdentical"]
            result["expectations"][key] = {
                "expectedChange": expected_change, "linearRgbChanged": rgb_changed,
                "pngAndLinearByteIdentical": invariant, "pass": rgb_changed if expected_change else invariant,
            }
    result["allExpectationsPass"] = all(item["pass"] for item in result["expectations"].values())
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def sheet(root):
    cell_w, cell_h, title_h = 384, 300, 46
    labels = [f"{fixture}\n{label}" for fixture, label, _ in RUNS]
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    canvas = Image.new("RGB", (len(RUNS) * cell_w, 80 + len(MODES) * (cell_h + title_h)), "#202020")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Priority-directed medium ownership | oblique overlapping ellipsoids | IOR 1.50 / 1.33", fill="white", font=title_font)
    draw.text((12, 40), "Zero and tie columns are invariance controls. +10/0 candidate is the intended ownership change. Fixed frames; no timing claim.", fill="white", font=font)
    for row, mode in enumerate(MODES):
        for col, ((fixture, label, _), title) in enumerate(zip(RUNS, labels)):
            png, _ = paths(root, mode, f"{fixture}/{label}")
            x, y = col * cell_w, 80 + row * (cell_h + title_h)
            draw.multiline_text((x + 7, y + 5), f"{mode} | {title}", fill="white", font=font, spacing=2)
            with Image.open(png) as image:
                canvas.paste(image.convert("RGB").resize((cell_w, cell_h)), (x, y + title_h))
    path = root / "review-sheet.png"
    canvas.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.analyse_only:
        executables = {"baseline": args.baseline.resolve(strict=True), "candidate": args.candidate.resolve(strict=True)}
        root.mkdir(parents=True, exist_ok=False)
        provenance = {"width": args.width, "height": args.height, "frames": args.frames, "runs": {}}
        for mode in MODES:
            for fixture, label, executable in RUNS:
                key = f"{mode}:{fixture}/{label}"
                provenance["runs"][key] = capture(executables[executable], fixture, mode, label, root, args.width, args.height, args.frames)
                (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    result = analyse(root)
    result["reviewSheet"] = str(sheet(root))
    print(json.dumps(result["expectations"], indent=2))
    if not result["allExpectationsPass"]:
        raise SystemExit("one or more invariance/expected-change comparisons failed")


if __name__ == "__main__":
    main()
