"""Capture and compare cut-owner persistence/inheritance states.

Run through native/tools/gpu_run.py. Complete the baseline before launching a
paired after-run so ordinary queue concurrency cannot race the analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

NAMES = ["saved", "restored", "recipient", "inherited"]
PAIRS = [("saved", "restored"), ("recipient", "inherited")]
W, H, FRAMES = 480, 360, 96


def read_hdr(path: Path) -> np.ndarray:
    return np.fromfile(path, "<f4", offset=8).reshape(H, W, 4)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", choices=["before", "after"], required=True)
    parser.add_argument("--compare-to", choices=["before"])
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    phase = args.output.resolve()
    native = Path(__file__).resolve().parents[2]
    exe = args.exe.resolve()
    out = phase / "renders" / args.label
    logs = phase / "logs"
    reports = phase / "reports"
    for folder in (out, logs, reports):
        folder.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        for name in NAMES:
            cmd = [
                str(exe), "--setup", str(phase / "fixtures" / f"{name}.json"),
                "rx=0", "ry=0", "rz=0", f"w={W}", f"h={H}", f"frames={FRAMES}",
                "spin=0", "meter=0", "seed=3", "grain=0", "bloom=0",
                "--hdr", str(out / f"{name}.rgba"), "--still", str(out / f"{name}.png"),
            ]
            run = subprocess.run(cmd, cwd=native, capture_output=True, text=True, timeout=300)
            (logs / f"{args.label}-{name}.log").write_text(
                "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + run.stdout + "\nSTDERR\n" + run.stderr,
                encoding="utf-8",
            )
            if run.returncode:
                raise RuntimeError(f"{name} failed with {run.returncode}")
            (out / f"{name}.report.json").write_text(run.stdout.splitlines()[-1] + "\n", encoding="utf-8")
        manifest = {"label": args.label, "executable": str(exe), "executable_sha256": digest(exe),
                    "width": W, "height": H, "frames": FRAMES}
        (reports / f"capture-{args.label}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result = {"label": args.label, "pairs": {}}
    for a, b in PAIRS:
        x, y = read_hdr(out / f"{a}.rgba"), read_hdr(out / f"{b}.rgba")
        delta = x - y
        result["pairs"][f"{a}_vs_{b}"] = {
            "max_abs_rgba": float(np.abs(delta).max()),
            "different_pixels": int((delta[:, :, :3] != 0).any(axis=2).sum()),
            "rgb_rms": float(np.sqrt(np.mean(delta[:, :, :3] ** 2))),
            "max_abs_depth": float(np.abs(delta[:, :, 3]).max()),
        }
    if args.compare_to:
        before = phase / "renders" / args.compare_to
        result["before_after"] = {}
        for name in NAMES:
            old, new = read_hdr(before / f"{name}.rgba"), read_hdr(out / f"{name}.rgba")
            result["before_after"][name] = {
                "max_abs_rgba": float(np.abs(new - old).max()),
                "rgb_rms": float(np.sqrt(np.mean((new[:, :, :3] - old[:, :, :3]) ** 2))),
                "max_abs_depth": float(np.abs(new[:, :, 3] - old[:, :, 3]).max()),
            }
    (reports / f"analysis-{args.label}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    if args.compare_to:
        columns = ["before saved", "before restored", "after saved", "after restored"]
        sheet = Image.new("RGB", (W * 4, (H + 34) * 2 + 42), (20, 23, 28))
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 12), "Cut ownership | saved/restored and recipient/inherited | matched fixtures", fill=(235, 238, 242))
        for row, (a, b) in enumerate(PAIRS):
            names = [a, b, a, b]
            labels = [f"before {a}", f"before {b}", f"after {a}", f"after {b}"]
            for col, (name, label) in enumerate(zip(names, labels)):
                label_name = args.compare_to if col < 2 else args.label
                image = Image.open(phase / "renders" / label_name / f"{name}.png").convert("RGB")
                x, y = col * W, 42 + row * (H + 34)
                draw.text((x + 10, y + 9), label, fill=(230, 233, 238))
                sheet.paste(image, (x, y + 34))
        review = phase.parent / "review"
        review.mkdir(parents=True, exist_ok=True)
        sheet.save(review / "cut-owner-comparison.png")


if __name__ == "__main__":
    main()
