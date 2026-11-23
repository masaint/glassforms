"""Baseline-A/candidate/baseline-B fixtures for smooth material dielectric interfaces.

Queue the outer invocation through tools/gpu_run.py. Every fixture is a single-material oblique scene;
spectral absorption selects trace_material for the clear curved/cut controls without enabling roughness,
birefringence or a random walk. Raw PNG/HDR/session data and provenance stay beneath the requested output.
These fixed-frame comparisons are not timing qualification.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import dielectric_render_repeat as shared
import numpy as np


SCENES = {
    "oblique-spectral-curve": (
        "scene=attractors preset=thomas material=crown spectrum=1 nb=8 "
        "rx=31 ry=-38 rz=11 scale=1 spin=0 grain=0"
    ).split(),
    "faceted-spectral-cut": (
        "scene=gems preset=brilliant material=diamond spectrum=1 nb=8 "
        "rx=38 ry=-28 rz=13 scale=1 spin=0 grain=0"
    ).split(),
    "coated-film-oblique": (
        "scene=knots preset=trefoil material=crown coat=1 coatIor=1.46 "
        "film=420 filmIor=1.32 filmVar=0.15 nb=8 rx=26 ry=-35 rz=9 "
        "scale=1 spin=0 grain=0"
    ).split(),
}

ROUTING = {
    "oblique-spectral-curve": {
        "geometry": "oblique curved Thomas attractor",
        "materialPath": "spectrum=1 selects trace_material; no roughness, birefringence or walk",
        "branches": "shadeM spectral entry/interior exit and beyondM clear re-entry/continuation",
    },
    "faceted-spectral-cut": {
        "geometry": "oblique faceted brilliant-cut gem",
        "materialPath": "spectrum=1 selects trace_material; cut-stone weaving remains active",
        "branches": "shadeM faceted internal reflection/exit, TIR cutoff and beyondM clear continuation",
    },
    "coated-film-oblique": {
        "geometry": "oblique curved trefoil",
        "materialPath": "coat/film select trace_material with smooth specialised layer weighting",
        "branches": "smooth substrate entry/exit directions; coat/film Fresnel and throughput remain controls",
    },
}


def load_hdr(path):
    raw = np.fromfile(path, dtype=np.uint8)
    width, height = np.frombuffer(raw[:8].tobytes(), dtype="<u4")
    pixels = np.frombuffer(raw[8:].tobytes(), dtype="<f4")
    return pixels.reshape(int(height), int(width), 4)


def hdr_statistics(before, after):
    a, b = load_hdr(before), load_hdr(after)
    if a.shape != b.shape:
        raise ValueError(f"HDR dimensions differ: {before}={a.shape}, {after}={b.shape}")
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "dimensions": [int(a.shape[1]), int(a.shape[0])],
        "differentPixelsRGB": int(np.count_nonzero(np.any(delta[..., :3] != 0.0, axis=2))),
        "differentPixelsRGBA": int(np.count_nonzero(np.any(delta != 0.0, axis=2))),
        "meanAbsoluteRGB": np.abs(delta[..., :3]).mean(axis=(0, 1)).tolist(),
        "signedMeanRGB": delta[..., :3].mean(axis=(0, 1)).tolist(),
        "maxRGB": np.abs(delta[..., :3]).max(axis=(0, 1)).tolist(),
        "meanAbsoluteAlpha": float(np.abs(delta[..., 3]).mean()),
        "maxAlpha": float(np.abs(delta[..., 3]).max()),
    }


def output_argument():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    return args.output


def main():
    if output_argument() is None:
        sys.argv.extend([
            "--output",
            str(shared.NATIVE / "out/unified_pbr/10-material-interfaces/renders/fixed-repeat"),
        ])
    shared.SCENES.clear()
    shared.SCENES.update(SCENES)
    shared.main()

    root = output_argument().resolve()
    comparison_path = root / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    routing = {name: ROUTING[name] for name in SCENES if name in comparison["scenes"]}
    (root / "fixture-routing.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
    for scene in routing:
        for pair, measurements in comparison["scenes"][scene]["comparisons"].items():
            left, right = pair.split("__")
            stem = f"{scene}.hdr"
            measurements["linearHdr"] = hdr_statistics(
                root / scene / left / stem,
                root / scene / right / stem,
            )
    archived_sheet = Path(comparison["reviewSheet"])
    local_sheet = root / "review-sheet.png"
    shutil.copy2(archived_sheet, local_sheet)
    comparison["archiveReviewSheet"] = str(archived_sheet)
    comparison["reviewSheet"] = str(local_sheet)
    comparison["fixtureRouting"] = routing
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
