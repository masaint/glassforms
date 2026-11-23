"""Baseline A / candidate / baseline B pixel-identity check for the unification steps.

    python tools/gpu_run.py --name unify-repeat -- python -I tools/pbr/unify_repeat.py \
        --baseline A.exe --candidate B.exe [--scenes all|name,name] [--out DIR] \
        [--width 1920 --height 1080 --frames 64] [--tolerance NAME=1e-6 ...]

Every scene is rendered three times as a fixed-seed still: baseline (A), candidate, baseline again (B).
A and B must match exactly, or the scene is NONDETERMINISTIC and fails: identity against the candidate
proves nothing otherwise. The candidate is then compared with A, the 8-bit PNG and the linear HDR (width,
height, then RGBA float32; alpha is the distance to the glass, 0 for none, so depth is compared with it
and reported apart from colour; the renderer has no other secondary target). By default both must be byte-identical (a NaN against a NaN at the same place
counts as identical). --tolerance NAME=REL allows the candidate's HDR to differ from A by at most REL
relative for that scene only; its PNG must still be identical. Exit status is non-zero on any failure.
Queue the outer invocation through tools/gpu_run.py; these are image checks, not timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image

NATIVE = Path(__file__).resolve().parents[2]
LABELS = ("baseline-a", "candidate", "baseline-b")
# A single preseeded render with no turntable, grain or adaptive sampling (opaque_paint_adapter.py COMMON,
# without the settings that change the scene itself). Appended last, so they win over scene arguments.
SEED = 5
FIXED = ["spin=0", "grain=0", "adapt=0", f"preseed={SEED}"]
PASS = ("IDENTICAL", "WITHIN_TOLERANCE")


def gate_scenes():
    """The perf gate's scenes, as {name: args}. perf_gate.py has no import-time side effects."""
    sys.path.insert(0, str(NATIVE / "tools"))
    try:
        import perf_gate
    finally:
        sys.path.pop(0)
    return {name: list(args) for name, args in perf_gate.SCENES}


# ---------------------------------------------------------------- comparison (no GPU)

def load_png(path):
    with Image.open(path) as image:
        image.load()
        return np.array(image)


def load_hdr(path):
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: {len(raw)} bytes, no header")
    width, height = struct.unpack("<II", raw[:8])
    if len(raw) != 8 + width * height * 16:
        raise ValueError(f"{path}: {len(raw)} bytes for {width}x{height} RGBA float32")
    return np.frombuffer(raw, dtype="<f4", offset=8).reshape(height, width, 4)


def diff_arrays(a, b):
    """Differences between two images (H, W[, C]). Float values count as equal only when bit-identical,
    or both NaN. Statistics over finite differences; a NaN against a number is counted separately."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"identical": False, "shapeMismatch": True,
                "shapes": [list(a.shape), list(b.shape)], "dtypes": [str(a.dtype), str(b.dtype)]}
    if a.dtype.kind == "f":
        bits = np.dtype(f"u{a.dtype.itemsize}")
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        same = (a.view(bits) == b.view(bits)) | (nan_a & nan_b)
        nan_mismatch = nan_a ^ nan_b
    else:
        same = a == b
        nan_mismatch = np.zeros(a.shape, bool)
    differ = ~same
    per_pixel = differ.any(axis=2) if differ.ndim == 3 else differ
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    numeric = differ & ~nan_mismatch
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        delta = np.where(numeric, np.abs(a64 - b64), 0.0)
        scale = np.maximum(np.abs(a64), np.abs(b64))
        rel = np.where(numeric, delta / np.where(scale > 0, scale, 1.0), 0.0)
    rel[~np.isfinite(rel)] = np.inf          # an infinity against a finite value
    finite = np.isfinite(delta)
    return {
        "identical": bool(not differ.any()),
        "shapeMismatch": False,
        "shape": list(a.shape),
        "differingValues": int(differ.sum()),
        "differingPixels": int(per_pixel.sum()),
        "nanMismatches": int(nan_mismatch.sum()),
        "maxAbs": float(delta[finite].max()) if finite.any() else 0.0,
        "meanAbs": float(delta[finite].mean()) if finite.any() else 0.0,
        "nonFiniteDiffs": int((~finite).sum()),
        "maxRel": float(rel.max()) if rel.size else 0.0,
    }


def within(stats, tolerance):
    """Identical, or (with a tolerance) every difference within it relative and no NaN mismatch."""
    if stats["identical"]:
        return True
    if tolerance is None or stats["shapeMismatch"]:
        return False
    return stats["nanMismatches"] == 0 and stats["maxRel"] <= tolerance


def compare_files(left, right):
    """left/right: {"png": path, "hdr": path}. Raises FileNotFoundError / ValueError when unreadable.
    "hdr" (all four channels) decides the verdict; "colour" and "depth" (alpha) split it for the report."""
    a, b = load_hdr(left["hdr"]), load_hdr(right["hdr"])
    out = {"png": diff_arrays(load_png(left["png"]), load_png(right["png"])), "hdr": diff_arrays(a, b)}
    if a.shape == b.shape:
        out["colour"] = diff_arrays(a[..., :3], b[..., :3])
        out["depth"] = diff_arrays(a[..., 3], b[..., 3])
    return out


def verdict(a_vs_b, candidate_vs_a, tolerance=None):
    if not (a_vs_b["png"]["identical"] and a_vs_b["hdr"]["identical"]):
        return "NONDETERMINISTIC"
    if candidate_vs_a["png"]["shapeMismatch"] or candidate_vs_a["hdr"]["shapeMismatch"]:
        return "SHAPE_MISMATCH"
    if not candidate_vs_a["png"]["identical"]:
        return "DIFFERENT"                    # a tolerance never covers the PNG
    if candidate_vs_a["hdr"]["identical"]:
        return "IDENTICAL"
    return "WITHIN_TOLERANCE" if within(candidate_vs_a["hdr"], tolerance) else "DIFFERENT"


def capture_files(root, scene, label):
    stem = Path(root) / scene / label / scene
    return {"png": stem.with_suffix(".png"), "hdr": stem.with_suffix(".hdr")}


def evaluate_scene(root, scene, tolerance=None):
    files = {label: capture_files(root, scene, label) for label in LABELS}
    missing = [str(p) for f in files.values() for p in f.values() if not p.is_file()]
    result = {"scene": scene, "tolerance": tolerance}
    if missing:
        return {**result, "verdict": "MISSING", "missing": missing}
    try:
        result["baselineA_vs_baselineB"] = compare_files(files["baseline-a"], files["baseline-b"])
        result["candidate_vs_baselineA"] = compare_files(files["baseline-a"], files["candidate"])
    except (OSError, ValueError) as error:
        return {**result, "verdict": "UNREADABLE", "error": str(error)}
    result["verdict"] = verdict(result["baselineA_vs_baselineB"], result["candidate_vs_baselineA"], tolerance)
    return result


def parse_tolerances(items, scenes):
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or name not in scenes:
            raise ValueError(f"--tolerance {item!r}: expected NAME=REL with NAME one of the selected scenes")
        rel = float(value)
        if not rel >= 0:
            raise ValueError(f"--tolerance {item!r}: REL must be a non-negative number")
        out[name] = rel
    return out


def table(results):
    lines = [f"{'scene':22s} {'verdict':17s} {'A=B':4s} {'png px':>8s} "
             f"{'hdr px':>8s} {'hdr maxAbs':>11s} {'hdr meanAbs':>11s} {'hdr maxRel':>11s} {'depth px':>8s}"]
    for r in results:
        cand = r.get("candidate_vs_baselineA")
        ab = r.get("baselineA_vs_baselineB")
        if not cand or cand["hdr"]["shapeMismatch"] or cand["png"]["shapeMismatch"]:
            lines.append(f"{r['scene']:22s} {r['verdict']:17s}")
            continue
        same = "yes" if ab["png"]["identical"] and ab["hdr"]["identical"] else "NO"
        h = cand["hdr"]
        lines.append(f"{r['scene']:22s} {r['verdict']:17s} {same:4s} {cand['png']['differingPixels']:8d} "
                     f"{h['differingPixels']:8d} {h['maxAbs']:11.3g} {h['meanAbs']:11.3g} {h['maxRel']:11.3g} "
                     f"{cand['depth']['differingPixels']:8d}")
    return "\n".join(lines)


# ---------------------------------------------------------------- capture (GPU)

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def capture(executable, root, scene, label, args, width, height, frames):
    files = capture_files(root, scene, label)
    folder = files["png"].parent
    folder.mkdir(parents=True, exist_ok=False)
    command = [str(executable), "--still", str(files["png"]), "--hdr", str(files["hdr"]),
               f"w={width}", f"h={height}", f"frames={frames}", *args, *FIXED]
    started = time.perf_counter()
    try:
        with (folder / "session.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=NATIVE, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=900).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    record = {"command": command, "exitCode": code, "seconds": round(time.perf_counter() - started, 2)}
    problems = [] if code == 0 else [f"exit {code}"]
    report = None
    for line in reversed((folder / "session.log").read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("{"):
            try:
                report = json.loads(line)
            except ValueError:
                pass
            break
    if code == 0:
        if report is None:
            problems.append("no renderer report")
        elif report.get("frames") != frames:
            problems.append(f"renderer reports frames={report.get('frames')}")
        elif "preseed" not in report or "renderPasses" not in report:
            record["warning"] = "renderer does not report preseed/renderPasses; fixed seed unverified"
        elif report["preseed"] != SEED or report["renderPasses"] != 1:
            problems.append(f"not one preseeded render: preseed={report['preseed']} "
                            f"renderPasses={report['renderPasses']}")
    for kind, path in files.items():
        if path.is_file():
            record[kind + "Sha256"] = digest(path)
        elif code == 0:
            problems.append(f"no {kind} written")
    record["renderer"] = report
    record["problems"] = problems
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scenes", default="all", help="all, or comma-separated perf_gate scene names")
    parser.add_argument("--out", type=Path, help="defaults to out/unify_repeat/<time>")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--tolerance", action="append", metavar="NAME=REL",
                        help="relative HDR tolerance for one scene (PNG must still be identical)")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.frames) <= 0:
        parser.error("width, height and frames must be positive")
    available = gate_scenes()
    names = list(available) if args.scenes == "all" else [s.strip() for s in args.scenes.split(",") if s.strip()]
    unknown = [n for n in names if n not in available]
    if unknown or not names:
        parser.error(f"unknown scenes {unknown}; choose from {', '.join(available)}")
    try:
        tolerances = parse_tolerances(args.tolerance, names)
    except ValueError as error:
        parser.error(str(error))
    baseline, candidate = args.baseline.resolve(strict=True), args.candidate.resolve(strict=True)
    root = (args.out or NATIVE / "out" / "unify_repeat" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    executables = {"baseline": {"path": str(baseline), "sha256": digest(baseline)},
                   "candidate": {"path": str(candidate), "sha256": digest(candidate)}}
    report = {"settings": {"width": args.width, "height": args.height, "frames": args.frames,
                           "fixed": FIXED, "tolerances": tolerances},
              "executables": executables, "scenes": {}}
    results = []
    for scene in names:
        captures = {}
        for label, exe in zip(LABELS, (baseline, candidate, baseline)):
            captures[label] = capture(exe, root, scene, label, available[scene],
                                      args.width, args.height, args.frames)
        failed = {label: c["problems"] for label, c in captures.items() if c["problems"]}
        result = evaluate_scene(root, scene, tolerances.get(scene))
        if failed:
            result["verdict"] = "CAPTURE_FAILED"
            result["captureProblems"] = failed
        result["captures"] = captures
        results.append(result)
        report["scenes"][scene] = result
        print(f"{scene}: {result['verdict']}", flush=True)
    for name, exe in (("baseline", baseline), ("candidate", candidate)):
        if digest(exe) != executables[name]["sha256"]:
            report["error"] = f"the {name} executable changed during the run"
    report["wallSeconds"] = round(time.perf_counter() - started, 1)
    ok = "error" not in report and all(r["verdict"] in PASS for r in results)
    report["passed"] = ok
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table(results))
    if "error" in report:
        print(report["error"])
    print(f"report: {root / 'report.json'}")
    print("UNIFY REPEAT " + ("PASSED" if ok else "FAILED: " + ", ".join(
        f"{r['scene']} {r['verdict']}" for r in results if r["verdict"] not in PASS)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
