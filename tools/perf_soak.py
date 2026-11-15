"""Run the real window off-screen for a long while, the way it is used, and log how fast it draws.

The window is driven by a script (see app/src/ui/script.rs) that goes round and round the program:
shapes, families, materials, studio, shadows, lenses, layers, the light room, animation. The window
itself appends a line a second to a log (GF_PERFLOG): frames drawn, the time between them, the time the
engine's call took, and what the engine keeps. Beside it this script samples, every two seconds, the
program's memory (CPU working set and GPU dedicated memory, from Windows' counters) and the GPU's
clocks, temperature and power (nvidia-smi).

    python tools/perf_soak.py <out folder> [--minutes 15] [--exe path] [--cycle name] [extra name=value ...]

Run it through the GPU queue (one long job). Writes <out>/frames.log, <out>/samples.csv, <out>/summary.txt.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)

# One cycle of use, about a minute and a half at the frame rates of a 4K window. Every action takes a
# frame; wait:<n> lets n frames go by. The view spins all along, so every frame is drawn.
CYCLES = {
    "tour": [
        "arg:spin=0.25", "family:attractors", "preset:thomas", "wait:400",
        "arg:material=jade", "wait:300", "arg:material=gold", "wait:300", "arg:material=frosted", "wait:300",
        "arg:material=truemilk", "wait:300", "arg:material=glass",
        "family:gems", "preset:brilliant", "wait:300",
        "arg:studio=lavender", "wait:300", "arg:cast=1", "wait:300", "arg:cast=0", "arg:studio=none",
        "arg:proj=fisheye", "wait:200", "arg:proj=perspective", "arg:aperture=0.3", "wait:200", "arg:aperture=0",
        "room:open", "wait:300", "room:close",
        "family:knots", "preset:trefoil", "play", "wait:500", "play",
        "family:surfaces", "preset:gyroid", "wait:300",
        "add-layer", "wait:300",
        "family:forms", "preset:superformula", "play", "wait:400", "play",
        "family:models", "preset:teapot", "ready", "wait:300",
    ],
    # layers joined and animated: the seam's radius travels, a body of another material, casting, the lights turning
    "joins": [
        "arg:spin=0.25", "arg:lspin=0.2", "family:knots", "preset:trefoil", "arg:layer=solids:icosahedron",
        "arg:track=J12.radius:0.02:0.4:6", "play", "wait:600", "arg:B2.material=gold", "wait:600", "arg:cast=1", "wait:600",
        "arg:cast=0", "play", "wait:200",
    ],
    # the plainest case: one shape spinning, nothing else touched
    "still": ["arg:spin=0.25", "wait:2000"],
}


def gpu_mem_of(pid):
    """Dedicated GPU memory of one process, MB (Windows' per-process counter), or None."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage').CounterSamples | Measure-Object -Property CookedValue -Sum | ForEach-Object {{ $_.Sum }}"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out) / 1048576.0 if out else None
    except Exception:
        return None


def working_set_of(pid):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).WorkingSet64"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out) / 1048576.0 if out else None
    except Exception:
        return None


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,clocks.mem,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        return [x.strip() for x in out.split(",")]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--exe", default=os.path.join(NATIVE, "target", "release", "glassforms.exe"))
    ap.add_argument("--cycle", default="tour")
    ap.add_argument("--size", default="3840x2160")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "frames.log")
    open(log, "w").close()
    cycle = CYCLES[a.cycle]
    # enough cycles to fill the time; the window leaves when its timeout comes
    # (as many as fit on a command line)
    # every cycle starts from the setup the window opened with (so layers added in one are gone in the
    # next), and says so in the log by a mark the window cannot miss: a wait of 1 frame
    base = os.path.abspath(os.path.join(a.out, "base.json")).replace("\\", "/")
    one = ";".join([f"load-setup:{base}"] + cycle)
    script = ";".join([f"save-setup:{base}"] + [one] * max(1, 28000 // (len(one) + 1)))
    w, h = a.size.split("x")
    env = dict(os.environ, GF_PERFLOG=log)
    cmd = [a.exe, f"--ui-do={script}", "--ui-timeout", str(int(a.minutes * 60)), "--ui-dump", os.path.join(a.out, "state.json"),
           f"w={w}", f"h={h}", "vsync=0"] + a.extra
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=NATIVE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True).start()
    with open(os.path.join(a.out, "samples.csv"), "w") as f:
        f.write("t,ws_mb,gpu_mb,temp,clk_gr,clk_mem,power,util,card_mb,throttle\n")
        while proc.poll() is None:
            t = time.time() - t0
            ws, gm, g = working_set_of(proc.pid), gpu_mem_of(proc.pid), gpu_state()
            f.write(",".join([f"{t:.1f}", f"{ws or 0:.0f}", f"{gm or 0:.0f}"] + g) + "\n")
            f.flush()
            # past the timeout with a margin: the window has hung, end it
            if t > a.minutes * 60 + 90:
                proc.kill()
                break
            time.sleep(2.0)
    proc.wait()
    sys.stdout.write("".join(lines[-20:]))
    # summary: frame time by minute
    rows = []
    for line in open(log):
        parts = line.split()
        if len(parts) >= 6:
            rows.append((float(parts[0]), int(parts[1]), float(parts[2]), float(parts[4]), " ".join(parts[6:])))
    with open(os.path.join(a.out, "summary.txt"), "w") as s:
        for m in range(int(a.minutes) + 2):
            r = [x for x in rows if m * 60 <= x[0] < (m + 1) * 60]
            if not r:
                continue
            frames = sum(x[1] for x in r)
            secs = len(r)
            line = f"minute {m:2d}: {frames / secs:7.1f} frames/s  mean gap {sum(x[2] * x[1] for x in r) / max(frames, 1):6.2f} ms  engine call {sum(x[3] * x[1] for x in r) / max(frames, 1):5.2f} ms  last: {r[-1][4]}"
            s.write(line + "\n")
            print(line)
    print("exit", proc.returncode)


if __name__ == "__main__":
    main()
