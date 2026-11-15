"""Run `--motion-seq` for a set of cases, playing and turning, then measure them (tools/motion_grain.py).

    python tools/motion_cases.py <out folder> [--exe path] [--only name,name] [--moves play,turn]

One GPU job: run it through the queue.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)

CASES = {
    "thomas": [],
    "gyroid": ["scene=surfaces", "preset=gyroid"],
    "truemilk": ["scene=surfaces", "preset=gyroid", "material=truemilk"],
    "frosted": ["material=frosted"],
    "jade": ["material=jade"],
    "gold": ["material=gold"],
    "brilliant": ["scene=gems", "preset=brilliant"],
    "studio": ["studio=lavender"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--exe", default=os.path.join(NATIVE, "target", "release", "glassforms.exe"))
    ap.add_argument("--only", default="")
    ap.add_argument("--moves", default="play,turn")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    names = [n for n in CASES if not a.only or n in a.only.split(",")]
    dirs = []
    for n in names:
        for mv in a.moves.split(","):
            d = os.path.join(a.out, f"{n}-{mv}")
            cmd = [a.exe, "--motion-seq", d, "w=800", "h=450", f"move={mv}"] + CASES[n] + a.extra
            r = subprocess.run(cmd, cwd=NATIVE, capture_output=True, text=True)
            if r.returncode != 0:
                print(n, mv, "failed:", r.stderr.strip()[-400:])
                continue
            dirs.append(d)
    subprocess.run([sys.executable, os.path.join(HERE, "motion_grain.py")] + dirs)


if __name__ == "__main__":
    main()
