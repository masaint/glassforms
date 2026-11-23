"""4K live sessions with modifiers (gf_core::mods), timed alone through the shared queue.

    python tools/mods_perf.py [frames] [only names,..]

Each scene runs `--session` at 3840x2160 as a timing job (tools/gpu_run.py --timing) and prints p80 of all its
frames and of turning, playing and settling, in ms. Each modified scene has its modifier-free twin beside
it. Results go to out/perf/mods_<time>.json.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'
OUT = os.path.join(NATIVE, 'out', 'perf')


def M(kind, **k):
    d = dict(kind=kind)
    for key, v in k.items():
        d[key] = {'v': v[0], 'link': v[1]} if key in ('a', 'b') else v
    return d


def mods(*m):
    return ['mods=' + json.dumps(list(m), separators=(',', ':'))]


BLOCK = ['scene=studio', 'preset=block', 'p.width=0.75', 'p.height=1.9', 'p.depth=0.75']
BALL = ['scene=solids', 'preset=superellipsoid']
PEAR = ['scene=gems', 'preset=pear']
CSG = ['trace=csg']
TWIST = M('twist', a=([0, 180, 0], False))
SWIRL = M('swirl', a=([0, 200, 0], False), size=0.75)
REPEAT = M('repeat', a=([3, 1, 2], False), b=([3.0, 2.2, 3.8], False))
SCENES = [
    ('block csg', BLOCK + CSG),
    ('block csg twist', BLOCK + CSG + mods(TWIST)),
    ('ball csg', BALL + CSG),
    ('ball csg swirl', BALL + CSG + mods(SWIRL)),
    ('pear csg', PEAR + CSG),
    ('pear csg repeat', PEAR + CSG + mods(M('mirror', a=([1, 0, 0], False)), REPEAT)),
    ('block csg twist+swirl+repeat', BLOCK + CSG + mods(TWIST, M('swirl', a=([0, 90, 0], False), size=0.75), M('repeat', a=([2, 1, 2], False), b=([2.4, 2.2, 2.4], False)))),
    ('block volume', BLOCK),
    ('block volume twist (baked)', BLOCK + mods(TWIST)),
    ('thomas', ['scene=attractors', 'preset=thomas']),
    ('thomas noise (baked)', ['scene=attractors', 'preset=thomas'] + mods(M('noise', a=([0.025, 0.025, 0.025], True), size=0.22, detail=3))),
]


def run(args, frames):
    cmd = [sys.executable, QUEUE, '--name', 'mods_perf', '--timing', '--', EXE, '--session', str(frames), 'w=3840', 'h=2160'] + args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=NATIVE)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{') and '"all"' in l), None)
    if r.returncode != 0 or line is None:
        return {'error': (r.stderr or r.stdout)[-800:]}
    return json.loads(line)


def main(argv):
    frames = int(argv[0]) if argv else 240
    only = set(argv[1].split(',')) if len(argv) > 1 else None
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for name, args in SCENES:
        if only and not any(o in name for o in only):
            continue
        res = run(args, frames)
        results[name] = res
        if 'error' in res:
            print(f'{name:32s} FAILED: {res["error"]}', flush=True)
            continue
        parts = '  '.join(f'{k} {res[k]["p80"]:.2f}' for k in ('turn', 'play', 'settle'))
        print(f'{name:32s} all p80 {res["all"]["p80"]:6.2f} ms  [{parts}]', flush=True)
    with open(os.path.join(OUT, f'mods_{time.strftime("%Y%m%d-%H%M%S")}.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
