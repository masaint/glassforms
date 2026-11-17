"""The performance gate: the owner's floor is 90 fps, 80% of the time, at 4K.

    python tools/perf_gate.py [--baseline] [--quick] [--exe path] [--only name] [--arg name=value ...]

For each scene a live session at 3840x2160 is timed frame by frame (`--session`: the view turned by hand,
the animation playing, then everything stopped and settling), and the 80th percentile of all its frames
must be 11.1 ms or less. Each run is written to out/perf/gate_<time>.json. With --baseline the run also
becomes out/perf/baseline.json; later runs say how far each scene moved from it, and a scene more than 5%
slower than its baseline is flagged even when it is under the floor.

Each scene runs as a timing job through tools/gpu_run.py, so it has the GPU to itself. Close the window
of the program first: a window drawing at the same time makes every number worthless.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
OUT = os.path.join(NATIVE, 'out', 'perf')
FLOOR = 1000.0 / 90.0
SLOWER = 1.05
SLOWER_MS = 0.3   # and by more than this: on a 2 ms scene 5% is noise
GARDEN = 'hdri=' + os.path.join(NATIVE, 'hdri', 'suburban_garden_4k.exr').replace('\\', '/')
LAMP = ['rig=', 'ambient=0,0,0', 'backdrop=0', 'floor=1', 'elev=25', 'cast=2',
        'lamp1=panel,1.4,1.8,0.6,0.6,0.4,1,1,1,25', 'lamp1.rx=90']
SCENES = [
    ('glass_thomas', ['scene=attractors', 'preset=thomas']),
    ('gem_cushion', ['scene=gems', 'preset=cushion']),
    ('milk_teapot', ['scene=models', 'preset=teapot', 'material=milk']),
    ('gold_knot', ['scene=knots', 'preset=trefoil', 'material=gold', 'studio=metal']),
    ('lamp_twist', ['scene=forms', 'preset=twistedring'] + LAMP),
    ('chrome_garden', ['scene=solids', 'preset=superellipsoid', 'material=chrome', GARDEN, 'hdriShow=1']),
]


EXTRA = []


def run(exe, args, frames):
    cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'perf_gate', '--timing', '--',
           exe, '--session', str(frames), 'w=3840', 'h=2160'] + args + EXTRA
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=NATIVE)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{') and '"all"' in l), None)
    if r.returncode != 0 or line is None:
        return {'error': (r.stderr or r.stdout)[-800:]}
    return json.loads(line)


def main(argv):
    exe = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
    if '--exe' in argv:
        exe = argv[argv.index('--exe') + 1]
    only = argv[argv.index('--only') + 1] if '--only' in argv else None
    EXTRA.extend(argv[i + 1] for i, a in enumerate(argv) if a == '--arg')
    frames = 120 if '--quick' in argv else 240
    os.makedirs(OUT, exist_ok=True)
    base_path = os.path.join(OUT, 'baseline.json')
    base = json.load(open(base_path)) if os.path.exists(base_path) and '--baseline' not in argv else {}
    results, failed = {}, []
    for name, args in SCENES:
        if only and only not in name:
            continue
        res = run(exe, args, frames)
        results[name] = res
        if 'error' in res:
            print(f'{name:14s} FAILED TO RUN: {res["error"]}', flush=True)
            failed.append(name)
            continue
        p80 = res['all']['p80']
        verdict = 'ok' if p80 <= FLOOR else 'OVER 11.1 ms'
        note = ''
        b = base.get(name, {}).get('all', {}).get('p80')
        if b:
            note = f'  ({(p80 / b - 1) * 100:+.1f}% vs baseline)'
            if p80 > b * SLOWER and p80 - b > SLOWER_MS:
                verdict += ' / SLOWER'
        if p80 > FLOOR or (b and p80 > b * SLOWER and p80 - b > SLOWER_MS):
            failed.append(name)
        parts = '  '.join(f'{k} p80 {res[k]["p80"]:.2f}' for k in ('turn', 'play', 'settle'))
        print(f'{name:14s} all p80 {p80:6.2f} ms  p95 {res["all"]["p95"]:6.2f}  max {res["all"]["max"]:7.2f}  [{parts}]  {verdict}{note}', flush=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    with open(os.path.join(OUT, f'gate_{stamp}.json'), 'w') as f:
        json.dump(results, f, indent=1)
    if '--baseline' in argv:
        with open(base_path, 'w') as f:
            json.dump(results, f, indent=1)
        print('written as the baseline')
    print('GATE ' + ('FAILED: ' + ', '.join(failed) if failed else 'PASSED'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
