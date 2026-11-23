"""Paths at 4K, timed as the gate times a scene (a live session: turned, played, settling), read from their
volume and from their formula.

    python tools/paths_time.py [--exe path] [--frames n] [name ...]

Each run goes through the GPU queue as a timing job. Prints each scene's p80 over all frames and per phase."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
METAL = ['material=copper', 'studio=metal']
SCENES = [
    ('strip_volume', ['scene=paths', 'preset=strippath'] + METAL),
    ('strip_formula', ['scene=paths', 'preset=strippath', 'trace=csg'] + METAL),
    ('tube_volume', ['scene=paths', 'preset=tubepath'] + METAL),
    ('tube_formula', ['scene=paths', 'preset=tubepath', 'trace=csg'] + METAL),
    ('tube_formula_glass', ['scene=paths', 'preset=tubepath', 'trace=csg']),
    ('tube_volume_glass', ['scene=paths', 'preset=tubepath']),
    ('arch_formula_glass', ['scene=studio', 'preset=arch', 'trace=csg']),
    ('arch_volume_glass', ['scene=studio', 'preset=arch']),
    ('star_formula', ['scene=paths', 'preset=outlinepath', 'trace=csg', 'p.shape=0'] + METAL),
    ('gear_volume', ['scene=paths', 'preset=outlinepath', 'p.shape=4'] + METAL),
    ('outline_formula', ['scene=paths', 'preset=outlinepath', 'trace=csg', 'p.shape=4'] + METAL),
]


def main(argv):
    exe = argv[argv.index('--exe') + 1] if '--exe' in argv else os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
    frames = argv[argv.index('--frames') + 1] if '--frames' in argv else '120'
    repeat = int(argv[argv.index('--repeat') + 1]) if '--repeat' in argv else 1
    names = [a for i, a in enumerate(argv) if not a.startswith('--') and (i == 0 or argv[i - 1] not in ('--exe', '--frames', '--repeat'))]
    for name, args in SCENES:
        if names and name not in names:
            continue
        runs = []
        for _ in range(repeat):
            cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'paths', '--timing', '--', exe, '--session', frames, 'w=3840', 'h=2160'] + args
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=NATIVE)
            line = next((l for l in r.stdout.splitlines() if l.startswith('{') and '"all"' in l), None)
            if line is None:
                print(name, 'error', (r.stderr or r.stdout)[-400:])
                continue
            d = json.loads(line)
            p80 = lambda k: (d.get(k).get('p80') if isinstance(d.get(k), dict) else d.get(k))
            runs.append((p80('all'), p80('turn'), p80('play'), p80('settle')))
        if not runs:
            continue
        runs.sort()
        best = runs[0]
        print(f"{name:20} all p80 best {best[0]:6.2f} ms (of {len(runs)}: {', '.join('%.2f' % r[0] for r in runs)})   turn {best[1]}  play {best[2]}  settle {best[3]}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
