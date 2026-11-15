"""What the studio costs: a turning 3840 x 2160 view with each thing switched on and without it, timed in
turn within one run of the program (`--bench-with`), several rounds, the quickest go of each kept. Other
work on the GPU can only make a go slower, and taking the two in turn keeps what it does to the one close
to what it does to the other.

usage: python tools/stage_bench.py [--old <program>] [rounds=40] [frames=12] [gyroid] [size=3840x2160]
       (from D:\\glassforms\\native)

  --old <program>   also time plain glass in the program as built before the studio went in, in turn with
                    this one: the two must take the same time
Prints a table; changes nothing."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.environ.get('GLASSFORMS_EXE') or os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
OLD = os.path.abspath(sys.argv[sys.argv.index('--old') + 1]) if '--old' in sys.argv else None
ROUNDS = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('rounds=')), 40)
FRAMES = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('frames=')), 12)
W, H = next((a.split('=')[1].split('x') for a in sys.argv if a.startswith('size=')), ('3840', '2160'))
SHAPE = ['scene=surfaces', 'preset=gyroid'] if 'gyroid' in sys.argv else ['scene=attractors', 'preset=thomas']
SIZE = ['w=' + W, 'h=' + H]

FLOOR = 'backdrop:1,floor:1,floorAuto:0,floorY:-1.1'
UP = ['elev=20']
# what is written; settings the same in both; what is switched on; and for the live floor, the environment
ROWS = [
    ('a backdrop', [], 'backdrop:1', None),
    ('a backdrop, the camera raised 20', UP, 'backdrop:1', None),
    ('a floor (height by hand)', UP, FLOOR, None),
    ('a floor under a tumbling shape, waited for', UP, 'backdrop:1,floor:1', None),
    ('a floor under a tumbling shape, as the window', UP, 'backdrop:1,floor:1', {'GLASSFORMS_FLOOR_LIVE': '1'}),
    ('a glossy floor', UP, FLOOR + ',floorGloss:0.6', None),
    ('the shape mirrored in it', UP, FLOOR + ',floorGloss:0.6,floorMirror:1', None),
    ('tone curve and grade', [], 'tone:1,contrast:1.2,saturation:1.1,wbTemp:0.2', None),
    ('a floor, under a lens (aperture 0.12)', UP + ['aperture=0.12'], FLOOR, None),
    ('a floor, under clay', UP + ['clay=1'], FLOOR, None),
    ('a floor, under frosted glass', UP + ['rough=0.25'], FLOOR, None),
]


def run(exe, *args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([exe, *args], capture_output=True, text=True, cwd=NATIVE, timeout=1800, env=e)
    lines = [l for l in r.stdout.splitlines() if l.startswith('{')]
    if not lines:
        raise RuntimeError((r.stderr or r.stdout).strip()[:300])
    return json.loads(lines[-1])


print('%s, a turning %s x %s frame, the quickest of %d goes of %d frames each way' % (' '.join(SHAPE), W, H, ROUNDS, FRAMES))
if OLD:
    best = [1e9, 1e9]
    for _ in range(ROUNDS):
        for k, exe in enumerate((OLD, EXE)):
            best[k] = min(best[k], run(exe, '--bench', str(FRAMES), *SIZE, *SHAPE)['ms'])
    print('%-48s %6.2f ms before, %6.2f ms now (%.3f times)' % ('plain glass, the program before and now', best[0], best[1], best[1] / best[0]), flush=True)
print('%-48s %10s %10s %8s' % ('', 'without', 'with', 'times'))
for name, same, with_, env in ROWS:
    r = run(EXE, '--bench-with', str(FRAMES), 'rounds=%d' % ROUNDS, *SIZE, *SHAPE, *same, 'with=' + with_, env=env)
    print('%-48s %7.2f ms %7.2f ms %8.2f' % (name, r['withoutMs'], r['withMs'], r['times']), flush=True)
