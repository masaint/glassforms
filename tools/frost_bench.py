"""What a rough surface costs: a turning frame at 3840 x 2160 against plain glass on the same shape.

usage: python tools/frost_bench.py [exe] [--rounds N] [--quick]

Each figure is the least of N rounds (default 4), each round timed straight after plain glass, because
other programs on the GPU can only make a run slower. Prints a table and writes out/frost_bench.json.
"""
import json
import os
import sys

import frost_lib as fl

argv = sys.argv[1:]
ROUNDS = 4
if '--rounds' in argv:
    i = argv.index('--rounds')
    ROUNDS = int(argv[i + 1])
    del argv[i:i + 2]
QUICK = '--quick' in argv
args = [a for a in argv if not a.startswith('--')]
EXE = os.path.abspath(args[0]) if args else fl.EXE

SHAPES = [('Thomas', fl.THOMAS), ('gyroid', fl.GYROID)]
if not QUICK:
    SHAPES += [('cut stone', fl.GEM)]
CASES = [('glass, frost %g' % r, 'rough=%g' % r) for r in (0.15, 0.3, 0.6, 1.0)]
CASES += [('mirror', 'material=mirror')] + [('mirror, frost %g' % r, 'material=mirror rough=%g' % r) for r in (0.15, 0.3, 0.6, 1.0)]
if not QUICK:
    CASES += [('satin gold', 'material=satingold'), ('satin steel', 'material=brushed'), ('frosted, random (roughRef=1)', 'rough=0.6 roughRef=1')]

out = {}
for shape, where in SHAPES:
    best = {name: 1e9 for name, _ in CASES}
    plain = 1e9
    for _ in range(ROUNDS):
        for name, s in CASES:
            p = fl.bench(where, exe=EXE)
            m = fl.bench(where + ' ' + s, exe=EXE)
            plain = min(plain, p)
            best[name] = min(best[name], m / p)
    print('%s: plain glass %.2f ms' % (shape, plain), flush=True)
    for name, _ in CASES:
        print('  %-30s x %.2f  (%.2f ms)' % (name, best[name], best[name] * plain), flush=True)
    out[shape] = {'plain_ms': plain, 'ratio': best}
with open(os.path.join(fl.ROOT, 'out', 'frost_bench.json'), 'w') as f:
    json.dump(out, f, indent=1)
