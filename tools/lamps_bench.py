"""What lamps cost: a turning 4K frame of plain glass without lamps, with one, and with four, in turn.

usage: python tools/lamps_bench.py <exe> [rounds=3]
Each case is timed `rounds` times, interleaved, and the quickest kept (other work on the GPU only ever
slows a run). Prints milliseconds a frame and the ratio to no lamps.
"""
import json
import subprocess
import sys

exe = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SHAPES = [('thomas', ['scene=attractors', 'preset=thomas']), ('gyroid', ['scene=surfaces', 'preset=gyroid'])]
CASES = [
    ('no lamps', []),
    ('one ball', ['lamp1=ball,1.3,0.9,1.4,0.2,0,1,0.9,0.8,15']),
    ('four', ['lamp1=ball,1.3,0.9,1.4,0.2,0,1,0.9,0.8,15', 'lamp2=tube,-1.6,-0.3,1.2,0.05,0.8,0.6,0.8,1,15',
              'lamp3=panel,0,2.2,1.2,0.6,0.4,1,1,1,5', 'lamp4=ball,-1.2,1.4,-1.6,0.3,0,1,0.6,0.4,10']),
]
best = {}
for _ in range(rounds):
    for shape, where in SHAPES:
        for name, args in CASES:
            p = subprocess.run([exe, '--bench', '80', 'w=3840', 'h=2160'] + where + args, capture_output=True, text=True)
            if p.returncode != 0:
                print('FAIL', shape, name, p.stderr[-400:])
                continue
            ms = json.loads(p.stdout.strip().splitlines()[-1])['ms']
            best[(shape, name)] = min(best.get((shape, name), 1e9), ms)
for shape, _ in SHAPES:
    base = best.get((shape, 'no lamps'))
    for name, _ in CASES:
        ms = best.get((shape, name))
        if ms is not None and base:
            print('%-7s %-9s %6.2f ms  x%.3f' % (shape, name, ms, ms / base))
