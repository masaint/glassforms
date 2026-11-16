"""4K sessions of the brushed metals (r4 metal), timing jobs through gpu_run: p80 of every frame.

    python tools/metal_perf.py <exe> [<exe> ...]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GARDEN = 'hdri=' + os.path.join(os.path.dirname(HERE), 'hdri', 'suburban_garden_4k.exr').replace('\\', '/')
SPH = ['scene=solids', 'preset=superellipsoid']
SCENES = [
    ('steel_studio', SPH + ['material=brushed', 'studio=metal']),
    ('steel_garden', SPH + ['material=brushed', GARDEN, 'hdriShow=1']),
    ('sgold_studio', SPH + ['material=satingold', 'studio=metal']),
    ('steel_teapot', ['scene=models', 'preset=teapot', 'material=brushed', 'studio=metal']),
]


def p80(v):
    v = sorted(v)
    return v[int(0.8 * (len(v) - 1))]


for exe in sys.argv[1:]:
    for name, args in SCENES:
        cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'metal', '--timing', '--',
               exe, '--session', '240', 'w=3840', 'h=2160'] + args
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith('{')]
        if not line:
            print(name, 'FAILED', (r.stderr or r.stdout)[-400:])
            continue
        d = json.loads(line[-1])
        parts = '  '.join(f'{k} {d[k]["p80"]:.2f}' for k in ('all', 'turn', 'play', 'settle') if k in d)
        print(exe.split('/')[-1], name, parts, flush=True)
