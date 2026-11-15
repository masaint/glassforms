"""What denoising costs: turning 3840 x 2160 frames of each case without it, with the tracer split but no
stage (GLASSFORMS_DENOISE=split), and with it, in turn, `rounds` times; the quickest of each is kept.
Run it as one timing job:  python tools/gpu_run.py --name denoise --timing -- python tools/denoise_bench.py

usage: python tools/denoise_bench.py [rounds] [frames]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FRAMES = sys.argv[2] if len(sys.argv) > 2 else '60'
CASES = {
    'truemilk': ['scene=surfaces', 'preset=gyroid', 'material=truemilk'],
    'truejade': ['scene=solids', 'preset=superellipsoid', 'material=truejade'],
    'roughref': ['scene=surfaces', 'preset=gyroid', 'material=frosted', 'roughRef=1'],
    'plain': ['scene=surfaces', 'preset=gyroid'],
}
MODES = [('off', 'denoise=0', ''), ('split', 'denoise=1', 'split'), ('on', 'denoise=1', '')]
best = {}
for _ in range(ROUNDS):
    for name, args in CASES.items():
        for mode, arg, env in MODES:
            if name == 'plain' and mode != 'off':
                continue
            e = dict(os.environ, GLASSFORMS_DENOISE=env)
            r = subprocess.run([EXE, '--bench', FRAMES, 'w=3840', 'h=2160', arg] + args, cwd=NATIVE, env=e, capture_output=True, text=True)
            ms = json.loads(r.stdout.strip().splitlines()[-1])['ms']
            k = (name, mode)
            best[k] = min(best.get(k, 1e9), ms)
for name in CASES:
    row = {m: best.get((name, m)) for m, _, _ in MODES}
    if row['on'] is None:
        print('%-9s off %.2f ms' % (name, row['off']))
        continue
    print('%-9s off %.2f ms   split %.2f (+%.2f)   on %.2f (stage +%.2f, all +%.2f)' % (
        name, row['off'], row['split'], row['split'] - row['off'], row['on'], row['on'] - row['split'], row['on'] - row['off']))
