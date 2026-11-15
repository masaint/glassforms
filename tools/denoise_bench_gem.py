"""What denoising clear glass (denoise=2) costs: a turning 3840 x 2160 brilliant, without and with, in turn,
three times; the quickest of each. Run as one timing job through gpu_run.py --timing."""
import json
import os
import subprocess

EXE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'target', 'release', 'glassforms.exe')
best = {}
for _ in range(3):
    for d in ('0', '2'):
        r = subprocess.run([EXE, '--bench', '60', 'w=3840', 'h=2160', 'scene=gems', 'preset=brilliant', 'denoise=' + d], capture_output=True, text=True)
        best[d] = min(best.get(d, 1e9), json.loads(r.stdout.strip().splitlines()[-1])['ms'])
print('brilliant: denoise=0 %.2f ms, denoise=2 %.2f ms (+%.2f)' % (best['0'], best['2'], best['2'] - best['0']))
