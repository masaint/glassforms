r"""Before (adapt=0) vs after at equal wall time, refining at 60 Hz pacing: stills after 0.25, 1 and 3 s,
every run a timing job in the shared queue. Prints each run's "refine:" line.

    python tools/r7_times.py <out dir> [scenes,..] [times,..] [extra args...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r7_run import run

out = sys.argv[1]
scenes = sys.argv[2].split(',') if len(sys.argv) > 2 else ['gem', 'pot', 'thomas', 'milky']
times = sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.25', '1', '3']
extra = sys.argv[4:]
os.makedirs(out, exist_ok=True)
for s in scenes:
    for t in times:
        for a in ('0', '1'):
            p = os.path.join(out, f'{s}_t{t}_a{a}.png')
            line = run(p, s, [f'refineTime={t}', 'hz=60', f'adapt={a}'] + extra, timing=True)
            print(s, t, 'after' if a == '1' else 'before', '->', line, flush=True)
