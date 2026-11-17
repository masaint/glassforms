r"""4K frame times of the tree forms, each as a timing job through the shared queue.

    python tools/tree_timing.py [names,..]

Prints p80 of the turn, play (the tree growing: rebuilt every frame) and settle phases, and of all frames,
against the floor of 11.1 ms. Close the program's window first: a window drawing at the same time makes
every number worthless. The first line is the default tree, to see whether the GPU is quiet (about 7 ms
when it is).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(os.path.dirname(HERE), 'target', 'release', 'glassforms.exe')
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'
XMAS = ['p.form=1', 'p.seed=5', 'p.bole=0.12', 'p.width=0.8']
CASES = [
    ('default', ['p.form=0', 'p.seed=1']),
    ('broadleaf leaves 0.3', ['p.form=0', 'p.seed=2', 'p.leaves=0.3']),
    ('broadleaf leaves 0.6', ['p.form=0', 'p.seed=2', 'p.leaves=0.6']),
    ('broadleaf leaves 1', ['p.form=0', 'p.seed=2', 'p.leaves=1']),
    ('conifer bare', XMAS),
    ('christmas tree 0.3', XMAS + ['p.leaves=0.3']),
    ('christmas tree 0.6', XMAS + ['p.leaves=0.6']),
    ('christmas tree 1', XMAS + ['p.leaves=1']),
    ('columnar', ['p.form=2', 'p.seed=3', 'p.bole=0.2']),
    ('weeping', ['p.form=3', 'p.seed=4', 'p.bole=0.4', 'p.width=1.6']),
    ('weeping leaves 0.6', ['p.form=3', 'p.seed=4', 'p.bole=0.4', 'p.width=1.6', 'p.leaves=0.6']),
    ('palm', ['p.form=4', 'p.seed=1', 'p.bole=2', 'p.lean=0.4', 'p.trunk=0.02', 'p.roots=0']),
]


def main(argv):
    only = argv[1].split(',') if len(argv) > 1 else None
    for name, args in CASES:
        if only and name not in only:
            continue
        cmd = [sys.executable, QUEUE, '--name', 'trees', '--timing', '--', EXE, '--session', '240', 'w=3840', 'h=2160', 'scene=trees', 'preset=tree'] + args
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith('{') and '"all"' in l), None)
        if line is None:
            print(f'{name:24} FAILED {(r.stderr or r.stdout)[-300:]}', flush=True)
            continue
        d = json.loads(line)
        p = {k: d[k]['p80'] for k in ('turn', 'play', 'settle', 'all')}
        flag = 'ok' if p['all'] <= 1000 / 90 else 'OVER'
        print(f"{name:24} all p80 {p['all']:6.2f}  turn {p['turn']:6.2f}  play {p['play']:6.2f}  settle {p['settle']:6.2f}  {flag}", flush=True)


if __name__ == '__main__':
    main(sys.argv)
