r"""Refining runs for docs/refine.md: one still through the shared GPU queue, printing the program's
"refine:" line.

    python tools/r7_run.py <out.png> [--timing] [--exe path] <scene> [program args...]

Scenes: gem (hammered cushion), pot (cells teapot), thomas (plain glass), milky, ceramic.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'
HAMMERED = {'layers': [{'kind': 'hammered', 'scale': 0.09, 'depth': 0.012, 'a': 1.2, 'b': 0.85, 'mode': 'carve'}]}
CELLS = {'layers': [{'kind': 'cells', 'scale': 0.14, 'depth': 0.1, 'a': 0.75, 'b': 0.9, 'mode': 'carve'}], 'round': 0.012, 'crumbs': 0.06}
SCENES = {
    'gem': ['scene=gems', 'preset=cushion', 'sculpt=' + json.dumps(HAMMERED, separators=(',', ':'))],
    'pot': ['scene=models', 'preset=teapot', 'sculpt=' + json.dumps(CELLS, separators=(',', ':'))],
    'thomas': ['scene=attractors', 'preset=thomas'],
    'milky': ['scene=models', 'preset=teapot', 'material=milk'],
    'ceramic': ['scene=models', 'preset=teapot', 'material=porcelain'],
}


def run(out, scene, args, timing=False, exe=EXE):
    cmd = [sys.executable, QUEUE, '--name', 'refine'] + (['--timing'] if timing else []) + ['--', exe, '--still', out, 'w=1920', 'h=1080'] + SCENES[scene] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = r.stderr + r.stdout
    lines = [l for l in text.splitlines() if l.startswith(('refine:', 'adapt:'))]
    line = ' | '.join(lines) or None
    if r.returncode != 0 or line is None:
        print(text[-2000:])
    return line


if __name__ == '__main__':
    a = sys.argv[1:]
    timing = '--timing' in a
    a = [x for x in a if x != '--timing']
    exe = EXE
    if '--exe' in a:
        i = a.index('--exe')
        exe = a[i + 1]
        del a[i:i + 2]
    print(run(a[0], a[1], a[2:], timing, exe))
