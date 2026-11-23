"""Quick stills for the materials round: one queue slot for a batch of named variants.

    python tools/mats_try.py <exe> <out dir> <spec file>

The spec file: one still per line, `name  studio  arg arg ... [+ arg ...]` (args after + go after the studio's) where studio is `soft`, `sky`, `metal` (the
program's own metal studio) or `none`. w, h, frames default to 960 720 96 (override with w= h= frames=).
Queue the whole batch: python D:\\glassforms\\native\\tools\\gpu_run.py --name mats -- python tools/mats_try.py ...
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mats_shots import STUDIO, SKY  # noqa: E402

GPU_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_run.py')
STUDIOS = {'soft': STUDIO, 'sky': SKY, 'metal': ['studio=metal', 'elev=22'], 'none': []}


def main(argv):
    exe, out, spec = argv[1], argv[2], argv[3]
    os.makedirs(out, exist_ok=True)
    for line in open(spec, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        name, studio, args = parts[0], parts[1], parts[2:]
        size = ['w=960', 'h=720', 'frames=96']
        keys = {a.split('=')[0] for a in args}
        size = [s for s in size if s.split('=')[0] not in keys]
        path = os.path.join(out, name + '.png')
        # (the shape and material first, then the studio; args after a lone + come last of all)
        pre, post = (args[:args.index('+')], args[args.index('+') + 1:]) if '+' in args else (args, [])
        cmd = [sys.executable, GPU_RUN, '--name', 'mats', '--', exe, '--still', path] + size + pre + STUDIOS[studio] + post
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print(name, 'FAILED', (r.stderr or r.stdout)[-800:], flush=True)
            continue
        print(name, 'ok', path, flush=True)


if __name__ == '__main__':
    main(sys.argv)
