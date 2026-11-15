"""What the thin film costs: a turning 4K frame against plain glass on the same shape, each timed
straight after plain glass; the least time of each over several rounds is what counts (other programs
share the GPU, and they only ever add).

usage: python tools/film_bench.py [exe] [rounds=5] [name=settings ...]

Without names it times the standard set. Prints one line a case: milliseconds and the ratio to plain glass.
"""
import json
import os
import subprocess
import sys

args = [a for a in sys.argv[1:] if '=' not in a]
pairs = [a.split('=', 1) for a in sys.argv[1:] if '=' in a]
EXE = os.path.abspath(args[0] if args else os.path.join('target', 'release', 'glassforms.exe'))
ROUNDS = 5
CASES = []
for k, v in pairs:
    if k == 'rounds':
        ROUNDS = int(v)
    else:
        CASES.append((k, v))
GARDEN = 'hdri=hdri/suburban_garden_4k.exr'
if not CASES:
    CASES = [
        ('a film on glass', 'film=400 filmVar=0.4'),
        ('a film on glass, draining', 'film=900 filmVar=0.2 filmDrain=0.4'),
        ('a film on glass under the garden picture', 'film=400 filmVar=0.4 ' + GARDEN),
        ('a soap bubble', 'material=soap'),
        ('a soap bubble under the garden picture', 'material=soap ' + GARDEN),
        ('a film on a metal', 'material=anodised'),
        ('oil on dark glass', 'material=oil'),
    ]
SHAPES = (('Thomas', 'scene=attractors preset=thomas'), ('the gyroid', 'scene=surfaces preset=gyroid'))


def idle():
    """Wait until nothing else is rendering (two readings of the GPU under a fifth busy, half a second
    apart), for six seconds at most: a timing taken while another program renders is noise."""
    import time
    calm = 0
    for _ in range(12):
        try:
            busy = int(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
        except (OSError, ValueError, IndexError):
            return
        calm = calm + 1 if busy < 20 else 0
        if calm >= 2:
            return
        time.sleep(0.5)


def ms(settings):
    idle()
    r = subprocess.run([EXE, '--bench', '40', 'w=3840', 'h=2160'] + settings.split(), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('%s\n%s' % (settings, r.stderr[-400:]))
    return json.loads(r.stdout.strip().splitlines()[-1])['ms']


for shape, where in SHAPES:
    # the least time of each over the rounds: other work on the GPU only ever adds
    plain = {False: 1e9, True: 1e9}
    best = {name: 1e9 for name, _ in CASES}
    for _ in range(ROUNDS):
        for name, s in CASES:
            # under a picture, the yardstick is plain glass under the same picture
            pic = GARDEN in s
            plain[pic] = min(plain[pic], ms(where + (' ' + GARDEN if pic else '')))
            best[name] = min(best[name], ms(where + ' ' + s))
    print('%s: plain glass %.2f ms%s' % (shape, plain[False], '' if plain[True] > 1e8 else ' (%.2f under the garden picture)' % plain[True]))
    for name, s in CASES:
        print('  %-44s %6.2f ms   x %.2f' % (name, best[name], best[name] / plain[GARDEN in s]), flush=True)
