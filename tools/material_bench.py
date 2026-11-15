"""What each material costs: a turning 4K frame, against plain glass on the same shape.

usage: python tools/material_bench.py <exe> [rounds=3] [frames=80] [w=3840] [h=2160] [ref=<another exe>]

For each shape (Thomas, the gyroid) every material is timed `rounds` times, each time straight after
plain glass, and the smaller of the times is kept (other programs use the GPU too, and they only ever
make a run slower). Then a scene of two bodies, each way of giving the second its own material
against the same scene as one material (only=bodies times just these). Prints milliseconds a frame and the ratio to plain glass. With ref=, plain glass in
that program is timed as well, to show that plain glass has not become slower.
"""
import json
import os
import subprocess
import sys

exe = os.path.abspath(sys.argv[1])
opts = dict(a.split('=', 1) for a in sys.argv[2:])
rounds, frames = int(opts.get('rounds', 3)), opts.get('frames', '80')
size = ['w=' + opts.get('w', '3840'), 'h=' + opts.get('h', '2160')]
ref = os.path.abspath(opts['ref']) if 'ref' in opts else None

SHAPES = [('thomas', ['scene=attractors', 'preset=thomas']), ('gyroid', ['scene=surfaces', 'preset=gyroid'])]
# name, settings
MATERIALS = [
    # a coat of index 1 is no coat at all: plain glass again, but traced the way every other material is
    ('nothing changed', ['coat=1', 'coatIor=1']),
    ('frost 0.3', ['rough=0.3']),
    ('frost 0.6', ['rough=0.6']),
    ('clear coat', ['coat=1']),
    ('thin film', ['film=400', 'filmVar=0.4']),
    ('colour by wavelength (ruby)', ['material=ruby']),
    ('milky, estimated (jade)', ['material=jade']),
    ('milky, true scattering (jade)', ['material=truejade']),
    ('glazed porcelain', ['material=porcelain']),
    ('pearl', ['material=pearl']),
    ('soap bubble', ['material=soap']),
    ('gold', ['material=gold']),
    ('satin gold', ['material=satingold']),
    ('anodised titanium', ['material=anodised']),
    ('bubbles', ['bubbles=0.2']),
    ('glow', ['glow=2']),
    ('double refraction', ['biref=-0.17']),
    ('graded index', ['grin=0.35']),
]
only = opts.get('only')
if only:
    MATERIALS = [m for m in MATERIALS if only in m[0]]


def bench(program, args):
    r = subprocess.run([program, '--bench', frames] + size + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('bench failed: %s\n%s' % (' '.join(args), r.stderr))
    return json.loads(r.stdout.strip().splitlines()[-1])['ms']


out = {}
for shape, where in SHAPES:
    plain = min(bench(exe, where) for _ in range(rounds))
    line = '%-8s plain glass %6.2f ms' % (shape, plain)
    if ref:
        was = min(bench(ref, where) for _ in range(rounds))
        now = min(plain, min(bench(exe, where) for _ in range(rounds)))
        plain = now
        line += '   (the program before materials: %.2f ms; now %.2f ms: %+.1f%%)' % (was, now, 100 * (now / was - 1))
    print(line, flush=True)
    out[shape] = {'plain': plain}
    for name, settings in MATERIALS:
        best, base = 1e9, 1e9
        for _ in range(rounds):
            base = min(base, bench(exe, where))
            best = min(best, bench(exe, where + settings))
        base = min(base, plain)
        out[shape][name] = (best, best / base)
        print('%-8s %-32s %6.2f ms   x %.2f' % (shape, name, best, best / base), flush=True)
# several bodies, each of its own material: against the same scene as one material
KNOT = ['scene=knots', 'preset=trefoil', 'layer=algebraic:heart', 'L2.size=0.7', 'J12.op=apart']
BODIES = [
    ('the same glass, drawn body by body', ['B2.own=1']),
    ('a gold heart in the glass knot', ['B2.material=gold']),
    ('a jade heart in the glass knot', ['B2.material=jade']),
    ('a glass heart in a gold knot', ['material=gold', 'B2.material=crown']),
    ('a soap-film heart in the glass knot', ['B2.material=soap']),
]
if not only or only == 'bodies':
    plain = min(bench(exe, KNOT) for _ in range(rounds))
    print('%-8s as one material %6.2f ms' % ('bodies', plain), flush=True)
    out['bodies'] = {'plain': plain}
    for name, settings in BODIES:
        best, base = 1e9, 1e9
        for _ in range(rounds):
            base = min(base, bench(exe, KNOT))
            best = min(best, bench(exe, KNOT + settings))
        base = min(base, plain)
        out['bodies'][name] = (best, best / base)
        print('%-8s %-36s %6.2f ms   x %.2f' % ('bodies', name, best, best / base), flush=True)
if 'json' in opts:
    json.dump(out, open(opts['json'], 'w'), indent=1)
