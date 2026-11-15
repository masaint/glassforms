"""Frame times of a turning 3840 x 2160 view for the materials this work touched, before and after.
usage: python tools/photo_bench.py <exe before> <exe after> [frames=60]
Run it alone on the GPU: python tools/gpu_run.py --name <who> --timing -- python tools/photo_bench.py ...
Each case is timed with both programs, twice, alternating; the better of the two runs is kept."""
import json
import os
import subprocess
import sys

before, after = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
frames = next((a[7:] for a in sys.argv[3:] if a.startswith('frames=')), '60')
HDRI = 'hdri=' + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hdri', 'suburban_garden_4k.exr').replace(os.sep, '/')
T = 'scene=attractors preset=thomas'
CASES = [
    ('plain glass', T),
    ('plain glass, picture', T + ' ' + HDRI),
    ('chrome, picture', T + ' material=chrome ' + HDRI),
    ('gold, studio', T + ' material=gold studio=white'),
    ('satin steel, picture', T + ' material=brushed ' + HDRI),
    ('satin gold, studio', T + ' material=satingold studio=white'),
    ('jade, picture', T + ' material=jade ' + HDRI),
    ('porcelain, studio', T + ' material=porcelain studio=lavender'),
    ('milk glass', T + ' material=milk'),
    ('ruby (clear, untouched)', T + ' material=ruby'),
]


def run(exe, st):
    r = subprocess.run([exe, '--bench', frames, 'w=3840', 'h=2160'] + st.split(), capture_output=True, text=True)
    if r.returncode:
        return float('nan')
    return json.loads(r.stdout.strip().splitlines()[-1])['ms']


for name, st in CASES:
    b = min(run(before, st), run(before, st))
    a = min(run(after, st), run(after, st))
    print('%-26s before %6.2f ms   after %6.2f ms   x %.3f' % (name, b, a, a / b), flush=True)
