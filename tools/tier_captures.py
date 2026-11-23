"""Window captures of the quality-tier row (WP8 mock-up), dark and light, 1080p and 4K, each a GPU job through gpu_run.

    python -I tools/tier_captures.py EXE [only=<word>] [out=<folder>]

One window at a time (gpu_run, SLOTS = 1), the exact command written to out/_last_gpu_job.txt first, 1080p before 4K. Each capture is the
whole window (shot) plus a 4x crop of the row and its dropdown (crop), from the rectangles the window says it drew.
"""
import json
import os
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
GPU_RUN = os.environ.get('GPU_RUN', r'D:\glassforms\native\tools\gpu_run.py')
args = [a for a in sys.argv[1:] if '=' not in a]
EXE = os.path.abspath(args[0])
only = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('only=')), None)
OUT = os.path.join(NATIVE, 'out', next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('out=')), 'tier_captures'))
os.makedirs(OUT, exist_ok=True)

# (name, size, ui scale, theme, scene pairs, script)
OPEN = 'wait:14;press:tier:menu;wait:8;'
SHOTS = [
    ('row_full_dark_1080', (1920, 1080), 1, 'dark', ('tierAll=1',), 'wait:16;'),
    ('row_full_light_1080', (1920, 1080), 1, 'light', ('tierAll=1',), 'wait:16;'),
    ('menu_matcap_dark_1080', (1920, 1080), 1, 'dark', ('tierAll=1',), 'wait:14;key:z;wait:4;press:tier:menu;wait:8;'),
    ('menu_matcap_light_1080', (1920, 1080), 1, 'light', ('tierAll=1',), 'wait:14;key:z;wait:4;press:tier:menu;wait:8;'),
    ('menu_live_gems_dark_1080', (1920, 1080), 1, 'dark', ('tierAll=1', 'scene=gems', 'preset=brilliant'), 'wait:14;press:tier:live;wait:4;press:tier:menu;wait:8;'),
    ('menu_auto_light_1080', (1920, 1080), 1, 'light', ('tierAll=1',), 'wait:14;press:tier:menu;wait:6;press:tier/auto:on;wait:8;'),
    ('row_matcap_auto_dark_1080', (1920, 1080), 1, 'dark', ('tierAll=1', 'tierAuto=1'), 'wait:14;key:z;wait:8;'),
    ('row_full_dark_4k', (3840, 2160), 2, 'dark', ('tierAll=1',), 'wait:16;'),
    ('row_full_light_4k', (3840, 2160), 2, 'light', ('tierAll=1',), 'wait:16;'),
    ('menu_matcap_dark_4k', (3840, 2160), 2, 'dark', ('tierAll=1',), 'wait:14;key:z;wait:4;press:tier:menu;wait:8;'),
]


def resolve(script, rects):
    """press:<name> -> a click at that rectangle's middle, from the rectangles a first run noted."""
    out = []
    for act in script.split(';'):
        if act.startswith('press:') and act[6:] in rects:
            r = rects[act[6:]]
            out.append('pdown:%.1f,%.1f;pup' % ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2))
        else:
            out.append(act)
    return ';'.join(out)


def run(cmd, label):
    full = [sys.executable, '-I', GPU_RUN, '--name', 'wp8', '--'] + cmd
    open(os.path.join(OUT, '_last_gpu_job.txt'), 'w', encoding='utf-8').write(' '.join(full) + '\n')
    done = subprocess.run(full, capture_output=True, text=True, cwd=NATIVE, env={**os.environ, 'GF_GPU_QUEUE': r'D:\glassforms\_gpu_queue'}, timeout=300)
    if done.returncode != 0:
        print(label, 'FAILED', (done.stderr or done.stdout)[-300:])
        return False
    return True


def base(size, scale, theme, pairs, dump, shot, script):
    return [EXE, 'w=%d' % size[0], 'h=%d' % size[1], 'spin=0', *pairs, '--ui-scale', str(scale), '--ui-frames', '5', '--ui-dump', dump, '--ui-out', OUT,
            '--ui-timeout', '120', '--ui-shot', shot, '--ui-do=ui:theme=%s;%s' % (theme, script)]


def main():
    for name, size, scale, theme, pairs, script in SHOTS:
        if only and only not in name:
            continue
        dump, shot = os.path.join(OUT, name + '.json'), os.path.join(OUT, name + '.png')
        # (a first pass to learn where the buttons stand: the scripts press things by name, and the window gives each its rectangle)
        rects = {}
        if 'press:' in script:
            probe = os.path.join(OUT, name + '_probe.json')
            first = script.split('press:')[0] + 'wait:2'
            if not run(base(size, scale, theme, pairs, probe, os.path.join(OUT, name + '_probe.png'), first), name + ' probe'):
                continue
            rects = json.load(open(probe, encoding='utf-8'))['rects']
            # (the dropdown's own buttons are only drawn once it is open: a second step)
            if 'tier/auto:on' in script:
                step = script.split('press:tier/auto:on')[0] + 'wait:2'
                probe2 = os.path.join(OUT, name + '_probe2.json')
                if run(base(size, scale, theme, pairs, probe2, os.path.join(OUT, name + '_probe2.png'), resolve(step, rects)), name + ' probe2'):
                    rects.update(json.load(open(probe2, encoding='utf-8'))['rects'])
        if not run(base(size, scale, theme, pairs, dump, shot, resolve(script, rects)), name):
            continue
        d = json.load(open(dump, encoding='utf-8'))
        r = d['rects']
        ppp = d['window']['pixelsPerPoint']
        im = Image.open(shot)
        box = [r['tier bar'][0], r['camera bar'][1], r['tier bar'][2] + 20, r['tier bar'][3] + 6]
        if 'tier/menu' in r:
            box[3] = max(box[3], r['tier/menu'][3] + 6)
            box[2] = max(box[2], r['tier/menu'][2] + 6)
        box = [int(box[0] * ppp) - 8, int(box[1] * ppp) - 8, int(box[2] * ppp) + 8, int(box[3] * ppp) + 8]
        crop = im.crop(box)
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS) if size[0] <= 1920 else crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        crop.save(os.path.join(OUT, name + '_crop.png'))
        print(name, 'ok', im.size, 'crop', crop.size)


if __name__ == '__main__':
    main()
