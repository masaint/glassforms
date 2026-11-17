r"""How fast a sculpted picture refines, before and after: several rays a pixel while refining, and the
studio's pixel pre-filter.

    python tools/converge_shots.py [out dir]

"before" is one ray a pixel a refining frame (refineMost=1) with the pre-filter off; "after" is as the
program now is. Stills after 8, 32 and 128 frames of the window's own refining loop (refine=N), at
1920x1080, side by side in <out dir>/converge_sheet.png. Then the time to 128 rays a pixel with each frame
taking at least a 60 Hz and a 144 Hz display's refresh (as the window's are, with vsync). Every run goes
through the shared GPU queue as a timing job.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = r'D:\glassforms\native\tools\gpu_run.py'
HAMMERED = {'layers': [{'kind': 'hammered', 'scale': 0.09, 'depth': 0.012, 'a': 1.2, 'b': 0.85, 'mode': 'carve'}]}
CELLS = {'layers': [{'kind': 'cells', 'scale': 0.14, 'depth': 0.1, 'a': 0.75, 'b': 0.9, 'mode': 'carve'}], 'round': 0.012, 'crumbs': 0.06}
SCENES = [('gem', ['scene=gems', 'preset=cushion'], HAMMERED, (560, 240, 1360, 740)), ('pot', ['scene=models', 'preset=teapot'], CELLS, (480, 200, 1440, 800))]
FRAMES = [8, 32, 128]


def run(args):
    r = subprocess.run([sys.executable, QUEUE, '--name', 'studio', '--timing', '--', EXE] + args, capture_output=True, text=True)
    m = re.search(r'refine: (\d+) frames in ([\d.]+) s, (\d+) rays', r.stderr + r.stdout)
    return m.groups() if m else None


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(NATIVE, 'out', 'r5')
    os.makedirs(os.path.join(out, 'conv'), exist_ok=True)
    for name, scene, stack, _ in SCENES:
        for v in ('before', 'after'):
            s = dict(stack, prefilter=(v == 'after'))
            extra = ['refineMost=1'] if v == 'before' else []
            for n in FRAMES:
                path = os.path.join(out, 'conv', f'{name}_{v}_{n}.png')
                got = run(['--still', path, 'w=1920', 'h=1080', f'refine={n}', 'sculpt=' + json.dumps(s)] + extra + scene)
                print(name, v, n, 'frames ->', got and f'{got[2]} rays in {got[1]} s', flush=True)
            for hz in (60, 144):
                got = run(['--still', os.path.join(out, 'conv', 't.png'), 'w=1920', 'h=1080', 'refineRays=128', f'hz={hz}', 'sculpt=' + json.dumps(s)] + extra + scene)
                print(name, v, f'{hz} Hz: 128 rays a pixel in', got and f'{got[1]} s ({got[0]} frames)', flush=True)
    from PIL import Image, ImageDraw
    W, H = 640, 400
    sheet = Image.new('RGB', (W * 3, (H + 18) * 4), (10, 10, 12))
    d = ImageDraw.Draw(sheet)
    for r, (name, v) in enumerate([(s[0], v) for s in SCENES for v in ('before', 'after')]):
        box = next(s[3] for s in SCENES if s[0] == name)
        for c, n in enumerate(FRAMES):
            im = Image.open(os.path.join(out, 'conv', f'{name}_{v}_{n}.png')).convert('RGB')
            sheet.paste(im.crop(box).resize((W, H)), (c * W, r * (H + 18) + 18))
            d.text((c * W + 6, r * (H + 18) + 3), f'{name} {v}, {n} refining frames', fill=(200, 205, 215))
    sheet.save(os.path.join(out, 'converge_sheet.png'))


if __name__ == '__main__':
    main(sys.argv)
