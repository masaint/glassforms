r"""Stills of the tree generator, for judging by eye, and a contact sheet of them.

    python tools/tree_shots.py <out dir> <set> [w h frames] [only names,..]

Every still goes through the shared GPU queue (tools/gpu_run.py), with this copy's
program. The sheet is <out dir>/trees_sheet.png. The sets of shots are written here only: the program
ships one tree and its sliders, no named kinds of tree.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
EXE = os.path.join(NATIVE, 'target', 'release', 'glassforms.exe')
QUEUE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_run.py')
T = ['scene=trees', 'preset=tree']

# a bark stack: long grooves up the wood, and a fine roughness
BARK = {'layers': [
    {'kind': 'crackle', 'scale': 0.045, 'depth': 0.006, 'stretch': 5, 'pitch': 90, 'a': 0.16, 'b': 0.9, 'mode': 'carve'},
    {'kind': 'noise', 'scale': 0.03, 'depth': 0.0025, 'a': 3, 'b': 0.5, 'mode': 'both'},
]}
# (only below the crown: the twigs are too thin to take grooves)
BARK_BARE_ARG = 'sculpt=' + json.dumps(BARK, separators=(',', ':'))
BARK['layers'][0].update(mask='thick', mask_at=0.02, mask_width=0.006)
BARK['layers'][1].update(mask='thick', mask_at=0.012, mask_width=0.004)
BARK_ARG = 'sculpt=' + json.dumps(BARK, separators=(',', ':'))
COLUMN = ['p.crown=0.85', 'p.width=0.5', 'p.shape=-0.8', 'p.leader=1', 'p.upturn=-0.3', 'p.angle=75', 'p.light=0']
SPREAD = ['p.crown=0.5', 'p.width=1.5', 'p.shape=0.6', 'p.leader=0.05', 'p.upturn=-0.1', 'p.light=0.5', 'p.angle=55']
WEEP = ['p.droop=0.7', 'p.width=1.0', 'p.crown=0.7', 'p.leader=0.3']

SETS = {
    'kinds': [
        ('glass_default', T + ['p.seed=1']),
        ('glass_column', T + ['p.seed=7', 'p.crown=0.82', 'p.width=0.42', 'p.shape=-0.7', 'p.leader=0.9', 'p.upturn=0.35', 'p.angle=30']),
        ('glass_spreading', T + ['p.seed=3', 'p.crown=0.5', 'p.width=1.5', 'p.shape=0.6', 'p.leader=0.05', 'p.upturn=-0.1', 'p.light=0.5', 'p.angle=55']),
        ('glass_weeping', T + ['p.seed=11', 'p.droop=0.7', 'p.width=1.0', 'p.crown=0.7', 'p.leader=0.3']),
        ('glass_dense', T + ['p.seed=5', 'p.twigs=3000', 'p.taper=2.2']),
        ('clay_default', T + ['p.seed=1', 'clay=1']),
    ],
    'final': [
        ('glass seed 1', T + ['p.seed=1']),
        ('glass seed 2', T + ['p.seed=2']),
        ('glass seed 3, lumpy', T + ['p.seed=3', 'p.lumps=0.9']),
        ('glass pointed', T + ['p.seed=7'] + COLUMN),
        ('glass spreading', T + ['p.seed=3'] + SPREAD),
        ('glass weeping', T + ['p.seed=11'] + WEEP),
        ('gold', T + ['p.seed=4', 'material=gold']),
        ('clay', T + ['p.seed=1', 'clay=1']),
        ('porcelain', T + ['p.seed=2', 'material=porcelain']),
        ('bark studio, clay', T + ['p.seed=1', 'clay=1', BARK_ARG]),
        ('leaves', T + ['p.seed=5', 'p.leaves=0.5']),
        ('grown 0.35', T + ['p.seed=1', 'p.grow=0.35']),
        ('grown 0.65', T + ['p.seed=1', 'p.grow=0.65']),
        ('dense twigs', T + ['p.seed=6', 'p.twigs=3000']),
        ('many roots, big flare', T + ['p.seed=8', 'p.roots=8', 'p.flare=1.6', 'clay=1']),
    ],
    'fix1': [
        ('leaves', T + ['p.seed=5', 'p.leaves=0.5']),
        ('grown 0.2', T + ['p.seed=1', 'p.grow=0.2']),
        ('grown 0.4', T + ['p.seed=1', 'p.grow=0.4']),
        ('glass seed 2', T + ['p.seed=2']),
        ('bark studio, clay', T + ['p.seed=1', 'clay=1', BARK_ARG]),
        ('clay', T + ['p.seed=1', 'clay=1']),
    ],
    'leaves': [
        ('leaves 0.5', T + ['p.seed=5', 'p.leaves=0.5']),
        ('leaves 1', T + ['p.seed=5', 'p.leaves=1']),
        ('leaves 0.5 clay', T + ['p.seed=5', 'p.leaves=0.5', 'clay=1']),
        ('grown 0.3', T + ['p.seed=1', 'p.grow=0.3']),
        ('grown 0.5', T + ['p.seed=1', 'p.grow=0.5']),
        ('grown 0.75', T + ['p.seed=1', 'p.grow=0.75']),
    ],
    'res': [
        ('res 256 (moving)', T + ['p.seed=2', 'res=256']),
        ('res 512', T + ['p.seed=2', 'res=512']),
        ('res 768 (still)', T + ['p.seed=2']),
        ('res 256 clay', T + ['p.seed=2', 'res=256', 'clay=1']),
        ('res 1024 clay', T + ['p.seed=2', 'res=1024', 'clay=1']),
        ('melt 0 (hard joins) clay', T + ['p.seed=2', 'melt=0', 'clay=1']),
    ],
    'close': [
        ('close clay melt 0.004', T + ['p.seed=2', 'clay=1', 'fov=9']),
        ('close clay melt 0', T + ['p.seed=2', 'clay=1', 'fov=9', 'melt=0']),
        ('close clay melt 0.01', T + ['p.seed=2', 'clay=1', 'fov=9', 'melt=0.01']),
        ('close glass', T + ['p.seed=2', 'fov=9']),
    ],
    'foot': [
        ('foot glass', T + ['p.seed=2', 'fov=8', 'shiftY=-2.3']),
        ('foot clay', T + ['p.seed=2', 'fov=8', 'shiftY=-2.3', 'clay=1']),
        ('whole glass', T + ['p.seed=2']),
    ],
    'bark': [
        ('bark, thick mask', T + ['p.seed=1', 'clay=1', BARK_ARG]),
        ('bark, no mask', T + ['p.seed=1', 'clay=1', BARK_BARE_ARG]),
        ('bark, thick mask, close', T + ['p.seed=1', 'clay=1', 'fov=9', BARK_ARG]),
        ('bark, thick mask, glass', T + ['p.seed=1', BARK_ARG]),
    ],
    'forms': [
        ('broadleaf glass', T + ['p.form=0', 'p.seed=2']),
        ('broadleaf clay', T + ['p.form=0', 'p.seed=2', 'clay=1']),
        ('conifer glass', T + ['p.form=1', 'p.seed=1', 'p.bole=0.15', 'p.width=0.85']),
        ('conifer clay', T + ['p.form=1', 'p.seed=1', 'p.bole=0.15', 'p.width=0.85', 'clay=1']),
        ('columnar glass', T + ['p.form=2', 'p.seed=3', 'p.bole=0.2']),
        ('columnar clay', T + ['p.form=2', 'p.seed=3', 'p.bole=0.2', 'clay=1']),
        ('weeping glass', T + ['p.form=3', 'p.seed=4', 'p.bole=0.4', 'p.width=1.6']),
        ('weeping clay', T + ['p.form=3', 'p.seed=4', 'p.bole=0.4', 'p.width=1.6', 'clay=1']),
        ('palm glass', T + ['p.form=4', 'p.seed=1', 'p.bole=2', 'p.lean=0.4', 'p.trunk=0.02', 'p.roots=0']),
        ('palm clay', T + ['p.form=4', 'p.seed=1', 'p.bole=2', 'p.lean=0.4', 'p.trunk=0.02', 'p.roots=0', 'clay=1']),
        ('christmas tree glass', T + ['p.form=1', 'p.seed=5', 'p.bole=0.12', 'p.width=0.8', 'p.leaves=0.6']),
        ('christmas tree clay', T + ['p.form=1', 'p.seed=5', 'p.bole=0.12', 'p.width=0.8', 'p.leaves=0.6', 'clay=1']),
        ('broadleaf leaves glass', T + ['p.form=0', 'p.seed=2', 'p.leaves=0.6']),
        ('broadleaf leaves clay', T + ['p.form=0', 'p.seed=2', 'p.leaves=0.6', 'clay=1']),
        ('leaf close-up clay', T + ['p.form=0', 'p.seed=2', 'p.leaves=0.6', 'clay=1', 'fov=4', 'shiftY=0.9', 'shiftX=0.6']),
        ('needle close-up clay', T + ['p.form=1', 'p.seed=5', 'p.bole=0.12', 'p.width=0.8', 'p.leaves=0.6', 'clay=1', 'fov=5', 'shiftX=0.8', 'shiftY=-0.6']),
        ('weeping leaves glass', T + ['p.form=3', 'p.seed=4', 'p.bole=0.4', 'p.width=1.6', 'p.leaves=0.6']),
        ('broadleaf trunk 1.2 lean', T + ['p.form=0', 'p.seed=2', 'p.bole=1.2', 'p.lean=0.5', 'clay=1']),
    ],
    'leafclose': [
        ('leaf close-up clay', T + ['p.form=0', 'p.seed=2', 'p.leaves=0.6', 'clay=1', 'fov=4', 'shiftY=0.9', 'shiftX=0.6']),
        ('leaf close-up glass', T + ['p.form=0', 'p.seed=2', 'p.leaves=0.6', 'fov=4', 'shiftY=0.9', 'shiftX=0.6']),
        ('columnar clay', T + ['p.form=2', 'p.seed=3', 'p.bole=0.2', 'clay=1']),
        ('broadleaf leaves clay', T + ['p.form=0', 'p.seed=2', 'p.leaves=0.6', 'clay=1']),
    ],
    'first': [
        ('glass_s1', T + ['p.seed=1']),
        ('clay_s1', T + ['p.seed=1', 'clay=1']),
    ],
}


def shot(out, w, h, frames, name, args):
    path = os.path.join(out, name.replace(' ', '_').replace(',', '') + '.png')
    cmd = [sys.executable, QUEUE, '--name', 'trees', '--', EXE, '--still', path, f'w={w}', f'h={h}', f'frames={frames}'] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(path)
    line = next((l for l in r.stdout.splitlines() if l.startswith('{')), '')
    ms = ''
    if '"volumeMs"' in line:
        ms = 'volume ' + line.split('"volumeMs":')[1].split(',')[0] + ' ms'
    print(name, 'ok' if ok else 'FAILED ' + (r.stderr or r.stdout)[-600:], ms, flush=True)
    return path if ok else None


def main(argv):
    out = argv[1]
    shots = SETS[argv[2]]
    w, h, frames = (argv[3], argv[4], argv[5]) if len(argv) > 5 else ('1920', '1080', '64')
    if len(argv) > 6:
        shots = [s for s in shots if s[0] in argv[6].split(',')]
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        paths = list(ex.map(lambda s: shot(out, w, h, frames, *s), shots))
    from PIL import Image, ImageDraw
    tw, th = 640, 360
    cols = 3
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new('RGB', (tw * cols, (th + 18) * rows), (10, 10, 12))
    d = ImageDraw.Draw(sheet)
    for k, ((name, _), p) in enumerate(zip(shots, paths)):
        r, c = divmod(k, cols)
        if p:
            sheet.paste(Image.open(p).convert('RGB').resize((tw, th), Image.LANCZOS), (c * tw, r * (th + 18) + 18))
        d.text((c * tw + 6, r * (th + 18) + 3), name, fill=(200, 205, 215))
    sheet.save(os.path.join(out, 'trees_sheet.png'))


if __name__ == '__main__':
    main(sys.argv)
