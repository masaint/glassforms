"""Stills for looking at the floor, its shadows and caustics (no checks: pictures to look at).

usage: python tools/ground_shots.py <exe> <out folder> <tag> [name=value ...]
Draws one still per entry of SHOTS (each the common settings, then the shot's own, then the extra ones
given) and lays them side by side, two rows, in <out>/<tag>_sheet.png.
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

exe, out, tag = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
os.makedirs(out, exist_ok=True)
COMMON = ['w=' + os.environ.get('GROUND_W', '480'), 'h=' + os.environ.get('GROUND_H', '270'), 'scene=studio', 'preset=sphere', 'floor=1', 'cast=2']
if os.environ.get('GROUND_COMMON') is not None:
    COMMON = ['w=' + os.environ.get('GROUND_W', '480'), 'h=' + os.environ.get('GROUND_H', '270')] + json.loads(os.environ['GROUND_COMMON'])
SHOTS = json.loads(os.environ.get('GROUND_SHOTS', 'null')) or [
    ['elev=60', 'rig=0,45,3,3,30,30,30'],
    ['elev=60', 'rig=90,45,3,3,30,30,30'],
    ['elev=60', 'rig=180,45,3,3,30,30,30'],
    ['elev=60', 'rig=270,45,3,3,30,30,30'],
]
tiles = []
for i, s in enumerate(SHOTS):
    path = os.path.join(out, '%s_%d.png' % (tag, i))
    r = subprocess.run([exe, '--still', path] + COMMON + s + extra, capture_output=True, text=True)
    if r.returncode != 0:
        print('FAIL', s, r.stderr[-600:])
        continue
    print(' '.join(s), r.stdout.strip().splitlines()[-1][:160])
    tiles.append((' '.join(s), Image.open(path).convert('RGB')))
if tiles:
    w, h = tiles[0][1].size
    cols = (len(tiles) + 1) // 2
    sheet = Image.new('RGB', (w * cols, h * 2), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        d.text((x + 6, y + 4), label, fill=(255, 255, 0))
    sheet.save(os.path.join(out, '%s_sheet.png' % tag))
    print('sheet', os.path.join(out, '%s_sheet.png' % tag))
