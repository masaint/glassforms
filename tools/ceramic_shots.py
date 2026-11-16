"""Stills of the ceramic materials under three lightings, and a sheet with the reference photographs.

    python tools/ceramic_shots.py <out dir> [materials comma list] [w h frames] [tag]

Each still goes through tools/gpu_run.py (name ceramic). sheet_<tag>.png: one row per material
(studio, key light, backlit), then a row of references.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = 'D:/glassforms/_r4_ceramic/native/target/release/glassforms.exe'
REFS = 'D:/glassforms/docs/refs/'
KEY = ['rig=100,30,5,5,8,8,8,0.5:-60,20,20,30,1,1,1.1,0.9', 'ambient=0.005,0.005,0.006']
BACK = ['rig=0,15,12,12,10,10,10,0.5:-120,30,30,30,0.4,0.4,0.42,0.9', 'ambient=0.005,0.005,0.006']
PHOTO = ['rig=-45,35,10,10,10,10,10,0.6:80,15,30,30,0.25,0.25,0.27,0.9', 'ambient=0.01,0.01,0.011']
LIGHTS = [('studio', []), ('key', KEY), ('photo', PHOTO), ('back', BACK)]
REF_ROW = ['scatter-porcelain-guanyin-frontlit-grey.jpg', 'scatter-milkglass-sugarbowl-frontlit-black.jpg',
           'scatter-milkglass-compote-frontlit-black.jpg', 'scatter-milkglass-lamp-unlit-frontlit.jpg',
           'scatter-porcelain-lithophane-backlit.jpg', 'scatter-porcelain-plate-held-to-sky-backlit.jpg']


def main(argv):
    out = argv[1]
    mats = (argv[2] if len(argv) > 2 else 'porcelain,milk,stoneware,bisque').split(',')
    w, h, frames = (argv[3], argv[4], argv[5]) if len(argv) > 5 else ('640', '480', '48')
    tag = argv[6] if len(argv) > 6 else 'x'
    os.makedirs(out, exist_ok=True)
    rows = []
    for m in mats:
        row = []
        for lname, largs in LIGHTS:
            path = os.path.join(out, f'{m}_{lname}_{tag}.png')
            cmd = [sys.executable, os.path.join(HERE, 'gpu_run.py'), '--name', 'ceramic', '--', EXE, '--still', path,
                   f'w={w}', f'h={h}', f'frames={frames}', 'scene=models', 'preset=teapot', f'material={m}'] + largs
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(m, lname, 'FAILED', (r.stderr or r.stdout)[-600:])
                continue
            print(m, lname, 'ok', path, flush=True)
            row.append(path)
        rows.append(row)
    from PIL import Image
    W, H = int(w), int(h)
    ref_h = H
    refs = []
    for r in REF_ROW:
        im = Image.open(REFS + r).convert('RGB')
        refs.append(im.resize((max(1, im.width * ref_h // im.height), ref_h)))
    ref_w = sum(i.width for i in refs)
    sheet = Image.new('RGB', (max(len(LIGHTS) * W, ref_w), H * len(rows) + ref_h), (40, 40, 40))
    for j, row in enumerate(rows):
        for i, p in enumerate(row):
            sheet.paste(Image.open(p).convert('RGB'), (i * W, j * H))
    x = 0
    for im in refs:
        sheet.paste(im, (x, H * len(rows)))
        x += im.width
    sp = os.path.join(out, f'sheet_{tag}.png')
    sheet.save(sp)
    print('sheet', sp)


if __name__ == '__main__':
    main(sys.argv)
