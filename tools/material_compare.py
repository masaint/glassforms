"""Capture fixed and adaptive 4K material frames on the same camera and animation path.

    python tools/material_compare.py rough_assembly
    python tools/material_compare.py rough_acrylic --exe path/to/glassforms.exe

Use perf_gate.py for frame times. PNG readbacks in this run serve visual comparison.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageChops, ImageDraw, ImageStat

from perf_gate import NATIVE, SCENES, executable_digest

FRAMES = {79: 'turn', 159: 'play', 160: 'stop', 161: 'second_settle', 2271: 'settled'}
DETAIL_BOX = {
    'rough_assembly': (1600, 850, 2200, 1290),
    'rough_acrylic': (2130, 635, 2770, 1115),
    'rough_nested_frost': (1610, 930, 2330, 1430),
}


def compare_frames(out, result):
    for index, phase in FRAMES.items():
        files = []
        for mode in ('fixed', 'adaptive'):
            path, = (out / mode).glob(f'f{index:04d}_s*.png')
            files.append(path)
        with Image.open(files[0]) as fixed, Image.open(files[1]) as adaptive:
            if fixed.size != (3840, 2160) or adaptive.size != fixed.size:
                raise ValueError(f'Unexpected output dimensions for {phase}')
            diff = ImageChops.difference(fixed.convert('RGB'), adaptive.convert('RGB'))
            info = {'frame': index, 'files': [str(p.relative_to(out)) for p in files],
                    'meanAbsoluteRGB': ImageStat.Stat(diff).mean}
            box = DETAIL_BOX[result['scene']]
            info['detailBox'] = box
            info['detailMeanAbsoluteRGB'] = ImageStat.Stat(diff.crop(box)).mean
            if phase in ('turn', 'play', 'settled'):
                width, height = box[2] - box[0], box[3] - box[1]
                detail = Image.new('RGB', (2 * width, height + 26), '#242424')
                for col, (mode, img) in enumerate([('Fixed', fixed), ('Adaptive', adaptive)]):
                    scale = files[col].stem.rsplit('_s', 1)[1]
                    ImageDraw.Draw(detail).text((col * width + 8, 6), f'{mode}   {scale}x', fill='white')
                    detail.paste(img.crop(box), (col * width, 26))
                name = 'playback' if phase == 'play' else phase
                detail.save(out / f'{name}-detail.png')
            result['frames'][phase] = info
    (out / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
    native = Path(NATIVE)
    scenes = {name: args for name, args in SCENES if name.startswith('rough_')}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', choices=scenes)
    parser.add_argument('--exe', type=Path, default=native / 'target/release/glassforms.exe')
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    out = native / 'out/material_compare' / (args.scene + '_' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    scene = scenes[args.scene]
    digest = executable_digest(exe)
    result = {'scene': args.scene, 'executable': str(exe), 'executableSha256': digest, 'args': scene, 'frames': {}}
    for mode, rate in [('fixed', 0), ('adaptive', 90)]:
        folder = out / mode
        folder.mkdir(parents=True)
        # Isolate the GPU so competing jobs cannot drive the adaptive quality controller.
        command = [sys.executable, str(native / 'tools/gpu_run.py'), '--name', f'{args.scene}-images-{mode}',
                   '--timing', '--', str(exe), '--interact', '1', 'w=3840', 'h=2160', *scene,
                   'scale=1', 'grain=0', 'spin=0', 'refineMost=1', f'keepRate={rate}',
                   'events=turn:80;play:80;idle:2112', f'shots={folder}', 'shotAt=' + ','.join(map(str, FRAMES))]
        with (folder / 'session.json').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=native, stdout=log, check=True)
        if executable_digest(exe) != digest:
            raise RuntimeError('The executable changed during capture. Repeat both modes using a stable copy.')
    compare_frames(out, result)
    print(out / 'comparison.json')


if __name__ == '__main__':
    main()
