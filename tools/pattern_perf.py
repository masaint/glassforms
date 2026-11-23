"""4K live sessions of the pattern builds and of strong exact warps, timed alone through the shared queue.

    python tools/pattern_perf.py [frames] [only names,..]

Prints p80 of all frames and of turning, playing and settling, in ms (see tools/mods_perf.py).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mods_perf as mp          # noqa: E402
import pattern_shots as ps      # noqa: E402

B = mp.BLOCK + mp.CSG
SCENES = [
    ('block csg', B),
    ('block csg twist 180', B + mp.mods(mp.TWIST)),
    ('ball csg swirl', mp.BALL + mp.CSG + mp.mods(mp.SWIRL)),
    ('cone lattice (wrap)', ps.CONE),
    ('dome lattice (wrap)', ps.DOME),
    ('ring of ribs (strokes, ring)', ps.RING),
    ('staggered wall', ps.WALL),
    ('lathe (revolve)', ps.LATHE),
]


def main(argv):
    frames = int(argv[0]) if argv else 240
    only = set(argv[1].split(',')) if len(argv) > 1 else None
    for name, args in SCENES:
        if only and not any(o in name for o in only):
            continue
        r = mp.run(args, frames)
        if 'error' in r:
            print(f'{name:30s} FAILED: {r["error"]}', flush=True)
            continue
        parts = '  '.join(f'{k} {r[k]["p80"]:.2f}' for k in ('turn', 'play', 'settle'))
        print(f'{name:30s} all p80 {r["all"]["p80"]:6.2f} ms  [{parts}]', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
