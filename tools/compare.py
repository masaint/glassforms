"""How alike are two pictures? Used to hold the native renderer to the web one.

usage: compare.py a.png b.png [diff.png] [--limit=<mean difference allowed, in grey levels>]

Prints the mean and the largest difference in grey levels (0-255), the share of pixels that differ by more
than 2, 8 and 32 levels, and how far apart the two pictures' overall brightness is. With a third file name,
writes the difference (amplified 8 times) as a picture. Exit status 1 if the mean exceeds the limit
(default 1.0), 2 if the sizes differ.

Two renderings of the same thing on different graphics APIs never agree bit for bit: each averages a
finite number of noisy frames, and edges land a hair differently. A mean difference well under one grey
level, with large differences confined to a few edge pixels, is "the same picture"."""
import sys
import numpy as np
from PIL import Image

args = [a for a in sys.argv[1:] if not a.startswith('--')]
limit = 1.0
for a in sys.argv[1:]:
    if a.startswith('--limit='):
        limit = float(a.split('=')[1])
if len(args) < 2:
    print(__doc__)
    sys.exit(2)
a = np.asarray(Image.open(args[0]).convert('RGB')).astype(np.float32)
b = np.asarray(Image.open(args[1]).convert('RGB')).astype(np.float32)
if a.shape != b.shape:
    print('sizes differ: %s against %s' % (a.shape[1::-1], b.shape[1::-1]))
    sys.exit(2)
d = np.abs(a - b)
worst = d.max(axis=2)
n = worst.size
print('mean %.3f  largest %.0f  |  pixels off by more than 2: %.2f%%, 8: %.2f%%, 32: %.3f%%  |  brightness %.2f against %.2f'
      % (d.mean(), d.max(), 100 * (worst > 2).sum() / n, 100 * (worst > 8).sum() / n, 100 * (worst > 32).sum() / n, a.mean(), b.mean()))
if len(args) > 2:
    Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(args[2])
sys.exit(0 if d.mean() <= limit else 1)
