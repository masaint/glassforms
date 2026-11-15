"""Environment pictures for the thin film's reference set-ups, written as Radiance .hdr panoramas.

usage: python tools/film_pictures.py [out folder]        (default: out/materials/film/env)

  garden_shade.hdr   the sample garden picture (hdri/suburban_garden_4k.exr) with its ground and everything
                     standing on it in shade, and its sky whitened: only the sky keeps its light. The soap
                     bubble reference was taken that way (its ground and trees are dark shapes against a
                     white sky), and a bubble shows it twice, so both mirror images lie on dark.
  studio.hdr         a photographer's studio: a large soft box overhead, a glow from above, two tall
                     panels behind to the sides, one soft box beside the camera; black elsewhere. Written
                     tilted (see TILT) so that the room can be pitched by the same angle, which leans the
                     film's "up" towards the camera while the lights stay where they are in the picture.
  room.hdr           a room papered in grey checks with one bright window, for the coated shell.

Needs only numpy. The garden picture is read by a small reader for OpenEXR files packed the common
"ZIP" way with 32-bit float channels, which is what that file is.
"""
import os
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'out', 'materials', 'film', 'env'))

# how far the studio is written leaning back (degrees): pitch the lights by this (lpitch=TILT) and the
# studio stands upright in the picture while the film's "up" leans that far towards the camera
TILT = 38.0


def read_exr(path):
    """An OpenEXR picture as (height, width, 3) floats, red green blue. Scanlines, ZIP-packed or plain,
    channels of 32-bit or 16-bit floats."""
    with open(path, 'rb') as f:
        d = f.read()
    if d[:4] != b'v/1\x01':
        raise ValueError('%s is not an OpenEXR file' % path)
    pos, attrs = 8, {}
    while True:
        e = d.index(b'\0', pos)
        name, pos = d[pos:e].decode(), e + 1
        if not name:
            break
        e = d.index(b'\0', pos)
        pos = e + 1
        size, = struct.unpack('<i', d[pos:pos + 4])
        attrs[name] = d[pos + 4:pos + 4 + size]
        pos += 4 + size
    x0, y0, x1, y1 = struct.unpack('<4i', attrs['dataWindow'])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    comp = attrs['compression'][0]
    if comp not in (0, 2, 3):
        raise ValueError('%s: packed in a way this reader does not know (%d)' % (path, comp))
    chans, p, v = [], 0, attrs['channels']
    while v[p] != 0:
        e = v.index(b'\0', p)
        kind, = struct.unpack('<i', v[e + 1:e + 5])
        chans.append((v[p:e].decode(), kind))
        p = e + 17
    width = {1: 2, 2: 4}
    line = sum(width[k] for _, k in chans) * w
    rows = {0: 1, 2: 1, 3: 16}[comp]
    n = (h + rows - 1) // rows
    offsets = struct.unpack('<%dQ' % n, d[pos:pos + 8 * n])
    out = {c: np.zeros((h, w), np.float32) for c, _ in chans}
    for o in offsets:
        y, size = struct.unpack('<ii', d[o:o + 8])
        lines = min(rows, y1 - y + 1)
        raw = d[o + 8:o + 8 + size]
        if size < lines * line:
            t = np.frombuffer(zlib.decompress(raw), np.uint8).astype(np.int64)
            # each byte was stored as its difference from the one before, the two halves interleaved
            t = ((np.cumsum(t) - 128 * np.arange(len(t))) % 256).astype(np.uint8)
            half = (len(t) + 1) // 2
            raw = np.empty(len(t), np.uint8)
            raw[0::2], raw[1::2] = t[:half], t[half:]
            raw = raw.tobytes()
        at = 0
        for j in range(lines):
            for c, k in chans:
                out[c][y - y0 + j] = np.frombuffer(raw, np.float16 if k == 1 else np.float32, w, at)
                at += width[k] * w
    return np.stack([out['R'], out['G'], out['B']], axis=-1)


def write_hdr(path, rgb):
    """Write (height, width, 3) floats as a Radiance picture (one shared exponent a pixel, not packed)."""
    rgb = np.maximum(np.asarray(rgb, np.float32), 0.0)
    top = rgb.max(axis=-1)
    e = np.where(top > 1e-32, np.floor(np.log2(np.maximum(top, 1e-32))) + 1, 0)
    scale = np.where(top > 1e-32, 256.0 / np.exp2(e), 0.0)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    out[..., 3] = np.where(top > 1e-32, e + 128, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n' % (rgb.shape[0], rgb.shape[1]))
        f.write(out.tobytes())
    print('wrote', path, rgb.shape[1], 'x', rgb.shape[0], flush=True)


def write_png(path, rgb, gain=1.0):
    from PIL import Image
    lin = np.clip(rgb * gain, 0.0, 1.0)
    Image.fromarray((np.where(lin < 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)).save(path)


def dirs(w, h):
    """The direction each texel of a panorama looks in: x to the right, y up, z towards the camera.
    (The middle of the picture is straight ahead of the camera, behind the shape: -z.)"""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    ph, th = (u[None, :] - 0.5) * 2 * np.pi, v[:, None] * np.pi
    # the program's own layout (env.rs paint_dir turned to the picture's: its middle is behind the shape)
    return np.stack([np.sin(th) * np.sin(ph), np.cos(th) * np.ones_like(ph), -np.sin(th) * np.cos(ph)], axis=-1)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def box(d, c, half, e, up=(0.0, 1.0, 0.0)):
    """A soft box seen along d: c points at its middle, half: the tangents of its half-width and
    half-height, e: how soft its edge is."""
    c = np.asarray(c, float)
    c = c / np.linalg.norm(c)
    r = np.cross(up, c)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    v = np.cross(c, r)
    z = d @ c
    a = np.abs(d @ r) / np.maximum(z, 1e-3)
    b = np.abs(d @ v) / np.maximum(z, 1e-3)
    return (1 - smooth(half[0] - e, half[0] + e, a)) * (1 - smooth(half[1] - e, half[1] + e, b)) * (z > 0)


def garden_shade():
    src = os.path.join(ROOT, 'hdri', 'suburban_garden_4k.exr')
    rgb = read_exr(src)
    h, w, _ = rgb.shape
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    up = np.cos((np.arange(h) + 0.5) / h * np.pi)[:, None] * np.ones((1, w))
    # sky: above the horizon, and either blue or bright cloud (not the green and brown of what stands on the ground)
    blue = rgb[..., 2] / np.maximum(rgb[..., 0], 1e-6)
    level = np.median(lum[: h // 3])
    sky = smooth(-0.02, 0.03, up) * np.maximum(smooth(1.05, 1.3, blue), smooth(1.2, 2.0, lum / level))
    # (a little spread, so that the mask has no ragged single pixels)
    k = 3
    pad = np.pad(sky, ((k, k), (k, k)), mode='edge')
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * k + 1
    soft = (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)
    sky = np.clip(np.minimum(sky, soft * 1.5), 0.0, 1.0)
    # the sky itself whitened most of the way, as under high thin cloud (the reference's sky is white: its
    # film colours are the film's own, not a blue sky's times the film's)
    white = rgb + 0.75 * (lum[..., None] - rgb)
    out = white * (0.03 + 0.97 * sky[..., None])
    write_hdr(os.path.join(OUT, 'garden_shade.hdr'), out)
    write_png(os.path.join(OUT, 'garden_shade.png'), out[::4, ::4], 1.0 / (4 * level))


def turn_x(d, deg):
    a = np.radians(deg)
    y = d[..., 1] * np.cos(a) - d[..., 2] * np.sin(a)
    z = d[..., 1] * np.sin(a) + d[..., 2] * np.cos(a)
    return np.stack([d[..., 0], y, z], axis=-1)


def studio():
    w, h = 4096, 2048
    # the studio as it should stand in the picture is s(d); the file holds it leaning, so that pitching
    # the lights by TILT stands it up again
    d = turn_x(dirs(w, h), TILT)
    # light from overhead, and light from behind that reaches down to the table; towards the camera and
    # low down it is dark, as a studio is where the photographer stands
    lit = 0.5 * smooth(0.3, 0.9, d[..., 1]) ** 1.2 + 0.35 * smooth(-0.15, 0.5, d[..., 1]) * np.clip(-d[..., 2], 0.0, 1.0)
    lit += 0.5 * box(d, (0.0, 1.0, -0.25), (0.84, 0.53), 0.012, up=(0.0, 0.0, 1.0))
    back = box(d, (-0.6428, 0.0, -0.766), (0.3, 0.75), 0.02) + box(d, (0.6428, 0.0, -0.766), (0.3, 0.75), 0.02)
    lit = np.maximum(lit, 0.6 * back)
    lit += 0.8 * box(d, (0.5736, 0.02, 0.8192), (0.34, 0.2), 0.012)
    out = np.repeat(lit[..., None], 3, axis=2).astype(np.float32)
    write_hdr(os.path.join(OUT, 'studio.hdr'), out)
    write_png(os.path.join(OUT, 'studio.png'), out[::4, ::4])


def room():
    w, h = 4096, 2048
    d = dirs(w, h)
    # a box room, the camera's eye a little above its middle: where each direction meets a wall
    half = np.array([4.0, 2.2, 4.0])
    eye = np.array([0.0, 0.4, 0.0])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(d > 0, (half - eye) / d, (-half - eye) / d)
    t = np.nan_to_num(t, nan=1e9, posinf=1e9)
    hit = t.min(axis=-1)
    wall = t.argmin(axis=-1)
    p = eye + d * hit[..., None]
    # grey checks with a fine grid of lines on them
    def checks(a, b):
        tile = (np.floor(a) + np.floor(b)) % 2
        fa, fb = np.abs((a * 4) % 1 - 0.5), np.abs((b * 4) % 1 - 0.5)
        lines = np.minimum(smooth(0.46, 0.5, fa) + smooth(0.46, 0.5, fb), 1.0)
        return (0.34 + 0.34 * tile) * (1 - 0.18 * lines)
    q = p * 1.0
    grey = np.where(wall == 1, checks(q[..., 0], q[..., 2]), np.where(wall == 0, checks(q[..., 2], q[..., 1]), checks(q[..., 0], q[..., 1])))
    # the ceiling is dark; the floor is lit; the walls fall off upwards
    ceiling = (wall == 1) & (d[..., 1] > 0)
    grey = np.where(ceiling, 0.02, grey * np.where(wall == 1, 1.0, 0.25 + 0.75 * smooth(2.2, -2.2, p[..., 1])))
    # one long bright window in the ceiling, behind the camera and to its left, and a small one opposite
    grey = grey + 14.0 * box(d, (-0.25, 0.86, 0.45), (0.95, 0.2), 0.01, up=(0.0, 0.0, 1.0)) + 3.0 * box(d, (0.75, 0.5, -0.45), (0.2, 0.3), 0.01)
    out = np.repeat(grey[..., None], 3, axis=2).astype(np.float32)
    write_hdr(os.path.join(OUT, 'room.hdr'), out)
    write_png(os.path.join(OUT, 'room.png'), out[::4, ::4])


if __name__ == '__main__':
    garden_shade()
    studio()
    room()
