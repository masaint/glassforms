"""Thin-film optics in plain numpy: the full-spectrum reference the tracer's film is checked against.

  spec_w(l)                         the tracer's own colour matching (CIE 1931 as fitted by Wyman, Sloan and
                                    Shirley 2013, to linear sRGB)
  airy(cos1, lam, d, n1, n2, n3)    reflectance of a film of index n2, d nm thick, between media n1 and n3
                                    (n3 may be complex), every reflection inside it counted, unpolarised
  colour(cos1, d, n1, n2, n3)       that reflectance as a colour under white light (white = 1, 1, 1), by
                                    integrating over 380 to 780 nm in 1 nm steps
"""
import numpy as np


def spec_w(l):
    l = np.asarray(l, dtype=np.float64)

    def g(x, m, a, b):
        t = (x - m) * np.where(x < m, a, b)
        return np.exp(-0.5 * t * t)
    x = 0.362 * g(l, 442.0, 0.0624, 0.0374) + 1.056 * g(l, 599.8, 0.0264, 0.0323) - 0.065 * g(l, 501.1, 0.0490, 0.0382)
    y = 0.821 * g(l, 568.8, 0.0213, 0.0247) + 0.286 * g(l, 530.9, 0.0613, 0.0322)
    z = 1.217 * g(l, 437.0, 0.0845, 0.0278) + 0.681 * g(l, 459.0, 0.0385, 0.0725)
    return np.stack([3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z], axis=-1)


def airy(cos1, lam, d, n1, n2, n3):
    """Unpolarised reflectance. All arguments broadcast; n3 may be complex (n + ik)."""
    cos1 = np.clip(np.asarray(cos1, dtype=np.float64), 1e-6, 1.0)
    n3 = np.asarray(n3, dtype=np.complex128)
    s1 = np.sqrt(1.0 - cos1 * cos1)
    c2 = np.sqrt((1.0 - (n1 * s1 / n2) ** 2).astype(np.complex128))
    c3 = np.sqrt(1.0 - (n1 * s1 / n3) ** 2)
    phi = 4.0 * np.pi * n2 * d * c2 / lam
    e = np.exp(1j * phi)
    out = 0.0
    for pol in (0, 1):
        if pol == 0:
            r12 = (n1 * cos1 - n2 * c2) / (n1 * cos1 + n2 * c2)
            r23 = (n2 * c2 - n3 * c3) / (n2 * c2 + n3 * c3)
        else:
            r12 = (n2 * cos1 - n1 * c2) / (n2 * cos1 + n1 * c2)
            r23 = (n3 * c2 - n2 * c3) / (n3 * c2 + n2 * c3)
        r = (r12 + r23 * e) / (1.0 + r12 * r23 * e)
        out = out + 0.5 * np.abs(r) ** 2
    return np.minimum(out, 1.0)


LAM = np.arange(380.0, 780.5, 1.0)
W = spec_w(LAM)
WSUM = W.sum(axis=0)


def colour(cos1, d, n1, n2, n3):
    """Reflectance as linear sRGB with white = (1, 1, 1). cos1 and d broadcast against each other;
    n3: a number, or an array of LAM's length (a metal's index at each wavelength)."""
    cos1 = np.asarray(cos1, dtype=np.float64)[..., None]
    d = np.asarray(d, dtype=np.float64)[..., None]
    r = airy(cos1, LAM, d, n1, n2, n3)
    return (r[..., None] * W).sum(axis=-2) / WSUM


def to_srgb8(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return (np.where(lin < 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)


if __name__ == '__main__':
    # the colour scale of a soap film, straight on: thickness against colour
    for d in (0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1200):
        c = colour(1.0, d, 1.0, 1.33, 1.0)
        print('%5d nm  %7.4f %7.4f %7.4f' % (d, c[0], c[1], c[2]))
