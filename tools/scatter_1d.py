"""A one-dimensional trial of the scheme the light volume uses, against the exact answer.

A slab of a scattering body, lit from the left by a beam that dies away as it goes in. The light scattered
out of the beam is the source of a diffusion equation with a boundary that lets light out on both faces.
The slab is laid over a grid at every offset within a cell: what matters is not only how near the grid's
answer is to the exact one, but that it does not change as the surface moves through the grid (that would
show as ripples on the picture).

usage: python tools/scatter_1d.py
"""
import numpy as np


def exact(L, ss, sa, A, xs):
    """Fluence of scattered light at depths xs, for a unit beam."""
    st = ss + sa
    D = 1.0 / (3.0 * st)
    zb = 2.0 * A * D
    k = np.sqrt(sa / D)
    C = ss / (sa - D * st * st)
    # u = C e^{-st x} + P e^{kx} + Q e^{-kx};  u - zb u' = 0 at 0;  u + zb u' = 0 at L
    m = np.array([[1 - zb * k, 1 + zb * k], [np.exp(k * L) * (1 + zb * k), np.exp(-k * L) * (1 - zb * k)]])
    r = np.array([-C * (1 + zb * st), -C * np.exp(-st * L) * (1 - zb * st)])
    P, Q = np.linalg.solve(m, r)
    return C * np.exp(-st * xs) + P * np.exp(k * xs) + Q * np.exp(-k * xs), D, zb


def grid(L, ss, sa, A, h, x0, n=None):
    """The same by the smoothed-boundary scheme on cells of size h, the slab starting at x0."""
    st = ss + sa
    D = 1.0 / (3.0 * st)
    zb = 2.0 * A * D
    n = n or int(np.ceil((x0 + L) / h)) + 4
    x = (np.arange(n) + 0.5) * h - 2 * h
    d = np.maximum(x0 - x, x - (x0 + L))
    ramp = lambda v: np.clip(0.5 - v / h, 0.0, 1.0)
    f = ramp(d)
    a = ramp(0.5 * (d[:-1] + d[1:]))            # faces between cells i and i+1
    ar = np.append(a, 0.0)
    al = np.insert(a, 0, 0.0)
    S = np.abs(ar - al)
    lo = np.clip(x - 0.5 * h - x0, 0.0, L)
    hi = np.clip(x + 0.5 * h - x0, 0.0, L)
    dep = (np.exp(-st * lo) - np.exp(-st * hi)) / (st * h)      # fluence of the beam, summed over the cell's part of the body, per cell
    M = np.zeros((n, n))
    rhs = ss * dep
    for i in range(n):
        M[i, i] = D / h ** 2 * (ar[i] + al[i]) + D / zb * S[i] / h + sa * f[i]
        if i + 1 < n:
            M[i, i + 1] = -D / h ** 2 * ar[i]
        if i > 0:
            M[i, i - 1] = -D / h ** 2 * al[i]
        if M[i, i] == 0.0:
            M[i, i] = 1.0
    u = np.linalg.solve(M, rhs)
    return x, u, D, zb


def report(L, mfp, albedo, A, h):
    ss = 1.0 / mfp
    sa = ss * (1 - albedo) / albedo
    rows = []
    for off in np.linspace(0.0, 1.0, 9)[:-1]:
        x0 = (3 + off) * h
        x, u, D, zb = grid(L, ss, sa, A, h, x0)
        front = np.interp(x0, x, u)
        back = np.interp(x0 + L, x, u)
        mid = np.interp(x0 + 0.5 * L, x, u)
        rows.append((front, mid, back))
    rows = np.array(rows)
    ue, D, zb = exact(L, ss, sa, A, np.array([0.0, 0.5 * L, L]))
    mean = rows.mean(axis=0)
    spread = (rows.max(axis=0) - rows.min(axis=0)) / np.maximum(mean, 1e-12)
    print('L %.2f mfp %.3f albedo %.4f  h/mfp %.2f  zb/h %.2f | exact front %.4f mid %.4f back %.4f | grid x %.3f %.3f %.3f | ripple %.1f%% %.1f%% %.1f%%'
          % (L, mfp, albedo, h / mfp, zb / h, ue[0], ue[1], ue[2], mean[0] / ue[0], mean[1] / ue[1], mean[2] / ue[2], 100 * spread[0], 100 * spread[1], 100 * spread[2]))


if __name__ == '__main__':
    A = 3.96
    for h in (3.0 / 256, 3.0 / 128):
        print('cell %.4f' % h)
        for L, mfp, alb in ((0.3, 0.06, 0.9998), (0.3, 0.1, 0.9), (0.3, 0.03, 0.9998), (0.06, 0.012, 0.9998), (0.06, 0.02, 0.9), (0.06, 0.032, 0.99), (0.3, 0.01, 0.99), (0.3, 0.3, 0.999)):
            report(L, mfp, alb, A, h)
