"""Step 11: compare the coat layer laws numerically (f64 ports of the WGSL)."""
import math

def fresnel(c, n1, n2):
    # dielectric unpolarised (pbr/dielectric.wgsl form): exact s/p average with TIR
    s2 = (n1 / n2) ** 2 * (1 - c * c)
    if s2 >= 1: return 1.0
    ct = math.sqrt(1 - s2)
    rs = (n1 * c - n2 * ct) / (n1 * c + n2 * ct)
    rp = (n1 * ct - n2 * c) / (n1 * ct + n2 * c)
    return 0.5 * (rs * rs + rp * rp)

def refr(c, ni, nt):
    if ni == nt: return c
    r = ni / nt
    return math.sqrt(max(0.0, 1 - r * r * (1 - c * c)))

def polar(ci, ct, ni, nt):
    if ni == nt: return (0.0, 0.0)
    if ct <= 0: return (1.0, 1.0)
    a = (ni * ci, ni * ct); b = (nt * ct, nt * ci)
    return tuple(((x - y) / (x + y)) ** 2 for x, y in zip(a, b))

def assembly(c, coat, body):  # paintLayerWeights, coated
    cc = refr(c, 1.0, coat); cb = refr(c, 1.0, body)
    top = polar(c, cc, 1.0, coat); base = polar(cc, cb, coat, body)
    out = [0.0, 0.0, 0.0]
    for t, b in zip(top, base):
        inv = 1 / max(1 - t * b, 1e-10)
        out[0] += 0.5 * t; out[1] += 0.5 * (1 - t) ** 2 * b * inv; out[2] += 0.5 * (1 - t) * (1 - b) * inv
    return out

def standalone(c, coat, body):  # trace_material 2129/2186: top1, under, no returns
    T = fresnel(c, 1.0, coat)
    U = fresnel(math.sqrt(max(1 - (1 - c * c) / coat ** 2, 0.0)), coat, body)
    return [T, (1 - T) ** 2 * U, (1 - T) * (1 - U)]

worst = (0, None)
rows = []
for coat in (1.45, 1.5, 1.6):
    for body in (1.3, 1.5, 1.7, 2.0):
        for c in (1.0, 0.7, 0.4, 0.2, 0.05):
            a = assembly(c, coat, body); s = standalone(c, coat, body)
            d = [x - y for x, y in zip(a, s)]
            m = max(abs(x) for x in d)
            rows.append((coat, body, c, a, s, d))
            if m > worst[0]: worst = (m, (coat, body, c, a, s))
print("worst |assembly - standalone| over shares:", worst)
for coat, body, c, a, s, d in rows:
    if c in (1.0, 0.2) and body in (1.5, 2.0) and coat == 1.5:
        print(f"coat {coat} body {body} cos {c}: assembly top/below/diffuse {[round(x,5) for x in a]}  standalone {[round(x,5) for x in s]}")
# isolate the two causes at one point: polarised vs averaged, and the internal-return series
c, coat, body = 0.2, 1.5, 2.0
cc = refr(c, 1.0, coat); cb = refr(c, 1.0, body)
top = polar(c, cc, 1.0, coat); base = polar(cc, cb, coat, body)
T = sum(top) / 2; U = sum(base) / 2
print("no-returns polarised below:", sum(0.5 * (1 - t) ** 2 * b for t, b in zip(top, base)), " averaged:", (1 - T) ** 2 * U,
      " with returns polarised:", assembly(c, coat, body)[1])
