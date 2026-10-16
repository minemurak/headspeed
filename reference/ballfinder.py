"""Reference for BallFinder.swift: finding the ball at rest (search on a 4x-downscaled frame,
then a full-resolution refine). search = face-on; search_rear (half-size frame) / refine_rear = rear view."""
import numpy as np, math
from analyzer import components

def mean_in(img, cx, cy, r0, r1):
    H, W = img.shape
    x0, x1 = max(0, int(cx - r1)), min(W - 1, int(cx + r1) + 1)
    y0, y1 = max(0, int(cy - r1)), min(H - 1, int(cy + r1) + 1)
    if x0 > x1 or y0 > y1: return 0.0
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    r2 = (xx - cx)**2 + (yy - cy)**2
    m = (r2 >= r0*r0) & (r2 <= r1*r1)
    return float(img[y0:y1 + 1, x0:x1 + 1][m].mean()) if m.any() else 0.0

def search(img, guide, minD, maxD):
    """The best bright blob (see search_all)."""
    c = search_all(img, guide, minD, maxD)
    return c[0] if c else None

def search_all(img, guide, minD, maxD):
    """Bright, round, compact blobs above a percentile threshold, best (contrast x fill) first."""
    H, W = img.shape
    gx0, gx1 = max(0, int(guide[0]*W)), min(W, int(guide[2]*W))
    gy0, gy1 = max(0, int(guide[1]*H)), min(H, int(guide[3]*H))
    g = img[gy0:gy1, gx0:gx1]
    hist = np.bincount(g.ravel(), minlength=256)
    acc = 0; p995 = 255
    for v in range(255, -1, -1):
        acc += hist[v]
        if acc >= g.size*0.005: p995 = v; break
    thr = max(140, p995)
    mask = np.zeros_like(img, bool); mask[gy0:gy1, gx0:gx1] = g >= thr
    return pick(img, mask, minD, maxD)

def refine(img, approx):
    """Face-on refine (BallFinder.refine): the bright blob at a threshold halfway between the
    ball's middle and its surroundings, which must be round and compact."""
    cx, cy, d = approx
    inside = mean_in(img, cx, cy, 0, d*0.3); ring = mean_in(img, cx, cy, d*0.9, d*1.4)
    if inside - ring < 30: return None
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx - cx)**2 + (yy - cy)**2 < d*d) & (img > (inside + ring)/2)
    cs = components(mask)
    if not cs: return None
    b = max(cs, key=lambda c: c['area'])
    if b['area'] <= 20: return None
    bw = b['maxx'] - b['minx'] + 1; bh = b['maxy'] - b['miny'] + 1
    if max(bw, bh)/min(bw, bh) > 1.3 or b['area']/(bw*bh) < 0.6: return None
    return b['sx']/b['area'], b['sy']/b['area'], 2*math.sqrt(b['area']/math.pi)

def pick(img, mask, minD, maxD):
    out = []
    for b in components(mask):
        d = 2*math.sqrt(b['area']/math.pi)
        if not (minD <= d <= maxD): continue
        bw = b['maxx'] - b['minx'] + 1; bh = b['maxy'] - b['miny'] + 1
        if max(bw, bh)/min(bw, bh) > 1.4 or b['area']/(bw*bh) < 0.6: continue
        cx = b['sx']/b['area']; cy = b['sy']/b['area']
        contrast = mean_in(img, cx, cy, 0, d*0.3) - mean_in(img, cx, cy, d*0.8, d*1.6)
        if contrast < 40: continue
        out.append((contrast*b['area']/(bw*bh), (cx, cy, d)))
    out.sort(key=lambda o: -o[0])
    return [o[1] for o in out]


def integral(img):
    H, W = img.shape
    I = np.zeros((H + 1, W + 1)); I[1:, 1:] = img.astype(float).cumsum(0).cumsum(1); return I

def box(I, x, y, h):
    H, W = I.shape[0] - 1, I.shape[1] - 1
    x0, x1 = max(0, x - h), min(W, x + h + 1); y0, y1 = max(0, y - h), min(H, y + h + 1)
    return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0], (x1 - x0)*(y1 - y0)


# ---- face-on fallback

def search_lit(img, guide, minD, maxD, anchor=None):
    """The preferred candidate of search_lit_all."""
    c = search_lit_all(img, guide, minD, maxD, anchor)
    return c[0] if c else None

def search_lit_all(img, guide, minD, maxD, anchor=None):
    """Fallback when the ball is only partly lit (sun from behind, the far half in its own
    shade, on sunlit grass as bright as that shade): centre-surround brightness, so the lit
    part still stands out on average, then a shape check so a white shoe does not pass."""
    H, W = img.shape
    gx0, gx1 = max(0, int(guide[0]*W)), min(W, int(guide[2]*W))
    gy0, gy1 = max(0, int(guide[1]*H)), min(H, int(guide[3]*H))
    I = integral(img)
    cands = []
    for r in range(max(2, int(minD/2)), int(maxD/2) + 1):
        hi, hr, ho = max(1, int(0.6*r + 0.5)), int(1.25*r + 0.5), 2*r   # rounded half up, as in Swift
        sc = np.full((gy1 - gy0, gx1 - gx0), -1e9)
        for y in range(gy0, gy1):
            for x in range(gx0, gx1):
                si, ni = box(I, x, y, hi); sm, nm = box(I, x, y, hr); so, no = box(I, x, y, ho)
                if no - nm <= 0: continue
                sc[y - gy0, x - gx0] = si/ni - (so - sm)/(no - nm)
        for y, x in np.argwhere(sc >= 60):
            if sc[y, x] < sc[max(0, y-1):y+2, max(0, x-1):x+2].max(): continue
            cands.append((float(sc[y, x]), x + gx0, y + gy0, r))
    cands.sort(reverse=True)
    valid = []
    for s, x, y, r in cands[:40]:
        w = int(math.ceil(2.5*r))
        x0, x1, y0, y1 = max(0, x - w), min(W, x + w + 1), max(0, y - w), min(H, y + w + 1)
        win = img[y0:y1, x0:x1]
        inner = mean_in(img, x, y, 0, 0.6*r); ring = mean_in(img, x, y, 1.25*r, 2.0*r)
        m = win >= (inner + ring)/2
        hit = None
        for c in components(m):
            if c['minx'] <= x - x0 <= c['maxx'] and c['miny'] <= y - y0 <= c['maxy']:
                if hit is None or c['area'] > hit['area']: hit = c
        if hit is None: continue
        bw = hit['maxx'] - hit['minx'] + 1; bh = hit['maxy'] - hit['miny'] + 1
        # compact and on its own: not touching the window (a shoe or a leg runs on)
        if hit['minx'] == 0 or hit['miny'] == 0 or hit['maxx'] == x1 - x0 - 1 or hit['maxy'] == y1 - y0 - 1: continue
        if max(bw, bh)/min(bw, bh) > 1.6 or hit['area']/(bw*bh) < 0.4: continue
        d = max(bw, bh)
        if not (minD <= d <= maxD): continue
        valid.append((s, float(x0 + hit['sx']/hit['area']), float(y0 + hit['sy']/hit['area']), float(d)))
    if not valid: return []
    if anchor is not None:
        valid.sort(key=lambda v: -v[0]/(1 + 3*math.hypot(v[1]/W - anchor[0], v[2]/H - anchor[1])))
    else:
        # face-on, the ball lies in front of the feet: the strong candidates lowest first
        # (a white shoe toe is about as bright and compact), then the rest
        top = max(v[0] for v in valid)
        valid.sort(key=lambda v: (v[0] < 0.8*top, -v[2] if v[0] >= 0.8*top else -v[0]))
    return [(x, y, d) for s, x, y, d in valid]


# ---- rear view: the ball sits on a tee, often in front of a split background (dark field
# above, bright path below) or backlit, so it is neither reliably brighter nor darker than its
# surroundings. Its outline is still a circle of edges: look for that instead of a blob.

BOTTOM = (math.radians(50), math.radians(130))     # the tee's side (image y grows downward)

def sobel(img):
    p = np.pad(img.astype(float), 1, mode='edge')
    gx = (p[:-2, 2:] + 2*p[1:-1, 2:] + p[2:, 2:] - p[:-2, :-2] - 2*p[1:-1, :-2] - p[2:, :-2])/8
    gy = (p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:] - p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:])/8
    return gx, gy

def coverage(gx, gy, mag, cx, cy, r, emin):
    """Share of directions (tee side skipped) with a radial edge on the circle."""
    H, W = mag.shape; hit = n = 0
    for a in np.radians(np.arange(0, 360, 10)):
        if BOTTOM[0] <= a <= BOTTOM[1]: continue
        ux, uy = math.cos(a), math.sin(a); n += 1
        for dr in (0, -1, 1, -1.5, 1.5):
            x = int(round(cx + ux*(r + dr))); y = int(round(cy + uy*(r + dr)))
            if not (0 <= x < W and 0 <= y < H): continue
            m = mag[y, x]
            if m >= emin and abs(gx[y, x]*ux + gy[y, x]*uy) >= 0.93*m: hit += 1; break
    return hit/n

def search_rear(img, guide, minD, maxD, anchor, emin=10.0):
    """The best candidate of search_rear_all."""
    c = search_rear_all(img, guide, minD, maxD, anchor, emin)
    return c[0] if c else None

def search_rear_all(img, guide, minD, maxD, anchor, emin=10.0):
    """Circle Hough on edge directions (either polarity) inside the guide, on a half-size
    frame. Candidates must have edges all round (a square sticker or a tee fails); among
    them, the one nearest the anchor (x, y in 0..1) wins; specks near the minimum size count half."""
    H, W = img.shape
    gx0, gx1 = max(0, int(guide[0]*W)), min(W, int(guide[2]*W))
    gy0, gy1 = max(0, int(guide[1]*H)), min(H, int(guide[3]*H))
    gx, gy = sobel(img); mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag[gy0:gy1, gx0:gx1] >= emin); ys = ys + gy0; xs = xs + gx0
    nx = gx[ys, xs]/mag[ys, xs]; ny = gy[ys, xs]/mag[ys, xs]
    gw, gh = gx1 - gx0, gy1 - gy0
    cands = []
    for r in np.arange(max(2.0, minD/2), maxD/2 + 0.01, 1.0):
        acc = np.zeros((gh, gw))
        for s in (1, -1):
            cx = np.rint(xs + s*r*nx).astype(int) - gx0; cy = np.rint(ys + s*r*ny).astype(int) - gy0
            ok = (cx >= 0) & (cx < gw) & (cy >= 0) & (cy < gh)
            np.add.at(acc, (cy[ok], cx[ok]), 1)
        a3 = np.zeros_like(acc)            # 3x3 sum: votes scatter by a pixel or so
        a3[1:-1, 1:-1] = sum(acc[1+dy:gh-1+dy, 1+dx:gw-1+dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        score = a3/(2*math.pi*r)
        for y, x in np.argwhere(score >= 0.6):
            if score[y, x] < score[max(0, y-1):y+2, max(0, x-1):x+2].max(): continue
            cands.append((score[y, x], x + gx0, y + gy0, r))
    cands.sort(reverse=True)
    out = []
    for sc, x, y, r in cands[:60]:
        cov = coverage(gx, gy, mag, x, y, r, emin)
        if cov < 0.75: continue
        key = cov*min(1.0, r/minD)/(1 + 3*math.hypot(x/W - anchor[0], y/H - anchor[1]))
        out.append((key, (float(x), float(y), 2.0*r)))
    out.sort(key=lambda o: -o[0])     # stable: equal keys keep the Hough order, as in Swift
    return [o[1] for o in out]

def bilinear(img, x, y):
    H, W = img.shape
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= W or y0 + 1 >= H: return None
    fx, fy = x - x0, y - y0
    return ((1-fx)*(1-fy)*img[y0, x0] + fx*(1-fy)*img[y0, x0+1] + (1-fx)*fy*img[y0+1, x0] + fx*fy*img[y0+1, x0+1])

def refine_rear(img, approx, strong=None, minpts=12):
    """Sub-pixel circle: on rays from the approximate centre (tee side skipped), the strongest
    radial edge with a parabolic peak, then a least-squares circle with outlier rejection.
    strong: keep only rays whose edge is at least this share of the strongest one."""
    cx0, cy0, d0 = approx
    rs = np.arange(0.2*d0, 0.85*d0, 0.5)
    pts = []
    for a in np.radians(np.arange(0, 360, 6)):
        if BOTTOM[0] <= a <= BOTTOM[1]: continue
        ux, uy = math.cos(a), math.sin(a)
        v = [bilinear(img, cx0 + ux*r, cy0 + uy*r) for r in rs]
        if any(q is None for q in v): continue
        g = np.abs(np.gradient(np.array(v, float), 0.5))
        i = int(np.argmax(g))
        if g[i] < 6 or i == 0 or i == len(g) - 1: continue
        den = g[i-1] - 2*g[i] + g[i+1]
        off = 0.5*(g[i-1] - g[i+1])/den if den < 0 else 0.0
        rr = rs[i] + 0.5*off
        pts.append((cx0 + ux*rr, cy0 + uy*rr, g[i]))
    if strong is not None and pts:
        gmax = max(p[2] for p in pts); pts = [p for p in pts if p[2] >= strong*gmax]
    if len(pts) < minpts: return None
    P = np.array([p[:2] for p in pts])
    keep = np.ones(len(P), bool)
    for _ in range(3):
        Q = P[keep]
        A = np.c_[2*Q[:, 0], 2*Q[:, 1], np.ones(len(Q))]
        sol = np.linalg.lstsq(A, (Q**2).sum(1), rcond=None)[0]
        cx, cy = sol[0], sol[1]; rad = math.sqrt(max(1e-9, sol[2] + cx*cx + cy*cy))
        res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - rad)
        keep = res <= max(0.6, 2.5*float(np.median(res[keep])))
        if keep.sum() < minpts: return None
    if np.median(res[keep]) > 0.04*rad + 0.3 or not (0.7*d0 < 2*rad < 1.3*d0): return None
    return cx, cy, 2*rad

def refine_lit(img, approx):
    """Face-on fallback refine. Only the lit arc has a real outline (the shaded side's edge is
    lost in the grass texture), so fit the circle to the strong radial edges alone; the lit
    part only hints at the size, so start from a few diameters."""
    for k in (1.0, 1.25, 0.8):
        r = refine_rear(img, (approx[0], approx[1], k*approx[2]), strong=0.5, minpts=8)
        if r: return r
    return None

