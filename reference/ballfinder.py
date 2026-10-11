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
    """Current (percentile threshold) version."""
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

def pick(img, mask, minD, maxD):
    best = None
    for b in components(mask):
        d = 2*math.sqrt(b['area']/math.pi)
        if not (minD <= d <= maxD): continue
        bw = b['maxx'] - b['minx'] + 1; bh = b['maxy'] - b['miny'] + 1
        if max(bw, bh)/min(bw, bh) > 1.4 or b['area']/(bw*bh) < 0.6: continue
        cx = b['sx']/b['area']; cy = b['sy']/b['area']
        contrast = mean_in(img, cx, cy, 0, d*0.3) - mean_in(img, cx, cy, d*0.8, d*1.6)
        if contrast < 40: continue
        score = contrast*b['area']/(bw*bh)
        if best is None or score > best[1]: best = ((cx, cy, d), score)
    return best[0] if best else None


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
    best = None
    for sc, x, y, r in cands[:60]:
        cov = coverage(gx, gy, mag, x, y, r, emin)
        if cov < 0.75: continue
        key = cov*min(1.0, r/minD)/(1 + 3*math.hypot(x/W - anchor[0], y/H - anchor[1]))
        if best is None or key > best[1]: best = ((float(x), float(y), 2.0*r), key)
    return best[0] if best else None

def bilinear(img, x, y):
    H, W = img.shape
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= W or y0 + 1 >= H: return None
    fx, fy = x - x0, y - y0
    return ((1-fx)*(1-fy)*img[y0, x0] + fx*(1-fy)*img[y0, x0+1] + (1-fx)*fy*img[y0+1, x0] + fx*fy*img[y0+1, x0+1])

def refine_rear(img, approx):
    """Sub-pixel circle: on rays from the approximate centre (tee side skipped), the strongest
    radial edge with a parabolic peak, then a least-squares circle with outlier rejection."""
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
        pts.append((cx0 + ux*rr, cy0 + uy*rr))
    if len(pts) < 12: return None
    P = np.array(pts)
    keep = np.ones(len(P), bool)
    for _ in range(3):
        Q = P[keep]
        A = np.c_[2*Q[:, 0], 2*Q[:, 1], np.ones(len(Q))]
        sol = np.linalg.lstsq(A, (Q**2).sum(1), rcond=None)[0]
        cx, cy = sol[0], sol[1]; rad = math.sqrt(max(1e-9, sol[2] + cx*cx + cy*cy))
        res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - rad)
        keep = res <= max(0.6, 2.5*float(np.median(res[keep])))
        if keep.sum() < 12: return None
    if np.median(res[keep]) > 0.04*rad + 0.3 or not (0.7*d0 < 2*rad < 1.3*d0): return None
    return cx, cy, 2*rad
