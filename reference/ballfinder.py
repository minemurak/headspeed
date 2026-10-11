"""Reference for BallFinder.swift: finding the ball at rest (search on a 4x-downscaled frame,
then a full-resolution refine). search = face-on; search_rear / refine_rear = rear view."""
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


# ---- rear view: the ball may be darker than its surroundings (backlit) and sit on a tee ----

def box_mean(img, r):
    """Mean over a (2r+1)^2 box, clamped at the borders (integral image)."""
    H, W = img.shape
    I = np.zeros((H + 1, W + 1)); I[1:, 1:] = img.astype(float).cumsum(0).cumsum(1)
    ys = np.arange(H); xs = np.arange(W)
    y0 = np.clip(ys - r, 0, H); y1 = np.clip(ys + r + 1, 0, H)
    x0 = np.clip(xs - r, 0, W); x1 = np.clip(xs + r + 1, 0, W)
    S = I[y1][:, x1] - I[y0][:, x1] - I[y1][:, x0] + I[y0][:, x0]
    return S/((y1 - y0)[:, None]*(x1 - x0)[None, :])

def open_h(mask, k):
    """Morphological opening with a horizontal segment of k pixels: removes anything narrower
    than k (a tee under the ball, thin lines) and keeps wider shapes."""
    if k <= 1: return mask.copy()
    H, W = mask.shape
    c = np.zeros((H, W + 1), int); c[:, 1:] = mask.cumsum(1)
    er = np.zeros_like(mask)
    er[:, :W - k + 1] = (c[:, k:] - c[:, :W - k + 1]) == k      # run of k starting at x
    out = np.zeros_like(mask)
    for j in range(k): out[:, j:W - k + 1 + j] |= er[:, :W - k + 1]
    return out

def search_rear(img, guide, minD, maxD, anchor):
    """Round blob of either polarity in the guide (a backlit ball looks dark); several opening
    widths cut it loose from a tee. Picks the candidate nearest the anchor (x, y in 0..1)."""
    H, W = img.shape
    gx0, gx1 = max(0, int(guide[0]*W)), min(W, int(guide[2]*W))
    gy0, gy1 = max(0, int(guide[1]*H)), min(H, int(guide[3]*H))
    loc = box_mean(img, max(4, int(maxD*0.7)))
    best = None
    for pol in (1, -1):
        raw = np.zeros((H, W), bool)
        raw[gy0:gy1, gx0:gx1] = pol*(img[gy0:gy1, gx0:gx1] - loc[gy0:gy1, gx0:gx1]) >= 25
        for k in (2, 3, 5, 7):
            lo, hi = max(minD, k/0.6), min(maxD, max(k/0.3, 6))
            if lo > hi: continue
            for b in components(open_h(raw, k)):
                bw = b['maxx'] - b['minx'] + 1; bh = b['maxy'] - b['miny'] + 1
                if not (lo <= bw <= hi) or not (0.7*bw <= bh <= 1.3*bw): continue
                if b['area']/(bw*bh) < 0.6: continue
                cx = b['sx']/b['area']; cy = b['sy']/b['area']; d = (bw + bh)/2
                contrast = pol*(mean_in(img, cx, cy, 0, d*0.3) - mean_in(img, cx, cy, d*0.8, d*1.6))
                if contrast < 35: continue
                dist = math.hypot(cx/W - anchor[0], cy/H - anchor[1])
                if best is None or dist < best[1]: best = ((cx, cy, d), dist)
    return best[0] if best else None

def refine_rear(img, approx):
    """Sub-pixel circle from the ball's upper edge (a tee or the ground can touch the lower
    half): rays from the approximate centre, threshold crossings, least-squares circle."""
    cx0, cy0, d0 = approx
    H, W = img.shape
    inside = mean_in(img, cx0, cy0, 0, d0*0.25); ring = mean_in(img, cx0, cy0, d0*0.9, d0*1.4)
    if abs(inside - ring) < 30: return None
    thr = (inside + ring)/2; pol = 1 if inside > ring else -1
    def at(x, y):
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        if x0 < 0 or y0 < 0 or x0 + 1 >= W or y0 + 1 >= H: return None
        fx, fy = x - x0, y - y0
        return ((1-fx)*(1-fy)*img[y0, x0] + fx*(1-fy)*img[y0, x0+1] + (1-fx)*fy*img[y0+1, x0] + fx*fy*img[y0+1, x0+1])
    pts = []
    for a in np.radians(np.arange(-170, -9, 8)):        # upper half (image y grows downward)
        ux, uy = math.cos(a), math.sin(a)
        prev = None; r = 0.15*d0
        while r < 1.0*d0:
            v = at(cx0 + ux*r, cy0 + uy*r)
            if v is None: break
            s = pol*(v - thr)
            if prev is not None and prev[1] > 0 >= s:
                rr = prev[0] + (r - prev[0])*prev[1]/(prev[1] - s)
                pts.append((cx0 + ux*rr, cy0 + uy*rr)); break
            prev = (r, s); r += 0.25
    if len(pts) < 10: return None
    P = np.array(pts)
    A = np.c_[2*P[:, 0], 2*P[:, 1], np.ones(len(P))]
    sol = np.linalg.lstsq(A, (P**2).sum(1), rcond=None)[0]
    cx, cy = sol[0], sol[1]; rad = math.sqrt(sol[2] + cx*cx + cy*cy)
    res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - rad)
    if np.median(res) > 0.06*rad + 0.3 or not (0.7*d0 < 2*rad < 1.3*d0): return None
    return cx, cy, 2*rad, pol
