"""Reference for BallLocator.swift / BallLock.swift: from frames of the ball at rest to a locked ball.

Every fix so far patched one more look of the ball (bright, dark, split background, half lit) or
one more thing mistaken for it (sky gap, sticker, white shoe, trousers). A single frame cannot
always tell them apart, so the lock now leans on what does not depend on the look:

- all candidates, then a choice: every detector returns everything that passes, and the choice
  is made on all of them (an all-round outline beats a bright blob without one; face-on, the ball
  lies in front of the feet, so lower wins; a tap or the spot of the last measured ball wins).
- geometry: the ball lies on the ground, so it cannot sit far above the horizon (camera tilt from
  the motion sensor, field of view from the format). Trousers, a sky gap or a sticker high on the
  ball dispenser are dropped.
- time: once chosen, the ball is followed on every few frames by matching its own look (cheap,
  and the same for any look of the ball) and locked once it has stayed put for LOCK_SPAN
  seconds; one missed look does not start over. Its size is the median over those looks.
- memory: the next ball usually goes on the same tee, so the spot of the last measured ball is
  looked at first.
"""
import math
import numpy as np
from ballfinder import (search_all, refine, search_lit_all, refine_lit, search_rear_all, refine_rear)

LOCK_HITS = 4       # fixes needed ...
LOCK_SPAN = 0.25    # ... spread over at least this many seconds
MAX_MISSES = 2      # consecutive looks without the ball that start over
TOL_POS = 0.2       # x ball diameter: how far a fix may sit from the run's first one
STILL = 12.0        # grey levels: mean change of the ball's look allowed within a run
GONE = 24.0         # grey levels: more change than this and the ball is not there (covered or taken)


def rnd(v):
    """Round half away from zero, as Swift's rounded()."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))

def front_guide():
    return (0.15, 0.30, 0.85, 0.97)

def rear_guide():
    return (0.25, 0.45, 0.75, 0.97)

def pick_area(p):
    """Search area around a tapped point (normalized x, y)."""
    return (max(0, p[0] - 0.05), max(0, p[1] - 0.07), min(1, p[0] + 0.05), min(1, p[1] + 0.07))

def horizon_row(W, H, fov_deg, pitch):
    """Image row of the horizon for a camera looking down by `pitch` radians."""
    f = W/2/math.tan(math.radians(fov_deg)/2)
    return H/2 - f*math.tan(pitch)

def crop_refine(full, approx, fn):
    """Refine on a full-resolution crop around approx = (x, y, d)."""
    x, y, d = approx
    H, W = full.shape
    half = int(d*2.1) + 2
    x0, x1 = max(0, int(x) - half), min(W, int(x) + half)
    y0, y1 = max(0, int(y) - half), min(H, int(y) + half)
    if x1 - x0 <= 4 or y1 - y0 <= 4: return None
    r = fn(full[y0:y1, x0:x1].astype(float), (x - x0, y - y0, d))
    return None if r is None else (r[0] + x0, r[1] + y0, r[2])

def refiner(kind):
    return {'round': refine, 'plain': refine, 'lit': refine_lit, 'rear': refine_rear}[kind]

def locate(full, rear, anchor=None, home=None, horizon=None):
    """Every ball candidate in the frame, best first: [(x, y, d, kind)], full-resolution pixels.
    anchor: tapped point (normalized); home: (x, y, d, kind) of the last measured ball, looked
    at directly first (a full search may not rank it among its top candidates);
    horizon: image row of the horizon, or None when the tilt is unknown."""
    H, W = full.shape
    guide = pick_area(anchor) if anchor is not None else (rear_guide() if rear else front_guide())
    f = 2 if rear else 4
    small = full[::f, ::f]
    def up(c): return (c[0]*f + (f - 1)/2, c[1]*f + (f - 1)/2, c[2]*f)
    found = []
    def add(r, kind):
        if r is None: return
        if any(math.hypot(r[0] - q[0], r[1] - q[1]) < 0.3*max(r[2], q[2]) for q in found): return
        found.append((r[0], r[1], r[2], kind))
    if home is not None and anchor is None:     # a tap says which ball, whatever was measured before
        add(track(full, home[:3], home[3]), home[3])
    if rear:
        for c in search_rear_all(small, guide, 14/f, 70/f, anchor if anchor is not None else (0.5, 0.75)):
            add(crop_refine(full, up(c), refine_rear), 'rear')
    else:
        for c in search_all(small, guide, 12/f, 110/f):
            r = crop_refine(full, up(c), refine)
            if r is None: continue
            # an all-round outline (tee side aside): a ball has one, a shoe toe or a trouser leg not
            add(r, 'round' if crop_refine(full, r, refine_rear) is not None else 'plain')
        if not any(q[3] == 'round' for q in found):
            # a ball lit only on one side is a crescent to the bright search
            for c in search_lit_all(small, guide, 12/f, 110/f, anchor):
                add(crop_refine(full, up(c), refine_lit), 'lit')
    if horizon is not None:
        # on the ground: not far above the horizon (margin for tilt error and a camera set low)
        found = [q for q in found if q[1] >= horizon - (0.15*H + 3*q[2])]
    def at_home(q):
        return home is not None and anchor is None and math.hypot(q[0] - home[0], q[1] - home[1]) < 2*home[2] and abs(q[2]/home[2] - 1) < 0.25
    rank = {'round': 0, 'rear': 0, 'lit': 1, 'plain': 2}
    order = []
    for i, q in enumerate(found):
        if anchor is not None: key = math.hypot(q[0]/W - anchor[0], q[1]/H - anchor[1])
        elif rear or q[3] == 'lit': key = i     # the detector's own ranking
        else: key = -q[1]                       # face-on bright blobs: lowest first
        order.append(((0 if at_home(q) else 1, rank[q[3]], key), q))
    order.sort(key=lambda o: o[0])
    return [o[1] for o in order]

def track(full, fix, kind):
    """The ball near fix = (x, y, d) on a new frame by its detector's refine, or None."""
    r = crop_refine(full, fix[:3], refiner(kind))
    if r is None or math.hypot(r[0] - fix[0], r[1] - fix[1]) > 0.5*fix[2]: return None
    return r

class Template:
    """The ball and a little of what is round it (a square 1.2 d wide, 11 x 11 samples) as first
    seen, found again on a later frame within a quarter ball either way. Works for any look of
    the ball (bright, dark, half lit, split background), since it only compares with itself."""
    N = 11
    def __init__(self, full, x, y, d):
        self.x, self.y, self.d = x, y, d
        h = (self.N - 1)/2
        self.offs = [(rnd((i - h)/h*0.6*d), rnd((j - h)/h*0.6*d)) for j in range(self.N) for i in range(self.N)]
        self.ref = self.sample(full, rnd(x), rnd(y))

    def sample(self, full, cx, cy):
        H, W = full.shape
        return np.array([float(full[min(H - 1, max(0, cy + v)), min(W - 1, max(0, cx + u))]) for u, v in self.offs])

    def match(self, full):
        """(x, y, change): the best place and the mean |difference| there, in grey levels."""
        R = max(1, int(0.25*self.d))
        cx, cy = rnd(self.x), rnd(self.y)
        best = None
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                c = float(np.abs(self.sample(full, cx + dx, cy + dy) - self.ref).mean())
                if best is None or c < best[2]: best = (self.x + dx, self.y + dy, c)
        return best

class BallLock:
    """When to lock, from one look at a time: update(t, fix, change) with fix = (x, y, d) or None
    (not seen) and change = how much the ball's look differs from the run's first look (grey
    levels). Returns the locked ball (medians of the run) once it has been seen LOCK_HITS times
    over LOCK_SPAN seconds without moving or changing; MAX_MISSES looks in a row without it
    start over, and so does a move or a change."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.run = []          # (t, fix)
        self.misses = 0

    def update(self, t, fix, change=0.0):
        if fix is None:
            self.misses += 1
            if self.misses >= MAX_MISSES: self.reset()
            return None
        self.misses = 0
        if self.run:
            x0, y0, d0 = self.run[0][1][:3]
            if math.hypot(fix[0] - x0, fix[1] - y0) > TOL_POS*d0 or change > STILL: self.reset()
        self.run.append((t, fix))
        if len(self.run) >= LOCK_HITS and t - self.run[0][0] >= LOCK_SPAN - 1e-9:
            def med(v):
                s = sorted(v); n = len(s)
                return s[n//2] if n % 2 else (s[n//2 - 1] + s[n//2])/2
            return tuple(med([f[k] for _, f in self.run]) for k in range(3))
        return None

class Follower:
    """From a chosen candidate to a lock, one frame at a time (CaptureEngine does the same with
    BallTracker): the template says where the ball is and whether it stayed put, the detector's
    refine measures its size where it can; the lock takes the median size of the run."""
    def __init__(self, full, cand):
        self.kind = cand[3]
        self.d = cand[2]
        self.tpl = Template(full, *cand[:3])
        self.lock = BallLock()

    def step(self, t, full):
        """('locked', (x, y, d)), ('following', (x, y, d)) or ('lost', None)."""
        x, y, change = self.tpl.match(full)
        if change > GONE:
            self.lock.update(t, None)
            return ('lost', None) if not self.lock.run and self.lock.misses == 0 else ('following', None)
        r = track(full, (x, y, self.d), self.kind)
        d = r[2] if r is not None and math.hypot(r[0] - x, r[1] - y) < 0.25*self.d else self.d
        n0 = len(self.lock.run)
        got = self.lock.update(t, (x, y, d), change)
        if len(self.lock.run) <= n0 and n0 > 0:
            # started over (moved or changed): look at it afresh from here
            self.tpl = Template(full, x, y, self.d)
        if got is not None: return 'locked', got
        return 'following', (x, y, d)
