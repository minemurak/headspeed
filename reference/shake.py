"""Was the camera itself moved (bumped, or the phone picked up) when the trigger fired?
A swing changes part of the picture over a still background; a moved camera shifts all of it,
so a small whole-frame shift lines the frames up much better than no shift at all."""
import numpy as np

STEP, MAXS, MARGIN = 4, 4, 8          # sample grid (px), largest shift tried (px), border kept clear

def best_shift(ref, cur):
    """(dx, dy, diff at best shift, diff at no shift): mean |cur - ref| on a sparse grid."""
    H, W = ref.shape
    ys = np.arange(MARGIN, H - MARGIN, STEP); xs = np.arange(MARGIN, W - MARGIN, STEP)
    c = cur[np.ix_(ys, xs)].astype(float)
    best = None
    for dy in range(-MAXS, MAXS + 1):
        for dx in range(-MAXS, MAXS + 1):
            dv = float(np.abs(c - ref[np.ix_(ys + dy, xs + dx)].astype(float)).mean())
            if dx == 0 and dy == 0: d0 = dv
            if best is None or dv < best[2]: best = (dx, dy, dv)
    return best + (d0,)

def camera_moved(frames, trig):
    """True when, in at least 2 of the frames from the trigger to 8 after (every other one),
    a whole-frame shift explains the change: shifted diff < 0.7 x unshifted, and > 1.5 grey
    levels of change to explain. Reference: 12 frames before the trigger."""
    if trig < 12 or len(frames) < 2: return False
    ref = frames[trig - 12]
    hits = 0
    for k in range(trig, min(len(frames), trig + 9), 2):
        dx, dy, db, d0 = best_shift(ref, frames[k])
        if (dx, dy) != (0, 0) and d0 > 1.5 and db < 0.7*d0: hits += 1
    return hits >= 2
