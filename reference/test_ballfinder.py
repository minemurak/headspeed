"""Rear-view ball search on synthetic frames: bright or backlit (dark) ball, on a tee,
with bright sky and round sky gaps elsewhere (python3 test_ballfinder.py)."""
import sys, math
import numpy as np
from PIL import Image
from sim_rear import render
from ballfinder import search_rear, refine_rear

GUIDE = (0.25, 0.45, 0.75, 0.97)
failures = []
for ball_val, sky_val, seed, cam_dist in ((232, 170, 0, 2.6), (45, 240, 1, 2.6), (60, 250, 2, 3.2), (220, 250, 3, 2.2)):
    s = render(ball_val=ball_val, sky_val=sky_val, seed=seed, cam_dist=cam_dist, body=False)
    f = s['frames'][0].astype(float)
    bu, bv, d = s['ball']
    # tee under the ball, a bright round gap in a dark net up in the sky, two flat (elliptical) holes in the mat
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    f[(np.abs(xx - bu) < 0.15*d) & (yy > bv) & (yy < bv + 0.9*d) & ((xx - bu)**2 + (yy - bv)**2 > (d/2)**2)] = 150
    f[(np.abs(xx - 680) < 18) & (np.abs(yy - 270) < 18)] = 40
    f[(xx - 680)**2 + (yy - 270)**2 < 10**2] = 250
    for hx in (560, 600):
        f[((xx - hx)/(1.2*d))**2 + ((yy - bv - 2*d)/(0.45*d))**2 < 1] = 20
    full = np.array(Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).resize((1920, 1080), Image.BILINEAR)).astype(float)
    small = full[::4, ::4]
    c = search_rear(small, GUIDE, 14/4, 70/4, (0.5, 0.75))
    fix = refine_rear(full, (c[0]*4 + 1.5, c[1]*4 + 1.5, c[2]*4)) if c else None
    tu, tv, td = bu*2 + 0.5, bv*2 + 0.5, d*2
    ok = fix is not None and math.hypot(fix[0] - tu, fix[1] - tv) < 1.0 and abs(fix[2]/td - 1) < 0.04
    print(dict(ball=ball_val, sky=sky_val), 'truth', (round(tu, 1), round(tv, 1), round(td, 1)),
          '->', None if fix is None else tuple(round(v, 2) for v in fix), 'OK' if ok else 'FAIL')
    if not ok: failures.append(seed)
print('OK' if not failures else f'{len(failures)} failed')
sys.exit(1 if failures else 0)
