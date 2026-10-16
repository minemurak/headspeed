"""Face-on ball search on real frames from a sunny range (fixtures/front_halflit*.jpg): the sun
is behind the ball, so only its near-left part is lit and the rest is as dark as the sunlit grass.
The bright-blob search misses it; the lit fallback (search_lit + refine_lit) must find it, and
not the white shoe toe beside it (python3 test_front_ball.py)."""
import os, math, sys
import numpy as np
from PIL import Image
from ballfinder import search, refine, search_lit, refine_lit

GUIDE = (0.15, 0.30, 0.85, 0.97)
TRUTH = (1041, 578, 56)   # measured by hand: centre and diameter of the ball's outline
here = os.path.dirname(__file__) or '.'
failures = []

def locate(full, guide, anchor=None):
    """As CaptureEngine.searchBall does face-on: the bright blob first, the lit fallback if
    that finds nothing or its blob does not refine."""
    s = full[::4, ::4]
    for find, fine in ((search, refine), (lambda *a: search_lit(*a, anchor), refine_lit)):
        c = find(s, guide, 12/4, 110/4)
        if not c: continue
        x, y, d = c[0]*4 + 1.5, c[1]*4 + 1.5, c[2]*4
        h = int(d*2.1) + 2
        x0, y0 = max(0, int(x) - h), max(0, int(y) - h)
        r = fine(full[y0:int(y) + h, x0:int(x) + h].astype(float), (x - x0, y - y0, d))
        if r: return r[0] + x0, r[1] + y0, r[2]
    return None

for name in ('front_halflit', 'front_halflit_shoe'):
    full = np.asarray(Image.open(f'{here}/fixtures/{name}.jpg'))
    s = full[::4, ::4]
    if search(s, GUIDE, 12/4, 110/4) is not None:
        print(name, 'bright search now finds something; check the fixture still covers the fallback')
    tx, ty = TRUTH[0]/1920, TRUTH[1]/1080
    for label, guide, anchor in (('auto', GUIDE, None), ('tap', (tx - .05, ty - .07, tx + .05, ty + .07), (tx, ty))):
        fix = locate(full, guide, anchor)
        ok = fix is not None and math.hypot(fix[0] - TRUTH[0], fix[1] - TRUTH[1]) < 5 and abs(fix[2]/TRUTH[2] - 1) < 0.05
        print(name, label, '->', None if fix is None else tuple(round(v, 1) for v in fix), 'OK' if ok else 'FAIL')
        if not ok: failures.append((name, label))
sys.exit(1 if failures else 0)
