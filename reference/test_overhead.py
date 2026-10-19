"""上からモード (overhead): face-on, looking down steeply (python3 test_overhead.py).
From above, the image's vertical motion mixes rising with moving toward or away from the camera,
so the analyzer uses only the along-target motion and gives no launch or attack angle.
Synthetic swings at 45-60 deg pitch, some balls starting off line (which the face-on mode would
read as a wrong launch angle and speed), then the real swing of test_front_edge.py (pitch 56 deg)."""
import math, os, sys
import numpy as np
from PIL import Image
from sim import render
from analyzer import analyze

cases = [dict(hs=42, bs=62, launch=13, attack=0, d=24, seed=11, pitch=56),
         dict(hs=38, bs=55, launch=12, attack=-1, d=22, direction=-1, seed=12, pitch=50, offline=6),
         dict(hs=46, bs=68, launch=14, attack=1, d=26, seed=13, pitch=60, offline=-5),
         dict(hs=35, bs=50, launch=13, attack=0, d=20, seed=14, pitch=45, offline=8, noise=6),
         dict(hs=36, bs=48, launch=17, attack=-4, d=22, seed=15, pitch=55, offline=4, club='7I')]
failures = []
for c in cases:
    s = render(**{k: v for k, v in c.items() if k != 'club'})
    r = analyze(s['frames'], s['times'], s['ball'], s['trig'], club=c.get('club', '1W'), pitch=c['pitch'], overhead=True)
    f = lambda k: (f"{r[k]:.2f}" if k in r else '-')
    print({k: c[k] for k in ('hs', 'bs', 'launch', 'pitch', 'offline') if k in c}, '->',
          'hs', f('hs'), 'bs', f('bs'), 'smash', f('smash'), r.get('headFrames'), r.get('ballFrames'), r['notes'])
    checks = [('hs', lambda v: abs(v/c['hs'] - 1) < 0.03), ('bs', lambda v: abs(v/c['bs'] - 1) < 0.03),
              ('smash', lambda v: abs(v - c['bs']/c['hs']) < 0.06)]
    for k, ok in checks:
        if k not in r or not ok(r[k]): failures.append((c, k, r.get(k)))
    if 'launch' in r or 'attack' in r: failures.append((c, 'launch/attack given', None))

img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__) or '.', 'fixtures', 'front_edge_exit.jpg')))
h = img.shape[0] // 21
sel = [img[i*h:(i+1)*h].copy() for i in range(21)]
frames = [sel[0]]*240
for j, k in enumerate((88, 103, 118, 133, 148)): frames[k] = sel[j]
for j, k in enumerate(range(190, 206)): frames[k] = sel[5 + j]
for k in range(206, 240): frames[k] = sel[-1]
r = analyze(frames, np.arange(240)/240.0, (781.94, 333.93, 28.71), 198, club='1W',
            pitch=math.degrees(0.9858), overhead=True)
print('real swing ->', {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
if not (34 < r.get('hs', 0) < 41 and 45 < r.get('bs', 0) < 58 and 'launch' not in r and 'attack' not in r):
    failures.append(('real swing', r.get('hs'), r.get('bs')))

for x in failures: print('FAIL', x)
print('OK' if not failures else f'{len(failures)} check(s) failed')
sys.exit(1 if failures else 0)
