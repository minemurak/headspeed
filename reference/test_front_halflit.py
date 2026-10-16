"""Face-on analyzer on a real driver swing where the sun is behind the ball (python3 test_front_halflit.py).
fixtures/front_halflit_swing.jpg: 29 strip frames stacked: five background frames (trigger - 110,
-95, -80, -65, -50), trigger - 8 ... trigger + 7, then trigger + 32 ... + 39 (ball gone). Only the
ball's far-right part was lit, and the app measured that cap (d 16.9 instead of ~30), which put
head and ball speed 1.8x too high (HS 64 for a golfer of about 40). The analyzer must size the ball
from its silhouette once it has left, and keep the shaft off the head's leading edge."""
import os, sys
import numpy as np
from PIL import Image
from analyzer import analyze

img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__) or '.', 'fixtures', 'front_halflit_swing.jpg')))
h = img.shape[0] // 29
sel = [img[i*h:(i+1)*h] for i in range(29)]
trig = 198
frames = [sel[0]]*240
for j, k in enumerate((88, 103, 118, 133, 148)): frames[k] = sel[j]
for j, k in enumerate(range(190, 206)): frames[k] = sel[5 + j]
for j, k in enumerate(range(230, 238)): frames[k] = sel[21 + j]
for k in range(238, 240): frames[k] = sel[-1]
times = np.arange(240)/240.0
r = analyze(frames, times, (524.0, 236.2, 16.85), trig, club='1W', pitch=4.5)   # as the app exported it
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, np.ndarray)) or k == 'notes'})
# by hand: the ball is ~30 px across; the head's leading edge advances ~100-120 px per frame
# before contact, the ball ~158 px per frame after
ok = ('ballCorrected' in r and abs(r['ballCorrected'][2] - 30) < 2.5 and 'hs' in r and 32 < r['hs'] < 41
      and 'bs' in r and 50 < r['bs'] < 58 and 1.35 < r['smash'] < 1.6)
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
