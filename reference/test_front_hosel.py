"""Face-on analyzer on a real driver swing where the hosel and shaft lead the face
(python3 test_front_hosel.py). fixtures/front_hosel_swing.jpg: 29 strip frames stacked: five
background frames (trigger - 110, -95, -80, -65, -50), trigger - 8 ... trigger + 7, then
trigger + 32 ... + 39. The leading-edge column caught the hosel ahead of the face in the early
frames, so the fitted head speed came out 34 m/s; the head's dark body moves ~110 px per frame
(about 43 m/s at this ball size). Matching the whole head between frames must see that."""
import os, sys
import numpy as np
from PIL import Image
from analyzer import analyze

img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__) or '.', 'fixtures', 'front_hosel_swing.jpg')))
h = img.shape[0] // 29
sel = [img[i*h:(i+1)*h] for i in range(29)]
trig = 198
frames = [sel[0]]*240
for j, k in enumerate((88, 103, 118, 133, 148)): frames[k] = sel[j]
for j, k in enumerate(range(190, 206)): frames[k] = sel[5 + j]
for j, k in enumerate(range(230, 238)): frames[k] = sel[21 + j]
for k in range(238, 240): frames[k] = sel[-1]
times = np.arange(240)/240.0
r = analyze(frames, times, (468.54, 335.42, 26.35), trig, club='1W', pitch=1.9)   # as the app exported it
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, np.ndarray)) or k == 'notes'})
ok = r.get('headMatched') and 'hs' in r and 39 < r['hs'] < 47 and 'bs' in r and 50 < r['bs'] < 55 and 'ballCorrected' not in r
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
