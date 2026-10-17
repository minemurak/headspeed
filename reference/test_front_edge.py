"""A face-on ball that leaves the picture after one frame must still give a ball speed
(python3 test_front_edge.py). fixtures/front_edge_exit.jpg: 21 strip frames stacked (5 background
frames, then 190..205) from a real driver swing filmed from above (pitch 56 deg), ball near the
right edge. The ball is in flight in frame 198 only (~136 px from the tee) and is out of the
picture in 199, so the tracker used to give up. With the head's contact time it comes out
~55 m/s, smash ~1.5."""
import math, os, sys
import numpy as np
from PIL import Image
from analyzer import analyze

img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__) or '.', 'fixtures', 'front_edge_exit.jpg')))
h = img.shape[0] // 21
sel = [img[i*h:(i+1)*h].copy() for i in range(21)]
trig = 198
frames = [sel[0]]*240
for j, k in enumerate((88, 103, 118, 133, 148)): frames[k] = sel[j]
for j, k in enumerate(range(190, 206)): frames[k] = sel[5 + j]
for k in range(206, 240): frames[k] = sel[-1]
times = np.arange(240)/240.0
r = analyze(frames, times, (781.94, 333.93, 28.71), trig, club='1W', pitch=math.degrees(0.9858))
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, np.ndarray)) or k == 'notes'})
ok = (r.get('ballFrames') == 1 and 50 < r['bs'] < 61 and 34 < r.get('hs', 0) < 41
      and 1.3 < r.get('smash', 0) < 1.65 and any('画面外' in n for n in r['notes']))
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
