"""Face-on ball tracking must ignore things that differ from the background but do not move
(python3 test_front_static.py). fixtures/front_static_blob.jpg: 29 strip frames stacked as in
test_front_hosel.py, from a real driver swing. After the hit a bright patch above the ball
(the golfer's glove/trousers, shifted since the background frames) sits still at (477, 180).
On the phone the ball's first flight frame fell just under the size limit, so the tracker took
that patch as the ball's first position: 83 m/s at -26 degrees. Here the ball is painted out of
that frame to reproduce it; the ball must still come out ~55 m/s at ~10 degrees."""
import os, sys
import numpy as np
from PIL import Image
from analyzer import analyze

img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__) or '.', 'fixtures', 'front_static_blob.jpg')))
h = img.shape[0] // 29
sel = [img[i*h:(i+1)*h].copy() for i in range(29)]
trig = 198
frames = [sel[0]]*240
for j, k in enumerate((88, 103, 118, 133, 148)): frames[k] = sel[j]
for j, k in enumerate(range(190, 206)): frames[k] = sel[5 + j]
for j, k in enumerate(range(230, 238)): frames[k] = sel[21 + j]
for k in range(238, 240): frames[k] = sel[-1]
# the ball leaving the tee in the impact frame, missed on the phone: paint the background over it
bg = np.median(np.stack(sel[:5]), axis=0).astype(np.uint8)
f = frames[198].copy(); f[255:295, 495:545] = bg[255:295, 495:545]; frames[198] = f
times = np.arange(240)/240.0
r = analyze(frames, times, (456.91, 288.82, 20.65), trig, club='1W', pitch=0.7)
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, np.ndarray)) or k == 'notes'})
ok = 'bs' in r and 50 < r['bs'] < 60 and 5 < r['launch'] < 16 and 1.3 < r.get('smash', 0) < 1.7
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
