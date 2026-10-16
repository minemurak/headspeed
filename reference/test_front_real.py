"""Face-on analyzer on a real driver swing (python3 test_front_real.py).
fixtures/front_sunny.jpg: 21 strip frames stacked: five background frames (impact - 110, -95, -80,
-65, -50) then impact - 8 ... impact + 7. Sunny tee: the golfer's right foot and shadows sit next to
the ball and shift during the downswing; the head must still be tracked."""
import os, sys
import numpy as np
from PIL import Image
from analyzer import analyze

img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__) or '.', 'fixtures', 'front_sunny.jpg')))
h = img.shape[0] // 21
sel = [img[i*h:(i+1)*h] for i in range(21)]
trig = 198
frames = [sel[0]]*240
for j, k in enumerate((88, 103, 118, 133, 148)): frames[k] = sel[j]
for j, k in enumerate(range(190, 206)): frames[k] = sel[5 + j]
for k in range(206, 240): frames[k] = sel[-1]
times = np.arange(240)/240.0
r = analyze(frames, times, (502.6, 212.6, 15.7), trig, club='1W')
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, np.ndarray)) or k == 'notes'})
# hand-measured from the frames: the head's leading edge advances ~60 px per frame just before
# impact, the ball ~111 px per frame just after
ok = 'headPxS' in r and 13000 < r['headPxS'] < 16000 and 'ballPxS' in r and 25500 < r['ballPxS'] < 28500
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
