"""Camera-moved check (python3 test_shake.py): a real clip where the phone was bumped after the
swing must be caught; synthetic swings from a still camera must not."""
import os, sys
import numpy as np
from PIL import Image
from shake import camera_moved
from sim_rear import render

failures = []
# fixtures/rear_bumped.jpg: 6 strip frames stacked (12 before the trigger, then trigger +0, +2 ... +8)
img = np.asarray(Image.open(os.path.join(os.path.dirname(__file__) or '.', 'fixtures', 'rear_bumped.jpg')))
h = img.shape[0] // 6
sel = [img[i*h:(i+1)*h] for i in range(6)]
frames = [sel[0]]*12 + [sel[1 + j//2] for j in range(9)]       # trigger at index 12
moved = camera_moved(frames, 12)
print('real bumped phone ->', moved); failures += [] if moved else ['real bumped']
still = camera_moved([sel[0]]*21, 12)
print('real still ->', still); failures += ['real still'] if still else []

for kw in (dict(), dict(handed=-1, seed=1), dict(cam_x=0.8, seed=8, handed=-1), dict(noise=7, seed=4),
           dict(hs=48, seed=2, cam_h=1.0, look=(0.0, 0.3, 3.0))):
    s = render(**kw); fr, tr = s['frames'], s['trig']
    m = camera_moved(fr, tr)
    print('synthetic swing', kw, '->', m); failures += [kw] if m else []

# on the real (textured) background: a dark shaft and a body-sized blob moving through a still
# picture is a swing; the same picture knocked a few pixels is a moved camera
bg = sel[0].astype(float); H, W = bg.shape
yy, xx = np.mgrid[0:H, 0:W]
swing = []
for k in range(21):
    f = bg.copy()
    a = 0.25*k; x0 = 300 + 25*k
    f[np.abs((xx - x0)*np.cos(a) + (yy - 150)*np.sin(a)) < 4] = 30          # shaft
    f[((xx - 820 - 3*k)/60)**2 + ((yy - 180)/150)**2 < 1] = 50             # body turning
    swing.append(np.clip(f, 0, 255).astype(np.uint8))
m = camera_moved(swing, 12); print('real background + swing ->', m); failures += ['swing'] if m else []
for dx, dy in ((3, 2), (-2, 1), (1, -3)):
    knocked = [sel[0]]*12 + [np.roll(np.roll(sel[0], dx, axis=1), dy, axis=0)]*9
    m = camera_moved(knocked, 12); print('real background knocked', (dx, dy), '->', m); failures += [] if m else [(dx, dy)]
print('OK' if not failures else f'{len(failures)} failed: {failures}')
sys.exit(1 if failures else 0)
