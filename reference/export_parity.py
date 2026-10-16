"""Writes frames and the reference's answers on them, for the Swift port to be checked against
(Tests/AnalysisTests/main.swift, run in CI on macOS):
    python3 export_parity.py OUTDIR
Each frame is OUTDIR/<name>.gray (raw 8-bit, row-major); OUTDIR/cases.json lists the frames
(width, height), the locate calls with their results, follower runs and BallLock runs."""
import json, math, os, sys
import numpy as np
from PIL import Image
from balllock import locate, Follower, BallLock, horizon_row

here = os.path.dirname(__file__) or '.'
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
frames = {}

def put(name, img):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    img.tofile(os.path.join(out, name + '.gray'))
    frames[name] = dict(width=int(img.shape[1]), height=int(img.shape[0]))
    return img

def fixture(name):
    return put(name, np.asarray(Image.open(f'{here}/fixtures/{name}.jpg')))

def cand(q):
    return dict(x=float(q[0]), y=float(q[1]), d=float(q[2]), kind=q[3])

# synthetic face-on frame: a white shoe toe above the ball (as test_balllock.py), and a sky gap
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:1080, 0:1920]
g = 105 + 10*np.sin(xx/13.0)*np.cos(yy/17.0) + rng.normal(0, 7, (1080, 1920))
g[((xx - 760)/44)**2 + ((yy - 640)/32)**2 < 1] = 245
g[(xx - 1010)**2 + (yy - 800)**2 < 23**2] = 228
g[(np.abs(xx - 1010) < 4) & (yy > 823) & (yy < 850)] = 200
put('synth_shoe', np.clip(g, 0, 255))
rng = np.random.default_rng(1)
g = 105 + 10*np.sin(xx/13.0)*np.cos(yy/17.0) + rng.normal(0, 7, (1080, 1920))
g[(xx - 900)**2 + (yy - 380)**2 < 20**2] = 235
put('synth_sky', np.clip(g, 0, 255))

imgs = {n: fixture(n) for n in ('rear_backlit', 'rear_split', 'rear_split2', 'rear_two_balls', 'front_halflit', 'front_halflit_shoe')}
imgs['synth_shoe'] = np.fromfile(os.path.join(out, 'synth_shoe.gray'), np.uint8).reshape(1080, 1920)
imgs['synth_sky'] = np.fromfile(os.path.join(out, 'synth_sky.gray'), np.uint8).reshape(1080, 1920)

locates = []
def L(frame, rear, anchor=None, home=None, horizon=None):
    r = locate(imgs[frame], rear, anchor=anchor, home=home, horizon=horizon)
    locates.append(dict(frame=frame, rear=rear, anchor=anchor and list(anchor),
                        home=home and cand(home), horizon=horizon, result=[cand(q) for q in r[:4]]))
    print(frame, rear, anchor, home, horizon, '->', [(round(q[0], 1), round(q[1], 1), round(q[2], 1), q[3]) for q in r[:2]])

for n in ('rear_backlit', 'rear_split', 'rear_split2'):
    L(n, True)
    L(n, True, horizon=horizon_row(1920, 1080, 62.0, math.radians(3)))
L('rear_two_balls', True, anchor=(900/1920, 745/1080))
L('rear_two_balls', True, home=(916.5, 735.0, 30.6, 'rear'))
L('rear_two_balls', True, home=(1053.5, 698.0, 31.6, 'rear'))
for n in ('front_halflit', 'front_halflit_shoe'):
    L(n, False)
    L(n, False, anchor=(1041/1920, 578/1080))
L('synth_shoe', False)
L('synth_shoe', False, horizon=horizon_row(1920, 1080, 70.0, math.radians(2)))
L('synth_sky', False)
L('synth_sky', False, horizon=horizon_row(1920, 1080, 70.0, math.radians(-6)))

# follower runs on still / moved / emptied frames (no noise, so both sides see the same pixels)
follows = []
for name, c, truth in (('rear_split', (888.0, 674.0, 33.9, 'rear'), (886, 676, 34)),
                       ('front_halflit', (1040.2, 576.9, 55.8, 'lit'), (1041, 578, 56))):
    base = imgs[name]
    shift = int(0.24*truth[2]) + 1
    put(name + '_moved', np.roll(base, shift, axis=1))
    gone = base.copy()
    gone[(xx - truth[0])**2 + (yy - truth[1])**2 < (0.7*truth[2])**2] = 90
    put(name + '_gone', gone)
    imgs[name + '_moved'] = np.roll(base, shift, axis=1); imgs[name + '_gone'] = gone
    for seq in ([name]*14, [name]*5 + [name + '_moved']*12, [name]*3 + [name + '_gone']*3):
        f = Follower(imgs[seq[0]], c)
        steps = []
        for k, fr in enumerate(seq[1:], 1):
            st, fx = f.step(k/40, imgs[fr])
            steps.append(dict(state=st, fix=None if fx is None else [float(v) for v in fx]))
        follows.append(dict(start=seq[0], cand=cand(c), frames=seq[1:], steps=steps))
        print('follow', name, len(seq), [s['state'][0] for s in steps])

# BallLock on number sequences
locks = []
seqs = [
    [[500, 400, 40, 2.0]]*12,
    [[500, 400, 40, 2.0]]*3 + [None] + [[500, 400, 40, 2.0]]*12,
    [[500, 400, 40, 2.0]]*3 + [None, None] + [[500, 400, 40, 2.0]]*12,
    [[500 + 1.5*math.sin(k), 400 + 1.5*math.cos(k), 40 + 2*math.sin(2*k), 3.0] for k in range(16)],
    [[500 + 4*k, 400, 40, 2.0] for k in range(16)],
    [[500, 400, 40, 2.0 if k % 3 else 20.0] for k in range(16)],
]
for seq in seqs:
    lk = BallLock(); res = []
    for k, s in enumerate(seq):
        r = lk.update(k/40, None if s is None else tuple(s[:3]), 0.0 if s is None else s[3])
        res.append(None if r is None else [float(v) for v in r])
    locks.append(dict(seq=seq, result=res))

json.dump(dict(frames=frames, locate=locates, follow=follows, lock=locks), open(os.path.join(out, 'cases.json'), 'w'), indent=1)
print('wrote', len(frames), 'frames,', len(locates), 'locate,', len(follows), 'follow,', len(locks), 'lock cases to', out)
