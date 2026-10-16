"""Ball lock (python3 test_balllock.py): the choice among candidates (a white shoe toe, a round
sky gap above the horizon, two balls with one at the last measured spot) and when a ball is
locked from a run of looks. Needs numpy and pillow."""
import math, os, sys
import numpy as np
from PIL import Image
from balllock import locate, track, BallLock, Follower, horizon_row, LOCK_SPAN
from ballfinder import search, refine

here = os.path.dirname(__file__) or '.'
failures = []
def check(name, ok, info=''):
    print(('OK  ' if ok else 'FAIL'), name, info)
    if not ok: failures.append(name)

def grass(seed, W=1920, H=1080):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = 105 + 10*np.sin(xx/13.0)*np.cos(yy/17.0) + rng.normal(0, 7, (H, W))
    return img, xx, yy

def disc(img, xx, yy, x, y, d, v):
    img[(xx - x)**2 + (yy - y)**2 < (d/2)**2] = v

def u8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

# 1. face-on: a white shoe toe (brighter, compact, but no round outline) above the ball
img, xx, yy = grass(0)
img[((xx - 760)/44)**2 + ((yy - 640)/32)**2 < 1] = 245        # shoe toe, 88 x 64
disc(img, xx, yy, 1010, 800, 46, 228)                           # ball
img[(np.abs(xx - 1010) < 4) & (yy > 823) & (yy < 850)] = 200    # tee
full = full1 = u8(img)
c = search(full[::4, ::4], (0.15, 0.30, 0.85, 0.97), 3, 27.5)
check('old search ranks the shoe first (what this fixes)', c is not None and abs(c[0]*4 - 760) < 30)
from balllock import crop_refine
old = crop_refine(full, (c[0]*4 + 1.5, c[1]*4 + 1.5, c[2]*4), refine)
check('... whose refine fails, so the ball was never tried', old is None, old)
cands = locate(full, rear=False)
top = cands[0] if cands else None
check('face-on: the ball found under the shoe', top is not None and math.hypot(top[0] - 1010, top[1] - 800) < 3
      and abs(top[2]/46 - 1) < 0.08 and top[3] == 'round',
      [(round(q[0]), round(q[1]), round(q[2], 1), q[3]) for q in cands])

# 2. horizon: with no ball in view, a round bright gap high in the frame (sky through a net)
img, xx, yy = grass(1)
disc(img, xx, yy, 900, 380, 40, 235)
full = u8(img)
hz = horizon_row(1920, 1080, 70.0, math.radians(-6))            # looking a little up
check('sky gap found without the tilt', len(locate(full, rear=False)) > 0)
check('sky gap dropped above the horizon', locate(full, rear=False, horizon=hz) == [], 'horizon row %.0f' % hz)
q = (locate(full1, rear=False, horizon=horizon_row(1920, 1080, 70.0, math.radians(2))) or [None])[0]
check('ball below the horizon kept', q is not None and abs(q[0] - 1010) < 3)

# 3. real rear frames: the ball comes first, also when the tilt is known (camera level)
for name, (tx, ty, td) in (('rear_backlit', (947, 777, 35)), ('rear_split', (886, 676, 34)), ('rear_split2', (827, 773, 34))):
    full = np.asarray(Image.open(f'{here}/fixtures/{name}.jpg'))
    cands = locate(full, rear=True, horizon=horizon_row(1920, 1080, 62.0, math.radians(3)))
    q = cands[0] if cands else None
    check(f'{name}: ball first', q is not None and math.hypot(q[0] - tx, q[1] - ty) < 2.5 and abs(q[2]/td - 1) < 0.08,
          None if q is None else (round(q[0], 1), round(q[1], 1), round(q[2], 1)))
    t = track(full, (tx + 4, ty - 3, td*1.1), 'rear')
    check(f'{name}: followed from a rough position', t is not None and math.hypot(t[0] - tx, t[1] - ty) < 2.5, t and tuple(round(v, 1) for v in t))

# 4. two balls: the one on the spot of the last measured ball comes first, without a tap
two = np.asarray(Image.open(f'{here}/fixtures/rear_two_balls.jpg'))
for home, (tx, ty) in (((1053.5, 698.0, 31.6, 'rear'), (1053.5, 698.0)), ((916.5, 735.0, 30.6, 'rear'), (916.5, 735.0))):
    q = (locate(two, rear=True, home=home) or [None])[0]
    check(f'two balls, home {home[:2]}', q is not None and math.hypot(q[0] - tx, q[1] - ty) < 2.5, q and tuple(round(v, 1) for v in q[:3]))

# 5. face-on, ball half lit by a low sun behind it (real frame): found by the lit fallback
for name in ('front_halflit', 'front_halflit_shoe'):
    full = np.asarray(Image.open(f'{here}/fixtures/{name}.jpg'))
    q = (locate(full, rear=False) or [None])[0]
    check(f'{name}: ball first', q is not None and math.hypot(q[0] - 1041, q[1] - 578) < 5 and abs(q[2]/56 - 1) < 0.06,
          q and (round(q[0], 1), round(q[1], 1), round(q[2], 1), q[3]))

# 6. when the lock is taken, looking every 1/40 s
def run(seq, dt=1/40):
    lock = BallLock()
    for k, (fix, change) in enumerate(seq):
        r = lock.update(k*dt, fix, change)
        if r is not None: return k*dt, r
    return None, None
ball = (500.0, 400.0, 40.0)
t, r = run([(ball, 2.0)]*40)
check('steady ball locks after %.2f s' % LOCK_SPAN, t is not None and abs(t - 0.25) < 1e-6 and r == ball, t)
t, _ = run([(ball, 2.0)]*3 + [(None, 0)] + [(ball, 2.0)]*40)
check('one missed look does not start over', t is not None and t <= 0.25 + 1e-6, t)
t, _ = run([(ball, 2.0)]*3 + [(None, 0)]*2 + [(ball, 2.0)]*40)
check('two missed looks start over', t is not None and abs(t - (5/40 + 0.25)) < 1e-6, t)
t, r = run([((500 + 1.5*math.sin(k), 400 + 1.5*math.cos(k), 40 + 2*math.sin(2*k)), 3.0) for k in range(40)])
check('fit jitter still locks, median size', t is not None and abs(t - 0.25) < 1e-6 and abs(r[2] - 40) < 1, (t, r))
check('something sliding never locks', run([((500 + 4*k, 400, 40), 2.0) for k in range(40)])[0] is None)
check('a changing look never locks', run([(ball, 2.0 if k % 3 else 20.0) for k in range(40)])[0] is None)

# 7. following real frames (sensor noise added each look): locks with the right size; a moved
# ball starts over; a ball taken away is lost
rng = np.random.default_rng(5)
for name, cand, truth in (('rear_split', (888, 674, 33.9, 'rear'), (886, 676, 34)),
                          ('front_halflit', (1040.2, 576.9, 55.8, 'lit'), (1041, 578, 56))):
    base = np.asarray(Image.open(f'{here}/fixtures/{name}.jpg')).astype(float)
    def frame(shift=0):
        f = np.roll(base, shift, axis=1) + rng.normal(0, 3, base.shape)
        return np.clip(f, 0, 255).astype(np.uint8)
    fol = Follower(frame(), cand)
    out = None
    for k in range(1, 30):
        st, fx = fol.step(k/40, frame())
        if st == 'locked': out = (k/40, fx); break
    check(f'{name}: follower locks', out is not None and out[0] <= 0.3 and math.hypot(out[1][0] - truth[0], out[1][1] - truth[1]) < 3
          and abs(out[1][2]/truth[2] - 1) < 0.06, out and (out[0], tuple(round(v, 1) for v in out[1])))
    fol = Follower(frame(), cand)
    for k in range(1, 6): fol.step(k/40, frame())
    st, fx = fol.step(6/40, frame(int(0.24*truth[2]) + 1))     # the ball moved over a fifth of itself (rolled on the tee)
    check(f'{name}: a moved ball starts the run over', st == 'following' and len(fol.lock.run) == 1, (st, len(fol.lock.run)))
    gone = base.copy(); x, y, d = truth
    yy, xx = np.mgrid[0:base.shape[0], 0:base.shape[1]]
    gone[(xx - x)**2 + (yy - y)**2 < (0.7*d)**2] = 90
    fol = Follower(frame(), cand)
    states = [fol.step(k/40, np.clip(gone + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8))[0] for k in range(1, 4)]
    check(f'{name}: a ball taken away is lost', states[:2] == ['following', 'lost'], states)

print('OK' if not failures else f'{len(failures)} failed: {failures}')
sys.exit(1 if failures else 0)
