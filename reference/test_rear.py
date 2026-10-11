"""Accuracy of the rear-view analyzer on synthetic down-the-line swings (python3 test_rear.py)."""
import sys
from sim_rear import render; from analyzer_rear import analyze_rear
cases = [
    dict(),
    dict(hs=38, bs=50, launch=17, plane=58, radius=1.4, club='7I', seed=1, handed=-1),
    dict(hs=48, bs=71, launch=10, side_deg=4, path_deg=2, cam_h=1.0, look=(0.0, 0.3, 3.0), seed=2),
    dict(hs=33, bs=46, launch=24, plane=60, radius=1.3, club='W', cam_dist=2.2, seed=3, drop={-3, 2}),
    dict(hs=45, bs=66, launch=13, radius=1.75, cam_x=0.3, side_deg=-3, seed=4, noise=7),
    dict(hs=42, bs=60, launch=12, radius=1.45, cam_dist=3.2, cam_h=0.9, look=(0.0, 0.3, 3.0), seed=5, handed=-1),
    dict(hs=40, bs=58, launch=14, decel=2.5, seed=6),
    dict(exposure=1/500, seed=7),
    dict(cam_x=0.8, seed=8, handed=-1),
    dict(hs=36, bs=53, radius=1.5, cam_x=-0.6, seed=9),
    dict(cam_h=0.3, look=(0.0, 0.4, 2.0), seed=10),
]
failures = []
for c in cases:
    club = c.pop('club', '1W')
    s = render(**c)
    r = analyze_rear(s['frames'], s['times'], s['ball'], s['trig'], s['f'], s['cx'], s['cy'], s['pitch'], club=club)
    hs, bs, la = c.get('hs', 42.0), c.get('bs', 62.0), c.get('launch', 12.0)
    f = lambda k: (f"{r[k]:.2f}" if k in r else '-')
    print(dict(hs=hs, bs=bs, launch=la, club=club), '->', 'hs', f('hs'), 'bs', f('bs'), 'smash', f('smash'),
          '(true %.2f)' % (bs/hs), 'launch', f('launch'), 'R', f('radius'), r.get('headFrames'), r.get('ballFrames'), r['notes'])
    checks = [('hs', lambda v: abs(v/hs - 1) < 0.04), ('bs', lambda v: abs(v/bs - 1) < 0.025),
              ('launch', lambda v: abs(v - la) < 1.5), ('smash', lambda v: abs(v - bs/hs) < 0.07)]
    for k, ok in checks:
        if k not in r or not ok(r[k]): failures.append((c, k, r.get(k)))
for c, k, v in failures: print('FAIL', c.get('seed', 0), k, v)
print('OK' if not failures else f'{len(failures)} check(s) failed')
sys.exit(1 if failures else 0)
