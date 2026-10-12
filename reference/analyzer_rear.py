"""Rear-view (down-the-line) analyzer. Mirrored by SwingAnalyzer.analyzeRear in Swift.

Seen from behind, the ball and the clubhead move mostly away from the camera, so pixel speed
alone says little. Instead everything is placed in 3D:
  * depth of the ball from its apparent size: Z = f * 42.67mm / diameter, with the diameter
    taken from the ball's summed coverage (unaffected by motion blur);
  * ball velocity: straight line from the ball's rest position, fitted to the ball's image
    position and size in the frames after impact;
  * clubhead: tracked in the image over the last ~9 frames before impact and fitted with a
    circular arc that ends at the ball, running along the ball's horizontal direction (within a
    few degrees) at contact. Unknowns: swing-plane tilt, radius, angular speed and its rise,
    path yaw. Head speed = radius * angular speed at contact.
The radius is only loosely known from the club, so head speed rests on the arc model; ball
speed does not.
"""
import numpy as np, math
from analyzer import components

BALL_M = 0.04267
CLUB_RADIUS_M = {'1W': 1.6, 'FW': 1.55, 'UT': 1.5, '7I': 1.4, 'W': 1.3}

def median_of(frames, idx):
    return np.median(np.stack([frames[i] for i in idx]), axis=0)

def arc_points(par, t, t0, Hc, t_hat, up):
    """Head centre positions at times t (before t0) for params
    (alpha: plane tilt, R: radius, omega: angular speed at impact, c: its relative growth per
    second, psi: club path yaw relative to t_hat)."""
    a, R, om, c, psi = par
    s_hat = np.cross(t_hat, up)
    th = math.cos(psi)*t_hat + math.sin(psi)*s_hat
    sh = np.cross(th, up)
    n = math.sin(a)*up + math.cos(a)*sh
    C = Hc + R*n
    tau = t0 - np.asarray(t)
    ph = om*(tau - c*tau*tau/2)
    return C[None, :] - R*(np.cos(ph)[:, None]*n[None, :] + np.sin(ph)[:, None]*th[None, :])

def project(P, f, cx, cy):
    return np.stack([f*P[:, 0]/P[:, 2] + cx, f*P[:, 1]/P[:, 2] + cy], axis=1)

def lm(resid, p, h, iters=60, clamp=None):
    """Levenberg-Marquardt with a forward-difference Jacobian. resid(p) -> array or None."""
    p = np.asarray(p, float).copy()
    r = resid(p)
    if r is None: return None
    cost = (r*r).sum(); lam = 1e-2
    for _ in range(iters):
        J = np.zeros((len(r), len(p)))
        for j in range(len(p)):
            q = p.copy(); q[j] += h[j]
            rq = resid(q)
            if rq is None: return p, r
            J[:, j] = (rq - r)/h[j]
        A = J.T @ J; g = J.T @ r
        step = np.linalg.solve(A + lam*np.diag(np.diag(A) + 1e-9), -g)
        q = p + step
        if clamp: q = clamp(q)
        rq = resid(q)
        if rq is not None and (rq*rq).sum() < cost:
            p, r, cost = q, rq, (rq*rq).sum(); lam = max(lam/3, 1e-7)
            if np.all(np.abs(step) < np.asarray(h)*1e-2): break
        else:
            lam *= 4
            if lam > 1e7: break
    return p, r

def fit_arc(t, uv, t0, Hc, t_hat, up, f, cx, cy, R0):
    """Least squares over arc_points' params, in pixels. Soft priors keep the radius near the
    club's typical value (+-15%), the path within ~5 deg of the ball's direction and the
    head still speeding up a little into the ball."""
    t = np.asarray(t); uv = np.asarray(uv)
    def resid(p):
        P = arc_points(p, t, t0, Hc, t_hat, up)
        if np.any(P[:, 2] < 0.2): return None
        r = (project(P, f, cx, cy) - uv).ravel()/0.7
        return np.concatenate([r, [(p[1] - R0)/(0.15*R0), (p[3] - 1.5)/3.0, p[4]/math.radians(5)]])
    best = None
    for a in np.radians(np.arange(-85, 90, 5)):
        for hs in (25.0, 35.0, 45.0, 55.0):
            p = np.array([a, R0, hs/R0, 1.5, 0.0])
            r = resid(p)
            if r is None: continue
            c = (r*r).sum()
            if best is None or c < best[0]: best = (c, p)
    if best is None: return None
    def clamp(q):
        q = q.copy(); q[1] = min(max(q[1], 0.6*R0), 1.5*R0); return q
    out = lm(resid, best[1], (1e-4, 1e-4, 1e-3, 1e-3, 1e-4), clamp=clamp)
    if out is None: return None
    p, r = out
    rms = math.sqrt(((r[:-3]*0.7)**2).mean())
    return p, rms

def analyze_rear(frames, times, ball, trig, f, cx, cy, pitch=0.0, club='1W'):
    bx, by, d = ball
    n = len(frames); H, W = frames[0].shape
    F = [fr.astype(np.float32) for fr in frames]
    times = np.asarray(times, float)
    res = dict(ok=False, notes=[])
    p = min(max(pitch, -0.6), 1.2)
    up = np.array([0.0, -math.cos(p), -math.sin(p)])    # world up in camera coordinates
    def ray(u, v): return np.array([(u - cx)/f, (v - cy)/f, 1.0])
    Bg = median_of(F, [max(0, trig - o) for o in (110, 95, 80, 65, 50)])
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (xx - bx)**2 + (yy - by)**2
    core = r2 < (0.3*d)**2
    rest = r2 < (0.75*d)**2
    # 1. impact: first frame whose ball core changes (the head may cover the ball a frame or two early)
    kImp = None
    for k in range(max(1, trig - 24), min(n, trig + 12)):
        if np.abs(F[k][core] - Bg[core]).mean() > 18:
            kImp = k; break
    if kImp is None:
        res['notes'].append('インパクトの瞬間を検出できませんでした'); return res
    # 2. ball diameter at rest from coverage: compare with the spot after the ball has gone
    Iball = float(Bg[core].mean())
    late = [min(n - 1, kImp + o) for o in (24, 30, 36)]
    G = median_of(F, late)
    win = r2 < (0.5*d + 2.5)**2
    den = Iball - G[win]
    # the ball may be brighter or (backlit) darker than what is behind it
    if abs(np.median(den)) > 25:
        # not clipped at 0, so background noise around the ball cancels out
        ringw = (r2 >= (0.5*d + 3.5)**2) & (r2 < (0.5*d + 7)**2)
        off = float(np.median(Bg[ringw] - G[ringw]))
        cov = np.clip((Bg[win] - G[win] - off)/np.where(np.abs(den) < 1, 1, den), -0.5, 1.2)
        d0 = 2*math.sqrt(cov.sum()/math.pi)
        if not (0.8*d < d0 < 1.2*d): d0 = d
    else:
        d0 = d
    Z0 = f*BALL_M/d0
    B3 = ray(bx, by)*Z0
    # 3. ball in flight: bright compact blobs, sized by summed coverage
    den = Iball - Bg
    okden = np.abs(den) > 20
    den = np.where(okden, den, 1)
    A0 = math.pi*d0*d0/4
    bpts = []
    fr = float(np.median(np.diff(times))) if n > 1 else 1/240
    for k in range(kImp, min(n, kImp + 16)):
        c = np.where(okden, (F[k] - Bg)/den, 0)
        c = np.clip(c, -0.5, 1.2)
        m = c > 0.5
        m[:max(0, int(by - 25*d0))] = False
        m[min(H, int(by + 1.5*d0)):] = False
        m[:, :max(0, int(bx - 15*d0))] = False
        m[:, min(W, int(bx + 15*d0)):] = False
        m &= ~rest
        cands = []
        for b in components(m):
            if not (0.08*A0 <= b['area'] <= 2.2*A0): continue
            bw = b['maxx'] - b['minx'] + 1; bh = b['maxy'] - b['miny'] + 1
            if max(bw, bh) > 2.6*d0 or b['area']/(bw*bh) < 0.4: continue
            # coverage summed over the blob plus a margin, after removing the frame's overall
            # brightness offset (flicker), measured on a ring just outside the margin
            x0, x1 = max(0, b['minx'] - 3), min(W, b['maxx'] + 4)
            y0, y1 = max(0, b['miny'] - 3), min(H, b['maxy'] + 4)
            X0, X1 = max(0, x0 - 3), min(W, x1 + 3); Y0, Y1 = max(0, y0 - 3), min(H, y1 + 3)
            ringm = np.ones((Y1 - Y0, X1 - X0), bool); ringm[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
            ringm &= okden[Y0:Y1, X0:X1] & ~rest[Y0:Y1, X0:X1]
            off = float(np.median((F[k] - Bg)[Y0:Y1, X0:X1][ringm])) if ringm.sum() > 8 else 0.0
            sub = np.where(okden[y0:y1, x0:x1], (F[k][y0:y1, x0:x1] - Bg[y0:y1, x0:x1] - off)/den[y0:y1, x0:x1], 0)
            sub = np.clip(sub, -0.5, 1.2)
            sub[rest[y0:y1, x0:x1]] = 0
            S = sub.sum()
            if S <= 0 or np.maximum(sub, 0).sum() <= 0: continue
            dk = 2*math.sqrt(S/math.pi)
            if dk > 1.1*d0: continue
            sp = np.maximum(sub, 0); Sp = sp.sum()
            u = (sp*xx[y0:y1, x0:x1]).sum()/Sp; v = (sp*yy[y0:y1, x0:x1]).sum()/Sp
            # must not touch the resting spot (coverage there is unreliable)
            if (u - bx)**2 + (v - by)**2 < (0.75*d0 + dk/2 + 1)**2: continue
            if sub.max() < 0.7: continue          # not as bright as the ball
            # size noise grows as the ball's contrast with what is behind it drops
            ds = max(1.0, 100/float(np.median(np.abs(den[y0:y1, x0:x1]))))
            cands.append((u, v, dk, ds))
        # follow the ball outward from its resting spot
        ref = (bx, by) if not bpts else bpts[-1][1:3]
        gap = 1 if not bpts else max(1.0, (times[k] - bpts[-1][0])/fr)
        lim = 6*d0 if not bpts else 5*d0*gap
        best = None
        for u, v, dk, ds in cands:
            dist = math.hypot(u - ref[0], v - ref[1])
            if dist > lim: continue
            if best is None or dist < best[0]: best = (dist, u, v, dk, ds)
        if best: bpts.append((times[k], best[1], best[2], best[3], best[4]))
    # Straight 3D line from the resting ball, fitted in image space: the rest position slides
    # along its ray (its apparent size at rest is only a soft hint for its depth), plus velocity
    # and launch time. Frames that do not fit are dropped one at a time (a ball still partly
    # hidden by the head looks too small, i.e. too far).
    ray0 = ray(bx, by)
    def fit_line(pts):
        t = np.array([q[0] for q in pts])
        P = np.array([ray(q[1], q[2])*(f*BALL_M/q[3]) for q in pts])
        w = np.array([(q[3]/d0)**2/q[4]**2 for q in pts])    # depth error grows as the ball shrinks
        best = None
        # start: launch after the ball core first changed (minus slack), before the ball was seen
        for tt0 in np.arange(times[kImp] - 2*fr, t[0], fr/20):
            dt = t - tt0
            if np.any(dt <= 0): continue
            Vc = ((P - B3)*(w*dt)[:, None]).sum(0)/(w*dt*dt).sum()
            e = (w[:, None]*(P - B3 - Vc[None, :]*dt[:, None])**2).sum()
            if best is None or e < best[0]: best = (e, Vc, tt0)
        if best is None: return None
        _, Vc, tt0 = best
        def resid(p):
            Z0p, t0p, Vp = p[0], p[1], p[2:5]
            out = []
            for q in pts:
                Pq = ray0*Z0p + Vp*(q[0] - t0p)
                if Pq[2] < 0.3: return None
                out += [(f*Pq[0]/Pq[2] + cx - q[1])/0.3, (f*Pq[1]/Pq[2] + cy - q[2])/0.3,
                        (f*BALL_M/Pq[2] - q[3])/(0.12*q[4])]
            out.append((Z0p - Z0)/(0.015*Z0))
            return np.array(out)
        fit = lm(resid, np.concatenate([[Z0, tt0], Vc]), (1e-4, 1e-6, 1e-3, 1e-3, 1e-3))
        if fit is None: return None
        p = fit[0]
        errs = []
        for q in pts:
            Pq = ray0*p[0] + p[2:5]*(q[0] - p[1])
            dq = f*BALL_M/Pq[2]; uq = f*Pq[0]/Pq[2] + cx; vq = f*Pq[1]/Pq[2] + cy
            errs.append(abs(dq - q[3])/(0.3*q[4]) + math.hypot(uq - q[1], vq - q[2])/0.7)
        return p[2:5], p[1], errs
    cand = [q for q in bpts if q[3] <= 1.05*d0][:10]
    V = None; t0 = times[kImp]
    while len(cand) >= 3:
        fl = fit_line(cand)
        if fl is None: break
        Vc, tt0, errs = fl
        i = int(np.argmax(errs))
        if errs[i] > max(2.5, 2.5*float(np.median(errs))) and len(cand) > 3:
            cand.pop(i); continue
        if errs[i] <= 6:
            V, t0 = Vc, tt0
        break
    if V is not None:
        res['bs'] = float(np.linalg.norm(V))
        res['launch'] = math.degrees(math.asin(max(-1, min(1, V @ up/np.linalg.norm(V)))))
        res['ballFrames'] = len(cand)
    if V is None:
        res['notes'].append('打ち出したボールを追跡できませんでした')
    res['impactTime'] = float(t0)
    # 4. club path direction at contact: horizontal part of the ball's direction
    if V is not None:
        th = V - (V @ up)*up
    else:
        th = np.array([0.0, 0.0, 1.0]) - up*up[2]
    t_hat = th/np.linalg.norm(th)
    Hc = B3 - 0.035*t_hat
    # 5. head: track backwards from contact against a per-frame moving background
    kFirst = int(np.searchsorted(times, t0, side='left')) - 1
    prev = []
    Zp = Z0
    for k in range(kFirst, max(12, kFirst - 9), -1):
        if k - 12 < 0: break
        M = median_of(F, [k - 12, k - 9, k - 6])
        mask = np.abs(F[k] - M) > 28      # the resting ball is in M too, so it drops out
        rk = max(3.0, f*0.06/Zp)       # about the clubhead's size
        if not prev:
            pu, pv = bx, by; sr = 3*rk
        elif len(prev) == 1:
            pu, pv = prev[-1][1], prev[-1][2]; sr = 4*rk
        else:
            a, b = prev[-1], prev[-2]
            # going back in time the head moves faster across the image
            pu = a[1] + (a[1] - b[1])*1.3; pv = a[2] + (a[2] - b[2])*1.3
            sr = max(2.5*rk, 0.6*math.hypot(a[1] - b[1], a[2] - b[2]))
        # densest head-sized cluster near the prediction, then mean-shift onto it
        x0, x1 = int(max(0, pu - sr)), int(min(W, pu + sr + 1))
        y0, y1 = int(max(0, pv - sr)), int(min(H, pv + sr + 1))
        if x0 >= x1 or y0 >= y1: break
        sub = mask[y0:y1, x0:x1]
        if sub.sum() < 3: break
        ys, xs = np.nonzero(sub); xs = xs + x0; ys = ys + y0
        # seed: the masked pixel with the most masked neighbours within rk
        step = max(1, len(xs)//400)
        seeds = np.arange(0, len(xs), step)
        cnt = [(((xs - xs[i])**2 + (ys - ys[i])**2) < rk*rk).sum() for i in seeds]
        if max(cnt) < 0.25*math.pi*rk*rk: break     # nothing head-sized
        i = seeds[int(np.argmax(cnt))]
        u, v = float(xs[i]), float(ys[i])
        for _ in range(6):
            sel = ((xs - u)**2 + (ys - v)**2) < rk*rk
            if sel.sum() == 0: break
            u, v = xs[sel].mean(), ys[sel].mean()
        prev.append((times[k], u, v))
        # rough depth for the next kernel size: the head comes toward the camera
        Zp = max(0.4, Z0 - 0.15*len(prev))
    hpts = list(reversed(prev))
    res['headFrames'] = len(hpts)
    if len(hpts) >= 4:
        R0 = CLUB_RADIUS_M.get(club, 1.5)
        fit = fit_arc([q[0] for q in hpts], [(q[1], q[2]) for q in hpts], t0, Hc, t_hat, up, f, cx, cy, R0)
        if fit is not None:
            par, rms = fit
            res['hs'] = float(par[1]*par[2]); res['radius'] = float(par[1])
            res['plane'] = float(math.degrees(par[0])); res['fitRms'] = rms
            if rms > 0.6*d0: res['notes'].append('ヘッドの軌道が円弧に合いません（参考値）')
    if 'hs' not in res:
        res['notes'].append('ヘッドを検出できませんでした')
    if 'hs' in res and 'bs' in res:
        res['smash'] = res['bs']/res['hs']; res['ok'] = True
    res['hpts'] = hpts
    return res
