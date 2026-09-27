"""Reference implementation of the swing analyzer. The Swift SwingAnalyzer mirrors this logic 1:1."""
import numpy as np, math

BALL_MM = 42.67
CLUB_RADIUS_M = {'1W': 1.6, 'FW': 1.55, 'UT': 1.5, '7I': 1.4, 'W': 1.3}

def components(mask):
    """4-neighbour connected components -> list of dict(area, sx, sy, minx, maxx, miny, maxy)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]: continue
        cid = len(out) + 1; stack = [(y0, x0)]; lab[y0, x0] = cid
        c = dict(area=0, sx=0.0, sy=0.0, minx=x0, maxx=x0, miny=y0, maxy=y0)
        while stack:
            y, x = stack.pop()
            c['area'] += 1; c['sx'] += x; c['sy'] += y
            c['minx'] = min(c['minx'], x); c['maxx'] = max(c['maxx'], x)
            c['miny'] = min(c['miny'], y); c['maxy'] = max(c['maxy'], y)
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cid; stack.append((ny, nx))
        out.append(c)
    return out

def linfit(t, v):
    t = np.asarray(t, float); v = np.asarray(v, float)
    tm = t.mean(); vm = v.mean()
    den = ((t - tm)**2).sum()
    slope = ((t - tm)*(v - vm)).sum()/den
    return slope, vm - slope*tm

def analyze(frames, times, ball, trig, club='1W', energy=None, pitch=0.0):
    # pitch: degrees the camera looks down (face-on from above); image vertical motion is
    # the true vertical motion times cos(pitch), so divide it back out
    ycorr = 1/math.cos(math.radians(min(max(pitch, 0.0), 60.0)))
    bx, by, d = ball
    n = len(frames); H, W = frames[0].shape
    F = [f.astype(np.int16) for f in frames]
    res = dict(ok=False, notes=[])
    # background: median of frames well before impact
    idx = [max(0, trig - o) for o in (110, 95, 80, 65, 50)]
    Bg = np.median(np.stack([F[i] for i in idx]), axis=0).astype(np.int16)
    yy, xx = np.mgrid[0:H, 0:W]
    core = ((xx - bx)**2 + (yy - by)**2) < (0.35*d)**2
    # 1. impact frame: first frame whose ball core differs from background
    kImp = None
    for k in range(max(1, trig - 24), min(n, trig + 12)):
        if np.abs(F[k][core] - Bg[core]).mean() > 18:
            kImp = k; break
    if kImp is None:
        res['notes'].append('インパクトの瞬間を検出できませんでした'); return res
    res['kImp'] = kImp
    rest = ((xx - bx)**2 + (yy - by)**2) < (0.75*d)**2
    # 2. head: leading edge of the dark moving region in a band around the ball height
    y0 = max(0, int(by - 2.5*d)); y1 = min(H, int(by + 0.8*d))
    minCnt = max(2, int(round(0.2*d)))
    direction = None; pts = []
    for k in range(max(0, kImp - 6), kImp):
        diff = np.abs(F[k][y0:y1] - Bg[y0:y1]) > 28
        diff &= ~rest[y0:y1]
        cnt = diff.sum(axis=0)
        cols = np.nonzero(cnt >= minCnt)[0]
        if len(cols) == 0: continue
        if direction is None:
            cx = (cols*cnt[cols]).sum()/cnt[cols].sum()
            direction = 1 if bx > cx else -1
        if direction > 0:
            cand = cols[cols <= bx - 0.2*d]
            if len(cand) == 0: continue
            e = cand.max()
            band = (xx[y0:y1] >= e - 0.5*d) & (xx[y0:y1] <= e) & diff
        else:
            cand = cols[cols >= bx + 0.2*d]
            if len(cand) == 0: continue
            e = cand.min()
            band = (xx[y0:y1] <= e + 0.5*d) & (xx[y0:y1] >= e) & diff
        ys = np.nonzero(band)[0]
        pts.append((times[k], float(e), float(ys.mean() + y0) if len(ys) else float('nan')))
    # keep the last frames that advance monotonically toward the ball
    clean = []
    for p in reversed(pts):
        if not clean or (direction*(clean[-1][1] - p[1]) > 0.1*d): clean.append(p)
        if len(clean) == 4: break
    clean.reverse()
    mmpx = BALL_MM/d
    if direction is not None and len(clean) >= 2:
        t = [p[0] for p in clean]
        vx, _ = linfit(t, [p[1] for p in clean])
        ysv = [p[2] for p in clean]
        vy = linfit(t, ysv)[0]*ycorr if not any(math.isnan(v) for v in ysv) else 0.0
        vy = max(-abs(vx)*0.27, min(abs(vx)*0.27, vy))
        res['headPxS'] = math.hypot(vx, vy)
        res['hs'] = res['headPxS']*mmpx/1000
        chord = math.degrees(math.atan2(-vy, abs(vx)))
        # the fitted line is a chord of the swing arc: rotate to the tangent at contact
        e_contact = bx - direction*0.5*d
        e_fit_b = np.mean([p[1] for p in clean]) - vx*np.mean(t)
        tContact = (e_contact - e_fit_b)/vx
        tMid = float(np.mean(t))
        R = CLUB_RADIUS_M.get(club, 1.5)
        res['attack'] = chord + math.degrees((res['hs']/R)*(tContact - tMid))
        res['headFrames'] = len(clean)
    else:
        res['notes'].append('ヘッドを検出できませんでした')
    # 3. ball: most forward bright compact blob on the target side
    if direction is None:
        return res
    A0 = math.pi*d*d/4
    def find_ball(thr, maxArea, maxSide, minFill):
        pts = []
        for k in range(kImp, min(n, kImp + 10)):
            diff = (F[k] - Bg) > thr
            diff &= ~rest
            if direction > 0: diff[:, :int(bx + 0.5*d)] = False
            else: diff[:, int(bx - 0.5*d):] = False
            diff[:max(0, int(by - 14*d))] = False
            diff[min(H, int(by + 1.0*d)):] = False
            best = None
            for c in components(diff):
                if not (0.35*A0 <= c['area'] <= maxArea*A0): continue
                bw = c['maxx'] - c['minx'] + 1; bh = c['maxy'] - c['miny'] + 1
                if max(bw, bh) > maxSide*d or c['area']/(bw*bh) < minFill: continue
                if c['minx'] == 0 or c['miny'] == 0 or c['maxx'] == W-1 or c['maxy'] == H-1: continue
                cx = c['sx']/c['area']; cy = c['sy']/c['area']
                if best is None or direction*cx > direction*best[0]: best = (cx, cy)
            if best: pts.append((times[k], best[0], best[1]))
        return pts
    bpts = find_ball(35, 4.0, 3.5, 0.45)
    if len(bpts) < 2:   # slow shutter: the ball is a long, faint streak
        bpts = find_ball(12, 12.0, 10.0, 0.25)
        if len(bpts) >= 2: res['notes'].append('シャッターが遅くボールがブレています（精度低下）')
    # enforce forward progression
    cleanb = []
    for p in bpts:
        if not cleanb or direction*(p[1] - cleanb[-1][1]) > 0.3*d: cleanb.append(p)
    cleanb = cleanb[:4]
    if len(cleanb) >= 2:
        t = [p[0] for p in cleanb]
        vx, _ = linfit(t, [p[1] for p in cleanb]); vy, _ = linfit(t, [p[2] for p in cleanb]); vy *= ycorr
        res['ballPxS'] = math.hypot(vx, vy)
        res['bs'] = res['ballPxS']*mmpx/1000
        res['launch'] = math.degrees(math.atan2(-vy, abs(vx)))
        res['ballFrames'] = len(cleanb)
    else:
        res['notes'].append('打ち出したボールを追跡できませんでした')
    if 'hs' in res and 'bs' in res:
        res['smash'] = res['ballPxS']/res['headPxS']
        res['ok'] = True
    return res
