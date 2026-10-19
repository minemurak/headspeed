"""Reference implementation of the swing analyzer. The Swift SwingAnalyzer mirrors this logic 1:1."""
import numpy as np, math

BALL_MM = 42.67
CLUB_RADIUS_M = {'1W': 1.6, 'FW': 1.55, 'UT': 1.5, '7I': 1.4, 'W': 1.3}
# typical launch angle (deg) per club: from above, only the along-target part of the ball's
# motion is trusted, and the full speed is that over cos(launch)
CLUB_LAUNCH_DEG = {'1W': 13.0, 'FW': 12.0, 'UT': 14.0, '7I': 17.0, 'W': 27.0}
# and typical attack angle (deg, down negative): the head's tilt at contact, unseen from above
CLUB_ATTACK_DEG = {'1W': 0.0, 'FW': -2.0, 'UT': -3.0, '7I': -4.0, 'W': -5.0}

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

def open_mask(m, k):
    """Binary opening with a k x k square: removes anything thinner than k (a shaft)."""
    if k <= 1: return m
    from numpy.lib.stride_tricks import sliding_window_view as sw
    p = k//2
    def run(a, axis, f):
        pad = np.pad(a, [(p, p) if i == axis else (0, 0) for i in range(2)], constant_values=False)
        return f(sw(pad, k, axis=axis), axis=-1)
    e = run(run(m, 0, np.all), 1, np.all)
    return run(run(e, 0, np.any), 1, np.any)

def dilate_mask(m, k):
    """Binary dilation with a k x k square."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    p = k//2
    def run(a, axis):
        pad = np.pad(a, [(p, p) if i == axis else (0, 0) for i in range(2)], constant_values=False)
        return sw(pad, k, axis=axis).any(axis=-1)
    return run(run(m, 0), 1)

def linfit(t, v):
    t = np.asarray(t, float); v = np.asarray(v, float)
    tm = t.mean(); vm = v.mean()
    den = ((t - tm)**2).sum()
    slope = ((t - tm)*(v - vm)).sum()/den
    return slope, vm - slope*tm

def silhouette(F, Bg, ball, trig):
    """The ball's size and place from what is gone once it has left: the frames at the end of the
    clip against the background, so a ball lit only on one side (its shaded part as dark as the
    grass, invisible to a brightness search) still shows whole. Width across, since the tee
    hangs below. None when the end frames are missing or the change there is not a lone ball."""
    bx, by, d = ball
    post = list(range(trig + 30, len(F)))[-8:]
    if len(post) < 5: return None
    H, W = Bg.shape
    x0, x1 = max(0, int(bx - 2*d)), min(W, int(bx + 2*d) + 1)
    y0, y1 = max(0, int(by - 2*d)), min(H, int(by + 2*d) + 1)
    P = np.median(np.stack([F[k][y0:y1, x0:x1] for k in post]), axis=0)
    m = np.abs(P - Bg[y0:y1, x0:x1]) > 25
    hit = None
    for c in components(m):
        if c['area'] < 10: continue
        if c['minx'] <= bx - x0 <= c['maxx'] and c['miny'] <= by - y0 <= c['maxy']:
            if hit is None or c['area'] > hit['area']: hit = c
    if hit is None or hit['minx'] == 0 or hit['miny'] == 0 or hit['maxx'] == x1 - x0 - 1 or hit['maxy'] == y1 - y0 - 1:
        return None
    w = hit['maxx'] - hit['minx'] + 1
    if not (0.5*d <= w <= 2.5*d): return None
    return x0 + (hit['minx'] + hit['maxx'])/2, y0 + hit['miny'] + w/2, float(w)

def head_match(F, Bg, times, ball, band, kImp, direction, shaft):
    """Head displacement between consecutive frames shortly before contact, by matching the head's
    own pixels (the moving region within a head's length of its front) in the next frame. Uses the
    whole head, so the hosel and shaft that lead the face no longer set the speed, as they can
    for a single leading-edge column. Returns [(t_mid, vx, vy)] in px/s."""
    H, W = Bg.shape
    bx, by, d = ball
    y0, y1 = band
    out = []
    yy, xx = np.mgrid[0:H, 0:W]
    rest = ((xx - bx)**2 + (yy - by)**2) < (0.75*d)**2
    # pairs ending two frames before contact: in the frame just before it the head's front is
    # already at the ball, and the front is what pins a plain head's position
    for k in range(max(1, kImp - 4), kImp - 2):
        m = np.zeros((H, W), bool)
        m[y0:y1] = (np.abs(F[k][y0:y1] - Bg[y0:y1]) > 28) & (np.abs(F[k][y0:y1] - F[k-1][y0:y1]) > 28)
        m &= ~rest
        m = open_mask(m, shaft)
        cs = [c for c in components(m) if c['area'] >= 0.3*d*d]
        if not cs: continue
        # the head: everything moving within a head's length of the front-most part
        if direction > 0:
            front = max(c['maxx'] for c in cs); x0, x1 = int(front - 3*d), front + 1
        else:
            front = min(c['minx'] for c in cs); x0, x1 = front, int(front + 3*d) + 1
        hm = np.zeros_like(m); hm[:, max(0, x0):min(W, x1)] = m[:, max(0, x0):min(W, x1)]
        if hm.sum() < 0.3*d*d: continue
        # with a margin of background round it, so a plain (untextured) head is pinned by its
        # outline; compared as differences from the background, which is ~0 off the head
        ys, xs = np.nonzero(dilate_mask(hm, max(3, int(0.6*d) | 1)) & ~rest)
        vals = (F[k][ys, xs] - Bg[ys, xs]).astype(float); nxt = F[k + 1] - Bg
        def cost(dx, dy):
            X = xs + dx; Y = ys + dy
            ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
            ok[ok] = ~rest[Y[ok], X[ok]]          # the ball (bright, then hit) is not the head's
            if ok.mean() < 0.6: return None
            return float(np.abs(nxt[Y[ok], X[ok]] - vals[ok]).mean())
        best = None
        r = int(1.5*d)
        for dy in range(-r, r + 1, 2):            # coarse, then refine around the best
            for a in range(int(0.5*d), int(8*d), 2):
                c0 = cost(direction*a, dy)
                if c0 is not None and (best is None or c0 < best[0]): best = (c0, direction*a, dy)
        if best is None: continue
        bdx, bdy = best[1], best[2]
        for dy in range(bdy - 2, bdy + 3):
            for dx in range(bdx - 2, bdx + 3):
                c0 = cost(dx, dy)
                if c0 is not None and c0 < best[0]: best = (c0, dx, dy)
        dt = times[k + 1] - times[k]
        out.append(((times[k] + times[k + 1])/2, best[1]/dt, best[2]/dt))
    return out

def analyze(frames, times, ball, trig, club='1W', energy=None, pitch=0.0, overhead=False):
    # pitch: degrees the camera looks down (face-on from above); image vertical motion is
    # the true vertical motion times cos(pitch), so divide it back out.
    # overhead: filmed face-on from well above (looking down steeply). Image vertical motion
    # then mixes rising with moving toward or away from the camera, so only the horizontal
    # (along the target line) motion is used: no launch or attack angle.
    ycorr = 1/math.cos(math.radians(min(max(pitch, 0.0), 60.0)))
    bx, by, d = ball
    n = len(frames); H, W = frames[0].shape
    F = [f.astype(np.int16) for f in frames]
    res = dict(ok=False, notes=[])
    # background: median of frames well before impact
    idx = [max(0, trig - o) for o in (110, 95, 80, 65, 50)]
    Bg = np.median(np.stack([F[i] for i in idx]), axis=0).astype(np.int16)
    sil = silhouette(F, Bg, ball, trig)
    if sil and abs(sil[2]/d - 1) > 0.1:
        # the finder saw only part of the ball (the lit side); every speed scales with d
        bx, by, d = sil
        res['ballCorrected'] = sil
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
    direction = None; pts = []; tContact = None
    shaft = max(3, int(0.17*d) | 1)   # odd, wider than the shaft
    for k in range(max(1, kImp - 6), kImp):
        # only what is moving fast right now: different from the background AND from the
        # previous frame (a golfer's foot or a shadow shifting slowly near the ball is not)
        diff = (np.abs(F[k][y0:y1] - Bg[y0:y1]) > 28) & (np.abs(F[k][y0:y1] - F[k-1][y0:y1]) > 28)
        diff &= ~rest[y0:y1]
        # the shaft leans ahead of the head and crosses the band first: keep only the head's bulk
        diff = open_mask(diff, shaft)
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
    # keep the last frames that advance monotonically toward the ball, at a steady pace
    # (an earlier frame can catch the shaft crossing the band ahead of the head)
    clean = []
    for p in reversed(pts):
        if clean and direction*(clean[-1][1] - p[1]) <= 0.1*d: continue
        if len(clean) >= 2:
            last = direction*(clean[-2][1] - clean[-1][1])/(clean[-2][0] - clean[-1][0])
            step = direction*(clean[-1][1] - p[1])/(clean[-1][0] - p[0])
            if not (0.6*last <= step <= 1.6*last): break
        clean.append(p)
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
        e_contact = bx - direction*0.5*d
        e_fit_b = np.mean([p[1] for p in clean]) - vx*np.mean(t)
        tContact = (e_contact - e_fit_b)/vx
        tMid = float(np.mean(t))
        # the head's whole-body displacement over the last frames, when it agrees roughly
        hm = head_match(F, Bg, times, (bx, by, d), (y0, y1), kImp, direction, shaft)
        if hm:
            mvx = float(np.mean([h[1] for h in hm])); mvy = float(np.mean([h[2] for h in hm]))*ycorr
            if 0.6 <= mvx/vx <= 2.0:
                # the whole velocity along the arc (a few frames out it still dips noticeably)
                vx, vy = mvx, max(-abs(mvx)*0.6, min(abs(mvx)*0.6, mvy))
                tMid = float(np.mean([h[0] for h in hm]))
                res['headPxS'] = math.hypot(vx, vy)
                res['hs'] = res['headPxS']*mmpx/1000
                res['headMatched'] = len(hm)
        R = CLUB_RADIUS_M.get(club, 1.5)
        if overhead:
            # the head a few frames before contact still moves along the arc, tilted from the
            # target line by the club's usual attack angle plus its turn until contact; its
            # horizontal part is the speed times the cosine of that tilt
            th = (abs(vx)*mmpx/1000/R)*max(0.0, tContact - tMid) - math.radians(CLUB_ATTACK_DEG.get(club, 0.0))
            th = min(math.radians(25), th)
            res['headPxS'] = abs(vx)/math.cos(th)
            res['hs'] = res['headPxS']*mmpx/1000
        else:
            chord = math.degrees(math.atan2(-vy, abs(vx)))
            # the fitted line is a chord of the swing arc: rotate to the tangent at contact
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
            # and changed since the previous frame: the ball in flight moves several of its own
            # widths per frame, while something that merely differs from the background (a glove
            # or trouser leg that shifted at address) sits still and is not the ball
            diff &= np.abs(F[k] - F[k - 1]) > thr
            diff &= ~rest
            if direction > 0: diff[:, :int(bx + 0.5*d)] = False
            else: diff[:, int(bx - 0.5*d):] = False
            diff[:max(0, int(by - 14*d))] = False
            # from above, a ball starting away from the golfer (toward the camera) goes down
            diff[min(H, int(by + (4.0 if overhead else 1.0)*d)):] = False
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
    def forward(pts):
        kept = []
        for p in pts:
            if not kept or direction*(p[1] - kept[-1][1]) > 0.3*d: kept.append(p)
        return kept[:4]
    strict = find_ball(35, 4.0, 3.5, 0.45)
    cleanb = forward(strict)
    if len(strict) < 2:   # slow shutter: the ball is a long, faint streak
        loose = forward(find_ball(12, 12.0, 10.0, 0.25))
        if len(loose) >= 2:
            cleanb = loose
            res['notes'].append('シャッターが遅くボールがブレています（精度低下）')
    dtFrame = float(np.median(np.diff(times)))
    if len(cleanb) >= 2:
        t = [p[0] for p in cleanb]
        vx, _ = linfit(t, [p[1] for p in cleanb]); vy, _ = linfit(t, [p[2] for p in cleanb]); vy *= ycorr
        res['ballFrames'] = len(cleanb)
    elif cleanb and tContact is not None and \
            0.3*dtFrame <= cleanb[0][0] - tContact <= 2.5*dtFrame and direction*(cleanb[0][1] - bx) > d:
        # one flight frame only (the ball left the picture straight away): it left the tee
        # when the head reached it, so take the displacement from there
        tb, xb, yb = cleanb[0]
        vx = (xb - bx)/(tb - tContact); vy = (yb - by)/(tb - tContact)*ycorr
        res['ballFrames'] = 1
        if 0 <= xb + vx*dtFrame < W:
            res['notes'].append('ボールが1コマしか写らず推定値です（精度低下）')
        else:
            res['notes'].append('ボールがすぐ画面外に出たため1コマから推定しました（精度低下）。ボールを画面の中央寄りに置いてください')
    else:
        res['notes'].append('打ち出したボールを追跡できませんでした')
    if 'ballFrames' in res:
        if overhead:
            res['ballPxS'] = abs(vx)/math.cos(math.radians(CLUB_LAUNCH_DEG.get(club, 13.0)))
        else:
            res['ballPxS'] = math.hypot(vx, vy)
            res['launch'] = math.degrees(math.atan2(-vy, abs(vx)))
        res['bs'] = res['ballPxS']*mmpx/1000
    if 'hs' in res and 'bs' in res:
        res['smash'] = res['ballPxS']/res['headPxS']
        res['ok'] = True
    return res
