"""Synthetic 240fps rear-view (down-the-line) swing generator for testing analyzer_rear.py.

World: x right (seen from the camera), y up, z along the target line. Ball centre at (0, r, 0).
The camera stands behind the ball; the image is what the app's rear-mode ring keeps
(the full frame at half resolution).
"""
import numpy as np, math

BALL_R = 0.042672/2

def _norm(v): return v/np.linalg.norm(v)

def camera(cam, look):
    fwd = _norm(np.asarray(look, float) - cam)
    right = _norm(np.cross([0.0, 1.0, 0.0], fwd))
    down = np.cross(right, fwd)
    return np.stack([right, down, fwd])

def render(hs=42.0, bs=62.0, launch=12.0, side_deg=0.0, path_deg=0.0, handed=+1,
           cam_dist=2.6, cam_x=0.0, cam_h=0.6, look=(0.0, 0.45, 2.0), plane=50.0, radius=1.6,
           decel=1.2, W=960, H=540, hfov=70.0, noise=4.0, drop=None, seed=0, exposure=1/1000,
           flicker=2.0, fps=240.0, body=True):
    """handed=+1 right-handed (golfer on the image left), -1 left-handed.
    plane: swing-plane angle above the ground (deg). decel: relative head-speed loss per second
    going back in time from impact (the head is still speeding up into the ball)."""
    rng = np.random.default_rng(seed)
    f = (W/2)/math.tan(math.radians(hfov)/2)
    cx, cy = W/2, H/2
    cam = np.array([cam_x, cam_h, -cam_dist])
    Rm = camera(cam, look)
    def proj(P):
        q = Rm @ (np.asarray(P, float) - cam)
        return f*q[0]/q[2] + cx, f*q[1]/q[2] + cy, q[2]
    B = np.array([0.0, BALL_R, 0.0])
    bu, bv, bz = proj(B)
    d = 2*f*BALL_R/bz
    # swing plane: contains the club path through the ball, tilted toward the golfer
    pa = math.radians(path_deg)
    t_hat = np.array([math.sin(pa), 0.0, math.cos(pa)])
    s_hat = handed*_norm(np.cross(t_hat, [0.0, 1.0, 0.0]))   # horizontal, ball -> golfer
    pl = math.radians(plane)
    n_hat = math.sin(pl)*np.array([0.0, 1.0, 0.0]) + math.cos(pl)*s_hat   # ball -> swing centre
    Hc0 = B - 0.035*t_hat                                 # clubhead centre at contact
    C = Hc0 + radius*n_hat
    om = hs/radius
    def phi(tau):          # arc angle before impact (tau > 0 before impact)
        return om*(tau - decel*tau*tau/2) if tau > 0 else om*tau
    def head(tau):
        p = phi(tau)
        return C - radius*(math.cos(p)*n_hat + math.sin(p)*t_hat), p
    def hands(tau):
        p = phi(tau)*0.92
        r = radius - 1.05
        return C - r*(math.cos(p)*n_hat + math.sin(p)*t_hat)
    la, sd = math.radians(launch), math.radians(side_deg)
    V = bs*np.array([math.cos(la)*math.sin(sd), math.sin(la), math.cos(la)*math.cos(sd)])
    def ball(tau):
        if tau >= 0: return B
        tt = -tau
        return B + V*tt + np.array([0, -4.9*tt*tt, 0])
    # background: ground texture below the horizon, brighter sky/net above
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    bg = 100 + 14*np.sin(xx/13.0)*np.cos(yy/19.0) + rng.normal(0, 6, (H, W))
    hor = proj(np.array([0.0, cam_h, 200.0]))[1]
    bg[yy < hor] = 165 + 10*np.sin(xx[yy < hor]/40.0)
    bg[(yy > bv + 0.6*d) & (np.abs(xx - bu) < 6*d)] += 15           # mat
    def disc(img, u, v, r, val):
        x0, x1 = int(max(0, u - r - 1)), int(min(W, u + r + 2))
        y0, y1 = int(max(0, v - r - 1)), int(min(H, v + r + 2))
        if x0 >= x1 or y0 >= y1: return
        sub = img[y0:y1, x0:x1]
        dist = np.sqrt((xx[y0:y1, x0:x1] - u)**2 + (yy[y0:y1, x0:x1] - v)**2)
        a = np.clip(r - dist + 0.5, 0, 1)         # anti-aliased edge, like real optics
        sub[:] = sub*(1 - a) + val*a
    def line(img, a, b, w, val):
        x0, x1 = int(max(0, min(a[0], b[0]) - w - 1)), int(min(W, max(a[0], b[0]) + w + 2))
        y0, y1 = int(max(0, min(a[1], b[1]) - w - 1)), int(min(H, max(a[1], b[1]) + w + 2))
        if x0 >= x1 or y0 >= y1: return
        X = xx[y0:y1, x0:x1]; Y = yy[y0:y1, x0:x1]
        dx, dy = b[0] - a[0], b[1] - a[1]; L2 = dx*dx + dy*dy + 1e-9
        t = np.clip(((X - a[0])*dx + (Y - a[1])*dy)/L2, 0, 1)
        m = (X - a[0] - t*dx)**2 + (Y - a[1] - t*dy)**2 < w*w
        img[y0:y1, x0:x1][m] = val
    frames, ts = [], []
    n_pre, n_post = 150, 40
    ks = [k for k in range(-n_pre, n_post) if not (drop and k in drop)]
    nsub = 6
    for k in ks:
        acc = np.zeros((H, W))
        for s in range(nsub):
            tau = -(k/fps - exposure*s/(nsub - 1))     # time before impact
            img = bg.copy()
            if body:   # torso and legs, turning slowly
                g = C + np.array([0.0, -0.35, 0.0]) + 0.25*math.sin(min(max(-tau, -0.4), 0.3)*5)*t_hat
                gu, gv, gz = proj(g); disc(img, gu, gv, f*0.22/gz, 55)
                lu, lv, lz = proj(g + np.array([0.0, -0.6, 0.0])); disc(img, lu, lv, f*0.16/lz, 60)
                su, sv, sz = proj(C + np.array([0.0, 0.25, 0.0])); hu, hv, hz = proj(hands(tau))
                line(img, (su, sv), (hu, hv), f*0.05/hz, 70)
            u, v, z = proj(ball(tau))
            disc(img, u, v, f*BALL_R/z, 232)
            P, _ = head(tau)          # the head is always nearer the camera than the ball
            if P[2] > cam[2] + 0.3:
                hu, hv, hz = proj(P); au, av, az = proj(hands(tau))
                line(img, (hu, hv), (au, av), max(0.8, f*0.006/hz), 45)
                disc(img, hu, hv, f*0.05/hz, 30)
            acc += img
        img = acc/nsub + rng.normal(0, noise, (H, W)) + rng.normal(0, flicker)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        ts.append(k/fps + 5.0)
    truth = dict(B=B, V=V, d=d, head=[(5.0 - tau, proj(head(tau)[0])[:2]) for tau in np.arange(0, 0.06, 1/fps)])
    pitch = math.asin(-Rm[2][1])
    return dict(frames=frames, times=np.array(ts), ball=(bu, bv, d), trig=ks.index(0), fps=fps,
                f=f, cx=cx, cy=cy, pitch=pitch, t_imp=5.0, truth=truth)
