"""Synthetic 240fps face-on swing strip generator (grayscale) for testing the analyzer."""
import numpy as np, math

def render(hs=42.0, bs=62.0, launch=12.0, attack=-3.0, d=24.0, direction=+1, fps=240.0,
           W=900, H=320, noise=4.0, drop=None, seed=0, exposure=1/1000, flicker=2.0):
    rng = np.random.default_rng(seed)
    mmpx = 42.67 / d
    by = H - 2.6*d            # ball center row
    bx = W/2 + (-4*d if direction > 0 else 4*d)
    # background: textured turf + mat edge + two dark feet
    yy, xx = np.mgrid[0:H, 0:W]
    bg = 105 + 12*np.sin(xx/17.0)*np.cos(yy/23.0) + rng.normal(0, 6, (H, W))
    bg[int(by + 0.5*d):, :] += 18   # mat
    for fx in (bx - direction*3.5*d, bx - direction*8*d):
        m = ((xx - fx)/(1.6*d))**2 + ((yy - (by+0.2*d))/(0.8*d))**2 < 1
        bg[m] = 40
    R = 1600/mmpx
    om = hs*1000/mmpx / R         # rad/s
    thI = math.radians(attack)
    face = (bx - direction*0.5*d, by)
    # circle param: p(th) = C + R*(dir*sin th, cos th); velocity dir at th: (dir*cos, -sin) -> attack = -th? choose th so that
    # screen-up slope = -(-sin th)... we want motion angle (up positive) = attack -> sin th = -tan? use th = -attack(rad)
    th_imp = -math.radians(attack) * (-1)  # motion y (down +) = -R*sin(th)*om ; up angle = sin(th) -> th = attack
    th_imp = math.radians(attack)
    Cx = face[0] - direction*R*math.sin(th_imp)
    Cy = face[1] - R*math.cos(th_imp)
    def head_pos(tau):
        th = th_imp + om*tau
        return Cx + direction*R*math.sin(th), Cy + R*math.cos(th), th
    bvpx = bs*1000/mmpx
    la = math.radians(launch)
    def ball_pos(tau):
        if tau <= 0: return bx, by
        return bx + direction*bvpx*tau*math.cos(la), by - bvpx*tau*math.sin(la)
    ts = []
    frames = []
    n_pre, n_post = 150, 36
    k_list = list(range(-n_pre, n_post))
    if drop: k_list = [k for k in k_list if k not in drop]
    for k in k_list:
        tau_end = k/fps
        img = bg.copy()
        acc = np.zeros((H, W)); nsub = 6
        for s in range(nsub):
            tau = tau_end - exposure*s/(nsub-1)
            f = bg.copy()
            # ball
            px, py = ball_pos(tau)
            f[((xx-px)**2 + (yy-py)**2) < (d/2)**2] = 232
            # head (only near impact; far away frames: out of strip)
            if tau > -0.25:
                hx, hy, th = head_pos(tau)
                # head body: rectangle behind face along motion dir, 2.3d long, 1.1d tall
                ux, uy = direction*math.cos(th), -math.sin(th)
                rx = (xx - hx)*ux + (yy - hy)*uy            # along motion (+ ahead)
                ry = -(xx - hx)*uy + (yy - hy)*ux
                hm = (rx < 0) & (rx > -2.4*d) & (np.abs(ry + 0.0) < 0.55*d)
                f[hm] = 28
                # shaft toward circle center
                vx, vy = Cx - hx, Cy - hy; L = math.hypot(vx, vy); vx/=L; vy/=L
                t = (xx - (hx - ux*1.8*d))*vx + (yy - (hy - uy*1.8*d))*vy
                dist = np.abs((xx - (hx - ux*1.8*d))*vy - (yy - (hy - uy*1.8*d))*vx)
                f[(t > 0) & (dist < 0.12*d)] = 45
            acc += f
        img = acc/nsub + rng.normal(0, noise, (H, W)) + rng.normal(0, flicker)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        ts.append(k/fps + 5.0)
    trig = k_list.index(0)  # frame where head reaches ball roughly
    return dict(frames=frames, times=np.array(ts), ball=(bx, by, d), trig=trig, fps=fps)
