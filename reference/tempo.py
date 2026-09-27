"""Reference tempo detection from per-frame whole-image motion energy. Mirrored by TempoAnalyzer.swift."""
def tempo(times, energy, t_imp):
    pts = [(t, e) for t, e in zip(times, energy) if t_imp - 4.0 <= t <= t_imp]
    if len(pts) < 60: return None
    ts = [p[0] for p in pts]; raw = [p[1] for p in pts]
    n = len(raw); w = 5
    sm = [sum(raw[max(0, i-w):min(n, i+w+1)]) / (min(n, i+w+1) - max(0, i-w)) for i in range(n)]
    def argext(lo, hi, better):
        best = None
        for i, t in enumerate(ts):
            if lo <= t <= hi and (best is None or better(sm[i], sm[best])): best = i
        return best
    iTop = argext(t_imp - 0.7, t_imp - 0.12, lambda a, b: a < b)
    if iTop is None: return None
    tTop = ts[iTop]
    iPeak = argext(tTop - 1.5, tTop - 0.1, lambda a, b: a > b)
    if iPeak is None: return None
    iBase = argext(tTop - 3.0, ts[iPeak], lambda a, b: a < b)
    if iBase is None: return None
    peak, base = sm[iPeak], sm[iBase]
    if peak <= base * 1.5: return None
    thr = base + 0.1 * (peak - base)
    i0 = None
    for i in range(iPeak, -1, -1):
        if sm[i] < thr: i0 = i; break
    if i0 is None: return None
    bsw, dsw = tTop - ts[i0], t_imp - tTop
    if not (0.4 <= bsw <= 2.0 and 0.15 <= dsw <= 0.6): return None
    return bsw, dsw, bsw / dsw

if __name__ == '__main__':
    import math, random
    random.seed(1)
    for Tbs, Tds in ((0.81, 0.27), (0.95, 0.30), (0.7, 0.28)):
        fps = 240; t_imp = 10.0; t0 = t_imp - Tds - Tbs
        times = [t_imp - 3.5 + i / fps for i in range(int(3.6 * fps))]
        def omega(t):  # angular speed of the club
            if t < t0: return 0.0
            if t < t_imp - Tds: s = (t - t0) / Tbs; return 3.5 / Tbs * math.pi / 2 * math.sin(math.pi * s)
            return 26 * (t - (t_imp - Tds)) / Tds
        energy = [30 + 400 * omega(t) + random.gauss(0, 15) for t in times]
        print((Tbs, Tds, round(Tbs / Tds, 2)), '->', tuple(round(v, 2) for v in tempo(times, energy, t_imp)))
