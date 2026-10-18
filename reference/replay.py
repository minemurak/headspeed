"""Re-run the analyzers on a swing exported from the app (result card -> 書き出す):
    python3 replay.py swing.mov swing.json
Needs ffmpeg on PATH to decode the movie. Prints the app's result next to this one, so
real swings can be used to check and tune the reference (and then the Swift port)."""
import sys, json, math, subprocess
import numpy as np

def load(mov, meta):
    info = json.load(open(meta))
    W, H = info['width'], info['height']
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', mov, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                         check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    n = min(len(frames), len(info['times']))
    return [frames[k] for k in range(n)], np.array(info['times'][:n]), info

def main(mov, meta):
    frames, times, info = load(mov, meta)
    ball = (info['ballX'], info['ballY'], info['ballD'])
    if info['rear']:
        from analyzer_rear import analyze_rear
        res = analyze_rear(frames, times, ball, info['triggerIndex'], info['focal'], info['centerX'],
                           info['centerY'], pitch=info['cameraPitch'], club=info['club'])
    else:
        from analyzer import analyze
        res = analyze(frames, times, ball, info['triggerIndex'], club=info['club'],
                      pitch=math.degrees(info['cameraPitch']), overhead=info.get('overhead', False))
    print('frames', len(frames), f'{frames[0].shape[1]}x{frames[0].shape[0]}',
          'fps %.1f' % (1/np.median(np.diff(times))), 'trigger', info['triggerIndex'], 'ball', ball)
    from shake import camera_moved
    print('camera moved at the trigger:', camera_moved(frames, info['triggerIndex']))
    print('app      ', {k: info.get(k) for k in ('headSpeed', 'ballSpeed', 'launch', 'attack', 'notes')})
    print('reference', {k: v for k, v in res.items() if not isinstance(v, (list, np.ndarray)) or k == 'notes'})

if __name__ == '__main__':
    main(*sys.argv[1:3])
