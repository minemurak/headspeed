from sim import render; from analyzer import analyze
cases=[dict(hs=42,bs=62,launch=12,attack=-3,d=24),dict(hs=38,bs=50,launch=18,attack=-5,d=20,direction=-1,seed=1),
dict(hs=48,bs=71,launch=10,attack=2,d=30,seed=2),dict(hs=33,bs=46,launch=24,attack=-6,d=22,seed=3,drop={-3,2}),
dict(hs=45,bs=66,launch=14,attack=0,d=18,direction=-1,seed=4,noise=7),
dict(hs=42,bs=62,launch=12,attack=-3,d=24,seed=5,pitch=30),dict(hs=36,bs=48,launch=22,attack=-5,d=22,direction=-1,seed=6,pitch=40)]
for c in cases:
    s=render(**c); r=analyze(s['frames'],s['times'],s['ball'],s['trig'],pitch=c.get('pitch',0))
    f=lambda k:(f"{r[k]:.2f}" if k in r else '-')
    print({k:c[k] for k in ('hs','bs','launch','attack','pitch') if k in c}, '->', 'hs',f('hs'),'bs',f('bs'),'smash',f('smash'),'(true %.2f)'%(c['bs']/c['hs']),'launch',f('launch'),'attack',f('attack'),r.get('headFrames'),r.get('ballFrames'),r['notes'])
