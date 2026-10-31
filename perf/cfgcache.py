"""Approximate: reuse the unconditional pass across steps. python -m perf.cfgcache VARIANT[,VARIANT] ; VARIANT = {u|d}{period}[w{warm}]
u: reuse v_uncond, d: reuse the guidance delta (v_cond - v_uncond). Fresh uncond on the first `warm` steps, the last 2,
and every `period`-th step."""
import json, os, sys, time, re, mlx.core as mx, numpy as np
from idol import mlx_dit
from perf.common import WEIGHTS, CONFIG, IMAGES
model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
ls, jn = mx.compile(model.latent_stream), mx.compile(model.joint)
def run(d, kind, period, warm, steps=50, w=3.0, rescale=3.0):
    sample = {'latent': mx.array(d['latent']), 'camera': mx.array(d['camera'])}
    cond = model.condition({'feature1': mx.array(d['feature1']), 'feature2': mx.array(d['feature2'])})
    neg = model.condition({'feature1': mx.zeros(d['feature1'].shape), 'feature2': mx.zeros(d['feature2'].shape)})
    t_seq = np.linspace(1, 0, steps + 1); t_seq = rescale * t_seq / (1 + (rescale - 1) * t_seq)
    cache = None; prev = None; fresh_n = 0
    for i in range(steps):
        s = ls(sample, mx.full((1,), 1000 * t_seq[i], dtype=mx.float32)); vc = jn(s, *cond)
        fresh = i < warm or i >= steps - 2 or (i - warm) % period == 0
        if fresh:
            vu = jn(s, *neg); fresh_n += 1
            cache = vu if kind == 'u' else {k: vc[k] - vu[k] for k in vc}
        if kind == 'u': v = {k: (1 + w) * vc[k] - w * cache[k] for k in vc}
        else: v = {k: vc[k] + w * cache[k] for k in vc}
        dt = mx.array(t_seq[i] - t_seq[i + 1], mx.float32)
        sample = {k: sample[k] - v[k] * dt for k in sample}
        mx.async_eval(sample, cache)
        if prev is not None: mx.eval(prev)
        prev = sample
    mx.eval(sample); return sample, fresh_n
for var in sys.argv[1].split(','):
    m = re.fullmatch(r'([ud])(\d+)(?:w(\d+))?', var); kind, period, warm = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    for name in IMAGES:
        d = np.load(f'perf/inputs_{name}.npz')
        if name == 'avocado': run(d, kind, period, warm, steps=3)
        t0 = time.perf_counter(); out, fresh_n = run(d, kind, period, warm); dt = time.perf_counter() - t0
        np.savez(f'perf/out_cfg-{var}_{name}.npz', latent=np.array(out['latent']), camera=np.array(out['camera']))
        row = dict(tag=f'cfg-{var}', image=name, median_s=round(dt, 3), fresh_uncond=fresh_n, time=time.strftime('%H:%M:%S'))
        print(json.dumps(row), flush=True); open('perf/results.jsonl', 'a').write(json.dumps(row) + '\n')
