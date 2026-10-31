import os, time, math, mlx.core as mx, mlx.nn as nn, numpy as np
from idol import mlx_dit
from perf.common import WEIGHTS, CONFIG
model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
repo = model.repo_layers[3]
def bench(f, *a, n=50):
    mx.eval(f(*a)); t = time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter() - t) / n * 1000
for L in (12294, 8195, 2048):
    h = (mx.random.normal((1, L, 1024))).astype(mx.float16)
    W = mx.concatenate([repo.gate_map.weight, repo.content_map.weight])
    f_norm = mx.compile(lambda h: repo.norm(h.astype(mx.float32)))
    hn = f_norm(h)
    f_mm = mx.compile(lambda hn: hn @ W.T)
    gc = f_mm(hn)
    def tail(gc):
        g, c = mx.split(gc, 2, axis=-1); out = repo.final_map(nn.silu(g) * c)
        B, L, _ = out.shape; d = out.reshape(B, L, 16, 3, 1)
        return mx.concatenate([mlx_dit.clamp_mul(d[..., i, :], f) * math.pi for i, f in enumerate((repo.freqs_0, repo.freqs_1, repo.freqs_2))], axis=-1)
    f_tail = mx.compile(tail)
    print(f'L={L} full {bench(mx.compile(repo), h):.3f} | norm {bench(f_norm, h):.3f} | mm {bench(f_mm, hn):.3f} | tail {bench(f_tail, gc):.3f} ms')
