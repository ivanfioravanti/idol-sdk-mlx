"""RePo layer variants: current vs pre-concatenated / pre-transposed FP32 weights. python -m perf.micro_repo2"""
import math, os, time
import mlx.core as mx, mlx.nn as nn, numpy as np
from idol import mlx_dit
from perf.common import WEIGHTS, CONFIG
model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
repo = model.repo_layers[3]


def bench(f, *a, n=50):
    mx.eval(f(*a)); t = time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter() - t) / n * 1000


Wgc_t = mx.contiguous(mx.concatenate([repo.gate_map.weight, repo.content_map.weight]).T)
Wf_t = mx.contiguous(repo.final_map.weight.T)
mx.eval(Wgc_t, Wf_t)


def variant(x):
    h = repo.norm(x.astype(mx.float32))
    gate, content = mx.split(h @ Wgc_t, 2, axis=-1)
    out = (nn.silu(gate) * content) @ Wf_t
    B, L, _ = out.shape
    delta = out.reshape(B, L, repo.num_heads, 3, 1)
    return mx.concatenate([mlx_dit.clamp_mul(delta[..., i, :], f) * math.pi
                           for i, f in enumerate((repo.freqs_0, repo.freqs_1, repo.freqs_2))], axis=-1)


for L in (12294, 8195):
    x = mx.random.normal((1, L, 1024)).astype(mx.float16)
    a, b = mx.compile(repo)(x), mx.compile(variant)(x)
    err = float(mx.abs(a - b).max())
    print(f'L={L} current {bench(mx.compile(repo), x):.3f} ms | transposed weights {bench(mx.compile(variant), x):.3f} ms | max diff {err:.2e}')
