"""Block LayerNorm + adaLN modulation: MLX ops (cast, fast.layer_norm, cast, mul, add) cost. python -m perf.micro_norm"""
import time
import mlx.core as mx, mlx.nn as nn
from idol import mlx_dit


def bench(f, *a, n=50):
    mx.eval(f(*a)); t = time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter() - t) / n * 1000


norm = nn.LayerNorm(1024, eps=1e-6, affine=False)
for L in (12294, 8195):
    x = mx.random.normal((1, L, 1024)).astype(mx.float16)
    sc, sh = [(mx.random.normal((1, 1, 1024)) * 0.1).astype(mx.float16) for _ in range(2)]
    f = mx.compile(lambda x, sc, sh: mlx_dit.norm32(norm, x) * (1 + sc) + sh)
    g = mx.compile(lambda x: x * 2)
    print(f'L={L} norm+mod {bench(f, x, sc, sh):.3f} ms | copy-like x*2 {bench(g, x):.3f} ms')
