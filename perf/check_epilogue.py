"""Fused GEMM epilogues vs the MLX ops they replace (compiled, as in the passes): bit-identity and time.
python -m perf.check_epilogue"""
import time
import mlx.core as mx
import mlx.nn as nn
from idol import mlx_dit


def bench(f, n=20):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n * 1000


gelu = nn.GELU(approx='tanh')
for L in (12294, 8195):
    x = (mx.random.normal((1, L, 1024)) * 2).astype(mx.float16)
    h = (mx.random.normal((1, L, 4096))).astype(mx.float16)
    res = (mx.random.normal((1, L, 1024)) * 3).astype(mx.float16)
    gate = (mx.random.normal((1, 1, 1024)) * 0.5).astype(mx.float16)
    w1, w2, wo = [(mx.random.normal(s) * s[0] ** -0.5).astype(mx.float16) for s in ((1024, 4096), (4096, 1024), (1024, 1024))]
    b1, b2, bo = [(mx.random.normal((n,)) * 0.3).astype(mx.float16) for n in (4096, 1024, 1024)]
    mx.eval(x, h, res, gate, w1, w2, wo, b1, b2, bo)
    ref_gelu = mx.compile(lambda x: gelu(mlx_dit.gemm_bias(x, w1, b1)))
    new_gelu = mx.compile(lambda x: mlx_dit.gemm_bias(x, w1, b1, 'gelu'))
    ref_res = mx.compile(lambda h, res: res + mlx_dit.gemm_bias(h, w2, b2) * gate)
    new_res = mx.compile(lambda h, res: mlx_dit.gemm_bias(h, w2, b2, 'gate_res', gate, res))
    ref_out = mx.compile(lambda x, res: res + mlx_dit.gemm_bias(x, wo, bo) * gate)
    new_out = mx.compile(lambda x, res: mlx_dit.gemm_bias(x, wo, bo, 'gate_res', gate, res))
    ref_mlp = mx.compile(lambda x, res: res + mlx_dit.gemm_bias(gelu(mlx_dit.gemm_bias(x, w1, b1)), w2, b2) * gate)
    new_mlp = mx.compile(lambda x, res: mlx_dit.gemm_bias(mlx_dit.gemm_bias(x, w1, b1, 'gelu'), w2, b2, 'gate_res', gate, res))
    for name, r, n, a in (('fc1+gelu', ref_gelu, new_gelu, (x,)), ('fc2+gate+res', ref_res, new_res, (h, res)),
                          ('out+gate+res', ref_out, new_out, (x, res)), ('mlp', ref_mlp, new_mlp, (x, res))):
        ro, no = r(*a), n(*a)
        ne = float((ro != no).astype(mx.float32).mean())
        print(f'L={L} {name}: MLX ops {bench(lambda: r(*a)):.3f} ms, fused {bench(lambda: n(*a)):.3f} ms, '
              f'differing {ne:.2e} max abs {float(mx.abs(ro.astype(mx.float32) - no.astype(mx.float32)).max()):.2e}', flush=True)
