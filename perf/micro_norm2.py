"""norm_mod / RePo LayerNorm: 256-thread threadgroup per token (production) vs one simdgroup per token with the same
reduction tree (bit-identical). python -m perf.micro_norm2"""
import time
import mlx.core as mx
from idol import mlx_dit

NORM_MOD_SG = """
    const uint t = threadgroup_position_in_grid.x * TPG + simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint C = 1024, b = t / x_shape[1];
    if (t >= x_shape[0] * x_shape[1]) return;
    float4 v[8];
    float part[8];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 8; j++) {
        v[j] = float4(*(const device half4*)(x + t * C + j * 128 + lane * 4));
        part[j] = simd_sum(v[j].x + v[j].y + v[j].z + v[j].w);
    }
    float mean = 0;
    for (int j = 0; j < 8; j++) mean += part[j];
    mean /= C;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 8; j++) {
        v[j] -= mean;
        part[j] = simd_sum(dot(v[j], v[j]));
    }
    float var = 0;
    for (int j = 0; j < 8; j++) var += part[j];
    const float r = metal::precise::rsqrt(var / C + 1e-6f);
    #pragma clang loop unroll(full)
    for (int j = 0; j < 8; j++) {
        const uint o = j * 128 + lane * 4;
        const half4 y = half4(v[j] * r);
        const half4 m = half4(1.0h) + *(const device half4*)(scale + b * C + o);
        const half4 p = y * m;
        *(device half4*)(out + t * C + o) = p + *(const device half4*)(shift + b * C + o);
    }
"""
_k = {}


def norm_mod_sg(x, scale, shift, tpg):
    if tpg not in _k:
        _k[tpg] = mx.fast.metal_kernel(name=f'nm_sg{tpg}', input_names=['x', 'scale', 'shift'], output_names=['out'],
                                       source=f'constexpr uint TPG = {tpg};\n' + NORM_MOD_SG)
    B, L, C = x.shape
    return _k[tpg](inputs=[x, scale, shift], grid=((B * L + tpg - 1) // tpg * tpg * 32, 1, 1), threadgroup=(tpg * 32, 1, 1),
                   output_shapes=[x.shape], output_dtypes=[mx.float16])[0]


def bench(f, n=50):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n * 1000


for L in (12294, 8195):
    x = (mx.random.normal((1, L, 1024)) * 3 + 1).astype(mx.float16)
    scale, shift = [(mx.random.normal((1, 1, 1024)) * 0.3).astype(mx.float16) for _ in range(2)]
    mx.eval(x, scale, shift)
    ref = mlx_dit.norm_mod(x, scale, shift)
    line = f'L={L} norm_mod production {bench(lambda: mlx_dit.norm_mod(x, scale, shift)):.3f} ms'
    for tpg in (1, 2, 4, 8):
        o = norm_mod_sg(x, scale, shift, tpg)
        line += f' | sg/token, {tpg} tokens/TG {bench(lambda: norm_mod_sg(x, scale, shift, tpg)):.3f} identical {bool(mx.array_equal(o, ref))}'
    print(line, flush=True)
