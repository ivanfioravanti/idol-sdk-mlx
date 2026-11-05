"""rope + RMSNorm kernel variants vs the production one (bit-identity and time) at the cond-pass shape.
python -m perf.micro_rope"""
import time
import mlx.core as mx
from idol import mlx_dit

# one thread per rotation pair for q and k of the same (token, head): one sincos, one angle read
SRC2 = """
    uint i = thread_position_in_grid.x;               // rotation pair
    uint h = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint H = gamma_q_shape[0], L = qkv_shape[1];
    uint src = (t * 3 * H + h) * 64 + 2 * i;
    const half2 xq = *(const device half2*)(qkv + src), xk = *(const device half2*)(qkv + src + H * 64);
    float a = ang[(t * H + h) * 32 + i];
    float c = metal::precise::cos(a), s = metal::precise::sin(a);
    float q0 = float(half(float(xq[0]) * c - float(xq[1]) * s));
    float q1 = float(half(float(xq[0]) * s + float(xq[1]) * c));
    float k0 = float(half(float(xk[0]) * c - float(xk[1]) * s));
    float k1 = float(half(float(xk[0]) * s + float(xk[1]) * c));
    float nq = metal::max(metal::sqrt(simd_sum(q0 * q0 + q1 * q1)), 1e-12f);
    float nk = metal::max(metal::sqrt(simd_sum(k0 * k0 + k1 * k1)), 1e-12f);
    uint g = h * 64 + 2 * i;
    uint dst = (((t / L) * H + h) * L + t % L) * 64 + 2 * i;
    *(device half2*)(q + dst) = half2(half(q0 / nq * gamma_q[g] * 8.0f), half(q1 / nq * gamma_q[g + 1] * 8.0f));
    *(device half2*)(k + dst) = half2(half(k0 / nk * gamma_k[g] * 8.0f), half(k1 / nk * gamma_k[g + 1] * 8.0f));
"""
k2 = mx.fast.metal_kernel(name='rope2', input_names=['qkv', 'ang', 'gamma_q', 'gamma_k'], output_names=['q', 'k'], source=SRC2)


def rope2(qkv, ang, gq, gk, tg=8):
    B, L, _, H, D = qkv.shape
    return k2(inputs=[qkv, ang, gq, gk], grid=(32, H, B * L), threadgroup=(32, tg, 1), output_shapes=[(B, H, L, D)] * 2,
              output_dtypes=[mx.float16] * 2)


def bench(f, n=50):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n * 1000


L, H = 12294, 16
qkv = mx.random.normal((1, L, 3, H, 64)).astype(mx.float16)
ang = mx.random.normal((1, L, H, 32)) * 30
gq, gk = mx.random.uniform(0.5, 1.5, (H, 64)), mx.random.uniform(0.5, 1.5, (H, 64))
mx.eval(qkv, ang, gq, gk)
ref = mlx_dit.rope_rmsnorm(qkv, ang, gq, gk)
print(f'production {bench(lambda: mlx_dit.rope_rmsnorm(qkv, ang, gq, gk)):.3f} ms')
for tg in (4, 8, 16):
    out = rope2(qkv, ang, gq, gk, tg)
    same = all(bool(mx.array_equal(a, b)) for a, b in zip(out, ref))
    print(f'rope2 tg{tg} {bench(lambda: rope2(qkv, ang, gq, gk, tg)):.3f} ms bit-identical {same}')
