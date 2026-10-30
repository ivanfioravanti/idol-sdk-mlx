"""RePo tail (silu(gate) * content -> final_map -> clamp_mul angles) as one Metal kernel vs the MLX ops.
python -m perf.repo_tail"""
import math, os, time
import mlx.core as mx, mlx.nn as nn, numpy as np
from idol import mlx_dit
from perf.common import WEIGHTS, CONFIG

SRC = """
    const uint t = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    threadgroup float u[128];
    threadgroup float dl[48];
    const float g = gc[t * 256 + lid], c = gc[t * 256 + 128 + lid];
    u[lid] = g / (1.0f + metal::precise::exp(-g)) * c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 48) {
        float acc = 0;
        for (int k = 0; k < 128; k++) acc = metal::fma(u[k], wf_t[k * 48 + lid], acc);
        dl[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = lid; o < 512; o += 128) {
        const uint h = o / 32, p = o % 32;
        const float d = dl[h * 3 + axis[p]], f = freqs[p], ft = metal::precise::tanh(f);
        ang[t * 512 + o] = (d * ft + d * (f - ft)) * float(M_PI_F);
    }
"""
LN_SRC = """
    // one threadgroup of 256 per token, 4 channels per thread: x FP16 [N, 1024] -> LayerNorm in FP32 with weight, bias
    const uint t = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint sg = lid / 32, lane = lid % 32;
    threadgroup float part[8];
    const float4 v = float4(*(const device half4*)(x + t * 1024 + lid * 4));
    float s = simd_sum(v.x + v.y + v.z + v.w);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0;
    for (int i = 0; i < 8; i++) mean += part[i];
    mean /= 1024.0f;
    const float4 dv = v - mean;
    float q = simd_sum(dot(dv, dv));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0;
    for (int i = 0; i < 8; i++) var += part[i];
    const float r = metal::precise::rsqrt(var / 1024.0f + 1e-5f);
    *(device float4*)(out + t * 1024 + lid * 4) = dv * r * *(const device float4*)(w + lid * 4) + *(const device float4*)(b + lid * 4);
"""
ln_kern = mx.fast.metal_kernel(name='repo_ln', input_names=['x', 'w', 'b'], output_names=['out'], source=LN_SRC)
kern = mx.fast.metal_kernel(name='repo_tail', input_names=['gc', 'wf_t', 'freqs', 'axis'], output_names=['ang'], source=SRC)

if __name__ == '__main__':
    model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
    repo = model.repo_layers[3]
    Wgc_t = mx.contiguous(mx.concatenate([repo.gate_map.weight, repo.content_map.weight]).T)
    Wf_t = mx.contiguous(repo.final_map.weight.T)
    freqs = mx.concatenate([repo.freqs_0, repo.freqs_1, repo.freqs_2])
    axis = mx.array(sum([[i] * f.size for i, f in enumerate((repo.freqs_0, repo.freqs_1, repo.freqs_2))], []), mx.int32)
    mx.eval(Wgc_t, Wf_t, freqs, axis)

    def bench(f, *a, n=50):
        mx.eval(f(*a)); t = time.perf_counter()
        for _ in range(n): mx.eval(f(*a))
        return (time.perf_counter() - t) / n * 1000

    def fused(x):
        B, L, _ = x.shape
        h = ln_kern(inputs=[x, repo.norm.weight, repo.norm.bias], grid=(B * L * 256, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(B, L, 1024)], output_dtypes=[mx.float32])[0] if LN else repo.norm(x.astype(mx.float32))
        gc = h @ Wgc_t
        return kern(inputs=[gc, Wf_t, freqs, axis], grid=(B * L * 128, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(B, L, 16, 32)], output_dtypes=[mx.float32])[0]

    LN = False
    for L in (12294, 8195):
        x = mx.random.normal((1, L, 1024)).astype(mx.float16)
        a, b = mx.compile(repo)(x), mx.compile(fused)(x)
        rel = float(mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(a.flatten()))
        print(f'L={L} current {bench(mx.compile(repo), x):.3f} ms | fused tail {bench(mx.compile(fused), x):.3f} ms | '
              f'rel {rel:.2e} max {float(mx.abs(a - b).max()):.2e} (|ang| max {float(mx.abs(a).max()):.1f})')
        LN = True
        b = mx.compile(fused)(x)
        rel = float(mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(a.flatten()))
        hn = repo.norm(x.astype(mx.float32))
        h2 = ln_kern(inputs=[x, repo.norm.weight, repo.norm.bias], grid=(L * 256, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(1, L, 1024)], output_dtypes=[mx.float32])[0]
        print(f'   + LN kernel {bench(mx.compile(fused), x):.3f} ms | rel {rel:.2e} max {float(mx.abs(a - b).max()):.2e} '
              f'| LN rel {float(mx.linalg.norm((hn - h2).flatten()) / mx.linalg.norm(hn.flatten())):.2e}')
        LN = False
