"""MLX port of the denoiser (models/dit.py) and its sampler (sampler.py): the 50-step loop that dominates a request.
Same module tree and parameter names as the torch model, so it loads denoiser.safetensors as is. The image encoders
and the decoders stay in torch; sample_flow takes and returns torch TensorDicts."""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from tensordict import TensorDict
from tqdm import tqdm

from .models.layers import PcdAbsolutePositionEmbedder

F16, F32 = mx.float16, mx.float32


def tf32_split(a):
    """a = hi + lo with hi exact in TF32 (the low 13 mantissa bits cleared)."""
    hi = mx.view(mx.view(a, mx.uint32) & 0xFFFFE000, F32)
    return hi, a - hi


def _fp32_matmul_is_exact():
    """Whether this GPU's FP32 matmul keeps full FP32 precision. On M5 (Neural Accelerators) MLX rounds FP32 operands
    to TF32 (~1e-3 relative error); M1-M4 multiply in full FP32 (~5e-7), where 3xTF32 only triples the cost."""
    rng = np.random.default_rng(0)
    x, w = rng.standard_normal((256, 1024), np.float32), rng.standard_normal((256, 1024), np.float32)
    ref = x.astype(np.float64) @ w.astype(np.float64).T
    y = np.array(mx.array(x) @ mx.array(w).T, np.float64)
    return float(np.linalg.norm(y - ref) / np.linalg.norm(ref)) < 1e-5


FP32_MATMUL_EXACT = _fp32_matmul_is_exact()


def matmul32(x, w):
    """x @ w at full FP32 precision: where MLX's GPU matmul rounds FP32 operands to TF32 (~1e-3 relative error,
    amplified by the RoPE angles), this runs three TF32-exact products ("3xTF32"); elsewhere one plain product."""
    if FP32_MATMUL_EXACT:
        return x @ w
    x_hi, x_lo = tf32_split(x)
    w_hi, w_lo = tf32_split(w)
    return x_hi @ w_hi + (x_hi @ w_lo + x_lo @ w_hi)


class Linear(nn.Linear):
    """nn.Linear, at full precision for FP32 inputs (matmul32). FP16 is exact in TF32 as is. After
    LatentSeqMMFlowModel.to_f16() the weight is held as weight_t [in, out], contiguous: MLX's GEMM is ~5% faster on it
    than on the weight.T view."""
    def __call__(self, x):
        if 'weight_t' in self:
            return mx.addmm(self.bias, x, self.weight_t) if 'bias' in self else x @ self.weight_t
        if x.dtype != F32:
            return super().__call__(x)
        y = matmul32(x, self.weight.T)
        return y + self.bias if 'bias' in self else y


def run(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def norm32(norm, x):
    return norm(x.astype(F32)).astype(x.dtype)


# layers.apply_rotary_emb then MultiHeadRMSNorm on q and k, fused: one SIMD group per (token, head, q or k), one
# thread per rotation pair. Rotates in FP32 and rounds to FP16 (torch's type_as), normalises in FP32, and writes the
# [B, H, L, D] layout SDPA takes. As separate MLX ops this was ~30% of a block for memory-bound work.
_rope_rmsnorm_kernel = mx.fast.metal_kernel(
    name='idol_rope_rmsnorm',
    input_names=['qkv', 'ang', 'gamma_q', 'gamma_k'],
    output_names=['q', 'k'],
    source="""
    uint i = thread_position_in_grid.x;               // rotation pair: elements 2i, 2i + 1 of the 64-wide head
    uint H = gamma_q_shape[0];
    uint which = thread_position_in_grid.y / H;       // 0: q, 1: k
    uint h = thread_position_in_grid.y % H;
    uint t = thread_position_in_grid.z;               // b * L + token
    uint L = qkv_shape[1];
    uint src = ((t * 3 + which) * H + h) * 64 + 2 * i;
    float x0 = qkv[src], x1 = qkv[src + 1];
    float a = ang[(t * H + h) * 32 + i];
    float c = metal::precise::cos(a), s = metal::precise::sin(a);
    float y0 = float(half(x0 * c - x1 * s));
    float y1 = float(half(x0 * s + x1 * c));
    float norm = metal::max(metal::sqrt(simd_sum(y0 * y0 + y1 * y1)), 1e-12f);
    uint g = h * 64 + 2 * i;
    uint dst = (((t / L) * H + h) * L + t % L) * 64 + 2 * i;
    if (which == 0) {
        q[dst] = half(y0 / norm * gamma_q[g] * 8.0f);
        q[dst + 1] = half(y1 / norm * gamma_q[g + 1] * 8.0f);
    } else {
        k[dst] = half(y0 / norm * gamma_k[g] * 8.0f);
        k[dst + 1] = half(y1 / norm * gamma_k[g + 1] * 8.0f);
    }
""")


def rope_rmsnorm(qkv, ang, gamma_q, gamma_k):
    """qkv [B, L, 3, H, 64] FP16, ang [B, L, H, 32] FP32 -> rotated, RMS-normalised q, k, each [B, H, L, 64] FP16"""
    B, L, _, H, D = qkv.shape
    assert D == 64 and qkv.dtype == F16
    return _rope_rmsnorm_kernel(inputs=[qkv, ang, gamma_q, gamma_k], grid=(32, 2 * H, B * L), threadgroup=(32, 8, 1),
                                output_shapes=[(B, H, L, D)] * 2, output_dtypes=[F16] * 2)


# Flash attention for head_dim 64, faster here than mx.fast.scaled_dot_product_attention (27.3 -> 26.1 ms at 12,294
# tokens, 13.1 -> 12.2 ms at 8,195 with the key bias): MLX's kernel gives each simdgroup 8 query rows and reloads Q and K
# fragments from threadgroup memory for every MMA; here a simdgroup keeps 16 query rows of Q in registers, so each K / V
# fragment feeds two MMAs. FP16 operands, FP32 scores, softmax and accumulation, the same online softmax as MLX. Reads
# V straight from the qkv projection and writes [B, L, H * 64], the layout the output projection takes.
_ATTN_SOURCE = """
    constexpr int D = 64, TD = 8, NSG = 4, TQ = 2, BQ = NSG * TQ * 8, BK = 32, TK = BK / 8, LD = D + 8;
    threadgroup half Ks[BK * LD];      // K block [key][d], read transposed by simdgroup_load
    threadgroup half Vs[BK * LD];      // V block [key][d]

    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int qb = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
    const int H = q_shape[1], L = q_shape[2], HD = H * D;
    const short fm = (lane / 4 & 4) + (lane / 2) % 4, fn = (lane / 4 & 2) * 2 + (lane % 2) * 2;  // this lane's fragment row, cols

    const device half* Q = q + (long)(b * H + h) * L * D;
    const device half* K = k + (long)(b * H + h) * L * D;
    const device half* V = qkv + (long)b * L * 3 * HD + 2 * HD + h * D;    // row stride 3 * HD
    const float scale = scale_[0] * M_LOG2E_F;

    simdgroup_half8x8 Qf[TQ][TD];
    simdgroup_float8x8 Of[TQ][TD], Sf[TQ][TK];
    float m[TQ], s[TQ];
    const int row0 = qb * BQ + sg * TQ * 8;
    #pragma clang loop unroll(full)
    for (int i = 0; i < TQ; i++) {
        const int r = metal::min(row0 + i * 8 + fm, L - 1);
        #pragma clang loop unroll(full)
        for (int d = 0; d < TD; d++) {
            const half2 x = *(const device half2*)(Q + (long)r * D + d * 8 + fn);
            thread auto& e = Qf[i][d].thread_elements();
            e[0] = x[0]; e[1] = x[1];
            Of[i][d] = simdgroup_float8x8(0);
        }
        m[i] = -FLT_MAX; s[i] = 0;
    }

    constexpr int PER = BK * D / (NSG * 32);    // halfs of K and of V each thread moves per block
    const int tid = sg * 32 + lane, lk = tid / (D / PER), ld0 = tid % (D / PER) * PER;
    for (int kb = 0; kb < (L + BK - 1) / BK; kb++) {
        const int key = kb * BK + lk;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int j = 0; j < PER; j += 4) {
            *(threadgroup half4*)(Ks + lk * LD + ld0 + j) = key < L ? *(const device half4*)(K + (long)key * D + ld0 + j) : half4(0);
            *(threadgroup half4*)(Vs + lk * LD + ld0 + j) = key < L ? *(const device half4*)(V + (long)key * 3 * HD + ld0 + j) : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int i = 0; i < TQ; i++)
            _Pragma("clang loop unroll(full)") for (int c = 0; c < TK; c++) Sf[i][c] = simdgroup_float8x8(0);
        #pragma clang loop unroll(full)
        for (int d = 0; d < TD; d++) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, Ks + c * 8 * LD + d * 8, LD, ulong2(0, 0), true);
                _Pragma("clang loop unroll(full)") for (int i = 0; i < TQ; i++) simdgroup_multiply_accumulate(Sf[i][c], Qf[i][d], kf, Sf[i][c]);
            }
        }

        const bool tail = (kb + 1) * BK > L;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TQ; i++) {
            float mx_ = m[i];
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                thread auto& e = Sf[i][c].thread_elements();
                const int col = kb * BK + c * 8 + fn;
                e[0] *= scale; e[1] *= scale;
                KEY_BIAS
                if (tail) {
                    if (col >= L) e[0] = -FLT_MAX;
                    if (col + 1 >= L) e[1] = -FLT_MAX;
                }
                mx_ = metal::max(mx_, metal::max(e[0], e[1]));
            }
            mx_ = metal::max(mx_, simd_shuffle_xor(mx_, 1));      // the 4 lanes holding a fragment row
            mx_ = metal::max(mx_, simd_shuffle_xor(mx_, 8));
            float sum = 0;
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                thread auto& e = Sf[i][c].thread_elements();
                e[0] = fast::exp2(e[0] - mx_); e[1] = fast::exp2(e[1] - mx_);
                sum += e[0] + e[1];
            }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 8);
            const float f = fast::exp2(m[i] - mx_);
            m[i] = mx_;
            s[i] = s[i] * f + sum;
            #pragma clang loop unroll(full)
            for (int d = 0; d < TD; d++) {
                thread auto& e = Of[i][d].thread_elements();
                e[0] *= f; e[1] *= f;
            }
        }

        #pragma clang loop unroll(full)
        for (int c = 0; c < TK; c++) {
            #pragma clang loop unroll(full)
            for (int d = 0; d < TD; d++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, Vs + c * 8 * LD + d * 8, LD);
                _Pragma("clang loop unroll(full)") for (int i = 0; i < TQ; i++) simdgroup_multiply_accumulate(Of[i][d], Sf[i][c], vf, Of[i][d]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < TQ; i++) {
        const int r = row0 + i * 8 + fm;
        if (r >= L) continue;
        #pragma clang loop unroll(full)
        for (int d = 0; d < TD; d++) {
            thread auto& e = Of[i][d].thread_elements();
            *(device half2*)(out + ((long)b * L + r) * HD + h * D + d * 8 + fn) = half2(half(e[0] / s[i]), half(e[1] / s[i]));
        }
    }
"""
_attn_kernels = {
    with_bias: mx.fast.metal_kernel(
        name=f'idol_attention{"_bias" if with_bias else ""}',
        input_names=['q', 'k', 'qkv', 'scale_'] + (['key_bias'] if with_bias else []), output_names=['out'],
        source=_ATTN_SOURCE.replace('KEY_BIAS', 'e[0] += M_LOG2E_F * float(key_bias[metal::min(col, L - 1)]);\n'
                                                '                e[1] += M_LOG2E_F * float(key_bias[metal::min(col + 1, L - 1)]);'
                                    if with_bias else ''))
    for with_bias in (False, True)}


def attention(q, k, qkv, scale, key_bias=None):
    """q, k [B, H, L, 64] FP16, qkv [B, L, 3, H, 64] FP16 (v = qkv[:, :, 2]), key_bias [L] (added to the scores) or
    None -> softmax(q k^T * scale + key_bias) v as [B, L, H * 64] FP16"""
    B, H, L, D = q.shape
    assert D == 64 and q.dtype == k.dtype == qkv.dtype == F16
    inputs = [q, k, qkv, mx.array([scale], F32)] + ([key_bias] if key_bias is not None else [])
    return _attn_kernels[key_bias is not None](inputs=inputs, grid=((L + 63) // 64 * 128, H, B), threadgroup=(128, 1, 1),
                                               output_shapes=[(B, L, H * D)], output_dtypes=[F16])[0]


# The two ends of a RePo layer, fused. LayerNorm of FP16 tokens in FP32 in one pass (one threadgroup per token, 4
# channels per thread), replacing a cast and mx.fast.layer_norm. Then, from the gate / content projections:
# silu(gate) * content -> final_map (128 -> 3 per head, FP32 FMAs) -> clamp_mul angles, written as [N, H, 32];
# replaces an FP32 matmul with 48 outputs and the elementwise ops around it. Same FP32 math (angles agree to ~2e-7).
_repo_norm_kernel = mx.fast.metal_kernel(
    name='idol_repo_norm',
    input_names=['x', 'w', 'b'],
    output_names=['out'],
    source="""
    const uint t = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint C = x_shape[2], sg = lid / 32, lane = lid % 32;
    threadgroup float part[8];
    const float4 v = float4(*(const device half4*)(x + t * C + lid * 4));
    float s = simd_sum(v.x + v.y + v.z + v.w);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0;
    for (int i = 0; i < 8; i++) mean += part[i];
    mean /= C;
    const float4 dv = v - mean;
    s = simd_sum(dot(dv, dv));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0;
    for (int i = 0; i < 8; i++) var += part[i];
    const float r = metal::precise::rsqrt(var / C + 1e-5f);
    *(device float4*)(out + t * C + lid * 4) = dv * r * *(const device float4*)(w + lid * 4) + *(const device float4*)(b + lid * 4);
""")

_repo_angles_kernel = mx.fast.metal_kernel(
    name='idol_repo_angles',
    input_names=['gc', 'wf_t', 'freqs', 'axis'],
    output_names=['ang'],
    source="""
    const uint t = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    threadgroup float u[128];
    threadgroup float delta[48];
    const float g = gc[t * 256 + lid], c = gc[t * 256 + 128 + lid];
    u[lid] = g / (1.0f + metal::precise::exp(-g)) * c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 48) {
        float acc = 0;
        for (int k = 0; k < 128; k++) acc = metal::fma(u[k], wf_t[k * 48 + lid], acc);
        delta[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = lid; o < 512; o += 128) {
        const uint h = o / 32, p = o % 32;
        const float d = delta[h * 3 + axis[p]], f = freqs[p], ft = metal::precise::tanh(f);
        ang[t * 512 + o] = (d * ft + d * (f - ft)) * float(M_PI_F);     // clamp_mul(d, f) * pi
    }
""")


# norm32(LayerNorm without affine, eps 1e-6) then adaLN modulation, as one pass: LayerNorm in FP32, rounded to FP16,
# then x * (1 + scale) + shift in FP16 arithmetic, as the separate MLX ops do (a cast, mx.fast.layer_norm, a cast and a
# fused multiply-add: three passes over the tokens).
_norm_mod_kernel = mx.fast.metal_kernel(
    name='idol_norm_mod',
    input_names=['x', 'scale', 'shift'],
    output_names=['out'],
    source="""
    const uint t = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint C = x_shape[2], b = t / x_shape[1], sg = lid / 32, lane = lid % 32;
    threadgroup float part[8];
    const float4 v = float4(*(const device half4*)(x + t * C + lid * 4));
    float s = simd_sum(v.x + v.y + v.z + v.w);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0;
    for (int i = 0; i < 8; i++) mean += part[i];
    mean /= C;
    const float4 dv = v - mean;
    s = simd_sum(dot(dv, dv));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0;
    for (int i = 0; i < 8; i++) var += part[i];
    const half4 y = half4(dv * metal::precise::rsqrt(var / C + 1e-6f));
    const half4 m = half4(1.0h) + *(const device half4*)(scale + b * C + lid * 4);
    const half4 p = y * m;
    *(device half4*)(out + t * C + lid * 4) = p + *(const device half4*)(shift + b * C + lid * 4);
""")


def norm_mod(x, scale, shift):
    """norm32(LayerNorm(eps 1e-6, no affine), x) * (1 + scale) + shift; x [B, L, 1024] FP16, scale, shift [B, 1, 1024]"""
    B, L, C = x.shape
    assert C == 1024 and x.dtype == scale.dtype == shift.dtype == F16
    return _norm_mod_kernel(inputs=[x, scale, shift], grid=(B * L * 256, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[x.shape], output_dtypes=[F16])[0]


def clamp_mul(x, f):
    f_t = mx.tanh(f)
    return x * f_t + x * (f - f_t)


class SinusoidalEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = [Linear(frequency_embedding_size, hidden_size), nn.SiLU(), Linear(hidden_size, hidden_size)]
        self.half = frequency_embedding_size // 2
        self.max_period = max_period

    def __call__(self, t):
        freqs = mx.exp(-math.log(self.max_period) * mx.arange(self.half, dtype=F32) / self.half)
        args = t[:, None].astype(F32) * freqs[None]
        return run(self.mlp, mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1))


class FeedForwardNet(nn.Module):
    """layers.FeedForwardNet, and layers.MLP with two layers (same layers and parameter names)."""
    def __init__(self, channels, hidden, channels_out):
        super().__init__()
        self.mlp = [Linear(channels, hidden), nn.GELU(approx='tanh'), Linear(hidden, channels_out)]

    def __call__(self, x):
        return run(self.mlp, x)


class MultiHeadRMSNorm(nn.Module):
    """Holds gamma; applied by rope_rmsnorm."""
    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = mx.ones((heads, dim))


class RePo3DRotaryEmbedding(nn.Module):
    def __init__(self, model_channels, num_heads, head_dim, repo_hidden_ratio=0.125):
        super().__init__()
        self.num_heads = num_heads
        hidden = int(model_channels * repo_hidden_ratio)
        self.norm = nn.LayerNorm(model_channels)
        self.gate_map = Linear(model_channels, hidden, bias=False)
        self.content_map = Linear(model_channels, hidden, bias=False)
        self.final_map = Linear(hidden, 3 * num_heads, bias=False)
        dim = 2 * (head_dim // 6)
        self.freqs_0 = mx.zeros((dim // 2,))
        self.freqs_1 = mx.zeros((dim // 2,))
        self.freqs_2 = mx.zeros(((head_dim - 2 * dim) // 2,))

    def prepare(self):
        """After loading: the fused kernels' weight layouts (see _repo_angles_kernel)."""
        assert self.num_heads == 16 and self.final_map.weight.shape == (48, 128) and self.norm.weight.shape == (1024,)
        freqs = (self.freqs_0, self.freqs_1, self.freqs_2)
        self.gc_t = mx.contiguous(mx.concatenate([self.gate_map.weight, self.content_map.weight]).T)
        self.final_t = mx.contiguous(self.final_map.weight.T)
        self.freqs = mx.concatenate(freqs)
        self.axis = mx.array([i for i, f in enumerate(freqs) for _ in range(f.size)], dtype=mx.int32)

    def __call__(self, x):
        """-> rotation angles [B, L, H, head_dim / 2]"""
        if 'gc_t' in self and x.dtype == F16:
            B, L, C = x.shape
            h = _repo_norm_kernel(inputs=[x, self.norm.weight, self.norm.bias], grid=(B * L * 256, 1, 1),
                                  threadgroup=(256, 1, 1), output_shapes=[(B, L, C)], output_dtypes=[F32])[0]
            return _repo_angles_kernel(inputs=[matmul32(h, self.gc_t), self.final_t, self.freqs, self.axis],
                                       grid=(B * L * 128, 1, 1), threadgroup=(128, 1, 1),
                                       output_shapes=[(B, L, self.num_heads, 32)], output_dtypes=[F32])[0]
        h = self.norm(x.astype(F32))
        gate_content = matmul32(h, mx.concatenate([self.gate_map.weight, self.content_map.weight]).T)   # one matmul
        gate, content = mx.split(gate_content, 2, axis=-1)
        out = self.final_map(nn.silu(gate) * content)
        B, L, _ = out.shape
        delta = out.reshape(B, L, self.num_heads, 3, 1)
        ang = mx.concatenate([clamp_mul(delta[..., i, :], f) * math.pi
                              for i, f in enumerate((self.freqs_0, self.freqs_1, self.freqs_2))], axis=-1)
        return ang


class RopeMultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = Linear(channels, channels)

    def __call__(self, x, rope, mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k = rope_rmsnorm(qkv, rope, self.q_norm.gamma, self.k_norm.gamma)
        return self.out(attention(q, k, qkv, self.head_dim ** -0.5, mask))


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0, modulation=True):
        super().__init__()
        self.modulation = modulation
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, affine=not modulation)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, affine=not modulation)
        self.attn = RopeMultiHeadAttention(channels, num_heads)
        self.mlp = FeedForwardNet(channels, int(channels * mlp_ratio), channels)
        if modulation:
            self.shift_table = mx.zeros((1, 6 * channels))

    def __call__(self, x, mod, rope, mask=None):
        if not self.modulation:
            x = x + self.attn(norm32(self.norm1, x), rope)
            return x + self.mlp(norm32(self.norm2, x))
        mod = (mod + self.shift_table.astype(mod.dtype))[:, None]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(mod, 6, axis=-1)
        x = x + self.attn(norm_mod(x, scale_msa, shift_msa), rope, mask) * gate_msa
        return x + self.mlp(norm_mod(x, scale_mlp, shift_mlp)) * gate_mlp


class LatentSeqMMFlowModel(nn.Module):
    def __init__(self, q_token_length, in_channels, cam_channels, model_channels, cond_channels, cond2_channels,
                 num_blocks, num_refiner_blocks=2, num_heads=16, num_head_channels=64, mlp_ratio=4):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.model_channels = model_channels

        def blocks(count, modulation=True):
            return [UnifiedTransformerBlock(model_channels, num_heads, mlp_ratio, modulation) for _ in range(count)]

        def rope_layers(count):
            return [RePo3DRotaryEmbedding(model_channels, num_heads, num_head_channels) for _ in range(count)]

        self.t_embedder = SinusoidalEmbedder(model_channels)
        self.adaLN_modulation = [nn.SiLU(), Linear(model_channels, 6 * model_channels)]
        self.input_layer = Linear(in_channels, model_channels)
        self.cond_embedder = Linear(cond_channels, model_channels)
        self.cond_embedder2 = Linear(cond2_channels, model_channels)
        self.pos_pe = mx.zeros((1, q_token_length, 3))
        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = FeedForwardNet(cam_channels, model_channels, model_channels)
        self.blocks = blocks(num_blocks)
        self.shift_table = mx.zeros((1, 2, model_channels))
        self.out_layer = Linear(model_channels, in_channels)
        self.cam_out_layer = Linear(model_channels, cam_channels)

    def to_f16(self):
        """The torch model's convert_module_to_f16: the transformer blocks' Linear layers in FP16, the rest FP32.
        Their weights are then held transposed (Linear.weight_t)."""
        for block in self.blocks + self.noise_refiner + self.context_refiner:
            for _, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    module.set_dtype(F16)
                    module.weight_t = mx.contiguous(module.pop('weight').T)

    def condition(self, cond):
        """cond: {'feature1' [B, L_c, 1280], 'feature2' [B, L_c, 128]} -> (image tokens after the context refiner,
        attention key bias [L] or None). The context refiner has no time input, so sample_flow runs this once per request.

        Identical image tokens (the all-zero unconditional input) stay identical through every block, since RoPE
        here comes from token features, not positions; and n copies of a key attend like one key with log(n) added to
        its score. So they run as two tokens whose fp16 biases b1, b2 satisfy e^b1 + e^b2 = n (one fp16 bias is off by
        up to 0.4%): the unconditional pass drops from 12,294 to 8,195 tokens, ~55% less attention, same result."""
        h = (self.cond_embedder(cond['feature1']) + self.cond_embedder2(cond['feature2'])).astype(F16)
        n = h.shape[1]
        mask = None
        if n > 2 and mx.all(h == h[:, :1]).item():
            b1 = np.float16(math.log(n))
            if math.exp(b1) > n:
                b1 = np.nextafter(b1, np.float16(0))
            rest = n - math.exp(b1)
            b2 = np.float16(math.log(rest)) if rest > 0 else np.float16(-np.inf)
            h = h[:, :2]
            mask = mx.zeros((self.q_token_length + 2 + 1,), dtype=F16)    # key bias: [latent, 2 image tokens, camera]
            mask[self.q_token_length:self.q_token_length + 2] = mx.array([float(b1), float(b2)], dtype=F16)
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h = block(h, None, repo(h))
        return h, mask

    def __call__(self, x, t, h_cond, mask=None):
        """x: {'latent' [B, 8192, 16], 'camera' [B, 1, 5]}; t [B] in [0, 1000]; h_cond, mask: from condition()
        -> the velocity, shaped like x"""
        return self.joint(self.latent_stream(x, t), h_cond, mask)

    def latent_stream(self, x, t):
        """The part of the forward pass that does not see the condition, so both guidance passes share it: the latent
        tokens through the noise refiner, the camera token, the time embeddings."""
        h_x = self.input_layer(x['latent']).astype(F16)
        t_emb = self.t_embedder(t)
        t_mod = run(self.adaLN_modulation, t_emb).astype(F16)
        h_x = h_x + self._pos_emb.astype(F16)
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, t_mod, repo(h_x))
        return h_x, self.cam_refiner(x['camera']).astype(F16), t_emb, t_mod

    def joint(self, stream, h_cond, mask=None):
        """latent_stream() output and condition() output -> the velocity"""
        h_x, h_cam, t_emb, t_mod = stream
        h = mx.concatenate([h_x, h_cond, h_cam], axis=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = block(h, t_mod, repo(h), mask)

        h_x = mx.fast.layer_norm(h[:, :h_x.shape[1]].astype(F32), None, None, 1e-5)
        h_cam = mx.fast.layer_norm(h[:, -h_cam.shape[1]:].astype(F32), None, None, 1e-5)
        shift, scale = mx.split(self.shift_table + t_emb[:, None], 2, axis=1)
        return {'latent': self.out_layer(h_x * (1 + scale) + shift),
                'camera': self.cam_out_layer(h_cam * (1 + scale) + shift)}


def load_denoiser(path, args):
    model = LatentSeqMMFlowModel(**args)
    model.load_weights(path)
    # the fixed position embedding, computed once by the torch module: its frequencies reach 2^16, where sin() of
    # fp32 arguments ~4e5 amplifies 1-ulp differences in any re-implementation to ~0.03
    pos_pe = torch.from_numpy(np.array(model.pos_pe))
    model._pos_emb = mx.array(PcdAbsolutePositionEmbedder(model.model_channels)(pos_pe).numpy())
    model.to_f16()
    for repo in model.noise_repo_layers + model.context_repo_layers + model.repo_layers:
        repo.prepare()
    mx.eval(model.parameters())
    return model


def sample_flow(model, noise, cond, neg_cond, steps=50, rescale_t=3.0, cfg_strength=3.0, verbose=True, batch_cfg=False):
    """idol.sampler.sample_flow in MLX, taking and returning torch TensorDicts. batch_cfg is ignored: cond and
    neg_cond always run as two passes (a batch-2 forward is slower on Apple GPUs)."""
    def to_mx(td):
        return {key: mx.array(value.float().cpu().numpy()) for key, value in td.items()}

    sample = to_mx(noise)
    cond, neg_cond = model.condition(to_mx(cond)), model.condition(to_mx(neg_cond))
    batch = sample['latent'].shape[0]

    # compiled separately: both joint passes in one compiled graph fail in MLX 0.32 (unordered_map::at)
    latent_stream, joint = mx.compile(model.latent_stream), mx.compile(model.joint)

    def step(sample, t, dt):
        stream = latent_stream(sample, t)
        v = joint(stream, *cond)
        if cfg_strength > 0:
            v_neg = joint(stream, *neg_cond)
            v = {key: (1 + cfg_strength) * v[key] - cfg_strength * v_neg[key] for key in v}
        return {key: sample[key] - v[key] * dt for key in sample}

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    previous = None
    for i in tqdm(range(steps), desc='Sampling', disable=not verbose):
        sample = step(sample, mx.full((batch,), 1000 * t_seq[i], dtype=F32), mx.array(t_seq[i] - t_seq[i + 1], F32))
        mx.async_eval(sample)
        if previous is not None:
            mx.eval(previous)   # wait for the step before: the CPU queues step i + 1 while the GPU runs step i
        previous = sample
    mx.eval(sample)
    return TensorDict({key: torch.from_numpy(np.array(value)) for key, value in sample.items()},
                      batch_size=noise.batch_size).to(noise.device)
