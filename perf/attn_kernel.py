"""Experimental flash attention for head_dim 64 (mx.fast.metal_kernel) vs mx.fast.scaled_dot_product_attention.
python -m perf.attn_kernel"""
import time
import mlx.core as mx
import numpy as np

SRC = """
    constexpr int D = 64, TD = 8, NSG = NSG_, BQ = NSG * TQ * 8, TK = BK / 8, NBUF = NBUF_;
    constexpr int LDK = D + 8, LDV = D + 8;
    threadgroup half Ks[NBUF][BK * LDK];     // K block: [key][d], read transposed by simdgroup_load
    threadgroup half Vs[NBUF][BK * LDV];     // V block: [key][d]

    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tid = sg * 32 + lane;
    const int qb = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
    const int H = q_shape[1], L = q_shape[2];
    const int HD = H * D;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;

    const device half* Q = q + ((long)(b * H + h) * L) * D;
    const device half* K = k + ((long)(b * H + h) * L) * D;
    const device half* V = qkv + (long)b * L * 3 * HD + 2 * HD + h * D;    // row stride 3 * HD

    QT_ Qf[TQ][TD];
    simdgroup_float8x8 Of[TQ][TD], Sf[TQ][TK];
    const float scale = scale_[0] * M_LOG2E_F;
    float m[TQ], s[TQ];
    const int row0 = qb * BQ + sg * TQ * 8;
    #pragma clang loop unroll(full)
    for (int i = 0; i < TQ; i++) {
        int r = metal::min(row0 + i * 8 + fm, L - 1);
        #pragma clang loop unroll(full)
        for (int d = 0; d < TD; d++) {
            half2 x = *(const device half2*)(Q + (long)r * D + d * 8 + fn);
            thread auto& e = Qf[i][d].thread_elements();
            QLOAD_CODE
            Of[i][d] = simdgroup_float8x8(0);
        }
        m[i] = -FLT_MAX; s[i] = 0;
    }

    constexpr int PER = BK * D / (NSG * 32);   // halfs of K and of V each thread moves per block
    constexpr int TPR = D / PER;               // threads per key row
    const int lk = tid / TPR, ld0 = (tid % TPR) * PER;
    auto load = [&](int kb, int buf) {
        const int key = kb * BK + lk;
        if (key < L) {
            _Pragma("clang loop unroll(full)") for (int j = 0; j < PER; j += 4) {
                *(threadgroup half4*)(&Ks[buf][lk * LDK + ld0 + j]) = *(const device half4*)(K + (long)key * D + ld0 + j);
                *(threadgroup half4*)(&Vs[buf][lk * LDV + ld0 + j]) = *(const device half4*)(V + (long)key * 3 * HD + ld0 + j);
            }
        } else {
            _Pragma("clang loop unroll(full)") for (int j = 0; j < PER; j += 4) {
                *(threadgroup half4*)(&Ks[buf][lk * LDK + ld0 + j]) = half4(0);
                *(threadgroup half4*)(&Vs[buf][lk * LDV + ld0 + j]) = half4(0);
            }
        }
    };

    const int NK = (L + BK - 1) / BK;
    if (NBUF == 2) load(0, 0);
    for (int kb = 0; kb < NK; kb++) {
        const int buf = NBUF == 2 ? kb & 1 : 0;
        if (NBUF == 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            load(kb, 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (NBUF == 2 && kb + 1 < NK) load(kb + 1, buf ^ 1);
        const threadgroup half* Kb = Ks[buf];
        const threadgroup half* Vb = Vs[buf];

        // S = Q K^T
        #pragma clang loop unroll(full)
        for (int i = 0; i < TQ; i++)
            _Pragma("clang loop unroll(full)") for (int c = 0; c < TK; c++) Sf[i][c] = simdgroup_float8x8(0);
        #pragma clang loop unroll(full)
        for (int d = 0; d < TD; d++) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, Kb + c * 8 * LDK + d * 8, LDK, ulong2(0, 0), true);
                _Pragma("clang loop unroll(full)") for (int i = 0; i < TQ; i++) simdgroup_multiply_accumulate(Sf[i][c], Qf[i][d], kf, Sf[i][c]);
            }
        }
        // online softmax (base 2). LAZY: keep a stale row max until a score exceeds it by 2^8 (P <= 256 in float):
        // same math, the rescale of O is skipped on most blocks. Row sums stay per thread until the end.
        const bool tail = (kb + 1) * BK > L;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TQ; i++) {
            float mx_ = -FLT_MAX;
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                thread auto& e = Sf[i][c].thread_elements();
                SCALE_CODE
                BIAS_CODE
                if (tail) {
                    int col = kb * BK + c * 8 + fn;
                    if (col >= L) e[0] = -FLT_MAX;
                    if (col + 1 >= L) e[1] = -FLT_MAX;
                }
                mx_ = metal::max(mx_, metal::max(e[0], e[1]));
            }
            mx_ = metal::max(mx_, simd_shuffle_xor(mx_, 1));
            mx_ = metal::max(mx_, simd_shuffle_xor(mx_, 8));
            if (simd_any(mx_ > m[i] + TAU)) {
                float mn = metal::max(m[i], mx_);
                float f = fast::exp2(m[i] - mn);
                m[i] = mn;
                s[i] *= f;
                #pragma clang loop unroll(full)
                for (int d = 0; d < TD; d++) {
                    thread auto& e = Of[i][d].thread_elements();
                    e[0] *= f; e[1] *= f;
                }
            }
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                thread auto& e = Sf[i][c].thread_elements();
                e[0] = fast::exp2(e[0] - m[i]); e[1] = fast::exp2(e[1] - m[i]);
                s[i] += e[0] + e[1];
            }
        }
        // O += P V
        #pragma clang loop unroll(full)
        for (int c = 0; c < TK; c++) {
            #pragma clang loop unroll(full)
            for (int d = 0; d < TD; d++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, Vb + c * 8 * LDV + d * 8, LDV);
                _Pragma("clang loop unroll(full)") for (int i = 0; i < TQ; i++) simdgroup_multiply_accumulate(Of[i][d], Sf[i][c], vf, Of[i][d]);
            }
        }
    }
    // out [B, L, H * D]
    #pragma clang loop unroll(full)
    for (int i = 0; i < TQ; i++) {
        s[i] += simd_shuffle_xor(s[i], 1);
        s[i] += simd_shuffle_xor(s[i], 8);
        int r = row0 + i * 8 + fm;
        if (r >= L) continue;
        #pragma clang loop unroll(full)
        for (int d = 0; d < TD; d++) {
            thread auto& e = Of[i][d].thread_elements();
            *(device half2*)(out + ((long)b * L + r) * HD + h * D + d * 8 + fn) = half2(half(e[0] / s[i]), half(e[1] / s[i]));
        }
    }
"""

BIAS_CODE = """{
                    int col = kb * BK + c * 8 + fn;
                    e[0] += M_LOG2E_F * float(kbias[metal::min(col, L - 1)]);
                    e[1] += M_LOG2E_F * float(kbias[metal::min(col + 1, L - 1)]);
                }"""
_kernels = {}
QT = 'simdgroup_half8x8'
FOLD = False
TAU = 0.0


def attention(q, k, qkv, scale, bias=None, tq=2, bk=32, nsg=4, nbuf=1):
    """q, k [B, H, L, 64] FP16, qkv [B, L, 3, H, 64] FP16 (v = qkv[:, :, 2]), bias [L] or None -> [B, L, H * 64]"""
    B, H, L, D = q.shape
    key = (bias is not None, tq, bk, nsg, nbuf, QT, FOLD, TAU)
    if key not in _kernels:
        names = ['q', 'k', 'qkv', 'scale_'] + (['kbias'] if bias is not None else [])
        _kernels[key] = mx.fast.metal_kernel(
            name=f'idol_attn_{QT[10:14]}_f{int(FOLD)}_t{int(TAU)}_tq{tq}_bk{bk}_nsg{nsg}_nb{nbuf}_b{int(bias is not None)}', input_names=names, output_names=['out'],
            source=SRC.replace('QT_', 'simdgroup_float8x8' if FOLD else QT).replace('TAU', f'{TAU:.1f}f')
            .replace('SCALE_CODE', '' if FOLD else 'e[0] *= scale; e[1] *= scale;')
            .replace('QLOAD_CODE', 'e[0] = float(x[0]) * scale; e[1] = float(x[1]) * scale;' if FOLD else 'e[0] = x[0]; e[1] = x[1];').replace('NSG_', str(nsg)).replace('NBUF_', str(nbuf)).replace('BIAS_CODE', BIAS_CODE if bias is not None else ''))
    bq = nsg * tq * 8
    inputs = [q, k, qkv, mx.array([scale], mx.float32)] + ([bias] if bias is not None else [])
    return _kernels[key](inputs=inputs, template=[('TQ', tq), ('BK', bk)],
                         grid=(((L + bq - 1) // bq) * nsg * 32, H, B), threadgroup=(nsg * 32, 1, 1),
                         output_shapes=[(B, L, H * D)], output_dtypes=[mx.float16])[0]


def bench(f, *a, n=10):
    mx.eval(f(*a)); t = time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter() - t) / n * 1000


if __name__ == '__main__':
    import sys
    H, D = 16, 64
    configs = [(2, 32, 4, 1), (2, 64, 4, 1)]
    for L, with_bias in ((12294, False), (8195, True)):
        qkv = (mx.random.normal((1, L, 3, H, D)) * 0.5).astype(mx.float16)
        q = mx.random.normal((1, H, L, D)).astype(mx.float16) * 2
        k = mx.random.normal((1, H, L, D)).astype(mx.float16) * 2
        bias = None
        mask = None
        if with_bias:
            bias = mx.zeros((L,), mx.float16)
            bias[8192:8194] = mx.array([8.3, 2.0], mx.float16)
            mask = bias.reshape(1, 1, 1, L)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        ref_f = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask).transpose(0, 2, 1, 3).reshape(1, L, H * D)
        ref = ref_f(q, k, v)
        # exact-ish reference in fp32
        s = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * 0.125
        if mask is not None: s = s + mask.astype(mx.float32)
        r32 = (mx.softmax(s, axis=-1) @ v.astype(mx.float32)).transpose(0, 2, 1, 3).reshape(1, L, H * D)
        mx.eval(ref, r32)
        rel = lambda a: float(mx.linalg.norm((a.astype(mx.float32) - r32).flatten()) / mx.linalg.norm(r32.flatten()))
        fl = 4 * L * L * D * H
        t = bench(ref_f, q, k, v)
        print(f'L={L} mlx sdpa {t:.2f} ms {fl / t / 1e9:.1f} TFLOPS rel {rel(ref):.3e}', flush=True)
        for FOLD, TAU, tq, bk, nsg, nbuf in [(fo, ta) + c for c in configs for fo, ta in ((False, 0.0), (True, 0.0), (False, 8.0), (True, 8.0))]:
            f = lambda q, k, qkv: attention(q, k, qkv, 0.125, bias, tq, bk, nsg, nbuf)
            o = f(q, k, qkv); mx.eval(o)
            t = bench(f, q, k, qkv)
            print(f'  fold{int(FOLD)} tau{TAU} tq{tq} bk{bk} nsg{nsg} nbuf{nbuf}: {t:.2f} ms {fl / t / 1e9:.1f} TFLOPS rel {rel(o):.3e} '
                  f'vs mlx {float(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32)).max()):.2e}', flush=True)
