"""Round-2 attention kernel variants (built from the production kernel), timed at the two pass shapes.
python -m perf.attn2 [variant ...]   variants: base ph fold bias0 kt (combine with '+', e.g. ph+fold+bias0)"""
import sys, time
import mlx.core as mx
import numpy as np

SRC = """
    constexpr int D = 64, TD = 8, TQ = 2, BQ = NSG * TQ * 8, TK = BK / 8, LD = D + LDPAD, LDT = BK + LDTPAD;
#if KT
    threadgroup half Ks[D * LDT];      // K block transposed [d][key]
#else
    threadgroup half Ks[BK * LD];      // K block [key][d], read transposed by simdgroup_load
#endif
    threadgroup half Vs[BK * LD];      // V block [key][d]

    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int qb = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
    const int H = q_shape[1], L = q_shape[2], HD = H * D;
    const short fm = (lane / 4 & 4) + (lane / 2) % 4, fn = (lane / 4 & 2) * 2 + (lane % 2) * 2;

    const device half* Q = q + (long)(b * H + h) * L * D;
    const device half* K = k + (long)(b * H + h) * L * D;
    const device half* V = qkv + (long)b * L * 3 * HD + 2 * HD + h * D;
    const float scale = scale_[0] * M_LOG2E_F;
#if BIAS0 && BIAS
    const int kb_special = metal::min(L / BK, bias_start[0] / BK);   // first block with a bias or the tail
#else
    const int kb_special = L / BK;
#endif

    simdgroup_half8x8 Qf[TQ][TD];
    simdgroup_float8x8 Of[TQ][TD], Sf[TQ][TK];
#if PH
    simdgroup_half8x8 Pf[TQ][TK];
#endif
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
#if NOMAX
        m[i] = rowbound[(long)(b * H + h) * L + metal::min(row0 + i * 8 + fm, L - 1)];
#endif
    }

    constexpr int PER = BK * D / (NSG * 32);
    const int tid = sg * 32 + lane, lk = tid / (D / PER), ld0 = tid % (D / PER) * PER;
    for (int kb = 0; kb < (L + BK - 1) / BK; kb++) {
        const int key = kb * BK + lk;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int j = 0; j < PER; j += 4) {
#if KT == 2
            {   // one key per lane, 16 dims per simdgroup: the transposed stores hit consecutive halfs
                const int kk = kb * BK + lane, dd = sg * PER + j;
                const half4 kv = kk < L ? *(const device half4*)(K + (long)kk * D + dd) : half4(0);
                Ks[dd * LDT + lane] = kv[0]; Ks[(dd + 1) * LDT + lane] = kv[1];
                Ks[(dd + 2) * LDT + lane] = kv[2]; Ks[(dd + 3) * LDT + lane] = kv[3];
            }
#elif KT
            const half4 kv = key < L ? *(const device half4*)(K + (long)key * D + ld0 + j) : half4(0);
            Ks[(ld0 + j) * LDT + lk] = kv[0]; Ks[(ld0 + j + 1) * LDT + lk] = kv[1];
            Ks[(ld0 + j + 2) * LDT + lk] = kv[2]; Ks[(ld0 + j + 3) * LDT + lk] = kv[3];
#else
            *(threadgroup half4*)(Ks + lk * LD + ld0 + j) = key < L ? *(const device half4*)(K + (long)key * D + ld0 + j) : half4(0);
#endif
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
#if KT
                simdgroup_load(kf, Ks + d * 8 * LDT + c * 8, LDT);
#else
                simdgroup_load(kf, Ks + c * 8 * LD + d * 8, LD, ulong2(0, 0), true);
#endif
                _Pragma("clang loop unroll(full)") for (int i = 0; i < TQ; i++) simdgroup_multiply_accumulate(Sf[i][c], Qf[i][d], kf, Sf[i][c]);
            }
        }

        const bool special = kb >= kb_special;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TQ; i++) {
            float mx_ = m[i];
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                thread auto& e = Sf[i][c].thread_elements();
                const int col = kb * BK + c * 8 + fn;
#if !FOLD
                e[0] *= scale; e[1] *= scale;
#endif
#if BIAS && !BIAS0
                e[0] += M_LOG2E_F * float(key_bias[metal::min(col, L - 1)]);
                e[1] += M_LOG2E_F * float(key_bias[metal::min(col + 1, L - 1)]);
#endif
                if (special) {
#if BIAS && BIAS0
                    e[0] += M_LOG2E_F * float(key_bias[metal::min(col, L - 1)]);
                    e[1] += M_LOG2E_F * float(key_bias[metal::min(col + 1, L - 1)]);
#endif
                    if (col >= L) e[0] = -FLT_MAX;
                    if (col + 1 >= L) e[1] = -FLT_MAX;
                }
#if !NOMAX
                mx_ = metal::max(mx_, metal::max(e[0], e[1]));
#endif
            }
#if !NOMAX
            mx_ = metal::max(mx_, simd_shuffle_xor(mx_, 1));
            mx_ = metal::max(mx_, simd_shuffle_xor(mx_, 8));
#endif
            float sum = 0;
            #pragma clang loop unroll(full)
            for (int c = 0; c < TK; c++) {
                thread auto& e = Sf[i][c].thread_elements();
                e[0] = fast::exp2(e[0] - mx_); e[1] = fast::exp2(e[1] - mx_);
                sum += e[0] + e[1];
#if PH
                thread auto& p = Pf[i][c].thread_elements();
                p[0] = half(e[0]); p[1] = half(e[1]);
#endif
            }
#if NOMAX
            s[i] += sum;
#else
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
#endif
        }

        #pragma clang loop unroll(full)
        for (int c = 0; c < TK; c++) {
            #pragma clang loop unroll(full)
            for (int d = 0; d < TD; d++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, Vs + c * 8 * LD + d * 8, LD);
#if PH
                _Pragma("clang loop unroll(full)") for (int i = 0; i < TQ; i++) simdgroup_multiply_accumulate(Of[i][d], Pf[i][c], vf, Of[i][d]);
#else
                _Pragma("clang loop unroll(full)") for (int i = 0; i < TQ; i++) simdgroup_multiply_accumulate(Of[i][d], Sf[i][c], vf, Of[i][d]);
#endif
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < TQ; i++) {
        const int r = row0 + i * 8 + fm;
#if NOMAX
        s[i] += simd_shuffle_xor(s[i], 1);
        s[i] += simd_shuffle_xor(s[i], 8);
#endif
        if (r >= L) continue;
        #pragma clang loop unroll(full)
        for (int d = 0; d < TD; d++) {
            thread auto& e = Of[i][d].thread_elements();
            *(device half2*)(out + ((long)b * L + r) * HD + h * D + d * 8 + fn) = half2(half(e[0] / s[i]), half(e[1] / s[i]));
        }
    }
"""
_k = {}


def attention(q, k, qkv, scale, bias=None, bias_start=0, flags=()):
    B, H, L, D = q.shape
    key = (tuple(sorted(flags)), bias is not None)
    if key not in _k:
        defs = {f: int(f in flags) for f in ('PH', 'FOLD', 'BIAS0', 'KT', 'NOMAX')}
        if 'KT2' in flags:
            defs['KT'] = 2
        defs['LDPAD'], defs['LDTPAD'], defs['NSG'], defs['BK'] = 8, 8, 4, 32
        for f in flags:
            if f.startswith('NSG'): defs['NSG'] = int(f[3:])
            elif f.startswith('BK'): defs['BK'] = int(f[2:])
            elif f.startswith('LDTP'): defs['LDTPAD'] = int(f[4:])
            elif f.startswith('LDP'): defs['LDPAD'] = int(f[3:])
        defs['BIAS'] = int(bias is not None)
        header = ''.join(f'#define {n} {v}\n' for n, v in defs.items())
        names = ['q', 'k', 'qkv', 'scale_'] + (['key_bias', 'bias_start'] if bias is not None else []) + (['rowbound'] if 'NOMAX' in flags else [])
        _k[key] = mx.fast.metal_kernel(name='r2attn_' + '_'.join(f'{n}{v}' for n, v in defs.items()),
                                       input_names=names, output_names=['out'], header='', source=header + SRC)
    inputs = [q, k, qkv, mx.array([scale], mx.float32)]
    if bias is not None:
        inputs += [bias, mx.array([bias_start], mx.int32)]
    if 'NOMAX' in flags:
        qn = mx.sqrt((q.astype(mx.float32) ** 2).sum(-1))
        kn = mx.sqrt((k.astype(mx.float32) ** 2).sum(-1)).max(-1, keepdims=True)
        bmax = 0.0 if bias is None else float(bias.astype(mx.float32).max())
        inputs += [(qn * kn * scale + bmax) * 1.4426950408889634]
    nsg = next((int(f[3:]) for f in flags if f.startswith('NSG')), 4)
    return _k[key](inputs=inputs, grid=((L + nsg * 16 - 1) // (nsg * 16) * nsg * 32, H, B), threadgroup=(nsg * 32, 1, 1),
                   output_shapes=[(B, L, H * D)], output_dtypes=[mx.float16])[0]


def bench(f, n=20):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n * 1000


if __name__ == '__main__':
    from idol import mlx_dit
    variants = sys.argv[1:] or ['base', 'ph', 'fold', 'bias0', 'kt']
    H, D = 16, 64
    mx.random.seed(0)
    for L, with_bias in ((12294, False), (8195, True)):
        qkv = (mx.random.normal((1, L, 3, H, D)) * 0.5).astype(mx.float16)
        q = (mx.random.normal((1, H, L, D)) * 2).astype(mx.float16)
        k = (mx.random.normal((1, H, L, D)) * 2).astype(mx.float16)
        bias = None
        if with_bias:
            bias = mx.zeros((L,), mx.float16)
            bias[8192:8194] = mx.array([8.3, 2.0], mx.float16)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        s = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * 0.125
        if bias is not None: s = s + bias.astype(mx.float32)
        r32 = (mx.softmax(s, axis=-1) @ v.astype(mx.float32)).transpose(0, 2, 1, 3).reshape(1, L, H * D)
        mx.eval(r32); del s
        rel = lambda a: float(mx.linalg.norm((a.astype(mx.float32) - r32).flatten()) / mx.linalg.norm(r32.flatten()))
        fl = 4 * L * L * D * H
        prod = lambda: mlx_dit.attention(q, k, qkv, 0.125, bias)
        o = prod(); t = bench(prod)
        print(f'L={L} production {t:.2f} ms {fl / t / 1e9:.1f} TFLOPS rel {rel(o):.3e}', flush=True)
        qf = (q.astype(mx.float32) * (0.125 * 1.4426950408889634)).astype(mx.float16)
        for var in variants:
            flags = [] if var == 'base' else [f.upper() for f in var.split('+')]
            qq, sc = (qf, 1.0) if 'FOLD' in flags else (q, 0.125)
            f = lambda: attention(qq, k, qkv, sc, bias, 8192, flags)
            o = f(); t = bench(f)
            print(f'  {var:16s} {t:.2f} ms {fl / t / 1e9:.1f} TFLOPS rel {rel(o):.3e}', flush=True)
