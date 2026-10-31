"""Experimental FP16 GEMM (x [M, K] @ w [K, N] + b) vs mx.addmm at the block linear shapes.
python -m perf.gemm"""
import time
import mlx.core as mx

SRC = """
    // TG tile BM x BN, WM x WN simdgroups, each SM x SN (in 8x8 fragments: FI x FJ)
    constexpr int BM = WM * FI * 8, BN = WN * FJ * 8, LDA = BK + PADA, LDB = BN + PADB;
    threadgroup half As[BM * LDA];
    threadgroup half Bs[BK * LDB];
    const int M = x_shape[0], K = x_shape[1], N = w_shape[1];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = sg * 32 + lane;
    const int sm = sg / WN, sn = sg % WN;
    const int m0 = threadgroup_position_in_grid.y * BM, n0 = threadgroup_position_in_grid.x * BN;
    const short fm = (lane / 4 & 4) + (lane / 2) % 4, fn = (lane / 4 & 2) * 2 + (lane % 2) * 2;
    constexpr int NT = WM * WN * 32;

    simdgroup_float8x8 C[FI][FJ];
    #pragma clang loop unroll(full)
    for (int i = 0; i < FI; i++)
        _Pragma("clang loop unroll(full)") for (int j = 0; j < FJ; j++) C[i][j] = simdgroup_float8x8(0);

    for (int k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A tile: BM rows x BK halfs, 8 halfs (16 bytes) per thread per pass
        #pragma clang loop unroll(full)
        for (int p = 0; p < BM * BK / 8 / NT; p++) {
            const int e = p * NT + tid, r = e / (BK / 8), c = e % (BK / 8) * 8;
            const int gr = metal::min(m0 + r, M - 1);
            *(threadgroup uint4*)(As + r * LDA + c) = *(const device uint4*)(x + (long)gr * K + k0 + c);
        }
        #pragma clang loop unroll(full)
        for (int p = 0; p < BK * BN / 8 / NT; p++) {
            const int e = p * NT + tid, r = e / (BN / 8), c = e % (BN / 8) * 8;
            *(threadgroup uint4*)(Bs + r * LDB + c) = *(const device uint4*)(w + (long)(k0 + r) * N + n0 + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < BK / 8; kk++) {
            simdgroup_half8x8 bf[FJ];
            #pragma clang loop unroll(full)
            for (int j = 0; j < FJ; j++) simdgroup_load(bf[j], Bs + kk * 8 * LDB + sn * FJ * 8 + j * 8, LDB);
            #pragma clang loop unroll(full)
            for (int i = 0; i < FI; i++) {
                simdgroup_half8x8 af;
                simdgroup_load(af, As + (sm * FI * 8 + i * 8) * LDA + kk * 8, LDA);
                _Pragma("clang loop unroll(full)") for (int j = 0; j < FJ; j++) simdgroup_multiply_accumulate(C[i][j], af, bf[j], C[i][j]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < FI; i++) {
        const int r = m0 + sm * FI * 8 + i * 8 + fm;
        if (r >= M) continue;
        #pragma clang loop unroll(full)
        for (int j = 0; j < FJ; j++) {
            const int c = n0 + sn * FJ * 8 + j * 8 + fn;
            thread auto& e = C[i][j].thread_elements();
            *(device half2*)(out + (long)r * N + c) = half2(half(e[0] + float(b[c])), half(e[1] + float(b[c + 1])));
        }
    }
"""
_k = {}


def gemm(x, w, b, wm=2, wn=2, fi=8, fj=4, bk=32, pada=8, padb=8):
    M, K = x.shape
    N = w.shape[1]
    key = (wm, wn, fi, fj, bk, pada, padb)
    if key not in _k:
        hdr = f'constexpr int WM = {wm}, WN = {wn}, FI = {fi}, FJ = {fj}, BK = {bk}, PADA = {pada}, PADB = {padb};\n'
        _k[key] = mx.fast.metal_kernel(name='gemm_' + '_'.join(map(str, key)), input_names=['x', 'w', 'b'],
                                       output_names=['out'], source=hdr + SRC)
    bm, bn = wm * fi * 8, wn * fj * 8
    assert N % bn == 0 and K % bk == 0
    return _k[key](inputs=[x, w, b], grid=(N // bn * wm * wn * 32, (M + bm - 1) // bm, 1),
                   threadgroup=(wm * wn * 32, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]


def bench(f, n=20):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n * 1000


if __name__ == '__main__':
    import sys
    configs = [(2, 2, 8, 4, 32, 8, 8), (2, 2, 4, 8, 32, 8, 8), (2, 2, 8, 4, 16, 8, 8), (1, 2, 8, 4, 32, 8, 8),
               (2, 2, 4, 4, 32, 8, 8), (4, 1, 4, 8, 32, 8, 8)]
    for M in (12294, 8195):
        for K, N, name in ((1024, 3072, 'qkv'), (1024, 1024, 'out'), (1024, 4096, 'fc1'), (4096, 1024, 'fc2')):
            x = mx.random.normal((M, K)).astype(mx.float16)
            w = (mx.random.normal((K, N)) * 0.03).astype(mx.float16)
            b = (mx.random.normal((N,)) * 0.1).astype(mx.float16)
            mx.eval(x, w, b)
            fl = 2 * M * K * N / 1e9
            ref = mx.addmm(b, x, w)
            t = bench(lambda: mx.addmm(b, x, w))
            line = f'M={M} {name}: addmm {t:.3f} ({fl / t:.1f} TF)'
            for cfg in configs:
                o = gemm(x, w, b, *cfg)
                t = bench(lambda: gemm(x, w, b, *cfg))
                diff = float((o != ref).astype(mx.float32).mean())
                line += f' | {cfg[:5]} {t:.3f} ({fl / t:.1f}) ne {diff:.1e}'
            print(line, flush=True)
