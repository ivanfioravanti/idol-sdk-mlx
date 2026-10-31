"""FP32 GEMM (the RePo gate/content projection, [M, 1024] @ [1024, 256]) vs MLX's FP32 matmul: bit-identity and time.
python -m perf.gemm32"""
import time
import mlx.core as mx

SRC = """
    constexpr int BM = WM * FI * 8, BN = WN * FJ * 8, LDA = BK + PADA, LDB = BN + PADB, NT = WM * WN * 32;
    threadgroup float As[BM * LDA];
    threadgroup float Bs[BK * LDB];
    const int M = x_shape[0], K = x_shape[1], N = w_shape[1];
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = sg * 32 + lane;
    const int sm = sg / WN, sn = sg % WN;
    const int m0 = threadgroup_position_in_grid.y * BM, n0 = threadgroup_position_in_grid.x * BN;
    const short fm = (lane / 4 & 4) + (lane / 2) % 4, fn = (lane / 4 & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 C[FI][FJ];
    #pragma clang loop unroll(full)
    for (int i = 0; i < FI; i++)
        _Pragma("clang loop unroll(full)") for (int j = 0; j < FJ; j++) C[i][j] = simdgroup_float8x8(0);
    for (int k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int p = 0; p < BM * BK / 4 / NT; p++) {
            const int e = p * NT + tid, r = e / (BK / 4), c = e % (BK / 4) * 4;
            *(threadgroup float4*)(As + r * LDA + c) = *(const device float4*)(x + (long)metal::min(m0 + r, M - 1) * K + k0 + c);
        }
        #pragma clang loop unroll(full)
        for (int p = 0; p < BK * BN / 4 / NT; p++) {
            const int e = p * NT + tid, r = e / (BN / 4), c = e % (BN / 4) * 4;
            *(threadgroup float4*)(Bs + r * LDB + c) = *(const device float4*)(w + (long)(k0 + r) * N + n0 + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < BK / 8; kk++) {
            simdgroup_float8x8 bf[FJ];
            #pragma clang loop unroll(full)
            for (int j = 0; j < FJ; j++) simdgroup_load(bf[j], Bs + kk * 8 * LDB + sn * FJ * 8 + j * 8, LDB);
            #pragma clang loop unroll(full)
            for (int i = 0; i < FI; i++) {
                simdgroup_float8x8 af;
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
            *(device float2*)(out + (long)r * N + c) = float2(e[0], e[1]);
        }
    }
"""
_k = {}


def gemm32(x, w, wm=2, wn=2, fi=4, fj=4, bk=32, pada=4, padb=4):
    M, K = x.shape
    N = w.shape[1]
    key = (wm, wn, fi, fj, bk, pada, padb)
    if key not in _k:
        hdr = f'constexpr int WM = {wm}, WN = {wn}, FI = {fi}, FJ = {fj}, BK = {bk}, PADA = {pada}, PADB = {padb};\n'
        _k[key] = mx.fast.metal_kernel(name='gemm32_' + '_'.join(map(str, key)), input_names=['x', 'w'],
                                       output_names=['out'], source=hdr + SRC)
    bm, bn = wm * fi * 8, wn * fj * 8
    return _k[key](inputs=[x, w], grid=(N // bn * wm * wn * 32, (M + bm - 1) // bm, 1), threadgroup=(wm * wn * 32, 1, 1),
                   output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]


def bench(f, n=10, k=10):
    mx.eval([f() for _ in range(k)]); t = time.perf_counter()
    for _ in range(n): mx.eval([f() for _ in range(k)])
    return (time.perf_counter() - t) / n / k * 1000


if __name__ == '__main__':
    for M in (12294, 8195, 8192):
        x = mx.random.normal((M, 1024)); w = mx.random.normal((1024, 256)) * 0.03
        mx.eval(x, w)
        ref = x @ w
        fl = 2 * M * 1024 * 256 / 1e9
        t = bench(lambda: x @ w)
        line = f'M={M}: MLX {t:.3f} ms ({fl / t:.1f} TF)'
        for cfg in ((2, 2, 4, 4, 32, 4, 4), (2, 2, 4, 4, 32, 8, 8), (2, 2, 4, 4, 16, 4, 4), (2, 2, 2, 4, 32, 4, 4),
                    (2, 2, 4, 2, 32, 4, 4), (1, 2, 4, 4, 32, 4, 4), (2, 2, 2, 2, 32, 4, 4)):
            o = gemm32(x, w, *cfg)
            t = bench(lambda: gemm32(x, w, *cfg))
            line += f' | {cfg[:5]} {t:.3f} ({fl / t:.1f}){"" if bool(mx.array_equal(o, ref)) else " DIFF"}'
        print(line, flush=True)
