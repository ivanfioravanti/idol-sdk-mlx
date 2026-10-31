"""Register-only simdgroup MMA throughput (the ceiling for the GEMMs and the attention kernel).
python -m perf.mma_peak"""
import time
import mlx.core as mx

SRC = """
    AT a[4]; BT bb[4]; simdgroup_float8x8 c[4][4];
    for (int i = 0; i < 4; i++) {
        a[i] = AT(float(x[0]) + i); bb[i] = BT(float(x[1]) - i);
        for (int j = 0; j < 4; j++) c[i][j] = simdgroup_float8x8(0);
    }
    for (int it = 0; it < ITERS; it++) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++)
            _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(c[i][j], a[i], bb[j], c[i][j]);
    }
    float acc = 0;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc += c[i][j].thread_elements()[0];
    out[thread_position_in_grid.x] = acc;
"""
ITERS = 4096
for at, bt in (('simdgroup_half8x8', 'simdgroup_half8x8'), ('simdgroup_float8x8', 'simdgroup_half8x8'),
               ('simdgroup_float8x8', 'simdgroup_float8x8')):
    k = mx.fast.metal_kernel(name=f'mma_{at[10:14]}_{bt[10:14]}', input_names=['x'], output_names=['out'],
                             source=SRC.replace('AT', at).replace('BT', bt).replace('ITERS', str(ITERS)))
    x = mx.array([0.001, 0.002], mx.float16)
    for tg in (128, 256):
        n_tg = 80 * 64
        f = lambda: k(inputs=[x], grid=(n_tg * tg, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(n_tg * tg,)],
                      output_dtypes=[mx.float32])[0]
        mx.eval(f()); t = time.perf_counter()
        for _ in range(5): mx.eval(f())
        dt = (time.perf_counter() - t) / 5
        flops = n_tg * tg / 32 * ITERS * 16 * 2 * 512
        print(f'{at[10:]} x {bt[10:]} tg{tg}: {flops / dt / 1e12:.2f} TFLOPS', flush=True)
