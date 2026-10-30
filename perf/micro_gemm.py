"""GEMM layout micro: the cond-pass linears as x @ w.T (nn.Linear) vs pre-transposed weights, addmm, aligned M.
python -m perf.micro_gemm"""
import time
import mlx.core as mx


def bench(f, *a, n=20):
    mx.eval(f(*a)); t = time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter() - t) / n * 1000


for M in (12294, 12288, 8195):
    for K, N, name in ((1024, 3072, 'qkv'), (1024, 1024, 'out'), (1024, 4096, 'fc1'), (4096, 1024, 'fc2')):
        x = mx.random.normal((1, M, K)).astype(mx.float16)
        w = (mx.random.normal((N, K)) * 0.03).astype(mx.float16)
        b = mx.zeros((N,), mx.float16)
        wt = mx.contiguous(w.T)
        mx.eval(x, w, b, wt)
        fl = 2 * M * K * N / 1e9
        r = [bench(lambda x, w, b: mx.addmm(b, x, w.T), x, w, b),
             bench(lambda x, w, b: mx.addmm(b, x, w), x, wt, b),
             bench(lambda x, w: x @ w.T, x, w),
             bench(lambda x, w: x @ w, x, wt)]
        print(f'M={M} {name} {K}->{N}: addmm(w.T) {r[0]:.3f} ms {fl / r[0]:.1f} TF | addmm(wt) {r[1]:.3f} {fl / r[1]:.1f} | '
              f'x@w.T {r[2]:.3f} {fl / r[2]:.1f} | x@wt {r[3]:.3f} {fl / r[3]:.1f}', flush=True)
