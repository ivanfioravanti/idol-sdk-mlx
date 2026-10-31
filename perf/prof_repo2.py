"""RePo layer pieces at the cond shape: LayerNorm kernel, FP32 gate/content matmul, angles kernel.
python -m perf.prof_repo2"""
import os, time
import mlx.core as mx
from idol import mlx_dit
from perf.common import WEIGHTS, CONFIG


def bench(f, n=50):
    mx.eval(f()); t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n * 1000


model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
repo = model.repo_layers[0]
for L in (12294, 8195):
    x = (mx.random.normal((1, L, 1024)) * 3).astype(mx.float16)
    mx.eval(x)
    ln = lambda: mlx_dit._repo_norm_kernel(inputs=[x, repo.norm.weight, repo.norm.bias], grid=(L * 256, 1, 1),
                                           threadgroup=(256, 1, 1), output_shapes=[(1, L, 1024)], output_dtypes=[mx.float32])[0]
    h = ln(); mx.eval(h)
    mm = lambda: mlx_dit.matmul32(h, repo.gc_t)
    gc = mm(); mx.eval(gc)
    ang = lambda: mlx_dit._repo_angles_kernel(inputs=[gc, repo.final_t, repo.freqs, repo.axis], grid=(L * 128, 1, 1),
                                              threadgroup=(128, 1, 1), output_shapes=[(1, L, 16, 32)], output_dtypes=[mx.float32])[0]
    full = mx.compile(lambda x: repo(x))
    print(f'L={L} repo {bench(lambda: full(x)):.3f} ms: LN {bench(ln):.3f} | fp32 matmul 1024->256 {bench(mm):.3f} '
          f'| angles {bench(ang):.3f}', flush=True)
