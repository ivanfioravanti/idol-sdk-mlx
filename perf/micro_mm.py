import time, mlx.core as mx, numpy as np
from idol.mlx_dit import tf32_split, matmul32
def bench(f, *a, n=20):
    mx.eval(f(*a)); t=time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter()-t)/n*1000
a=mx.random.normal((8192,8192)).astype(mx.float16); print(f'fp16 8192^3 {2*8192**3/bench(lambda a: a@a, a)/1e9:.1f} TFLOPS')
a=a.astype(mx.float32); print(f'fp32 8192^3 {2*8192**3/bench(lambda a: a@a, a)/1e9:.1f} TFLOPS')
L=12294
x=mx.random.normal((L,1024)); w=mx.random.normal((256,1024))*0.03
ref = np.array(x).astype(np.float64) @ np.array(w).astype(np.float64).T
whi, wlo = tf32_split(w.T)
S=2.0**11
w_hi16 = whi.astype(mx.float16); w_lo16 = (wlo*S).astype(mx.float16)
wcat = mx.concatenate([w_hi16, w_lo16], axis=0)  # [2048, 256]
def v3(x): return matmul32(x, w)
def v3pre(x):
    xh, xl = tf32_split(x); return xh@whi + (xh@wlo + xl@whi)
def vf16corr(x):
    xh, xl = tf32_split(x)
    corr = mx.concatenate([(xl*S).astype(mx.float16), xh.astype(mx.float16)], axis=-1) @ wcat
    return xh@whi + corr.astype(mx.float32)/S
def plain(x): return x@w.T
for nm,f in [('plain tf32',plain),('3xTF32',v3),('3xTF32 presplit',v3pre),('tf32+fp16corr',vf16corr)]:
    f=mx.compile(f); y=np.array(f(x)).astype(np.float64)
    print(f'{nm:18s} {bench(f,x):.3f} ms  rel {np.linalg.norm(y-ref)/np.linalg.norm(ref):.2e}')
