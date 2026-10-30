import time, mlx.core as mx
def bench(f, *a, n=10):
    mx.eval(f(*a)); t=time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter()-t)/n*1000
H,D=16,64
for L in (12294, 8195):
    q,k,v=[mx.random.normal((1,H,L,D)).astype(mx.float16) for _ in range(3)]
    sd = lambda q,k,v: mx.fast.scaled_dot_product_attention(q,k,v,scale=D**-0.5)
    fl = 4*L*L*D*H
    t=bench(sd,q,k,v); print(f'L={L} sdpa {t:.1f} ms {fl/t/1e9:.1f} TFLOPS', flush=True)
    a=mx.random.normal((L,1024)).astype(mx.float16); w=mx.random.normal((4096,1024)).astype(mx.float16)
    t=bench(lambda a,w: a@w.T, a, w); print(f'  linear 1024->4096 {t:.2f} ms {2*L*1024*4096/t/1e9:.1f} TFLOPS', flush=True)
