set -x
.venv/bin/python - <<'PY'
import time, mlx.core as mx
def bench(f, *a, n=20):
    mx.eval(f(*a)); t=time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter()-t)/n*1000
for dt in (mx.float16, mx.bfloat16):
    x=mx.random.normal((12294,1024)).astype(dt); w=mx.random.normal((4096,1024)).astype(dt)
    q=mx.random.normal((1,16,12294,64)).astype(dt)
    print(dt, 'linear', round(bench(lambda x: x@w.T, x),2), 'sdpa', round(bench(lambda q: mx.fast.scaled_dot_product_attention(q,q,q,scale=0.125), q),2))
PY
.venv/bin/python -m perf.onepass fused
.venv/bin/python -m perf.prof_dit
.venv/bin/python -m perf.bench_dit fused 2
.venv/bin/python -m perf.compare fused
echo ALLDONE
