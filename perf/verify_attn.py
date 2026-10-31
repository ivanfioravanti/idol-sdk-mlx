"""Custom attention kernel vs mx.fast.scaled_dot_product_attention (FP16) and an FP64 reference."""
import mlx.core as mx, numpy as np
from idol import mlx_dit
mx.random.seed(0)
for B, L, bias in [(1, 12294, False), (1, 8195, True), (2, 777, False), (2, 777, True), (1, 65, True), (1, 1, False)]:
    qkv = (mx.random.normal((B, L, 3, 16, 64)) * 1.5).astype(mx.float16)
    q, k = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in (0, 1)); v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    kb = (mx.random.normal((L,)) * 3).astype(mx.float16) if bias else None
    out = mlx_dit.attention(q, k, qkv, 0.125, kb)
    mask = kb.reshape(1, 1, 1, L) if bias else None
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask).transpose(0, 2, 1, 3).reshape(B, L, -1)
    a, r = np.array(out.astype(mx.float32), np.float64), np.array(ref.astype(mx.float32), np.float64)
    line = f'B={B} L={L:5d} bias={bias!s:5s} vs mx.fast.sdpa: rel {np.linalg.norm(a-r)/np.linalg.norm(r):.2e} max abs {np.abs(a-r).max():.2e}'
    if L <= 777:
        qn, kn, vn = (np.array(t.astype(mx.float32), np.float64) for t in (q, k, v))
        s = qn @ kn.transpose(0, 1, 3, 2) * 0.125 + (np.array(kb.astype(mx.float32), np.float64) if bias else 0)
        p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        f = (p @ vn).transpose(0, 2, 1, 3).reshape(B, L, -1)
        line += f' | FP64 ref: kernel rel {np.linalg.norm(a-f)/np.linalg.norm(f):.2e}, sdpa rel {np.linalg.norm(r-f)/np.linalg.norm(f):.2e}'
    print(line, flush=True)
