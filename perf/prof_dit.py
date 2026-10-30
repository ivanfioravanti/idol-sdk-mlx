"""Time pieces of one joint pass (cond 12,294 tokens) and one uncond pass (8,195)."""
import os, time, mlx.core as mx, numpy as np
from idol import mlx_dit
from perf.common import WEIGHTS, CONFIG
model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
d = np.load('perf/inputs_avocado.npz')
x = {'latent': mx.array(d['latent']), 'camera': mx.array(d['camera'])}
cond = model.condition({'feature1': mx.array(d['feature1']), 'feature2': mx.array(d['feature2'])})
neg = model.condition({'feature1': mx.zeros(d['feature1'].shape), 'feature2': mx.zeros(d['feature2'].shape)})
def bench(f, *a, n=10):
    mx.eval(f(*a)); t = time.perf_counter()
    for _ in range(n): mx.eval(f(*a))
    return (time.perf_counter() - t) / n * 1000
t = mx.array([500.0])
ls = mx.compile(model.latent_stream); jn = mx.compile(model.joint)
s = ls(x, t); mx.eval(s)
print(f'latent_stream {bench(ls, x, t):.1f} ms')
print(f'joint cond {bench(jn, s, *cond):.1f} ms  uncond {bench(jn, s, *neg):.1f} ms')
blk, repo = model.blocks[0], model.repo_layers[0]
for name, hc, mask in (('cond', cond[0], cond[1]), ('uncond', neg[0], neg[1])):
    h = mx.concatenate([s[0], hc, s[1]], axis=1); mx.eval(h); t_mod = s[3]
    rope = mx.compile(repo); r = rope(h); mx.eval(r)
    fb = mx.compile(lambda h, r: blk(h, t_mod, r, mask))
    attn = mx.compile(lambda h, r: blk.attn(h, r, mask))
    mlp = mx.compile(lambda h: blk.mlp(h))
    B, L, C = h.shape
    qkv = blk.attn.qkv(h).reshape(B, L, 3, 16, 64); mx.eval(qkv)
    kern = lambda qkv, r: mlx_dit.rope_rmsnorm(qkv, r, blk.attn.q_norm.gamma, blk.attn.k_norm.gamma)
    q, k = kern(qkv, r); v = qkv[:, :, 2].transpose(0, 2, 1, 3); mx.eval(q, k, v)
    sd = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
    print(f'[{name} L={L}] repo {bench(rope, h):.2f} | block(no repo) {bench(fb, h, r):.2f} | attn {bench(attn, h, r):.2f} '
          f'| qkv lin {bench(blk.attn.qkv, h):.2f} | rope_rms kern {bench(kern, qkv, r):.2f} | sdpa {bench(sd, q, k, v):.2f} '
          f'| out lin {bench(blk.attn.out, h):.2f} | mlp {bench(mlp, h):.2f} ms')
