"""MLX denoiser pieces against their torch originals. Run: python tests/test_mlx_dit.py (Apple Silicon, needs mlx)."""
import mlx.core as mx
import numpy as np
import torch

from idol import mlx_dit
from idol.models.layers import MultiHeadRMSNorm, apply_rotary_emb


def rel(a, b):
    return float((a - b).norm() / a.norm())


def test_rope_rmsnorm_kernel():
    B, L, H, D = 2, 300, 16, 64
    torch.manual_seed(0)
    qkv = torch.randn(B, L, 3, H, D).half()
    ang = torch.randn(B, L, H, D // 2) * 20
    gq, gk = MultiHeadRMSNorm(D, H), MultiHeadRMSNorm(D, H)
    with torch.no_grad():
        gq.gamma.uniform_(0.5, 1.5), gk.gamma.uniform_(0.5, 1.5)
        rope = torch.stack([ang.cos(), ang.sin()], dim=-1)
        q_ref = gq(apply_rotary_emb(qkv[:, :, 0].float(), rope).half()).permute(0, 2, 1, 3).float()
        k_ref = gk(apply_rotary_emb(qkv[:, :, 1].float(), rope).half()).permute(0, 2, 1, 3).float()
    q, k = mlx_dit.rope_rmsnorm(mx.array(qkv.numpy()), mx.array(ang.numpy()),
                                mx.array(gq.gamma.detach().numpy()), mx.array(gk.gamma.detach().numpy()))
    q, k = torch.from_numpy(np.array(q.astype(mx.float32))), torch.from_numpy(np.array(k.astype(mx.float32)))
    assert rel(q_ref, q) < 1e-3 and rel(k_ref, k) < 1e-3, (rel(q_ref, q), rel(k_ref, k))   # FP16 output rounding


def test_attention_kernel():
    B, H, D = 2, 16, 64
    for L, with_bias in ((300, False), (77, True), (2, False)):     # lengths off the 64-query / 32-key blocks
        qkv = mx.random.normal((B, L, 3, H, D)).astype(mx.float16)
        q, k = [(mx.random.normal((B, H, L, D)) * 2).astype(mx.float16) for _ in range(2)]
        bias = mx.random.uniform(-3, 3, (L,)).astype(mx.float16) if with_bias else None
        s = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * 0.125
        if bias is not None:
            s = s + bias.astype(mx.float32)
        ref = (mx.softmax(s, axis=-1) @ qkv[:, :, 2].astype(mx.float32).transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        out = mlx_dit.attention(q, k, qkv, 0.125, bias).astype(mx.float32)
        ref, out = torch.from_numpy(np.array(ref.reshape(B, L, H * D))), torch.from_numpy(np.array(out))
        assert rel(ref, out) < 1e-3, (L, rel(ref, out))   # FP16 output rounding


def test_repo_kernels():
    repo = mlx_dit.RePo3DRotaryEmbedding(1024, 16, 64)
    for f in ('freqs_0', 'freqs_1', 'freqs_2'):
        setattr(repo, f, mx.random.uniform(0.5, 2.0, getattr(repo, f).shape))
    repo.norm.weight, repo.norm.bias = mx.random.uniform(0.5, 1.5, (1024,)), mx.random.normal((1024,)) * 0.1
    x = (mx.random.normal((2, 300, 1024)) * 3 + 1).astype(mx.float16)
    ref = repo(x)                  # MLX ops
    repo.prepare()
    out = repo(x)                  # fused kernels
    assert float(mx.linalg.norm((out - ref).flatten()) / mx.linalg.norm(ref.flatten())) < 1e-6


def test_norm_mod_kernel():
    norm = mlx_dit.nn.LayerNorm(1024, eps=1e-6, affine=False)
    x = (mx.random.normal((2, 300, 1024)) * 3 + 1).astype(mx.float16)
    scale, shift = [(mx.random.normal((2, 1, 1024)) * 0.3).astype(mx.float16) for _ in range(2)]
    ref = (mlx_dit.norm32(norm, x) * (1 + scale) + shift).astype(mx.float32)
    out = mlx_dit.norm_mod(x, scale, shift).astype(mx.float32)
    assert float(mx.linalg.norm((out - ref).flatten()) / mx.linalg.norm(ref.flatten())) < 1e-4
    print('norm_mod max abs diff', float(mx.abs(out - ref).max()), 'mismatch fraction', float((out != ref).mean()))


def test_linear_full_fp32():
    layer = mlx_dit.Linear(1024, 512)
    x = mx.random.normal((4096, 1024))
    ref = np.array(x).astype(np.float64) @ np.array(layer.weight).astype(np.float64).T + np.array(layer.bias)
    assert np.linalg.norm(np.array(layer(x)) - ref) / np.linalg.norm(ref) < 1e-5   # plain MLX GPU matmul: ~8e-4


if __name__ == '__main__':
    test_rope_rmsnorm_kernel()
    test_attention_kernel()
    test_repo_kernels()
    test_norm_mod_kernel()
    test_linear_full_fp32()
    print('ok')
