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


def test_linear_full_fp32():
    layer = mlx_dit.Linear(1024, 512)
    x = mx.random.normal((4096, 1024))
    ref = np.array(x).astype(np.float64) @ np.array(layer.weight).astype(np.float64).T + np.array(layer.bias)
    assert np.linalg.norm(np.array(layer(x)) - ref) / np.linalg.norm(ref) < 1e-5   # plain MLX GPU matmul: ~8e-4


if __name__ == '__main__':
    test_rope_rmsnorm_kernel()
    test_linear_full_fp32()
    print('ok')
