"""MLX port of the denoiser (models/dit.py) and its sampler (sampler.py): the 50-step loop that dominates a request.
Same module tree and parameter names as the torch model, so it loads denoiser.safetensors as is. The image encoders
and the decoders stay in torch; sample_flow takes and returns torch TensorDicts."""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from tensordict import TensorDict
from tqdm import tqdm

from .models.layers import PcdAbsolutePositionEmbedder

F16, F32 = mx.float16, mx.float32


def tf32_split(a):
    """a = hi + lo with hi exact in TF32 (the low 13 mantissa bits cleared)."""
    hi = mx.view(mx.view(a, mx.uint32) & 0xFFFFE000, F32)
    return hi, a - hi


def matmul32(x, w):
    """x @ w.T at full FP32 precision: MLX's GPU matmul rounds FP32 operands to TF32 (~1e-3 relative error, amplified
    by the RoPE angles), so this runs three TF32-exact products ("3xTF32")."""
    x_hi, x_lo = tf32_split(x)
    w_hi, w_lo = tf32_split(w.T)
    return x_hi @ w_hi + (x_hi @ w_lo + x_lo @ w_hi)


class Linear(nn.Linear):
    """nn.Linear, at full precision for FP32 inputs (matmul32). FP16 is exact in TF32 as is."""
    def __call__(self, x):
        if x.dtype != F32:
            return super().__call__(x)
        y = matmul32(x, self.weight)
        return y + self.bias if 'bias' in self else y


def run(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def norm32(norm, x):
    return norm(x.astype(F32)).astype(x.dtype)


# layers.apply_rotary_emb then MultiHeadRMSNorm on q and k, fused: one SIMD group per (token, head, q or k), one
# thread per rotation pair. Rotates in FP32 and rounds to FP16 (torch's type_as), normalises in FP32, and writes the
# [B, H, L, D] layout SDPA takes. As separate MLX ops this was ~30% of a block for memory-bound work.
_rope_rmsnorm_kernel = mx.fast.metal_kernel(
    name='idol_rope_rmsnorm',
    input_names=['qkv', 'ang', 'gamma_q', 'gamma_k'],
    output_names=['q', 'k'],
    source="""
    uint i = thread_position_in_grid.x;               // rotation pair: elements 2i, 2i + 1 of the 64-wide head
    uint H = gamma_q_shape[0];
    uint which = thread_position_in_grid.y / H;       // 0: q, 1: k
    uint h = thread_position_in_grid.y % H;
    uint t = thread_position_in_grid.z;               // b * L + token
    uint L = qkv_shape[1];
    uint src = ((t * 3 + which) * H + h) * 64 + 2 * i;
    float x0 = qkv[src], x1 = qkv[src + 1];
    float a = ang[(t * H + h) * 32 + i];
    float c = metal::precise::cos(a), s = metal::precise::sin(a);
    float y0 = float(half(x0 * c - x1 * s));
    float y1 = float(half(x0 * s + x1 * c));
    float norm = metal::max(metal::sqrt(simd_sum(y0 * y0 + y1 * y1)), 1e-12f);
    uint g = h * 64 + 2 * i;
    uint dst = (((t / L) * H + h) * L + t % L) * 64 + 2 * i;
    if (which == 0) {
        q[dst] = half(y0 / norm * gamma_q[g] * 8.0f);
        q[dst + 1] = half(y1 / norm * gamma_q[g + 1] * 8.0f);
    } else {
        k[dst] = half(y0 / norm * gamma_k[g] * 8.0f);
        k[dst + 1] = half(y1 / norm * gamma_k[g + 1] * 8.0f);
    }
""")


def rope_rmsnorm(qkv, ang, gamma_q, gamma_k):
    """qkv [B, L, 3, H, 64] FP16, ang [B, L, H, 32] FP32 -> rotated, RMS-normalised q, k, each [B, H, L, 64] FP16"""
    B, L, _, H, D = qkv.shape
    assert D == 64 and qkv.dtype == F16
    return _rope_rmsnorm_kernel(inputs=[qkv, ang, gamma_q, gamma_k], grid=(32, 2 * H, B * L), threadgroup=(32, 8, 1),
                                output_shapes=[(B, H, L, D)] * 2, output_dtypes=[F16] * 2)


def clamp_mul(x, f):
    f_t = mx.tanh(f)
    return x * f_t + x * (f - f_t)


class SinusoidalEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = [Linear(frequency_embedding_size, hidden_size), nn.SiLU(), Linear(hidden_size, hidden_size)]
        self.half = frequency_embedding_size // 2
        self.max_period = max_period

    def __call__(self, t):
        freqs = mx.exp(-math.log(self.max_period) * mx.arange(self.half, dtype=F32) / self.half)
        args = t[:, None].astype(F32) * freqs[None]
        return run(self.mlp, mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1))


class FeedForwardNet(nn.Module):
    """layers.FeedForwardNet, and layers.MLP with two layers (same layers and parameter names)."""
    def __init__(self, channels, hidden, channels_out):
        super().__init__()
        self.mlp = [Linear(channels, hidden), nn.GELU(approx='tanh'), Linear(hidden, channels_out)]

    def __call__(self, x):
        return run(self.mlp, x)


class MultiHeadRMSNorm(nn.Module):
    """Holds gamma; applied by rope_rmsnorm."""
    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = mx.ones((heads, dim))


class RePo3DRotaryEmbedding(nn.Module):
    def __init__(self, model_channels, num_heads, head_dim, repo_hidden_ratio=0.125):
        super().__init__()
        self.num_heads = num_heads
        hidden = int(model_channels * repo_hidden_ratio)
        self.norm = nn.LayerNorm(model_channels)
        self.gate_map = Linear(model_channels, hidden, bias=False)
        self.content_map = Linear(model_channels, hidden, bias=False)
        self.final_map = Linear(hidden, 3 * num_heads, bias=False)
        dim = 2 * (head_dim // 6)
        self.freqs_0 = mx.zeros((dim // 2,))
        self.freqs_1 = mx.zeros((dim // 2,))
        self.freqs_2 = mx.zeros(((head_dim - 2 * dim) // 2,))

    def __call__(self, x):
        """-> rotation angles [B, L, H, head_dim / 2]"""
        h = self.norm(x.astype(F32))
        gate_content = matmul32(h, mx.concatenate([self.gate_map.weight, self.content_map.weight]))   # one matmul
        gate, content = mx.split(gate_content, 2, axis=-1)
        out = self.final_map(nn.silu(gate) * content)
        B, L, _ = out.shape
        delta = out.reshape(B, L, self.num_heads, 3, 1)
        ang = mx.concatenate([clamp_mul(delta[..., i, :], f) * math.pi
                              for i, f in enumerate((self.freqs_0, self.freqs_1, self.freqs_2))], axis=-1)
        return ang


class RopeMultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = Linear(channels, channels)

    def __call__(self, x, rope, mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k = rope_rmsnorm(qkv, rope, self.q_norm.gamma, self.k_norm.gamma)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask)
        return self.out(out.transpose(0, 2, 1, 3).reshape(B, L, C))


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0, modulation=True):
        super().__init__()
        self.modulation = modulation
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, affine=not modulation)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, affine=not modulation)
        self.attn = RopeMultiHeadAttention(channels, num_heads)
        self.mlp = FeedForwardNet(channels, int(channels * mlp_ratio), channels)
        if modulation:
            self.shift_table = mx.zeros((1, 6 * channels))

    def __call__(self, x, mod, rope, mask=None):
        if not self.modulation:
            x = x + self.attn(norm32(self.norm1, x), rope)
            return x + self.mlp(norm32(self.norm2, x))
        mod = (mod + self.shift_table.astype(mod.dtype))[:, None]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(mod, 6, axis=-1)
        x = x + self.attn(norm32(self.norm1, x) * (1 + scale_msa) + shift_msa, rope, mask) * gate_msa
        return x + self.mlp(norm32(self.norm2, x) * (1 + scale_mlp) + shift_mlp) * gate_mlp


class LatentSeqMMFlowModel(nn.Module):
    def __init__(self, q_token_length, in_channels, cam_channels, model_channels, cond_channels, cond2_channels,
                 num_blocks, num_refiner_blocks=2, num_heads=16, num_head_channels=64, mlp_ratio=4):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.model_channels = model_channels

        def blocks(count, modulation=True):
            return [UnifiedTransformerBlock(model_channels, num_heads, mlp_ratio, modulation) for _ in range(count)]

        def rope_layers(count):
            return [RePo3DRotaryEmbedding(model_channels, num_heads, num_head_channels) for _ in range(count)]

        self.t_embedder = SinusoidalEmbedder(model_channels)
        self.adaLN_modulation = [nn.SiLU(), Linear(model_channels, 6 * model_channels)]
        self.input_layer = Linear(in_channels, model_channels)
        self.cond_embedder = Linear(cond_channels, model_channels)
        self.cond_embedder2 = Linear(cond2_channels, model_channels)
        self.pos_pe = mx.zeros((1, q_token_length, 3))
        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = FeedForwardNet(cam_channels, model_channels, model_channels)
        self.blocks = blocks(num_blocks)
        self.shift_table = mx.zeros((1, 2, model_channels))
        self.out_layer = Linear(model_channels, in_channels)
        self.cam_out_layer = Linear(model_channels, cam_channels)

    def to_f16(self):
        """The torch model's convert_module_to_f16: the transformer blocks' Linear layers in FP16, the rest FP32."""
        for block in self.blocks + self.noise_refiner + self.context_refiner:
            for _, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    module.set_dtype(F16)

    def condition(self, cond):
        """cond: {'feature1' [B, L_c, 1280], 'feature2' [B, L_c, 128]} -> (image tokens after the context refiner,
        attention mask or None). The context refiner has no time input, so sample_flow runs this once per request.

        Identical image tokens (the all-zero unconditional input) stay identical through every block, since RoPE
        here comes from token features, not positions; and n copies of a key attend like one key with log(n) added to
        its score. So they run as two tokens whose fp16 biases b1, b2 satisfy e^b1 + e^b2 = n (one fp16 bias is off by
        up to 0.4%): the unconditional pass drops from 12,294 to 8,195 tokens, ~55% less attention, same result."""
        h = (self.cond_embedder(cond['feature1']) + self.cond_embedder2(cond['feature2'])).astype(F16)
        n = h.shape[1]
        mask = None
        if n > 2 and mx.all(h == h[:, :1]).item():
            b1 = np.float16(math.log(n))
            if math.exp(b1) > n:
                b1 = np.nextafter(b1, np.float16(0))
            rest = n - math.exp(b1)
            b2 = np.float16(math.log(rest)) if rest > 0 else np.float16(-np.inf)
            h = h[:, :2]
            mask = mx.zeros((1, 1, 1, self.q_token_length + 2 + 1), dtype=F16)    # [latent, 2 image tokens, camera]
            mask[..., self.q_token_length:self.q_token_length + 2] = mx.array([float(b1), float(b2)], dtype=F16)
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h = block(h, None, repo(h))
        return h, mask

    def __call__(self, x, t, h_cond, mask=None):
        """x: {'latent' [B, 8192, 16], 'camera' [B, 1, 5]}; t [B] in [0, 1000]; h_cond, mask: from condition()
        -> the velocity, shaped like x"""
        return self.joint(self.latent_stream(x, t), h_cond, mask)

    def latent_stream(self, x, t):
        """The part of the forward pass that does not see the condition, so both guidance passes share it: the latent
        tokens through the noise refiner, the camera token, the time embeddings."""
        h_x = self.input_layer(x['latent']).astype(F16)
        t_emb = self.t_embedder(t)
        t_mod = run(self.adaLN_modulation, t_emb).astype(F16)
        h_x = h_x + self._pos_emb.astype(F16)
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, t_mod, repo(h_x))
        return h_x, self.cam_refiner(x['camera']).astype(F16), t_emb, t_mod

    def joint(self, stream, h_cond, mask=None):
        """latent_stream() output and condition() output -> the velocity"""
        h_x, h_cam, t_emb, t_mod = stream
        h = mx.concatenate([h_x, h_cond, h_cam], axis=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = block(h, t_mod, repo(h), mask)

        h_x = mx.fast.layer_norm(h[:, :h_x.shape[1]].astype(F32), None, None, 1e-5)
        h_cam = mx.fast.layer_norm(h[:, -h_cam.shape[1]:].astype(F32), None, None, 1e-5)
        shift, scale = mx.split(self.shift_table + t_emb[:, None], 2, axis=1)
        return {'latent': self.out_layer(h_x * (1 + scale) + shift),
                'camera': self.cam_out_layer(h_cam * (1 + scale) + shift)}


def load_denoiser(path, args):
    model = LatentSeqMMFlowModel(**args)
    model.load_weights(path)
    # the fixed position embedding, computed once by the torch module: its frequencies reach 2^16, where sin() of
    # fp32 arguments ~4e5 amplifies 1-ulp differences in any re-implementation to ~0.03
    pos_pe = torch.from_numpy(np.array(model.pos_pe))
    model._pos_emb = mx.array(PcdAbsolutePositionEmbedder(model.model_channels)(pos_pe).numpy())
    model.to_f16()
    mx.eval(model.parameters())
    return model


def sample_flow(model, noise, cond, neg_cond, steps=50, rescale_t=3.0, cfg_strength=3.0, verbose=True, batch_cfg=False):
    """idol.sampler.sample_flow in MLX, taking and returning torch TensorDicts. batch_cfg is ignored: cond and
    neg_cond always run as two passes (a batch-2 forward is slower on Apple GPUs)."""
    def to_mx(td):
        return {key: mx.array(value.float().cpu().numpy()) for key, value in td.items()}

    sample = to_mx(noise)
    cond, neg_cond = model.condition(to_mx(cond)), model.condition(to_mx(neg_cond))
    batch = sample['latent'].shape[0]

    # compiled separately: both joint passes in one compiled graph fail in MLX 0.32 (unordered_map::at)
    latent_stream, joint = mx.compile(model.latent_stream), mx.compile(model.joint)

    def step(sample, t, dt):
        stream = latent_stream(sample, t)
        v = joint(stream, *cond)
        if cfg_strength > 0:
            v_neg = joint(stream, *neg_cond)
            v = {key: (1 + cfg_strength) * v[key] - cfg_strength * v_neg[key] for key in v}
        return {key: sample[key] - v[key] * dt for key in sample}

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    previous = None
    for i in tqdm(range(steps), desc='Sampling', disable=not verbose):
        sample = step(sample, mx.full((batch,), 1000 * t_seq[i], dtype=F32), mx.array(t_seq[i] - t_seq[i + 1], F32))
        mx.async_eval(sample)
        if previous is not None:
            mx.eval(previous)   # wait for the step before: the CPU queues step i + 1 while the GPU runs step i
        previous = sample
    mx.eval(sample)
    return TensorDict({key: torch.from_numpy(np.array(value)) for key, value in sample.items()},
                      batch_size=noise.batch_size).to(noise.device)
