"""PyTorch FP32 reference of the 50-step sampling (MPS, everything in FP32)."""
import os, sys, numpy as np, torch
from safetensors.torch import load_file
from tensordict import TensorDict
from idol.models import LatentSeqMMFlowModel
from idol.models.layers import RePo3DRotaryEmbedding
from idol.sampler import sample_flow
from perf.common import WEIGHTS, CONFIG
dev = sys.argv[2] if len(sys.argv) > 2 else 'mps'
m = LatentSeqMMFlowModel(**CONFIG['denoiser']['args'])
m.load_state_dict(load_file(os.path.join(WEIGHTS, 'denoiser.safetensors')))
m = m.float().eval().to(dev); m.dtype = torch.float32
for mod in m.modules():
    if isinstance(mod, RePo3DRotaryEmbedding): mod.real_valued = True
name = sys.argv[1]
d = np.load(f'perf/inputs_{name}.npz')
T = lambda k: torch.from_numpy(d[k]).to(dev)
noise = TensorDict({'latent': T('latent'), 'camera': T('camera')}, batch_size=1, device=dev)
cond = TensorDict({'feature1': T('feature1'), 'feature2': T('feature2')}, batch_size=1, device=dev)
with torch.no_grad():
    out = sample_flow(m, noise, cond, cond.apply(torch.zeros_like), steps=50, verbose=True)
np.savez(f'perf/ref_fp32_{name}.npz', latent=out['latent'].cpu().numpy(), camera=out['camera'].cpu().numpy())
print('done', name)
