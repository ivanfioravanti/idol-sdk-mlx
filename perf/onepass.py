"""One guided step velocity (cond, uncond) at t in {900, 500, 100} on the avocado inputs.
python -m perf.onepass torch   -> perf/onepass_fp32.npz  (torch FP32, MPS)
python -m perf.onepass TAG     -> compares MLX against it"""
import os, sys, numpy as np, torch
from tensordict import TensorDict
from perf.common import WEIGHTS, CONFIG
tag = sys.argv[1]
d = np.load('perf/inputs_avocado.npz')
TS = [900.0, 500.0, 100.0]
if tag == 'torch':
    from safetensors.torch import load_file
    from idol.models import LatentSeqMMFlowModel
    from idol.models.layers import RePo3DRotaryEmbedding
    m = LatentSeqMMFlowModel(**CONFIG['denoiser']['args']); m.load_state_dict(load_file(os.path.join(WEIGHTS, 'denoiser.safetensors')))
    m = m.float().eval().to('mps'); m.dtype = torch.float32
    for mod in m.modules():
        if isinstance(mod, RePo3DRotaryEmbedding): mod.real_valued = True
    T = lambda k: torch.from_numpy(d[k]).to('mps')
    x = TensorDict({'latent': T('latent'), 'camera': T('camera')}, batch_size=1)
    c = TensorDict({'feature1': T('feature1'), 'feature2': T('feature2')}, batch_size=1)
    out = {}
    with torch.no_grad():
        for t in TS:
            for nm, cc in (('c', c), ('u', c.apply(torch.zeros_like))):
                v = m(x, torch.tensor([t], device='mps'), cc)
                out[f'{nm}{int(t)}_latent'] = v['latent'].cpu().numpy(); out[f'{nm}{int(t)}_camera'] = v['camera'].cpu().numpy()
    np.savez('perf/onepass_fp32.npz', **out); print('saved'); sys.exit()
import mlx.core as mx
from idol import mlx_dit
model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
ref = np.load('perf/onepass_fp32.npz')
x = {'latent': mx.array(d['latent']), 'camera': mx.array(d['camera'])}
cond = model.condition({'feature1': mx.array(d['feature1']), 'feature2': mx.array(d['feature2'])})
neg = model.condition({'feature1': mx.zeros(d['feature1'].shape), 'feature2': mx.zeros(d['feature2'].shape)})
rels = {}
for t in TS:
    s = model.latent_stream(x, mx.array([t]))
    for nm, cc in (('c', cond), ('u', neg)):
        v = model.joint(s, *cc)
        for k in ('latent', 'camera'):
            a = np.array(v[k]).astype(np.float64); b = ref[f'{nm}{int(t)}_{k}'].astype(np.float64)
            rels.setdefault(k, []).append(np.linalg.norm(a - b) / np.linalg.norm(b))
print(f"{tag:14s} onepass rel err vs FP32  latent mean {np.mean(rels['latent']):.3e} max {np.max(rels['latent']):.3e} | camera mean {np.mean(rels['camera']):.3e} max {np.max(rels['camera']):.3e}")
