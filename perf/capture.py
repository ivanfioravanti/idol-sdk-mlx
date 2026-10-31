"""Save the denoiser inputs (image tokens + seed-42 noise) for each test image, exactly as the pipeline makes them."""
import numpy as np, torch
from idol import IdolPipeline
from perf.common import IMAGES
p = IdolPipeline.from_pretrained(backend='mlx')
for name, path in IMAGES.items():
    prepared = p.preprocess(path, mode='synthetic')
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(p.device)
    torch.manual_seed(42)
    cond = p.image_encoder(pixels)
    lat = torch.randn(1, 8192, 16, device=p.device); cam = torch.randn(1, 1, 5, device=p.device)
    np.savez(f'perf/inputs_{name}.npz', feature1=cond['feature1'].cpu().numpy(), feature2=cond['feature2'].cpu().numpy(),
             latent=lat.cpu().numpy(), camera=cam.cpu().numpy())
    print(name, cond['feature1'].shape, cond['feature2'].shape, flush=True)
