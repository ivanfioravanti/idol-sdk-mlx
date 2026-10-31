"""python -m perf.compare TAG [BASE_TAG]: errors of out_TAG vs the torch FP32 reference and vs out_BASE."""
import math, sys, numpy as np
from idol.camera import decode_camera
import torch
from perf.common import IMAGES
tag = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
def metrics(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return dict(cos=float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), rel=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                max_abs=float(np.abs(a - b).max()), mean_abs=float(np.abs(a - b).mean()))
def cam(x):
    c = decode_camera(torch.from_numpy(x['camera'][0, 0])); return c
for name in IMAGES:
    try: o = np.load(f'perf/out_{tag}_{name}.npz')
    except FileNotFoundError: continue
    for ref_name in [f'ref_fp32_{name}', f'out_{base}_{name}']:
        try: r = np.load(f'perf/{ref_name}.npz')
        except FileNotFoundError: continue
        m = metrics(o['latent'], r['latent']); co, cr = cam(o), cam(r)
        dyaw = math.degrees(abs(math.remainder(co['yaw'] - cr['yaw'], 2 * math.pi))); dp = math.degrees(abs(co['pitch'] - cr['pitch']))
        print(f"{tag:14s} {name:8s} vs {ref_name:22s} cos {m['cos']:.5f} rel {m['rel']:.3e} max {m['max_abs']:.3e} mean {m['mean_abs']:.3e} dyaw {dyaw:.3f}° dpitch {dp:.3f}°")
