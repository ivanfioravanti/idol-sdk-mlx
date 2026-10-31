"""Time the MLX denoiser (50-step sample_flow) on saved inputs; save outputs for the quality check.
python -m perf.bench_dit TAG [runs]"""
import json, os, statistics, subprocess, sys, time
import mlx.core as mx, numpy as np, torch
from tensordict import TensorDict
from idol import mlx_dit
from perf.common import WEIGHTS, CONFIG, IMAGES
tag = sys.argv[1]; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
names = sys.argv[3].split(',') if len(sys.argv) > 3 else list(IMAGES)
commit = subprocess.run(['git', 'describe', '--always', '--dirty'], capture_output=True, text=True).stdout.strip()
model = mlx_dit.load_denoiser(os.path.join(WEIGHTS, 'denoiser.safetensors'), CONFIG['denoiser']['args'])
for name in names:
    d = np.load(f'perf/inputs_{name}.npz')
    T = lambda k: torch.from_numpy(d[k])
    noise = TensorDict({'latent': T('latent'), 'camera': T('camera')}, batch_size=1)
    cond = TensorDict({'feature1': T('feature1'), 'feature2': T('feature2')}, batch_size=1)
    neg = cond.apply(torch.zeros_like)
    mlx_dit.sample_flow(model, noise, cond, neg, steps=50, verbose=False)   # warm-up (compile)
    times = []
    mx.reset_peak_memory()
    for r in range(runs):
        t0 = time.perf_counter()
        out = mlx_dit.sample_flow(model, noise, cond, neg, steps=50, verbose=False)
        times.append(time.perf_counter() - t0)
    peak = mx.get_peak_memory() / 2**30
    np.savez(f'perf/out_{tag}_{name}.npz', latent=out['latent'].numpy(), camera=out['camera'].numpy())
    row = dict(tag=tag, commit=commit, mlx=mx.__version__, image=name, runs=runs, times=[round(t, 3) for t in times],
               median_s=round(statistics.median(times), 3), mean_s=round(statistics.mean(times), 3),
               s_per_step=round(statistics.median(times) / 50, 4), peak_gb=round(peak, 2), time=time.strftime('%H:%M:%S'))
    print(json.dumps(row), flush=True)
    open('perf/results.jsonl', 'a').write(json.dumps(row) + '\n')
