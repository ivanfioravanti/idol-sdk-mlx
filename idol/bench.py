"""Time the denoiser backends on the same images and log every run.

    python -m idol.bench --image demo/examples/food-avocado.webp --runs 3

One warm-up per backend (torch.compile / mx.compile), then the backends alternate run by run, so thermal throttling
does not favour whichever runs first. Each run appends a JSON line (commit, chip, stage timings, camera) to --log;
the summary prints medians and how far each backend's camera is from the first backend's (same seed)."""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time

import torch

from .pipeline import IdolPipeline


def device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    if platform.system() == 'Darwin':
        return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', nargs='+', default=['demo/examples/food-avocado.webp'])
    parser.add_argument('--backends', nargs='+', choices=['torch', 'mlx'], default=['torch', 'mlx'])
    parser.add_argument('--runs', type=int, default=3, help='timed runs per backend and image')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_gaussians', type=int, default=262144)
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    parser.add_argument('--log', default='benchmarks.jsonl')
    args = parser.parse_args()

    commit = subprocess.run(['git', 'describe', '--always', '--dirty'], capture_output=True, text=True,
                            cwd=os.path.dirname(__file__)).stdout.strip()
    common = {'commit': commit, 'device': device_name(), 'steps': args.steps, 'num_gaussians': args.num_gaussians}
    pipelines, load_s = {}, {}
    for backend in args.backends:
        start = time.perf_counter()
        pipelines[backend] = IdolPipeline.from_pretrained(backend=backend)
        load_s[backend] = time.perf_counter() - start

    rows = []
    for image in args.image:
        for pipeline in pipelines.values():
            pipeline(image, mode=args.mode, steps=2, num_gaussians=args.num_gaussians)   # warm-up: compiles
        for run in range(args.runs):
            for backend in (args.backends if run % 2 == 0 else args.backends[::-1]):
                output = pipelines[backend](image, mode=args.mode, steps=args.steps, num_gaussians=args.num_gaussians)
                row = {**common, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'image': os.path.basename(image),
                       'backend': backend, 'run': run, 'load_s': round(load_s[backend], 2),
                       **{f'{stage}_s': round(seconds, 3) for stage, seconds in output.timings.items()},
                       'total_s': round(sum(output.timings.values()), 3), 'camera': output.camera}
                rows.append(row)
                with open(args.log, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"{row['image']} {backend} run {run}: {row['total_s']:.1f}s (sample {row['sample_s']:.1f}s)")

    print(f"\n{common['device']}, {commit}, {args.steps} steps, {args.num_gaussians:,} Gaussians, "
          f"median of {args.runs} (seconds)")
    print(f"{'image':28s} {'backend':7s} {'total':>6s} {'sample':>6s} {'s/step':>6s} {'encode':>6s} {'decode':>6s} "
          f"{'speedup':>7s} {'d_yaw':>6s} {'d_pitch':>7s}")
    for image in dict.fromkeys(row['image'] for row in rows):
        def median(backend, key):
            return statistics.median(row[key] for row in rows if row['image'] == image and row['backend'] == backend)
        base = args.backends[0]
        camera = next(row['camera'] for row in rows if row['image'] == image and row['backend'] == base)
        for backend in args.backends:
            other = next(row['camera'] for row in rows if row['image'] == image and row['backend'] == backend)
            d_yaw = math.degrees(abs(math.remainder(other['yaw'] - camera['yaw'], 2 * math.pi)))
            d_pitch = math.degrees(abs(other['pitch'] - camera['pitch']))
            print(f"{image:28s} {backend:7s} {median(backend, 'total_s'):6.1f} {median(backend, 'sample_s'):6.1f} "
                  f"{median(backend, 'sample_s') / args.steps:6.2f} {median(backend, 'encode_s'):6.2f} "
                  f"{median(backend, 'decode_s'):6.2f} {median(base, 'total_s') / median(backend, 'total_s'):6.2f}x "
                  f"{d_yaw:5.2f}° {d_pitch:6.2f}°")


if __name__ == '__main__':
    main()
