# MLX denoiser perf harness
- `python -m perf.capture` saves the denoiser inputs (image tokens, seed-42 noise) for avocado and fox.
- `python -m perf.ref_torch NAME` writes the 50-step PyTorch FP32 (MPS) reference; `python -m perf.onepass torch` the one-pass reference.
- `python -m perf.bench_dit TAG [RUNS]`: warm-up + timed 50-step MLX sampling, appends to perf/results.jsonl, saves outputs.
- `python -m perf.compare TAG` (50-step cos / rel / camera vs FP32 and baseline), `python -m perf.onepass TAG` (one-pass rel error).
- `python -m perf.prof_dit`, `perf.prof_repo`, `perf.micro_mm`, `perf/micro_attn.py`: per-component timings.
- `python -m perf.cfgcache`: the (rejected, lossy) unconditional-pass reuse experiment.
