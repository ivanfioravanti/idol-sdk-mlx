# Task: make the MLX denoiser sampling of idol-sdk-mlx faster, keeping quality

You are in ~/github/idol-sdk-mlx on a Mac Studio M3 Ultra (512 GB), on git branch `perf/claude-opus` (from 15ac77f).
Idol turns one image into 3D Gaussians. 95% of a request is the 50-step flow-matching sampler of a 370M DiT, ported to
MLX in `idol/mlx_dit.py` (`LatentSeqMMFlowModel`, `sample_flow`). Goal: make that 50-step MLX sampling faster with the
output staying numerically equivalent. You have about 2 hours. Work autonomously; do not ask questions.

## Read first
- `PERF_NOTES.md` (full log of what was measured and tried), `README.md` (section "What the MLX port does"),
  `idol/mlx_dit.py`, `perf/README.md`, `tests/test_mlx_dit.py`.

## Environment and harness
- Use `.venv/bin/python` (mlx 0.32.4, torch 2.15). `.venv-main` has MLX main (a21f892): same speed. Do not modify the system Python.
- Inputs already captured: `perf/inputs_{avocado,fox}.npz`. FP32 references already exist: `perf/ref_fp32_{avocado,fox}.npz`
  (50 steps, `perf/ref_torch.py`) and `perf/onepass_fp32.npz` (`python -m perf.onepass torch`). Do not regenerate them.
- Speed: `python -m perf.bench_dit TAG RUNS` (warm-up + RUNS timed 50-step runs per image, appends to perf/results.jsonl,
  saves perf/out_TAG_*.npz). One run is ~80 s per image, so RUNS=3 on both images is ~9 min: run it with nohup to a log
  file and poll it. Quick checks: `python -m perf.prof_dit` (per-pass and per-component ms, ~30 s).
- Quality: `python -m perf.onepass TAG` (one-pass rel error vs FP32) and `python -m perf.compare TAG` (50-step cos/rel/camera).
  Also `python tests/test_mlx_dit.py` must pass.
- Only one GPU job at a time: concurrent jobs distort timings 2-4x.

## Current state (15ac77f), the bar to beat
- 50-step sampling, median of 3: avocado 79.42 s (1.588 s/step), fox 81.55 s (1.631 s/step). Original 0fbe4f1: 83.14 / 83.53 s.
- Per step: latent_stream 44 ms, joint cond pass (12,294 tokens) 1005 ms, uncond pass (8,195 tokens) 550 ms.
  Cond block 41.4 ms = sdpa 27.3 + qkv 3.7 + out 1.4 + mlp 9.3 (+ RePo 1.24 ms). fp16 GEMM peak 24.7 TFLOPS, sdpa runs at ~22.7.
- Quality of 15ac77f vs FP32: one-pass latent rel mean 2.42e-3 (camera 2.08e-3); 50 steps avocado cos 0.99985 rel 1.73e-2
  dyaw 0.075° dpitch 0.025°, fox cos 0.99987 rel 1.64e-2 dyaw 0.070° dpitch 0.121°.

## Acceptance criteria for every kept change
- One-pass latent rel error vs FP32 <= 2.7e-3; 50-step cos vs FP32 >= 0.9998 on both images;
  camera error (dyaw/dpitch and one-pass camera rel) not meaningfully worse than above.
- A measured speedup with the harness (bench_dit, ideally RUNS>=2 on both images), not just a microbenchmark.
- Lossy approximations (step skipping, reusing passes across steps, fewer steps, quantization) are out of scope.

## Already tried (see PERF_NOTES.md for numbers)
- KEPT: plain FP32 matmul on M1-M4 (full precision there), 3xTF32 only on M5 (probe at import).
- No gain: MLX main; quantized linears (compute bound at 12k tokens); fusing the RePo angle computation into the
  rope+RMSNorm kernel (-0.3%, patch in perf/fused_repo_rejected.patch); more mx.compile (both passes already compiled;
  both in one graph hits an MLX `unordered_map::at` bug); host syncs (none idle: 1.588 s/step vs 1.599 s pass sum).
- Rejected: CFG/uncond reuse across steps (lossy, rel 0.1); batch cond+uncond as B=2 (different lengths, padding costs more);
  bf16 (fewer mantissa bits, error budget already FP16-dominated).
- So the code is close to the sdpa + GEMM time. Real wins need new ideas, e.g.: a custom Metal attention kernel via
  mx.fast.metal_kernel tuned for head_dim 64 / these lengths, or attention restructuring (e.g. exact block-wise handling
  of the 2-token / masked part, splitting key sets with log-sum-exp merging, reusing K/V work that is truly identical);
  fusing modulation/LayerNorm/GELU around GEMMs; GEMM shape/layout tricks (merging projections, contiguous layouts,
  avoiding transposes/copies); anything exact that reduces FLOPs. Verify each with the harness; measure before believing.

## Rules
- Commit each kept change on `perf/claude-opus` with a clear message, and log every experiment (kept or rejected, with
  speed and error numbers) by appending to PERF_NOTES.md under a "## Claude Code (Opus) session" header.
- Never push, never open PRs, never change git remotes, never touch other branches (main, perf/mlx-opt).
- Do not kill or touch processes you did not start (there are mlx-serve, LiteLLM and other services on this Mac).
- Do not install system-wide packages. Do not regenerate the FP32 references.
- At the end, print a short summary: each experiment, kept or rejected, speed and error numbers, and final commit hash.
