# Task (round 2): make the MLX denoiser sampling of idol-sdk-mlx faster again, keeping quality

You are in ~/github/idol-sdk-mlx on a Mac Studio M3 Ultra (512 GB), on git branch `perf/round2` (from main e31b970,
which already contains round 1). Idol turns one image into 3D Gaussians; 95% of a request is the 50-step flow-matching
sampler of a 370M DiT in `idol/mlx_dit.py` (`LatentSeqMMFlowModel`, `sample_flow`). Goal: make the 50-step MLX sampling
faster while the output stays numerically equivalent. You have about 2 hours. Work autonomously; do not ask questions.

## Read first
- `PERF_NOTES.md` (full log of rounds 0 and 1, experiments 1-15), `perf/claude_result.md` (your round-1 summary),
  `README.md` ("What the MLX port does"), `idol/mlx_dit.py`, `perf/README.md`, `tests/test_mlx_dit.py`, `perf/attn_kernel.py`.

## Environment and harness
- `.venv/bin/python` (mlx 0.32.4, torch 2.15). `.venv-main` = MLX main a21f892 (same speed). Do not touch the system Python.
- Inputs and FP32 references already exist: `perf/inputs_{avocado,fox}.npz`, `perf/ref_fp32_{avocado,fox}.npz`,
  `perf/onepass_fp32.npz`. Do not regenerate them.
- Speed: `python -m perf.bench_dit TAG RUNS` (warm-up + timed 50-step runs, both images; ~80 s per run). Run it with nohup
  to a log and poll. Quick: `python -m perf.prof_dit` (~30 s). Attention check: `python -m perf.verify_attn`.
- Quality: `python -m perf.onepass TAG`, `python -m perf.compare TAG [BASE_TAG]`; `python tests/test_mlx_dit.py` must pass.
- One GPU job at a time. Other things may load the Mac (mlx-serve, mediaanalysisd...); check `uptime`/`ps` and re-run a
  benchmark if the run-to-run spread is large. Compare against a fresh measurement of the branch start in the same session.

## Current state (main e31b970), the bar to beat
- 50-step sampling, median of 3, independently verified: avocado 76.2 s (1.523 s/step), fox 78.8 s (1.576 s/step).
  (In your own round-1 runs both were 75.9 s; the fox slowdown showed up when the Mac had other load.)
- Per step (round-1 end): latent_stream ~43 ms, cond pass (12,294 tokens) ~971 ms, uncond pass (8,195 tokens, 2-token key
  bias) ~517 ms. Attention is ~59% of a step: 26.8 ms cond / 12.5 ms uncond per block. fp16 GEMM peak ~24.7 TFLOPS.
- Quality of main vs FP32: one-pass latent rel mean 2.417e-3 (max 3.17e-3), camera mean 2.23e-3 max 6.81e-3;
  50 steps avocado cos 0.99983 rel 1.85e-2 dyaw 0.037° dpitch 0.089°; fox cos 0.99984 rel 1.78e-2 dyaw 0.133° dpitch 0.029°.

## Acceptance criteria for every kept change (stricter on camera this round)
- One-pass latent rel mean vs FP32 <= 2.7e-3; one-pass camera max <= 7e-3 (mean not meaningfully above 2.3e-3).
- 50-step cos vs FP32 >= 0.9998 on both images; no single camera angle error (dyaw or dpitch, either image) > 0.15°.
- A real speedup measured with bench_dit (RUNS>=2, both images), not only a microbenchmark.
- No lossy approximations (step skipping, pass reuse across steps, fewer steps, quantization, bf16).

## Already tried — do not repeat (numbers in PERF_NOTES.md)
- KEPT: plain FP32 matmul on M1-M4; custom hd64 flash attention (TQ=2, 16 query rows/simdgroup, BK=32, 4 simdgroups);
  pre-transposed contiguous FP16 weights with addmm; fused RePo (one-pass LN kernel + angles kernel); LN + adaLN modulation kernel.
- REJECTED / no gain: MLX main; quantized linears; CFG/uncond reuse (lossy); batching cond+uncond as B=2 (different lengths);
  bf16; more mx.compile (passes already compiled, both passes in one graph hits an MLX unordered_map::at bug); host syncs (none idle);
  attention variants: half fragments with transposed loads, double-buffered K/V, BK=64, BK=16, TQ=1, TQ=3, TQ=4x2sg,
  8 simdgroups, scale folded into float Q, lazy softmax rescaling (kernel is MMA-bound).

## Directions to explore this round (verify everything with the harness)
- The attention kernel itself (~59% of the step): tile shapes not yet tried, K/V tiles shared across simdgroups via
  threadgroup memory (reuse), better MMA scheduling/instruction mix, a specialised path for the uncond pass where 2 keys carry a
  bias (e.g. handle those keys separately with an exact log-sum-exp merge so the main loop has no bias/mask work), L tails.
- Fusing the qkv projection output / rope + RMSNorm into the attention prologue (fewer round trips through memory).
- MLP: fuse GELU (tanh) into the fc1 GEMM epilogue or into fc2's input; fewer kernel launches per block in general.
- fp16 GEMM layout/shape tricks for these sizes (M=12294/8195, K=1024/4096), e.g. merging projections, split/stream-K.
- Explain why fox was slower than avocado in the independent runs (same shapes, so likely machine load or data-dependent
  behaviour: check with repeated alternating runs) and fix it if it is something in the code.

## Rules
- Commit each kept change on `perf/round2` with a clear message; append every experiment (kept or rejected, speed + error
  numbers) to PERF_NOTES.md under "## Claude Code (Opus) round 2". Update the README table for kept changes.
- Never push, never open PRs, never change git remotes, never touch other branches.
- Do not kill or touch processes you did not start (mlx-serve, LiteLLM, other services). No system-wide installs.
- At the end, print a short summary: each experiment kept/rejected with speed and error numbers, and the final commit hash.
  Also write it to perf/claude_result_round2.md and commit it.
