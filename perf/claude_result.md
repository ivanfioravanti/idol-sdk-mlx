50-step sampling is now 4.5% faster on avocado (79.42 → 75.88 s) and 6.9% faster on fox (81.55 → 75.89 s), about 1.518 s/step, with output equivalent to before. Each kept change passed the acceptance criteria and was measured with `bench_dit` (3 runs, both images). Compared with the original 0fbe4f1 (83.14 / 83.53 s) it's about 9% faster. `tests/test_mlx_dit.py` passes, and I added tests for each new kernel.

**Kept** (times are cumulative medians, avocado / fox):

| # | Change | Commit | 50-step time | One-pass rel (latent / camera) | 50-step cos | dyaw / dpitch (avocado; fox) |
|---|---|---|---|---|---|---|
| — | starting point | 15ac77f | 79.42 / 81.55 s | 2.42e-3 / 2.08e-3 | 0.99985 / 0.99987 | 0.075°/0.025°; 0.070°/0.121° |
| 11 | Own flash-attention kernel for head_dim 64 (16 query rows per simdgroup instead of MLX's 8; output written in the layout the next layer needs) | a09ffab | 78.48 / 78.79 | 2.421e-3 / 1.84e-3 | 0.99994 / 0.99988 | 0.081°/0.006°; 0.012°/0.026° |
| 12 | FP16 weights stored pre-transposed (MLX's GEMM is ~5% faster on them) | 41c34dd | 77.26 / 77.28 | 2.421e-3 / 2.10e-3 | 0.99987 / 0.99986 | 0.109°/0.007°; 0.089°/0.052° |
| 14 | RePo layer fused: a one-pass LayerNorm kernel and one kernel from the projections to the angles (angles agree to 2.5e-7) | 2d59aff | 76.89 / 76.97 | 2.422e-3 / 2.05e-3 | 0.99985 / 0.99987 | 0.118°/0.016°; 0.025°/0.027° |
| 15 | LayerNorm + modulation in one kernel (same FP16 rounding steps) | 7287533 | **75.88 / 75.89** | 2.417e-3 / 2.23e-3 | 0.99983 / 0.99984 | 0.037°/0.089°; 0.133°/0.029° |

- **Attention speed:** 27.3 → 26.8 ms on the conditional pass (12,294 tokens) and 13.1 → 12.5 ms on the unconditional one (8,195 tokens, with the key bias), with the same error vs FP32.
- **Camera drift:** the camera angles move around between changes. The worst is fox dyaw 0.133° in the last change, against 0.070° before. That change only shifts some FP16 values by one ulp (0.004% of them), and the four angles summed over both images are 0.288° vs 0.291° at 15ac77f. One-pass camera max went from 4.0e-3 to 6.8e-3; one-pass camera mean across these equivalent variants stayed between 1.8e-3 and 2.2e-3. I read this as FP16 noise rather than a trend, but it's the one number you might want to look at.
- **Memory:** peak use dropped from 1.93 to 1.82 GB.

**Rejected** (all attention variants measured on random data at 12,294 tokens; the kept kernel ran 26.1–26.5 ms in the same runs):

| # | Variant | Time | Why it lost |
|---|---|---|---|
| 11 | Half-precision fragments with hardware transposed loads | 26.1 ms | same speed |
| 11 | Double-buffered K/V | 26.4 ms | no gain |
| 11 | 64 keys per block | 30.0 ms | slower |
| 11 | 16 keys per block | 27.4 ms | slower |
| 11 | 8 query rows per simdgroup | 27.1 ms | slower |
| 11 | Scale folded into float Q | 30.5 ms | the mixed float×half multiply is slower |
| 11 | Lazy softmax rescaling | 26.4 ms | no gain: the kernel is limited by matrix-multiply throughput, not the softmax |
| 13 | 24 query rows per simdgroup | 29.5 ms | too many registers in use |
| 13 | 8 simdgroups per threadgroup | 27.3 ms | slower |

Everything is logged with numbers under "## Claude Code (Opus) session" in `PERF_NOTES.md`. Note that `PERF_NOTES.md`, `perf/results.jsonl` and the logs are listed in `.git/info/exclude`, so those notes are local only and not in the commits. I also added the kept changes to the README's "What the MLX port does" table. Nothing was pushed, and I left the untracked `perf/claude_watch.py` alone because I didn't create it.

Final commit: **7287533** on `perf/claude-opus`.