Round 2: 50-step MLX sampling is **5.0% faster on avocado (75.87 → 72.11 s) and 5.2% faster on fox (76.05 → 72.11 s)**,
about 1.442 s/step, and the output is **bit-identical** to the branch start (main e31b970) on both images. Every kept
change was checked for bit-identity: unit tests against the MLX ops, the one-pass check, and `perf.compare` of the
50-step outputs against the baseline run (0 difference). So the quality numbers are exactly main's: one-pass latent rel
mean 2.417e-3 (max 3.171e-3), camera mean 2.228e-3 (max 6.806e-3); 50 steps vs FP32 avocado cos 0.99983, rel 1.85e-2,
dyaw 0.037°, dpitch 0.089°; fox cos 0.99984, rel 1.78e-2, dyaw 0.133°, dpitch 0.029°. All within the acceptance
criteria, camera included. `tests/test_mlx_dit.py` passes, with new tests for each new kernel.

The baseline was measured fresh in this session (bench_dit, 2 runs): avocado 75.87 s, fox 76.05 s. Each step below is
a bench_dit median of 2 runs, both images.

**Kept**

| # | Change | Commit | avocado / fox | Per pass (prof_dit, cond / uncond) |
|---|---|---|---|---|
| — | start (main e31b970) | 574204e | 75.87 / 76.05 s | 971 / 517 ms |
| 16 | Attention: K stored transposed in threadgroup memory (plain instead of transposed fragment loads); the uncond key bias only added in the last key block (it is zero on the 8,192 latent keys) | 54ee0bc | 74.52 / 74.52 | 955.7 / 505.0 |
| 17 | Own FP16 GEMM + bias kernel for the block linears (64×64 tile, 4 simdgroups, K step 32, against MLX's 2 simdgroups and K step 16); same MMAs in the same order, bit-identical to `mx.addmm` on the shapes routed to it | 701ffd5 | 73.37 / 73.38 | 943.6 / 496.5 |
| 18 | GEMM epilogues: fc1 → GELU, out projection and fc2 → `x + y * gate` (each FP16 op rounded as MLX: FMA contraction off, tanh in FP32) | 0bad711 | 72.73 / 72.73 | 935.4 / 492.4 |
| 21 | Last block: the image tokens only serve as keys and values, since their outputs are unused (8,193 queries over 12,294 keys, and the out projection and MLP run on those rows only); rope kernel with one sincos per q/k pair | 0182519 | **72.11 / 72.11** | 922.6 / 491.4 |

Final check (HEAD, bench_dit 3 runs): avocado 72.100 / 72.109 / 72.118 s (median 72.11), fox 72.107 / 72.111 / 72.114 s (median 72.11); `perf.compare final2 r2base`: 0 difference on both images.

**Rejected** (numbers in PERF_NOTES.md, experiments 16-24)
- Attention: P rounded to FP16 before the PV MMA (27.05 vs 26.8 ms, slower); scale·log2e folded into FP16 Q (−0.7%,
  but it changes Q's rounding, not worth the risk for 0.2 ms); a fixed per-row softmax bound with no online max or rescale
  (no gain: the kernel is MMA/load-bound); one lane per key for the transposed stores, threadgroup padding 0-16
  (noise); 2 or 8 simdgroups, BK=16 or BK=64 with the new layout (all slower).
- GEMM: other tiles (2×4 / 4×2 / 1×2 / 2×1 simdgroups, 8×4 fragments, K step 16 / 64) and register prefetch of the next
  K tile: no gain. With batched timing the block GEMMs run at 25.2-25.9 TFLOPS against a register-only MMA ceiling of
  ~26 (perf/mma_peak.py).
- FP32 GEMM for the RePo projection: bit-identical and 0.40 → 0.32 ms in isolation, but no change in the passes.
- One-simdgroup-per-token LayerNorm kernels: bit-identical, but the production kernels already run at ~570 GB/s
  (the per-call micro numbers were eval overhead).

**Fox slower than avocado?** It's machine load, not the data. Same shapes, no data-dependent code, and in this session
the two are equal: 75.87 / 76.05 s at the start, 74.52 / 74.52 with fox run first, then 73.37 / 73.38, 72.73 / 72.73 and
72.11 / 72.11. In the independent runs avocado's third run also drifted to 77.9-78.2 s and fox, run after it, stayed at
~78.8 s: the Mac was under other load (at the start of this session a system process, shown truncated by ps as /System/Library/PrivateFrameworks/Me..., was at 99% CPU).

**Where the time goes now:** attention ≈ 63% of a step at ~24 TFLOPS (~92% of the MMA ceiling; flat from L=6k to 12k,
so no wave-quantization loss). The GEMMs ≈ 30% are at 97-99% of it. What's left outside those is ~1 ms per block.
Further exact gains would need fewer FLOPs, not faster kernels.

Code: 0182519; experiment scripts ad67d6b; this summary is the last commit on perf/round2 (not pushed). PERF_NOTES.md (excluded from git by .git/info/exclude) has the full log under "Claude Code (Opus) round 2".
