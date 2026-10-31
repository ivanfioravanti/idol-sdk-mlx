# Idol

**A single image to 3DGS object model**

Idol turns one picture of an object into a 3D Gaussian splat. Give it a render, a generated image or a photo: it removes the background, frames the object and generates up to 1,048,576 Gaussians, including the sides the picture never showed. It runs on consumer GPUs, using 7.2 GB of VRAM and can run in under 2 seconds.

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/speridlabs/idol)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/speridlabs/idol)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

| ![](assets/gallery/armchair.webp) | ![](assets/gallery/avocado.webp) |
|---|---|
| ![](assets/gallery/lamp.webp) | ![](assets/gallery/axolotl.webp) |

Each pair: the input image and the generated object turning. The
[demo](https://huggingface.co/spaces/speridlabs/idol) has 93 more examples ready to run and takes your own images as well.

## Model

| | |
|---|---|
| Output | 3D Gaussians: 32,768 to 1,048,576 (default 262,144), standard 3DGS `.ply` |
| DiT | 370M parameters |
| Image conditioning | DINOv3 ViT-H+ and FLUX.2 VAE |
| Decoder | 288M parameters |
| Background removal | BiRefNet |
| Sampler | Euler, 50 steps, classifier-free guidance 3.0, time shift 3.0 |

## Install

```bash
uv sync                       # Python >= 3.10: PyTorch, MLX (Apple Silicon) and Gradio for the demo
# or with pip:
pip install -e .              # PyTorch >= 2.5, and MLX on Apple Silicon
pip install -e ".[demo]"      # + Gradio, to run the demo locally
```

The weights are downloaded from Hugging Face on first use, you need to request access to
[facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
and [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) first, then log in with
`hf auth login`. 

`flash-attn`, if installed, is used for attention; otherwise PyTorch's `scaled_dot_product_attention` is (same
result, slower). Set `ATTN_BACKEND=sdpa` or `ATTN_BACKEND=flash_attn` to choose.

## Inference

```bash
python -m idol.sample --image photo.jpg --mode real
python -m idol.sample --image renders/ --mode synthetic --out outputs
```

For each image this writes `<name>.ply` (open it in any Gaussian splat viewer), `<name>_input.png` (the image the model saw, after background
removal and cropping) and `<name>_camera.json` (the predicted camera of the input).

**Pick the mode for your input:** 

| `--mode` | for | 
|---|---|
| `synthetic` (default) | computer generated images| 
| `real` | real world photographs | 

Other flags:

| flag | default | |
|---|---|---|
| `--seed` | 42 | same seed, same object |
| `--steps` | 50 | sampling steps |
| `--cfg_strength` | 3.0 | classifier-free guidance |
| `--rescale_t` | 3.0 | time shift: higher spends more steps at high noise |
| `--num_gaussians` | 262144 | 32,768 to 1,048,576, a multiple of 32 |
| `--dinov3_dtype` | bf16 | `fp32` runs the DINOv3 encoder in full precision (1.7 GB more memory) |
| `--weights` | `speridlabs/idol` | a Hugging Face repo or a local folder |
| `--no_optimize` | optimized | turn off the faster sampling (see below) |
| `--backend` | `torch` | `mlx` runs the denoiser in MLX on Apple Silicon (see [Apple Silicon](#apple-silicon-mps-and-mlx)) |

**Apple Silicon:** runs on MPS automatically; `--backend mlx` is faster. See [Apple Silicon](#apple-silicon-mps-and-mlx).

**Faster sampling is on by default on GPU** The first image in a process is slower while it pytorch compiles; the rest run at full
speed. 

### Python API

```python
from idol import IdolPipeline

pipeline = IdolPipeline.from_pretrained("speridlabs/idol")
output = pipeline("photo.jpg", mode="real", seed=42, num_gaussians=1_048_576)
output.gaussian.save_ply("photo.ply")
output.preprocessed_image.save("photo_input.png")
```

### Demo

The Gradio app behind the [Hugging Face Space](https://huggingface.co/spaces/speridlabs/idol) is in `demo/`, with
the example images. In this fork it is adapted to run locally on Apple Silicon with the MLX backend: see
[the Gradio demo on Apple Silicon](#gradio-demo-on-apple-silicon).

## Apple Silicon (MPS and MLX)

On a Mac the pipeline runs on the GPU through PyTorch MPS, picked automatically. With the MLX backend the denoiser,
the 50-step loop that takes ~95% of a request, runs in [MLX](https://github.com/ml-explore/mlx) instead
(`idol/mlx_dit.py`). Background removal, DINOv3, the FLUX.2 VAE and the two decoders stay on MPS: they run once per
image, about 2 s together.

```bash
uv sync                       # or: pip install -e .
uv run python -m idol.sample --image photo.jpg --backend mlx
```

```python
pipeline = IdolPipeline.from_pretrained("speridlabs/idol", backend="mlx")
output = pipeline("photo.jpg")
print(output.timings)   # seconds per stage: preprocess, encode, sample, decode (every backend)
```

The MLX denoiser loads the same `denoiser.safetensors`, no conversion step. Same seed, same noise: the two backends
generate the same object (see [Accuracy](#accuracy)).

### Speed

One image to 262,144 Gaussians at 50 steps on an M5 Max (128 GB), median of 3 runs with the backends alternating:

| Image | PyTorch MPS | MLX | Speedup |
|---|---|---|---|
| `food-avocado.webp` | 169.4 s | 102.8 s | 1.65x |
| `fig-fox.webp` | 171.5 s | 89.4 s | 1.92x |

Other work was using the GPU during these runs, so the absolute times are slower than on an idle Mac; the backends
alternated run by run, so the speedup is the number to compare. They predate the last two optimizations below (the
shared latent stream and the merged RePo matmul). Measure your machine:

```bash
python -m idol.bench --image demo/examples/food-avocado.webp demo/examples/fig-fox.webp --runs 3
```

It warms each backend up, alternates them run by run, appends every run (commit, chip, stage timings, camera) to
`benchmarks.jsonl` and prints medians with how far MLX's predicted camera is from PyTorch's. Nothing else should use the
GPU meanwhile: a second job slows both backends 2-4x.

### What the MLX port does

Profiled per component. Each change below was checked against the PyTorch FP32 reference before it was kept.

| Change | Why | Effect |
|---|---|---|
| Same module tree and parameter names as `models/dit.py`; FP16 transformer blocks, FP32 elsewhere | loads the PyTorch checkpoint as is, same precision layout as the original | |
| Full-precision FP32 matmuls ("3xTF32") | MLX's GPU matmul rounds FP32 operands to TF32 (~8e-4 relative error), which the FP32 RoPE angle layers amplify. Each operand is split into a TF32-exact part and its remainder, and three products are summed | 50-step agreement with FP32: cos 0.968 → 0.995; costs ~3% |
| 3xTF32 only where needed | M1–M4 GPUs multiply FP32 in full precision (5e-7 relative error); only M5 rounds to TF32. A probe at import picks plain FP32 matmuls when they are exact | M3 Ultra: RePo layer 2.1 → 1.2 ms per block, sampling −4.5%, same error |
| Fixed position embedding computed by the PyTorch module at load | its frequencies reach 2^16: sin() of FP32 arguments up to ~4e5 turns 1-ulp differences into errors of 0.03 | exact |
| Fused RoPE + RMSNorm Metal kernel | rotating and normalising q and k as separate MLX ops took 7.6 ms of a 24 ms block (profiled alone) for memory-bound work | 0.43 ms per block |
| Image tokens computed once per request | the context refiner has no time input, so its output is the same at all 50 steps | skips 2 blocks per pass |
| Unconditional pass on 8,195 tokens instead of 12,294 | its image input is all zeros, so the 4,101 image tokens stay identical through every block (RoPE here comes from token features, not positions), and n identical keys attend like one key with log(n) added to its score. Kept as two tokens whose FP16 biases sum to n, as one FP16 bias would be off by 0.4% | unconditional pass 519 → 304 ms |
| Latent stream shared by both guidance passes | the noise refiner, time and camera embeddings do not see the condition | skips 2 blocks per step |
| One precise matmul for each RePo layer's gate and content projections | halves the operand splits and matmul launches | |
| `mx.compile` per pass, next step queued while the GPU runs the current one | | |
| Own flash-attention Metal kernel for head_dim 64 (M1–M4) | MLX's kernel gives each simdgroup 8 query rows and reloads Q and K fragments for every MMA; here 16 rows stay in registers, so each K / V fragment feeds two MMAs. Same FP32 online softmax; reads V from the qkv projection, writes the layout the output projection takes | attention 27.3 → 26.8 ms (12,294 tokens), 13.1 → 12.5 ms (8,195, with the key bias); sampling −1.2 to −3.4% |
| Flash attention on the M5 Neural Accelerators | the kernel above runs on `simdgroup_matrix`, which M5 executes ~3x slower than its Neural Accelerators. On M5 a second kernel uses Metal 4 tensor ops (`mpp::tensor_ops::matmul2d`): QKᵀ as one 16x32x64 op and PV as one 16x64x32 op per simdgroup, K / V read by the op from device memory, the score tile handed in registers from one op to the next. P goes in FP16: the accelerators truncate FP32 operands to TF32, which MLX's own NAX attention accepts (4x the error, before output rounding); FP16 is as fast. Picked by the same import-time probe as 3xTF32 | M5 Max: 2.37x faster guided step than with the simdgroup kernel, 5% faster than with MLX's NAX SDPA; attention runs at the large-GEMM throughput (31.5 vs 31.7 TFLOPS at sustained clocks); 50 steps 117.4 → 63–72 s (avocado), 120.0 → 65.8 s (fox); one-pass error unchanged (2.41e-3) |
| FP16 linear weights stored transposed and contiguous | MLX's GEMM is ~5% faster on `[in, out]` weights than on the `weight.T` view | sampling −1.6% |
| Fused RePo layer: one-pass LayerNorm kernel and one kernel from the gate / content projections to the angles | replaces a cast, a 48-output FP32 matmul and the elementwise ops around it; same FP32 math (angles agree to 2.5e-7) | RePo 1.17 → 0.82 ms per block; sampling −0.4% |
| LayerNorm + adaLN modulation in one kernel | the cast, `mx.fast.layer_norm`, cast and multiply-add were three passes over the tokens; same FP16 rounding steps (1-ulp differences on 0.004% of values) | sampling −1.3% |

Tried and dropped: running both guidance passes as one batch-2 forward (no faster in PyTorch MPS, slower in MLX:
1.40 s against 2 x 0.61 s); bfloat16 kernels (attention as fast as float16, matmuls ~12% slower).

The PyTorch path needed three changes to run on MPS: picking `mps` as the device, BiRefNet in FP16, and the denoiser's
real-valued RoPE (complex tensors are poorly supported on MPS; the math is the same).

### Accuracy

| Check | PyTorch MPS (FP16) | MLX |
|---|---|---|
| One forward pass, relative error vs PyTorch FP32 (latent / camera) | 3.43e-3 / 4.12e-3 | 3.53e-3 / 4.37e-3 |
| 50 sampling steps, cosine similarity to PyTorch FP32 (latent) | 0.9959 | 0.9922 |
| Real images, input camera vs PyTorch MPS (yaw / pitch) | | avocado 0.10° / 0.04°, fox 0.04° / 0.02° |

The first two rows use random tokens in place of the image encoders. All 50-step FP16 runs differ from FP32 by a
comparable amount: rounding compounds over the steps in PyTorch's own FP16 path too. `python tests/test_mlx_dit.py`
checks the Metal kernel and the FP32 matmul against PyTorch.

Known MLX limitations (0.32.4 and `main` at `1ba7f969`) worked around here: two guidance passes in one compiled graph
fail with `unordered_map::at: key not found`, so each pass compiles separately; attention masks must match the query
dtype (FP16), hence the two-token log-count bias above.

### Gradio demo on Apple Silicon

```bash
uv sync                           # or: pip install -e ".[demo]"
cd demo && uv run python app.py   # http://127.0.0.1:7862
```

The model loads on the first Generate (MLX denoiser, everything else on MPS) and stays loaded; requests run one at a
time. Every result (`.ply`, input images, the settings and the real stage timings) is saved to `outputs-demo/`, and the
latest one is shown again after a restart. Settings:

| Variable | Default | |
|---|---|---|
| `IDOL_PORT` | `7862` | port, on 127.0.0.1 |
| `IDOL_WEIGHTS` | `speridlabs/idol` | a Hugging Face repo or a local folder |
| `IDOL_OUTPUT_DIR` | `outputs-demo/` | where results go |
| `IDOL_PRELOAD_DIR` | `~/github/idol-sdk-mlx/outputs-mlx` | a folder with `food-avocado.ply` and `food-avocado_input.png` to show before the first Generate |

Once the weights are cached, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` runs it without network access. Setup
details and a walkthrough are in [`demo/README.md`](demo/README.md).

## More details about the model

| Component | Parameters | Precision on GPU | Weights in VRAM |
|---|---|---|---|
| DiT (denoiser) | 370.6M | FP16 transformer blocks; FP32 embeddings and norms | ~0.8–1.0 GB |
| Octree decoder | 50.8M | FP16 transformer blocks; FP32 embeddings and norms | ~0.1–0.2 GB |
| Gaussian decoder | 237.3M | FP16 transformer blocks; FP32 embeddings and norms | ~0.5–0.9 GB |
| BiRefNet (background removal) | 220.7M | FP16 | ~0.44 GB |
| DINOv3 ViT-H+/16  | 840.6M | bf16 | ~1.68 GB |
| FLUX.2 VAE  | 84.1M  | bf16 (FP32 checkpoint, cast on load) | ~0.17 GB |
| **Full request** | **~1.80B** | mostly FP16 / bf16, some FP32 | **~3.7–4.4 GB of weights, 7.2 GB peak** |

## Benchmark

Idol, TripoSplat, TRELLIS and LGM, scored at the input view (PSNR, SSIM, LPIPS, IoU) and at seven new views
(DINOv2, CLIP, sharpness distance). Mean scores, 1 × H100 per model. Bold is the best in each column.

| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | DINOv2 ↑ | CLIP ↑ | Sharpness distance ↓ | Request time ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Idol** | 50 steps, 262,144 Gaussians | 12.25 | **0.726** | **0.213** | **0.862** | **0.781** | **0.885** | **0.81** | 6.8 s |
| TripoSplat | 50 steps, 262,144 Gaussians | 11.76 | 0.712 | 0.225 | 0.860 | 0.743 | 0.868 | 1.03 | 9.9 s |
| TRELLIS | Paper setting, 50 + 50 steps | 11.89 | 0.716 | 0.235 | 0.845 | 0.754 | 0.868 | 0.97 | 7.1 s |
| LGM | Default, 65,536 Gaussians | **12.46**\* | 0.722 | 0.237 | 0.834 | 0.575 | 0.784 | 1.22 | **1.1 s** |

\* LGM has the highest PSNR despite visibly blurry results, because PSNR averages squared pixel error.

## Across GPUs

One image to 262,144 Gaussians in 3.4 s. Request time at 50 steps: median over 8 images, with the same code and
settings on every GPU, one GPU per machine.

| GPU | VRAM | Request time |
|---|---|---|
| B300 | 269 GB | 3.4 s |
| B200 | 179 GB | 3.6 s |
| H200 | 140 GB | 6.6 s |
| H100 | 80 GB | 6.7 s |
| RTX 5090 | 32 GB | 14.1 s |
| A100 | 80 GB | 16.3 s |
| L40S | 45 GB | 16.6 s |
| RTX 4090 | 24 GB | 19.1 s |
| RTX 3080 | 10 GB | 46.2 s |
| L4 | 22 GB | 54.6 s |

## Acknowledgements

Idol builds on [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat). It uses
[DINOv3](https://github.com/facebookresearch/dinov3), the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev)
VAE and [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

**Made with ❤️ by [speridlabs.com](https://speridlabs.com)**
