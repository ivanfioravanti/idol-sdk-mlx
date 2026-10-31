---
title: Idol
emoji: 🗿
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.49.1
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: One image to a 3D object (Gaussian splats)
---

# Idol demo

Speridlabs' original Gradio interface, presets and `Model3D` Gaussian-splat renderer, adapted to run locally on Apple
Silicon with the MLX backend. `app.py` owns the UI; `model.py` calls the pipeline without modifying inference code.
`examples/presets.json` lists the original example settings.

## Run

From the repository root:

```bash
uv sync                    # or: pip install -e ".[demo]"
cd demo
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False uv run python app.py   # http://127.0.0.1:7862
```

`model.py` imports `idol` from this repository and refuses any other installed copy. Do not install
`idol @ git+https://github.com/speridlabs/idol-sdk`: it would replace the MLX port.

No model is loaded on startup. The first Generate calls
`IdolPipeline.from_pretrained(..., backend="mlx", device="mps", optimize=False)`: the denoiser runs in MLX, while
background removal, the encoders and the decoders stay on PyTorch MPS. Further requests reuse that pipeline, one at a
time.

| Variable | Default | |
|---|---|---|
| `IDOL_PORT` | `7862` | port, on 127.0.0.1 |
| `IDOL_WEIGHTS` | `speridlabs/idol` | a Hugging Face repo or a local folder |
| `IDOL_OUTPUT_DIR` | `outputs-demo/` | where results go |
| `IDOL_PRELOAD_DIR` | `~/github/idol-sdk-mlx/outputs-mlx` | a folder with `food-avocado.ply` and `food-avocado_input.png`, shown before the first Generate |

## Cached weights

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` keeps the app from downloading anything: missing cache entries produce an
explicit generation failure, not a download. Besides the Idol weights, the DINOv3 and FLUX.2 encoder weights must be
cached.

To fill the caches, request access to `facebook/dinov3-vith16plus-pretrain-lvd1689m` and
`black-forest-labs/FLUX.2-dev`, accept their terms, run `hf auth login`, and start the app once without the two
offline variables. Viewing the preloaded splat needs no downloads.

On startup the viewer shows the preloaded pair from `IDOL_PRELOAD_DIR` (the splat and its processed input, in the
viewer's accordion), with `examples/food-avocado.webp` as the input image. It is labelled **previously generated**: its
original settings and timings are unknown.

## Using it

1. Orbit the splat by dragging; scroll to zoom.
2. Use the avocado or upload another image. Suggested settings: **Synthetic**, **262,144 Gaussians**, **50 steps**,
   **guidance 3**, **time shift 3**, **seed 42**, with **Random seed** unchecked.
3. Press Generate. The old splat stays visible and interactive and an elapsed clock starts. The first generation also
   loads the models. A progress bar on the status line follows the real sampling steps (e.g. 23/50); there is no
   estimated percentage.
4. On success the new splat replaces the old one. The UI reports the pipeline's `preprocess`, `encode`, `sample` and
   `decode` timings, the total wall time (including loading and export), and the model-load time on the first run.
5. On failure an error and the elapsed time appear; the previous splat, processed input and result details stay. The
   terminal shows the traceback.

Each successful result (the `.ply`, the original and processed inputs, the settings and timings) is written to
`outputs-demo/`. `latest.json` is replaced only after a successful export, and the latest result is shown again after
a page reload or an app restart. A request does not survive stopping the server.

## Scripted request

With the app running, this sends the same request as the Generate button (it **runs real inference**):

```bash
python - <<'PY'
from gradio_client import Client, handle_file
client = Client("http://127.0.0.1:7862/")
result = client.predict(
    handle_file("examples/food-avocado.webp"),
    "Synthetic", 50, 3.0, 3.0, 42, False, "262,144",
    api_name="/generate",
)
print(result)
PY
```

It returns the splat file, the seed, the result details (HTML), the status (HTML) and the processed input file. Check
the **status**: failures are returned in it, so the previous result can stay in the browser. A successful run also
writes `outputs-demo/latest.json`; reload the page afterwards to see it there.

## Attribution

Model, original app, examples and viewer integration: [Speridlabs Idol SDK](https://github.com/speridlabs/idol-sdk),
[original Hugging Face Space](https://huggingface.co/spaces/speridlabs/idol), and
[weights](https://huggingface.co/speridlabs/idol). The SDK/demo use Apache-2.0; individual model dependencies retain
their own terms. The local MLX port does not imply upstream endorsement.
