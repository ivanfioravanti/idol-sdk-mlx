import contextvars
import json
import os
import sys
import time
import uuid
from pathlib import Path

# The environment may have another copy of idol installed: always import this checkout's.
WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))
import idol
from idol import IdolPipeline

if Path(idol.__file__).resolve().parent != WORKTREE / "idol":
    raise RuntimeError(f"Wrong Idol checkout imported: {idol.__file__}; expected {WORKTREE / 'idol'}")

MODES = {'Synthetic': 'synthetic', 'Real photo': 'real'}
GAUSSIANS = {'65,536': 65536, '131,072': 131072, '262,144': 262144, '524,288': 524288, '1,048,576': 1048576}
DEFAULTS = {'steps': 50, 'cfg': 3.0, 'rescale_t': 3.0, 'gaussians': '262,144', 'seed': 42}

# Load on the first Generate, not on page load: the saved splat can be explored immediately.
pipeline = None
OUTPUT_DIR = Path(os.environ.get("IDOL_OUTPUT_DIR", WORKTREE / "outputs-demo")).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LATEST = OUTPUT_DIR / "latest.json"


def previous():
    if not LATEST.exists():
        return None
    saved = json.loads(LATEST.read_text())
    for key in ("ply", "input", "prepared"):
        if not Path(saved[key]).is_file():
            raise FileNotFoundError(f"Saved Idol result is missing {key}: {saved[key]}")
    return saved


def generate(image, mode, steps, cfg_strength, rescale_t, seed, num_gaussians):
    global pipeline
    started = time.perf_counter()
    load_seconds = None
    if pipeline is None:
        # in an empty context, so Gradio's track_tqdm doesn't show the loading bars: the hub's download bars never
        # close and would sit on the progress row at 0/0 B for the whole first run
        pipeline = contextvars.Context().run(
            IdolPipeline.from_pretrained, os.environ.get("IDOL_WEIGHTS", "speridlabs/idol"),
            device="mps", backend="mlx", optimize=False,
        )
        load_seconds = time.perf_counter() - started
    output = pipeline(image, mode=mode, seed=seed, steps=steps, cfg_strength=cfg_strength, rescale_t=rescale_t,
                      num_gaussians=num_gaussians, verbose=True)   # tqdm per sampling step, shown by app.py
    stem = OUTPUT_DIR / f"idol-seed{seed}-{num_gaussians}-{uuid.uuid4().hex[:8]}"
    ply, source, prepared = (str(stem) + suffix for suffix in (".ply", "_source.png", "_input.png"))
    output.gaussian.save_ply(ply)
    image.save(source)
    output.preprocessed_image.save(prepared)
    saved = {
        "ply": ply, "input": source, "prepared": prepared,
        "mode": mode, "steps": steps, "cfg": cfg_strength, "rescale_t": rescale_t,
        "seed": seed, "num_gaussians": num_gaussians,
        "timings": output.timings, "load_seconds": load_seconds,
        "wall_seconds": time.perf_counter() - started,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "mlx", "device": "mps", "checkout": str(WORKTREE),
    }
    temporary = LATEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2))
    temporary.replace(LATEST)
    return saved
