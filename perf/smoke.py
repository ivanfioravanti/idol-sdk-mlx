import time, sys
from idol import IdolPipeline
t=time.perf_counter(); p=IdolPipeline.from_pretrained(backend='mlx'); print('load', time.perf_counter()-t, flush=True)
o=p('demo/examples/food-avocado.webp', steps=50, verbose=True)
print('timings', o.timings, 'camera', o.camera, flush=True)
o.gaussian.save_ply('perf/smoke_avocado.ply')
