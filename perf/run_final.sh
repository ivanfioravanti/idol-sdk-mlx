set -x
.venv/bin/python -m perf.bench_dit opt 3
.venv/bin/python -m perf.compare opt
(cd /tmp/idol-base && ~/github/idol-sdk-mlx/.venv/bin/python -m idol.bench --backends mlx --runs 2 --image demo/examples/food-avocado.webp demo/examples/fig-fox.webp --log ~/github/idol-sdk-mlx/perf/e2e.jsonl)
.venv/bin/python -m idol.bench --backends mlx --runs 2 --image demo/examples/food-avocado.webp demo/examples/fig-fox.webp --log perf/e2e.jsonl
echo ALLDONE
