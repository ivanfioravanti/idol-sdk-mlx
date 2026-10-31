set -x
.venv/bin/python -m perf.bench_dit baseline 3
.venv/bin/python -m perf.ref_torch avocado
.venv/bin/python -m perf.ref_torch fox
.venv/bin/python -m perf.compare baseline
echo ALLDONE
