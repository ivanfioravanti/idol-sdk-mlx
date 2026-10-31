set -x
.venv/bin/python -m perf.onepass torch
.venv/bin/python -m perf.onepass baseline
.venv/bin/python -m perf.profile
.venv/bin/python -m perf.ref_torch avocado
.venv/bin/python -m perf.ref_torch fox
.venv/bin/python -m perf.compare baseline
echo ALLDONE
