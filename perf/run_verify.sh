set -x
cd ~/github/idol-sdk-mlx
.venv/bin/python -m perf.bench_dit verify 3
.venv/bin/python -m perf.onepass verify
cd /tmp/idol-15ac
~/github/idol-sdk-mlx/.venv/bin/python -m perf.bench_dit rerun15ac 3
~/github/idol-sdk-mlx/.venv/bin/python -m perf.onepass rerun15ac
cp perf/out_rerun15ac_*.npz ~/github/idol-sdk-mlx/perf/
cd ~/github/idol-sdk-mlx
.venv/bin/python -m perf.compare verify rerun15ac
.venv/bin/python -m perf.compare rerun15ac baseline
echo ALLDONE
