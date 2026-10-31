.venv/bin/python -m perf.cfgcache u1,u2w3,d2w3,u3w3,d3w3
for v in u1 u2w3 d2w3 u3w3 d3w3; do .venv/bin/python -m perf.compare cfg-$v; done
echo ALLDONE
