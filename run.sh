#!/usr/bin/env bash
set -euo pipefail

# One-command runner for Graph-WaveNet (uv-managed env)
# Usage:
#   ./run.sh smoke            # synthetic quick check
#   ./run.sh prepare-real     # generate npz from data/metr-la.h5
#   ./run.sh train-real       # train on data/METR-LA

MODE=${1:-smoke}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERR] uv not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv .venv
  source .venv/bin/activate
  uv pip install -r requirements.txt
  uv pip install tables
else
  source .venv/bin/activate
fi

mkdir -p data/METR-LA data/sensor_graph garage

# Ensure adjacency file exists
if [ ! -f data/sensor_graph/adj_mx.pkl ]; then
  python - <<'PY'
from urllib.request import urlopen
url='https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl'
out='data/sensor_graph/adj_mx.pkl'
with urlopen(url, timeout=30) as r:
    open(out,'wb').write(r.read())
print('saved', out)
PY
fi

if [ "$MODE" = "smoke" ]; then
  echo "[INFO] running smoke test with synthetic metr-la.h5"
  python - <<'PY'
import numpy as np, pandas as pd
np.random.seed(42)
n_steps=1500
n_nodes=207
idx=pd.date_range('2012-01-01', periods=n_steps, freq='5min')
base=np.random.rand(n_steps, n_nodes)*20+30
t=np.arange(n_steps)[:,None]
season=10*np.sin(2*np.pi*t/288)
data=np.clip(base+season+np.random.randn(n_steps,n_nodes)*2,0,None)
df=pd.DataFrame(data,index=idx,columns=[f's{i}' for i in range(n_nodes)])
df.to_hdf('data/metr-la.h5',key='df')
print('saved data/metr-la.h5', df.shape)
PY
  printf 'y\n' | python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
  python train.py --device cpu --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --epochs 1 --batch_size 16 --print_every 20 --save ./garage/metr_smoke
  exit 0
fi

if [ "$MODE" = "prepare-real" ]; then
  if [ ! -f data/metr-la.h5 ]; then
    echo "[ERR] data/metr-la.h5 not found. Please place real METR-LA h5 here first." >&2
    exit 2
  fi
  printf 'y\n' | python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
  echo "[OK] generated data/METR-LA/{train,val,test}.npz"
  exit 0
fi

if [ "$MODE" = "train-real" ]; then
  python train.py --device cpu --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --epochs 10 --batch_size 64 --print_every 50 --save ./garage/metr_real
  exit 0
fi

echo "Unknown mode: $MODE"
exit 1
